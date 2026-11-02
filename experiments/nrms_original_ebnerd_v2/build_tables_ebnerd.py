"""
[EB-NeRD 적용본] 정민교 ~/minkyo/nrms/build_tables.py(md5 6bc488c4…)를 복사해 아래 두 곳만 바꿨다.
  변경 1) TOKEN_RE: [a-z0-9]+ → \\w+ (덴마크어 æ·ø·å가 단어를 쪼개지 않도록. 영어 MIND에서는 결과가 같다)
  변경 2) 테이블 행 수 N: EB-NeRD의 embedding_row는 0~19,906 중 12,860개만 쓰여 연속이 아니다.
          N = max(embedding_row)+1로 두고, 기사가 없는 행은 0(pad와 같은 영벡터/코드)으로 남긴다.
  변경 3) sequence에서 SID를 모을 때 메모리 부족(Killed)이 나서 컬럼 4개만 나눠 읽음 (모으는 값은 같음)
모델·데이터셋·학습 코드(model.py, dataset.py, train.py)는 원본 그대로다.

기사 조회 테이블 생성 (train + validation 전체 기사).

행 번호는 master parquet의 embedding_row를 그대로 쓴다 (= article_embeddings.npy의 행 번호).

시맨틱 ID는 두 가지 방법 중 하나로 가져온다.
  (a) --seq_paths : 시퀀스 parquet(train/validation/test)의 history_c1~c3, candidate_c1~c3, target_c1~c3 에서 모음.
                    V1이 실제로 입력으로 쓴 코드와 똑같아서 비교가 가장 정확하다. (권장)
  (b) --sid_parquet : article_semantic_ids.parquet 를 직접 읽음.

[이전 버전의 문제 수정]
이전 data_prep.py는 train용 article_master.parquet만 읽어서, validation에만 있는 기사
(validation 후보 기사의 약 48%)의 제목이 비어(영벡터) 있었다. 지금은 validation_article_master까지 합친다.

출력 (out_dir):
  vocab.pkl, title_tokens.npy, sid_codes.npy, sid_shuffled.npy, article_id2row.pkl

사용 예 (서버):
  python3 build_tables.py --raw_dir ~/shared/raw/mind --out_dir . \
     --seq_paths ~/shared/datasets/mind/train_sequences_1pos4neg.parquet \
                 ~/shared/datasets/mind/validation_sequences_1pos4neg_half.parquet
"""
import argparse
import pickle
import re

import numpy as np
import pandas as pd

MAX_TITLE_LEN = 20
MIN_WORD_FREQ = 2
TOKEN_RE = re.compile(r"\w+")  # [EB-NeRD 변경 1] 원본: r"[a-z0-9]+"


def extract_title(model_text: str) -> str:
    text = model_text
    if text.startswith("query:"):
        text = text[len("query:"):].strip()
    return text.split("\n", 1)[0]


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


def norm_id(a):
    return int(str(a).lstrip("N"))


def sid_from_sequences(paths, N, id2row):
    """시퀀스 parquet에서 기사별 (c1,c2,c3)를 모은다. 반환: codes (N,3), has_sid (N,) bool"""
    # [EB-NeRD 변경 3] 원본은 parquet 전체를 읽어 pandas explode로 펼쳤다. EB-NeRD는 history가 길어
    # AWS에서 메모리 부족(Killed)이 나서, 필요한 4개 컬럼만 2만 행씩 읽어 펼치고 바로 중복을 없앤다.
    # 모으는 값((article_id, c1, c2, c3) 고유 조합)은 원본과 같다.
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    parts = []
    for p in paths:
        pf = pq.ParquetFile(p)
        names = set(pf.schema_arrow.names)
        for pre in ("history", "candidate", "target"):
            cols = [f"{pre}_article_ids", f"{pre}_c1", f"{pre}_c2", f"{pre}_c3"]
            if not all(c in names for c in cols):
                continue
            seen = None
            for batch in pf.iter_batches(batch_size=20000, columns=cols):
                flat = [pc.list_flatten(batch.column(i)) for i in range(4)]
                lens = {len(f) for f in flat}
                if len(lens) != 1:
                    raise ValueError(f"{p} {pre}: 리스트 길이가 컬럼마다 다릅니다 {lens}")
                arr = np.stack([np.asarray(f.cast("int64").fill_null(-1)) for f in flat], 1)
                arr = np.unique(arr[(arr >= 0).all(1)], axis=0)
                seen = arr if seen is None else np.unique(np.concatenate([seen, arr]), axis=0)
            if seen is not None:
                parts.append(pd.DataFrame(seen, columns=["aid", "c1", "c2", "c3"]))
        print(f"[build] 시퀀스 읽음: {p} ({pf.metadata.num_rows}행)")
    allp = pd.concat(parts).drop_duplicates()
    conflict = allp.groupby("aid").size()
    n_conf = int((conflict > 1).sum())
    print(f"[build] SID가 있는 기사 = {allp.aid.nunique()}, 한 기사에 코드가 2종 이상인 경우 = {n_conf} (0이어야 정상)")
    allp = allp.drop_duplicates("aid")
    codes = np.zeros((N, 3), dtype=np.int64)
    has = np.zeros(N, dtype=bool)
    miss = 0
    for aid, c1, c2, c3 in allp.itertuples(index=False):
        r = id2row.get(int(aid))
        if r is None:
            miss += 1
            continue
        codes[r] = (c1, c2, c3)
        has[r] = True
    print(f"[build] 기사 테이블에 없는 SID 기사 = {miss} (0이어야 정상), SID 있는 행 = {int(has.sum())}/{N}")
    return codes, has


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default="/mnt/user-data/uploads", help="article_master/validation_article_master 폴더")
    ap.add_argument("--out_dir", default=".")
    ap.add_argument("--seq_paths", nargs="*", default=[])
    ap.add_argument("--sid_parquet", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    R, O = args.raw_dir, args.out_dir

    tm = pd.read_parquet(f"{R}/article_master.parquet")[["article_id", "embedding_row", "model_text"]]
    vm = pd.read_parquet(f"{R}/validation_article_master.parquet")[["article_id", "embedding_row", "model_text"]]
    tm["is_train"] = True
    vm["is_train"] = False
    m = pd.concat([tm, vm]).reset_index(drop=True)
    N = int(m["embedding_row"].max()) + 1
    # [EB-NeRD 변경 2] 원본: assert ... and N == len(m). EB-NeRD는 embedding_row가 연속이 아니다.
    assert m["embedding_row"].is_unique, f"embedding_row 중복: rows={len(m)}"
    print(f"[build] embedding_row 0~{N - 1} 중 기사 {len(m)}개 사용 (나머지 행은 영벡터)")
    m = m.sort_values("embedding_row").reset_index(drop=True)
    print(f"[build] articles = {len(m)}  (train {int(m.is_train.sum())}, validation-only {int((~m.is_train).sum())})")

    ids = [norm_id(a) for a in m["article_id"]]
    id2row = {i: int(r) for i, r in zip(ids, m["embedding_row"])}
    titles = [extract_title(t) for t in m["model_text"]]

    freq = {}
    for t, tr in zip(titles, m["is_train"]):
        if tr:
            for w in tokenize(t):
                freq[w] = freq.get(w, 0) + 1
    vocab = {"<pad>": 0, "<unk>": 1}
    for w, c in sorted(freq.items(), key=lambda x: -x[1]):
        if c >= MIN_WORD_FREQ:
            vocab[w] = len(vocab)
    print(f"[build] vocab size = {len(vocab)}")

    tok = np.zeros((N, MAX_TITLE_LEN), dtype=np.int64)
    for r, t in zip(m["embedding_row"], titles):
        w = [vocab.get(x, 1) for x in tokenize(t)][:MAX_TITLE_LEN]
        tok[r, :len(w)] = w

    if args.seq_paths:
        codes, has = sid_from_sequences(args.seq_paths, N, id2row)
    elif args.sid_parquet:
        s = pd.read_parquet(args.sid_parquet)
        codes = np.zeros((N, 3), dtype=np.int64)
        has = np.zeros(N, dtype=bool)
        for a, r, c1, c2, c3 in zip(s["article_id"], s["embedding_row"], s["c1"], s["c2"], s["c3"]):
            codes[int(r)] = (c1, c2, c3)
            has[int(r)] = True
    else:
        raise SystemExit("--seq_paths 또는 --sid_parquet 중 하나가 필요합니다")

    # 대조군: SID가 있는 기사들끼리만 코드를 섞는다 (그룹 크기 구조 유지, 의미만 제거)
    shuffled = codes.copy()
    idx = np.where(has)[0]
    perm = np.random.default_rng(args.seed).permutation(len(idx))
    shuffled[idx] = codes[idx][perm]

    np.save(f"{O}/title_tokens.npy", tok)
    np.save(f"{O}/sid_codes.npy", codes)
    np.save(f"{O}/sid_shuffled.npy", shuffled)
    with open(f"{O}/vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    with open(f"{O}/article_id2row.pkl", "wb") as f:
        pickle.dump(id2row, f)
    print(f"[build] c1 max={codes[has,0].max()}, c2 max={codes[has,1].max()}, c3 max={codes[has,2].max()}")
    print("[build] saved: title_tokens.npy sid_codes.npy sid_shuffled.npy vocab.pkl article_id2row.pkl")


if __name__ == "__main__":
    main()
