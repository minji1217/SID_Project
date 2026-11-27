"""우리 1pos4neg (train / validation 절반 / test) → Microsoft Recommenders NRMS 가 읽는 MIND 형식.

    python experiments/nrms_recommenders_v1/prepare_mind.py \
        --train <train 1pos4neg> --validation <validation 절반> --test <test 1pos4neg> \
        --articles data/raw/articles.parquet --fasttext ~/minji/fasttext/cc.da.300.vec.gz --out <폴더>

결과 (폴더가 이미 있으면 멈춤)
  news.tsv                     모든 split 의 기사 하나로 (기사 번호, -, -, 제목, -, -, -, -). 공개 코드 반복자는 처음 읽은
                               기사 파일을 기억하므로 split 마다 다른 파일을 쓰지 않는다.
  {train,valid,test}_behaviors.tsv   1pos4neg 한 행 = 한 줄: impression_id, user_id, 시각, history(공백), 후보-정답표시(공백)
  {train,valid,test}_meta.parquet    줄 순서 그대로 impression_id · user_id · 시각 · 후보 · 정답 · 후보 c1 · c2 (점수 파일 만들 때 씀)
  word_dict.pkl, embedding.npy, uid2index.pkl   공개 코드가 요구하는 사전들
  vocab_stats.json
단어: 공개 코드의 word_tokenize (소문자, [\\w]+|[.,!?;|]) 로 자른 제목 단어. 사전 = 모든 기사 제목 단어 중 fastText 에 있는 것
  + train 기사 제목에만 있고 fastText 에 없는 것 (0 벡터로 시작, 학습 중 갱신). 그 밖은 0번 (모름).
  fastText 조회 순서: 소문자 → 첫 글자 대문자 → 대문자. 정답 라벨은 쓰지 않으므로 test 기사 제목 단어의 사전 벡터는 누설이 아니다.
"""
import argparse
import gzip
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import polars as pl
from recommenders.models.newsrec.newsrec_utils import word_tokenize

COLS = ["impression_id", "user_id", "impression_time", "history_article_ids", "candidate_article_ids", "candidate_labels",
        "candidate_c1", "candidate_c2"]


def clean(s) -> str:
    return " ".join(str(s or "").split())   # 탭 · 줄바꿈 제거


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--validation", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--articles", required=True)
    ap.add_argument("--fasttext", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="(테스트용) 각 파일 앞 N행만")
    a = ap.parse_args()
    out = Path(a.out).expanduser()
    if out.exists():
        sys.exit(f"멈춤: {out} 가 이미 있습니다 (덮어쓰지 않음)")

    D = {}
    for k, f in [("train", a.train), ("valid", a.validation), ("test", a.test)]:
        d = pl.read_parquet(f, columns=COLS)
        if a.limit:
            d = d.head(a.limit)
        lab = np.array(d["candidate_labels"].to_list())
        if lab.shape[1] != 5 or not (lab.sum(1) == 1).all():
            sys.exit(f"{f}: 행마다 후보 5개 · 정답 1개여야 합니다")
        if (d["history_article_ids"].list.len() == 0).any():
            sys.exit(f"{f}: history 가 빈 행이 있습니다")
        D[k] = d
        print(f"{k}: {d.height:,}행")

    ids = {k: set(x for col in ("history_article_ids", "candidate_article_ids") for row in d[col].to_list() for x in row) for k, d in D.items()}
    all_ids = sorted(set().union(*ids.values()))
    arts = pl.read_parquet(a.articles, columns=["article_id", "title"])
    title = dict(zip(arts["article_id"].to_list(), arts["title"].to_list()))
    miss = [x for x in all_ids if x not in title]
    toks = {x: word_tokenize(clean(title.get(x, ""))) for x in all_ids}
    all_words = {w for t in toks.values() for w in t}
    train_words = {w for x in ids["train"] for w in toks[x]}

    want = {}
    for w in all_words:
        for rank, f in enumerate(dict.fromkeys((w, w.capitalize(), w.upper()))):
            want.setdefault(f, []).append((w, rank))
    found, dim = {}, None
    ft = str(Path(a.fasttext).expanduser())
    with (gzip.open if ft.endswith(".gz") else open)(ft, "rt", encoding="utf-8", errors="ignore") as fh:
        first = fh.readline().split()
        if len(first) != 2:
            fh.seek(0)
        for line in fh:
            sp = line.rstrip().split(" ")
            hits = want.get(sp[0])
            if not hits:
                continue
            vec = np.asarray(sp[1:], dtype=np.float32)
            dim = dim or len(vec)
            if len(vec) != dim:
                continue
            for w, rank in hits:
                if w not in found or found[w][0] > rank:
                    found[w] = (rank, vec)
    if not found:
        sys.exit("fastText 에서 제목 단어를 하나도 못 찾았습니다")
    vocab = sorted(set(found) | train_words)
    word_dict = {w: i + 1 for i, w in enumerate(vocab)}          # 0 = 패딩 · 모름
    emb = np.zeros((len(vocab) + 1, dim), dtype=np.float32)
    for w, (_, v) in found.items():
        emb[word_dict[w]] = v

    out.mkdir(parents=True)
    with open(out / "news.tsv", "w", encoding="utf-8") as f:
        for x in all_ids:
            f.write(f"{x}\t-\t-\t{clean(title.get(x, ''))}\t\t\t[]\t[]\n")
    users = sorted({u for d in D.values() for u in d["user_id"].to_list()})
    uid2index = {str(u): i + 1 for i, u in enumerate(users)}
    for k, d in D.items():
        with open(out / f"{k}_behaviors.tsv", "w", encoding="utf-8") as f:
            for imp, u, t, h, c, y in zip(d["impression_id"].to_list(), d["user_id"].to_list(), d["impression_time"].to_list(),
                                          d["history_article_ids"].to_list(), d["candidate_article_ids"].to_list(), d["candidate_labels"].to_list()):
                f.write(f"{imp}\t{u}\t{t}\t{' '.join(map(str, h))}\t{' '.join(f'{ci}-{yi}' for ci, yi in zip(c, y))}\n")
        d.drop("history_article_ids").write_parquet(out / f"{k}_meta.parquet")
    pickle.dump(word_dict, open(out / "word_dict.pkl", "wb"))
    pickle.dump(uid2index, open(out / "uid2index.pkl", "wb"))
    np.save(out / "embedding.npy", emb)
    n_tok = sum(len(t) for t in toks.values())
    stats = {"rows": {k: d.height for k, d in D.items()}, "articles": len(all_ids), "articles_without_title": len(miss),
             "title_words_unique": len(all_words), "vocab": len(vocab), "fasttext_found": len(found),
             "token_coverage_fasttext": sum(1 for t in toks.values() for w in t if w in found) / max(1, n_tok),
             "token_unknown": sum(1 for t in toks.values() for w in t if w not in word_dict) / max(1, n_tok),
             "embedding_dim": dim, "train": a.train, "validation": a.validation, "test": a.test, "fasttext": ft}
    (out / "vocab_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
