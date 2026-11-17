"""ver1(기존 UNI) vs ver2(고정 클러스터 전처리로 재학습)를 같은 기사 집합에서 비교한다. 학습 없이 평가만 한다.

기사 집합 (ver2 전처리 결과 기준)
  V      validation-only 2,741   두 모델 모두 학습에 안 쓴 기사 (ver1 validation 3,122 의 부분집합)  ← 공정 비교의 핵심
  T_old  ver1 Train 9,738        두 모델 모두 학습한 기사 (raw 에서 history ∪ 현재 ∪ 클릭 으로 다시 계산)
  T_new  늘어난 Train 681        ver2 만 학습 (ver1 에게는 381개는 validation, 300개는 처음 보는 기사)

지표 (모델 · 집합마다)
  rec / total / codebook / commitment / uniqueness loss
      학습 코드 evaluate() 와 같은 계산: 256개 묶음, 순서 고정, 묶음 크기 가중 평균.
      rec 는 기사별 값의 평균이라 묶음 구성과 무관하다. uniqueness 는 묶음 안의 쌍으로 계산하므로 집합이 다르면 값도 달라진다.
  SID (기사 단위 c1,c2,c3 = 학습 때와 같은 nearest code)
      collision rate = (c1,c2,c3) 가 같은 기사가 2개 이상인 그룹에 속한 기사 ÷ 기사 수
      고유 SID 수, Q2 · Q3 사용 코드 수
      주의: validation 기사의 c2 도 여기서는 기사 단위 nearest 다. 실제 SID 생성(timeupdate, Train 사건 배정)과는 다르다.

자기 검증: ver2 checkpoint 의 validation_state(학습 중 기록한 valid rec)와 V 에서 다시 계산한 rec 가 같아야 한다.

사용 (AWS):
  python experiments/rqvae_uni_fixedcluster_v1/compare_ver1_ver2.py \
      --code_dir ~/minji/rqvae_uni_code_127ae7e/RQVAE \
      --ver1 ~/minji/checkpoint_best_rec.pt \
      --ver2 data/output/rqvae/uni_fixedcluster_lu0.05_m0.5_s42/checkpoint_best_rec.pt \
      --input_dir data/output/fixedcluster_v1/exports/rqvae_inputs \
      --out data/output/rqvae/compare_ver1_ver2
"""
import argparse
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def old_train_ids(raw: Path, valid_ids: set) -> set:
    """ver1(수정 전) Train 기사 = train history ∪ behaviors 현재 ∪ 클릭, 유효 기사만 (src/build_train.py 수정 전 규칙)."""
    h = pd.read_parquet(raw / "train/history.parquet", columns=["user_id", "article_id_fixed", "impression_time_fixed"])
    dup_u = set(h.user_id[h.user_id.notna() & h.user_id.duplicated(keep=False)])
    ids = set()
    for u, a, t in zip(h.user_id, h.article_id_fixed, h.impression_time_fixed):
        if u is None or pd.isna(u) or u in dup_u or a is None or t is None or len(a) != len(t):
            continue
        ids.update(int(x) for x, y in zip(a, t) if x is not None and y is not None and not pd.isna(y))
    b = pd.read_parquet(raw / "train/behaviors.parquet",
                        columns=["impression_id", "user_id", "impression_time", "article_id", "article_ids_clicked"])
    dup_i = set(b.impression_id[b.impression_id.notna() & b.impression_id.duplicated(keep=False)])
    for i, u, t, cur, clk in zip(b.impression_id, b.user_id, b.impression_time, b.article_id, b.article_ids_clicked):
        if pd.isna(i) or i in dup_i or pd.isna(u) or pd.isna(t) or clk is None:
            continue
        clk = [int(x) for x in clk if x is not None]
        if not clk:
            continue
        ids.update(clk)
        if cur is not None and not pd.isna(cur):
            ids.add(int(cur))
    return ids & valid_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code_dir", required=True, help="uniqueness-loss-experiment 의 RQVAE 폴더 (modules import 용)")
    ap.add_argument("--ver1", required=True)
    ap.add_argument("--ver2", required=True)
    ap.add_argument("--input_dir", required=True, help="ver2 전처리 exports/rqvae_inputs")
    ap.add_argument("--raw_dir", default="data/raw")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch_size", type=int, default=256)
    a = ap.parse_args()

    sys.path.insert(0, str(Path(a.code_dir).expanduser().resolve()))
    import torch
    from modules.rqvae import RqVae

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    d = Path(a.input_dir)
    tr = pd.read_parquet(d / "article_master.parquet")
    va = pd.read_parquet(d / "validation_article_master.parquet")
    emb = np.load(d / "article_embeddings.npy", mmap_mode="r")
    master = pd.concat([tr, va]).set_index("article_id")

    t_new_all = set(tr.article_id.astype(int))
    # 유효 기사 중 수정 전 규칙(history ∪ 현재 ∪ 클릭)에 걸리는 기사는 모두 새 Train 에도 있으므로
    # 새 Train 기사 집합을 유효 기사 필터로 쓴다 (제목·부제 빈 기사 등은 여기서 빠짐)
    t_old = old_train_ids(Path(a.raw_dir), t_new_all)
    sets = {
        "V (validation-only 2,741 · 둘 다 학습 안 함)": sorted(va.article_id.astype(int)),
        "T_old (ver1 Train 9,738 · 둘 다 학습)": sorted(t_old),
        "T_new (늘어난 Train 681 · ver2만 학습)": sorted(t_new_all - t_old),
    }
    n = {k: len(v) for k, v in sets.items()}
    print("[집합]", n)
    assert len(t_old) == 9738, f"ver1 Train 재계산이 9,738 이 아닙니다 ({len(t_old)}). raw 데이터를 확인하세요"
    assert len(t_new_all - t_old) == 681

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load(path):
        ck = torch.load(Path(path).expanduser(), map_location="cpu", weights_only=False)
        cfg = dict(ck["model_config"])
        acc = set(inspect.signature(RqVae.__init__).parameters) - {"self"}
        m = RqVae(**{k: v for k, v in cfg.items() if k in acc})
        m.load_state_dict(ck["model"], strict=True)
        return m.to(dev).eval(), ck

    @torch.no_grad()
    def evaluate(model, ids):
        sums = dict(loss=0.0, reconstruction_loss=0.0, codebook_loss=0.0, commitment_loss=0.0, uniqueness_loss=0.0, rqvae_loss=0.0)
        sids = []
        for s in range(0, len(ids), a.batch_size):
            chunk = ids[s:s + a.batch_size]
            rows = master.loc[chunk]
            x = torch.from_numpy(np.asarray(emb[rows.embedding_row.to_numpy()], dtype=np.float32)).to(dev)
            c = torch.from_numpy(rows.model_category_id.to_numpy().astype(np.int64)).to(dev)
            o = model(x=x, category_ids=c)
            for k in sums:
                sums[k] += float(getattr(o, k).detach().float().item()) * len(chunk)
            sids.append(model.get_semantic_ids(x=x, category_ids=c).sem_ids.detach().cpu().numpy())
        res = {k: v / len(ids) for k, v in sums.items()}
        sid = np.concatenate(sids)
        df = pd.DataFrame(sid[:, :3], columns=["c1", "c2", "c3"])
        g = df.groupby(["c1", "c2", "c3"]).size()
        res.update({
            "articles": len(ids),
            "unique_sid": int(len(g)),
            "collision_rate": float(g[g >= 2].sum() / len(ids)),
            "q2_used": int(df.c2.nunique()),
            "q3_used": int(df.c3.nunique()),
        })
        return res

    report = {"sets": n, "models": {}}
    for name, path in [("ver1", a.ver1), ("ver2", a.ver2)]:
        model, ck = load(path)
        info = {"epoch": int(ck.get("epoch", -1)) + 1, "validation_state_at_save": ck.get("validation_state")}
        info["results"] = {k: evaluate(model, v) for k, v in sets.items()}
        report["models"][name] = info
        print(f"[{name}] epoch {info['epoch']} · 저장 당시 {info['validation_state_at_save']}")

    # 자기 검증: ver2 의 학습 중 valid rec 와 V 재계산 값
    vs = report["models"]["ver2"]["validation_state_at_save"] or {}
    rec_v2 = report["models"]["ver2"]["results"][next(iter(sets))]["reconstruction_loss"]
    if "valid_reconstruction_loss" in vs:
        diff = abs(rec_v2 - vs["valid_reconstruction_loss"])
        report["self_check"] = {"ver2_logged_valid_rec": vs["valid_reconstruction_loss"], "recomputed_on_V": rec_v2, "abs_diff": diff}
        print(f"[자기 검증] ver2 학습 기록 valid rec {vs['valid_reconstruction_loss']:.6f} · V 재계산 {rec_v2:.6f} · 차이 {diff:.2e}")

    rows = []
    for s in sets:
        for k in ["reconstruction_loss", "loss", "codebook_loss", "commitment_loss", "uniqueness_loss",
                  "collision_rate", "unique_sid", "q2_used", "q3_used"]:
            v1 = report["models"]["ver1"]["results"][s][k]
            v2 = report["models"]["ver2"]["results"][s][k]
            rows.append({"set": s, "metric": k, "ver1": v1, "ver2": v2, "ver2-ver1": v2 - v1})
    tab = pd.DataFrame(rows)
    tab.to_csv(out / "compare_ver1_ver2.csv", index=False)
    (out / "compare_ver1_ver2.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 60)
    print()
    print(tab.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"\n저장: {out}/compare_ver1_ver2.csv, compare_ver1_ver2.json")


if __name__ == "__main__":
    main()
