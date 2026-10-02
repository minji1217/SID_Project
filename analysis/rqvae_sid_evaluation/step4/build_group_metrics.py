"""Step 4 보조 — Step 2에서 뽑은 정성평가 group 각각의 내부 pair 유사도 계산.

정성평가(어떤 기사가 묶였나)와 정량평가(얼마나 비슷한가)를 한 화면에서
연결하기 위한 값. Step 3의 main analysis 수치는 건드리지 않고,
동일한 TF-IDF space(9,338 body corpus 1회 fit)를 다시 써서 group 내부만 계산한다.
"""
from __future__ import annotations
import json, sys, itertools
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "step3"))
sys.stdout.reconfigure(line_buffering=True)
from sid_pairs import load, tokenize, jaccard, build_tfidf   # noqa: E402


def main():
    print("모집단 + TF-IDF (Step 3과 동일 설정) ...")
    m = load()
    tr = m[m.split == "train"].reset_index(drop=True)
    pop = tr[tr.body.fillna("").astype(str).str.strip() != ""].reset_index(drop=True)
    vec, X = build_tfidf([tokenize(b) for b in pop.body])
    row = {a: i for i, a in enumerate(pop.aid)}
    tset = {a: set(tokenize(t)) for a, t in zip(pop.aid, pop.title)}
    bset = {a: set(tokenize(b)) for a, b in zip(pop.aid, pop.body)}
    print(f"  {len(pop):,} 기사 · vocab {X.shape[1]:,}")

    def pair_metrics(aa: str, bb: str) -> dict:
        return {
            "title_jaccard": jaccard(tset[aa], tset[bb]),
            "body_jaccard": jaccard(bset[aa], bset[bb]),
            "body_tfidf_cosine": float(X[row[aa]].multiply(X[row[bb]]).sum()),
        }

    S2 = ROOT / "step2"
    out_pairs, out_groups = [], []
    for stem, kind in [("qualitative_prefix3_groups", "prefix_3"),
                       ("qualitative_prefix2_groups", "prefix_2"),
                       ("qualitative_semantic_different", "semantic_different"),
                       ("qualitative_semantic_different_allcodes_supplementary",
                        "semantic_different_supp")]:
        df = pd.read_parquet(S2 / f"{stem}.parquet")
        for gid, g in df.groupby("qualitative_group_id", sort=True):
            ids = list(g.article_id)
            ms = []
            for aa, bb in itertools.combinations(ids, 2):
                pm = pair_metrics(aa, bb)
                out_pairs.append({"kind": kind, "qualitative_group_id": gid,
                                  "article_id_a": aa, "article_id_b": bb, **pm})
                ms.append(pm)
            rec = {"kind": kind, "qualitative_group_id": gid,
                   "n_articles": len(ids), "n_pairs": len(ms)}
            for k in ("title_jaccard", "body_jaccard", "body_tfidf_cosine"):
                v = np.array([x[k] for x in ms], dtype=float)
                v = v[~np.isnan(v)]
                rec[f"{k}_mean"] = float(v.mean()) if v.size else float("nan")
                rec[f"{k}_min"] = float(v.min()) if v.size else float("nan")
                rec[f"{k}_max"] = float(v.max()) if v.size else float("nan")
            out_groups.append(rec)
        print(f"  {kind:26s} group {df.qualitative_group_id.nunique():>3}")

    gp = pd.DataFrame(out_groups)
    pp = pd.DataFrame(out_pairs)
    gp.to_parquet(HERE / "qualitative_group_metrics.parquet", index=False)
    gp.round(6).to_csv(HERE / "qualitative_group_metrics.csv", index=False,
                       encoding="utf-8-sig")
    pp.to_parquet(HERE / "qualitative_group_pair_metrics.parquet", index=False)
    print(f"\n  group {len(gp):,} · 내부 pair {len(pp):,} 저장")
    for kind, g in gp.groupby("kind"):
        print(f"  {kind:26s} TF-IDF cosine group mean  "
              f"min {g.body_tfidf_cosine_mean.min():.3f} / "
              f"median {g.body_tfidf_cosine_mean.median():.3f} / "
              f"max {g.body_tfidf_cosine_mean.max():.3f}")


if __name__ == "__main__":
    main()
