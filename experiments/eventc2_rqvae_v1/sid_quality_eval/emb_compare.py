import sys, json
from math import comb
import numpy as np, pandas as pd
S = sys.argv[1]
E = f"{S}/sideval/analysis/rqvae_sid_evaluation"
sys.path.insert(0, f"{E}/step3"); sys.path.insert(0, f"{E}/step6_prefix1")
import sid_pairs as sp
from sid_pairs import PairSet, all_prefix3_pairs, prefix2_group_balanced, semantic_different_group_balanced

# step6c의 prefix1 sampler (모듈 import 시 matplotlib 폰트 설정만 하므로 함수만 복사)
def prefix1_group_balanced(d, n, rng):
    groups = {}
    for key, idx in d.groupby("c1").indices.items():
        sub = d.iloc[idx].groupby("c2").indices
        subs = {c2: idx[v] for c2, v in sub.items()}
        if len(subs) >= 2: groups[key] = subs
    keys = sorted(groups); ps, stale = PairSet(), 0
    while len(ps) < n and stale < len(keys) * 50:
        rng.shuffle(keys); progressed = False
        for key in keys:
            if len(ps) >= n: break
            subs = groups[key]
            a2, b2 = rng.choice(list(subs), size=2, replace=False)
            if ps.add(int(rng.choice(subs[a2])), int(rng.choice(subs[b2]))): progressed = True
        stale = 0 if progressed else stale + 1
    return ps

X = np.load(f"{S}/real_data/article_embeddings.npy").astype(np.float64)
X /= np.linalg.norm(X, axis=1, keepdims=True)
master = pd.read_parquet(f"{S}/real_data/article_master.parquet")[["article_id", "event_id"]]
master["article_id"] = master.article_id.astype(str)

SIDS = {"UNI": f"{S}/real_A2/article_semantic_ids.parquet",
        "EventC2": f"{S}/aws_res/out/semantic_ids/eventc2_lu0.05_m0.5/article_semantic_ids.parquet"}
P = lambda s: int(sum(comb(int(n), 2) for n in s if n >= 2))
out = {}
for name, path in SIDS.items():
    d = pd.read_parquet(path); d["article_id"] = d.article_id.astype(str)
    d = d[d.split == "train"].sort_values("article_id").reset_index(drop=True)
    d = d.drop(columns=["event_id"]).merge(master, on="article_id", how="left", validate="1:1")
    A, AB, ABC = P(d.groupby("c1").size()), P(d.groupby(["c1","c2"]).size()), P(d.groupby(["c1","c2","c3"]).size())
    counts = {"prefix_3": ABC, "prefix_2": AB - ABC, "prefix_1": A - AB,
              "different": sp.exact_counts(d)["semantic_different"]}
    rng = np.random.default_rng(42)
    sets = {"prefix_3": all_prefix3_pairs(d),
            "prefix_2": prefix2_group_balanced(d, min(50000, counts["prefix_2"]), rng),
            "prefix_1": prefix1_group_balanced(d, min(50000, counts["prefix_1"]), rng),
            "different": semantic_different_group_balanced(d, 50000, rng)}
    rows = d.embedding_row.to_numpy(); ev = d.event_id.to_numpy()
    res = {}
    for cond, ps in sets.items():
        f = pd.DataFrame(ps.rows); ia, ib = f.idx_a.to_numpy(), f.idx_b.to_numpy()
        cos = (X[rows[ia]] * X[rows[ib]]).sum(1)
        same_ev = ev[ia] == ev[ib]
        res[cond] = {"population_pairs": counts[cond], "sampled": len(f),
                     "cos_mean": cos.mean(), "cos_median": float(np.median(cos)),
                     "same_event_ratio": same_ev.mean(),
                     "cos_mean_same_event": cos[same_ev].mean() if same_ev.any() else None,
                     "cos_mean_diff_event": cos[~same_ev].mean() if (~same_ev).any() else None}
    out[name] = res
    print(f"\n== {name} (train {len(d):,})")
    for cond in ("different", "prefix_1", "prefix_2", "prefix_3"):
        r = res[cond]
        print(f"  {cond:10s} pairs {r['population_pairs']:>11,}  sampled {r['sampled']:>6,}  "
              f"cos mean {r['cos_mean']:.4f} median {r['cos_median']:.4f}  "
              f"same-event {r['same_event_ratio']:.1%}  "
              f"(same-ev {r['cos_mean_same_event'] if r['cos_mean_same_event'] is None else round(r['cos_mean_same_event'],4)}, "
              f"diff-ev {r['cos_mean_diff_event'] if r['cos_mean_diff_event'] is None else round(r['cos_mean_diff_event'],4)})")
json.dump(out, open(f"{S}/ec_eval/emb_compare.json", "w"), indent=2, default=float)
