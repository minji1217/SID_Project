"""결과 모으기: RQ-VAE(summary.json)와 Transformer(test_metrics.json)를 표 하나로.

    python experiments/v2_yeomin_rerun/collect_results.py ~/minji/SID_Project/data/output/fixedcluster_v1/v2

결과 표를 <V2>/results_rqvae.csv, <V2>/results_transformer.csv 로도 저장한다.
"""
import json
import sys
from pathlib import Path

import pandas as pd

v2 = Path(sys.argv[1]).expanduser()

rows = []
for d in sorted((v2 / "rqvae").glob("*_s*")):
    f = d / "summary.json"
    if f.exists() and "smoke" not in d.name:
        s = json.loads(f.read_text())
        variant, seed = d.name.split("_s", 1)
        rows.append({"실험": variant.upper(), "seed": seed, "best_rec": s.get("best_rec"), "best_rec_epoch": s.get("best_rec_epoch"),
                     "마지막 epoch": s.get("last_epoch"), "조기 종료": s.get("early_stopped")})
if rows:
    r = pd.DataFrame(rows)
    print("== RQ-VAE (validation 복원 손실 최소 checkpoint)")
    print(r.to_string(index=False))
    r.to_csv(v2 / "results_rqvae.csv", index=False)

rows = []
for d in sorted((v2 / "transformer").glob("*_seed*")):
    f = d / "test_metrics.json"
    if f.exists() and "_ep" not in d.name:
        m = json.loads(f.read_text())
        run, seed = d.name.split("_seed", 1)
        rows.append({"run": run.upper(), "seed": seed, "test 행": m.get("num_impressions"), "Top-1": m.get("top1_accuracy"),
                     "AUC": m.get("auc"), "MRR": m.get("mrr"), "nDCG@5": m.get("ndcg@5")})
if rows:
    t = pd.DataFrame(rows)
    print("\n== Transformer test (seed별)")
    print(t.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    g = t.groupby("run")[["Top-1", "AUC", "MRR", "nDCG@5"]].agg(["mean", "std", "count"])
    print("\n== Transformer test (seed 평균 ± 표준편차)")
    print(g.to_string(float_format=lambda x: f"{x:.4f}"))
    t.to_csv(v2 / "results_transformer.csv", index=False)
