"""같은 EventC2 test set에서 V1 candidate ranker vs TIGER-style generative 비교 (집계만)

    python experiments/eventc2_tiger_generative_v1/compare_v1_vs_tiger.py \
        --v1-scores <EXP>/transformer_runs/seed42/test_candidate_scores.parquet \
        --tiger-run <EXP>/eventc2_tiger_generative_v1/seed42 \
        --out <EXP>/eventc2_tiger_generative_v1/compare_v1_vs_tiger.md

두 score 파일의 test sample이 같은지 (impression_id, 후보 article 5개, positive) 먼저 확인한다.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_tiger import ranking_metrics  # noqa: E402

METRICS = ["top1_accuracy", "mrr", "ndcg@5", "auc", "preference_loss", "positive_probability"]


def load(path: Path):
    df = pd.read_parquet(path).sort_values(["sample_index"], kind="stable")
    n = df.sample_index.nunique()
    if len(df) != n * 5:
        raise ValueError(f"{path}: sample마다 후보 5개가 아닙니다.")
    scores = df.candidate_score.to_numpy(float).reshape(n, 5)
    labels = df.label.to_numpy(float).astype(int).reshape(n, 5)
    arts = df.article_id.astype(str).to_numpy().reshape(n, 5)
    imps = df.impression_id.to_numpy().reshape(n, 5)[:, 0]
    keys = Counter((str(i), tuple(sorted(a)), a[l.argmax()]) for i, a, l in zip(imps, arts, labels))
    return scores, labels, keys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--v1-scores", type=Path, required=True)
    p.add_argument("--tiger-run", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    v1_s, v1_l, v1_k = load(a.v1_scores)
    tg_s, tg_l, tg_k = load(a.tiger_run / "test_candidate_scores.parquet")
    same = v1_k == tg_k
    print(f"test sample 동일: {same} (V1 {sum(v1_k.values()):,} / TIGER {sum(tg_k.values()):,})")
    if not same:
        raise SystemExit("두 모델의 test sample 구성이 다릅니다. 비교를 중단합니다.")

    v1 = ranking_metrics(v1_s, v1_l)
    tg = ranking_metrics(tg_s, tg_l)
    t = json.loads((a.tiger_run / "test_metrics.json").read_text())
    summary = json.loads((a.tiger_run / "run_summary.json").read_text())

    lines = ["# EventC2 test: V1 candidate ranker vs TIGER-style generative", "",
             f"같은 test sample {sum(v1_k.values()):,}개 (impression·후보 5개·positive 일치 확인)", "",
             "| 지표 | V1 ranker | TIGER-style | 차이 |", "|---|---|---|---|"]
    for m in METRICS:
        lines.append(f"| {m} | {v1[m]:.6f} | {tg[m]:.6f} | {tg[m] - v1[m]:+.6f} |")
    lines += ["", f"TIGER best epoch {summary['best_epoch']} / {summary['epochs_run']} epoch", "",
              "## TIGER 진단 (test)", "", "| level | CE | token accuracy |", "|---|---|---|"]
    for l in ("c1", "c2", "c3", "c4"):
        lines.append(f"| {l} | {t['level_ce'][l]:.4f} | {t['level_token_accuracy'][l]:.4f} |")
    lines += ["", f"- positive 평균 log-prob {t['positive_mean_log_prob']:.4f}",
              f"- negative 평균 log-prob {t['negative_mean_log_prob']:.4f}",
              f"- gap {t['positive_negative_score_gap']:.4f}", "",
              "| 누적 점수 | Top-1 | AUC | MRR | nDCG@5 |", "|---|---|---|---|---|"]
    for k, m in t["cumulative_levels"].items():
        lines.append(f"| {k} | {m['top1_accuracy']:.4f} | {m['auc']:.4f} | {m['mrr']:.4f} | {m['ndcg@5']:.4f} |")
    text = "\n".join(lines) + "\n"
    a.out.write_text(text, encoding="utf-8")
    (a.out.with_suffix(".json")).write_text(json.dumps({"v1": v1, "tiger": tg, "tiger_diagnostics": t}, indent=2))
    print(text)


if __name__ == "__main__":
    main()
