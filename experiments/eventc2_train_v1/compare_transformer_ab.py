"""A(기존 article-level Train c2)와 B(event-level Train c2) Transformer 결과를 같은 표로 비교한다.

각 run 폴더는 train_transformer.py의 save_dir이다.
  - run_summary.json            : Validation(best epoch) 지표
  - test_candidate_scores.parquet: predict_sid.py 출력 (있으면 Test 지표 계산)

    python experiments/eventc2_train_v1/compare_transformer_ab.py \
        --a-run <A seed42 run 폴더> \
        --b-run <exp>/transformer_runs/seed42 \
        --out-dir <exp>/reports/transformer_ab
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


METRICS = [
    ("top1_accuracy", "Top-1 Accuracy", True),
    ("mrr", "MRR", True),
    ("ndcg@5", "nDCG@5", True),
    ("auc", "AUC", True),
    ("preference_loss", "Preference Loss", False),
    ("positive_prob", "Positive Probability", True),
    ("score_gap", "Positive-Negative Score Gap", True),
]


def validation_metrics(run_dir: Path) -> Dict[str, Optional[float]]:
    summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    best = summary.get("best_metrics", {})

    def get(key: str) -> Optional[float]:
        value = best.get(f"val_{key}")
        return None if value is None else float(value)

    positive_score = get("positive_score")
    negative_score = get("negative_score")

    return {
        "best_epoch": summary.get("best_epoch"),
        "num_impressions": get("num_impressions"),
        "top1_accuracy": get("top1_accuracy"),
        "mrr": get("mrr"),
        "ndcg@5": get("ndcg@5"),
        "auc": get("auc"),
        "preference_loss": get("preference_loss"),
        "positive_prob": get("positive_prob"),
        "score_gap": (
            positive_score - negative_score
            if positive_score is not None and negative_score is not None
            else None
        ),
    }


def test_metrics(run_dir: Path) -> Optional[Dict[str, Optional[float]]]:
    path = run_dir / "test_candidate_scores.parquet"
    if not path.exists():
        return None

    df = pd.read_parquet(path)
    group = "sample_index" if "sample_index" in df.columns else "impression_id"

    top1, mrr, ndcg5, auc, loss, pos_prob = [], [], [], [], [], []
    pos_scores, neg_scores = [], []

    for _, g in df.groupby(group, sort=False):
        labels = g["label"].to_numpy(dtype=np.int64)
        scores = g["candidate_score"].to_numpy(dtype=np.float64)

        # evaluate/metrics.py와 같은 정의 (stable argsort, 동점은 앞 index 우선)
        order = np.argsort(-scores, kind="stable")
        ranks = np.empty(len(scores), dtype=np.int64)
        ranks[order] = np.arange(1, len(scores) + 1)

        positive_rank = int(ranks[labels == 1].min())
        top1.append(float(labels[int(np.argmax(scores))] == 1))
        mrr.append(1.0 / positive_rank)
        ndcg5.append(1.0 / np.log2(positive_rank + 1) if positive_rank <= 5 else 0.0)

        positive = scores[labels == 1]
        negative = scores[labels == 0]
        comparisons = positive[:, None] - negative[None, :]
        auc.append(float(((comparisons > 0).sum() + 0.5 * (comparisons == 0).sum()) / comparisons.size))

        shifted = scores - scores.max()
        log_probs = shifted - np.log(np.exp(shifted).sum())
        loss.append(float(-log_probs[labels == 1].mean()))
        pos_prob.append(float(np.exp(log_probs[labels == 1]).mean()))

        pos_scores.extend(positive.tolist())
        neg_scores.extend(negative.tolist())

    return {
        "num_impressions": len(top1),
        "top1_accuracy": float(np.mean(top1)),
        "mrr": float(np.mean(mrr)),
        "ndcg@5": float(np.mean(ndcg5)),
        "auc": float(np.mean(auc)),
        "preference_loss": float(np.mean(loss)),
        "positive_prob": float(np.mean(pos_prob)),
        "score_gap": float(np.mean(pos_scores) - np.mean(neg_scores)),
    }


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def table(title: str, a: Dict[str, Any], b: Dict[str, Any]) -> list[str]:
    lines = [f"### {title}", "", "| metric | A | B | B − A | better |", "|---|---|---|---|---|"]
    lines.append(f"| impressions | {fmt(a.get('num_impressions'))} | {fmt(b.get('num_impressions'))} | | |")
    for key, label, higher_is_better in METRICS:
        av, bv = a.get(key), b.get(key)
        if av is None or bv is None:
            lines.append(f"| {label} | {fmt(av)} | {fmt(bv)} | - | - |")
            continue
        diff = bv - av
        better = "B" if (diff > 0) == higher_is_better and diff != 0 else ("A" if diff != 0 else "=")
        lines.append(f"| {label} | {fmt(av)} | {fmt(bv)} | {diff:+.6f} | {better} |")
    return lines + [""]


def main() -> None:
    parser = argparse.ArgumentParser(description="Transformer A/B 비교")
    parser.add_argument("--a-run", type=Path, required=True)
    parser.add_argument("--b-run", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    report = {
        "a_run": str(args.a_run),
        "b_run": str(args.b_run),
        "validation": {"A": validation_metrics(args.a_run), "B": validation_metrics(args.b_run)},
        "test": {"A": test_metrics(args.a_run), "B": test_metrics(args.b_run)},
    }

    lines = table("Validation (best epoch, Top-1 기준 선택)", report["validation"]["A"], report["validation"]["B"])
    if report["test"]["A"] is not None and report["test"]["B"] is not None:
        lines += table("Test", report["test"]["A"], report["test"]["B"])
    else:
        lines.append("Test: 한쪽 run에 test_candidate_scores.parquet이 없어 비교하지 않았습니다.")

    lines += [
        "",
        "주의: A와 B는 1pos4neg sampling을 각자의 SID로 다시 했으므로",
        "탈락 row 수와 negative 구성이 다를 수 있다. impressions 수와 sampling 리포트를 함께 확인할 것.",
    ]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "transformer_ab_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    (args.out_dir / "transformer_ab_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
