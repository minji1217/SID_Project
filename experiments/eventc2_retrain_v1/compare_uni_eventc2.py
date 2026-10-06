"""UNI baseline vs EventC2 Transformer 비교 (집계만, 학습/추론 없음)

    python experiments/eventc2_retrain_v1/compare_uni_eventc2.py \
        --uni-train <UNI train 1pos4neg> --uni-val-half <...> --uni-test <...> \
        --uni-scores <UNI test_candidate_scores.parquet> --uni-run <UNI run 폴더> \
        --uni-neg-report <UNI 1pos4neg 단계별 리포트 json (선택)> \
        --ec-train ... --ec-val-half ... --ec-test ... --ec-scores ... --ec-run ... --ec-neg-report ... \
        --out-dir <EXP>/reports/uni_vs_eventc2

출력
  1. split별 row / impression 수
  2. Test impression 겹침: 공통 / UNI-only / EventC2-only,
     공통 impression 중 candidate article_id 구성(positive 포함)이 완전히 같은 수/비율
  3. Test 6개 지표: 전체 test / 공통 impression / 공통 + candidate 구성 동일
     Top-1, MRR, nDCG@5, AUC: evaluate/metrics.py와 같은 정의 (sample 평균)
     Preference Loss = mean(-log softmax(score)[positive])  (modules/loss.py의 cross_entropy와 같음)
     Positive Probability = positive candidate의 softmax 확률 평균 (predict_sid.py와 같음)
  4. Validation(학습 중 best) 지표 (run_summary.json)
  5. 1pos4neg 단계별 감소량 (P-N, N-N, drop)
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ["top1_accuracy", "mrr", "ndcg@5", "auc", "preference_loss", "positive_probability"]


def split_counts(path: Path) -> dict:
    df = pd.read_parquet(path, columns=["impression_id"])
    return {"rows": int(len(df)), "impressions": int(df["impression_id"].nunique())}


def candidate_composition(path: Path) -> dict:
    """impression -> 각 row의 (positive, 정렬된 candidate article_id) multiset"""
    df = pd.read_parquet(path, columns=["impression_id", "candidate_article_ids", "candidate_labels"])
    result: dict = {}
    for impression_id, candidates, labels in zip(
        df["impression_id"], df["candidate_article_ids"], df["candidate_labels"]
    ):
        candidates = [str(c) for c in candidates]
        positive = candidates[list(labels).index(1)]
        result.setdefault(impression_id, Counter())[(positive, tuple(sorted(candidates)))] += 1
    return result


def ranking_metrics(scores: pd.DataFrame) -> dict:
    df = scores.sort_values(["sample_index"], kind="stable")
    sizes = df.groupby("sample_index", sort=False).size()
    if not (sizes == 5).all():
        raise ValueError("sample마다 candidate 5개여야 합니다.")
    n = len(sizes)
    labels = df["label"].to_numpy(dtype=np.int64).reshape(n, 5)
    score = df["candidate_score"].to_numpy(dtype=np.float64).reshape(n, 5)
    if not (labels.sum(1) == 1).all():
        raise ValueError("sample마다 positive 1개여야 합니다.")

    pos = labels.argmax(1)
    order = np.argsort(-score, axis=1, kind="stable")
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.arange(1, 6)[None, :].repeat(n, 0), axis=1)
    pos_rank = ranks[np.arange(n), pos]
    pos_score = score[np.arange(n), pos][:, None]
    neg = labels == 0
    wins = ((pos_score > score) & neg).sum(1)
    ties = ((pos_score == score) & neg).sum(1)

    shifted = score - score.max(1, keepdims=True)
    log_prob = shifted - np.log(np.exp(shifted).sum(1, keepdims=True))
    pos_log_prob = log_prob[np.arange(n), pos]

    return {
        "num_samples": int(n),
        "num_impressions": int(df["impression_id"].nunique()),
        "top1_accuracy": float((score.argmax(1) == pos).mean()),
        "mrr": float((1.0 / pos_rank).mean()),
        "ndcg@5": float((1.0 / np.log2(pos_rank + 1)).mean()),
        "auc": float(((wins + 0.5 * ties) / 4).mean()),
        "preference_loss": float((-pos_log_prob).mean()),
        "positive_probability": float(np.exp(pos_log_prob).mean()),
    }


def stage_table(report_path: Path | None) -> list:
    if report_path is None or not report_path.exists():
        return []
    keep = [
        "split_name", "impression_count", "positive_count", "pair_count_total",
        "pn_same_sid_removed_pair_count", "pn_same_sid_removed_pair_ratio",
        "nn_dedup_removed_pair_count", "nn_dedup_removed_pair_ratio",
        "dropped_positive_count", "dropped_positive_ratio",
        "output_row_count", "output_impression_count",
    ]
    return [{k: r.get(k) for k in keep} for r in json.loads(report_path.read_text(encoding="utf-8"))]


def main():
    parser = argparse.ArgumentParser()
    for prefix in ("uni", "ec"):
        for name in ("train", "val-half", "test", "scores", "run"):
            parser.add_argument(f"--{prefix}-{name}", type=Path, required=True)
        parser.add_argument(f"--{prefix}-neg-report", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"이미 존재합니다. 덮어쓰지 않습니다: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    sides = {
        "UNI": dict(train=args.uni_train, val=args.uni_val_half, test=args.uni_test,
                    scores=args.uni_scores, run=args.uni_run, neg=args.uni_neg_report),
        "EventC2": dict(train=args.ec_train, val=args.ec_val_half, test=args.ec_test,
                        scores=args.ec_scores, run=args.ec_run, neg=args.ec_neg_report),
    }

    report: dict = {"splits": {}, "test_overlap": {}, "test_metrics": {}, "validation_best": {}, "1pos4neg_stages": {}}

    for name, side in sides.items():
        report["splits"][name] = {
            "train": split_counts(side["train"]),
            "validation_half": split_counts(side["val"]),
            "test": split_counts(side["test"]),
        }
        summary = json.loads((side["run"] / "run_summary.json").read_text(encoding="utf-8"))
        report["validation_best"][name] = {
            "best_epoch": summary.get("best_epoch"),
            **{k: v for k, v in summary.get("best_metrics", {}).items() if k.startswith("val_")},
        }
        report["1pos4neg_stages"][name] = stage_table(side["neg"])

    # test 겹침
    uni_comp = candidate_composition(args.uni_test)
    ec_comp = candidate_composition(args.ec_test)
    common = set(uni_comp) & set(ec_comp)
    identical = {i for i in common if uni_comp[i] == ec_comp[i]}
    report["test_overlap"] = {
        "common_impressions": len(common),
        "uni_only_impressions": len(set(uni_comp) - common),
        "eventc2_only_impressions": len(set(ec_comp) - common),
        "common_identical_candidates": len(identical),
        "common_identical_candidates_ratio": len(identical) / len(common) if common else None,
    }

    # test 지표
    for name, side in sides.items():
        scores = pd.read_parquet(side["scores"])
        test_ids = set(pd.read_parquet(side["test"], columns=["impression_id"])["impression_id"])
        if set(scores["impression_id"]) != test_ids:
            raise ValueError(f"{name}: score 파일의 impression이 test 파일과 다릅니다.")
        full = ranking_metrics(scores)
        official = side["run"] / "test_metrics.json"
        if official.exists():
            official_metrics = json.loads(official.read_text(encoding="utf-8"))
            for key in ("top1_accuracy", "mrr", "ndcg@5", "auc"):
                if abs(official_metrics[key] - full[key]) > 1e-9:
                    raise ValueError(f"{name}: {key} 재계산 {full[key]} != test_metrics.json {official_metrics[key]}")
        report["test_metrics"][name] = {
            "full_test": full,
            "common_impressions": ranking_metrics(scores[scores["impression_id"].isin(common)]),
            "common_identical_candidates": ranking_metrics(scores[scores["impression_id"].isin(identical)]),
        }

    (args.out_dir / "uni_vs_eventc2.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    lines = ["# UNI vs EventC2 Transformer (seed 42, V1 config)", ""]
    lines += ["## Test 지표", "", "| 범위 | 지표 | UNI | EventC2 | 차이 |", "|---|---|---|---|---|"]
    for scope, label in (("full_test", "각자 전체 Test"), ("common_impressions", "공통 impression"),
                         ("common_identical_candidates", "공통 + candidate 동일")):
        u, e = report["test_metrics"]["UNI"][scope], report["test_metrics"]["EventC2"][scope]
        lines.append(f"| {label} | samples | {u['num_samples']:,} | {e['num_samples']:,} | |")
        for m in METRICS:
            lines.append(f"| {label} | {m} | {u[m]:.6f} | {e[m]:.6f} | {e[m] - u[m]:+.6f} |")
    o = report["test_overlap"]
    lines += ["", "## Test impression 겹침", "",
              f"- 공통 impression: {o['common_impressions']:,}",
              f"- UNI-only: {o['uni_only_impressions']:,}",
              f"- EventC2-only: {o['eventc2_only_impressions']:,}",
              f"- 공통 중 candidate 구성 동일: {o['common_identical_candidates']:,} "
              f"({(o['common_identical_candidates_ratio'] or 0):.2%})"]
    lines += ["", "## split 크기", "", "| split | UNI rows / impressions | EventC2 rows / impressions |", "|---|---|---|"]
    for split in ("train", "validation_half", "test"):
        u, e = report["splits"]["UNI"][split], report["splits"]["EventC2"][split]
        lines.append(f"| {split} | {u['rows']:,} / {u['impressions']:,} | {e['rows']:,} / {e['impressions']:,} |")
    lines += ["", "## 1pos4neg 단계별 (positive, negative) 쌍 기준", "",
              "| 모델 | split | 쌍 | P-N 제거 | N-N 제거 | drop positive | 출력 rows / impressions |",
              "|---|---|---|---|---|---|---|"]
    for name in ("UNI", "EventC2"):
        for r in report["1pos4neg_stages"][name]:
            if r.get("pair_count_total") is None:
                continue
            lines.append(
                f"| {name} | {r['split_name']} | {r['pair_count_total']:,} | "
                f"{r['pn_same_sid_removed_pair_count']:,} ({r['pn_same_sid_removed_pair_ratio']:.2%}) | "
                f"{r['nn_dedup_removed_pair_count']:,} ({r['nn_dedup_removed_pair_ratio']:.2%}) | "
                f"{r['dropped_positive_count']:,} / {r['positive_count']:,} ({r['dropped_positive_ratio']:.2%}) | "
                f"{r['output_row_count']:,} / {r['output_impression_count']:,} |"
            )
    lines += ["", "## Validation (학습 중 best checkpoint)", "", "```",
              json.dumps(report["validation_best"], ensure_ascii=False, indent=2), "```"]
    (args.out_dir / "uni_vs_eventc2.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
