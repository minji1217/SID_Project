"""UNI baseline run에 test_candidate_scores.parquet이 없을 때만 쓰는 추론 전용 스크립트.

학습하지 않는다. baseline run 폴더에는 아무것도 쓰지 않는다.
baseline run_summary.json의 gin_config를 그대로 config로 써서 같은 모델 구조로 checkpoint_best.pt를 읽고,
같은 test 파일을 예측한 뒤 metrics.py 결과가 baseline test_metrics.json과 같은지 확인한다.

    python experiments/eventc2_retrain_v1/predict_uni_baseline.py \
        --transformer-root <Transformer> --uni-run <UNI run> --uni-test <UNI test parquet> \
        --out-dir <EXP>/reports/uni_baseline_inference
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--transformer-root", type=Path, required=True)
    parser.add_argument("--uni-run", type=Path, required=True)
    parser.add_argument("--uni-test", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    scores = out_dir / "test_candidate_scores.parquet"
    metrics = out_dir / "test_metrics.json"
    if metrics.exists():
        print(f"이미 존재: {metrics}")
        return
    if out_dir.resolve() == args.uni_run.resolve():
        raise ValueError("baseline run 폴더에는 쓰지 않습니다.")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = json.loads((args.uni_run / "run_summary.json").read_text(encoding="utf-8"))
    config_path = out_dir / "uni_run_config.gin"
    config_path.write_text(summary["gin_config"], encoding="utf-8")

    root = args.transformer_root.resolve()
    with open(out_dir / "predict.log", "w", encoding="utf-8") as log:
        subprocess.run(
            [sys.executable, "-u", "predict_sid.py", "--config", str(config_path),
             "--checkpoint", str((args.uni_run / "checkpoint_best.pt").resolve()),
             "--test_path", str(args.uni_test.resolve()), "--output_path", str(scores),
             "--batch_size", "128", "--num_workers", str(args.num_workers)],
            cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True,
        )
    subprocess.run(
        [sys.executable, "-u", "evaluate/metrics.py", "--prediction_path", str(scores),
         "--output_path", str(metrics)],
        cwd=root, check=True,
    )

    baseline_metrics = args.uni_run / "test_metrics.json"
    if baseline_metrics.exists():
        a = json.loads(baseline_metrics.read_text(encoding="utf-8"))
        b = json.loads(metrics.read_text(encoding="utf-8"))
        diff = {k: (a[k], b[k]) for k in ("top1_accuracy", "mrr", "ndcg@5", "auc") if abs(a[k] - b[k]) > 1e-6}
        print("baseline test_metrics.json과 일치" if not diff else f"WARNING: baseline과 다름 {diff}")


if __name__ == "__main__":
    main()
