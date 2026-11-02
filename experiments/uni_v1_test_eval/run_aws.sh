#!/usr/bin/env bash
# UNI V1 baseline(shuffle_check/seed42)과 level weight 설정을 validation half / test에서 다시 평가 (재학습 없음)
#   cd ~/minji/SID_Project && CUDA_VISIBLE_DEVICES=0 bash experiments/uni_v1_test_eval/run_aws.sh 2>&1 | tee data/output/experiments/uni_v1_test_eval.log
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
T="${TRANSFORMER_ROOT:-$HOME/minji/SID_Project-transformer/Transformer}"
RUN="$T/sweep_out/ebnerd_v2/shuffle_check/seed42"
DATA="$T/datasets/ebnerd"
OUT="$REPO_ROOT/data/output/experiments/uni_v1_test_eval"
PYTHON="${PYTHON:-python}"
export PYTHONUNBUFFERED=1

for split in validation_half:validation_sequences_1pos4neg_half test:test_sequences_1pos4neg; do
    name="${split%%:*}"; file="${split##*:}"
    echo "== ${name} 예측 (checkpoint ${RUN}/checkpoint_best.pt)"
    "$PYTHON" "$REPO_ROOT/experiments/eventc2_retrain_v1/predict_uni_baseline.py" \
        --transformer-root "$T" --uni-run "$RUN" --uni-test "$DATA/${file}.parquet" --out-dir "$OUT/$name"
    "$PYTHON" "$REPO_ROOT/experiments/uni_v1_test_eval/score_level_weights.py" \
        --scores "$OUT/$name/test_candidate_scores.parquet" --name "$name"
done
