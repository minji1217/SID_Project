#!/usr/bin/env bash
# 공개 TIGER 재현 모델(수정 없음)을 EventC2 데이터로 학습 + Test + V1 비교 (single GPU)
#   cd ~/minji/SID_Project && CUDA_VISIBLE_DEVICES=0 nohup bash experiments/eventc2_tiger_reference_v1/run_aws.sh > <로그> 2>&1 &
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP="${EXP_ROOT:-$REPO_ROOT/data/output/experiments/normalize_v2_uni_lu005_m05_eventc2_retrain_v1}"
OUT="$EXP/eventc2_tiger_reference_v1/seed42"
PYTHON="${PYTHON:-python}"
export PYTHONUNBUFFERED=1

"$PYTHON" "$REPO_ROOT/experiments/eventc2_tiger_reference_v1/run_reference.py" \
    --train "$EXP/post_rqvae/train_sequences.parquet" \
    --val "$EXP/transformer_datasets/validation_sequences_1pos4neg_half.parquet" \
    --test "$EXP/transformer_datasets/test_sequences_1pos4neg.parquet" \
    --out-dir "$OUT" --seed 42

"$PYTHON" "$REPO_ROOT/experiments/eventc2_tiger_generative_v1/compare_v1_vs_tiger.py" \
    --v1-scores "$EXP/transformer_runs/seed42/test_candidate_scores.parquet" \
    --v1-run "$EXP/transformer_runs/seed42" --tiger-run "$OUT" \
    --out "$EXP/eventc2_tiger_reference_v1/compare_v1_vs_reference.md"
