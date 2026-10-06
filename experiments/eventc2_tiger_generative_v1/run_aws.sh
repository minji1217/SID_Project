#!/usr/bin/env bash
# EventC2 데이터 그대로 TIGER-style generative Transformer 학습 + Test + V1 비교 (single GPU)
#   cd ~/minji/SID_Project && CUDA_VISIBLE_DEVICES=0 nohup bash experiments/eventc2_tiger_generative_v1/run_aws.sh > <로그> 2>&1 &
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP="${EXP_ROOT:-$REPO_ROOT/data/output/experiments/normalize_v2_uni_lu005_m05_eventc2_retrain_v1}"
OUT="$EXP/eventc2_tiger_generative_v1/seed42"
PYTHON="${PYTHON:-python}"
export PYTHONUNBUFFERED=1

for f in "$EXP/candidate_1pos4neg_shuffled/train_sequences_1pos4neg.parquet" \
         "$EXP/transformer_datasets/validation_sequences_1pos4neg_half.parquet" \
         "$EXP/transformer_datasets/test_sequences_1pos4neg.parquet"; do
    [ -f "$f" ] || { echo "없음: $f" >&2; exit 1; }
done

"$PYTHON" "$REPO_ROOT/experiments/eventc2_tiger_generative_v1/train_tiger.py" \
    --train "$EXP/candidate_1pos4neg_shuffled/train_sequences_1pos4neg.parquet" \
    --val "$EXP/transformer_datasets/validation_sequences_1pos4neg_half.parquet" \
    --test "$EXP/transformer_datasets/test_sequences_1pos4neg.parquet" \
    --out-dir "$OUT" --seed 42

V1_SCORES="$EXP/transformer_runs/seed42/test_candidate_scores.parquet"
if [ -f "$V1_SCORES" ]; then
    "$PYTHON" "$REPO_ROOT/experiments/eventc2_tiger_generative_v1/compare_v1_vs_tiger.py" \
        --v1-scores "$V1_SCORES" --tiger-run "$OUT" \
        --out "$EXP/eventc2_tiger_generative_v1/compare_v1_vs_tiger.md"
else
    echo "V1 test score 파일이 없어 비교를 건너뜀: $V1_SCORES"
fi
