#!/usr/bin/env bash
# 공개 TIGER 재현 모델(수정 없음)을 UNI SID로 학습 + Test + V1 비교 + 진단 (single GPU, seed 42)
#
#   cd ~/minji/SID_Project
#   UNI_TRAIN=<1P4N 이전 UNI train_sequences.parquet> CUDA_VISIBLE_DEVICES=0 \
#     nohup bash experiments/uni_tiger_reference_v1/run_aws.sh > data/output/experiments/uni_tiger_reference_v1.log 2>&1 &
#
# 모델·학습·채점 코드는 eventc2_tiger_reference_v1/run_reference.py 그대로 (경로만 UNI).
# 입력은 읽기만 하고, 출력 폴더가 이미 있으면 시작하지 않는다.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HERE="$REPO_ROOT/experiments/uni_tiger_reference_v1"
SHARED="${SHARED_DIR:-$HOME/shared/datasets/ebnerd}"
VAL="${UNI_VAL:-$SHARED/validation_sequences_1pos4neg_half.parquet}"
TEST="${UNI_TEST:-$SHARED/test_sequences_1pos4neg.parquet}"
V1_SCORES="${V1_SCORES:-$HOME/minji/SID_Project/data/output/experiments/uni_v1_test_eval/test/test_candidate_scores.parquet}"
V1_RUN="${V1_RUN:-$HOME/minji/SID_Project-transformer/Transformer/sweep_out/ebnerd_v2/shuffle_check/seed42}"
SEED="${SEED:-42}"
EXP_DIR="${EXP_DIR:-$REPO_ROOT/data/output/experiments/uni_tiger_reference_v1}"
OUT="$EXP_DIR/seed$SEED"
PYTHON="${PYTHON:-python}"
export PYTHONUNBUFFERED=1
# 공개 구현 data/schemas.py는 .gitignore(data) 때문에 EventC2 폴더에 커밋되지 않았다.
# EventC2 폴더는 고치지 않고, 이 폴더의 원본 사본(byte 동일)을 import 경로에 추가한다.
export PYTHONPATH="$HERE/reference${PYTHONPATH:+:$PYTHONPATH}"

[ -n "${UNI_TRAIN:-}" ] || { echo "UNI_TRAIN(1P4N 이전 UNI train_sequences.parquet)을 지정하세요" >&2; exit 1; }
for f in "$UNI_TRAIN" "$VAL" "$TEST" "$V1_SCORES" "$V1_RUN/run_summary.json"; do
    [ -f "$f" ] || { echo "없음: $f" >&2; exit 1; }
done
[ -e "$OUT" ] && { echo "출력 폴더가 이미 있습니다. 덮어쓰지 않습니다: $OUT" >&2; exit 1; }
case "$(realpath -m "$OUT")" in "$HOME/shared"*|"$HOME/yeomin"*|"$HOME/minkyo"*) echo "쓰면 안 되는 위치: $OUT" >&2; exit 1 ;; esac
mkdir -p "$EXP_DIR"

echo "== [1/4] SID 일치 확인 (train / validation half / test) $(date)"
"$PYTHON" "$HERE/check_uni_sid.py" --seq "$UNI_TRAIN" --seq "$VAL" --seq "$TEST"

echo "== [2/4] 공개 TIGER 모델 학습 + Test $(date)"
"$PYTHON" "$REPO_ROOT/experiments/eventc2_tiger_reference_v1/run_reference.py" \
    --train "$UNI_TRAIN" --val "$VAL" --test "$TEST" --out-dir "$OUT" --seed "$SEED"

echo "== [3/4] V1과 비교 (같은 test sample인지 먼저 확인) $(date)"
CMP="$EXP_DIR/compare_v1_vs_reference_seed$SEED.md"
"$PYTHON" "$REPO_ROOT/experiments/eventc2_tiger_generative_v1/compare_v1_vs_tiger.py" \
    --v1-scores "$V1_SCORES" --v1-run "$V1_RUN" --tiger-run "$OUT" --out "$CMP"
sed -i '1s/^# EventC2 test: V1 candidate ranker vs TIGER-style generative/# UNI test: V1 candidate ranker vs 공개 TIGER (reference)/' "$CMP"

echo "== [4/4] 정답·오답이 train에서 클릭된 적 있는지 $(date)"
"$PYTHON" "$HERE/diagnose_seen_in_train.py" --train "$UNI_TRAIN" --eval "$VAL" --eval "$TEST" \
    | tee "$EXP_DIR/diagnose_seen_in_train.txt"
echo "끝 $(date) -> $EXP_DIR"
