#!/usr/bin/env bash
# validation 앞 절반 예측 (인기 결합의 λ 를 고를 때 씀). 학습한 best checkpoint 로 점수만 계산한다.
#   RUN=h SEED=42 CUDA_VISIBLE_DEVICES=0 bash experiments/v2_yeomin_rerun/5_predict_val.sh
#
#   RUN   v1 | h | r      SEED  42 / 123 / 2026
#   (테스트용) RUN_DIR, VAL_PATH 로 경로를 바꿀 수 있음
# 결과: $V2/transformer/<RUN>_seed<SEED>/val_candidate_scores.parquet
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

RUN="${RUN:?RUN=v1, h, r}"; SEED="${SEED:-42}"
case "$RUN" in
    v1) VAL=/home/ubuntu/shared/datasets/ebnerd/validation_sequences_1pos4neg_half.parquet ;;
    h|r) VAL="$PRE/experiments/${EXP_PREFIX}${RUN}/split/validation_sequences_1pos4neg_half.parquet" ;;
    *) die "RUN 은 v1, h, r" ;;
esac
VAL="$(realpath -m "${VAL_PATH:-$VAL}")"
OUT="$(realpath -m "${RUN_DIR:-$V2/transformer/${RUN}_seed${SEED}}")"
DST="$OUT/val_candidate_scores.parquet"

echo "== validation 앞 절반 예측: $RUN · seed $SEED → $DST"
check_yeomin_code
for f in "$VAL" "$OUT/transformer.gin" "$OUT/checkpoint_best.pt"; do [[ -f "$f" ]] || die "없음: $f"; done
[[ -e "$DST" ]] && die "$DST 가 이미 있습니다"

(cd "$YM/Transformer" && "$PYTHON" predict_sid.py --config "$OUT/transformer.gin" --checkpoint "$OUT/checkpoint_best.pt" \
    --test_path "$VAL" --output_path "$DST") > "$OUT/predict_val.log" 2>&1 || { tail -20 "$OUT/predict_val.log"; die "예측 실패"; }
echo "validation_pred $VAL → $DST ($(date -u +%Y-%m-%dT%H:%M:%SZ))" >> "$OUT/provenance.txt"
echo "== 완료: $DST"
