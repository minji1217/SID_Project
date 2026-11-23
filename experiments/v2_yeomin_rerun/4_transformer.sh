#!/usr/bin/env bash
# ⑤ Transformer 학습 → test 예측 → 지표 (여민 코드 + 우리 V1 gin, fp32)
#   RUN=h SEED=42 CUDA_VISIBLE_DEVICES=0 nohup bash experiments/v2_yeomin_rerun/4_transformer.sh > ~/minji/v2_tf_h_s42.log 2>&1 &
#
#   RUN   v1 (예전 UNI SID, 공유 폴더 1pos4neg = 비교 기준) | h | r
#   SEED  42 / 123 / 2026
#   (테스트용) TRAIN_PATH / VAL_PATH / TEST_PATH / SID_PATH 로 입력을 바꿀 수 있음. NUM_EPOCHS 로 epoch 수 변경
# 결과: $V2/transformer/<RUN>_seed<SEED>/ (transformer.gin, checkpoint_best.pt, test_candidate_scores.parquet, test_metrics.json, 로그)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

RUN="${RUN:?RUN=v1, h, r}"; SEED="${SEED:-42}"
case "$RUN" in
    v1) SHARED=/home/ubuntu/shared/datasets/ebnerd
        TRAIN="$SHARED/train_sequences_1pos4neg.parquet"
        VAL="$SHARED/validation_sequences_1pos4neg_half.parquet"
        TEST="$SHARED/test_sequences_1pos4neg.parquet"
        SID="$HOME/minji/uni_post_rqvae/article_semantic_ids.parquet" ;;
    h|r) E="$PRE/experiments/${EXP_PREFIX}${RUN}"
        TRAIN="$E/candidate_1pos4neg_shuffled/train_sequences_1pos4neg.parquet"
        VAL="$E/split/validation_sequences_1pos4neg_half.parquet"
        TEST="$E/split/test_sequences_1pos4neg.parquet"
        SID="$V2/sid/$RUN/article_semantic_ids.parquet" ;;
    *) die "RUN 은 v1, h, r" ;;
esac
TRAIN="$(realpath -m "${TRAIN_PATH:-$TRAIN}")"; VAL="$(realpath -m "${VAL_PATH:-$VAL}")"
TEST="$(realpath -m "${TEST_PATH:-$TEST}")"; SID="$(realpath -m "${SID_PATH:-$SID}")"
SUFFIX=""; [[ -n "${NUM_EPOCHS:-}" ]] && SUFFIX="_ep${NUM_EPOCHS}"
OUT="$V2/transformer/${RUN}_seed${SEED}${SUFFIX}"

echo "== ⑤ Transformer: $RUN · seed $SEED → $OUT"
check_yeomin_code; check_gpu
for f in "$TRAIN" "$VAL" "$TEST" "$SID"; do [[ -f "$f" ]] || die "없음: $f"; done
"$PYTHON" "$CODE/check_candidates.py" "$TRAIN" "$VAL" "$TEST"
new_out_dir "$OUT"

# 우리 V1 gin 에 경로·seed 만 바꿔 넣는다 (여민 코드는 ~ 를 펼치지 않으므로 전부 절대 경로)
"$PYTHON" - "$REPO_ROOT/experiments/transformer_v1_config/transformer_ebnerd_v1.gin" "$OUT/transformer.gin" \
    "$TRAIN" "$VAL" "$OUT" "$SEED" "${NUM_EPOCHS:-}" <<'EOF'
import re, sys
src, dst, train, val, save, seed, epochs = sys.argv[1:8]
t = open(src, encoding="utf-8").read()
rep = {"train.train_path": f'"{train}"', "train.validation_path": f'"{val}"', "train.save_dir": f'"{save}"', "train.seed": seed}
if epochs:
    rep["train.num_epochs"] = epochs
for k, v in rep.items():
    t, n = re.subn(rf"^{re.escape(k)}\s*=.*$", f"{k} = {v}", t, flags=re.M)
    assert n == 1, k
open(dst, "w", encoding="utf-8").write(t)
EOF
provenance "$OUT" "run         $RUN" "seed        $SEED" "train       $TRAIN" "validation  $VAL" "test        $TEST" \
    "sid         $SID ($(sha256sum "$SID" | cut -c1-12))"

STATUS=0
cd "$YM/Transformer"
echo "-- 학습 (로그 $OUT/train.log)"
"$PYTHON" train_transformer.py --config "$OUT/transformer.gin" --sid-path "$SID" > "$OUT/train.log" 2>&1 || STATUS=$?
[[ $STATUS -eq 0 ]] || { finish_provenance "$OUT" "$STATUS"; tail -30 "$OUT/train.log"; die "학습 실패"; }
grep -E "vocab size" "$OUT/train.log" || true

echo "-- test 예측 (같은 gin, best checkpoint)"
"$PYTHON" predict_sid.py --config "$OUT/transformer.gin" --checkpoint "$OUT/checkpoint_best.pt" \
    --test_path "$TEST" --output_path "$OUT/test_candidate_scores.parquet" > "$OUT/predict.log" 2>&1 || STATUS=$?
[[ $STATUS -eq 0 ]] || { finish_provenance "$OUT" "$STATUS"; tail -30 "$OUT/predict.log"; die "예측 실패"; }

echo "-- 지표"
"$PYTHON" evaluate/metrics.py --prediction_path "$OUT/test_candidate_scores.parquet" \
    --output_path "$OUT/test_metrics.json" > "$OUT/metrics.log" 2>&1 || STATUS=$?
finish_provenance "$OUT" "$STATUS"
[[ $STATUS -eq 0 ]] || { tail -30 "$OUT/metrics.log"; die "지표 계산 실패"; }
cat "$OUT/test_metrics.json"
echo "== 완료: $OUT"
