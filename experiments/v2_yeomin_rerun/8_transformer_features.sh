#!/usr/bin/env bash
# ⑥ 인기·나이를 함께 학습하는 Transformer (여민 코드 22028ba + 우리 패치 transformer_features.patch)
#   RUN=h SEED=42 CUDA_VISIBLE_DEVICES=0 nohup bash experiments/v2_yeomin_rerun/8_transformer_features.sh > ~/minji/v2_tffeat_h_s42.log 2>&1 &
#
# 후보 점수 = SID 점수 + w1 · candidate_logpop + w2 · candidate_logage   (w 를 같이 학습, 시작값 5, −1 = 재학습 없는 결합에서 고른 값)
# 나머지는 4_transformer.sh 와 같다 (V1 gin, fp32, best = validation 앞 절반 Top-1, test 한 번).
#
#   RUN   v1 | h | r      SEED  42 / 123 / 2026
#   입력  $V2/features/<RUN>/ (add_candidate_features.py 로 만든 파일) + SID 파일
#   코드  ~/minji/yeomin_code_22028ba_features (없으면 여민 코드를 복사해 패치를 적용해서 만든다. 원래 여민 코드 폴더는 건드리지 않음)
#   (선택) FEATURE_INIT="5.0, -1.0"  FEATURE_LR=0.01  NUM_EPOCHS=<n>
#          FEATURE_LR=0 이면 가중치를 시작값(5, −1)에 고정하고 Transformer 만 학습 → 결과 폴더 <RUN>_seed<SEED>_flr0
# 결과: $V2/transformer_feat/<RUN>_seed<SEED>/ (transformer.gin, checkpoint_best.pt, test_candidate_scores.parquet(sid_score 포함), test_metrics.json)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

RUN="${RUN:?RUN=v1, h, r}"; SEED="${SEED:-42}"
PATCH="$CODE/transformer_features.patch"
YMF="$(realpath -m "${YMF_DIR:-$HOME/minji/yeomin_code_${YM_REF:0:7}_features}")"
F="$V2/features/$RUN"
TRAIN="$F/train_sequences_1pos4neg.parquet"; VAL="$F/validation_sequences_1pos4neg_half.parquet"; TEST="$F/test_sequences_1pos4neg.parquet"
case "$RUN" in
    v1) SID="$HOME/minji/uni_post_rqvae/article_semantic_ids.parquet" ;;
    h|r) SID="$V2/sid/$RUN/article_semantic_ids.parquet" ;;
    *) die "RUN 은 v1, h, r" ;;
esac
SID="$(realpath -m "${SID_PATH:-$SID}")"
SUFFIX=""; [[ -n "${NUM_EPOCHS:-}" ]] && SUFFIX="_ep${NUM_EPOCHS}"
# 가중치를 고정(FEATURE_LR=0)하거나 시작값을 바꾸면 결과 폴더를 따로 쓴다 (예: h_seed42_flr0)
[[ -n "${FEATURE_LR:-}" && "${FEATURE_LR}" != "0.01" ]] && SUFFIX="${SUFFIX}_flr${FEATURE_LR}"
[[ -n "${FEATURE_INIT:-}" ]] && SUFFIX="${SUFFIX}_init$(echo "$FEATURE_INIT" | tr -d ' ' | tr ',' '_')"
OUT="$V2/transformer_feat/${RUN}_seed${SEED}${SUFFIX}"

echo "== ⑥ 인기·나이 함께 학습: $RUN · seed $SEED → $OUT"
check_yeomin_code
check_gpu
for f in "$TRAIN" "$VAL" "$TEST" "$SID" "$PATCH"; do [[ -f "$f" ]] || die "없음: $f (특징 파일은 add_candidate_features.py --run $RUN)"; done

# 패치를 적용한 여민 코드 복사본 (처음 한 번 만든다)
if [[ ! -d "$YMF" ]]; then
    git clone -q "$YM" "$YMF"
    git -C "$YMF" checkout -q --detach "$YM_REF"
    git -C "$YMF" apply "$PATCH"
    echo "패치 적용한 코드: $YMF"
fi
[[ "$(git -C "$YMF" rev-parse HEAD)" == "$YM_REF" ]] || die "$YMF 가 $YM_REF 가 아닙니다"
git -C "$YMF" apply --check -R "$PATCH" 2>/dev/null || die "$YMF 에 패치가 그대로 적용된 상태가 아닙니다"
"$PYTHON" "$CODE/check_candidates.py" "$TRAIN" "$VAL" "$TEST"
new_out_dir "$OUT"

"$PYTHON" - "$REPO_ROOT/experiments/transformer_v1_config/transformer_ebnerd_v1.gin" "$OUT/transformer.gin" \
    "$TRAIN" "$VAL" "$OUT" "$SEED" "${NUM_EPOCHS:-}" "${FEATURE_INIT:-5.0, -1.0}" "${FEATURE_LR:-0.01}" <<'EOF'
import re, sys
src, dst, train, val, save, seed, epochs, init, lr = sys.argv[1:10]
t = open(src, encoding="utf-8").read()
rep = {"train.train_path": f'"{train}"', "train.validation_path": f'"{val}"', "train.save_dir": f'"{save}"', "train.seed": seed}
if epochs:
    rep["train.num_epochs"] = epochs
for k, v in rep.items():
    t, n = re.subn(rf"^{re.escape(k)}\s*=.*$", f"{k} = {v}", t, flags=re.M)
    assert n == 1, k
t += f"""
# 후보 특징 (transformer_features.patch)
NewsSequenceDataset.candidate_feature_columns = ["candidate_logpop", "candidate_logage"]
NewsEncoderDecoderTransformer.num_candidate_features = 2
NewsEncoderDecoderTransformer.candidate_feature_init = [{init}]
train.feature_learning_rate = {lr}
"""
open(dst, "w", encoding="utf-8").write(t)
EOF
provenance "$OUT" "run         $RUN (인기·나이 함께 학습)" "seed        $SEED" "code        $YMF + $(sha256sum "$PATCH" | cut -c1-12)" \
    "train       $TRAIN" "validation  $VAL" "test        $TEST" "sid         $SID"

STATUS=0
cd "$YMF/Transformer"
echo "-- 학습 (로그 $OUT/train.log)"
"$PYTHON" train_transformer.py --config "$OUT/transformer.gin" --sid-path "$SID" > "$OUT/train.log" 2>&1 || STATUS=$?
[[ $STATUS -eq 0 ]] || { finish_provenance "$OUT" "$STATUS"; tail -30 "$OUT/train.log"; die "학습 실패"; }
grep -E "Feature weights" "$OUT/train.log" | tail -n 3 || true

echo "-- test 예측"
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
