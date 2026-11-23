#!/usr/bin/env bash
# ① RQ-VAE 학습 (여민 코드 + 우리 UNI gin). 실험 H 또는 R, seed 하나.
#   VARIANT=h SEED=42 CUDA_VISIBLE_DEVICES=0 nohup bash experiments/v2_yeomin_rerun/1_train_rqvae.sh > ~/minji/v2_rqvae_h_s42.log 2>&1 &
#
#   VARIANT  h (Q2 초기화 = 사건별 h 평균) | r (사건별 r1 평균)
#   SEED     42 / 123 / 2026
#   A_CKPT   (선택) 예전 UNI checkpoint. 주면 gin 설정이 같은지 먼저 비교하고 다르면 멈춤
#   SMOKE=1  1 epoch 만 (연결 확인용)
# 결과: $V2/rqvae/<VARIANT>_s<SEED>/checkpoint_best_rec.pt
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

VARIANT="${VARIANT:?VARIANT=h 또는 r}"; SEED="${SEED:-42}"
case "$VARIANT" in
    h) SCRIPT=train_rqvae_hver.py ;;
    r) SCRIPT=train_rqvae_r1ver.py ;;
    *) die "VARIANT 는 h 또는 r" ;;
esac
SUFFIX=""; [[ "${SMOKE:-0}" == "1" ]] && SUFFIX="_smoke"
OUT="$V2/rqvae/${VARIANT}_s${SEED}${SUFFIX}"

echo "== ① RQ-VAE 학습: 실험 $VARIANT · seed $SEED → $OUT"
check_yeomin_code; check_inputs; check_gpu
new_out_dir "$OUT"

git -C "$REPO_ROOT" show "$UNI_REF:RQVAE/configs/rqvae_ebnerd_wd001_uni.gin" > "$OUT/uni_src.gin" \
    || die "UNI gin 원본을 못 꺼냈습니다 (0_setup.sh 먼저)"
EXTRA=()
if [[ "${SMOKE:-0}" == "1" ]]; then
    EXTRA=(--extra train.epochs=1 train.eval_every_epochs=1 train.early_stopping=False train.save_model_every_epochs=1000)
    [[ "${ALLOW_CPU:-0}" == "1" ]] && EXTRA+=(train.amp=False)
fi
"$PYTHON" "$REPO_ROOT/experiments/rqvae_uni_fixedcluster_v1/make_gin.py" --src "$OUT/uni_src.gin" \
    --out "$OUT/rqvae.gin" --dataset_folder "$INPUT" --save_dir "$OUT" --seed "$SEED" "${EXTRA[@]}"

provenance "$OUT" "variant     $VARIANT ($SCRIPT)" "seed        $SEED"
(cd "$INPUT" && sha256sum article_master.parquet validation_article_master.parquet article_embeddings.npy event_master.parquet) > "$OUT/inputs.sha256"

if [[ -n "${A_CKPT:-}" && "${SMOKE:-0}" != "1" ]]; then
    echo "-- 예전 UNI checkpoint 설정과 비교"
    "$PYTHON" "$REPO_ROOT/experiments/rqvae_uni_fixedcluster_v1/check_against_a.py" "$A_CKPT" "$OUT/rqvae.gin" "$YM/RQVAE" \
        | tee "$OUT/check_against_a.txt" || die "예전 UNI 와 설정이 다릅니다"
fi

echo "-- 학습 (로그 $OUT/train.log)"
STATUS=0
(cd "$YM/RQVAE" && "$PYTHON" "$SCRIPT" "$OUT/rqvae.gin") > "$OUT/train.log" 2>&1 || STATUS=$?
if [[ "$VARIANT" == "r" ]] && ! grep -q "Encoding train r1 for Q2 initialization" "$OUT/train.log"; then
    echo "r1 초기화 로그가 없습니다"; [[ $STATUS -ne 0 ]] || STATUS=1
fi
finish_provenance "$OUT" "$STATUS"
"$PYTHON" "$REPO_ROOT/experiments/rqvae_uni_fixedcluster_v1/parse_log.py" "$OUT" || true
[[ $STATUS -eq 0 ]] || { tail -30 "$OUT/train.log"; die "학습 실패 (exit $STATUS)"; }
ls "$OUT" | grep -E "checkpoint_best_rec" >/dev/null || die "checkpoint_best_rec.pt 가 없습니다"
echo "== 완료: $OUT/checkpoint_best_rec.pt"
