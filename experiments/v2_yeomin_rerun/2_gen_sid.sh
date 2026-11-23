#!/usr/bin/env bash
# ② SID 생성 (여민 코드). seed 42 RQ-VAE 모델로 한 번.
#   VARIANT=h CUDA_VISIBLE_DEVICES=0 bash experiments/v2_yeomin_rerun/2_gen_sid.sh > ~/minji/v2_sid_h.log 2>&1
#
#   VARIANT  h (hver) | r (r1ver)
#   RQ_SEED  SID 를 만들 RQ-VAE seed (기본 42, 미리 정한 기준)
# 결과: $V2/sid/<VARIANT>/article_semantic_ids.parquet (+ 매칭 결과, 요약)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

VARIANT="${VARIANT:?VARIANT=h 또는 r}"; RQ_SEED="${RQ_SEED:-42}"
case "$VARIANT" in
    h) SCRIPT=generate_semantic_ids_timeupdate_hver.py ;;
    r) SCRIPT=generate_semantic_ids_timeupdate_r1ver.py ;;
    *) die "VARIANT 는 h 또는 r" ;;
esac
CKPT="$V2/rqvae/${VARIANT}_s${RQ_SEED}/checkpoint_best_rec.pt"
OUT="$V2/sid/${VARIANT}"

echo "== ② SID 생성: 실험 $VARIANT · checkpoint $CKPT → $OUT"
check_yeomin_code; check_inputs
[[ -f "$CKPT" ]] || die "checkpoint 없음: $CKPT (① 먼저)"
new_out_dir "$OUT"
provenance "$OUT" "variant     $VARIANT ($SCRIPT)" "checkpoint  $CKPT ($(sha256sum "$CKPT" | cut -c1-12))"

STATUS=0
(cd "$YM/RQVAE" && "$PYTHON" "$SCRIPT" --data_dir "$INPUT" --checkpoint "$CKPT" --output_dir "$OUT") \
    > "$OUT/sid.log" 2>&1 || STATUS=$?
finish_provenance "$OUT" "$STATUS"
[[ $STATUS -eq 0 ]] || { tail -30 "$OUT/sid.log"; die "SID 생성 실패 (exit $STATUS)"; }

# R 일 때: R1 코드를 H checkpoint 로 한 번 더 돌려서, 사건 배정이 H 와 100% 같은지 본다 (코사인이 h 인지 코드 확인)
HCKPT="$V2/rqvae/h_s${RQ_SEED}/checkpoint_best_rec.pt"
CHK="$V2/sid/_check_r1code_with_h_ckpt"
if [[ "$VARIANT" == "r" && -f "$HCKPT" && -f "$V2/sid/h/article_semantic_ids.parquet" && ! -e "$CHK" ]]; then
    echo "-- 코드 확인: R1 코드 + H checkpoint → $CHK"
    mkdir -p "$CHK"
    (cd "$YM/RQVAE" && "$PYTHON" "$SCRIPT" --data_dir "$INPUT" --checkpoint "$HCKPT" --output_dir "$CHK") \
        > "$CHK/sid.log" 2>&1 || die "코드 확인용 SID 생성 실패 ($CHK/sid.log)"
fi

"$PYTHON" "$CODE/check_sid.py" "$V2/sid" | tee "$OUT/check_sid.txt"
[[ ${PIPESTATUS[0]} -eq 0 ]] || die "SID 확인 실패"
echo "== 완료: $OUT/article_semantic_ids.parquet"
