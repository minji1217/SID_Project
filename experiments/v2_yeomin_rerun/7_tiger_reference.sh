#!/usr/bin/env bash
# 공개 TIGER 재현 모델(수정 없음)을 v2 SID 로 학습 + test (비교용 기준선)
#   RUN=h SEED=42 CUDA_VISIBLE_DEVICES=0 nohup bash experiments/v2_yeomin_rerun/7_tiger_reference.sh > ~/minji/v2_tiger_h_s42.log 2>&1 &
#
# 모델·학습·채점: experiments/eventc2_tiger_reference_v1/run_reference.py 그대로
#   (EdoardoBotta/RQ-VAE-Recommender@957d32b modules/model.py 원본. d384·6 head·4층, AdamW 1e-3, batch 640, history 20)
#   학습 = 1pos4neg 이전 train 시퀀스의 정답 기사 SID 를 생성(level별 CE). 채점 = 1pos4neg 후보 5개를 log P(c1..c3) 합으로.
#   2026-11-03 예전 UNI SID 로 돌린 결과: test Top-1 17.36% / AUC 0.487 (무작위보다 낮음, 정답 대부분이 새 기사라서)
#
#   RUN   h | r | v1(예전 UNI SID, 공유 폴더 1pos4neg)       SEED 42 기본
#   (테스트용) LIMIT=<행 수>, EPOCHS=<n>
# 결과: $V2/tiger/<RUN>_seed<SEED>/ (test_metrics.json, test_candidate_scores.parquet, epoch_history.csv, run_summary.json),
#       같은 폴더 diagnose_seen_in_train.txt (정답·오답이 train 에서 클릭된 적 있는지)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

RUN="${RUN:?RUN=h, r, v1}"; SEED="${SEED:-42}"
case "$RUN" in
    v1) TRAIN="$HOME/minji/uni_post_rqvae/train_sequences.parquet"
        VAL=/home/ubuntu/shared/datasets/ebnerd/validation_sequences_1pos4neg_half.parquet
        TEST=/home/ubuntu/shared/datasets/ebnerd/test_sequences_1pos4neg.parquet ;;
    h|r) E="$PRE/experiments/${EXP_PREFIX}${RUN}"
        TRAIN="$E/post_rqvae/train_sequences.parquet"
        VAL="$E/split/validation_sequences_1pos4neg_half.parquet"
        TEST="$E/split/test_sequences_1pos4neg.parquet" ;;
    *) die "RUN 은 h, r, v1" ;;
esac
OUT="$V2/tiger/${RUN}_seed${SEED}"
T="$REPO_ROOT/experiments"
export PYTHONPATH="$T/uni_tiger_reference_v1/reference${PYTHONPATH:+:$PYTHONPATH}"   # 공개 구현 data/schemas.py

echo "== 공개 TIGER: $RUN · seed $SEED → $OUT"
check_gpu
for f in "$TRAIN" "$VAL" "$TEST"; do [[ -f "$f" ]] || die "없음: $f"; done
new_out_dir "$OUT"
provenance "$OUT" "run         $RUN (공개 TIGER, run_reference.py)" "seed        $SEED" "train       $TRAIN" "validation  $VAL" "test        $TEST"

echo "-- SID 일치 확인 (train / validation / test 에서 같은 기사는 같은 SID)"
"$PYTHON" "$T/uni_tiger_reference_v1/check_uni_sid.py" --seq "$TRAIN" --seq "$VAL" --seq "$TEST" | tee "$OUT/check_sid.txt"

EXTRA=()
[[ -n "${LIMIT:-}" ]] && EXTRA+=(--limit "$LIMIT")
[[ -n "${EPOCHS:-}" ]] && EXTRA+=(--epochs "$EPOCHS")
echo "-- 학습 + test"
STATUS=0
"$PYTHON" "$T/eventc2_tiger_reference_v1/run_reference.py" --train "$TRAIN" --val "$VAL" --test "$TEST" \
    --out-dir "$OUT" --seed "$SEED" "${EXTRA[@]}" || STATUS=$?
[[ $STATUS -eq 0 ]] || { finish_provenance "$OUT" "$STATUS"; die "학습 실패"; }

echo "-- 정답·오답이 train 에서 클릭된 적 있는지"
"$PYTHON" "$T/uni_tiger_reference_v1/diagnose_seen_in_train.py" --train "$TRAIN" --eval "$VAL" --eval "$TEST" \
    | tee "$OUT/diagnose_seen_in_train.txt" || true
finish_provenance "$OUT" 0
cat "$OUT/test_metrics.json"
echo "== 완료: $OUT"
