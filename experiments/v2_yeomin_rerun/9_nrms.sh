#!/usr/bin/env bash
# 오리지널 NRMS (Microsoft Recommenders 공개 코드, 모델 수정 없음) 를 우리 1pos4neg 로: 변환 → 그리드서치 → seed 3개.
#   RUN=h CUDA_VISIBLE_DEVICES=0 nohup bash experiments/v2_yeomin_rerun/9_nrms.sh > ~/minji/v2_nrms_h.log 2>&1 &
#
# 준비 (한 번)
#   1) TensorFlow 2.15 환경 (공개 코드가 TF < 2.16 필요, 지금 PyTorch 환경과 따로):
#      python3.10 -m venv ~/minji/nrms_env
#      ~/minji/nrms_env/bin/pip install "tensorflow[and-cuda]==2.15.*" "git+https://github.com/recommenders-team/recommenders@cc47866b0599721f8544d1cc850a5a1db091acb3" polars pyarrow pandas
#   2) 덴마크어 fastText:  mkdir -p ~/minji/fasttext && wget -c -P ~/minji/fasttext https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.da.300.vec.gz
#
#   RUN   h(기본) | r | v1   어느 실험의 1pos4neg 행을 쓸지 (NRMS 는 SID 를 안 쓴다)
#   NRMS_PY  기본 ~/minji/nrms_env/bin/python     FT  기본 ~/minji/fasttext/cc.da.300.vec.gz
#   PARALLEL 동시 실행 수 (기본 2)                 (테스트용) LIMIT=<행 수>, EPOCHS=<n>
# 결과: $V2/nrms/mind_<RUN>/ (변환), $V2/nrms/<RUN>/grid/…, final/…, summary.csv, summary.json
# NRMS + 인기 + 나이 (끝난 뒤): python experiments/v2_yeomin_rerun/popularity_window.py --runs h \
#     --pred-template '{v2}/nrms/{run}/final/best_seed{seed}/{split}_candidate_scores.parquet' --out-name nrms_window
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

RUN="${RUN:-h}"; NRMS_PY="${NRMS_PY:-$HOME/minji/nrms_env/bin/python}"; FT="${FT:-$HOME/minji/fasttext/cc.da.300.vec.gz}"
PARALLEL="${PARALLEL:-2}"
case "$RUN" in
    v1) D=/home/ubuntu/shared/datasets/ebnerd
        TRAIN="$D/train_sequences_1pos4neg.parquet"; VAL="$D/validation_sequences_1pos4neg_half.parquet"; TEST="$D/test_sequences_1pos4neg.parquet" ;;
    h|r) E="$PRE/experiments/${EXP_PREFIX}${RUN}"
        TRAIN="$E/candidate_1pos4neg_shuffled/train_sequences_1pos4neg.parquet"
        VAL="$E/split/validation_sequences_1pos4neg_half.parquet"; TEST="$E/split/test_sequences_1pos4neg.parquet" ;;
    *) die "RUN 은 h, r, v1" ;;
esac
T="$REPO_ROOT/experiments/nrms_recommenders_v1"
DATA="$V2/nrms/mind_${RUN}${LIMIT:+_limit$LIMIT}"
ROOT="$V2/nrms/${RUN}${LIMIT:+_limit$LIMIT}"

echo "== NRMS (Microsoft Recommenders): 행 $RUN → $ROOT"
[[ -x "$NRMS_PY" ]] || die "TensorFlow 환경이 없습니다: $NRMS_PY (위 준비 1)"
"$NRMS_PY" -c "import tensorflow as tf, sys; g=tf.config.list_physical_devices('GPU'); print('TF', tf.__version__, 'GPU', g); sys.exit(0 if g or '${ALLOW_CPU:-0}'=='1' else 1)" \
    || die "TensorFlow 가 GPU 를 못 찾았습니다"
for f in "$TRAIN" "$VAL" "$TEST" "$REPO_ROOT/data/raw/articles.parquet" "$FT"; do [[ -f "$f" ]] || die "없음: $f"; done

if [[ ! -f "$DATA/vocab_stats.json" ]]; then
    [[ -e "$DATA" ]] && die "$DATA 가 끝나지 않은 채 있습니다 (지우고 다시)"
    echo "-- MIND 형식으로 변환"
    "$NRMS_PY" "$T/prepare_mind.py" --train "$TRAIN" --validation "$VAL" --test "$TEST" \
        --articles "$REPO_ROOT/data/raw/articles.parquet" --fasttext "$FT" --out "$DATA" ${LIMIT:+--limit "$LIMIT"}
else
    echo "-- 변환 결과 있음: $DATA"
fi
mkdir -p "$ROOT"
provenance "$ROOT" "run         $RUN (NRMS, Microsoft Recommenders @cc47866)" "data        $DATA" "train       $TRAIN" "validation  $VAL" "test        $TEST" "fasttext    $FT"
echo "-- 그리드서치 (seed 42) → 최종 seed 123 · 2026"
"$NRMS_PY" "$T/grid_nrms.py" --data "$DATA" --root "$ROOT" --python "$NRMS_PY" --parallel "$PARALLEL" ${EPOCHS:+--epochs "$EPOCHS"}
echo "== 완료: $ROOT/summary.csv"
