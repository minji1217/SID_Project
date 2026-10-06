#!/usr/bin/env bash
# ============================================================
# eventc2_train_v1: Train c2를 event-level로 고정한 SID로 downstream 전체 재생성
#
#   기존 final RQ-VAE checkpoint (재학습 없음)
#     -> generate_semantic_ids.py --train_c2_policy event
#     -> post-rqvae (전체 train/validation sequences)
#     -> 1pos4neg sampling (seed 42, c1c2c3, negative 4)
#     -> candidate order shuffle (seed 42)
#     -> validation 50:50 split (validation half / test)
#     -> Transformer V1 최종 config 학습 (seed 42) + Test 평가
#     -> A/B 비교 리포트
#
# 모든 출력은 ${SID_OUTPUT_DIR:-data/output}/experiments/$EXP 아래에만 쓴다.
# 각 단계는 결과가 이미 있으면 건너뛰고, 어떤 파일도 덮어쓰지 않는다.
#
# 필수 환경변수
#   RQVAE_DATA_DIR   : A의 SID를 만들 때 쓴 --data_dir
#                      (article_master / validation_article_master / article_embeddings.npy)
#   CKPT             : A의 SID를 만들 때 쓴 final checkpoint (.pt)
#   TRANSFORMER_ROOT : sid_project-transformer/Transformer 폴더 (claude/hopeful-mendel-fhc2n4 브랜치)
#
# 선택 환경변수
#   EVENT_REPR       : mean_h (B, 교수님 설계, 기본) | mean_r1 (B-r1, 원인 분리 진단용)
#   EXP              : 실험 이름 (기본 B: normalize_v2_uni_lu005_m05_eventc2_train_v1,
#                                   B-r1: normalize_v2_uni_lu005_m05_eventc2r1_train_v1)
#   SID_OUTPUT_DIR   : A의 post-rqvae를 돌릴 때와 같은 값 (src/config.py의 OUTPUT_DIR)
#   A_SID_DIR        : A의 generate_semantic_ids --output_dir (주면 SID A/B 리포트 생성)
#   A_RUN_DIR        : A의 Transformer seed42 run 폴더 (주면 Transformer A/B 리포트 생성)
#   SEEDS            : 기본 "42"
#   NUM_WORKERS      : 기본 4
#
# 실행 (SID_Project 레포 루트에서)
#   RQVAE_DATA_DIR=... CKPT=... TRANSFORMER_ROOT=... \
#   A_SID_DIR=... A_RUN_DIR=... \
#   bash experiments/eventc2_train_v1/run_pipeline.sh
# ============================================================

set -euo pipefail

: "${RQVAE_DATA_DIR:?RQVAE_DATA_DIR를 지정하세요}"
: "${CKPT:?CKPT를 지정하세요}"
: "${TRANSFORMER_ROOT:?TRANSFORMER_ROOT를 지정하세요}"

EVENT_REPR="${EVENT_REPR:-mean_h}"
case "$EVENT_REPR" in
    mean_h)  DEFAULT_EXP="normalize_v2_uni_lu005_m05_eventc2_train_v1" ;;
    mean_r1) DEFAULT_EXP="normalize_v2_uni_lu005_m05_eventc2r1_train_v1" ;;
    *) echo "EVENT_REPR는 mean_h 또는 mean_r1이어야 합니다: $EVENT_REPR" >&2; exit 1 ;;
esac
EXP="${EXP:-$DEFAULT_EXP}"
SEEDS="${SEEDS:-42}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PYTHON="${PYTHON:-python}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HERE="$REPO_ROOT/experiments/eventc2_train_v1"
OUTPUT_ROOT="$(cd "$REPO_ROOT" && realpath -m "${SID_OUTPUT_DIR:-data/output}")"
EXP_ROOT="$OUTPUT_ROOT/experiments/$EXP"

SID_DIR="$EXP_ROOT/semantic_ids"
POST_DIR="$EXP_ROOT/post_rqvae"
NEG_DIR="$EXP_ROOT/candidate_1pos4neg"
SHUF_DIR="$EXP_ROOT/candidate_1pos4neg_shuffled"
DATASET_DIR="$EXP_ROOT/transformer_datasets"
RUNS_DIR="$EXP_ROOT/transformer_runs"
REPORT_DIR="$EXP_ROOT/reports"

step() { echo; echo "=================================================================="; echo "$1"; echo "=================================================================="; }

# 입력이 실험 폴더 안을 가리키지 않는지 확인 (원본 read-only 보장)
for path in "$RQVAE_DATA_DIR" "$CKPT" "${A_SID_DIR:-}" "${A_RUN_DIR:-}"; do
    [ -z "$path" ] && continue
    resolved="$(realpath -m "$path")"
    case "$resolved" in
        "$EXP_ROOT"*) echo "입력 경로가 새 실험 폴더 안에 있습니다: $path" >&2; exit 1 ;;
    esac
done

mkdir -p "$REPORT_DIR"
cd "$REPO_ROOT"

# ------------------------------------------------------------
step "1. SID 생성 (train_c2_policy=event, event_repr=$EVENT_REPR, 기존 checkpoint 그대로)"
# ------------------------------------------------------------
if [ -f "$SID_DIR/article_semantic_ids.parquet" ]; then
    echo "이미 존재: $SID_DIR (건너뜀)"
else
    (cd RQVAE && "$PYTHON" generate_semantic_ids.py \
        --data_dir "$(realpath "$RQVAE_DATA_DIR")" \
        --checkpoint "$(realpath "$CKPT")" \
        --output_dir "$SID_DIR" \
        --train_c2_policy event \
        --event_repr "$EVENT_REPR") 2>&1 | tee "$REPORT_DIR/01_generate_semantic_ids.log"
fi

if [ -n "${A_SID_DIR:-}" ] && [ ! -f "$REPORT_DIR/sid_ab/sid_ab_report.json" ]; then
    step "1-b. SID A/B 비교"
    "$PYTHON" "$HERE/compare_sid_ab.py" \
        --a-dir "$A_SID_DIR" --b-dir "$SID_DIR" --out-dir "$REPORT_DIR/sid_ab" \
        --checkpoint "$CKPT" --data-dir "$RQVAE_DATA_DIR" 2>&1 | tee "$REPORT_DIR/01b_sid_ab.log"
fi

# ------------------------------------------------------------
step "2. post-rqvae: 새 SID로 전체 train/validation sequences"
# ------------------------------------------------------------
if [ -f "$POST_DIR/train_sequences.parquet" ] && [ -f "$POST_DIR/validation_sequences.parquet" ]; then
    echo "이미 존재: $POST_DIR (건너뜀)"
else
    SID_OUTPUT_DIR="$OUTPUT_ROOT" "$PYTHON" -m src.main post-rqvae \
        --sid-path "$SID_DIR/article_semantic_ids.parquet" \
        --experiment "$EXP" 2>&1 | tee "$REPORT_DIR/02_post_rqvae.log"
fi

# ------------------------------------------------------------
step "3. 1pos4neg sampling (새 SID 기준, seed 42)"
# ------------------------------------------------------------
if [ -f "$NEG_DIR/train_sequences_1pos4neg.parquet" ] && [ -f "$NEG_DIR/validation_sequences_1pos4neg.parquet" ]; then
    echo "이미 존재: $NEG_DIR (건너뜀)"
else
    # history/candidate SID 모두 새 SID로 만든 sequences를 쓴다.
    # (--semantic-ids는 candidate SID만 바꾸므로 사용하지 않는다.)
    SID_OUTPUT_DIR="$OUTPUT_ROOT" "$PYTHON" -m src.build_1pos4neg_sequences \
        --out-dir "$NEG_DIR" \
        --path "$POST_DIR/train_sequences.parquet" \
        --path "$POST_DIR/validation_sequences.parquet" \
        --negatives 4 --seed 42 --level c1c2c3 \
        --report "$REPORT_DIR/03_1pos4neg_report.json" 2>&1 | tee "$REPORT_DIR/03_1pos4neg.log"
fi

# ------------------------------------------------------------
step "4. candidate order shuffle (seed 42)"
# ------------------------------------------------------------
if [ -f "$SHUF_DIR/train_sequences_1pos4neg.parquet" ] && [ -f "$SHUF_DIR/validation_sequences_1pos4neg.parquet" ]; then
    echo "이미 존재: $SHUF_DIR (건너뜀)"
else
    "$PYTHON" -m src.shuffle_candidate_order \
        --input-dir "$NEG_DIR" --output-dir "$SHUF_DIR" --seed 42 \
        --report "$REPORT_DIR/04_shuffle_report.json" 2>&1 | tee "$REPORT_DIR/04_shuffle.log"
fi

# ------------------------------------------------------------
step "5. validation 50:50 split (validation half / test)"
# ------------------------------------------------------------
if [ -f "$DATASET_DIR/validation_sequences_1pos4neg_half.parquet" ] && [ -f "$DATASET_DIR/test_sequences_1pos4neg.parquet" ]; then
    echo "이미 존재: $DATASET_DIR (건너뜀)"
else
    "$PYTHON" "$HERE/split_validation_half.py" \
        --input "$SHUF_DIR/validation_sequences_1pos4neg.parquet" \
        --out-dir "$DATASET_DIR" 2>&1 | tee "$REPORT_DIR/05_split.log"
fi

# ------------------------------------------------------------
step "6. Transformer 학습 + Test (V1 최종 config, seed: $SEEDS)"
# ------------------------------------------------------------
# shellcheck disable=SC2086
"$PYTHON" "$HERE/run_transformer.py" \
    --transformer-root "$TRANSFORMER_ROOT" \
    --train-path "$SHUF_DIR/train_sequences_1pos4neg.parquet" \
    --dataset-dir "$DATASET_DIR" \
    --sid-path "$SID_DIR/article_semantic_ids.parquet" \
    --out-dir "$RUNS_DIR" \
    --seeds $SEEDS \
    --num-workers "$NUM_WORKERS" 2>&1 | tee -a "$REPORT_DIR/06_transformer.log"

# ------------------------------------------------------------
if [ -n "${A_RUN_DIR:-}" ]; then
    step "7. Transformer A/B 비교"
    first_seed="${SEEDS%% *}"
    "$PYTHON" "$HERE/compare_transformer_ab.py" \
        --a-run "$A_RUN_DIR" \
        --b-run "$RUNS_DIR/seed$first_seed" \
        --out-dir "$REPORT_DIR/transformer_ab" 2>&1 | tee "$REPORT_DIR/07_transformer_ab.log"
fi

step "완료: $EXP_ROOT"
