#!/usr/bin/env bash
# ============================================================
# eventc2_retrain_v1: EventC2로 재학습한 RQ-VAE의 SID로 downstream 전체 재생성 + UNI baseline 비교
#
#   EventC2 checkpoint_best_rec.pt (재학습 없음)
#     -> generate_semantic_ids.py --train_c2_policy event  (AWS에서 만든 SID와 내용 일치 확인)
#     -> post-rqvae (전체 train/validation sequences)
#     -> 1pos4neg (seed 42, c1c2c3, negative 4, 단계별 감소량 기록)
#     -> candidate shuffle (seed 42)
#     -> validation 50:50 split (validation half / test)
#     -> Transformer V1 최종 config (seed 42, c4_vocab_size는 max c4 + 1 = 32)
#     -> UNI baseline과 비교 (전체 test / 공통 impression / 공통 + candidate 동일)
#
# UNI baseline 파일과 run 폴더는 읽기만 한다. 모든 출력은 $EXP_ROOT 아래.
# 각 단계는 결과가 있으면 건너뛰고, 어떤 파일도 덮어쓰지 않는다.
#
# 필수 환경변수
#   CKPT             : EventC2 checkpoint_best_rec.pt
#   RQVAE_DATA_DIR   : RQ-VAE 입력 3개 폴더 (RQVAE/datasets/ebnerd)
#   TRANSFORMER_ROOT : UNI baseline을 학습한 Transformer 폴더
#   UNI_TRAIN, UNI_VAL_HALF, UNI_TEST : UNI baseline이 쓴 shuffled_v2 train / validation half / test
#   UNI_RUN          : UNI baseline seed42 run 폴더 (run_summary.json, checkpoint_best.pt)
# 선택
#   UNI_SEQ_DIR      : UNI의 1pos4neg 이전 train/validation_sequences.parquet 폴더 (단계별 감소량 재집계)
#   EXP              : 기본 normalize_v2_uni_lu005_m05_eventc2_retrain_v1
#   SID_OUTPUT_DIR   : 기본 data/output
#   NUM_WORKERS      : 기본 4
# ============================================================

set -euo pipefail

for name in CKPT RQVAE_DATA_DIR TRANSFORMER_ROOT UNI_TRAIN UNI_VAL_HALF UNI_TEST UNI_RUN; do
    [ -n "${!name:-}" ] || { echo "$name를 지정하세요" >&2; exit 1; }
done

EXP="${EXP:-normalize_v2_uni_lu005_m05_eventc2_retrain_v1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PYTHON="${PYTHON:-python}"
EXPECTED_SID_SIGNATURE="97a65313b5d597f0e6edc6ace459ebc492e4680c35eae3a598a54ee63b18676d"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HERE="$REPO_ROOT/experiments/eventc2_retrain_v1"
V1="$REPO_ROOT/experiments/eventc2_train_v1"
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

for path in "$CKPT" "$RQVAE_DATA_DIR" "$UNI_TRAIN" "$UNI_VAL_HALF" "$UNI_TEST" "$UNI_RUN" "${UNI_SEQ_DIR:-}"; do
    [ -z "$path" ] && continue
    [ -e "$path" ] || { echo "없음: $path" >&2; exit 1; }
    case "$(realpath -m "$path")" in "$EXP_ROOT"*) echo "입력이 새 실험 폴더 안에 있습니다: $path" >&2; exit 1 ;; esac
done

mkdir -p "$REPORT_DIR"
cd "$REPO_ROOT"

step "1. SID 생성 (EventC2 checkpoint, train_c2_policy=event)"
if [ -f "$SID_DIR/article_semantic_ids.parquet" ]; then
    echo "이미 존재: $SID_DIR (건너뜀)"
else
    (cd RQVAE && "$PYTHON" generate_semantic_ids.py \
        --data_dir "$(realpath "$RQVAE_DATA_DIR")" --checkpoint "$(realpath "$CKPT")" \
        --output_dir "$SID_DIR" --train_c2_policy event) 2>&1 | tee "$REPORT_DIR/01_generate_semantic_ids.log"
fi
"$PYTHON" - "$SID_DIR/article_semantic_ids.parquet" "$EXPECTED_SID_SIGNATURE" <<'PY'
import hashlib, sys
import pandas as pd
df = pd.read_parquet(sys.argv[1], columns=["article_id", "c1", "c2", "c3", "c4"])
df = df.astype({"article_id": str, "c1": "int64", "c2": "int64", "c3": "int64", "c4": "int64"}).sort_values("article_id")
sig = hashlib.sha256(df.to_csv(index=False).encode()).hexdigest()
if sig != sys.argv[2]:
    sys.exit(f"SID가 AWS에서 평가한 EventC2 SID와 다릅니다: {sig}")
print(f"SID 내용 일치 (EventC2 평가 SID와 동일), max c4 = {int(df['c4'].max())}")
PY

step "2. post-rqvae: EventC2 SID로 전체 train/validation sequences"
if [ -f "$POST_DIR/train_sequences.parquet" ] && [ -f "$POST_DIR/validation_sequences.parquet" ]; then
    echo "이미 존재: $POST_DIR (건너뜀)"
else
    SID_OUTPUT_DIR="$OUTPUT_ROOT" "$PYTHON" -m src.main post-rqvae \
        --sid-path "$SID_DIR/article_semantic_ids.parquet" --experiment "$EXP" 2>&1 | tee "$REPORT_DIR/02_post_rqvae.log"
fi

step "3. 1pos4neg (EventC2 SID 기준, seed 42, 단계별 감소량 기록)"
if [ -f "$NEG_DIR/train_sequences_1pos4neg.parquet" ] && [ -f "$NEG_DIR/validation_sequences_1pos4neg.parquet" ]; then
    echo "이미 존재: $NEG_DIR (건너뜀)"
else
    SID_OUTPUT_DIR="$OUTPUT_ROOT" "$PYTHON" -m src.build_1pos4neg_sequences --out-dir "$NEG_DIR" \
        --path "$POST_DIR/train_sequences.parquet" --path "$POST_DIR/validation_sequences.parquet" \
        --negatives 4 --seed 42 --level c1c2c3 \
        --report "$REPORT_DIR/03_1pos4neg_report.json" 2>&1 | tee "$REPORT_DIR/03_1pos4neg.log"
fi

step "4. candidate shuffle (seed 42)"
if [ -f "$SHUF_DIR/train_sequences_1pos4neg.parquet" ] && [ -f "$SHUF_DIR/validation_sequences_1pos4neg.parquet" ]; then
    echo "이미 존재: $SHUF_DIR (건너뜀)"
else
    "$PYTHON" -m src.shuffle_candidate_order --input-dir "$NEG_DIR" --output-dir "$SHUF_DIR" --seed 42 \
        --report "$REPORT_DIR/04_shuffle_report.json" 2>&1 | tee "$REPORT_DIR/04_shuffle.log"
fi

step "5. validation 50:50 split"
if [ -f "$DATASET_DIR/validation_sequences_1pos4neg_half.parquet" ] && [ -f "$DATASET_DIR/test_sequences_1pos4neg.parquet" ]; then
    echo "이미 존재: $DATASET_DIR (건너뜀)"
else
    "$PYTHON" "$V1/split_validation_half.py" --input "$SHUF_DIR/validation_sequences_1pos4neg.parquet" \
        --out-dir "$DATASET_DIR" 2>&1 | tee "$REPORT_DIR/05_split.log"
fi

step "6. Transformer 학습 + Test (V1 최종 config, seed 42)"
echo "c4_vocab_size: EventC2 SID의 max c4가 31이라 32를 쓴다 (기존 28은 max c4 27까지)." | tee "$REPORT_DIR/06_c4_vocab_note.txt"
echo "튜닝이 아니라 새 SID의 c4 범위가 늘어서 필요한 변경이다. UNI baseline은 기존 결과(vocab 28) 그대로." | tee -a "$REPORT_DIR/06_c4_vocab_note.txt"
"$PYTHON" "$V1/run_transformer.py" --transformer-root "$TRANSFORMER_ROOT" \
    --train-path "$SHUF_DIR/train_sequences_1pos4neg.parquet" --dataset-dir "$DATASET_DIR" \
    --sid-path "$SID_DIR/article_semantic_ids.parquet" --out-dir "$RUNS_DIR" \
    --seeds 42 --num-workers "$NUM_WORKERS" 2>&1 | tee -a "$REPORT_DIR/06_transformer.log"

UNI_NEG_REPORT_ARG=()
if [ -n "${UNI_SEQ_DIR:-}" ]; then
    step "7. UNI 1pos4neg 단계별 감소량 재집계 (parquet 쓰지 않음)"
    if [ ! -f "$REPORT_DIR/uni_1pos4neg_stats.json" ]; then
        SID_OUTPUT_DIR="$OUTPUT_ROOT" "$PYTHON" -m src.build_1pos4neg_sequences --stats-only --out-dir "$REPORT_DIR" \
            --path "$UNI_SEQ_DIR/train_sequences.parquet" --path "$UNI_SEQ_DIR/validation_sequences.parquet" \
            --negatives 4 --seed 42 --level c1c2c3 \
            --report "$REPORT_DIR/uni_1pos4neg_stats.json" 2>&1 | tee "$REPORT_DIR/07_uni_1pos4neg_stats.log"
    fi
    # 재집계가 baseline 데이터와 같은 규모인지 확인 (train rows, validation half + test rows)
    "$PYTHON" - "$REPORT_DIR/uni_1pos4neg_stats.json" "$UNI_TRAIN" "$UNI_VAL_HALF" "$UNI_TEST" <<'PY'
import json, sys
import pyarrow.parquet as pq
report = {r["split_name"]: r for r in json.load(open(sys.argv[1]))}
rows = lambda p: pq.read_metadata(p).num_rows
train_ok = report["train_sequences"]["output_row_count"] == rows(sys.argv[2])
valid_ok = report["validation_sequences"]["output_row_count"] == rows(sys.argv[3]) + rows(sys.argv[4])
print(f"UNI 재집계 rows == baseline: train {train_ok}, validation {valid_ok}")
if not (train_ok and valid_ok):
    print("WARNING: UNI_SEQ_DIR가 baseline을 만든 sequences가 아닐 수 있습니다.")
PY
    UNI_NEG_REPORT_ARG=(--uni-neg-report "$REPORT_DIR/uni_1pos4neg_stats.json")
fi

step "8. UNI baseline test score"
UNI_SCORES="$UNI_RUN/test_candidate_scores.parquet"
UNI_SCORE_RUN="$UNI_RUN"
if [ ! -f "$UNI_SCORES" ]; then
    echo "baseline run에 test_candidate_scores.parquet이 없어 추론만 다시 한다 (baseline 폴더에는 쓰지 않음)."
    "$PYTHON" "$HERE/predict_uni_baseline.py" --transformer-root "$TRANSFORMER_ROOT" --uni-run "$UNI_RUN" \
        --uni-test "$UNI_TEST" --out-dir "$REPORT_DIR/uni_baseline_inference" --num-workers "$NUM_WORKERS"
    UNI_SCORES="$REPORT_DIR/uni_baseline_inference/test_candidate_scores.parquet"
fi

step "9. UNI vs EventC2 비교"
if [ -f "$REPORT_DIR/uni_vs_eventc2/uni_vs_eventc2.md" ]; then
    cat "$REPORT_DIR/uni_vs_eventc2/uni_vs_eventc2.md"
else
    "$PYTHON" "$HERE/compare_uni_eventc2.py" \
        --uni-train "$UNI_TRAIN" --uni-val-half "$UNI_VAL_HALF" --uni-test "$UNI_TEST" \
        --uni-scores "$UNI_SCORES" --uni-run "$UNI_SCORE_RUN" "${UNI_NEG_REPORT_ARG[@]}" \
        --ec-train "$SHUF_DIR/train_sequences_1pos4neg.parquet" \
        --ec-val-half "$DATASET_DIR/validation_sequences_1pos4neg_half.parquet" \
        --ec-test "$DATASET_DIR/test_sequences_1pos4neg.parquet" \
        --ec-scores "$RUNS_DIR/seed42/test_candidate_scores.parquet" --ec-run "$RUNS_DIR/seed42" \
        --ec-neg-report "$REPORT_DIR/03_1pos4neg_report.json" \
        --out-dir "$REPORT_DIR/uni_vs_eventc2" 2>&1 | tee "$REPORT_DIR/09_compare.log"
fi

step "완료: $EXP_ROOT"
