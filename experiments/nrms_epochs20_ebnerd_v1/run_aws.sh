#!/usr/bin/env bash
# NRMS(정민교 코드, nrms_original_ebnerd_v2) 를 최대 20 epoch으로 다시 돌린다. 5 epoch에서 E5 · recon이 아직 오르는 중이라서.
# 코드 · 데이터 · 설정 · seed는 v2 / nrms_recon_ebnerd_v1과 같고 --epochs 만 다르다. best = validation Top-1 최고 epoch.
#
# 모드마다 결과 폴더가 따로라서 여러 모드를 동시에 돌려도 된다 (GPU 하나에 작은 모델 여러 개).
#   cd ~/minji/SID_Project
#   for m in e5 recon sid sid_shuffled; do
#     MODES=$m RAW_DIR=~/minji/e5_input CUDA_VISIBLE_DEVICES=0 \
#       nohup bash experiments/nrms_epochs20_ebnerd_v1/run_aws.sh > data/output/experiments/nrms_epochs20_$m.log 2>&1 &
#   done
#   tail -f data/output/experiments/nrms_epochs20_*.log
#   python experiments/nrms_epochs20_ebnerd_v1/summarize.py        # 끝난 뒤 요약
#
# 기존 파일(원본 npy/parquet, v2 tables, recon 표)은 읽기만 한다. 결과는 OUT_ROOT/<mode>/ 에만 쓴다(이미 있으면 멈춤).
set -euo pipefail
CODE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$CODE/../.." && pwd)"
V2="$REPO_ROOT/experiments/nrms_original_ebnerd_v2"
RAW="${RAW_DIR:-$HOME/minji/e5_input}"
SEQ="${SEQ_DIR:-$HOME/shared/datasets/ebnerd}"
TABLES="${TABLES_DIR:-$REPO_ROOT/data/output/experiments/nrms_original_ebnerd_v2/tables}"
RECON="${RECON_PATH:-$REPO_ROOT/data/output/experiments/nrms_recon_ebnerd_v1/recon_embeddings.npy}"
OUT_ROOT="$(realpath -m "${OUT_ROOT:-$REPO_ROOT/data/output/experiments/nrms_epochs20_ebnerd_v1}")"
EPOCHS="${EPOCHS:-20}"
MODES="${MODES:-e5 recon sid sid_shuffled}"
SEEDS="${SEEDS:-42 123 2026}"
PYTHON="${PYTHON:-python}"
export PYTHONUNBUFFERED=1

TRAIN="$SEQ/train_sequences_1pos4neg.parquet"
VAL="$SEQ/validation_sequences_1pos4neg_half.parquet"
TEST="$SEQ/test_sequences_1pos4neg.parquet"
E5="$RAW/article_embeddings.npy"
for f in "$E5" "$TRAIN" "$VAL" "$TEST" "$TABLES/sid_codes.npy" "$TABLES/article_id2row.pkl"; do
    [[ -f "$f" ]] || { echo "없음: $f"; exit 1; }
done

for mode in $MODES; do
    case "$mode" in
        e5)    imode=e5;  emb="$E5" ;;
        recon) imode=e5;  emb="$RECON"; [[ -f "$RECON" ]] || { echo "없음: $RECON (nrms_recon_ebnerd_v1 먼저)"; exit 1; } ;;
        sid|sid_shuffled|title) imode=$mode; emb="$E5" ;;
        *) echo "모르는 모드: $mode"; exit 1 ;;
    esac
    OUT="$OUT_ROOT/$mode"
    [[ -e "$OUT" ]] && { echo "$OUT 가 이미 있습니다 (덮어쓰지 않음)"; exit 1; }
    mkdir -p "$OUT"
    cd "$OUT"   # train.py는 checkpoint를 현재 폴더에 저장한다
    for s in $SEEDS; do
        echo "== train  mode=$mode seed=$s epochs=$EPOCHS"
        "$PYTHON" "$V2/train.py" --input_mode "$imode" --epochs "$EPOCHS" --batch_size 64 --num_workers 4 --seed "$s" \
            --tag "_${mode}_s$s" --tables_dir "$TABLES" --e5_path "$emb" \
            --train_path "$TRAIN" --val_path "$VAL" --test_path "$TEST" 2>&1 | tee "log_${mode}_s${s}.txt"
        echo "== evaluate mode=$mode seed=$s"
        "$PYTHON" "$V2/evaluate_test.py" --input_mode "$imode" --tag "_${mode}_s$s" --tables_dir "$TABLES" --e5_path "$emb" \
            --val_path "$VAL" --test_path "$TEST" --out "metrics_${mode}_s${s}.json"
    done
    echo "== DONE mode=$mode"
done
