#!/usr/bin/env bash
# 교수님 제안: NRMS 뉴스 입력 = UNI RQ-VAE 복원 벡터(recon). 정민교 NRMS 코드(nrms_original_ebnerd_v2)를 수정 없이
# 'e5' 모드로 돌리고, 표 파일만 article_embeddings.npy 대신 recon_embeddings.npy를 넣는다.
# 데이터 · 설정 · seed는 nrms_original_ebnerd_v2와 같다 (5 epoch, batch 64, seed 42 123 2026).
#
#   cd ~/minji/SID_Project && CKPT=<UNI checkpoint_best_rec.pt 경로> CUDA_VISIBLE_DEVICES=0 \
#       nohup bash experiments/nrms_recon_ebnerd_v1/run_aws.sh > data/output/experiments/nrms_recon_ebnerd_v1.log 2>&1 &
#   tail -f data/output/experiments/nrms_recon_ebnerd_v1.log
#
# 기존 파일(원본 parquet/npy/checkpoint, v2 tables)은 읽기만 한다. 결과는 OUT 폴더에만 쓴다(이미 있으면 멈춤).
set -euo pipefail
CODE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$CODE/../.." && pwd)"
V2="$REPO_ROOT/experiments/nrms_original_ebnerd_v2"
RAW="${RAW_DIR:-$HOME/shared/raw/ebnerd}"
SEQ="${SEQ_DIR:-$HOME/shared/datasets/ebnerd}"
TABLES="${TABLES_DIR:-$REPO_ROOT/data/output/experiments/nrms_original_ebnerd_v2/tables}"
OUT="$(realpath -m "${OUT_DIR:-$REPO_ROOT/data/output/experiments/nrms_recon_ebnerd_v1}")"
CKPT="${CKPT:?CKPT=<UNI checkpoint_best_rec.pt 경로> 를 지정하세요}"
PYTHON="${PYTHON:-python}"
RQVAE_PYTHON="${RQVAE_PYTHON:-$PYTHON}"   # RQ-VAE 코드는 gin, huggingface_hub 필요
SEEDS="${SEEDS:-42 123 2026}"
export PYTHONUNBUFFERED=1

TRAIN="$SEQ/train_sequences_1pos4neg.parquet"
VAL="$SEQ/validation_sequences_1pos4neg_half.parquet"
TEST="$SEQ/test_sequences_1pos4neg.parquet"
E5="$RAW/article_embeddings.npy"
for f in "$CKPT" "$E5" "$TRAIN" "$VAL" "$TEST" "$TABLES/sid_codes.npy" "$TABLES/article_id2row.pkl"; do
    [[ -f "$f" ]] || { echo "없음: $f"; exit 1; }
done
[[ -e "$OUT" ]] && { echo "$OUT 가 이미 있습니다 (덮어쓰지 않음)"; exit 1; }
mkdir -p "$OUT"
RECON="$OUT/recon_embeddings.npy"

echo "== 1. recon 표 생성 (UNI RQ-VAE decoder(Q1[c1]+Q2[c2]+Q3[c3]))"
"$RQVAE_PYTHON" "$CODE/build_recon_table.py" --ckpt "$CKPT" --tables_dir "$TABLES" --e5_path "$E5" --out "$RECON"

cd "$OUT"   # train.py는 checkpoint를 현재 폴더에 저장한다
for s in $SEEDS; do
    echo "== 2. train  recon seed=$s"
    "$PYTHON" "$V2/train.py" --input_mode e5 --epochs 5 --batch_size 64 --num_workers 4 --seed "$s" --tag "_recon_s$s" \
        --tables_dir "$TABLES" --e5_path "$RECON" --train_path "$TRAIN" --val_path "$VAL" --test_path "$TEST" \
        2>&1 | tee "log_recon_s${s}.txt"
    echo "== 3. evaluate recon seed=$s"
    "$PYTHON" "$V2/evaluate_test.py" --input_mode e5 --tag "_recon_s$s" --tables_dir "$TABLES" --e5_path "$RECON" \
        --val_path "$VAL" --test_path "$TEST" --out "metrics_recon_s${s}.json"
done

echo "== 4. 요약 (seed 평균) · 비교: v2 E5 29.92% / SID 26.09% / Title 26.06% / 섞은 SID 20.71%"
"$PYTHON" - "$OUT" <<'EOF'
import json, glob, sys, statistics as st
out = sys.argv[1]
js = [json.load(open(f)) for f in sorted(glob.glob(f"{out}/metrics_recon_s*.json"))]
m = lambda split, k: st.mean(j[split][k] for j in js)
s = {"seeds": len(js), "validation_top1": m("validation", "top1_accuracy"),
     **{f"test_{k}": m("test", k) for k in ("top1_accuracy", "mrr", "ndcg@5", "auc")},
     "test_top1_per_seed": [j["test"]["top1_accuracy"] for j in js]}
print(json.dumps(s, indent=2))
json.dump(s, open(f"{out}/summary.json", "w"), indent=2)
EOF
echo "== DONE"
