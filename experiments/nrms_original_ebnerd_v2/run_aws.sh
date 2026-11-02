#!/usr/bin/env bash
# 정민교 NRMS(원 논문 구조) 코드를 그대로 EB-NeRD에 적용: title / e5 / sid / sid_shuffled × seed 42,123,2026
# 설정은 정민교 MIND 실행과 같다: --epochs 5 --batch_size 64 (lr 1e-3, embed 300, head 20, dropout 0.2 = 코드 기본값)
#
#   cd ~/minji/SID_Project && CUDA_VISIBLE_DEVICES=0 nohup bash experiments/nrms_original_ebnerd_v2/run_aws.sh \
#       > data/output/experiments/nrms_original_ebnerd_v2.log 2>&1 &
#   tail -f data/output/experiments/nrms_original_ebnerd_v2.log
#
# 경로는 환경변수로 바꿀 수 있다. 기존 파일은 읽기만 하고, 결과는 OUT 폴더에만 쓴다(이미 있으면 멈춤).
set -euo pipefail
CODE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$CODE/../.." && pwd)"
RAW="${RAW_DIR:-$HOME/shared/raw/ebnerd}"                 # article_master / validation_article_master / article_embeddings.npy
SEQ="${SEQ_DIR:-$HOME/shared/datasets/ebnerd}"            # UNI SID 1pos4neg (V1·9장 NRMS와 같은 파일)
OUT="${OUT_DIR:-$REPO_ROOT/data/output/experiments/nrms_original_ebnerd_v2}"
OUT="$(realpath -m "$OUT")"   # 아래에서 cd "$OUT" 하므로 상대경로면 tables 경로가 깨진다
PYTHON="${PYTHON:-python}"
MODES="${MODES:-title e5 sid sid_shuffled}"
SEEDS="${SEEDS:-42 123 2026}"  # EB-NeRD의 다른 실험(9장 NRMS, V1 seed 확인)과 같은 seed. 민교 MIND는 0 1 2
export PYTHONUNBUFFERED=1

TRAIN="$SEQ/train_sequences_1pos4neg.parquet"
VAL="$SEQ/validation_sequences_1pos4neg_half.parquet"
TEST="$SEQ/test_sequences_1pos4neg.parquet"
E5="$RAW/article_embeddings.npy"
for f in "$RAW/article_master.parquet" "$RAW/validation_article_master.parquet" "$E5" "$TRAIN" "$VAL" "$TEST"; do
    [[ -f "$f" ]] || { echo "없음: $f  (RAW_DIR / SEQ_DIR 환경변수로 경로 지정)"; exit 1; }
done
[[ -e "$OUT" ]] && { echo "$OUT 가 이미 있습니다 (덮어쓰지 않음)"; exit 1; }
mkdir -p "$OUT/tables"

echo "== 1. 조회 테이블 (SID는 V1이 쓴 sequence 파일의 c1~c3에서 모음)"
"$PYTHON" "$CODE/build_tables_ebnerd.py" --raw_dir "$RAW" --out_dir "$OUT/tables" --seq_paths "$TRAIN" "$VAL" "$TEST"

cd "$OUT"   # train.py는 checkpoint를 현재 폴더에 저장한다
for mode in $MODES; do
  for s in $SEEDS; do
    echo "== 2. train  mode=$mode seed=$s"
    "$PYTHON" "$CODE/train.py" --input_mode "$mode" --epochs 5 --batch_size 64 --num_workers 4 --seed "$s" --tag "_s$s" \
        --tables_dir "$OUT/tables" --e5_path "$E5" --train_path "$TRAIN" --val_path "$VAL" --test_path "$TEST" \
        2>&1 | tee "log_${mode}_s${s}.txt"
    echo "== 3. evaluate mode=$mode seed=$s"
    "$PYTHON" "$CODE/evaluate_test.py" --input_mode "$mode" --tag "_s$s" --tables_dir "$OUT/tables" --e5_path "$E5" \
        --val_path "$VAL" --test_path "$TEST" --out "metrics_${mode}_s${s}.json"
  done
done

echo "== 4. 요약 (seed 평균)"
"$PYTHON" - "$OUT" <<'EOF'
import json, glob, sys, statistics as st
out = sys.argv[1]
rows = {}
for f in sorted(glob.glob(f"{out}/metrics_*_s*.json")):
    j = json.load(open(f)); rows.setdefault(j["input_mode"], []).append(j)
print(f"{'mode':14s} {'n':>2s} {'Val Top-1':>10s} {'Test Top-1':>11s} {'MRR':>7s} {'nDCG@5':>7s} {'AUC':>7s}")
summary = {}
for mode, js in rows.items():
    m = lambda split, k: st.mean(j[split][k] for j in js)
    summary[mode] = {"seeds": len(js), "validation_top1": m("validation", "top1_accuracy"),
                     **{f"test_{k}": m("test", k) for k in ("top1_accuracy", "mrr", "ndcg@5", "auc")}}
    s = summary[mode]
    print(f"{mode:14s} {len(js):>2d} {s['validation_top1']:>10.2%} {s['test_top1_accuracy']:>11.2%} "
          f"{s['test_mrr']:>7.4f} {s['test_ndcg@5']:>7.4f} {s['test_auc']:>7.4f}")
json.dump(summary, open(f"{out}/summary.json", "w"), indent=2)
EOF
