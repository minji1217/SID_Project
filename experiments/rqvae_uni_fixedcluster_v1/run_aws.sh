#!/usr/bin/env bash
# UNI 설정(WD-001-UNI-lu0.05-m0.5) 그대로 RQ-VAE 재학습. 입력만 고정 클러스터 전처리 결과(exports/rqvae_inputs)로 바꾼다.
#
#   cd ~/minji/SID_Project
#   SEED=42 CUDA_VISIBLE_DEVICES=0 nohup bash experiments/rqvae_uni_fixedcluster_v1/run_aws.sh \
#       > data/output/rqvae_uni_fixedcluster_s42.log 2>&1 &
#   tail -f data/output/rqvae_uni_fixedcluster_s42.log
#
# 먼저 같은 브랜치에서 python -m src.main preprocess 를 끝내서 data/output/exports/rqvae_inputs/ 가 있어야 한다.
# 학습 코드는 uniqueness-loss-experiment 브랜치(UNI 를 만든 코드)를 별도 worktree 로 꺼내 쓴다 (현재 checkout 은 건드리지 않음).
# 결과는 OUT 폴더에만 쓴다 (이미 있으면 멈춤). 원본/기존 결과는 읽기만 한다.
#
# 환경변수
#   SEED        기본 42 (A 와 같음)
#   INPUT_DIR   기본 <repo>/data/output/exports/rqvae_inputs
#   OUT_DIR     기본 <repo>/data/output/rqvae/uni_fixedcluster_lu0.05_m0.5_s<SEED>
#   UNI_REF     기본 127ae7e (uniqueness-loss-experiment "gin 내용 추가". 학습 코드는 7c022f3·브랜치 최신과 같고 UNI gin 내용이 처음 들어간 커밋)
#   UNI_DIR     기본 ~/minji/rqvae_uni_code_<UNI_REF>  (학습 코드 worktree)
#   PYTHON      기본 python   (gin-config, accelerate, einops, scikit-learn, huggingface-hub, wandb 필요)
#   SMOKE=1     1 epoch 만 돌려 입력/코드 연결만 확인 (OUT 이름에 _smoke)
#   ALLOW_CPU=1 GPU 없이도 실행 (스모크 테스트용)
#   A_CKPT      기존 UNI checkpoint_best_rec.pt 경로. 주면 그 안의 gin_config 와 하이퍼파라미터를 비교해서 다르면 멈춤 (권장)
#   VARIANT     h (기본, 실험 H = 지금 ver2) | r1 (실험 R: Q2 K-means 초기화를 사건별 r1(a)=h(a)-Q1[c1] 평균으로. 나머지 같음)
#               r1 이면 OUT 기본 이름에 _r1 이 붙고 train_rqvae_r1.py 로 학습한다.
set -euo pipefail
CODE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$CODE/../.." && pwd)"
SEED="${SEED:-42}"
INPUT="$(realpath -m "${INPUT_DIR:-$REPO_ROOT/data/output/exports/rqvae_inputs}")"
UNI_REF="${UNI_REF:-127ae7e}"
UNI_DIR="$(realpath -m "${UNI_DIR:-$HOME/minji/rqvae_uni_code_$UNI_REF}")"
PYTHON="${PYTHON:-python}"
VARIANT="${VARIANT:-h}"
case "$VARIANT" in
    h)  VTAG="" ;;
    r1) VTAG="_r1" ;;
    *)  echo "VARIANT 는 h 또는 r1 ($VARIANT)"; exit 1 ;;
esac
SUFFIX=""; [[ "${SMOKE:-0}" == "1" ]] && SUFFIX="_smoke"
OUT="$(realpath -m "${OUT_DIR:-$REPO_ROOT/data/output/rqvae/uni_fixedcluster_lu0.05_m0.5${VTAG}_s${SEED}${SUFFIX}}")"
export PYTHONUNBUFFERED=1

echo "== 0. 설정"
echo "   repo    $REPO_ROOT ($(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD) $(git -C "$REPO_ROOT" rev-parse --short HEAD))"
echo "   input   $INPUT"
echo "   code    $UNI_DIR (uniqueness-loss-experiment @ $UNI_REF)"
echo "   out     $OUT"
echo "   seed    $SEED   smoke ${SMOKE:-0}   variant $VARIANT"

echo "== 1. 입력 확인"
for f in article_master.parquet validation_article_master.parquet article_embeddings.npy event_master.parquet manifest.json; do
    [[ -f "$INPUT/$f" ]] || { echo "없음: $INPUT/$f  (먼저 python -m src.main preprocess)"; exit 1; }
done
"$PYTHON" - "$INPUT" <<'EOF'
import json, sys
import numpy as np, pandas as pd
d = sys.argv[1]
man = json.load(open(f"{d}/manifest.json", encoding="utf-8"))
v = man["validation"]
assert v["event_entity"]["status"] == "PASS", "manifest 검사 실패"
tr = pd.read_parquet(f"{d}/article_master.parquet")
va = pd.read_parquet(f"{d}/validation_article_master.parquet")
emb = np.load(f"{d}/article_embeddings.npy", mmap_mode="r")
ev = tr["event_id"].nunique()
print(f"   train 기사 {len(tr):,} · validation-only {len(va):,} · train 사건 {ev:,} · 임베딩 {emb.shape}")
print(f"   c1 범위 {tr['model_category_id'].min()}..{tr['model_category_id'].max()} (gin num_categories=25)")
assert set(tr.columns) >= {"article_id", "embedding_row", "model_category_id", "event_id"}
assert tr["model_category_id"].max() < 25 and va["model_category_id"].max() < 25
assert ev >= 128, "Q2 K-means 초기화에는 train 사건이 c2 코드북 크기(128) 이상 필요"
assert not set(tr["article_id"]) & set(va["article_id"])
if (len(tr), len(va), ev) != (10419, 2741, 9794):
    print("   주의: 기대값(10,419 / 2,741 / 9,794)과 다릅니다. 전처리 브랜치·원본을 확인하세요.")
EOF

[[ -e "$OUT" ]] && { echo "$OUT 가 이미 있습니다 (덮어쓰지 않음)"; exit 1; }

echo "== 2. 학습 코드 (uniqueness-loss-experiment @ $UNI_REF)"
git -C "$REPO_ROOT" fetch -q origin uniqueness-loss-experiment
WANT="$(git -C "$REPO_ROOT" rev-parse "$UNI_REF^{commit}")"
if [[ -d "$UNI_DIR" ]]; then
    HAVE="$(git -C "$UNI_DIR" rev-parse HEAD)"
    [[ "$HAVE" == "$WANT" ]] || { echo "$UNI_DIR 가 $HAVE 입니다 ($WANT 이어야 함)"; exit 1; }
    [[ -z "$(git -C "$UNI_DIR" status --porcelain -- RQVAE)" ]] || { echo "$UNI_DIR/RQVAE 에 수정된 파일이 있습니다"; exit 1; }
else
    git -C "$REPO_ROOT" worktree add --detach "$UNI_DIR" "$WANT"
fi
echo "   $(git -C "$UNI_DIR" log -1 --format='%h %ad %s' --date=short)"

if [[ "${ALLOW_CPU:-0}" != "1" ]]; then
    "$PYTHON" -c "import torch, sys; ok = torch.cuda.is_available(); print('   cuda', ok, torch.__version__); sys.exit(0 if ok else 1)" \
        || { echo "GPU 를 못 찾았습니다 (CUDA_VISIBLE_DEVICES 확인)"; exit 1; }
fi

echo "== 3. gin (UNI 그대로 + 입력/저장 경로/seed)"
mkdir -p "$OUT"
EXTRA=()
if [[ "${SMOKE:-0}" == "1" ]]; then
    EXTRA=(--extra train.epochs=1 train.eval_every_epochs=1 train.early_stopping=False train.save_model_every_epochs=1000)
    [[ "${ALLOW_CPU:-0}" == "1" ]] && EXTRA+=(train.amp=False)
fi
"$PYTHON" "$CODE/make_gin.py" --src "$UNI_DIR/RQVAE/configs/rqvae_ebnerd_wd001_uni.gin" \
    --out "$OUT/rqvae_uni.gin" --dataset_folder "$INPUT" --save_dir "$OUT" --seed "$SEED" "${EXTRA[@]}"

{
    echo "preprocess_repo $(git -C "$REPO_ROOT" rev-parse HEAD) ($(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD))"
    echo "train_code      $WANT (uniqueness-loss-experiment)"
    echo "input_dir       $INPUT"
    echo "seed            $SEED"
    echo "variant         $VARIANT (h: Q2 초기화 = 사건별 h 평균 / r1: 사건별 r1 = h - Q1[c1] 평균)"
    date -u +"started_utc     %Y-%m-%dT%H:%M:%SZ"
} > "$OUT/provenance.txt"
(cd "$INPUT" && sha256sum article_master.parquet validation_article_master.parquet article_embeddings.npy event_master.parquet) > "$OUT/inputs.sha256"
cp "$INPUT/manifest.json" "$OUT/inputs_manifest.json"

if [[ -n "${A_CKPT:-}" && "${SMOKE:-0}" != "1" ]]; then
    echo "== 3-1. 기존 UNI(A) checkpoint 설정과 비교"
    "$PYTHON" "$CODE/check_against_a.py" "$A_CKPT" "$OUT/rqvae_uni.gin" "$UNI_DIR/RQVAE" | tee "$OUT/check_against_a.txt" \
        || { echo "A 와 설정이 다릅니다. 학습하지 않고 멈춥니다."; exit 1; }
    echo "a_checkpoint    $A_CKPT ($(sha256sum "$A_CKPT" | cut -c1-12))" >> "$OUT/provenance.txt"
fi

echo "== 4. 학습 (로그: $OUT/train.log · 진행 중 지표 표: python $CODE/parse_log.py $OUT)"
cd "$UNI_DIR/RQVAE"
STATUS=0
if [[ "$VARIANT" == "r1" ]]; then
    cp "$CODE/train_rqvae_r1.py" "$OUT/train_rqvae_r1.py"
    "$PYTHON" "$OUT/train_rqvae_r1.py" "$OUT/rqvae_uni.gin" > "$OUT/train.log" 2>&1 || STATUS=$?
    if ! grep -q "^\[R\] Q2 K-means 입력" "$OUT/train.log"; then
        echo "r1 초기화 로그가 없습니다"
        [[ $STATUS -ne 0 ]] || STATUS=1
    fi
else
    "$PYTHON" train_rqvae.py "$OUT/rqvae_uni.gin" > "$OUT/train.log" 2>&1 || STATUS=$?
fi
date -u +"finished_utc    %Y-%m-%dT%H:%M:%SZ (exit $STATUS)" >> "$OUT/provenance.txt"

echo "== 5. 지표 CSV (metrics_train / metrics_validation / early_stopping / summary.json)"
"$PYTHON" "$CODE/parse_log.py" "$OUT" || true
if [[ $STATUS -ne 0 ]]; then echo "학습 실패 (exit $STATUS). 마지막 로그:"; tail -30 "$OUT/train.log"; exit $STATUS; fi
ls -la "$OUT" | grep -E "checkpoint|gin|log" || true
echo "SID 생성에 쓸 checkpoint: $OUT/checkpoint_best_rec.pt"
echo "== DONE"
