# 공통 설정. 각 단계 스크립트가 source 한다. 직접 실행하지 않는다.
#
# 환경변수 (기본값)
#   PRE_DIR   새 전처리 결과 폴더      ~/minji/SID_Project/data/output/fixedcluster_v1  (= SID_OUTPUT_DIR)
#   V2_DIR    이번 실험 결과 폴더      $PRE_DIR/v2
#   YM_DIR    여민 코드 checkout       ~/minji/yeomin_code_22028ba
#   PYTHON    python
#   ALLOW_CPU 1 이면 GPU 없이도 실행 (스모크 테스트용)

CODE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$CODE/../.." && pwd)"
YM_REF="22028ba1e8762802d2c3c8dd56926f1d180f1961"   # 여민 main "Update RQ-VAE and SID generation" (2026-11-23 11:43)
UNI_REF="127ae7e"                                   # UNI gin 원본 (minji1217/SID_Project uniqueness-loss-experiment)

PRE="$(realpath -m "${PRE_DIR:-$REPO_ROOT/data/output/fixedcluster_v1}")"
INPUT="$PRE/exports/rqvae_inputs"
V2="$(realpath -m "${V2_DIR:-$PRE/v2}")"
YM="$(realpath -m "${YM_DIR:-$HOME/minji/yeomin_code_${YM_REF:0:7}}")"
PYTHON="${PYTHON:-python}"
EXP_PREFIX="${EXP_PREFIX:-v2_}"                     # 시퀀스·1pos4neg 실험 폴더 이름 앞부분
export PYTHONUNBUFFERED=1

die() { echo "멈춤: $*" >&2; exit 1; }

check_yeomin_code() {
    [[ -d "$YM/RQVAE" && -d "$YM/Transformer" ]] || die "여민 코드가 없습니다: $YM (먼저 0_setup.sh)"
    local have; have="$(git -C "$YM" rev-parse HEAD)"
    [[ "$have" == "$YM_REF" ]] || die "$YM 가 $have 입니다 ($YM_REF 이어야 함)"
    [[ -z "$(git -C "$YM" status --porcelain -- RQVAE Transformer)" ]] || die "$YM 에 수정된 파일이 있습니다"
}

check_inputs() {
    for f in article_master.parquet validation_article_master.parquet article_embeddings.npy event_master.parquet \
             article_entities.parquet entity_idf.parquet entity_idf_meta.json manifest.json; do
        [[ -f "$INPUT/$f" ]] || die "입력 없음: $INPUT/$f"
    done
}

check_gpu() {
    [[ "${ALLOW_CPU:-0}" == "1" ]] && return 0
    "$PYTHON" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
        || die "GPU 를 못 찾았습니다 (CUDA_VISIBLE_DEVICES 확인)"
}

new_out_dir() {   # 이미 있으면 멈춘다 (덮어쓰지 않음)
    [[ -e "$1" ]] && die "$1 가 이미 있습니다 (덮어쓰지 않음). 지우거나 다른 이름을 쓰세요."
    mkdir -p "$1"
}

provenance() {    # $1 = 결과 폴더, 나머지 = 추가 줄
    local out="$1"; shift
    {
        echo "minji_repo  $(git -C "$REPO_ROOT" rev-parse HEAD) ($(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD))"
        echo "yeomin_code $YM_REF ($YM)"
        echo "pre_dir     $PRE"
        for line in "$@"; do echo "$line"; done
        date -u +"started_utc %Y-%m-%dT%H:%M:%SZ"
    } > "$out/provenance.txt"
}

finish_provenance() {   # $1 = 결과 폴더, $2 = 종료 코드
    date -u +"finished_utc %Y-%m-%dT%H:%M:%SZ (exit $2)" >> "$1/provenance.txt"
}
