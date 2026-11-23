#!/usr/bin/env bash
# ③④ 시퀀스 → 1pos4neg → 후보 순서 섞기 (우리 코드) → validation 앞 절반 / test 나누기 (여민 split_validation.py)
#   VARIANT=h bash experiments/v2_yeomin_rerun/3_build_candidates.sh > ~/minji/v2_cand_h.log 2>&1
#
# 결과: $PRE/experiments/v2_<VARIANT>/
#   post_rqvae/                      시퀀스 (train / validation)
#   candidate_1pos4neg/              1pos4neg (정답이 첫 자리)
#   candidate_1pos4neg_shuffled/     후보 순서 섞음 ← Transformer train 파일은 여기
#   split/                           validation 앞 절반 · test ← Transformer validation / test 파일은 여기
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

VARIANT="${VARIANT:?VARIANT=h 또는 r}"
SID="$V2/sid/$VARIANT/article_semantic_ids.parquet"
EXP="${EXP_PREFIX}${VARIANT}"
E="$PRE/experiments/$EXP"
export SID_OUTPUT_DIR="$PRE"    # 새 전처리 결과의 category_mapping 등을 쓰도록

echo "== ③ 시퀀스·1pos4neg·섞기·split: 실험 $VARIANT · SID $SID → $E"
check_yeomin_code
[[ -f "$SID" ]] || die "SID 파일 없음: $SID (② 먼저)"
[[ -e "$E" ]] && die "$E 가 이미 있습니다 (덮어쓰지 않음)"
cd "$REPO_ROOT"

"$PYTHON" -m src.main post-rqvae --experiment "$EXP" --sid-path "$SID"
provenance "$E" "variant     $VARIANT" "sid         $SID ($(sha256sum "$SID" | cut -c1-12))"

"$PYTHON" -m src.build_1pos4neg_sequences --out-dir "$E/candidate_1pos4neg" \
    --path "$E/post_rqvae/train_sequences.parquet" --path "$E/post_rqvae/validation_sequences.parquet" \
    --report "$E/candidate_1pos4neg/report.json"

"$PYTHON" -m src.shuffle_candidate_order --input-dir "$E/candidate_1pos4neg" \
    --output-dir "$E/candidate_1pos4neg_shuffled" --report "$E/candidate_1pos4neg_shuffled/report.json"

(cd "$YM/Transformer" && "$PYTHON" split_validation.py --input_dir "$E/candidate_1pos4neg_shuffled" --output_dir "$E/split")

"$PYTHON" "$CODE/check_candidates.py" \
    "$E/candidate_1pos4neg_shuffled/train_sequences_1pos4neg.parquet" \
    "$E/split/validation_sequences_1pos4neg_half.parquet" \
    "$E/split/test_sequences_1pos4neg.parquet" | tee "$E/check_candidates.txt"
finish_provenance "$E" 0
echo "== 완료: $E"
