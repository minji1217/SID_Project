#!/usr/bin/env bash
# 여민 코드(baekyeomin/SID_Project @ 22028ba)를 ~/minji 아래에 꺼낸다. 이미 있으면 커밋만 확인한다.
#   bash experiments/v2_yeomin_rerun/0_setup.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

if [[ ! -d "$YM" ]]; then
    git clone -q https://github.com/baekyeomin/SID_Project "$YM"
fi
git -C "$YM" fetch -q origin main
git -C "$YM" checkout -q --detach "$YM_REF"
check_yeomin_code
echo "여민 코드: $YM ($(git -C "$YM" log -1 --format='%h %ad %s' --date=short))"

# UNI gin 원본이 들어 있는 커밋을 받아 둔다 (1단계에서 씀)
git -C "$REPO_ROOT" fetch -q origin uniqueness-loss-experiment
git -C "$REPO_ROOT" cat-file -e "$UNI_REF:RQVAE/configs/rqvae_ebnerd_wd001_uni.gin" && echo "UNI gin 원본: $UNI_REF 확인"

check_inputs
"$PYTHON" - "$INPUT" <<'EOF'
import sys, pandas as pd
d = sys.argv[1]
tr = pd.read_parquet(f"{d}/article_master.parquet"); va = pd.read_parquet(f"{d}/validation_article_master.parquet")
print(f"입력: train 기사 {len(tr):,} · validation-only {len(va):,} · train 사건 {tr.event_id.nunique():,}")
if (len(tr), len(va), tr.event_id.nunique()) != (10419, 2741, 9794):
    print("주의: 기대값(10,419 / 2,741 / 9,794)과 다릅니다. PRE_DIR 을 확인하세요.")
EOF
echo "결과 폴더: $V2"
