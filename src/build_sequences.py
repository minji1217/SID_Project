from __future__ import annotations

from pprint import pprint
from typing import Any

import polars as pl

from src import config

# STEP 11. Transformer Sequence build
# 목적
# --------------------
# RQ-VAE가 모든 필요한 기사에 대해 SID를 만든 뒤, article_id -> (c1,c2,c3)
# 정보를 실제 사용자 행동 순서와 결합해 transformer가 학습/평가에 사용할 seq 데이터 생성

# 실행시점
# -------------------
# 아래 과정이 모두 끝난 뒤 실행
# 1. Train / Validation article 전처리 완료
# 2. RQ-VAE Train 완료
# 3. validation-only 기사에 대한 frozen inference 완료
# 4. 최종 article_semantic_ids.parquet 생성 완료

# 입력
# -----------------
# 1. article_semantic_ids.parquet
#         article_id | c1 | c2 | c3 
# 2. train/history.parquet
# 3. train/behaviors.parquet
# 4. validation/history.parquet
# 5. validation/behiavors.parquet

# 출력 
# ------------
# 1. train_sequences.parquet
# 2. validation_sequences.parquet


# Running History 정책
# ---------------------------------- 
# Train / Validation running history는 서로 완전히 별개로 구성한다.
#
# 각 split에서:
#
#   history.parquet의 과거 기사
#           ↓
#   user별 initial running_history
#           ↓
#   behaviors를 user_id -> impression_time -> impression_id 순으로 처리
#
# EB-NeRD behaviors.article_id는 해당 impression이 발생했을 때
# 사용자가 현재 보고 있던 기사(current)를 의미한다.
#
# current가 None이면 추가하지 않는다.
#
# current가 존재하면 running history의 "바로 마지막 기사"와만 비교한다.
#
# 예 1)
#   running = [10, 20, 30]
#   current = 30
#
#   -> 바로 직전과 같음
#   -> current 추가 SKIP
#
# 이유:
# 이전 impression의 clicked target이 다음 impression의 current가 되는
# 연속 중복을 한 번 더 넣지 않기 위해서다.
#
# 예 2)
#   running = [10, 20, 30]
#   current = 20
#
#   -> 과거에는 20이 있었지만 바로 직전은 30
#   -> current 20을 다시 추가
#   -> [10, 20, 30, 20]
#
# 이유:
# 사용자가 과거에 봤던 기사를 나중에 다시 방문했을 수 있기 때문이다.
# 따라서 running history 전체를 대상으로 global dedup은 하지 않는다.

# clicked target 처리 규칙
# article_ids_clicked는 한 impression 내부에서 stable dedup한다.
#
# 예:
#   [100, 100, 200, 100]
#   -> [100, 200]
#
# 하지만 sample 생성 후 running history에 target을 넣을 때는
# 과거 history에 동일 기사가 있더라도 조건 없이 append한다.
#
# 이유:
# target은 "이번 impression에서 실제로 새로 발생한 클릭 행동"이므로
# 과거에 같은 기사를 클릭했더라도 이번 클릭은 새로운 이벤트다.
#
#
#  Target Leakage 방지
# ------------------------------------------------------------
# 현재 clicked target은 Transformer input snapshot을 만든 뒤에만
# running history에 추가한다.
#
# 예:
#   기존 running = [10, 20]
#   current       = 30
#   targets       = [40, 50]
#
# current 처리 후:
#   input history = [10, 20, 30]
#   targets       = [40, 50]
#
# sample 생성 후:
#   running = [10, 20, 30, 40, 50]
#
# 절대로 현재 input에 [40, 50]을 미리 넣지 않는다.
#
#
#  Multi-target 정책
# ------------------------------------------------------------
# 한 impression에서 clicked target이 여러 개일 수 있다.
#
#   target_article_ids = [100, 200]
#
# Transformer 예측이 target set 안에 존재하면 정답으로 판단한다.
# 실제 multi-positive loss는 Transformer 구현 단계에서 결정한다.
#
#
#  Validation Candidate 정책
# ------------------------------------------------------------
# article_ids_inview도 stable dedup한다.
# 모든 target이 candidate 안에 있어야 ranking sample을 만든다.
#
# 예:
#   targets    = [100, 300]
#   candidates = [100, 200, 300, 400]
#
#   candidate_labels = [1, 0, 1, 0]

