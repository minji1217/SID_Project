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

ARTICLE_SEMANTIC_IDS_PATH = (
    config.ARTICLE_SEMANTIC_IDS_PATH
)

TRAIN_SEQUENCES_PATH = (
    config.TRAIN_SEQUENCES_PATH
)

VALIDATION_SEQUENCES_PATH = (
    config.VALIDATION_SEQUENCES_PATH
)

# STEP 11-2. 최종 Sequence 컬럼 정의
# history에는 현재 impression의 current까지 반영된 seq가 들어감
# current를 별도 컬럼으로 저장하지 않고 history의 마지막 문맥으로 포함
# target은 scalar가 아닌 list

TRAIN_SEQUENCE_COLUMNS = [
    "impression_id",
    "user_id",
    "impression_time",

    "history_article_ids",
    "history_c1",
    "history_c2",
    "history_c3",

    "target_article_ids",

]

VALIDATION_SEQUENCE_COLUMNS = TRAIN_SEQUENCE_COLUMNS + [
    "candidate_article_ids",
    "candidate_c1",
    "candidate_c2",
    "candidate_c3",
    "candidate_labels",
]

# STEP 11-3. Stable dedup 

def _stable_unique_ints(values: list[Any]) -> list[int]:
    """
    리스트의 원순서 유지하면서 중복 ID만 제거 
    예 : [100, 100, 200, 100, 300] -> [100, 200, 300]

    사용 위치
    -------------
    1. article_ids_clicked
    2. validation article_ids_inview

    주의
    ---------------
    running history 전체에 이 함수를 적용하지 않는다.
    running history에선 같은 기사가 나중에 다시 등장하는 것이
    실제 재방문 / 재클릭일 수 있기 떄문임
    """

    result: list[int] = []
    seen: set[int] = set()

    for value in values:
        value_int = int(value)

        if value_int in seen:
            continue 

        seen.add(value_int)
        result.append(value_int)

    return result 

# STEP 11-4. RQ-VAE SID Lookup 생성
def _load_sid_lookup() -> dict[int, tuple[int, int, int]]:
    """
    article_semantic_ids.parquet을 읽어 article_id -> (c1,c2,c3)

    lookup dict로 변환
    예 : article_id | c1 | c2 | c3
        ---------------------------
         100       |  3 | 10 | 8
         200       |  5 | 10 | 21
    -> {
        100: (3, 10, 8),
        200: (5, 10, 21),
        }
    """

    # STEP 11-4-1. SID 파일 존재 여부 확인
    if not ARTICLE_SEMANTIC_IDS_PATH.exists():
        raise FileNotFoundError(
            "article_semantic_ids.parquet 파일이 없습니다. "
            "RQ-VAE Train과 validation frozen inference를 먼저 완료해야 합니다. "
            f"경로={ARTICLE_SEMANTIC_IDS_PATH}"
        )

    # STEP 11-4-2. 필요한 컬럼만 읽고 타입 통일
    semantic_ids = (
        pl.read_parquet(
            ARTICLE_SEMANTIC_IDS_PATH, 
            columns=[
                "article_id","c1","c2","c3"
            ]
        ).with_columns([
            pl.col("article_id").cast(pl.Int64),
            pl.col("c1").cast(pl.Int32),
            pl.col("c2").cast(pl.Int32),
            pl.col("c3").cast(pl.Int32),
        ]).sort("article_id")
    )

    # STEP 11-4-3. 빈 SID 파일 방지 
    if semantic_ids.height == 0:
        raise ValueError("article_semantic_ids.parquet가 비어 있습니다.")

    # STEP 11-4-4. article_id 중복 검사
    duplicate_article_count = (semantic_ids.select(
        pl.col("article_id").is_duplicated().sum().alias("count")
    ).item())

    if duplicate_article_count != 0:
        raise ValueError(
            "article_semantic_ids.parquet에 중복 article_id가 존재합니다. "
            f"중복 행 수={duplicate_article_count}"
        )

    # STEP 11-4-5. 필수 컬럼 null 검사
    for column_name in ["article_id", "c1", "c2", "c3",]:
        null_count = (
            semantic_ids.get_column(column_name).null_count()
        )

        if null_count != 0:
            raise ValueError(
                "article_semantic_ids.parquet에 null 값이 존재합니다. "
                f"column={column_name}, null_count={null_count}"
            )
    # STEP 11-4-6. Python dict 생성
    sid_lookup = {
        int(article_id): (int(c1), int(c2), int(c3),)
     for article_id, c1, c2, c3 in semantic_ids.iter_rows()}

    return sid_lookup

# STEP 11-5. Initial History 준비 
def _prepare_inital_histories(
        history_path: Any, 
        sid_lookup: dict[int, tuple[int, int, int]],
)-> tuple[
    dict[int, list[int]],
    dict[str, Any],
]:
    """
    raw history.parquet을 사용자별 running history 시작 상태로 만든다

    EB-NeRD history 핵심 컬럼
    --------------------------------
    user_id
    article_id_fixed
    impression_time_fixed

    처리규칙
    -------------------------------------------
    1. user_id null 행 제외
    2. duplicate user_id 행 제외
    3. article/time list null 제외
    4. article/time list 길이 불일치 제외
    5. list 내부 null 제외
    6. impression_time_fixed 오름차순 정렬
    7. 같은 시간은 article_id로 deterministic tie-break
    8. initial history 전체에 global dedup은 하지 않음
    9. SID가 없는 history article만 개별 제거

    왜 initial history를 dedup하지 않는가?
    --------------------------------------------------------
    같은 기사가 시간적으로 떨어진 시점에 다시 나타나는 것은
    실제 반복 소비 행동일 수 있기 때문이다.

    current에 대해서도 전체 history membership이 아니라
    "바로 마지막 기사"와만 비교하는 것과 같은 철학이다.
    """

    # STEP 11-5-1. 필요한 history 컬럼 읽기
    history = pl.read_parquet(
        history_path, columns=["user_id", "article_id_fixed", "impression_time_fixed",],
    )

    # STEP 11-5-2. 중복 user_id 목록 생성
    duplicated_user_ids = set(
        history.filter(pl.col("user_id").is_not_null() &
                       pl.col("user_id").is_duplicated()
    ).get_column("user_id").to_list())

    histories: dict[int, list[int]] = {}

    invalid_history_row_count = 0 # user_id, article_id, impression_time_fixed가 null, 두 리스트 길이 안맞는 경우 
    duplicate_user_history_row_count = 0 # 같은 유저가 여러 행에 나온 경우
    reordered_history_row_count = 0 # 시간순이 아니어서 다시 정렬한 행 

    filtered_missing_history_sid_count = 0 # SID 없어서 제거한 기사 수 
    missing_history_sid_examples:list[int] = [] # 그 기사들의 실제 ID 샘플


    # STEP 11-5-3. history 한 행씩 처리
    for row in history.ier_rows(named=True):
        user_id = row["user_id"]
        article_ids = row["article_id_fixed"]
        impression_times = row["impression_time_fixed"]

        if user_id is None:
            invalid_history_row_count += 1
            continue 
        if user_id in duplicated_user_ids:
            duplicate_user_history_row_count += 1
            continue 
        if article_ids is None or impression_times is None:
            invalid_history_row_count += 1
            continue 
        if len(article_ids) != len(impression_times):
            invalid_history_row_count += 1
            continue 
        if any(article_id is None for article_id in article_ids):
            invalid_history_row_count += 1
            continue 
        if any(impression_time is None for impression_time in impression_times):
            invalid_history_row_count += 1
            continue 

        # STEP 11-5-4. (time, article_id) pair 생성
        