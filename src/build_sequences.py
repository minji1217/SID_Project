from __future__ import annotations

from pprint import pprint
from typing import Any

import polars as pl

from src import config

# ============================================================
# STEP 11. Transformer Sequence Build
# ============================================================
#
# 목적
# ------------------------------------------------------------
# RQ-VAE가 필요한 모든 기사에 대해 최종 Semantic ID를 만든 뒤,
#
#     article_id -> (c1, c2, c3)
#
# 매핑을 EB-NeRD의 history / behaviors와 결합하여
# Transformer 학습/평가용 사용자 sequence를 만든다.
#
#
# 실행 시점
# ------------------------------------------------------------
# 아래가 모두 끝난 뒤 실행한다.
#
# 1. Train 전처리 완료
# 2. Validation 전처리 완료
# 3. RQ-VAE Train 완료
# 4. Validation-only 기사 Frozen Inference 완료
# 5. article_semantic_ids.parquet 생성 완료
#
#
# 입력
# ------------------------------------------------------------
# 1. article_semantic_ids.parquet
#       article_id | c1 | c2 | c3
#
# 2. train/history.parquet
# 3. train/behaviors.parquet
# 4. validation/history.parquet
# 5. validation/behaviors.parquet
#
#
# 출력
# ------------------------------------------------------------
# 1. train_sequences.parquet
# 2. validation_sequences.parquet
#
#
# ============================================================
# ★ 최종 Running History 정책
# ============================================================
#
# Train / Validation running history는 서로 완전히 별개다.
#
# 각 split에서:
#
#   history.parquet
#       ↓
#   user별 initial running_history 구성
#       ↓
#   behaviors를
#   user_id -> impression_time -> impression_id 순으로 처리
#
#
# 각 behavior에서는:
#
#   기존 running history
#       ↓
#   current(article_id) 처리
#       ↓
#   current까지 반영된 history를
#   현재 Transformer input으로 snapshot
#       ↓
#   clicked target은 정답으로 별도 저장
#       ↓
#   sample 생성
#       ↓
#   clicked target들을 running history에 추가
#       ↓
#   같은 user의 다음 impression
#
#
# ------------------------------------------------------------
# 1. Initial History 정책
# ------------------------------------------------------------
#
# history.parquet의
#
#   article_id_fixed
#   impression_time_fixed
#
# 는 같은 index끼리 하나의 pair로 본다.
#
# 예:
#
#   article_id_fixed      = [100, None, 300, 400]
#   impression_time_fixed = [10:00, 11:00, None, 13:00]
#
# 사용 가능한 pair:
#
#   (100, 10:00)
#   (400, 13:00)
#
# 따라서 내부 null 때문에 user history 전체를 버리지 않고
# 문제 있는 pair만 제거한다.
#
# 단,
#
# - user_id null
# - duplicate user_id
# - history list 자체 null
# - article/time list 길이 불일치
#
# 인 history 행은 initial history로 사용하지 않는다.
# 해당 user의 정상 behavior가 존재하면 빈 history에서 시작할 수 있다.
#
# History 전체 global dedup은 하지 않는다.
# 같은 기사가 시간적으로 떨어져 다시 등장하는 것은
# 실제 재방문/재클릭일 수 있기 때문이다.
#
#
# ------------------------------------------------------------
# 2. Current(article_id) 정책
# ------------------------------------------------------------
#
# current는 behavior가 발생했을 때 사용자가 보고 있던 기사다.
#
# current == None
#   -> behavior 행은 유지
#   -> current만 running history에 추가하지 않는다.
#
# current가 존재하면
# running_history 전체를 검색하지 않고
# "바로 마지막 article_id"와만 비교한다.
#
# 예:
#
# running = [10, 20, 30]
# current = 30
#
# -> 직전과 같으므로 skip
#
# 이유:
# 이전 impression의 clicked target이 다음 impression의 current가 되는
# 연속 중복을 한 번 더 넣지 않기 위해서다.
#
#
# 반대로:
#
# running = [10, 20, 30]
# current = 20
#
# -> 과거에는 20이 있었지만 마지막은 30
# -> current 20 append
#
# 결과:
#
# [10, 20, 30, 20]
#
# 즉 "연속 중복만 제거"하고 재방문은 보존한다.
#
#
# ------------------------------------------------------------
# 3. Clicked Target 정책
# ------------------------------------------------------------
#
# article_ids_clicked 내부 null은 해당 원소만 제거한다.
#
# 예:
#   [100, None, 200]
#   -> [100, 200]
#
# 이후 한 impression 내부에서는 stable dedup한다.
#
# 예:
#   [100, 100, 200, 100]
#   -> [100, 200]
#
# stable dedup 후 target이 하나 이상이면 사용한다.
#
# 여러 target:
#
#   [100, 200]
#
# 도 그대로 유지한다.
#
# Transformer에서는 prediction이 target set 안에 있으면
# 정답으로 판단할 수 있도록 downstream에서 multi-positive objective를 적용한다.
#
#
# ★ 중요한 차이:
#
# current:
#   running history의 "바로 직전"과 같으면 skip
#
# target:
#   실제 이번 impression에서 발생한 새 클릭 행동이므로
#   과거 history에 같은 article이 있어도 무조건 append
#
#
# ------------------------------------------------------------
# 4. Target Leakage 방지
# ------------------------------------------------------------
#
# 현재 clicked target은 현재 Transformer input에 미리 넣지 않는다.
#
# 예:
#
# 기존 running = [10, 20]
# current       = 30
# clicked       = [40, 50]
#
# current 처리:
#
#   [10, 20, 30]
#
# 현재 sample:
#
#   history = [10, 20, 30]
#   target  = [40, 50]
#
# sample 생성 후에만:
#
#   running = [10, 20, 30, 40, 50]
#
#
# ------------------------------------------------------------
# 5. Validation Candidate 정책
# ------------------------------------------------------------
#
# article_ids_inview:
#
# - list 자체 null/empty -> ranking sample 제외
# - 내부 null -> null 원소만 제거
# - stable dedup 수행
#
# 모든 target이 candidate 안에 있어야 한다.
#
# 예:
#
# targets    = [100, 300]
# candidates = [100, 200, 300, 400]
#
# candidate_labels:
#
# [1, 0, 1, 0]
#
# candidate 문제 때문에 현재 ranking sample을 만들지 못하더라도,
# target SID가 정상이라면 실제 클릭 행동은 발생했으므로
# target은 다음 running history 진행에 반영한다.
#
#
# ------------------------------------------------------------
# 6. 여기서 하지 않는 것
# ------------------------------------------------------------
#
# - Padding
# - Truncation
# - Attention Mask
#
# 위 처리는 Transformer Dataset / DataLoader 단계에서 수행한다.
# ============================================================


# ============================================================
# STEP 11-1. 최종 Sequence Schema
# ============================================================
#
# 경로는 config.py에서 중앙 관리한다.
#
# config.py에 필요한 경로:
#
# ARTICLE_SEMANTIC_IDS_PATH
# TRAIN_SEQUENCES_PATH
# VALIDATION_SEQUENCES_PATH
#
#
# target은 이제 scalar가 아니라 List다.

ARTICLE_SEMANTIC_IDS_PATH = (
    config.ARTICLE_SEMANTIC_IDS_PATH
)

TRAIN_SEQUENCES_PATH = (
    config.TRAIN_SEQUENCES_PATH
)

VALIDATION_SEQUENCES_PATH = (
    config.VALIDATION_SEQUENCES_PATH
)


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
    "target_c1",
    "target_c2",
    "target_c3",

]

VALIDATION_SEQUENCE_COLUMNS = TRAIN_SEQUENCE_COLUMNS + [
    "candidate_article_ids",
    "candidate_c1",
    "candidate_c2",
    "candidate_c3",
    "candidate_labels",
]

# STEP 11-2. Stable dedup 

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

# STEP 11-3. RQ-VAE SID Lookup 생성
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

    # STEP 11-3-1. SID 파일 존재 여부 확인
    if not ARTICLE_SEMANTIC_IDS_PATH.exists():
        raise FileNotFoundError(
            "article_semantic_ids.parquet 파일이 없습니다. "
            "RQ-VAE Train과 validation frozen inference를 먼저 완료해야 합니다. "
            f"경로={ARTICLE_SEMANTIC_IDS_PATH}"
        )

    # STEP 11-3-2. 필요한 컬럼만 읽고 타입 통일
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

    # STEP 11-3-3. 빈 SID 파일 방지 
    if semantic_ids.height == 0:
        raise ValueError("article_semantic_ids.parquet가 비어 있습니다.")

    # STEP 11-3-4. article_id 중복 검사
    duplicate_article_count = (semantic_ids.select(
        pl.col("article_id").is_duplicated().sum().alias("count")
    ).item())

    if duplicate_article_count != 0:
        raise ValueError(
            "article_semantic_ids.parquet에 중복 article_id가 존재합니다. "
            f"중복 행 수={duplicate_article_count}"
        )

    # STEP 11-3-5. 필수 컬럼 null 검사
    for column_name in ["article_id", "c1", "c2", "c3",]:
        null_count = (
            semantic_ids.get_column(column_name).null_count()
        )

        if null_count != 0:
            raise ValueError(
                "article_semantic_ids.parquet에 null 값이 존재합니다. "
                f"column={column_name}, null_count={null_count}"
            )
    # STEP 11-3-6. Python dict 생성
    sid_lookup = {
        int(article_id): (int(c1), int(c2), int(c3),)
     for article_id, c1, c2, c3 in semantic_ids.iter_rows()}

    return sid_lookup

# STEP 11-4. Initial History 준비 
def _prepare_inital_histories(
        history_path: Any, 
        sid_lookup: dict[int, tuple[int, int, int]],
)-> tuple[
    dict[int, list[int]],
    dict[str, Any],
]:
    """
    raw history.parquet을 각 사용자의 running history 시작 상태로 만든다.

    처리 정책
    --------------------------------------------------------
    1. user_id null
       -> 해당 history 행 사용 안 함

    2. 동일 user_id가 여러 history 행에 존재
       -> 해당 user의 initial history 사용 안 함

    3. article/time list 자체 null
       -> 해당 history 행 사용 안 함

    4. article/time list 길이 불일치
       -> pair 대응 관계가 깨졌으므로 해당 history 행 사용 안 함

    5. 내부 article_id/time null
       -> 그 pair만 제거

    6. 남은 유효 pair를 impression_time_fixed 오름차순 정렬

    7. 같은 시간인 pair는 Python stable sort에 의해
       원래 list의 상대 순서를 그대로 유지

    8. History 전체 global dedup은 하지 않음

    9. SID가 없는 history article은 그 item만 제거

    history 행을 사용하지 못한 user라도
    이후 정상 behavior가 있으면 []에서 running history를 시작할 수 있다.
    """

    # STEP 11-4-1. 필요한 history 컬럼 읽기
    history = pl.read_parquet(
        history_path, columns=["user_id", "article_id_fixed", "impression_time_fixed",],
    )

    # STEP 11-4-2. 중복 user_id 목록 생성
    duplicated_user_ids = set(
        history.filter(pl.col("user_id").is_not_null() &
                       pl.col("user_id").is_duplicated()
    ).get_column("user_id").to_list())

    histories: dict[int, list[int]] = {}

    invalid_history_row_count = 0 # user_id, article_id, impression_time_fixed가 null, 두 리스트 길이 안맞는 경우 
    duplicate_user_history_row_count = 0 # 같은 유저가 여러 행에 나온 경우
    reordered_history_row_count = 0 # 시간순이 아니어서 다시 정렬한 행 
    removed_null_history_pair_count = 0

    filtered_missing_history_sid_count = 0 # SID 없어서 제거한 기사 수 
    missing_history_sid_examples:list[int] = [] # 그 기사들의 실제 ID 샘플


    # STEP 11-5-3. history 한 행씩 처리
    for row in history.iter_rows(named=True):
        user_id = row["user_id"]
        article_ids = row["article_id_fixed"]
        impression_times = row["impression_time_fixed"]

        # user 식별 불가
        if user_id is None:
            invalid_history_row_count += 1
            continue 
        # 어떤 history 행이 기준인지 결정 불가 
        if user_id in duplicated_user_ids:
            duplicate_user_history_row_count += 1
            continue 
        # list 구조 자체사용 불가
        if article_ids is None or impression_times is None:
            invalid_history_row_count += 1
            continue 
        # article/time 1:1 대응 관계가 깨짐 
        if len(article_ids) != len(impression_times):
            invalid_history_row_count += 1
            continue 

        # STEP 11-4-4. 내부 null pair만 제거
        # 각 pair는 (impression_time, article_id) 형태로 저장한다.
        # 이후 timestamp 기준으로 정렬하며,
        # Python sorted()는 stable sort이므로 같은 timestamp의 원래 상대 순서는 유지된다.
        valid_pairs: list[tuple[Any, int]] = []

        for article_id, impression_time in zip(article_ids, impression_times):
            if article_id is None or impression_time is None : 
                removed_null_history_pair_count += 1
                continue 

            valid_pairs.append((impression_time, int(article_id)))

        # STEP 11-4-5. 시간순 정렬 
        # python sorted()는 stable sort이므로 
        # 같은 timestamp라면 원래 list의 순서를 유지한다. 

        sorted_pairs = sorted(valid_pairs, key=lambda pair: pair[0])

        if valid_pairs != sorted_pairs: 
            reordered_history_row_count += 1

        # STEP 11-4-6. SID 없는 history article item만 제거
        filtered_article_ids : list[int] = []

        for _, article_id in sorted_pairs:
            if article_id not in sid_lookup:
                filtered_missing_history_sid_count += 1

                if len(missing_history_sid_examples) < 10 and article_id not in missing_history_sid_examples:
                    missing_history_sid_examples.append(article_id)
                continue 

            # global dedup 하지 않고 실제 반복 행동 보존
            filtered_article_ids.append(article_id)


        histories[int(user_id)] = filtered_article_ids

    # STEP 11-4-7. 결과 반환
    return histories, {
        "history_row_count": int(
            history.height
        ),
        "history_user_count": int(
            len(histories)
        ),
        "invalid_history_row_count": int(
            invalid_history_row_count
        ),
        "duplicate_user_history_row_count": int(
            duplicate_user_history_row_count
        ),
        "removed_null_history_pair_count": int(
            removed_null_history_pair_count
        ),
        "reordered_history_row_count": int(
            reordered_history_row_count
        ),
        "filtered_missing_history_sid_count": int(
            filtered_missing_history_sid_count
        ),
        "missing_history_sid_examples": (
            missing_history_sid_examples
        ),
    }

# STEP 11-5. Behavior 읽기 / 사용자 시간순 정렬

def _load_behaviors(
        behaviors_path: Any, 
        split_name: str, 
)-> tuple[pl.DataFrame, set[int]]:
    """
    behaviors.parquet을 seq 생성 순서로 준비
    정렬 : user_id -> impression_time -> impression_id 

    같은 user의 이전 behavior 결과를 다음 behavior의 running history에 계속 누적하기 위해 필요하다. 
    validation에선 ranking 평가를 위해 article_ids_inview도 추가로 읽는다. 
    
    """
    # STEP 11-5-1. Schema 확인
    schema = pl.scan_parquet(behaviors_path).collect_schema()

    column_names = set(schema.names())

    required_columns = {
        "impression_id", "user_id", "impression_time", "article_id", "article_ids_clicked"
    }

    if split_name == "validation":
        required_columns.add("article_ids_inview")

    missing_columns = required_columns - column_names 
    if missing_columns:
        raise ValueError(
            f"{split_name} behaviors에 필요한 컬럼이 없습니다: "
                        + ", ".join(
                            sorted(missing_columns)
                        )
        )

    # STEP 11-5-2. 필요한 컬럼만 읽기
    select_columns = [
        "impression_id", "user_id", "impression_time", "article_id", "article_ids_clicked"
    ]

    if split_name == "validation":
        select_columns.append("article_ids_inview")

    behaviors = pl.read_parquet(
        behaviors_path, columns = select_columns,
    )

    # STEP 11-5-3. 중복 impression_id 탐지 
    duplicated_impression_ids = set(
        behaviors.filter(pl.col("impression_id").is_not_null() 
                         &
                         pl.col("impression_id").is_duplicated())
    ).get_column("impression_id").to_list()

    # STEP 11-5-4. user - time - id 순 정렬 
    # impression time에 null 섞여있으면 맨 뒤로 
    behaviors = behaviors.sort(
        ["user_id", "impression_time", "impression_id"], nulls_last=  True, 
    )

    return (behaviors, duplicated_impression_ids)

# STEP 11-6. Article ID list -> SID list 변환
def _article_ids_to_codes(
        article_ids: list[int],
        sid_lookup: dict[int, tuple[int, int, int]],
)-> tuple[list[int], list[int], list[int]]:
    """
    article_id list를 동일 순서의 c1/c2/c3 리스트로 변환
    
    예:
        article_ids = [100, 200]

        100 -> (1, 10, 7)
        200 -> (2, 10, 9)

    결과:
        c1 = [1, 2]
        c2 = [10, 10]
        c3 = [7, 9]
    
    
    """

    c1_list: list[int] = []
    c2_list: list[int] = []
    c3_list: list[int] = []

    for article_id in article_ids: 
        c1,c2,c3 = sid_lookup[article_id]
        c1_list.append(c1);c2_list.append(c2);c3_list.append(c3)

    return (c1_list, c2_list, c3_list)


# STEP 11-7. Current를 Running history에 반영
def _append_current_if_needed(
        running_history: list[int],
        current_article_id: Any,
        sid_lookup: dict[int, tuple[int, int, int]],
)-> tuple[ str, int | None]:
    """
    current article를 현재 user의 running history에 반영

    반환 status
    -------------------------------
    NULL : current가 None -> current만 추가 x
    MISSING_SID : current article의 SID X -> current만 추가 x
    SKIPPED_CONSECUTIVE_DUPLICATE : current == running_history[-1] -> 연속 중복이므로 추가 x
    APPENDED : 바로 직전과 다름 -> 과거 history 어디엔가 같은 article이 있어도 append

    핵심
    -----------------
    global 검사 x -> if current not in running_history 와 같은 검사 x
    오직 바로 직전과만 검사 ㅇ
    
    """

    # current null은 EB-NERD에서 허용함
    if current_article_id is None : return ("NULL", None)
    current_article_id_int = int(current_article_id)

    # current를 SID seq로 표현 못하면 input에선 제외하고 behavior 자체는 살림
    if current_article_id_int not in sid_lookup:
        return ("MISSING_SID", current_article_id_int)

    # 연속 중복만 제거 
    if len(running_history) > 0 and running_history[-1] == current_article_id_int:
        return ("SKIPPED_CONSECUTIVE_DUPLICATE", current_article_id_int)

    # 과거 somewhere에 동일 article이 있어도 바로 직전과 다르면 재방문으로 보고 append
    running_history.append(current_article_id_int)

    return ("APPENDED", current_article_id_int)


# STEP 11-8. Python row -> polars seq df
def _make_sequence_df(rows: list[dict[str, Any]], split_name :str)-> pl.DataFrame:
    """
    생성된 python dict rows를 최종 parquet schema로 변환
    target은 multi 정책으로 list 타입
    """

    if split_name == "train":
        columns = TRAIN_SEQUENCE_COLUMNS
    else:
        columns = VALIDATION_SEQUENCE_COLUMNS

    # STEP 11-8-1. 0행이어도 schema 유지
    if not rows:
        schema: dict[str, Any] = {
            "impression_id": pl.Int64,
            "user_id": pl.Int64,
            "impression_time": (
                pl.Datetime(
                    time_unit="us"
                )
            ),

            "history_article_ids": (
                pl.List(pl.Int64)
            ),
            "history_c1": (
                pl.List(pl.Int32)
            ),
            "history_c2": (
                pl.List(pl.Int32)
            ),
            "history_c3": (
                pl.List(pl.Int32)
            ),

            "target_article_ids": (
                pl.List(pl.Int64)
            ),
            "target_c1": (
                pl.List(pl.Int32)
            ),
            "target_c2": (
                pl.List(pl.Int32)
            ),
            "target_c3": (
                pl.List(pl.Int32)
            ),
        }

        if split_name == "validation":
            schema.update({
                "candidate_article_ids": (
                    pl.List(pl.Int64)
                ),
                "candidate_c1": (
                    pl.List(pl.Int32)
                ),
                "candidate_c2": (
                    pl.List(pl.Int32)
                ),
                "candidate_c3": (
                    pl.List(pl.Int32)
                ),
                "candidate_labels": (
                    pl.List(pl.Int32)
                ),
            })

        return (
            pl.DataFrame(
                schema=schema
            )
            .select(columns)
        )

    # STEP 11-8-2. 실제 rows -> DataFrame
    sequence_df = (
        pl.DataFrame(rows)
        .with_columns([
            pl.col("impression_id")
            .cast(pl.Int64),

            pl.col("user_id")
            .cast(pl.Int64),

            pl.col("impression_time")
            .cast(
                pl.Datetime(
                    time_unit="us"
                )
            ),

            pl.col("history_article_ids")
            .cast(
                pl.List(pl.Int64)
            ),

            pl.col("history_c1")
            .cast(
                pl.List(pl.Int32)
            ),

            pl.col("history_c2")
            .cast(
                pl.List(pl.Int32)
            ),

            pl.col("history_c3")
            .cast(
                pl.List(pl.Int32)
            ),

            pl.col("target_article_ids")
            .cast(
                pl.List(pl.Int64)
            ),

            pl.col("target_c1")
            .cast(
                pl.List(pl.Int32)
            ),

            pl.col("target_c2")
            .cast(
                pl.List(pl.Int32)
            ),

            pl.col("target_c3")
            .cast(
                pl.List(pl.Int32)
            ),
        ])
    )

    # STEP 11-8-3. Validation candidate 컬럼 타입 통일
    if split_name == "validation":
        sequence_df = (
            sequence_df
            .with_columns([
                pl.col(
                    "candidate_article_ids"
                )
                .cast(
                    pl.List(pl.Int64)
                ),

                pl.col(
                    "candidate_c1"
                )
                .cast(
                    pl.List(pl.Int32)
                ),

                pl.col(
                    "candidate_c2"
                )
                .cast(
                    pl.List(pl.Int32)
                ),

                pl.col(
                    "candidate_c3"
                )
                .cast(
                    pl.List(pl.Int32)
                ),

                pl.col(
                    "candidate_labels"
                )
                .cast(
                    pl.List(pl.Int32)
                ),
            ])
        )

    return sequence_df.select(
        columns
    )

# STEP 11-9. 생성된 seq 정합성 검사
def _validate_sequence_integrity(sequence_df:pl.DataFrame, split_name:str)-> None:
    """
    만들어진 seq의 기본 list 정합성 검사 
    
    History:
        len(history_article_ids)
        == len(history_c1)
        == len(history_c2)
        == len(history_c3)

    Target:
        len(target_article_ids)
        == len(target_c1)
        == len(target_c2)
        == len(target_c3)
        >= 1

    Validation Candidate:
        len(candidate_article_ids)
        == len(candidate_c1)
        == len(candidate_c2)
        == len(candidate_c3)
        == len(candidate_labels)

    validation candidate_labels에는 최소 하나 이상의 positive가 있어야함
    """

    if sequence_df.height == 0: return 

    # STEP 11-9-1. History list 길이 검사
    # 연산자 중 하나라도 true면 true 
    # 조건에 맞는 행만 남긴 표에서 그 표의 행 수 
    invalid_history_length_count = sequence_df.filter(
        (pl.col("history_article_ids").list.len() != pl.col("history_c1").list.len()) | 
        (pl.col("history_article_ids").list.len() != pl.col("history_c2").list.len()) |
        (pl.col("history_article_ids").list.len() != pl.col("history_c3").list.len())
    ).height 

    if invalid_history_length_count != 0:
        raise ValueError(
            f"{split_name} sequence의 history list 길이가 일치하지 않습니다. "
            f"문제 행 수={invalid_history_length_count}"
        )

    # STEP 11-9-2. Target list 길이 검사
    invalid_target_length_count = sequence_df.filter(
        (pl.col("target_article_ids").list.len() != pl.col("target_c1").list.len()) | 
        (pl.col("target_article_ids").list.len() != pl.col("target_c2").list.len()) |
        (pl.col("target_article_ids").list.len() != pl.col("target_c3").list.len())
    ).height 

    if invalid_target_length_count != 0:
        raise ValueError(
            f"{split_name} sequence의 target list 길이가 일치하지 않습니다. "
            f"문제 행 수={invalid_target_length_count}"
        )

    # STEP 11-9-3. Validation candidate 정합성 검사
    if split_name == "validation":
        invalid_candidate_length_count = sequence_df.filter(
        (pl.col("candidate_article_ids").list.len() != pl.col("candidate_c1").list.len()) | 
        (pl.col("candidate_article_ids").list.len() != pl.col("candidate_c2").list.len()) |
        (pl.col("candidate_article_ids").list.len() != pl.col("candidate_c3").list.len()) |
        (pl.col("candidate_article_ids").list.len() != pl.col("candidate_labels").list.len())
    ).height 

        if invalid_candidate_length_count != 0:
            raise ValueError(
                "validation sequence의 candidate list 길이가 일치하지 않습니다. "
                f"문제 행 수={invalid_candidate_length_count}"
            )

        no_positive_candidate_count = sequence_df.filter(
            pl.col("candidate_labels").list.sum() <= 0
        ).height 

        if no_positive_candidate_count != 0:
            raise ValueError(
                "validation sequence에 positive candidate가 없는 행이 존재합니다. "
                f"문제 행 수={no_positive_candidate_count}"
            )

# STEP 11-10. Train / validation 공통 seq 생성 
def _build_split_sequences(
        *, 
        split_name: str, 
        history_path: Any, 
        behaviors_path : Any, 
        output_path: Any, 
        sid_lookup: dict[int, tuple[int, int, int]],
)-> dict[str, Any]:
    """
    train 또는 validation 한 split의 seq 생성

    핵심 순서 
    -----------------------------
    1. history.parquet에서 user별 initial running history 생성
    2. behaviors를 user/time 순서로 처리
    3. clicked 내부 null 제거 
    4. clicked stable dedup
    5. current를 바로 직전 기사와만 비교해 history 반영
    6. current까지 반영된 history를 현재 input으로 snapshot
    7. target은 정답으로 별도 저장
    8. sample 생성 후 target은 조건 없이 running history에 append
    9. 같은 user의 다음 behavior에서 이전 running history 계속 사용

    train과 validation은 이 함수를 각각 별도로 호출하므로
    running history 상태는 서로 공유하지 않는다.

    """

    # STEP 11-10-1. split별 initial history 준비 
    running_histories, history_stats = (
        _prepare_inital_histories(
            history_path=history_path,sid_lookup=sid_lookup,
        )
    )

    # STEP 11-10-2. behavior 읽기 /정렬
    behaviors, duplicated_impression_ids = _load_behaviors(behaviors_path=behaviors_path, split_name=split_name)

    rows: list[dict[str, Any]]=[]

    # 공통 통계 
    structurally_usable_behavior_row_count = 0
    invalid_behavior_row_count = 0
    duplicate_impression_row_count = 0

    clicked_null_element_row_count = 0
    duplicate_clicked_behavior_row_count = 0

    single_target_behavior_row_count = 0
    multi_target_behavior_row_count = 0

    current_null_count = 0
    appended_current_count = 0
    skipped_consecutive_current_count = 0

    missing_current_sid_count = 0
    missing_current_sid_examples: list[int] = []

    missing_target_sid_sample_count = 0
    missing_target_sid_article_count = 0
    missing_target_sid_examples: list[int] = []

    appended_target_article_count = 0

    # Validation 전용 통계
   
    candidate_null_or_empty_count = 0
    candidate_null_element_row_count = 0
    duplicate_candidate_behavior_row_count = 0

    target_not_in_candidates_sample_count = 0
    target_not_in_candidates_article_count = 0
    target_not_in_candidates_examples: list[int] = []

    missing_candidate_sid_sample_count = 0
    missing_candidate_sid_article_count = 0
    missing_candidate_sid_examples: list[int] = []

    # candidate 문제로 sequence row는 만들지 못했지만
    # target은 실제 클릭이므로 running history만 진행한 행 수
    # 원래 정상이라면, target도 candidate에 있어서 평가 샘플을 생성해서 running history 개선
    # 근데 candidate가 null이거나 비어있거나 target이 candidate 안에 없거나 하면
    # -> 평가 샘플을 못만들지만 사용자가 클릭한 건 사실이기에 running history엔 추가 
    history_only_update_count = 0

    # STEP 11-10-3. Behavior 한 행씩 시간순 처리
    for row in behaviors.iter_rows(
        named=True
    ):
        impression_id = row["impression_id"]
        user_id = row["user_id"]
        impression_time = row["impression_time"]
        current_article_id = row["article_id"]
        clicked_article_ids = row["article_ids_clicked"]

        # STEP 11-10-3-1. Behavior 기본 구조 검사 
        if impression_id is None:
            invalid_behavior_row_count += 1
            continue

        if (
            impression_id
            in duplicated_impression_ids
        ):
            duplicate_impression_row_count += 1
            continue

        if user_id is None:
            invalid_behavior_row_count += 1
            continue

        if impression_time is None:
            invalid_behavior_row_count += 1
            continue

        # STEP 11-10-3-2. Clicked list 자체 검사 
        if clicked_article_ids is None:
            invalid_behavior_row_count += 1
            continue

        if len(clicked_article_ids) == 0:
            invalid_behavior_row_count += 1
            continue

        # STEP 11-10-3-3. Clicked 내부 null 원소만 제거 
        if any(
            article_id is None
            for article_id
            in clicked_article_ids
        ):
            clicked_null_element_row_count += 1

        valid_clicked_article_ids = [
            article_id
            for article_id
            in clicked_article_ids
            if article_id is not None
        ]

        # null 제거 후 target이 하나도 없으면 사용 불가
        if len(valid_clicked_article_ids) == 0:
            invalid_behavior_row_count += 1
            continue

        # STEP 11-10-3-4. Clicked stable dedup
        target_article_ids = _stable_unique_ints(valid_clicked_article_ids)

        if len(target_article_ids) != len(valid_clicked_article_ids):
            duplicate_clicked_behavior_row_count += 1

        if len(target_article_ids) == 0:
            invalid_behavior_row_count += 1
            continue 

        structurally_usable_behavior_row_count += 1

        if len(target_article_ids) == 1:
            single_target_behavior_row_count += 1
        else:
            multi_target_behavior_row_count += 1

        impression_id_int = int(impression_id)
        user_id_int = int(user_id)

        # STEP 11-10-3-5. 현재 user running history 가져오기
        # history.parquet에서 initial history를 만들지 못했거나
        # 원래 history가 없던 user라면 []에서 시작
        running_history = running_histories.setdefault(user_id_int,[])

        # STEP 11-10-3-6. Current 처리
        # current는 전체 history 검사 x
        # 바로 직전이랑만 같을 때 연속 중복으로 skip
        current_status, current_id_int = _append_current_if_needed(
            running_history=running_history, current_article_id=current_article_id,sid_lookup=sid_lookup
        )

        if current_status == "NULL":
            current_null_count += 1

        elif current_status == "APPENDED":
            appended_current_count += 1

        elif (
            current_status
            ==
            "SKIPPED_CONSECUTIVE_DUPLICATE"
        ):
            skipped_consecutive_current_count += 1

        elif current_status == "MISSING_SID":
            missing_current_sid_count += 1

            if (current_id_int is not None and len(missing_current_sid_examples) < 10
                and current_id_int not in missing_current_sid_examples):
                missing_current_sid_examples.append(
                    current_id_int
                )
        # STEP 11-10-3-7. Target SID 존재 여부 검사 
        # multi-target 중 하나라도 SID 없으면 완전한 정답 set을 만들 수 없기에
        # 현재 sample은 생성 x
        missing_target_ids = [target_article_id for target_article_id in target_article_ids if target_article_id not in sid_lookup]

        if missing_target_ids:
            missing_target_sid_sample_count += 1
            missing_target_sid_article_count += len(missing_target_ids)

            for missing_id in missing_target_ids:
                if len(missing_target_sid_examples) >= 10:
                    break 
                if missing_id not in missing_target_sid_examples:
                    missing_target_sid_examples.append(missing_id)

            # SID가 없는 target은 이후 SID history에도 넣을 수 없기에 target append도 x
            continue 

        # STEP 11-10-3-8. Current까지 반영된 history 스냅샷
        # 현재 transformer의 input은 여기까지 들어가는 것
        # 아직 현재 clicked targets는 안들어감
        history_article_ids = list(running_history)

        (history_c1, history_c2, history_c3)=_article_ids_to_codes(article_ids=history_article_ids, sid_lookup=sid_lookup)

        # STEP 11-10-3-9. Multi target sid list 생성
        (target_c1, target_c2, target_c3) = _article_ids_to_codes(article_ids=target_article_ids, sid_lookup=sid_lookup)

        # Train / Validation 공통 row
        base_row: dict[str, Any] = {
            "impression_id" : impression_id_int,
            "user_id" : user_id_int,
            "impression_time": impression_time,
            "history_article_ids": history_article_ids,
            "history_c1": history_c1,
            "history_c2": history_c2,
            "history_c3": history_c3,
            "target_article_ids": target_article_ids,
            "target_c1": target_c1,
            "target_c2": target_c2,
            "target_c3": target_c3,
        }

        # STEP 11-10-4. Train sample 생성
        if split_name == "train":
            # current까지 반영된 history + target set 저장
            rows.append(base_row)

            # sample 생성 후 clicked target은 과거 중복 여부와 상관없이 무조건 append 

            # article_ids_clicked의 원래 list 순서를 stable dedup한 deterministic 순서 그대로 사용
            # 이 순서를 실제 클릭 timestamp 순서로 가정하진 x (clicked_ids)
            for target_article_id in target_article_ids:
                running_history.append(target_article_id)

                appended_target_article_count += 1

            continue 

        # STEP 11-10-5. Validation candidate 처리
        candidate_raw = row["article_ids_inview"]
        should_emit = True # seq row 생성 대상 

        # STEP 11-10-5-1. Candidate list 자체 검사
        if candidate_raw is None or len(candidate_raw) == 0:
            candidate_null_or_empty_count += 1
            should_emit = False 

        candidate_article_ids: list[int] = []

        # STEP 11-10-5-2. Candidate 내부 null 제거 + stable dedup
        if should_emit:
            if any(article_id is None for article_id in candidate_raw):
                candidate_null_element_row_count += 1

            valid_candidate_ids = [article_id for article_id in candidate_raw if article_id is not None]

            if len(valid_candidate_ids) == 0:
                candidate_null_or_empty_count += 1
                should_emit = False

            else:
                candidate_article_ids = _stable_unique_ints(valid_candidate_ids)

                if len(candidate_article_ids)!=len(valid_candidate_ids):
                    duplicate_candidate_behavior_row_count += 1

        # STEP 11-10-5-3. 모든 target이 candidate 안에 있는지 ? 
        if should_emit:
            candidate_set = set(candidate_article_ids)
            targets_missing_from_candidates = [
                target_article_id for target_article_id in target_article_ids if target_article_id not in candidate_set
            ]
            if targets_missing_from_candidates:
                target_not_in_candidates_sample_count += 1
                target_not_in_candidates_article_count += len(targets_missing_from_candidates)
                for missing_id in targets_missing_from_candidates:
                    if len(target_not_in_candidates_examples) >= 10: break 
                    if missing_id not in target_not_in_candidates_examples:
                        target_not_in_candidates_examples.append(missing_id)
                should_emit = False 


        # STEP 11-10-5-4. Candidate SID 존재 여부 
        if should_emit:
            missing_candidate_ids = [
                candidate_article_id for candidate_article_id in candidate_article_ids if candidate_article_id not in sid_lookup
            ]
            if missing_candidate_ids:
                missing_candidate_sid_sample_count+=1
                missing_candidate_sid_article_count += len(missing_candidate_ids)
                for missing_id in missing_candidate_ids:
                    if len(missing_candidate_sid_examples)>=10:
                        break 
                    if missing_id not in missing_candidate_sid_examples:
                        missing_candidate_sid_examples.append(missing_id)
                
                should_emit = False
        # STEP 11-10-5-5. Candidate SID + Multi-positive label 생성
        if should_emit:
            (candidate_c1, candidate_c2, candidate_c3)\
                = _article_ids_to_codes(article_ids=candidate_article_ids, sid_lookup=sid_lookup)
            target_set = set(target_article_ids)
            candidate_labels = [(1 if candidate_article_id in target_set else 0) for candidate_article_id in candidate_article_ids]
            rows.append({**base_row, 
                         "candidate_article_ids": candidate_article_ids,
                         "candidate_c1": candidate_c1,
                         "candidate_c2": candidate_c2,
                         "candidate_c3": candidate_c3,
                         "candidate_labels": candidate_labels})
        else:
            # 현재 ranking sample은 출력하지 못했어도 
            # 실제 click 행동은 발생했기에 history는 진행
            history_only_update_count += 1

        # STEP 11-10-5-6. Validation target을 history에 추가
        # train과 동일 (실제 새 클릭이므로 과거 중복 여부 확인 x)
        for target_article_id in target_article_ids:
            running_history.append(target_article_id)
            appended_target_article_count += 1

    # STEP 11-10-6. Python rows -> df
    sequence_df = _make_sequence_df(rows=rows, split_name=split_name)

    # STEP 11-10-7. 최종 내부 정합성 검사
    _validate_sequence_integrity(
        sequence_df=sequence_df,
        split_name=split_name,
    )

    # output impression_id 하나당 정확히 한 row 보장
    duplicate_output_impression_count = sequence_df.select(
        pl.col("impression_id").is_duplicated().sum()
    ).item() if sequence_df.height > 0 else 0 
    # height >0 이면 계산 , 아니면 계산 자체 안하고 그냥 0 바로 쓰기 

    if duplicate_output_impression_count != 0:
        raise ValueError(
            f"{split_name}_sequences에 중복 impression_id가 존재합니다. "
            f"중복 행 수={duplicate_output_impression_count}"
        )

    # STEP 11-10-8. 출력 순서 고정 (한 번 더!)
    if sequence_df.height > 0:
        sequence_df = sequence_df.sort([
            "user_id", "impression_time", "impression_id"
        ])  
    

    # STEP 11-10-9. Parquet 저장
    output_path.parent.mkdir(parents=True, exist_ok = True,)

    sequence_df.write_parquet(
            output_path,
            compression="zstd",
        )

    # STEP 11-10-10. 결과 통계 계산 
    # history 길이 
    history_lengths = sequence_df.get_column("history_article_ids").list.len().to_list() if sequence_df.height > 0 else []
    if history_lengths:
        minimum_history_length = int(
            min(history_lengths)
        )

        maximum_history_length = int(
            max(history_lengths)
        )

        average_history_length = float(
            sum(history_lengths)
            / len(history_lengths)
        )

    else:
        minimum_history_length = None
        maximum_history_length = None
        average_history_length = None

    # Target 개수
    target_counts = (
        sequence_df
        .get_column(
            "target_article_ids"
        )
        .list.len()
        .to_list()
        if sequence_df.height > 0
        else []
    )

    if target_counts:
        minimum_target_count = int(
            min(target_counts)
        )

        maximum_target_count = int(
            max(target_counts)
        )

        average_target_count = float(
            sum(target_counts)
            / len(target_counts)
        )

    else:
        minimum_target_count = None
        maximum_target_count = None
        average_target_count = None 

    # STEP 11-10-11. Split 결과 반환
    result: dict[str, Any] = {
        "status": "SUCCESS",
        "split": split_name,

        "sequence_path": str(
            output_path
        ),

        "behavior_row_count": int(
            behaviors.height
        ),

        "structurally_usable_behavior_row_count": int(
            structurally_usable_behavior_row_count
        ),

        "sequence_row_count": int(
            sequence_df.height
        ),

        "invalid_behavior_row_count": int(
            invalid_behavior_row_count
        ),

        "duplicate_impression_row_count": int(
            duplicate_impression_row_count
        ),

        "clicked_null_element_row_count": int(
            clicked_null_element_row_count
        ),

        "duplicate_clicked_behavior_row_count": int(
            duplicate_clicked_behavior_row_count
        ),

        "single_target_behavior_row_count": int(
            single_target_behavior_row_count
        ),

        "multi_target_behavior_row_count": int(
            multi_target_behavior_row_count
        ),

        # Current 통계
        "current_null_count": int(
            current_null_count
        ),

        "appended_current_count": int(
            appended_current_count
        ),

        "skipped_consecutive_current_count": int(
            skipped_consecutive_current_count
        ),

        "missing_current_sid_count": int(
            missing_current_sid_count
        ),

        "missing_current_sid_examples": (
            missing_current_sid_examples
        ),

        # Target 통계
        "missing_target_sid_sample_count": int(
            missing_target_sid_sample_count
        ),

        "missing_target_sid_article_count": int(
            missing_target_sid_article_count
        ),

        "missing_target_sid_examples": (
            missing_target_sid_examples
        ),

        "appended_target_article_count": int(
            appended_target_article_count
        ),

        # Sequence 길이
        "minimum_history_length": (
            minimum_history_length
        ),

        "maximum_history_length": (
            maximum_history_length
        ),

        "average_history_length": (
            average_history_length
        ),

        "minimum_target_count": (
            minimum_target_count
        ),

        "maximum_target_count": (
            maximum_target_count
        ),

        "average_target_count": (
            average_target_count
        ),

        **history_stats,
    }

    # Validation 전용 통계
    if split_name == "validation":
        result.update({
            "candidate_null_or_empty_count": int(
                candidate_null_or_empty_count
            ),

            "candidate_null_element_row_count": int(
                candidate_null_element_row_count
            ),

            "duplicate_candidate_behavior_row_count": int(
                duplicate_candidate_behavior_row_count
            ),

            "target_not_in_candidates_sample_count": int(
                target_not_in_candidates_sample_count
            ),

            "target_not_in_candidates_article_count": int(
                target_not_in_candidates_article_count
            ),

            "target_not_in_candidates_examples": (
                target_not_in_candidates_examples
            ),

            "missing_candidate_sid_sample_count": int(
                missing_candidate_sid_sample_count
            ),

            "missing_candidate_sid_article_count": int(
                missing_candidate_sid_article_count
            ),

            "missing_candidate_sid_examples": (
                missing_candidate_sid_examples
            ),

            "history_only_update_count": int(
                history_only_update_count
            ),
        })

    return result


# STEP 11-11. Train seq 생성 wrapper
def build_train_sequences(sid_lookup: (dict[int, tuple[int, int, int]] | None)= None,)-> dict[str, Any]:
    """
    train history/behaviors를 이용해 train_sequences.parquet 생성
    validation running history와 공유 x
    """

    if sid_lookup is None : sid_lookup = _load_sid_lookup()

    return _build_split_sequences(split_name="train", history_path=config.TRAIN_HISTORY_PATH, 
                                  behaviors_path=config.TRAIN_BEHAVIORS_PATH,
                                  output_path=config.TRAIN_SEQUENCES_PATH, sid_lookup=sid_lookup)


# STEP 11-12. Validation seq 생성 wrapper
def build_validation_sequences(sid_lookup: (dict[int, tuple[int, int, int]]|None)=None)-> dict[str, Any]:
    """
    validation running history는 train running history 이어받지 않음 
    추가 생성 : 
    candidate_article_ids
    candidate_c1/c2/c3
    candidate_labels
    """

    if sid_lookup is None: sid_lookup = _load_sid_lookup()
    
    return _build_split_sequences(split_name="validation", history_path=config.VALIDATION_HISTORY_PATH,
                                  behaviors_path=config.VALIDATION_BEHAVIORS_PATH,
                                  output_path=config.VALIDATION_SEQUENCES_PATH,
                                  sid_lookup=sid_lookup)

# STEP 11-13. Train + validation seq 전체 실행
def build_sequences() -> dict[str, Any]:
    """
    RQ-VAE 최종 Article SID 이용해 Transformer용 Train/validation seq 모두 생성

    실행 순서
    ------------------
    1. 입력 파일 존재 확인
    2. article_id -> (c1,c2,c3) lookup 생성
    3. train running history / seq 생성
    4. validation running history / seq 생성
    5. 전체 결과 반환

    중요
    ------------------

    두 split이 공유하는 것은 SID lookup
    """

    # STEP 11-13-1. 출력 디렉토리 생성 
    config.create_output_directories()
    
    # STEP 11-13-2. 필요한 입력 파일 존재 검사
    required_paths = {
        "article_semantic_ids": (config.ARTICLE_SEMANTIC_IDS_PATH),

        "train_history": (config.TRAIN_HISTORY_PATH),

        "train_behaviors": (config.TRAIN_BEHAVIORS_PATH),

        "validation_history": (config.VALIDATION_HISTORY_PATH),

        "validation_behaviors": (config.VALIDATION_BEHAVIORS_PATH),
    }

    for (file_name, file_path) in required_paths.items():
        if not file_path.exists():
            raise FileNotFoundError(
                f"{file_name} 파일이 없습니다. "
                f"경로 ={file_path}"
            )

    # STEP 11-13-3. SID lookup은 한 번만 생성
    sid_lookup = _load_sid_lookup()

    # STEP 11-13-4. Train seq 생성
    train_result = build_train_sequences(sid_lookup=sid_lookup)

    # STEP 11-13-5. Validation seq 생성
    validation_result = build_validation_sequences(sid_lookup=sid_lookup)

    # STEP 11-13-6. 전체 결과 반환
    return {
        "status": "SUCCESS",
    
        "article_semantic_id_count": int(len(sid_lookup)),
    
        "article_semantic_ids_path": str(config.ARTICLE_SEMANTIC_IDS_PATH),
    
        "train_result": train_result,
    
        "validation_result": validation_result,
        }

# STEP 11-14. 직접 실행
# 실행 : python -m src.build_sequences
# 전제 : article_semantic_ids.parquet 먼저 존재해야함

if __name__ == "__main__":
    result = build_sequences()

    print()
    print("=" * 70)
    print("Sequence Build 완료")
    print("=" * 70)

    pprint(result)