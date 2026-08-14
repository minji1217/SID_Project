from __future__ import annotations

import math
from datetime import timedelta
from pprint import pprint
from typing import Any

import numpy as np
import polars as pl

from src import config


# ==============================================
# train event 생성에서 사용했던 entity 처리 로직 그대로 사용
# train과 validation에서 entity 정규화 방식과 weighted jaccard 계산 방식이 달라지면 안됨
# 현재는 entity_linking을 적용하지 않으므로 build_train.py에 이미 구현된 아래 두 함수 그대로 사용

from src.build_train import(
    _normalize_entity_set,
    _idf_weighted_jaccard,
)

# STEP 10. Validation article / event build
# train 전처리가 끝난 뒤 validation에서 새롭게 필요한 기사들을
# RQ-VAE inference에 사용할 수 있는 형태로 준비한다

# validation에서는 아래 5가지 금지 
# 1. category mapping 재생성
# 2. entity idf 재계산
# 3. high-df 기준 재계산
# 4. train event 재클러스터링


# 시간 기준
# event 처리 시간: article.published_time
# 사용자 행동 처리 시간 : imprssesion_time -> build_sequences.py 에서 사용

# STEP 10-1.
# Validation에서 실제 참조되는 기사 ID 수집 

def _collect_validation_used_article_ids(
        valid_article_ids: set[int],
)-> dict[str, Any]:
    """
    validation에서 실제 사용하는 모든 article_id 수집

    수집 대상
    -----------------------------
    1. validation/history.parquet - article_id_fixed
    2. validation/behaviors.parquet - usable behavior의 현재 article_id, 클릭 target article_id
    3. article_ids_inview 컬럼이 존재하는 경우 - impression candidate 전체 
    
    참고사항 (candidate를 같이 수집하는 이유)
    -------------------------
    이후 transformer validation에서 NDCG, AUC, MAP 등 계산하려면
    impression 내 candidate들의 SID도 필요하기 때문에 
    
    예: candidate=[A,B,C,D] (정답:A)
    => B,C,D도 SID가 있어야 RANKING 평가 가능 

    behavior 사용 조건
    -----------------------------------
    train의 collect_train_used_article_ids()와 동일하게 맞춘다.
    1. impression_id != null
    2. impression_id 중복 x
    3. used_id != null
    4. impression_time != null
    5. article_ids_clicked != null
    6. article_ids_clicked 빈 리스트 x
    7. clicked list 내부 null x
    8. stable dedup 후 unique click이 정확히 1개 

    반환
    ------------------------------------
    used_article_ids: build_valid_articles()을 통과한 최종 validation 참조 기사 ID 집합
    raw_used_article_count : valid 여부 확인 전 전체 참조 기사 수 
    excluded_article_count : validation에서는 참조됐지만 articles_base.parquet에 존재하지않는 기사 수
    """

    # STEP 10-1-1. Validation history 읽기 
    validation_history = pl.read_parquet(
        config.VALIDATION_HISTORY_PATH
    ).select(
        [
            "user_id",
            "article_id_fixed",
            "impression_time_fixed",
        ]
    )

    # STEP 10-1-2. Validation behavior 스키마 확인
    behavior_schema = pl.scan_parquet(config.VALIDATION_BEHAVIORS_PATH).collect_schema()

    behavior_column_names = set(behavior_schema.names())

    required_behavior_columns = {
        "impression_id",
        "user_id",
        "impression_time",
        "article_id",
        "article_ids_clicked",  
    }

    missing_behavior_columns = required_behavior_columns - behavior_column_names


    if missing_behavior_columns:
        raise ValueError(
            "validation behaviors에 필요한 컬럼이 없습니다: "
            + ", ".join(
                sorted(
                    missing_behavior_columns
                )
            )
        )

    # EB-Nerd ranking candidate 컬럼이 존재하면 사용
    candidate_column_name : str | None = None 

    if "article_ids_inview" in behavior_column_names:
        candidate_column_name =  "article_ids_inview"

    behavior_select_columns = [
        "impression_id",
        "user_id",
        "impression_time",
        "article_id",
        "article_ids_clicked",
    ]

    if candidate_column_name is not None: 
        behavior_select_columns.append(
            candidate_column_name
        )

    validation_behaviors = pl.read_parquet(
        config.VALIDATION_BEHAVIORS_PATH
    ).select(behavior_select_columns)

    # 이 코드의 최종 데이터 형태 
    # 1. validation_behaviors 
    # impression_id | user_id | impression_time | article_id | article_ids_clicked | article_ids_inview

    # 2. validation_history
    # user_id | article_id_fixed 

    # STEP 10-1-3. History 기사 ID 수집
    # Train과 동일한 history 유효성 정책 사용

    duplicated_history_user_ids = set(
        validation_history
        .filter(
            pl.col("user_id").is_not_null()
            &
            pl.col("user_id").is_duplicated()
        )
        .get_column("user_id")
        .to_list()
    )

    history_article_ids: set[int] = set()

    for row in validation_history.iter_rows(named=True):
        user_id = row["user_id"]
        article_ids = row["article_id_fixed"]
        impression_times = row["impression_time_fixed"]

        if user_id is None:
            continue

        if user_id in duplicated_history_user_ids:
            continue

        if article_ids is None or impression_times is None:
            continue

        if len(article_ids) != len(impression_times):
            continue

        for article_id, impression_time in zip(
            article_ids,
            impression_times,
        ):
            # 내부 null은 pair만 제거
            if article_id is None:
                continue

            if impression_time is None:
                continue

            history_article_ids.add(
                int(article_id)
            )


    # STEP 10-1-4. 중복 impression_id 찾기
    # impression_id가 여러 행에 존재한다면 usable behavior에서 제외한다

    duplicated_impression_ids = set(
        validation_behaviors.filter(
            pl.col("impression_id").is_not_null() 
            &
            pl.col("impression_id").is_duplicated()
        ).get_column("impression_id").to_list()
    )

    # STEP 10-1-5. Behavior 참조 기사 저장 공간

    # 사용자가 그때 보고 있던 기사  
    current_article_ids: set[int] = set() 

    # 클릭한 기사 
    target_article_ids: set[int] = set()

    # 후보로 노출됐던 기사 전체 (★★ validation에서 추가 ★★)
    candidate_article_ids: set[int] = set()

    # 유효 행 개수 카운터 
    usable_behavior_row_count = 0

    # STEP 10-1-6. Validation behavior 한 행씩 처리 
    for row in validation_behaviors.iter_rows(named=True):
        impression_id = row["impression_id"]
        user_id = row["user_id"]
        impression_time = row["impression_time"]
        current_article_id = row["article_id"]
        clicked_article_ids = row["article_ids_clicked"]

        # 기본 식별 정보 검사 
        if impression_id is None: continue

        if (impression_id in duplicated_impression_ids): continue

        if user_id is None: continue

        if impression_time is None: continue 

        # 클릭 목록 검사 
        if clicked_article_ids is None : continue 
        if len(clicked_article_ids) == 0:
            continue
        
        # clicked 내부 null은 원소만 제거
        valid_clicked_article_ids = [
            article_id
            for article_id in clicked_article_ids
            if article_id is not None
        ]

        # null 제거 후 target이 하나도 없으면 제외
        if len(valid_clicked_article_ids) == 0:
            continue


        # stable dedup
        # 예 : [100, 100, 200, 100] -> [100, 200]
        # 처음 등장한 순서 유지

        deduplicated_clicked_ids: list[int] = []

        seen_clicked_ids: set[int] = set()

        for article_id in valid_clicked_article_ids:

            article_id = int(article_id) 

            if article_id in seen_clicked_ids:
                continue

            seen_clicked_ids.add(article_id)

            deduplicated_clicked_ids.append(article_id) 

        

        if len(deduplicated_clicked_ids) ==0 : continue 
        usable_behavior_row_count += 1

        target_article_ids.update(deduplicated_clicked_ids)


        # 현재 보고 있던 기사
        if current_article_id is not None : current_article_ids.add(int(current_article_id))

        # impression candidate 기사 
        # article_ids_inview가 존재하는 경우만 처리 
        if candidate_column_name is not None : 
            candidate_ids = row[candidate_column_name]

            if candidate_ids is not None:
                for candidate_article_id in candidate_ids:
                    if candidate_article_id is None:
                        continue 
                    candidate_article_ids.add(int(candidate_article_id))


          

    # STEP 10-1-7.
    # validation에서 참조한 모든 기사 합치기
    raw_used_article_ids = (
        history_article_ids 
        | current_article_ids
        | target_article_ids
        | candidate_article_ids
    ) 

    # STEP 10-1-8. 
    # article_base.parquet의 유효 기사와 교집합
    # 원본 behaviors/history에 참조되더라도
    # build_valid_articles() 단계에서 제거된 기사는 모델 입력으로 사용 불가

    used_article_ids = raw_used_article_ids & valid_article_ids

    excluded_article_ids = raw_used_article_ids - valid_article_ids

    # STEP 10-1-9. 결과 반환
    return {
    "used_article_ids": used_article_ids,
    "raw_used_article_count": int(len(raw_used_article_ids)),
    "valid_used_article_count": int(len(used_article_ids)),
    "excluded_article_count": int(len(excluded_article_ids)),
    "excluded_article_examples": sorted(excluded_article_ids)[:10],
    "history_article_count": int(len(history_article_ids)),
    "current_article_count": int(len(current_article_ids)),
    "target_article_count": int(len(target_article_ids)),
    "candidate_column_present": candidate_column_name is not None,
    "candidate_article_count": int(len(candidate_article_ids)),
    "usable_behavior_row_count": int(usable_behavior_row_count),
    }

# STEP 10-2. 
# article_id 집합을 parquet으로 저장
def _write_article_id_set(article_ids: set[int], output_path:Any)-> None:
    """
    article_id python set을 정렬해 한 행당 article_id 하나인 parquet으로 저장
    결과 예: 
    article_id
    ----------
    100
    200
    300
    """

    # validation에서 SID 필요한 기사들의 id 모아둔 article_ids (파이썬 set)
    # -> parquet 파일로 저장하는 코드 

    article_id_df = (
        pl.DataFrame({"article_id": sorted(article_ids)})
        .with_columns(
            pl.col("article_id").cast(pl.Int64)
        )
        .sort("article_id")
    )

    article_id_df.write_parquet(output_path, compression="zstd")

# STEP 10-3.
# validation only 기사 dynamic event assignment 
def _assign_validation_events(
    validation_only_article_ids: set[int],
    entity_similarity_threshold: float = 
        config.EVENT_ENTITY_SIMILARITY_THRESHOLD
    , time_window_hours: int = (
        config.EVENT_TIME_WINDOW_HOURS
    )) -> dict[str, Any]:
    """
    validation-only 기사에 event id 부여 

    train event를 다시 클러스터링 하지 않는다.
    train에서 만들어진 event_master.parquet, entity_idf.parquet 그대로 시작 상태로 사용

    dynamic 방식
    ----------------------------
    validation 기사 V1 
        ↓
    기존 Event와 비교
        ↓
    Event 10에 배정
        ↓
    Event 10 상태 업데이트
        event_entities 갱신
        event_last_added_time 갱신
        validation_article_count += 1 
        ↓
    validation 기사 V2는 V1까지 반영된 최신 Event 10과 비교한다.

    시간 기준
    ------------------------------------------
    event 처리는 impression_time이 아니라 article.published_time 기준

    또한 현재 구현에서는 : 
    article.published_time >= event_last_added_time인 Event만 후보로 사용

    즉 Validation 기사보다 미래의 Event 상태를 이용해 
    과거 기사에 Event를 붙이는 것은 막는다. 

    Event 선택
    -------------------------------------------
    1. 0<=time_gap<=time_window
    2. IDF Weighted Jaccard >= threshold
    3. 조건을 만족한 event 중 sim 가장 높은 event 
    4. sim이 같으면 event_id가 작은 event 

    어떤 event에도 연결되지 않으면 새로운 validation-origin event 만든다 
    """

    # STEP 10-3-1. 파라미터 검사 
    if not 0.0 <= entity_similarity_threshold <= 1.0:
        raise ValueError(
            "entity_similarity_threshold는 0과 1 사이여야 합니다. "
            f"현재 값={entity_similarity_threshold}"
        )

    if time_window_hours <= 0:
        raise ValueError(
            "time_window_hours는 0보다 커야 합니다. "
            f"현재 값={time_window_hours}"
        )

    # 시간 조건을 timedelta 객체로 변환
    # 예: 72 -> timedelta(hours=72)
    time_window = timedelta(hours=time_window_hours)


    # STEP 10-3-2. train entity idf 읽기
    # validation에서 IDF를 다시 계산하면 안됨
    # TRAIN: Train 기사 -> Entity DF / IDF 계산
    # VALIDATION : 위에서 계산된 IDF 그대로 사용 

    entity_idf_df = pl.read_parquet(config.ENTITY_IDF_PATH)

    required_entity_idf_columns = {"entity", "idf", "is_high_df"}
    missing_entity_idf_columns = required_entity_idf_columns - set(entity_idf_df.columns)
    if missing_entity_idf_columns:
        raise ValueError(
            "entity_idf.parquet에 필요한 컬럼이 없습니다: "
            + ", ".join(sorted(missing_entity_idf_columns))
        )

    # entity -> idf 형태의 python dictionary 생성
    # "PER::trump" : 521, "ORG::openai": 4.81, ...
    entity_idf: dict[str, float]={
        str(entity): float(idf) for entity, idf in entity_idf_df.select(["entity", "idf"]).iter_rows()
    }

    # train에서 너무 자주 등장해서 event 계산에서 제외하기로 한 entity
    # validation에서도 같은 목록 그대로 사용 
    high_df_entities: set[str] = set(
        str(entity) for entity in (entity_idf_df.filter(pl.col("is_high_df")).get_column("entity").to_list())
    )

    # STEP 10-3-3. 
    # Validation에서 처음 등장한 entity의 IDF 기본값 계산
    # Validation 신규 entity는 train idf dict에 없다
    # 이때 train에서 사용했던 정책과 동일하게 : 
    # unseen_entity_idf = log(N+1)+1 (N도 Validation이 아니라 train 기사 수)

    train_article_count = pl.read_parquet(config.TRAIN_USED_ARTICLE_IDS_PATH).height 
    if train_article_count <= 0:
        raise ValueError("Train 사용 기사 수가 0입니다.")

    unseen_entity_idf = math.log(train_article_count + 1) + 1.0

    # STEP 10-3-4. Train event master 읽기 
    # event_master.parquet은 train event의 현재 상태 저장한다.
    # event_id | event_origin_split | event_start_time | event_last_added_time
    # event_article_count | train_article_count | event_entities | first_article_id 
    # validation은 이 상태에서 시작 

    train_event_master = pl.read_parquet(config.EVENT_MASTER_PATH)

    required_event_columns = {
        "event_id",
        "event_origin_split",
        "event_start_time",
        "event_last_added_time",
        "event_article_count",
        "train_article_count",
        "event_entities",
        "event_entity_count",
        "first_article_id",
    }

    missing_event_columns = required_event_columns - set(train_event_master.columns)

    if missing_event_columns: 
        raise ValueError(
            "event_master.parquet에 필요한 컬럼이 없습니다: " + ", ".join(sorted(missing_event_columns))
        )

    # event_id는 event 하나당 정확히 한 행이어야 함
    duplicate_event_id_count = train_event_master.select(
        pl.col("event_id").is_duplicated().sum()
    ).item()

    if duplicate_event_id_count != 0:
        raise ValueError(
            "event_master.parquet에 중복 event_id가 존재합니다. "
            f"중복 행 수={duplicate_event_id_count}"
        )

    # STEP 10-3-5. Event Master를 python 상태로 
    # validation 기사 하나를 처리할 때마다 event 상태를 수정해야하기에
    # python dict 형태로 관리한다
    # events:
    # {
    #     10: {
    #         "event_id": 10,
    #         "event_entities": {...},
    #         "event_last_added_time": ...,
    #         ...
    #     }
    # }

    events: dict[int, dict[str, Any]] = {}

    for row in train_event_master.iter_rows(named=True):
        event_id = int(row["event_id"])
        event_entities = set(row["event_entities"] or [])
        # 그 사건에 속한 train 기사 수 
        train_count = int(row["train_article_count"])
        # 그 사건에 속한 전체 기사 수 (train + validation)
        event_article_count = int(row["event_article_count"])

        # train 상태에선 train 데이터로만 event 만들었기 때문에 일반적으로 0
        # 혹시 이미 Validation 데이터가 들어간 잘못된 파일이라면
        # 여기서 값이 0보다 클 수 있으므로 계산해 확인한다.
        validation_article_count = event_article_count - train_count

        if validation_article_count < 0:
            raise ValueError(
                "event_master의 기사 수 정보가 잘못되었습니다. "
                f"event_id={event_id}, "
                f"event_article_count={event_article_count}, "
                f"train_article_count={train_count}"
            )

        events[event_id] = {
            "event_id": event_id,
            "event_origin_split": str(row["event_origin_split"]),
            "event_start_time": row["event_start_time"],
            "event_last_added_time": row["event_last_added_time"],
            "event_article_count": event_article_count,
            "train_article_count": train_count,
            "validation_article_count": validation_article_count,
            "event_entities": event_entities,
            "first_article_id": int(row["first_article_id"]),
        }
    if not events: 
        raise ValueError("Train Event가 하나도 존재하지 않습니다.")

    # 새 Validation Event 만들 경우
    # 기존 가장 큰 event_id 다음 번호부터 사용
    next_event_id = max(events)+1 

    # STEP 10-3-6. Validation-only 기사 metadata 읽기
    # articles_with_category.parquet에는 전체 유효 기사와 함께 published_time, NER 정보 존재
    # 여기선 validation_only에 해당하는 기사만 사용

    articles= pl.read_parquet(config.ARTICLES_WITH_CATEGORY_PATH).select(
        ["article_id", "published_time", "ner_clusters", "entity_groups"]
    )

    available_article_ids = set(
        int(article_id) for article_id in articles.get_column("article_id").to_list()
    )


    missing_validation_article_ids = validation_only_article_ids - available_article_ids
    if missing_validation_article_ids:
        raise ValueError(
            "validation_only 기사 중 articles_with_category.parquet에 없는 기사가 있습니다. "
            f"예시 = {sorted(missing_validation_article_ids)[:10]}"
        )

    # STEP 10-3-7. Validation-only 기사별 entity set 생성
    # raw entity_set : 원본 NER 정규화한 전체 entity
    # clustering_entity_set : high-df entity 제외한 event clustering용 entity 
    # raw{PER::trump, LOC::usa} 
    # -> 이때 usa가 너무 흔한 high-df entity라면?
    # clustering{PER::trump}

    validation_articles: list[dict[str, Any]] = []
    
    for row in articles.iter_rows(named=True):
        article_id = int(row["article_id"])

        if article_id not in validation_only_article_ids:
            continue 

        # 해당 기사에서 나오는 개체들 정규화 
        entity_set = _normalize_entity_set(
            row["ner_clusters"], row["entity_groups"],
        )

        # entity_set에서 흔한 것 뺌 
        clustering_entity_set = entity_set - high_df_entities

        validation_articles.append({
            "article_id":article_id,
            "published_time": row["published_time"],
            "entity_set": entity_set,
            # 처음 보는 개체는 clustering_entity_set에 포함되고, 다음 단계에서 
            # unseen_entity_idf로 처리됨 
            "clustering_entity_set":clustering_entity_set, 
        })

    # event 생성 순서는 article published_time 기준
    # 같은 publisehd_time이면 article_id를 다음 기준으로 
    validation_articles.sort(
        key=lambda article: (
            article["published_time"],
            article["article_id"],
        )
    )

    # STEP 10-3-8. Dynamic event assignment 
    validation_article_event_rows: list[dict[str, Any]] = []

    # 통계용 카운터 
    # 시간 조건(72시간 이내)을 통과해서, 실제로 유사도 계산까지 한 후보쌍의 개수
    time_candidate_pair_count = 0

    # 유사도(θ)까지 통과해서, 사건 매칭 후보로 채택된 쌍의 개수
    similarity_candidate_pair_count = 0

    # 새 기사가 "기존에 이미 있던 사건"에 병합된 횟수 (2-a 케이스 전체)
    matched_existing_event_count = 0

    # 그중에서도, train에서 만들어진 사건에 병합된 횟수
    matched_train_origin_event_count = 0

    # 그중에서도, validation 처리 중 새로 생긴 사건에 병합된 횟수
    matched_validation_origin_event_count = 0

    # 병합할 사건을 못 찾아서, 새 사건을 만든 횟수 (2-b 케이스)
    new_validation_event_count = 0

    # entity_set(원본, high_df 포함)이 아예 빈 기사 수
    validation_empty_entity_article_count = 0

    # clustering_entity_set(high_df 제외한 버전)이 아예 빈 기사 수
    validation_clustering_empty_entity_article_count = 0


    # (validation-only)validation 기사를 published_time 순서로 하나씩 처리
    for article in validation_articles: 
        article_id = article["article_id"]
        article_time = article["published_time"]

        raw_article_entities = article["entity_set"]
        article_entities  = article["clustering_entity_set"]

        if not raw_article_entities:
            validation_empty_entity_article_count += 1

        if not article_entities:
            validation_clustering_empty_entity_article_count += 1

        # 현재 기사와 가장 잘 맞는 기존 event 찾기
        best_event_id: int | None = None 
        best_similarity = -1.0

        # event_id 오름차순으로 검사
        # similarity가 완전히 같은 event가 여러 개라면
        # 작은 event_id가 deterministic하게 선택됨
        for event_id in sorted(events):
            event = events[event_id]
            event_last_added_time = event["event_last_added_time"]

            # 시간 차 계산
            # article_time < event_last_added_time 인 경우:
            # 현재 기사보다 미래 상태의 event 이용하는 것이므로 제외
            time_gap = article_time - event_last_added_time

            if time_gap < timedelta(0):
                continue 
            if time_gap > time_window: continue 
            time_candidate_pair_count += 1

            # entity sim 계산
            similarity = _idf_weighted_jaccard(
                article_entities,
                event["event_entities"],
                entity_idf,
                unseen_entity_idf, 
            )

            if similarity < entity_similarity_threshold: continue 
            similarity_candidate_pair_count += 1

            # 현재까지 가장 sim이 가장 높은 event 저장 
            if best_event_id is None or similarity > best_similarity:
                best_event_id = event_id 
                best_similarity = similarity 

        # 기존 Event와 연결되는 경우
        if best_event_id is not None:
            event = events[best_event_id]
            matched_existing_event_count += 1

            if event["event_origin_split"] == "train":
                matched_train_origin_event_count += 1
            else:
                matched_validation_origin_event_count += 1
            
            # validation 기사 entity를 기존 event entity union에 추가
            event["event_entities"].update(article_entities)

            # impression_time이 아닌 현재 기사의 published_time
            event["event_last_added_time"]=article_time

            # event에 포함된 전체기사 수 증가
            event["event_article_count"]+=1

            # validation에서 추가된 기사 수 증가
            event["validation_article_count"] +=1 

            assigned_event_id = best_event_id

        # 연결되는 event가 없으면 새 validation event 생성
        else:
            assigned_event_id = next_event_id

            events[assigned_event_id] = {
                "event_id": assigned_event_id,
                "event_origin_split": "validation",
                "event_start_time": article_time,
                "event_last_added_time": article_time,
                "event_article_count": 1,
                "train_article_count": 0,
                "validation_article_count": 1,
                "event_entities": set(article_entities),
                "first_article_id": article_id,
            }

            next_event_id += 1
            new_validation_event_count += 1

               
        # 현재 기사 → Event mapping 저장
        validation_article_event_rows.append({
            "article_id": article_id,
            "event_id": assigned_event_id,
            "assignment_split": "validation",
            "published_time": article_time,
        })

        # STEP 10-3-9. Validation article event df 생성
        
    if validation_article_event_rows:
        validation_article_events_df = (
            pl.DataFrame(validation_article_event_rows)
            .with_columns([
                pl.col("article_id").cast(pl.Int64),
                pl.col("event_id").cast(pl.Int64),
            ])
            .sort("article_id")
        )

    else:
        # Validation-only 기사가 0개여도 파일 자체는 생성한다.
        # downstream에서 FileNotFoundError가 발생하지 않도록 하기 위함.
        validation_article_events_df = pl.DataFrame(
            schema={
                "article_id": pl.Int64,
                "event_id": pl.Int64,
                "assignment_split": pl.Utf8,
                "published_time": articles.schema["published_time"],
            }
        )

    # STEP 10-3-10. Validation Article Event 정합성 검사
    if validation_article_events_df.height != len(validation_only_article_ids):
        raise ValueError(
            "Validation Article Event 수와 validation_only 기사 수가 다릅니다. "
            f"validation_only={len(validation_only_article_ids)}, "
            f"article_events={validation_article_events_df.height}"
        )

    duplicate_article_count = (
        validation_article_events_df
        .select(pl.col("article_id").is_duplicated().sum())
        .item()
    )

    if duplicate_article_count != 0:
        raise ValueError(
            "Validation Article Event에 중복 article_id가 존재합니다. "
            f"중복 행 수={duplicate_article_count}"
        )

    # Validation-only article → event mapping 저장
    validation_article_events_df.write_parquet(
        config.VALIDATION_ARTICLE_EVENTS_PATH,
        compression="zstd",
    )

    # STEP 10-3-11. validation까지 반영된 event master 생성
    event_master_rows: list[dict[str, Any]] = []

    for event_id in sorted(events):
        event = events[event_id]
        event_entities = event["event_entities"]

        event_master_rows.append({
            "event_id": event_id,
            "event_origin_split": event["event_origin_split"],
            "event_start_time": event["event_start_time"],
            "event_last_added_time": event["event_last_added_time"],
            "event_article_count": event["event_article_count"],
            "train_article_count": event["train_article_count"],
            "validation_article_count": event["validation_article_count"],
            "event_entities": sorted(event_entities),
            "event_entity_count": len(event_entities),
            "first_article_id": event["first_article_id"],
        })

    event_master_with_validation_df = (
        pl.DataFrame(event_master_rows)
        .with_columns([
            pl.col("event_id").cast(pl.Int64),
            pl.col("event_article_count").cast(pl.Int64),
            pl.col("train_article_count").cast(pl.Int64),
            pl.col("validation_article_count").cast(pl.Int64),
            pl.col("event_entity_count").cast(pl.Int64),
            pl.col("first_article_id").cast(pl.Int64),
        ])
        .sort("event_id")
    )

    # STEP 10-3-12. Event 기사 수 정합성 검사
    # event_article_count = train_article_count + validation_article_count 여야함
    invalid_event_article_count = (
        event_master_with_validation_df
        .filter(
            pl.col("event_article_count")
            != pl.col("train_article_count") + pl.col("validation_article_count")
        )
        .height
    )

    if invalid_event_article_count != 0:
        raise ValueError(
            "Event 전체 기사 수와 Train + Validation 기사 수가 다릅니다. "
            f"문제 Event 수={invalid_event_article_count}"
        )

    # STEP 10-3-13. Event 시간 정합성 검사
    invalid_event_time_count = (
        event_master_with_validation_df
        .filter(pl.col("event_start_time") > pl.col("event_last_added_time"))
        .height
    )

    if invalid_event_time_count != 0:
        raise ValueError(
            "event_start_time보다 event_last_added_time이 이전인 Event가 있습니다. "
            f"문제 Event 수={invalid_event_time_count}"
        )

    # STEP 10-3-14. 
    # 실제 article -> event mapping 기준 기사 수 재검사 
    # article_events.parquet 과 validation_article_events.parquet
    # 합쳐서 실제 event별 기사 수 다시 계산 

    # article_id | event_id 
    # train+validation 전체 기사가 어느 사건에 속하는지 모임
    # -> event_id 별로 실제 기사 수 세기 
    train_article_events_df = (
        pl.read_parquet(config.ARTICLE_EVENTS_PATH)
        .select(["article_id", "event_id"])
        .with_columns([
            pl.col("article_id").cast(pl.Int64),
            pl.col("event_id").cast(pl.Int64),
        ])
    )

    validation_event_count_df = validation_article_events_df.select(
        ["article_id", "event_id"]
    )

    combined_article_events = pl.concat(
        [train_article_events_df, validation_event_count_df],
        how="vertical",
    )

    actual_event_article_counts = (
        combined_article_events
        .group_by("event_id")
        .agg(pl.len().alias("_actual_event_article_count"))
    )

    event_count_check = (
        event_master_with_validation_df
        .select(["event_id", "event_article_count"])
        .join(actual_event_article_counts, on="event_id", how="left")
    )

    missing_actual_count = (
        event_count_check
        .get_column("_actual_event_article_count")
        .null_count()
    )

    if missing_actual_count != 0:
        raise ValueError(
            "Event Master에는 존재하지만 Article Event에는 기사가 없는 Event가 있습니다. "
            f"문제 Event 수={missing_actual_count}"
        )
    

    mismatched_event_count = (
        event_count_check
        .filter(
            pl.col("event_article_count")
            != pl.col("_actual_event_article_count")
        )
        .height
    )

    if mismatched_event_count != 0:
        raise ValueError(
            "Event Master의 기사 수와 실제 Article Event 기사 수가 다릅니다. "
            f"문제 Event 수={mismatched_event_count}"
        )
    
    # STEP 10-3-15. Validation Event Master 저장
    event_master_with_validation_df.write_parquet(
        config.EVENT_MASTER_WITH_VALIDATION_PATH,
        compression="zstd",
    )

    # event_id | event_origin_split | event_start_time    | event_last_added_time 
    # event_article_count | train_article_count | validation_article_count
    # event_entity_count | first_article_id

    # STEP 10-3-16. 결과 반환
    

    return {
        "status": "SUCCESS",
        "validation_only_article_count": int(len(validation_only_article_ids)),
        "validation_article_event_count": int(validation_article_events_df.height),

        "matched_existing_event_count": int(matched_existing_event_count),
        "matched_train_origin_event_count": int(matched_train_origin_event_count),
        "matched_validation_origin_event_count": int(matched_validation_origin_event_count),
        "new_validation_event_count": int(new_validation_event_count),

        "final_event_count": int(event_master_with_validation_df.height),

        "time_candidate_pair_count": int(time_candidate_pair_count),
        "similarity_candidate_pair_count": int(similarity_candidate_pair_count),

        "validation_empty_entity_article_count": int(
            validation_empty_entity_article_count
        ),
        "validation_clustering_empty_entity_article_count": int(
            validation_clustering_empty_entity_article_count
        ),

        "validation_article_events_path": str(
            config.VALIDATION_ARTICLE_EVENTS_PATH
        ),
        "event_master_with_validation_path": str(
            config.EVENT_MASTER_WITH_VALIDATION_PATH
        ),
    }


# STEP 10-4. Validation Article Master 생성
def build_validation_article_master() -> dict[str, Any]:
    """
    RQ-VAE validation frozen inference에 사용할 
    validation-only 기사 metadata를 생성한다.

    train article master와 동일한 컬럼 구조 : 
    article_id | embedding_row | model_category_id 
    event_id | published_time | model_text

    차이점
    -----------------------
    Train article master : RQ-VAE 학습용

    Validation article master : 이미 학습된 RQ-VAE frozen inference용

    따라서 validation_article_master.parquet은 
    RQ-VAE Train DataLoader에 포함하면 안됨
    """

    # STEP 10-4-1. 필요한 선행 산출물 존재 여부 검사 
    required_paths = {
        "validation_only_article_ids": config.VALIDATION_ONLY_ARTICLE_IDS_PATH,
        "articles_with_category": config.ARTICLES_WITH_CATEGORY_PATH,
        "article_embedding_input": config.ARTICLE_EMBEDDING_INPUT_PATH,
        "article_embeddings": config.ARTICLE_EMBEDDINGS_PATH,
        "validation_article_events": config.VALIDATION_ARTICLE_EVENTS_PATH,
    }

    for file_name, file_path in required_paths.items():
        if not file_path.exists():
            raise FileNotFoundError(
                f"{file_name} 파일이 없습니다. 경로 ={file_path}"
            )
        
    # STEP 10-4-2. validaion-only 기사 id 읽기
    validation_only_articles = (
        pl.read_parquet(config.VALIDATION_ONLY_ARTICLE_IDS_PATH)
        .select("article_id")
        .with_columns(pl.col("article_id").cast(pl.Int64))
        .sort("article_id")
    )

    validation_article_count = validation_only_articles.height 

    # STEP 10-4-3. 기사 Metadata 읽기
    # model_category_id는 이미 train category mapping을 
    # 전체 유효 기사에 적용한 결과이므로,
    # validation에서 category mapping을 다시 만들지 않는다.

    article_metadata = pl.read_parquet(config.ARTICLES_WITH_CATEGORY_PATH)\
    .select([
        "article_id",
        "model_category_id",
        "published_time",
        "model_text",
    ])\
    .with_columns(pl.col("article_id").cast(pl.Int64))

    # STEP 10-4-4. Article embedding 위치 정보 읽기
    # 실제 768차원 벡터는 article_embeddings.npy에 존재 
    # 여기선 article_id -> embedding_row 매핑만 가져오기
    article_embedding_input = (
        pl.read_parquet(config.ARTICLE_EMBEDDING_INPUT_PATH)
        .select(["article_id", "embedding_row"])
        .with_columns([
            pl.col("article_id").cast(pl.Int64),
            pl.col("embedding_row").cast(pl.Int64),
        ])
    )

    # STEP 10-4-5. Validation-only event mapping 읽기
    # article_id | event_id 
    validation_article_events = (
        pl.read_parquet(config.VALIDATION_ARTICLE_EVENTS_PATH)
        .select(["article_id", "event_id"])
        .with_columns([
            pl.col("article_id").cast(pl.Int64),
            pl.col("event_id").cast(pl.Int64),
        ])
    )

    # STEP 10-4-6. JOIN Source의 article_id 중복 검사 
    join_sources = {
        "article_metadata": article_metadata,
        "article_embedding_input": article_embedding_input,
        "validation_article_events": validation_article_events,
    }

    for source_name, source_df in join_sources.items():
        duplicate_count = (
            source_df
            .select(pl.col("article_id").is_duplicated().sum())
            .item()
        )

        if duplicate_count != 0:
            raise ValueError(
                f"{source_name}에 중복 article_id가 존재합니다. "
                f"중복 행 수={duplicate_count}"
            )

    # STEP 10-4-7. Validation article master join
    validation_article_master = (
        validation_only_articles
        .join(article_metadata, on="article_id", how="left")
        .join(article_embedding_input, on="article_id", how="left")
        .join(validation_article_events, on="article_id", how="left")
        .select([
            "article_id",
            "embedding_row",
            "model_category_id",
            "event_id",
            "published_time",
            "model_text",
        ])
        .sort("article_id")
    )

    # STEP 10-4-8. 기사 수 정합성 검사
    if validation_article_master.height != validation_article_count:
        raise ValueError(
            "Validation Article Master 기사 수와 validation_only 기사 수가 다릅니다. "
            f"validation_only={validation_article_count}, "
            f"master={validation_article_master.height}"    
        )   

    # STEP 10-4-9. 필수 컬럼 null 검사
    required_columns = [
        "article_id",
        "embedding_row",
        "model_category_id",
        "event_id",
        "published_time",
        "model_text",
    ]

    null_counts: dict[str, int] = {}

    for column_name in required_columns:
        null_count = int(
            validation_article_master
            .get_column(column_name)
            .null_count()
        )

        null_counts[column_name] = null_count

        if null_count != 0:
            raise ValueError(
                "Validation Article Master의 "
                f"{column_name}에 null 값이 존재합니다. "
                f"null 개수={null_count}"
            )

    # STEP 10-4-10. model_text 빈 문자열 검사
    empty_model_text_count = (
        validation_article_master.filter(
            pl.col("model_text").cast(pl.Utf8, strict=False)
            .str.strip_chars() == ""
        ).height
    )

    if empty_model_text_count != 0:
        raise ValueError(
            "Validation Article Master에 빈 model_text가 존재합니다. "
            f"빈 문자열 수={empty_model_text_count}"
        )

    # STEP 10-4-11. article_id 중복 검사
    duplicate_article_count = (
        validation_article_master
        .select(pl.col("article_id").is_duplicated().sum())
        .item()
    )

    if duplicate_article_count != 0:
        raise ValueError(
            "Validation Article Master에 중복 article_id가 존재합니다. "
            f"중복 행 수={duplicate_article_count}"
        )

    # STEP 10-4-12.
    # embedding_row가 실제 .npy 범위 안인지 검사 
    article_embeddings = np.load(
        config.ARTICLE_EMBEDDINGS_PATH,
        mmap_mode="r",
        allow_pickle=False,
    )

    if article_embeddings.ndim != 2:
        raise ValueError(
            "article_embeddings.npy는 2차원 배열이어야 합니다. "
            f"현재 ndim={article_embeddings.ndim}"
        )

    embedding_array_row_count = int(article_embeddings.shape[0])

    if validation_article_count > 0:
        minimum_embedding_row = int(
            validation_article_master.get_column("embedding_row").min()
        )

        maximum_embedding_row = int(
            validation_article_master.get_column("embedding_row").max()
        )

        if minimum_embedding_row < 0:
            raise ValueError(
                "Validation embedding_row에 음수가 존재합니다. "
                f"최솟값={minimum_embedding_row}"
            )

        if maximum_embedding_row >= embedding_array_row_count:
            raise ValueError(
                "Validation embedding_row가 article_embeddings.npy 범위를 벗어났습니다. "
                f"최대 row={maximum_embedding_row}, "
                f"embedding 행 수={embedding_array_row_count}"
            )

    else:
        minimum_embedding_row = None
        maximum_embedding_row = None

    # STEP 10-4-13. 데이터 타입 정리
    validation_article_master = validation_article_master.with_columns([
        pl.col("article_id").cast(pl.Int64),
        pl.col("embedding_row").cast(pl.Int64),
        pl.col("model_category_id").cast(pl.Int32),
        pl.col("event_id").cast(pl.Int64),
    ])

    # STEP 10-4-14. Validation Article Master 저장
    validation_article_master.write_parquet(
        config.VALIDATION_ARTICLE_MASTER_PATH,
        compression="zstd",
    )

    # STEP 10-4-15. 결과 통계
    unknown_category_article_count = (
        validation_article_master
        .filter(pl.col("model_category_id") == 0)
        .height
    )

    unique_event_count = (
        validation_article_master
        .select("event_id")
        .unique()
        .height
    )

    return {
        "status": "SUCCESS",
        "validation_article_master_path": str(
            config.VALIDATION_ARTICLE_MASTER_PATH
        ),

        "validation_article_count": int(validation_article_count),
        "duplicate_article_count": int(duplicate_article_count),
        "empty_model_text_count": int(empty_model_text_count),
        "unknown_category_article_count": int(unknown_category_article_count),
        "unique_event_count": int(unique_event_count),

        "minimum_embedding_row": minimum_embedding_row,
        "maximum_embedding_row": maximum_embedding_row,

        "article_embedding_array_shape": [
            int(article_embeddings.shape[0]),
            int(article_embeddings.shape[1]),
        ],
    }

# STEP 10-5. Validation Build 전체 실행 함수
def build_validation() -> dict[str, Any]:
    """
    Validation 전처리 전체 과정을 한 번에 실행한다. 

    처리 순서
    --------------------------------------------------
    1. train 전처리 산출물 존재 여부 검사
    2. validation에서 실제 사용하는 기사 id 수집
    3. validation 기사와 train 기사 비교
        - train과 겹치는 기사
        - validation-only 기사
    4. validation-only 기사에 dynamic event assignment 수행
    5. RQ-VAE frozen inference용 
        validation_article_master.parquet 생성
    
    중요
    ----------------------------------------
    train에서 이미 처리된 기사는 validation에서 다시 처리 x
    
    """

    # STEP 10-5-1. 출력 폴더 생성
    config.create_output_directories()

    # STEP 10-5-2. 필요한 선행 파일 검사
    required_paths = {
        "articles_base": config.ARTICLES_BASE_PATH,
        "train_used_article_ids": config.TRAIN_USED_ARTICLE_IDS_PATH,
        "articles_with_category": config.ARTICLES_WITH_CATEGORY_PATH,
        "article_embedding_input": config.ARTICLE_EMBEDDING_INPUT_PATH,
        "article_embeddings": config.ARTICLE_EMBEDDINGS_PATH,
        "train_article_events": config.ARTICLE_EVENTS_PATH,
        "train_event_master": config.EVENT_MASTER_PATH,
        "train_entity_idf": config.ENTITY_IDF_PATH,
        "validation_behaviors": config.VALIDATION_BEHAVIORS_PATH,
        "validation_history": config.VALIDATION_HISTORY_PATH,
    }

    for file_name, file_path in required_paths.items():
        if not file_path.exists():
            raise FileNotFoundError(
                f"{file_name} 파일이 없습니다. 경로={file_path}"
            )

    # STEP 10-5-3. 전체 유효 기사 ID 읽기
    valid_articles = pl.read_parquet(config.ARTICLES_BASE_PATH).select("article_id")

    valid_article_ids = set(int(article_id) for article_id in valid_articles.get_column("article_id").to_list())

    # STEP 10-5-4. validation에서 사용하는 기사 ID 수집
    # STEP 10-1에서 만든 함수로, history, current article, clicked target, article_ids_inview candidate 모음

    validation_usage_result = _collect_validation_used_article_ids(
        valid_article_ids=valid_article_ids
    )

    validation_used_article_ids = validation_usage_result["used_article_ids"]

    # STEP 10-5-5. Train에서 사용한 기사 id 읽기 
    train_used_article_ids = set(
        int(article_id)
        for article_id in (
            pl.read_parquet(config.TRAIN_USED_ARTICLE_IDS_PATH)
            .get_column("article_id")
            .to_list()
        )
    )

    # STEP 10-5-6. Validation 기사 분리
    # Validation에서 사용하면서 Train에도 있던 기사
    validation_train_overlap_article_ids = (
        validation_used_article_ids & train_used_article_ids
    )

    # Validation에서 처음 등장한 기사
    validation_only_article_ids = (
        validation_used_article_ids - train_used_article_ids
    )

    # STEP 10-5-7. Validation 기사 ID 목록 저장
    # Validation에서 실제 참조되는 전체 유효 기사
    _write_article_id_set(
        validation_used_article_ids,
        config.VALIDATION_USED_ARTICLE_IDS_PATH,
    )

    # Validation에서 사용하지만 Train에서는 사용하지 않은 기사
    _write_article_id_set(
        validation_only_article_ids,
        config.VALIDATION_ONLY_ARTICLE_IDS_PATH,
    )

    # STEP 10-5-8. Validation-only dynamic event 배정
    validation_event_result = _assign_validation_events(
        validation_only_article_ids=validation_only_article_ids
    )


    # STEP 10-5-9. Validation article master 생성
    validation_master_result = build_validation_article_master()

    # STEP 10-5-10. 최종 결과 반환 
    usage_stats = {
        key: value
        for key, value in validation_usage_result.items()
        if key != "used_article_ids"
    }

    return {
        "status": "SUCCESS",

        "validation_usage": usage_stats,

        "validation_used_article_count": int(
            len(validation_used_article_ids)
        ),
        "validation_train_overlap_article_count": int(
            len(validation_train_overlap_article_ids)
        ),
        "validation_only_article_count": int(
            len(validation_only_article_ids)
        ),

        "validation_used_article_ids_path": str(
            config.VALIDATION_USED_ARTICLE_IDS_PATH
        ),
        "validation_only_article_ids_path": str(
            config.VALIDATION_ONLY_ARTICLE_IDS_PATH
        ),

        "event_result": validation_event_result,
        "article_master_result": validation_master_result,
        
    }

# STEP 10-6. build_validation.py 실행
if __name__ == "__main__":
    result = build_validation()

    print()
    print("=" * 70)
    print("Validation Build 완료")
    print("=" * 70)

    pprint(result)