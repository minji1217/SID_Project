import polars as pl
from typing import Any 
import numpy as np
import torch
import torch.nn.functional as F
from collections import Counter
from datetime import timedelta
import math 
import unicodedata # entity 문자열 정규화용 

from rich.progress import track
from transformers import AutoModel, AutoTokenizer

from src.config import (
    ARTICLES_PATH,
    TRAIN_BEHAVIORS_PATH,
    TRAIN_HISTORY_PATH,

    ARTICLES_BASE_PATH,
    TRAIN_USED_ARTICLE_IDS_PATH,
    CATEGORY_MAPPING_PATH,
    ARTICLES_WITH_CATEGORY_PATH,

    ARTICLE_EMBEDDING_INPUT_PATH,
    ARTICLE_EMBEDDINGS_PATH,

    ARTICLE_EVENTS_PATH,
    EVENT_MASTER_PATH,
    ENTITY_IDF_PATH,

    ARTICLE_EMBEDDING_MODEL_NAME,
    ARTICLE_EMBEDDING_MAX_LENGTH,
    ARTICLE_EMBEDDING_BATCH_SIZE,

    EVENT_ENTITY_SIMILARITY_THRESHOLD,
    EVENT_TIME_WINDOW_HOURS,

    EVENT_MAX_ENTITY_DF_RATIO, 

    create_output_directories,
)



# STEP2. 모델에서 사용할 유효 기사 데이터 생성
def build_valid_articles() -> dict[str, Any]:
    """
    원본 articles.parquet에서 모델이 사용할 수 있는 기사만 선별하고,
    임베딩 모델에 입력할 model_text 컬럼 생성 
    
    제외 조건
    ---------
    다음 조건에 하나라도 해당하는 기사 행은 제외함
    1. article_id가 null인 기사 
    2. article_id가 중복된 기사 
    3. title과 subtitle이 모두 비어있는 기사 
    4. published_time이 null인 기사
    5. ner_clusters와 entity_groups의 길이가 다른 기사 

    유지 조건 (예외)
    ---------------
    category 또는 category_str이 없는 기사는 현재 단계에서 제외 x 
    # -> category_str을 기준으로 model_category_id를 만들고
    # 이후 Semantic ID의 c1로 직접 사용하므로 원본 category 누락은 허용

    model_text 생성 규칙
    -------------------
    1. title과 subtitle 모두 있는 경우 
    -> query: 제목 부제목

    2. title만 있는 경우
    -> query: 제목

    3. subtitle만 있는 경우
    -> query: 부제목 

    이때, None, null, [SEP] 같은 문자열은 model_text에 넣지 않는다.

    반환값
    ------------
    원본 행 수, 유효 기사 수, 제외 기사 수,
    제외 사유별 행 수와 저장 경로 딕셔너리로 반환
    """

    # STEP 2-1. 출력 디렉토리 생성 
    # data/output/reports와 data/ouput/model_inputs 경로가 존재하지 않는 경우 생성하도록
    create_output_directories() # 미리 만들어두긴 했음 

    # STEP 2-2. 원본 articles.parquet 파일 읽기 
    # 원본 데이터의 모든 컬럼을 읽고, model_text 컬럼만 새롭게 추가
    articles = pl.read_parquet(ARTICLES_PATH)

    # 원본 컬럼 순서 나중에 그대로 유지하기 위해 저장
    original_columns = articles.columns 

    # STEP 2-3. 필요한 컬럼 존재 여부 확인
    # 유효 기사 판별과 model_text 생성에 필요한 컬럼들
    required_columns = [
        "article_id",
        "title",
        "subtitle",
        "category",
        "category_str",
        "published_time",
        "ner_clusters",
        "entity_groups",
    ]

    # 원본 파일에 없는 필수 컬럼 찾기 
    missing_columns = [
        column_name
        for column_name in required_columns 
        if column_name not in articles.columns 
    ]

    # 필요한 컬럼이 하나라도 없으면 중단 
    if missing_columns: 
        raise ValueError(
            "articles.parquet에 필요한 컬럼이 없습니다: " + ", ".join(missing_columns)
        )

    # STEP 2-4. 계산용 임시 컬럼 생성 
    # 빈 값 판정, 리스트 길이 불일치 검사 등 하려고 
    # 처리 방식 (아래 2줄)
    # title과 subtitle은 문자열 형식으로 변환 -> null은 빈 문자열로 -> 문자열 앞뒤 공백 제거 
    # ner_clusters와 entity_groups가 null이면 길이를 0으로

    prepared_articles = articles.with_columns(
        [
            pl.col("title").cast(pl.Utf8, strict=False).fill_null("").str.strip_chars().alias("_clean_title"),
            pl.col("subtitle").cast(pl.Utf8, strict=False).fill_null("").str.strip_chars().alias("_clean_subtitle"),
            pl.col("ner_clusters").list.len().fill_null(0).alias("_ner_length"),
            pl.col("entity_groups").list.len().fill_null(0).alias("_entity_group_length"),
        ]
    )

    # STEP 2-5. 제외 사유별 임시 컬럼 생성 (위에서 생성한 표 처리)
    # 제외 안되면 False 
    flagged_articles = prepared_articles.with_columns(
        [
            pl.col("article_id").is_null().alias("_exclude_article_id_null"),
            # 같은 article_id가 2번 이상 존재하는 경우, 중복된 ID 가진 모든 행 True로 표시 
            pl.col("article_id").is_duplicated().alias("_exclude_article_id_duplicate"),
            # title과 subtitle 모두 빈 문자열인지 확인
            (
                (pl.col("_clean_title") == "")
                &
                (pl.col("_clean_subtitle") == "")
            ).alias("_exclude_empty_text"), 
            # 기사 발행 시각이 null인지 확인
            pl.col("published_time").is_null().alias("_exclude_published_time_null"),
            # 같은 기사 행에서 ner_clusters와 entity_groups의 리스트 길이가 다른지 확인
            (
                pl.col("_ner_length") != pl.col("_entity_group_length")
            ).alias("_exclude_ner_entity_length_mismatch"),

            # E5 모델에 입력할 model_text 생성
            # 1. 제목, 부제목 모두 존재하는 경우
            pl.when(
                (pl.col("_clean_title")!= "")& (pl.col("_clean_subtitle")!="")
            ).then(pl.concat_str(
                [
                    pl.lit("query: "),
                    pl.col("_clean_title"),
                    pl.lit("\n"),
                    pl.col("_clean_subtitle"),
                ]
                )
            )
            # 2. 제목만 존재하는 경우 
            .when(
                pl.col("_clean_title")!=""
            ).then(
                pl.concat_str(
                    [
                        pl.lit("query: "),
                        pl.col("_clean_title"),
                    ]
                )
            )
            # 3. 부제목만 존재하는 경우
            .otherwise(
                pl.concat_str(
                    [
                        pl.lit("query: "),
                        pl.col("_clean_subtitle"),
                    ]
                )
            ).alias("model_text"),
        ]
    )
    # _exclude_article_id_null	_exclude_article_id_duplicate	
    # _exclude_empty_text	_exclude_published_time_null	
    # _exclude_ner_entity_length_mismatch	model_text 
    # 총 6개 컬럼 추가됨

    # STEP 2-6. 최종 제외 여부 계산 
    # 위의 5개 컬럼 하나라도 True면 True로 만들기 
    flagged_articles = flagged_articles.with_columns(
        (
            pl.col("_exclude_article_id_null")
            | pl.col("_exclude_article_id_duplicate")
            | pl.col("_exclude_empty_text")
            | pl.col("_exclude_published_time_null")
            | pl.col("_exclude_ner_entity_length_mismatch")
        ).alias("_exclude_any")
    )

    # STEP 2-7. 원본 전체 기사 행 수 계산
    # validate_data.py 에서도 원본 행 수 확인했지만, 실제 제외 행 수 계산 위해 재계산
    original_row_count = articles.height 

    # STEP 2-8. 모델에서 사용할 유효 기사만 남기기 
    # _exclude_any가 False인 행만 선택 
    # 아래 5개를 모두 통과한 기사만 남음
    # 1. article_id_null
    # 2. article_id 중복
    # 3. title, subtitle 모두 비어있음
    # 4. published_time null
    # 5. NER과 entity type 리스트 길이 불일치

    valid_articles = (
        # exclude_any가 False인 행만 남기기 (살아남은 행)
        flagged_articles.filter(~pl.col("_exclude_any"))
        .select(
            [
                # 원본 articles.parquet 불러오고, _exclude 컬럼들 다 지우고 model_text만 추가 
                *original_columns, 
                # 새롭게 생성한 입력 테스트 컬럼 추가
                "model_text",
            ]
        )
    )

    # STEP 2-9. 최종 유효 기사 수와 제외 기사 수 계산 
    # 즉, 최종적으로 모델 입력으로 사용할 수 있는 기사 행 수 
    valid_row_count = valid_articles.height 

    # 실제 제외된 전체 기사 행 수 
    exclude_row_count = (
        original_row_count-valid_row_count
    )

    # STEP 2-10. 유효 기사 데이터를 parquet 파일로 저장
    # 저장 위치 : data/output/model_inputs/articles_base.parquet 
    # 원본 data/raw/articles.parquet은 수정 안함
    valid_articles.write_parquet(
        ARTICLES_BASE_PATH,
        compression="zstd", # 압축알고리즘 중 하나 (압축률 굳, 속도도 굳) 
    )

    # STEP 2-11. 기사 빌드 결과 반환
    # 빌드 전후 개수와 저장 경로만 반환
    return {
        # 함수 실행 완료 상태 뜻함
        "status": "SUCCESS", 
        # 읽은 원본 파일 경로
        "input_path": str(ARTICLES_PATH),
        # 생성한 결과 파일 경로
        "output_path": str(ARTICLES_BASE_PATH),
        # 원본 articles.parquet 전체 행 수
        "original_row_count": int(
            original_row_count
        ),
        # 최종 모델 입력에 사용할 수 있는 기사 수
        "valid_row_count": int(valid_row_count),
        # 실제 제외된 전체 기사 수
        "excluded_row_count": int(exclude_row_count),
    }




# STEP 3. train에서 실제 사용하는 기사 ID 수집 
def collect_train_used_article_ids() -> dict[str,Any]:
    """
    train history와 behaviors에서 실제 학습에 사용되는 기사 ID 수집 

    수집 대상
    -----------
    1. train history의 article_id_fixed
    2. 사용 가능한 클릭 behavior의 현재 article_id
    3. 사용 가능한 클릭 behavior의 클릭 target article_id

    behavior 처리 규칙
    -----------------
    아래 7개 조건 만족하는 behavior만 사용
    1. impression_id가 null이 아님
    2. impression_id가 중복되지 않음
    3. user_id가 null이 아님
    4. impression_time이 null이 아님
    5. 클릭 리스트가 null 또는 빈 리스트 아님
    6. 클릭 리스트 내부에 null 없음
    7. stable dedup후 고유 클릭 target 기사 ID (multi-target도 모두포함)


    마지막엔 articles_base.parquet에 존재하는 유효 article_id와 교집합 계산
    """

    # STEP 3-1. 출력 디렉토리 생성
    create_output_directories()

    # STEP 3-2. 유효 기사 파일 존재 여부 확인
    if not ARTICLES_BASE_PATH.exists():
        raise FileNotFoundError(
            "articles_base.parquet 파일이 없습니다."
            " build_valid_articles()를 먼저 실행하세요."
        )

    # STEP 3-3. 필요한 데이터 읽기 
    # 앞단에서 확정한 유효 기사 ID 목록
    valid_articles = pl.read_parquet(
        ARTICLES_BASE_PATH,
        columns=[
            "article_id",
        ],
    )

    # train 사용자의 과거 기사 목록
    # train 사용자의 과거 기사 목록
    train_history = pl.read_parquet(
        TRAIN_HISTORY_PATH,
        columns=[
            "user_id",
            "article_id_fixed",
            "impression_time_fixed",
        ],
    )

    # train behavior에서 실제 클릭 샘플 고르기 위한 컬럼
    train_behaviors = pl.read_parquet(
        TRAIN_BEHAVIORS_PATH,
        columns=[
            "impression_id",
            "user_id",
            "impression_time",
            "article_id",
            "article_ids_clicked",
        ]
    )

    # STEP 3-4. 유효 기사 ID 집합 생성
    valid_article_ids = set(
        valid_articles.get_column("article_id").to_list()
    )

    # 결과 parquet의 article_id 데이터 타입을 원본 기사 ID 타입과 동일하게 유지하기 위해
    article_id_dtype = (
        valid_articles.get_column("article_id").dtype
    )

    # STEP 3-5. train history 기사 ID 수집 
    #
    # build_sequences.py와 동일한 history 유효성 정책을 적용한다.
    # - user_id null / 중복 -> 해당 history 행 사용 안 함
    # - list 자체 null -> 해당 history 행 사용 안 함
    # - article/time 길이 불일치 -> 해당 history 행 사용 안 함
    # - 내부 article/time null -> 해당 pair만 제거

    duplicated_history_user_ids = set(
        train_history
        .filter(
            pl.col("user_id").is_not_null()
            &
            pl.col("user_id").is_duplicated()
        )
        .get_column("user_id")
        .to_list()
    )

    history_article_ids: set[int] = set()

    for row in train_history.iter_rows(named=True):
        user_id = row["user_id"]
        article_ids = row["article_id_fixed"]
        impression_times = row["impression_time_fixed"]

        # history 행 자체를 사용할 수 없는 경우
        if user_id is None:
            continue

        if user_id in duplicated_history_user_ids:
            continue

        if article_ids is None or impression_times is None:
            continue

        if len(article_ids) != len(impression_times):
            continue

        # 내부 null은 pair만 제거
        for article_id, impression_time in zip(
            article_ids,
            impression_times,
        ):
            if article_id is None:
                continue

            if impression_time is None:
                continue

            history_article_ids.add(
                int(article_id)
            )

    # STEP 3-6. 중복 impression_id 목록 생성 
    # 동일한 impression_id가 여러 행에 있음 안되기에 해당 ID에 속한 모든 행 제외
    duplicated_impression_ids = set(
        train_behaviors.filter(
            pl.col("impression_id").is_not_null()
            &
            pl.col("impression_id").is_duplicated()
        ).get_column("impression_id").to_list()
    )

    # STEP 3-7. behavior 기사 ID 저장 공간 생성
    # 조건을 모두 통과한 behavior에서 현재 기사와 클릭 target 기사를 모으기 위한 집합 준비

    # 사용자가 behavior 발생 당시 보고 있던 현재기사 ID
    current_article_ids: set[Any] = set()

    # stable dedup 후 고유 클릭이 1개인 target 기사 ID
    target_article_ids: set[Any] = set()

    # 실제 클릭 학습 샘플로 사용할 수 있는 behavior 행 수
    usable_behavior_row_count = 0 

    # STEP 3-8. train behavior 한 행씩 처리
    # 학습에 사용할 수 있는 클릭 행만 선택 
    for row in train_behaviors.iter_rows(
        named= True # 각 행을 컬럼명이 붙은 딕셔너리로 가져오기 
    ):
        # 현재 behavior 행의 주요값들을 가져옴
        impression_id = row["impression_id"]
        user_id = row["user_id"]
        impression_time = row["impression_time"]

        # behavior 발생 당시 사용자가 보고 있던 현재 기사
        current_article_id = row["article_id"]

        # 해당 impression에서 클릭한 기사 ID 목록
        clicked_ids = row["article_ids_clicked"]

        # STEP 3-8-1. 기본 식별 정보 검사
        # 1. impression_id 없는 경우
        if impression_id is None: continue 
        # 2. 중복 impression_id인 경우 
        if impression_id in duplicated_impression_ids: continue 
        # 3. user_id 없는 경우 
        if user_id is None: continue 
        # 4. impression_time이 없는 경우 
        if impression_time is None: continue 

        # STEP 3-8-2. 클릭 리스트 기본 검사 
        # 학습 target 없거나 클릭 id가 잘못된 경우 제외 
        # 1. 클릭 리스트 자체가 null인 경우
        if clicked_ids is None: continue 
        # 2. 클릭 리스트가 빈 리스트인 경우 
        if len(clicked_ids) == 0: continue 
        # 3. 클릭 리스트 내부에 null이 존재하는 경우 
        valid_clicked_ids = [
            clicked_article_id for clicked_article_id in clicked_ids if clicked_article_id is not None 
        ]

        # null 제거 후 클릭 기사가 하나도 없으면 target이 없으므로 제외
        if len(valid_clicked_ids) == 0: continue 

        # STEP 3-8-3. 클릭 목록 stable dedup
        # 클릭 목록의 원순서는 유지하며 동일한 기사 ID가 반복된 경우 중복 제거 
        # 지금까지 한 번이라도 등장한 클릭 ID 저장
        seen_clicked_ids: set[Any] = set()

        # 원순서 유지하며 중복 제거한 결과 저장
        unique_clicked_ids: list[Any] = []

        # 클릭 ID를 원순서로 확인
        for clicked_article_id in valid_clicked_ids:
            # 아직 등장하지 않은 기사 ID만 결과에 추가 
            if clicked_article_id not in seen_clicked_ids:
                seen_clicked_ids.add(
                    clicked_article_id
                )

                unique_clicked_ids.append(clicked_article_id)

        # STEP 3-8-4. 클릭 behavior만 선택
        # stable dedup 후 target이 하나 이상이면 사용
        # 여러 target도 multi-positive 정답으로 모두 유지

        # 즉, stable dedup 결과가 하나 이상이면 사용 
        if len(unique_clicked_ids) == 0:
            continue 
        

        # 현재 행이 모든 조건 통과했기에 실제 사용 가능한 behavior에 포함됨
        usable_behavior_row_count += 1

        # 유일한 클릭 기사를 target 기사로 저장
        target_article_ids.update(unique_clicked_ids)

        # 현재 artciel_id는 원본 데이터에서 null이 허용되므로,
        # null이 아닌 경우만 train 사용 기사 목록에 추가됨
        if current_article_id is not None: 
            current_article_ids.add(
                current_article_id
            )

    # STEP 3-9. train에서 참조된 전체 기사 ID 합치기 
    # train history, 현재 기사, 클릭 target에서 수집한 기사 ID를 
    # 하나의 전체 집합으로 합치기 
    train_referenced_article_ids = (
        history_article_ids
        | current_article_ids
        | target_article_ids
    )

    # STEP 3-10. 유효 기사 ID와 교집합 계산 
    # train에서 참조되었더라도 build_valid_articles()에서 
    # 제외된 기사는 모델 입력으로 사용 불가
    # -> train 참조 기사와 유효 기사 집합의 교집합만 train 사용 기사로 확정 
    train_used_article_ids = (
        train_referenced_article_ids & valid_article_ids
    )

    # 제외 기사 ID도 별도로 확인 
    excluded_reference_article_ids = (
        train_referenced_article_ids - valid_article_ids
    )

    # STEP 3-11. 결과 DataFrame 생성 
    # 최종 train 사용 기사 ID를 parquet으로 저장할 수 있는 Polars DF 형태로
    # article_id 정렬
    train_used_articles = pl.DataFrame(
        {
            "article_id": pl.Series(name="article_id",
                                     values=sorted(train_used_article_ids),
                                       dtype=article_id_dtype,)
        }
    )

    # STEP 3-12. train 사용 기사 ID 저장
    # train 사용 기사들의 카테고리로만 매핑표 만들때 필요 
    # train 사용 기사들의 임베딩 만들때 필요 
    # history/behaviors로 시퀀스 만들 때 이 기사가 실제 SID 갖고 있는지 확인 위함 용도 
    # -> 다시 순회하지않고 이 파일 읽어서 즉시 확인하는 용도 
    train_used_articles.write_parquet(
        TRAIN_USED_ARTICLE_IDS_PATH,
        compression="zstd",
    )

    # STEP 3-13. 최종 상태 결정
    # train에서 참조한 기사 중 유효 기사 집합에서 제외된 기사 존재하는지
    if excluded_reference_article_ids:
        status = "WARNING"
    else: 
        status = "SUCCESS"

    # STEP 3-14. 실행 결과 반환
    return {
        # SUCCESS 또는 WARNING
        "status": status,

        # 생성한 Parquet 파일 경로
        "output_path": str(
            TRAIN_USED_ARTICLE_IDS_PATH
        ),

        # articles_base.parquet에 존재하는 전체 유효 기사 수
        "valid_article_count": int(
            len(valid_article_ids)
        ),

        # 모든 조건을 통과한 단일 클릭 behavior 행 수
        "usable_behavior_row_count": int(
            usable_behavior_row_count
        ),

        # train history와 behavior에서 참조된 고유 기사 수
        "train_referenced_article_count": int(
            len(train_referenced_article_ids)
        ),

        # 유효 기사와 교집합을 구한 최종 train 사용 기사 수
        "train_used_article_count": int(
            len(train_used_article_ids)
        ),

        # train에서 참조됐지만 유효 기사에서는 제외된 기사 수
        "excluded_reference_article_count": int(
            len(excluded_reference_article_ids)
        ),

        # 제외된 참조 기사 ID를 최대 10개까지 보여준다.
        "excluded_reference_article_examples": sorted(
            excluded_reference_article_ids
        )[:10],
    }    

# STEP4. train 기준 카테고리 매핑 생성
def build_category_mapping()-> dict[str, Any]:
    """
    train에서 실제 사용하는 기사만 기준으로 category_str을 model_category_id로 변환할 매핑 생성

    매핑 원칙
    -----------
    1. <UNK> 카테고리는 model_category_id 0으로
    2. train 기사에 등장한 정상 category_str만 매핑 수행
    3. 정상 카테고리는 문자열 기준으로 정렬한 뒤 1부터 ID 부여
    4. validation 데이터는 매핑 생성에 사용 x
    5. 이후 train에서 보지 못한 카테고리는 <UNK>=0으로 처리
    
    생성 결과 예시
    -------------
    category_str | model_category_id
    --------------------------------
    <UNK>        | 0
    culture      | 1
    economy      | 2
    sport        | 3    

    """

    # STEP 4-1. 출력 디렉토리 생성
    create_output_directories()

    # STEP 4-2. 선행 결과 파일 존재 여부 확인
    # 카테고리 매핑 만들기 위해 필요한 유효 기사 파일과 train 사용 기사 ID 파일 먼저 생성됐는지 확인
    if not ARTICLES_BASE_PATH.exists():
        raise FileNotFoundError(
            "articles_base.parquet 파일이 없습니다. "
            "build_valid_articles()를 먼저 실행해야 합니다."
        )

    if not TRAIN_USED_ARTICLE_IDS_PATH.exists():
        raise FileNotFoundError(
            "train_used_article_ids.parquet 파일이 없습니다. "
            "collect_train_used_article_ids()를 먼저 실행해야 합니다."
        )

    # STEP 4-3. 필요 데이터 읽기
    # 전체 유효 기사에서 article_id와 category_str만 읽고
    # train_used_article_ids에서는 실제 학습에 사용하는 기사 ID만 읽는다. 
    article_categories = pl.read_parquet(
        ARTICLES_BASE_PATH,
        columns=[
            "article_id",
            "category_str",
        ],
    )

    train_used_article_ids=pl.read_parquet(
        TRAIN_USED_ARTICLE_IDS_PATH,
        columns=[
            "article_id",
        ],
    )

    # STEP 4-4. train 사용 기사에 articles_base의 category_str 연결
    # article_id | category_str
    #     201    |   "정치"
    #     202    |   "스포츠"
  
    
    train_articles = (
        train_used_article_ids.join(
            article_categories,
            on="article_id",
            how="inner"
        )
    )

    # STEP 4-5. category_str 정리
    # null과 빈 문자열 동일하게 처리하고, 앞뒤 공백 차이로 같은 카테고리가 별도 ID 받는 것 방지
    # "sport" "  sport" -> "sport"
    # null -> ""

    train_articles = train_articles.with_columns(
        pl.col("category_str").cast(pl.Utf8, strict=False)
        .fill_null("")
        .str.strip_chars()
        .alias("_clean_category_str")
    )

    # STEP 4-6. <UNK> 처리 대상 기사 수 계산
    # category_str이 null, 빈문자열인 기사가 train 사용 기사 중 몇개 존재?
    # 이 기사들은 이후 model_category_id 0으로 처리
    train_unknown_category_article_count = (
        train_articles.select(
            pl.col("_clean_category_str")
            .eq("") # 각 값이 빈 문자열인지 true/false로 
            .sum()  # tru를 1, false를 0으로 세서 더함
            .alias("count")
        ).item()
    )

    # STEP 4-7. train의 정상 카테고리 추출
    # train 기사에 등장한 비어 있지 않은 카테고리만 추출하고,
    # 중복 제거하여 고유 카테고리 목록 만듦
    # 문자열 기준으로 정렬해서 같은 카테고리에 같은 정수 ID가 부여되도록
    known_categories = (
        train_articles.filter(
            pl.col("_clean_category_str") != ""
        ).select(
            pl.col("_clean_category_str").alias("category_str")
        ).unique().sort("category_str")
    )

    # STEP 4-8. 실제 원본 카테고리에 <UNK> 존재 ? 
    reserved_token_count = (
        known_categories.filter(pl.col("category_str")=="<UNK>").height)
    if reserved_token_count > 0:
        raise ValueError("원본 category_str에 예약 문자열 <UNK> 존재")

    # STEP 4-9. 정상 카테고리에 정수 ID 부여
    # 정렬된 정상 카테고리에 1부터 시작하는 연속된 model_category_id 부여 
    # 0은 <UNK>으로 사용하기에 offset=1로 
    known_category_mapping= (
        known_categories
        .with_row_index(
            name="model_category_id",
            offset=1
        ).select(
            [
                "category_str",
                "model_category_id"
            ]
        ).with_columns(
            pl.col("model_category_id").cast(pl.Int32)
        )
    )

    # STEP 4-10. <UNK>=0 매핑 
    # train에서 카테고리가 비어있거나, 이후 validation에만 등장하는 
    # 새 카테고리를 model_cateogry_id 0으로 변환할 수 있게 
    unknown_category_mapping = pl.DataFrame(
        {
            "category_str": pl.Series(
                name="category_str",
                values=[
                    "<UNK>",
                ],
                dtype=pl.Utf8
            ),
            "model_category_id": pl.Series(
                name="model_category_id",
                values=[
                    0,
                ],
                dtype=pl.Int32
            )
        }
    )

    # STEP 4-11. 최종 카테고리 매핑 결합
    # <UNK>=0행과 train에서 추출한 정상 카테고리 매핑을 하나의 최종 매핑 테이블로
    category_mapping = pl.concat(
        [
            unknown_category_mapping,
            known_category_mapping,
        ], how= "vertical",
    )

    # STEP 4-12. 카테고리 매핑 저장
    # 이후 전체 기사에 model_category_id 붙일 때 
    # 동일한 train 기준 매핑 재사용할 수 있도록 저장
    category_mapping.write_parquet(
        CATEGORY_MAPPING_PATH,
        compression="zstd",
    )

    # STEP 4-13. 실행 결과 반환
    # 매핑 생성에 사용된 기사 수와 카테고리 수, 저장 위치 출력
    return {
        # 함수가 정상적으로 끝났음을 의미한다.
        "status": "SUCCESS",

        # 생성한 카테고리 매핑 파일 경로
        "output_path": str(
            CATEGORY_MAPPING_PATH
        ),

        # 매핑 생성 기준이 된 train 사용 기사 수
        "train_used_article_count": int(
            train_used_article_ids.height
        ),

        # 카테고리가 비어 있어 <UNK>=0으로 처리될 train 기사 수
        "train_unknown_category_article_count": int(
            train_unknown_category_article_count
        ),

        # <UNK>를 제외한 train의 정상 고유 카테고리 수
        "known_category_count": int(
            known_category_mapping.height
        ),

        # <UNK> 행까지 포함한 최종 매핑 행 수
        "mapping_row_count": int(
            category_mapping.height
        ),
    }    

# STEP 5. 전체 유효 기사에 카테고리 매핑 적용
def apply_category_mapping_to_articles() -> dict[str, Any]:
    """
    train 기사만 기준으로 만든 category_mapping.parquet을
    전체 유효 기사에 적용

    적용 대상
    ----------
    articles_base.parquet에 포함된 전체 유효 기사

    적용 규칙
    ----------
    1. train에서 발견된 category_str이면 기존 ID 사용
    2. train에서 발견되지 않은 category_str이면 0 사용
    3. category_str이 null 또는 빈 문자열이어도 0 사용
    4. 원본 category_str 컬럼은 그대로 보존
    5. model_category_id 컬럼 새로 추가 

    결과 예시
    category_str | model_category_id
    --------------------------------
    economy      | 1
    sport        | 2
    new_category | 0
    null         | 0    
    """

    # STEP 5-1. 출력 디렉토리 생성
    create_output_directories()

    # STEP 5-2. 선행 결과 파일 존재 여부 확인
    if not ARTICLES_BASE_PATH.exists():
        raise FileNotFoundError(
            "articles_base.parquet 파일이 없습니다. "
            "build_valid_articles()를 먼저 실행해야 합니다."
        )

    if not CATEGORY_MAPPING_PATH.exists():
        raise FileNotFoundError(
            "category_mapping.parquet 파일이 없습니다. "
            "build_category_mapping()을 먼저 실행해야 합니다."
        )

    # STEP 5-3. 전체 유효 기사 읽기 
    # train, val에서 공통으로 사용할 전체 유효 기사 20,719개 읽기 
    # 순서 유지하도록 순서 컬럼 함께 생성 
    articles=(
        pl.read_parquet(
            ARTICLES_BASE_PATH
        ).with_row_index(
            name="_article_order"
        )
    )

    # STEP 5-4. train 기준 카테고리 매핑 읽기
    # train 기사만 기준으로 만든 category_str과 model_category_id 매핑 읽기
    category_mapping = pl.read_parquet(
        CATEGORY_MAPPING_PATH,
        columns=[
            "category_str",
            "model_category_id",
        ]
    )

    # STEP 5-5. 정상 카테고리 매핑만 분리 
    # <UNK>=0행은 실제 category_str과 직접 연결하지않고
    # 매핑되지 않은 모든 기사에 나중에 0 채우는 용도로 사용
    known_category_mapping = (
        category_mapping.
        filter(
            pl.col("category_str")!="<UNK>"
        ).rename(
            {
                "category_str": "_clean_category_str",
            }
        )
    )

    # STEP 5-6. 전체 기사의 category_str 정리
    # 예:
    # "sport"   -> "sport"
    # " sport " -> "sport"
    # null      -> ""
    articles = articles.with_columns(
        pl.col("category_str")
        .cast(
            pl.Utf8,
            strict=False,
        )
        .fill_null("")
        .str.strip_chars()
        .alias("_clean_category_str")
    )

    # STEP 5-7. 전체 기사에 train 카테고리 매핑 연결
    # 전체 유효 기사의 정리된 category_str을 기준으로
    # train에서 만든 model_category_id를 붙인다.
    articles_with_category = (
        articles
        .join(
            known_category_mapping,
            on="_clean_category_str",
            how="left",
        )
    )    

    # STEP 5-8. 매핑되지않은 카테고리는 <UNK>=0으로 처리
    # join 결과 null이 된 행은 예약 ID인 0으로 변환
    articles_with_category = (
        articles_with_category
        .with_columns(
            pl.col("model_category_id")
            .fill_null(0)
            .cast(pl.Int32)
        )
    )

    # STEP 5-9. 빈 카테고리 기사 수 계산
    # model_category_id 0으로 처리된 기사 수 확인
    blank_category_article_count = (
        articles_with_category
        .filter(
            pl.col("_clean_category_str") == ""
        )
        .height
    )

    # STEP 5-10. train에서 보지 못한 카테고리 확인
    # validation에서만 사용되는 새 카테고리나
    # train 학습 기사엔 등장하지 않은 카테고리 포함 가능
    unseen_category_articles = (
        articles_with_category
        .filter(
            (
                pl.col("_clean_category_str") != ""
            )
            & (
                pl.col("model_category_id") == 0
            )
        )
    )

    # STEP 5-11. 보지 못한 카테고리 통계 계산
    # train에서 보지 못한 카테고리에 해당하는 기사 수와 
    # 실제 고유 카테고리 종류 수 각각 계산 
    unseen_category_article_count = (
        unseen_category_articles.height 
    ) 

    unseen_category_count = (
        unseen_category_articles.
        select(
            "_clean_category_str"
        ).unique().height
    )

    # train에서 보지 못한 카테고리 직접 확인 (최대 10개)
    unseen_category_examples = (
        unseen_category_articles
        .select(
            "_clean_category_str"
        )
        .unique()
        .sort(
            "_clean_category_str"
        )
        .head(10)
        .get_column(
            "_clean_category_str"
        )
        .to_list()
    )

    # STEP 5-12. 알려진 카테고리(우리가 매핑한)와 <UNK> 기사 수 계산
    known_category_article_count = (
        articles_with_category
        .filter(
            pl.col("model_category_id") != 0
        )
        .height
    )

    unknown_category_article_count = (
        articles_with_category
        .filter(
            pl.col("model_category_id") == 0
        )
        .height
    )

    # STEP 5-13. 임시 컬럼 제거
    # 정리용 컬럼과 순서보존용 컬럼 제거 
    # 예: " sport", "sport" -> "sport"
    final_articles = (
        articles_with_category
        .sort(
            "_article_order"
        )
        .drop(
            [
                "_article_order",
                "_clean_category_str",
            ]
        )
    )

    # STEP 5-14. 전체 기사 결과 저장 
    # 이후 임베딩, 이벤트 클러스터링, RQ-VAE 입력 생성에서
    # 모든 기사가 동일한 model_category_id 사용하도록 저장
    final_articles.write_parquet(
        ARTICLES_WITH_CATEGORY_PATH,
        compression="zstd",
    )

    # STEP 5-15. 실행 결과 반환
    return {
        # 함수가 정상적으로 완료됐음을 의미한다.
        "status": "SUCCESS",

        # 생성된 전체 기사 카테고리 파일 경로
        "output_path": str(
            ARTICLES_WITH_CATEGORY_PATH
        ),

        # 카테고리 매핑을 적용한 전체 유효 기사 수
        "article_count": int(
            final_articles.height
        ),

        # <UNK>를 포함한 전체 카테고리 매핑 행 수
        "mapping_row_count": int(
            category_mapping.height
        ),

        # train에서 학습한 정상 카테고리 ID를 받은 기사 수
        "known_category_article_count": int(
            known_category_article_count
        ),

        # model_category_id 0으로 처리된 전체 기사 수
        "unknown_category_article_count": int(
            unknown_category_article_count
        ),

        # category_str 자체가 비어 있어서 0으로 처리된 기사 수
        "blank_category_article_count": int(
            blank_category_article_count
        ),

        # category_str은 있지만 train에서 보지 못한 기사 수
        "unseen_category_article_count": int(
            unseen_category_article_count
        ),

        # train에서 보지 못한 고유 카테고리 종류 수
        "unseen_category_count": int(
            unseen_category_count
        ),

        # train에서 보지 못한 카테고리 예시 최대 10개
        "unseen_category_examples": (
            unseen_category_examples
        ),
    }

# STEP 6. 기사 임베딩 입력 데이터 생성
def build_article_embedding_input() -> dict[str, Any]:
    """
    전체 유효 기사를 article_id 기준으로 정렬하고,
    각 기사에 고정된 embedding_row 부여 

    생성 컬럼
    ---------
    embedding_row : 임베딩 배열에서 해당 기사가 저장될 행 번호(0부터)
    article_id : 원본 기사 id
    model_text : E5 임베딩 모델에 입력할 최종 텍스트

    처리 원칙
    ---------
    1. train과 validation 기사를 따로 임베딩하지 않는다.
    2. 전체 유효 기사 20,719개를 한 번만 임베딩한다.
    3. article_id 기준으로 정렬해 실행할 때마다 순서를 고정한다.
    4. embedding_row와 article_id의 관계를 이후 모든 단계에서 유지한다.
    """

    # STEP 6-1. 출력 디렉토리 생성
    create_output_directories()

    # STEP 6-2. 선행 결과 파일 존재 여부 확인
    if not ARTICLES_WITH_CATEGORY_PATH.exists():
        raise FileNotFoundError(
            "articles_with_category.parquet 파일이 없습니다. "
            "apply_category_mapping_to_articles()를 먼저 실행해야 합니다."
        )

    # STEP 6-3. 임베딩에 필요한 기사 컬럼 읽기
    # 기사 ID와 모델 입력 텍스트만 읽기
    articles=pl.read_parquet(
        ARTICLES_WITH_CATEGORY_PATH,
        columns=[
            "article_id",
            "model_text",
        ]
    )

    # STEP 6-4. article_id 기준으로 기사 순서 고정
    # 프로그램 여러 번 실행해도 동일한 기사가 동일한 embedding_row 받도록 article_id로 정렬
    articles = articles.sort(
        "article_id",
    )

    # STEP 6-5. embedding_row 부여
    # 임베딩 배열의 실제 행 번호와 article_id 연결 위해 
    # 0부터 시작하는 embedding_row 만듦
    # 예:
    # embedding_row=0 -> 첫 번째 기사의 임베딩
    # embedding_row=1 -> 두 번째 기사의 임베딩
    article_embedding_input = (
        articles
        .with_row_index(
            name="embedding_row",
            offset=0,
        )
        .with_columns(
            # 이후 NumPy 배열의 행 인덱스로 사용하기 쉽도록
            # embedding_row 타입을 Int64로 통일한다.
            pl.col("embedding_row")
            .cast(pl.Int64)
        )
        .select(
            [
                "embedding_row",
                "article_id",
                "model_text",
            ]
        )
    )

    # STEP 6-6. 임베딩 입력 파일 저장
    # 별도의 embed_articles.py에서 E5 임베딩을 생성할 때 
    # 동일한 기사 순서와 model_text 사용할 수 있게 저장
    article_embedding_input.write_parquet(
        ARTICLE_EMBEDDING_INPUT_PATH,
        compression="zstd",
    )

    # STEP 6-7. 실행 결과 반환
    return {
        # 함수가 정상적으로 완료됐음을 의미한다.
        "status": "SUCCESS",

        # 생성된 임베딩 입력 파일 경로
        "output_path": str(
            ARTICLE_EMBEDDING_INPUT_PATH
        ),

        # 임베딩을 생성할 전체 유효 기사 수
        "article_count": int(
            article_embedding_input.height
        ),

        # 첫 번째 임베딩 행 번호
        "first_embedding_row": (
            int(
                article_embedding_input
                .get_column("embedding_row")
                .min()
            )
            if article_embedding_input.height > 0
            else None
        ),

        # 마지막 임베딩 행 번호
        "last_embedding_row": (
            int(
                article_embedding_input
                .get_column("embedding_row")
                .max()
            )
            if article_embedding_input.height > 0
            else None
        ),

        # 첫 번째 article_id
        "first_article_id": (
            article_embedding_input
            .get_column("article_id")
            .first()
            if article_embedding_input.height > 0
            else None
        ),

        # 마지막 article_id
        "last_article_id": (
            article_embedding_input
            .get_column("article_id")
            .last()
            if article_embedding_input.height > 0
            else None
        ),
    }

# 평균 pooling 보조 함수 정의
def _average_pool(
    last_hidden_state: torch.Tensor,
    attention_mask:torch.Tensor,
)-> torch.Tensor:
    """
    padding 토큰을 제외하고 실제 텍스트 토큰의 벡터만 평균 낸다.

    파라미터
    ---------
    last_hidden_state : E5 모델이 각 토큰에 대해 출력한 벡터
    -> 형태 : [batch_size, token_length, embedding_dim]
    attention_mask : 실제 토큰은 1, padding 토큰은 0으로 표시한 값
    -> 형태 : [batch_size, token_length]

    반환값
    -------
    torch.Tensor : 문장별 평균 임베딩
    -> 형태 : [batch_size, embedding_dim]
    """

    # attention_mask가 0인 padding 위치를 False로 변환하고,
    # 임베딩 차원에 맞게 마지막 축을 하나 추가
    expanded_attention_mask = (
        attention_mask.unsqueeze(-1).bool()
    )

    # padding 토큰의 벡터는 평균 계산에 포함되지 않도록
    # 모든 값을 0으로 변경한다.
    masked_hidden_state = (
        last_hidden_state
        .masked_fill(
            ~expanded_attention_mask,
            0.0,
        )
    )

    # 각 문장에 실제로 존재하는 토큰 수를 계산한다.
    token_count = (
        attention_mask
        .sum(dim=1)
        .unsqueeze(-1)
        .clamp(min=1)
    )

    # 실제 토큰의 벡터를 모두 더한 후
    # 실제 토큰 수로 나누어 문장 임베딩을 만든다.
    return (
        masked_hidden_state.sum(dim=1)
        / token_count
    )

# STEP 6. 전체 기사 텍스트 임베딩 생성
def generate_article_embeddings(
        batch_size: int = ARTICLE_EMBEDDING_BATCH_SIZE,
)-> dict[str, Any]:
    """
    전체 유효 기사의 model_text를 multilingual-e5-base에 입력해
    768차원 L2 정규화 임베딩 생성

    처리 순서 
    ------------
    1. article_embedding_input.parquet 읽음
    2. embedding_row가 0부터 연속적으로 부여됐는지 확인
    3. model_text가 비어있지 않고 query: 접두어 가지는지 확인
    4. multilingual-e5-base 모델과 tokenizer 불러옴
    5. model_text를 batch 단위로 토큰화
    6. 모델의 토큰별 출력에 average pooling 적용
    7. 각 기사 임베딩 L2 정규화
    8. float32 Numpy 배열로 저장 

    출력
    ----
    article_embeddings.npy

    배열 형태
    ---------
    [기사 수, 임베딩 차원]

    연결 방법
    ---------
    article_embeddings[embedding_row]
    → 해당 embedding_row를 가진 기사의 임베딩
    """
    # STEP 6-1. 출력 디렉터리 생성
    create_output_directories()

    # STEP 6-2. batch_size 검사
    if batch_size <= 0:
        raise ValueError(
            "batch_size는 1 이상의 정수여야 합니다."
        )

    # STEP 6-3. 임베딩 입력 파일 존재 여부 확인
    if not ARTICLE_EMBEDDING_INPUT_PATH.exists():
        raise FileNotFoundError(
            "article_embedding_input.parquet 파일이 없습니다. "
            "build_article_embedding_input()을 먼저 실행해야 합니다."
        )

    # STEP 6-4. 임베딩 입력 데이터 읽기
    embedding_input = pl.read_parquet(
        ARTICLE_EMBEDDING_INPUT_PATH,
        columns=[
            "embedding_row",
            "article_id",
            "model_text",
        ],
    )

    article_count = embedding_input.height

    # 임베딩 대상 기사가 하나도 없으면 모델을 실행할 수 없다.
    if article_count == 0:
        raise ValueError(
            "임베딩을 생성할 기사가 없습니다."
        )
    
    # STEP 6-5. embedding_row 연속성 확인
    # embedding_row가 0부터 기사수-1까지 연속되어야
    # numpy 배열의 실제 행 번호와 일치할 수 있음 
    actual_embedding_rows = (
        embedding_input
        .get_column("embedding_row")
        .to_list()
    )

    expected_embedding_rows = list(
        range(article_count)
    )

    if actual_embedding_rows != expected_embedding_rows:
        raise ValueError(
            "embedding_row가 0부터 연속적으로 구성되어 있지 않습니다."
        )
    
    # STEP 6-6. model_text 검사
    # null 또는 빈 문자열은 정상적인 임베딩 만들 수 없기에 이중확인
    model_text_values = (
        embedding_input
        .get_column("model_text")
        .to_list()
    )

    null_model_text_count = sum(
        text is None
        for text in model_text_values
    )

    if null_model_text_count > 0:
        raise ValueError(
            "model_text에 null 값이 존재합니다. "
            f"null 개수: {null_model_text_count}"
        )

    # null 검사를 통과했으므로 이후 처리에서는 문자열로 사용한다.
    model_texts = [
        str(text)
        for text in model_text_values
    ]

    empty_model_text_count = sum(
        text.strip() == ""
        for text in model_texts
    )

    if empty_model_text_count > 0:
        raise ValueError(
            "model_text에 빈 문자열이 존재합니다. "
            f"빈 문자열 개수: {empty_model_text_count}"
        )

    # STEP 6-7. E5 query 접두어 확인
    # 모든 입력 텍스트가 query: 접두어 형식인지 확인
    missing_query_prefix_count = sum(
        not text.startswith("query: ")
        for text in model_texts
    )

    if missing_query_prefix_count > 0:
        raise ValueError(
            "query: 접두어가 없는 model_text가 존재합니다. "
            f"문제 행 수: {missing_query_prefix_count}"
        )

    # STEP 6-8. 실행 장치 선택 
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # STEP 6-9. tokenizer와 E5 모델 로드
    # model_text를 토큰으로 변환할 tokenizer와 실제 768차원 임베딩 생성할 E5 모델 불러옴
    tokenizer = AutoTokenizer.from_pretrained(
        ARTICLE_EMBEDDING_MODEL_NAME
    )

    model = AutoModel.from_pretrained(
        ARTICLE_EMBEDDING_MODEL_NAME
    )

    # 모델을 CPU 또는 GPU 장치로 이동한다.
    model = model.to(device)

    # dropout과 같은 학습용 동작을 비활성화하고
    # 추론 전용 상태로 전환한다.
    model.eval()

    # 모델 설정에서 실제 출력 임베딩 차원을 가져온다.
    embedding_dim = int(
        model.config.hidden_size
    )

    # STEP 6-10. 최종 임베딩 배열 공간 생성
    # 모든 기사 임베딩 저장할 Numpy 배열 공간 미리 만들기
    # embedding_row가 곧 배열 행 번호이기에 각 batch의 결과를 해당 위치에 바로 저장
    article_embeddings= np.empty(
        (article_count,
        embedding_dim,
    ), dtype=np.float32,)

    # STEP 6-11. batch 단위 임베딩 생성
    batch_start_positions=range(0, article_count, batch_size)

    # track : 진행바 화면에 그려줌 
    for start_index in track(batch_start_positions, description="기사 임베딩 생성 중...",):
        end_index = min(start_index+batch_size, article_count,)

        # 현재 batch에 포함되는 기사 텍스트 가져옴
        batch_texts=model_texts[
            start_index:end_index
        ]

        # STEP 6-11-1. 기사 텍스트 토큰화
        # 문자열을 E5 모델이 처리할 수 있는 token ID와 attention mask 형태로 변환
        tokenized_batch = tokenizer(
            batch_texts,
            max_length = ARTICLE_EMBEDDING_MAX_LENGTH,
            padding = True,
            truncation=True,
            return_tensors="pt", # 파이토치 
        )
        # "input_ids", "attention_mask" 형태 
        # tensor([[101, 4521,...],...]) 
        # tensor([[1,1,...],...])


        # tokenizer 출력도 모델과 같은 CPU 또는 GPU로 
        tokenized_batch = {
            key: value.to(device)
            for key, value in tokenized_batch.items()
        }

        # STEP 6-11.12 E5 모델 추론
        # 학습용 gradient 만들지 않고 토큰별 임베딩만 계산
        with torch.inference_mode():
            model_output = model(
                **tokenized_batch
            )

            # STEP 6-11-3. average pooling 
            # 모델은 각 토큰마다 하나의 벡터 출력하므로
            # padding 제외한 실제 토큰 벡터를 평균 내
            # 기사 하나당 벡터 하나로 만듦 
            batch_embeddings = _average_pool(
                last_hidden_state=(
                    model_output.last_hidden_state
                ),
                attention_mask=(
                    tokenized_batch["attention_mask"]
                ),
            )

            # STEP 6-11-4. L2 정규화
            # 각 임베딩 벡터의 길이를 1로 맞춤
            batch_embeddings = F.normalize(
                batch_embeddings,
                p=2, # L2
                dim=1, # 기사(행)마다 정규화 
            )

        # STEP 6-11-5. Numpy float32 배열로 변환
        # GPU 또는 CPU의 파이토치 텐서를 cpu로 옮기고,
        # 최종 저장 형식인 numpy float32로 변환
        batch_embeddings_numpy = (
            batch_embeddings
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.float32,
                copy=False,
            )
        )

        # 현재 batch의 결과를 embedding_row와 동일한 위치에 저장
        article_embeddings[
            start_index:end_index
        ]= batch_embeddings_numpy 

    # STEP 6-12. 임베딩 배열 형태 확인
    # 1. 생성된 임베딩 행 수 = 기사 수?
    # 2. 열 수 = 모델 출력 차원?
    expected_shape = (
        article_count,
        embedding_dim,
    )

    if article_embeddings.shape != expected_shape:
        raise ValueError(
            "기사 임베딩 배열의 형태가 예상과 다릅니다. "
            f"예상: {expected_shape}, "
            f"실제: {article_embeddings.shape}"
        )

    # STEP 6-13. NaN과 무한대 검사 
    # NaN 또는 무한대가 포함된 벡터는 이후 클러스터링이나
    # 모델 학습을 망가뜨릴 수 있으므로 저장 전에 차단
    if not np.isfinite(
        article_embeddings
    ).all():
        raise ValueError(
            "기사 임베딩에 NaN 또는 무한대 값이 존재합니다."
        )

    # STEP 6-14. L2 정규화 결과 확인
    # 모든 기사 벡터의 길이가 약 1인지 확인 
    embedding_norms = np.linalg.norm(
        article_embeddings,
        axis=1,
    )

    minimum_l2_norm = float(
        embedding_norms.min()
    )

    maximum_l2_norm = float(
        embedding_norms.max()
    )

    if not np.allclose(
        embedding_norms,
        1.0,
        atol=1e-4,
    ):
        raise ValueError(
            "일부 기사 임베딩의 L2 norm이 1이 아닙니다. "
            f"최소 norm: {minimum_l2_norm}, "
            f"최대 norm: {maximum_l2_norm}"
        )

    # STEP 6-15. 실제 임베딩 배열 저장
    # .npy로 저장 
    np.save(
        ARTICLE_EMBEDDINGS_PATH,
        article_embeddings,
        allow_pickle=False, # 순수 숫자 행렬만 저장 
    )

    # STEP 6-16. 실행 결과 반환
    return {
        # 함수가 정상적으로 완료됐음을 의미한다.
        "status": "SUCCESS",

        # 생성한 실제 기사 임베딩 파일 경로
        "output_path": str(
            ARTICLE_EMBEDDINGS_PATH
        ),

        # 사용한 E5 모델 이름
        "model_name": (
            ARTICLE_EMBEDDING_MODEL_NAME
        ),

        # 실제 추론에 사용한 장치
        "device": str(
            device
        ),

        # 임베딩 생성에 사용한 batch 크기
        "batch_size": int(
            batch_size
        ),

        # 토큰화 최대 길이
        "max_length": int(
            ARTICLE_EMBEDDING_MAX_LENGTH
        ),

        # 임베딩이 생성된 기사 수
        "article_count": int(
            article_count
        ),

        # 기사 한 개당 임베딩 차원
        "embedding_dim": int(
            embedding_dim
        ),

        # 최종 NumPy 배열 형태
        "embedding_shape": [
            int(article_embeddings.shape[0]),
            int(article_embeddings.shape[1]),
        ],

        # 최종 저장 자료형
        "dtype": str(
            article_embeddings.dtype
        ),

        # L2 정규화 후 가장 작은 벡터 길이
        "minimum_l2_norm": (
            minimum_l2_norm
        ),

        # L2 정규화 후 가장 큰 벡터 길이
        "maximum_l2_norm": (
            maximum_l2_norm
        ),

        # 저장된 파일 크기
        "file_size_mb": round(
            ARTICLE_EMBEDDINGS_PATH.stat().st_size
            / (1024 * 1024),
            2,
        ),
    }



# STEP 7-1. 기사 하나의 NER 목록을 entity set으로 변환
# ner_clusters와 entity_groups를 위치별로 대응 
# null/ 빈 문자열/문자열 앞뒤 공백/ 같은 기사 내부의 중복 개체 제거 / 대소문자 정규화
# 모든 entity type 그대로 사용 
# 예:
# ner_clusters  = ["Trump", "White House", "Denmark"]
# entity_groups = ["PER", "ORG", "LOC"]
#
# 결과:
# {
#     "PER::trump",
#     "ORG::white house",
#     "LOC::denmark",
# }

def _normalize_entity_set(
        raw_entities:Any, 
        raw_entity_groups: Any, 
)-> set[str]:
    """
    ner_clusters 값을 사건 클러스터링에서 사용할
    python set[str] 형태로 변환 

    예:
    ["Trump", "Trump", " withe house", None, ""]
    -> ["Trump", "withe house"]
    """

    if raw_entities is None: return set()
    if raw_entity_groups is None: return set()

    # build_valid_articles()에서 이미 제외했어야 하는 데이터이므로
    # 여기서 다시 길이 불일치 발견되면 오류로 처리
    if len(raw_entities) != len(raw_entity_groups):
        raise ValueError(
            "ner_clusters와 entity_groups의 길이가 다릅니다. "
            f"ner_length={len(raw_entities)}, "
            f"entity_group_length={len(raw_entity_groups)}"
        )


    normalized_entities: set[str] = set()

    # ner_clusters[i]와 entity_groups[i]는 같은 위치끼리 대응 
    for entity, entity_group in zip(
        raw_entities, 
        raw_entity_groups, 
    ):
        if entity is None: continue 
        if entity_group is None: continue 

        # Entity 문자열 정규화 (모두 소문자로)
        # " Trump", "Trump" -> "trump"
        # NFKC : 유니코드 정규화 방식
        entity_text = unicodedata.normalize("NFKC", str(entity),)

        # 연속된 공백도 하나로 
        entity_text = " ".join(entity_text.strip().split())

        # 대소문자 차이 제거
        entity_text = entity_text.lower()
        if entity_text == "": continue 

        # entity type 정규화 (모두 대문자로)
        # per -> PER
        entity_type = unicodedata.normalize("NFKC", str(entity_group))

        entity_type = (entity_type.strip().upper())

        if entity_type == "": continue 

        # type namespace 포함한 최종 entity key
        # 예 : Trump + PER -> PER:trump 
        entity_key = f"{entity_type}::{entity_text}"
        normalized_entities.add(entity_key)
    return normalized_entities


# STEP 7-2. Train 기사 기준 entity IDF 계산
# validation 정보 사용하지 않고 train 기사만으로 IDF 고정 
# 흔한 개체는 낮은 가중치, 드문 개체는 높은 가중치
def _build_train_entity_idf(
        train_entity_sets: list[set[str]],
)-> tuple[dict[str, float], dict[str, int], float]:
    """
    train 기사들의 entity set을 이용해 IDF 계산

    IDF 공식
    ------------
    idf(e) = log((N+1)/df(e)+1)+1

    validation에서 처음 등장한 entity는 df=0으로 보고 unseen_idf 사용
    """

    train_article_count = len(train_entity_sets)

    if train_article_count == 0:
        raise ValueError(
            "Train 사건 생성에 사용할 기사가 없습니다."
        )

    # 한 기사에서 같은 entity가 여러번 나와도 entity_set이므로
    # document frequency엔 1번만 반영됨
    document_frequency: Counter[str] = Counter() # 각 값이 몇 번 나왔는지 자동으로 세어줌 

    for entity_set in train_entity_sets:
        document_frequency.update(entity_set) # Counter에서 이 집합 안의 각 원소를 1씩 세라 

    entity_idf: dict[str, float] = {}

    for entity, df in document_frequency.items():
        entity_idf[entity] = (
            math.log(
                (train_article_count + 1) / (df+1)
            ) + 1.0 
        )

    # validation에서 train에 없던 entity가 등장했을 때 사용
    # df = 0이라면 train에서 1번밖에 안 나온 것보다도 더 희귀한 취급 받음 (값이 더큼)

    unseen_entity_idf = (
        math.log(train_article_count + 1) + 1.0 
    )

    return (
        entity_idf, 
        dict(document_frequency),
        unseen_entity_idf,
    )
    
# STEP 7-3. IDF Weighted Jaccard 계산
# 기사 <-> 기사 
# validation 신규 기사 <-> 사건 
# 사이의 개체 집합 유사도 계산 

def _idf_weighted_jaccard(
        left_entities: set[str],
        right_entities: set[str],
        entity_idf:dict[str, float],
        unseen_entity_idf:float,
)-> float:
    """
    IDF Weighted Jaccard sim을 계산한다.

    분자
    -----------
    두 집합의 교집합 entity IDF 합

    분모
    -----------
    두 집합의 합집합 entity IDF 합
    """

    # 둘 중 하나라도 entity가 하나도 없으면 같은 사건으로 판단하지 않음 
    if not left_entities or not right_entities: 
        return 0.0

    intersection = ( left_entities & right_entities)

    if not intersection: return 0.0

    union = (left_entities | right_entities)

    # 분자 (교집합 IDF 합 )
    # intersection = {"Zlatan", "Severige", "AC Milan"}
    # entity_idf에서 entity:"Zlatan"를 키로 해서 딕셔너리에서 찾아라
    # 없으면 기본으로 쓸 값은 unseen_entity_idf이다. 

    numerator = sum(
        entity_idf.get(
            entity,
            unseen_entity_idf,
        ) for entity in intersection 
    )

    # 분모 (합집합 IDF 합)
    denominator = sum(
        entity_idf.get(
            entity,
            unseen_entity_idf,
        ) for entity in union
    )

    if denominator <= 0.0: return 0.0

    return float(numerator / denominator)


# STEP 7-4. Union-Find 
# 인접 리스트 전체를 저장하지 않고 연결요소를 효율적으로 계산
# size : 몇 개의 기사를 다룰지 (train 전체 기사 수)
# 참고 -------
# 이때, union-find는 article_id(큰숫자)를 다루는 것이 아닌, 
# 0부터 시작하는 순번을 다룸
# 예 : article_id   N24512, N33078, N55291, ...
#    union-find용       0       1       2

# rank : union()이 알아서 갱신함 (처음엔 모든 기사들이 각자 다 대표라서 size만큼 생성)

class _UnionFind:
    def __init__(
            self,
            size:int,
    )-> None:
        self.parent = list(range(size))
        self.rank = [0 for _ in range(size)]

    def find(self, node: int,) -> int:
        # 이 노드의 부모가 자기 자신이 아니면, 부모 따라 더 올라가서 찾아봐라
        # 자기 자신이면 그게 대표
        if self.parent[node] != node: 
            self.parent[node] = self.find(self.parent[node])

        return self.parent[node]

    def union(self, left_node: int, right_node: int,)-> None:
        left_root = self.find(left_node)
        right_root = self.find(right_node)

        if left_root==right_root: return 
        # union (rank에 따라)
        # rank가 더 높은 쪽에 얕은 트리를 붙여야함 
        if(self.rank[left_root] < self.rank[right_root]):
            self.parent[left_root]= right_root 
        elif(self.rank[left_root]>self.rank[right_root]):
            self.parent[right_root]=left_root
        # 두 그룹의 깊이가 같은 경우 
        else:
            self.parent[right_root] = left_root 
            self.rank[left_root]+=1 

# STEP 7-6. 기사 Event 생성

# Train 
# ------------
# 1. train 사용 기사 전체로 IDF계산
# 2. 72시간 이내(하이퍼파라미터) 기사쌍만 비교
# 3. IDF Weighted jaccard >= 0.3이면 edge
# 4. union-find connected component = event 


# 출력
# ------------------
# - article_events.parquet
# - event_master.parquet
# - entity_idf.parquet

def build_article_events(
        entity_similarity_threshold: float=(
            EVENT_ENTITY_SIMILARITY_THRESHOLD
        ),
        time_window_hours: int=(
            EVENT_TIME_WINDOW_HOURS
        ),
        max_entity_df_ratio:float=(EVENT_MAX_ENTITY_DF_RATIO)
)-> dict[str, Any]:
    """
    train event 생성만 수행 
    """

    # STEP 7-6-1. 기본 입력값 검사
    # 잘못된 threshold/시간 설정으로 실행되는 것 방지
    if not (
        0.0
        <= entity_similarity_threshold
        <= 1.0
    ):
        raise ValueError(
            "entity_similarity_threshold는 "
            "0과 1 사이여야 합니다."
        )

    if time_window_hours <= 0:
        raise ValueError(
            "time_window_hours는 "
            "0보다 커야 합니다."
        )

    # 0% 이하 또는 100% 초과와 같은 잘못된 비율 차단
    if not (0.0 < max_entity_df_ratio <= 1.0):
        raise ValueError(
            "max_entity_df_ratio는 "
            "0보다 크고 1 이하여야 합니다."    
        )

    create_output_directories()

    time_window = timedelta(
        hours=time_window_hours # 기사의 시각 차이는 숫자가 아닌 timedelta 객체로 나옴 
    )

    # STEP 7-6-2. 기사 / Train 사용 기사 로드
    # 전체 valid 기사 메타데이터와 이미 확정한 train 사용 기사 ID 연결 
    articles = pl.read_parquet(
        ARTICLES_WITH_CATEGORY_PATH
    ).select(
        [
            # 사건 클러스터링에 필요 
            "article_id",       # 어느 기사인지 식별
            "published_time",   # 시간윈도우 조건 비교 
            "ner_clusters",      # 개체 유사도 비교
            "entity_groups", 
        ]
    )

    train_used_ids_df = pl.read_parquet(
        TRAIN_USED_ARTICLE_IDS_PATH
    ).select("article_id")

    train_used_article_ids = set(
        int(article_id)
        for article_id in (train_used_ids_df.get_column("article_id").to_list())
    )

    valid_article_ids = set(
        int(article_id)
        for article_id in (articles.get_column("article_id").to_list())
    )

    # 0이 되어야 정상
    missing_train_article_ids = sorted(
        train_used_article_ids - valid_article_ids
    )

    if missing_train_article_ids:
        raise ValueError(
            "train_used_article_ids.parquet에 " \
            "유효 기사에 존재하지 않는 article_id 존재." \
            f"예시: {missing_train_article_ids[:10]}"
        )

    # STEP 7-6-3. 파이썬 LOOKUP 생성
    # 이후 그래프/validation 순차처리에서 article_id로 entity set 빠르게 조회 
    article_lookup: dict[
        int, 
        dict[str, Any],
    ] = {}

    for row in articles.iter_rows(named=True):
        article_id = int(row["article_id"])
        article_lookup[article_id]={"article_id":article_id, 
                                    "published_time":row["published_time"],
                                    "entity_set": _normalize_entity_set(row["ner_clusters"],
                                                                        row["entity_groups"])}

    # STEP 7-6-4. Train 기사 정렬
    # train_articles = [
    #     {"article_id": 100, "published_time": "01-01 09:00", "entity_set": {"PER::Zlatan", "LOC::Sverige"}},  # index 0
    #     {"article_id": 200, "published_time": "01-02 08:00", "entity_set": {"PER::Zlatan", "LOC::Milan"}},     # index 1
    #     {"article_id": 300, "published_time": "01-05 10:00", "entity_set": {"PER::Zlatan"}},               # index 2
    #     {"article_id": 400, "published_time": "01-05 11:00", "entity_set": {"PER::Messi"}},                 # index 3
    # ]

    train_articles = [article_lookup[article_id] for article_id in train_used_article_ids]
    train_articles.sort(key = lambda article: (article["published_time"], article["article_id"]))

    if not train_articles: raise ValueError("Train 사건 생성에 사용할 유효 기사 존재하지 않습니다.")
    
    # STEP 7-6-5. Train Entity IDF 계산
    # entity -> PER:trump 이 정규화된 개체로 계산 
    train_article_count = len(train_articles)
    train_entity_sets = [ article["entity_set"] for article in train_articles]

    (entity_idf, entity_documnet_frequency, unseen_entity_idf,) = \
        _build_train_entity_idf(train_entity_sets)

    # 너무 많은 기사에서 반복 등장해서 구체적인 사건 구분력 낮은 entity 찾기
    # 예 : ORG::ekstra bladet
    # 원본 ner_clusters/entity_groups에선 삭제 x
    # event sim 계산에서만 제외
    high_df_entities: set[str] = set()
    for(
        entity, document_frequency,
    ) in entity_documnet_frequency.items():
        entity_df_ratio = (document_frequency / train_article_count)
        if(entity_df_ratio >= max_entity_df_ratio):
            high_df_entities.add(entity)

    # entity_idf.items() 딕셔너리를 (키,값)쌍의 리스트처럼 순회 
    # [("PER::zlatan", 1.405), ("LOC::sverige", 1.405), ("ORG::ac milan", 1.693), ("PER::messi", 2.099)]
    
    # entity idf 저장 행 생성 
    # build_validtion.py에서도 동일한 train high-df entity 기준을 재사용할 수 있도록
    # df 비율과 제외 여부까지 저장 

    entity_idf_rows = []

    for (
        entity,
        idf_value,
    ) in sorted(
        entity_idf.items(),
        key=lambda item: item[0],
    ):

        document_frequency = int(
            entity_documnet_frequency[
                entity
            ]
        )

        document_frequency_ratio = (
            document_frequency
            / train_article_count
        )

        entity_idf_rows.append(
            {
                "entity": entity,

                "document_frequency": (
                    document_frequency
                ),

                "document_frequency_ratio": float(
                    document_frequency_ratio
                ),

                "idf": float(
                    idf_value
                ),

                # Event 유사도에서 제외되는 Entity인지
                "is_high_df": (
                    entity
                    in high_df_entities
                ),
            }
        )

    entity_idf_df = pl.DataFrame(
        entity_idf_rows,
        schema={
            "entity": pl.Utf8,
            "document_frequency": pl.Int64,

            
            "document_frequency_ratio": pl.Float64,

            "idf": pl.Float64,

            
            "is_high_df": pl.Boolean,
        },
    )
    
    # Event 계산용 Entity Set 생성
    # 원본 entity_set은 그대로 유지하고, high-df entity만 제외한 별도 집합 생성
    # 예:
    #
    # 원본
    # {
    #     "ORG::ekstra bladet",
    #     "PER::zlatan ibrahimovic",
    #     "ORG::ac milan",
    # }
    #
    # Event 계산용
    # {
    #     "PER::zlatan ibrahimovic",
    #     "ORG::ac milan",
    # }

    for article in train_articles:
        article["clustering_entity_set"] = article["entity_set"] - high_df_entities


    # entity | document_frequency | idf
    # LOC::sverige    |  3  | 1.405
    # ORG::ac milan   |  2  | 1.693
    # PER::messi      |  1  | 2.099
    # PER::zlatan     |  3  | 1.405
    entity_idf_df.write_parquet(
        ENTITY_IDF_PATH,
        compression="zstd",
    )

    # STEP 7-6-6. Train graph 생성


    union_find = _UnionFind(train_article_count)
    time_candidate_pair_count = 0
    similarity_edge_count = 0

    for left_index in range(train_article_count):
        left_article = train_articles[left_index]
        left_time = left_article["published_time"]
        left_entities=left_article["clustering_entity_set"]

        for right_index in range(left_index+1, train_article_count):
            right_article = train_articles[right_index]


            time_gap = right_article["published_time"] - left_time 

            # 시간순 정렬되어 있기에 72시간보다 커지면 이후 기사 비교 필요 X
            if time_gap > time_window : break 

            time_candidate_pair_count += 1

            right_entities = right_article["clustering_entity_set"]
            similarity = _idf_weighted_jaccard(left_entities, right_entities, entity_idf, unseen_entity_idf)

            if similarity < entity_similarity_threshold: continue 

            union_find.union(left_index, right_index)

            similarity_edge_count += 1

    # STEP 7-6-7. train connected component 추출
    # 해당 그룹에 속한 index들의 리스트 담을 거임 
    # 0: [0,1], 2: [2], 3: [3]
    component_members: dict[int, list[int]] = {}

    for article_index in range(train_article_count):
        root = union_find.find(article_index)

        if root not in component_members:
            component_members[root] = []

        component_members[root].append(article_index)

    # STEP 7-6-8. Train Event ID 결정 (index 번호 묶음에 실제 event_id 부여하는 최종)
    train_components: list[dict[str, Any]]=[]

    # member_indices : [0,1], [2], [3]
    # 각 사건 그룹 내 기사들을 발행시간순으로 정렬 
    # 그룹내 멤버들 기사들 id, 가장 발행시각 빠른 멤버의 발행시각, 그 멤버의 id 
    for member_indices in component_members.values():
        member_articles = [train_articles[index] for index in member_indices]

        member_articles.sort(key=lambda article: (article["published_time"], article["article_id"]))
        train_components.append({"member_articles": member_articles,
                                "event_start_time": member_articles[0]["published_time"],
                                "first_article_id": member_articles[0]["article_id"]})

    # 사건들끼리 event_start_time 순으로 정렬 
    # 가장 발행시각 빠른 
    train_components.sort(
        key=lambda component:( component["event_start_time"], component["first_article_id"])
    )

    # STEP 7-6-9. Train event 상태 생성 
    events: dict[ int, dict[str, Any]] = {}
    article_event_rows: list[dict[str, Any]]= []
    


    #train_components = [
    #    {
    #        "member_articles": [
    #            {"article_id": 100, "published_time": "01-01 09:00", "entity_set": {"PER::zlatan", "LOC::sverige"}},
    #            {"article_id": 200, "published_time": "01-02 08:00", "entity_set": {"PER::zlatan", "ORG::milan"}},
    #        ],
    #        "event_start_time": "01-01 09:00",
    #        "first_article_id": 100,
    #    },
    #    {
    #        "member_articles": [
    #            {"article_id": 300, "published_time": "01-05 10:00", "entity_set": {"PER::messi"}},
    #        ],
    #        "event_start_time": "01-05 10:00",
    #        "first_article_id": 300,
    #    },
    #]


    for event_id, component in enumerate(train_components):
        member_articles = component["member_articles"]
        event_entity_set: set[str] = set()

        for article in member_articles:
            event_entity_set.update(article["clustering_entity_set"])

        event_start_time = component["event_start_time"]
        first_article_id = component["first_article_id"]

        # member_articles는 이미 published_time -> article_id 순으로 정렬되어있기에
        # 마지막 기사의 발행시각 = event의 마지막 기사 발행시각 
        event_last_added_time = member_articles[-1]["published_time"]

        

        # 현재 train event의 상태 저장
        # validation 신규 기사가 들어오면 이 상태 기준으로
        # 살아있는 event인지 판단하고 entity sim 계산 
        events[event_id] = {
            "event_id":event_id,
            "origin_split": "train",
            "event_start_time": event_start_time, 
            "last_added_time": event_last_added_time,
            "first_article_id": first_article_id,

            # event에 속한 모든 기사 entity의 union
            # validation에서 신규 기사와 이 집합 비교
            "entity_set": event_entity_set,
            # 현재는 train 기사만 들어있음
            "train_article_count": len(member_articles),
            "validation_article_count" : 0,
        }

        # 각 train 기사에 해당 event_id 저장
        for article in member_articles:
            article_event_rows.append(
                {
                    "article_id": article["article_id"],
                    "event_id": event_id, 
                    "assignment_split": "train",
                    "published_time": article["published_time"],
                }
            )

    # STEP 7-6-10. Train event 통계 계산
    # 이벤트 크기, 클러스터링 결과 확인 용도
    train_event_count = len(events)
    train_singleton_event_count = sum(1 for event in events.values() if event["train_article_count"]==1)
    train_max_event_article_count = max(event["train_article_count"] for event in events.values())


    # STEP 7-6-12. Train article -> event mapping 저장
    # train에서 실제 사용하는 각 article_id가 어떤 event_id에 속하는지 저장
    article_events_df = (pl.DataFrame(article_event_rows).with_columns([
        pl.col("article_id").cast(pl.Int64),
        pl.col("event_id").cast(pl.Int64),
    ]).sort("article_id"))

    # STEP Article event 기본 정합성 검사 
    # 목적:
    # 기사 하나가 여러 Event에 배정되거나
    # Event ID가 없는 상태를 방지
    if article_events_df.height != train_article_count:
        raise ValueError(
            "article_events의 기사 수와 "
            "Train 사용 기사 수가 다릅니다. "
            f"Train 기사 수={train_article_count}, "
            f"article_events 기사 수={article_events_df.height}"
        )

    if (
        article_events_df
        .get_column(
            "article_id"
        )
        .null_count()
        != 0
    ):
        raise ValueError(
            "article_events에 null article_id가 존재합니다."
        )

    if (
        article_events_df
        .get_column(
            "event_id"
        )
        .null_count()
        != 0
    ):
        raise ValueError(
            "article_events에 null event_id가 존재합니다."
        )

    # article_id 하나당 하나의 Event만 가져야 함
    duplicate_article_count = (
        article_events_df
        .select(
            pl.col(
                "article_id"
            )
            .is_duplicated()
            .sum()
        )
        .item()
    )

    if duplicate_article_count != 0:
        raise ValueError(
            "article_events에 중복 article_id가 존재합니다."
        )

    # 실제 저장
    article_events_df.write_parquet(
        ARTICLE_EVENTS_PATH,
        compression="zstd",
    )

    # STEP 7-6-13. Train event master 생성
    # event 하나당 한 행으로 event의 최종 상태 저장
    # build_validation.py에서는
    # Validation 신규 기사와 기존 Train Event의
    # Entity UNION을 비교해야 한다.
    #
    # 따라서 event_entity_count뿐 아니라
    # 실제 event_entities도 저장한다.

    event_master_rows: list[
        dict[str, Any]
    ] = []

    for event_id in sorted(
        events
    ):

        event = events[
            event_id
        ]

        event_master_rows.append(
            {
                # 실제 Event 고유 ID
                "event_id": (
                    event_id
                ),

                # 현재 Event는 모두 Train에서 생성됨
                "event_origin_split": (
                    "train"
                ),

                # Event에서 가장 먼저 발행된 기사 시각
                "event_start_time": (
                    event[
                        "event_start_time"
                    ]
                ),

                # Event에서 가장 최근에 발행된 기사 시각
                "event_last_added_time": (
                    event[
                        "last_added_time"
                    ]
                ),

                # 현재 Event에 속한 Train 기사 수
                "event_article_count": (
                    event[
                        "train_article_count"
                    ]
                ),

                # 명시적으로 Train 기사 수도 보존
                "train_article_count": (
                    event[
                        "train_article_count"
                    ]
                ),

                # Event Entity UNION의 실제 개체 목록
                #
                # set은 parquet에 바로 안정적으로 저장하기보다는
                # 정렬된 list[str]로 변환
                #
                # 예:
                # [
                #     "LOC::denmark",
                #     "ORG::white house",
                #     "PER::trump",
                # ]
                "event_entities": (
                    sorted(
                        event[
                            "entity_set"
                        ]
                    )
                ),

                # Event의 unique entity 개수
                "event_entity_count": (
                    len(
                        event[
                            "entity_set"
                        ]
                    )
                ),

                # Event에서 가장 먼저 등장한 article_id
                "first_article_id": (
                    event[
                        "first_article_id"
                    ]
                ),
            }
        )
    # STEP 7-6-13-1. Event Master DataFrame 생성
    event_master_df = (
        pl.DataFrame(
            event_master_rows
        )
        .with_columns(
            [
                pl.col(
                    "event_id"
                ).cast(
                    pl.Int64
                ),

                pl.col(
                    "event_article_count"
                ).cast(
                    pl.Int64
                ),

                pl.col(
                    "train_article_count"
                ).cast(
                    pl.Int64
                ),

                pl.col(
                    "event_entity_count"
                ).cast(
                    pl.Int64
                ),

                pl.col(
                    "first_article_id"
                ).cast(
                    pl.Int64
                ),
            ]
        )
        .sort(
            "event_id"
        )
    )
    # STEP 7-6-14. Event 개수 정합성 검사
    # 목적:
    # 실제 events 딕셔너리 Event 수와
    # event_master 행 수가 같은지 확인
    if event_master_df.height != train_event_count:
        raise ValueError(
            "event_master의 Event 수와 "
            "실제 Train Event 수가 다릅니다. "
            f"Train Event 수={train_event_count}, "
            f"event_master 행 수={event_master_df.height}"
        )

    # STEP 7-6-15. Event별 기사 수 정합성 검사
    # 목적:
    # event_master에 기록한 event_article_count와
    # article_events에서 실제 확인되는 기사 수가 동일한지 확인
    actual_event_article_counts = (
        article_events_df
        .group_by(
            "event_id"
        )
        .agg(
            pl.len().alias(
                "_actual_article_count"
            )
        )
    )

    event_count_check = (
        event_master_df
        .select(
            [
                "event_id",
                "event_article_count",
            ]
        )
        .join(
            actual_event_article_counts,
            on="event_id",
            how="left",
        )
    )

    mismatched_event_count = (
        event_count_check
        .filter(
            pl.col(
                "event_article_count"
            )
            !=
            pl.col(
                "_actual_article_count"
            )
        )
        .height
    )

    if mismatched_event_count != 0:
        raise ValueError(
            "event_master의 event_article_count와 "
            "article_events의 실제 기사 수가 "
            "일치하지 않는 Event가 존재합니다."
        )

    # STEP 7-6-16. Event 시간 정합성 검사
    # 목적:
    # Event 시작 시각보다 최근 기사 시각이
    # 더 과거인 잘못된 Event가 없는지 확인
    invalid_event_time_count = (
        event_master_df
        .filter(
            pl.col(
                "event_start_time"
            )
            >
            pl.col(
                "event_last_added_time"
            )
        )
        .height
    )

    if invalid_event_time_count != 0:
        raise ValueError(
            "event_start_time보다 "
            "event_last_added_time이 이전인 Event가 존재합니다."
        )
    # STEP 7-6-17. Event Master 저장
    

    event_master_df.write_parquet(
        EVENT_MASTER_PATH,
        compression="zstd",
    )

    # STEP 7-6-18. Empty Entity 기사 통계
    # 목적:
    # NER 정규화 결과 entity가 하나도 없던 Train 기사 수 확인
    #
    # 이런 기사는 다른 기사와 Jaccard 유사도 0이므로
    # 기본적으로 singleton Event가 됨
    train_empty_entity_article_count = sum(
        1
        for article
        in train_articles
        if not article[
            "entity_set"
        ]
    )
    # High-DF 제거 후 Event 계산에 사용할 Entity가
    # 하나도 남지 않은 기사 수
    # ========================================================

    train_clustering_empty_entity_article_count = sum(
        1
        for article in train_articles
        if not article[
            "clustering_entity_set"
        ]
    )


    # 원래 Entity는 있었지만
    # 모두 High-DF Entity여서 제거된 기사 수
    high_df_only_article_count = sum(
        1
        for article in train_articles
        if (
            article[
                "entity_set"
            ]
            and not article[
                "clustering_entity_set"
            ]
        )
    )
    # STEP 7-6-19. Event Entity가 비어있는 Event 수 확인
    empty_entity_event_count = (
        event_master_df
        .filter(
            pl.col(
                "event_entity_count"
            )
            == 0
        )
        .height
    )

    # STEP 7-6-20. 최종 결과 반환

    return {
        "status": "SUCCESS",

        # 사용한 하이퍼파라미터
        "entity_similarity_threshold": float(
            entity_similarity_threshold
        ),

        "time_window_hours": int(
            time_window_hours
        ),

        # Train 기사 수
        "train_used_article_count": int(
            train_article_count
        ),

        # Train에서 계산된 고유 Entity 수
        "entity_idf_count": int(
            len(
                entity_idf
            )
        ),

        # 72시간 조건을 만족해서
        # 실제 similarity 계산 대상이 된 기사쌍 수
        "time_candidate_pair_count": int(
            time_candidate_pair_count
        ),

        # Weighted Jaccard threshold를 통과해
        # 실제 Graph Edge가 된 기사쌍 수
        "similarity_edge_count": int(
            similarity_edge_count
        ),

        # 최종 Train Event 수
        "train_event_count": int(
            train_event_count
        ),

        # 기사 하나만 들어있는 Event 수
        "train_singleton_event_count": int(
            train_singleton_event_count
        ),

        # 가장 큰 Event의 기사 수
        "train_max_event_article_count": int(
            train_max_event_article_count
        ),

        # Entity가 없는 Train 기사 수
        "train_empty_entity_article_count": int(
            train_empty_entity_article_count
        ),

        # Entity UNION이 비어있는 Event 수
        "empty_entity_event_count": int(
            empty_entity_event_count
        ),

        # 생성된 article -> event mapping 수
        "article_event_row_count": int(
            article_events_df.height
        ),

        # 저장 경로
        "article_events_path": str(
            ARTICLE_EVENTS_PATH
        ),

        "event_master_path": str(
            EVENT_MASTER_PATH
        ),

        "entity_idf_path": str(
            ENTITY_IDF_PATH
        ),

        # High-DF 기준
        "max_entity_df_ratio": float(
            max_entity_df_ratio
        ),

        # Event 계산에서 제외된 High-DF Entity 수
        "high_df_entity_count": int(
            len(
                high_df_entities
            )
        ),


        # High-DF 제거 후 Entity가 하나도 남지 않은 기사 수
        "train_clustering_empty_entity_article_count": int(
            train_clustering_empty_entity_article_count
        ),

        # Entity는 있었지만 전부 High-DF여서
        # Event 계산에서는 빈 Entity가 된 기사 수
        "high_df_only_article_count": int(
            high_df_only_article_count
        ),
    }







