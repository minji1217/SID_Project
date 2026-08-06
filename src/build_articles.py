import polars as pl
from typing import Any 
from src.config import(
    ARTICLES_PATH,
    MODEL_INPUT_DIR,
    TRAIN_HISTORY_PATH,
    TRAIN_BEHAVIORS_PATH,
    create_output_directories,
)

# STEP1. 가공된 기사 데이터 저장 경로
# build_valid_articels()가 만든 유효 기사 데이터를 아래 경로에 Parquet 파일로 저장
ARTICLES_BASE_PATH = (
    MODEL_INPUT_DIR/"articles_base.parquet"
)

# train에서 실제 사용하는 기사 ID 저장 경로
TRAIN_USED_ARTICLE_IDS_PATH = (
    MODEL_INPUT_DIR/"train_used_article_ids.parquet"
)

# train 데이터만 기준으로 생성한 카테고리 매핑 저장 경로 
CATEGORY_MAPPING_PATH = (
    MODEL_INPUT_DIR/"category_mapping.parquet"
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
    -> category_str이 없는 경우엔 이후 카테고리 매핑 단계에서 <UNK> 카테고리로 처리 예정
    -> 어차피 category_str로 tag prediction loss 정답 정수 만들기에 category 없어도 상관 x

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
    2. 사용 가능한 단일 클릭 behavior의 현재 article_id
    3. 사용 가능한 단일 클릭 behavior의 클릭 target article_id

    behavior 처리 규칙
    -----------------
    아래 7개 조건 만족하는 behavior만 사용
    1. impression_id가 null이 아님
    2. impression_id가 중복되지 않음
    3. user_id가 null이 아님
    4. impression_time이 null이 아님
    5. 클릭 리스트가 null 또는 빈 리스트 아님
    6. 클릭 리스트 내부에 null 없음
    7. stable dedup후 고유 클릭 기사가 정확히 1개임

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
    train_history = pl.read_parquet(
        TRAIN_HISTORY_PATH, 
        columns= [
            "article_id_fixed",
        ],
    )

    # train behavior에서 실제 단일 클릭 샘플 고르기 위한 컬럼
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
    # train history 전체에서 등장한 고유 기사 ID 수집 
    history_article_ids = set(
        train_history.select(
            pl.col("article_id_fixed").explode().alias("article_id")
        ).drop_nulls("article_id").get_column("article_id").to_list()
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

    # 실제 단일 클릭 학습 샘플로 사용할 수 있는 behavior 행 수
    usable_behavior_row_count = 0 

    # STEP 3-8. train behavior 한 행씩 처리
    # 학습에 사용할 수 있는 단일 클릭 행만 선택 
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
        has_null_clicked_id = any(
            clicked_article_id is None 
            for clicked_article_id in clicked_ids
        )

        if has_null_clicked_id: continue 

        # STEP 3-8-3. 클릭 목록 stable dedup
        # 클릭 목록의 원순서는 유지하며 동일한 기사 ID가 반복된 경우 중복 제거 
        # 지금까지 한 번이라도 등장한 클릭 ID 저장
        seen_clicked_ids: set[Any] = set()

        # 원순서 유지하며 중복 제거한 결과 저장
        unique_clicked_ids: list[Any] = []

        # 클릭 ID를 원순서로 확인
        for clicked_article_id in clicked_ids:
            # 아직 등장하지 않은 기사 ID만 결과에 추가 
            if clicked_article_id not in seen_clicked_ids:
                seen_clicked_ids.add(
                    clicked_article_id
                )

                unique_clicked_ids.append(clicked_article_id)

        # STEP 3-8-4. 단일 클릭 behavior만 선택
        # 현재 baseline은 하나의 behavior에 대해 target 기사 하나만 정답으로 사용

        # 즉, stable dedup 후 고유 클릭 기사가 정확히 1개인 행만 사용
        if len(unique_clicked_ids) != 1:
            continue 

        # 현재 행이 모든 조건 통과했기에 실제 사용 가능한 behavior에 포함됨
        usable_behavior_row_count += 1

        # 유일한 클릭 기사를 target 기사로 저장
        target_article_ids.add(unique_clicked_ids[0])

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

