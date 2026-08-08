

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





# STEP 7-5. validation에서 실제 사용하는 기사 ID 수집
# validation history 기사 
# stable dedup 후 단일 클릭 behavior의 current 기사
# stable dedup 후 단일 클릭 behavior의 target 기사
# 를 하나의 unique article 집합으로 수집하기 위해 
# 이전에 우리가 구했던 train_used_article_ids처럼 

def _collect_validation_used_article_ids(valid_article_ids:set[int],)-> dict[str, Any]:
    """
    validation에서 실제 사용하는 article_id 수집
    train의 collect_train_used_article_ids()와 동일한 방법
    behavior는 stable dedup 후 클릭 target이 정확히 1개인 usable behavior만 사용
    """

    validation_history = pl.read_parquet(
        VALIDATION_HISTORY_PATH
    ).select([
        "user_id", "article_id_fixed",
    ])

    validation_behaviors = pl.read_parquet(
        VALIDATION_BEHAVIORS_PATH
    ).select([
        "impression_id","article_id","article_ids_clicked",
    ])

    raw_used_article_ids: set[int]= set()

    # STEP 7-5-1. validation history 기사 수집
    # 사용자의 초기 history에 등장한 모든 기사 id 수집
    for row in validation_history.iter_rows(named=True):
        article_ids=(row["article_id_fixed"])

        if article_ids is None: continue 

        for article_id in article_ids: 
            if article_id is None: continue 

            raw_used_article_ids.add(
                int(article_id)
            )
    usable_behavior_row_count = 0 

    # STEP 7-5-2. Validation usable behavior 기사 수집 
    # clicked list를 stable dedup한 뒤 target이 하나인 행만 사용
    for row in validation_behaviors.iter_rows(named=True):
        clicked_article_ids = row["article_ids_clicked"]

        # stable dedup : 처음 등장한 순서는 유지하며 중복 제거
        deduplicated_clicked_ids: list[int] = []
        seen_clicked_ids: set[int] = set()

        if clicked_article_ids is not None: 
            for article_id in clicked_article_ids:
                if article_id is None:
                    continue 

                article_id = int(article_id)
                if article_id in seen_clicked_ids: continue 
                seen_clicked_ids.add(article_id) # 지금까지 본 것들 
                deduplicated_clicked_ids.append(article_id) # 결과 리스트 

        # baseline : stable dedup 후 클릭 기사가 정확히 1개인 행만
        if len(deduplicated_clicked_ids) != 1: continue 
        usable_behavior_row_count += 1

        # target 기사 
        target_article_id = deduplicated_clicked_ids[0]

        raw_used_article_ids.add(target_article_id)

        # behavior의 current article
        current_article_id = row["article_id"]

        if current_article_id is not None:
            raw_used_article_ids.add(int(current_article_id))

    # STEP 7-5-3. 최종 valid article과 교집합 
    # build_valid_articles()에서 제외된 기사는 사건 생성에서도 제외해야함
    valid_used_article_ids = (raw_used_article_ids & valid_article_ids)

    excluded_article_ids = sorted(raw_used_article_ids - valid_article_ids)

    return {
        "used_article_ids": (
            valid_used_article_ids
        ),
        "raw_used_article_count": (
            len(raw_used_article_ids)
        ),
        "valid_used_article_count": (
            len(valid_used_article_ids)
        ),
        "excluded_article_count": (
            len(excluded_article_ids)
        ),
        "excluded_article_examples": (
            excluded_article_ids[:10]
        ),
        "usable_behavior_row_count": (
            usable_behavior_row_count
        ),
    }
