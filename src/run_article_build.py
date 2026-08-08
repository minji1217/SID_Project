from pprint import pprint

from src import config

from src.build_articles import (
    build_valid_articles,
    collect_train_used_article_ids,
    build_category_mapping,
    apply_category_mapping_to_articles,
    build_article_embedding_input,
    generate_article_embeddings,
    build_article_events,
)

from src.build_article_master import (
    build_article_master,
)


# ============================================================
# STEP 9. Article Build 전체 실행
#
# 목적:
# STEP 2 ~ STEP 8의 Article 관련 전처리를
# 정해진 순서대로 한 번에 실행한다.
#
# 중요:
# - 이 파일 자체에서는 데이터 처리 로직을 구현하지 않는다.
# - 각 src 파일에 존재하는 함수를 순서대로 호출하는 역할만 한다.
# - Validation behaviors/history 전처리는 여기서 수행하지 않는다.
#
# 실행 순서:
#
# STEP 2  build_valid_articles()
# STEP 3  collect_train_used_article_ids()
# STEP 4  build_category_mapping()
# STEP 5  apply_category_mapping_to_articles()
# STEP 6  build_article_embedding_input()
#         generate_article_embeddings()
# STEP 7  build_article_events()
# STEP 8  build_article_master()
#
# 현재는 파일만 작성해두고 실행하지 않는다.
# ============================================================


def _print_step_result(
    step_name: str,
    result: dict,
) -> None:
    """
    각 STEP의 실행 결과를
    터미널에서 보기 쉽게 출력한다.
    """

    print()

    print(
        "=" * 70
    )

    print(
        step_name
    )

    print(
        "=" * 70
    )

    pprint(
        result
    )


def main() -> None:

    # ========================================================
    # STEP 9-1. 출력 디렉토리 생성
    #
    # 목적:
    # 전체 Article Build 과정에서 사용할
    # output 디렉토리를 미리 생성한다.
    # ========================================================

    config.create_output_directories()


    # ========================================================
    # STEP 9-2. 유효 기사 데이터 생성
    #
    # 목적:
    # 원본 articles.parquet에서
    # 모델 입력으로 사용할 수 없는 기사를 제외하고
    # articles_base.parquet을 생성한다.
    #
    # 중요:
    # 여기서 valid는 Validation split이 아니다.
    #
    # valid article
    # = 전처리 기준을 통과한 유효 기사
    # ========================================================

    result = (
        build_valid_articles()
    )


    _print_step_result(
        "STEP 2 - 유효 기사 데이터 생성",
        result,
    )


    # ========================================================
    # STEP 9-3. Train 사용 기사 ID 수집
    #
    # 목적:
    # Train history와 usable behaviors에서
    # 실제 Train 과정에서 참조되는 기사 ID를 수집한다.
    # ========================================================

    result = (
        collect_train_used_article_ids()
    )


    _print_step_result(
        "STEP 3 - Train 사용 기사 ID 수집",
        result,
    )


    # ========================================================
    # STEP 9-4. Train 기준 Category Mapping 생성
    #
    # 목적:
    # Train에서 실제 사용하는 category_str만 이용해서
    # model_category_id mapping을 생성한다.
    #
    # Validation 정보로 mapping을 fit하지 않는다.
    # ========================================================

    result = (
        build_category_mapping()
    )


    _print_step_result(
        "STEP 4 - Train Category Mapping 생성",
        result,
    )


    # ========================================================
    # STEP 9-5. 전체 유효 기사에 Category Mapping 적용
    #
    # 목적:
    # STEP 4에서 Train으로 확정한 mapping을
    # 전체 유효 기사에 적용한다.
    #
    # Train에서 보지 못한 category는
    # <UNK> = 0으로 변환된다.
    #
    # mapping 자체를 다시 학습하는 단계가 아니다.
    # ========================================================

    result = (
        apply_category_mapping_to_articles()
    )


    _print_step_result(
        "STEP 5 - Category Mapping 적용",
        result,
    )


    # ========================================================
    # STEP 9-6. Article Embedding 입력 생성
    #
    # 목적:
    # 전체 유효 기사를 article_id 기준으로 정렬하고
    # 각 기사에 embedding_row를 부여한다.
    # ========================================================

    result = (
        build_article_embedding_input()
    )


    _print_step_result(
        "STEP 6-1 - Article Embedding 입력 생성",
        result,
    )


    # ========================================================
    # STEP 9-7. Article Embedding 생성
    #
    # 목적:
    # multilingual-e5-base를 사용하여
    # 전체 유효 기사의 768차원 embedding을 생성한다.
    #
    # 주의:
    # 이 STEP은 E5 추론을 다시 수행하기 때문에
    # 전체 pipeline 중 시간이 가장 오래 걸릴 수 있다.
    # ========================================================

    result = (
        generate_article_embeddings()
    )


    _print_step_result(
        "STEP 6-2 - Article Embedding 생성",
        result,
    )


    # ========================================================
    # STEP 9-8. Train Event 생성
    #
    # 목적:
    # Train 사용 기사만 가지고 실제 Event를 생성한다.
    #
    # 현재 config 기준:
    #
    # - Entity similarity threshold = 0.3
    # - Time window = 72시간
    # - High-DF Entity ratio = 0.01
    #
    # Validation Event 처리는 여기서 수행하지 않는다.
    # ========================================================

    result = (
        build_article_events()
    )


    _print_step_result(
        "STEP 7 - Train Article Event 생성",
        result,
    )


    # ========================================================
    # STEP 9-9. Train Article Master 생성
    #
    # 목적:
    # 지금까지 생성한
    #
    # - article_id
    # - embedding_row
    # - model_category_id
    # - event_id
    # - published_time
    # - model_text
    #
    # 정보를 Train 기사 단위로 하나의 파일에 결합한다.
    # ========================================================

    result = (
        build_article_master()
    )


    _print_step_result(
        "STEP 8 - Train Article Master 생성",
        result,
    )


  
    # STEP 9-10. 전체 Article Build 완료 출력
    

    print()

    print(
        "=" * 70
    )

    print(
        "Article Build Pipeline 완료"
    )

    print(
        "=" * 70
    )

    print(
        "최종 Article Master:"
    )

    print(
        config.ARTICLE_MASTER_PATH
    )


if __name__ == "__main__":
    main()