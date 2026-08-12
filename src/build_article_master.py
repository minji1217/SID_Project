from typing import Any
import numpy as np
import polars as pl
from src import config 

# STEP 8. Train article master 생성

# 이전 스텝에서 각각 따로 생성한 TRAIN 기사 정보를 article_id 기준으로 하나의 최종 테이블로 합치기
# 최종 출력 : 
# article_master.parquet

# 최종 컬럼
# 1. article_id
# 2. embedding_row
# 3. model_category_id
# 4. event_id
# 5. published_time
# 6. model_text

# 중요 :
# 1. train에서 실제 사용하는 기사만 포함된다.
# 2. 새로운 category/event/embedding을 여기서 만들지 않는다.
# 3. 앞 단계에서 이미 확정한 결과를 JOIN만 한다
# 4. JOIN 과정에서 정보가 누락되면 오류 처리한다.

def build_train_article_master() -> dict[str, Any]:
    # STEP 8-1. 출력 디렉토리 생성
    # 목적:
    # article_master.parquet을 저장할 model_inputs 디렉토리 생성
    config.create_output_directories()

    # STEP 8-2. 선행 산출물 존재 여부 확인
    # Article master 만들기 위해 필요한 step 3-7 결과가 모두 만들어졌는지 확인 
    required_paths = {
        "train_used_article_ids": (
            config.TRAIN_USED_ARTICLE_IDS_PATH
        ),
        "articles_with_category": (
            config.ARTICLES_WITH_CATEGORY_PATH
        ),
        "article_embedding_input": (
            config.ARTICLE_EMBEDDING_INPUT_PATH
        ),
        "article_embeddings": (
            config.ARTICLE_EMBEDDINGS_PATH
        ),
        "article_events": (
            config.ARTICLE_EVENTS_PATH
        ),
    }

    for file_name, file_path in required_paths.items():

        if not file_path.exists():
            raise FileNotFoundError(
                f"{file_name} 파일이 없습니다. "
                f"경로: {file_path}"
            )


    # STEP 8-3. Train 사용 기사 ID 읽기
    # ARTICLE MASTER의 기준 집합 확정
    # 여기 있는 기사만 최종 Train article master에 포함
    train_used_articles = pl.read_parquet(config.TRAIN_USED_ARTICLE_IDS_PATH, columns=["article_id"])\
    .select("article_id").sort("article_id")

    train_article_count = train_used_articles.height 
    if train_article_count == 0:
        raise ValueError("Train에서 사용하는 기사가 없습니다.")

    # STEP 8-4.Train 기사 ID 기본 정합성 검사
    # 같은 article_id 여러번 존재 ? 
    train_null_article_id_count = train_used_articles.get_column("article_id").null_count()
    if train_null_article_id_count != 0:
        raise ValueError(
            "train_used_article_ids에 null article_id 존재합니다."
        )

    train_duplicate_article_count=train_used_articles.select(pl.col("article_id").is_duplicated().sum().alias("count")).item()
    if train_duplicate_article_count != 0:
        raise ValueError("train_used_article_ids에 중복 article_id가 존재합니다.")


    # STEP 8-5. 기사 메타데이터 읽기
    # Train 기준으로 이미 확정된 model_cateogry_id와
    # 기사 시각, model_text 가져오기
    # 이때, category_str이 아니라 모델 입력용 model_category_id 사용


    article_metadata = (
        pl.read_parquet(
            config.ARTICLES_WITH_CATEGORY_PATH,
            columns=[
                "article_id",
                "model_category_id",
                "published_time",
                "model_text",
            ],
        )
    )

    # STEP 8-6. Article embedding 위치 정보 읽기
    # article_embeddings.npy에서 해당 기사의 벡터를 찾을 수 있도록
    # embedding_row 연결
    # 예:
    # embedding_row = 35 -> article_embeddings[35]

    article_embedding_input = pl.read_parquet(
        config.ARTICLE_EMBEDDING_INPUT_PATH,
        columns = [
            "article_id", "embedding_row"
        ]
    )

    # STEP 8-7. Train event 정보 읽기
    # STEP 7에서 확정한 event_id를 각 train 기사에 연결
    # validation data는 미포함
    article_events=pl.read_parquet(config.ARTICLE_EVENTS_PATH,
                                   columns=[
                                       "article_id",
                                       "event_id",
                                       "assignment_split"
                                   ]).filter(
                                       pl.col("assignment_split")=="train"
                                   ).select(["article_id", "event_id"])

    # STEP 8-8. JOIN 대상 article_id 중복 검사
    # article_id 기준 JOIN 전에 각 파일에서 article_id가 하나의 행만 가지는지 확인
    join_sources = {
        "articles_with_category": article_metadata,
        "article_embedding_input": article_embedding_input,
        "article_events": article_events,
    }

    for source_name, source_df in join_sources.items():
        duplicate_count = source_df.select(
            pl.col("article_id").is_duplicated().sum().alias("count")
        ).item()

        if duplicate_count != 0:
            raise ValueError(
                f"{source_name}에 "
                f"중복 article_id가 존재합니다. "
                f"중복 행 수={duplicate_count}"
            )

    # STEP 8-9. Train 기사 기준으로 정보 결합
    # train 사용 기사 9738개 기준으로 category/embedding 위치/event 정보 붙이기

    article_master = (
        train_used_articles.join(
            article_metadata,
            on="article_id",
            how="left",
        )
        .join(
            article_embedding_input,
            on="article_id",
            how="left",
        )
        .join(article_events,
        on="article_id",
        how="left",
    ).select([
        "article_id",
        "embedding_row",
        "model_category_id",
        "event_id",
        "published_time",
        "model_text",
    ]).sort("article_id"))

    # STEP 8-10 ~ 8-14 JOIN 후 잘못된 거 있는지 검사 

    # STEP 8-10. JOIN 후 기사 수 검사
    # JOIN 과정에서 행 증가 OR 감소?
    if (
        article_master.height
        != train_article_count
    ):
        raise ValueError(
            "Article Master 기사 수가 "
            "Train 사용 기사 수와 다릅니다. "
            f"Train 기사 수={train_article_count}, "
            f"Article Master 기사 수={article_master.height}"
        )

    # STEP 8-11. 필수 컬럼 NULL검사 
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

        null_count = (
            article_master
            .get_column(
                column_name
            )
            .null_count()
        )

        null_counts[
            column_name
        ] = int(
            null_count
        )

        if null_count != 0:
            raise ValueError(
                f"Article Master의 {column_name}에 "
                f"null 값이 존재합니다. "
                f"null 개수={null_count}"
            )

    # STEP 8-12. model_text 빈 문자열 검사 
    # 실제 텍스트가 비어있는 잘못된 기사 입력 차단 
    empty_model_text_count = (
        article_master
        .filter(
            pl.col(
                "model_text"
            )
            .cast(
                pl.Utf8,
                strict=False,
            )
            .str.strip_chars()
            == ""
        )
        .height
    )

    if empty_model_text_count != 0:
        raise ValueError(
            "Article Master에 빈 model_text가 존재합니다. "
            f"빈 문자열 기사 수={empty_model_text_count}"
        )

    # STEP 8-13. 최종 article_id 중복 검사 
    # 기사 하나가 Article master에서 정확히 한 행만 가지도록 보장
    duplicate_article_count = article_master.select(
        pl.col("article_id").is_duplicated().sum().alias("count")
    ).item()

    if duplicate_article_count != 0:
        raise ValueError(
            "Article Master에 중복 article_id 존재합니다. "
        )

    # STEP 8-14. embedding_row 범위 검사
    # article_master의 embedding_row가 실제 article_embeddings.npy 범위 안에 있는지 
    article_embeddings = np.load(
        config.ARTICLE_EMBEDDINGS_PATH,
        mmap_mode="r",
        allow_pickle=False,
    )

    if article_embeddings.ndim != 2:
        raise ValueError(
            "article_embeddings.npy는 "
            "2차원 배열이어야 합니다."
        )

    embedding_array_row_count = int(
        article_embeddings.shape[0]
    )

    minimum_embedding_row = int(
        article_master
        .get_column(
            "embedding_row"
        )
        .min()
    )

    maximum_embedding_row = int(
        article_master
        .get_column(
            "embedding_row"
        )
        .max()
    )

    if minimum_embedding_row < 0:
        raise ValueError(
            "embedding_row에 음수가 존재합니다."
        )

    if (
        maximum_embedding_row
        >= embedding_array_row_count
    ):
        raise ValueError(
            "embedding_row가 실제 "
            "article_embeddings.npy 범위를 벗어났습니다. "
            f"최대 embedding_row={maximum_embedding_row}, "
            f"임베딩 배열 행 수={embedding_array_row_count}"
        )

    # STEP 8-15. 모델 입력용 데이터 타입 정리
    # 이후 RQ-VAE 입력에서 사용할 ID 타입을 정수형으로 고정 
    article_master = (
        article_master
        .with_columns(
            [
                pl.col(
                    "article_id"
                ).cast(
                    pl.Int64
                ),

                pl.col(
                    "embedding_row"
                ).cast(
                    pl.Int64
                ),

                pl.col(
                    "model_category_id"
                ).cast(
                    pl.Int32
                ),

                pl.col(
                    "event_id"
                ).cast(
                    pl.Int64
                ),
            ]
        )
    )

    # STEP 8-16. Article master 저장
    # 이후 RQ-VAE에서 Train 기사 단위의 기준 테이블로 사용할 결과 저장
    article_master.write_parquet(
        config.ARTICLE_MASTER_PATH,
        compression="zstd",
    )

    # STEP 8-17. 결과 통계 계산
    # 생성 결과를 실행 직후 간단하게 확인할 수 있도록 

    unique_event_count = (
        article_master
        .select(
            "event_id"
        )
        .unique()
        .height
    )

    unique_category_count = (
        article_master
        .select(
            "model_category_id"
        )
        .unique()
        .height
    )

    unknown_category_article_count = (
        article_master
        .filter(
            pl.col(
                "model_category_id"
            )
            == 0
        )
        .height
    )

    # STEP 8-18. 최종 결과 반환

    return {
        "status": "SUCCESS",

        "article_master_path": str(
            config.ARTICLE_MASTER_PATH
        ),

        "train_article_count": int(
            train_article_count
        ),

        "article_master_row_count": int(
            article_master.height
        ),

        "duplicate_article_count": int(
            duplicate_article_count
        ),

        "null_article_id_count": int(
            null_counts[
                "article_id"
            ]
        ),

        "null_embedding_row_count": int(
            null_counts[
                "embedding_row"
            ]
        ),

        "null_model_category_id_count": int(
            null_counts[
                "model_category_id"
            ]
        ),

        "null_event_id_count": int(
            null_counts[
                "event_id"
            ]
        ),

        "null_published_time_count": int(
            null_counts[
                "published_time"
            ]
        ),

        "null_model_text_count": int(
            null_counts[
                "model_text"
            ]
        ),

        "empty_model_text_count": int(
            empty_model_text_count
        ),

        "unique_event_count": int(
            unique_event_count
        ),

        "unique_category_count": int(
            unique_category_count
        ),

        "unknown_category_article_count": int(
            unknown_category_article_count
        ),

        "minimum_embedding_row": int(
            minimum_embedding_row
        ),

        "maximum_embedding_row": int(
            maximum_embedding_row
        ),

        "article_embedding_array_shape": [
            int(
                article_embeddings.shape[0]
            ),
            int(
                article_embeddings.shape[1]
            ),
        ],
    }