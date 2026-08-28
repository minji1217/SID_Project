from __future__ import annotations

from collections import defaultdict
from pprint import pformat

import polars as pl

from . import config
from .build_train_events import build_train_events
from .build_validation_events import build_validation_events


def _build_normalize_v2_compatible_input() -> None:
    """
    Frozen normalize_v2/article_entities.parquet을
    현재 Entity-Linking Event runner가 읽는 형식으로 변환한다.

    입력:
        article_entities.parquet
        article_id | canonical_entity_key | ...

    출력:
        baseline_article_linked_entities.parquet
        article_id | linked_entities

    여기서 linked_entities라는 컬럼명만 runner 호환용으로 사용하는 것이고,
    실제 값은 GPT/Wikidata 결과가 아니라 normalize_v2 canonical entity key다.
    """
    source_path = (
        config.FROZEN_NORMALIZE_V2_DIR
        / "article_entities.parquet"
    )

    if not source_path.exists():
        raise FileNotFoundError(
            "normalize_v2 article_entities.parquet이 없습니다. "
            f"경로={source_path}"
        )

    articles_path = config.ARTICLES_WITH_CATEGORY_PATH

    if not articles_path.exists():
        raise FileNotFoundError(
            "articles_with_category.parquet이 없습니다. "
            f"경로={articles_path}"
        )

    article_ids = (
        pl.read_parquet(
            articles_path,
            columns=["article_id"],
        )
        .get_column("article_id")
        .cast(pl.Int64)
        .to_list()
    )

    entity_df = pl.read_parquet(
        source_path,
        columns=[
            "article_id",
            "canonical_entity_key",
        ],
    )

    required_columns = {
        "article_id",
        "canonical_entity_key",
    }
    missing = required_columns - set(entity_df.columns)

    if missing:
        raise ValueError(
            "normalize_v2 article_entities.parquet에 필요한 컬럼이 없습니다: "
            + ", ".join(sorted(missing))
        )

    by_article: dict[int, set[str]] = defaultdict(set)

    for article_id, entity_key in entity_df.iter_rows():
        if article_id is None:
            continue
        if entity_key is None:
            continue

        text = str(entity_key).strip()
        if not text:
            continue

        by_article[int(article_id)].add(text)

    rows = [
        {
            "article_id": int(article_id),
            "linked_entities": sorted(
                by_article.get(int(article_id), set())
            ),
        }
        for article_id in article_ids
    ]

    baseline_input_df = pl.DataFrame(
        rows,
        schema={
            "article_id": pl.Int64,
            "linked_entities": pl.List(pl.Utf8),
        },
    )

    baseline_input_df.write_parquet(
        config.ARTICLE_LINKED_ENTITIES_PATH,
        compression="zstd",
    )


def _redirect_config_to_normalize_v2_rerun() -> None:
    """
    기존 Entity-Linking Event 결과와 normalize_v2 frozen 결과를
    절대 덮어쓰지 않도록 출력 경로를 별도 디렉터리로 변경한다.
    """
    rerun_dir = (
        config.EXPERIMENTS_DIR
        / "event_normalize_v2_rerun"
    )
    rerun_dir.mkdir(parents=True, exist_ok=True)

    # 현재 Event 엔진의 입력 포맷과 맞춘 임시/실험 입력
    config.ARTICLE_LINKED_ENTITIES_PATH = (
        rerun_dir
        / "baseline_article_linked_entities.parquet"
    )

    # 모든 Event 출력도 별도 경로로
    config.EVENT_OUTPUT_DIR = rerun_dir
    config.ARTICLE_EVENTS_PATH = (
        rerun_dir / "article_events.parquet"
    )
    config.EVENT_MASTER_PATH = (
        rerun_dir / "event_master.parquet"
    )
    config.ENTITY_IDF_PATH = (
        rerun_dir / "entity_idf.parquet"
    )
    config.VALIDATION_ARTICLE_EVENTS_PATH = (
        rerun_dir / "validation_article_events.parquet"
    )
    config.EVENT_MASTER_WITH_VALIDATION_PATH = (
        rerun_dir / "event_master_with_validation.parquet"
    )
    config.ALL_ARTICLE_EVENTS_PATH = (
        rerun_dir / "all_article_events.parquet"
    )
    config.EVENT_BUILD_SUMMARY_PATH = (
        rerun_dir / "event_build_summary.txt"
    )


def _write_summary(
    train_result: dict,
    validation_result: dict,
) -> None:
    lines = [
        "SID_Project Normalize-v2 Event Rerun Summary",
        "=" * 60,
        f"entity_source={config.ARTICLE_LINKED_ENTITIES_PATH}",
        f"frozen_upstream={config.FROZEN_NORMALIZE_V2_DIR}",
        f"output_dir={config.EVENT_OUTPUT_DIR}",
        f"similarity={config.EVENT_ENTITY_SIMILARITY_THRESHOLD}",
        f"time_window_hours={config.EVENT_TIME_WINDOW_HOURS}",
        f"max_entity_df_ratio={config.EVENT_MAX_ENTITY_DF_RATIO}",
        "entity_mode=normalize_v2",
        "openai_api_used=False",
        "wikidata_api_used=False",
        "",
        "[Train Result]",
        pformat(train_result, sort_dicts=False),
        "",
        "[Validation Result]",
        pformat(validation_result, sort_dicts=False),
        "",
        "[Outputs]",
        f"article_events={config.ARTICLE_EVENTS_PATH}",
        f"event_master={config.EVENT_MASTER_PATH}",
        f"entity_idf={config.ENTITY_IDF_PATH}",
        f"validation_article_events={config.VALIDATION_ARTICLE_EVENTS_PATH}",
        f"event_master_with_validation={config.EVENT_MASTER_WITH_VALIDATION_PATH}",
        f"all_article_events={config.ALL_ARTICLE_EVENTS_PATH}",
    ]

    config.EVENT_BUILD_SUMMARY_PATH.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    # 매우 중요:
    # build_train_events / build_validation_events의 알고리즘은 수정하지 않는다.
    # 입력 Entity만 normalize_v2 canonical key로 바꾼다.
    _redirect_config_to_normalize_v2_rerun()

    print("=" * 80)
    print("SID_Project Normalize-v2 Event RERUN")
    print("=" * 80)
    print(f"Frozen upstream : {config.FROZEN_NORMALIZE_V2_DIR}")
    print(f"New output      : {config.EVENT_OUTPUT_DIR}")
    print(
        "Frozen params   : "
        f"similarity={config.EVENT_ENTITY_SIMILARITY_THRESHOLD}, "
        f"time_window={config.EVENT_TIME_WINDOW_HOURS}h, "
        f"max_entity_df_ratio={config.EVENT_MAX_ENTITY_DF_RATIO}"
    )
    print("Event algorithm  : SAME as Entity-Linking Event runner")
    print("GPT/Wikidata API : NOT USED")
    print()

    print("[STEP 0] Building normalize_v2 Event input adapter...")
    _build_normalize_v2_compatible_input()
    print(
        "[OK] Event input created: "
        f"{config.ARTICLE_LINKED_ENTITIES_PATH}"
    )
    print()

    print("[STEP 1] Building Train Events...")
    train_result = build_train_events()
    print("[OK] Train Event build completed.")
    print(pformat(train_result, sort_dicts=False))
    print()

    print("[STEP 2] Building Validation Events...")
    validation_result = build_validation_events()
    print("[OK] Validation Event build completed.")
    print(pformat(validation_result, sort_dicts=False))
    print()

    _write_summary(
        train_result=train_result,
        validation_result=validation_result,
    )

    print("=" * 80)
    print("COMPLETED")
    print(f"Outputs: {config.EVENT_OUTPUT_DIR}")
    print("Frozen normalize_v2 and Entity-Linking Event outputs were not modified.")
    print("=" * 80)


if __name__ == "__main__":
    main()
