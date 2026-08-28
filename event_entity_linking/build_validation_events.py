from __future__ import annotations

from datetime import timedelta
import math
from typing import Any

import polars as pl

from . import config
from .build_train_events import _idf_weighted_jaccard
from .linked_entity_input import load_linked_entity_lookup


def _require_paths() -> None:
    required = {
        "validation_only_article_ids": config.VALIDATION_ONLY_ARTICLE_IDS_PATH,
        "articles_with_category": config.ARTICLES_WITH_CATEGORY_PATH,
        "train_used_article_ids": config.TRAIN_USED_ARTICLE_IDS_PATH,
        "article_linked_entities": config.ARTICLE_LINKED_ENTITIES_PATH,
        "entity_linking_train_article_events": config.ARTICLE_EVENTS_PATH,
        "entity_linking_train_event_master": config.EVENT_MASTER_PATH,
        "entity_linking_train_entity_idf": config.ENTITY_IDF_PATH,
    }
    for label, path in required.items():
        if not path.exists():
            raise FileNotFoundError(f"{label} 파일이 없습니다. 경로={path}")


def _load_validation_only_article_ids() -> set[int]:
    df = (
        pl.read_parquet(config.VALIDATION_ONLY_ARTICLE_IDS_PATH)
        .select("article_id")
        .with_columns(pl.col("article_id").cast(pl.Int64))
    )
    if df.get_column("article_id").null_count() != 0:
        raise ValueError("validation_only_article_ids.parquet에 null article_id가 있습니다.")
    return {int(x) for x in df.get_column("article_id").to_list()}


def build_validation_events(
    entity_similarity_threshold: float = config.EVENT_ENTITY_SIMILARITY_THRESHOLD,
    time_window_hours: int = config.EVENT_TIME_WINDOW_HOURS,
) -> dict[str, Any]:
    """Assign Validation-only articles dynamically using Entity-Linking keys.

    Same policy as latest src/build_validation.py:
    - do NOT recompute Train IDF/high-DF
    - do NOT recluster Train Events
    - compare in published_time order
    - 0 <= article_time - event_last_added_time <= 72h
    - choose highest weighted-Jaccard event above threshold
    - tie is deterministic because event_id is traversed ascending
    - otherwise create a new validation-origin Event
    """
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

    _require_paths()
    config.create_output_directories()

    validation_only_article_ids = _load_validation_only_article_ids()
    time_window = timedelta(hours=time_window_hours)

    entity_idf_df = pl.read_parquet(config.ENTITY_IDF_PATH)
    required_entity_idf_columns = {"entity", "idf", "is_high_df"}
    missing_entity_idf_columns = (
        required_entity_idf_columns - set(entity_idf_df.columns)
    )
    if missing_entity_idf_columns:
        raise ValueError(
            "entity_idf.parquet에 필요한 컬럼이 없습니다: "
            + ", ".join(sorted(missing_entity_idf_columns))
        )

    entity_idf: dict[str, float] = {
        str(entity): float(idf)
        for entity, idf in entity_idf_df.select(
            ["entity", "idf"]
        ).iter_rows()
    }
    high_df_entities: set[str] = {
        str(entity)
        for entity in (
            entity_idf_df.filter(pl.col("is_high_df"))
            .get_column("entity")
            .to_list()
        )
    }

    train_article_count = (
        pl.read_parquet(config.TRAIN_USED_ARTICLE_IDS_PATH)
        .select("article_id")
        .height
    )
    if train_article_count <= 0:
        raise ValueError("Train 사용 기사 수가 0입니다.")

    unseen_entity_idf = math.log(train_article_count + 1) + 1.0

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
    missing_event_columns = required_event_columns - set(
        train_event_master.columns
    )
    if missing_event_columns:
        raise ValueError(
            "event_master.parquet에 필요한 컬럼이 없습니다: "
            + ", ".join(sorted(missing_event_columns))
        )

    duplicate_event_id_count = (
        train_event_master.select(
            pl.col("event_id").is_duplicated().sum()
        ).item()
    )
    if duplicate_event_id_count != 0:
        raise ValueError(
            "event_master.parquet에 중복 event_id가 존재합니다. "
            f"중복 행 수={duplicate_event_id_count}"
        )

    events: dict[int, dict[str, Any]] = {}
    for row in train_event_master.iter_rows(named=True):
        event_id = int(row["event_id"])
        train_count = int(row["train_article_count"])
        event_article_count = int(row["event_article_count"])
        validation_article_count = event_article_count - train_count
        if validation_article_count < 0:
            raise ValueError(
                "event_master의 기사 수 정보가 잘못되었습니다. "
                f"event_id={event_id}, event_article_count={event_article_count}, "
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
            "event_entities": set(row["event_entities"] or []),
            "first_article_id": int(row["first_article_id"]),
        }

    if not events:
        raise ValueError("Train Event가 하나도 존재하지 않습니다.")

    next_event_id = max(events) + 1

    articles = (
        pl.read_parquet(config.ARTICLES_WITH_CATEGORY_PATH)
        .select(["article_id", "published_time"])
        .with_columns(pl.col("article_id").cast(pl.Int64))
    )
    available_article_ids = {
        int(article_id)
        for article_id in articles.get_column("article_id").to_list()
    }

    missing_validation_article_ids = (
        validation_only_article_ids - available_article_ids
    )
    if missing_validation_article_ids:
        raise ValueError(
            "validation_only 기사 중 articles_with_category.parquet에 없는 기사가 있습니다. "
            f"예시={sorted(missing_validation_article_ids)[:10]}"
        )

    linked_entity_lookup = load_linked_entity_lookup(
        required_article_ids=validation_only_article_ids
    )

    validation_articles: list[dict[str, Any]] = []
    for row in articles.iter_rows(named=True):
        article_id = int(row["article_id"])
        if article_id not in validation_only_article_ids:
            continue

        entity_set = set(linked_entity_lookup.get(article_id, set()))
        clustering_entity_set = entity_set - high_df_entities

        validation_articles.append(
            {
                "article_id": article_id,
                "published_time": row["published_time"],
                "entity_set": entity_set,
                "clustering_entity_set": clustering_entity_set,
            }
        )

    validation_articles.sort(
        key=lambda article: (
            article["published_time"],
            article["article_id"],
        )
    )

    validation_article_event_rows: list[dict[str, Any]] = []
    time_candidate_pair_count = 0
    similarity_candidate_pair_count = 0
    matched_existing_event_count = 0
    matched_train_origin_event_count = 0
    matched_validation_origin_event_count = 0
    new_validation_event_count = 0
    validation_empty_entity_article_count = 0
    validation_clustering_empty_entity_article_count = 0

    for article in validation_articles:
        article_id = article["article_id"]
        article_time = article["published_time"]
        raw_article_entities = article["entity_set"]
        article_entities = article["clustering_entity_set"]

        if not raw_article_entities:
            validation_empty_entity_article_count += 1
        if not article_entities:
            validation_clustering_empty_entity_article_count += 1

        best_event_id: int | None = None
        best_similarity = -1.0

        for event_id in sorted(events):
            event = events[event_id]
            time_gap = article_time - event["event_last_added_time"]

            if time_gap < timedelta(0):
                continue
            if time_gap > time_window:
                continue

            time_candidate_pair_count += 1

            similarity = _idf_weighted_jaccard(
                article_entities,
                event["event_entities"],
                entity_idf,
                unseen_entity_idf,
            )

            if similarity < entity_similarity_threshold:
                continue

            similarity_candidate_pair_count += 1

            if best_event_id is None or similarity > best_similarity:
                best_event_id = event_id
                best_similarity = similarity

        if best_event_id is not None:
            event = events[best_event_id]
            matched_existing_event_count += 1

            if event["event_origin_split"] == "train":
                matched_train_origin_event_count += 1
            else:
                matched_validation_origin_event_count += 1

            event["event_entities"].update(article_entities)
            event["event_last_added_time"] = article_time
            event["event_article_count"] += 1
            event["validation_article_count"] += 1
            assigned_event_id = best_event_id
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

        validation_article_event_rows.append(
            {
                "article_id": article_id,
                "event_id": assigned_event_id,
                "assignment_split": "validation",
                "published_time": article_time,
            }
        )

    if validation_article_event_rows:
        validation_article_events_df = (
            pl.DataFrame(validation_article_event_rows)
            .with_columns(
                [
                    pl.col("article_id").cast(pl.Int64),
                    pl.col("event_id").cast(pl.Int64),
                ]
            )
            .sort("article_id")
        )
    else:
        validation_article_events_df = pl.DataFrame(
            schema={
                "article_id": pl.Int64,
                "event_id": pl.Int64,
                "assignment_split": pl.Utf8,
                "published_time": articles.schema["published_time"],
            }
        )

    if validation_article_events_df.height != len(
        validation_only_article_ids
    ):
        raise ValueError(
            "Validation Article Event 수와 validation_only 기사 수가 다릅니다. "
            f"validation_only={len(validation_only_article_ids)}, "
            f"article_events={validation_article_events_df.height}"
        )

    duplicate_article_count = (
        validation_article_events_df.select(
            pl.col("article_id").is_duplicated().sum()
        ).item()
    )
    if duplicate_article_count != 0:
        raise ValueError(
            "Validation Article Event에 중복 article_id가 존재합니다. "
            f"중복 행 수={duplicate_article_count}"
        )

    validation_article_events_df.write_parquet(
        config.VALIDATION_ARTICLE_EVENTS_PATH,
        compression="zstd",
    )

    event_master_rows: list[dict[str, Any]] = []
    for event_id in sorted(events):
        event = events[event_id]
        event_entities = event["event_entities"]
        event_master_rows.append(
            {
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
            }
        )

    event_master_with_validation_df = (
        pl.DataFrame(event_master_rows)
        .with_columns(
            [
                pl.col("event_id").cast(pl.Int64),
                pl.col("event_article_count").cast(pl.Int64),
                pl.col("train_article_count").cast(pl.Int64),
                pl.col("validation_article_count").cast(pl.Int64),
                pl.col("event_entity_count").cast(pl.Int64),
                pl.col("first_article_id").cast(pl.Int64),
            ]
        )
        .sort("event_id")
    )

    invalid_event_article_count = (
        event_master_with_validation_df.filter(
            pl.col("event_article_count")
            != pl.col("train_article_count")
            + pl.col("validation_article_count")
        ).height
    )
    if invalid_event_article_count != 0:
        raise ValueError(
            "Event 전체 기사 수와 Train + Validation 기사 수가 다릅니다. "
            f"문제 Event 수={invalid_event_article_count}"
        )

    invalid_event_time_count = (
        event_master_with_validation_df.filter(
            pl.col("event_start_time") > pl.col("event_last_added_time")
        ).height
    )
    if invalid_event_time_count != 0:
        raise ValueError(
            "event_start_time보다 event_last_added_time이 이전인 Event가 있습니다. "
            f"문제 Event 수={invalid_event_time_count}"
        )

    train_article_events_df = (
        pl.read_parquet(config.ARTICLE_EVENTS_PATH)
        .select(["article_id", "event_id"])
        .with_columns(
            [
                pl.col("article_id").cast(pl.Int64),
                pl.col("event_id").cast(pl.Int64),
            ]
        )
    )
    validation_event_count_df = validation_article_events_df.select(
        ["article_id", "event_id"]
    )
    combined_article_events = pl.concat(
        [train_article_events_df, validation_event_count_df],
        how="vertical",
    )

    actual_event_article_counts = (
        combined_article_events.group_by("event_id")
        .agg(pl.len().alias("_actual_event_article_count"))
    )
    event_count_check = (
        event_master_with_validation_df
        .select(["event_id", "event_article_count"])
        .join(actual_event_article_counts, on="event_id", how="left")
    )

    missing_actual_count = (
        event_count_check.get_column("_actual_event_article_count").null_count()
    )
    if missing_actual_count != 0:
        raise ValueError(
            "Event Master에는 존재하지만 Article Event에는 기사가 없는 Event가 있습니다. "
            f"문제 Event 수={missing_actual_count}"
        )

    mismatched_event_count = (
        event_count_check.filter(
            pl.col("event_article_count")
            != pl.col("_actual_event_article_count")
        ).height
    )
    if mismatched_event_count != 0:
        raise ValueError(
            "Event Master의 기사 수와 실제 Article Event 기사 수가 다릅니다. "
            f"문제 Event 수={mismatched_event_count}"
        )

    event_master_with_validation_df.write_parquet(
        config.EVENT_MASTER_WITH_VALIDATION_PATH,
        compression="zstd",
    )

    all_article_events_df = pl.concat(
        [
            pl.read_parquet(config.ARTICLE_EVENTS_PATH),
            validation_article_events_df,
        ],
        how="vertical_relaxed",
    ).sort("article_id")

    if all_article_events_df.select(
        pl.col("article_id").is_duplicated().sum()
    ).item() != 0:
        raise ValueError("Train + Validation 전체 mapping에 중복 article_id가 있습니다.")

    all_article_events_df.write_parquet(
        config.ALL_ARTICLE_EVENTS_PATH,
        compression="zstd",
    )

    return {
        "status": "SUCCESS",
        "entity_source": str(config.ARTICLE_LINKED_ENTITIES_PATH),
        "validation_only_article_count": int(len(validation_only_article_ids)),
        "validation_article_event_count": int(
            validation_article_events_df.height
        ),
        "matched_existing_event_count": int(matched_existing_event_count),
        "matched_train_origin_event_count": int(
            matched_train_origin_event_count
        ),
        "matched_validation_origin_event_count": int(
            matched_validation_origin_event_count
        ),
        "new_validation_event_count": int(new_validation_event_count),
        "final_event_count": int(event_master_with_validation_df.height),
        "time_candidate_pair_count": int(time_candidate_pair_count),
        "similarity_candidate_pair_count": int(
            similarity_candidate_pair_count
        ),
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
        "all_article_events_path": str(config.ALL_ARTICLE_EVENTS_PATH),
    }
