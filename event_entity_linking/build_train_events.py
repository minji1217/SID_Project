from __future__ import annotations

from collections import Counter
from datetime import timedelta
import math
from typing import Any

import polars as pl

from . import config
from .linked_entity_input import load_linked_entity_lookup


def _build_train_entity_idf(
    train_entity_sets: list[set[str]],
) -> tuple[dict[str, float], dict[str, int], float]:
    """Same IDF policy as the latest src/build_train.py.

    idf(e) = log((N + 1) / (df(e) + 1)) + 1
    unseen_entity_idf = log(N + 1) + 1
    """
    train_article_count = len(train_entity_sets)
    if train_article_count == 0:
        raise ValueError("Train 사건 생성에 사용할 기사가 없습니다.")

    document_frequency: Counter[str] = Counter()
    for entity_set in train_entity_sets:
        document_frequency.update(entity_set)

    entity_idf: dict[str, float] = {}
    for entity, df in document_frequency.items():
        entity_idf[entity] = (
            math.log((train_article_count + 1) / (df + 1)) + 1.0
        )

    unseen_entity_idf = math.log(train_article_count + 1) + 1.0

    return entity_idf, dict(document_frequency), unseen_entity_idf


def _idf_weighted_jaccard(
    left_entities: set[str],
    right_entities: set[str],
    entity_idf: dict[str, float],
    unseen_entity_idf: float,
) -> float:
    """Same IDF Weighted Jaccard policy as the latest src/build_train.py."""
    if not left_entities or not right_entities:
        return 0.0

    intersection = left_entities & right_entities
    if not intersection:
        return 0.0

    union = left_entities | right_entities

    numerator = sum(
        entity_idf.get(entity, unseen_entity_idf)
        for entity in intersection
    )
    denominator = sum(
        entity_idf.get(entity, unseen_entity_idf)
        for entity in union
    )

    if denominator <= 0.0:
        return 0.0

    return float(numerator / denominator)


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0 for _ in range(size)]

    def find(self, node: int) -> int:
        if self.parent[node] != node:
            self.parent[node] = self.find(self.parent[node])
        return self.parent[node]

    def union(self, left_node: int, right_node: int) -> None:
        left_root = self.find(left_node)
        right_root = self.find(right_node)

        if left_root == right_root:
            return

        if self.rank[left_root] < self.rank[right_root]:
            self.parent[left_root] = right_root
        elif self.rank[left_root] > self.rank[right_root]:
            self.parent[right_root] = left_root
        else:
            self.parent[right_root] = left_root
            self.rank[left_root] += 1


def _require_paths() -> None:
    required = {
        "articles_with_category": config.ARTICLES_WITH_CATEGORY_PATH,
        "train_used_article_ids": config.TRAIN_USED_ARTICLE_IDS_PATH,
        "article_linked_entities": config.ARTICLE_LINKED_ENTITIES_PATH,
    }
    for label, path in required.items():
        if not path.exists():
            raise FileNotFoundError(f"{label} 파일이 없습니다. 경로={path}")


def build_train_events(
    entity_similarity_threshold: float = config.EVENT_ENTITY_SIMILARITY_THRESHOLD,
    time_window_hours: int = config.EVENT_TIME_WINDOW_HOURS,
    max_entity_df_ratio: float = config.EVENT_MAX_ENTITY_DF_RATIO,
) -> dict[str, Any]:
    """Build Train Events from final Entity-Linking keys only.

    Event algorithm is intentionally unchanged:
    Train DF -> high-DF removal -> IDF -> 72h candidate window ->
    IDF Weighted Jaccard -> threshold -> Union-Find connected components.
    """
    if not 0.0 <= entity_similarity_threshold <= 1.0:
        raise ValueError("entity_similarity_threshold는 0과 1 사이여야 합니다.")
    if time_window_hours <= 0:
        raise ValueError("time_window_hours는 0보다 커야 합니다.")
    if not 0.0 < max_entity_df_ratio <= 1.0:
        raise ValueError("max_entity_df_ratio는 0보다 크고 1 이하여야 합니다.")

    _require_paths()
    config.create_output_directories()

    time_window = timedelta(hours=time_window_hours)

    articles = (
        pl.read_parquet(config.ARTICLES_WITH_CATEGORY_PATH)
        .select(["article_id", "published_time"])
        .with_columns(pl.col("article_id").cast(pl.Int64))
    )

    train_used_ids_df = (
        pl.read_parquet(config.TRAIN_USED_ARTICLE_IDS_PATH)
        .select("article_id")
        .with_columns(pl.col("article_id").cast(pl.Int64))
    )
    train_used_article_ids = {
        int(article_id)
        for article_id in train_used_ids_df.get_column("article_id").to_list()
    }

    valid_article_ids = {
        int(article_id)
        for article_id in articles.get_column("article_id").to_list()
    }
    missing_train_article_ids = sorted(
        train_used_article_ids - valid_article_ids
    )
    if missing_train_article_ids:
        raise ValueError(
            "train_used_article_ids.parquet에 유효 기사에 존재하지 않는 article_id가 있습니다. "
            f"예시={missing_train_article_ids[:10]}"
        )

    linked_entity_lookup = load_linked_entity_lookup(
        required_article_ids=train_used_article_ids
    )

    article_lookup: dict[int, dict[str, Any]] = {}
    for row in articles.iter_rows(named=True):
        article_id = int(row["article_id"])
        article_lookup[article_id] = {
            "article_id": article_id,
            "published_time": row["published_time"],
            "entity_set": set(linked_entity_lookup.get(article_id, set())),
        }

    train_articles = [
        article_lookup[article_id]
        for article_id in train_used_article_ids
    ]
    train_articles.sort(
        key=lambda article: (
            article["published_time"],
            article["article_id"],
        )
    )

    if not train_articles:
        raise ValueError("Train 사건 생성에 사용할 유효 기사가 존재하지 않습니다.")

    train_article_count = len(train_articles)
    train_entity_sets = [article["entity_set"] for article in train_articles]

    (
        entity_idf,
        entity_document_frequency,
        unseen_entity_idf,
    ) = _build_train_entity_idf(train_entity_sets)

    high_df_entities: set[str] = set()
    for entity, document_frequency in entity_document_frequency.items():
        entity_df_ratio = document_frequency / train_article_count
        if entity_df_ratio >= max_entity_df_ratio:
            high_df_entities.add(entity)

    entity_idf_rows: list[dict[str, Any]] = []
    for entity, idf_value in sorted(entity_idf.items()):
        document_frequency = int(entity_document_frequency[entity])
        document_frequency_ratio = document_frequency / train_article_count
        entity_idf_rows.append(
            {
                "entity": entity,
                "document_frequency": document_frequency,
                "document_frequency_ratio": float(document_frequency_ratio),
                "idf": float(idf_value),
                "is_high_df": entity in high_df_entities,
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
    entity_idf_df.write_parquet(
        config.ENTITY_IDF_PATH,
        compression="zstd",
    )

    for article in train_articles:
        article["clustering_entity_set"] = (
            article["entity_set"] - high_df_entities
        )

    union_find = _UnionFind(train_article_count)
    time_candidate_pair_count = 0
    similarity_edge_count = 0

    for left_index in range(train_article_count):
        left_article = train_articles[left_index]
        left_time = left_article["published_time"]
        left_entities = left_article["clustering_entity_set"]

        for right_index in range(left_index + 1, train_article_count):
            right_article = train_articles[right_index]
            time_gap = right_article["published_time"] - left_time

            if time_gap > time_window:
                break

            time_candidate_pair_count += 1

            right_entities = right_article["clustering_entity_set"]
            similarity = _idf_weighted_jaccard(
                left_entities,
                right_entities,
                entity_idf,
                unseen_entity_idf,
            )

            if similarity < entity_similarity_threshold:
                continue

            union_find.union(left_index, right_index)
            similarity_edge_count += 1

    component_members: dict[int, list[int]] = {}
    for article_index in range(train_article_count):
        root = union_find.find(article_index)
        component_members.setdefault(root, []).append(article_index)

    train_components: list[dict[str, Any]] = []
    for member_indices in component_members.values():
        member_articles = [
            train_articles[index] for index in member_indices
        ]
        member_articles.sort(
            key=lambda article: (
                article["published_time"],
                article["article_id"],
            )
        )
        train_components.append(
            {
                "member_articles": member_articles,
                "event_start_time": member_articles[0]["published_time"],
                "first_article_id": member_articles[0]["article_id"],
            }
        )

    train_components.sort(
        key=lambda component: (
            component["event_start_time"],
            component["first_article_id"],
        )
    )

    events: dict[int, dict[str, Any]] = {}
    article_event_rows: list[dict[str, Any]] = []

    for event_id, component in enumerate(train_components):
        member_articles = component["member_articles"]
        event_entity_set: set[str] = set()
        for article in member_articles:
            event_entity_set.update(article["clustering_entity_set"])

        event_start_time = component["event_start_time"]
        first_article_id = component["first_article_id"]
        event_last_added_time = member_articles[-1]["published_time"]

        events[event_id] = {
            "event_id": event_id,
            "event_start_time": event_start_time,
            "last_added_time": event_last_added_time,
            "first_article_id": first_article_id,
            "entity_set": event_entity_set,
            "train_article_count": len(member_articles),
        }

        for article in member_articles:
            article_event_rows.append(
                {
                    "article_id": article["article_id"],
                    "event_id": event_id,
                    "assignment_split": "train",
                    "published_time": article["published_time"],
                }
            )

    train_event_count = len(events)
    train_singleton_event_count = sum(
        1
        for event in events.values()
        if event["train_article_count"] == 1
    )
    train_max_event_article_count = max(
        event["train_article_count"] for event in events.values()
    )

    article_events_df = (
        pl.DataFrame(article_event_rows)
        .with_columns(
            [
                pl.col("article_id").cast(pl.Int64),
                pl.col("event_id").cast(pl.Int64),
            ]
        )
        .sort("article_id")
    )

    if article_events_df.height != train_article_count:
        raise ValueError(
            "article_events의 기사 수와 Train 사용 기사 수가 다릅니다. "
            f"Train={train_article_count}, article_events={article_events_df.height}"
        )
    if article_events_df.get_column("article_id").null_count() != 0:
        raise ValueError("article_events에 null article_id가 존재합니다.")
    if article_events_df.get_column("event_id").null_count() != 0:
        raise ValueError("article_events에 null event_id가 존재합니다.")

    duplicate_article_count = (
        article_events_df.select(pl.col("article_id").is_duplicated().sum()).item()
    )
    if duplicate_article_count != 0:
        raise ValueError("article_events에 중복 article_id가 존재합니다.")

    article_events_df.write_parquet(
        config.ARTICLE_EVENTS_PATH,
        compression="zstd",
    )

    event_master_rows: list[dict[str, Any]] = []
    for event_id in sorted(events):
        event = events[event_id]
        event_master_rows.append(
            {
                "event_id": event_id,
                "event_origin_split": "train",
                "event_start_time": event["event_start_time"],
                "event_last_added_time": event["last_added_time"],
                "event_article_count": event["train_article_count"],
                "train_article_count": event["train_article_count"],
                "event_entities": sorted(event["entity_set"]),
                "event_entity_count": len(event["entity_set"]),
                "first_article_id": event["first_article_id"],
            }
        )

    event_master_df = (
        pl.DataFrame(event_master_rows)
        .with_columns(
            [
                pl.col("event_id").cast(pl.Int64),
                pl.col("event_article_count").cast(pl.Int64),
                pl.col("train_article_count").cast(pl.Int64),
                pl.col("event_entity_count").cast(pl.Int64),
                pl.col("first_article_id").cast(pl.Int64),
            ]
        )
        .sort("event_id")
    )

    if event_master_df.height != train_event_count:
        raise ValueError(
            "event_master의 Event 수와 실제 Train Event 수가 다릅니다. "
            f"Train Event={train_event_count}, event_master={event_master_df.height}"
        )

    actual_event_article_counts = (
        article_events_df.group_by("event_id")
        .agg(pl.len().alias("_actual_article_count"))
    )
    event_count_check = (
        event_master_df.select(["event_id", "event_article_count"])
        .join(actual_event_article_counts, on="event_id", how="left")
    )
    mismatched_event_count = (
        event_count_check.filter(
            pl.col("event_article_count")
            != pl.col("_actual_article_count")
        ).height
    )
    if mismatched_event_count != 0:
        raise ValueError(
            "event_master의 event_article_count와 article_events의 실제 기사 수가 다릅니다."
        )

    invalid_event_time_count = (
        event_master_df.filter(
            pl.col("event_start_time") > pl.col("event_last_added_time")
        ).height
    )
    if invalid_event_time_count != 0:
        raise ValueError(
            "event_start_time보다 event_last_added_time이 이전인 Event가 존재합니다."
        )

    event_master_df.write_parquet(
        config.EVENT_MASTER_PATH,
        compression="zstd",
    )

    train_empty_entity_article_count = sum(
        1 for article in train_articles if not article["entity_set"]
    )
    train_clustering_empty_entity_article_count = sum(
        1
        for article in train_articles
        if not article["clustering_entity_set"]
    )
    high_df_only_article_count = sum(
        1
        for article in train_articles
        if article["entity_set"]
        and not article["clustering_entity_set"]
    )
    empty_entity_event_count = (
        event_master_df.filter(pl.col("event_entity_count") == 0).height
    )

    return {
        "status": "SUCCESS",
        "entity_source": str(config.ARTICLE_LINKED_ENTITIES_PATH),
        "entity_similarity_threshold": float(entity_similarity_threshold),
        "time_window_hours": int(time_window_hours),
        "max_entity_df_ratio": float(max_entity_df_ratio),
        "train_used_article_count": int(train_article_count),
        "entity_idf_count": int(len(entity_idf)),
        "high_df_entity_count": int(len(high_df_entities)),
        "time_candidate_pair_count": int(time_candidate_pair_count),
        "similarity_edge_count": int(similarity_edge_count),
        "train_event_count": int(train_event_count),
        "train_singleton_event_count": int(train_singleton_event_count),
        "train_max_event_article_count": int(train_max_event_article_count),
        "train_empty_entity_article_count": int(train_empty_entity_article_count),
        "train_clustering_empty_entity_article_count": int(
            train_clustering_empty_entity_article_count
        ),
        "high_df_only_article_count": int(high_df_only_article_count),
        "empty_entity_event_count": int(empty_entity_event_count),
        "article_event_row_count": int(article_events_df.height),
        "article_events_path": str(config.ARTICLE_EVENTS_PATH),
        "event_master_path": str(config.EVENT_MASTER_PATH),
        "entity_idf_path": str(config.ENTITY_IDF_PATH),
    }
