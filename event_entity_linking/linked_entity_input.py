from __future__ import annotations

from pathlib import Path
from typing import Iterable

import polars as pl

from . import config


REQUIRED_COLUMNS = {
    "article_id",
    "linked_entities",
}


def _require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} 파일이 없습니다. 경로={path}")


def load_linked_entity_lookup(
    path: Path | None = None,
    *,
    required_article_ids: Iterable[int] | None = None,
) -> dict[int, set[str]]:
    """Load GPT Entity-Linking Event input as article_id -> set[final entity key].

    Final keys are already resolved by the linking runner:
    - LINKED     -> WD::<QID>
    - AMBIGUOUS  -> original normalize_v2 TYPE::entity key
    - UNLINKED   -> original normalize_v2 TYPE::entity key

    This function performs no extra normalization or linking.
    """
    if path is None:
        path = config.ARTICLE_LINKED_ENTITIES_PATH

    path = Path(path)
    _require_file(path, "article_linked_entities.parquet")

    df = pl.read_parquet(path)

    missing_columns = REQUIRED_COLUMNS - set(df.columns)
    if missing_columns:
        raise ValueError(
            "article_linked_entities.parquet에 필요한 컬럼이 없습니다: "
            + ", ".join(sorted(missing_columns))
        )

    if df.get_column("article_id").null_count() != 0:
        raise ValueError("article_linked_entities.parquet에 null article_id가 존재합니다.")

    duplicate_article_count = (
        df.select(pl.col("article_id").is_duplicated().sum()).item()
    )
    if duplicate_article_count != 0:
        raise ValueError(
            "article_linked_entities.parquet에 중복 article_id가 존재합니다. "
            f"중복 행 수={duplicate_article_count}"
        )

    lookup: dict[int, set[str]] = {}

    for article_id, linked_entities in df.select(
        ["article_id", "linked_entities"]
    ).iter_rows():
        article_id = int(article_id)

        entity_set: set[str] = set()
        for value in linked_entities or []:
            if value is None:
                continue
            text = str(value).strip()
            if not text:
                continue
            entity_set.add(text)

        lookup[article_id] = entity_set

    if required_article_ids is not None:
        required = {int(article_id) for article_id in required_article_ids}
        available = set(lookup)
        missing = sorted(required - available)
        if missing:
            raise ValueError(
                "Entity Linking 결과에 Event 대상 article_id가 누락되어 있습니다. "
                f"missing_count={len(missing)}, examples={missing[:10]}"
            )

    return lookup
