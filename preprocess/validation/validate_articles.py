"""Validation for the raw article catalogue."""

from __future__ import annotations

from typing import Any

import polars as pl

from preprocess.utils import ensure_required_columns, null_ratio


def validate_articles(frame: pl.DataFrame, required_columns: list[str]) -> dict[str, Any]:
    """Validate article identifiers and required article schema fields."""

    ensure_required_columns(frame, required_columns, "articles")
    article_ids = frame.get_column("article_id")
    non_null_ids = article_ids.drop_nulls()
    duplicate_excess = non_null_ids.len() - non_null_ids.n_unique()
    return {
        "row_count": frame.height,
        "column_count": frame.width,
        "missing_required_columns": [],
        "article_id_null_count": article_ids.null_count(),
        "article_id_null_ratio": null_ratio(article_ids.null_count(), frame.height),
        "article_id_duplicate_excess_rows": duplicate_excess,
        "article_id_unique_non_null_count": non_null_ids.n_unique(),
    }
