"""Read-only validation for positionally aligned user history lists."""

from __future__ import annotations

from typing import Any

import polars as pl

from preprocess.utils import ensure_required_columns, numeric_summary


def _sequence_has_decrease(values: list[Any]) -> bool:
    return any(current < previous for previous, current in zip(values, values[1:]) if previous is not None and current is not None)


def _sequence_has_equal_adjacent(values: list[Any]) -> bool:
    return any(current == previous for previous, current in zip(values, values[1:]) if previous is not None and current is not None)


def validate_history(
    frame: pl.DataFrame,
    required_columns: list[str],
    list_columns: list[str],
    quantiles: list[float],
    unsorted_user_sample_size: int = 20,
) -> dict[str, Any]:
    """Report history alignment, time order, and repeat-visit characteristics."""

    ensure_required_columns(frame, required_columns, "history")
    user_ids = frame.get_column("user_id")
    non_null_users = user_ids.drop_nulls()
    null_list_counts = {column: frame.get_column(column).null_count() for column in list_columns}
    mismatch_rows = 0
    empty_history_rows = 0
    inverted_users: list[Any] = []
    same_timestamp_users: list[Any] = []
    inner_article_null_count = 0
    repeat_visit_count = 0
    consecutive_repeat_count = 0
    history_lengths: list[int] = []

    for row in frame.select(["user_id", *list_columns]).iter_rows(named=True):
        lists = [row[column] for column in list_columns]
        lengths = [len(value) if value is not None else None for value in lists]
        known_lengths = [length for length in lengths if length is not None]
        if len(known_lengths) == len(list_columns) and len(set(known_lengths)) > 1:
            mismatch_rows += 1
        article_ids = row["article_id_fixed"]
        timestamps = row["impression_time_fixed"]
        if article_ids is not None:
            history_lengths.append(len(article_ids))
            if len(article_ids) == 0:
                empty_history_rows += 1
            inner_article_null_count += sum(value is None for value in article_ids)
            usable_ids = [value for value in article_ids if value is not None]
            repeat_visit_count += len(usable_ids) - len(set(usable_ids))
            consecutive_repeat_count += sum(
                current == previous
                for previous, current in zip(article_ids, article_ids[1:])
                if previous is not None and current is not None
            )
        if timestamps is not None:
            if _sequence_has_decrease(timestamps):
                inverted_users.append(row["user_id"])
            if _sequence_has_equal_adjacent(timestamps):
                same_timestamp_users.append(row["user_id"])

    return {
        "row_count": frame.height,
        "user_id_null_count": user_ids.null_count(),
        "user_id_duplicate_excess_rows": non_null_users.len() - non_null_users.n_unique(),
        "null_list_counts": null_list_counts,
        "empty_history_rows": empty_history_rows,
        "list_length_mismatch_rows": mismatch_rows,
        "impression_time_not_ascending_user_count": len(inverted_users),
        "same_impression_timestamp_user_count": len(same_timestamp_users),
        "article_id_fixed_inner_null_count": inner_article_null_count,
        "repeat_visit_count": repeat_visit_count,
        "consecutive_same_article_count": consecutive_repeat_count,
        "history_length": numeric_summary(history_lengths, quantiles),
        "unsorted_user_id_samples": inverted_users[:unsorted_user_sample_size],
    }
