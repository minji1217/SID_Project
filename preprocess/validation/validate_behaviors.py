"""Read-only validation for behavior impressions and clicked article lists."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import polars as pl

from preprocess.utils import distribution, ensure_required_columns, null_ratio, stable_deduplicate


def validate_behaviors(frame: pl.DataFrame, required_columns: list[str]) -> dict[str, Any]:
    """Report identifiers, click-list quality, ordering, and in-view list statistics."""

    ensure_required_columns(frame, required_columns, "behaviors")
    impression_ids = frame.get_column("impression_id")
    users = frame.get_column("user_id")
    article_ids = frame.get_column("article_id")
    original_lengths: list[int] = []
    deduplicated_lengths: list[int] = []
    clicked_duplicate_rows = 0
    single_click_rows = 0
    multi_click_rows = 0
    empty_click_rows = 0
    inview_lengths: list[int] = []
    user_times: dict[Any, list[Any]] = defaultdict(list)
    same_timestamp_rows = 0

    for row in frame.select(["user_id", "impression_time", "article_ids_clicked", "article_ids_inview"]).iter_rows(named=True):
        clicked = row["article_ids_clicked"]
        original_length = len(clicked) if clicked is not None else 0
        stable_clicked = stable_deduplicate(clicked)
        original_lengths.append(original_length)
        deduplicated_lengths.append(len(stable_clicked))
        if len(stable_clicked) < original_length:
            clicked_duplicate_rows += 1
        if len(stable_clicked) == 1:
            single_click_rows += 1
        elif len(stable_clicked) > 1:
            multi_click_rows += 1
        else:
            empty_click_rows += 1
        inview = row["article_ids_inview"]
        if inview is not None:
            inview_lengths.append(len(inview))
        if row["user_id"] is not None and row["impression_time"] is not None:
            user_times[row["user_id"]].append(row["impression_time"])

    unsorted_users = [user_id for user_id, times in user_times.items() if any(current < previous for previous, current in zip(times, times[1:]))]
    for times in user_times.values():
        same_timestamp_rows += sum(current == previous for previous, current in zip(times, times[1:]))
    non_null_impressions = impression_ids.drop_nulls()
    return {
        "row_count": frame.height,
        "impression_id_null_count": impression_ids.null_count(),
        "impression_id_duplicate_excess_rows": non_null_impressions.len() - non_null_impressions.n_unique(),
        "user_id_null_count": users.null_count(),
        "impression_time_null_count": frame.get_column("impression_time").null_count(),
        "article_id_null_count": article_ids.null_count(),
        "article_id_null_ratio": null_ratio(article_ids.null_count(), frame.height),
        "clicked_original_length_distribution": distribution(original_lengths),
        "clicked_duplicate_rows": clicked_duplicate_rows,
        "clicked_stable_deduplicated_length_distribution": distribution(deduplicated_lengths),
        "single_unique_click_rows": single_click_rows,
        "multi_unique_click_rows": multi_click_rows,
        "empty_unique_click_rows": empty_click_rows,
        "user_impression_time_not_ascending_user_count": len(unsorted_users),
        "same_impression_timestamp_rows": same_timestamp_rows,
        "article_ids_inview": {
            "null_list_count": frame.get_column("article_ids_inview").null_count(),
            "length_distribution": distribution(inview_lengths),
            "model_input_note": "Inspected only; article_ids_inview is not used as model input in STEP 0.",
        },
    }
