"""Shared helpers for validation reports, schema checks, and logging."""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable

import polars as pl


def configure_logging() -> None:
    """Configure consistent structured logging for the validation run."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def ensure_required_columns(frame: pl.DataFrame, required_columns: Iterable[str], dataset_name: str) -> None:
    """Fail fast when a dataset is structurally incompatible with validation."""

    missing = sorted(set(required_columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{dataset_name} is missing required columns: {missing}")


def null_ratio(null_count: int, row_count: int) -> float:
    """Return a null ratio, with zero for an empty table."""

    return 0.0 if row_count == 0 else null_count / row_count


def stable_deduplicate(values: list[Any] | None) -> list[Any]:
    """Deduplicate a list while preserving first-occurrence order and null values."""

    if values is None:
        return []
    result: list[Any] = []
    seen: set[Any] = set()
    null_seen = False
    for value in values:
        if value is None:
            if not null_seen:
                result.append(value)
                null_seen = True
        elif value not in seen:
            result.append(value)
            seen.add(value)
    return result


def numeric_summary(values: list[int | float], quantiles: Iterable[float]) -> dict[str, float | int | None]:
    """Summarize numeric values without introducing an additional dependency."""

    if not values:
        return {"min": None, "mean": None, "median": None, "max": None, "quantiles": {}}
    sorted_values = sorted(values)
    count = len(sorted_values)

    def percentile(probability: float) -> float:
        position = (count - 1) * probability
        lower = int(position)
        upper = min(lower + 1, count - 1)
        weight = position - lower
        return float(sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight)

    return {
        "min": min(values),
        "mean": sum(values) / count,
        "median": percentile(0.5),
        "max": max(values),
        "quantiles": {f"p{int(probability * 100):02d}": percentile(probability) for probability in quantiles},
    }


def distribution(values: Iterable[int]) -> dict[str, int]:
    """Return a JSON-safe frequency distribution using string keys."""

    result: dict[str, int] = {}
    for value in values:
        key = str(value)
        result[key] = result.get(key, 0) + 1
    return dict(sorted(result.items(), key=lambda item: int(item[0])))


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__} to JSON")


def write_json_report(report: dict[str, Any], path: Path) -> None:
    """Write a UTF-8 JSON report, creating the parent directory when needed."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_handle:
        json.dump(report, file_handle, indent=2, ensure_ascii=False, default=_json_default)
        file_handle.write("\n")
