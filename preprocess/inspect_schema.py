"""Raw Parquet schema inspection used before validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from preprocess.utils import null_ratio, numeric_summary


def inspect_parquet(path: Path, required_columns: list[str], quantiles: list[float]) -> dict[str, Any]:
    """Inspect one Parquet file without changing its contents."""

    report: dict[str, Any] = {"path": str(path), "file_exists": path.is_file()}
    if not path.is_file():
        report.update({"schema": {}, "row_count": None, "column_count": None, "missing_required_columns": required_columns})
        return report

    frame = pl.read_parquet(path)
    missing = sorted(set(required_columns).difference(frame.columns))
    list_columns = [name for name, dtype in frame.schema.items() if isinstance(dtype, pl.List)]
    list_statistics: dict[str, dict[str, float | int | None]] = {}
    for column in list_columns:
        lengths = [len(value) for value in frame.get_column(column).to_list() if value is not None]
        list_statistics[column] = numeric_summary(lengths, quantiles)

    row_count = frame.height
    report.update(
        {
            "schema": {name: str(dtype) for name, dtype in frame.schema.items()},
            "row_count": row_count,
            "column_count": frame.width,
            "nulls": {
                column: {"count": frame.get_column(column).null_count(), "ratio": null_ratio(frame.get_column(column).null_count(), row_count)}
                for column in frame.columns
            },
            "list_column_lengths": list_statistics,
            "missing_required_columns": missing,
        }
    )
    return report
