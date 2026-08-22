"""Cross-file referential-integrity checks with Parquet audit outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from preprocess.utils import ensure_required_columns


def _missing_ids(frame: pl.DataFrame, column: str, known_ids: pl.Series, source: str) -> pl.DataFrame:
    values = frame.get_column(column).drop_nulls()
    missing = values.filter(~values.is_in(known_ids)).unique().sort()
    return pl.DataFrame({"source": [source] * missing.len(), "article_id": missing})


def _missing_list_ids(frame: pl.DataFrame, column: str, known_ids: pl.Series, source: str) -> pl.DataFrame:
    values = frame.get_column(column).explode().drop_nulls()
    missing = values.filter(~values.is_in(known_ids)).unique().sort()
    return pl.DataFrame({"source": [source] * missing.len(), "article_id": missing})


def _write_audit(frame: pl.DataFrame, path: Path, article_dtype: pl.DataType) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if frame.is_empty():
        frame = pl.DataFrame(schema={"source": pl.String, "article_id": article_dtype})
    frame.write_parquet(path)


def _user_overlap(history: pl.DataFrame, behaviors: pl.DataFrame) -> dict[str, int]:
    history_users = set(history.get_column("user_id").drop_nulls().to_list())
    behavior_users = set(behaviors.get_column("user_id").drop_nulls().to_list())
    return {
        "history_unique_users": len(history_users),
        "behavior_unique_users": len(behavior_users),
        "intersection_users": len(history_users & behavior_users),
        "history_only_users": len(history_users - behavior_users),
        "behavior_only_users": len(behavior_users - history_users),
    }


def validate_cross_files(
    articles: pl.DataFrame,
    train_history: pl.DataFrame,
    validation_history: pl.DataFrame,
    train_behaviors: pl.DataFrame,
    validation_behaviors: pl.DataFrame,
    audit_dir: Path,
) -> dict[str, Any]:
    """Check article references and train/validation overlap, then write audit Parquets."""

    ensure_required_columns(articles, ["article_id"], "articles")
    for name, frame in (("train_history", train_history), ("validation_history", validation_history)):
        ensure_required_columns(frame, ["user_id", "article_id_fixed"], name)
    for name, frame in (("train_behaviors", train_behaviors), ("validation_behaviors", validation_behaviors)):
        ensure_required_columns(frame, ["user_id", "article_id", "article_ids_clicked"], name)

    known_ids = articles.get_column("article_id").drop_nulls().unique()
    article_dtype = articles.schema["article_id"]
    history_missing = pl.concat(
        [
            _missing_list_ids(train_history, "article_id_fixed", known_ids, "train_history"),
            _missing_list_ids(validation_history, "article_id_fixed", known_ids, "validation_history"),
        ]
    )
    current_missing = pl.concat(
        [
            _missing_ids(train_behaviors, "article_id", known_ids, "train_behaviors"),
            _missing_ids(validation_behaviors, "article_id", known_ids, "validation_behaviors"),
        ]
    )
    clicked_missing = pl.concat(
        [
            _missing_list_ids(train_behaviors, "article_ids_clicked", known_ids, "train_behaviors"),
            _missing_list_ids(validation_behaviors, "article_ids_clicked", known_ids, "validation_behaviors"),
        ]
    )
    _write_audit(history_missing, audit_dir / "missing_history_article_ids.parquet", article_dtype)
    _write_audit(current_missing, audit_dir / "missing_current_article_ids.parquet", article_dtype)
    _write_audit(clicked_missing, audit_dir / "missing_clicked_article_ids.parquet", article_dtype)

    train_current_ids = set(train_behaviors.get_column("article_id").drop_nulls().to_list())
    validation_current_ids = set(validation_behaviors.get_column("article_id").drop_nulls().to_list())
    combined_history = pl.concat([train_history, validation_history])
    combined_behaviors = pl.concat([train_behaviors, validation_behaviors])
    return {
        "known_article_id_count": known_ids.len(),
        "missing_history_article_ids": {"unique_id_count": history_missing.get_column("article_id").n_unique(), "audit_file": "missing_history_article_ids.parquet"},
        "missing_current_article_ids": {"unique_id_count": current_missing.get_column("article_id").n_unique(), "audit_file": "missing_current_article_ids.parquet"},
        "missing_clicked_article_ids": {"unique_id_count": clicked_missing.get_column("article_id").n_unique(), "audit_file": "missing_clicked_article_ids.parquet"},
        "user_overlap": {
            "train": _user_overlap(train_history, train_behaviors),
            "validation": _user_overlap(validation_history, validation_behaviors),
            "combined": _user_overlap(combined_history, combined_behaviors),
        },
        "train_validation_current_article_id_overlap": {
            "train_unique_article_ids": len(train_current_ids),
            "validation_unique_article_ids": len(validation_current_ids),
            "intersection_article_ids": len(train_current_ids & validation_current_ids),
            "train_only_article_ids": len(train_current_ids - validation_current_ids),
            "validation_only_article_ids": len(validation_current_ids - train_current_ids),
        },
    }
