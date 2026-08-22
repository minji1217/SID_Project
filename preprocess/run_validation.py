"""CLI entry point for the STEP 0, read-only EB-NeRD validation run."""

from __future__ import annotations

import logging

import polars as pl

from preprocess.config import load_validation_settings
from preprocess.inspect_schema import inspect_parquet
from preprocess.utils import configure_logging, write_json_report
from preprocess.validation.validate_articles import validate_articles
from preprocess.validation.validate_behaviors import validate_behaviors
from preprocess.validation.validate_cross_files import validate_cross_files
from preprocess.validation.validate_history import validate_history


def main() -> None:
    """Run all STEP 0 inspections and validations without modifying raw data."""

    configure_logging()
    logger = logging.getLogger(__name__)
    settings = load_validation_settings()
    datasets = {
        "articles": (settings.paths["articles"], "articles"),
        "train_behaviors": (settings.paths["train_behaviors"], "behaviors"),
        "train_history": (settings.paths["train_history"], "history"),
        "validation_behaviors": (settings.paths["validation_behaviors"], "behaviors"),
        "validation_history": (settings.paths["validation_history"], "history"),
    }
    schema_report = {
        name: inspect_parquet(path, settings.required_columns[schema_name], settings.report_quantiles)
        for name, (path, schema_name) in datasets.items()
    }
    write_json_report(schema_report, settings.reports_dir / "raw_schema_report.json")
    missing_files = [name for name, report in schema_report.items() if not report["file_exists"]]
    missing_columns = [name for name, report in schema_report.items() if report["missing_required_columns"]]
    if missing_files or missing_columns:
        raise ValueError(f"Raw schema validation failed: missing_files={missing_files}, missing_columns={missing_columns}")

    frames = {name: pl.read_parquet(path) for name, (path, _) in datasets.items()}
    article_report = validate_articles(frames["articles"], settings.required_columns["articles"])
    write_json_report(article_report, settings.reports_dir / "articles_validation_report.json")
    for split in ("train", "validation"):
        history_report = validate_history(
            frames[f"{split}_history"], settings.required_columns["history"], settings.history_list_columns,
            settings.report_quantiles, settings.unsorted_user_sample_size,
        )
        behavior_report = validate_behaviors(frames[f"{split}_behaviors"], settings.required_columns["behaviors"])
        write_json_report(history_report, settings.reports_dir / f"{split}_history_validation_report.json")
        write_json_report(behavior_report, settings.reports_dir / f"{split}_behaviors_validation_report.json")
        logger.info("Validated %s rows: history=%s behaviors=%s", split, frames[f"{split}_history"].height, frames[f"{split}_behaviors"].height)
    cross_report = validate_cross_files(
        frames["articles"], frames["train_history"], frames["validation_history"],
        frames["train_behaviors"], frames["validation_behaviors"], settings.audit_dir,
    )
    write_json_report(cross_report, settings.reports_dir / "cross_file_validation_report.json")
    logger.info("STEP 0 validation completed. Reports: %s", settings.reports_dir)


if __name__ == "__main__":
    main()
