"""Configuration loading for the read-only raw-data validation stage."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ValidationSettings:
    """Resolved paths and validation settings loaded from YAML files."""

    repository_root: Path
    paths: dict[str, Path]
    required_columns: dict[str, list[str]]
    history_list_columns: list[str]
    report_quantiles: list[float]
    unsorted_user_sample_size: int

    @property
    def reports_dir(self) -> Path:
        return self.paths["outputs_dir"] / "reports"

    @property
    def audit_dir(self) -> Path:
        return self.paths["outputs_dir"] / "audit"


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file_handle:
        loaded = yaml.safe_load(file_handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a mapping in YAML configuration: {path}")
    return loaded


def load_validation_settings(config_dir: Path | None = None) -> ValidationSettings:
    """Load YAML configuration and resolve all project-relative paths."""

    resolved_config_dir = config_dir or Path(__file__).resolve().parents[1] / "configs"
    resolved_config_dir = resolved_config_dir.resolve()
    repository_root = resolved_config_dir.parent
    path_config = _load_yaml(resolved_config_dir / "paths.yaml")
    validation_config = _load_yaml(resolved_config_dir / "preprocessing.yaml")

    data_dir = repository_root / str(path_config["data_dir"])
    outputs_dir = repository_root / str(path_config["outputs_dir"])
    paths = {
        "articles": data_dir / str(path_config["articles"]),
        "train_behaviors": data_dir / str(path_config["train_behaviors"]),
        "train_history": data_dir / str(path_config["train_history"]),
        "validation_behaviors": data_dir / str(path_config["validation_behaviors"]),
        "validation_history": data_dir / str(path_config["validation_history"]),
        "outputs_dir": outputs_dir,
    }
    return ValidationSettings(
        repository_root=repository_root,
        paths=paths,
        required_columns={key: list(value) for key, value in validation_config["required_columns"].items()},
        history_list_columns=list(validation_config["history_list_columns"]),
        report_quantiles=[float(value) for value in validation_config["report_quantiles"]],
        unsorted_user_sample_size=int(validation_config["unsorted_user_sample_size"]),
    )
