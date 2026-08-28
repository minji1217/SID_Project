from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent

FROZEN_V2_DIR = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "experiments"
    / "normalize_v2"
    / "model_inputs"
)

GRID_ROOT = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "experiments"
    / "normalize_v2_event_grid"
)

SIMILARITIES = (0.3, 0.4, 0.5, 0.6)
TIME_WINDOWS = (24, 48, 72, 168)
MAX_ENTITY_DF_RATIO = 0.01


# Event 재생성에 필요한 v2 frozen upstream만 각 run에 연결한다.
# Event/master/validation 결과는 절대 복사하지 않는다.
FROZEN_INPUT_NAMES = (
    "articles_base.parquet",
    "train_used_article_ids.parquet",
    "category_mapping.parquet",
    "articles_with_category.parquet",
    "article_embedding_input.parquet",
    "article_embeddings.npy",
    "article_entities.parquet",
    "entity_normalization_map.parquet",
)


def _fmt_float(value: float) -> str:
    return (
        f"{value:.6f}"
        .rstrip("0")
        .rstrip(".")
        .replace(".", "p")
    )


def _run_name(similarity: float, window: int) -> str:
    return (
        f"sim_{_fmt_float(similarity)}"
        f"_win_{window}h"
        f"_df_{_fmt_float(MAX_ENTITY_DF_RATIO)}"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Frozen normalize_v2 Event 4x4 parameter grid "
            "(similarity 0.3/0.4/0.5/0.6 x "
            "window 24/48/72/168h)"
        )
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="완료된 조합은 건너뛰고 나머지만 실행합니다.",
    )
    return parser.parse_args()


def _validate_frozen_v2() -> None:
    missing = [
        FROZEN_V2_DIR / name
        for name in FROZEN_INPUT_NAMES
        if not (FROZEN_V2_DIR / name).exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Frozen normalize_v2 입력이 부족합니다:\n"
            + "\n".join(f"- {path}" for path in missing)
        )


def _link_or_copy(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        raise FileExistsError(
            f"대상 파일이 이미 존재합니다: {destination}"
        )

    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def _prepare_run_inputs(run_dir: Path) -> dict[str, str]:
    model_input_dir = run_dir / "model_inputs"
    model_input_dir.mkdir(parents=True, exist_ok=False)

    materialization: dict[str, str] = {}

    for name in FROZEN_INPUT_NAMES:
        source = FROZEN_V2_DIR / name
        destination = model_input_dir / name
        materialization[name] = _link_or_copy(
            source,
            destination,
        )

    return materialization


def _worker_env(
    run_dir: Path,
    similarity: float,
    window: int,
) -> dict[str, str]:
    env = os.environ.copy()

    # 핵심: worker Python 프로세스가 src를 import하기 전에 환경변수를 준다.
    env["SID_OUTPUT_DIR"] = str(run_dir)
    env["SID_EVENT_SIMILARITY_THRESHOLD"] = str(similarity)
    env["SID_EVENT_TIME_WINDOW_HOURS"] = str(window)
    env["SID_EVENT_MAX_ENTITY_DF_RATIO"] = str(
        MAX_ENTITY_DF_RATIO
    )

    return env


def _read_summary(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "run_summary.json"
    if not path.exists():
        raise FileNotFoundError(
            f"worker summary가 없습니다: {path}"
        )

    return json.loads(path.read_text(encoding="utf-8"))


def _write_grid_summary(rows: list[dict[str, Any]]) -> None:
    GRID_ROOT.mkdir(parents=True, exist_ok=True)

    json_path = GRID_ROOT / "grid_summary.json"
    json_path.write_text(
        json.dumps(
            rows,
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    csv_path = GRID_ROOT / "grid_summary.csv"

    fieldnames = [
        "similarity",
        "time_window_hours",
        "max_entity_df_ratio",
        "similarity_edge_count",
        "train_event_count",
        "train_singleton_event_count",
        "train_max_event_article_count",
        "time_candidate_pair_count",
        "validation_matched_existing_event_count",
        "validation_new_event_count",
        "final_event_count",
        "run_dir",
    ]

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    key: row.get(key)
                    for key in fieldnames
                }
            )


def main() -> None:
    args = _parse_args()

    _validate_frozen_v2()
    GRID_ROOT.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Normalize-v2 Event Parameter Grid")
    print("=" * 80)
    print(f"Frozen v2 : {FROZEN_V2_DIR}")
    print(f"Grid root : {GRID_ROOT}")
    print("Similarity: 0.3 / 0.4 / 0.5 / 0.6")
    print("Window    : 24 / 48 / 72 / 168 hours")
    print("High-DF   : 0.01")
    print("Total     : 16 runs")
    print("Overwrite : NEVER")
    print("=" * 80)

    summary_rows: list[dict[str, Any]] = []

    for window in TIME_WINDOWS:
        for similarity in SIMILARITIES:
            name = _run_name(similarity, window)
            run_dir = GRID_ROOT / name
            complete_marker = run_dir / "_COMPLETE.json"

            if run_dir.exists():
                if complete_marker.exists() and args.skip_existing:
                    print(f"[SKIP] {name}")
                    summary = _read_summary(run_dir)
                else:
                    raise FileExistsError(
                        "동일 조합 폴더가 이미 존재하므로 "
                        "절대 덮어쓰지 않습니다.\n"
                        f"경로={run_dir}\n"
                        "완료된 조합을 건너뛰려면 "
                        "--skip-existing를 사용하세요."
                    )
            else:
                print()
                print("#" * 80)
                print(f"[RUN] {name}")
                print("#" * 80)

                run_dir.mkdir(parents=True, exist_ok=False)

                try:
                    materialization = _prepare_run_inputs(
                        run_dir
                    )

                    (
                        run_dir
                        / "frozen_input_materialization.json"
                    ).write_text(
                        json.dumps(
                            materialization,
                            ensure_ascii=False,
                            indent=2,
                        ),
                        encoding="utf-8",
                    )

                    subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "event_parameter_sweep.worker",
                        ],
                        cwd=PROJECT_ROOT,
                        env=_worker_env(
                            run_dir,
                            similarity,
                            window,
                        ),
                        check=True,
                    )
                except Exception:
                    # partial 폴더는 지우지 않는다.
                    # _COMPLETE.json이 없으므로 완료 결과와 구분된다.
                    print(
                        "[FAILED] 부분 결과를 진단용으로 보존합니다: "
                        f"{run_dir}"
                    )
                    raise

                summary = _read_summary(run_dir)

            metrics = summary.get("metrics") or {}

            summary_rows.append(
                {
                    "similarity": similarity,
                    "time_window_hours": window,
                    "max_entity_df_ratio": (
                        MAX_ENTITY_DF_RATIO
                    ),
                    "similarity_edge_count": metrics.get(
                        "similarity_edge_count"
                    ),
                    "train_event_count": metrics.get(
                        "train_event_count"
                    ),
                    "train_singleton_event_count": metrics.get(
                        "train_singleton_event_count"
                    ),
                    "train_max_event_article_count": metrics.get(
                        "train_max_event_article_count"
                    ),
                    "time_candidate_pair_count": metrics.get(
                        "time_candidate_pair_count"
                    ),
                    "validation_matched_existing_event_count": (
                        metrics.get(
                            "validation_matched_existing_event_count"
                        )
                    ),
                    "validation_new_event_count": metrics.get(
                        "validation_new_event_count"
                    ),
                    "final_event_count": metrics.get(
                        "final_event_count"
                    ),
                    "run_dir": str(run_dir),
                }
            )

            _write_grid_summary(summary_rows)

    print()
    print("=" * 80)
    print("ALL 16 RUNS COMPLETE")
    print(f"CSV  : {GRID_ROOT / 'grid_summary.csv'}")
    print(f"JSON : {GRID_ROOT / 'grid_summary.json'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
