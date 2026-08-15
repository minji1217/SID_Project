from __future__ import annotations

import argparse
from pathlib import Path
from pprint import pprint
import shutil

from src import config
from src.build_article_master import build_train_article_master
from src.build_train import build_article_events
from src.build_validation import build_validation
from src.entity_processing import build_article_entities


# =============================================================================
# Entity-only Experiment Runner
#
# 목적
# -----------------------------------------------------------------------------
# Entity 표현만 바꾸는 Before/After 실험에서 E5 embedding을 다시 생성하지 않고
# 아래 범위만 재실행한다.
#
# Entity Processing
#   -> Train DF / high-DF / IDF
#   -> Train Event clustering
#   -> Train article_master 갱신
#   -> Validation dynamic Event assignment
#   -> Validation article_master 갱신
#
# 기존 Category mapping / E5 embedding은 그대로 재사용한다.
# RQ-VAE / SID / Sequence는 이 실험 단계에서 실행하지 않는다.
# =============================================================================


# Event 효과 확인에 필요한 파일만 snapshot한다.
# 옛 article_semantic_ids / sequence 같은 downstream 결과를 섞지 않기 위해
# MODEL_INPUT_DIR 전체를 무조건 복사하지 않는다.
_SNAPSHOT_PATH_NAMES = [
    "articles_base.parquet",
    "train_used_article_ids.parquet",
    "category_mapping.parquet",
    "articles_with_category.parquet",
    "article_embedding_input.parquet",
    "article_embeddings.npy",
    "article_entities.parquet",
    "entity_normalization_map.parquet",
    "article_events.parquet",
    "event_master.parquet",
    "entity_idf.parquet",
    "article_master.parquet",
    "validation_used_article_ids.parquet",
    "validation_only_article_ids.parquet",
    "validation_article_events.parquet",
    "event_master_with_validation.parquet",
    "validation_article_master.parquet",
]


def _print_result(
    title: str,
    result: dict,
) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)
    pprint(result)



def _assert_prerequisites() -> None:
    """
    Entity-only rerun 전에 기존 article/category/embedding 산출물이 존재하는지 확인한다.
    """

    required_paths = [
        config.ARTICLES_WITH_CATEGORY_PATH,
        config.TRAIN_USED_ARTICLE_IDS_PATH,
        config.ARTICLE_EMBEDDING_INPUT_PATH,
        config.ARTICLE_EMBEDDINGS_PATH,
    ]

    missing = [
        str(path)
        for path in required_paths
        if not Path(path).exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Entity-only 실험에 필요한 기존 preprocess 산출물이 없습니다. "
            "먼저 baseline preprocess를 완료해야 합니다. "
            f"missing={missing}"
        )



def _snapshot_results(
    experiment_name: str,
    overwrite: bool,
) -> Path:
    """
    현재 model_inputs 중 Entity/Event 비교에 필요한 파일만 실험 폴더에 보존한다.
    """

    experiment_root = (
        config.OUTPUT_DIR
        / "experiments"
        / experiment_name
    )

    snapshot_dir = (
        experiment_root
        / "model_inputs"
    )

    if snapshot_dir.exists():
        if not overwrite:
            raise FileExistsError(
                "실험 snapshot이 이미 존재합니다. "
                f"경로={snapshot_dir}. "
                "덮어쓰려면 --overwrite를 명시하세요."
            )

        shutil.rmtree(
            snapshot_dir
        )

    snapshot_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    copied_files: list[str] = []

    for file_name in _SNAPSHOT_PATH_NAMES:
        source = (
            config.MODEL_INPUT_DIR
            / file_name
        )

        if not source.exists():
            # Validation 단계 전/특정 환경에서 없는 보조 파일은 건너뛴다.
            continue

        destination = (
            snapshot_dir
            / file_name
        )

        shutil.copy2(
            source,
            destination,
        )

        copied_files.append(
            file_name
        )

    manifest_path = (
        experiment_root
        / "experiment_config.txt"
    )

    manifest_lines = [
        f"experiment={experiment_name}",
        f"entity_processing_mode={config.ENTITY_PROCESSING_MODE}",
        f"similarity_threshold={config.EVENT_ENTITY_SIMILARITY_THRESHOLD}",
        f"time_window_hours={config.EVENT_TIME_WINDOW_HOURS}",
        f"max_entity_df_ratio={config.EVENT_MAX_ENTITY_DF_RATIO}",
        "fit_split=train_used_articles_only",
        "copied_files=" + ",".join(copied_files),
    ]

    manifest_path.write_text(
        "\n".join(manifest_lines) + "\n",
        encoding="utf-8",
    )

    return snapshot_dir



def run_entity_experiment(
    snapshot: bool = True,
    overwrite: bool = False,
) -> dict:
    """
    현재 config.ENTITY_PROCESSING_MODE로 Entity/Event 단계만 재실행한다.
    """

    _assert_prerequisites()
    config.create_output_directories()

    mode = str(
        config.ENTITY_PROCESSING_MODE
    ).strip().lower()

    if mode == "normalize_and_link":
        experiment_name = "entity_linked"
    else:
        experiment_name = mode

    entity_result = (
        build_article_entities()
    )

    _print_result(
        "ENTITY STEP 1 - Entity Processing",
        entity_result,
    )

    train_event_result = (
        build_article_events()
    )

    _print_result(
        "ENTITY STEP 2 - Train Event Rebuild",
        train_event_result,
    )

    train_master_result = (
        build_train_article_master()
    )

    _print_result(
        "ENTITY STEP 3 - Train Article Master",
        train_master_result,
    )

    validation_result = (
        build_validation()
    )

    _print_result(
        "ENTITY STEP 4 - Validation Event Assignment",
        validation_result,
    )

    snapshot_dir: str | None = None

    if snapshot:
        snapshot_path = _snapshot_results(
            experiment_name=experiment_name,
            overwrite=overwrite,
        )

        snapshot_dir = str(
            snapshot_path
        )

        print()
        print(
            "Experiment snapshot:",
            snapshot_path,
        )

    return {
        "status": "SUCCESS",
        "entity_processing_mode": mode,
        "experiment_name": experiment_name,
        "entity_result": entity_result,
        "train_event_result": train_event_result,
        "train_master_result": train_master_result,
        "validation_result": validation_result,
        "snapshot_dir": snapshot_dir,
    }

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Entity representation만 변경하여 Event 단계만 재실행합니다."
        )
    )

    parser.add_argument(
        "--no-snapshot",
        action="store_true",
        help="실험 결과 snapshot을 만들지 않습니다.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="동일 experiment snapshot이 있으면 덮어씁니다.",
    )

    args = parser.parse_args()

    result = run_entity_experiment(
        snapshot=(
            not args.no_snapshot
        ),
        overwrite=args.overwrite,
    )

    print()
    print("=" * 80)
    print("Entity Experiment 완료")
    print("=" * 80)
    pprint(result)


if __name__ == "__main__":
    main()
