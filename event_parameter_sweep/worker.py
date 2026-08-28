from __future__ import annotations

import json
import os
from pathlib import Path
from pprint import pformat
from typing import Any

from src import config
from src.build_article_master import build_train_article_master
from src.build_train import build_article_events
from src.build_validation import build_validation
from src.export_inputs import export_rqvae_inputs


def _replace_with_hardlink(source: Path, destination: Path) -> str:
    """
    export_inputs.py가 article_embeddings.npy를 복사한 뒤,
    같은 디스크라면 hard-link로 바꿔 중복 용량을 줄인다.
    실패하면 기존 copy를 그대로 유지한다.
    """
    if not source.exists() or not destination.exists():
        return "not_applicable"

    temp_link = destination.with_name(destination.name + ".hardlink_tmp")

    try:
        if temp_link.exists():
            temp_link.unlink()

        os.link(source, temp_link)

        # hard-link 생성에 성공한 뒤에만 기존 copy를 제거한다.
        destination.unlink()
        temp_link.replace(destination)
        return "hardlink"
    except OSError:
        if temp_link.exists():
            temp_link.unlink()
        return "copy"


def _extract_validation_event_result(validation_result: dict[str, Any]) -> dict[str, Any]:
    value = validation_result.get("event_result")
    if isinstance(value, dict):
        return value
    return {}


def main() -> None:
    output_dir = Path(config.OUTPUT_DIR)
    model_input_dir = Path(config.MODEL_INPUT_DIR)

    # ------------------------------------------------------------------
    # Guardrails
    # ------------------------------------------------------------------
    mode = str(getattr(config, "ENTITY_PROCESSING_MODE", "")).strip().lower()
    version = str(
        getattr(config, "ENTITY_NORMALIZATION_VERSION", "")
    ).strip().lower()

    if mode != "normalize_only" or version != "v2":
        raise RuntimeError(
            "이 worker는 frozen Safe Normalization v2 전용입니다. "
            f"현재 mode={mode}, version={version}"
        )

    required_frozen_inputs = [
        config.ARTICLES_BASE_PATH,
        config.TRAIN_USED_ARTICLE_IDS_PATH,
        config.CATEGORY_MAPPING_PATH,
        config.ARTICLES_WITH_CATEGORY_PATH,
        config.ARTICLE_EMBEDDING_INPUT_PATH,
        config.ARTICLE_EMBEDDINGS_PATH,
        config.ARTICLE_ENTITIES_PATH,
    ]

    missing = [
        str(path)
        for path in required_frozen_inputs
        if not Path(path).exists()
    ]
    if missing:
        raise FileNotFoundError(
            "run model_inputs에 필요한 frozen normalize_v2 입력이 없습니다:\n"
            + "\n".join(f"- {p}" for p in missing)
        )

    print("=" * 80)
    print("Normalize-v2 Event grid worker")
    print("=" * 80)
    print(f"OUTPUT_DIR  : {output_dir}")
    print(f"MODEL_INPUT : {model_input_dir}")
    print(
        "PARAMS      : "
        f"similarity={config.EVENT_ENTITY_SIMILARITY_THRESHOLD}, "
        f"window={config.EVENT_TIME_WINDOW_HOURS}h, "
        f"high_df={config.EVENT_MAX_ENTITY_DF_RATIO}"
    )
    print("Entity      : frozen normalize_v2")
    print("API         : OpenAI/Wikidata NOT USED")
    print()

    # ------------------------------------------------------------------
    # 1. Train Event
    # build_article_events()에는 값을 명시적으로 전달한다.
    # ------------------------------------------------------------------
    print("[1/4] Train Event build")
    train_event_result = build_article_events(
        entity_similarity_threshold=(
            config.EVENT_ENTITY_SIMILARITY_THRESHOLD
        ),
        time_window_hours=config.EVENT_TIME_WINDOW_HOURS,
        max_entity_df_ratio=config.EVENT_MAX_ENTITY_DF_RATIO,
    )
    print(pformat(train_event_result, sort_dicts=False))
    print()

    # ------------------------------------------------------------------
    # 2. Train RQ-VAE article_master
    # ------------------------------------------------------------------
    print("[2/4] Train article_master build")
    train_master_result = build_train_article_master()
    print(pformat(train_master_result, sort_dicts=False))
    print()

    # ------------------------------------------------------------------
    # 3. Validation
    #
    # build_validation.py의 _assign_validation_events default argument는
    # import 시점의 config 값을 사용한다.
    # 이 worker 자체가 조합마다 별도 subprocess에서 시작하므로
    # import 시점부터 해당 조합 값이 적용된다.
    # ------------------------------------------------------------------
    print("[3/4] Validation Event + article_master build")
    validation_result = build_validation()
    print(pformat(validation_result, sort_dicts=False))
    print()

    # ------------------------------------------------------------------
    # 4. Current RQ-VAE packages
    # ------------------------------------------------------------------
    print("[4/4] RQ-VAE package export")
    export_result = export_rqvae_inputs()

    train_export_dir = Path(export_result["rqvae_train_export_dir"])
    validation_export_dir = Path(
        export_result["rqvae_validation_export_dir"]
    )

    train_embedding_mode = _replace_with_hardlink(
        Path(config.ARTICLE_EMBEDDINGS_PATH),
        train_export_dir / "article_embeddings.npy",
    )
    validation_embedding_mode = _replace_with_hardlink(
        Path(config.ARTICLE_EMBEDDINGS_PATH),
        validation_export_dir / "article_embeddings.npy",
    )

    validation_event_result = _extract_validation_event_result(
        validation_result
    )

    summary = {
        "status": "SUCCESS",
        "entity_mode": "normalize_v2",
        "similarity": config.EVENT_ENTITY_SIMILARITY_THRESHOLD,
        "time_window_hours": config.EVENT_TIME_WINDOW_HOURS,
        "max_entity_df_ratio": config.EVENT_MAX_ENTITY_DF_RATIO,
        "output_dir": str(output_dir),
        "model_input_dir": str(model_input_dir),
        "train_event_result": train_event_result,
        "train_master_result": train_master_result,
        "validation_result": validation_result,
        "rqvae_export_result": export_result,
        "article_embeddings_export_mode": {
            "train": train_embedding_mode,
            "validation": validation_embedding_mode,
        },
        "metrics": {
            "similarity_edge_count": train_event_result.get(
                "similarity_edge_count"
            ),
            "train_event_count": train_event_result.get(
                "train_event_count"
            ),
            "train_singleton_event_count": train_event_result.get(
                "train_singleton_event_count"
            ),
            "train_max_event_article_count": train_event_result.get(
                "train_max_event_article_count"
            ),
            "time_candidate_pair_count": train_event_result.get(
                "time_candidate_pair_count"
            ),
            "validation_matched_existing_event_count": (
                validation_event_result.get(
                    "matched_existing_event_count"
                )
            ),
            "validation_new_event_count": (
                validation_event_result.get(
                    "new_validation_event_count"
                )
            ),
            "final_event_count": validation_event_result.get(
                "final_event_count"
            ),
        },
    }

    summary_path = output_dir / "run_summary.json"
    summary_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    complete_path = output_dir / "_COMPLETE.json"
    complete_path.write_text(
        json.dumps(
            {
                "status": "SUCCESS",
                "similarity": config.EVENT_ENTITY_SIMILARITY_THRESHOLD,
                "time_window_hours": config.EVENT_TIME_WINDOW_HOURS,
                "max_entity_df_ratio": config.EVENT_MAX_ENTITY_DF_RATIO,
                "run_summary": str(summary_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print("WORKER COMPLETE")
    print(f"Summary : {summary_path}")
    print(f"RQ-VAE Train      : {train_export_dir}")
    print(f"RQ-VAE Validation : {validation_export_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
