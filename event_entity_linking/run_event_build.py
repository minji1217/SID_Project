from __future__ import annotations

import argparse
from pprint import pformat

from . import config
from .build_train_events import build_train_events
from .build_validation_events import build_validation_events


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build Entity-Linking Event experiment without modifying src/ or "
            "frozen normalize_v2 artifacts."
        )
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--train-only",
        action="store_true",
        help="Build only Train Events.",
    )
    mode.add_argument(
        "--validation-only",
        action="store_true",
        help="Build Validation Events from existing Entity-Linking Train Event outputs.",
    )
    return parser.parse_args()


def _write_summary(text: str) -> None:
    config.create_output_directories()
    config.EVENT_BUILD_SUMMARY_PATH.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()

    print("=" * 80)
    print("SID_Project Entity-Linking Event experiment")
    print("=" * 80)
    print(f"Entity source     : {config.ARTICLE_LINKED_ENTITIES_PATH}")
    print(f"Frozen upstream   : {config.FROZEN_NORMALIZE_V2_DIR}")
    print(f"New output        : {config.EVENT_OUTPUT_DIR}")
    print(
        "Frozen params     : "
        f"similarity={config.EVENT_ENTITY_SIMILARITY_THRESHOLD}, "
        f"time_window={config.EVENT_TIME_WINDOW_HOURS}h, "
        f"max_entity_df_ratio={config.EVENT_MAX_ENTITY_DF_RATIO}"
    )
    print("OpenAI/Wikidata API: NOT USED")
    print()

    train_result = None
    validation_result = None

    if not args.validation_only:
        print("[STEP 1] Building Train Events...")
        train_result = build_train_events()
        print("[OK] Train Event build completed.")
        print(pformat(train_result, sort_dicts=False))
        print()

    if not args.train_only:
        print("[STEP 2] Building Validation Events...")
        validation_result = build_validation_events()
        print("[OK] Validation Event build completed.")
        print(pformat(validation_result, sort_dicts=False))
        print()

    lines = [
        "SID_Project Entity-Linking Event Build Summary",
        "=" * 60,
        f"entity_source={config.ARTICLE_LINKED_ENTITIES_PATH}",
        f"frozen_upstream={config.FROZEN_NORMALIZE_V2_DIR}",
        f"output_dir={config.EVENT_OUTPUT_DIR}",
        f"similarity={config.EVENT_ENTITY_SIMILARITY_THRESHOLD}",
        f"time_window_hours={config.EVENT_TIME_WINDOW_HOURS}",
        f"max_entity_df_ratio={config.EVENT_MAX_ENTITY_DF_RATIO}",
        "openai_api_used=False",
        "wikidata_api_used=False",
        "",
        "[Train Result]",
        pformat(train_result, sort_dicts=False) if train_result is not None else "SKIPPED",
        "",
        "[Validation Result]",
        pformat(validation_result, sort_dicts=False)
        if validation_result is not None
        else "SKIPPED",
        "",
        "[Outputs]",
        f"article_events={config.ARTICLE_EVENTS_PATH}",
        f"event_master={config.EVENT_MASTER_PATH}",
        f"entity_idf={config.ENTITY_IDF_PATH}",
        f"validation_article_events={config.VALIDATION_ARTICLE_EVENTS_PATH}",
        f"event_master_with_validation={config.EVENT_MASTER_WITH_VALIDATION_PATH}",
        f"all_article_events={config.ALL_ARTICLE_EVENTS_PATH}",
    ]
    _write_summary("\n".join(lines) + "\n")

    print("=" * 80)
    print("COMPLETED")
    print(f"Outputs: {config.EVENT_OUTPUT_DIR}")
    print("Frozen normalize_v2/src files were not modified by this runner.")
    print("=" * 80)


if __name__ == "__main__":
    main()
