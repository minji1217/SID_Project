from pathlib import Path


# This package is expected at:
# SID_Project/event_entity_linking/config.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = DATA_DIR / "output"
EXPERIMENTS_DIR = OUTPUT_DIR / "experiments"

# -----------------------------------------------------------------------------
# Frozen/read-only upstream artifacts from the completed normalize_v2 experiment.
# Do not write into this directory from the Entity-Linking Event experiment.
# -----------------------------------------------------------------------------
FROZEN_NORMALIZE_V2_DIR = EXPERIMENTS_DIR / "normalize_v2" / "model_inputs"

ARTICLES_WITH_CATEGORY_PATH = (
    FROZEN_NORMALIZE_V2_DIR / "articles_with_category.parquet"
)
TRAIN_USED_ARTICLE_IDS_PATH = (
    FROZEN_NORMALIZE_V2_DIR / "train_used_article_ids.parquet"
)
VALIDATION_ONLY_ARTICLE_IDS_PATH = (
    FROZEN_NORMALIZE_V2_DIR / "validation_only_article_ids.parquet"
)

# Optional frozen normalize_v2 Event outputs. These are never overwritten.
NORMALIZE_V2_ARTICLE_EVENTS_PATH = (
    FROZEN_NORMALIZE_V2_DIR / "article_events.parquet"
)
NORMALIZE_V2_EVENT_MASTER_PATH = (
    FROZEN_NORMALIZE_V2_DIR / "event_master.parquet"
)
NORMALIZE_V2_ENTITY_IDF_PATH = (
    FROZEN_NORMALIZE_V2_DIR / "entity_idf.parquet"
)
NORMALIZE_V2_VALIDATION_ARTICLE_EVENTS_PATH = (
    FROZEN_NORMALIZE_V2_DIR / "validation_article_events.parquet"
)
NORMALIZE_V2_EVENT_MASTER_WITH_VALIDATION_PATH = (
    FROZEN_NORMALIZE_V2_DIR / "event_master_with_validation.parquet"
)

# -----------------------------------------------------------------------------
# Completed GPT/Wikidata Entity Linking output.
# This is the only Entity representation used by this experiment.
# -----------------------------------------------------------------------------
ENTITY_LINKING_DIR = EXPERIMENTS_DIR / "entity_linking_full"
ARTICLE_LINKED_ENTITIES_PATH = (
    ENTITY_LINKING_DIR / "article_linked_entities.parquet"
)

# -----------------------------------------------------------------------------
# New Event outputs. Completely separate from src/model_inputs and normalize_v2.
# -----------------------------------------------------------------------------
EVENT_OUTPUT_DIR = EXPERIMENTS_DIR / "event_entity_linking"

ARTICLE_EVENTS_PATH = EVENT_OUTPUT_DIR / "article_events.parquet"
EVENT_MASTER_PATH = EVENT_OUTPUT_DIR / "event_master.parquet"
ENTITY_IDF_PATH = EVENT_OUTPUT_DIR / "entity_idf.parquet"

VALIDATION_ARTICLE_EVENTS_PATH = (
    EVENT_OUTPUT_DIR / "validation_article_events.parquet"
)
EVENT_MASTER_WITH_VALIDATION_PATH = (
    EVENT_OUTPUT_DIR / "event_master_with_validation.parquet"
)
ALL_ARTICLE_EVENTS_PATH = EVENT_OUTPUT_DIR / "all_article_events.parquet"
EVENT_BUILD_SUMMARY_PATH = EVENT_OUTPUT_DIR / "event_build_summary.txt"

# -----------------------------------------------------------------------------
# Frozen experiment hyperparameters.
# Before/After comparison changes Entity representation only.
# -----------------------------------------------------------------------------
EVENT_ENTITY_SIMILARITY_THRESHOLD = 0.3
EVENT_TIME_WINDOW_HOURS = 72
EVENT_MAX_ENTITY_DF_RATIO = 0.01


def create_output_directories() -> None:
    EVENT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
