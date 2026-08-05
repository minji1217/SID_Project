from pathlib import Path


# 현재 파일:
# SID_Project/src/config.py
#
# parent       -> SID_Project/src
# parent.parent -> SID_Project
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# 데이터 폴더
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
OUTPUT_DIR = DATA_DIR / "output"


# 원본 파일 경로
ARTICLES_PATH = RAW_DIR / "articles.parquet"

TRAIN_BEHAVIORS_PATH = RAW_DIR / "train" / "behaviors.parquet"
TRAIN_HISTORY_PATH = RAW_DIR / "train" / "history.parquet"

VALIDATION_BEHAVIORS_PATH = (
    RAW_DIR / "validation" / "behaviors.parquet"
)
VALIDATION_HISTORY_PATH = (
    RAW_DIR / "validation" / "history.parquet"
)


# 결과 저장 폴더
REPORT_DIR = OUTPUT_DIR / "reports"
MODEL_INPUT_DIR = OUTPUT_DIR / "model_inputs"


def create_output_directories() -> None:
    """
    결과 저장 폴더가 없으면 생성한다.
    이미 존재해도 오류가 발생하지 않는다.
    """

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_INPUT_DIR.mkdir(parents=True, exist_ok=True)