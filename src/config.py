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


# =================================
# ARTICLE BUILD OUTPUT PATH (build_articles.py 관련)
# build_articles.py에서 생성하는 기사 관련 결과 파일의 저장 경로 
# =================================


# STEP1. 가공된 기사 데이터 저장 경로
# build_valid_articels()가 만든 유효 기사 데이터를 아래 경로에 Parquet 파일로 저장
ARTICLES_BASE_PATH = (
    MODEL_INPUT_DIR/"articles_base.parquet"
)

# train에서 실제 사용하는 기사 ID 저장 경로
TRAIN_USED_ARTICLE_IDS_PATH = (
    MODEL_INPUT_DIR/"train_used_article_ids.parquet"
)

# train 데이터만 기준으로 생성한 카테고리 매핑 저장 경로 
CATEGORY_MAPPING_PATH = (
    MODEL_INPUT_DIR/"category_mapping.parquet"
)

# 전체 유효 기사에 카테고리 ID 연결한 결과 저장 경로 
ARTICLES_WITH_CATEGORY_PATH = (
    MODEL_INPUT_DIR/"articles_with_category.parquet"
)

# E5 임베딩 모델에 입력할 기사 순서와 텍스트 저장 경로 
ARTICLE_EMBEDDING_INPUT_PATH = (
    MODEL_INPUT_DIR / "article_embedding_input.parquet"
)

# 실제 768차원 기사 임베딩 배열 저장 경로
ARTICLE_EMBEDDINGS_PATH = (
    MODEL_INPUT_DIR / "article_embeddings.npy"
)

# ======================================
# EVENT BUILD OUTPUT PATHS (build_articles.py)
# 사건 클러스터링 결과 


# 기사별 실제 사건 ID 저장
ARTICLE_EVENTS_PATH = (
    MODEL_INPUT_DIR / "article_events.parquet"
)

# 사건별 메타데이터 저장 (사건 하나에 대해 몇개 기사 있는지, 첫번째 기사 발행시각은 언젠지 등)
# 이 사건이 아직 살아있는지 확인 여부 
EVENT_MASTER_PATH = (
    MODEL_INPUT_DIR / "event_master.parquet"
)

# train 기준 entity IDF 저장 
# 각 entity_key (PER::zlatan - 5.2) : idf_value 
ENTITY_IDF_PATH = (
    MODEL_INPUT_DIR / "entity_idf.parquet"
)


# 기사 임베딩 생성에 사용하는 모델 설정

# 기사 임베딩에 사용할 허깅페이스 모델 이름
ARTICLE_EMBEDDING_MODEL_NAME = (
    "intfloat/multilingual-e5-base"
)

# 기사 텍스트를 토큰화할 때 사용할 최대 토큰 길이
ARTICLE_EMBEDDING_MAX_LENGTH = 256

# 한 번에 모델에 전달할 기사 수
# 메모리 부족 오류가 발생하면 16에서 8 또는 4로 줄인다.
ARTICLE_EMBEDDING_BATCH_SIZE = 16



# 클러스터링 하이퍼파라미터 
EVENT_ENTITY_SIMILARITY_THRESHOLD = 0.3
EVENT_TIME_WINDOW_HOURS = 72 

# event 연결 계산에서 제외할 entity 기준 
# train 기사 중 1% 이상에서 등장하면 event sim 계산에서 제외
EVENT_MAX_ENTITY_DF_RATIO = 0.01 

# TRAIN에서 실제 사용하는 기사들의 최종 학습용 메타데이터 저장
ARTICLE_MASTER_PATH = (MODEL_INPUT_DIR / "article_master.parquet")

# =========================================== 여기까지 build_train.py에 등장 

# ==========================================
# build_validation.py 
# ==========================================

# validation history/behaviors/candidate에서 
# 실제 참조되는 전체 유효 기사 ID 
VALIDATION_USED_ARTICLE_IDS_PATH = (
    MODEL_INPUT_DIR / "validation_used_article_ids.parquet"
)

# validation에서 사용되지만 train에서 한 번도 사용되지 않은 기사 ID
# 이 기사들은 학습 완료된 RQ-VAE에 넣어서 frozen inference로 SID 생성
VALIDATION_ONLY_ARTICLE_IDS_PATH = (
    MODEL_INPUT_DIR / "validation_only_article_ids.parquet"
)

# validation_only 기사에 dynamic event assignment를 적용한 결과
VALIDATION_ARTICLE_EVENTS_PATH = (
    MODEL_INPUT_DIR / "validation_article_events.parquet"
)

# train event 상태에 validation 기사를 시간순으로 반영한 
# 최종 event 상태 

# 기존 EVENT_MASTER_PATH는 train 결과이므로 덮어쓰지 않는다.
EVENT_MASTER_WITH_VALIDATION_PATH = (
    MODEL_INPUT_DIR / "event_master_with_validation.parquet"
)

# RQ-VAE Validation inference에 전달할 기사 단위 metadata
VALIDATION_ARTICLE_MASTER_PATH = (
    MODEL_INPUT_DIR / "validation_article_master.parquet"
)
