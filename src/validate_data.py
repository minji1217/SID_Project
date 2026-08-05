# 원본 데이터가 정상인지 검사하는 입구 역할 

"""
검사 대상
1. data/raw/articles.parquet
2. data/raw/train/behaviors.parquet
3. data/raw/train/history.parquet
4. data/raw/validation/behaviors.parquet
5. data/raw/validation/history.parquet
--------------------------------------
현재 단계에서 검사하는 내용
1. 파일이 실제 경로에 존재?
2. Polars로 Parquet 파일을 읽을 수 있는지?
3. 각 파일의 전체 행 수
4. 각 파일의 컬럼 수
5. 각 파일의 컬럼명과 데이터 타입
6. 필수 컬럼 누락 여부 
--------------------------------------
검사 결과는 다음 경로에 JSON 파일로 저장
- data/output/reports/raw_basic_validation.json
"""

# STEP1. 외부 라이브러리 불러오기 
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import polars as pl
from src.config import(
    ARTICLES_PATH,
    MODEL_INPUT_DIR,
    REPORT_DIR,
    TRAIN_BEHAVIORS_PATH,
    TRAIN_HISTORY_PATH,
    VALIDATION_BEHAVIORS_PATH,
    VALIDATION_HISTORY_PATH,
    create_output_directories,
)

# 2. 검사할 원본 파일 5개 등록
# 데이터셋 이름과 실제 parquet 파일 경로 연결
# key : 프로그램 안에서 사용할 데이터셋 이름, value : config.py에 정의한 실제 파일 경로 
DATASET_PATHS: dict[str, Path] = {
    "articles": ARTICLES_PATH,
    "train_behaviors": TRAIN_BEHAVIORS_PATH,
    "train_history": TRAIN_HISTORY_PATH,
    "validation_behaviors": VALIDATION_BEHAVIORS_PATH,
    "validation_history": VALIDATION_HISTORY_PATH,
}

# 3. 데이터 파일별 필수 컬럼 정의 
# 각 데이터 파일에 반드시 존재해야 하는 컬럼 정의 
# 이때, 실제 컬럼과 필수 컬럼 사이 차집합 계산 위해 set 사용 
REQUIRED_COLUMNS: dict[str, set[str]] = {
    # 3-1. articles.parquet 필수 컬럼
    "articles": {
        "article_id",       # 기사 고유 ID 
        "title",            # 기사 제목 
        "subtitle",         # 기사 부제목 또는 요약문
        "published_time",   # 기사 발행 시각
        "category_str",     # 기사 카테고리 문자열
        "ner_clusters",     # 기사에 등장하는 개체명 목록
        "entity_groups",    # 각 개체명의 타입 목록 (예: 인물, 장소, 기관)
    },
    # 3-2. train/behaviors.parquet 필수 컬럼
    "train_behaviors": {
        "impression_id",        # 노출 이벤트 고유 ID
        "user_id",              # 사용자 고유 ID
        "impression_time",      # 해당 행동이 발생한 시각
        "article_id",           # 행동 당시 사용자가 보고 있던 현재 기사 ID (null 가능)
        "article_ids_clicked",  # 해당 impression에서 사용자가 클릭한 기사 ID 목록 
    },
    # 3-3. validation/behaviors.parquet 필수 컬럼
    "validation_behaviors": {
        "impression_id",        
        "user_id",              
        "impression_time",      
        "article_id",           
        "article_ids_clicked",  
    },
    # 3-4. train/history.parquet 필수 컬럼
    "train_history": {
        "user_id",                  # 사용자 고유 ID
        "article_id_fixed",         # 사용자가 과거에 읽은 기사 ID 목록 
        "impression_time_fixed",    # 각 과거 기사 행동의 발생 시각 목록
    },
    # 3-5. validation/history.parquet 필수 컬럼
     "validation_history": {
            "user_id",                  
            "article_id_fixed",         
            "impression_time_fixed",    
    },
}

# 4. parquet 파일 하나 검사하는 함수 
def validate_parquet_file(
        dataset_name: str, # 검사할 데이터셋 이름 (예: train_behaviors)
        file_path: Path,   # 검사할 parquet 파일의 실제 경로 
) -> dict[str, Any]:       # 파일 존재 여부, 행/컬럼 수, 필수 컬럼 누락 여부 등이 들어있는 검사 결과
    # 4-1. 검사 결과의 기본 정보 만들기
    # 각 파일의 검사 결과 저장할 딕셔너리 
    result : dict[str, Any] = {
        # 현재 검사하고 있는 데이터셋 이름
        "dataset_name": dataset_name,
        "file_path": str(file_path),
        "exists": file_path.exists(), # 파일이 실제 경로에 존재하는지 chk
    }

    # 4-2. 파일이 존재하지 않으면 즉시 FAIL 처리 
    if not file_path.exists():
        # 해당 파일의 검사 상태를 FAIL로 기록
        result["status"] = "FAIL"
        # 실패 원인을 검사 결과에 기록
        result["error"] = "파일이 존재하지 않습니다."

        # 현재까지 만든 결과를 반환하고 함수 종료
        return result

    # 4-3. parquet 파일 읽고 스키마 검사 
    try:
        # pl.scan_parquet()은 원본 데이터를 즉시 전부 메모리에
        # 올리지 않고 LazyFrame 형태로 읽기 계획 만듦
        lazy_frame = pl.scan_parquet(file_path)

        # 4-4. 파일의 스키마 가져옴
        # 스키마에는 컬럼 이름, 컬럼별 데이터 타입 정보 들어있음
        # 예:
        # article_id -> Int32
        # title -> String
        # published_time -> Datetime
        schema = lazy_frame.collect_schema()

        # 4-5. 실제 컬럼 이름을 리스트로 가져옴
        # schema.names()를 사용하면 데이터 타입 제외 컬럼 이름만 리스트 형태로 읽어올 수 있음
        column_names = schema.names()

        # 4-6. 컬럼 이름을 set으로 변환
        # 필수 컬럼과 실제 컬럼 사이의 차집합을 계산하려면 실제 컬럼도 set 형태로 변환하는 것이 편리 
        existing_columns = set(column_names)

        # 4-7. 전체 행 수 계산
        # pl.len() : 전체 행의 개수 계산하는 Polars 표현식
        # alias("row_count") : 계산 결과 컬럼에 row_count 이름 붙임
        # select() : 원본 전체 컬럼 가져오지 않고 행 개수만 계산
        # collect() : LazyFrame에 작성한 계산을 실제로 실행
        # item() : 1행 1열 결과에서 python 숫자 하나만 꺼냄
        row_count = (
            lazy_frame.select(
                pl.len().alias("row_count")
            )
        ).collect().item()

        # 4-8. 현재 데이터셋의 필수 컬럼 가져옴
        # 예: dataset_name이 train_history라면 REQUIRED_COLUMNS["train_history"]에 정의된 컬럼들이
        # required_columns에 저장됨
        required_columns = REQUIRED_COLUMNS[dataset_name]

        # 4-9. 누락된 필수 컬럼 계산 (필수 컬럼 set - 실제 컬럼 set)
        # 예:
        # 필수 컬럼 = {"user_id", "article_id_fixed"}
        # 실제 컬럼 = {"user_id"}
        #
        # 결과:
        # {"article_id_fixed"}
        missing_columns = sorted(
            required_columns - existing_columns
        )
        # sorted 사용해서 결과를 이름순으로 정렬해 JSON 출력 순서 일정하게

        # 4-10. Polars 데이터 타입을 문자열로 변환 (JSON으로 저장하기 위해)
        schema_dict = {
            column_name : str(data_type)
            for column_name, data_type in schema.items()
        }

        # 4-11. 검사 결과를 result 딕셔너리에 추가
        # 누락된 필수 컬럼 없으면 PASS 처리
        status = ("PASS" if not missing_columns else "FAIL")

        # 처음에 만든 result 딕셔너리에 검사 결과 추가
        result.update({
            "status": status,   # 현재 파일의 최종 검사 상태
            "row_count": int(row_count), # 전체 행 수
            "column_count": len(column_names), # 전체 컬럼 수
            "columns": column_names, # 실제 파일에 존재하는 모든 컬럼 이름
            "schema": schema_dict, # 컬럼별 데이터 타입
            "required_columns": sorted(required_columns), # 이 파일에 필요하다고 정의한 필수 컬럼 목록
            "missing_required_columns": missing_columns, # 필요하지만 실제 파일에는 없는 컬럼 목록

        })

    # 4-12. parquet 읽기 중 발생한 오류 처리 
    # 예:
    # - 손상된 Parquet 파일
    # - 읽기 권한 문제
    # - 지원하지 않는 파일 형식
    # - Polars 실행 오류
    except Exception as error:
        # 검사 상태를 FAIL로 기록한다.
        result["status"] = "FAIL"

        # 오류 클래스 이름과 실제 메시지를 함께 기록한다.
        #
        # 예:
        # ComputeError: invalid parquet file
        result["error"] = (
            f"{type(error).__name__}: {error}"
        )

    # 4-13. 파일 하나의 검사 결과 반환
    # 이 결과는 run_validation()에서 받아 전체 데이터셋 검사 결과에 누적
    return result 

# 5. 전체 검사 결과를 JSON 파일로 저장하는 함수 
def save_report(
        report: dict[str,Any], # 저장할 전체 검사 결과 딕셔너리
        output_path: Path,     # JSON 파일이 저장될 실제 경로 
)->None:

    # 5-1. JSON 파일의 부모 폴더 확인 
    # 예: data/output/reports 만드려는데 data/ouput 없으면 이 순서로 다 만들도록
    # 예: 이미 폴더 존재한다면 아무것도 안함 (exist_ok)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 5-2. JSON 파일을 쓰기 모드로 열기
    # 덮어쓰는 대상은 원본 parquet이 아닌 검사 보고서 JSON
    with output_path.open(
        mode="w",
        encoding="utf-8",
    ) as file:
        # 5-3. 파이썬 딕셔너리를 JSON으로 저장
        # 한글을 실제 한글로 저장, JSON 내용 2칸 등려쓰기 
        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=2,
        )

# 6. 파일 하나의 검사 결과를 터미널에 출력하는 함수 
def print_validation_result(
        dataset_name:str,
        result: dict[str, Any],
) -> None:
    # 6-1. 검사 결과에서 PASS 또는 FAIL 가져옴
    status = result["status"]

    # 6-2. 데이터셋 이름과 상태 출력
    # 예:
    # [PASS] articles, [FAIL] train_history
    print()
    print("="*70)
    print(f"[{status}] {dataset_name}")
    print("="*70)
    print(f"파일 경로: {result['file_path']}") # 현재 검사한 원본 파일 경로 출력

    # 6-3. 파일 부재 또는 읽기 오류 출력 
    if "error" in result:
        print(f"오류: {result['error']}")

        return # 다음 데이터 파일로 진행 

    # 6-4. 정상적으로 읽힌 파일의 기본 정보 출력 
    # 행 수, 컬럼 수 출력
    print(f"행 수: {result['row_count']:,}") # 천 단위 구분기호 사용

    # 전체 컬럼 수를 출력한다.
    print(f"컬럼 수: {result['column_count']}")

    # 6-5. 누락된 필수 컬럼 확인 
    # 검사 결과에서 누락된 필수 컬럼 목록을 가져온다.
    missing_columns = result["missing_required_columns"]

    # 누락된 컬럼이 하나 이상 있다면 컬럼명을 출력한다.
    if missing_columns:
        print(
            "누락된 필수 컬럼: "
            + ", ".join(missing_columns)
        )

    # 누락 컬럼이 없다면 필수 컬럼이 모두 있다고 출력한다.
    else:
        print("필수 컬럼: 모두 존재")


# STEP7. articles.parquet 핵심 데이터 검사하는 함수
def validate_articles() -> dict[str, Any]:
    '''
    검사 항목
    -----------
    1. article_id가 null인 기사 수
    2. 같은 article_id 가진 중복 행 수
    3. title과 subtitle이 모두 비어 있는 기사 수
    4. category가 null인 기사 수
    5. cateogry_str이 null이거나 빈 문자열인 수 
    6. published_time이 null인 기사 수
    7. ner_clusters와 entity_groups의 리스트 길이가 다른 기사 수 
   
    처리 원칙
    -----------
    문제가 있는 데이터 개수만 확인하고, 실제 제외 처리는 이후 전처리 모듈에서 수행
    
    상태
    -----------
    FAIL : article_id가 null이거나 중복
    WARNING : 제외 가능한 기사 데이터 발견된 경우
    PASS : 핵심 문제 발견되지 않은 경우 
    '''

    # STEP 7-1. 기존 기본 검증 함수 먼저 실행
    basic_result = validate_parquet_file(
        dataset_name="articles",
        file_path=ARTICLES_PATH,
    )

    # STEP 7-2. 검증 실패 시 상세 검증도 중단 

    if basic_result["status"] != "PASS":
        return {
            # 상세 기사 검증도 실패로 기록한다.
            "status": "FAIL",

            # 실패 원인을 확인할 수 있도록
            # 기존 기본 검증 결과를 그대로 포함한다.
            "basic_validation": basic_result,
        }

    # STEP 7-3. 기사 검증에 필요한 컬럼만 읽음 
    articles = pl.read_parquet(
        ARTICLES_PATH,
        columns=[
            "article_id",
            "title",
            "subtitle",
            "category",
            "category_str",
            "published_time",
            "ner_clusters",
            "entity_groups",
        ],
    )

    # STEP 7-4. article_id null 검사 
    # article_id는 기사 임베딩, event_id, semantic id, 사용자 history, 클릭 target에 사용됨
    article_id_null_count = (
        articles.select(pl.col("article_id").is_null().sum().alias("count"))
        .item()
    )

    # STEP 7-5. article_id 중복 검사 
    # 하나의 article_id는 하나의 기사만 가리켜야함 
    # is_duplicated()는 중복에 포함된 모든 행 True로 반환
    # 예: [10,10,20] -> 2
    article_id_duplicate_row_count = (
        articles.
        select(
            (
                pl.col("article_id").is_not_null() & pl.col("article_id").is_duplicated()
            ).sum().alias("count")
        ).item()

    )

    # STEP 7-6. title과 subtitle이 모두 빈 기사 검사 
    # 기사 임베딩은 title+subtitle로 구성 (두 컬럼이 모두 비어 있는 경우는 문제됨)
    # fill_null("") : null을 빈 문자열로 바꿈
    # str.strip_chars() : 문자열 앞뒤의 공백 제거
    # eq(""): 공백 제거 후 빈 문자열인지 확인
    empty_title_and_subtitle_count = (
        articles.select(
            (
                pl.col("title").fill_null("").str.strip_chars().eq("")
                &
                pl.col("subtitle").fill_null("").str.strip_chars().eq("")
            ).sum().alias("count")
        ).item()
    )

    # STEP 7-7. 원본 category ID null 검사 
    category_null_count = (
        articles.select(pl.col("category").is_null().sum().alias("count")).item()
    )

    # STEP 7-8. category_str 빈값 검사 
    # 이후 model_category_id 만드는 기준 (tag prediction loss의 정답)
    # 예 : 빈 카테고리 <UNK> ID 0, "sport" -> model_category_id 1
    # null, 빈 문자열 "", 공백만 있는 문자열 " " -> 빈 카테고리로 판단
    # category_str이 비어있으면 이후 카테고리 매핑 단계에서 <UNK>=0으로 처리 (걍 model_category_id 0임)
    empty_category_str_count = (
        articles.select(
            pl.col("category_str").fill_null("").str.strip_chars().eq("").sum().alias("count")
        ).item()
    )

    # STEP 7-9. published_time null 검사 
    # 발행 시간이 없는 클러스터링 처리 불가
    published_time_null_count = (
        articles.select(
            pl.col("published_time").is_null().sum().alias("count")
        ).item()
    )

    # STEP 7-10. NER과 entity type 리스트 길이 검사 
    # 동일한 리스트 위치끼리 서로 대응
    # 예:
    # ner_clusters = ["OpenAI", "Seoul"]
    # entity_groups = ["ORG", "LOC"]

    # -> 각 행의 리스트 길이 계산해서 리스트 자체가 null이면 0으로 처리하고, 길이가 다른 기사 확인
    # 이후 NER 기반 event 구성 단계에서 제외 ㅇ
    ner_entity_length_mismatch_count = (
        articles.select(
            (
                pl.col("ner_clusters").list.len().fill_null(0)
                !=
                pl.col("entity_groups").list.len().fill_null(0)
            ).sum().alias("count")
        ).item()
    )

    # STEP 7-11. 구조적인 실패 조건 확인
    # article_id가 null이거나 중복되면 안됨 
    has_fatal_issue = (
        article_id_null_count > 0
        or article_id_duplicate_row_count > 0 
    )

    # STEP 7-12. 경고 조건 확인 
    # 1. title과 subtitle 모두 빈 기사: 임베딩 생성 대상에서 제외
    # 2. category 또는 category_str 누락: 원본 확인 또는 <UNK>=0 처리
    # 3. published_time 누락: 시간 기반 event clustering에서 제외
    # 4. NER 길이 불일치: NER 기반 event 구성에서 제외
    has_warning = (
        empty_title_and_subtitle_count > 0
        or category_null_count > 0
        or empty_category_str_count > 0
        or published_time_null_count > 0
        or ner_entity_length_mismatch_count > 0
    )

    # STEP 7-13. 최종 검증 상태 결정
    if has_fatal_issue: status = "FAIL"
    elif has_warning: status = "WARNING"
    else: status = "PASS"

    # STEP 7-14. 기사 검증 결과 반환
    # 각 검사 결과를 딕셔너리로 반환
    # 현재 함수에서 JSON파일 직접 저장하지 않고 
    # 이후 run_validation에서 다른 검증 결과와 합쳐 보고서로 저장
    return {
        # articles.parquet 파일 경로
        "file_path": str(ARTICLES_PATH),

        # 기사 데이터의 최종 검증 상태
        "status": status,

        # 전체 기사 행 수
        "row_count": articles.height,

        # article_id가 null인 기사 수
        "article_id_null_count": int(
            article_id_null_count
        ),

        # 중복 article_id에 포함된 전체 행 수
        "article_id_duplicate_row_count": int(
            article_id_duplicate_row_count
        ),

        # title과 subtitle이 모두 비어 있는 기사 수
        "empty_title_and_subtitle_count": int(
            empty_title_and_subtitle_count
        ),

        # 원본 category ID가 null인 기사 수
        "category_null_count": int(
            category_null_count
        ),

        # category_str이 null, 빈 문자열 또는 공백인 기사 수
        "empty_category_str_count": int(
            empty_category_str_count
        ),

        # published_time이 null인 기사 수
        "published_time_null_count": int(
            published_time_null_count
        ),

        # ner_clusters와 entity_groups의
        # 리스트 길이가 다른 기사 수
        "ner_entity_length_mismatch_count": int(
            ner_entity_length_mismatch_count
        ),
    }

# STEP 8. history.parquet 핵심 데이터 검증
def validate_history(
        split_name: str,
        file_path: Path,
)-> dict[str, Any]:
    """
    검사 대상 컬럼
    ----------------
    1. user_id
    2. article_id_fixed
    3. impression_time_fixed
    
    검사 항목 (-> X는 이후 해당하는 행은 제거한다는 의미)
    ----------------
    1. user_id가 null인 행 수 (-> X)
    2. user_id가 중복된 행 수  (-> X)
    3. history 리스트 자체가 null인 행 수 (-> X)
    4. 기사 ID 목록과 시간 목록의 길이가 다른 행 수 (-> X)
    5. 기사 ID 목록 내부에 null이 있는 행 수 (-> X)
    6. 시간 목록 내부에 null이 있는 행 수 (-> X)
    7. history가 시간 오름차순이 아닌 행 수 

    처리 원칙
    ----------------
    article_id_fixed와 impression_time_fixed를 같은 위치끼리 묶은 뒤
    두 리스트를 함께 시간순으로 정렬

    상태 기준 
    ----------------
    FAIL : 파일이 없거나 필수 컬럼 누락된 경우
    WARNING : 제외 또는 재정렬이 필요한 history 행 있는 경우
    PASS : 핵심 문제가 발견되지 않은 경우 
    """

    # STEP 8-1. 현재 검사할 데이터셋 이름 만들기
    # split_name이 train이면 train_history, validation이면 validation_history가 됨
    # 이 이름은 REQUIRED_COLUMNS의 key와 동일
    dataset_name = f"{split_name}_history"

    # STEP 8-2. 기존 기본 구조 검사 실행 
    basic_result = validate_parquet_file(
        dataset_name=dataset_name,
        file_path=file_path,
    )

    # STEP 8-3. 기본 구조 검사 실패 시 함수 종료 
    if basic_result["status"] != "PASS":
        return {
            # 현재 검사한 split
            "split": split_name,

            # 현재 검사한 파일 경로
            "file_path": str(file_path),

            # 상세 검사를 수행할 수 없으므로 FAIL
            "status": "FAIL",

            # 기본 검사에서 발생한 문제를 함께 반환한다.
            "basic_validation": basic_result,
        }

    # STEP 8-4. history 검증에 필요한 컬럼만 읽기 
    history = pl.read_parquet(
        file_path,
        columns=[
            "user_id",
            "article_id_fixed",
            "impression_time_fixed",
        ],
    )

    # STEP 8-5. user_id가 null인 행 수 계산
    # user_id가 null이면 해당 history를 어떤 사용자에게 연결? 모름 
    user_id_null_count = (
        history
        .select(
            pl.col("user_id")
            .is_null()
            .sum()
            .alias("count")
        )
        .item()
    )

    # STEP 8-6. 중복 user_id 찾기 
    # history.parquet은 기본적으로 사용자당 한 행이어야함.  
    duplicated_user_rows = (
        history
        .filter(
            pl.col("user_id").is_not_null()
            &
            pl.col("user_id").is_duplicated()
        )
    )
    # 중복 user_id에 포함된 전체 행 수 계산 
    user_id_duplicate_row_count = duplicated_user_rows.height # height : 행 몇개 ? 

    # 중복 user_id를 파이썬 set으로 만듦
    # 이후 각 history 행을 검사하며 현재 사용자가 중복 사용자에 해당하는지 chk
    duplicated_user_ids = set(
        duplicated_user_rows
        .get_column("user_id")
        .to_list()
    )

    # STEP 8-7. history 행별 검사 결과 카운터 생성 

    # article_id_fixed 또는 impression_time_fixed
    # 리스트 자체가 null인 행 수
    null_history_list_row_count = 0

    # article_id_fixed와 impression_time_fixed의
    # 리스트 길이가 다른 행 수
    history_length_mismatch_row_count = 0

    # article_id_fixed 리스트 내부에
    # null article_id가 들어 있는 행 수
    null_article_element_row_count = 0

    # impression_time_fixed 리스트 내부에
    # null 시간이 들어 있는 행 수
    null_time_element_row_count = 0

    # impression_time_fixed가 시간 오름차순이 아닌 행 수
    unsorted_history_row_count = 0

    # 이후 build_sequences.py에서
    # 제외해야 할 history 행 수
    # 한 행에 문제가 여러 개 있어도 한 번만 계산한다.
    exclusion_candidate_row_count = 0

    # 데이터는 사용할 수 있지만
    # article_id와 시간을 함께 정렬해야 하는 행 수
    reorder_candidate_row_count = 0

    # STEP 8-8. 사용자 history를 한 행씩 검사
    # iter_rows(named=True)를 사용하면 각 행을 다음 형태의 python 딕셔너리로 가져올 수 ㅇ
    #
    # {
    #     "user_id": 123,
    #     "article_id_fixed": [...],
    #     "impression_time_fixed": [...]
    # }   

    for row in history.iter_rows(named=True): # 딕셔너리로 가져옴 
        # 현재 행의 사용자 ID 가져옴
        user_id = row["user_id"]

        # 현재 사용자의 과거 기사 ID 리스트 가져옴
        article_ids = row["article_id_fixed"]

        # 각 과거 기사에 대응하는 시간 리스트 가져옴
        impression_times = row["impression_time_fixed"]

        # 현재 행을 이후 단계에서 제외해야 하는지 기록
        should_exclude_row = False # 문제 없다고 초기화 

        # STEP 8-9. 문제 확인
        # 1. user_id가 null이면 사용자와 history를 연결할 수 없음
        if user_id is None:
            should_exclude_row = True

        # 2. 같은 user_id가 여러 history에 존재하는 경우
        if user_id in duplicated_user_ids:
            should_exclude_row = True 

        # 3. 기사 리스트나 시간 리스트 자체가 null인 경우
        if article_ids is None or impression_times is None:
            null_history_list_row_count += 1
            should_exclude_row = True 

        # 리스트가 null이면 아래의 길이와 내부 값 검사 불가하기에 skip
        else:
            # 3-1. 두 리스트의 길이 일치 여부 확인
            if len(article_ids) != len(impression_times):
                history_length_mismatch_row_count += 1
                should_exclude_row = True

            # 3-2. 기사 ID 리스트 내부 null 검사
            # 두 리스트의 길이가 같을 때만 내부 값과 시간 순서 추가 검사
            else:
                has_null_article = any(
                    article_id is None 
                    for article_id in article_ids
                )

                # null article_id가 발견된 행 수 기록
                if has_null_article:
                    null_article_element_row_count += 1
                    should_exclude_row = True

                # 3-3. 시간 리스트 내부 null 검사 
                # impression_time_fixed 안에 null 시간이 하나라도 존재 ? 
                has_null_time = any(
                    impression_time is None
                    for impression_time in impression_times
                )

                # null 시간이 발견된 행 수 기록
                if has_null_time:
                    null_time_element_row_count += 1
                    should_exclude_row = True

                # STEP 8-10. history 시간 오름차순 검사
                # 기사 ID와 시간이 모두 정상인 경우
                if not has_null_article and not has_null_time:
                    # 현재 시간과 바로 다음 시간을 차례로 비교
                    # 같은 시각의 history는 허용
                    # 예:
                    # [10:00, 11:00, 11:00, 13:00] -> 정상
                    # [10:00, 13:00, 12:00]        -> 비정상

                    is_time_sorted = all(
                        impression_times[index]
                        <= impression_times[index + 1]
                        for index in range(
                            len(impression_times) - 1
                        )
                    )

                    # 특별 케이스 : 
                    # 시간이 오름차순이 아니라면 해당 행은 삭제하지 않고 재정렬 후보로 기록
                    if not is_time_sorted:
                        unsorted_history_row_count += 1
                        reorder_candidate_row_count += 1


        # 한 행에 여러 문제가 동시에 존재하더라도 
        # exclusion_candidate_row_count에는 한 번만 추가
        if should_exclude_row:
            exclusion_candidate_row_count += 1

        # STEP 8-11. 경고 존재 여부 확인
        # 제외 대상이나 재정렬 대상이 하나라도 있는 경우
        # history 데이터 후처리 필요 
        has_warning = (exclusion_candidate_row_count > 0 or reorder_candidate_row_count > 0)


        # STEP 8-12. 최종 검증 상태 결정 
        # 제외 또는 재정렬 후보가 있으면 WARNING이다.
        if has_warning:
            status = "WARNING"

        # 모든 핵심 문제가 0이면 PASS다.
        else:
            status = "PASS"

        # STEP 8-13. history 검증 결과 반환 
        return {
        # train 또는 validation 구분
        "split": split_name,

        # 검사한 history.parquet 경로
        "file_path": str(file_path),

        # history 데이터의 최종 검증 상태
        "status": status,

        # 전체 history 행 수
        "row_count": history.height,

        # user_id가 null인 행 수
        "user_id_null_count": int(
            user_id_null_count
        ),

        # 중복 user_id 그룹에 포함된 전체 행 수
        "user_id_duplicate_row_count": int(
            user_id_duplicate_row_count
        ),

        # 기사 또는 시간 리스트 자체가 null인 행 수
        "null_history_list_row_count": int(
            null_history_list_row_count
        ),

        # 기사 리스트와 시간 리스트 길이가 다른 행 수
        "history_length_mismatch_row_count": int(
            history_length_mismatch_row_count
        ),

        # 기사 ID 리스트 내부에 null이 있는 행 수
        "null_article_element_row_count": int(
            null_article_element_row_count
        ),

        # 시간 리스트 내부에 null이 있는 행 수
        "null_time_element_row_count": int(
            null_time_element_row_count
        ),

        # 시간 오름차순이 아닌 history 행 수
        "unsorted_history_row_count": int(
            unsorted_history_row_count
        ),

        # 이후 build_sequences.py에서 제외할 후보 행 수
        #
        # 여러 문제가 겹쳐도 행 단위로 한 번만 집계한다.
        "exclusion_candidate_row_count": int(
            exclusion_candidate_row_count
        ),

        # 제외하지 않고 기사와 시간을 함께 정렬할 후보 행 수
        "reorder_candidate_row_count": int(
            reorder_candidate_row_count
        ),
    }        


        
                







# 7. 원본 파일 5개 전체 검사 실행하는 중심 함수
def run_validation() -> dict[str, Any]:
    '''
    처리 순서
    -----------
    1. 출력 폴더 생성
    2. DATASET_PATHS의 파일 5개 반복
    3. validate_parquet_file() 호출
    4. 결과를 전체 딕셔너리에 저장
    5. 터미널에 파일별 결과 출력
    6. 전체 PASS 또는 FAIL 결정
    7. JSON 보고서 저장
    '''
   
    # 7-1. 검사 보고서 저장할 출력 폴더 생성
    # config.py에 정의한 함수
    create_output_directories()

    # 7-2. 파일별 결과를 저장할 빈 딕셔너리 생성 
    # 예 : {
    #        "articles" : {...}, "train_behaviors":{...}, 
    dataset_results: dict[str, Any] = {}

    # 7-3. 원본 파일 5개를 하나씩 반복 
    for dataset_name, file_path in DATASET_PATHS.items():
        # 7-4. 현재 parquet 파일 기본 검사 
        # 파일 하나를 validate_parquet_file()에 전달
        result = validate_parquet_file(
            dataset_name=dataset_name,
            file_path=file_path,
        )

        # 7-5. 현재 파일의 결과 저장
        # 현재 검사 결과를 전체 결과 딕셔너리에 저장
        dataset_results[dataset_name] = result

        # 7-6. 현재 파일의 결과를 터미널에 출력
        print_validation_result(
            dataset_name=dataset_name,
            result=result,
        )

    # 7-7. 전체 PASS 또는 FAIL 결정
    # 파일 5개가 모두 PASS인지 확인
    # all()은 모든 조건이 True일 때만 True를 반환한다.
    all_files_passed = all(
        result.get("status") == "PASS"
        for result in dataset_results.values()
    )

    # 모든 파일이 PASS라면 전체 결과도 PASS다.
    if all_files_passed:
        overall_status = "PASS"

    # 하나라도 FAIL이면 전체 결과는 FAIL이다.
    else:
        overall_status = "FAIL"

    # 7-8. 최종 JSON 보고서 구성 
    report: dict[str, Any] = {
        # 전체 기본 검사 상태
        "overall_status": overall_status,
        # 파일별 상세 기본 검사 결과
        "datasets": dataset_results,
    }

    # 7-9. JSON 보고서 저장 경로 생성 및 저장 
    # 실제 저장 경로 : data/output/reports/raw_basic_validation.json
    report_path = (
        REPORT_DIR / "raw_basic_validation.json"
    )
    save_report(
        report=report,
        output_path=report_path,
    )

    # 7-10. 전체 결과 터미널에 출력
    print()
    print("=" * 70)
    print(f"전체 결과: {overall_status}")
    print(f"상세 보고서: {report_path}")
    print("=" * 70)


    # 7-12. 전체 검사 결과 반환
    return report


# 8. 이 파일을 직접 실행했을 때 검증 시작
# 다음 명령으로 실행하면:
#
# python -m src.validate_data
#
# __name__ 값이 "__main__"이 되므로
# run_validation() 함수가 실행된다.
if __name__ == "__main__":
    run_validation()

    


    