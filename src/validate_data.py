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
        "category", 
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
        "article_ids_inview",
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


# 공통 데이터 검사 함수 

def count_null_rows(
    dataframe: pl.DataFrame,
    column_name: str,
) -> int:
    """
    지정한 컬럼의 값이 null인 행 수를 반환

    매개변수
    ----------
    dataframe:
        검사할 Polars DataFrame

    column_name:
        null 개수를 검사할 컬럼 이름

    반환값
    -------
    int:
        해당 컬럼이 null인 행의 개수

    주의
    ----
    리스트 컬럼에 사용하면 리스트 자체가 null인 행만 계산

    예:
        None        -> null로 계산
        []          -> null이 아님
        [101, None] -> 리스트 자체는 null이 아니므로 계산되지 않음
    """

    # 지정한 컬럼에서 null인 값을 True로 변환하고, True의 총개수를 합산
    null_count = (
        dataframe
        .select(
            pl.col(column_name)
            .is_null()
            .sum()
            .alias("null_count")
        )
        .item()
    )

    # Polars에서 반환한 숫자를 일반 Python int로 변환
    return int(null_count)

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
    article_id_null_count = count_null_rows(
        dataframe=articles,
        column_name="article_id",
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
    category_null_count = count_null_rows(
    dataframe=articles,
    column_name="category",
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
    published_time_null_count = count_null_rows(
    dataframe=articles,
    column_name="published_time",
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
    user_id_null_count = count_null_rows(
    dataframe=history,
    column_name="user_id",
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

# STEP 9. behaviors.parquet 핵심 데이터 검증

def validate_behaviors(
        split_name: str,
        file_path: Path,
)-> dict[str, Any]:
    """
    검사 대상 컬럼
    --------------
    1. impression_id
    2. user_id
    3. impression_time
    4. article_id
    5. article_ids_clicked

    검사 항목
    --------
    1. impression_id가 null인 행 수
    2. impression_id가 중복된 행 수
    3. user_id가 null인 행 수
    4. impression_time이 null인 행 수
    5. 현재 article_id가 null인 행 수
    6. 클릭 리스트 자체가 null인 행 수
    7. 클릭 리스트가 빈 리스트인 행 수
    8. 클릭 리스트 내부에 null이 있는 행 수
    9. 클릭 리스트 내부에 중복 ID가 있는 행 수
    10. stable dedup 후 클릭 기사가 0개인 행 수
    11. stable dedup 후 클릭 기사가 1개인 행 수
    12. stable dedup 후 클릭 기사가 2개 이상인 행 수
    13. 실제 baseline 학습에 사용할 수 있는 단일 클릭 행 수

    stable dedup : 클릭 리스트의 원래 순서 유지하며 중복만 제거하는 방식 

    이후 실제 제외되는 경우
    - impression_id가 null
    - impression_id가 중복
    - user_id가 null
    - impression_time이 null
    - 클릭 리스트가 null
    - 클릭 리스트가 비어 있음
    - 클릭 리스트 내부에 null이 있음
    - stable dedup 후 클릭 기사가 0개
    - stable dedup 후 클릭 기사가 2개 이상   

    상태 기준
    --------
    FAIL:
        파일이 없거나 필수 컬럼이 누락된 경우

    WARNING:
        제외 대상 또는 stable dedup이 필요한 행이 있는 경우

    PASS:
        핵심 문제가 발견되지 않은 경우        
    """

    # STEP 9-1. 현재 검사할 데이터셋 이름 만들기 
    # 이 이름은 REQUIRED_COLUMNS의 key와 동일
    dataset_name = f"{split_name}_behaviors" 

    # STEP 9-2. 기존 기본 구조 검사 실행   
    basic_result = validate_parquet_file(
        dataset_name=dataset_name,
        file_path=file_path,
    )

    # STEP 9-3. 기본 구조 검사 실패 시 함수 종료
    # 파일이 없거나 필수 컬럼이 누락된 경우에는
    # behavior 내부 데이터를 검사할 수 없다.
    if basic_result["status"] != "PASS":
        return {
            # train 또는 validation 구분
            "split": split_name,

            # 검사한 파일 경로
            "file_path": str(file_path),

            # 상세 검사를 수행할 수 없으므로 FAIL
            "status": "FAIL",

            # 기본 검사에서 발생한 문제를 함께 반환한다.
            "basic_validation": basic_result,
        }    

    # STEP 9-4. behavior 검증에 필요한 컬럼만 읽기 
    # TRAIN / validation 공통으로 필요한 기본 컬럼
    behavior_columns = [
        "impression_id",
        "user_id",
        "impression_time",
        "article_id",
        "article_ids_clicked",
    ]

    # validation에서는 ranking 평가 위해 실제 impression에 노출된 
    # candidate 기사 목록도 검사한다. 

    # article_ids_inview 예 : [100, 200,300,400] -> [300] : target
    # 300 = positive, 100, 200, 400 = negative candidate

    if split_name == "validation":
        behavior_columns.append("article_ids_inview")

    behaviors = pl.read_parquet(
        file_path, 
        columns = behavior_columns, 
    )   

    # STEP 9-5. impression_id가 null인 행 수 계산 
    # 각 노출 행동을 구분하는 ID (핵심)
    impression_id_null_count = count_null_rows(
    dataframe=behaviors,
    column_name="impression_id",
    )    

    # STEP 9-6. 중복 impression_id 찾기         
    # is_duplicated()는 중복 그룹에 속하는 모든 행을 찾음
    duplicated_impression_rows = (
        behaviors
        .filter(
            pl.col("impression_id").is_not_null()
            &
            pl.col("impression_id").is_duplicated()
        )
    )

    # 중복 impression_id 그룹에 포함된 전체 행 수를 계산
    impression_id_duplicate_row_count = (
        duplicated_impression_rows.height
    )

    # 중복된 impression_id들을 set으로 만든다.
    # 이후 행별 검사에서 현재 행의 impression_id가
    # 중복 ID인지 빠르게 확인하기 위해 사용
    duplicated_impression_ids = set(
        duplicated_impression_rows
        .get_column("impression_id")
        .to_list()
    )

    # STEP 9-7. user_id가 null인 행 수 계산 
    user_id_null_count = count_null_rows(
        dataframe=behaviors,
        column_name="user_id",
    )

    # STEP 9-8. impression_time이 null인 행 수 계산
    impression_time_null_count = count_null_rows(
        dataframe=behaviors,
        column_name="impression_time",
    )

    # STEP 9-9. 현재 article_id가 null인 행 수 계산
    # behavior가 발생했을 때 사용자가 보고 있던 현재 기사 ID
    current_article_id_null_count = count_null_rows(
        dataframe=behaviors,
        column_name="article_id",
    )

    # STEP 9-10. 클릭 데이터 검사 카운터 생성 
 
    # article_ids_clicked 리스트 자체가 null인 행 수
    clicked_list_null_count = 0

    # article_ids_clicked가 빈 리스트인 행 수
    clicked_list_empty_count = 0

    # 클릭 리스트 내부에 null article_id가 있는 행 수
    clicked_null_element_row_count = 0

    # 클릭 리스트 내부에 중복 article_id가 있는 행 수
    duplicate_clicked_row_count = 0

    # stable dedup 후 고유 클릭 기사가 0개인 행 수
    zero_click_after_dedup_row_count = 0

    # stable dedup 후 고유 클릭 기사가 1개인 행 수
    single_click_after_dedup_row_count = 0

    # stable dedup 후 고유 클릭 기사가 2개 이상인 행 수
    multi_click_after_dedup_row_count = 0

    # 모든 핵심 조건을 만족해서
    # 실제 baseline 단일 클릭 데이터로 사용할 수 있는 행 수
    usable_single_click_row_count = 0

    # 이후 build_sequences.py에서 제외해야 하는 행 수
    #
    # 한 행에 여러 문제가 있더라도 한 번만 계산한다.
    exclusion_candidate_row_count = 0

    # validation candidate(article_ids_inview) 검사용 카운터
    # train에선 모두 0으로 유지
    inview_list_null_count = 0
    inview_list_empty_count = 0
    inview_null_element_row_count = 0
    duplicate_inview_row_count = 0

    # 클릭 target이 실제 노출 candidate 안에 없는 이상한 행
    clicked_not_inview_row_count = 0

    # STEP 9-11. behavior를 한 행씩 검사 
    # iter_rows(named=True) 사용하여 각 behavior 행을 파이썬 딕셔너리 형태로 가져옴
    for row in behaviors.iter_rows(named=True):

        # 현재 behavior의 노출 ID 가져옴
        impression_id = row["impression_id"]

        # 현재 behavior의 사용자 ID 가져옴
        user_id = row["user_id"]

        # 현재 behavior가 발생한 시각을 가져옴
        impression_time = row["impression_time"]

        # 클릭한 기사 ID 리스트 가져옴
        clicked_ids = row["article_ids_clicked"]

        # validation인 경우
        # 실제 impression에 노출된 candidate 기사 목록 가져옴
        # train에서는 article_ids_inview 컬럼 읽지 않았기에 None으로
        if split_name == "validation":
            inview_ids = row["article_ids_inview"]
        else:
            inview_ids = None 

        # 아래에서 validation candidate stable dedup 결과를 저장하기 위한 빈 리스트
        # train에서도 변수가 항상 존재하도록 반복문 시작 시 빈 리스트로 초기화
        unique_inview_ids: list[Any] = []

        # candidate 목록이 정상이라 target 포함 여부 검사할 수 있는지 여부
        can_check_inview_membership = False 


        # 현재 behavior 행을 이후 단계에서 제외해야하는지 기록
        should_exclude_row = False # 문제 없다고 일단 가정

        # STEP 9-12. behavior 식별자와 시간 문제 확인
        # 1. impression_id가 null인 경우 
        if impression_id is None:
            should_exclude_row = True

        # 2. 같은 impression_id가 여러 행에 존재하는 경우
        if impression_id in duplicated_impression_ids:
            should_exclude_row = True 

        # 3. user_id가 null인 경우
        if user_id is None: 
            should_exclude_row = True 

        # 4. impression_time이 null인 경우
        if impression_time is None : 
            should_exclude_row = True 

        # + validation candidate(article_ids_inview) 검사 
        # train에서는 candidate ranking 평가하지않기에 validation에서만 검사
        # 예 : article_ids_inview [100, 200, 300, 400]
        # article_ids_clicked : [300]
        # 이후 build_sequences.py에선 
        # candidate_article_ids = [100, 200, 300, 400]
        # candidate_labels      = [0, 0, 1, 0] 형태로 만들 예정 

        if split_name == "validation":
            # 1. candidate list 자체가 null
            if inview_ids is None : 
                inview_list_null_count += 1
                should_exclude_row = True 

            # 2. candidate list가 빈 리스트
            elif len(inview_ids) == 0:
                inview_list_empty_count += 1
                should_exclude_row = True 
            else:
                # 3. candidate list 내부 null 검사
                # 예 : [100, 200, None, 400]
                has_null_inview_id = any(
                    article_id is None for article_id in inview_ids
                )

                if has_null_inview_id :
                    inview_null_element_row_count += 1
                    should_exclude_row = True 

                # null값을 제외한 candidate ID만 임시로 사용
                valid_inview_ids = [
                    article_id for article_id in inview_ids 
                    if article_id is not None 
                ]

                # 4. candidate stable dedup
                seen_inview_ids: set[Any] = set()

                for article_id in valid_inview_ids:
                    if article_id not in seen_inview_ids:
                        seen_inview_ids.add(article_id)
                        unique_inview_ids.append(article_id)

                # stable dedup 전후 길이가 다르면
                # 원본 candidate 리스트에 중복 ID가 있었다는 뜻
                if len(valid_inview_ids) != len(unique_inview_ids):
                    duplicate_inview_row_count += 1

                if not has_null_inview_id:
                    can_check_inview_membership = True 

        # 5. 클릭 리스트 자체가 null인지 ? 
        if clicked_ids is None : 
            clicked_list_null_count += 1
            should_exclude_row = True 

            # 클릭 리스트가 없으므로 아래 클릭 검사 진행 x
            exclusion_candidate_row_count += 1
            continue 

        # 6. 클릭 리스트가 비어 있는지 ? 
        # 빈 리스트에는 클릭 target이 없으므로 학습 샘플 생성 불가 
        if len(clicked_ids) == 0:
            clicked_list_empty_count += 1
            zero_click_after_dedup_row_count += 1
            should_exclude_row = True 

            # 클릭할 기사 없으므로 아래 검사 진행 x 
            exclusion_candidate_row_count += 1
            continue 

        # 7. 클릭 리스트 내부 null 검사 
        # article_ids_clicked 내부에 null article_id 존재 ? 
        has_null_clicked_id = any(
            article_id is None 
            for article_id in clicked_ids 
        )

        # null 클릭 ID가 있으면 해당 행을 제외 후보로 기록 
        if has_null_clicked_id:
            clicked_null_element_row_count += 1
            should_exclude_row = True 

        # STEP 9-16. null 클릭 ID를 제외한 임시 리스트 생성 
        valid_clicked_ids = [
            article_id for article_id in clicked_ids if article_id is not None 
        ]

        # STEP 9-17. stable dedup 수행 
        # 지금까지 확인한 article_id 저장
        # set[Any] : set 안에 아무 타입이나 들어갈 수 있는 집합임
        # set() : 위에껀 그냥 미리 말만, 실질적으론 일단 set()으로 초기화
        seen_clicked_ids: set[Any] = set() # 있는지 빨리 확인하려고 

        # 원래 순서 유지하며 중복 제거한 클릭 ID 저장
        unique_clicked_ids: list[Any] = []

        # 클릭 리스트를 앞에서부터 확인
        for article_id in valid_clicked_ids:
            # 아직 등장하지 않은 article_id만 결과에 추가
            if article_id not in seen_clicked_ids:
                seen_clicked_ids.add(article_id)
                unique_clicked_ids.append(article_id)

        # STEP 9-18. 클릭 리스트 내부 중복 여부 검사 
        # stable dedup 전후의 길이가 다르면 원복 클릭 리스트 내부에 중복 ID가 있었다는 것
        # stable dedup 후 고유 클릭 ID가 하나라면 단일 클릭 샘플로 사용 가능 
        if len(valid_clicked_ids) != len(unique_clicked_ids):
            duplicate_clicked_row_count += 1

        # STEP 9-19. stable dedup 후 클릭 수 분류
        # 유효한 고유 클릭 기사 없는 경우
        if len(unique_clicked_ids) == 0:
            zero_click_after_dedup_row_count += 1
            should_exclude_row = True 

        # 유효한 고유 클릭 기사 정확히 하나인 경우 (-> 학습에 사용ㅇ)
        elif len(unique_clicked_ids) == 1:
            single_click_after_dedup_row_count += 1

            # validation에선 clicked target이 실제 impression candidate 안에 있었는지 검사 
            if split_name == "validation" and can_check_inview_membership:
                target_article_id = unique_clicked_ids[0]
                if target_article_id not in unique_inview_ids:
                    clicked_not_inview_row_count += 1
                    should_exclude_row = True 

            # 다른 구조 문제가 없으면 실제 단일 클릭 샘플로 사용 가능
            if not should_exclude_row:
                usable_single_click_row_count += 1

        # 유효한 고유 클릭 기사가 둘 이상인 경우
        else:
            multi_click_after_dedup_row_count += 1

            # 클릭 간 정확한 순서를 알 수 없으므로 현재 baseline에서 일단 제외함
            should_exclude_row = True 

        # STEP 9-20. 제외 후보 행 수 계산
        # 한 behavior 행에 문제가 여럿 있어도 제외되는 실제 행은 하나니까 1번만 증가시킴
        if should_exclude_row:
            exclusion_candidate_row_count += 1

    # STEP 9-21. 경고 조건 확인
    # 제외 후보 행 또는 stable dedup이 필요한 행이 있다면 
    # 이후 후처리 필요 (build_sequences.py에서)
    has_warning = (
        exclusion_candidate_row_count > 0
        or 
        duplicate_clicked_row_count > 0
        # validation candidate에 중복이 있는 경우
        # # stable dedup 필요하므로  
        or duplicate_inview_row_count > 0
    )

    # STEP 9-22. 최종 상태 결정 
    # 제외 또는 stable dedup 대상 있는 경우
    if has_warning: status = "WARNING"
    # 모든 핵심 검사 결과에 문제 없다면 PASS
    else: status = "PASS"

    # STEP 9-23. behaviors 결과 반환 
    return {
        # train 또는 validation 구분
        "split": split_name,

        # 검사한 behaviors.parquet 경로
        "file_path": str(file_path),

        # behaviors 데이터의 최종 검증 상태
        "status": status,

        # 전체 behavior 행 수
        "row_count": behaviors.height,

        # impression_id가 null인 행 수
        "impression_id_null_count": int(
            impression_id_null_count
        ),

        # 중복 impression_id 그룹에 포함된 전체 행 수
        "impression_id_duplicate_row_count": int(
            impression_id_duplicate_row_count
        ),

        # user_id가 null인 행 수
        "user_id_null_count": int(
            user_id_null_count
        ),

        # impression_time이 null인 행 수
        "impression_time_null_count": int(
            impression_time_null_count
        ),

        # 현재 article_id가 null인 행 수
        #
        # 이 값은 허용되므로 제외 후보 수에는 포함하지 않는다.
        "current_article_id_null_count": int(
            current_article_id_null_count
        ),

        # 클릭 리스트 자체가 null인 행 수
        "clicked_list_null_count": int(
            clicked_list_null_count
        ),

        # 클릭 리스트가 빈 리스트인 행 수
        "clicked_list_empty_count": int(
            clicked_list_empty_count
        ),

        # 클릭 리스트 내부에 null article_id가 있는 행 수
        "clicked_null_element_row_count": int(
            clicked_null_element_row_count
        ),

        # 클릭 리스트 내부에 중복 article_id가 있는 행 수
        "duplicate_clicked_row_count": int(
            duplicate_clicked_row_count
        ),

        # stable dedup 후 고유 클릭 기사가 0개인 행 수
        "zero_click_after_dedup_row_count": int(
            zero_click_after_dedup_row_count
        ),

        # stable dedup 후 고유 클릭 기사가 1개인 행 수
        "single_click_after_dedup_row_count": int(
            single_click_after_dedup_row_count
        ),

        # stable dedup 후 고유 클릭 기사가 2개 이상인 행 수
        "multi_click_after_dedup_row_count": int(
            multi_click_after_dedup_row_count
        ),

        # 현재 정책상 실제 baseline 학습에 사용할 수 있는 행 수
        "usable_single_click_row_count": int(
            usable_single_click_row_count
        ),

        # 이후 build_sequences.py에서 제외할 후보 행 수
        #
        # 여러 문제가 동시에 있어도 행 단위로 한 번만 집계한다.
        "exclusion_candidate_row_count": int(
            exclusion_candidate_row_count
        ),
        # Validation candidate(article_ids_inview) 검사 결과
        # ========================================================

        # article_ids_inview 리스트 자체가 null인 행 수
        "inview_list_null_count": int(
            inview_list_null_count
        ),

        # article_ids_inview가 빈 리스트인 행 수
        "inview_list_empty_count": int(
            inview_list_empty_count
        ),

        # article_ids_inview 내부에 null article_id가 있는 행 수
        "inview_null_element_row_count": int(
            inview_null_element_row_count
        ),

        # candidate 리스트 내부에 중복 article_id가 있는 행 수
        "duplicate_inview_row_count": int(
            duplicate_inview_row_count
        ),

        # clicked target이 article_ids_inview 안에 없는 행 수
        "clicked_not_inview_row_count": int(
            clicked_not_inview_row_count
        ),
    }    


# STEP 10. 파일 간 참조 관계 검증
def validate_cross_file_references() -> dict[str, Any]:
    """
    articles, history, behaviors 파일 사이의 참조 관계 검사

    검사 관계
    --------
    1. history.article_id_fixed -> articles.article_id
    2. behaviors.article_id -> articles.article_id
    3. behaviors.article_ids_clicked -> articles.article_id
    4. behaviors.user_id -> 같은 split(train/val)의 history.user_id

    처리 원칙
    --------
    존재하지 않는 기사 ID 또는 history에 없는 user_id 발견하면
    해당 개수와 영향을 받는 행 수만 반환

    실제 제외 처리는 이후 build_train.py, build_sequences.py에서 함
    
    FAIL:
        파일이 없거나 필수 컬럼이 누락되어
        참조 관계를 검사할 수 없는 경우

    WARNING:
        존재하지 않는 기사 ID 또는 history에 없는 사용자가
        하나 이상 발견된 경우

    PASS:
        모든 참조값이 정상적으로 연결되는 경우    
    """

    # STEP 10-1. 참조 검사에 필요한 파일의 기본 구조 검사 
    # validate_parquet_file() 사용해 파일 존재 여부와 필수 컬럼 존재 여부 검사
    basic_results = {
        "articles": validate_parquet_file(
            dataset_name="articles",
            file_path=ARTICLES_PATH,
        ),
        "train_history": validate_parquet_file(
            dataset_name="train_history",
            file_path=TRAIN_HISTORY_PATH,
        ),
        "train_behaviors": validate_parquet_file(
            dataset_name="train_behaviors",
            file_path=TRAIN_BEHAVIORS_PATH,
        ),
        "validation_history": validate_parquet_file(
            dataset_name="validation_history",
            file_path=VALIDATION_HISTORY_PATH,
        ),
        "validation_behaviors": validate_parquet_file(
            dataset_name="validation_behaviors",
            file_path=VALIDATION_BEHAVIORS_PATH,
        ),
    }

    # STEP 10-2. 기본 구조 검사 실패 여부 chk
    # 5개 파일 중 하나라도 기본 검증 통과못하면 파일 간 참조 관계 정확히 검사 불가
    has_basic_failure = any(
        result["status"] != "PASS"
        for result in basic_results.values()
    )

    # 기본 구조에 문제가 있다면 참조 검사 미수행
    if has_basic_failure:
        return {
            "status": "FAIL",
            "basic_validation": basic_results,
        }
    
    # STEP 10-3. articles.parquet의 유효한 article_id 읽기 
    # 참조 검사의 기준은 articles.article_id
    # 이때 article_id가 null인 행을 고려해 drop_nulls()로 제외
    # unique() 통해 중복값도 제거 

    article_ids = (
        pl.read_parquet(
            ARTICLES_PATH,
            columns=["article_id"],
        ).drop_nulls("article_id")
        .unique()
    )

    # 전체 고유 article_id 개수를 기록
    article_id_count = article_ids.height 

    # STEP 10-4. split 하나를 검사하는 내부 함수 정의
    # validate_cross_file_references()가 실행되는 동안에만 사용됨
    def inspect_split_references(
        split_name:str,
        history_path: Path,
        behaviors_path: Path,
    ) -> dict[str, Any]:
        # STEP 10-4-1. history 핵심 컬럼 읽기
        # user_id : behaviors의 사용자가 history에 존재 ? 
        # article_id_fixed : 과거 기사 ID가 articles에 존재 ? 

        history = pl.read_parquet(
            history_path,
            columns=[
                "user_id",
                "article_id_fixed",
            ],
        )

        # STEP 10-4-2. behaviors 핵심 컬럼 읽기
        # impression_id : 문제가 발생한 behavior 행 구분 위해 
        # user_id : 같은 split의 history에 존재 ? 
        # article_id : 사용자가 행동 당시 보고 있던 현재 기사 
        # article_ids_clicked : 실제 클릭 target 기사 목록
        behavior_columns = [
            "impression_id",
            "user_id",
            "article_id",
            "article_ids_clicked",
        ]

        # Validation에서만 candidate 기사 참조 검사
        if split_name == "validation":
            behavior_columns.append(
                "article_ids_inview"
            )

        behaviors = pl.read_parquet(
            behaviors_path,
            columns=behavior_columns,
        )

        # STEP 10-4-3. history에 존재하는 사용자 목록 
        # 같은 split의 history에 존재하는 고유 user_id만 추출
        # null user_id는 제외
        history_user_ids = (
            history.select("user_id").drop_nulls("user_id").unique()
        )        

        # STEP 10-4-4. history에 없는 behavior 사용자 찾기 
        # history가 있어야 트랜스포머 학습시키는데 behaviors에는 있는 사용자가 history는 없다면?
        # 그 사용자는 시작 이력이 없는 것으로, 시퀀스를 만들 수 없거나 빈 이력으로 시작해야함 

        # behaviors의 user_id 중 history_user_ids에 존재하지 않는 행 찾기 
        # anti join : 왼쪽 DF엔 있지만 오른쪽 DF에 없는 행만 남기기
        # 이때 user_id가 아예 NULL인 경우는 여기서의 문제가 아니므로 제외
        missing_behaviors_users = (
            behaviors.filter(pl.col("user_id").is_not_null())
            .join(
                history_user_ids,
                on="user_id", # user_id가 같은 것끼리 짝지어 붙임
                how="anti",   # 이때, 그중에서도 짝을 못찾은 애들만 남김
            )
        )

        # history에 없는 user_id 때문에 영향 받는 behavior 전체 행 수 계산
        behavior_user_missing_history_row_count = (
            missing_behaviors_users.height 
        )

        # history에 없는 고유 사용자 수 계산
        # 같은 사용자가 여러 behavior 행 가질 수 있기 때문에
        behavior_user_missing_history_user_count = (
            missing_behaviors_users.get_column("user_id").n_unique()
        )

        # 위에는 몇개, 몇명인지만 알기 때문에 이번엔 그게 직접 누구인지 확인 
        # history에 없는 user_id 예시로 최대 10개 저장 (for 확인용)
        # get_column : Series [55,77,99]로 변환
        # to_list() : Series -> 순수 파이썬 리스트로 
        missing_history_user_examples = (
            missing_behaviors_users.select("user_id").unique().sort("user_id").head(10).get_column("user_id").to_list()
        )

        # STEP 10-4-5. history 기사 ID 리스트를 행 단위로 펼치기 
        # 각 사용자의 history 리스트를 풀어서 그 안에 있는 기사 ID 하나하나가
        # 진짜 존재하는 기사인지 확인하기 좋은 형태로 바꿈 
        # 현재 예시:
        # user_id = 10
        # article_id_fixed = [101,202, 303]
        # explode() 적용 후:
        # user_id | article_id
        # 10      | 101
        # 10      | 202
        # 10      | 303

        # with_row_index("_history_row")는 원래 어떤 history행에서
        # 나온 기사인지 확인하기 위한 임시 행 번호임 
        # 뒤에서 explode하면 원래 몇번째 사용자 것이었는지 헷갈리기에 출신 남기는 것과 동일
        # 검증할 때 history에서 user id 중복 없는 것 확인했으니 위의 문장처럼 말해도 ok
        history_article_references = (
            history.with_row_index("_history_row").select([
                "_history_row",
                "user_id",
                "article_id_fixed",
            ]).explode("article_id_fixed").rename(
                {"article_id_fixed":"article_id"}
            ).drop_nulls("article_id")
        )

        # STEP 10-4-6. articles에 없는 history 기사 찾기 
        # history에서 사용된 article_id 중 articles.parquet에 존재하지 않는 ID만 찾는다
        missing_history_article_references = (
            history_article_references.join(
                article_ids, 
                on="article_id",
                how="anti",
            )
        )

        # 존재하지 않는 article_id가 history 리스트에서
        # 총 몇 번 참조되었는지 계산
        # 같은 누락 ID가 여러 사용자의 history에 존재하면 여러번 계산되는 것임
        missing_history_article_reference_count = (
            missing_history_article_references.height 
        )

        # 존재하지 않는 기사 ID 때문에 영향 받는 history 원본 행 수(사용자 수) 계산
        missing_history_article_row_count = (
            missing_history_article_references
            .get_column("_history_row")
            .n_unique()
        )

        # articles에 존재하지 않는 고유 기사 ID수 계산
        missing_history_article_id_count = (
            missing_history_article_references.
            get_column("article_id").n_unique()
        )

        # 누락된 history article_id 예시를 최대 10개 저장 
        # 누락된 history article_id 예시를 최대 10개 저장한다.
        missing_history_article_id_examples = (
            missing_history_article_references
            .select("article_id")
            .unique()
            .sort("article_id")
            .head(10)
            .get_column("article_id")
            .to_list()
        )

        # STEP 10-4-7. behaviors의 현재 article_id 참조 만들기
        # 이번엔 현재 보고 있던 기사 하나가 실제 존재하는지 확인 
        # 이때 article_id는 리스트가 아닌 스칼라이기에 history처럼 explode 필요 x
        # behaviors.article_id는 사용자가 행동 당시 보고 있던 현재 기사 ID
        # null값 허용되므로 참조 검사에서 제외
        current_article_references = (
            behaviors.with_row_index("_behavior_row").
            select(
                [
                    "_behavior_row",
                    "impression_id",
                    "article_id",
                ]
            )
            .drop_nulls("article_id")
        )

        # STEP 10-4-8. articles에 없는 현재 article_id 찾기 
        # null이 아닌 현재 article_id 중 articles.parquet에 존재하지 않는 ID 찾는다
        missing_current_article_references = (
            current_article_references.join(
                article_ids,
                on="article_id",
                how="anti",
            )
        )
        # 현재 article_id가 articles에 존재하지 않는 behavior 행 수 계산
        missing_current_article_row_count = (
            missing_current_article_references.height 
        )

        # 존재하지 않는 고유한 현재 article_id 수 계산 
        missing_current_article_id_count = (
            missing_current_article_references.get_column("article_id").n_unique()
        )

        # 누락된 현재 article_id 예시 최대 10개 저장 
        missing_current_article_id_examples = (
            missing_current_article_references
            .select("article_id")
            .unique()
            .sort("article_id")
            .head(10)
            .get_column("article_id")
            .to_list()
        )

        # STEP 10-4-9. 클릭 기사 리스트를 행 단위로 펼치기 
        # 이번엔 클릭 기사들이 실존하는지 확인
        # 이 경우엔 explode 필요 
        clicked_article_references = (
            behaviors.with_row_index("_behavior_row").select(
                [
                    "_behavior_row",
                    "impression_id",
                    "article_ids_clicked",
                ]
            ).explode("article_ids_clicked").rename(
                {
                    "article_ids_clicked":"article_id",
                }
            ).drop_nulls("article_id")
        )

        # STEP 10-4-10. articles에 없는 클릭 기사 찾기
        # 실제 클릭된 article_id 중 articles.parquet에 존재하지 않는 ID만 찾는다
        missing_clicked_article_references = (
            clicked_article_references.join(
                article_ids,
                on="article_id",
                how="anti",
            )
        )

        # 존재하지 않는 클릭 기사 ID가 총 몇번 등장했는지 계산
        # 원본 리스트 내부에 같은 ID 반복되면 반복된 횟수도 포함됨
        missing_clicked_article_reference_count = (
            missing_clicked_article_references.height 
        )

        # 존재하지 않는 클릭 기사 때문에 영향 받는 behavior 원본 행 수 계산
        missing_clicked_behavior_row_count = (
            missing_clicked_article_references.get_column("_behavior_row").n_unique()
        )

        # 존재하지 않는 고유 클릭 article_id 수 계산 
        # 원본 행 수는 클릭 기사 수가 리스트로 구성된 경우도 고려한 것
        missing_clicked_article_id_count = (
            missing_clicked_article_references.get_column("article_id").n_unique()
        )

        # 누락된 클릭 article_id 예시를 최대 10개 저장
        missing_clicked_article_id_examples = (
            missing_clicked_article_references
            .select("article_id")
            .unique()
            .sort("article_id")
            .head(10)
            .get_column("article_id")
            .to_list()
        )


        # STEP 10-4-11.
        # Validation article_ids_inview -> articles.article_id
        # 참조 관계 검사
 

        # Train에서는 사용하지 않으므로 기본값 0 / 빈 리스트
        missing_inview_article_reference_count = 0
        missing_inview_behavior_row_count = 0
        missing_inview_article_id_count = 0
        missing_inview_article_id_examples = []

        if split_name == "validation":

            # ----------------------------------------------------
            # candidate 리스트를 article_id 단위로 펼친다.
            #
            # 예:
            #
            # impression_id = 10
            # article_ids_inview = [100,200,300]
            #
            # ↓ explode
            #
            # impression_id | article_id
            # 10            | 100
            # 10            | 200
            # 10            | 300
            # ----------------------------------------------------

            inview_article_references = (
                behaviors
                .with_row_index(
                    "_behavior_row"
                )
                .select(
                    [
                        "_behavior_row",
                        "impression_id",
                        "article_ids_inview",
                    ]
                )
                .explode(
                    "article_ids_inview"
                )
                .rename(
                    {
                        "article_ids_inview":
                        "article_id"
                    }
                )
                .drop_nulls(
                    "article_id"
                )
            )

            # ----------------------------------------------------
            # articles.parquet에 존재하지 않는
            # candidate article_id만 남긴다.
            # ----------------------------------------------------

            missing_inview_article_references = (
                inview_article_references
                .join(
                    article_ids,
                    on="article_id",
                    how="anti",
                )
            )

            # 누락 candidate가 총 몇 번 참조됐는지
            missing_inview_article_reference_count = (
                missing_inview_article_references.height
            )

            # 누락 candidate 때문에 영향을 받은 behavior 행 수
            missing_inview_behavior_row_count = (
                missing_inview_article_references
                .get_column(
                    "_behavior_row"
                )
                .n_unique()
            )

            # 실제 존재하지 않는 고유 candidate article_id 수
            missing_inview_article_id_count = (
                missing_inview_article_references
                .get_column(
                    "article_id"
                )
                .n_unique()
            )

            # 확인용 예시 최대 10개
            missing_inview_article_id_examples = (
                missing_inview_article_references
                .select(
                    "article_id"
                )
                .unique()
                .sort(
                    "article_id"
                )
                .head(
                    10
                )
                .get_column(
                    "article_id"
                )
                .to_list()
            )


        # STEP 10-4-11. 현재 split의 경고 여부 확인
        # 아래 항목 중 하나라도 1 이상이면 파일 간 참조 완전 일치는 아니라는 의미
        has_warning = (
            behavior_user_missing_history_row_count > 0
            or missing_history_article_reference_count > 0
            or missing_current_article_row_count > 0
            or missing_clicked_article_reference_count > 0
            or missing_inview_article_reference_count > 0
        )

        # 참조 누락이 하나라도 있으면 WARNING, 아니면 PASS
        if has_warning:
            status = "WARNING"
        else:
            status = "PASS"

        # STEP 10-4-12. 현재 split의 참조 검사 결과 반환

        return {
            # train 또는 validation 구분
            "split": split_name,

            # 현재 split의 최종 상태
            "status": status,

            # 전체 history 행 수
            "history_row_count": history.height,

            # 전체 behavior 행 수
            "behavior_row_count": behaviors.height,

            # history에 user_id가 없는 behavior 행 수
            "behavior_user_missing_history_row_count": int(
                behavior_user_missing_history_row_count
            ),

            # history에 존재하지 않는 고유 사용자 수
            "behavior_user_missing_history_user_count": int(
                behavior_user_missing_history_user_count
            ),

            # history에 존재하지 않는 user_id 예시
            "missing_history_user_examples": (
                missing_history_user_examples
            ),

            # articles에 없는 history 기사 ID의 전체 참조 횟수
            "missing_history_article_reference_count": int(
                missing_history_article_reference_count
            ),

            # 존재하지 않는 기사 ID가 포함된 history 행 수
            "missing_history_article_row_count": int(
                missing_history_article_row_count
            ),

            # articles에 존재하지 않는 고유 history 기사 ID 수
            "missing_history_article_id_count": int(
                missing_history_article_id_count
            ),

            # 누락된 history article_id 예시
            "missing_history_article_id_examples": (
                missing_history_article_id_examples
            ),

            # articles에 없는 현재 article_id를 가진 behavior 행 수
            "missing_current_article_row_count": int(
                missing_current_article_row_count
            ),

            # articles에 존재하지 않는 고유 현재 article_id 수
            "missing_current_article_id_count": int(
                missing_current_article_id_count
            ),

            # 누락된 현재 article_id 예시
            "missing_current_article_id_examples": (
                missing_current_article_id_examples
            ),

            # articles에 없는 클릭 기사 ID의 전체 참조 횟수
            #
            # 동일 ID가 원본 클릭 리스트에서 반복되면
            # 반복 횟수도 포함된다.
            "missing_clicked_article_reference_count": int(
                missing_clicked_article_reference_count
            ),

            # 존재하지 않는 클릭 기사가 포함된 behavior 행 수
            "missing_clicked_behavior_row_count": int(
                missing_clicked_behavior_row_count
            ),

            # articles에 존재하지 않는 고유 클릭 기사 ID 수
            "missing_clicked_article_id_count": int(
                missing_clicked_article_id_count
            ),

            # 누락된 클릭 article_id 예시
            "missing_clicked_article_id_examples": (
                missing_clicked_article_id_examples
            ),
            # Validation candidate 중 articles.parquet에
            # 존재하지 않는 참조의 총 횟수
            "missing_inview_article_reference_count": int(
                missing_inview_article_reference_count
            ),

            # 존재하지 않는 candidate가 포함된 behavior 행 수
            "missing_inview_behavior_row_count": int(
                missing_inview_behavior_row_count
            ),

            # articles.parquet에 존재하지 않는
            # 고유 candidate article_id 수
            "missing_inview_article_id_count": int(
                missing_inview_article_id_count
            ),

            # 누락 candidate article_id 예시
            "missing_inview_article_id_examples": (
                missing_inview_article_id_examples
            ),
        }

    # STEP 10-5. train 파일 간 참조 검사 
    train_result = inspect_split_references(
        split_name="train",
        history_path=TRAIN_HISTORY_PATH,
        behaviors_path=TRAIN_BEHAVIORS_PATH,
    )

    # STEP 10-6. validation 파일 간 참조 검사 
    validation_result = inspect_split_references(
        split_name="validation",
        history_path=VALIDATION_HISTORY_PATH,
        behaviors_path=VALIDATION_BEHAVIORS_PATH,
    )

    # STEP 10-7. 전체 참조 검사 (train 또는 validation 중 하나라도 WARNING이면 전체도 WARNING)
    if(
        train_result["status"]=="WARNING"
        or validation_result["status"]=="WARNING"
    ):
        status = "WARNING"
    else:
        status = "PASS"

    # STEP 10-8. 전체 참조 검사 결과 반환

    return {
        # 전체 파일 간 참조 검증 상태
        "status": status,

        # articles.parquet에 존재하는 고유 article_id 수
        "article_id_count": int(article_id_count),

        # train 참조 검증 결과
        "train": train_result,

        # validation 참조 검증 결과
        "validation": validation_result,
    }                                    






                


# STEP11. 원본 파일 5개 전체 검사 실행하는 중심 함수
def run_validation() -> dict[str, Any]:
    """
    2. train/history.parquet 상세 검증
    3. validation/history.parquet 상세 검증
    4. train/behaviors.parquet 상세 검증
    5. validation/behaviors.parquet 상세 검증
    6. 파일 간 사용자 및 기사 참조 검증

    전체 상태 기준
    --------------
    FAIL:
        하나 이상의 검증 결과가 FAIL인 경우

    WARNING:
        FAIL은 없지만 하나 이상의 검증 결과가 WARNING인 경우

    PASS:
        모든 검증 결과가 PASS인 경우

    처리 원칙
    --------
    이 함수에서는 원본 데이터를 수정하거나 삭제하지 않는다.

    모든 검사 결과를 하나의 딕셔너리로 합친 뒤
    JSON 검증 보고서로 저장한다.
    """
    
    # STEP 11-1. articles.parquet 상세 검증
    # 기사 ID, 텍스트, 카테고리, 발행 시각, NER 리스트 정합성 검사 
    articles_result = validate_articles()

    # STEP 11-2. train history.parquet 상세 검증
    # train 사용자의 과거 기사 목록과 시간 목록의 정합성 검사
    train_history_result = validate_history(
        split_name = "train",
        file_path = TRAIN_HISTORY_PATH,
    )

    # STEP 11-3. validation history.parquet 상세 검증
    # validation 사용자의 과거 기사 목록과 시간 목록의 정합성 검사
    validation_history_result = validate_history(
        split_name="validation",
        file_path=VALIDATION_HISTORY_PATH,
    )

    # STEP 11-4. train behavior.parquet 상세 검증
    # train behavior의 사용자, 시간, 현재 기사와 클릭 기사 목록 검사
    train_behaviors_result = validate_behaviors(
        split_name="train",
        file_path=TRAIN_BEHAVIORS_PATH,
    )

    # STEP 11-5. validation behaviors.parquet 상세 검증
    # validation behavior의 사용자, 시간, 현재 기사와 클릭 기사 목록 검사
    validation_behaviors_result = validate_behaviors(
        split_name="validation",
        file_path=VALIDATION_BEHAVIORS_PATH,
    )

    # STEP 11-6. 파일 간 참조 관계 검증
    # history와 behaviors에서 참조하는 사용자/기사 ID가 실제 다른 파일에 존재하는지?
    cross_file_result = validate_cross_file_references()

    # STEP 11-7. 모든 검증 결과 하나로 묶기 
    # 검증 이름을 key로 사용
    validation_results = {
        "articles": articles_result,
        "train_history": train_history_result,
        "validation_history": validation_history_result,
        "train_behaviors": train_behaviors_result,
        "validation_behaviors": validation_behaviors_result,
        "cross_file_reference": cross_file_result,
    }

    # STEP 11-8. 각 검증 결과의 상태만 추출
    # 예 : ["WARNING", "PASS", ...]
    validation_statuses = [
        result["status"]
        for result in validation_results.values()
    ]

    # STEP 11-9. 전체 검증 상태 결정 
    # 하나라도 FAIL이면 전체 검증을 FAIL로 처리한다.
    #
    # 파일 누락이나 필수 컬럼 누락처럼
    # 이후 단계를 진행할 수 없는 문제가 있다는 뜻이다.
    if "FAIL" in validation_statuses:
        overall_status = "FAIL"

    # FAIL은 없지만 WARNING이 하나라도 있으면
    # 전체 검증을 WARNING으로 처리
    # -> 데이터 파일은 사용할 수 있지 이후 전처리에서 제외 또는 정제가 필요하다는 뜻
    elif "WARNING" in validation_statuses:
        overall_status = "WARNING"

    # 모든 결과가 PASS라면 전체 검증도 PASS
    else:
        overall_status = "PASS"   

    # STEP 11-10. 최종 JSON 보고서 구성 
    # cf. 총 몇 개의 검증을 실행했는지 계산
    total_validation_count = len(validation_statuses)

    # results에는 각 파일별 상세 검증 결과 저장 
    report = {
        # 전체 원본 데이터 검증 상태
        "status": overall_status,

        # 전체 검증 결과 요약
        "summary": {
            # 실행한 전체 검증 개수
            "total_validation_count": int(
                total_validation_count
            )
        },
        # 각 데이터별 상세 검증 결과
        "results": validation_results, 
    }

    # STEP 11-11. 검증 결과 터미널에 출력 
    print()
    print("=" * 70)
    print("원본 데이터 전체 검증 결과")
    print("=" * 70)

    # 각 검증 결과를 기존 출력 함수로 출력
    for dataset_name, result in validation_results.items():
        print(
            f"{dataset_name}: "
            f"{result['status']}"
        )   

    # STEP 11-12. JSON 보고서 저장
    # 기존 save_report() 함수 사용해 전체 검증 결과를 JSON 파일로 저장
    report_path = (REPORT_DIR / "raw_validation_report.json")
    save_report(report=report, output_path=report_path,)
    
    # 저장된 보고서 경로를 터미널에 출력
    print(
        "검증 보고서 : "
        f"{report_path}"
    ) 
    print("=" * 70)

    # STEP 11-13. 전체 검증 결과 반환
    return report # 최종 보고서 딕셔너리 반환


# STEP12. 이 파일을 직접 실행했을 때 검증 시작
# 다음 명령으로 실행하면:
#
# python -m src.validate_data
#
# __name__ 값이 "__main__"이 되므로
# run_validation() 함수가 실행된다.
if __name__ == "__main__":
    run_validation()

    


    