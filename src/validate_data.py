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

    


    