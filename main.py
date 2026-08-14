from __future__ import annotations

import argparse
from pprint import pprint
from typing import Any

from src import config
from src.build_sequences import build_sequences
from src.build_validation import build_validation
from src.export_inputs import (
    export_rqvae_inputs,
    export_transformer_inputs,
)
from src.run_article_build import (
    main as run_article_build,
)
from src.validate_data import run_validation

# STEP 13. SID Project 전체 pipeline 
# main.py의 역할 
# 지금까지 작성한 각 모듈에는 실제 처리 로직이 들어있음
# 예 
# =====================
# validate_data.py -> Raw 데이터 정합성 검사
# run_article_build.py -> train 기사 전처리 전체 실행
# build_validation.py -> validation-only 기사 / event 전처리
# build_sequences.py -> transformer용 running history seq 생성
# export_inputs.py -> 모델별 전달 package 생성
# main.py는 위 로직을 다시 구현하는 게 아니라 어떤 순서로 어떤 모듈을 실행할지 책임지는 실행 스크립

# 각 모듈 = 실제 작업자, main.py = 작업 순서 지시자 
# 이 프로젝트 중간엔 별도의 RQ-VAE 모델 학습 단계가 끼어 있기에 한 번에 전체 실행 X

# 전체 흐름:
#
# Raw EB-NeRD
#     ↓
# 전처리
#     ↓
# RQ-VAE Train
#     ↓
# Validation Frozen Inference
#     ↓
# article_semantic_ids.parquet
#     ↓
# build_sequences.py
#     ↓
# Transformer
#
# 따라서 main.py는 안전하게 두 구간으로 분리한다.
#
#
# [1단계: preprocess]
# ------------------------------------------------------------
# Raw 데이터
#     ↓
# validate_data.py
#     ↓
# run_article_build.py
#     ↓
# build_validation.py
#     ↓
# export_rqvae_inputs()
#     ↓
# "RQ-VAE가 학습할 수 있는 package 준비 완료"
#
#
# [중간: RQ-VAE 팀/모듈]
# ------------------------------------------------------------
# rqvae_train_inputs
#     ↓
# RQ-VAE Train
#     ↓
# validation-only Frozen Inference
#     ↓
# article_semantic_ids.parquet 생성
#
#
# [2단계: post-rqvae]
# ------------------------------------------------------------
# article_semantic_ids.parquet
#     ↓
# build_sequences.py
#     ↓
# train_sequences.parquet
# validation_sequences.parquet
#     ↓
# export_transformer_inputs()
#     ↓
# "Transformer가 학습할 수 있는 package 준비 완료"
#
#
# 왜 preprocess 후 post-rqvae를 자동 실행하면 안 되는가?
# ------------------------------------------------------------
# 예전에 생성한 article_semantic_ids.parquet가 폴더에 남아 있을 수 있다.
#
# 그런데 오늘 Event clustering을 새로 만들었다면
# 옛 SID는 새 Event 결과와 맞지 않을 수 있다.
#
# 만약 단순히
#
#     if article_semantic_ids.parquet exists:
#         build_sequences()
#
# 같은 방식으로 자동 실행하면
#
#     새 전처리 결과 + 옛 SID
#
# 를 섞어버릴 위험이 있다.
#
# 그래서 RQ-VAE 완료 후 사용자가 명시적으로
#
#     python -m src.main post-rqvae
#
# 를 실행하도록 두 단계로 나눈다.
# ============================================================

# STEP 13-0. Stage 결과 출력 헬퍼 
def _print_stage_result(stage_name:str, result:Any)-> None : 
    """
    각 큰 단계의 실행 결과를 터미널에서 구분해서 출력
    데이터 처리 로직은 전혀 없고 로그 가독성 위한 출력 헬퍼임
    예 : 
    ==========================
    STEP 10 - Validation Build
    ==========================
    {"status" : "SUCCESS", ...}
    
    """
    print(); print("="*80); print(stage_name); print("="*80); pprint(result)

# STEP 13-1. PREPROCESS Pipeline
# python -m src.main preprocess
# 목적 : 
# RQ-VAE 학습을 시작할 수 있는 상태까지 전처리 관련 작업 순서대로 한 번에 실행

# 완료 지점 : 
# rqvae_train_inputs/
# rqvae_validation_inputs/
# package가 만들어진 상태 

def run_preprocess_pipeline() -> dict[str, Any]:
    """
    Raw EB-NeRD부터 RQ-VAE 전달 패키지 생성까지 실행

    실행 순서
    -------------------
    STEP1. Raw Validation 
        raw parquet 자체가 사용가능한지 검사
    STEP2-9. Train Article Build
        valid article 생성
        train-used article 수집
        category mapping
        e5 embedding
        train event clustering
        article_master 생성

    STEP10. Validation build
        validation-used/validation-only 분리
        validation dynamic event assignment
        validation_article_master 생성

    STEP12. RQ-VAE INPUT EXPORT
         위 산출물들을 RQ-VAE Train/frozen infer용 전달 패키지로 묶음 
    
    반환값 : 각 단계 결과를 하나의 dict로 묶어 반환
    
    """

    # STEP 13-1-0. 공통 output 폴더 생성
    # reports, model_inputs 폴더가 없으면 미리 만든다
    config.create_output_directories()

    # STEP 13-1-1. Raw Data validation 
    # validate_data.py에서 작성한 전체 원본 검증 함수 실행
    # 확인 예 : 
    # - articles article_id null/duplicate
    # - history list 구조 
    # - behavior 필수 컬럼
    # - clicked/candidate 구조 
    # - cross-file re
    validation_result = run_validation()

    _print_stage_result("STEP 1 - Raw Data Validation", validation_result)

    # run_validation() 결과의 최상위 status확인
    validation_status = str(validation_result.get("status", "UNKNOWN")).upper()

    # FAIL이면 원본 데이터 자체에 심각한 문제가 있단 뜻이기에 뒷 단계 진행 X
    # WARNING은 후처리 정책으로 다룰 수 있는 문제이기에 진행 가능하도록
    if validation_status == "FAIL":
        raise RuntimeError(
            "Raw validation이 FAIL이므로 "
            "Article Build를 실행하지 않습니다."
        )


    # STEP 13-1-2. Train article build
    # run_article_build.py의 main() 호출
    # 이 함수 내부에서 이미 다음 단계들을 순서대로 실행한다.
    #
    # build_valid_articles()
    #     ↓
    # collect_train_used_article_ids()
    #     ↓
    # build_category_mapping()
    #     ↓
    # apply_category_mapping_to_articles()
    #     ↓
    # build_article_embedding_input()
    #     ↓
    # generate_article_embeddings()
    #     ↓
    # build_article_events()
    #     ↓
    # build_train_article_master()
    #
    # 따라서 main.py에서 이 내부 로직을 다시 작성하지 않는다.

    run_article_build()
    # run_article_build.main() 자체는 각 STEP 결과를 내부에서 출력하고
    # 전체 결과 dict를 반환하지 않으므로
    # main.py에서는 완료 여부만 간단히 기록한다.
    article_build_result = {
        "status": "SUCCESS",
        "message": (
            "run_article_build.main() 완료"
        ),
    }

    # STEP 13-1-3. Validation build
    # train event를 다시 학습/클러스터링하지 않고
    # validation-only 기사를 시간순으로 기존 event에 배정 or 새로운 validation-origin event 생성

    # 주요 결과 : 
    # validation_used_article_ids.parquet
    # validation_only_article_ids.parquet
    # validation_article_events.parquet
    # event_master_with_validation.parquet
    # validation_article_master.parquet
    validation_build_result = (
        build_validation()
    )

    _print_stage_result(
        "STEP 10 - Validation Build",
        validation_build_result,
    )


    # STEP 13-1-4. RQ-VAE INPUT EXPORT
    # model_inputs에 흩어져있는 파일을 RQ-VAE 파트에서 바로 사용할 수 있는 패키지로 묶기 
    # 결과 예:
    # data/output/exports/
    #   rqvae_train_inputs/
    #   rqvae_validation_inputs/

    rqvae_export_result = export_rqvae_inputs()

    _print_stage_result(
        "STEP 12 - RQ-VAE Input Export", rqvae_export_result,
    )

    # STEP 13-1-5. PREPROCESS 전체 결과 반환
    # 여기까지 온 건 RQ-VAE 입력 패키지 만들 준비 끝났다는 것
    # 다음 단계는 RQ-VAE Train
    return {
        "status": "SUCCESS",
        "pipeline_stage": (
            "PREPROCESS_COMPLETE"
        ),
        "raw_validation": validation_result,
        "article_build": article_build_result,
        "validation_build": validation_build_result,
        "rqvae_export": rqvae_export_result,
        "next_action": (
            "RQ-VAE Train + Validation Frozen Inference를 실행한 뒤 "
            "article_semantic_ids.parquet를 생성하고 "
            "python -m src.main post-rqvae 를 실행하세요."
        ),
    }

# STEP 13-2. POST-RQVAE 파이프라인
# python -m src.main post-rqvae
# 실행 시점:
# RQ-VAE Train 및 Validation frozen inference가 끝나서 article_semantic_ids.parquet이 생성된 뒤.
# 목적:
# 기사 단위 SID를 사용자 행동 history/behavior와 결합하여
# transformer 학습/평가용 seq를 만들고, 최종 트랜스포머 전달 package까지 생성

def run_post_rqvae_pipeline() -> dict[str, Any]:
    """
    RQ-VAE 결과 받은 뒤 트랜스포머 입력 생성까지 실행

    실행 순서
    -------------------
    1. article_semantic_ids.parquet 존재 확인
    2. build_sequences()
    3. export_transformer_inputs()
    """

    # STEP 13-2-0. output 폴더 보장
    config.create_output_directories()

    # STEP 13-2-1. RQ-VAE 최종 SID 파일 존재확인
    # build_sequences.py는 반드시 article_id -> (c1,c2,c3) lookup 필요
    # 따라서 SID 파일이 없으면 sequence 생성 자체를 시작하지 않는다.
    if not config.ARTICLE_SEMANTIC_IDS_PATH.exists():
        raise FileNotFoundError(
            "RQ-VAE 최종 결과가 없습니다. "
            "Train + Validation Frozen Inference를 먼저 완료해야 합니다. "
            f"경로={config.ARTICLE_SEMANTIC_IDS_PATH}"
        )


    # STEP 13-2-2. Transformer seq build
    # build_sequences.py가 하는 일 : 
    # article_semantic_ids.parquet 
    # + history.parquet
    # + behaviors.parquet
    #       ↓
    # user별 running history 구성
    #       ↓
    # train_sequences.parquet
    # validation_sequences.parquet
    #
    # Train / Validation running history는 서로 독립이다.

    sequence_result = build_sequences()
    _print_stage_result(
        "STEP 11 - Transformer Sequence Build", 
        sequence_result,
    )


    # STEP 13-2-3. Transformer input export
    # build_sequences 결과와 SID 파일을
    # Transformer 파트에 전달하기 좋은 별도 패키지로 묶음 

    # 결과 예:
    # data/output/exports/transformer_inputs/
    #   article_semantic_ids.parquet
    #   train_sequences.parquet
    #   validation_sequences.parquet
    #   category_mapping.parquet
    #   manifest.json

    transformer_export_result = export_transformer_inputs()

    _print_stage_result(
        "STEP 12 - Transformer Input Export",
        transformer_export_result,
    )

    # STEP 13-2-4. POST-RQVAE 전체 결과 반환
    return {
        "status": "SUCCESS",
        "pipeline_stage": (
            "TRANSFORMER_INPUTS_COMPLETE"
        ),
        "sequence_build": sequence_result,
        "transformer_export": (
            transformer_export_result
        ),
    }


# STEP 13-3. CLI 
# 터미널에서 main.py 실행시 어떤 구간 실행할지 stage 인자로 선택
# 사용 예
# ------------------------------------------------------------
# 전처리 ~ RQ-VAE package 생성:
#
#     python -m src.main preprocess
#
# RQ-VAE 완료 후 ~ Transformer package 생성:
#
#     python -m src.main post-rqvae
#
# stage를 생략하면 preprocess가 기본값이다.
#
#     python -m src.main == python -m src.main preprocess

def main() -> None : 
    """
    터미널 인자 읽고 알맞은 파이프라인 stage 호출하는 최상위 함수

    """

    # STEP 13-3-1. Argument parser 생성
    parser = argparse.ArgumentParser(
        description=(
            "EB-NeRD → RQ-VAE → Transformer 데이터 파이프라인"
        )
    )

    # STEP 13-3-2. stage 인자 정의
    # choices를 두어 오타/지원하지 않는 stage 막음
    # 가능값 : preprocess, post-rqvae
    # nargs ?는 stage (preprocess, post-rqvae) 인자 안써도 ok

    parser.add_argument("stage", nargs="?", default="preprocess",
                        choices=["preprocess", "post-rqvae"],
                        help=("preprocess: RQ-VAE 입력 생성까지 / "
                              "post-rqvae: SID 수신 후 Transformer 입력 생성"))

    # 사용자가 입력한 CLI argument 실제 해석
    args = parser.parse_args()

    print()
    print("#" * 80)
    print("SID_Project Pipeline 시작")
    print("#" * 80)

    
    # STEP 13-3-3. 선택한 stage 실행
    
    if args.stage == "preprocess":
        result = (
            run_preprocess_pipeline()
        )
    else:
        result = (
            run_post_rqvae_pipeline()
        )

    
    # STEP 13-3-4. 전체 결과 출력
    
    print()
    print("#" * 80)
    print("SID_Project Pipeline 완료")
    print("#" * 80)
    pprint(result)
    
# STEP 13-4. main.py 직접 실행
# ============================================================
#
# python -m src.main preprocess
# python -m src.main post-rqvae
#
# 로 실행했을 때만 main() 호출.
# 다른 Python 파일에서 import할 때는 자동 실행하지 않는다.
# ============================================================

if __name__ == "__main__":
    main()