from __future__ import annotations

import json
import shutil
from pathlib import Path
from pprint import pprint
from typing import Any

import numpy as np
import polars as pl

from src import config

# ============================================================
# STEP 12. Model Input Export
# ============================================================
#
# 이 파일의 역할
# ------------------------------------------------------------
# 지금까지의 전처리 과정에서는 모든 산출물을
#
#     data/output/model_inputs/
#
# 아래에 계속 저장해 왔다.
#
# 예:
#
#     article_master.parquet
#     article_embeddings.npy
#     article_events.parquet
#     validation_article_master.parquet
#     article_semantic_ids.parquet
#     train_sequences.parquet
#     validation_sequences.parquet
#
# 하지만 실제 모델 담당자가 RQ-VAE나 Transformer를 실행할 때는
# model_inputs 폴더 전체가 필요한 것이 아니다.
#
# 예를 들어 RQ-VAE Train은 주로
#
#     article_master.parquet
#     article_embeddings.npy
#
# 를 사용하고,
# Transformer는
#
#     train_sequences.parquet
#     validation_sequences.parquet
#
# 를 사용한다.
#
# 따라서 이 파일은
#
#     "현재까지 만들어 둔 model_inputs 산출물을 검사한 뒤,
#      모델별로 필요한 파일만 별도의 전달 폴더에 복사한다."
#
# 는 역할을 한다.
#
# ★ 매우 중요
# ------------------------------------------------------------
# export_inputs.py는 새로운 학습 데이터를 계산하는 파일이 아니다.
#
# 즉 여기서는:
#
#     Event clustering X
#     E5 embedding 생성 X
#     RQ-VAE 학습 X
#     Semantic ID 생성 X
#     Sequence 생성 X
#
# 위 작업들을 하지 않는다.
#
# 이미 생성된 결과를
#
#     1. 존재 여부 검사
#     2. 최소 정합성 검사
#     3. 모델별 폴더에 복사
#     4. manifest.json 기록
#
# 하는 "최종 전달 패키징 단계"다.
#
#
# 전체 흐름에서 위치
# ------------------------------------------------------------
#
# [전처리]
#   build_train.py
#   build_article_master.py
#   build_validation.py
#          ↓
# data/output/model_inputs/
#          ↓
#     export_inputs.py
#          ↓
# data/output/exports/
#   ├─ rqvae_train_inputs/
#   ├─ rqvae_validation_inputs/
#   └─ transformer_inputs/
#
#
# 왜 그냥 model_inputs 폴더를 통째로 전달하지 않는가?
# ------------------------------------------------------------
# 1. 각 모델에 필요 없는 내부 전처리 파일까지 같이 전달되는 것을 방지
# 2. 모델 담당자가 어떤 파일을 사용해야 하는지 명확하게 함
# 3. 전달 직전에 데이터 정합성을 한 번 더 검사할 수 있음
# 4. manifest.json으로 전달 당시 파일 구성을 기록할 수 있음
#
#
# 원칙
# ------------------------------------------------------------
# 1. data/output/model_inputs 안의 원본 산출물은 수정하지 않는다.
# 2. export 폴더에 복사본만 만든다.
# 3. export 전에 최소 정합성 검사를 한다.
# 4. RQ-VAE Train / Validation Frozen Inference / Transformer를 분리한다.
# ============================================================


# ============================================================
# STEP 12-0. Export 경로 정의
# ============================================================
#
# 기존 model_inputs와 별도로 exports 폴더를 만든다.
#
# 결과 예:
#
# data/output/exports/
# ├─ rqvae_train_inputs/
# │   ├─ article_master.parquet
# │   ├─ article_embeddings.npy
# │   └─ manifest.json
# │
# ├─ rqvae_validation_inputs/
# │   ├─ validation_article_master.parquet
# │   ├─ article_embeddings.npy
# │   └─ manifest.json
# │
# └─ transformer_inputs/
#     ├─ article_semantic_ids.parquet
#     ├─ train_sequences.parquet
#     ├─ validation_sequences.parquet
#     └─ manifest.json
# ============================================================


EXPORT_ROOT_DIR = config.OUTPUT_DIR / "exports"
RQ_VAE_TRAIN_EXPORT_DIR = EXPORT_ROOT_DIR / "rqvae_train_inputs"
RQ_VAE_VALIDATION_EXPORT_DIR = EXPORT_ROOT_DIR / "rqvae_validation_inputs"
TRANSFORMER_EXPORT_DIR = EXPORT_ROOT_DIR / "transformer_inputs"

# STEP 12-1. 공통 헬퍼 함수
def _require_paths(paths: dict[str,Path])-> None:
    # 필수 파일이 하나라도 없으면 export 시작 x

    missing = {name: path for name, path in paths.items() if not path.exists()}

    if missing: 
        message = "\n".join(
        f"- {name}: {path}" for name, path in missing.items()
    ) 
        raise FileNotFoundError(
        "필수 입력 파일이 없습니다.\n" + message
        )

def _reset_directory(directory: Path)-> None:
    """
    export dir 안의 과거 파일이 새 package와 섞이지 않도록 초기화

    주의 : data/output/exports 아래의 해당 package 폴더만 삭제
    원본 model_inputs는 절대 삭제x 
    """
    if directory.exists():
        shutil.rmtree(directory)

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


def _copy_file(
        source: Path, destination_dir: Path,
)-> Path: 
    """
    필수 파일 하나를 export 폴더로 복사 
    예:
    --------------------------------------------------------
    source
      = data/output/model_inputs/article_master.parquet

    destination_dir
      = data/output/exports/rqvae_train_inputs

    결과
      = data/output/exports/rqvae_train_inputs/article_master.parquet

    shutil.copy2()를 사용하여 파일 내용뿐 아니라
    가능한 범위에서 수정 시간 등의 메타데이터도 함께 복사한다.
    """

    destination = (
        destination_dir / source.name
    )

    shutil.copy2(
        source,
        destination,
    )

    return destination

def _copy_optional_file(source: Path, destination_dir: Path)-> Path | None:
    """
    존재하면 복사하고, 없으면 그냥 넘어가는 보조 파일용 함수.

    필수 파일과 차이:
    --------------------------------------------------------
    필수 파일
      -> 없으면 모델 실행 자체가 불가능하므로 오류

    optional 파일
      -> 분석/해석/디버깅에는 유용하지만
         핵심 모델 입력이 아니므로 없다고 export 전체를 실패시키지는 않음

    반환:
    --------------------------------------------------------
    복사했으면 destination Path
    없으면 None
    
    """
    if not source.exists():
        return None

    return _copy_file(
        source,
        destination_dir,
    )


def _write_manifest(
        export_dir: Path, package_name:str, validation_result: dict[str, Any], exported_files: list[Path],
)-> Path:
    """
    export package에 manifest.json을 생성한다.

    manifest란?
    --------------------------------------------------------
    "이 package에 어떤 파일이 들어 있고,
     export 직전 검증 결과가 어떠했는지"
    기록하는 목록 파일이다.

    예:
    --------------------------------------------------------
    {
      "package_name": "transformer_inputs",
      "files": [
        {
          "name": "train_sequences.parquet",
          "size_bytes": 123456
        }
      ],
      "validation": {
        "status": "PASS",
        "train_sequence_count": 232887
      }
    }

    모델 학습 코드가 manifest를 반드시 읽어야 하는 것은 아니다.
    전달/재현/확인용 메타데이터다.
    """

    manifest_path = (
        export_dir / "manifest.json"
    )

    manifest = {
        "package_name": package_name,
        "export_dir": str(export_dir),

        # 실제 export된 파일들의 이름 / 경로 / 크기 기록
        "files": [
            {
                "name": path.name,
                "path": str(path),
                "size_bytes": int(
                    path.stat().st_size
                ),
            }
            for path in exported_files
        ],

        # export 직전에 수행한 정합성 검사 결과
        "validation": validation_result,
    }

    with manifest_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            ensure_ascii=False,
            indent=2,
        )

    return manifest_path

def _validate_required_columns(
        dataframe: pl.DataFrame, required_columns: set[str], dataset_name: str,
)-> None:
    """
    parquet에 모델이 기대하는 필수 컬럼이 모두 있는지 확인한다.

    예:
    --------------------------------------------------------
    article_master.parquet에 반드시

        article_id
        embedding_row
        model_category_id
        event_id

    가 있어야 하는데 event_id가 빠져 있다면
    RQ-VAE가 정상적으로 Event membership을 알 수 없다.

    그래서 export 전에 즉시 차단한다.
    """


    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            f"{dataset_name}에 필요한 컬럼이 없습니다: "
            + ", ".join(
                sorted(missing_columns)
            )
        )


def _validate_unique_non_null_id(
    dataframe: pl.DataFrame,
    id_column: str,
    dataset_name: str,
)-> None: 
    """
    ID 컬럼이 null 없이 unique한지 검사한다.

    왜 필요한가?
    --------------------------------------------------------
    예를 들어 article_semantic_ids.parquet에서

        article_id=100 -> (1,2,3)
        article_id=100 -> (4,5,6)

    처럼 같은 article_id가 두 번 나오면
    어느 SID가 맞는지 결정할 수 없다.

    따라서 article_id / impression_id / event_id 같은
    식별자 컬럼은 export 전에 unique + non-null을 확인한다.
    """


    null_count = (
        dataframe
        .get_column(id_column)
        .null_count()
    )

    if null_count != 0:
        raise ValueError(
            f"{dataset_name}.{id_column}에 null이 존재합니다. "
            f"null_count={null_count}"
        )

    duplicate_count = (
        dataframe
        .select(
            pl.col(id_column)
            .is_duplicated()
            .sum()
        )
        .item()
    )

    if duplicate_count != 0:
        raise ValueError(
            f"{dataset_name}.{id_column}에 중복이 존재합니다. "
            f"duplicate_row_count={duplicate_count}"
        )


# STEP 12-2. RQ-VAE Train 입력 검증
# article_master.parquet
#
#   article_id
#   embedding_row
#   model_category_id
#   event_id
#          │
#          │ embedding_row
#          ▼
# article_embeddings.npy
#          │
#          ▼
#        x(a)
#          │
#          ▼
#       Encoder
#          │
#          ▼
#        h(a)
#
# 그리고 같은 event_id를 가진 Train 기사들의 h(a)를 모아
# RQ-VAE 내부에서 z(E)를 계산한다.
#
# 즉 여기서는 "article_master와 article_embeddings가 서로
# 정상적으로 연결 가능한가?"를 검사하는 것이 핵심이다.


def _validate_rqvae_train_inputs() -> dict[str, Any]:
    """
    RQ-VAE train package 생성 전에 필요한 최소 정합성 검사
    """

    # STEP 12-2-1. Train article master 읽기 
    article_master = pl.read_parquet(
        config.ARTICLE_MASTER_PATH
    )

    # STEP 12-2-2. RQ-VAE가 기대하는 컬럼 존재 확인 
    _validate_required_columns(
        article_master,
        {
            "article_id",
            "embedding_row",
            "model_category_id",
            "event_id",
            "published_time",
            "model_text",
        },
        "article_master.parquet",
    )

    # article_id 하나당 정확히 한 행이어야 한다.
    _validate_unique_non_null_id(
        article_master,
        "article_id",
        "article_master.parquet",
    )

    # STEP 12-2-3. RQ-VAE 핵심 metadata null 검사
    # embedding_row가 null이면 article_embeddings.npy에서 x(a)를 못 찾음
    # model_category_id가 null이면 c1 direct mapping 불가
    # event_id가 null이면 어떤 Event에 속하는지 알 수 없음
    for column_name in [
        "embedding_row",
        "model_category_id",
        "event_id",
    ]:
        null_count = (
            article_master
            .get_column(column_name)
            .null_count()
        )

        if null_count != 0:
            raise ValueError(
                f"article_master.{column_name}에 null이 존재합니다. "
                f"null_count={null_count}"
            )

    # STEP 12-2-4. E5 article embedding 배열 읽기
    # mmap_mode="r"
    # -> 큰 npy를 전부 메모리에 복사하지 않고 읽기 전용 mapping으로 연다.
    article_embeddings = np.load(
        config.ARTICLE_EMBEDDINGS_PATH,
        mmap_mode="r",
    )

    # 기사 임베딩은 반드시
    # [기사 수, embedding dimension]
    # 형태의 2차원 행렬이어야 한다.
    if article_embeddings.ndim != 2:
        raise ValueError(
            "article_embeddings.npy는 2차원 배열이어야 합니다. "
            f"shape={article_embeddings.shape}"
        )

    # E5 embedding을 생성할 때 float32로 저장하기로 했으므로 확인
    if article_embeddings.dtype != np.float32:
        raise ValueError(
            "article_embeddings.npy dtype은 float32여야 합니다. "
            f"dtype={article_embeddings.dtype}"
        )

    embedding_row_count = int(
        article_embeddings.shape[0]
    )

    embedding_dim = int(
        article_embeddings.shape[1]
    )



    # STEP 12-2-5. embedding_row 범위 검사 
    # 예:
    # article_embeddings.shape = (20719, 768)
    #
    # 사용 가능한 embedding_row:
    # 0 ~ 20718
    #
    # article_master.embedding_row=30000 같은 값이 있으면
    # numpy indexing이 불가능하므로 export 전에 차단한다.
    if article_master.height > 0:
        minimum_embedding_row = int(
            article_master
            .get_column("embedding_row")
            .min()
        )

        maximum_embedding_row = int(
            article_master
            .get_column("embedding_row")
            .max()
        )

        if minimum_embedding_row < 0:
            raise ValueError(
                "article_master.embedding_row에 음수가 존재합니다. "
                f"min={minimum_embedding_row}"
            )

        if maximum_embedding_row >= embedding_row_count:
            raise ValueError(
                "article_master.embedding_row가 "
                "article_embeddings.npy 범위를 벗어납니다. "
                f"max_row={maximum_embedding_row}, "
                f"embedding_rows={embedding_row_count}"
            )
    else:
        minimum_embedding_row = None
        maximum_embedding_row = None

    # 검사를 통과한 내용을 manifest에 기록할 수 있도록 반환
    return {
        "status": "PASS",
        "article_count": int(
            article_master.height
        ),
        "unique_event_count": int(
            article_master
            .select("event_id")
            .unique()
            .height
        ),
        "minimum_embedding_row": minimum_embedding_row,
        "maximum_embedding_row": maximum_embedding_row,
        "article_embeddings_shape": [
            embedding_row_count,
            embedding_dim,
        ],
        "article_embeddings_dtype": str(
            article_embeddings.dtype
        ),
    }


# STEP 12-3. RQ-VAE Validation frozen infer 입력 검증
# Validation-only 기사는 RQ-VAE 학습에 넣지 않는다.
# 이미 Train으로 학습된 checkpoint를 고정한 뒤 inference만 수행한다.
#
# 주요 입력:
#
# validation_article_master.parquet
#   article_id
#   embedding_row
#   model_category_id
#   event_id
#
# article_embeddings.npy
#   -> validation-only 기사 x(a) 조회
#
# event_master_with_validation.parquet
#   -> event_id가 Train-origin인지 Validation-origin인지,
#      first_article_id가 무엇인지 확인 가능
#
# 특히 Validation-origin Event는 birth 시점에 c2를 한 번만 정하고
# 이후에는 freeze해야 하므로 Event metadata가 중요하다.

def _validate_rqvae_validation_inputs() -> dict[str, Any]:
    """
    validation-only frozen inference package 생성 전 정합성 검사
    """

    # STEP 12-3-1. validation article master 읽기 
    validation_master = pl.read_parquet(
        config.VALIDATION_ARTICLE_MASTER_PATH
    )

    _validate_required_columns(
        validation_master,
        {
            "article_id",
            "embedding_row",
            "model_category_id",
            "event_id",
            "published_time",
            "model_text",
        },
        "validation_article_master.parquet",
    )

    # validation-only article 하나당 한 행이어야 한다.
    _validate_unique_non_null_id(
        validation_master,
        "article_id",
        "validation_article_master.parquet",
    )

    # STEP 12-3-2. validation까지 반영된 event master 읽기 
    event_master = pl.read_parquet(
        config.EVENT_MASTER_WITH_VALIDATION_PATH
    )

    _validate_required_columns(
        event_master,
        {
            "event_id",
            "event_origin_split",
            "first_article_id",
        },
        "event_master_with_validation.parquet",
    )

    # Event 하나당 event_master 한 행
    _validate_unique_non_null_id(
        event_master,
        "event_id",
        "event_master_with_validation.parquet",
    )

    # STEP 12-3-3. article_embeddings.npy 연결 범위 검사
    article_embeddings = np.load(
        config.ARTICLE_EMBEDDINGS_PATH,
        mmap_mode="r",
    )

    embedding_row_count = int(
        article_embeddings.shape[0]
    )

    if validation_master.height > 0:
        minimum_embedding_row = int(
            validation_master
            .get_column("embedding_row")
            .min()
        )

        maximum_embedding_row = int(
            validation_master
            .get_column("embedding_row")
            .max()
        )

        if minimum_embedding_row < 0:
            raise ValueError(
                "validation embedding_row에 음수가 존재합니다."
            )

        if maximum_embedding_row >= embedding_row_count:
            raise ValueError(
                "validation embedding_row가 "
                "article_embeddings.npy 범위를 벗어납니다. "
                f"max_row={maximum_embedding_row}, "
                f"embedding_rows={embedding_row_count}"
            )
    else:
        minimum_embedding_row = None
        maximum_embedding_row = None



    # STEP 12-3-4. Validation article의 event_id 참조 정합성 검사
    # validation_article_master의 event_id는 반드시
    # event_master_with_validation.parquet 안에 존재해야함
    event_ids = set(
        int(event_id)
        for event_id
        in event_master
        .get_column("event_id")
        .to_list()
    )

    missing_event_ids = sorted(
        {
            int(event_id)
            for event_id
            in validation_master
            .get_column("event_id")
            .to_list()
            if int(event_id) not in event_ids
        }
    )

    if missing_event_ids:
        raise ValueError(
            "validation_article_master의 event_id 중 "
            "Event Master에 없는 값이 있습니다. "
            f"예시={missing_event_ids[:10]}"
        )

    return {
        "status": "PASS",
        "validation_only_article_count": int(
            validation_master.height
        ),
        "event_count": int(
            event_master.height
        ),
        "minimum_embedding_row": minimum_embedding_row,
        "maximum_embedding_row": maximum_embedding_row,
    }


# STEP 12-4. Transformer 입력 검증 
# Transformer 단계에서는 article embedding이 아니라
# RQ-VAE가 만든 Semantic ID sequence를 사용한다.
#
# 핵심 파일:
#
# article_semantic_ids.parquet
#   article_id -> (c1,c2,c3)
#
# train_sequences.parquet
#   history SID sequence -> target SID set
#
# validation_sequences.parquet
#   history SID sequence
#   + target SID set
#   + candidate SID sequence
#   + candidate_labels
#
# 여기서는 build_sequences.py에서 이미 검사한 내용을
# "전달 직전" 한 번 더 최소한으로 확인한다.

def _validate_transformer_inputs() -> dict[str, Any]:
    """
    Transformer 전달 package 생성 전 최소 정합성 검사.
    """

    # STEP 12-4-1. Article sid 검사
    semantic_ids = pl.read_parquet(
        config.ARTICLE_SEMANTIC_IDS_PATH
    )

    _validate_required_columns(
        semantic_ids,
        {
            "article_id",
            "c1",
            "c2",
            "c3",
        },
        "article_semantic_ids.parquet",
    )

    # 기사 하나 -> SID 하나
    _validate_unique_non_null_id(
        semantic_ids,
        "article_id",
        "article_semantic_ids.parquet",
    )

    # SID 구성 요소는 하나라도 null이면 안 된다.
    for column_name in [
        "c1",
        "c2",
        "c3",
    ]:
        null_count = (
            semantic_ids
            .get_column(column_name)
            .null_count()
        )

        if null_count != 0:
            raise ValueError(
                f"article_semantic_ids.{column_name}에 null이 존재합니다."
            )

    # STEP 12-4-2. Train / validation seq 읽기
    train_sequences = pl.read_parquet(
        config.TRAIN_SEQUENCES_PATH
    )

    validation_sequences = pl.read_parquet(
        config.VALIDATION_SEQUENCES_PATH
    )

    # Train / Validation 공통으로 필요한 컬럼
    common_columns = {
        "impression_id",
        "user_id",
        "impression_time",
        "history_article_ids",
        "history_c1",
        "history_c2",
        "history_c3",
        "target_article_ids",
        "target_c1",
        "target_c2",
        "target_c3",
    }

    _validate_required_columns(
        train_sequences,
        common_columns,
        "train_sequences.parquet",
    )

    # Validation에는 ranking candidate 관련 컬럼이 추가로 필요
    _validate_required_columns(
        validation_sequences,
        common_columns
        | {
            "candidate_article_ids",
            "candidate_c1",
            "candidate_c2",
            "candidate_c3",
            "candidate_labels",
        },
        "validation_sequences.parquet",
    )



    # STEP 12-4-3. Train / validation 공통 list 정합성 검사 
    for split_name, dataframe in [
        ("train", train_sequences),
        ("validation", validation_sequences),
    ]:
        # impression 하나당 sequence row 하나
        _validate_unique_non_null_id(
            dataframe,
            "impression_id",
            f"{split_name}_sequences.parquet",
        )

        # history_article_ids와 history_c1/c2/c3는
        # 반드시 같은 길이어야 한다.
        #
        # 예:
        # history_article_ids = [A,B,C]
        # history_c1          = [1,2,3]
        # history_c2          = [4,5,6]
        # history_c3          = [7,8,9]
        invalid_history_count = dataframe.filter(
            (
                pl.col("history_article_ids").list.len()
                != pl.col("history_c1").list.len()
            )
            | (
                pl.col("history_article_ids").list.len()
                != pl.col("history_c2").list.len()
            )
            | (
                pl.col("history_article_ids").list.len()
                != pl.col("history_c3").list.len()
            )
        ).height

        if invalid_history_count != 0:
            raise ValueError(
                f"{split_name} history list 길이 불일치: "
                f"{invalid_history_count}행"
            )

        # target은 최소 1개 이상이어야 하고
        # article_id / c1 / c2 / c3 list 길이가 같아야 한다.
        invalid_target_count = dataframe.filter(
            (
                pl.col("target_article_ids").list.len()
                <= 0
            )
            | (
                pl.col("target_article_ids").list.len()
                != pl.col("target_c1").list.len()
            )
            | (
                pl.col("target_article_ids").list.len()
                != pl.col("target_c2").list.len()
            )
            | (
                pl.col("target_article_ids").list.len()
                != pl.col("target_c3").list.len()
            )
        ).height

        if invalid_target_count != 0:
            raise ValueError(
                f"{split_name} target list 정합성 오류: "
                f"{invalid_target_count}행"
            )


    # STEP 12-4-4. Validation candidate 정합성 검사
    # candidate_article_ids와 candidate c1/c2/c3/label도
    # index 기준 1:1 대응이므로 길이가 모두 같아야 한다.
    #
    # 예:
    # candidate_article_ids = [A,B,C,D]
    # candidate_labels      = [0,1,0,1]
    #
    # target이 candidate에 반드시 포함되도록 build_sequences.py에서
    # 필터링했으므로 label sum도 최소 1이어야 한다.

    invalid_candidate_count = validation_sequences.filter(
        (
            pl.col("candidate_article_ids").list.len()
            <= 0
        )
        | (
            pl.col("candidate_article_ids").list.len()
            != pl.col("candidate_c1").list.len()
        )
        | (
            pl.col("candidate_article_ids").list.len()
            != pl.col("candidate_c2").list.len()
        )
        | (
            pl.col("candidate_article_ids").list.len()
            != pl.col("candidate_c3").list.len()
        )
        | (
            pl.col("candidate_article_ids").list.len()
            != pl.col("candidate_labels").list.len()
        )
        | (
            pl.col("candidate_labels").list.sum()
            <= 0
        )
    ).height

    if invalid_candidate_count != 0:
        raise ValueError(
            "validation candidate list 정합성 오류가 존재합니다. "
            f"문제 행 수={invalid_candidate_count}"
        )

    return {
        "status": "PASS",
        "semantic_id_count": int(
            semantic_ids.height
        ),
        "train_sequence_count": int(
            train_sequences.height
        ),
        "validation_sequence_count": int(
            validation_sequences.height
        ),
    }


# STEP 12-5. RQ-VAE 입력 패키지 EXPORT
# ============================================================
#
# 이 함수는 RQ-VAE에 필요한 전달물을 두 package로 나눈다.
#
# 1. rqvae_train_inputs
#    -> Train 기사로 RQ-VAE를 학습할 때 사용
#
# 2. rqvae_validation_inputs
#    -> 학습 완료 checkpoint를 고정하고
#       Validation-only 기사 SID를 inference할 때 사용
#
# 왜 둘을 나누는가?
# ------------------------------------------------------------
# Validation-only 기사를 Train DataLoader에 넣으면
# validation 정보가 학습에 들어가는 leakage가 발생할 수 있기 때문이다.
# ============================================================


def export_rqvae_inputs() -> dict[str, Any]:
    """
    RQ-VAE Train package + Validation Frozen Inference package 생성.

    Train 핵심 파일
    --------------------------------------------------------
    article_master.parquet
      -> 어떤 Train article을 사용할지
      -> embedding_row / category / event_id 연결

    article_embeddings.npy
      -> embedding_row로 x(a) 조회

    Train 보조 파일
    --------------------------------------------------------
    category_mapping.parquet
      -> model_category_id 해석

    article_events.parquet
      -> article_id -> event_id 관계 확인

    event_master.parquet
      -> Train Event 분석/검증

    Validation 핵심 파일
    --------------------------------------------------------
    validation_article_master.parquet
      -> validation-only 기사 목록 및 metadata

    article_embeddings.npy
      -> 동일한 전체 기사 embedding 배열 재사용

    event_master_with_validation.parquet
      -> Train-origin / Validation-origin Event 상태 확인

    Validation 보조
    --------------------------------------------------------
    validation_article_events.parquet
      -> validation-only article -> event_id mapping 확인
    """

    # STEP 12-5-1. 반드시 있어야 하는 TRAIN 입력 
    required_train_paths = {
        "article_master": (
            config.ARTICLE_MASTER_PATH
        ),
        "article_embeddings": (
            config.ARTICLE_EMBEDDINGS_PATH
        ),
    }

    # STEP 12-5-2. 반드시 있어야 하는 VALIDATION inference 입력
    required_validation_paths = {
        "validation_article_master": (
            config.VALIDATION_ARTICLE_MASTER_PATH
        ),
        "article_embeddings": (
            config.ARTICLE_EMBEDDINGS_PATH
        ),
        "event_master_with_validation": (
            config.EVENT_MASTER_WITH_VALIDATION_PATH
        ),
    }


    # 파일이 없으면 여기서 즉시 중단
    _require_paths(
        required_train_paths
    )

    _require_paths(
        required_validation_paths
    )

    # STEP 12-5-3. 복사하기 전 정합성 검사
    train_validation = (
        _validate_rqvae_train_inputs()
    )

    validation_validation = (
        _validate_rqvae_validation_inputs()
    )

    # STEP 12-5-4. 이전 export 폴더 초기화 
    _reset_directory(
        RQ_VAE_TRAIN_EXPORT_DIR
    )

    _reset_directory(
        RQ_VAE_VALIDATION_EXPORT_DIR
    )


    # STEP 12-5-5. RQ-VAE train 핵심 파일 복사
    train_exported_files = [
        _copy_file(
            config.ARTICLE_MASTER_PATH,
            RQ_VAE_TRAIN_EXPORT_DIR,
        ),
        _copy_file(
            config.ARTICLE_EMBEDDINGS_PATH,
            RQ_VAE_TRAIN_EXPORT_DIR,
        ),
    ]

    # STEP 12-5-6. Train 분석 / 검증용 보조 파일 복사
    # 없어도 핵심 RQ-VAE Train package 생성은 가능하도록 optional 처리
    for optional_path in [
        config.CATEGORY_MAPPING_PATH,
        config.ARTICLE_EVENTS_PATH,
        config.EVENT_MASTER_PATH,
    ]:
        copied = _copy_optional_file(
            optional_path,
            RQ_VAE_TRAIN_EXPORT_DIR,
        )

        if copied is not None:
            train_exported_files.append(
                copied
            )

    # STEP 12-5-7. Validation frozen infer 핵심 파일 복사 
    validation_exported_files = [
        _copy_file(
            config.VALIDATION_ARTICLE_MASTER_PATH,
            RQ_VAE_VALIDATION_EXPORT_DIR,
        ),
        _copy_file(
            config.ARTICLE_EMBEDDINGS_PATH,
            RQ_VAE_VALIDATION_EXPORT_DIR,
        ),
        _copy_file(
            config.EVENT_MASTER_WITH_VALIDATION_PATH,
            RQ_VAE_VALIDATION_EXPORT_DIR,
        ),
    ]

    # validation article -> event mapping은 보조 파일
    copied_validation_events = (
        _copy_optional_file(
            config.VALIDATION_ARTICLE_EVENTS_PATH,
            RQ_VAE_VALIDATION_EXPORT_DIR,
        )
    )

    if copied_validation_events is not None:
        validation_exported_files.append(
            copied_validation_events
        )

    # STEP 12-5-8. Train package manifest 저장
    train_manifest_path = _write_manifest(
        RQ_VAE_TRAIN_EXPORT_DIR,
        "rqvae_train_inputs",
        train_validation,
        train_exported_files,
    )

    # STEP 12-5-9. Validation package manifest 저장
    validation_manifest_path = _write_manifest(
        RQ_VAE_VALIDATION_EXPORT_DIR,
        "rqvae_validation_inputs",
        validation_validation,
        validation_exported_files,
    )

    # 호출한 main.py가 결과를 로그로 보여줄 수 있게
    # 생성된 경로와 validation 결과 반환
    return {
        "status": "SUCCESS",
        "rqvae_train_export_dir": str(
            RQ_VAE_TRAIN_EXPORT_DIR
        ),
        "rqvae_validation_export_dir": str(
            RQ_VAE_VALIDATION_EXPORT_DIR
        ),
        "train_manifest_path": str(
            train_manifest_path
        ),
        "validation_manifest_path": str(
            validation_manifest_path
        ),
        "train_validation": train_validation,
        "validation_validation": validation_validation,
    }


# STEP 12-7. Transformer 입력 package export
# 실행 전제:
# ------------------------------------------------------------
# RQ-VAE가 이미
#
#     article_semantic_ids.parquet
#
# 을 생성했고,
# 그 결과를 이용해 build_sequences.py가
#
#     train_sequences.parquet
#     validation_sequences.parquet
#
# 을 생성한 뒤여야 한다.
#
# 즉 이 함수는 파이프라인 뒤쪽에서 실행된다.

def export_transformer_inputs() -> dict[str, Any]:
    """
    Transformer 팀에 전달할 최종 package를 생성한다.

    핵심
    --------------------------------------------------------
    train_sequences.parquet
      -> Transformer 학습 sample

    validation_sequences.parquet
      -> ranking validation sample

    함께 전달
    --------------------------------------------------------
    article_semantic_ids.parquet
      -> SID ↔ 실제 article_id 연결

    category_mapping.parquet
      -> c1/category 해석 보조
    """

    # STEP 12-6-1. Transformer 필수 파일 존재 검사
    required_paths = {
        "article_semantic_ids": (
            config.ARTICLE_SEMANTIC_IDS_PATH
        ),
        "train_sequences": (
            config.TRAIN_SEQUENCES_PATH
        ),
        "validation_sequences": (
            config.VALIDATION_SEQUENCES_PATH
        ),
    }

    _require_paths(
        required_paths
    )

    # STEP 12-6-2. 최종 전달 직전 정합성 검사
    validation_result = (
        _validate_transformer_inputs()
    )

    # STEP 12-6-3. 과거 트랜스포머 EXPORT 폴더 제거 
    _reset_directory(
        TRANSFORMER_EXPORT_DIR
    )

    # STEP 12-6-4. 트랜스포머 핵심 파일 복사 
    exported_files = [
        _copy_file(
            config.ARTICLE_SEMANTIC_IDS_PATH,
            TRANSFORMER_EXPORT_DIR,
        ),
        _copy_file(
            config.TRAIN_SEQUENCES_PATH,
            TRANSFORMER_EXPORT_DIR,
        ),
        _copy_file(
            config.VALIDATION_SEQUENCES_PATH,
            TRANSFORMER_EXPORT_DIR,
        ),
    ]

    # category_mapping은 모델 계산에 필수는 아니지만 c1/category 결과 해석시 유용하므로 존재하면 함께 전달하도록
    copied_category_mapping = (
        _copy_optional_file(
            config.CATEGORY_MAPPING_PATH,
            TRANSFORMER_EXPORT_DIR,
        )
    )

    if copied_category_mapping is not None:
        exported_files.append(
            copied_category_mapping
        )


    # STEP 12-6-5. Transformer package manifest 저장
    manifest_path = _write_manifest(
        TRANSFORMER_EXPORT_DIR,
        "transformer_inputs",
        validation_result,
        exported_files,
    )

    return {
        "status": "SUCCESS",
        "transformer_export_dir": str(
            TRANSFORMER_EXPORT_DIR
        ),
        "manifest_path": str(
            manifest_path
        ),
        "validation": validation_result,
    }


# STEP 12-7. 현재 준비된 package 전체 export
# 이 함수는 편의용 래퍼함수
# 상황 1. 전처리만 끝난 직후
# ------------------------------------------------------------
# article_semantic_ids.parquet가 아직 없음
#
# -> RQ-VAE package만 export
# -> Transformer package는 건너뜀
#
# 상황 2. RQ-VAE + build_sequences까지 끝난 뒤
# ------------------------------------------------------------
# article_semantic_ids.parquet
# train_sequences.parquet
# validation_sequences.parquet
# 모두 존재
#
# -> RQ-VAE package
# -> Transformer package
# 둘 다 export 가능

def export_inputs() -> dict[str, Any]:
    """
    현재 파일 준비 상태를 보고 가능한 package를 export한다.
    """

    # 기본 output 폴더 생성
    config.create_output_directories()

    EXPORT_ROOT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 전처리 결과만 있으면 만들 수 있는 RQ-VAE package는 항상 시도
    rqvae_result = (
        export_rqvae_inputs()
    )

    # Transformer package를 만들려면 아래 세 파일이 모두 필요
    transformer_required_paths = [
        config.ARTICLE_SEMANTIC_IDS_PATH,
        config.TRAIN_SEQUENCES_PATH,
        config.VALIDATION_SEQUENCES_PATH,
    ]

    transformer_ready = all(
        path.exists()
        for path in transformer_required_paths
    )

    # 준비되어 있을 때만 Transformer export
    if transformer_ready:
        transformer_result: (
            dict[str, Any] | None
        ) = export_transformer_inputs()
    else:
        transformer_result = None

    return {
        "status": "SUCCESS",
        "rqvae_result": rqvae_result,
        "transformer_ready": transformer_ready,
        "transformer_result": transformer_result,
    }

# STEP 12-8. 직접 실행 CLI

# 사용 예
# ------------------------------------------------------------
# 1. RQ-VAE 전달 package만 만들기
#
#     python -m src.export_inputs rqvae
#
# 2. Transformer 전달 package만 만들기
#
#     python -m src.export_inputs transformer
#
# 3. 현재 준비된 파일을 보고 가능한 package 모두 만들기
#
#     python -m src.export_inputs available
#
# 아무 mode도 주지 않으면 available이 기본값이다.

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "RQ-VAE / Transformer 모델 전달용 입력 package export"
        )
    )

    parser.add_argument(
        "mode",
        nargs="?",
        default="available",
        choices=[
            "rqvae",
            "transformer",
            "available",
        ],
        help=(
            "rqvae: RQ-VAE package만 / "
            "transformer: Transformer package만 / "
            "available: 현재 준비된 package 자동 export"
        ),
    )

    args = parser.parse_args()

    if args.mode == "rqvae":
        result = export_rqvae_inputs()

    elif args.mode == "transformer":
        result = export_transformer_inputs()

    else:
        result = export_inputs()

    print()
    print("=" * 70)
    print("Model Input Export 완료")
    print("=" * 70)
    pprint(result)
