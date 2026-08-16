from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any
import unicodedata

import polars as pl

from src import config

# Entity processing
# 목적
# ------------------------------------------
# event clustering 자체를 다시 설계하지 않고, event 입력으로 사용할 entity 표현만 독립적으로 생성

# baseline
# - 기존 NFKC + whitespace + lowercase + TYPE namespace만 적용

# normalize_only
# - baseline 처리
# - train-used article만 이용해 보수적인 danish possessive(-s) mapping 
# - fit된 mapping을 전체 valid article에 적용

# normalize_and_link
# - 다음 단계에서 구현 예정

SUPPORTED_ENTITY_PROCESSING_MODES = {
    "baseline",
    "normalize_only",
    "normalize_and_link",
}


# normalize_only v1에서 자동 적용할 TYPE. (소유격이 문법적으로 자연스러운대상)
_SAFE_POSSESSIVE_ENTITY_GROUPS = {
    "PER",
    "ORG",
    "LOC",
}

# =============================================================================
# Safe Normalization version
# =============================================================================
#
# v1
#   baseline
#     ↓
#   safe possessive
#
# v2
#   baseline
#     ↓
#   safe possessive
#     ↓
#   safe hyphen/space
#
# config.ENTITY_NORMALIZATION_VERSION으로 선택한다.
# =============================================================================

SUPPORTED_ENTITY_NORMALIZATION_VERSIONS = {
    "v1",
    "v2",
}


# v2에서 hyphen/space 후보를 만들 때 동일하게 처리할 dash 문자들.
#
# 실제 entity를 무조건 바꾸는 용도가 아니다.
# 뒤의 guardrail을 통과한 경우에만 최종 mapping으로 사용한다.
_V2_DASH_TRANSLATION = str.maketrans(
    {
        "\u2010": "-",  # HYPHEN
        "\u2011": "-",  # NON-BREAKING HYPHEN
        "\u2012": "-",  # FIGURE DASH
        "\u2013": "-",  # EN DASH
        "\u2014": "-",  # EM DASH
        "\u2212": "-",  # MINUS SIGN
        "\uFE58": "-",  # SMALL EM DASH
        "\uFE63": "-",  # SMALL HYPHEN-MINUS
        "\uFF0D": "-",  # FULLWIDTH HYPHEN-MINUS
    }
)

# 1. Baseline-compatible normalization

def normalize_entity_text(entity: Any) -> str | None : 
    """
    현재 baseline과 동일한 entity 문자열 normalization.

    수행:
    1. unicode NFKC
    2. 앞뒤 공백 제거
    3. 연속 whitespace 한 칸으로 축약
    4. lowercase
    예:
        "  Ekstra   Bladet " -> "ekstra bladet"
        "TikTok"              -> "tiktok"
    """

    if entity is None: return None 

    text = unicodedata.normalize("NFKC", str(entity))

    text = " ".join(text.strip().split())

    text = text.lower()

    if text == "": return None 

    return text 

def normalize_entity_group(entity_group: Any) -> str | None: 
    """
    entity Type을 baseline과 동일하게 정규화
    예 : "per" -> "PER"
    """

    if entity_group is None: return None 

    group = unicodedata.normalize("NFKC", str(entity_group))

    group = group.strip().upper()

    if group=="": return None 

    return group 

def make_entity_key(
        entity_group : str, entity_text : str, 
)-> str: 
    """
    event clustering에서 사용하는 type namespace 포함 key 생성
    예 : PER + vladimir putin -> PER::vladimir putin 
    """

    return f"{entity_group}::{entity_text}"

def build_baseline_entity_set(
        raw_entities: Any, raw_entity_groups: Any, 
)-> set[str]:
    """
    기존 build_train._normalize_entity_set()과 동일한 결과 생성
    이 함수는 baseline 재현/ 호환용으로 normalize_only event에선
    article_entities.parquet의 canonical key를 읽음
    """

    if raw_entities is None : return set()
    if raw_entity_groups is None: return set()
    if len(raw_entities) != len(raw_entity_groups):
        raise ValueError(
            "ner_clusters와 entity_groups의 길이가 다릅니다. "
            f"ner_length={len(raw_entities)}, "
            f"entity_group_length={len(raw_entity_groups)}"
        )

    entity_set: set[str] = set()

    for raw_entity, raw_group in zip(raw_entities, raw_entity_groups):
        normalized_entity = normalize_entity_text(raw_entity)
        entity_group = normalize_entity_group(raw_group)
        if normalized_entity is None: continue 
        if entity_group is None: continue 
        entity_set.add(make_entity_key(entity_group, normalized_entity))

        return entity_set 

# 2. Train mention table 

def _build_entity_mention_rows(
        articles: pl.DataFrame, train_used_article_ids: set[int],
)-> pl.DataFrame:
    """
    valid article의 NER list를 mention 단위 long table로 펼친다
    출력 예 : 
        article_id
        entity_position
        raw_entity
        raw_entity_group
        entity_group
        normalized_entity
        baseline_entity_key
        is_train_used
    """

    """
    article_id | ner_clusters                    | entity_groups
       100     | ["Joe Biden", "USA", "NATO"]    | ["PER", "LOC", "ORG"]
       205     | ["Mette Frederiksen"]            | ["PER"]
    ->
    article_id | raw_entity          | raw_entity_group
       100     | Joe Biden           | PER
       100     | USA                 | LOC
       100     | NATO                | ORG
       205     | Mette Frederiksen   | PER    
    
    """

    
    rows: list[dict[str, Any]]= []

    for row in articles.iter_rows(named=True):
        article_id = int(row["article_id"])
        raw_entities = row["ner_clusters"]
        raw_groups = row["entity_groups"]

        if raw_entities is None or raw_groups is None : continue 

        if len(raw_entities) != len(raw_groups):
            raise ValueError(
                "유효 기사에서 ner_clusters와 entity_groups의 길이가 다릅니다. "
                f"article_id={article_id}, "
                f"ner_length={len(raw_entities)}, "
                f"entity_group_length={len(raw_groups)}"
            )

        for entity_position, (raw_entity, raw_group) in enumerate(zip(raw_entities, raw_groups)):
            normalized_entity = normalize_entity_text(raw_entity)
            entity_group = normalize_entity_group(raw_group)
            if normalized_entity is None: continue 
            if entity_group is None: continue 

            rows.append(
                {
                    "article_id": article_id,
                    "entity_position": int(entity_position),
                    "raw_entity": str(raw_entity),
                    "raw_entity_group": str(raw_group),
                    "entity_group": entity_group,
                    "normalized_entity": normalized_entity,
                    "baseline_entity_key": make_entity_key(
                        entity_group,
                        normalized_entity,
                    ),
                    "is_train_used": (
                        article_id in train_used_article_ids
                    ),
                }
            )

    schema = {
        "article_id": pl.Int64,
        "entity_position": pl.Int64,
        "raw_entity": pl.String,
        "raw_entity_group": pl.String,
        "entity_group": pl.String,
        "normalized_entity": pl.String,
        "baseline_entity_key": pl.String,
        "is_train_used": pl.Boolean,
    }

    if not rows:
        return pl.DataFrame(schema=schema)

    return pl.DataFrame(rows, schema=schema)

# 3. Safe possessive normalization mapping 

def _is_safe_possessive_candidate(
        entity_group:str, variant:str, base:str, variant_df:int, base_df: int,
)-> tuple[bool, str]:
    """
    normalize_only v1의 보수적인 trailing-s 판단

    inpsection에서 확인한 중요 반례 :
        Andreas -> Andrea
        Sky Sports -> Sky Sport 
    
    때문에 끝이 s면 제거 규칙을 무조건 사용하지 않는다.

    현재 가이드라인
    --------------------------------------------------
    1. PER/ORG/LOC만 자동 정규화 대상
    2. base도 같은 TYPE의 train vocab에 실존해야함 (호출 전에 보장)
    -> train data에서 정규화 거치고 난 고유 ORG::ekstra bladet이 있어야함 
    3. base DF >= variant DF 
        - canonical(대표)/base 표현이 variant(s 있는 쪽) 보다 적게 등장하면 제외
    4. base가 2개 이상 기사에 등장해야함
    5. PER은 두 단어 이상만 
    6. base가 3글자 이상이어야함 
    """
    if entity_group not in _SAFE_POSSESSIVE_ENTITY_GROUPS:
        return False, "unsupported_entity_group"

    if not variant.endswith("s"):
        return False, "not_trailing_s"

    if base == "" or len(base) < 3:
        return False, "base_too_short"

    if base_df < 2:
        return False, "base_df_too_low"

    if base_df < variant_df:
        return False, "base_less_frequent_than_variant"

    token_count = len(variant.split())

    if (
        entity_group == "PER"
        and token_count < 2
    ):
        return False, "single_token_person_name"

    return True, "safe_possessive_v1"


def _build_safe_normalization_mapping(
        train_mentions: pl.DataFrame,
)-> tuple[dict [str, str], pl.DataFrame, dict[str, int]]:
    """
    train-used article mention만 이용해 normalize_only mapping

    반환
    -------------------
    normalization_mapping:
        baseline_entity_key -> canonical_entity_key

    mapping_df:
        디버깅/재현용 상세 mapping

    stats:
        candidate/accepted/rejected 개수
    """

    if train_mentions.height == 0:
        empty_mapping_df = pl.DataFrame(
            schema={
                "entity_group": pl.String,
                "variant_entity": pl.String,
                "base_entity": pl.String,
                "variant_entity_key": pl.String,
                "canonical_entity_key": pl.String,
                "variant_article_df": pl.Int64,
                "base_article_df": pl.Int64,
                "rule": pl.String,
            }
        )

        return {}, empty_mapping_df, {
            "possessive_candidate_count": 0,
            "possessive_accepted_count": 0,
            "possessive_rejected_count": 0,
        }

    # article DF는 같은 기사 내부 mention 중복을 한 번만 세야함
    # 기사별로 개체를 집합에 모으기
    """
    ("ORG","ekstra bladet")  → 2      (기사 100, 205)
    ("ORG","ekstra bladets") → 1      (기사 307)
    ("PER","andreas")        → 1
    """
    # 모든 개체를 하나씩 검사해 사전검사 3개 (s로 끝나는지, 한글자면 스킵,...)
    entity_articles: dict[tuple[str, str], set[int]]={}
    for row in train_mentions.select([
        "article_id", "entity_group", "normalized_entity"
    ]).iter_rows(named=True):
        key = (str(row["entity_group"]), str(row["normalized_entity"]))
        entity_articles.setdefault(key, set()).add(int(row["article_id"]))

    entity_df: dict[tuple[str, str], int] = {
        key: len(article_ids) for key, article_ids in entity_articles.items()
    }

    mapping: dict[str, str] = {}
    mapping_rows: list[dict[str, Any]] = []

    candidate_count = 0; accepted_count=0

    # 정렬해 deterministic하게 mapping 생성
    for (entity_group, variant) , variant_article_df in sorted(entity_df.items()):
        if not variant.endswith("s"): continue 
        if len(variant) <= 1: continue 

        base = variant[:-1]
        base_key_tuple = (entity_group, base)

        if base_key_tuple not in entity_df: continue 

        candidate_count += 1
        base_article_df = int(entity_df[base_key_tuple])

        is_safe, rule =  (
            _is_safe_possessive_candidate(
                entity_group=entity_group,
                variant=variant,
                base=base,
                variant_df=int(variant_article_df),
                base_df=base_article_df,
            )
        )

        if not is_safe: continue 
        variant_entity_key = make_entity_key(entity_group, variant)
        canonical_entity_key = make_entity_key(entity_group, base)
        mapping[variant_entity_key] = canonical_entity_key

        mapping_rows.append(
            {
                "entity_group": entity_group,
                "variant_entity": variant,
                "base_entity": base,
                "variant_entity_key": variant_entity_key,
                "canonical_entity_key": canonical_entity_key,
                "variant_article_df": int(variant_article_df),
                "base_article_df": base_article_df,
                "rule": rule,
            } 
        )

        accepted_count+=1 

    mapping_schema = {
        "entity_group": pl.String,
        "variant_entity": pl.String,
        "base_entity": pl.String,
        "variant_entity_key": pl.String,
        "canonical_entity_key": pl.String,
        "variant_article_df": pl.Int64,
        "base_article_df": pl.Int64,
        "rule": pl.String,
    }

    if mapping_rows:
        mapping_df = (
            pl.DataFrame(
                mapping_rows,
                schema=mapping_schema,
            )
            .sort(
                [
                    "entity_group",
                    "variant_entity",
                ]
            )
        )
    else:
        mapping_df = pl.DataFrame(
            schema=mapping_schema
        )

    return (
        mapping,
        mapping_df,
        {
            "possessive_candidate_count": int(
                candidate_count
            ),
            "possessive_accepted_count": int(
                accepted_count
            ),
            "possessive_rejected_count": int(
                candidate_count - accepted_count
            ),
        },
    )

# =============================================================================
# 4. Safe hyphen/space normalization mapping - v2
# =============================================================================


def _hyphen_to_space_candidate(
    entity_text: str,
) -> str:
    """
    v2 hyphen/space 후보 문자열을 만든다.

    이 함수 자체는 entity를 확정적으로 변경하지 않는다.
    단지 "이렇게 바꿨을 때 어떤 문자열이 되는가?"를 계산한다.

    예:
        "jyllands - posten"
        -> "jyllands posten"

        "helle thorning-schmidt"
        -> "helle thorning schmidt"

        "covid - 19"
        -> "covid 19"

    이후 _build_safe_hyphen_v2_mapping()에서
    target이 실제 Train vocabulary에 존재하는지 등을 다시 검사한다.
    """

    # Unicode dash를 먼저 일반 '-'로 통일
    text = entity_text.translate(
        _V2_DASH_TRANSLATION
    )

    # hyphen을 space로 변경
    text = text.replace(
        "-",
        " ",
    )

    # hyphen 주변에 공백이 이미 있었으면
    # 여러 개의 공백이 생길 수 있으므로 다시 한 칸으로 축약
    text = " ".join(
        text.split()
    )

    return text


def _has_dangling_hyphen(
    entity_text: str,
) -> bool:
    """
    NER이 중간에서 잘린 것처럼 보이는 entity를 찾는다.

    실제 v2 후보 검토에서 발견된 위험 사례:

        "midt -"
        "olsen -"

    이런 것은 정상적인:
        "jyllands - posten"

    같은 표기 차이와 다르다.

    따라서 entity 시작이나 끝이 hyphen이면
    v2 자동 normalization에서 제외한다.
    """

    text = entity_text.translate(
        _V2_DASH_TRANSLATION
    ).strip()

    return (
        text.startswith("-")
        or text.endswith("-")
    )


def _build_safe_hyphen_v2_mapping(
    train_mentions: pl.DataFrame,
    possessive_mapping: dict[str, str],
) -> tuple[
    dict[str, str],
    pl.DataFrame,
    dict[str, int],
]:
    """
    Safe Normalization v2의 hyphen/space mapping을 Train에서 fit한다.

    매우 중요
    -------------------------------------------------------------------------
    v2는 raw entity에 바로 적용하지 않는다.

        baseline
            ↓
        safe possessive v1
            ↓
        safe hyphen v2

    순서로 적용한다.

    즉 여기서는 v1 적용 후의 canonical vocabulary를 대상으로
    hyphen/space 차이를 다시 찾는다.


    Guardrail
    -------------------------------------------------------------------------
    1. Train-used article만 사용

    2. 같은 entity TYPE 안에서만 mapping

       예:
           ORG -> ORG
           PER -> PER

    3. hyphen을 space로 변경한 최종 target이
       같은 TYPE의 Train vocabulary에 실제 존재해야 한다.

       예:

           ORG::jyllands - posten
           ORG::jyllands posten

       둘 다 Train에서 관측
       -> 허용 가능

    4. dangling hyphen 제외

       예:
           ORG::midt -
           PROD::olsen -

       -> 제외

    5. terminal punctuation은 처리하지 않는다.

       예:
           allan j.
           -> allan j

       이런 것은 이번 v2에서 보류한다.


    반환
    -------------------------------------------------------------------------
    hyphen_mapping

        v1 canonical entity key
        ->
        v2 canonical entity key


    mapping_df

        어떤 mapping이 생성됐는지 확인하는 상세 테이블


    stats

        collision group 수
        candidate 수
        accepted 수
        rejected 수
    """

    # -------------------------------------------------------------------------
    # STEP 4-1.
    # Train mention에 v1 possessive를 먼저 적용한 뒤,
    #
    #   (entity_group, v1 canonical surface)
    #       -> {article_id, ...}
    #
    # 형태로 Train vocabulary를 만든다.
    #
    # 같은 article에서 entity가 여러 번 나와도
    # DF는 article 단위로 한 번만 센다.
    # -------------------------------------------------------------------------

    entity_articles: dict[
        tuple[str, str],
        set[int],
    ] = {}

    for row in train_mentions.select(
        [
            "article_id",
            "baseline_entity_key",
        ]
    ).iter_rows(named=True):

        article_id = int(
            row["article_id"]
        )

        baseline_entity_key = str(
            row["baseline_entity_key"]
        )

        # 먼저 v1 possessive 적용
        v1_entity_key = (
            possessive_mapping.get(
                baseline_entity_key,
                baseline_entity_key,
            )
        )

        # 예:
        # ORG::jyllands - posten
        #
        # entity_group = ORG
        # entity_surface = jyllands - posten
        entity_group, entity_surface = (
            v1_entity_key.split(
                "::",
                1,
            )
        )

        key = (
            entity_group,
            entity_surface,
        )

        entity_articles.setdefault(
            key,
            set(),
        ).add(
            article_id
        )

    # -------------------------------------------------------------------------
    # STEP 4-2.
    # hyphen/space transform 결과가 같은 entity들을 bucket으로 묶는다.
    #
    # 예:
    #
    #   jyllands - posten
    #   jyllands posten
    #
    # 각각 transform:
    #
    #   jyllands posten
    #   jyllands posten
    #
    # 따라서 같은 collision group이 된다.
    # -------------------------------------------------------------------------

    collision_buckets: dict[
        tuple[str, str],
        set[str],
    ] = {}

    for (
        entity_group,
        entity_surface,
    ) in entity_articles:

        candidate_surface = (
            _hyphen_to_space_candidate(
                entity_surface
            )
        )

        bucket_key = (
            entity_group,
            candidate_surface,
        )

        collision_buckets.setdefault(
            bucket_key,
            set(),
        ).add(
            entity_surface
        )

    mapping: dict[
        str,
        str,
    ] = {}

    mapping_rows: list[
        dict[str, Any]
    ] = []

    collision_group_count = 0
    candidate_count = 0
    accepted_count = 0

    # -------------------------------------------------------------------------
    # STEP 4-3.
    # 실제 collision group만 검사한다.
    # -------------------------------------------------------------------------

    for (
        entity_group,
        candidate_surface,
    ), surfaces in sorted(
        collision_buckets.items()
    ):

        # 서로 다른 현재 surface가 최소 2개 있어야 한다.
        #
        # surface 1개밖에 없다면
        # "비슷한 표현이 Train에 같이 존재한다"는 증거가 없으므로 제외.
        if len(surfaces) < 2:
            continue

        # 최소 하나는 실제로 변해야 한다.
        #
        # 모두 동일 문자열이면 normalization 후보가 아니다.
        changed_surfaces = [
            surface
            for surface in surfaces
            if surface != candidate_surface
        ]

        if not changed_surfaces:
            continue

        collision_group_count += 1

        # 최종 target
        target_tuple = (
            entity_group,
            candidate_surface,
        )

        # target 자체가 Train vocabulary에 실제 존재하는지
        target_observed_in_train = (
            target_tuple in entity_articles
        )

        target_article_df = (
            len(
                entity_articles[
                    target_tuple
                ]
            )
            if target_observed_in_train
            else 0
        )

        # 한 collision group 안에서도
        # 변해야 하는 variant가 여러 개일 수 있다.
        for variant_surface in sorted(
            changed_surfaces
        ):

            candidate_count += 1

            variant_tuple = (
                entity_group,
                variant_surface,
            )

            variant_article_df = len(
                entity_articles[
                    variant_tuple
                ]
            )

            # -------------------------------------------------------------
            # Guardrail 1.
            # "midt -" 같은 dangling fragment는 제외
            # -------------------------------------------------------------
            if _has_dangling_hyphen(
                variant_surface
            ):
                continue

            # -------------------------------------------------------------
            # Guardrail 2.
            # 변환 후 target이 Train에 실제 존재해야 함
            #
            # 예:
            #
            #   esben jean - pierre - blum
            #   esben jean - pierre blum
            #
            # 둘이 같은 가상 target으로 모이더라도,
            #
            #   esben jean pierre blum
            #
            # 이 Train에 실제 없다면 mapping하지 않는다.
            # -------------------------------------------------------------
            if not target_observed_in_train:
                continue

            variant_entity_key = (
                make_entity_key(
                    entity_group,
                    variant_surface,
                )
            )

            canonical_entity_key = (
                make_entity_key(
                    entity_group,
                    candidate_surface,
                )
            )

            mapping[
                variant_entity_key
            ] = canonical_entity_key

            mapping_rows.append(
                {
                    "entity_group": entity_group,
                    "variant_entity": (
                        variant_surface
                    ),
                    "base_entity": (
                        candidate_surface
                    ),
                    "variant_entity_key": (
                        variant_entity_key
                    ),
                    "canonical_entity_key": (
                        canonical_entity_key
                    ),
                    "variant_article_df": int(
                        variant_article_df
                    ),
                    "base_article_df": int(
                        target_article_df
                    ),
                    "rule": (
                        "safe_hyphen_v2"
                    ),
                }
            )

            accepted_count += 1

    # v1 mapping parquet과 같은 schema를 사용한다.
    # 따라서 뒤에서 v1 + v2 mapping_df를 그대로 concat할 수 있다.
    mapping_schema = {
        "entity_group": pl.String,
        "variant_entity": pl.String,
        "base_entity": pl.String,
        "variant_entity_key": pl.String,
        "canonical_entity_key": pl.String,
        "variant_article_df": pl.Int64,
        "base_article_df": pl.Int64,
        "rule": pl.String,
    }

    if mapping_rows:
        mapping_df = (
            pl.DataFrame(
                mapping_rows,
                schema=mapping_schema,
            )
            .sort(
                [
                    "entity_group",
                    "variant_entity",
                ]
            )
        )

    else:
        mapping_df = pl.DataFrame(
            schema=mapping_schema
        )

    return (
        mapping,
        mapping_df,
        {
            "hyphen_collision_group_count": int(
                collision_group_count
            ),
            "hyphen_candidate_count": int(
                candidate_count
            ),
            "hyphen_accepted_count": int(
                accepted_count
            ),
            "hyphen_rejected_count": int(
                candidate_count
                - accepted_count
            ),
        },
    )

# 4. Main builder
def build_article_entities(mode: str | None = None)->dict[str, Any]:
    """
    전체 valid article에 대해 event 입력용 canonical entity 생성

    fit : train-used article만 사용
    transform : 전체 valid article에 적용 (train +validation-only 포함한 전체 valid article)
        출력
    -------------------------------------------------------------------------
    article_entities.parquet
        mention-level 상세 결과

    entity_normalization_map.parquet
        normalize_only에서 사용한 Train-fit mapping

    article_entities.parquet schema 핵심
    -------------------------------------------------------------------------
    article_id
    raw_entity
    entity_group
    normalized_entity
    baseline_entity_key
    canonical_entity
    canonical_entity_key
    processing_method
    """   

    if mode is None : mode = config.ENTITY_PROCESSING_MODE

    mode = str(mode).strip().lower()

    if mode not in SUPPORTED_ENTITY_PROCESSING_MODES:
        raise ValueError(
            "지원하지 않는 ENTITY_PROCESSING_MODE입니다. "
            f"현재 값={mode}, "
            f"지원 값={sorted(SUPPORTED_ENTITY_PROCESSING_MODES)}"
        )

    if mode == "normalize_and_link":
        raise NotImplementedError(
            "ENTITY_PROCESSING_MODE='normalize_and_link'는 "
            "Entity Linking decision layer 구현 후 활성화합니다. "
            "현재는 baseline 또는 normalize_only를 사용하세요."
        )
    if mode == "normalize_and_link":
        raise NotImplementedError(
            "ENTITY_PROCESSING_MODE='normalize_and_link'는 "
            "Entity Linking decision layer 구현 후 활성화합니다. "
            "현재는 baseline 또는 normalize_only를 사용하세요."
        )

    # -------------------------------------------------------------------------
    # normalize_only 안에서 v1 / v2 선택
    #
    # config에 변수가 없으면 기존 동작을 깨지 않도록 v1이 기본값.
    # -------------------------------------------------------------------------
    if mode == "normalize_only":
        normalization_version = str(
            getattr(
                config,
                "ENTITY_NORMALIZATION_VERSION",
                "v1",
            )
        ).strip().lower()

        if (
            normalization_version
            not in SUPPORTED_ENTITY_NORMALIZATION_VERSIONS
        ):
            raise ValueError(
                "지원하지 않는 ENTITY_NORMALIZATION_VERSION입니다. "
                f"현재 값={normalization_version}, "
                f"지원 값={sorted(SUPPORTED_ENTITY_NORMALIZATION_VERSIONS)}"
            )

    else:
        normalization_version = "baseline"

    
    config.create_output_directories()

    required_paths = [
        config.ARTICLES_WITH_CATEGORY_PATH,
        config.TRAIN_USED_ARTICLE_IDS_PATH,
    ]

    missing_paths = [
        str(path)
        for path in required_paths
        if not Path(path).exists()
    ]

    if missing_paths:
        raise FileNotFoundError(
            "Entity Processing 입력 파일이 없습니다: "
            + ", ".join(missing_paths)
        )

    articles = pl.read_parquet(
        config.ARTICLES_WITH_CATEGORY_PATH
    ).select(
        [
            "article_id",
            "ner_clusters",
            "entity_groups",
        ]
    )

    train_used_ids_df = pl.read_parquet(
        config.TRAIN_USED_ARTICLE_IDS_PATH
    ).select(
        "article_id"
    )

    train_used_article_ids = {
        int(article_id)
        for article_id in train_used_ids_df
        .get_column("article_id")
        .to_list()
    }

    article_ids = {
        int(article_id)
        for article_id in articles
        .get_column("article_id")
        .to_list()
    }

    missing_train_ids = sorted(
        train_used_article_ids - article_ids
    )

    if missing_train_ids:
        raise ValueError(
            "Train-used article 중 articles_with_category에 없는 ID가 있습니다. "
            f"예시={missing_train_ids[:10]}"
        )
    
        

    mentions = _build_entity_mention_rows(
        articles=articles,
        train_used_article_ids=train_used_article_ids,
    )

    train_mentions = mentions.filter(
        pl.col("is_train_used")
    )

    # -------------------------------------------------------------------------
    # v1 mapping과 v2 mapping을 따로 관리한다.
    #
    # 최종 적용 순서:
    #
    # baseline key
    #     ↓
    # possessive_mapping
    #     ↓
    # hyphen_mapping
    #     ↓
    # final canonical key
    # -------------------------------------------------------------------------

    possessive_mapping: dict[
        str,
        str,
    ] = {}

    hyphen_mapping: dict[
        str,
        str,
    ] = {}

    empty_mapping_schema = {
        "entity_group": pl.String,
        "variant_entity": pl.String,
        "base_entity": pl.String,
        "variant_entity_key": pl.String,
        "canonical_entity_key": pl.String,
        "variant_article_df": pl.Int64,
        "base_article_df": pl.Int64,
        "rule": pl.String,
    }

    mapping_df = pl.DataFrame(
        schema=empty_mapping_schema
    )

    mapping_stats = {
        "possessive_candidate_count": 0,
        "possessive_accepted_count": 0,
        "possessive_rejected_count": 0,

        "hyphen_collision_group_count": 0,
        "hyphen_candidate_count": 0,
        "hyphen_accepted_count": 0,
        "hyphen_rejected_count": 0,
    }

    if mode == "normalize_only":

        # -------------------------------------------------------------
        # STEP 1. 기존 v1 possessive
        # -------------------------------------------------------------
        (
            possessive_mapping,
            possessive_mapping_df,
            possessive_stats,
        ) = _build_safe_normalization_mapping(
            train_mentions=train_mentions
        )

        mapping_stats.update(
            possessive_stats
        )

        # v1까지만 실행하는 경우에는
        # 기존과 동일하게 possessive mapping만 저장
        mapping_df = (
            possessive_mapping_df
        )

        # -------------------------------------------------------------
        # STEP 2. v2일 때만 safe hyphen mapping 추가
        # -------------------------------------------------------------
        if normalization_version == "v2":
            (
                hyphen_mapping,
                hyphen_mapping_df,
                hyphen_stats,
            ) = _build_safe_hyphen_v2_mapping(
                train_mentions=train_mentions,
                possessive_mapping=possessive_mapping,
            )

            mapping_stats.update(
                hyphen_stats
            )

            # entity_normalization_map.parquet에는
            # v1 + v2 mapping을 모두 저장
            if hyphen_mapping_df.height > 0:
                mapping_df = (
                    pl.concat(
                        [
                            possessive_mapping_df,
                            hyphen_mapping_df,
                        ],
                        how="vertical",
                    )
                    .sort(
                        [
                            "rule",
                            "entity_group",
                            "variant_entity",
                        ]
                    )
                )

    output_rows: list[dict[str, Any]] = []

    changed_mention_count = 0
    changed_article_ids: set[int] = set()

    for row in mentions.iter_rows(named=True):
        baseline_entity_key = str(
            row["baseline_entity_key"]
        )

        # -------------------------------------------------------------
        # STEP 1. 기존 Safe Normalization v1
        #
        # 예:
        #   PER::mette frederiksens
        #   -> PER::mette frederiksen
        # -------------------------------------------------------------
        after_possessive_key = (
            possessive_mapping.get(
                baseline_entity_key,
                baseline_entity_key,
            )
        )

        # -------------------------------------------------------------
        # STEP 2. Safe Normalization v2
        #
        # v2가 비활성화된 경우 hyphen_mapping={}이므로
        # after_possessive_key가 그대로 유지된다.
        #
        # 예:
        #   ORG::jyllands - posten
        #   -> ORG::jyllands posten
        # -------------------------------------------------------------
        canonical_entity_key = (
            hyphen_mapping.get(
                after_possessive_key,
                after_possessive_key,
            )
        )

        possessive_changed = (
            after_possessive_key
            != baseline_entity_key
        )

        hyphen_changed = (
            canonical_entity_key
            != after_possessive_key
        )

        # -------------------------------------------------------------
        # 어떤 normalization이 적용됐는지 기록
        # -------------------------------------------------------------
        if (
            possessive_changed
            and hyphen_changed
        ):
            processing_method = (
                "safe_possessive_v1"
                "+safe_hyphen_v2"
            )

        elif possessive_changed:
            processing_method = (
                "safe_possessive_v1"
            )

        elif hyphen_changed:
            processing_method = (
                "safe_hyphen_v2"
            )

        else:
            processing_method = (
                "baseline"
            )

        if (
            canonical_entity_key
            != baseline_entity_key
        ):
            changed_mention_count += 1

            changed_article_ids.add(
                int(row["article_id"])
            )

        # TYPE::surface에서 surface만 분리.
        # split(..., 1)이라 surface 내부 '::'가 있더라도 첫 구분자만 사용.
        canonical_entity = canonical_entity_key.split(
            "::",
            1,
        )[1]

        output_rows.append(
            {
                "article_id": int(row["article_id"]),
                "entity_position": int(row["entity_position"]),
                "raw_entity": str(row["raw_entity"]),
                "raw_entity_group": str(row["raw_entity_group"]),
                "entity_group": str(row["entity_group"]),
                "normalized_entity": str(row["normalized_entity"]),
                "baseline_entity_key": baseline_entity_key,
                "canonical_entity": canonical_entity,
                "canonical_entity_key": canonical_entity_key,
                "processing_method": processing_method,
                "is_train_used": bool(row["is_train_used"]),
            }
        )

    output_schema = {
        "article_id": pl.Int64,
        "entity_position": pl.Int64,
        "raw_entity": pl.String,
        "raw_entity_group": pl.String,
        "entity_group": pl.String,
        "normalized_entity": pl.String,
        "baseline_entity_key": pl.String,
        "canonical_entity": pl.String,
        "canonical_entity_key": pl.String,
        "processing_method": pl.String,
        "is_train_used": pl.Boolean,
    }

    if output_rows:
        article_entities_df = (
            pl.DataFrame(
                output_rows,
                schema=output_schema,
            )
            .sort(
                [
                    "article_id",
                    "entity_position",
                ]
            )
        )
    else:
        article_entities_df = pl.DataFrame(
            schema=output_schema
        )

    article_entities_df.write_parquet(
        config.ARTICLE_ENTITIES_PATH,
        compression="zstd",
    )

    mapping_df.write_parquet(
        config.ENTITY_NORMALIZATION_MAP_PATH,
        compression="zstd",
    )

    # article 단위 canonical set 수. 같은 기사 내부 alias가 normalization으로
    # 하나로 합쳐졌는지 확인할 수 있는 진단 통계.
    baseline_unique_pair_count = (
        mentions.select(
            [
                "article_id",
                "baseline_entity_key",
            ]
        )
        .unique()
        .height
    )

    canonical_unique_pair_count = (
        article_entities_df.select(
            [
                "article_id",
                "canonical_entity_key",
            ]
        )
        .unique()
        .height
    )

    return {
        "status": "SUCCESS",
        "entity_processing_mode": mode,
        "fit_split": "train_used_articles_only",
        "normalization_version": normalization_version,
        "valid_article_count": int(
            articles.height
        ),
        "train_used_article_count": int(
            len(train_used_article_ids)
        ),
        "entity_mention_count": int(
            article_entities_df.height
        ),
        "changed_mention_count": int(
            changed_mention_count
        ),
        "changed_article_count": int(
            len(changed_article_ids)
        ),
        "baseline_article_entity_pair_count": int(
            baseline_unique_pair_count
        ),
        "canonical_article_entity_pair_count": int(
            canonical_unique_pair_count
        ),
        "article_entity_pair_reduction_count": int(
            baseline_unique_pair_count
            - canonical_unique_pair_count
        ),
        **mapping_stats,
        "article_entities_path": str(
            config.ARTICLE_ENTITIES_PATH
        ),
        "entity_normalization_map_path": str(
            config.ENTITY_NORMALIZATION_MAP_PATH
        ),
    }

# 5. Event clustering lookup loader
def load_canonical_entity_lookup(
        article_entities_path: Path | None = None,
)-> dict[int, set[str]]:
    """
    articel_entities.parquet을 event clustering에서 사용하기 쉬운
    article_id -> set[canonical_entity_key] 형태로 읽는다

    entity가 없는 article은 이 lookup에 없을 수 있으므로 caller는
    lookup.get(article_id, set())사용
    """

    if article_entities_path is None:
        article_entities_path = (
            config.ARTICLE_ENTITIES_PATH
        )

    if not Path(article_entities_path).exists():
        raise FileNotFoundError(
            "article_entities.parquet이 없습니다. "
            "build_article_entities()를 먼저 실행하세요. "
            f"경로={article_entities_path}"
        )

    entity_df = pl.read_parquet(
        article_entities_path
    ).select(
        [
            "article_id",
            "canonical_entity_key",
        ]
    )
    lookup: dict[int, set[str]] = {}

    for row in entity_df.iter_rows(named=True):
        article_id = int(row["article_id"])
        entity_key = row["canonical_entity_key"]

        if entity_key is None: continue 

        lookup.setdefault(article_id, set()).add(str(entity_key))

    return lookup 

if __name__ == "__main__":
    from pprint import pprint

    pprint(
        build_article_entities()
    )


