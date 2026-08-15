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

    normalization_mapping: dict[str, str] = {}

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
    }
    
    if mode == "normalize_only":
        (
            normalization_mapping,
            mapping_df,
            mapping_stats,
        ) = _build_safe_normalization_mapping(
            train_mentions=train_mentions
        )

    output_rows: list[dict[str, Any]] = []

    changed_mention_count = 0
    changed_article_ids: set[int] = set()

    for row in mentions.iter_rows(named=True):
        baseline_entity_key = str(
            row["baseline_entity_key"]
        )

        canonical_entity_key = (
            normalization_mapping.get(
                baseline_entity_key,
                baseline_entity_key,
            )
        )

        if canonical_entity_key != baseline_entity_key:
            processing_method = "safe_possessive_v1"
            changed_mention_count += 1
            changed_article_ids.add(
                int(row["article_id"])
            )
        else:
            processing_method = "baseline"

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


