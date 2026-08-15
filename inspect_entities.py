from pathlib import Path
import unicodedata
from itertools import combinations

import polars as pl


# =============================================================================
# 0. 경로 설정
# =============================================================================

ARTICLES_PATH = Path(
    "data/raw/articles.parquet"
)

# Before/After 실험의 Train article universe를 완전히 동일하게 유지하기 위해
# 이미 보존해 둔 baseline snapshot의 Train-used article ID를 사용한다.
BASELINE_TRAIN_IDS_PATH = Path(
    "data/output/experiments/baseline/model_inputs/train_used_article_ids.parquet"
)

OUTPUT_DIR = Path(
    "data/output/experiments/entity_inspection"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# 1. 현재 baseline에서 사용하는 normalization
#
# 중요:
# 아직 새로운 Entity Normalization을 적용하는 단계가 아니다.
#
# 현재 baseline이 실제로 어떤 entity representation을 사용하고 있는지
# 동일하게 재현해서 분석하기 위한 함수다.
# =============================================================================

def baseline_normalize_entity(entity) -> str | None:
    """
    현재 baseline entity 문자열 normalization.

    예:
        "  Ekstra   Bladet "
        -> "ekstra bladet"

        "TikTok"
        -> "tiktok"

    수행:
        1. Unicode NFKC
        2. 앞뒤 공백 제거
        3. 연속 공백 정리
        4. lowercase
    """

    if entity is None:
        return None

    text = unicodedata.normalize(
        "NFKC",
        str(entity),
    )

    text = " ".join(
        text.strip().split()
    )

    text = text.lower()

    if text == "":
        return None

    return text


def normalize_group(group) -> str | None:
    """
    Entity group normalization.

    예:
        "per" -> "PER"
        " org " -> "ORG"
    """

    if group is None:
        return None

    text = unicodedata.normalize(
        "NFKC",
        str(group),
    )

    text = text.strip().upper()

    if text == "":
        return None

    return text


# =============================================================================
# 2. 입력 파일 존재 확인
# =============================================================================

if not ARTICLES_PATH.exists():
    raise FileNotFoundError(
        f"articles.parquet를 찾을 수 없습니다: {ARTICLES_PATH}"
    )

if not BASELINE_TRAIN_IDS_PATH.exists():
    raise FileNotFoundError(
        "baseline Train-used article ID를 찾을 수 없습니다.\n"
        f"확인 경로: {BASELINE_TRAIN_IDS_PATH}"
    )


# =============================================================================
# 3. 데이터 로드
# =============================================================================

articles = pl.read_parquet(
    ARTICLES_PATH,
    columns=[
        "article_id",
        "title",
        "ner_clusters",
        "entity_groups",
    ],
)

train_ids = (
    pl.read_parquet(
        BASELINE_TRAIN_IDS_PATH
    )
    .select("article_id")
    .to_series()
    .to_list()
)

train_id_set = set(train_ids)


print()
print("=" * 100)
print("1. articles schema")
print("=" * 100)

print(articles.schema)

print()
print(
    f"전체 article 수          : {articles.height:,}"
)

print(
    f"baseline Train article 수: {len(train_id_set):,}"
)


# =============================================================================
# 4. Article-level NER 기본 상태
#
# 확인:
#   - ner_clusters null 여부
#   - entity_groups null 여부
#   - entity가 하나도 없는 기사
#   - 두 list 길이가 다른 기사
#
# ner_clusters[i]와 entity_groups[i]를 안전하게 pairing할 수 있는지 검증한다.
# =============================================================================

article_stats = articles.select(

    pl.len()
    .alias("article_count"),

    pl.col("ner_clusters")
    .is_null()
    .sum()
    .alias("ner_list_null_count"),

    pl.col("entity_groups")
    .is_null()
    .sum()
    .alias("group_list_null_count"),

    (
        pl.col("ner_clusters")
        .list.len()
        .fill_null(0)
        == 0
    )
    .sum()
    .alias("ner_empty_article_count"),

    (
        pl.col("entity_groups")
        .list.len()
        .fill_null(0)
        == 0
    )
    .sum()
    .alias("group_empty_article_count"),

    (
        pl.col("ner_clusters")
        .list.len()
        .fill_null(0)
        !=
        pl.col("entity_groups")
        .list.len()
        .fill_null(0)
    )
    .sum()
    .alias("length_mismatch_count"),
)


print()
print("=" * 100)
print("2. Article-level NER 상태")
print("=" * 100)

print(article_stats)


# =============================================================================
# 5. ner_clusters[i] ↔ entity_groups[i]를 개별 mention row로 펼치기
#
# 예:
#
# article_id = 100
# ner_clusters  = ["Joe Biden", "USA"]
# entity_groups = ["PER", "LOC"]
#
# ↓
#
# 100 | Joe Biden | PER
# 100 | USA       | LOC
# =============================================================================

records = []


for row in articles.iter_rows(named=True):

    article_id = row["article_id"]

    entities = row["ner_clusters"]
    groups = row["entity_groups"]

    if entities is None or groups is None:
        continue

    if len(entities) != len(groups):
        continue

    for entity, group in zip(
        entities,
        groups,
    ):

        normalized_entity = baseline_normalize_entity(
            entity
        )

        normalized_group = normalize_group(
            group
        )

        if normalized_entity is None:
            continue

        if normalized_group is None:
            continue

        records.append(
            {
                "article_id": int(article_id),
                "title": row["title"],

                "is_train_used":
                    article_id in train_id_set,

                # 원본 NER 결과
                "raw_entity":
                    str(entity),

                "raw_entity_group":
                    str(group),

                # 현재 baseline normalization 결과
                "entity_group":
                    normalized_group,

                "normalized_entity":
                    normalized_entity,

                # 현재 Event clustering에서 사용하는 개념과 동일
                "baseline_entity_key":
                    f"{normalized_group}::{normalized_entity}",
            }
        )


pairs = pl.DataFrame(
    records,
    schema={
        "article_id": pl.Int64,
        "title": pl.String,
        "is_train_used": pl.Boolean,
        "raw_entity": pl.String,
        "raw_entity_group": pl.String,
        "entity_group": pl.String,
        "normalized_entity": pl.String,
        "baseline_entity_key": pl.String,
    },
)


train_pairs = pairs.filter(
    pl.col("is_train_used")
)


print()
print("=" * 100)
print("3. 실제 Entity mention 수")
print("=" * 100)

print(
    f"전체 entity mention 수 : {pairs.height:,}"
)

print(
    f"Train entity mention 수: {train_pairs.height:,}"
)


# =============================================================================
# 6. Entity Group 종류
#
# PER / ORG / LOC만 있다고 가정하지 않고 실제 데이터를 확인한다.
# =============================================================================

group_stats_all = (
    pairs
    .group_by(
        "entity_group"
    )
    .agg(

        pl.len()
        .alias("mention_count"),

        pl.col("article_id")
        .n_unique()
        .alias("article_count"),

        pl.col("normalized_entity")
        .n_unique()
        .alias("unique_entity_count"),

    )
    .sort(
        "mention_count",
        descending=True,
    )
)


print()
print("=" * 100)
print("4. entity_groups 실제 종류 - 전체")
print("=" * 100)

print(group_stats_all)


group_stats_train = (
    train_pairs
    .group_by(
        "entity_group"
    )
    .agg(

        pl.len()
        .alias("mention_count"),

        pl.col("article_id")
        .n_unique()
        .alias("article_count"),

        pl.col("normalized_entity")
        .n_unique()
        .alias("unique_entity_count"),

    )
    .sort(
        "mention_count",
        descending=True,
    )
)


print()
print("=" * 100)
print("5. entity_groups 실제 종류 - baseline Train")
print("=" * 100)

print(group_stats_train)


group_stats_train.write_parquet(
    OUTPUT_DIR / "entity_group_stats.parquet"
)


# =============================================================================
# 7. 실제 raw pair 예시
# =============================================================================

print()
print("=" * 100)
print("6. 실제 ner_clusters / entity_groups pair 예시")
print("=" * 100)

sample = (
    train_pairs
    .select(
        "article_id",
        "title",
        "raw_entity",
        "raw_entity_group",
        "normalized_entity",
        "baseline_entity_key",
    )
    .head(50)
)

print(sample)


# =============================================================================
# 8. Entity별 Train DF 계산
#
# article_df:
#   이 entity가 서로 다른 Train article 몇 개에 등장했는지
#
# mention_count:
#   중복 mention까지 포함해서 전체 몇 번 등장했는지
# =============================================================================

entity_df = (
    train_pairs
    .group_by(
        "entity_group",
        "normalized_entity",
        "baseline_entity_key",
    )
    .agg(

        pl.len()
        .alias("mention_count"),

        pl.col("article_id")
        .n_unique()
        .alias("article_df"),

        pl.col("raw_entity")
        .unique()
        .alias("raw_forms"),

    )
)


entity_df.write_parquet(
    OUTPUT_DIR / "entity_df.parquet"
)


# =============================================================================
# 9. Train DF TOP Entity
# =============================================================================

top_entities = (
    entity_df
    .sort(
        [
            "article_df",
            "mention_count",
        ],
        descending=[
            True,
            True,
        ],
    )
    .head(50)
)


print()
print("=" * 100)
print("7. Train에서 article DF가 높은 Entity TOP 50")
print("=" * 100)

print(top_entities)


# =============================================================================
# 10. 현재 baseline normalization으로 이미 합쳐지고 있는 raw 표현
#
# 예:
#
# TikTok
# Tiktok
# tiktok
#
# ↓
#
# ORG::tiktok
#
# 이건 새로운 Entity Linking의 효과가 아니라
# 이미 baseline normalization이 해결하고 있는 영역이다.
# =============================================================================

normalization_variants = (
    train_pairs
    .group_by(
        "entity_group",
        "normalized_entity",
    )
    .agg(

        pl.col("raw_entity")
        .n_unique()
        .alias("raw_variant_count"),

        pl.col("raw_entity")
        .unique()
        .alias("raw_variants"),

        pl.col("article_id")
        .n_unique()
        .alias("article_df"),

    )
    .filter(
        pl.col("raw_variant_count") > 1
    )
    .sort(
        [
            "raw_variant_count",
            "article_df",
        ],
        descending=[
            True,
            True,
        ],
    )
)


print()
print("=" * 100)
print("8. 현재 baseline normalization으로 이미 합쳐지는 raw 표현 TOP 100")
print("=" * 100)

print(
    normalization_variants.head(100)
)


normalization_variants.write_parquet(
    OUTPUT_DIR / "baseline_normalization_variants.parquet"
)


# =============================================================================
# 11. Entity type별 TOP Entity
# =============================================================================

for target_group in [
    "PER",
    "ORG",
    "LOC",
    "PROD",
    "MISC",
    "EVENT",
]:

    group_entities = (
        entity_df
        .filter(
            pl.col("entity_group")
            == target_group
        )
        .select(
            "normalized_entity",
            "raw_forms",
            "article_df",
            "mention_count",
        )
        .sort(
            "article_df",
            descending=True,
        )
        .head(40)
    )

    print()
    print("=" * 100)
    print(
        f"9. {target_group} 실제 Entity TOP 40"
    )
    print("=" * 100)

    print(group_entities)


# =============================================================================
# 12. Fragmentation Diagnostic ①
#
# DF=1 Entity 비율
#
# article_df == 1이면 현재 baseline에서는
# 해당 entity key 자체로는 다른 기사와 entity overlap을 만들 수 없다.
#
# 단:
# DF=1이라고 해서 전부 잘못된 entity는 아니다.
#
# 실제로 corpus에서 단 한 번만 등장한 정상 entity도 포함된다.
#
# 따라서 이 수치는
# "Entity Fragmentation Indicator"로 해석한다.
# =============================================================================

total_entity_count = entity_df.height

df1_entities = entity_df.filter(
    pl.col("article_df") == 1
)

df1_entity_count = df1_entities.height


if total_entity_count > 0:
    df1_entity_ratio = (
        df1_entity_count
        / total_entity_count
    )
else:
    df1_entity_ratio = 0.0


total_mentions = (
    entity_df
    ["mention_count"]
    .sum()
)

df1_mentions = (
    df1_entities
    ["mention_count"]
    .sum()
)


if total_mentions > 0:
    df1_mention_ratio = (
        df1_mentions
        / total_mentions
    )
else:
    df1_mention_ratio = 0.0


print()
print("=" * 100)
print("10. DF=1 Entity Fragmentation")
print("=" * 100)

print(
    f"전체 unique entity   : {total_entity_count:,}"
)

print(
    f"DF=1 entity          : {df1_entity_count:,}"
)

print(
    f"DF=1 entity 비율     : {df1_entity_ratio:.2%}"
)

print()

print(
    f"전체 entity mention  : {total_mentions:,}"
)

print(
    f"DF=1 entity mention  : {df1_mentions:,}"
)

print(
    f"DF=1 mention 비율    : {df1_mention_ratio:.2%}"
)


# -----------------------------------------------------------------------------
# Group별 DF=1 비율
# -----------------------------------------------------------------------------

df1_by_group = (
    entity_df
    .group_by(
        "entity_group"
    )
    .agg(

        pl.len()
        .alias("unique_entity_count"),

        (
            pl.col("article_df") == 1
        )
        .sum()
        .alias("df1_entity_count"),

    )
    .with_columns(

        (
            pl.col("df1_entity_count")
            / pl.col("unique_entity_count")
        )
        .alias("df1_entity_ratio")

    )
    .sort(
        "df1_entity_ratio",
        descending=True,
    )
)


print()
print("Group별 DF=1 비율")
print(df1_by_group)


df1_by_group.write_parquet(
    OUTPUT_DIR / "df1_by_group.parquet"
)

df1_entities.write_parquet(
    OUTPUT_DIR / "df1_entities.parquet"
)


# =============================================================================
# 13. Fragmentation Diagnostic ②
#
# 동일 TYPE 안의 Token 포함관계
#
# 예:
#
# PER::putin
# PER::vladimir putin
#
# "putin"이라는 하나의 token이 full entity 안에 존재한다.
#
# 중요한 점:
# 이것은 자동 merge 규칙이 아니다.
# 단지 Entity Linking 후보를 찾기 위한 진단이다.
# =============================================================================

entities_by_group = {}


for row in entity_df.iter_rows(named=True):

    group = row["entity_group"]

    entities_by_group.setdefault(
        group,
        []
    ).append(
        {
            "entity":
                row["normalized_entity"],

            "article_df":
                row["article_df"],
        }
    )


containment_records = []


for group, group_entities in entities_by_group.items():

    # 한 단어 entity
    single_entities = [
        item
        for item in group_entities
        if len(
            item["entity"].split()
        ) == 1
    ]

    # 2단어 이상 entity
    multi_entities = [
        item
        for item in group_entities
        if len(
            item["entity"].split()
        ) >= 2
    ]

    for short_item in single_entities:

        short_entity = short_item["entity"]
        short_df = short_item["article_df"]

        for full_item in multi_entities:

            full_entity = full_item["entity"]
            full_df = full_item["article_df"]

            full_tokens = set(
                full_entity.split()
            )

            # substring이 아니라 token 단위로 검사한다.
            if short_entity not in full_tokens:
                continue

            containment_records.append(
                {
                    "entity_group":
                        group,

                    "short_entity":
                        short_entity,

                    "full_entity":
                        full_entity,

                    "short_article_df":
                        short_df,

                    "full_article_df":
                        full_df,
                }
            )


containment_schema = {
    "entity_group": pl.String,
    "short_entity": pl.String,
    "full_entity": pl.String,
    "short_article_df": pl.UInt32,
    "full_article_df": pl.UInt32,
}


if containment_records:

    containment_candidates = pl.DataFrame(
        containment_records
    )

else:

    containment_candidates = pl.DataFrame(
        schema=containment_schema
    )


containment_candidates = (
    containment_candidates
    .sort(
        [
            "full_article_df",
            "short_article_df",
        ],
        descending=[
            True,
            True,
        ],
    )
)


print()
print("=" * 100)
print("11. Same-Type Token Containment")
print("=" * 100)

print(
    f"포함관계 candidate pair 수: "
    f"{containment_candidates.height:,}"
)


if containment_candidates.height > 0:

    participating_entities = set()

    for row in containment_candidates.iter_rows(
        named=True
    ):

        participating_entities.add(
            (
                row["entity_group"],
                row["short_entity"],
            )
        )

        participating_entities.add(
            (
                row["entity_group"],
                row["full_entity"],
            )
        )

    containment_entity_count = len(
        participating_entities
    )

else:

    containment_entity_count = 0


if total_entity_count > 0:

    containment_entity_ratio = (
        containment_entity_count
        / total_entity_count
    )

else:

    containment_entity_ratio = 0.0


print(
    f"포함관계에 참여한 unique entity 수: "
    f"{containment_entity_count:,}"
)

print(
    f"전체 unique entity 대비 비율      : "
    f"{containment_entity_ratio:.2%}"
)

print()

print(
    containment_candidates.head(150)
)


containment_candidates.write_parquet(
    OUTPUT_DIR / "token_containment_candidates.parquet"
)


# =============================================================================
# 14. Fragmentation Diagnostic ③
#
# 동일 normalized surface가 서로 다른 TYPE으로 존재하는 경우
#
# 예:
#
# PER::bakhmut
# LOC::bakhmut
#
# 현재 TYPE::name 구조에서는 서로 완전히 다른 entity key다.
#
# 하지만 이것도 자동으로 NER 오류라고 판단하지 않는다.
# 실제 동명이 entity가 존재할 수도 있기 때문이다.
# =============================================================================

cross_type = (
    train_pairs
    .group_by(
        "normalized_entity"
    )
    .agg(

        pl.col("entity_group")
        .unique()
        .sort()
        .alias("entity_groups"),

        pl.col("entity_group")
        .n_unique()
        .alias("type_count"),

        pl.col("article_id")
        .n_unique()
        .alias("article_df"),

        pl.len()
        .alias("mention_count"),

    )
    .filter(
        pl.col("type_count") > 1
    )
    .sort(
        [
            "article_df",
            "mention_count",
        ],
        descending=[
            True,
            True,
        ],
    )
)


all_surface_count = (
    train_pairs
    ["normalized_entity"]
    .n_unique()
)

cross_type_surface_count = (
    cross_type.height
)


if all_surface_count > 0:

    cross_type_ratio = (
        cross_type_surface_count
        / all_surface_count
    )

else:

    cross_type_ratio = 0.0


print()
print("=" * 100)
print("12. Same Surface / Different TYPE")
print("=" * 100)

print(
    f"전체 normalized surface : "
    f"{all_surface_count:,}"
)

print(
    f"TYPE 충돌 surface       : "
    f"{cross_type_surface_count:,}"
)

print(
    f"TYPE 충돌 surface 비율  : "
    f"{cross_type_ratio:.2%}"
)

print()

print(
    cross_type.head(100)
)


cross_type.write_parquet(
    OUTPUT_DIR / "cross_type_candidates.parquet"
)


# =============================================================================
# 15. Cross-Type 조합별 개수
#
# 예:
#
# PER + LOC
# PER + ORG
# ORG + LOC
#
# 어떤 TYPE 사이에서 동일 surface 충돌이 많이 발생하는지 확인한다.
# =============================================================================

cross_type_pair_records = []


for row in cross_type.iter_rows(named=True):

    groups = row["entity_groups"]

    for group_a, group_b in combinations(
        groups,
        2,
    ):

        cross_type_pair_records.append(
            {
                "type_a":
                    group_a,

                "type_b":
                    group_b,

                "normalized_entity":
                    row["normalized_entity"],
            }
        )


if cross_type_pair_records:

    cross_type_pairs = (
        pl.DataFrame(
            cross_type_pair_records
        )
        .group_by(
            "type_a",
            "type_b",
        )
        .agg(
            pl.len()
            .alias("surface_count")
        )
        .sort(
            "surface_count",
            descending=True,
        )
    )

else:

    cross_type_pairs = pl.DataFrame(
        schema={
            "type_a": pl.String,
            "type_b": pl.String,
            "surface_count": pl.UInt32,
        }
    )


print()
print("TYPE 조합별 동일 surface 충돌")
print(
    cross_type_pairs
)


cross_type_pairs.write_parquet(
    OUTPUT_DIR / "cross_type_pair_stats.parquet"
)


# =============================================================================
# 16. Fragmentation Diagnostic ④
#
# 포함관계 entity가 같은 기사 안에서 실제 co-occurrence하는지 확인
#
# 예:
#
# 기사 A:
#
# PER::vladimir putin
# PER::putin
#
# corpus 전체에서 둘 다 존재하기만 하는 것보다
# 같은 기사 안에서 같이 등장하면
# alias 가능성의 훨씬 강한 evidence가 된다.
# =============================================================================

article_entity_sets = {}


for row in train_pairs.select(
    "article_id",
    "entity_group",
    "normalized_entity",
).iter_rows(named=True):

    article_id = row["article_id"]

    article_entity_sets.setdefault(
        article_id,
        set()
    ).add(
        (
            row["entity_group"],
            row["normalized_entity"],
        )
    )


cooccurrence_records = []


for row in containment_candidates.iter_rows(
    named=True
):

    group = row["entity_group"]
    short_entity = row["short_entity"]
    full_entity = row["full_entity"]

    cooccurrence_count = 0
    example_article_ids = []

    for article_id, entity_set in (
        article_entity_sets.items()
    ):

        short_key = (
            group,
            short_entity,
        )

        full_key = (
            group,
            full_entity,
        )

        if (
            short_key in entity_set
            and
            full_key in entity_set
        ):

            cooccurrence_count += 1

            if len(example_article_ids) < 10:
                example_article_ids.append(
                    article_id
                )

    if cooccurrence_count == 0:
        continue

    cooccurrence_records.append(
        {
            "entity_group":
                group,

            "short_entity":
                short_entity,

            "full_entity":
                full_entity,

            "short_article_df":
                row["short_article_df"],

            "full_article_df":
                row["full_article_df"],

            "cooccurrence_article_count":
                cooccurrence_count,

            "example_article_ids":
                example_article_ids,
        }
    )


cooccurrence_schema = {
    "entity_group": pl.String,
    "short_entity": pl.String,
    "full_entity": pl.String,
    "short_article_df": pl.UInt32,
    "full_article_df": pl.UInt32,
    "cooccurrence_article_count": pl.Int64,
    "example_article_ids": pl.List(
        pl.Int64
    ),
}


if cooccurrence_records:

    containment_cooccurrence = (
        pl.DataFrame(
            cooccurrence_records
        )
        .sort(
            [
                "cooccurrence_article_count",
                "full_article_df",
            ],
            descending=[
                True,
                True,
            ],
        )
    )

else:

    containment_cooccurrence = pl.DataFrame(
        schema=cooccurrence_schema
    )


print()
print("=" * 100)
print("13. Token Containment + Same Article Co-occurrence")
print("=" * 100)

print(
    f"전체 containment pair       : "
    f"{containment_candidates.height:,}"
)

print(
    f"같은 기사에서도 등장한 pair : "
    f"{containment_cooccurrence.height:,}"
)


if containment_candidates.height > 0:

    cooccurrence_pair_ratio = (
        containment_cooccurrence.height
        / containment_candidates.height
    )

else:

    cooccurrence_pair_ratio = 0.0


print(
    f"co-occurrence pair 비율      : "
    f"{cooccurrence_pair_ratio:.2%}"
)

print()

print(
    containment_cooccurrence.head(100)
)


containment_cooccurrence.write_parquet(
    OUTPUT_DIR
    / "token_containment_cooccurrence.parquet"
)


# =============================================================================
# 17. Possessive / trailing-s 후보
#
# 예:
#
# ekstra bladet
# ekstra bladets
#
# 단:
#
# 모든 끝의 s를 제거하면 안 된다.
#
# 실제 base entity가 corpus 안에 존재하는 경우만
# "조사 후보"로 만든다.
#
# 자동 normalization 규칙이 아니다.
# =============================================================================

entity_lookup = {
    (
        row["entity_group"],
        row["normalized_entity"],
    )
    for row in entity_df.iter_rows(
        named=True
    )
}


possessive_records = []


for row in entity_df.iter_rows(
    named=True
):

    group = row["entity_group"]
    entity = row["normalized_entity"]

    if len(entity) <= 2:
        continue

    if not entity.endswith("s"):
        continue

    base_candidate = entity[:-1]

    if (
        group,
        base_candidate,
    ) not in entity_lookup:
        continue

    possessive_records.append(
        {
            "entity_group":
                group,

            "variant":
                entity,

            "base_candidate":
                base_candidate,

            "variant_article_df":
                row["article_df"],
        }
    )


possessive_schema = {
    "entity_group": pl.String,
    "variant": pl.String,
    "base_candidate": pl.String,
    "variant_article_df": pl.UInt32,
}


if possessive_records:

    possessive_candidates = (
        pl.DataFrame(
            possessive_records
        )
        .sort(
            "variant_article_df",
            descending=True,
        )
    )

else:

    possessive_candidates = pl.DataFrame(
        schema=possessive_schema
    )


print()
print("=" * 100)
print("14. Trailing-s / Possessive 후보")
print("=" * 100)

print(
    f"후보 pair 수: "
    f"{possessive_candidates.height:,}"
)

print()

print(
    possessive_candidates.head(100)
)


possessive_candidates.write_parquet(
    OUTPUT_DIR / "possessive_candidates.parquet"
)


# =============================================================================
# 18. 최종 진단 Summary
# =============================================================================

summary = {
    "total_article_count":
        articles.height,

    "baseline_train_article_count":
        len(train_id_set),

    "total_entity_mention_count":
        pairs.height,

    "train_entity_mention_count":
        train_pairs.height,

    "total_unique_train_entity_count":
        total_entity_count,

    "df1_entity_count":
        df1_entity_count,

    "df1_entity_ratio":
        df1_entity_ratio,

    "df1_mention_count":
        df1_mentions,

    "df1_mention_ratio":
        df1_mention_ratio,

    "token_containment_pair_count":
        containment_candidates.height,

    "token_containment_participating_entity_count":
        containment_entity_count,

    "token_containment_participating_entity_ratio":
        containment_entity_ratio,

    "containment_cooccurrence_pair_count":
        containment_cooccurrence.height,

    "containment_cooccurrence_pair_ratio":
        cooccurrence_pair_ratio,

    "cross_type_surface_count":
        cross_type_surface_count,

    "cross_type_surface_ratio":
        cross_type_ratio,

    "possessive_candidate_count":
        possessive_candidates.height,
}


summary_df = pl.DataFrame(
    [
        {
            "metric":
                key,

            "value":
                str(value),
        }
        for key, value in summary.items()
    ]
)


summary_df.write_csv(
    OUTPUT_DIR / "entity_diagnostic_summary.csv"
)


# =============================================================================
# 19. 전체 pair 데이터도 저장
#
# 이후 entity_processing.py 구현 전 qualitative inspection에 재사용한다.
# =============================================================================

pairs.write_parquet(
    OUTPUT_DIR / "entity_pairs.parquet"
)


# =============================================================================
# 20. 완료 출력
# =============================================================================

print()
print("=" * 100)
print("15. Entity Diagnostic Summary")
print("=" * 100)

for key, value in summary.items():

    if isinstance(value, float):

        print(
            f"{key:50s}: {value:.4f}"
        )

    else:

        print(
            f"{key:50s}: {value}"
        )


print()
print("=" * 100)
print("Entity Inspection 완료")
print("=" * 100)

print()

print(
    f"결과 저장 위치: {OUTPUT_DIR}"
)

print()

print("생성 파일:")

for path in sorted(
    OUTPUT_DIR.iterdir()
):

    print(
        f"  - {path.name}"
    )

