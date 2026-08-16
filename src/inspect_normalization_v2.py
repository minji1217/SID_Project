from __future__ import annotations

"""
inspect_normalization_v2.py

Safe Normalization v2 후보를 실제 Train entity에서 진단하는 실험용 스크립트.

왜 바로 entity_processing.py를 수정하지 않는가?
------------------------------------------------------------
v1에서는 Danish possessive(-s) 규칙을 적용하기 전에 실제 후보를 조사했고,
그 결과를 검증한 뒤 안전한 규칙만 구현했다.

v2도 같은 원칙을 사용한다.

이번 스크립트는 v1 normalize_only 결과인 article_entities.parquet을 읽어서
아래와 같은 "표기 차이" 후보가 실제로 얼마나 존재하는지 찾는다.

예:
    COVID - 19  ↔ COVID-19
    Foo’s       ↔ Foo's
    A–B         ↔ A-B
    "Name"      ↔ “Name”
    invisible zero-width character 차이

중요:
    이 파일은 entity를 실제로 변경하지 않는다.
    후보만 추출한다.

즉 흐름은:

    normalize_only(v1) 결과
        ↓
    inspect_normalization_v2.py
        ↓
    v2 candidate parquet 생성
        ↓
    후보 검증
        ↓
    안전한 규칙만 entity_processing.py에 구현

Train/Validation leakage 방지
------------------------------------------------------------
후보 discovery는 반드시 Train-used article만 사용한다.
Validation-only entity를 보고 normalization 규칙을 결정하지 않는다.
"""

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable
import argparse
import re
import unicodedata

import polars as pl

from src import config


# =============================================================================
# 0. 기본 경로
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# v1의 확정된 snapshot을 입력으로 사용한다.
#
# data/output/model_inputs/article_entities.parquet을 바로 사용하면
# 이후 다른 실험이 해당 파일을 덮어썼을 때 분석 기준이 바뀔 수 있다.
#
# 따라서 experiments/normalize_only snapshot을 기본값으로 사용한다.
DEFAULT_V1_ARTICLE_ENTITIES_PATH = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "experiments"
    / "normalize_only"
    / "model_inputs"
    / "article_entities.parquet"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "experiments"
    / "normalization_v2_inspection"
)


# =============================================================================
# 1. Unicode 문자 정의
# =============================================================================
#
# 같은 눈 모양이어도 Unicode code point가 다른 문자가 존재한다.
#
# 예:
#   ASCII apostrophe    '
#   RIGHT SINGLE QUOTE  ’
#
# 사람 눈에는 거의 같지만 문자열 비교에서는 완전히 다른 값이다.
# 이런 차이는 Entity fragmentation의 원인이 될 수 있다.
# =============================================================================


APOSTROPHE_TRANSLATION = str.maketrans(
    {
        "\u2018": "'",   # LEFT SINGLE QUOTATION MARK
        "\u2019": "'",   # RIGHT SINGLE QUOTATION MARK
        "\u201B": "'",   # SINGLE HIGH-REVERSED-9 QUOTATION MARK
        "\u02BC": "'",   # MODIFIER LETTER APOSTROPHE
        "\uFF07": "'",   # FULLWIDTH APOSTROPHE
        "\u00B4": "'",   # ACUTE ACCENT
        "`": "'",        # GRAVE ACCENT / backtick
    }
)

DOUBLE_QUOTE_TRANSLATION = str.maketrans(
    {
        "\u201C": '"',   # LEFT DOUBLE QUOTATION MARK
        "\u201D": '"',   # RIGHT DOUBLE QUOTATION MARK
        "\u201E": '"',   # DOUBLE LOW-9 QUOTATION MARK
        "\u201F": '"',   # DOUBLE HIGH-REVERSED-9 QUOTATION MARK
        "\uFF02": '"',   # FULLWIDTH QUOTATION MARK
    }
)

DASH_TRANSLATION = str.maketrans(
    {
        "\u2010": "-",   # HYPHEN
        "\u2011": "-",   # NON-BREAKING HYPHEN
        "\u2012": "-",   # FIGURE DASH
        "\u2013": "-",   # EN DASH
        "\u2014": "-",   # EM DASH
        "\u2212": "-",   # MINUS SIGN
        "\uFE58": "-",   # SMALL EM DASH
        "\uFE63": "-",   # SMALL HYPHEN-MINUS
        "\uFF0D": "-",   # FULLWIDTH HYPHEN-MINUS
    }
)

ZERO_WIDTH_CHARACTERS = {
    "\u200B",  # ZERO WIDTH SPACE
    "\u200C",  # ZERO WIDTH NON-JOINER
    "\u200D",  # ZERO WIDTH JOINER
    "\u2060",  # WORD JOINER
    "\uFEFF",  # ZERO WIDTH NO-BREAK SPACE / BOM
}


# =============================================================================
# 2. 공통 문자열 helper
# =============================================================================


def _collapse_whitespace(text: str) -> str:
    """
    연속 whitespace를 한 칸으로 만든다.

    v1 baseline normalization에서도 이미 수행하지만,
    v2 transform 과정에서 문자를 제거하거나 바꾸면서
    새로운 이중 공백이 생길 수 있으므로 마지막에 다시 사용한다.
    """

    return " ".join(text.strip().split())


def _remove_zero_width(text: str) -> str:
    """
    눈에 보이지 않는 zero-width 문자를 제거한다.

    예:
        "open\u200bai" -> "openai"

    이런 문자는 의미를 바꾸는 문장부호라기보다
    encoding / copy-paste 차이에 가까워 Safe format 후보로 본다.
    """

    return "".join(
        character
        for character in text
        if character not in ZERO_WIDTH_CHARACTERS
    )


def _normalize_apostrophe_unicode(text: str) -> str:
    """
    apostrophe 모양만 다른 Unicode 문자를 ASCII apostrophe로 통일한다.

    예:
        foo’s -> foo's
    """

    return _collapse_whitespace(
        text.translate(APOSTROPHE_TRANSLATION)
    )


def _normalize_double_quote_unicode(text: str) -> str:
    """
    curly / fullwidth double quote를 ASCII double quote로 통일한다.

    예:
        “zanka” -> "zanka"
    """

    return _collapse_whitespace(
        text.translate(DOUBLE_QUOTE_TRANSLATION)
    )


def _normalize_dash_unicode(text: str) -> str:
    """
    서로 다른 Unicode dash code point를 '-'로만 통일한다.

    여기서는 dash 주변 공백은 아직 제거하지 않는다.

    예:
        a–b -> a-b
    """

    return _collapse_whitespace(
        text.translate(DASH_TRANSLATION)
    )


def _normalize_hyphen_spacing(text: str) -> str:
    """
    dash 종류를 '-'로 통일한 뒤,
    '-' 양옆의 whitespace만 제거한다.

    예:
        "covid - 19" -> "covid-19"
        "a - b"      -> "a-b"

    의미가 달라질 가능성이 Unicode apostrophe보다 조금 더 있기 때문에
    자동 적용 전 반드시 후보 검증을 한다.
    """

    normalized = text.translate(DASH_TRANSLATION)

    normalized = re.sub(
        r"\s*-\s*",
        "-",
        normalized,
    )

    return _collapse_whitespace(normalized)


def _normalize_hyphen_to_space(text: str) -> str:
    """
    hyphen과 space 차이로만 갈라진 entity를 찾기 위한 진단용 transform.

    예:
        covid-19 -> covid 19

    주의:
        이 규칙은 자동 적용 규칙이 아니다.

        고유명사에서는 hyphen이 실제 이름의 일부일 수 있으므로
        candidate discovery 용도로만 사용한다.
    """

    normalized = text.translate(DASH_TRANSLATION)

    normalized = re.sub(
        r"\s*-\s*",
        " ",
        normalized,
    )

    return _collapse_whitespace(normalized)


def _strip_outer_quotes(text: str) -> str:
    """
    문자열 전체를 둘러싼 quotation mark 차이를 조사한다.

    예:
        '"zanka"' -> 'zanka'
        '“zanka”' -> 'zanka'

    이것도 자동 적용 전 검토가 필요하다.
    """

    normalized = (
        text
        .translate(APOSTROPHE_TRANSLATION)
        .translate(DOUBLE_QUOTE_TRANSLATION)
    )

    normalized = _collapse_whitespace(normalized)

    if len(normalized) >= 2:
        quote_pairs = {
            ('"', '"'),
            ("'", "'"),
        }

        if (
            normalized[0],
            normalized[-1],
        ) in quote_pairs:
            normalized = normalized[1:-1]

    return _collapse_whitespace(normalized)


def _strip_terminal_punctuation(text: str) -> str:
    """
    끝 문장부호 차이를 진단한다.

    예:
        "openai." -> "openai"

    이 규칙은 매우 조심해야 한다.

    예:
        "Yahoo!" 같은 고유명사에서는 !가 이름의 일부일 수 있다.

    따라서 결과는 RISKY_DIAGNOSTIC으로 분류하고
    자동 normalization에 바로 넣지 않는다.
    """

    normalized = _collapse_whitespace(text)

    normalized = re.sub(
        r"[\.,;:!?]+$",
        "",
        normalized,
    )

    return _collapse_whitespace(normalized)


# =============================================================================
# 3. v2 후보 규칙 정의
# =============================================================================
#
# safety_tier
# -----------------------------------------------------------------------------
# SAFE_FORMAT
#   거의 encoding / typography 차이에 가까운 후보.
#   그래도 실제 데이터 검증 후 적용한다.
#
# REVIEW
#   같은 entity일 가능성이 있지만 고유명사 표기 의미가 달라질 수도 있음.
#
# RISKY_DIAGNOSTIC
#   자동 적용 목적이 아니라 fragmentation 규모를 보기 위한 진단.
# =============================================================================


V2_RULES: list[dict[str, Any]] = [
    {
        "rule_name": "zero_width",
        "safety_tier": "SAFE_FORMAT",
        "recommendation": "REVIEW_THEN_AUTO",
        "transform": lambda text: _collapse_whitespace(
            _remove_zero_width(text)
        ),
        "description": "zero-width/invisible Unicode 문자 제거",
    },
    {
        "rule_name": "apostrophe_unicode",
        "safety_tier": "SAFE_FORMAT",
        "recommendation": "REVIEW_THEN_AUTO",
        "transform": _normalize_apostrophe_unicode,
        "description": "Unicode apostrophe 모양 차이 통일",
    },
    {
        "rule_name": "double_quote_unicode",
        "safety_tier": "SAFE_FORMAT",
        "recommendation": "REVIEW_THEN_AUTO",
        "transform": _normalize_double_quote_unicode,
        "description": "Unicode double quote 모양 차이 통일",
    },
    {
        "rule_name": "dash_unicode",
        "safety_tier": "SAFE_FORMAT",
        "recommendation": "REVIEW_THEN_AUTO",
        "transform": _normalize_dash_unicode,
        "description": "Unicode dash code point 차이 통일",
    },
    {
        "rule_name": "hyphen_spacing",
        "safety_tier": "REVIEW",
        "recommendation": "MANUAL_REVIEW",
        "transform": _normalize_hyphen_spacing,
        "description": "hyphen 양옆 whitespace 차이 통일",
    },
    {
        "rule_name": "hyphen_vs_space",
        "safety_tier": "REVIEW",
        "recommendation": "MANUAL_REVIEW",
        "transform": _normalize_hyphen_to_space,
        "description": "hyphen과 일반 space 차이 후보",
    },
    {
        "rule_name": "outer_quotes",
        "safety_tier": "REVIEW",
        "recommendation": "MANUAL_REVIEW",
        "transform": _strip_outer_quotes,
        "description": "entity 전체를 감싸는 quote 차이",
    },
    {
        "rule_name": "terminal_punctuation",
        "safety_tier": "RISKY_DIAGNOSTIC",
        "recommendation": "DO_NOT_AUTO_APPLY",
        "transform": _strip_terminal_punctuation,
        "description": "문자열 끝 punctuation 차이 진단",
    },
]


# =============================================================================
# 4. 입력 데이터 읽기
# =============================================================================


def _load_train_v1_entities(
    article_entities_path: Path,
) -> pl.DataFrame:
    """
    v1 article_entities.parquet에서 Train-used mention만 읽는다.

    왜 canonical_entity를 사용하는가?
    ------------------------------------------------------------
    v2는 baseline raw entity가 아니라
    v1 Safe Normalization이 이미 적용된 결과 위에 추가되어야 한다.

    즉:
        raw
        -> baseline normalization
        -> possessive v1
        -> [여기서 v2 후보 조사]
    """

    if not article_entities_path.exists():
        raise FileNotFoundError(
            "normalize_only의 article_entities.parquet이 없습니다. "
            f"경로={article_entities_path}"
        )

    entity_df = pl.read_parquet(
        article_entities_path
    )

    required_columns = {
        "article_id",
        "entity_group",
        "canonical_entity",
        "canonical_entity_key",
        "is_train_used",
    }

    missing_columns = (
        required_columns - set(entity_df.columns)
    )

    if missing_columns:
        raise ValueError(
            "article_entities.parquet에 필요한 컬럼이 없습니다: "
            + ", ".join(sorted(missing_columns))
        )

    train_entities = (
        entity_df
        .filter(
            pl.col("is_train_used")
        )
        .select(
            [
                "article_id",
                "entity_group",
                "canonical_entity",
                "canonical_entity_key",
            ]
        )
    )

    return train_entities


def _load_article_titles() -> dict[int, str]:
    """
    후보를 사람이 검토할 때 문맥을 볼 수 있도록
    article_id -> title lookup을 만든다.

    후보 fit 자체에는 title을 사용하지 않는다.
    title은 오직 review/debug 용도다.
    """

    path = Path(
        config.ARTICLES_WITH_CATEGORY_PATH
    )

    if not path.exists():
        raise FileNotFoundError(
            "articles_with_category.parquet이 없습니다. "
            f"경로={path}"
        )

    schema_names = set(
        pl.scan_parquet(path)
        .collect_schema()
        .names()
    )

    if "title" not in schema_names:
        # title이 없더라도 후보 생성 자체는 가능하게 한다.
        return {}

    title_df = (
        pl.read_parquet(
            path,
            columns=[
                "article_id",
                "title",
            ],
        )
    )

    title_lookup: dict[int, str] = {}

    for article_id, title in title_df.iter_rows():
        if article_id is None:
            continue

        title_lookup[int(article_id)] = (
            ""
            if title is None
            else str(title)
        )

    return title_lookup


# =============================================================================
# 5. 현재 Train entity 통계 만들기
# =============================================================================


def _build_surface_statistics(
    train_entities: pl.DataFrame,
) -> tuple[
    dict[tuple[str, str], set[int]],
    Counter[tuple[str, str]],
]:
    """
    각 (TYPE, entity surface)의 article DF와 mention count를 계산한다.

    예:
        ("PER", "joe biden")
            article_ids = {10, 20, 30}
            mention_count = 5

    Event의 IDF/DF와 마찬가지로,
    article DF는 같은 기사 내부에서 여러 번 등장해도 1번만 센다.
    """

    surface_articles: dict[
        tuple[str, str],
        set[int],
    ] = defaultdict(set)

    mention_counter: Counter[
        tuple[str, str]
    ] = Counter()

    for row in train_entities.iter_rows(
        named=True
    ):
        article_id = int(
            row["article_id"]
        )

        entity_group = str(
            row["entity_group"]
        )

        entity_surface = str(
            row["canonical_entity"]
        )

        key = (
            entity_group,
            entity_surface,
        )

        surface_articles[
            key
        ].add(article_id)

        mention_counter[
            key
        ] += 1

    return (
        dict(surface_articles),
        mention_counter,
    )


# =============================================================================
# 6. 규칙별 collision group / mapping 후보 생성
# =============================================================================


def _build_rule_candidates(
    *,
    rule_name: str,
    safety_tier: str,
    recommendation: str,
    description: str,
    transform: Callable[[str], str],
    surface_articles: dict[
        tuple[str, str],
        set[int],
    ],
    mention_counter: Counter[
        tuple[str, str]
    ],
    title_lookup: dict[int, str],
    max_title_examples: int,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    """
    하나의 v2 규칙에 대해 실제 collision을 찾는다.

    핵심 아이디어
    ------------------------------------------------------------
    예를 들어 현재 Train vocabulary에:

        ORG::foo’s
        ORG::foo's

    가 있다고 하자.

    apostrophe_unicode transform을 적용하면 둘 다:

        foo's

    로 바뀐다.

    즉 "현재는 다른 entity지만 v2 format normalization 후에는 같은 값"이 된다.
    이것이 실제 fragmentation candidate다.

    반대로 어떤 entity 하나만 형태가 바뀌고
    같은 transformed value를 가진 다른 surface가 전혀 없다면,
    현재 데이터에서 alias fragmentation 증거가 약하므로
    collision candidate에는 포함하지 않는다.
    """

    # (TYPE, transformed surface) -> 원래 surface 목록
    collision_buckets: dict[
        tuple[str, str],
        set[str],
    ] = defaultdict(set)

    for (
        entity_group,
        surface,
    ) in surface_articles:
        transformed = transform(
            surface
        )

        if transformed == "":
            continue

        collision_buckets[
            (
                entity_group,
                transformed,
            )
        ].add(surface)

    collision_rows: list[
        dict[str, Any]
    ] = []

    mapping_rows: list[
        dict[str, Any]
    ] = []

    affected_article_ids: set[int] = set()
    collision_group_count = 0

    for (
        entity_group,
        canonical_candidate,
    ), surfaces in sorted(
        collision_buckets.items()
    ):
        distinct_surfaces = sorted(
            surfaces
        )

        # 최소 2개의 현재 surface가 같은 transformed value로
        # collapse되어야 fragmentation 후보라고 본다.
        if len(distinct_surfaces) < 2:
            continue

        # 그리고 실제로 rule에 의해 바뀌는 surface가 최소 하나 있어야 한다.
        changed_surfaces = [
            surface
            for surface in distinct_surfaces
            if transform(surface) != surface
        ]

        if not changed_surfaces:
            continue

        collision_group_count += 1

        surface_stats: list[
            tuple[str, int, int]
        ] = []

        group_article_ids: set[int] = set()

        for surface in distinct_surfaces:
            key = (
                entity_group,
                surface,
            )

            article_ids = (
                surface_articles[
                    key
                ]
            )

            article_df = len(
                article_ids
            )

            mention_count = int(
                mention_counter[
                    key
                ]
            )

            group_article_ids.update(
                article_ids
            )

            surface_stats.append(
                (
                    surface,
                    article_df,
                    mention_count,
                )
            )

        affected_article_ids.update(
            group_article_ids
        )

        # 사람이 보기 쉽게 frequency 순으로 variant를 정렬한다.
        surface_stats.sort(
            key=lambda item: (
                -item[1],
                -item[2],
                item[0],
            )
        )

        # transformed 결과 자체가 이미 현재 vocabulary에 있다면
        # 그것을 observed target으로 본다.
        target_observed = (
            canonical_candidate
            in distinct_surfaces
        )

        title_examples: list[str] = []

        for article_id in sorted(
            group_article_ids
        ):
            title = title_lookup.get(
                article_id,
                "",
            )

            if title:
                title_examples.append(
                    title
                )

            if (
                len(title_examples)
                >= max_title_examples
            ):
                break

        collision_rows.append(
            {
                "rule_name": rule_name,
                "safety_tier": safety_tier,
                "recommendation": recommendation,
                "description": description,
                "entity_group": entity_group,
                "canonical_candidate": canonical_candidate,
                "target_observed_in_train": target_observed,
                "surface_count": int(
                    len(distinct_surfaces)
                ),
                "surfaces": [
                    item[0]
                    for item in surface_stats
                ],
                "surface_article_dfs": [
                    int(item[1])
                    for item in surface_stats
                ],
                "surface_mention_counts": [
                    int(item[2])
                    for item in surface_stats
                ],
                "union_article_df": int(
                    len(group_article_ids)
                ),
                "title_examples": (
                    title_examples
                ),
            }
        )

        # 실제 mapping 후보는 바뀌는 variant마다 한 행씩 만든다.
        for variant_surface in changed_surfaces:
            key = (
                entity_group,
                variant_surface,
            )

            variant_article_ids = (
                surface_articles[
                    key
                ]
            )

            variant_titles: list[str] = []

            for article_id in sorted(
                variant_article_ids
            ):
                title = title_lookup.get(
                    article_id,
                    "",
                )

                if title:
                    variant_titles.append(
                        title
                    )

                if (
                    len(variant_titles)
                    >= max_title_examples
                ):
                    break

            canonical_key = (
                entity_group,
                canonical_candidate,
            )

            canonical_article_df = (
                len(
                    surface_articles[
                        canonical_key
                    ]
                )
                if canonical_key
                in surface_articles
                else 0
            )

            same_article_cooccurrence_count = 0

            if canonical_key in surface_articles:
                same_article_cooccurrence_count = len(
                    variant_article_ids
                    & surface_articles[
                        canonical_key
                    ]
                )

            mapping_rows.append(
                {
                    "rule_name": rule_name,
                    "safety_tier": safety_tier,
                    "recommendation": recommendation,
                    "entity_group": entity_group,
                    "variant_entity": variant_surface,
                    "canonical_candidate": canonical_candidate,
                    "variant_entity_key": (
                        f"{entity_group}::{variant_surface}"
                    ),
                    "canonical_candidate_key": (
                        f"{entity_group}::{canonical_candidate}"
                    ),
                    "variant_article_df": int(
                        len(
                            variant_article_ids
                        )
                    ),
                    "canonical_existing_article_df": int(
                        canonical_article_df
                    ),
                    "same_article_cooccurrence_count": int(
                        same_article_cooccurrence_count
                    ),
                    "canonical_observed_in_train": (
                        canonical_key
                        in surface_articles
                    ),
                    "variant_title_examples": (
                        variant_titles
                    ),
                }
            )

    stats = {
        "rule_name": rule_name,
        "safety_tier": safety_tier,
        "recommendation": recommendation,
        "description": description,
        "collision_group_count": int(
            collision_group_count
        ),
        "mapping_candidate_count": int(
            len(mapping_rows)
        ),
        "affected_article_count": int(
            len(affected_article_ids)
        ),
    }

    return (
        collision_rows,
        mapping_rows,
        stats,
    )


# =============================================================================
# 7. 특수문자 inventory
# =============================================================================


def _build_character_inventory(
    train_entities: pl.DataFrame,
) -> pl.DataFrame:
    """
    현재 canonical entity 안에 어떤 punctuation / format 문자가 실제로 있는지 조사한다.

    예:
        char = ’
        codepoint = U+2019
        unicode_name = RIGHT SINGLE QUOTATION MARK

    이 표를 보면 우리가 미리 생각하지 못한 특수문자가
    데이터 안에 있는지 확인할 수 있다.
    """

    character_mention_count: Counter[
        str
    ] = Counter()

    character_article_ids: dict[
        str,
        set[int],
    ] = defaultdict(set)

    character_surfaces: dict[
        str,
        set[str],
    ] = defaultdict(set)

    for row in train_entities.iter_rows(
        named=True
    ):
        article_id = int(
            row["article_id"]
        )

        surface = str(
            row["canonical_entity"]
        )

        # 같은 entity 문자열 안에서 같은 특수문자가 여러 번 나와도
        # article DF 관점에서는 한 번만 추가한다.
        seen_characters: set[str] = set()

        for character in surface:
            category = (
                unicodedata.category(
                    character
                )
            )

            # P*: punctuation
            # Cf: format character (zero-width 등)
            #
            # ASCII 영숫자/일반 whitespace는 제외.
            is_interesting = (
                category.startswith("P")
                or category == "Cf"
                or character
                in ZERO_WIDTH_CHARACTERS
            )

            if not is_interesting:
                continue

            character_mention_count[
                character
            ] += 1

            character_surfaces[
                character
            ].add(surface)

            seen_characters.add(
                character
            )

        for character in seen_characters:
            character_article_ids[
                character
            ].add(article_id)

    rows: list[dict[str, Any]] = []

    for character in sorted(
        character_mention_count,
        key=lambda value: (
            -character_mention_count[
                value
            ],
            ord(value),
        ),
    ):
        surface_examples = sorted(
            character_surfaces[
                character
            ]
        )[:10]

        try:
            unicode_name = (
                unicodedata.name(
                    character
                )
            )
        except ValueError:
            unicode_name = "<UNKNOWN>"

        rows.append(
            {
                "character": character,
                "codepoint": (
                    f"U+{ord(character):04X}"
                ),
                "unicode_name": unicode_name,
                "unicode_category": (
                    unicodedata.category(
                        character
                    )
                ),
                "mention_occurrence_count": int(
                    character_mention_count[
                        character
                    ]
                ),
                "article_df": int(
                    len(
                        character_article_ids[
                            character
                        ]
                    )
                ),
                "entity_surface_count": int(
                    len(
                        character_surfaces[
                            character
                        ]
                    )
                ),
                "surface_examples": (
                    surface_examples
                ),
            }
        )

    schema = {
        "character": pl.String,
        "codepoint": pl.String,
        "unicode_name": pl.String,
        "unicode_category": pl.String,
        "mention_occurrence_count": pl.Int64,
        "article_df": pl.Int64,
        "entity_surface_count": pl.Int64,
        "surface_examples": pl.List(
            pl.String
        ),
    }

    if not rows:
        return pl.DataFrame(
            schema=schema
        )

    return pl.DataFrame(
        rows,
        schema=schema,
    )


# =============================================================================
# 8. DataFrame 변환 helper
# =============================================================================


def _rows_to_collision_df(
    rows: list[dict[str, Any]],
) -> pl.DataFrame:
    """
    collision group rows -> Polars DataFrame
    """

    schema = {
        "rule_name": pl.String,
        "safety_tier": pl.String,
        "recommendation": pl.String,
        "description": pl.String,
        "entity_group": pl.String,
        "canonical_candidate": pl.String,
        "target_observed_in_train": pl.Boolean,
        "surface_count": pl.Int64,
        "surfaces": pl.List(
            pl.String
        ),
        "surface_article_dfs": pl.List(
            pl.Int64
        ),
        "surface_mention_counts": pl.List(
            pl.Int64
        ),
        "union_article_df": pl.Int64,
        "title_examples": pl.List(
            pl.String
        ),
    }

    if not rows:
        return pl.DataFrame(
            schema=schema
        )

    return (
        pl.DataFrame(
            rows,
            schema=schema,
        )
        .sort(
            [
                "safety_tier",
                "rule_name",
                "entity_group",
                "canonical_candidate",
            ]
        )
    )


def _rows_to_mapping_df(
    rows: list[dict[str, Any]],
) -> pl.DataFrame:
    """
    variant -> canonical candidate mapping 후보 rows -> DataFrame
    """

    schema = {
        "rule_name": pl.String,
        "safety_tier": pl.String,
        "recommendation": pl.String,
        "entity_group": pl.String,
        "variant_entity": pl.String,
        "canonical_candidate": pl.String,
        "variant_entity_key": pl.String,
        "canonical_candidate_key": pl.String,
        "variant_article_df": pl.Int64,
        "canonical_existing_article_df": pl.Int64,
        "same_article_cooccurrence_count": pl.Int64,
        "canonical_observed_in_train": pl.Boolean,
        "variant_title_examples": pl.List(
            pl.String
        ),
    }

    if not rows:
        return pl.DataFrame(
            schema=schema
        )

    return (
        pl.DataFrame(
            rows,
            schema=schema,
        )
        .sort(
            [
                "safety_tier",
                "rule_name",
                "entity_group",
                "variant_entity",
            ]
        )
    )


def _rows_to_rule_stats_df(
    rows: list[dict[str, Any]],
) -> pl.DataFrame:
    """
    규칙별 후보 규모 요약.
    """

    schema = {
        "rule_name": pl.String,
        "safety_tier": pl.String,
        "recommendation": pl.String,
        "description": pl.String,
        "collision_group_count": pl.Int64,
        "mapping_candidate_count": pl.Int64,
        "affected_article_count": pl.Int64,
    }

    if not rows:
        return pl.DataFrame(
            schema=schema
        )

    return pl.DataFrame(
        rows,
        schema=schema,
    )


# =============================================================================
# 9. summary text 저장
# =============================================================================


def _write_summary(
    *,
    output_path: Path,
    source_path: Path,
    train_entity_row_count: int,
    unique_train_entity_count: int,
    rule_stats: list[dict[str, Any]],
    collision_df: pl.DataFrame,
    mapping_df: pl.DataFrame,
    character_inventory_df: pl.DataFrame,
) -> None:
    """
    터미널 결과를 나중에 다시 찾지 않아도 되도록
    사람이 읽는 summary txt 파일도 함께 저장한다.
    """

    lines: list[str] = []

    lines.append(
        "=" * 80
    )
    lines.append(
        "Safe Normalization v2 Candidate Inspection"
    )
    lines.append(
        "=" * 80
    )
    lines.append("")
    lines.append(
        f"source_article_entities={source_path}"
    )
    lines.append(
        "fit_split=train_used_articles_only"
    )
    lines.append(
        f"train_entity_mention_rows={train_entity_row_count}"
    )
    lines.append(
        f"unique_train_entity_surfaces={unique_train_entity_count}"
    )
    lines.append("")

    lines.append(
        "-" * 80
    )
    lines.append(
        "Rule Statistics"
    )
    lines.append(
        "-" * 80
    )

    for stat in rule_stats:
        lines.append(
            (
                f"{stat['rule_name']}: "
                f"tier={stat['safety_tier']}, "
                f"collision_groups={stat['collision_group_count']}, "
                f"mapping_candidates={stat['mapping_candidate_count']}, "
                f"affected_articles={stat['affected_article_count']}"
            )
        )

    lines.append("")
    lines.append(
        "-" * 80
    )
    lines.append(
        "Overall"
    )
    lines.append(
        "-" * 80
    )
    lines.append(
        f"collision_group_rows={collision_df.height}"
    )
    lines.append(
        f"mapping_candidate_rows={mapping_df.height}"
    )
    lines.append(
        f"character_inventory_rows={character_inventory_df.height}"
    )
    lines.append("")

    lines.append(
        "주의: 이 결과는 후보 진단이다. "
        "아직 entity_processing.py에 자동 적용하지 않는다."
    )

    output_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# =============================================================================
# 10. Main
# =============================================================================


def inspect_normalization_v2(
    *,
    article_entities_path: Path,
    output_dir: Path,
    max_title_examples: int = 3,
) -> dict[str, Any]:
    """
    전체 v2 candidate inspection 실행.

    생성 파일
    ------------------------------------------------------------
    normalization_v2_summary.txt
        규칙별 후보 개수 요약

    normalization_v2_rule_stats.parquet
        rule 단위 통계

    normalization_v2_collision_groups.parquet
        여러 현재 surface가 하나의 v2 canonical candidate로 합쳐지는 그룹

    normalization_v2_mapping_candidates.parquet
        variant -> canonical candidate 형태의 검토용 후보

    normalization_v2_character_inventory.parquet
        현재 entity 안에 실제로 존재하는 punctuation/Unicode 문자 inventory
    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_entities = (
        _load_train_v1_entities(
            article_entities_path
        )
    )

    title_lookup = (
        _load_article_titles()
    )

    (
        surface_articles,
        mention_counter,
    ) = _build_surface_statistics(
        train_entities
    )

    all_collision_rows: list[
        dict[str, Any]
    ] = []

    all_mapping_rows: list[
        dict[str, Any]
    ] = []

    rule_stats: list[
        dict[str, Any]
    ] = []

    for rule in V2_RULES:
        (
            collision_rows,
            mapping_rows,
            stats,
        ) = _build_rule_candidates(
            rule_name=rule[
                "rule_name"
            ],
            safety_tier=rule[
                "safety_tier"
            ],
            recommendation=rule[
                "recommendation"
            ],
            description=rule[
                "description"
            ],
            transform=rule[
                "transform"
            ],
            surface_articles=surface_articles,
            mention_counter=mention_counter,
            title_lookup=title_lookup,
            max_title_examples=(
                max_title_examples
            ),
        )

        all_collision_rows.extend(
            collision_rows
        )

        all_mapping_rows.extend(
            mapping_rows
        )

        rule_stats.append(
            stats
        )

    collision_df = (
        _rows_to_collision_df(
            all_collision_rows
        )
    )

    mapping_df = (
        _rows_to_mapping_df(
            all_mapping_rows
        )
    )

    rule_stats_df = (
        _rows_to_rule_stats_df(
            rule_stats
        )
    )

    character_inventory_df = (
        _build_character_inventory(
            train_entities
        )
    )

    collision_path = (
        output_dir
        / "normalization_v2_collision_groups.parquet"
    )

    mapping_path = (
        output_dir
        / "normalization_v2_mapping_candidates.parquet"
    )

    rule_stats_path = (
        output_dir
        / "normalization_v2_rule_stats.parquet"
    )

    character_inventory_path = (
        output_dir
        / "normalization_v2_character_inventory.parquet"
    )

    summary_path = (
        output_dir
        / "normalization_v2_summary.txt"
    )

    collision_df.write_parquet(
        collision_path,
        compression="zstd",
    )

    mapping_df.write_parquet(
        mapping_path,
        compression="zstd",
    )

    rule_stats_df.write_parquet(
        rule_stats_path,
        compression="zstd",
    )

    character_inventory_df.write_parquet(
        character_inventory_path,
        compression="zstd",
    )

    unique_train_entity_count = len(
        surface_articles
    )

    _write_summary(
        output_path=summary_path,
        source_path=article_entities_path,
        train_entity_row_count=int(
            train_entities.height
        ),
        unique_train_entity_count=int(
            unique_train_entity_count
        ),
        rule_stats=rule_stats,
        collision_df=collision_df,
        mapping_df=mapping_df,
        character_inventory_df=(
            character_inventory_df
        ),
    )

    return {
        "status": "SUCCESS",
        "fit_split": (
            "train_used_articles_only"
        ),
        "source_article_entities_path": str(
            article_entities_path
        ),
        "train_entity_mention_row_count": int(
            train_entities.height
        ),
        "unique_train_entity_surface_count": int(
            unique_train_entity_count
        ),
        "collision_group_count": int(
            collision_df.height
        ),
        "mapping_candidate_count": int(
            mapping_df.height
        ),
        "character_inventory_count": int(
            character_inventory_df.height
        ),
        "output_dir": str(
            output_dir
        ),
        "summary_path": str(
            summary_path
        ),
        "rule_stats_path": str(
            rule_stats_path
        ),
        "collision_groups_path": str(
            collision_path
        ),
        "mapping_candidates_path": str(
            mapping_path
        ),
        "character_inventory_path": str(
            character_inventory_path
        ),
    }


# =============================================================================
# 11. CLI
# =============================================================================


def _parse_args() -> argparse.Namespace:
    """
    명령행 옵션.

    기본 실행:
        python -m src.inspect_normalization_v2

    다른 v1 snapshot을 보고 싶을 때:
        python -m src.inspect_normalization_v2 \
            --article-entities <path>
    """

    parser = argparse.ArgumentParser(
        description=(
            "Safe Normalization v2 후보를 "
            "Train v1 canonical entity에서 진단합니다."
        )
    )

    parser.add_argument(
        "--article-entities",
        type=Path,
        default=(
            DEFAULT_V1_ARTICLE_ENTITIES_PATH
        ),
        help=(
            "v1 article_entities.parquet 경로"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "v2 inspection 결과 저장 폴더"
        ),
    )

    parser.add_argument(
        "--max-title-examples",
        type=int,
        default=3,
        help=(
            "후보 하나당 저장할 기사 제목 예시 수"
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.max_title_examples < 0:
        raise ValueError(
            "--max-title-examples는 0 이상이어야 합니다."
        )

    print(
        "=" * 80
    )
    print(
        "Safe Normalization v2 Candidate Inspection 시작"
    )
    print(
        "=" * 80
    )
    print(
        f"article_entities = {args.article_entities}"
    )
    print(
        f"output_dir       = {args.output_dir}"
    )
    print()

    result = inspect_normalization_v2(
        article_entities_path=(
            args.article_entities
        ),
        output_dir=args.output_dir,
        max_title_examples=(
            args.max_title_examples
        ),
    )

    print(
        "=" * 80
    )
    print(
        "Safe Normalization v2 Candidate Inspection 완료"
    )
    print(
        "=" * 80
    )

    for key, value in result.items():
        print(
            f"{key}: {value}"
        )

    print()
    print(
        "다음 파일을 먼저 확인하세요:"
    )
    print(
        "1. normalization_v2_summary.txt"
    )
    print(
        "2. normalization_v2_rule_stats.parquet"
    )
    print(
        "3. normalization_v2_mapping_candidates.parquet"
    )
    print(
        "4. normalization_v2_collision_groups.parquet"
    )
    print(
        "5. normalization_v2_character_inventory.parquet"
    )


if __name__ == "__main__":
    main()