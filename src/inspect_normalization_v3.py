from __future__ import annotations

"""
inspect_normalization_v3.py
===========================

목적
----
Safe Entity Normalization v2까지 적용한 뒤에도 남아 있는
"같은 개체의 다른 표기(alias)" 후보를 Train 데이터 안에서 찾는다.

중요
----
이 파일은 "조사(inspection)" 전용이다.

즉:
    - article_entities.parquet 수정 X
    - entity_normalization_map.parquet 수정 X
    - Event clustering 재실행 X
    - RQ-VAE 재실행 X
    - 실제 v3 mapping 적용 X

오직 후보를 찾아서 parquet / txt로 저장한다.

조사할 v3 후보
--------------
1. ACRONYM
   예:
       ORG::fck
       ORG::fc københavn

2. ORG_PUNCTUATION
   예:
       ORG::f.c. københavn
       ORG::fc københavn

3. ORG_LEGAL_SUFFIX
   예:
       ORG::novo nordisk a/s
       ORG::novo nordisk

4. ORG_SHORT_FORM
   예:
       ORG::manchester united fc
       ORG::manchester united

   이 규칙은 가장 위험하므로 자동 적용 후보가 아니라
   REVIEW_ONLY 진단용으로만 본다.

데이터 누수 방지
----------------
candidate fit / 통계 계산은 반드시 Train-used article만 사용한다.

입력
----
data/output/experiments/normalize_v2/model_inputs/
    article_entities.parquet
    articles_base.parquet

핵심 입력 컬럼
--------------
article_entities.parquet:
    article_id
    entity_group
    canonical_entity
    canonical_entity_key
    is_train_used

articles_base.parquet:
    article_id
    title

출력
----
data/output/experiments/normalization_v3_inspection/

    normalization_v3_candidates.parquet
    acronym_candidates.parquet
    org_punctuation_candidates.parquet
    org_legal_suffix_candidates.parquet
    org_short_form_candidates.parquet
    normalization_v3_summary.txt

실행
----
프로젝트 루트에서:

    python -m src.inspect_normalization_v3
"""

import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from pprint import pprint
from typing import Any

import polars as pl


# =============================================================================
# 1. 경로
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

NORMALIZE_V2_MODEL_INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "experiments"
    / "normalize_v2"
    / "model_inputs"
)

ARTICLE_ENTITIES_PATH = (
    NORMALIZE_V2_MODEL_INPUT_DIR
    / "article_entities.parquet"
)

ARTICLES_BASE_PATH = (
    NORMALIZE_V2_MODEL_INPUT_DIR
    / "articles_base.parquet"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "experiments"
    / "normalization_v3_inspection"
)

ALL_CANDIDATES_PATH = (
    OUTPUT_DIR
    / "normalization_v3_candidates.parquet"
)

ACRONYM_CANDIDATES_PATH = (
    OUTPUT_DIR
    / "acronym_candidates.parquet"
)

PUNCTUATION_CANDIDATES_PATH = (
    OUTPUT_DIR
    / "org_punctuation_candidates.parquet"
)

LEGAL_SUFFIX_CANDIDATES_PATH = (
    OUTPUT_DIR
    / "org_legal_suffix_candidates.parquet"
)

SHORT_FORM_CANDIDATES_PATH = (
    OUTPUT_DIR
    / "org_short_form_candidates.parquet"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "normalization_v3_summary.txt"
)


# =============================================================================
# 2. 조사 정책
# =============================================================================

# v3 alias normalization은 우선 precision을 지키기 위해 ORG에서만 조사한다.
#
# PER surname:
#   Vladimir Putin ↔ Putin
# 은 ambiguity 위험이 크므로 여기서는 조사 대상에서 제외.
#
# LOC ↔ ORG cross-type:
#   Barcelona(도시) ↔ FC Barcelona(구단)
# 같은 문제가 있으므로 제외.
TARGET_ENTITY_GROUP = "ORG"


# 같은 기사에서 두 표현이 함께 나타난 횟수를 evidence로 사용한다.
#
# Acronym은 최소 2회 이상 co-occurrence하면 상대적으로 강한 evidence로 본다.
ACRONYM_STRONG_MIN_COOCCURRENCE = 2

# Acronym 자체가 Train에서 너무 희귀한 1회성 표현이면
# 자동 적용에는 위험하므로 strong으로 분류하지 않는다.
ACRONYM_STRONG_MIN_VARIANT_DF = 2

# punctuation / legal suffix에서 target base가 Train에서 최소 2개 기사 이상
# 등장했을 때만 SAFE_WITH_GUARDRAIL 후보로 본다.
SAFE_BASE_MIN_DF = 2

# short form 후보가 너무 짧으면 일반 단어 충돌 위험이 커진다.
SHORT_FORM_MIN_CHAR_LENGTH = 5

# short form 후보가 긴 형태와 다른 token 수.
# 지나치게 많은 token이 빠지는 generic containment는 조사하지 않는다.
SHORT_FORM_MAX_BOUNDARY_TOKEN_DIFF = 2

# 리뷰용 evidence article/title을 몇 개 저장할지.
MAX_SAMPLE_ARTICLES = 5


# =============================================================================
# 3. Acronym helper 설정
# =============================================================================

# full organization name의 initials 계산에서 무시할 수 있는
# 매우 흔한 function word.
#
# 예:
#   "foreningen for ..."에서 for까지 acronym에 넣으면
#   실제 약어와 어긋나는 경우가 많다.
ACRONYM_STOPWORDS = {
    "a",
    "af",
    "and",
    "av",
    "de",
    "den",
    "der",
    "det",
    "for",
    "i",
    "in",
    "of",
    "og",
    "the",
    "til",
    "von",
    "van",
}

# FC København → FCK 처럼
# 첫 token 자체가 이미 "FC"라는 약어인 경우가 있다.
#
# 이런 2~3자 조직 marker는 첫 글자 하나만 쓰지 않고
# token 전체를 acronym signature에 넣는다.
ACRONYM_PREFIX_TOKENS = {
    "ac",
    "afc",
    "bk",
    "cf",
    "fc",
    "fk",
    "hc",
    "if",
    "ik",
    "rc",
    "sc",
    "sk",
    "sv",
    "tv",
    "uc",
}


# =============================================================================
# 4. Legal suffix 설정
# =============================================================================

# 조직명 끝에서 제거해 볼 법인 suffix.
#
# punctuation normalization 후 비교하기 때문에:
#   A/S
#   A.S.
# 같은 형태도 normalize된 token sequence를 이용해 처리한다.
LEGAL_SUFFIX_PATTERNS = [
    ("a", "s"),
    ("as",),
    ("aps",),
    ("a", "m", "b", "a"),
    ("amba",),
    ("ab",),
    ("ag",),
    ("bv",),
    ("corp",),
    ("corporation",),
    ("gmbh",),
    ("inc",),
    ("incorporated",),
    ("limited",),
    ("llc",),
    ("ltd",),
    ("nv",),
    ("oy",),
    ("oyj",),
    ("plc",),
    ("sa",),
]


# =============================================================================
# 5. Short-form boundary descriptor
# =============================================================================

# "FC Barcelona" ↔ "Barcelona"
# "Manchester United FC" ↔ "Manchester United"
# 같은 후보를 진단하기 위한 boundary descriptor.
#
# 다만 이 규칙은 잘못 합치기 쉬우므로 결과는 무조건 REVIEW_ONLY.
SHORT_FORM_BOUNDARY_TOKENS = {
    "ac",
    "afc",
    "bk",
    "club",
    "fc",
    "fk",
    "football",
    "hc",
    "if",
    "ik",
    "klub",
    "rc",
    "sc",
    "sk",
    "sport",
    "sports",
    "team",
}


# =============================================================================
# 6. 공통 text helper
# =============================================================================


def _basic_text(value: Any) -> str:
    """
    Entity surface를 비교용 문자열로 정리한다.

    v2 canonical_entity가 이미 lower-case인 경우가 대부분이지만
    inspection script 자체를 독립적으로 안전하게 만들기 위해
    NFKC + lower + whitespace collapse를 한 번 더 적용한다.
    """

    text = unicodedata.normalize(
        "NFKC",
        str(value or ""),
    )

    text = text.strip().lower()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text


def _alnum_tokens(value: str) -> list[str]:
    """
    Unicode 문자/숫자 token만 추출한다.

    예:
        "f.c. københavn"
        -> ["f", "c", "københavn"]

        "novo nordisk a/s"
        -> ["novo", "nordisk", "a", "s"]
    """

    text = _basic_text(
        value
    )

    return re.findall(
        r"[^\W_]+",
        text,
        flags=re.UNICODE,
    )


def _compact_alnum(value: str) -> str:
    """
    문자/숫자만 남긴 compact form.

    예:
        "F.C.K."
        -> "fck"

        "F C K"
        -> "fck"
    """

    return "".join(
        _alnum_tokens(
            value
        )
    )


# =============================================================================
# 7. Input 검사 / Load
# =============================================================================


def _require_file(
    path: Path,
    description: str,
) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"{description} 파일이 없습니다: {path}"
        )


def _require_columns(
    df: pl.DataFrame,
    required: set[str],
    description: str,
) -> None:
    missing = (
        required
        - set(
            df.columns
        )
    )

    if missing:
        raise ValueError(
            f"{description}에 필요한 컬럼이 없습니다: "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )


def _load_train_entity_pairs() -> pl.DataFrame:
    """
    Train-used article의 canonical entity pair만 읽는다.

    mention-level 중복은 제거한다.

    즉 같은 기사 안에서:
        ORG::fck
        ORG::fck
    가 여러 mention으로 반복되어도 article/entity pair는 1개만 센다.
    """

    _require_file(
        ARTICLE_ENTITIES_PATH,
        "normalize_v2 article_entities",
    )

    df = pl.read_parquet(
        ARTICLE_ENTITIES_PATH
    )

    _require_columns(
        df,
        {
            "article_id",
            "entity_group",
            "canonical_entity",
            "canonical_entity_key",
            "is_train_used",
        },
        "article_entities.parquet",
    )

    return (
        df.filter(
            pl.col(
                "is_train_used"
            )
        )
        .select(
            [
                pl.col(
                    "article_id"
                ).cast(
                    pl.Int64
                ),
                pl.col(
                    "entity_group"
                ).cast(
                    pl.String
                ),
                pl.col(
                    "canonical_entity"
                ).cast(
                    pl.String
                ),
                pl.col(
                    "canonical_entity_key"
                ).cast(
                    pl.String
                ),
            ]
        )
        .unique()
        .sort(
            [
                "entity_group",
                "canonical_entity",
                "article_id",
            ]
        )
    )


def _load_title_lookup() -> dict[int, str]:
    """
    review evidence용 article title lookup.
    """

    _require_file(
        ARTICLES_BASE_PATH,
        "normalize_v2 articles_base",
    )

    df = pl.read_parquet(
        ARTICLES_BASE_PATH
    )

    _require_columns(
        df,
        {
            "article_id",
            "title",
        },
        "articles_base.parquet",
    )

    result: dict[
        int,
        str,
    ] = {}

    for row in (
        df.select(
            [
                "article_id",
                "title",
            ]
        )
        .unique(
            subset=[
                "article_id"
            ],
            keep="first",
        )
        .iter_rows(
            named=True
        )
    ):
        result[
            int(
                row[
                    "article_id"
                ]
            )
        ] = str(
            row.get(
                "title",
                "",
            )
            or ""
        )

    return result


# =============================================================================
# 8. Train DF / article set lookup
# =============================================================================


def _build_entity_article_sets(
    train_pairs: pl.DataFrame,
) -> dict[str, set[int]]:
    """
    canonical_entity_key
        -> 이 entity가 등장한 Train article_id 집합
    """

    result: dict[
        str,
        set[int],
    ] = defaultdict(
        set
    )

    for row in train_pairs.iter_rows(
        named=True
    ):
        result[
            str(
                row[
                    "canonical_entity_key"
                ]
            )
        ].add(
            int(
                row[
                    "article_id"
                ]
            )
        )

    return dict(
        result
    )


def _build_surface_key_lookup(
    train_pairs: pl.DataFrame,
    entity_group: str,
) -> dict[str, str]:
    """
    같은 TYPE 안에서:
        canonical surface -> canonical key
    """

    result: dict[
        str,
        str,
    ] = {}

    subset = (
        train_pairs.filter(
            pl.col(
                "entity_group"
            )
            == entity_group
        )
        .select(
            [
                "canonical_entity",
                "canonical_entity_key",
            ]
        )
        .unique()
    )

    for row in subset.iter_rows(
        named=True
    ):
        result[
            _basic_text(
                row[
                    "canonical_entity"
                ]
            )
        ] = str(
            row[
                "canonical_entity_key"
            ]
        )

    return result


def _build_key_surface_lookup(
    train_pairs: pl.DataFrame,
) -> dict[str, str]:
    result = {}

    for row in (
        train_pairs.select(
            [
                "canonical_entity_key",
                "canonical_entity",
            ]
        )
        .unique()
        .iter_rows(
            named=True
        )
    ):
        result[
            str(
                row[
                    "canonical_entity_key"
                ]
            )
        ] = _basic_text(
            row[
                "canonical_entity"
            ]
        )

    return result


# =============================================================================
# 9. Candidate row 공통 builder
# =============================================================================


def _evidence(
    *,
    left_key: str,
    right_key: str,
    entity_article_sets: dict[
        str,
        set[int],
    ],
    title_lookup: dict[
        int,
        str,
    ],
) -> dict[str, Any]:
    """
    두 entity가 Train에서 얼마나 등장하고,
    같은 기사에 얼마나 같이 등장하는지 계산한다.
    """

    left_articles = (
        entity_article_sets.get(
            left_key,
            set(),
        )
    )

    right_articles = (
        entity_article_sets.get(
            right_key,
            set(),
        )
    )

    overlap = sorted(
        left_articles
        & right_articles
    )

    left_df = len(
        left_articles
    )

    right_df = len(
        right_articles
    )

    minimum_df = min(
        left_df,
        right_df,
    )

    overlap_ratio_min_df = (
        len(
            overlap
        )
        / minimum_df
        if minimum_df
        else 0.0
    )

    sample_article_ids = overlap[
        :MAX_SAMPLE_ARTICLES
    ]

    sample_titles = [
        title_lookup.get(
            article_id,
            "",
        )
        for article_id
        in sample_article_ids
    ]

    return {
        "left_article_df": int(
            left_df
        ),
        "right_article_df": int(
            right_df
        ),
        "cooccurrence_article_count": int(
            len(
                overlap
            )
        ),
        "cooccurrence_ratio_min_df": float(
            overlap_ratio_min_df
        ),
        "sample_cooccurrence_article_ids": (
            sample_article_ids
        ),
        "sample_cooccurrence_titles": (
            sample_titles
        ),
    }


def _make_candidate_row(
    *,
    rule: str,
    safety_tier: str,
    entity_group: str,
    variant_entity: str,
    candidate_entity: str,
    variant_key: str,
    candidate_key: str,
    rationale: str,
    entity_article_sets: dict[
        str,
        set[int],
    ],
    title_lookup: dict[
        int,
        str,
    ],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:

    evidence = _evidence(
        left_key=variant_key,
        right_key=candidate_key,
        entity_article_sets=(
            entity_article_sets
        ),
        title_lookup=(
            title_lookup
        ),
    )

    row = {
        "rule": rule,
        "safety_tier": (
            safety_tier
        ),
        "entity_group": (
            entity_group
        ),
        "variant_entity": (
            variant_entity
        ),
        "candidate_entity": (
            candidate_entity
        ),
        "variant_entity_key": (
            variant_key
        ),
        "candidate_entity_key": (
            candidate_key
        ),
        "rationale": (
            rationale
        ),
        **evidence,
    }

    if extra:
        row.update(
            extra
        )

    return row


# =============================================================================
# 10. Acronym 후보
# =============================================================================


def _looks_like_acronym_surface(
    surface: str,
) -> bool:
    """
    lower-case canonical surface만 보고 acronym 후보를 추정한다.

    예:
        fck   -> True
        eu    -> True
        dr    -> True
        tv2   -> True

        fc københavn -> False
        european union -> False
    """

    compact = (
        _compact_alnum(
            surface
        )
    )

    if not compact:
        return False

    # 공백/다중 단어 full name은 acronym surface로 보지 않는다.
    if len(
        _alnum_tokens(
            surface
        )
    ) != 1:
        return False

    if not (
        2
        <= len(
            compact
        )
        <= 8
    ):
        return False

    return True


def _acronym_signatures(
    full_surface: str,
) -> set[str]:
    """
    full organization surface가 가질 수 있는 acronym signature를 만든다.

    예:
        "european union"
        -> {"eu"}

        "fc københavn"
        -> {"fk", "fck"}

    왜 여러 signature?
    ------------------
    "FC" 자체가 이미 약어 token이므로:
        first-char 방식: f + k = fk
        prefix-preserving 방식: fc + k = fck

    둘 다 검사한다.
    """

    tokens = [
        token
        for token in _alnum_tokens(
            full_surface
        )
        if token not in (
            ACRONYM_STOPWORDS
        )
    ]

    if len(
        tokens
    ) < 2:
        return set()

    signatures = set()

    # 가장 기본적인 initials.
    initials = "".join(
        token[
            0
        ]
        for token in tokens
        if token
    )

    if (
        2
        <= len(
            initials
        )
        <= 8
    ):
        signatures.add(
            initials
        )

    # FC København -> FCK 같은 경우.
    expanded_parts = []

    for token in tokens:
        if token in (
            ACRONYM_PREFIX_TOKENS
        ):
            expanded_parts.append(
                token
            )
        else:
            expanded_parts.append(
                token[
                    0
                ]
            )

    expanded = "".join(
        expanded_parts
    )

    if (
        2
        <= len(
            expanded
        )
        <= 8
    ):
        signatures.add(
            expanded
        )

    return signatures


def _inspect_acronym_candidates(
    *,
    train_pairs: pl.DataFrame,
    entity_article_sets: dict[
        str,
        set[int],
    ],
    title_lookup: dict[
        int,
        str,
    ],
) -> list[dict[str, Any]]:

    org_entities = (
        train_pairs.filter(
            pl.col(
                "entity_group"
            )
            == TARGET_ENTITY_GROUP
        )
        .select(
            [
                "canonical_entity",
                "canonical_entity_key",
            ]
        )
        .unique()
        .sort(
            "canonical_entity"
        )
    )

    surfaces = [
        {
            "surface": _basic_text(
                row[
                    "canonical_entity"
                ]
            ),
            "key": str(
                row[
                    "canonical_entity_key"
                ]
            ),
        }
        for row in org_entities.iter_rows(
            named=True
        )
    ]

    acronym_rows = [
        row
        for row in surfaces
        if _looks_like_acronym_surface(
            row[
                "surface"
            ]
        )
    ]

    full_rows = [
        row
        for row in surfaces
        if len(
            _alnum_tokens(
                row[
                    "surface"
                ]
            )
        )
        >= 2
    ]

    # signature -> full candidate list
    full_by_signature: dict[
        str,
        list[
            dict[str, str]
        ],
    ] = defaultdict(
        list
    )

    for full_row in full_rows:
        signatures = (
            _acronym_signatures(
                full_row[
                    "surface"
                ]
            )
        )

        for signature in signatures:
            full_by_signature[
                signature
            ].append(
                full_row
            )

    rows = []

    for acronym_row in acronym_rows:
        acronym_surface = (
            acronym_row[
                "surface"
            ]
        )

        acronym_compact = (
            _compact_alnum(
                acronym_surface
            )
        )

        matches = (
            full_by_signature.get(
                acronym_compact,
                [],
            )
        )

        # 자기 자신/동일 surface는 제외.
        matches = [
            match
            for match in matches
            if match[
                "key"
            ]
            != acronym_row[
                "key"
            ]
        ]

        if not matches:
            continue

        candidate_count = len(
            matches
        )

        for full_row in matches:
            evidence = _evidence(
                left_key=(
                    acronym_row[
                        "key"
                    ]
                ),
                right_key=(
                    full_row[
                        "key"
                    ]
                ),
                entity_article_sets=(
                    entity_article_sets
                ),
                title_lookup=(
                    title_lookup
                ),
            )

            variant_df = int(
                evidence[
                    "left_article_df"
                ]
            )

            cooccur = int(
                evidence[
                    "cooccurrence_article_count"
                ]
            )

            if (
                candidate_count
                == 1
                and cooccur
                >= ACRONYM_STRONG_MIN_COOCCURRENCE
                and variant_df
                >= ACRONYM_STRONG_MIN_VARIANT_DF
            ):
                tier = (
                    "SAFE_WITH_GUARDRAIL"
                )

                rationale = (
                    "acronym signature 일치 + "
                    "full-form 후보 유일 + "
                    "Train same-article co-occurrence 반복"
                )

            elif (
                candidate_count
                == 1
                and cooccur
                >= 1
            ):
                tier = (
                    "STRONG_REVIEW"
                )

                rationale = (
                    "acronym signature 일치 + "
                    "후보 유일 + co-occurrence 존재"
                )

            elif candidate_count > 1:
                tier = (
                    "AMBIGUOUS_REVIEW"
                )

                rationale = (
                    "같은 acronym signature에 "
                    "여러 full-form 후보가 존재"
                )

            else:
                tier = (
                    "WEAK_REVIEW"
                )

                rationale = (
                    "acronym signature는 일치하지만 "
                    "same-article evidence가 약함"
                )

            row = _make_candidate_row(
                rule="ACRONYM",
                safety_tier=tier,
                entity_group=(
                    TARGET_ENTITY_GROUP
                ),
                variant_entity=(
                    acronym_surface
                ),
                candidate_entity=(
                    full_row[
                        "surface"
                    ]
                ),
                variant_key=(
                    acronym_row[
                        "key"
                    ]
                ),
                candidate_key=(
                    full_row[
                        "key"
                    ]
                ),
                rationale=(
                    rationale
                ),
                entity_article_sets=(
                    entity_article_sets
                ),
                title_lookup=(
                    title_lookup
                ),
                extra={
                    "signature": (
                        acronym_compact
                    ),
                    "candidate_count_for_variant": int(
                        candidate_count
                    ),
                },
            )

            rows.append(
                row
            )

    return rows


# =============================================================================
# 11. ORG punctuation 후보
# =============================================================================


def _punctuation_signature(
    surface: str,
) -> str:
    """
    punctuation 차이만 비교하기 위한 signature.

    token들을 공백 하나로 연결한다.

    예:
        "f.c. københavn"
        -> "f c københavn"

        "f c københavn"
        -> "f c københavn"

    별도로 compact token pattern도 사용해:
        "f.c. københavn"
        -> ["f", "c", "københavn"]
        -> "fc københavn"

    를 비교할 수 있게 한다.
    """

    return " ".join(
        _alnum_tokens(
            surface
        )
    )


def _collapsed_initial_punctuation_signature(
    surface: str,
) -> str:
    """
    연속 single-letter token을 붙인다.

    예:
        ["f", "c", "københavn"]
        -> ["fc", "københavn"]
        -> "fc københavn"

        ["a", "p", "møller"]
        -> "ap møller"
    """

    tokens = _alnum_tokens(
        surface
    )

    if not tokens:
        return ""

    result = []
    buffer = []

    for token in tokens:
        if (
            len(
                token
            )
            == 1
            and token.isalpha()
        ):
            buffer.append(
                token
            )
            continue

        if buffer:
            result.append(
                "".join(
                    buffer
                )
            )
            buffer = []

        result.append(
            token
        )

    if buffer:
        result.append(
            "".join(
                buffer
            )
        )

    return " ".join(
        result
    )


def _inspect_punctuation_candidates(
    *,
    train_pairs: pl.DataFrame,
    entity_article_sets: dict[
        str,
        set[int],
    ],
    title_lookup: dict[
        int,
        str,
    ],
) -> list[dict[str, Any]]:

    org_entities = (
        train_pairs.filter(
            pl.col(
                "entity_group"
            )
            == TARGET_ENTITY_GROUP
        )
        .select(
            [
                "canonical_entity",
                "canonical_entity_key",
            ]
        )
        .unique()
    )

    rows_raw = [
        {
            "surface": _basic_text(
                row[
                    "canonical_entity"
                ]
            ),
            "key": str(
                row[
                    "canonical_entity_key"
                ]
            ),
        }
        for row in org_entities.iter_rows(
            named=True
        )
    ]

    groups: dict[
        str,
        list[
            dict[str, str]
        ],
    ] = defaultdict(
        list
    )

    for row in rows_raw:
        signature = (
            _collapsed_initial_punctuation_signature(
                row[
                    "surface"
                ]
            )
        )

        if signature:
            groups[
                signature
            ].append(
                row
            )

    rows = []

    for signature, members in (
        groups.items()
    ):
        unique_surfaces = sorted(
            {
                member[
                    "surface"
                ]
                for member in members
            }
        )

        if len(
            unique_surfaces
        ) < 2:
            continue

        # pairwise comparison.
        for i in range(
            len(
                members
            )
        ):
            for j in range(
                i + 1,
                len(
                    members
                ),
            ):
                left = members[
                    i
                ]

                right = members[
                    j
                ]

                if left[
                    "surface"
                ] == right[
                    "surface"
                ]:
                    continue

                # punctuation 차이인지 확인:
                # alnum compact가 동일해야 한다.
                if (
                    _compact_alnum(
                        left[
                            "surface"
                        ]
                    )
                    != _compact_alnum(
                        right[
                            "surface"
                        ]
                    )
                ):
                    continue

                evidence = (
                    _evidence(
                        left_key=(
                            left[
                                "key"
                            ]
                        ),
                        right_key=(
                            right[
                                "key"
                            ]
                        ),
                        entity_article_sets=(
                            entity_article_sets
                        ),
                        title_lookup=(
                            title_lookup
                        ),
                    )
                )

                left_df = int(
                    evidence[
                        "left_article_df"
                    ]
                )

                right_df = int(
                    evidence[
                        "right_article_df"
                    ]
                )

                # canonical target은 더 자주 쓰인 쪽.
                if (
                    right_df
                    > left_df
                ):
                    variant = left
                    canonical = right
                    canonical_df = (
                        right_df
                    )
                else:
                    variant = right
                    canonical = left
                    canonical_df = (
                        left_df
                    )

                group_size = len(
                    members
                )

                if (
                    group_size
                    == 2
                    and canonical_df
                    >= SAFE_BASE_MIN_DF
                ):
                    tier = (
                        "SAFE_WITH_GUARDRAIL"
                    )

                    rationale = (
                        "punctuation 제거 후 동일 표기 + "
                        "collision group 2개 + "
                        "canonical DF guardrail 통과"
                    )

                else:
                    tier = (
                        "REVIEW"
                    )

                    rationale = (
                        "punctuation collision group이 크거나 "
                        "canonical DF가 낮아 수동 검토 필요"
                    )

                rows.append(
                    _make_candidate_row(
                        rule=(
                            "ORG_PUNCTUATION"
                        ),
                        safety_tier=(
                            tier
                        ),
                        entity_group=(
                            TARGET_ENTITY_GROUP
                        ),
                        variant_entity=(
                            variant[
                                "surface"
                            ]
                        ),
                        candidate_entity=(
                            canonical[
                                "surface"
                            ]
                        ),
                        variant_key=(
                            variant[
                                "key"
                            ]
                        ),
                        candidate_key=(
                            canonical[
                                "key"
                            ]
                        ),
                        rationale=(
                            rationale
                        ),
                        entity_article_sets=(
                            entity_article_sets
                        ),
                        title_lookup=(
                            title_lookup
                        ),
                        extra={
                            "signature": (
                                signature
                            ),
                            "candidate_count_for_variant": int(
                                group_size
                                - 1
                            ),
                        },
                    )
                )

    return rows


# =============================================================================
# 12. ORG legal suffix 후보
# =============================================================================


def _strip_legal_suffix(
    surface: str,
) -> tuple[
    str | None,
    str | None,
]:
    """
    surface 끝의 legal suffix를 제거해 base surface를 만든다.

    예:
        "novo nordisk a/s"
        tokens = ["novo", "nordisk", "a", "s"]

        suffix=("a","s")
        -> base="novo nordisk"
    """

    tokens = _alnum_tokens(
        surface
    )

    if not tokens:
        return (
            None,
            None,
        )

    # 긴 suffix부터 먼저 검사.
    suffixes = sorted(
        LEGAL_SUFFIX_PATTERNS,
        key=len,
        reverse=True,
    )

    for suffix in suffixes:
        suffix_list = list(
            suffix
        )

        if (
            len(
                tokens
            )
            <= len(
                suffix_list
            )
        ):
            continue

        if (
            tokens[
                -len(
                    suffix_list
                ):
            ]
            != suffix_list
        ):
            continue

        base_tokens = tokens[
            :-len(
                suffix_list
            )
        ]

        base = " ".join(
            base_tokens
        ).strip()

        if not base:
            continue

        suffix_text = " ".join(
            suffix_list
        )

        return (
            base,
            suffix_text,
        )

    return (
        None,
        None,
    )


def _inspect_legal_suffix_candidates(
    *,
    train_pairs: pl.DataFrame,
    entity_article_sets: dict[
        str,
        set[int],
    ],
    title_lookup: dict[
        int,
        str,
    ],
) -> list[dict[str, Any]]:

    surface_to_key = (
        _build_surface_key_lookup(
            train_pairs,
            TARGET_ENTITY_GROUP,
        )
    )

    rows = []

    for variant_surface, variant_key in (
        surface_to_key.items()
    ):
        base_surface, suffix = (
            _strip_legal_suffix(
                variant_surface
            )
        )

        if (
            not base_surface
            or not suffix
        ):
            continue

        base_key = (
            surface_to_key.get(
                base_surface
            )
        )

        # base가 Train vocab에 실제 존재하는 경우만 후보.
        if not base_key:
            continue

        if (
            base_key
            == variant_key
        ):
            continue

        evidence = (
            _evidence(
                left_key=(
                    variant_key
                ),
                right_key=(
                    base_key
                ),
                entity_article_sets=(
                    entity_article_sets
                ),
                title_lookup=(
                    title_lookup
                ),
            )
        )

        base_df = int(
            evidence[
                "right_article_df"
            ]
        )

        variant_df = int(
            evidence[
                "left_article_df"
            ]
        )

        if (
            base_df
            >= SAFE_BASE_MIN_DF
            and base_df
            >= variant_df
        ):
            tier = (
                "SAFE_WITH_GUARDRAIL"
            )

            rationale = (
                "legal suffix 제거 base가 같은 ORG Train vocab에 존재 + "
                "base DF >= variant DF + base DF guardrail"
            )
        else:
            tier = (
                "REVIEW"
            )

            rationale = (
                "legal suffix base는 존재하지만 DF guardrail이 약함"
            )

        rows.append(
            _make_candidate_row(
                rule=(
                    "ORG_LEGAL_SUFFIX"
                ),
                safety_tier=(
                    tier
                ),
                entity_group=(
                    TARGET_ENTITY_GROUP
                ),
                variant_entity=(
                    variant_surface
                ),
                candidate_entity=(
                    base_surface
                ),
                variant_key=(
                    variant_key
                ),
                candidate_key=(
                    base_key
                ),
                rationale=(
                    rationale
                ),
                entity_article_sets=(
                    entity_article_sets
                ),
                title_lookup=(
                    title_lookup
                ),
                extra={
                    "signature": (
                        suffix
                    ),
                    "candidate_count_for_variant": 1,
                },
            )
        )

    return rows


# =============================================================================
# 13. ORG short form 후보
# =============================================================================


def _boundary_short_form_base(
    surface: str,
) -> list[
    tuple[
        str,
        str,
    ]
]:
    """
    organization descriptor가 앞/뒤에 붙은 경우만
    short-form base 후보를 만든다.

    generic containment:
        "abc" in "abc something"
    같은 것은 하지 않는다.

    예:
        "fc barcelona"
        -> "barcelona"

        "manchester united fc"
        -> "manchester united"

        "football club copenhagen"
        -> "copenhagen" 또는 "club copenhagen" 같은 난폭한 제거는 하지 않는다.

    반환:
        [(base_surface, removed_descriptor), ...]
    """

    tokens = _alnum_tokens(
        surface
    )

    if len(
        tokens
    ) < 2:
        return []

    candidates: list[
        tuple[
            str,
            str,
        ]
    ] = []

    # 앞쪽 1~2 token 제거.
    for diff in range(
        1,
        SHORT_FORM_MAX_BOUNDARY_TOKEN_DIFF
        + 1,
    ):
        if len(
            tokens
        ) <= diff:
            continue

        removed = tokens[
            :diff
        ]

        if not all(
            token in (
                SHORT_FORM_BOUNDARY_TOKENS
            )
            for token in removed
        ):
            continue

        base_tokens = tokens[
            diff:
        ]

        base = " ".join(
            base_tokens
        )

        candidates.append(
            (
                base,
                "PREFIX:"
                + " ".join(
                    removed
                ),
            )
        )

    # 뒤쪽 1~2 token 제거.
    for diff in range(
        1,
        SHORT_FORM_MAX_BOUNDARY_TOKEN_DIFF
        + 1,
    ):
        if len(
            tokens
        ) <= diff:
            continue

        removed = tokens[
            -diff:
        ]

        if not all(
            token in (
                SHORT_FORM_BOUNDARY_TOKENS
            )
            for token in removed
        ):
            continue

        base_tokens = tokens[
            :-diff
        ]

        base = " ".join(
            base_tokens
        )

        candidates.append(
            (
                base,
                "SUFFIX:"
                + " ".join(
                    removed
                ),
            )
        )

    # 중복 제거.
    unique = []
    seen = set()

    for item in candidates:
        if item in seen:
            continue

        seen.add(
            item
        )

        unique.append(
            item
        )

    return unique


def _inspect_short_form_candidates(
    *,
    train_pairs: pl.DataFrame,
    entity_article_sets: dict[
        str,
        set[int],
    ],
    title_lookup: dict[
        int,
        str,
    ],
) -> list[dict[str, Any]]:

    surface_to_key = (
        _build_surface_key_lookup(
            train_pairs,
            TARGET_ENTITY_GROUP,
        )
    )

    rows = []

    for long_surface, long_key in (
        surface_to_key.items()
    ):
        base_candidates = (
            _boundary_short_form_base(
                long_surface
            )
        )

        for (
            short_surface,
            removed_descriptor,
        ) in base_candidates:

            if (
                len(
                    short_surface
                )
                < SHORT_FORM_MIN_CHAR_LENGTH
            ):
                continue

            short_key = (
                surface_to_key.get(
                    short_surface
                )
            )

            if not short_key:
                continue

            if short_key == long_key:
                continue

            evidence = _evidence(
                left_key=(
                    short_key
                ),
                right_key=(
                    long_key
                ),
                entity_article_sets=(
                    entity_article_sets
                ),
                title_lookup=(
                    title_lookup
                ),
            )

            cooccur = int(
                evidence[
                    "cooccurrence_article_count"
                ]
            )

            # short form은 절대 SAFE로 두지 않는다.
            #
            # co-occurrence가 있으면 "상대적으로 볼 가치가 큰 후보",
            # 없으면 "약한 진단 후보".
            if cooccur >= 1:
                tier = (
                    "REVIEW_ONLY_WITH_COOCCURRENCE"
                )

                rationale = (
                    "ORG boundary descriptor 제거 후 base가 Train vocab에 존재 + "
                    "same-article evidence 존재. "
                    "하지만 short form은 ambiguity 위험 때문에 자동 적용 금지."
                )
            else:
                tier = (
                    "REVIEW_ONLY_WEAK"
                )

                rationale = (
                    "ORG boundary descriptor 제거 후 base가 존재하지만 "
                    "same-article evidence 없음. 자동 적용 금지."
                )

            rows.append(
                _make_candidate_row(
                    rule=(
                        "ORG_SHORT_FORM"
                    ),
                    safety_tier=(
                        tier
                    ),
                    entity_group=(
                        TARGET_ENTITY_GROUP
                    ),
                    variant_entity=(
                        short_surface
                    ),
                    candidate_entity=(
                        long_surface
                    ),
                    variant_key=(
                        short_key
                    ),
                    candidate_key=(
                        long_key
                    ),
                    rationale=(
                        rationale
                    ),
                    entity_article_sets=(
                        entity_article_sets
                    ),
                    title_lookup=(
                        title_lookup
                    ),
                    extra={
                        "signature": (
                            removed_descriptor
                        ),
                        "candidate_count_for_variant": 1,
                    },
                )
            )

    return rows


# =============================================================================
# 14. 중복 후보 제거
# =============================================================================


def _deduplicate_candidate_rows(
    rows: list[
        dict[str, Any]
    ],
) -> list[
    dict[str, Any]
]:
    """
    같은 rule + variant + candidate pair가 중복 생성되는 것을 막는다.
    """

    result = []
    seen = set()

    for row in rows:
        key = (
            str(
                row[
                    "rule"
                ]
            ),
            str(
                row[
                    "variant_entity_key"
                ]
            ),
            str(
                row[
                    "candidate_entity_key"
                ]
            ),
        )

        if key in seen:
            continue

        seen.add(
            key
        )

        result.append(
            row
        )

    return result


# =============================================================================
# 15. DataFrame 변환
# =============================================================================


CANDIDATE_SCHEMA = {
    "rule": pl.String,
    "safety_tier": pl.String,
    "entity_group": pl.String,
    "variant_entity": pl.String,
    "candidate_entity": pl.String,
    "variant_entity_key": pl.String,
    "candidate_entity_key": pl.String,
    "rationale": pl.String,
    "left_article_df": pl.Int64,
    "right_article_df": pl.Int64,
    "cooccurrence_article_count": pl.Int64,
    "cooccurrence_ratio_min_df": pl.Float64,
    "sample_cooccurrence_article_ids": pl.List(
        pl.Int64
    ),
    "sample_cooccurrence_titles": pl.List(
        pl.String
    ),
    "signature": pl.String,
    "candidate_count_for_variant": pl.Int64,
}


def _rows_to_df(
    rows: list[
        dict[str, Any]
    ],
) -> pl.DataFrame:

    if not rows:
        return pl.DataFrame(
            schema=(
                CANDIDATE_SCHEMA
            )
        )

    normalized = []

    for row in rows:
        normalized.append(
            {
                "rule": str(
                    row.get(
                        "rule",
                        "",
                    )
                ),
                "safety_tier": str(
                    row.get(
                        "safety_tier",
                        "",
                    )
                ),
                "entity_group": str(
                    row.get(
                        "entity_group",
                        "",
                    )
                ),
                "variant_entity": str(
                    row.get(
                        "variant_entity",
                        "",
                    )
                ),
                "candidate_entity": str(
                    row.get(
                        "candidate_entity",
                        "",
                    )
                ),
                "variant_entity_key": str(
                    row.get(
                        "variant_entity_key",
                        "",
                    )
                ),
                "candidate_entity_key": str(
                    row.get(
                        "candidate_entity_key",
                        "",
                    )
                ),
                "rationale": str(
                    row.get(
                        "rationale",
                        "",
                    )
                ),
                "left_article_df": int(
                    row.get(
                        "left_article_df",
                        0,
                    )
                ),
                "right_article_df": int(
                    row.get(
                        "right_article_df",
                        0,
                    )
                ),
                "cooccurrence_article_count": int(
                    row.get(
                        "cooccurrence_article_count",
                        0,
                    )
                ),
                "cooccurrence_ratio_min_df": float(
                    row.get(
                        "cooccurrence_ratio_min_df",
                        0.0,
                    )
                ),
                "sample_cooccurrence_article_ids": [
                    int(
                        value
                    )
                    for value in (
                        row.get(
                            "sample_cooccurrence_article_ids",
                            [],
                        )
                        or []
                    )
                ],
                "sample_cooccurrence_titles": [
                    str(
                        value
                    )
                    for value in (
                        row.get(
                            "sample_cooccurrence_titles",
                            [],
                        )
                        or []
                    )
                ],
                "signature": str(
                    row.get(
                        "signature",
                        "",
                    )
                ),
                "candidate_count_for_variant": int(
                    row.get(
                        "candidate_count_for_variant",
                        0,
                    )
                ),
            }
        )

    return (
        pl.DataFrame(
            normalized,
            schema=(
                CANDIDATE_SCHEMA
            ),
        )
        .sort(
            [
                "rule",
                "safety_tier",
                "cooccurrence_article_count",
                "variant_entity",
                "candidate_entity",
            ],
            descending=[
                False,
                False,
                True,
                False,
                False,
            ],
        )
    )


# =============================================================================
# 16. Summary
# =============================================================================


def _count_by_column(
    df: pl.DataFrame,
    column: str,
) -> dict[str, int]:

    if df.height == 0:
        return {}

    result = {}

    grouped = (
        df.group_by(
            column
        )
        .len()
        .sort(
            column
        )
    )

    for row in grouped.iter_rows(
        named=True
    ):
        result[
            str(
                row[
                    column
                ]
            )
        ] = int(
            row[
                "len"
            ]
        )

    return result


def _top_examples(
    df: pl.DataFrame,
    count: int = 15,
) -> list[str]:

    if df.height == 0:
        return []

    top = (
        df.sort(
            [
                "cooccurrence_article_count",
                "cooccurrence_ratio_min_df",
                "left_article_df",
            ],
            descending=[
                True,
                True,
                True,
            ],
        )
        .head(
            count
        )
    )

    lines = []

    for row in top.iter_rows(
        named=True
    ):
        lines.append(
            (
                f"- [{row['rule']}] "
                f"{row['variant_entity_key']} "
                f"-> {row['candidate_entity_key']} "
                f"| tier={row['safety_tier']} "
                f"| df={row['left_article_df']}/{row['right_article_df']} "
                f"| cooccur={row['cooccurrence_article_count']} "
                f"| ratio={row['cooccurrence_ratio_min_df']:.3f}"
            )
        )

    return lines


def _write_summary(
    *,
    train_pairs: pl.DataFrame,
    all_df: pl.DataFrame,
    acronym_df: pl.DataFrame,
    punctuation_df: pl.DataFrame,
    legal_df: pl.DataFrame,
    short_df: pl.DataFrame,
) -> None:

    unique_train_entities = (
        train_pairs.select(
            "canonical_entity_key"
        )
        .unique()
        .height
    )

    unique_train_org_entities = (
        train_pairs.filter(
            pl.col(
                "entity_group"
            )
            == TARGET_ENTITY_GROUP
        )
        .select(
            "canonical_entity_key"
        )
        .unique()
        .height
    )

    safe_df = (
        all_df.filter(
            pl.col(
                "safety_tier"
            )
            == "SAFE_WITH_GUARDRAIL"
        )
        if all_df.height
        else pl.DataFrame(
            schema=(
                CANDIDATE_SCHEMA
            )
        )
    )

    lines = [
        "=" * 100,
        "Normalization v3 Candidate Inspection",
        "=" * 100,
        "",
        "IMPORTANT:",
        "- inspection only",
        "- normalize_v2 files are NOT modified",
        "- Event clustering is NOT rerun",
        "- Train-used articles only",
        "",
        f"train_article_entity_pair_count={train_pairs.height}",
        f"unique_train_entity_count={unique_train_entities}",
        f"unique_train_org_entity_count={unique_train_org_entities}",
        "",
        "Candidate counts:",
        f"all_candidate_count={all_df.height}",
        f"acronym_candidate_count={acronym_df.height}",
        f"org_punctuation_candidate_count={punctuation_df.height}",
        f"org_legal_suffix_candidate_count={legal_df.height}",
        f"org_short_form_candidate_count={short_df.height}",
        f"safe_with_guardrail_candidate_count={safe_df.height}",
        "",
        "Rule counts:",
    ]

    for (
        rule,
        count,
    ) in _count_by_column(
        all_df,
        "rule",
    ).items():
        lines.append(
            f"- {rule}: {count}"
        )

    lines.extend(
        [
            "",
            "Safety tier counts:",
        ]
    )

    for (
        tier,
        count,
    ) in _count_by_column(
        all_df,
        "safety_tier",
    ).items():
        lines.append(
            f"- {tier}: {count}"
        )

    lines.extend(
        [
            "",
            "Top evidence examples:",
            *_top_examples(
                all_df
            ),
            "",
            "Interpretation guide:",
            "- SAFE_WITH_GUARDRAIL: 다음 수동 검토 우선 후보. 아직 실제 적용 전 검증 필요.",
            "- STRONG_REVIEW: 꽤 유망하지만 자동 mapping 전 확인 필요.",
            "- AMBIGUOUS_REVIEW: acronym 하나에 여러 full-form 후보 등 ambiguity 존재.",
            "- REVIEW / REVIEW_ONLY*: 자동 적용 금지. 정성 검토용.",
            "",
            "Next step:",
            "1. 후보 수와 실제 예시를 확인한다.",
            "2. SAFE_WITH_GUARDRAIL / STRONG_REVIEW를 사람이 검토한다.",
            "3. precision이 충분한 rule만 normalize_v3 mapping으로 구현한다.",
            "4. 그 후에만 canonical DF/high-DF/IDF/Event를 재계산한다.",
        ]
    )

    SUMMARY_PATH.write_text(
        "\n".join(
            lines
        )
        + "\n",
        encoding="utf-8",
    )


# =============================================================================
# 17. Main
# =============================================================================


def inspect_normalization_v3() -> dict[str, Any]:

    print(
        "=" * 100
    )
    print(
        "Normalization v3 Candidate Inspection 시작"
    )
    print(
        "=" * 100
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # STEP 1. Train-used canonical entity pair
    # -------------------------------------------------------------------------
    print()
    print(
        "[STEP 1] normalize_v2 Train entity universe 로드"
    )

    train_pairs = (
        _load_train_entity_pairs()
    )

    title_lookup = (
        _load_title_lookup()
    )

    entity_article_sets = (
        _build_entity_article_sets(
            train_pairs
        )
    )

    unique_entity_count = (
        train_pairs.select(
            "canonical_entity_key"
        )
        .unique()
        .height
    )

    unique_org_count = (
        train_pairs.filter(
            pl.col(
                "entity_group"
            )
            == TARGET_ENTITY_GROUP
        )
        .select(
            "canonical_entity_key"
        )
        .unique()
        .height
    )

    print(
        f"train_article_entity_pair_count = {train_pairs.height}"
    )
    print(
        f"unique_train_entity_count = {unique_entity_count}"
    )
    print(
        f"unique_train_org_entity_count = {unique_org_count}"
    )

    # -------------------------------------------------------------------------
    # STEP 2. Acronym
    # -------------------------------------------------------------------------
    print()
    print(
        "[STEP 2] ACRONYM 후보 조사"
    )

    acronym_rows = (
        _inspect_acronym_candidates(
            train_pairs=(
                train_pairs
            ),
            entity_article_sets=(
                entity_article_sets
            ),
            title_lookup=(
                title_lookup
            ),
        )
    )

    acronym_rows = (
        _deduplicate_candidate_rows(
            acronym_rows
        )
    )

    acronym_df = (
        _rows_to_df(
            acronym_rows
        )
    )

    acronym_df.write_parquet(
        ACRONYM_CANDIDATES_PATH,
        compression="zstd",
    )

    print(
        f"acronym_candidate_count = {acronym_df.height}"
    )

    # -------------------------------------------------------------------------
    # STEP 3. Punctuation
    # -------------------------------------------------------------------------
    print()
    print(
        "[STEP 3] ORG_PUNCTUATION 후보 조사"
    )

    punctuation_rows = (
        _inspect_punctuation_candidates(
            train_pairs=(
                train_pairs
            ),
            entity_article_sets=(
                entity_article_sets
            ),
            title_lookup=(
                title_lookup
            ),
        )
    )

    punctuation_rows = (
        _deduplicate_candidate_rows(
            punctuation_rows
        )
    )

    punctuation_df = (
        _rows_to_df(
            punctuation_rows
        )
    )

    punctuation_df.write_parquet(
        PUNCTUATION_CANDIDATES_PATH,
        compression="zstd",
    )

    print(
        f"org_punctuation_candidate_count = {punctuation_df.height}"
    )

    # -------------------------------------------------------------------------
    # STEP 4. Legal suffix
    # -------------------------------------------------------------------------
    print()
    print(
        "[STEP 4] ORG_LEGAL_SUFFIX 후보 조사"
    )

    legal_rows = (
        _inspect_legal_suffix_candidates(
            train_pairs=(
                train_pairs
            ),
            entity_article_sets=(
                entity_article_sets
            ),
            title_lookup=(
                title_lookup
            ),
        )
    )

    legal_rows = (
        _deduplicate_candidate_rows(
            legal_rows
        )
    )

    legal_df = (
        _rows_to_df(
            legal_rows
        )
    )

    legal_df.write_parquet(
        LEGAL_SUFFIX_CANDIDATES_PATH,
        compression="zstd",
    )

    print(
        f"org_legal_suffix_candidate_count = {legal_df.height}"
    )

    # -------------------------------------------------------------------------
    # STEP 5. Short form
    # -------------------------------------------------------------------------
    print()
    print(
        "[STEP 5] ORG_SHORT_FORM 후보 조사"
    )

    short_rows = (
        _inspect_short_form_candidates(
            train_pairs=(
                train_pairs
            ),
            entity_article_sets=(
                entity_article_sets
            ),
            title_lookup=(
                title_lookup
            ),
        )
    )

    short_rows = (
        _deduplicate_candidate_rows(
            short_rows
        )
    )

    short_df = (
        _rows_to_df(
            short_rows
        )
    )

    short_df.write_parquet(
        SHORT_FORM_CANDIDATES_PATH,
        compression="zstd",
    )

    print(
        f"org_short_form_candidate_count = {short_df.height}"
    )

    # -------------------------------------------------------------------------
    # STEP 6. 전체 합치기
    # -------------------------------------------------------------------------
    print()
    print(
        "[STEP 6] 전체 후보 통합"
    )

    all_rows = (
        acronym_rows
        + punctuation_rows
        + legal_rows
        + short_rows
    )

    all_rows = (
        _deduplicate_candidate_rows(
            all_rows
        )
    )

    all_df = (
        _rows_to_df(
            all_rows
        )
    )

    all_df.write_parquet(
        ALL_CANDIDATES_PATH,
        compression="zstd",
    )

    safe_count = (
        all_df.filter(
            pl.col(
                "safety_tier"
            )
            == "SAFE_WITH_GUARDRAIL"
        ).height
        if all_df.height
        else 0
    )

    print(
        f"all_candidate_count = {all_df.height}"
    )
    print(
        f"safe_with_guardrail_candidate_count = {safe_count}"
    )

    # -------------------------------------------------------------------------
    # STEP 7. Summary
    # -------------------------------------------------------------------------
    _write_summary(
        train_pairs=(
            train_pairs
        ),
        all_df=(
            all_df
        ),
        acronym_df=(
            acronym_df
        ),
        punctuation_df=(
            punctuation_df
        ),
        legal_df=(
            legal_df
        ),
        short_df=(
            short_df
        ),
    )

    result = {
        "status": "SUCCESS",
        "input_article_entities_path": str(
            ARTICLE_ENTITIES_PATH
        ),
        "train_article_entity_pair_count": int(
            train_pairs.height
        ),
        "unique_train_entity_count": int(
            unique_entity_count
        ),
        "unique_train_org_entity_count": int(
            unique_org_count
        ),
        "all_candidate_count": int(
            all_df.height
        ),
        "acronym_candidate_count": int(
            acronym_df.height
        ),
        "org_punctuation_candidate_count": int(
            punctuation_df.height
        ),
        "org_legal_suffix_candidate_count": int(
            legal_df.height
        ),
        "org_short_form_candidate_count": int(
            short_df.height
        ),
        "safe_with_guardrail_candidate_count": int(
            safe_count
        ),
        "all_candidates_path": str(
            ALL_CANDIDATES_PATH
        ),
        "summary_path": str(
            SUMMARY_PATH
        ),
    }

    print()
    print(
        "=" * 100
    )
    print(
        "Normalization v3 Candidate Inspection 완료"
    )
    print(
        "=" * 100
    )

    pprint(
        result
    )

    return result


def main() -> None:
    inspect_normalization_v3()


if __name__ == "__main__":
    main()
