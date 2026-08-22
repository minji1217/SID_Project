from __future__ import annotations

"""
Inspect Wikidata Context Disambiguation
=======================================

목적
----
Wikidata TYPE filtering까지 통과한 후보가 여러 개인 Entity를
"기사 문맥(article context)"으로 다시 순위화한다.

현재 단계 예:

    PER::putin

    TYPE_MATCH 후보:
      Q1 Vladimir Putin
      Q2 Igor Putin
      Q3 Vera Putina

여기까지는 모두 "사람"이라 TYPE만으로 구분할 수 없다.

그래서 이 파일은 해당 Entity가 나온 기사에서:

    - 기사 제목(title)
    - 같은 기사에 같이 나온 다른 canonical Entity들

을 가져와서 각 Wikidata 후보의:

    - label
    - description
    - 검색 match_text

와 얼마나 관련이 있는지 계산한다.

중요
----
이 스크립트는 아직 최종 Entity Linking을 확정하지 않는다.

즉:
- GPT 호출 안 함
- 최종 QID 확정 안 함
- entity_processing.py 수정 안 함
- Event clustering 재실행 안 함

이번 단계의 목적:

    TYPE_MATCH 후보 여러 개
           ↓
    article context 기반 후보 ranking
           ↓
    어느 정도 자동으로 좁혀지는지 검사
           ↓
    애매한 것만 다음 GPT 단계로 보낼지 결정

입력
----
1. normalize_v2/model_inputs/article_entities.parquet
2. normalize_v2/model_inputs/articles_base.parquet
3. wikidata_type_filter/wikidata_type_matched_candidates.parquet
4. wikidata_type_filter/wikidata_entity_type_filter_status.parquet

출력
----
data/output/experiments/wikidata_context_disambiguation/

1. article_candidate_context_scores.parquet
   - 기사 × Entity × 후보 QID별 문맥 점수

2. article_entity_context_status.parquet
   - 기사 × Entity별 top 후보 / margin / 진단 상태

3. entity_context_summary.parquet
   - canonical Entity 단위로 여러 기사에서 후보가 얼마나 일관되게 top인지 요약

4. context_review_examples.parquet
   - 검토하기 좋은 예시

5. wikidata_context_disambiguation_summary.txt
   - 전체 숫자 요약

실행
----
    python -m src.inspect_wikidata_context_disambiguation
"""

import argparse
import math
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import polars as pl


# =============================================================================
# 1. 경로
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SOURCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "experiments"
    / "normalize_v2"
    / "model_inputs"
)

DEFAULT_ARTICLE_ENTITIES_PATH = (
    DEFAULT_SOURCE_DIR
    / "article_entities.parquet"
)

DEFAULT_ARTICLES_BASE_PATH = (
    DEFAULT_SOURCE_DIR
    / "articles_base.parquet"
)

DEFAULT_TYPE_FILTER_DIR = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "experiments"
    / "wikidata_type_filter"
)

DEFAULT_MATCHED_CANDIDATES_PATH = (
    DEFAULT_TYPE_FILTER_DIR
    / "wikidata_type_matched_candidates.parquet"
)

DEFAULT_ENTITY_FILTER_STATUS_PATH = (
    DEFAULT_TYPE_FILTER_DIR
    / "wikidata_entity_type_filter_status.parquet"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "experiments"
    / "wikidata_context_disambiguation"
)

ARTICLE_CANDIDATE_SCORES_FILENAME = (
    "article_candidate_context_scores.parquet"
)

ARTICLE_ENTITY_STATUS_FILENAME = (
    "article_entity_context_status.parquet"
)

ENTITY_SUMMARY_FILENAME = (
    "entity_context_summary.parquet"
)

REVIEW_EXAMPLES_FILENAME = (
    "context_review_examples.parquet"
)

SUMMARY_FILENAME = (
    "wikidata_context_disambiguation_summary.txt"
)


# =============================================================================
# 2. 문맥 점수 설정
# =============================================================================

# 기사 title token과 후보 description token이 겹치는 정도
DESCRIPTION_OVERLAP_WEIGHT = 1.0

# 후보 label의 token이 기사 context에 실제로 등장하면 강한 신호
LABEL_CONTEXT_WEIGHT = 1.5

# Wikidata 검색 match_text가 원래 surface와 정확히 맞으면 작은 보너스
SEARCH_MATCH_WEIGHT = 0.5

# Wikidata label 자체가 canonical surface와 정확히 같으면 강한 lexical signal
EXACT_LABEL_WEIGHT = 1.0

# top1과 top2의 context score 차이가 이 값 이상이면
# "문맥상 어느 정도 차이가 난다"는 진단 표시를 한다.
# 아직 자동 LINK 확정 기준은 아니다.
CONTEXT_MARGIN_HINT = 0.15

# 긴 본문이 articles_base에 존재할 경우 너무 많은 token을 쓰지 않도록 제한.
MAX_OPTIONAL_TEXT_CHARS = 3000

# 리뷰용 예시 최대 row
DEFAULT_REVIEW_EXAMPLES = 300


# =============================================================================
# 3. 아주 기본적인 stopwords
# =============================================================================
#
# 완전한 NLP stopword 사전이 목적이 아니다.
# description/title에서 지나치게 흔한 기능어가 overlap 점수를
# 부풀리는 것을 막기 위한 최소 목록이다.
# =============================================================================

STOPWORDS = {
    # Danish
    "og", "i", "på", "af", "for", "til", "med", "en", "et", "den",
    "det", "de", "der", "som", "fra", "er", "var", "har", "om", "at",
    "ved", "eller", "ikke", "sin", "sit", "sine", "the",

    # English
    "a", "an", "and", "of", "in", "on", "for", "to", "with", "by",
    "from", "is", "was", "are", "were", "as", "at", "or", "that",
    "this", "his", "her", "their",
}


# =============================================================================
# 4. 입력 필수 컬럼
# =============================================================================

REQUIRED_ARTICLE_ENTITY_COLUMNS = {
    "article_id",
    "entity_group",
    "canonical_entity",
    "canonical_entity_key",
    "is_train_used",
}

REQUIRED_MATCHED_CANDIDATE_COLUMNS = {
    "entity_group",
    "canonical_entity",
    "canonical_entity_key",
    "candidate_rank",
    "qid",
    "label",
    "description",
    "search_language",
    "search_rank",
    "type_filter_status",
}

REQUIRED_ENTITY_STATUS_COLUMNS = {
    "entity_group",
    "canonical_entity",
    "canonical_entity_key",
    "diagnostic_status",
    "type_match_count",
}


# =============================================================================
# 5. 공통 검증
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
    missing = required - set(
        df.columns
    )

    if missing:
        raise ValueError(
            f"{description}에 필요한 컬럼이 없습니다: "
            + ", ".join(
                sorted(missing)
            )
        )


# =============================================================================
# 6. 문자열 정규화 / tokenization
# =============================================================================


def _normalize_text(
    text: Any,
) -> str:
    """
    비교용 텍스트 정규화.

    예:
        "  Vladimir   Putin "
            ↓
        "vladimir putin"
    """

    value = unicodedata.normalize(
        "NFKC",
        str(text or ""),
    )

    value = value.casefold()

    value = re.sub(
        r"\s+",
        " ",
        value,
    ).strip()

    return value


def _tokenize(
    text: Any,
) -> list[str]:
    """
    Unicode 문자/숫자 중심의 간단 tokenization.

    예:
        "Russian politician and president"
            ↓
        ["russian", "politician", "president"]

    너무 짧은 1글자 token과 stopword는 제거한다.
    """

    normalized = _normalize_text(
        text
    )

    tokens = re.findall(
        r"[^\W_]+",
        normalized,
        flags=re.UNICODE,
    )

    return [
        token
        for token in tokens
        if len(token) >= 2
        and token not in STOPWORDS
    ]


def _unique_tokens(
    text: Any,
) -> set[str]:
    return set(
        _tokenize(
            text
        )
    )


# =============================================================================
# 7. Train article_entities 로드
# =============================================================================


def _load_train_article_entities(
    path: Path,
) -> pl.DataFrame:
    """
    normalize_v2의 Train-used Entity mention만 읽는다.

    같은 canonical_entity_key가 여러 기사에서 등장할 수 있다.

    예:
        article 10  PER::putin
        article 20  PER::putin
        article 30  PER::putin

    context disambiguation은 이 세 article을 각각 볼 수 있도록
    article_id를 유지한다.
    """

    _require_file(
        path,
        "normalize_v2 article_entities",
    )

    df = pl.read_parquet(
        path
    )

    _require_columns(
        df,
        REQUIRED_ARTICLE_ENTITY_COLUMNS,
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
                ).cast(pl.Int64),

                pl.col(
                    "entity_group"
                ).cast(pl.String),

                pl.col(
                    "canonical_entity"
                ).cast(pl.String),

                pl.col(
                    "canonical_entity_key"
                ).cast(pl.String),
            ]
        )
        .unique()
    )


# =============================================================================
# 8. articles_base에서 article text/context 로드
# =============================================================================


def _load_article_text_lookup(
    path: Path,
) -> dict[int, dict[str, str]]:
    """
    articles_base.parquet에서 기사 context를 읽는다.

    title은 반드시 사용한다.

    그리고 아래처럼 본문 비슷한 컬럼이 실제 파일에 존재하면
    자동으로 추가 사용한다:

        subtitle
        sub_title
        lead
        summary
        description
        body
        text
        content
        article_text

    즉 현재 프로젝트 파일에 title밖에 없어도 실행 가능하고,
    나중에 text 컬럼이 추가되어도 코드 수정 없이 활용할 수 있다.
    """

    _require_file(
        path,
        "articles_base",
    )

    df = pl.read_parquet(
        path
    )

    _require_columns(
        df,
        {
            "article_id",
            "title",
        },
        "articles_base.parquet",
    )

    optional_candidates = [
        "subtitle",
        "sub_title",
        "lead",
        "summary",
        "description",
        "body",
        "text",
        "content",
        "article_text",
    ]

    optional_columns = [
        column
        for column in optional_candidates
        if column in df.columns
    ]

    selected_columns = [
        "article_id",
        "title",
        *optional_columns,
    ]

    lookup: dict[
        int,
        dict[str, str],
    ] = {}

    for row in df.select(
        selected_columns
    ).iter_rows(
        named=True
    ):
        article_id = int(
            row[
                "article_id"
            ]
        )

        title = str(
            row.get(
                "title",
                "",
            )
            or ""
        )

        optional_parts = []

        for column in optional_columns:
            value = str(
                row.get(
                    column,
                    "",
                )
                or ""
            ).strip()

            if value:
                optional_parts.append(
                    value
                )

        optional_text = " ".join(
            optional_parts
        )

        if len(
            optional_text
        ) > MAX_OPTIONAL_TEXT_CHARS:
            optional_text = (
                optional_text[
                    :MAX_OPTIONAL_TEXT_CHARS
                ]
            )

        lookup[
            article_id
        ] = {
            "title": title,
            "optional_text": (
                optional_text
            ),
            "optional_columns_used": (
                ",".join(
                    optional_columns
                )
            ),
        }

    return lookup


# =============================================================================
# 9. article_id -> 같이 나온 Entity context
# =============================================================================


def _build_article_entity_context(
    train_entities: pl.DataFrame,
) -> dict[
    int,
    list[
        dict[str, str]
    ],
]:
    """
    article_id마다 같은 기사에 등장한 canonical Entity들을 모은다.

    예:
        article_id = 100

        PER::putin
        LOC::ukraine
        LOC::rusland
        ORG::nato

    PER::putin을 판별할 때 context로:

        ukraine / rusland / nato

    를 활용할 수 있다.
    """

    result: dict[
        int,
        list[
            dict[str, str]
        ],
    ] = defaultdict(
        list
    )

    for row in train_entities.iter_rows(
        named=True
    ):
        article_id = int(
            row[
                "article_id"
            ]
        )

        result[
            article_id
        ].append(
            {
                "entity_group": str(
                    row[
                        "entity_group"
                    ]
                ),
                "canonical_entity": str(
                    row[
                        "canonical_entity"
                    ]
                ),
                "canonical_entity_key": str(
                    row[
                        "canonical_entity_key"
                    ]
                ),
            }
        )

    return dict(
        result
    )


# =============================================================================
# 10. TYPE_MATCH candidate 로드
# =============================================================================


def _load_matched_candidates(
    path: Path,
) -> pl.DataFrame:
    """
    이전 단계에서 TYPE_MATCH 판정을 받은 후보만 읽는다.

    파일 자체가 matched subset이지만,
    안전하게 type_filter_status도 다시 확인한다.
    """

    _require_file(
        path,
        "Wikidata TYPE matched candidates",
    )

    df = pl.read_parquet(
        path
    )

    _require_columns(
        df,
        REQUIRED_MATCHED_CANDIDATE_COLUMNS,
        "wikidata_type_matched_candidates.parquet",
    )

    return (
        df.filter(
            pl.col(
                "type_filter_status"
            )
            == "TYPE_MATCH"
        )
    )


# =============================================================================
# 11. Entity TYPE-filter status 로드
# =============================================================================


def _load_entity_filter_status(
    path: Path,
) -> pl.DataFrame:
    _require_file(
        path,
        "Wikidata entity TYPE filter status",
    )

    df = pl.read_parquet(
        path
    )

    _require_columns(
        df,
        REQUIRED_ENTITY_STATUS_COLUMNS,
        "wikidata_entity_type_filter_status.parquet",
    )

    return df


# =============================================================================
# 12. Candidate lookup
# =============================================================================


def _build_candidate_lookup(
    matched_candidates: pl.DataFrame,
) -> dict[
    str,
    list[
        dict[str, Any]
    ],
]:
    """
    canonical_entity_key -> TYPE_MATCH Wikidata 후보들.
    """

    result: dict[
        str,
        list[
            dict[str, Any]
        ],
    ] = defaultdict(
        list
    )

    for row in matched_candidates.iter_rows(
        named=True
    ):
        key = str(
            row[
                "canonical_entity_key"
            ]
        )

        result[
            key
        ].append(
            row
        )

    for key in result:
        result[
            key
        ].sort(
            key=lambda row: (
                int(
                    row.get(
                        "candidate_rank",
                        999999,
                    )
                ),
                str(
                    row.get(
                        "qid",
                        "",
                    )
                ),
            )
        )

    return dict(
        result
    )


# =============================================================================
# 13. 한 article에서 target Entity 제외한 co-entity context
# =============================================================================


def _co_entity_context(
    *,
    article_id: int,
    target_key: str,
    article_entity_context: dict[
        int,
        list[
            dict[str, str]
        ],
    ],
) -> tuple[
    str,
    list[str],
]:
    rows = (
        article_entity_context.get(
            article_id,
            [],
        )
    )

    surfaces = sorted(
        {
            str(
                row[
                    "canonical_entity"
                ]
            )
            for row in rows
            if str(
                row[
                    "canonical_entity_key"
                ]
            )
            != target_key
        }
    )

    return (
        " ".join(
            surfaces
        ),
        surfaces,
    )


# =============================================================================
# 14. Candidate별 context score 계산
# =============================================================================


def _score_candidate(
    *,
    surface: str,
    title: str,
    optional_text: str,
    co_entity_text: str,
    candidate: dict[str, Any],
) -> dict[str, Any]:
    """
    매우 단순하고 해석 가능한 lexical context score.

    사용 신호
    ---------
    1. EXACT LABEL
       canonical surface와 Wikidata label이 정확히 같은가?

    2. SEARCH MATCH
       원래 Wikidata 검색에서 match_text가 surface와 같은가?

    3. LABEL CONTEXT
       후보 label의 token이 기사 title/본문/co-entity context에도 나오는가?

    4. DESCRIPTION OVERLAP
       후보 description token과 기사 context token이 얼마나 겹치는가?

    이 점수는 "최종 Entity Linking 점수"가 아니다.
    다음 GPT 단계 전에 deterministic context가 얼마나 도움이 되는지
    검사하기 위한 diagnostic score다.
    """

    candidate_label = str(
        candidate.get(
            "label",
            "",
        )
        or ""
    )

    candidate_description = str(
        candidate.get(
            "description",
            "",
        )
        or ""
    )

    wikidata_label_da = str(
        candidate.get(
            "wikidata_label_da",
            "",
        )
        or ""
    )

    wikidata_label_en = str(
        candidate.get(
            "wikidata_label_en",
            "",
        )
        or ""
    )

    wikidata_description_da = str(
        candidate.get(
            "wikidata_description_da",
            "",
        )
        or ""
    )

    wikidata_description_en = str(
        candidate.get(
            "wikidata_description_en",
            "",
        )
        or ""
    )

    match_text = str(
        candidate.get(
            "match_text",
            "",
        )
        or ""
    )

    # -------------------------------------------------------------------------
    # Article context
    # -------------------------------------------------------------------------
    article_context = " ".join(
        [
            title,
            optional_text,
            co_entity_text,
        ]
    )

    article_tokens = (
        _unique_tokens(
            article_context
        )
    )

    # -------------------------------------------------------------------------
    # Candidate label / description
    # -------------------------------------------------------------------------
    label_variants = [
        candidate_label,
        wikidata_label_da,
        wikidata_label_en,
    ]

    description_variants = [
        candidate_description,
        wikidata_description_da,
        wikidata_description_en,
    ]

    label_text = " ".join(
        value
        for value in label_variants
        if value
    )

    description_text = " ".join(
        value
        for value in description_variants
        if value
    )

    label_tokens = (
        _unique_tokens(
            label_text
        )
    )

    description_tokens = (
        _unique_tokens(
            description_text
        )
    )

    # -------------------------------------------------------------------------
    # 1) surface-label exact
    # -------------------------------------------------------------------------
    surface_norm = (
        _normalize_text(
            surface
        )
    )

    normalized_labels = {
        _normalize_text(
            value
        )
        for value in label_variants
        if value
    }

    exact_label_match = int(
        surface_norm
        in normalized_labels
    )

    # -------------------------------------------------------------------------
    # 2) search match exact
    # -------------------------------------------------------------------------
    search_match_exact = int(
        bool(
            match_text
        )
        and _normalize_text(
            match_text
        )
        == surface_norm
    )

    # -------------------------------------------------------------------------
    # 3) label-context overlap
    # -------------------------------------------------------------------------
    label_context_overlap = (
        label_tokens
        & article_tokens
    )

    label_context_ratio = (
        len(
            label_context_overlap
        )
        / max(
            len(
                label_tokens
            ),
            1,
        )
    )

    # -------------------------------------------------------------------------
    # 4) description-context overlap
    # -------------------------------------------------------------------------
    description_context_overlap = (
        description_tokens
        & article_tokens
    )

    description_overlap_ratio = (
        len(
            description_context_overlap
        )
        / max(
            len(
                description_tokens
            ),
            1,
        )
    )

    # description이 매우 길면 ratio가 너무 작아질 수 있으므로
    # raw overlap count도 약하게 반영한다.
    description_overlap_count_score = (
        min(
            len(
                description_context_overlap
            ),
            5,
        )
        / 5.0
    )

    description_signal = (
        0.7
        * description_overlap_ratio
        + 0.3
        * description_overlap_count_score
    )

    # -------------------------------------------------------------------------
    # 최종 diagnostic context score
    # -------------------------------------------------------------------------
    raw_score = (
        EXACT_LABEL_WEIGHT
        * exact_label_match
        +
        SEARCH_MATCH_WEIGHT
        * search_match_exact
        +
        LABEL_CONTEXT_WEIGHT
        * label_context_ratio
        +
        DESCRIPTION_OVERLAP_WEIGHT
        * description_signal
    )

    # 해석하기 쉽게 0~1 범위로 압축
    max_possible = (
        EXACT_LABEL_WEIGHT
        + SEARCH_MATCH_WEIGHT
        + LABEL_CONTEXT_WEIGHT
        + DESCRIPTION_OVERLAP_WEIGHT
    )

    context_score = (
        raw_score
        / max_possible
    )

    return {
        "context_score": float(
            context_score
        ),

        "exact_label_match": int(
            exact_label_match
        ),

        "search_match_exact": int(
            search_match_exact
        ),

        "label_context_overlap_count": int(
            len(
                label_context_overlap
            )
        ),

        "label_context_overlap_tokens": sorted(
            label_context_overlap
        ),

        "label_context_ratio": float(
            label_context_ratio
        ),

        "description_context_overlap_count": int(
            len(
                description_context_overlap
            )
        ),

        "description_context_overlap_tokens": sorted(
            description_context_overlap
        ),

        "description_context_ratio": float(
            description_overlap_ratio
        ),
    }


# =============================================================================
# 15. Article × Entity × Candidate score table
# =============================================================================


def _build_article_candidate_scores(
    *,
    train_entities: pl.DataFrame,
    article_text_lookup: dict[
        int,
        dict[str, str]
    ],
    article_entity_context: dict[
        int,
        list[
            dict[str, str]
        ],
    ],
    candidate_lookup: dict[
        str,
        list[
            dict[str, Any]
        ],
    ],
) -> pl.DataFrame:
    """
    핵심 출력.

    같은 Entity라도 article_id가 다르면 별도 context score를 만든다.

    이유:
        동일한 surface가 기사마다 다른 실제 Entity를 가리킬 가능성을
        억지로 하나의 global QID로 고정하지 않기 위해서다.
    """

    rows: list[
        dict[str, Any]
    ] = []

    # candidate가 실제로 있는 Entity만 처리
    relevant_keys = set(
        candidate_lookup
    )

    target_mentions = (
        train_entities.filter(
            pl.col(
                "canonical_entity_key"
            ).is_in(
                sorted(
                    relevant_keys
                )
            )
        )
    )

    for mention in target_mentions.iter_rows(
        named=True
    ):
        article_id = int(
            mention[
                "article_id"
            ]
        )

        group = str(
            mention[
                "entity_group"
            ]
        )

        surface = str(
            mention[
                "canonical_entity"
            ]
        )

        key = str(
            mention[
                "canonical_entity_key"
            ]
        )

        article_text = (
            article_text_lookup.get(
                article_id,
                {
                    "title": "",
                    "optional_text": "",
                    "optional_columns_used": "",
                },
            )
        )

        title = str(
            article_text.get(
                "title",
                "",
            )
        )

        optional_text = str(
            article_text.get(
                "optional_text",
                "",
            )
        )

        (
            co_entity_text,
            co_entity_surfaces,
        ) = _co_entity_context(
            article_id=article_id,
            target_key=key,
            article_entity_context=(
                article_entity_context
            ),
        )

        candidates = (
            candidate_lookup.get(
                key,
                [],
            )
        )

        for candidate in candidates:
            score_info = (
                _score_candidate(
                    surface=surface,
                    title=title,
                    optional_text=(
                        optional_text
                    ),
                    co_entity_text=(
                        co_entity_text
                    ),
                    candidate=(
                        candidate
                    ),
                )
            )

            rows.append(
                {
                    "article_id": (
                        article_id
                    ),

                    "entity_group": (
                        group
                    ),

                    "canonical_entity": (
                        surface
                    ),

                    "canonical_entity_key": (
                        key
                    ),

                    "title": (
                        title
                    ),

                    "co_entity_count": int(
                        len(
                            co_entity_surfaces
                        )
                    ),

                    "co_entity_surfaces": (
                        co_entity_surfaces
                    ),

                    "optional_text_used": bool(
                        optional_text
                    ),

                    "qid": str(
                        candidate[
                            "qid"
                        ]
                    ),

                    "candidate_label": str(
                        candidate.get(
                            "label",
                            "",
                        )
                        or ""
                    ),

                    "candidate_description": str(
                        candidate.get(
                            "description",
                            "",
                        )
                        or ""
                    ),

                    "candidate_rank": int(
                        candidate.get(
                            "candidate_rank",
                            999999,
                        )
                    ),

                    "search_language": str(
                        candidate.get(
                            "search_language",
                            "",
                        )
                        or ""
                    ),

                    "search_rank": int(
                        candidate.get(
                            "search_rank",
                            999999,
                        )
                    ),

                    "p31_qids": list(
                        candidate.get(
                            "p31_qids",
                            [],
                        )
                        or []
                    ),

                    "p31_labels": list(
                        candidate.get(
                            "p31_labels",
                            [],
                        )
                        or []
                    ),

                    **score_info,
                }
            )

    if not rows:
        return pl.DataFrame()

    df = pl.DataFrame(
        rows,
        infer_schema_length=None,
    )

    # article/entity 내부에서 score desc, 기존 candidate rank asc
    return df.sort(
        [
            "article_id",
            "canonical_entity_key",
            "context_score",
            "candidate_rank",
        ],
        descending=[
            False,
            False,
            True,
            False,
        ],
    )


# =============================================================================
# 16. Article × Entity top candidate summary
# =============================================================================


def _build_article_entity_status(
    scores_df: pl.DataFrame,
) -> pl.DataFrame:
    """
    기사 하나에서 Entity 후보들의 ranking 결과를 요약.

    상태
    ----
    SINGLE_TYPE_CANDIDATE
        TYPE_MATCH 후보가 애초에 하나.

    CONTEXT_UNIQUE_TOP
        후보가 여러 개이고 context score 1등이 유일하며
        top2와 margin이 CONTEXT_MARGIN_HINT 이상.

    CONTEXT_WEAK_TOP
        1등은 있지만 차이가 작음.

    CONTEXT_TIE
        top score가 동점.

    CONTEXT_ZERO
        모든 후보 context score가 0.

    다시 강조:
        CONTEXT_UNIQUE_TOP도 최종 LINKED 확정이 아니다.
    """

    if not scores_df.height:
        return pl.DataFrame()

    grouped: dict[
        tuple[int, str],
        list[
            dict[str, Any]
        ],
    ] = defaultdict(
        list
    )

    for row in scores_df.iter_rows(
        named=True
    ):
        grouped[
            (
                int(
                    row[
                        "article_id"
                    ]
                ),
                str(
                    row[
                        "canonical_entity_key"
                    ]
                ),
            )
        ].append(
            row
        )

    output_rows: list[
        dict[str, Any]
    ] = []

    for (
        article_id,
        key,
    ), candidate_rows in grouped.items():

        candidate_rows = sorted(
            candidate_rows,
            key=lambda row: (
                -float(
                    row[
                        "context_score"
                    ]
                ),
                int(
                    row[
                        "candidate_rank"
                    ]
                ),
                str(
                    row[
                        "qid"
                    ]
                ),
            ),
        )

        candidate_count = len(
            candidate_rows
        )

        top = candidate_rows[
            0
        ]

        top_score = float(
            top[
                "context_score"
            ]
        )

        if candidate_count >= 2:
            second_score = float(
                candidate_rows[
                    1
                ][
                    "context_score"
                ]
            )
        else:
            second_score = None

        if second_score is None:
            margin = None
        else:
            margin = (
                top_score
                - second_score
            )

        top_tie_count = sum(
            1
            for row in candidate_rows
            if math.isclose(
                float(
                    row[
                        "context_score"
                    ]
                ),
                top_score,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        )

        if candidate_count == 1:
            diagnostic_status = (
                "SINGLE_TYPE_CANDIDATE"
            )

        elif top_score == 0:
            diagnostic_status = (
                "CONTEXT_ZERO"
            )

        elif top_tie_count > 1:
            diagnostic_status = (
                "CONTEXT_TIE"
            )

        elif (
            margin is not None
            and margin
            >= CONTEXT_MARGIN_HINT
        ):
            diagnostic_status = (
                "CONTEXT_UNIQUE_TOP"
            )

        else:
            diagnostic_status = (
                "CONTEXT_WEAK_TOP"
            )

        output_rows.append(
            {
                "article_id": (
                    article_id
                ),

                "entity_group": str(
                    top[
                        "entity_group"
                    ]
                ),

                "canonical_entity": str(
                    top[
                        "canonical_entity"
                    ]
                ),

                "canonical_entity_key": (
                    key
                ),

                "title": str(
                    top[
                        "title"
                    ]
                ),

                "type_match_candidate_count": int(
                    candidate_count
                ),

                "top_qid": str(
                    top[
                        "qid"
                    ]
                ),

                "top_label": str(
                    top[
                        "candidate_label"
                    ]
                ),

                "top_description": str(
                    top[
                        "candidate_description"
                    ]
                ),

                "top_context_score": float(
                    top_score
                ),

                "second_context_score": (
                    float(
                        second_score
                    )
                    if second_score
                    is not None
                    else None
                ),

                "top_margin": (
                    float(
                        margin
                    )
                    if margin
                    is not None
                    else None
                ),

                "top_tie_count": int(
                    top_tie_count
                ),

                "diagnostic_status": (
                    diagnostic_status
                ),

                "co_entity_surfaces": list(
                    top[
                        "co_entity_surfaces"
                    ]
                ),
            }
        )

    return (
        pl.DataFrame(
            output_rows,
            infer_schema_length=None,
        )
        .sort(
            [
                "article_id",
                "canonical_entity_key",
            ]
        )
    )


# =============================================================================
# 17. Entity 전체 기사에서 top QID 일관성 요약
# =============================================================================


def _build_entity_context_summary(
    article_status_df: pl.DataFrame,
) -> pl.DataFrame:
    """
    같은 canonical surface가 여러 기사에서 어떤 QID를 top으로 선택했는지 요약.

    예:
        PER::putin

        article 1 -> Q7747
        article 2 -> Q7747
        article 3 -> Q7747
        article 4 -> Qxxxx

    이 정보를 보면:

    - 대부분 같은 QID를 가리키는 안정적인 surface인지
    - 기사마다 다른 QID가 top으로 나오는 ambiguous surface인지

    확인할 수 있다.
    """

    if not article_status_df.height:
        return pl.DataFrame()

    grouped: dict[
        str,
        list[
            dict[str, Any]
        ],
    ] = defaultdict(
        list
    )

    for row in article_status_df.iter_rows(
        named=True
    ):
        key = str(
            row[
                "canonical_entity_key"
            ]
        )

        grouped[
            key
        ].append(
            row
        )

    rows = []

    for key, article_rows in grouped.items():
        first = article_rows[
            0
        ]

        top_qid_counter = Counter(
            str(
                row[
                    "top_qid"
                ]
            )
            for row in article_rows
        )

        (
            dominant_qid,
            dominant_count,
        ) = top_qid_counter.most_common(
            1
        )[0]

        article_count = len(
            article_rows
        )

        dominant_ratio = (
            dominant_count
            / article_count
        )

        unique_top_qid_count = len(
            top_qid_counter
        )

        status_counter = Counter(
            str(
                row[
                    "diagnostic_status"
                ]
            )
            for row in article_rows
        )

        rows.append(
            {
                "entity_group": str(
                    first[
                        "entity_group"
                    ]
                ),

                "canonical_entity": str(
                    first[
                        "canonical_entity"
                    ]
                ),

                "canonical_entity_key": (
                    key
                ),

                "article_count": int(
                    article_count
                ),

                "unique_top_qid_count": int(
                    unique_top_qid_count
                ),

                "dominant_top_qid": (
                    dominant_qid
                ),

                "dominant_top_qid_article_count": int(
                    dominant_count
                ),

                "dominant_top_qid_ratio": float(
                    dominant_ratio
                ),

                "single_type_candidate_article_count": int(
                    status_counter.get(
                        "SINGLE_TYPE_CANDIDATE",
                        0,
                    )
                ),

                "context_unique_top_article_count": int(
                    status_counter.get(
                        "CONTEXT_UNIQUE_TOP",
                        0,
                    )
                ),

                "context_weak_top_article_count": int(
                    status_counter.get(
                        "CONTEXT_WEAK_TOP",
                        0,
                    )
                ),

                "context_tie_article_count": int(
                    status_counter.get(
                        "CONTEXT_TIE",
                        0,
                    )
                ),

                "context_zero_article_count": int(
                    status_counter.get(
                        "CONTEXT_ZERO",
                        0,
                    )
                ),
            }
        )

    return (
        pl.DataFrame(
            rows,
            infer_schema_length=None,
        )
        .sort(
            [
                "article_count",
                "canonical_entity_key",
            ],
            descending=[
                True,
                False,
            ],
        )
    )


# =============================================================================
# 18. 리뷰용 example
# =============================================================================


def _build_review_examples(
    article_status_df: pl.DataFrame,
    *,
    max_rows: int,
) -> pl.DataFrame:
    """
    사람이 검토하기 좋은 순서:

    1. MULTIPLE 후보 중 CONTEXT_UNIQUE_TOP
    2. CONTEXT_WEAK_TOP
    3. CONTEXT_TIE
    4. CONTEXT_ZERO

    SINGLE_TYPE_CANDIDATE는 이미 TYPE 단계에서 후보가 하나라
    context disambiguation 리뷰 우선순위가 낮다.
    """

    if not article_status_df.height:
        return pl.DataFrame()

    priority_map = {
        "CONTEXT_UNIQUE_TOP": 1,
        "CONTEXT_WEAK_TOP": 2,
        "CONTEXT_TIE": 3,
        "CONTEXT_ZERO": 4,
        "SINGLE_TYPE_CANDIDATE": 5,
    }

    rows = []

    for row in article_status_df.iter_rows(
        named=True
    ):
        status = str(
            row[
                "diagnostic_status"
            ]
        )

        rows.append(
            {
                **row,
                "review_priority": int(
                    priority_map.get(
                        status,
                        99,
                    )
                ),
            }
        )

    return (
        pl.DataFrame(
            rows,
            infer_schema_length=None,
        )
        .sort(
            [
                "review_priority",
                "top_margin",
                "top_context_score",
            ],
            descending=[
                False,
                True,
                True,
            ],
            nulls_last=True,
        )
        .head(
            max_rows
        )
    )


# =============================================================================
# 19. Summary
# =============================================================================


def _write_summary(
    *,
    output_path: Path,
    scores_df: pl.DataFrame,
    article_status_df: pl.DataFrame,
    entity_summary_df: pl.DataFrame,
    matched_candidates_df: pl.DataFrame,
    entity_filter_status_df: pl.DataFrame,
) -> None:

    def article_status_count(
        status: str,
    ) -> int:
        if not article_status_df.height:
            return 0

        return int(
            article_status_df.filter(
                pl.col(
                    "diagnostic_status"
                )
                == status
            ).height
        )

    lines = [
        "=" * 90,
        "Wikidata Context Disambiguation Inspection",
        "=" * 90,
        "",
        "주의:",
        "- 아직 최종 QID를 확정하지 않는다.",
        "- context score는 diagnostic lexical score다.",
        "- 이 결과를 보고 GPT가 정말 필요한 범위를 결정한다.",
        "",
        f"type_matched_candidate_row_count={matched_candidates_df.height}",
        f"article_candidate_score_row_count={scores_df.height}",
        f"article_entity_context_case_count={article_status_df.height}",
        f"context_target_unique_entity_count={entity_summary_df.height}",
        "",
        "Article × Entity status:",
        f"single_type_candidate_count={article_status_count('SINGLE_TYPE_CANDIDATE')}",
        f"context_unique_top_count={article_status_count('CONTEXT_UNIQUE_TOP')}",
        f"context_weak_top_count={article_status_count('CONTEXT_WEAK_TOP')}",
        f"context_tie_count={article_status_count('CONTEXT_TIE')}",
        f"context_zero_count={article_status_count('CONTEXT_ZERO')}",
        "",
        "현재 TYPE-filter 100 Entity probe 참고:",
        f"source_one_type_match_entity_count="
        f"{entity_filter_status_df.filter(pl.col('diagnostic_status') == 'ONE_TYPE_MATCH').height}",
        f"source_multiple_type_match_entity_count="
        f"{entity_filter_status_df.filter(pl.col('diagnostic_status') == 'MULTIPLE_TYPE_MATCH').height}",
        "",
        "점수 신호:",
        "- exact_label_match",
        "- search_match_exact",
        "- candidate label token ↔ article context token overlap",
        "- candidate description token ↔ article context token overlap",
        "",
        "다음 판단:",
        "- CONTEXT_UNIQUE_TOP이 충분히 많고 수동 검토 품질이 좋으면 deterministic evidence로 사용 가능.",
        "- CONTEXT_WEAK_TOP/TIE/ZERO는 GPT disambiguation 우선 대상.",
        "- SAME surface가 여러 article에서 서로 다른 top_qid를 보이면 article-level linking을 유지해야 한다.",
    ]

    output_path.write_text(
        "\n".join(
            lines
        )
        + "\n",
        encoding="utf-8",
    )


# =============================================================================
# 20. 전체 실행
# =============================================================================


def inspect_wikidata_context_disambiguation(
    *,
    article_entities_path: Path,
    articles_base_path: Path,
    matched_candidates_path: Path,
    entity_filter_status_path: Path,
    output_dir: Path,
    review_examples: int,
) -> dict[str, Any]:

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    scores_path = (
        output_dir
        / ARTICLE_CANDIDATE_SCORES_FILENAME
    )

    article_status_path = (
        output_dir
        / ARTICLE_ENTITY_STATUS_FILENAME
    )

    entity_summary_path = (
        output_dir
        / ENTITY_SUMMARY_FILENAME
    )

    review_path = (
        output_dir
        / REVIEW_EXAMPLES_FILENAME
    )

    summary_path = (
        output_dir
        / SUMMARY_FILENAME
    )

    # -------------------------------------------------------------------------
    # STEP 1. 입력 로드
    # -------------------------------------------------------------------------
    print(
        "=" * 100
    )
    print(
        "Wikidata Context Disambiguation Inspection 시작"
    )
    print(
        "=" * 100
    )

    train_entities = (
        _load_train_article_entities(
            article_entities_path
        )
    )

    article_text_lookup = (
        _load_article_text_lookup(
            articles_base_path
        )
    )

    matched_candidates_df = (
        _load_matched_candidates(
            matched_candidates_path
        )
    )

    entity_filter_status_df = (
        _load_entity_filter_status(
            entity_filter_status_path
        )
    )

    print(
        f"train_entity_article_pair_count = {train_entities.height}"
    )
    print(
        f"type_matched_candidate_row_count = {matched_candidates_df.height}"
    )
    print(
        f"type_filter_entity_count = {entity_filter_status_df.height}"
    )
    print()

    # -------------------------------------------------------------------------
    # STEP 2. lookup 구성
    # -------------------------------------------------------------------------
    print(
        "[STEP 1] Article context 구성"
    )

    article_entity_context = (
        _build_article_entity_context(
            train_entities
        )
    )

    candidate_lookup = (
        _build_candidate_lookup(
            matched_candidates_df
        )
    )

    print(
        f"context_target_unique_entity_count = {len(candidate_lookup)}"
    )
    print()

    # -------------------------------------------------------------------------
    # STEP 3. Article × Candidate score
    # -------------------------------------------------------------------------
    print(
        "[STEP 2] Article × Entity × Candidate context score 계산"
    )

    scores_df = (
        _build_article_candidate_scores(
            train_entities=(
                train_entities
            ),
            article_text_lookup=(
                article_text_lookup
            ),
            article_entity_context=(
                article_entity_context
            ),
            candidate_lookup=(
                candidate_lookup
            ),
        )
    )

    print(
        f"article_candidate_score_row_count = {scores_df.height}"
    )
    print()

    # -------------------------------------------------------------------------
    # STEP 4. Article × Entity top ranking
    # -------------------------------------------------------------------------
    print(
        "[STEP 3] Article × Entity top candidate 요약"
    )

    article_status_df = (
        _build_article_entity_status(
            scores_df
        )
    )

    print(
        f"article_entity_context_case_count = {article_status_df.height}"
    )
    print()

    # -------------------------------------------------------------------------
    # STEP 5. Entity-level consistency
    # -------------------------------------------------------------------------
    print(
        "[STEP 4] Entity별 article 간 top QID 일관성 요약"
    )

    entity_summary_df = (
        _build_entity_context_summary(
            article_status_df
        )
    )

    # -------------------------------------------------------------------------
    # STEP 6. Review examples
    # -------------------------------------------------------------------------
    review_df = (
        _build_review_examples(
            article_status_df,
            max_rows=(
                review_examples
            ),
        )
    )

    # -------------------------------------------------------------------------
    # STEP 7. 저장
    # -------------------------------------------------------------------------
    scores_df.write_parquet(
        scores_path,
        compression="zstd",
    )

    article_status_df.write_parquet(
        article_status_path,
        compression="zstd",
    )

    entity_summary_df.write_parquet(
        entity_summary_path,
        compression="zstd",
    )

    review_df.write_parquet(
        review_path,
        compression="zstd",
    )

    _write_summary(
        output_path=(
            summary_path
        ),
        scores_df=(
            scores_df
        ),
        article_status_df=(
            article_status_df
        ),
        entity_summary_df=(
            entity_summary_df
        ),
        matched_candidates_df=(
            matched_candidates_df
        ),
        entity_filter_status_df=(
            entity_filter_status_df
        ),
    )

    # -------------------------------------------------------------------------
    # STEP 8. 결과 숫자
    # -------------------------------------------------------------------------
    def article_count(
        status: str,
    ) -> int:
        if not article_status_df.height:
            return 0

        return int(
            article_status_df.filter(
                pl.col(
                    "diagnostic_status"
                )
                == status
            ).height
        )

    return {
        "status": "SUCCESS",

        "type_matched_candidate_row_count": int(
            matched_candidates_df.height
        ),

        "context_target_unique_entity_count": int(
            len(
                candidate_lookup
            )
        ),

        "article_candidate_score_row_count": int(
            scores_df.height
        ),

        "article_entity_context_case_count": int(
            article_status_df.height
        ),

        "single_type_candidate_count": (
            article_count(
                "SINGLE_TYPE_CANDIDATE"
            )
        ),

        "context_unique_top_count": (
            article_count(
                "CONTEXT_UNIQUE_TOP"
            )
        ),

        "context_weak_top_count": (
            article_count(
                "CONTEXT_WEAK_TOP"
            )
        ),

        "context_tie_count": (
            article_count(
                "CONTEXT_TIE"
            )
        ),

        "context_zero_count": (
            article_count(
                "CONTEXT_ZERO"
            )
        ),

        "scores_path": str(
            scores_path
        ),

        "article_status_path": str(
            article_status_path
        ),

        "entity_summary_path": str(
            entity_summary_path
        ),

        "review_examples_path": str(
            review_path
        ),

        "summary_path": str(
            summary_path
        ),
    }


# =============================================================================
# 21. CLI
# =============================================================================


def _parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Wikidata TYPE_MATCH 후보를 article context로 "
            "순위화하는 diagnostic inspection입니다."
        )
    )

    parser.add_argument(
        "--article-entities",
        type=Path,
        default=(
            DEFAULT_ARTICLE_ENTITIES_PATH
        ),
    )

    parser.add_argument(
        "--articles-base",
        type=Path,
        default=(
            DEFAULT_ARTICLES_BASE_PATH
        ),
    )

    parser.add_argument(
        "--matched-candidates",
        type=Path,
        default=(
            DEFAULT_MATCHED_CANDIDATES_PATH
        ),
    )

    parser.add_argument(
        "--entity-filter-status",
        type=Path,
        default=(
            DEFAULT_ENTITY_FILTER_STATUS_PATH
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            DEFAULT_OUTPUT_DIR
        ),
    )

    parser.add_argument(
        "--review-examples",
        type=int,
        default=(
            DEFAULT_REVIEW_EXAMPLES
        ),
    )

    args = (
        parser.parse_args()
    )

    if args.review_examples <= 0:
        parser.error(
            "--review-examples는 1 이상이어야 합니다."
        )

    return args


def main() -> None:

    args = (
        _parse_args()
    )

    result = (
        inspect_wikidata_context_disambiguation(
            article_entities_path=(
                args.article_entities
            ),
            articles_base_path=(
                args.articles_base
            ),
            matched_candidates_path=(
                args.matched_candidates
            ),
            entity_filter_status_path=(
                args.entity_filter_status
            ),
            output_dir=(
                args.output_dir
            ),
            review_examples=(
                args.review_examples
            ),
        )
    )

    print()
    print(
        "=" * 100
    )
    print(
        "Wikidata Context Disambiguation Inspection 완료"
    )
    print(
        "=" * 100
    )

    for key, value in (
        result.items()
    ):
        print(
            f"{key}: {value}"
        )


if __name__ == "__main__":
    main()