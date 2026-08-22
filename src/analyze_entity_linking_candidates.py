from __future__ import annotations

"""
Entity Linking Candidate Analysis
=================================

목적
----
inspect_entity_linking_candidates.py가 만든 전체 후보를 그대로 자동 linking하지 않고,
"어떤 후보부터 사람이 검토해야 하는가?"를 체계적으로 좁힌다.

현재 전체 후보 예
----------------
PER full name <-> surname
ORG full organization <-> acronym

이 분석 파일은 다음을 수행한다.

1. 후보별 ambiguity를 구분한다.
   - short 표현이 long 후보 하나에만 연결되는가?
   - 여러 long 후보와 연결되는가?

2. 후보별 직접 evidence를 구분한다.
   - 같은 기사에서 full/short가 함께 등장했는가?
   - 같은 Event에서 full/short가 등장했는가?
   - 주변 context entity가 겹치는가?

3. 임의의 숫자 threshold를 먼저 정하지 않고
   실제 데이터 분포를 저장한다.
   - article DF
   - same-article cooccurrence
   - shared Event count
   - context Jaccard
   - 기존 review priority score

4. 사람이 먼저 검토할 후보를 Tier로 나눈다.

중요
----
이 파일 역시 실제 Entity Linking을 수행하지 않는다.

즉:

    2,967 candidate
        ↓
    evidence / ambiguity 분석
        ↓
    우선 검토 후보 축소
        ↓
    사람이 SAFE / AMBIGUOUS / REJECT 판정
        ↓
    그 다음에만 실제 normalize_and_link 구현

순서다.


Evidence Tier
-------------
Tier A1
    short가 한 long 후보에만 연결
    +
    같은 기사에서 full/short가 같이 등장

    예:
        PER::vladimir putin
        PER::putin
        같은 기사에서 둘 다 NER됨

    현재 조사 단계에서 가장 먼저 볼 후보.


Tier A2
    short가 한 long 후보에만 연결
    +
    같은 기사 직접 동시등장은 없지만
    같은 Event에서 둘 다 등장

    A1 다음으로 검토.


Tier B
    short가 한 long 후보에만 연결
    +
    같은 기사 / 같은 Event evidence는 없음
    +
    주변 context entity가 일부 겹침

    가능성은 있지만 직접 evidence가 약하므로 후순위.


Tier C1
    short가 여러 long 후보에 연결됨
    +
    같은 기사 동시등장 evidence가 있음

    evidence는 있지만 short 자체가 ambiguous하므로 자동 linking 금지.


Tier C2
    short가 여러 long 후보에 연결됨
    +
    같은 Event evidence가 있음

    역시 수동검토 대상.


Tier D
    위 evidence가 거의 없음.

    지금 단계에서는 linking 대상으로 잡지 않는다.


실행
----
프로젝트 루트:

    python -m src.analyze_entity_linking_candidates


기본 입력
---------
data/output/experiments/entity_linking_inspection/
    entity_linking_candidates.parquet


기본 출력
---------
data/output/experiments/entity_linking_analysis/

    entity_linking_analysis_summary.txt
    candidate_evidence_tiers.parquet
    priority_review_candidates.parquet
    per_priority_review_candidates.parquet
    org_priority_review_candidates.parquet
    ambiguous_evidence_candidates.parquet
    evidence_distribution.parquet
    tier_statistics.parquet
"""

import argparse
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import polars as pl


# =============================================================================
# 1. 기본 경로
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "experiments"
    / "entity_linking_inspection"
)

DEFAULT_CANDIDATES_PATH = (
    DEFAULT_INPUT_DIR
    / "entity_linking_candidates.parquet"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "experiments"
    / "entity_linking_analysis"
)


# =============================================================================
# 2. 입력 검증
# =============================================================================


REQUIRED_COLUMNS = [
    "candidate_id",
    "candidate_type",
    "entity_group",
    "long_entity",
    "short_entity",
    "long_entity_key",
    "short_entity_key",
    "long_article_df",
    "short_article_df",
    "same_article_cooccurrence_count",
    "long_event_count",
    "short_event_count",
    "shared_event_count",
    "short_candidate_long_count",
    "competing_long_entities",
    "shared_context_entity_count",
    "context_entity_jaccard",
    "shared_context_entities",
    "long_title_examples",
    "short_title_examples",
    "same_article_title_examples",
    "review_priority_score",
]


def _require_file(
    path: Path,
    description: str,
) -> None:
    """필수 입력 파일 존재 여부 확인."""

    if not path.exists():
        raise FileNotFoundError(
            f"{description} 파일이 없습니다. 경로={path}"
        )


def _require_columns(
    df: pl.DataFrame,
    required_columns: Iterable[str],
    description: str,
) -> None:
    """입력 parquet schema 확인."""

    missing = (
        set(required_columns)
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"{description}에 필요한 컬럼이 없습니다: "
            + ", ".join(
                sorted(missing)
            )
        )


# =============================================================================
# 3. Quantile / 분포 계산
# =============================================================================


def _quantile(
    values: list[float],
    q: float,
) -> float:
    """
    외부 통계 패키지에 의존하지 않고 간단한 linear-interpolation quantile 계산.

    q:
        0.0 ~ 1.0

    예:
        q=0.50 -> median
        q=0.90 -> 90 percentile
    """

    if not values:
        return 0.0

    ordered = sorted(
        float(value)
        for value in values
    )

    if len(ordered) == 1:
        return ordered[0]

    position = (
        (len(ordered) - 1)
        * q
    )

    left_index = int(
        position
    )

    right_index = min(
        left_index + 1,
        len(ordered) - 1,
    )

    fraction = (
        position
        - left_index
    )

    left_value = (
        ordered[
            left_index
        ]
    )

    right_value = (
        ordered[
            right_index
        ]
    )

    return float(
        left_value
        + (
            right_value
            - left_value
        )
        * fraction
    )


def _distribution_row(
    *,
    candidate_type: str,
    metric_name: str,
    values: list[float],
) -> dict[str, Any]:
    """
    하나의 metric 분포를 한 행으로 정리한다.

    threshold를 먼저 정하지 않고
    실제 값의 분포를 보기 위한 표이다.
    """

    if not values:
        return {
            "candidate_type": candidate_type,
            "metric_name": metric_name,
            "count": 0,
            "min": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0,
            "mean": 0.0,
        }

    float_values = [
        float(value)
        for value in values
    ]

    return {
        "candidate_type": candidate_type,
        "metric_name": metric_name,
        "count": int(
            len(float_values)
        ),
        "min": float(
            min(float_values)
        ),
        "p25": _quantile(
            float_values,
            0.25,
        ),
        "p50": _quantile(
            float_values,
            0.50,
        ),
        "p75": _quantile(
            float_values,
            0.75,
        ),
        "p90": _quantile(
            float_values,
            0.90,
        ),
        "p95": _quantile(
            float_values,
            0.95,
        ),
        "p99": _quantile(
            float_values,
            0.99,
        ),
        "max": float(
            max(float_values)
        ),
        "mean": float(
            sum(float_values)
            / len(float_values)
        ),
    }


def _build_distribution_df(
    candidates: pl.DataFrame,
) -> pl.DataFrame:
    """
    전체 / PER / ORG 각각에 대해 evidence metric 분포를 저장한다.

    여기서 보는 값:
    - long / short article DF
    - same article cooccurrence
    - shared event
    - shared context entity count
    - context Jaccard
    - review priority score
    """

    metrics = [
        "long_article_df",
        "short_article_df",
        "same_article_cooccurrence_count",
        "shared_event_count",
        "shared_context_entity_count",
        "context_entity_jaccard",
        "review_priority_score",
    ]

    groups: list[
        tuple[str, pl.DataFrame]
    ] = [
        (
            "ALL",
            candidates,
        )
    ]

    for candidate_type in [
        "PER_FULL_TO_SURNAME",
        "ORG_FULL_TO_ACRONYM",
    ]:
        groups.append(
            (
                candidate_type,
                candidates.filter(
                    pl.col(
                        "candidate_type"
                    )
                    == candidate_type
                ),
            )
        )

    rows: list[
        dict[str, Any]
    ] = []

    for (
        candidate_type,
        group_df,
    ) in groups:
        for metric_name in metrics:
            values = [
                float(value)
                for value
                in group_df
                .get_column(
                    metric_name
                )
                .to_list()
                if value is not None
            ]

            rows.append(
                _distribution_row(
                    candidate_type=(
                        candidate_type
                    ),
                    metric_name=(
                        metric_name
                    ),
                    values=values,
                )
            )

    schema = {
        "candidate_type": pl.String,
        "metric_name": pl.String,
        "count": pl.Int64,
        "min": pl.Float64,
        "p25": pl.Float64,
        "p50": pl.Float64,
        "p75": pl.Float64,
        "p90": pl.Float64,
        "p95": pl.Float64,
        "p99": pl.Float64,
        "max": pl.Float64,
        "mean": pl.Float64,
    }

    return (
        pl.DataFrame(
            rows,
            schema=schema,
        )
        .sort(
            [
                "candidate_type",
                "metric_name",
            ]
        )
    )


# =============================================================================
# 4. Evidence Tier
# =============================================================================


def _assign_evidence_tier(
    row: dict[str, Any],
) -> tuple[
    str,
    str,
]:
    """
    후보를 사람이 먼저 검토할 순서대로 Tier에 배치한다.

    매우 중요
    ---------
    이것은 SAFE / REJECT 판정이 아니다.

    즉 A1이라고 자동 linking하는 것이 아니라:
        "A1부터 먼저 사람이 보자"
    라는 의미다.
    """

    short_candidate_long_count = int(
        row[
            "short_candidate_long_count"
        ]
    )

    same_article_count = int(
        row[
            "same_article_cooccurrence_count"
        ]
    )

    shared_event_count = int(
        row[
            "shared_event_count"
        ]
    )

    shared_context_count = int(
        row[
            "shared_context_entity_count"
        ]
    )

    is_unambiguous_short = (
        short_candidate_long_count
        == 1
    )

    # -------------------------------------------------------------------------
    # A1. short가 유일 + 같은 기사 직접 cooccurrence
    # -------------------------------------------------------------------------
    if (
        is_unambiguous_short
        and same_article_count > 0
    ):
        return (
            "A1_UNIQUE_SAME_ARTICLE",
            (
                "short 표현이 하나의 long 후보에만 연결되고, "
                "같은 기사에서 두 표현이 직접 함께 등장함"
            ),
        )

    # -------------------------------------------------------------------------
    # A2. short가 유일 + 같은 Event evidence
    # -------------------------------------------------------------------------
    if (
        is_unambiguous_short
        and shared_event_count > 0
    ):
        return (
            "A2_UNIQUE_SHARED_EVENT",
            (
                "short 표현이 하나의 long 후보에만 연결되고, "
                "같은 기사 직접 동시등장은 없지만 같은 Event에서 등장함"
            ),
        )

    # -------------------------------------------------------------------------
    # B. short가 유일 + context overlap만 존재
    # -------------------------------------------------------------------------
    if (
        is_unambiguous_short
        and shared_context_count > 0
    ):
        return (
            "B_UNIQUE_CONTEXT_ONLY",
            (
                "short 표현은 유일하지만 같은 기사/Event 직접 evidence 없이 "
                "주변 context entity만 겹침"
            ),
        )

    # -------------------------------------------------------------------------
    # C1. short 자체가 ambiguous하지만 같은 기사 evidence 존재
    # -------------------------------------------------------------------------
    if (
        not is_unambiguous_short
        and same_article_count > 0
    ):
        return (
            "C1_AMBIGUOUS_SAME_ARTICLE",
            (
                "같은 기사 evidence는 있으나 short 표현이 "
                "여러 long 후보에 연결되어 자동 linking 위험"
            ),
        )

    # -------------------------------------------------------------------------
    # C2. ambiguous short + shared Event evidence
    # -------------------------------------------------------------------------
    if (
        not is_unambiguous_short
        and shared_event_count > 0
    ):
        return (
            "C2_AMBIGUOUS_SHARED_EVENT",
            (
                "같은 Event evidence는 있으나 short 표현이 "
                "여러 long 후보에 연결되어 자동 linking 위험"
            ),
        )

    # -------------------------------------------------------------------------
    # C3. ambiguous short + context overlap
    # -------------------------------------------------------------------------
    if (
        not is_unambiguous_short
        and shared_context_count > 0
    ):
        return (
            "C3_AMBIGUOUS_CONTEXT_ONLY",
            (
                "short 표현이 여러 long 후보에 연결되고 "
                "context overlap만 존재"
            ),
        )

    # -------------------------------------------------------------------------
    # D. 지금 단계에서 identity evidence가 거의 없음
    # -------------------------------------------------------------------------
    return (
        "D_WEAK_EVIDENCE",
        (
            "같은 기사/Event/context evidence가 거의 없어 "
            "현재 단계의 linking 우선 검토 대상이 아님"
        ),
    )


def _tier_rank(
    tier: str,
) -> int:
    """Tier 정렬 순서."""

    order = {
        "A1_UNIQUE_SAME_ARTICLE": 1,
        "A2_UNIQUE_SHARED_EVENT": 2,
        "B_UNIQUE_CONTEXT_ONLY": 3,
        "C1_AMBIGUOUS_SAME_ARTICLE": 4,
        "C2_AMBIGUOUS_SHARED_EVENT": 5,
        "C3_AMBIGUOUS_CONTEXT_ONLY": 6,
        "D_WEAK_EVIDENCE": 7,
    }

    return int(
        order.get(
            tier,
            999,
        )
    )


def _build_tiered_candidates(
    candidates: pl.DataFrame,
) -> pl.DataFrame:
    """
    전체 후보에 분석용 컬럼을 추가한다.

    추가 컬럼
    ---------
    evidence_tier
    evidence_tier_rank
    evidence_tier_reason

    is_unique_short
    has_same_article_evidence
    has_shared_event_evidence
    has_context_overlap

    cooccurrence_long_coverage
        long entity가 나온 기사 중
        short entity와 같이 나온 기사 비율

    cooccurrence_short_coverage
        short entity가 나온 기사 중
        long entity와 같이 나온 기사 비율
    """

    rows: list[
        dict[str, Any]
    ] = []

    for row in candidates.iter_rows(
        named=True
    ):
        (
            evidence_tier,
            evidence_tier_reason,
        ) = _assign_evidence_tier(
            row
        )

        long_df = int(
            row["long_article_df"]
        )

        short_df = int(
            row["short_article_df"]
        )

        same_article_count = int(
            row[
                "same_article_cooccurrence_count"
            ]
        )

        new_row = dict(
            row
        )

        new_row.update(
            {
                "evidence_tier": (
                    evidence_tier
                ),
                "evidence_tier_rank": (
                    _tier_rank(
                        evidence_tier
                    )
                ),
                "evidence_tier_reason": (
                    evidence_tier_reason
                ),

                "is_unique_short": (
                    int(
                        row[
                            "short_candidate_long_count"
                        ]
                    )
                    == 1
                ),

                "has_same_article_evidence": (
                    same_article_count
                    > 0
                ),

                "has_shared_event_evidence": (
                    int(
                        row[
                            "shared_event_count"
                        ]
                    )
                    > 0
                ),

                "has_context_overlap": (
                    int(
                        row[
                            "shared_context_entity_count"
                        ]
                    )
                    > 0
                ),

                "cooccurrence_long_coverage": (
                    float(
                        same_article_count
                        / long_df
                    )
                    if long_df > 0
                    else 0.0
                ),

                "cooccurrence_short_coverage": (
                    float(
                        same_article_count
                        / short_df
                    )
                    if short_df > 0
                    else 0.0
                ),
            }
        )

        rows.append(
            new_row
        )

    if not rows:
        # 입력이 비어있는 경우에도 기존 schema를 유지한다.
        return candidates

    tiered_df = (
        pl.DataFrame(
            rows
        )
        .sort(
            [
                "evidence_tier_rank",
                "same_article_cooccurrence_count",
                "shared_event_count",
                "context_entity_jaccard",
                "review_priority_score",
                "candidate_type",
                "short_entity",
                "long_entity",
            ],
            descending=[
                False,
                True,
                True,
                True,
                True,
                False,
                False,
                False,
            ],
        )
    )

    return tiered_df


# =============================================================================
# 5. Tier 통계
# =============================================================================


def _build_tier_statistics(
    tiered_df: pl.DataFrame,
) -> pl.DataFrame:
    """
    PER/ORG별 각 Tier 후보 수를 센다.
    """

    if tiered_df.height == 0:
        return pl.DataFrame(
            schema={
                "candidate_type": pl.String,
                "evidence_tier": pl.String,
                "evidence_tier_rank": pl.Int64,
                "candidate_count": pl.Int64,
                "unique_short_entity_count": pl.Int64,
            }
        )

    rows: list[
        dict[str, Any]
    ] = []

    candidate_types = [
        "ALL",
        "PER_FULL_TO_SURNAME",
        "ORG_FULL_TO_ACRONYM",
    ]

    tier_names = sorted(
        set(
            str(value)
            for value in tiered_df
            .get_column(
                "evidence_tier"
            )
            .to_list()
        ),
        key=_tier_rank,
    )

    for candidate_type in (
        candidate_types
    ):
        if candidate_type == "ALL":
            group_df = tiered_df
        else:
            group_df = (
                tiered_df.filter(
                    pl.col(
                        "candidate_type"
                    )
                    == candidate_type
                )
            )

        for tier_name in tier_names:
            tier_df = (
                group_df.filter(
                    pl.col(
                        "evidence_tier"
                    )
                    == tier_name
                )
            )

            rows.append(
                {
                    "candidate_type": (
                        candidate_type
                    ),
                    "evidence_tier": (
                        tier_name
                    ),
                    "evidence_tier_rank": (
                        _tier_rank(
                            tier_name
                        )
                    ),
                    "candidate_count": int(
                        tier_df.height
                    ),
                    "unique_short_entity_count": int(
                        tier_df
                        .select(
                            "short_entity_key"
                        )
                        .unique()
                        .height
                        if tier_df.height
                        else 0
                    ),
                }
            )

    return (
        pl.DataFrame(
            rows,
            schema={
                "candidate_type": pl.String,
                "evidence_tier": pl.String,
                "evidence_tier_rank": pl.Int64,
                "candidate_count": pl.Int64,
                "unique_short_entity_count": pl.Int64,
            },
        )
        .sort(
            [
                "candidate_type",
                "evidence_tier_rank",
            ]
        )
    )


# =============================================================================
# 6. 우선 검토 subset
# =============================================================================


def _build_priority_review_df(
    tiered_df: pl.DataFrame,
) -> pl.DataFrame:
    """
    실제로 가장 먼저 사람이 볼 후보만 추린다.

    우선 검토:
        A1
        A2

    이 기준에는 임의의 context Jaccard threshold를 사용하지 않는다.

    이유:
    -----
    A1/A2는
        unique short
        +
        same article 또는 shared event

    라는 해석 가능한 evidence가 있기 때문이다.

    이 파일을 먼저 검토한 뒤
    SAFE / AMBIGUOUS / REJECT 비율을 보고
    실제 Linking guardrail을 결정한다.
    """

    if tiered_df.height == 0:
        return tiered_df

    return (
        tiered_df.filter(
            pl.col(
                "evidence_tier"
            ).is_in(
                [
                    "A1_UNIQUE_SAME_ARTICLE",
                    "A2_UNIQUE_SHARED_EVENT",
                ]
            )
        )
        .sort(
            [
                "evidence_tier_rank",
                "same_article_cooccurrence_count",
                "shared_event_count",
                "context_entity_jaccard",
                "review_priority_score",
            ],
            descending=[
                False,
                True,
                True,
                True,
                True,
            ],
        )
    )


def _build_ambiguous_evidence_df(
    tiered_df: pl.DataFrame,
) -> pl.DataFrame:
    """
    short가 ambiguous하지만 evidence는 있는 후보를 별도로 모은다.

    현재는 자동 linking 대상이 아니며
    향후 context-aware linker를 고민할 때 참고할 진단 자료다.
    """

    if tiered_df.height == 0:
        return tiered_df

    return (
        tiered_df.filter(
            pl.col(
                "evidence_tier"
            ).is_in(
                [
                    "C1_AMBIGUOUS_SAME_ARTICLE",
                    "C2_AMBIGUOUS_SHARED_EVENT",
                    "C3_AMBIGUOUS_CONTEXT_ONLY",
                ]
            )
        )
        .sort(
            [
                "evidence_tier_rank",
                "same_article_cooccurrence_count",
                "shared_event_count",
                "context_entity_jaccard",
            ],
            descending=[
                False,
                True,
                True,
                True,
            ],
        )
    )


# =============================================================================
# 7. Summary text
# =============================================================================


def _count_tier(
    df: pl.DataFrame,
    tier_name: str,
) -> int:
    """특정 Tier 후보 수."""

    if df.height == 0:
        return 0

    return int(
        df.filter(
            pl.col(
                "evidence_tier"
            )
            == tier_name
        ).height
    )


def _write_summary(
    *,
    output_path: Path,
    source_path: Path,
    tiered_df: pl.DataFrame,
    priority_df: pl.DataFrame,
    per_priority_df: pl.DataFrame,
    org_priority_df: pl.DataFrame,
    ambiguous_df: pl.DataFrame,
    distribution_df: pl.DataFrame,
) -> None:
    """
    분석의 핵심 수치를 사람이 빠르게 읽을 수 있는 txt로 저장한다.
    """

    lines: list[str] = []

    lines.extend(
        [
            "=" * 80,
            "Entity Linking Candidate Analysis",
            "=" * 80,
            "",
            f"source_candidates={source_path}",
            f"total_candidate_count={tiered_df.height}",
            "",
            "-" * 80,
            "Evidence Tier Counts",
            "-" * 80,
        ]
    )

    tier_order = [
        "A1_UNIQUE_SAME_ARTICLE",
        "A2_UNIQUE_SHARED_EVENT",
        "B_UNIQUE_CONTEXT_ONLY",
        "C1_AMBIGUOUS_SAME_ARTICLE",
        "C2_AMBIGUOUS_SHARED_EVENT",
        "C3_AMBIGUOUS_CONTEXT_ONLY",
        "D_WEAK_EVIDENCE",
    ]

    for tier_name in tier_order:
        lines.append(
            (
                f"{tier_name}="
                f"{_count_tier(tiered_df, tier_name)}"
            )
        )

    lines.extend(
        [
            "",
            "-" * 80,
            "Priority Review",
            "-" * 80,
            (
                "priority_definition="
                "A1_UNIQUE_SAME_ARTICLE + A2_UNIQUE_SHARED_EVENT"
            ),
            f"priority_review_candidate_count={priority_df.height}",
            f"per_priority_candidate_count={per_priority_df.height}",
            f"org_priority_candidate_count={org_priority_df.height}",
            f"ambiguous_evidence_candidate_count={ambiguous_df.height}",
            "",
            "-" * 80,
            "Interpretation",
            "-" * 80,
            (
                "A1/A2는 자동 SAFE 판정이 아니다. "
                "사람이 가장 먼저 볼 후보 집합이다."
            ),
            (
                "A1은 unique short + same-article evidence로 "
                "현재 가장 강한 검토 우선순위다."
            ),
            (
                "A2는 unique short + shared-event evidence로 "
                "A1 다음 검토 대상이다."
            ),
            (
                "C 계열은 evidence가 있어도 short가 여러 long 후보와 연결되어 "
                "현재 자동 linking 대상으로 사용하지 않는다."
            ),
            (
                "context Jaccard / DF threshold는 아직 임의로 고정하지 않았다. "
                "evidence_distribution.parquet의 실제 분포를 보고 결정한다."
            ),
            "",
            "-" * 80,
            "Next Step",
            "-" * 80,
            (
                "1. priority_review_candidates.parquet을 검토하여 "
                "SAFE / AMBIGUOUS / REJECT를 판정한다."
            ),
            (
                "2. PER와 ORG를 따로 평가한다."
            ),
            (
                "3. SAFE 패턴이 확인되면 그때 normalize_and_link용 "
                "Train-fit mapping guardrail을 구현한다."
            ),
        ]
    )

    # 전체 context jaccard 분포 중 p50/p90/p95를 summary에도 짧게 기록
    all_context_rows = (
        distribution_df.filter(
            (
                pl.col(
                    "candidate_type"
                )
                == "ALL"
            )
            &
            (
                pl.col(
                    "metric_name"
                )
                == "context_entity_jaccard"
            )
        )
    )

    if all_context_rows.height == 1:
        row = (
            all_context_rows
            .row(
                0,
                named=True,
            )
        )

        lines.extend(
            [
                "",
                "-" * 80,
                "Context Jaccard Distribution (ALL)",
                "-" * 80,
                f"p50={row['p50']:.6f}",
                f"p90={row['p90']:.6f}",
                f"p95={row['p95']:.6f}",
                f"p99={row['p99']:.6f}",
                f"max={row['max']:.6f}",
            ]
        )

    output_path.write_text(
        "\n".join(
            lines
        )
        + "\n",
        encoding="utf-8",
    )


# =============================================================================
# 8. Main analysis
# =============================================================================


def analyze_entity_linking_candidates(
    *,
    candidates_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """
    전체 Entity Linking 후보 분석 실행.
    """

    # -------------------------------------------------------------------------
    # STEP 8-1. 입력 확인
    # -------------------------------------------------------------------------
    _require_file(
        candidates_path,
        "Entity Linking candidates",
    )

    candidates = (
        pl.read_parquet(
            candidates_path
        )
    )

    _require_columns(
        candidates,
        REQUIRED_COLUMNS,
        "entity_linking_candidates.parquet",
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # STEP 8-2. 실제 metric 분포 계산
    # -------------------------------------------------------------------------
    distribution_df = (
        _build_distribution_df(
            candidates
        )
    )

    # -------------------------------------------------------------------------
    # STEP 8-3. Evidence Tier 부여
    # -------------------------------------------------------------------------
    tiered_df = (
        _build_tiered_candidates(
            candidates
        )
    )

    tier_statistics_df = (
        _build_tier_statistics(
            tiered_df
        )
    )

    # -------------------------------------------------------------------------
    # STEP 8-4. 우선 검토 집합
    # -------------------------------------------------------------------------
    priority_df = (
        _build_priority_review_df(
            tiered_df
        )
    )

    if priority_df.height:
        per_priority_df = (
            priority_df.filter(
                pl.col(
                    "candidate_type"
                )
                == "PER_FULL_TO_SURNAME"
            )
        )

        org_priority_df = (
            priority_df.filter(
                pl.col(
                    "candidate_type"
                )
                == "ORG_FULL_TO_ACRONYM"
            )
        )
    else:
        per_priority_df = (
            priority_df
        )
        org_priority_df = (
            priority_df
        )

    ambiguous_df = (
        _build_ambiguous_evidence_df(
            tiered_df
        )
    )

    # -------------------------------------------------------------------------
    # STEP 8-5. 저장 경로
    # -------------------------------------------------------------------------
    tiered_path = (
        output_dir
        / "candidate_evidence_tiers.parquet"
    )

    priority_path = (
        output_dir
        / "priority_review_candidates.parquet"
    )

    per_priority_path = (
        output_dir
        / "per_priority_review_candidates.parquet"
    )

    org_priority_path = (
        output_dir
        / "org_priority_review_candidates.parquet"
    )

    ambiguous_path = (
        output_dir
        / "ambiguous_evidence_candidates.parquet"
    )

    distribution_path = (
        output_dir
        / "evidence_distribution.parquet"
    )

    tier_statistics_path = (
        output_dir
        / "tier_statistics.parquet"
    )

    summary_path = (
        output_dir
        / "entity_linking_analysis_summary.txt"
    )

    # -------------------------------------------------------------------------
    # STEP 8-6. Parquet 저장
    # -------------------------------------------------------------------------
    tiered_df.write_parquet(
        tiered_path,
        compression="zstd",
    )

    priority_df.write_parquet(
        priority_path,
        compression="zstd",
    )

    per_priority_df.write_parquet(
        per_priority_path,
        compression="zstd",
    )

    org_priority_df.write_parquet(
        org_priority_path,
        compression="zstd",
    )

    ambiguous_df.write_parquet(
        ambiguous_path,
        compression="zstd",
    )

    distribution_df.write_parquet(
        distribution_path,
        compression="zstd",
    )

    tier_statistics_df.write_parquet(
        tier_statistics_path,
        compression="zstd",
    )

    # -------------------------------------------------------------------------
    # STEP 8-7. Summary 저장
    # -------------------------------------------------------------------------
    _write_summary(
        output_path=summary_path,
        source_path=candidates_path,
        tiered_df=tiered_df,
        priority_df=priority_df,
        per_priority_df=per_priority_df,
        org_priority_df=org_priority_df,
        ambiguous_df=ambiguous_df,
        distribution_df=distribution_df,
    )

    return {
        "status": "SUCCESS",
        "source_candidate_count": int(
            candidates.height
        ),

        "tier_a1_count": int(
            _count_tier(
                tiered_df,
                "A1_UNIQUE_SAME_ARTICLE",
            )
        ),

        "tier_a2_count": int(
            _count_tier(
                tiered_df,
                "A2_UNIQUE_SHARED_EVENT",
            )
        ),

        "tier_b_count": int(
            _count_tier(
                tiered_df,
                "B_UNIQUE_CONTEXT_ONLY",
            )
        ),

        "tier_c1_count": int(
            _count_tier(
                tiered_df,
                "C1_AMBIGUOUS_SAME_ARTICLE",
            )
        ),

        "tier_c2_count": int(
            _count_tier(
                tiered_df,
                "C2_AMBIGUOUS_SHARED_EVENT",
            )
        ),

        "tier_c3_count": int(
            _count_tier(
                tiered_df,
                "C3_AMBIGUOUS_CONTEXT_ONLY",
            )
        ),

        "tier_d_count": int(
            _count_tier(
                tiered_df,
                "D_WEAK_EVIDENCE",
            )
        ),

        "priority_review_candidate_count": int(
            priority_df.height
        ),

        "per_priority_candidate_count": int(
            per_priority_df.height
        ),

        "org_priority_candidate_count": int(
            org_priority_df.height
        ),

        "ambiguous_evidence_candidate_count": int(
            ambiguous_df.height
        ),

        "summary_path": str(
            summary_path
        ),

        "tiered_candidates_path": str(
            tiered_path
        ),

        "priority_review_path": str(
            priority_path
        ),

        "per_priority_review_path": str(
            per_priority_path
        ),

        "org_priority_review_path": str(
            org_priority_path
        ),

        "ambiguous_evidence_path": str(
            ambiguous_path
        ),

        "distribution_path": str(
            distribution_path
        ),

        "tier_statistics_path": str(
            tier_statistics_path
        ),
    }


# =============================================================================
# 9. CLI
# =============================================================================


def _parse_args() -> argparse.Namespace:
    """
    기본 실행:
        python -m src.analyze_entity_linking_candidates

    경로 직접 지정:
        python -m src.analyze_entity_linking_candidates \
            --candidates <entity_linking_candidates.parquet> \
            --output-dir <output_dir>
    """

    parser = argparse.ArgumentParser(
        description=(
            "Entity Linking 전체 후보를 evidence/ambiguity 기준으로 "
            "분석하고 우선 검토 후보를 추립니다."
        )
    )

    parser.add_argument(
        "--candidates",
        type=Path,
        default=(
            DEFAULT_CANDIDATES_PATH
        ),
        help=(
            "inspect_entity_linking_candidates.py가 생성한 "
            "entity_linking_candidates.parquet 경로"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            DEFAULT_OUTPUT_DIR
        ),
        help=(
            "Entity Linking 후보 분석 결과 저장 폴더"
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    print(
        "=" * 80
    )
    print(
        "Entity Linking Candidate Analysis 시작"
    )
    print(
        "=" * 80
    )

    print(
        f"candidates = {args.candidates}"
    )

    print(
        f"output_dir = {args.output_dir}"
    )

    print()

    result = (
        analyze_entity_linking_candidates(
            candidates_path=(
                args.candidates
            ),
            output_dir=(
                args.output_dir
            ),
        )
    )

    print(
        "=" * 80
    )
    print(
        "Entity Linking Candidate Analysis 완료"
    )
    print(
        "=" * 80
    )

    for key, value in (
        result.items()
    ):
        print(
            f"{key}: {value}"
        )

    print()
    print(
        "다음 파일을 먼저 확인하세요:"
    )

    print(
        "1. entity_linking_analysis_summary.txt"
    )

    print(
        "2. priority_review_candidates.parquet"
    )

    print(
        "3. per_priority_review_candidates.parquet"
    )

    print(
        "4. org_priority_review_candidates.parquet"
    )

    print(
        "5. ambiguous_evidence_candidates.parquet"
    )

    print(
        "6. evidence_distribution.parquet"
    )

    print(
        "7. tier_statistics.parquet"
    )


if __name__ == "__main__":
    main()
