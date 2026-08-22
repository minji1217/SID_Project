from __future__ import annotations

"""
Build Wikidata Context Review Excel
===================================

목적
----
inspect_wikidata_context_disambiguation.py 결과를 사람이 직접 검토하기 쉬운
Excel 파일로 만든다.

왜 필요한가?
------------
현재 context 결과에는:

    CONTEXT_UNIQUE_TOP
    CONTEXT_WEAK_TOP
    CONTEXT_TIE

가 있다.

하지만:
    CONTEXT_UNIQUE_TOP = "점수상 1등이 꽤 앞섰다"
이지
    CONTEXT_UNIQUE_TOP = "정답 QID다"
는 아니다.

따라서 각 그룹에서 표본을 뽑아 사람이 직접:

    CORRECT
    WRONG
    AMBIGUOUS

를 검수한다.

입력
----
data/output/experiments/wikidata_context_disambiguation/

1. article_candidate_context_scores.parquet
2. article_entity_context_status.parquet

출력
----
data/output/experiments/wikidata_context_disambiguation/
wikidata_context_manual_review.xlsx

Workbook 시트
-------------
README
UNIQUE_TOP
WEAK_TOP
TIE
CANDIDATE_DETAIL

기본 표본
---------
각 상태별 30개.

같은 Entity가 너무 많은 기사를 차지하지 않도록
한 canonical_entity_key당 최대 3개 기사만 뽑는다.

실행
----
프로젝트 루트:

    python -m src.build_wikidata_context_review_xlsx

필요 패키지
-----------
    polars
    XlsxWriter

XlsxWriter가 없다면:

    pip install XlsxWriter
"""

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import polars as pl

try:
    import xlsxwriter
except ImportError as exc:
    raise ImportError(
        "Excel 생성에 XlsxWriter가 필요합니다.\n"
        "다음 명령으로 설치하세요:\n\n"
        "    pip install XlsxWriter"
    ) from exc


# =============================================================================
# 1. 경로
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SOURCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "experiments"
    / "wikidata_context_disambiguation"
)

DEFAULT_SCORES_PATH = (
    DEFAULT_SOURCE_DIR
    / "article_candidate_context_scores.parquet"
)

DEFAULT_ARTICLE_STATUS_PATH = (
    DEFAULT_SOURCE_DIR
    / "article_entity_context_status.parquet"
)

DEFAULT_OUTPUT_PATH = (
    DEFAULT_SOURCE_DIR
    / "wikidata_context_manual_review.xlsx"
)


# =============================================================================
# 2. 리뷰 표본 설정
# =============================================================================

DEFAULT_SAMPLE_PER_STATUS = 30

# 같은 canonical Entity가 기사 수가 많다는 이유만으로
# 리뷰 표본을 독식하지 않도록 제한.
DEFAULT_MAX_PER_ENTITY = 3

REVIEW_STATUSES = (
    "CONTEXT_UNIQUE_TOP",
    "CONTEXT_WEAK_TOP",
    "CONTEXT_TIE",
)

SHEET_BY_STATUS = {
    "CONTEXT_UNIQUE_TOP": "UNIQUE_TOP",
    "CONTEXT_WEAK_TOP": "WEAK_TOP",
    "CONTEXT_TIE": "TIE",
}


# =============================================================================
# 3. 입력 필수 컬럼
# =============================================================================

REQUIRED_STATUS_COLUMNS = {
    "article_id",
    "entity_group",
    "canonical_entity",
    "canonical_entity_key",
    "title",
    "type_match_candidate_count",
    "top_qid",
    "top_label",
    "top_description",
    "top_context_score",
    "second_context_score",
    "top_margin",
    "top_tie_count",
    "diagnostic_status",
    "co_entity_surfaces",
}

REQUIRED_SCORE_COLUMNS = {
    "article_id",
    "entity_group",
    "canonical_entity",
    "canonical_entity_key",
    "qid",
    "candidate_label",
    "candidate_description",
    "candidate_rank",
    "context_score",
    "exact_label_match",
    "search_match_exact",
    "label_context_overlap_tokens",
    "description_context_overlap_tokens",
}


# =============================================================================
# 4. 공통 검사
# =============================================================================


def _require_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"{description} 파일이 없습니다: {path}"
        )


def _require_columns(
    df: pl.DataFrame,
    required: set[str],
    description: str,
) -> None:
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{description}에 필요한 컬럼이 없습니다: "
            + ", ".join(sorted(missing))
        )


# =============================================================================
# 5. 입력 로드
# =============================================================================


def _load_inputs(
    scores_path: Path,
    article_status_path: Path,
) -> tuple[pl.DataFrame, pl.DataFrame]:

    _require_file(
        scores_path,
        "article candidate context scores",
    )

    _require_file(
        article_status_path,
        "article entity context status",
    )

    scores_df = pl.read_parquet(
        scores_path
    )

    status_df = pl.read_parquet(
        article_status_path
    )

    _require_columns(
        scores_df,
        REQUIRED_SCORE_COLUMNS,
        "article_candidate_context_scores.parquet",
    )

    _require_columns(
        status_df,
        REQUIRED_STATUS_COLUMNS,
        "article_entity_context_status.parquet",
    )

    return (
        scores_df,
        status_df,
    )


# =============================================================================
# 6. 표본을 골고루 뽑기
# =============================================================================


def _spread_indices(
    n: int,
    k: int,
) -> list[int]:
    """
    정렬된 데이터에서 앞/중간/뒤를 골고루 선택한다.

    예:
        n=100, k=5
        -> 대략 [0, 25, 50, 74, 99]

    random sampling을 쓰지 않아 실행할 때마다 같은 결과를 만든다.
    """

    if n <= 0 or k <= 0:
        return []

    if k >= n:
        return list(range(n))

    if k == 1:
        return [n // 2]

    indices = []

    for i in range(k):
        idx = round(
            i
            * (n - 1)
            / (k - 1)
        )

        if idx not in indices:
            indices.append(idx)

    return indices


def _select_diverse_sample(
    status_df: pl.DataFrame,
    *,
    status: str,
    sample_count: int,
    max_per_entity: int,
) -> list[dict[str, Any]]:
    """
    한 상태에서 review case를 골고루 뽑는다.

    1. 상태 필터
    2. 점수/margin 기준 정렬
    3. 정렬 결과 전체 구간에서 골고루 후보를 순회
    4. canonical entity당 최대 max_per_entity만 선택

    UNIQUE_TOP:
        margin 큰 것부터 작은 것까지 골고루.

    WEAK_TOP:
        weak margin의 다양한 구간.

    TIE:
        top_context_score 높은 tie부터 낮은 tie까지 골고루.
    """

    subset = (
        status_df.filter(
            pl.col(
                "diagnostic_status"
            )
            == status
        )
    )

    if not subset.height:
        return []

    if status in {
        "CONTEXT_UNIQUE_TOP",
        "CONTEXT_WEAK_TOP",
    }:
        subset = subset.sort(
            [
                "top_margin",
                "top_context_score",
                "canonical_entity_key",
                "article_id",
            ],
            descending=[
                True,
                True,
                False,
                False,
            ],
            nulls_last=True,
        )

    else:
        subset = subset.sort(
            [
                "top_context_score",
                "canonical_entity_key",
                "article_id",
            ],
            descending=[
                True,
                False,
                False,
            ],
        )

    rows = subset.iter_rows(
        named=True
    )

    rows = list(rows)

    # 한 번에 sample_count만 골라버리면 같은 Entity가 몰릴 수 있어
    # 전체에서 넉넉히 spread 후보를 만든다.
    candidate_index_count = min(
        len(rows),
        max(
            sample_count * 8,
            sample_count,
        ),
    )

    initial_indices = _spread_indices(
        len(rows),
        candidate_index_count,
    )

    # spread 순서를 먼저 본 후,
    # 혹시 diversity 제한 때문에 부족하면 전체를 다시 순회한다.
    order = (
        initial_indices
        + [
            i
            for i in range(
                len(rows)
            )
            if i not in set(
                initial_indices
            )
        ]
    )

    selected = []

    entity_counts: dict[
        str,
        int,
    ] = defaultdict(int)

    seen_cases: set[
        tuple[int, str]
    ] = set()

    for idx in order:
        row = rows[idx]

        article_id = int(
            row[
                "article_id"
            ]
        )

        key = str(
            row[
                "canonical_entity_key"
            ]
        )

        case_key = (
            article_id,
            key,
        )

        if case_key in seen_cases:
            continue

        if entity_counts[
            key
        ] >= max_per_entity:
            continue

        selected.append(
            row
        )

        seen_cases.add(
            case_key
        )

        entity_counts[
            key
        ] += 1

        if len(
            selected
        ) >= sample_count:
            break

    return selected


# =============================================================================
# 7. Candidate lookup
# =============================================================================


def _build_score_lookup(
    scores_df: pl.DataFrame,
) -> dict[
    tuple[int, str],
    list[dict[str, Any]],
]:
    """
    (article_id, canonical_entity_key)
        -> 후보 QID 목록
    """

    result: dict[
        tuple[int, str],
        list[dict[str, Any]],
    ] = defaultdict(list)

    for row in scores_df.iter_rows(
        named=True
    ):
        case_key = (
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

        result[
            case_key
        ].append(
            row
        )

    for case_key in result:
        result[
            case_key
        ].sort(
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
            )
        )

    return dict(
        result
    )


# =============================================================================
# 8. Excel에 보여줄 후보 요약
# =============================================================================


def _candidate_summary_text(
    candidate_rows: list[
        dict[str, Any]
    ],
) -> str:
    """
    한 셀에서 전체 후보를 읽기 쉽게 보여준다.

    예:
        #1 Q7747 | Vladimir Putin | score=0.6250
           Russian politician...
        #2 Q...   | Igor Putin     | score=0.3125
           Russian businessman...
    """

    lines = []

    for index, row in enumerate(
        candidate_rows,
        start=1,
    ):
        qid = str(
            row.get(
                "qid",
                "",
            )
        )

        label = str(
            row.get(
                "candidate_label",
                "",
            )
            or ""
        )

        description = str(
            row.get(
                "candidate_description",
                "",
            )
            or ""
        )

        score = float(
            row.get(
                "context_score",
                0.0,
            )
            or 0.0
        )

        lines.append(
            (
                f"#{index} {qid} | {label} "
                f"| score={score:.4f}"
            )
        )

        if description:
            lines.append(
                f"   {description}"
            )

    return "\n".join(
        lines
    )


def _list_text(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(
        value,
        (list, tuple),
    ):
        return ", ".join(
            str(item)
            for item in value
        )

    return str(value)


# =============================================================================
# 9. Review row 만들기
# =============================================================================


def _build_review_rows(
    selected_cases: list[
        dict[str, Any]
    ],
    score_lookup: dict[
        tuple[int, str],
        list[
            dict[str, Any]
        ],
    ],
) -> list[list[Any]]:
    """
    Excel 리뷰 시트용 row.

    마지막 4개 컬럼은 사용자가 직접 입력:
        REVIEW_LABEL
        CORRECT_QID
        CORRECT_LABEL
        REVIEW_NOTE
    """

    rows = []

    for case_no, case in enumerate(
        selected_cases,
        start=1,
    ):
        article_id = int(
            case[
                "article_id"
            ]
        )

        key = str(
            case[
                "canonical_entity_key"
            ]
        )

        candidate_rows = (
            score_lookup.get(
                (
                    article_id,
                    key,
                ),
                [],
            )
        )

        rows.append(
            [
                case_no,
                article_id,
                str(
                    case[
                        "entity_group"
                    ]
                ),
                str(
                    case[
                        "canonical_entity"
                    ]
                ),
                key,
                str(
                    case[
                        "diagnostic_status"
                    ]
                ),
                str(
                    case[
                        "title"
                    ]
                    or ""
                ),
                _list_text(
                    case.get(
                        "co_entity_surfaces"
                    )
                ),
                int(
                    case[
                        "type_match_candidate_count"
                    ]
                ),
                str(
                    case[
                        "top_qid"
                    ]
                ),
                str(
                    case[
                        "top_label"
                    ]
                    or ""
                ),
                str(
                    case[
                        "top_description"
                    ]
                    or ""
                ),
                float(
                    case[
                        "top_context_score"
                    ]
                    or 0.0
                ),
                (
                    float(
                        case[
                            "second_context_score"
                        ]
                    )
                    if case.get(
                        "second_context_score"
                    )
                    is not None
                    else None
                ),
                (
                    float(
                        case[
                            "top_margin"
                        ]
                    )
                    if case.get(
                        "top_margin"
                    )
                    is not None
                    else None
                ),
                int(
                    case[
                        "top_tie_count"
                    ]
                ),
                _candidate_summary_text(
                    candidate_rows
                ),

                # 사람이 직접 입력
                "",
                "",
                "",
                "",
            ]
        )

    return rows


# =============================================================================
# 10. Excel 생성
# =============================================================================


REVIEW_HEADERS = [
    "CASE_NO",
    "ARTICLE_ID",
    "ENTITY_GROUP",
    "CANONICAL_ENTITY",
    "CANONICAL_ENTITY_KEY",
    "CONTEXT_STATUS",
    "ARTICLE_TITLE",
    "CO_ENTITIES",
    "TYPE_MATCH_CANDIDATE_COUNT",
    "TOP_QID",
    "TOP_LABEL",
    "TOP_DESCRIPTION",
    "TOP_SCORE",
    "SECOND_SCORE",
    "TOP_MARGIN",
    "TOP_TIE_COUNT",
    "ALL_TYPE_MATCH_CANDIDATES",
    "REVIEW_LABEL",
    "CORRECT_QID",
    "CORRECT_LABEL",
    "REVIEW_NOTE",
]


def _write_readme(
    workbook: xlsxwriter.Workbook,
    formats: dict[str, Any],
) -> None:

    sheet = workbook.add_worksheet(
        "README"
    )

    sheet.hide_gridlines(
        2
    )

    sheet.set_column(
        "A:A",
        3,
    )

    sheet.set_column(
        "B:B",
        25,
    )

    sheet.set_column(
        "C:C",
        85,
    )

    sheet.set_row(
        1,
        30,
    )

    sheet.write(
        "B2",
        "Wikidata Context Manual Review",
        formats[
            "title"
        ],
    )

    guide_rows = [
        (
            "목적",
            (
                "Context score가 실제로 올바른 QID를 위로 올리는지 "
                "사람이 표본 검수하는 Excel입니다."
            ),
        ),
        (
            "UNIQUE_TOP",
            (
                "문맥 점수 1등이 2등보다 꽤 앞선 사례. "
                "이 그룹의 CORRECT 비율이 높아야 자동화 신호로 쓸 수 있습니다."
            ),
        ),
        (
            "WEAK_TOP",
            (
                "1등은 있지만 2등과 점수 차이가 작은 사례. "
                "GPT disambiguation 후보가 될 가능성이 높습니다."
            ),
        ),
        (
            "TIE",
            (
                "최고점 후보가 동점인 사례. 현재 lexical context만으로는 구분이 어렵습니다."
            ),
        ),
        (
            "REVIEW_LABEL",
            (
                "CORRECT = 현재 TOP_QID가 기사 문맥상 올바름\n"
                "WRONG = TOP_QID가 틀렸고 다른 후보가 더 맞음\n"
                "AMBIGUOUS = 현재 정보로 사람도 확정하기 어려움"
            ),
        ),
        (
            "CORRECT_QID",
            (
                "WRONG인 경우 후보 목록에서 실제로 더 적절한 QID를 적습니다. "
                "후보 안에 정답이 없다면 NO_CORRECT_CANDIDATE라고 적어도 됩니다."
            ),
        ),
        (
            "검수 기준",
            (
                "ARTICLE_TITLE + CO_ENTITIES를 먼저 보고, "
                "ALL_TYPE_MATCH_CANDIDATES에서 각 QID/label/description을 비교하세요."
            ),
        ),
        (
            "중요",
            (
                "CONTEXT_UNIQUE_TOP이라고 해서 자동으로 정답이 아닙니다. "
                "이번 표본 검수가 정확도를 확인하기 위한 단계입니다."
            ),
        ),
    ]

    start_row = 4

    for offset, (
        name,
        description,
    ) in enumerate(
        guide_rows
    ):
        row = (
            start_row
            + offset
        )

        sheet.write(
            row,
            1,
            name,
            formats[
                "guide_label"
            ],
        )

        sheet.write(
            row,
            2,
            description,
            formats[
                "guide_text"
            ],
        )

        sheet.set_row(
            row,
            50,
        )

    sheet.write(
        "B15",
        "추천 검수 순서",
        formats[
            "section"
        ],
    )

    sheet.write(
        "C15",
        (
            "1) UNIQUE_TOP 30건 → 2) WEAK_TOP 30건 → "
            "3) TIE 30건 → 4) 결과를 ChatGPT에 전달"
        ),
        formats[
            "guide_text"
        ],
    )


def _write_review_sheet(
    workbook: xlsxwriter.Workbook,
    *,
    sheet_name: str,
    rows: list[
        list[Any]
    ],
    formats: dict[
        str,
        Any,
    ],
) -> None:

    sheet = workbook.add_worksheet(
        sheet_name
    )

    sheet.hide_gridlines(
        2
    )

    sheet.freeze_panes(
        1,
        6,
    )

    # 헤더
    for col, header in enumerate(
        REVIEW_HEADERS
    ):
        sheet.write(
            0,
            col,
            header,
            formats[
                "header"
            ],
        )

    # 데이터
    for row_idx, row in enumerate(
        rows,
        start=1,
    ):
        for col_idx, value in enumerate(
            row
        ):
            if col_idx in {
                12,
                13,
                14,
            } and value is not None:
                sheet.write_number(
                    row_idx,
                    col_idx,
                    float(
                        value
                    ),
                    formats[
                        "score"
                    ],
                )

            elif col_idx in {
                6,
                7,
                11,
                16,
                20,
            }:
                sheet.write(
                    row_idx,
                    col_idx,
                    value,
                    formats[
                        "long_text"
                    ],
                )

            elif col_idx >= 17:
                sheet.write(
                    row_idx,
                    col_idx,
                    value,
                    formats[
                        "review_input"
                    ],
                )

            else:
                sheet.write(
                    row_idx,
                    col_idx,
                    value,
                    formats[
                        "body"
                    ],
                )

        sheet.set_row(
            row_idx,
            96,
        )

    # 열 너비
    widths = {
        0: 8,
        1: 13,
        2: 12,
        3: 24,
        4: 31,
        5: 22,
        6: 52,
        7: 45,
        8: 14,
        9: 15,
        10: 24,
        11: 45,
        12: 12,
        13: 12,
        14: 12,
        15: 12,
        16: 65,
        17: 16,
        18: 18,
        19: 25,
        20: 42,
    }

    for col, width in (
        widths.items()
    ):
        sheet.set_column(
            col,
            col,
            width,
        )

    # REVIEW_LABEL dropdown
    if rows:
        sheet.data_validation(
            1,
            17,
            len(rows),
            17,
            {
                "validate": "list",
                "source": [
                    "CORRECT",
                    "WRONG",
                    "AMBIGUOUS",
                ],
                "input_title": "검수 결과",
                "input_message": (
                    "TOP_QID가 맞으면 CORRECT, "
                    "틀리면 WRONG, 판단 불가면 AMBIGUOUS"
                ),
            },
        )

        # 색상 조건부서식
        sheet.conditional_format(
            1,
            17,
            len(rows),
            17,
            {
                "type": "text",
                "criteria": "containing",
                "value": "CORRECT",
                "format": formats[
                    "correct"
                ],
            },
        )

        sheet.conditional_format(
            1,
            17,
            len(rows),
            17,
            {
                "type": "text",
                "criteria": "containing",
                "value": "WRONG",
                "format": formats[
                    "wrong"
                ],
            },
        )

        sheet.conditional_format(
            1,
            17,
            len(rows),
            17,
            {
                "type": "text",
                "criteria": "containing",
                "value": "AMBIGUOUS",
                "format": formats[
                    "ambiguous"
                ],
            },
        )

    sheet.autofilter(
        0,
        0,
        max(
            len(rows),
            1,
        ),
        len(
            REVIEW_HEADERS
        )
        - 1,
    )


def _write_candidate_detail(
    workbook: xlsxwriter.Workbook,
    *,
    selected_case_keys: set[
        tuple[int, str]
    ],
    scores_df: pl.DataFrame,
    formats: dict[
        str,
        Any,
    ],
) -> None:
    """
    선택된 review case의 후보 하나하나를 별도 시트에 펼친다.
    """

    sheet = workbook.add_worksheet(
        "CANDIDATE_DETAIL"
    )

    headers = [
        "ARTICLE_ID",
        "ENTITY_GROUP",
        "CANONICAL_ENTITY",
        "CANONICAL_ENTITY_KEY",
        "QID",
        "CANDIDATE_LABEL",
        "CANDIDATE_DESCRIPTION",
        "CONTEXT_SCORE",
        "EXACT_LABEL_MATCH",
        "SEARCH_MATCH_EXACT",
        "LABEL_CONTEXT_OVERLAP",
        "DESCRIPTION_CONTEXT_OVERLAP",
    ]

    for col, header in enumerate(
        headers
    ):
        sheet.write(
            0,
            col,
            header,
            formats[
                "header"
            ],
        )

    filtered_rows = []

    for row in scores_df.iter_rows(
        named=True
    ):
        case_key = (
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

        if case_key not in (
            selected_case_keys
        ):
            continue

        filtered_rows.append(
            row
        )

    filtered_rows.sort(
        key=lambda row: (
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
            -float(
                row[
                    "context_score"
                ]
            ),
        )
    )

    for r, row in enumerate(
        filtered_rows,
        start=1,
    ):
        values = [
            int(
                row[
                    "article_id"
                ]
            ),
            str(
                row[
                    "entity_group"
                ]
            ),
            str(
                row[
                    "canonical_entity"
                ]
            ),
            str(
                row[
                    "canonical_entity_key"
                ]
            ),
            str(
                row[
                    "qid"
                ]
            ),
            str(
                row[
                    "candidate_label"
                ]
                or ""
            ),
            str(
                row[
                    "candidate_description"
                ]
                or ""
            ),
            float(
                row[
                    "context_score"
                ]
                or 0.0
            ),
            int(
                row[
                    "exact_label_match"
                ]
            ),
            int(
                row[
                    "search_match_exact"
                ]
            ),
            _list_text(
                row[
                    "label_context_overlap_tokens"
                ]
            ),
            _list_text(
                row[
                    "description_context_overlap_tokens"
                ]
            ),
        ]

        for c, value in enumerate(
            values
        ):
            if c == 7:
                sheet.write_number(
                    r,
                    c,
                    value,
                    formats[
                        "score"
                    ],
                )

            elif c in {
                6,
                10,
                11,
            }:
                sheet.write(
                    r,
                    c,
                    value,
                    formats[
                        "long_text"
                    ],
                )

            else:
                sheet.write(
                    r,
                    c,
                    value,
                    formats[
                        "body"
                    ],
                )

        sheet.set_row(
            r,
            52,
        )

    widths = [
        13,
        12,
        24,
        31,
        15,
        25,
        50,
        13,
        14,
        14,
        35,
        40,
    ]

    for col, width in enumerate(
        widths
    ):
        sheet.set_column(
            col,
            col,
            width,
        )

    sheet.freeze_panes(
        1,
        4,
    )

    sheet.autofilter(
        0,
        0,
        max(
            len(
                filtered_rows
            ),
            1,
        ),
        len(headers)
        - 1,
    )


def _build_formats(
    workbook: xlsxwriter.Workbook,
) -> dict[str, Any]:

    return {
        "title": workbook.add_format(
            {
                "bold": True,
                "font_size": 20,
                "font_color": "#FFFFFF",
                "bg_color": "#283593",
                "align": "left",
                "valign": "vcenter",
            }
        ),

        "section": workbook.add_format(
            {
                "bold": True,
                "font_size": 12,
                "font_color": "#FFFFFF",
                "bg_color": "#5C6BC0",
                "valign": "vcenter",
            }
        ),

        "guide_label": workbook.add_format(
            {
                "bold": True,
                "font_color": "#263238",
                "bg_color": "#E8EAF6",
                "border": 1,
                "border_color": "#D5D8E5",
                "valign": "top",
            }
        ),

        "guide_text": workbook.add_format(
            {
                "text_wrap": True,
                "valign": "top",
                "border": 1,
                "border_color": "#E5E7EB",
            }
        ),

        "header": workbook.add_format(
            {
                "bold": True,
                "font_color": "#FFFFFF",
                "bg_color": "#3949AB",
                "border": 1,
                "border_color": "#D5D8E5",
                "align": "center",
                "valign": "vcenter",
                "text_wrap": True,
            }
        ),

        "body": workbook.add_format(
            {
                "border": 1,
                "border_color": "#E5E7EB",
                "valign": "top",
            }
        ),

        "long_text": workbook.add_format(
            {
                "border": 1,
                "border_color": "#E5E7EB",
                "valign": "top",
                "text_wrap": True,
            }
        ),

        "score": workbook.add_format(
            {
                "border": 1,
                "border_color": "#E5E7EB",
                "valign": "top",
                "num_format": "0.0000",
            }
        ),

        "review_input": workbook.add_format(
            {
                "border": 1,
                "border_color": "#B0BEC5",
                "valign": "top",
                "text_wrap": True,
                "bg_color": "#FFFDE7",
            }
        ),

        "correct": workbook.add_format(
            {
                "bg_color": "#DCFCE7",
                "font_color": "#166534",
                "bold": True,
            }
        ),

        "wrong": workbook.add_format(
            {
                "bg_color": "#FEE2E2",
                "font_color": "#991B1B",
                "bold": True,
            }
        ),

        "ambiguous": workbook.add_format(
            {
                "bg_color": "#FEF3C7",
                "font_color": "#92400E",
                "bold": True,
            }
        ),
    }


# =============================================================================
# 11. 전체 실행
# =============================================================================


def build_review_workbook(
    *,
    scores_path: Path,
    article_status_path: Path,
    output_path: Path,
    sample_per_status: int,
    max_per_entity: int,
) -> dict[str, Any]:

    (
        scores_df,
        status_df,
    ) = _load_inputs(
        scores_path,
        article_status_path,
    )

    score_lookup = (
        _build_score_lookup(
            scores_df
        )
    )

    selected_by_status: dict[
        str,
        list[
            dict[str, Any]
        ],
    ] = {}

    for status in (
        REVIEW_STATUSES
    ):
        selected_by_status[
            status
        ] = (
            _select_diverse_sample(
                status_df,
                status=status,
                sample_count=(
                    sample_per_status
                ),
                max_per_entity=(
                    max_per_entity
                ),
            )
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    workbook = xlsxwriter.Workbook(
        str(
            output_path
        )
    )

    formats = (
        _build_formats(
            workbook
        )
    )

    _write_readme(
        workbook,
        formats,
    )

    selected_case_keys: set[
        tuple[int, str]
    ] = set()

    for status in (
        REVIEW_STATUSES
    ):
        cases = (
            selected_by_status[
                status
            ]
        )

        rows = (
            _build_review_rows(
                cases,
                score_lookup,
            )
        )

        _write_review_sheet(
            workbook,
            sheet_name=(
                SHEET_BY_STATUS[
                    status
                ]
            ),
            rows=rows,
            formats=formats,
        )

        for case in cases:
            selected_case_keys.add(
                (
                    int(
                        case[
                            "article_id"
                        ]
                    ),
                    str(
                        case[
                            "canonical_entity_key"
                        ]
                    ),
                )
            )

    _write_candidate_detail(
        workbook,
        selected_case_keys=(
            selected_case_keys
        ),
        scores_df=scores_df,
        formats=formats,
    )

    workbook.close()

    return {
        "status": "SUCCESS",
        "output_path": str(
            output_path
        ),
        "unique_top_sample_count": len(
            selected_by_status[
                "CONTEXT_UNIQUE_TOP"
            ]
        ),
        "weak_top_sample_count": len(
            selected_by_status[
                "CONTEXT_WEAK_TOP"
            ]
        ),
        "tie_sample_count": len(
            selected_by_status[
                "CONTEXT_TIE"
            ]
        ),
        "total_review_case_count": int(
            len(
                selected_case_keys
            )
        ),
    }


# =============================================================================
# 12. CLI
# =============================================================================


def _parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Wikidata context disambiguation 결과에서 "
            "UNIQUE/WEAK/TIE 표본을 뽑아 수동검수 Excel을 만듭니다."
        )
    )

    parser.add_argument(
        "--scores",
        type=Path,
        default=(
            DEFAULT_SCORES_PATH
        ),
    )

    parser.add_argument(
        "--article-status",
        type=Path,
        default=(
            DEFAULT_ARTICLE_STATUS_PATH
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=(
            DEFAULT_OUTPUT_PATH
        ),
    )

    parser.add_argument(
        "--sample-per-status",
        type=int,
        default=(
            DEFAULT_SAMPLE_PER_STATUS
        ),
        help=(
            "UNIQUE_TOP / WEAK_TOP / TIE 각각 뽑을 표본 수. 기본 30"
        ),
    )

    parser.add_argument(
        "--max-per-entity",
        type=int,
        default=(
            DEFAULT_MAX_PER_ENTITY
        ),
        help=(
            "한 canonical Entity가 한 시트에서 차지할 최대 기사 수. 기본 3"
        ),
    )

    args = parser.parse_args()

    if (
        args.sample_per_status
        <= 0
    ):
        parser.error(
            "--sample-per-status는 1 이상이어야 합니다."
        )

    if (
        args.max_per_entity
        <= 0
    ):
        parser.error(
            "--max-per-entity는 1 이상이어야 합니다."
        )

    return args


def main() -> None:

    args = _parse_args()

    print(
        "=" * 100
    )
    print(
        "Wikidata Context Manual Review Excel 생성 시작"
    )
    print(
        "=" * 100
    )

    result = (
        build_review_workbook(
            scores_path=(
                args.scores
            ),
            article_status_path=(
                args.article_status
            ),
            output_path=(
                args.output
            ),
            sample_per_status=(
                args.sample_per_status
            ),
            max_per_entity=(
                args.max_per_entity
            ),
        )
    )

    print()
    print(
        "=" * 100
    )
    print(
        "완료"
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
