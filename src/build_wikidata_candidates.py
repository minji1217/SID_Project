from __future__ import annotations

"""
Build Wikidata Candidates
=========================

목적
----
normalize_v2의 Train-used canonical entity 전체를 대상으로
Wikidata QID 후보를 검색하고 캐시한다.

중요
----
이 단계는 아직 "Entity Linking 결정" 단계가 아니다.

즉:
    PER::putin
        ↓
    Wikidata 후보
        Q7747  Vladimir Putin
        Q30524893 Putin (surname)
        ...

까지만 저장한다.

아직 하지 않는 것
-----------------
- 후보 중 정답 QID 자동 선택
- GPT disambiguation
- entity_processing.py 수정
- Event clustering 재실행

입력
----
data/output/experiments/normalize_v2/model_inputs/article_entities.parquet

Train-used row만 사용한다.

필요 컬럼:
- article_id
- entity_group
- canonical_entity
- canonical_entity_key
- is_train_used

출력
----
data/output/experiments/wikidata_candidates/

1. wikidata_search_cache.jsonl
   - 중간 checkpoint
   - 한 Entity 검색이 끝날 때마다 append
   - 실행 중 끊겨도 다음 실행에서 이미 처리한 Entity는 skip

2. wikidata_entity_status.parquet
   - Entity별 검색 상태
   - article_df / mention_count 포함

3. wikidata_candidates.parquet
   - Entity별 Wikidata 후보 QID 목록

4. wikidata_candidate_summary.txt
   - 전체 요약

권장 실행 순서
--------------
1) 먼저 100개만:

    python -m src.build_wikidata_candidates --limit 100

2) 결과 확인 후 전체:

    python -m src.build_wikidata_candidates

이미 cache에 있는 100개는 다시 호출하지 않는다.
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import polars as pl


# =============================================================================
# 1. 기본 경로
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_ARTICLE_ENTITIES_PATH = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "experiments"
    / "normalize_v2"
    / "model_inputs"
    / "article_entities.parquet"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "experiments"
    / "wikidata_candidates"
)

CACHE_FILENAME = "wikidata_search_cache.jsonl"
STATUS_FILENAME = "wikidata_entity_status.parquet"
CANDIDATES_FILENAME = "wikidata_candidates.parquet"
SUMMARY_FILENAME = "wikidata_candidate_summary.txt"


# =============================================================================
# 2. Wikidata API 설정
# =============================================================================

WIKIDATA_API = "https://www.wikidata.org/w/api.php"

# 환경변수로 원하는 User-Agent를 넣을 수 있다.
#
# PowerShell 예:
#   $env:WIKIDATA_USER_AGENT="SID-Project/0.1 (mailto:your@email.com)"
#
# 설정하지 않아도 실행은 가능하지만,
# Wikimedia는 연락 가능한 정보가 들어간 User-Agent를 권장한다.
USER_AGENT = os.environ.get(
    "WIKIDATA_USER_AGENT",
    "SID-Project-Wikidata-Linking/0.2 (undergraduate research prototype)",
)

SEARCH_LIMIT = 5

# 이전 probe에서 0.2초 간격으로 429가 발생했기 때문에
# 보수적으로 순차 요청한다.
DEFAULT_REQUEST_SLEEP_SECONDS = 1.5

MAX_RETRIES = 5

# Wikidata label/alias 검색 언어.
DEFAULT_LANGUAGES = ("da", "en")


# =============================================================================
# 3. 필요한 컬럼
# =============================================================================

REQUIRED_COLUMNS = {
    "article_id",
    "entity_group",
    "canonical_entity",
    "canonical_entity_key",
    "is_train_used",
}


# =============================================================================
# 4. 공통 검증
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
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{description}에 필요한 컬럼이 없습니다: "
            + ", ".join(sorted(missing))
        )


# =============================================================================
# 5. Train Entity Universe 생성
# =============================================================================


def _load_train_entity_universe(
    article_entities_path: Path,
    *,
    groups: set[str] | None,
) -> pl.DataFrame:
    """
    normalize_v2 article_entities에서 Train-used entity만 읽고
    canonical_entity_key 단위로 하나의 검색 대상 row를 만든다.

    예
    --
    원본 mention rows:

        article 1  PER  vladimir putin  PER::vladimir putin
        article 2  PER  vladimir putin  PER::vladimir putin
        article 3  PER  putin           PER::putin

    검색 universe:

        PER::vladimir putin
            article_df=2
            mention_count=2

        PER::putin
            article_df=1
            mention_count=1

    즉 같은 canonical entity를 기사마다 반복해서 API 호출하지 않는다.
    """

    _require_file(
        article_entities_path,
        "normalize_v2 article_entities",
    )

    df = pl.read_parquet(
        article_entities_path
    )

    _require_columns(
        df,
        REQUIRED_COLUMNS,
        "article_entities.parquet",
    )

    train_df = (
        df.filter(
            pl.col("is_train_used")
        )
        .select(
            [
                pl.col("article_id").cast(pl.Int64),
                pl.col("entity_group").cast(pl.String),
                pl.col("canonical_entity").cast(pl.String),
                pl.col("canonical_entity_key").cast(pl.String),
            ]
        )
    )

    if groups is not None:
        train_df = train_df.filter(
            pl.col("entity_group").is_in(
                sorted(groups)
            )
        )

    universe = (
        train_df
        .group_by(
            [
                "entity_group",
                "canonical_entity",
                "canonical_entity_key",
            ]
        )
        .agg(
            [
                pl.col("article_id")
                .n_unique()
                .alias("article_df"),

                pl.len()
                .alias("mention_count"),
            ]
        )
        # 먼저 영향력이 큰 Entity부터 처리하면
        # --limit 100 probe가 좀 더 유용하다.
        .sort(
            [
                "article_df",
                "mention_count",
                "canonical_entity_key",
            ],
            descending=[
                True,
                True,
                False,
            ],
        )
    )

    return universe


# =============================================================================
# 6. Wikidata API 한 번 검색
# =============================================================================


def _search_once(
    *,
    surface: str,
    language: str,
    request_sleep_seconds: float,
) -> list[dict[str, Any]]:
    """
    wbsearchentities 한 번 호출.

    429 / 503:
    - Retry-After 헤더가 있으면 그 값을 사용
    - 없으면 exponential backoff

    반환값은 검색 후보 목록이다.
    """

    params = {
        "action": "wbsearchentities",
        "search": surface,
        "language": language,
        "uselang": language,
        "type": "item",
        "limit": SEARCH_LIMIT,
        "format": "json",
        "maxlag": 5,
    }

    url = (
        WIKIDATA_API
        + "?"
        + urlencode(params)
    )

    # gzip을 명시적으로 요청하지 않는다.
    # urllib은 gzip을 자동 decode하지 않기 때문에
    # 이전 probe에서 UnicodeDecodeError가 발생했다.
    request = Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
        },
    )

    raw: bytes | None = None

    for attempt in range(
        MAX_RETRIES
    ):
        try:
            with urlopen(
                request,
                timeout=30,
            ) as response:
                raw = response.read()

            break

        except HTTPError as exc:
            if exc.code not in {
                429,
                503,
            }:
                raise RuntimeError(
                    "Wikidata HTTP 오류: "
                    f"status={exc.code}, "
                    f"surface={surface!r}, "
                    f"language={language}"
                ) from exc

            retry_after = (
                exc.headers.get(
                    "Retry-After"
                )
            )

            if retry_after is not None:
                try:
                    wait_seconds = float(
                        retry_after
                    )
                except ValueError:
                    wait_seconds = (
                        5.0
                        * (2 ** attempt)
                    )
            else:
                wait_seconds = (
                    5.0
                    * (2 ** attempt)
                )

            # 기본 호출 간격보다 더 짧은 retry는 하지 않는다.
            wait_seconds = max(
                wait_seconds,
                request_sleep_seconds,
            )

            print(
                "    "
                f"HTTP {exc.code} "
                f"→ {wait_seconds:.1f}초 후 재시도 "
                f"({attempt + 1}/{MAX_RETRIES})"
            )

            time.sleep(
                wait_seconds
            )

        except URLError as exc:
            raise RuntimeError(
                "Wikidata 연결 오류: "
                f"surface={surface!r}, "
                f"language={language}, "
                f"reason={exc.reason}"
            ) from exc

    else:
        raise RuntimeError(
            "Wikidata API 재시도 횟수를 초과했습니다. "
            f"surface={surface!r}, "
            f"language={language}"
        )

    if raw is None:
        raise RuntimeError(
            "Wikidata 응답을 받지 못했습니다. "
            f"surface={surface!r}, "
            f"language={language}"
        )

    try:
        data = json.loads(
            raw.decode("utf-8")
        )
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as exc:
        raise RuntimeError(
            "Wikidata 응답 JSON 파싱 실패: "
            f"surface={surface!r}, "
            f"language={language}"
        ) from exc

    rows: list[
        dict[str, Any]
    ] = []

    for item in data.get(
        "search",
        [],
    ):
        match = (
            item.get("match")
            or {}
        )

        rows.append(
            {
                "qid": str(
                    item.get(
                        "id",
                        "",
                    )
                ),
                "label": str(
                    item.get(
                        "label",
                        "",
                    )
                ),
                "description": str(
                    item.get(
                        "description",
                        "",
                    )
                    or ""
                ),
                "match_type": str(
                    match.get(
                        "type",
                        "",
                    )
                    or ""
                ),
                "match_text": str(
                    match.get(
                        "text",
                        "",
                    )
                    or ""
                ),
            }
        )

    # 성공한 요청 뒤 기본 간격
    time.sleep(
        request_sleep_seconds
    )

    return rows


# =============================================================================
# 7. 한 Entity의 da + en 후보 검색
# =============================================================================


def _search_entity(
    *,
    entity_group: str,
    surface: str,
    canonical_entity_key: str,
    article_df: int,
    mention_count: int,
    languages: tuple[str, ...],
    request_sleep_seconds: float,
) -> dict[str, Any]:
    """
    하나의 canonical entity를 여러 language로 검색한다.

    같은 QID가 da/en 양쪽에 나오면 후보 하나로 deduplicate한다.
    최초로 발견된 언어와 rank를 보존한다.
    """

    candidates: list[
        dict[str, Any]
    ] = []

    seen_qids: set[str] = set()

    for language in languages:
        language_candidates = (
            _search_once(
                surface=surface,
                language=language,
                request_sleep_seconds=(
                    request_sleep_seconds
                ),
            )
        )

        for search_rank, candidate in enumerate(
            language_candidates,
            start=1,
        ):
            qid = candidate[
                "qid"
            ]

            if not qid:
                continue

            if qid in seen_qids:
                continue

            seen_qids.add(
                qid
            )

            candidates.append(
                {
                    **candidate,
                    "search_language": (
                        language
                    ),
                    "search_rank": int(
                        search_rank
                    ),
                }
            )

    return {
        "entity_group": entity_group,
        "canonical_entity": surface,
        "canonical_entity_key": (
            canonical_entity_key
        ),
        "article_df": int(
            article_df
        ),
        "mention_count": int(
            mention_count
        ),
        "status": (
            "FOUND"
            if candidates
            else "NO_CANDIDATE"
        ),
        "candidate_count": int(
            len(candidates)
        ),
        "candidates": candidates,
    }


# =============================================================================
# 8. JSONL Cache
# =============================================================================


def _load_cache(
    cache_path: Path,
) -> dict[str, dict[str, Any]]:
    """
    기존 JSONL cache를 읽는다.

    key:
        canonical_entity_key

    중간에 프로그램이 종료되어도
    이전에 완료한 Entity 검색 결과를 재사용한다.
    """

    cache: dict[
        str,
        dict[str, Any],
    ] = {}

    if not cache_path.exists():
        return cache

    with cache_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line_number, line in enumerate(
            f,
            start=1,
        ):
            text = line.strip()

            if not text:
                continue

            try:
                row = json.loads(
                    text
                )
            except json.JSONDecodeError:
                print(
                    "WARNING: "
                    f"cache line {line_number} JSON 파싱 실패 → skip"
                )
                continue

            key = str(
                row.get(
                    "canonical_entity_key",
                    "",
                )
            )

            if not key:
                continue

            # 동일 key가 여러 번 있으면 가장 마지막 기록 사용
            cache[key] = row

    return cache


def _append_cache_row(
    cache_path: Path,
    row: dict[str, Any],
) -> None:
    """
    검색 완료 직후 한 Entity 결과를 JSONL에 append.

    Parquet을 Entity마다 다시 쓰지 않기 위해
    checkpoint는 JSONL로 유지한다.
    """

    cache_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with cache_path.open(
        "a",
        encoding="utf-8",
    ) as f:
        f.write(
            json.dumps(
                row,
                ensure_ascii=False,
            )
            + "\n"
        )

        f.flush()


# =============================================================================
# 9. Cache → Parquet 변환
# =============================================================================


def _build_output_frames(
    cache: dict[
        str,
        dict[str, Any],
    ],
) -> tuple[
    pl.DataFrame,
    pl.DataFrame,
]:
    """
    cache 전체를:

    1) Entity status
    2) candidate rows

    두 개 DataFrame으로 변환한다.
    """

    status_rows: list[
        dict[str, Any]
    ] = []

    candidate_rows: list[
        dict[str, Any]
    ] = []

    for key in sorted(
        cache
    ):
        entity_row = cache[
            key
        ]

        status_rows.append(
            {
                "entity_group": str(
                    entity_row[
                        "entity_group"
                    ]
                ),
                "canonical_entity": str(
                    entity_row[
                        "canonical_entity"
                    ]
                ),
                "canonical_entity_key": str(
                    entity_row[
                        "canonical_entity_key"
                    ]
                ),
                "article_df": int(
                    entity_row[
                        "article_df"
                    ]
                ),
                "mention_count": int(
                    entity_row[
                        "mention_count"
                    ]
                ),
                "status": str(
                    entity_row[
                        "status"
                    ]
                ),
                "candidate_count": int(
                    entity_row[
                        "candidate_count"
                    ]
                ),
            }
        )

        for candidate_rank, candidate in enumerate(
            entity_row.get(
                "candidates",
                [],
            ),
            start=1,
        ):
            candidate_rows.append(
                {
                    "entity_group": str(
                        entity_row[
                            "entity_group"
                        ]
                    ),
                    "canonical_entity": str(
                        entity_row[
                            "canonical_entity"
                        ]
                    ),
                    "canonical_entity_key": str(
                        entity_row[
                            "canonical_entity_key"
                        ]
                    ),
                    "article_df": int(
                        entity_row[
                            "article_df"
                        ]
                    ),
                    "mention_count": int(
                        entity_row[
                            "mention_count"
                        ]
                    ),
                    "candidate_rank": int(
                        candidate_rank
                    ),
                    "qid": str(
                        candidate[
                            "qid"
                        ]
                    ),
                    "label": str(
                        candidate[
                            "label"
                        ]
                    ),
                    "description": str(
                        candidate[
                            "description"
                        ]
                    ),
                    "search_language": str(
                        candidate[
                            "search_language"
                        ]
                    ),
                    "search_rank": int(
                        candidate[
                            "search_rank"
                        ]
                    ),
                    "match_type": str(
                        candidate[
                            "match_type"
                        ]
                    ),
                    "match_text": str(
                        candidate[
                            "match_text"
                        ]
                    ),
                }
            )

    status_schema = {
        "entity_group": pl.String,
        "canonical_entity": pl.String,
        "canonical_entity_key": pl.String,
        "article_df": pl.Int64,
        "mention_count": pl.Int64,
        "status": pl.String,
        "candidate_count": pl.Int64,
    }

    candidate_schema = {
        "entity_group": pl.String,
        "canonical_entity": pl.String,
        "canonical_entity_key": pl.String,
        "article_df": pl.Int64,
        "mention_count": pl.Int64,
        "candidate_rank": pl.Int64,
        "qid": pl.String,
        "label": pl.String,
        "description": pl.String,
        "search_language": pl.String,
        "search_rank": pl.Int64,
        "match_type": pl.String,
        "match_text": pl.String,
    }

    if status_rows:
        status_df = (
            pl.DataFrame(
                status_rows,
                schema=status_schema,
            )
            .sort(
                [
                    "article_df",
                    "canonical_entity_key",
                ],
                descending=[
                    True,
                    False,
                ],
            )
        )
    else:
        status_df = pl.DataFrame(
            schema=status_schema
        )

    if candidate_rows:
        candidates_df = (
            pl.DataFrame(
                candidate_rows,
                schema=candidate_schema,
            )
            .sort(
                [
                    "article_df",
                    "canonical_entity_key",
                    "candidate_rank",
                ],
                descending=[
                    True,
                    False,
                    False,
                ],
            )
        )
    else:
        candidates_df = pl.DataFrame(
            schema=candidate_schema
        )

    return (
        status_df,
        candidates_df,
    )


# =============================================================================
# 10. Summary
# =============================================================================


def _write_summary(
    *,
    output_path: Path,
    universe_df: pl.DataFrame,
    status_df: pl.DataFrame,
    candidates_df: pl.DataFrame,
    groups: set[str] | None,
    languages: tuple[str, ...],
) -> None:
    searched_count = int(
        status_df.filter(
            pl.col("status").is_in(
                ["FOUND", "NO_CANDIDATE"]
            )
        ).height
    )

    found_count = int(
        status_df.filter(
            pl.col("status")
            == "FOUND"
        ).height
    )

    no_candidate_count = int(
        status_df.filter(
            pl.col("status")
            == "NO_CANDIDATE"
        ).height
    )

    remaining_count = max(
        int(universe_df.height)
        - searched_count,
        0,
    )

    lines = [
        "=" * 80,
        "Wikidata Candidate Build Summary",
        "=" * 80,
        "",
        "fit_split=train_used_articles_only",
        "source_representation=normalize_v2_canonical_entity",
        f"groups={sorted(groups) if groups is not None else 'ALL'}",
        f"languages={list(languages)}",
        "",
        f"train_entity_universe_count={universe_df.height}",
        f"cached_searched_entity_count={searched_count}",
        f"remaining_entity_count={remaining_count}",
        f"found_candidate_entity_count={found_count}",
        f"no_candidate_entity_count={no_candidate_count}",
        f"candidate_row_count={candidates_df.height}",
        "",
        "주의:",
        "- 이 단계는 후보 생성(candidate generation)만 한다.",
        "- candidate_rank=1을 정답 QID로 간주하지 않는다.",
        "- 다음 단계에서 Wikidata TYPE(P31 등)과 article context를 이용해 후보를 줄인다.",
        "- QID를 확정하지 못한 Entity는 UNLINKED로 유지한다.",
    ]

    output_path.write_text(
        "\n".join(lines)
        + "\n",
        encoding="utf-8",
    )


# =============================================================================
# 11. Main Builder
# =============================================================================


def build_wikidata_candidates(
    *,
    article_entities_path: Path,
    output_dir: Path,
    limit: int | None,
    groups: set[str] | None,
    languages: tuple[str, ...],
    request_sleep_seconds: float,
) -> dict[str, Any]:

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cache_path = (
        output_dir
        / CACHE_FILENAME
    )

    status_path = (
        output_dir
        / STATUS_FILENAME
    )

    candidates_path = (
        output_dir
        / CANDIDATES_FILENAME
    )

    summary_path = (
        output_dir
        / SUMMARY_FILENAME
    )

    # -------------------------------------------------------------------------
    # STEP 1. 전체 Train Entity universe
    # -------------------------------------------------------------------------
    universe_df = (
        _load_train_entity_universe(
            article_entities_path,
            groups=groups,
        )
    )

    # -------------------------------------------------------------------------
    # STEP 2. 이전 cache 로드
    # -------------------------------------------------------------------------
    cache = (
        _load_cache(
            cache_path
        )
    )

    # FOUND / NO_CANDIDATE만 "완료"로 본다.
    # ERROR는 다음 실행에서 자동 재시도해야 한다.
    already_done = {
        key
        for key, value in cache.items()
        if str(value.get("status", ""))
        in {"FOUND", "NO_CANDIDATE"}
    }

    pending_df = (
        universe_df.filter(
            ~pl.col(
                "canonical_entity_key"
            ).is_in(
                list(
                    already_done
                )
            )
        )
    )

    if limit is not None:
        pending_df = (
            pending_df.head(
                limit
            )
        )

    print(
        "=" * 100
    )
    print(
        "Wikidata Candidate Build 시작"
    )
    print(
        "=" * 100
    )

    print(
        f"train_entity_universe_count = {universe_df.height}"
    )
    print(
        f"already_cached_count = {len(already_done)}"
    )
    print(
        f"this_run_target_count = {pending_df.height}"
    )
    print(
        f"languages = {languages}"
    )
    print(
        f"request_sleep_seconds = {request_sleep_seconds}"
    )
    print()

    # -------------------------------------------------------------------------
    # STEP 3. 미처리 Entity 검색
    # -------------------------------------------------------------------------
    total_this_run = int(
        pending_df.height
    )

    for index, row in enumerate(
        pending_df.iter_rows(
            named=True
        ),
        start=1,
    ):
        entity_group = str(
            row[
                "entity_group"
            ]
        )
        surface = str(
            row[
                "canonical_entity"
            ]
        )
        key = str(
            row[
                "canonical_entity_key"
            ]
        )

        print(
            f"[{index}/{total_this_run}] "
            f"{key} "
            f"(df={row['article_df']})"
        )

        try:
            result = (
                _search_entity(
                    entity_group=(
                        entity_group
                    ),
                    surface=surface,
                    canonical_entity_key=key,
                    article_df=int(
                        row[
                            "article_df"
                        ]
                    ),
                    mention_count=int(
                        row[
                            "mention_count"
                        ]
                    ),
                    languages=languages,
                    request_sleep_seconds=(
                        request_sleep_seconds
                    ),
                )
            )

        except Exception as exc:
            # 한 Entity 오류 때문에 전체 작업을 잃지 않도록
            # ERROR 상태 자체도 cache에 기록한다.
            print(
                "    ERROR: "
                f"{type(exc).__name__}: {exc}"
            )

            result = {
                "entity_group": (
                    entity_group
                ),
                "canonical_entity": (
                    surface
                ),
                "canonical_entity_key": (
                    key
                ),
                "article_df": int(
                    row[
                        "article_df"
                    ]
                ),
                "mention_count": int(
                    row[
                        "mention_count"
                    ]
                ),
                "status": "ERROR",
                "candidate_count": 0,
                "candidates": [],
                "error_type": (
                    type(exc).__name__
                ),
                "error_message": str(
                    exc
                ),
            }

        _append_cache_row(
            cache_path,
            result,
        )

        cache[
            key
        ] = result

        print(
            f"    status={result['status']}, "
            f"candidate_count={result['candidate_count']}"
        )

    # -------------------------------------------------------------------------
    # STEP 4. cache 전체를 Parquet으로 export
    # -------------------------------------------------------------------------
    (
        status_df,
        candidates_df,
    ) = (
        _build_output_frames(
            cache
        )
    )

    status_df.write_parquet(
        status_path,
        compression="zstd",
    )

    candidates_df.write_parquet(
        candidates_path,
        compression="zstd",
    )

    _write_summary(
        output_path=summary_path,
        universe_df=universe_df,
        status_df=status_df,
        candidates_df=candidates_df,
        groups=groups,
        languages=languages,
    )

    error_count = int(
        status_df.filter(
            pl.col("status")
            == "ERROR"
        ).height
    )

    found_count = int(
        status_df.filter(
            pl.col("status")
            == "FOUND"
        ).height
    )

    no_candidate_count = int(
        status_df.filter(
            pl.col("status")
            == "NO_CANDIDATE"
        ).height
    )

    completed_count = int(
        status_df.filter(
            pl.col("status").is_in(
                ["FOUND", "NO_CANDIDATE"]
            )
        ).height
    )

    remaining_count = max(
        int(universe_df.height)
        - completed_count,
        0,
    )

    return {
        "status": "SUCCESS",
        "train_entity_universe_count": int(
            universe_df.height
        ),
        "cached_searched_entity_count": (
            completed_count
        ),
        "remaining_entity_count": (
            remaining_count
        ),
        "found_candidate_entity_count": (
            found_count
        ),
        "no_candidate_entity_count": (
            no_candidate_count
        ),
        "error_entity_count": (
            error_count
        ),
        "candidate_row_count": int(
            candidates_df.height
        ),
        "cache_path": str(
            cache_path
        ),
        "status_path": str(
            status_path
        ),
        "candidates_path": str(
            candidates_path
        ),
        "summary_path": str(
            summary_path
        ),
    }


# =============================================================================
# 12. CLI
# =============================================================================


def _parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "normalize_v2 Train canonical entity 전체의 "
            "Wikidata 검색 후보를 cache합니다."
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
        "--output-dir",
        type=Path,
        default=(
            DEFAULT_OUTPUT_DIR
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "이번 실행에서 새로 검색할 최대 Entity 수. "
            "예: --limit 100. "
            "미지정 시 아직 cache되지 않은 전체 Entity."
        ),
    )

    parser.add_argument(
        "--groups",
        nargs="*",
        default=None,
        help=(
            "검색할 Entity TYPE. "
            "예: --groups PER ORG LOC. "
            "미지정 시 전체 TYPE."
        ),
    )

    parser.add_argument(
        "--languages",
        nargs="+",
        default=list(
            DEFAULT_LANGUAGES
        ),
        help=(
            "Wikidata 검색 언어. 기본: da en"
        ),
    )

    parser.add_argument(
        "--sleep",
        type=float,
        default=(
            DEFAULT_REQUEST_SLEEP_SECONDS
        ),
        help=(
            "성공한 API 요청 사이 기본 대기 시간(초). "
            "기본 1.5"
        ),
    )

    args = parser.parse_args()

    if (
        args.limit is not None
        and args.limit <= 0
    ):
        parser.error(
            "--limit은 1 이상이어야 합니다."
        )

    if args.sleep < 0:
        parser.error(
            "--sleep은 0 이상이어야 합니다."
        )

    return args


def main() -> None:

    args = (
        _parse_args()
    )

    groups = (
        set(args.groups)
        if args.groups
        else None
    )

    languages = tuple(
        args.languages
    )

    result = (
        build_wikidata_candidates(
            article_entities_path=(
                args.article_entities
            ),
            output_dir=(
                args.output_dir
            ),
            limit=(
                args.limit
            ),
            groups=groups,
            languages=languages,
            request_sleep_seconds=(
                args.sleep
            ),
        )
    )

    print()
    print(
        "=" * 100
    )
    print(
        "Wikidata Candidate Build 완료"
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