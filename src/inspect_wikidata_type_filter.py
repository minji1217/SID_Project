from __future__ import annotations

"""
Inspect Wikidata Type Filter
============================

목적
----
build_wikidata_candidates.py가 만든 Wikidata 후보들에 대해
"NER TYPE과 Wikidata TYPE이 서로 말이 되는가?"를 진단한다.

예:
    PER::vladimir putin
        후보 Q7747 Vladimir Putin
            P31 = Q5 (human)
            -> PER과 잘 맞음 -> TYPE_MATCH

    PER::vladimir putin
        후보 Q122033860 book edition
            P31 = book edition 계열
            -> PER과 맞지 않음 -> TYPE_MISMATCH

중요
----
이 스크립트는 아직 최종 Entity Linking을 하지 않는다.

즉:
- QID를 확정하지 않음
- candidate_rank=1을 정답이라고 보지 않음
- GPT 사용 안 함
- entity_processing.py 수정 안 함
- Event clustering 수정 안 함

이번 단계의 목적은 오직:
    Wikidata 검색 후보
        ↓
    P31(instance of)
        ↓
    P279(subclass of) 상위 클래스 추적
        ↓
    NER TYPE과 호환되는 후보인지 진단

이다.

입력
----
data/output/experiments/wikidata_candidates/
    wikidata_candidates.parquet
    wikidata_entity_status.parquet

출력
----
data/output/experiments/wikidata_type_filter/

1. wikidata_type_metadata_cache.jsonl
   - QID별 Wikidata P31/P279 metadata cache

2. wikidata_qid_metadata.parquet
   - 이번까지 수집한 QID metadata

3. wikidata_candidates_with_types.parquet
   - 원래 후보 + P31 + type-filter 판정

4. wikidata_type_matched_candidates.parquet
   - TYPE_MATCH 후보만 모은 진단 파일

5. wikidata_entity_type_filter_status.parquet
   - canonical entity별 후보 감소 현황

6. wikidata_type_filter_summary.txt
   - 전체 요약

실행
----
프로젝트 루트에서:

    python -m src.inspect_wikidata_type_filter

현재 100개 probe 결과에 바로 적용할 수 있다.
나중에 wikidata_candidates.parquet가 전체 Train 결과로 커져도
같은 스크립트를 다시 실행할 수 있다.
"""

import argparse
import json
import os
import time
from collections import deque
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import polars as pl


# =============================================================================
# 1. 경로
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CANDIDATES_PATH = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "experiments"
    / "wikidata_candidates"
    / "wikidata_candidates.parquet"
)

DEFAULT_ENTITY_STATUS_PATH = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "experiments"
    / "wikidata_candidates"
    / "wikidata_entity_status.parquet"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "experiments"
    / "wikidata_type_filter"
)

METADATA_CACHE_FILENAME = (
    "wikidata_type_metadata_cache.jsonl"
)

QID_METADATA_FILENAME = (
    "wikidata_qid_metadata.parquet"
)

CANDIDATES_WITH_TYPES_FILENAME = (
    "wikidata_candidates_with_types.parquet"
)

TYPE_MATCHED_FILENAME = (
    "wikidata_type_matched_candidates.parquet"
)

ENTITY_STATUS_FILENAME = (
    "wikidata_entity_type_filter_status.parquet"
)

SUMMARY_FILENAME = (
    "wikidata_type_filter_summary.txt"
)


# =============================================================================
# 2. Wikidata API 설정
# =============================================================================

WIKIDATA_API = "https://www.wikidata.org/w/api.php"

USER_AGENT = os.environ.get(
    "WIKIDATA_USER_AGENT",
    (
        "SID-Project-Wikidata-Type-Inspection/0.1 "
        "(undergraduate research prototype)"
    ),
)

# wbgetentities는 여러 QID를 한 요청에 묶을 수 있으므로
# 검색 단계처럼 QID 하나씩 호출하지 않는다.
BATCH_SIZE = 40

REQUEST_SLEEP_SECONDS = 1.5

MAX_RETRIES = 5

# P31 class -> P279 parent -> ... 를 얼마나 위까지 따라갈지.
# 너무 깊게 무한히 올라가지 않도록 diagnostic 단계에서는 제한한다.
MAX_ANCESTOR_DEPTH = 8


# =============================================================================
# 3. NER TYPE별 Wikidata root class
# =============================================================================
#
# 이 root들은 "자동 정답 판정"용이 아니라
# 후보의 ontology가 NER TYPE과 대체로 맞는지 보는 진단 기준이다.
#
# PER:
#   human Q5
#
# ORG:
#   organization Q43229
#
# LOC:
#   위치 표현은 도시/국가/행정구역/지리 지형 등 다양하므로
#   여러 넓은 root를 허용한다.
#
# EVENT:
#   occurrence Q1190554를 진단 root로 쓰지만,
#   EB-NeRD EVENT에는 Champions League 같은 반복 competition 이름도
#   들어갈 수 있으므로 "hard reject" 기준으로 쓰지 않는다.
#
# PROD:
#   product Q2424752
#
# MISC:
#   범위가 너무 넓어서 Wikidata ontology만으로 hard filtering하지 않는다.
# =============================================================================

TYPE_ROOTS: dict[str, dict[str, str]] = {
    "PER": {
        "Q5": "human",
    },
    "ORG": {
        "Q43229": "organization",
    },
    "LOC": {
        "Q2221906": "geographic location",
        "Q618123": "geographical feature",
        "Q56061": "administrative territorial entity",
        "Q486972": "human settlement",
        "Q6256": "country",
    },
    "EVENT": {
        "Q1190554": "occurrence",
    },
    "PROD": {
        "Q2424752": "product",
    },
}

# STRICT:
#   MATCH/MISMATCH를 비교적 강한 진단 신호로 사용 가능.
#
# CONSERVATIVE:
#   MATCH는 좋은 신호지만 MISMATCH를 곧바로 제거 근거로 쓰지 않음.
#
# DIAGNOSTIC_ONLY:
#   ontology 차이 가능성이 커서 참고만 함.
#
# UNCHECKED:
#   자동 TYPE 판정 자체를 하지 않음.
TYPE_POLICIES = {
    "PER": "STRICT",
    "ORG": "CONSERVATIVE",
    "LOC": "CONSERVATIVE",
    "EVENT": "DIAGNOSTIC_ONLY",
    "PROD": "CONSERVATIVE",
    "MISC": "UNCHECKED",
}


# =============================================================================
# 4. 입력 컬럼
# =============================================================================

REQUIRED_CANDIDATE_COLUMNS = {
    "entity_group",
    "canonical_entity",
    "canonical_entity_key",
    "article_df",
    "mention_count",
    "candidate_rank",
    "qid",
    "label",
    "description",
    "search_language",
    "search_rank",
}

REQUIRED_STATUS_COLUMNS = {
    "entity_group",
    "canonical_entity",
    "canonical_entity_key",
    "article_df",
    "mention_count",
    "status",
    "candidate_count",
}


# =============================================================================
# 5. 기본 검증
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
# 6. Wikidata claim에서 item QID 뽑기
# =============================================================================


def _extract_item_qids_from_claims(
    claims: dict[str, Any],
    property_id: str,
) -> list[str]:
    """
    Wikidata claims에서
    P31 또는 P279처럼 값이 item(QID)인 statement만 추출한다.

    예:
        claims["P31"]
            ↓
        mainsnak
            ↓
        datavalue
            ↓
        value.id = "Q5"

    반환:
        ["Q5", ...]
    """

    result: list[str] = []

    for claim in claims.get(
        property_id,
        [],
    ):
        mainsnak = (
            claim.get("mainsnak")
            or {}
        )

        datavalue = (
            mainsnak.get("datavalue")
            or {}
        )

        value = (
            datavalue.get("value")
            or {}
        )

        qid = value.get(
            "id"
        )

        if (
            isinstance(qid, str)
            and qid.startswith("Q")
        ):
            result.append(
                qid
            )

    return sorted(
        set(result)
    )


# =============================================================================
# 7. label / description 선택
# =============================================================================


def _language_value(
    values: dict[str, Any],
    language: str,
) -> str:
    item = (
        values.get(language)
        or {}
    )

    return str(
        item.get(
            "value",
            "",
        )
        or ""
    )


# =============================================================================
# 8. wbgetentities batch 요청
# =============================================================================


def _fetch_entities_batch(
    qids: list[str],
) -> dict[str, dict[str, Any]]:
    """
    여러 QID를 한 번의 wbgetentities 요청으로 조회한다.

    필요한 정보:
    - labels
    - descriptions
    - claims
        P31 = instance of
        P279 = subclass of

    429/503이면 자동 retry한다.
    """

    if not qids:
        return {}

    params = {
        "action": "wbgetentities",
        "ids": "|".join(qids),
        "props": "labels|descriptions|claims",
        "languages": "da|en",
        "languagefallback": "1",
        "format": "json",
        "maxlag": 5,
    }

    url = (
        WIKIDATA_API
        + "?"
        + urlencode(params)
    )

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
                timeout=40,
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
                    f"batch_size={len(qids)}"
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

            wait_seconds = max(
                wait_seconds,
                REQUEST_SLEEP_SECONDS,
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
                f"batch_size={len(qids)}, "
                f"reason={exc.reason}"
            ) from exc

    else:
        raise RuntimeError(
            "Wikidata API batch 재시도 횟수를 초과했습니다. "
            f"batch_size={len(qids)}"
        )

    if raw is None:
        raise RuntimeError(
            "Wikidata 응답을 받지 못했습니다."
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
            "Wikidata wbgetentities JSON 파싱 실패"
        ) from exc

    entities = (
        data.get("entities")
        or {}
    )

    parsed: dict[
        str,
        dict[str, Any],
    ] = {}

    for qid in qids:
        entity = (
            entities.get(qid)
            or {}
        )

        missing = bool(
            entity.get(
                "missing",
                False,
            )
        )

        claims = (
            entity.get("claims")
            or {}
        )

        labels = (
            entity.get("labels")
            or {}
        )

        descriptions = (
            entity.get(
                "descriptions"
            )
            or {}
        )

        parsed[qid] = {
            "qid": qid,
            "missing": missing,
            "label_da": _language_value(
                labels,
                "da",
            ),
            "label_en": _language_value(
                labels,
                "en",
            ),
            "description_da": _language_value(
                descriptions,
                "da",
            ),
            "description_en": _language_value(
                descriptions,
                "en",
            ),
            "p31_qids": (
                _extract_item_qids_from_claims(
                    claims,
                    "P31",
                )
            ),
            "p279_qids": (
                _extract_item_qids_from_claims(
                    claims,
                    "P279",
                )
            ),
        }

    time.sleep(
        REQUEST_SLEEP_SECONDS
    )

    return parsed


# =============================================================================
# 9. metadata cache
# =============================================================================


def _load_metadata_cache(
    cache_path: Path,
) -> dict[str, dict[str, Any]]:
    """
    QID metadata JSONL cache를 읽는다.
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
                    f"metadata cache line {line_number} "
                    "JSON 파싱 실패 → skip"
                )
                continue

            qid = str(
                row.get(
                    "qid",
                    "",
                )
            )

            if not qid:
                continue

            cache[qid] = row

    return cache


def _append_metadata_cache_rows(
    cache_path: Path,
    rows: list[
        dict[str, Any]
    ],
) -> None:
    """
    새로 조회한 QID metadata를 JSONL에 append한다.
    """

    if not rows:
        return

    cache_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with cache_path.open(
        "a",
        encoding="utf-8",
    ) as f:
        for row in rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

        f.flush()


# =============================================================================
# 10. 필요한 QID batch fetch
# =============================================================================


def _ensure_metadata(
    qids: set[str],
    *,
    cache: dict[
        str,
        dict[str, Any],
    ],
    cache_path: Path,
) -> int:
    """
    qids 중 아직 cache에 없는 것만 batch로 조회한다.

    반환:
        이번 호출에서 새로 fetch한 QID 수
    """

    missing_qids = sorted(
        qid
        for qid in qids
        if qid not in cache
    )

    if not missing_qids:
        return 0

    fetched_count = 0

    for start in range(
        0,
        len(missing_qids),
        BATCH_SIZE,
    ):
        batch = (
            missing_qids[
                start:
                start + BATCH_SIZE
            ]
        )

        print(
            "  metadata fetch "
            f"{start + 1}-"
            f"{min(start + len(batch), len(missing_qids))}"
            f"/{len(missing_qids)}"
        )

        fetched = (
            _fetch_entities_batch(
                batch
            )
        )

        new_rows = list(
            fetched.values()
        )

        _append_metadata_cache_rows(
            cache_path,
            new_rows,
        )

        cache.update(
            fetched
        )

        fetched_count += len(
            fetched
        )

    return fetched_count


# =============================================================================
# 11. P31 class의 P279 ancestry 확보
# =============================================================================


def _expand_class_hierarchy(
    initial_class_qids: set[str],
    *,
    cache: dict[
        str,
        dict[str, Any],
    ],
    cache_path: Path,
    max_depth: int,
) -> dict[str, set[str]]:
    """
    candidate의 P31 class들이 어떤 상위 class(P279)에 연결되는지
    최대 max_depth까지 metadata를 확보한다.

    예:
        association football club
            P279 -> football club
            P279 -> sports club
            P279 -> organization
                         ↑
                      Q43229

    이 구조를 알아야 ORG 후보를
    "직접 P31=organization이 아니어도"
    organization 계열로 볼 수 있다.

    반환:
        class_qid -> ancestor closure(set)
    """

    current_frontier = set(
        initial_class_qids
    )

    visited: set[str] = set()

    for depth in range(
        max_depth + 1
    ):
        frontier = (
            current_frontier
            - visited
        )

        if not frontier:
            break

        print(
            f"Hierarchy depth {depth}: "
            f"class_count={len(frontier)}"
        )

        _ensure_metadata(
            frontier,
            cache=cache,
            cache_path=cache_path,
        )

        visited.update(
            frontier
        )

        next_frontier: set[str] = set()

        for qid in frontier:
            row = cache.get(
                qid,
                {},
            )

            for parent in row.get(
                "p279_qids",
                [],
            ):
                if parent not in visited:
                    next_frontier.add(
                        parent
                    )

        current_frontier = (
            next_frontier
        )

    # -------------------------------------------------------------------------
    # 각 class별 ancestor closure 계산
    # -------------------------------------------------------------------------
    memo: dict[
        str,
        set[str],
    ] = {}

    def collect(
        start_qid: str,
    ) -> set[str]:
        if start_qid in memo:
            return set(
                memo[start_qid]
            )

        ancestors: set[str] = {
            start_qid
        }

        queue: deque[
            tuple[str, int]
        ] = deque(
            [
                (
                    start_qid,
                    0,
                )
            ]
        )

        seen: set[str] = {
            start_qid
        }

        while queue:
            current, depth = (
                queue.popleft()
            )

            if depth >= max_depth:
                continue

            row = cache.get(
                current,
                {},
            )

            parents = row.get(
                "p279_qids",
                [],
            )

            for parent in parents:
                ancestors.add(
                    parent
                )

                if parent not in seen:
                    seen.add(
                        parent
                    )
                    queue.append(
                        (
                            parent,
                            depth + 1,
                        )
                    )

        memo[start_qid] = set(
            ancestors
        )

        return ancestors

    return {
        class_qid: collect(
            class_qid
        )
        for class_qid in initial_class_qids
    }


# =============================================================================
# 12. QID label helper
# =============================================================================


def _best_label(
    qid: str,
    cache: dict[
        str,
        dict[str, Any],
    ],
) -> str:
    row = cache.get(
        qid,
        {},
    )

    return (
        str(
            row.get(
                "label_en",
                "",
            )
            or ""
        )
        or str(
            row.get(
                "label_da",
                "",
            )
            or ""
        )
        or qid
    )


# =============================================================================
# 13. 한 후보 TYPE 판정
# =============================================================================


def _classify_candidate_type(
    *,
    entity_group: str,
    candidate_qid: str,
    metadata_cache: dict[
        str,
        dict[str, Any],
    ],
    class_ancestors: dict[
        str,
        set[str],
    ],
) -> dict[str, Any]:
    """
    NER TYPE과 Wikidata P31/P279를 비교한다.

    결과:
        TYPE_MATCH
        TYPE_MISMATCH
        TYPE_UNKNOWN
        TYPE_UNCHECKED

    주의:
    TYPE_MISMATCH라고 바로 후보를 삭제하면 안 된다.
    이 파일은 inspection용이다.
    """

    policy = TYPE_POLICIES.get(
        entity_group,
        "UNCHECKED",
    )

    if (
        policy == "UNCHECKED"
        or entity_group not in TYPE_ROOTS
    ):
        return {
            "type_filter_policy": policy,
            "type_filter_status": (
                "TYPE_UNCHECKED"
            ),
            "type_match_root_qids": [],
            "type_match_root_labels": [],
        }

    metadata = metadata_cache.get(
        candidate_qid,
        {},
    )

    p31_qids = list(
        metadata.get(
            "p31_qids",
            [],
        )
    )

    if not p31_qids:
        return {
            "type_filter_policy": policy,
            "type_filter_status": (
                "TYPE_UNKNOWN"
            ),
            "type_match_root_qids": [],
            "type_match_root_labels": [],
        }

    allowed_roots = TYPE_ROOTS[
        entity_group
    ]

    matched_roots: set[str] = set()

    for p31_qid in p31_qids:
        ancestry = (
            class_ancestors.get(
                p31_qid,
                {
                    p31_qid
                },
            )
        )

        for root_qid in (
            allowed_roots
        ):
            if root_qid in ancestry:
                matched_roots.add(
                    root_qid
                )

    if matched_roots:
        status = "TYPE_MATCH"
    else:
        status = "TYPE_MISMATCH"

    matched_root_qids = sorted(
        matched_roots
    )

    return {
        "type_filter_policy": policy,
        "type_filter_status": (
            status
        ),
        "type_match_root_qids": (
            matched_root_qids
        ),
        "type_match_root_labels": [
            TYPE_ROOTS[
                entity_group
            ][qid]
            for qid in matched_root_qids
        ],
    }


# =============================================================================
# 14. 후보 DataFrame에 TYPE 정보 추가
# =============================================================================


def _build_typed_candidates(
    candidates_df: pl.DataFrame,
    *,
    metadata_cache: dict[
        str,
        dict[str, Any],
    ],
    class_ancestors: dict[
        str,
        set[str],
    ],
) -> pl.DataFrame:
    rows: list[
        dict[str, Any]
    ] = []

    for row in candidates_df.iter_rows(
        named=True
    ):
        qid = str(
            row[
                "qid"
            ]
        )

        entity_group = str(
            row[
                "entity_group"
            ]
        )

        metadata = metadata_cache.get(
            qid,
            {},
        )

        p31_qids = list(
            metadata.get(
                "p31_qids",
                [],
            )
        )

        type_result = (
            _classify_candidate_type(
                entity_group=(
                    entity_group
                ),
                candidate_qid=qid,
                metadata_cache=(
                    metadata_cache
                ),
                class_ancestors=(
                    class_ancestors
                ),
            )
        )

        p31_labels = [
            _best_label(
                p31_qid,
                metadata_cache,
            )
            for p31_qid in p31_qids
        ]

        rows.append(
            {
                **row,

                "wikidata_label_da": str(
                    metadata.get(
                        "label_da",
                        "",
                    )
                    or ""
                ),
                "wikidata_label_en": str(
                    metadata.get(
                        "label_en",
                        "",
                    )
                    or ""
                ),

                "wikidata_description_da": str(
                    metadata.get(
                        "description_da",
                        "",
                    )
                    or ""
                ),
                "wikidata_description_en": str(
                    metadata.get(
                        "description_en",
                        "",
                    )
                    or ""
                ),

                "p31_qids": p31_qids,
                "p31_labels": p31_labels,

                **type_result,
            }
        )

    if not rows:
        return pl.DataFrame()

    return (
        pl.DataFrame(
            rows,
            infer_schema_length=None,
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


# =============================================================================
# 15. Entity 단위 후보 감소 통계
# =============================================================================


def _build_entity_filter_status(
    entity_status_df: pl.DataFrame,
    typed_candidates_df: pl.DataFrame,
) -> pl.DataFrame:
    """
    canonical entity 하나마다:

        원래 후보 몇 개?
        TYPE_MATCH 몇 개?
        TYPE_MISMATCH 몇 개?
        TYPE_UNKNOWN 몇 개?

    를 만든다.

    예:
        PER::vladimir putin

        original = 5
        match    = 2
        mismatch = 3

        -> MULTIPLE_TYPE_MATCH

    여기서 ONE_TYPE_MATCH라도 아직 자동 QID 확정은 아니다.
    """

    count_rows: list[
        dict[str, Any]
    ] = []

    typed_by_key: dict[
        str,
        list[dict[str, Any]],
    ] = {}

    if typed_candidates_df.height:
        for row in typed_candidates_df.iter_rows(
            named=True
        ):
            key = str(
                row[
                    "canonical_entity_key"
                ]
            )

            typed_by_key.setdefault(
                key,
                [],
            ).append(
                row
            )

    for status_row in entity_status_df.iter_rows(
        named=True
    ):
        key = str(
            status_row[
                "canonical_entity_key"
            ]
        )

        candidate_rows = (
            typed_by_key.get(
                key,
                [],
            )
        )

        match_rows = [
            row
            for row in candidate_rows
            if row[
                "type_filter_status"
            ]
            == "TYPE_MATCH"
        ]

        mismatch_count = sum(
            1
            for row in candidate_rows
            if row[
                "type_filter_status"
            ]
            == "TYPE_MISMATCH"
        )

        unknown_count = sum(
            1
            for row in candidate_rows
            if row[
                "type_filter_status"
            ]
            == "TYPE_UNKNOWN"
        )

        unchecked_count = sum(
            1
            for row in candidate_rows
            if row[
                "type_filter_status"
            ]
            == "TYPE_UNCHECKED"
        )

        group = str(
            status_row[
                "entity_group"
            ]
        )

        policy = TYPE_POLICIES.get(
            group,
            "UNCHECKED",
        )

        source_status = str(
            status_row[
                "status"
            ]
        )

        if source_status != "FOUND":
            diagnostic_status = (
                source_status
            )

        elif policy == "UNCHECKED":
            diagnostic_status = (
                "UNCHECKED_GROUP"
            )

        elif len(match_rows) == 1:
            diagnostic_status = (
                "ONE_TYPE_MATCH"
            )

        elif len(match_rows) > 1:
            diagnostic_status = (
                "MULTIPLE_TYPE_MATCH"
            )

        elif unknown_count > 0:
            diagnostic_status = (
                "NO_MATCH_WITH_UNKNOWN"
            )

        else:
            diagnostic_status = (
                "NO_TYPE_MATCH"
            )

        one_match_qid = (
            str(
                match_rows[0]["qid"]
            )
            if len(match_rows) == 1
            else None
        )

        one_match_label = (
            str(
                match_rows[0]["label"]
            )
            if len(match_rows) == 1
            else None
        )

        count_rows.append(
            {
                "entity_group": group,
                "canonical_entity": str(
                    status_row[
                        "canonical_entity"
                    ]
                ),
                "canonical_entity_key": (
                    key
                ),
                "article_df": int(
                    status_row[
                        "article_df"
                    ]
                ),
                "mention_count": int(
                    status_row[
                        "mention_count"
                    ]
                ),
                "source_search_status": (
                    source_status
                ),
                "original_candidate_count": int(
                    status_row[
                        "candidate_count"
                    ]
                ),
                "type_filter_policy": policy,
                "type_match_count": int(
                    len(match_rows)
                ),
                "type_mismatch_count": int(
                    mismatch_count
                ),
                "type_unknown_count": int(
                    unknown_count
                ),
                "type_unchecked_count": int(
                    unchecked_count
                ),
                "diagnostic_status": (
                    diagnostic_status
                ),
                "single_type_match_qid": (
                    one_match_qid
                ),
                "single_type_match_label": (
                    one_match_label
                ),
            }
        )

    if not count_rows:
        return pl.DataFrame()

    return (
        pl.DataFrame(
            count_rows,
            infer_schema_length=None,
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


# =============================================================================
# 16. Metadata parquet
# =============================================================================


def _build_metadata_df(
    metadata_cache: dict[
        str,
        dict[str, Any],
    ],
) -> pl.DataFrame:
    rows = []

    for qid in sorted(
        metadata_cache
    ):
        row = metadata_cache[
            qid
        ]

        rows.append(
            {
                "qid": qid,
                "missing": bool(
                    row.get(
                        "missing",
                        False,
                    )
                ),
                "label_da": str(
                    row.get(
                        "label_da",
                        "",
                    )
                    or ""
                ),
                "label_en": str(
                    row.get(
                        "label_en",
                        "",
                    )
                    or ""
                ),
                "description_da": str(
                    row.get(
                        "description_da",
                        "",
                    )
                    or ""
                ),
                "description_en": str(
                    row.get(
                        "description_en",
                        "",
                    )
                    or ""
                ),
                "p31_qids": list(
                    row.get(
                        "p31_qids",
                        [],
                    )
                ),
                "p279_qids": list(
                    row.get(
                        "p279_qids",
                        [],
                    )
                ),
            }
        )

    if not rows:
        return pl.DataFrame()

    return pl.DataFrame(
        rows,
        infer_schema_length=None,
    )


# =============================================================================
# 17. Summary
# =============================================================================


def _write_summary(
    *,
    output_path: Path,
    candidates_df: pl.DataFrame,
    typed_candidates_df: pl.DataFrame,
    entity_filter_df: pl.DataFrame,
    candidate_qid_count: int,
    metadata_cache_count: int,
) -> None:
    def count_entity_status(
        status: str,
    ) -> int:
        if not entity_filter_df.height:
            return 0

        return int(
            entity_filter_df.filter(
                pl.col(
                    "diagnostic_status"
                )
                == status
            ).height
        )

    lines = [
        "=" * 88,
        "Wikidata Type Filter Inspection",
        "=" * 88,
        "",
        "주의:",
        "- 이 결과는 diagnostic이다.",
        "- ONE_TYPE_MATCH라고 해서 아직 QID를 자동 확정하지 않는다.",
        "- TYPE_MISMATCH도 특히 LOC/EVENT/PROD에서는 즉시 삭제 근거로 쓰지 않는다.",
        "",
        f"input_candidate_row_count={candidates_df.height}",
        f"unique_candidate_qid_count={candidate_qid_count}",
        f"metadata_cache_qid_count={metadata_cache_count}",
        "",
        f"type_match_candidate_row_count="
        f"{typed_candidates_df.filter(pl.col('type_filter_status') == 'TYPE_MATCH').height if typed_candidates_df.height else 0}",
        f"type_mismatch_candidate_row_count="
        f"{typed_candidates_df.filter(pl.col('type_filter_status') == 'TYPE_MISMATCH').height if typed_candidates_df.height else 0}",
        f"type_unknown_candidate_row_count="
        f"{typed_candidates_df.filter(pl.col('type_filter_status') == 'TYPE_UNKNOWN').height if typed_candidates_df.height else 0}",
        f"type_unchecked_candidate_row_count="
        f"{typed_candidates_df.filter(pl.col('type_filter_status') == 'TYPE_UNCHECKED').height if typed_candidates_df.height else 0}",
        "",
        "Entity-level:",
        f"one_type_match_entity_count={count_entity_status('ONE_TYPE_MATCH')}",
        f"multiple_type_match_entity_count={count_entity_status('MULTIPLE_TYPE_MATCH')}",
        f"no_type_match_entity_count={count_entity_status('NO_TYPE_MATCH')}",
        f"no_match_with_unknown_entity_count={count_entity_status('NO_MATCH_WITH_UNKNOWN')}",
        f"unchecked_group_entity_count={count_entity_status('UNCHECKED_GROUP')}",
        f"no_candidate_entity_count={count_entity_status('NO_CANDIDATE')}",
        f"error_entity_count={count_entity_status('ERROR')}",
        "",
        "TYPE policy:",
        f"PER={TYPE_POLICIES['PER']} root=Q5(human)",
        f"ORG={TYPE_POLICIES['ORG']} root=Q43229(organization)",
        "LOC=CONSERVATIVE roots="
        "Q2221906/Q618123/Q56061/Q486972/Q6256",
        "EVENT=DIAGNOSTIC_ONLY root=Q1190554(occurrence)",
        "PROD=CONSERVATIVE root=Q2424752(product)",
        "MISC=UNCHECKED",
        "",
        "다음 단계 판단:",
        "- ONE_TYPE_MATCH가 얼마나 많은지 확인한다.",
        "- PER에서 TYPE filtering이 후보를 얼마나 잘 줄이는지 우선 본다.",
        "- ORG/LOC/PROD는 MATCH는 좋은 신호지만 MISMATCH를 바로 제거하지 않는다.",
        "- MULTIPLE_TYPE_MATCH만 article context/GPT가 필요한 후보가 될 가능성이 높다.",
        "- NO_TYPE_MATCH는 검색 recall/ontology mismatch 여부를 별도로 본다.",
    ]

    output_path.write_text(
        "\n".join(lines)
        + "\n",
        encoding="utf-8",
    )


# =============================================================================
# 18. 전체 실행
# =============================================================================


def inspect_wikidata_type_filter(
    *,
    candidates_path: Path,
    entity_status_path: Path,
    output_dir: Path,
    max_ancestor_depth: int,
) -> dict[str, Any]:
    """
    Wikidata candidate TYPE inspection 전체 실행.
    """

    # -------------------------------------------------------------------------
    # STEP 1. 입력
    # -------------------------------------------------------------------------
    _require_file(
        candidates_path,
        "Wikidata candidates",
    )

    _require_file(
        entity_status_path,
        "Wikidata entity status",
    )

    candidates_df = (
        pl.read_parquet(
            candidates_path
        )
    )

    entity_status_df = (
        pl.read_parquet(
            entity_status_path
        )
    )

    _require_columns(
        candidates_df,
        REQUIRED_CANDIDATE_COLUMNS,
        "wikidata_candidates.parquet",
    )

    _require_columns(
        entity_status_df,
        REQUIRED_STATUS_COLUMNS,
        "wikidata_entity_status.parquet",
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cache_path = (
        output_dir
        / METADATA_CACHE_FILENAME
    )

    metadata_path = (
        output_dir
        / QID_METADATA_FILENAME
    )

    typed_candidates_path = (
        output_dir
        / CANDIDATES_WITH_TYPES_FILENAME
    )

    matched_path = (
        output_dir
        / TYPE_MATCHED_FILENAME
    )

    entity_filter_path = (
        output_dir
        / ENTITY_STATUS_FILENAME
    )

    summary_path = (
        output_dir
        / SUMMARY_FILENAME
    )

    # -------------------------------------------------------------------------
    # STEP 2. metadata cache
    # -------------------------------------------------------------------------
    metadata_cache = (
        _load_metadata_cache(
            cache_path
        )
    )

    candidate_qids = set(
        str(qid)
        for qid in candidates_df[
            "qid"
        ].to_list()
        if qid
    )

    print(
        "=" * 100
    )
    print(
        "Wikidata Type Filter Inspection 시작"
    )
    print(
        "=" * 100
    )

    print(
        f"candidate_row_count = {candidates_df.height}"
    )
    print(
        f"unique_candidate_qid_count = {len(candidate_qids)}"
    )
    print(
        f"metadata_cache_before = {len(metadata_cache)}"
    )
    print()

    # -------------------------------------------------------------------------
    # STEP 3. 후보 QID 자체 metadata 확보
    # -------------------------------------------------------------------------
    print(
        "[STEP 1] Candidate QID의 P31 metadata 조회"
    )

    _ensure_metadata(
        candidate_qids,
        cache=metadata_cache,
        cache_path=cache_path,
    )

    # -------------------------------------------------------------------------
    # STEP 4. 모든 P31 class 수집
    # -------------------------------------------------------------------------
    initial_class_qids: set[str] = set()

    for qid in candidate_qids:
        row = metadata_cache.get(
            qid,
            {},
        )

        initial_class_qids.update(
            row.get(
                "p31_qids",
                [],
            )
        )

    # root 자체도 metadata에 넣어두면 label 확인이 쉬움
    for roots in TYPE_ROOTS.values():
        initial_class_qids.update(
            roots.keys()
        )

    print()
    print(
        "[STEP 2] P31 class의 P279 hierarchy 조회"
    )

    print(
        f"initial_class_qid_count = {len(initial_class_qids)}"
    )

    class_ancestors = (
        _expand_class_hierarchy(
            initial_class_qids,
            cache=metadata_cache,
            cache_path=cache_path,
            max_depth=(
                max_ancestor_depth
            ),
        )
    )

    # -------------------------------------------------------------------------
    # STEP 5. 후보별 TYPE 판정
    # -------------------------------------------------------------------------
    print()
    print(
        "[STEP 3] NER TYPE ↔ Wikidata TYPE 비교"
    )

    typed_candidates_df = (
        _build_typed_candidates(
            candidates_df,
            metadata_cache=(
                metadata_cache
            ),
            class_ancestors=(
                class_ancestors
            ),
        )
    )

    # -------------------------------------------------------------------------
    # STEP 6. TYPE_MATCH subset
    # -------------------------------------------------------------------------
    if typed_candidates_df.height:
        matched_df = (
            typed_candidates_df.filter(
                pl.col(
                    "type_filter_status"
                )
                == "TYPE_MATCH"
            )
        )
    else:
        matched_df = (
            typed_candidates_df
        )

    # -------------------------------------------------------------------------
    # STEP 7. Entity-level status
    # -------------------------------------------------------------------------
    entity_filter_df = (
        _build_entity_filter_status(
            entity_status_df,
            typed_candidates_df,
        )
    )

    # -------------------------------------------------------------------------
    # STEP 8. metadata parquet
    # -------------------------------------------------------------------------
    metadata_df = (
        _build_metadata_df(
            metadata_cache
        )
    )

    # -------------------------------------------------------------------------
    # STEP 9. 저장
    # -------------------------------------------------------------------------
    metadata_df.write_parquet(
        metadata_path,
        compression="zstd",
    )

    typed_candidates_df.write_parquet(
        typed_candidates_path,
        compression="zstd",
    )

    matched_df.write_parquet(
        matched_path,
        compression="zstd",
    )

    entity_filter_df.write_parquet(
        entity_filter_path,
        compression="zstd",
    )

    _write_summary(
        output_path=summary_path,
        candidates_df=candidates_df,
        typed_candidates_df=(
            typed_candidates_df
        ),
        entity_filter_df=(
            entity_filter_df
        ),
        candidate_qid_count=(
            len(candidate_qids)
        ),
        metadata_cache_count=(
            len(metadata_cache)
        ),
    )

    # -------------------------------------------------------------------------
    # STEP 10. 결과 숫자
    # -------------------------------------------------------------------------
    def candidate_count(
        status: str,
    ) -> int:
        if not typed_candidates_df.height:
            return 0

        return int(
            typed_candidates_df.filter(
                pl.col(
                    "type_filter_status"
                )
                == status
            ).height
        )

    def entity_count(
        status: str,
    ) -> int:
        if not entity_filter_df.height:
            return 0

        return int(
            entity_filter_df.filter(
                pl.col(
                    "diagnostic_status"
                )
                == status
            ).height
        )

    return {
        "status": "SUCCESS",

        "input_candidate_row_count": int(
            candidates_df.height
        ),

        "unique_candidate_qid_count": int(
            len(candidate_qids)
        ),

        "metadata_cache_qid_count": int(
            len(metadata_cache)
        ),

        "type_match_candidate_row_count": (
            candidate_count(
                "TYPE_MATCH"
            )
        ),

        "type_mismatch_candidate_row_count": (
            candidate_count(
                "TYPE_MISMATCH"
            )
        ),

        "type_unknown_candidate_row_count": (
            candidate_count(
                "TYPE_UNKNOWN"
            )
        ),

        "type_unchecked_candidate_row_count": (
            candidate_count(
                "TYPE_UNCHECKED"
            )
        ),

        "one_type_match_entity_count": (
            entity_count(
                "ONE_TYPE_MATCH"
            )
        ),

        "multiple_type_match_entity_count": (
            entity_count(
                "MULTIPLE_TYPE_MATCH"
            )
        ),

        "no_type_match_entity_count": (
            entity_count(
                "NO_TYPE_MATCH"
            )
        ),

        "no_match_with_unknown_entity_count": (
            entity_count(
                "NO_MATCH_WITH_UNKNOWN"
            )
        ),

        "unchecked_group_entity_count": (
            entity_count(
                "UNCHECKED_GROUP"
            )
        ),

        "no_candidate_entity_count": (
            entity_count(
                "NO_CANDIDATE"
            )
        ),

        "metadata_cache_path": str(
            cache_path
        ),

        "qid_metadata_path": str(
            metadata_path
        ),

        "typed_candidates_path": str(
            typed_candidates_path
        ),

        "matched_candidates_path": str(
            matched_path
        ),

        "entity_filter_status_path": str(
            entity_filter_path
        ),

        "summary_path": str(
            summary_path
        ),
    }


# =============================================================================
# 19. CLI
# =============================================================================


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Wikidata 후보 QID의 P31/P279를 조회하여 "
            "NER TYPE과 호환되는 후보인지 진단합니다."
        )
    )

    parser.add_argument(
        "--candidates",
        type=Path,
        default=(
            DEFAULT_CANDIDATES_PATH
        ),
    )

    parser.add_argument(
        "--entity-status",
        type=Path,
        default=(
            DEFAULT_ENTITY_STATUS_PATH
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
        "--max-ancestor-depth",
        type=int,
        default=(
            MAX_ANCESTOR_DEPTH
        ),
        help=(
            "P279 상위 class를 따라갈 최대 깊이. "
            f"기본={MAX_ANCESTOR_DEPTH}"
        ),
    )

    args = parser.parse_args()

    if args.max_ancestor_depth < 0:
        parser.error(
            "--max-ancestor-depth는 0 이상이어야 합니다."
        )

    return args


def main() -> None:
    args = (
        _parse_args()
    )

    result = (
        inspect_wikidata_type_filter(
            candidates_path=(
                args.candidates
            ),
            entity_status_path=(
                args.entity_status
            ),
            output_dir=(
                args.output_dir
            ),
            max_ancestor_depth=(
                args.max_ancestor_depth
            ),
        )
    )

    print()
    print(
        "=" * 100
    )
    print(
        "Wikidata Type Filter Inspection 완료"
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