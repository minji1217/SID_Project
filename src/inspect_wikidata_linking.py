from __future__ import annotations

"""
Wikidata Entity Linking - Candidate Search Probe
================================================

목적
----
normalize_v2 Entity를 실제 Wikidata에 연결하기 전에,
Wikidata 검색 API가 우리 Entity 이름에 대해
어떤 QID 후보를 반환하는지 소량 테스트한다.

현재 단계에서는:
- GPT 사용 안 함
- 실제 Entity Linking 적용 안 함
- entity_processing.py 수정 안 함
- Event clustering 수정 안 함

오직:
    entity surface
        ↓
    Wikidata 후보 검색
        ↓
    QID / label / description 확인

까지만 수행한다.

실행
----
프로젝트 루트에서:

    python -m src.inspect_wikidata_linking

출력
----
data/output/experiments/wikidata_linking_inspection/
    wikidata_candidate_probe.parquet
"""

import json
import time
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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "experiments"
    / "wikidata_linking_inspection"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "wikidata_candidate_probe.parquet"
)


# =============================================================================
# 2. Wikidata API 설정
# =============================================================================

WIKIDATA_API = "https://www.wikidata.org/w/api.php"

# 실제 프로젝트 공개 시에는 이메일 또는 프로젝트 URL을 넣는 것이 좋다.
USER_AGENT = (
    "SID-Project-Wikidata-Linking/0.1 "
    "(undergraduate research prototype)"
)

# 언어별 최대 후보 수
SEARCH_LIMIT = 5

# 요청 간 기본 대기 시간
# 0.2초는 너무 빨라서 429가 발생했으므로 느리게 설정
REQUEST_SLEEP_SECONDS = 1.5

# 429 / 503 발생 시 최대 재시도 횟수
MAX_RETRIES = 4


# =============================================================================
# 3. 소량 테스트 Entity
# =============================================================================

# 이미 어느 정도 정답을 예상할 수 있는 Entity들로 probe.
# full name과 short form을 둘 다 넣어서 같은 QID 후보가 나오는지 확인한다.
TEST_ENTITIES = [
    ("PER", "vladimir putin"),
    ("PER", "putin"),

    ("PER", "kevin magnussen"),
    ("PER", "magnussen"),

    ("PER", "carlo ancelotti"),
    ("PER", "ancelotti"),

    ("PER", "nico hülkenberg"),
    ("PER", "hülkenberg"),

    ("PER", "esteban ocon"),
    ("PER", "ocon"),

    ("ORG", "fc københavn"),
    ("ORG", "fck"),
]


# =============================================================================
# 4. Wikidata 한 번 검색
# =============================================================================


def _search_once(
    *,
    surface: str,
    language: str,
) -> list[dict[str, Any]]:
    """
    Wikidata wbsearchentities API를 한 번 호출한다.

    예:
        surface="carlo ancelotti"
        language="da"

    반환 예:
        [
            {
                "qid": "Q174614",
                "label": "Carlo Ancelotti",
                "description": "...",
                "match_type": "label",
                "match_text": "Carlo Ancelotti",
            }
        ]

    429 / 503이 발생하면:
        Retry-After 헤더가 있으면 그 시간만큼 대기
        없으면 5, 10, 20, 40초 형태로 exponential backoff
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

    # 중요:
    # Accept-Encoding: gzip 을 넣지 않는다.
    # urllib은 gzip 응답을 자동으로 풀어주지 않아
    # 이전에 UnicodeDecodeError가 발생했기 때문.
    request = Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
        },
    )

    raw: bytes | None = None

    # -------------------------------------------------------------------------
    # 요청 + 429/503 재시도
    # -------------------------------------------------------------------------
    for attempt in range(MAX_RETRIES):

        try:
            with urlopen(
                request,
                timeout=20,
            ) as response:
                raw = response.read()

            # 성공했으면 retry loop 종료
            break

        except HTTPError as exc:

            # 429 = Too Many Requests
            # 503 = Service Unavailable / maxlag 등 일시적 제한
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

            # Retry-After가 숫자로 오면 그 값을 사용
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
                # 헤더가 없으면
                # 5초 → 10초 → 20초 → 40초
                wait_seconds = (
                    5.0
                    * (2 ** attempt)
                )

            print(
                "  "
                f"Wikidata HTTP {exc.code} "
                f"→ {wait_seconds:.1f}초 대기 후 재시도 "
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
        # for loop를 break 없이 끝냈다는 뜻
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

    # -------------------------------------------------------------------------
    # JSON decode
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # 필요한 필드만 추출
    # -------------------------------------------------------------------------
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

    return rows


# =============================================================================
# 5. Danish + English 후보 합치기
# =============================================================================


def search_wikidata_candidates(
    *,
    surface: str,
) -> list[dict[str, Any]]:
    """
    하나의 Entity surface를 Danish(da), English(en) 두 번 검색한다.

    이유:
    EB-NeRD는 덴마크어 뉴스지만
    어떤 Entity는 영어 label / alias에서 더 잘 검색될 수 있기 때문.

    동일 QID가 da/en 양쪽에서 나오면 한 번만 남긴다.
    """

    merged: list[
        dict[str, Any]
    ] = []

    seen_qids: set[str] = set()

    for language in [
        "da",
        "en",
    ]:

        candidates = (
            _search_once(
                surface=surface,
                language=language,
            )
        )

        for search_rank, candidate in enumerate(
            candidates,
            start=1,
        ):
            qid = candidate[
                "qid"
            ]

            if not qid:
                continue

            # da에서 이미 나온 QID면
            # en에서 또 나와도 중복 저장하지 않음
            if qid in seen_qids:
                continue

            seen_qids.add(
                qid
            )

            merged.append(
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

        # Wikidata API를 너무 빠르게 연속 호출하지 않도록 대기
        time.sleep(
            REQUEST_SLEEP_SECONDS
        )

    return merged


# =============================================================================
# 6. Main
# =============================================================================


def main() -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_rows: list[
        dict[str, Any]
    ] = []

    print(
        "=" * 100
    )
    print(
        "Wikidata Candidate Search Probe 시작"
    )
    print(
        "=" * 100
    )

    for (
        entity_group,
        surface,
    ) in TEST_ENTITIES:

        print()
        print(
            "-" * 100
        )
        print(
            f"{entity_group}::{surface}"
        )
        print(
            "-" * 100
        )

        candidates = (
            search_wikidata_candidates(
                surface=surface
            )
        )

        # ---------------------------------------------------------------------
        # 후보 없음
        # ---------------------------------------------------------------------
        if not candidates:

            print(
                "검색 후보 없음"
            )

            output_rows.append(
                {
                    "entity_group": (
                        entity_group
                    ),
                    "surface": (
                        surface
                    ),
                    "candidate_rank": None,
                    "qid": None,
                    "label": None,
                    "description": None,
                    "search_language": None,
                    "search_rank": None,
                    "match_type": None,
                    "match_text": None,
                }
            )

            continue

        # ---------------------------------------------------------------------
        # 후보 출력
        # ---------------------------------------------------------------------
        for (
            candidate_rank,
            candidate,
        ) in enumerate(
            candidates,
            start=1,
        ):

            print(
                f"[{candidate_rank}] "
                f"{candidate['qid']} | "
                f"{candidate['label']} | "
                f"{candidate['description']} | "
                f"lang={candidate['search_language']}"
            )

            output_rows.append(
                {
                    "entity_group": (
                        entity_group
                    ),
                    "surface": (
                        surface
                    ),
                    "candidate_rank": int(
                        candidate_rank
                    ),
                    "qid": (
                        candidate[
                            "qid"
                        ]
                    ),
                    "label": (
                        candidate[
                            "label"
                        ]
                    ),
                    "description": (
                        candidate[
                            "description"
                        ]
                    ),
                    "search_language": (
                        candidate[
                            "search_language"
                        ]
                    ),
                    "search_rank": int(
                        candidate[
                            "search_rank"
                        ]
                    ),
                    "match_type": (
                        candidate[
                            "match_type"
                        ]
                    ),
                    "match_text": (
                        candidate[
                            "match_text"
                        ]
                    ),
                }
            )

    # =========================================================================
    # 7. Parquet 저장
    # =========================================================================

    schema = {
        "entity_group": pl.String,
        "surface": pl.String,
        "candidate_rank": pl.Int64,
        "qid": pl.String,
        "label": pl.String,
        "description": pl.String,
        "search_language": pl.String,
        "search_rank": pl.Int64,
        "match_type": pl.String,
        "match_text": pl.String,
    }

    if output_rows:
        output_df = (
            pl.DataFrame(
                output_rows,
                schema=schema,
            )
            .sort(
                [
                    "entity_group",
                    "surface",
                    "candidate_rank",
                ]
            )
        )
    else:
        output_df = (
            pl.DataFrame(
                schema=schema
            )
        )

    output_df.write_parquet(
        OUTPUT_PATH,
        compression="zstd",
    )

    print()
    print(
        "=" * 100
    )
    print(
        "Wikidata Candidate Search Probe 완료"
    )
    print(
        "=" * 100
    )

    print(
        f"candidate_row_count = {output_df.height}"
    )

    print(
        f"output = {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()