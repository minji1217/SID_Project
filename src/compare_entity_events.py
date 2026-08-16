

from __future__ import annotations

"""
Baseline Event vs Entity Normalization Event 비교 스크립트
==========================================================

이 파일의 목적
--------------
Entity Normalization을 적용한 뒤 Event 수가 줄었다고 해서 바로
"좋아졌다"고 결론 내리면 안 된다.

예를 들어:
    baseline에서는
        article 100 -> event 10
        article 200 -> event 20

    normalize_only에서는
        article 100 -> event 8
        article 200 -> event 8

이라면 두 기사가 새롭게 같은 Event가 된 것은 맞다.
하지만 정말 같은 사건이어서 합쳐진 것인지, 잘못된 정규화 때문에
합쳐진 것인지는 기사 제목과 어떤 entity가 연결고리였는지 확인해야 한다.

이 스크립트는 따라서 다음 4가지를 한 번에 한다.

1. 숫자 비교
   - Event 수
   - singleton 수 / 비율
   - 2개 이상 기사 Event 수
   - 평균 / 중앙값 / 최대 Event 크기
   - Entity vocabulary 수
   - High-DF Entity 수
   - Validation Event 결과

2. Event membership 비교
   - 새롭게 같은 Event가 된 기사쌍(added article pairs)
   - baseline에서는 같았지만 normalize_only에서 갈라진 기사쌍(removed article pairs)

3. 새 Merge Event 정성검사용 자료 생성
   - normalize_only Event 하나가 baseline Event 여러 개를 포함하는지
   - 어떤 기사들이 들어있는지
   - 기사 제목
   - 어떤 normalization이 사용됐는지
   - 공통 canonical entity가 무엇인지

4. Entity DF / High-DF 변화 추적
   - normalization 때문에 어떤 entity의 DF가 올라갔는지
   - 새롭게 High-DF가 된 entity가 무엇인지

중요
----
Event ID 자체는 baseline과 normalize_only 사이에서 직접 비교하지 않는다.
Event ID는 connected component를 다시 만든 뒤 시간순으로 다시 번호를 부여하기 때문에
"baseline event 100 == normalize event 100"이라고 가정하면 안 된다.

대신 article membership을 기준으로 비교한다.
"어떤 기사들이 같은 그룹에 속하느냐"가 실제 Event 비교 기준이다.
"""

import argparse
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from pprint import pprint
from statistics import mean, median
from typing import Any, Iterable

import polars as pl


# =============================================================================
# 1. 기본 경로 / 실험 파일 정의
# =============================================================================


@dataclass(frozen=True)
class ExperimentFiles:
    """
    실험 snapshot 한 개에서 비교에 필요한 파일 경로를 묶어 관리한다.

    baseline과 normalize_only 모두 동일한 파일명을 사용하므로
    snapshot_dir만 다르게 주면 같은 방식으로 읽을 수 있다.
    """

    snapshot_dir: Path

    @property
    def article_events(self) -> Path:
        return self.snapshot_dir / "article_events.parquet"

    @property
    def event_master(self) -> Path:
        return self.snapshot_dir / "event_master.parquet"

    @property
    def entity_idf(self) -> Path:
        return self.snapshot_dir / "entity_idf.parquet"

    @property
    def event_master_with_validation(self) -> Path:
        return self.snapshot_dir / "event_master_with_validation.parquet"

    @property
    def validation_article_events(self) -> Path:
        return self.snapshot_dir / "validation_article_events.parquet"

    @property
    def articles_base(self) -> Path:
        return self.snapshot_dir / "articles_base.parquet"

    @property
    def article_entities(self) -> Path:
        return self.snapshot_dir / "article_entities.parquet"

    @property
    def entity_normalization_map(self) -> Path:
        return self.snapshot_dir / "entity_normalization_map.parquet"


# =============================================================================
# 2. 공통 유틸 함수
# =============================================================================


def _require_file(path: Path, description: str) -> None:
    """필수 파일이 없으면 뒤에서 애매한 오류가 나기 전에 즉시 중단한다."""

    if not path.exists():
        raise FileNotFoundError(
            f"{description} 파일이 없습니다. 경로={path}"
        )



def _require_columns(
    df: pl.DataFrame,
    required_columns: Iterable[str],
    description: str,
) -> None:
    """Parquet schema가 예상과 다른 경우 명확한 메시지로 중단한다."""

    required = set(required_columns)
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{description}에 필요한 컬럼이 없습니다: "
            + ", ".join(sorted(missing))
        )



def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    """0으로 나누는 문제 없이 비율을 계산한다."""

    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)



def _sorted_unique_strings(values: Iterable[str]) -> list[str]:
    """문자열 목록을 deterministic하게 중복 제거 + 정렬한다."""

    return sorted({str(value) for value in values if value is not None})


# =============================================================================
# 3. Train Event membership 읽기
# =============================================================================


def _load_article_event_mapping(
    article_events_path: Path,
) -> tuple[dict[int, int], dict[int, list[int]]]:
    """
    article_events.parquet을 두 방향 lookup으로 변환한다.

    반환 1
    ------
    article_to_event
        article_id -> event_id

    반환 2
    ------
    event_to_articles
        event_id -> [article_id, ...]

    왜 두 개가 필요한가?
    -------------------
    - article_to_event:
      normalize Event 안의 기사들이 baseline에서 어느 Event에 있었는지 찾을 때 사용

    - event_to_articles:
      Event 크기, singleton, article pair 생성 등에 사용
    """

    _require_file(
        article_events_path,
        "article_events",
    )

    df = pl.read_parquet(article_events_path).select(
        ["article_id", "event_id"]
    )

    _require_columns(
        df,
        ["article_id", "event_id"],
        "article_events.parquet",
    )

    duplicate_article_count = (
        df.select(pl.col("article_id").is_duplicated().sum()).item()
    )

    if duplicate_article_count != 0:
        raise ValueError(
            "article_events.parquet에 중복 article_id가 존재합니다. "
            f"중복 행 수={duplicate_article_count}"
        )

    article_to_event: dict[int, int] = {}
    event_to_articles: dict[int, list[int]] = defaultdict(list)

    for article_id, event_id in df.iter_rows():
        article_id = int(article_id)
        event_id = int(event_id)

        article_to_event[article_id] = event_id
        event_to_articles[event_id].append(article_id)

    # 결과가 실행 환경에 따라 흔들리지 않도록 article_id 정렬
    for article_ids in event_to_articles.values():
        article_ids.sort()

    return article_to_event, dict(event_to_articles)


# =============================================================================
# 4. Train Event 기본 통계
# =============================================================================


def _build_train_event_summary(
    event_to_articles: dict[int, list[int]],
) -> dict[str, float | int]:
    """
    Event membership만으로 Train Event 핵심 지표를 계산한다.

    Event ID 값은 비교하지 않고 Event별 기사 수만 사용한다.
    """

    event_sizes = [
        len(article_ids)
        for article_ids in event_to_articles.values()
    ]

    if not event_sizes:
        raise ValueError("Train Event가 하나도 존재하지 않습니다.")

    event_count = len(event_sizes)
    singleton_count = sum(size == 1 for size in event_sizes)
    multi_article_event_count = sum(size >= 2 for size in event_sizes)

    return {
        "train_event_count": int(event_count),
        "train_singleton_event_count": int(singleton_count),
        "train_singleton_event_ratio": _safe_ratio(
            singleton_count,
            event_count,
        ),
        "train_multi_article_event_count": int(
            multi_article_event_count
        ),
        "train_mean_event_size": float(mean(event_sizes)),
        "train_median_event_size": float(median(event_sizes)),
        "train_max_event_size": int(max(event_sizes)),
    }


# =============================================================================
# 5. Validation Event 기본 통계
# =============================================================================


def _build_validation_summary(
    experiment: ExperimentFiles,
) -> dict[str, float | int]:
    """
    event_master_with_validation.parquet을 이용해 Validation 결과를 복원한다.

    run_entity_experiment 로그에 있었던 다음 값들을 snapshot만으로 다시 계산한다.

    - final_event_count
    - new_validation_event_count
    - matched_existing_event_count
    - matched_train_origin_event_count
    - matched_validation_origin_event_count

    계산 원리
    ---------
    Train-origin Event에 들어간 validation 기사는 모두 기존 Train Event에 매칭된 기사다.

    Validation-origin Event는 최초 1개 기사가 Event를 새로 만들고,
    그 뒤 붙는 기사들은 validation-origin Event에 매칭된 기사다.
    """

    _require_file(
        experiment.event_master_with_validation,
        "event_master_with_validation",
    )
    _require_file(
        experiment.validation_article_events,
        "validation_article_events",
    )

    master = pl.read_parquet(
        experiment.event_master_with_validation
    )

    required_columns = [
        "event_id",
        "event_origin_split",
        "event_article_count",
        "train_article_count",
        "validation_article_count",
    ]

    _require_columns(
        master,
        required_columns,
        "event_master_with_validation.parquet",
    )

    validation_article_count = pl.read_parquet(
        experiment.validation_article_events
    ).height

    train_origin_rows = master.filter(
        pl.col("event_origin_split") == "train"
    )
    validation_origin_rows = master.filter(
        pl.col("event_origin_split") == "validation"
    )

    matched_train_origin_count = int(
        train_origin_rows
        .get_column("validation_article_count")
        .sum()
    )

    # Validation-origin Event 하나는 첫 기사 1개가 새 Event를 생성한다.
    # 그 Event에 validation 기사가 3개라면:
    #   첫 기사 1개 = new event
    #   나머지 2개 = matched_validation_origin
    matched_validation_origin_count = int(
        sum(
            max(int(count) - 1, 0)
            for count in validation_origin_rows
            .get_column("validation_article_count")
            .to_list()
        )
    )

    new_validation_event_count = int(
        validation_origin_rows.height
    )

    matched_existing_event_count = (
        matched_train_origin_count
        + matched_validation_origin_count
    )

    return {
        "validation_article_count": int(
            validation_article_count
        ),
        "final_event_count": int(master.height),
        "new_validation_event_count": int(
            new_validation_event_count
        ),
        "matched_existing_event_count": int(
            matched_existing_event_count
        ),
        "matched_train_origin_event_count": int(
            matched_train_origin_count
        ),
        "matched_validation_origin_event_count": int(
            matched_validation_origin_count
        ),
    }


# =============================================================================
# 6. Entity IDF / High-DF 비교
# =============================================================================


def _load_entity_idf(
    entity_idf_path: Path,
) -> dict[str, dict[str, Any]]:
    """entity_idf.parquet을 entity 문자열 기준 dictionary로 읽는다."""

    _require_file(entity_idf_path, "entity_idf")

    df = pl.read_parquet(entity_idf_path)

    _require_columns(
        df,
        [
            "entity",
            "document_frequency",
            "document_frequency_ratio",
            "idf",
            "is_high_df",
        ],
        "entity_idf.parquet",
    )

    result: dict[str, dict[str, Any]] = {}

    for row in df.iter_rows(named=True):
        entity = str(row["entity"])
        result[entity] = {
            "document_frequency": int(
                row["document_frequency"]
            ),
            "document_frequency_ratio": float(
                row["document_frequency_ratio"]
            ),
            "idf": float(row["idf"]),
            "is_high_df": bool(row["is_high_df"]),
        }

    return result



def _compare_entity_idf(
    baseline_idf: dict[str, dict[str, Any]],
    normalize_idf: dict[str, dict[str, Any]],
) -> pl.DataFrame:
    """
    Entity vocabulary / DF / High-DF 상태가 어떻게 바뀌었는지 한 행씩 비교한다.

    특히 확인하고 싶은 것
    ---------------------
    1. possessive variant가 vocabulary에서 사라졌는가?
    2. canonical base entity의 DF가 올라갔는가?
    3. 그 결과 새롭게 High-DF가 된 entity가 있는가?
    """

    all_entities = sorted(
        set(baseline_idf) | set(normalize_idf)
    )

    rows: list[dict[str, Any]] = []

    for entity in all_entities:
        baseline = baseline_idf.get(entity)
        normalize = normalize_idf.get(entity)

        baseline_df = (
            baseline["document_frequency"]
            if baseline is not None
            else None
        )
        normalize_df = (
            normalize["document_frequency"]
            if normalize is not None
            else None
        )

        baseline_high = (
            bool(baseline["is_high_df"])
            if baseline is not None
            else False
        )
        normalize_high = (
            bool(normalize["is_high_df"])
            if normalize is not None
            else False
        )

        # 아무 변화도 없는 entity는 결과 파일을 불필요하게 크게 만들므로 제외
        if (
            baseline is not None
            and normalize is not None
            and baseline_df == normalize_df
            and baseline_high == normalize_high
        ):
            continue

        if baseline is None:
            change_type = "VOCABULARY_ADDED"
        elif normalize is None:
            change_type = "VOCABULARY_REMOVED"
        elif not baseline_high and normalize_high:
            change_type = "BECAME_HIGH_DF"
        elif baseline_high and not normalize_high:
            change_type = "LEFT_HIGH_DF"
        else:
            change_type = "DF_CHANGED"

        rows.append(
            {
                "entity": entity,
                "change_type": change_type,
                "baseline_document_frequency": baseline_df,
                "normalize_document_frequency": normalize_df,
                "document_frequency_delta": (
                    None
                    if baseline_df is None or normalize_df is None
                    else int(normalize_df - baseline_df)
                ),
                "baseline_is_high_df": baseline_high,
                "normalize_is_high_df": normalize_high,
            }
        )

    schema = {
        "entity": pl.String,
        "change_type": pl.String,
        "baseline_document_frequency": pl.Int64,
        "normalize_document_frequency": pl.Int64,
        "document_frequency_delta": pl.Int64,
        "baseline_is_high_df": pl.Boolean,
        "normalize_is_high_df": pl.Boolean,
    }

    if rows:
        return pl.DataFrame(rows, schema=schema).sort(
            ["change_type", "entity"]
        )

    return pl.DataFrame(schema=schema)


# =============================================================================
# 7. Event 안의 article pair 생성
# =============================================================================


def _build_coclustered_article_pairs(
    event_to_articles: dict[int, list[int]],
) -> set[tuple[int, int]]:
    """
    같은 Event에 속한 모든 기사쌍을 만든다.

    예
    --
    Event 10 = [100, 200, 300]

    결과:
        (100, 200)
        (100, 300)
        (200, 300)

    이 집합을 baseline과 normalize_only에서 각각 만든 뒤 차집합을 구하면:

    normalize_pairs - baseline_pairs
        = 새롭게 같은 Event가 된 기사쌍

    baseline_pairs - normalize_pairs
        = 원래 같은 Event였는데 갈라진 기사쌍
    """

    pairs: set[tuple[int, int]] = set()

    for article_ids in event_to_articles.values():
        if len(article_ids) < 2:
            continue

        for left_id, right_id in combinations(
            sorted(article_ids),
            2,
        ):
            pairs.add((int(left_id), int(right_id)))

    return pairs


# =============================================================================
# 8. 기사 제목 / Entity Normalization 증거 읽기
# =============================================================================


def _load_article_titles(
    baseline_experiment: ExperimentFiles,
    normalize_experiment: ExperimentFiles,
) -> dict[int, str]:
    """
    정성 검사용 article_id -> title lookup을 만든다.

    보통 baseline snapshot의 articles_base.parquet을 사용한다.
    baseline에 없다면 normalize snapshot을 fallback으로 사용한다.
    """

    candidate_paths = [
        baseline_experiment.articles_base,
        normalize_experiment.articles_base,
    ]

    source_path = next(
        (path for path in candidate_paths if path.exists()),
        None,
    )

    if source_path is None:
        raise FileNotFoundError(
            "기사 제목 확인용 articles_base.parquet을 찾을 수 없습니다."
        )

    articles = pl.read_parquet(source_path)

    _require_columns(
        articles,
        ["article_id", "title"],
        "articles_base.parquet",
    )

    return {
        int(article_id): str(title or "")
        for article_id, title in articles
        .select(["article_id", "title"])
        .iter_rows()
    }



def _load_normalization_evidence(
    normalize_experiment: ExperimentFiles,
) -> tuple[
    dict[int, set[str]],
    dict[int, list[str]],
    set[str],
    pl.DataFrame,
]:
    """
    normalize_only에서 만들어진 article_entities.parquet과
    entity_normalization_map.parquet을 정성 검사용 lookup으로 바꾼다.

    반환
    ----
    article_canonical_entities
        article_id -> canonical entity key 집합

    article_changes
        article_id -> 실제 적용된 normalization 문자열 목록
        예: ["PER::mette frederiksens -> PER::mette frederiksen"]

    normalization_targets
        mapping의 canonical target 전체 집합
        공통 entity가 실제 normalization 대상이었는지 확인할 때 사용

    normalization_map_df
        원본 mapping DataFrame
    """

    _require_file(
        normalize_experiment.article_entities,
        "normalize_only article_entities",
    )
    _require_file(
        normalize_experiment.entity_normalization_map,
        "normalize_only entity_normalization_map",
    )

    article_entities_df = pl.read_parquet(
        normalize_experiment.article_entities
    )

    _require_columns(
        article_entities_df,
        [
            "article_id",
            "baseline_entity_key",
            "canonical_entity_key",
            "processing_method",
        ],
        "article_entities.parquet",
    )

    normalization_map_df = pl.read_parquet(
        normalize_experiment.entity_normalization_map
    )

    _require_columns(
        normalization_map_df,
        [
            "variant_entity_key",
            "canonical_entity_key",
            "rule",
        ],
        "entity_normalization_map.parquet",
    )

    article_canonical_entities: dict[int, set[str]] = defaultdict(set)
    article_changes: dict[int, list[str]] = defaultdict(list)

    for row in article_entities_df.iter_rows(named=True):
        article_id = int(row["article_id"])
        baseline_key = str(row["baseline_entity_key"])
        canonical_key = str(row["canonical_entity_key"])

        article_canonical_entities[article_id].add(
            canonical_key
        )

        if baseline_key != canonical_key:
            article_changes[article_id].append(
                f"{baseline_key} -> {canonical_key}"
            )

    # 같은 기사에 동일 변화가 중복 기록될 수도 있으므로 정리
    article_changes = {
        article_id: _sorted_unique_strings(changes)
        for article_id, changes in article_changes.items()
    }

    normalization_targets = {
        str(value)
        for value in normalization_map_df
        .get_column("canonical_entity_key")
        .to_list()
    }

    return (
        dict(article_canonical_entities),
        article_changes,
        normalization_targets,
        normalization_map_df,
    )


# =============================================================================
# 9. Added / Removed article pair 상세 분석
# =============================================================================


def _build_pair_detail_df(
    pairs: set[tuple[int, int]],
    pair_type: str,
    baseline_article_to_event: dict[int, int],
    normalize_article_to_event: dict[int, int],
    titles: dict[int, str],
    article_canonical_entities: dict[int, set[str]],
    article_changes: dict[int, list[str]],
    normalization_targets: set[str],
) -> pl.DataFrame:
    """
    article pair에 사람이 검토할 수 있는 설명 정보를 붙인다.

    pair_type
    ---------
    ADDED:
        normalize_only에서 새롭게 같은 Event가 된 기사쌍

    REMOVED:
        baseline에서는 같은 Event였는데 normalize_only에서 갈라진 기사쌍
    """

    rows: list[dict[str, Any]] = []

    for left_id, right_id in sorted(pairs):
        left_entities = article_canonical_entities.get(
            left_id,
            set(),
        )
        right_entities = article_canonical_entities.get(
            right_id,
            set(),
        )

        common_entities = sorted(
            left_entities & right_entities
        )

        normalization_common_entities = [
            entity
            for entity in common_entities
            if entity in normalization_targets
        ]

        rows.append(
            {
                "pair_type": pair_type,
                "left_article_id": int(left_id),
                "right_article_id": int(right_id),
                "baseline_left_event_id": int(
                    baseline_article_to_event[left_id]
                ),
                "baseline_right_event_id": int(
                    baseline_article_to_event[right_id]
                ),
                "normalize_left_event_id": int(
                    normalize_article_to_event[left_id]
                ),
                "normalize_right_event_id": int(
                    normalize_article_to_event[right_id]
                ),
                "left_title": titles.get(left_id, ""),
                "right_title": titles.get(right_id, ""),
                "common_canonical_entities": common_entities,
                "normalization_common_entities": (
                    normalization_common_entities
                ),
                "left_normalization_changes": article_changes.get(
                    left_id,
                    [],
                ),
                "right_normalization_changes": article_changes.get(
                    right_id,
                    [],
                ),
            }
        )

    schema = {
        "pair_type": pl.String,
        "left_article_id": pl.Int64,
        "right_article_id": pl.Int64,
        "baseline_left_event_id": pl.Int64,
        "baseline_right_event_id": pl.Int64,
        "normalize_left_event_id": pl.Int64,
        "normalize_right_event_id": pl.Int64,
        "left_title": pl.String,
        "right_title": pl.String,
        "common_canonical_entities": pl.List(pl.String),
        "normalization_common_entities": pl.List(pl.String),
        "left_normalization_changes": pl.List(pl.String),
        "right_normalization_changes": pl.List(pl.String),
    }

    if rows:
        return pl.DataFrame(rows, schema=schema)

    return pl.DataFrame(schema=schema)


# =============================================================================
# 10. Event 단위 Merge / Split 분석
# =============================================================================


def _build_new_merge_events_df(
    baseline_article_to_event: dict[int, int],
    normalize_event_to_articles: dict[int, list[int]],
    titles: dict[int, str],
    article_changes: dict[int, list[str]],
    article_canonical_entities: dict[int, set[str]],
    normalization_targets: set[str],
    added_pairs: set[tuple[int, int]],
) -> pl.DataFrame:
    """
    normalize_only Event 하나 안에 baseline Event가 2개 이상 들어오면
    "새 merge/reconfiguration 후보"로 기록한다.

    중요한 점
    ---------
    이것이 무조건 좋은 merge라는 뜻은 아니다.
    사람이 기사 제목과 normalization 근거를 보고 확인해야 한다.
    """

    # normalize event별 새롭게 추가된 co-cluster pair 수를 먼저 센다.
    added_pair_count_by_normalize_event: dict[int, int] = defaultdict(int)

    # event -> articles 구조를 article -> event 구조로 한 번 뒤집어 둔다.
    # added pair는 normalize_only에서 반드시 같은 Event에 속하므로
    # pair의 왼쪽 article_id만 조회해도 해당 normalize Event를 알 수 있다.
    article_to_normalize_event: dict[int, int] = {}
    for event_id, article_ids in normalize_event_to_articles.items():
        for article_id in article_ids:
            article_to_normalize_event[int(article_id)] = int(event_id)

    for left_id, _ in added_pairs:
        added_pair_count_by_normalize_event[
            article_to_normalize_event[left_id]
        ] += 1

    rows: list[dict[str, Any]] = []

    for normalize_event_id, article_ids in sorted(
        normalize_event_to_articles.items()
    ):
        baseline_event_ids = sorted(
            {
                baseline_article_to_event[article_id]
                for article_id in article_ids
            }
        )

        if len(baseline_event_ids) <= 1:
            continue

        changed_article_ids = sorted(
            article_id
            for article_id in article_ids
            if article_id in article_changes
        )

        normalization_changes = _sorted_unique_strings(
            change
            for article_id in article_ids
            for change in article_changes.get(article_id, [])
        )

        # Event 전체에 등장한 canonical entity 중
        # 실제 normalization target이었던 것들을 확인한다.
        event_entity_union: set[str] = set()

        for article_id in article_ids:
            event_entity_union.update(
                article_canonical_entities.get(article_id, set())
            )

        event_normalization_targets = sorted(
            event_entity_union & normalization_targets
        )

        rows.append(
            {
                "normalize_event_id": int(normalize_event_id),
                "normalize_event_article_count": int(len(article_ids)),
                "baseline_event_count": int(len(baseline_event_ids)),
                "baseline_event_ids": [
                    int(value) for value in baseline_event_ids
                ],
                "article_ids": [int(value) for value in article_ids],
                "article_titles": [
                    titles.get(article_id, "")
                    for article_id in article_ids
                ],
                "changed_article_count": int(
                    len(changed_article_ids)
                ),
                "changed_article_ids": [
                    int(value) for value in changed_article_ids
                ],
                "normalization_changes": normalization_changes,
                "normalization_target_entities": (
                    event_normalization_targets
                ),
                "added_article_pair_count": int(
                    added_pair_count_by_normalize_event.get(
                        normalize_event_id,
                        0,
                    )
                ),
            }
        )

    schema = {
        "normalize_event_id": pl.Int64,
        "normalize_event_article_count": pl.Int64,
        "baseline_event_count": pl.Int64,
        "baseline_event_ids": pl.List(pl.Int64),
        "article_ids": pl.List(pl.Int64),
        "article_titles": pl.List(pl.String),
        "changed_article_count": pl.Int64,
        "changed_article_ids": pl.List(pl.Int64),
        "normalization_changes": pl.List(pl.String),
        "normalization_target_entities": pl.List(pl.String),
        "added_article_pair_count": pl.Int64,
    }

    if rows:
        return (
            pl.DataFrame(rows, schema=schema)
            .sort(
                [
                    "added_article_pair_count",
                    "normalize_event_article_count",
                ],
                descending=[True, True],
            )
        )

    return pl.DataFrame(schema=schema)



def _build_split_events_df(
    baseline_event_to_articles: dict[int, list[int]],
    normalize_article_to_event: dict[int, int],
    titles: dict[int, str],
    article_changes: dict[int, list[str]],
) -> pl.DataFrame:
    """
    baseline Event 하나가 normalize_only에서 여러 Event로 갈라졌는지 찾는다.

    normalization은 보통 merge를 유도하지만,
    canonical entity DF가 증가해 새롭게 High-DF가 되는 경우에는
    오히려 기존 연결고리가 Event 계산에서 제거되어 split이 생길 수도 있다.

    그래서 merge만 보지 않고 split도 반드시 같이 검사한다.
    """

    rows: list[dict[str, Any]] = []

    for baseline_event_id, article_ids in sorted(
        baseline_event_to_articles.items()
    ):
        normalize_event_ids = sorted(
            {
                normalize_article_to_event[article_id]
                for article_id in article_ids
            }
        )

        if len(normalize_event_ids) <= 1:
            continue

        changed_article_ids = sorted(
            article_id
            for article_id in article_ids
            if article_id in article_changes
        )

        rows.append(
            {
                "baseline_event_id": int(baseline_event_id),
                "baseline_event_article_count": int(len(article_ids)),
                "normalize_event_count": int(len(normalize_event_ids)),
                "normalize_event_ids": [
                    int(value) for value in normalize_event_ids
                ],
                "article_ids": [int(value) for value in article_ids],
                "article_titles": [
                    titles.get(article_id, "")
                    for article_id in article_ids
                ],
                "changed_article_count": int(
                    len(changed_article_ids)
                ),
                "changed_article_ids": [
                    int(value) for value in changed_article_ids
                ],
                "normalization_changes": _sorted_unique_strings(
                    change
                    for article_id in article_ids
                    for change in article_changes.get(article_id, [])
                ),
            }
        )

    schema = {
        "baseline_event_id": pl.Int64,
        "baseline_event_article_count": pl.Int64,
        "normalize_event_count": pl.Int64,
        "normalize_event_ids": pl.List(pl.Int64),
        "article_ids": pl.List(pl.Int64),
        "article_titles": pl.List(pl.String),
        "changed_article_count": pl.Int64,
        "changed_article_ids": pl.List(pl.Int64),
        "normalization_changes": pl.List(pl.String),
    }

    if rows:
        return (
            pl.DataFrame(rows, schema=schema)
            .sort(
                [
                    "normalize_event_count",
                    "baseline_event_article_count",
                ],
                descending=[True, True],
            )
        )

    return pl.DataFrame(schema=schema)


# =============================================================================
# 11. Summary metric 테이블 만들기
# =============================================================================


def _make_summary_rows(
    baseline_train: dict[str, float | int],
    normalize_train: dict[str, float | int],
    baseline_validation: dict[str, float | int],
    normalize_validation: dict[str, float | int],
    baseline_idf: dict[str, dict[str, Any]],
    normalize_idf: dict[str, dict[str, Any]],
    added_pair_count: int,
    removed_pair_count: int,
    merge_event_count: int,
    split_event_count: int,
) -> pl.DataFrame:
    """콘솔/파일에서 한눈에 볼 수 있는 Before/After 표를 만든다."""

    baseline_high_df_count = sum(
        bool(row["is_high_df"])
        for row in baseline_idf.values()
    )
    normalize_high_df_count = sum(
        bool(row["is_high_df"])
        for row in normalize_idf.values()
    )

    metric_values: list[
        tuple[str, float | int, float | int]
    ] = [
        (
            "entity_vocabulary_count",
            len(baseline_idf),
            len(normalize_idf),
        ),
        (
            "high_df_entity_count",
            baseline_high_df_count,
            normalize_high_df_count,
        ),
    ]

    for metric_name in [
        "train_event_count",
        "train_singleton_event_count",
        "train_singleton_event_ratio",
        "train_multi_article_event_count",
        "train_mean_event_size",
        "train_median_event_size",
        "train_max_event_size",
    ]:
        metric_values.append(
            (
                metric_name,
                baseline_train[metric_name],
                normalize_train[metric_name],
            )
        )

    for metric_name in [
        "final_event_count",
        "new_validation_event_count",
        "matched_existing_event_count",
        "matched_train_origin_event_count",
        "matched_validation_origin_event_count",
    ]:
        metric_values.append(
            (
                metric_name,
                baseline_validation[metric_name],
                normalize_validation[metric_name],
            )
        )

    # pair / merge / split은 baseline 값이 0이라는 의미보다
    # "비교 결과 자체"이므로 baseline=0, normalize=실제 개수 형태로 기록한다.
    metric_values.extend(
        [
            ("added_cocluster_article_pair_count", 0, added_pair_count),
            ("removed_cocluster_article_pair_count", 0, removed_pair_count),
            ("new_merge_or_reconfigured_event_count", 0, merge_event_count),
            ("split_or_reconfigured_event_count", 0, split_event_count),
        ]
    )

    rows: list[dict[str, Any]] = []

    for metric_name, baseline_value, normalize_value in metric_values:
        baseline_float = float(baseline_value)
        normalize_float = float(normalize_value)
        delta = normalize_float - baseline_float

        rows.append(
            {
                "metric": metric_name,
                "baseline": baseline_float,
                "normalize_only": normalize_float,
                "delta": delta,
                "delta_ratio": (
                    None
                    if baseline_float == 0
                    else delta / baseline_float
                ),
            }
        )

    return pl.DataFrame(
        rows,
        schema={
            "metric": pl.String,
            "baseline": pl.Float64,
            "normalize_only": pl.Float64,
            "delta": pl.Float64,
            "delta_ratio": pl.Float64,
        },
    )


# =============================================================================
# 12. 결과 저장
# =============================================================================


def _write_summary_text(
    output_path: Path,
    summary_df: pl.DataFrame,
    entity_df_changes: pl.DataFrame,
    merge_events: pl.DataFrame,
    split_events: pl.DataFrame,
) -> None:
    """터미널을 다시 찾지 않아도 되도록 핵심 결과를 txt로 저장한다."""

    lines: list[str] = []

    lines.append("=" * 80)
    lines.append("Baseline vs normalize_only Entity Event Comparison")
    lines.append("=" * 80)
    lines.append("")

    for row in summary_df.iter_rows(named=True):
        lines.append(
            f"{row['metric']}: "
            f"baseline={row['baseline']:.6f}, "
            f"normalize_only={row['normalize_only']:.6f}, "
            f"delta={row['delta']:+.6f}"
        )

    lines.append("")
    lines.append("-" * 80)
    lines.append("High-DF 변화")
    lines.append("-" * 80)

    became_high_df = entity_df_changes.filter(
        pl.col("change_type") == "BECAME_HIGH_DF"
    )

    if became_high_df.height == 0:
        lines.append("새롭게 High-DF가 된 entity 없음")
    else:
        for row in became_high_df.iter_rows(named=True):
            lines.append(
                f"{row['entity']}: "
                f"df {row['baseline_document_frequency']} "
                f"-> {row['normalize_document_frequency']}"
            )

    lines.append("")
    lines.append("-" * 80)
    lines.append("정성 검토 대상")
    lines.append("-" * 80)
    lines.append(
        f"new merge/reconfigured events = {merge_events.height}"
    )
    lines.append(
        f"split/reconfigured events = {split_events.height}"
    )

    output_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# =============================================================================
# 13. 전체 비교 실행 함수
# =============================================================================


def compare_entity_events(
    baseline_dir: Path,
    normalize_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """
    Baseline과 normalize_only snapshot을 실제로 비교하는 메인 함수.

    이 함수는 기존 Event 파일을 수정하지 않는다.
    비교 결과만 별도 output_dir에 저장한다.
    """

    baseline = ExperimentFiles(
        snapshot_dir=baseline_dir
    )
    normalize = ExperimentFiles(
        snapshot_dir=normalize_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # STEP 1. Train article -> Event membership 읽기
    # -------------------------------------------------------------------------
    (
        baseline_article_to_event,
        baseline_event_to_articles,
    ) = _load_article_event_mapping(
        baseline.article_events
    )

    (
        normalize_article_to_event,
        normalize_event_to_articles,
    ) = _load_article_event_mapping(
        normalize.article_events
    )

    # 같은 Train article universe를 비교하는 실험이어야 한다.
    baseline_article_ids = set(
        baseline_article_to_event
    )
    normalize_article_ids = set(
        normalize_article_to_event
    )

    if baseline_article_ids != normalize_article_ids:
        raise ValueError(
            "baseline과 normalize_only의 Train article 집합이 다릅니다. "
            f"baseline_only={len(baseline_article_ids - normalize_article_ids)}, "
            f"normalize_only={len(normalize_article_ids - baseline_article_ids)}"
        )

    # -------------------------------------------------------------------------
    # STEP 2. 숫자 지표 계산
    # -------------------------------------------------------------------------
    baseline_train_summary = _build_train_event_summary(
        baseline_event_to_articles
    )
    normalize_train_summary = _build_train_event_summary(
        normalize_event_to_articles
    )

    baseline_validation_summary = _build_validation_summary(
        baseline
    )
    normalize_validation_summary = _build_validation_summary(
        normalize
    )

    baseline_idf = _load_entity_idf(
        baseline.entity_idf
    )
    normalize_idf = _load_entity_idf(
        normalize.entity_idf
    )

    entity_df_changes = _compare_entity_idf(
        baseline_idf,
        normalize_idf,
    )

    # -------------------------------------------------------------------------
    # STEP 3. Event ID가 아니라 article pair 기준으로 실제 grouping 변화 비교
    # -------------------------------------------------------------------------
    baseline_pairs = _build_coclustered_article_pairs(
        baseline_event_to_articles
    )
    normalize_pairs = _build_coclustered_article_pairs(
        normalize_event_to_articles
    )

    added_pairs = normalize_pairs - baseline_pairs
    removed_pairs = baseline_pairs - normalize_pairs

    # -------------------------------------------------------------------------
    # STEP 4. 정성검토용 제목 / normalization 증거 준비
    # -------------------------------------------------------------------------
    titles = _load_article_titles(
        baseline,
        normalize,
    )

    (
        article_canonical_entities,
        article_changes,
        normalization_targets,
        normalization_map_df,
    ) = _load_normalization_evidence(
        normalize
    )

    # -------------------------------------------------------------------------
    # STEP 5. 새롭게 같은 Event가 된 pair / 갈라진 pair 상세 테이블
    # -------------------------------------------------------------------------
    added_pair_df = _build_pair_detail_df(
        pairs=added_pairs,
        pair_type="ADDED",
        baseline_article_to_event=baseline_article_to_event,
        normalize_article_to_event=normalize_article_to_event,
        titles=titles,
        article_canonical_entities=article_canonical_entities,
        article_changes=article_changes,
        normalization_targets=normalization_targets,
    )

    removed_pair_df = _build_pair_detail_df(
        pairs=removed_pairs,
        pair_type="REMOVED",
        baseline_article_to_event=baseline_article_to_event,
        normalize_article_to_event=normalize_article_to_event,
        titles=titles,
        article_canonical_entities=article_canonical_entities,
        article_changes=article_changes,
        normalization_targets=normalization_targets,
    )

    # -------------------------------------------------------------------------
    # STEP 6. Event 단위 merge / split 후보 생성
    # -------------------------------------------------------------------------
    merge_events_df = _build_new_merge_events_df(
        baseline_article_to_event=baseline_article_to_event,
        normalize_event_to_articles=normalize_event_to_articles,
        titles=titles,
        article_changes=article_changes,
        article_canonical_entities=article_canonical_entities,
        normalization_targets=normalization_targets,
        added_pairs=added_pairs,
    )

    split_events_df = _build_split_events_df(
        baseline_event_to_articles=baseline_event_to_articles,
        normalize_article_to_event=normalize_article_to_event,
        titles=titles,
        article_changes=article_changes,
    )

    # -------------------------------------------------------------------------
    # STEP 7. Summary 표 생성
    # -------------------------------------------------------------------------
    summary_df = _make_summary_rows(
        baseline_train=baseline_train_summary,
        normalize_train=normalize_train_summary,
        baseline_validation=baseline_validation_summary,
        normalize_validation=normalize_validation_summary,
        baseline_idf=baseline_idf,
        normalize_idf=normalize_idf,
        added_pair_count=len(added_pairs),
        removed_pair_count=len(removed_pairs),
        merge_event_count=merge_events_df.height,
        split_event_count=split_events_df.height,
    )

    # -------------------------------------------------------------------------
    # STEP 8. 결과 저장
    # -------------------------------------------------------------------------
    summary_parquet_path = output_dir / "summary_metrics.parquet"
    summary_csv_path = output_dir / "summary_metrics.csv"
    entity_df_changes_path = output_dir / "entity_df_changes.parquet"
    added_pairs_path = output_dir / "added_article_pairs.parquet"
    removed_pairs_path = output_dir / "removed_article_pairs.parquet"
    merge_events_path = output_dir / "new_merge_events.parquet"
    split_events_path = output_dir / "split_events.parquet"
    normalization_map_path = output_dir / "normalization_map_used.parquet"
    summary_text_path = output_dir / "comparison_summary.txt"

    summary_df.write_parquet(
        summary_parquet_path,
        compression="zstd",
    )
    summary_df.write_csv(summary_csv_path)

    entity_df_changes.write_parquet(
        entity_df_changes_path,
        compression="zstd",
    )
    added_pair_df.write_parquet(
        added_pairs_path,
        compression="zstd",
    )
    removed_pair_df.write_parquet(
        removed_pairs_path,
        compression="zstd",
    )
    merge_events_df.write_parquet(
        merge_events_path,
        compression="zstd",
    )
    split_events_df.write_parquet(
        split_events_path,
        compression="zstd",
    )
    normalization_map_df.write_parquet(
        normalization_map_path,
        compression="zstd",
    )

    _write_summary_text(
        output_path=summary_text_path,
        summary_df=summary_df,
        entity_df_changes=entity_df_changes,
        merge_events=merge_events_df,
        split_events=split_events_df,
    )

    became_high_df_count = entity_df_changes.filter(
        pl.col("change_type") == "BECAME_HIGH_DF"
    ).height

    return {
        "status": "SUCCESS",
        "baseline_dir": str(baseline_dir),
        "normalize_dir": str(normalize_dir),
        "output_dir": str(output_dir),
        "summary_metrics_path": str(summary_parquet_path),
        "comparison_summary_path": str(summary_text_path),
        "entity_df_changes_path": str(entity_df_changes_path),
        "added_article_pairs_path": str(added_pairs_path),
        "removed_article_pairs_path": str(removed_pairs_path),
        "new_merge_events_path": str(merge_events_path),
        "split_events_path": str(split_events_path),
        "entity_vocabulary_baseline": int(len(baseline_idf)),
        "entity_vocabulary_normalize_only": int(len(normalize_idf)),
        "added_cocluster_pair_count": int(len(added_pairs)),
        "removed_cocluster_pair_count": int(len(removed_pairs)),
        "new_merge_event_count": int(merge_events_df.height),
        "split_event_count": int(split_events_df.height),
        "became_high_df_entity_count": int(became_high_df_count),
    }


# =============================================================================
# 14. CLI
# =============================================================================


def _parse_args() -> argparse.Namespace:
    """python -m src.compare_entity_events 실행 옵션 정의."""

    project_root = Path(__file__).resolve().parents[1]

    default_baseline_dir = (
        project_root
        / "data"
        / "output"
        / "experiments"
        / "baseline"
        / "model_inputs"
    )

    default_normalize_dir = (
        project_root
        / "data"
        / "output"
        / "experiments"
        / "normalize_only"
        / "model_inputs"
    )

    default_output_dir = (
        project_root
        / "data"
        / "output"
        / "experiments"
        / "entity_comparison"
        / "baseline_vs_normalize_only"
    )

    parser = argparse.ArgumentParser(
        description=(
            "Baseline Event와 normalize_only Event를 article membership 기준으로 비교합니다."
        )
    )

    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=default_baseline_dir,
        help=(
            "baseline snapshot model_inputs 경로. "
            f"기본값={default_baseline_dir}"
        ),
    )

    parser.add_argument(
        "--normalize-dir",
        type=Path,
        default=default_normalize_dir,
        help=(
            "normalize_only snapshot model_inputs 경로. "
            f"기본값={default_normalize_dir}"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help=(
            "비교 결과 저장 경로. "
            f"기본값={default_output_dir}"
        ),
    )

    return parser.parse_args()



def main() -> None:
    """
    CLI 진입점.

    실행 예:
        python -m src.compare_entity_events
    """

    args = _parse_args()

    print("=" * 80)
    print("Entity Event Comparison 시작")
    print("=" * 80)
    print(f"baseline_dir  = {args.baseline_dir}")
    print(f"normalize_dir = {args.normalize_dir}")
    print(f"output_dir    = {args.output_dir}")
    print()

    result = compare_entity_events(
        baseline_dir=args.baseline_dir,
        normalize_dir=args.normalize_dir,
        output_dir=args.output_dir,
    )

    print("=" * 80)
    print("Entity Event Comparison 완료")
    print("=" * 80)
    pprint(result)

    print()
    print("다음 파일을 먼저 확인하세요:")
    print("1. comparison_summary.txt")
    print("2. summary_metrics.csv")
    print("3. new_merge_events.parquet")
    print("4. added_article_pairs.parquet")
    print("5. split_events.parquet")
    print("6. entity_df_changes.parquet")


if __name__ == "__main__":
    main()
