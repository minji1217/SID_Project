from __future__ import annotations

"""
Entity Linking 후보 조사기
==========================

Safe Normalization v2가 끝난 canonical entity를 대상으로
문자열은 다르지만 같은 실제 개체일 가능성이 있는 후보를 찾는다.

중요:
- 이 파일은 실제 linking을 적용하지 않는다.
- Train-used article만 후보 생성에 사용한다.
- 후보를 발견했다고 해서 같은 개체로 확정하지 않는다.
- 사람이 SAFE / AMBIGUOUS / REJECT 검토하기 위한 evidence를 저장한다.

현재 찾는 후보
1) PER_FULL_TO_SURNAME
   vladimir putin <-> putin
   kevin magnussen <-> magnussen

2) ORG_FULL_TO_ACRONYM
   fc københavn <-> fck
   danmarks radio <-> dr

실행:
    python -m src.inspect_entity_linking_candidates

기본 입력:
    data/output/experiments/normalize_v2/model_inputs/article_entities.parquet
    data/output/experiments/normalize_v2/model_inputs/articles_base.parquet
    data/output/experiments/normalize_v2/model_inputs/article_events.parquet

기본 출력:
    data/output/experiments/entity_linking_inspection/
"""

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

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

DEFAULT_ARTICLE_ENTITIES_PATH = DEFAULT_SOURCE_DIR / "article_entities.parquet"
DEFAULT_ARTICLES_BASE_PATH = DEFAULT_SOURCE_DIR / "articles_base.parquet"
DEFAULT_ARTICLE_EVENTS_PATH = DEFAULT_SOURCE_DIR / "article_events.parquet"

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "experiments"
    / "entity_linking_inspection"
)

DEFAULT_MAX_TITLE_EXAMPLES = 5
DEFAULT_MAX_CONTEXT_EXAMPLES = 10


# =============================================================================
# 2. 공통 검사 / 로드
# =============================================================================


def _require_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} 파일이 없습니다. 경로={path}")


def _require_columns(
    df: pl.DataFrame,
    required_columns: Iterable[str],
    description: str,
) -> None:
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(
            f"{description}에 필요한 컬럼이 없습니다: "
            + ", ".join(sorted(missing))
        )


def _load_train_entities(path: Path) -> pl.DataFrame:
    """
    normalize_v2 article_entities에서 Train-used mention만 읽는다.

    사용하는 컬럼:
      article_id
      entity_group
      canonical_entity
      canonical_entity_key
      is_train_used
    """
    _require_file(path, "normalize_v2 article_entities")
    df = pl.read_parquet(path)

    _require_columns(
        df,
        [
            "article_id",
            "entity_group",
            "canonical_entity",
            "canonical_entity_key",
            "is_train_used",
        ],
        "article_entities.parquet",
    )

    return (
        df.filter(pl.col("is_train_used"))
        .select(
            [
                pl.col("article_id").cast(pl.Int64),
                pl.col("entity_group").cast(pl.String),
                pl.col("canonical_entity").cast(pl.String),
                pl.col("canonical_entity_key").cast(pl.String),
            ]
        )
    )


def _load_title_lookup(path: Path) -> dict[int, str]:
    """article_id -> title"""
    _require_file(path, "articles_base")
    df = pl.read_parquet(path)
    _require_columns(df, ["article_id", "title"], "articles_base.parquet")

    return {
        int(article_id): str(title or "")
        for article_id, title in df.select(["article_id", "title"]).iter_rows()
    }


def _load_article_event_lookup(path: Path) -> dict[int, int]:
    """Train article_id -> event_id"""
    _require_file(path, "article_events")
    df = pl.read_parquet(path)
    _require_columns(df, ["article_id", "event_id"], "article_events.parquet")

    return {
        int(article_id): int(event_id)
        for article_id, event_id in df.select(["article_id", "event_id"]).iter_rows()
    }


# =============================================================================
# 3. 기본 통계
# =============================================================================


def _build_statistics(
    train_entities: pl.DataFrame,
    article_event_lookup: dict[int, int],
) -> tuple[
    dict[str, set[int]],
    Counter[str],
    dict[int, set[str]],
    dict[str, set[int]],
    dict[str, set[str]],
]:
    """
    반환:
    - entity_articles: entity_key -> article_id set
    - entity_mentions: entity_key -> mention count
    - article_entities: article_id -> entity_key set
    - entity_events: entity_key -> event_id set
    - vocab_by_group: PER/ORG/... -> canonical surface set
    """
    entity_articles: dict[str, set[int]] = defaultdict(set)
    entity_mentions: Counter[str] = Counter()
    article_entities: dict[int, set[str]] = defaultdict(set)
    vocab_by_group: dict[str, set[str]] = defaultdict(set)

    for row in train_entities.iter_rows(named=True):
        article_id = int(row["article_id"])
        group = str(row["entity_group"])
        surface = str(row["canonical_entity"])
        key = str(row["canonical_entity_key"])

        entity_articles[key].add(article_id)
        entity_mentions[key] += 1
        article_entities[article_id].add(key)
        vocab_by_group[group].add(surface)

    entity_events: dict[str, set[int]] = defaultdict(set)

    for key, article_ids in entity_articles.items():
        for article_id in article_ids:
            event_id = article_event_lookup.get(article_id)
            if event_id is not None:
                entity_events[key].add(event_id)

    return (
        dict(entity_articles),
        entity_mentions,
        dict(article_entities),
        dict(entity_events),
        dict(vocab_by_group),
    )


# =============================================================================
# 4. PER 후보: full name <-> surname
# =============================================================================


def _discover_per_candidates(per_vocab: set[str]) -> list[tuple[str, str]]:
    """
    multi-token PER의 마지막 token이 one-token PER로 Train에 실제 존재하면 후보.

    예:
      "vladimir putin" -> "putin"

    주의:
      후보일 뿐 동일 인물 확정이 아니다.
    """
    rows: list[tuple[str, str]] = []

    for full_name in sorted(per_vocab):
        tokens = [token for token in full_name.split() if token]
        if len(tokens) < 2:
            continue

        short_name = tokens[-1]
        if short_name not in per_vocab:
            continue

        rows.append((full_name, short_name))

    return rows


# =============================================================================
# 5. ORG 후보: full organization <-> acronym
# =============================================================================


_ORG_STOPWORDS = {
    "a",
    "af",
    "and",
    "for",
    "i",
    "of",
    "og",
    "the",
    "til",
}


def _clean_token(token: str) -> str:
    return "".join(char for char in token if char.isalnum()).lower()


def _org_acronym(full_name: str) -> str | None:
    """
    ORG full name에서 diagnostic acronym을 만든다.

    일반 단어는 첫 글자 사용.
    3글자 이하의 이미 짧은 token은 token 전체를 유지.

    예:
      danmarks radio -> dr
      fc københavn   -> fc + k -> fck
    """
    tokens = [
        cleaned
        for token in full_name.split()
        if (cleaned := _clean_token(token))
        and cleaned not in _ORG_STOPWORDS
    ]

    if len(tokens) < 2:
        return None

    parts: list[str] = []

    for token in tokens:
        if len(token) <= 3 and (token.isalpha() or token.isdigit()):
            parts.append(token)
        else:
            parts.append(token[0])

    acronym = "".join(parts)

    if len(acronym) < 2 or acronym == full_name:
        return None

    return acronym


def _discover_org_candidates(org_vocab: set[str]) -> list[tuple[str, str]]:
    """
    diagnostic acronym이 같은 ORG Train vocabulary에 실제 존재하면 후보.
    """
    rows: list[tuple[str, str]] = []

    for full_name in sorted(org_vocab):
        short_name = _org_acronym(full_name)
        if short_name is None:
            continue
        if short_name not in org_vocab:
            continue

        rows.append((full_name, short_name))

    return rows


# =============================================================================
# 6. 후보 ambiguity
# =============================================================================


def _build_short_to_longs(
    per_candidates: list[tuple[str, str]],
    org_candidates: list[tuple[str, str]],
) -> dict[tuple[str, str, str], list[str]]:
    """
    하나의 short 표현에 long 후보가 몇 개 붙는지 계산한다.

    예:
      hansen -> [anders hansen, mikkel hansen, ...]

    여러 개면 자동 linking에 특히 위험할 수 있다.
    """
    result: dict[tuple[str, str, str], list[str]] = defaultdict(list)

    for long_name, short_name in per_candidates:
        result[("PER_FULL_TO_SURNAME", "PER", short_name)].append(long_name)

    for long_name, short_name in org_candidates:
        result[("ORG_FULL_TO_ACRONYM", "ORG", short_name)].append(long_name)

    return {
        key: sorted(set(values))
        for key, values in result.items()
    }


# =============================================================================
# 7. 검토 evidence
# =============================================================================


def _title_examples(
    article_ids: set[int],
    title_lookup: dict[int, str],
    limit: int,
) -> list[str]:
    result: list[str] = []

    for article_id in sorted(article_ids):
        title = title_lookup.get(article_id, "").strip()
        if not title:
            continue

        result.append(title)
        if len(result) >= limit:
            break

    return result


def _context_counter(
    entity_key: str,
    entity_articles: dict[str, set[int]],
    article_entities: dict[int, set[str]],
) -> Counter[str]:
    """
    entity가 나온 기사들에서 같이 등장한 다른 entity 빈도를 센다.
    """
    counter: Counter[str] = Counter()

    for article_id in entity_articles.get(entity_key, set()):
        for other_key in article_entities.get(article_id, set()):
            if other_key != entity_key:
                counter[other_key] += 1

    return counter


def _top_context(counter: Counter[str], limit: int) -> list[str]:
    return [
        f"{entity} ({count})"
        for entity, count in counter.most_common(limit)
    ]


def _build_candidate_row(
    *,
    candidate_type: str,
    entity_group: str,
    long_surface: str,
    short_surface: str,
    competing_longs: list[str],
    entity_articles: dict[str, set[int]],
    entity_mentions: Counter[str],
    article_entities: dict[int, set[str]],
    entity_events: dict[str, set[int]],
    title_lookup: dict[int, str],
    max_title_examples: int,
    max_context_examples: int,
) -> dict[str, Any]:
    """
    후보 한 쌍에 사람이 판정하기 위한 evidence를 붙인다.
    """
    long_key = f"{entity_group}::{long_surface}"
    short_key = f"{entity_group}::{short_surface}"

    long_articles = entity_articles.get(long_key, set())
    short_articles = entity_articles.get(short_key, set())
    same_articles = long_articles & short_articles

    long_events = entity_events.get(long_key, set())
    short_events = entity_events.get(short_key, set())
    shared_events = long_events & short_events

    long_context = _context_counter(long_key, entity_articles, article_entities)
    short_context = _context_counter(short_key, entity_articles, article_entities)

    long_context_set = set(long_context)
    short_context_set = set(short_context)
    shared_context = long_context_set & short_context_set
    union_context = long_context_set | short_context_set

    context_jaccard = (
        len(shared_context) / len(union_context)
        if union_context
        else 0.0
    )

    shared_context_ranked = sorted(
        shared_context,
        key=lambda key: (
            -(long_context[key] + short_context[key]),
            key,
        ),
    )[:max_context_examples]

    # 동일 entity 확률 점수가 아니다.
    # 사람이 먼저 볼 후보를 정렬하기 위한 evidence priority일 뿐이다.
    ambiguity_penalty = max(len(competing_longs) - 1, 0)
    review_priority_score = (
        3.0 * len(same_articles)
        + 1.5 * len(shared_events)
        + 5.0 * context_jaccard
        - 2.0 * ambiguity_penalty
    )

    return {
        "candidate_type": candidate_type,
        "entity_group": entity_group,
        "long_entity": long_surface,
        "short_entity": short_surface,
        "long_entity_key": long_key,
        "short_entity_key": short_key,
        "long_article_df": len(long_articles),
        "short_article_df": len(short_articles),
        "long_mention_count": int(entity_mentions.get(long_key, 0)),
        "short_mention_count": int(entity_mentions.get(short_key, 0)),
        "same_article_cooccurrence_count": len(same_articles),
        "long_event_count": len(long_events),
        "short_event_count": len(short_events),
        "shared_event_count": len(shared_events),
        "shared_event_ids": sorted(shared_events)[:10],
        "short_candidate_long_count": len(competing_longs),
        "competing_long_entities": competing_longs,
        "shared_context_entity_count": len(shared_context),
        "context_entity_jaccard": float(context_jaccard),
        "shared_context_entities": shared_context_ranked,
        "long_top_context_entities": _top_context(long_context, max_context_examples),
        "short_top_context_entities": _top_context(short_context, max_context_examples),
        "long_title_examples": _title_examples(
            long_articles, title_lookup, max_title_examples
        ),
        "short_title_examples": _title_examples(
            short_articles, title_lookup, max_title_examples
        ),
        "same_article_title_examples": _title_examples(
            same_articles, title_lookup, max_title_examples
        ),
        "review_priority_score": float(review_priority_score),
        "manual_decision": "",
        "manual_reason": "",
    }


# =============================================================================
# 8. DataFrame 생성
# =============================================================================


def _candidate_schema() -> dict[str, pl.DataType]:
    return {
        "candidate_id": pl.String,
        "candidate_type": pl.String,
        "entity_group": pl.String,
        "long_entity": pl.String,
        "short_entity": pl.String,
        "long_entity_key": pl.String,
        "short_entity_key": pl.String,
        "long_article_df": pl.Int64,
        "short_article_df": pl.Int64,
        "long_mention_count": pl.Int64,
        "short_mention_count": pl.Int64,
        "same_article_cooccurrence_count": pl.Int64,
        "long_event_count": pl.Int64,
        "short_event_count": pl.Int64,
        "shared_event_count": pl.Int64,
        "shared_event_ids": pl.List(pl.Int64),
        "short_candidate_long_count": pl.Int64,
        "competing_long_entities": pl.List(pl.String),
        "shared_context_entity_count": pl.Int64,
        "context_entity_jaccard": pl.Float64,
        "shared_context_entities": pl.List(pl.String),
        "long_top_context_entities": pl.List(pl.String),
        "short_top_context_entities": pl.List(pl.String),
        "long_title_examples": pl.List(pl.String),
        "short_title_examples": pl.List(pl.String),
        "same_article_title_examples": pl.List(pl.String),
        "review_priority_score": pl.Float64,
        "manual_decision": pl.String,
        "manual_reason": pl.String,
    }


def _rows_to_candidate_df(rows: list[dict[str, Any]]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(schema=_candidate_schema())

    rows = sorted(
        rows,
        key=lambda row: (
            -float(row["review_priority_score"]),
            row["candidate_type"],
            row["short_entity"],
            row["long_entity"],
        ),
    )

    type_counters: Counter[str] = Counter()

    for row in rows:
        candidate_type = row["candidate_type"]
        type_counters[candidate_type] += 1
        prefix = "PER" if candidate_type == "PER_FULL_TO_SURNAME" else "ORG"
        row["candidate_id"] = (
            f"LINK-{prefix}-{type_counters[candidate_type]:04d}"
        )

    return pl.DataFrame(rows, schema=_candidate_schema())


def _build_ambiguity_df(
    short_to_longs: dict[tuple[str, str, str], list[str]],
) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []

    for (candidate_type, group, short_name), long_names in sorted(
        short_to_longs.items()
    ):
        if len(long_names) <= 1:
            continue

        rows.append(
            {
                "candidate_type": candidate_type,
                "entity_group": group,
                "short_entity": short_name,
                "candidate_long_count": len(long_names),
                "candidate_long_entities": long_names,
            }
        )

    schema = {
        "candidate_type": pl.String,
        "entity_group": pl.String,
        "short_entity": pl.String,
        "candidate_long_count": pl.Int64,
        "candidate_long_entities": pl.List(pl.String),
    }

    if not rows:
        return pl.DataFrame(schema=schema)

    return pl.DataFrame(rows, schema=schema).sort(
        ["candidate_long_count", "candidate_type", "short_entity"],
        descending=[True, False, False],
    )


# =============================================================================
# 9. Summary
# =============================================================================


def _write_summary(
    output_path: Path,
    *,
    article_entities_path: Path,
    train_mention_count: int,
    unique_train_entity_count: int,
    per_df: pl.DataFrame,
    org_df: pl.DataFrame,
    all_df: pl.DataFrame,
    ambiguity_df: pl.DataFrame,
) -> None:
    same_article_positive = (
        all_df.filter(pl.col("same_article_cooccurrence_count") > 0).height
        if all_df.height
        else 0
    )

    shared_event_positive = (
        all_df.filter(pl.col("shared_event_count") > 0).height
        if all_df.height
        else 0
    )

    unambiguous_candidates = (
        all_df.filter(pl.col("short_candidate_long_count") == 1).height
        if all_df.height
        else 0
    )

    lines = [
        "=" * 80,
        "Entity Linking Candidate Inspection",
        "=" * 80,
        "",
        f"source_article_entities={article_entities_path}",
        "fit_split=train_used_articles_only",
        "source_representation=normalize_v2_canonical_entity",
        "",
        f"train_entity_mention_count={train_mention_count}",
        f"unique_train_entity_count={unique_train_entity_count}",
        "",
        f"per_full_to_surname_candidate_count={per_df.height}",
        f"org_full_to_acronym_candidate_count={org_df.height}",
        f"total_candidate_count={all_df.height}",
        f"ambiguous_short_entity_count={ambiguity_df.height}",
        f"unambiguous_candidate_count={unambiguous_candidates}",
        f"same_article_positive_candidate_count={same_article_positive}",
        f"shared_event_positive_candidate_count={shared_event_positive}",
        "",
        "주의:",
        "- 아직 어떤 후보도 자동 linking하지 않는다.",
        "- short_candidate_long_count > 1이면 특히 위험 후보로 본다.",
        "- same_article/shared_event/context overlap은 evidence이지 ground truth가 아니다.",
        "- 후보 검토 후 SAFE/AMBIGUOUS/REJECT 정책을 정한 뒤 실제 linking을 구현한다.",
    ]

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# =============================================================================
# 10. 전체 실행
# =============================================================================


def inspect_entity_linking_candidates(
    *,
    article_entities_path: Path,
    articles_base_path: Path,
    article_events_path: Path,
    output_dir: Path,
    max_title_examples: int = DEFAULT_MAX_TITLE_EXAMPLES,
    max_context_examples: int = DEFAULT_MAX_CONTEXT_EXAMPLES,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    train_entities = _load_train_entities(article_entities_path)
    title_lookup = _load_title_lookup(articles_base_path)
    article_event_lookup = _load_article_event_lookup(article_events_path)

    (
        entity_articles,
        entity_mentions,
        article_entities,
        entity_events,
        vocab_by_group,
    ) = _build_statistics(train_entities, article_event_lookup)

    per_candidates = _discover_per_candidates(vocab_by_group.get("PER", set()))
    org_candidates = _discover_org_candidates(vocab_by_group.get("ORG", set()))

    short_to_longs = _build_short_to_longs(per_candidates, org_candidates)

    rows: list[dict[str, Any]] = []

    for long_name, short_name in per_candidates:
        rows.append(
            _build_candidate_row(
                candidate_type="PER_FULL_TO_SURNAME",
                entity_group="PER",
                long_surface=long_name,
                short_surface=short_name,
                competing_longs=short_to_longs[
                    ("PER_FULL_TO_SURNAME", "PER", short_name)
                ],
                entity_articles=entity_articles,
                entity_mentions=entity_mentions,
                article_entities=article_entities,
                entity_events=entity_events,
                title_lookup=title_lookup,
                max_title_examples=max_title_examples,
                max_context_examples=max_context_examples,
            )
        )

    for long_name, short_name in org_candidates:
        rows.append(
            _build_candidate_row(
                candidate_type="ORG_FULL_TO_ACRONYM",
                entity_group="ORG",
                long_surface=long_name,
                short_surface=short_name,
                competing_longs=short_to_longs[
                    ("ORG_FULL_TO_ACRONYM", "ORG", short_name)
                ],
                entity_articles=entity_articles,
                entity_mentions=entity_mentions,
                article_entities=article_entities,
                entity_events=entity_events,
                title_lookup=title_lookup,
                max_title_examples=max_title_examples,
                max_context_examples=max_context_examples,
            )
        )

    all_df = _rows_to_candidate_df(rows)

    if all_df.height:
        per_df = all_df.filter(
            pl.col("candidate_type") == "PER_FULL_TO_SURNAME"
        )
        org_df = all_df.filter(
            pl.col("candidate_type") == "ORG_FULL_TO_ACRONYM"
        )
    else:
        per_df = pl.DataFrame(schema=_candidate_schema())
        org_df = pl.DataFrame(schema=_candidate_schema())

    ambiguity_df = _build_ambiguity_df(short_to_longs)

    summary_path = output_dir / "entity_linking_summary.txt"
    all_path = output_dir / "entity_linking_candidates.parquet"
    per_path = output_dir / "per_name_candidates.parquet"
    org_path = output_dir / "org_acronym_candidates.parquet"
    ambiguity_path = output_dir / "short_entity_ambiguity.parquet"

    all_df.write_parquet(all_path, compression="zstd")
    per_df.write_parquet(per_path, compression="zstd")
    org_df.write_parquet(org_path, compression="zstd")
    ambiguity_df.write_parquet(ambiguity_path, compression="zstd")

    _write_summary(
        summary_path,
        article_entities_path=article_entities_path,
        train_mention_count=train_entities.height,
        unique_train_entity_count=len(entity_articles),
        per_df=per_df,
        org_df=org_df,
        all_df=all_df,
        ambiguity_df=ambiguity_df,
    )

    return {
        "status": "SUCCESS",
        "fit_split": "train_used_articles_only",
        "source_representation": "normalize_v2_canonical_entity",
        "train_entity_mention_count": int(train_entities.height),
        "unique_train_entity_count": int(len(entity_articles)),
        "per_full_to_surname_candidate_count": int(per_df.height),
        "org_full_to_acronym_candidate_count": int(org_df.height),
        "total_candidate_count": int(all_df.height),
        "ambiguous_short_entity_count": int(ambiguity_df.height),
        "summary_path": str(summary_path),
        "all_candidates_path": str(all_path),
        "per_candidates_path": str(per_path),
        "org_candidates_path": str(org_path),
        "ambiguity_path": str(ambiguity_path),
    }


# =============================================================================
# 11. CLI
# =============================================================================


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "normalize_v2 Train canonical entity에서 "
            "Entity Linking 후보를 진단합니다."
        )
    )

    parser.add_argument(
        "--article-entities",
        type=Path,
        default=DEFAULT_ARTICLE_ENTITIES_PATH,
    )
    parser.add_argument(
        "--articles-base",
        type=Path,
        default=DEFAULT_ARTICLES_BASE_PATH,
    )
    parser.add_argument(
        "--article-events",
        type=Path,
        default=DEFAULT_ARTICLE_EVENTS_PATH,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--max-title-examples",
        type=int,
        default=DEFAULT_MAX_TITLE_EXAMPLES,
    )
    parser.add_argument(
        "--max-context-examples",
        type=int,
        default=DEFAULT_MAX_CONTEXT_EXAMPLES,
    )

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.max_title_examples < 0:
        raise ValueError("--max-title-examples는 0 이상이어야 합니다.")
    if args.max_context_examples < 0:
        raise ValueError("--max-context-examples는 0 이상이어야 합니다.")

    print("=" * 80)
    print("Entity Linking Candidate Inspection 시작")
    print("=" * 80)
    print(f"article_entities = {args.article_entities}")
    print(f"articles_base    = {args.articles_base}")
    print(f"article_events   = {args.article_events}")
    print(f"output_dir       = {args.output_dir}")
    print()

    result = inspect_entity_linking_candidates(
        article_entities_path=args.article_entities,
        articles_base_path=args.articles_base,
        article_events_path=args.article_events,
        output_dir=args.output_dir,
        max_title_examples=args.max_title_examples,
        max_context_examples=args.max_context_examples,
    )

    print("=" * 80)
    print("Entity Linking Candidate Inspection 완료")
    print("=" * 80)

    for key, value in result.items():
        print(f"{key}: {value}")

    print()
    print("다음 파일을 먼저 확인하세요:")
    print("1. entity_linking_summary.txt")
    print("2. entity_linking_candidates.parquet")
    print("3. per_name_candidates.parquet")
    print("4. org_acronym_candidates.parquet")
    print("5. short_entity_ambiguity.parquet")


if __name__ == "__main__":
    main()