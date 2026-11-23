import argparse
import json

from pathlib import Path
from typing import Any

import polars as pl

from src import config

from src.analyze_candidate_sid_collision import (
    PREFIX_LEVELS,
    PRIMARY_LEVEL,
    resolve_semantic_ids_path,
    _load_sid_lookup,
    _safe_ratio,
)


# ============================================================
# STEP 15. 1 positive + N negative sequences 생성
#
# 목적:
# 기존 sequences parquet을 읽어
# "impression당 positive 1개 + negative N개, N+1개가 모두 서로 다른 SID"
# 형태의 학습 데이터를 만든다.
#
# 확정된 정책 (2026-09-18 교수님 회신):
#   1. positive가 2개 이상인 impression은 positive마다 별도 row로 분리한다.
#      분리된 row들이 서로 다른 negative를 갖도록 강제하지 않는다.
#   2. 조건을 못 채우는 row는 제외한다.
#   3. negative 비율은 1:4로 고정한다.
#
# 처리 순서 (positive 하나당):
#   negative 전체 (label == 0)
#     -> positive와 (c1,c2,c3)가 같은 negative 제거
#     -> negative끼리 (c1,c2,c3) 중복 제거 (그룹당 1개만 남김)
#     -> N개 미만이면 drop
#     -> 남은 것 중 seed 기반으로 N개 추출
#     -> [P, N1, ..., Nn]
#
# 중요:
# negative 후보는 반드시 label == 0인 것만 쓴다.
# 같은 impression의 다른 positive는 실제로 클릭된 기사이므로
# negative로 넣으면 안 된다.
#
# 출력:
#   기존 sequences와 동일한 컬럼 구성이며
#   target_* 는 그 row의 positive 하나만 담는다.
# ============================================================


DEFAULT_NEGATIVE_COUNT = 4
DEFAULT_SEED = 42
DEFAULT_CHUNK_SIZE = 20_000


HISTORY_COLUMNS = [
    "impression_id",
    "user_id",
    "impression_time",
    "history_article_ids",
    "history_c1",
    "history_c2",
    "history_c3",
    "history_c4",
]


CANDIDATE_COLUMNS = [
    "candidate_article_ids",
    "candidate_c1",
    "candidate_c2",
    "candidate_c3",
    "candidate_c4",
    "candidate_labels",
]


SID_SUFFIXES = ["c1", "c2", "c3", "c4"]


def _build_chunk(
    chunk_df: pl.DataFrame,
    sid_columns: list[str],
    negative_count: int,
    seed: int,
    offset: int,
) -> tuple[pl.DataFrame, dict[str, int]]:
    """
    chunk 하나를 1 positive + N negative 형태로 바꾼다.

    반환: (결과 DataFrame, 통계 dict)
    """

    # STEP 15-1. candidate를 long format으로 펼친다
    long_df = (
        chunk_df
        .select(CANDIDATE_COLUMNS)
        .with_row_index("row_index", offset=offset)
        .explode(CANDIDATE_COLUMNS, empty_as_null=False)
    )

    # STEP 15-2. positive / negative 분리
    # negative는 반드시 label == 0인 것만 쓴다.
    positive_df = (
        long_df
        .filter(pl.col("candidate_labels") == 1)
        .drop("candidate_labels")
        .with_row_index("sample_id")
    )

    negative_df = (
        long_df
        .filter(pl.col("candidate_labels") == 0)
        .drop("candidate_labels")
    )

    positive_total = positive_df.height

    # STEP 15-3. (positive, negative) 쌍을 만든다
    pair_df = positive_df.join(
        negative_df,
        on="row_index",
        how="inner",
        suffix="_neg",
    )

    # STEP 15-4. positive와 SID가 같은 negative 제거
    same_sid_expression = pl.lit(True)

    for column_name in sid_columns:
        same_sid_expression = same_sid_expression & (
            pl.col(column_name) == pl.col(f"{column_name}_neg")
        )

    pair_df = pair_df.filter(~same_sid_expression)

    # STEP 15-5. 재현 가능한 난수 부여
    # (전역 row_index, positive article_id, negative article_id)의 해시를
    # 우선순위로 쓴다. chunk 크기를 바꿔도 같은 값이 나온다.
    pair_df = pair_df.with_columns(
        pl.struct([
            "row_index",
            "candidate_article_ids",
            "candidate_article_ids_neg",
        ])
        .hash(seed=seed)
        .alias("priority")
    )

    # STEP 15-6. negative끼리 SID 중복 제거
    # 같은 SID 그룹에서는 priority가 가장 작은 하나만 남긴다.
    negative_sid_columns = [
        f"{column_name}_neg" for column_name in sid_columns
    ]

    pair_df = (
        pair_df
        .sort("priority")
        .unique(
            subset=["sample_id"] + negative_sid_columns,
            keep="first",
            maintain_order=True,
        )
    )

    # STEP 15-7. negative가 N개 미만인 sample 제외
    usable_df = (
        pair_df
        .group_by("sample_id")
        .len()
        .filter(pl.col("len") >= negative_count)
        .select("sample_id")
    )

    feasible_sample_count = usable_df.height

    pair_df = pair_df.join(usable_df, on="sample_id", how="semi")

    # STEP 15-8. sample마다 N개 추출
    picked_df = (
        pair_df
        .sort(["sample_id", "priority"])
        .group_by("sample_id", maintain_order=True)
        .head(negative_count)
    )

    # STEP 15-9. negative를 list로 묶는다
    negative_list_df = (
        picked_df
        .group_by("sample_id", maintain_order=True)
        .agg([
            pl.col("candidate_article_ids_neg").alias("neg_article_ids"),
            *[
                pl.col(f"candidate_{suffix}_neg").alias(f"neg_{suffix}")
                for suffix in SID_SUFFIXES
            ],
        ])
    )

    # STEP 15-10. positive를 앞에 붙여 최종 candidate list를 만든다
    kept_positive_df = positive_df.join(
        usable_df,
        on="sample_id",
        how="semi",
    )

    sample_df = (
        kept_positive_df
        .join(negative_list_df, on="sample_id", how="inner")
        .with_columns([
            pl.concat_list([
                pl.col("candidate_article_ids"),
                pl.col("neg_article_ids"),
            ]).alias("candidate_article_ids"),

            *[
                pl.concat_list([
                    pl.col(f"candidate_{suffix}"),
                    pl.col(f"neg_{suffix}"),
                ]).alias(f"candidate_{suffix}")
                for suffix in SID_SUFFIXES
            ],
        ])
        .with_columns(
            pl.concat_list([
                pl.lit(1, dtype=pl.Int32),
                *[pl.lit(0, dtype=pl.Int32)] * negative_count,
            ]).alias("candidate_labels")
        )
    )

    # STEP 15-11. history 컬럼을 다시 붙이고 target을 그 positive 하나로 둔다
    history_df = (
        chunk_df
        .select(HISTORY_COLUMNS)
        .with_row_index("row_index", offset=offset)
    )

    result_df = (
        sample_df
        .join(history_df, on="row_index", how="inner")
        .with_columns([
            pl.concat_list([
                pl.col("candidate_article_ids").list.first()
            ]).alias("target_article_ids"),

            *[
                pl.concat_list([
                    pl.col(f"candidate_{suffix}").list.first()
                ]).alias(f"target_{suffix}")
                for suffix in SID_SUFFIXES
            ],
        ])
        .select(HISTORY_COLUMNS + [
            "target_article_ids",
            *[f"target_{suffix}" for suffix in SID_SUFFIXES],
            *CANDIDATE_COLUMNS,
        ])
    )

    stats = {
        "impression_count": chunk_df.height,
        "positive_count": positive_total,
        "feasible_sample_count": feasible_sample_count,
        "output_row_count": result_df.height,
    }

    return result_df, stats


def build_1pos_n_neg(
    sequences_path: Path,
    output_path: Path,
    split_name: str,
    negative_count: int = DEFAULT_NEGATIVE_COUNT,
    seed: int = DEFAULT_SEED,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    limit: int | None = None,
    semantic_ids_path: Path | None = None,
    level_name: str = PRIMARY_LEVEL,
) -> dict[str, Any]:
    """
    sequences parquet 하나를 1 positive + N negative 형태로 변환해 저장한다.
    """

    sid_columns = dict(PREFIX_LEVELS).get(level_name)

    if sid_columns is None:
        raise ValueError(f"알 수 없는 level: {level_name}")

    use_external_sid = semantic_ids_path is not None

    if use_external_sid:
        semantic_ids_path = resolve_semantic_ids_path(semantic_ids_path)
        sid_lookup_df = _load_sid_lookup(semantic_ids_path)

    lazy_frame = pl.scan_parquet(sequences_path)

    schema_names = lazy_frame.collect_schema().names()

    missing_columns = [
        column_name
        for column_name in HISTORY_COLUMNS + CANDIDATE_COLUMNS
        if column_name not in schema_names
    ]

    if missing_columns:
        raise ValueError(
            "필요한 컬럼이 없습니다: " + ", ".join(missing_columns)
        )

    total_row_count = lazy_frame.select(pl.len()).collect().item()

    if limit is not None:
        total_row_count = min(total_row_count, limit)

    totals = {
        "impression_count": 0,
        "positive_count": 0,
        "feasible_sample_count": 0,
        "output_row_count": 0,
    }

    output_frames = []

    for offset in range(0, total_row_count, chunk_size):
        current_chunk_size = min(chunk_size, total_row_count - offset)

        chunk_df = (
            lazy_frame
            .slice(offset, current_chunk_size)
            .select(HISTORY_COLUMNS + CANDIDATE_COLUMNS)
            .collect()
        )

        # 외부 SID를 쓰는 경우 candidate SID를 교체한다.
        if use_external_sid:
            chunk_df = _replace_candidate_sid(chunk_df, sid_lookup_df)

        result_df, stats = _build_chunk(
            chunk_df,
            sid_columns,
            negative_count,
            seed,
            offset,
        )

        output_frames.append(result_df)

        for key in totals:
            totals[key] += stats[key]

    output_df = pl.concat(output_frames, how="vertical")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.write_parquet(output_path)

    dropped = totals["positive_count"] - totals["output_row_count"]

    return {
        "split_name": split_name,
        "sequences_path": str(sequences_path),
        "output_path": str(output_path),
        "semantic_ids_path": (
            str(semantic_ids_path) if use_external_sid else None
        ),
        "level": level_name,
        "negative_count": negative_count,
        "seed": seed,
        "impression_count": totals["impression_count"],
        "positive_count": totals["positive_count"],
        "output_row_count": totals["output_row_count"],
        "dropped_positive_count": dropped,
        "dropped_positive_ratio": _safe_ratio(
            dropped,
            totals["positive_count"],
        ),
        "candidate_position_count": (
            totals["output_row_count"] * (negative_count + 1)
        ),
    }


def _replace_candidate_sid(
    chunk_df: pl.DataFrame,
    sid_lookup_df: pl.DataFrame,
) -> pl.DataFrame:
    """
    candidate SID를 외부 article_semantic_ids.parquet 값으로 교체한다.
    """

    long_df = (
        chunk_df
        .select(["candidate_article_ids", "candidate_labels"])
        .with_row_index("row_index")
        .explode(
            ["candidate_article_ids", "candidate_labels"],
            empty_as_null=False,
        )
        .with_columns(pl.col("candidate_article_ids").cast(pl.Int64))
        .join(sid_lookup_df, on="candidate_article_ids", how="left")
        .filter(pl.col("candidate_c1").is_not_null())
        .group_by("row_index", maintain_order=True)
        .agg([
            pl.col("candidate_article_ids"),
            *[pl.col(f"candidate_{suffix}") for suffix in SID_SUFFIXES],
            pl.col("candidate_labels"),
        ])
    )

    return (
        chunk_df
        .drop(CANDIDATE_COLUMNS)
        .with_row_index("row_index")
        .join(long_df, on="row_index", how="inner")
        .drop("row_index")
    )


def _print_result(result: dict[str, Any]) -> None:
    n = result["negative_count"]

    print()
    print("=" * 70)
    print(f"[{result['split_name']}] 1 positive + {n} negative 생성")
    print("=" * 70)

    print(f"입력                 : {result['sequences_path']}")
    print(f"출력                 : {result['output_path']}")
    print(f"seed                 : {result['seed']}")
    print(f"impression 수        : {result['impression_count']:,}")
    print(f"positive 수          : {result['positive_count']:,}")
    print(f"생성된 학습 row      : {result['output_row_count']:,}")
    print(
        "탈락 positive        : "
        f"{result['dropped_positive_count']:,} "
        f"({result['dropped_positive_ratio']:.4%})"
    )
    print(
        "candidate position   : "
        f"{result['candidate_position_count']:,} "
        f"(= row × {n + 1})"
    )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "sequences parquet을 1 positive + N negative 형태로 변환한다. "
            "N+1개가 모두 서로 다른 SID를 갖는다."
        )
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="출력 폴더. train/validation parquet이 이 안에 생성된다.",
    )

    parser.add_argument("--path", type=Path, action="append", default=None)
    parser.add_argument("--semantic-ids", type=Path, default=None)
    parser.add_argument(
        "--negatives", type=int, default=DEFAULT_NEGATIVE_COUNT
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--level", type=str, default=PRIMARY_LEVEL)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--report", type=Path, default=None)

    args = parser.parse_args()

    if args.path:
        targets = [(path.stem, path) for path in args.path]
    else:
        targets = [
            ("train_sequences", config.TRAIN_SEQUENCES_PATH),
            ("validation_sequences", config.VALIDATION_SEQUENCES_PATH),
        ]

    results = []

    for split_name, sequences_path in targets:
        output_path = (
            args.out_dir
            / f"{split_name}_1pos{args.negatives}neg.parquet"
        )

        result = build_1pos_n_neg(
            sequences_path,
            output_path,
            split_name,
            negative_count=args.negatives,
            seed=args.seed,
            chunk_size=args.chunk_size,
            limit=args.limit,
            semantic_ids_path=args.semantic_ids,
            level_name=args.level,
        )

        _print_result(result)
        results.append(result)

    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(results, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"리포트 저장 : {args.report}")


if __name__ == "__main__":
    main()
