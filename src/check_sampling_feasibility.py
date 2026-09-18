import argparse
import json

from pathlib import Path
from typing import Any

import polars as pl

from src import config

from src.analyze_candidate_sid_collision import (
    DEFAULT_CHUNK_SIZE,
    PRIMARY_LEVEL,
    PREFIX_LEVELS,
    _explode_candidates,
    _required_candidate_columns,
    _safe_ratio,
    _validate_candidate_columns,
    resolve_semantic_ids_path,
    _load_sid_lookup,
)


# ============================================================
# STEP 14. 1 positive + N negative 샘플링 가능 여부 점검
#
# 목적:
# "각 impression에서 positive 1개 + negative N개를 뽑되
#  N+1개가 모두 서로 다른 SID를 갖도록" 구성할 때
# 몇 개의 impression이 조건을 못 채우는지 미리 센다.
#
# 판정 방법:
# 한 impression에서 (c1,c2,c3) 단위로 묶으면
#   D = negative가 1개 이상 있는 SID 그룹 수
#     = 서로 다른 negative SID의 개수
# 이고, positive 하나를 고르면 그 positive의 SID와 같은
# negative는 쓸 수 없으므로
#   사용 가능한 negative SID 수 = D - (positive의 SID 그룹에 negative가 있으면 1)
# 이 값이 N 이상이면 그 positive로 샘플을 만들 수 있다.
#
# 실패 사유는 두 가지로 나눈다.
#   1. negative 자체가 N개 미만
#   2. negative는 충분하지만 SID 중복/충돌로 서로 다른 SID가 N개 미만
# ============================================================


DEFAULT_NEGATIVE_COUNT = 4


def _feasibility_for_chunk(
    candidate_df: pl.DataFrame,
    sid_columns: list[str],
    negative_count: int,
) -> dict[str, int]:
    """
    chunk 하나의 샘플링 가능 여부를 센다.
    """

    # STEP 14-1. (impression, SID) 그룹별 positive / negative 수
    group_df = (
        candidate_df
        .group_by(["row_index"] + sid_columns)
        .agg([
            (pl.col("candidate_labels") == 1).sum().alias("positive_count"),
            (pl.col("candidate_labels") == 0).sum().alias("negative_count"),
        ])
    )

    # STEP 14-2. impression 단위 요약
    #   negative_total          : negative 총 개수
    #   distinct_negative_sid   : 서로 다른 negative SID 수 (= D)
    #   clean_positive_count    : negative와 SID가 겹치지 않는 positive 수
    #   dirty_positive_count    : negative와 SID가 겹치는 positive 수
    row_df = (
        group_df
        .group_by("row_index")
        .agg([
            pl.col("negative_count").sum().alias("negative_total"),

            (pl.col("negative_count") >= 1)
            .sum()
            .alias("distinct_negative_sid"),

            pl.col("positive_count")
            .filter(pl.col("negative_count") == 0)
            .sum()
            .alias("clean_positive_count"),

            pl.col("positive_count")
            .filter(pl.col("negative_count") >= 1)
            .sum()
            .alias("dirty_positive_count"),
        ])
        .with_columns([
            pl.col("clean_positive_count").fill_null(0),
            pl.col("dirty_positive_count").fill_null(0),
        ])
    )

    # STEP 14-3. positive 종류별 사용 가능한 negative SID 수
    # clean positive : 자기 SID에 negative가 없으므로 D를 그대로 쓴다
    # dirty positive : 자기 SID 그룹 하나를 못 쓰므로 D - 1
    row_df = row_df.with_columns([
        pl.col("distinct_negative_sid").alias("usable_for_clean"),
        (pl.col("distinct_negative_sid") - 1).alias("usable_for_dirty"),
    ])

    # STEP 14-4. impression 단위 판정
    # 하나라도 쓸 수 있는 positive가 있으면 그 impression은 성공이다.
    row_df = row_df.with_columns(
        (
            (
                (pl.col("clean_positive_count") >= 1)
                & (pl.col("usable_for_clean") >= negative_count)
            )
            | (
                (pl.col("dirty_positive_count") >= 1)
                & (pl.col("usable_for_dirty") >= negative_count)
            )
        ).alias("is_feasible")
    )

    feasible_row_count = row_df.filter(pl.col("is_feasible")).height

    # STEP 14-5. 실패 사유 분해
    failed_df = row_df.filter(~pl.col("is_feasible"))

    lack_negative_count = (
        failed_df
        .filter(pl.col("negative_total") < negative_count)
        .height
    )

    lack_distinct_sid_count = (
        failed_df.height
        - lack_negative_count
    )

    # STEP 14-6. positive 단위 판정
    # positive마다 별도 샘플로 나눌 때 몇 개의 샘플을 만들 수 있는지.
    sample_count = int(
        row_df
        .select(
            (
                pl.when(pl.col("usable_for_clean") >= negative_count)
                .then(pl.col("clean_positive_count"))
                .otherwise(0)
                + pl.when(pl.col("usable_for_dirty") >= negative_count)
                .then(pl.col("dirty_positive_count"))
                .otherwise(0)
            ).sum()
        )
        .item()
        or 0
    )

    positive_total = int(
        row_df
        .select(
            (
                pl.col("clean_positive_count")
                + pl.col("dirty_positive_count")
            ).sum()
        )
        .item()
        or 0
    )

    return {
        "row_count": row_df.height,
        "feasible_row_count": feasible_row_count,
        "lack_negative_count": lack_negative_count,
        "lack_distinct_sid_count": lack_distinct_sid_count,
        "positive_total": positive_total,
        "feasible_sample_count": sample_count,
    }


def check_sampling_feasibility(
    sequences_path: Path,
    split_name: str,
    negative_count: int = DEFAULT_NEGATIVE_COUNT,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    limit: int | None = None,
    semantic_ids_path: Path | None = None,
    level_name: str = PRIMARY_LEVEL,
) -> dict[str, Any]:
    """
    sequences parquet 하나에 대해
    1 positive + negative_count개 샘플링이 가능한 impression 수를 센다.
    """

    sid_columns = dict(PREFIX_LEVELS).get(level_name)

    if sid_columns is None:
        raise ValueError(
            f"알 수 없는 level: {level_name}. "
            "사용 가능: " + ", ".join(name for name, _ in PREFIX_LEVELS)
        )

    use_external_sid = semantic_ids_path is not None

    if use_external_sid:
        semantic_ids_path = resolve_semantic_ids_path(semantic_ids_path)
        sid_lookup_df = _load_sid_lookup(semantic_ids_path)
    else:
        sid_lookup_df = None

    read_columns = _required_candidate_columns(use_external_sid)

    lazy_frame = pl.scan_parquet(sequences_path)

    _validate_candidate_columns(
        lazy_frame.collect_schema().names(),
        use_external_sid=use_external_sid,
    )

    total_row_count = lazy_frame.select(pl.len()).collect().item()

    if limit is not None:
        total_row_count = min(total_row_count, limit)

    totals = {
        "row_count": 0,
        "feasible_row_count": 0,
        "lack_negative_count": 0,
        "lack_distinct_sid_count": 0,
        "positive_total": 0,
        "feasible_sample_count": 0,
    }

    for offset in range(0, total_row_count, chunk_size):
        current_chunk_size = min(chunk_size, total_row_count - offset)

        chunk_df = (
            lazy_frame
            .slice(offset, current_chunk_size)
            .select(read_columns)
            .collect()
        )

        candidate_df = _explode_candidates(chunk_df, read_columns)

        if use_external_sid:
            candidate_df = (
                candidate_df
                .with_columns(
                    pl.col("candidate_article_ids").cast(pl.Int64)
                )
                .join(sid_lookup_df, on="candidate_article_ids", how="left")
                .filter(pl.col("candidate_c1").is_not_null())
            )

        chunk_result = _feasibility_for_chunk(
            candidate_df,
            sid_columns,
            negative_count,
        )

        for key in totals:
            totals[key] += chunk_result[key]

    failed_row_count = (
        totals["row_count"]
        - totals["feasible_row_count"]
    )

    return {
        "split_name": split_name,
        "sequences_path": str(sequences_path),
        "semantic_ids_path": (
            str(semantic_ids_path) if use_external_sid else None
        ),
        "level": level_name,
        "negative_count": negative_count,
        "impression_count": totals["row_count"],
        "feasible_impression_count": totals["feasible_row_count"],
        "feasible_impression_ratio": _safe_ratio(
            totals["feasible_row_count"],
            totals["row_count"],
        ),
        "failed_impression_count": failed_row_count,
        "failed_impression_ratio": _safe_ratio(
            failed_row_count,
            totals["row_count"],
        ),
        # 실패 사유 1 : negative 자체가 모자람
        "lack_negative_count": totals["lack_negative_count"],
        # 실패 사유 2 : SID 중복/충돌로 서로 다른 SID가 모자람
        "lack_distinct_sid_count": totals["lack_distinct_sid_count"],
        "positive_count": totals["positive_total"],
        # positive마다 별도 샘플로 나눌 때 만들 수 있는 샘플 수
        "feasible_sample_count": totals["feasible_sample_count"],
        "feasible_sample_ratio": _safe_ratio(
            totals["feasible_sample_count"],
            totals["positive_total"],
        ),
    }


def _print_result(result: dict[str, Any]) -> None:
    n = result["negative_count"]

    print()
    print("=" * 70)
    print(
        f"[{result['split_name']}] "
        f"1 positive + {n} negative 샘플링 가능 여부 "
        f"({result['level']} 기준)"
    )
    print("=" * 70)

    print(f"파일                 : {result['sequences_path']}")

    if result["semantic_ids_path"]:
        print(f"SID 출처             : {result['semantic_ids_path']}")

    print(f"impression 수        : {result['impression_count']:,}")
    print(
        "구성 가능            : "
        f"{result['feasible_impression_count']:,} "
        f"({result['feasible_impression_ratio']:.4%})"
    )
    print(
        "구성 불가            : "
        f"{result['failed_impression_count']:,} "
        f"({result['failed_impression_ratio']:.4%})"
    )

    print()
    print("  실패 사유")
    print(
        f"    negative가 {n}개 미만            : "
        f"{result['lack_negative_count']:>8,}"
    )
    print(
        f"    서로 다른 SID가 {n}개 미만       : "
        f"{result['lack_distinct_sid_count']:>8,}"
    )

    print()
    print(
        "  positive마다 별도 샘플로 나눌 때 : "
        f"{result['feasible_sample_count']:,} 샘플 "
        f"(전체 positive {result['positive_count']:,}개 중 "
        f"{result['feasible_sample_ratio']:.4%})"
    )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "각 impression에서 1 positive + N negative를 "
            "모두 다른 SID로 뽑을 수 있는지 점검한다."
        )
    )

    parser.add_argument("--path", type=Path, action="append", default=None)

    parser.add_argument(
        "--negatives",
        type=int,
        default=DEFAULT_NEGATIVE_COUNT,
        help=f"뽑을 negative 개수. 기본값 {DEFAULT_NEGATIVE_COUNT}",
    )

    parser.add_argument(
        "--level",
        type=str,
        default=PRIMARY_LEVEL,
        help=(
            "SID 비교 기준. "
            + ", ".join(name for name, _ in PREFIX_LEVELS)
            + f". 기본값 {PRIMARY_LEVEL}"
        ),
    )

    parser.add_argument("--semantic-ids", type=Path, default=None)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--report", type=Path, default=None)

    args = parser.parse_args()

    if args.path:
        targets = [(path.stem, path) for path in args.path]
    else:
        targets = [
            ("train", config.TRAIN_SEQUENCES_PATH),
            ("validation", config.VALIDATION_SEQUENCES_PATH),
        ]

    results = []

    for split_name, sequences_path in targets:
        result = check_sampling_feasibility(
            sequences_path,
            split_name,
            negative_count=args.negatives,
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
