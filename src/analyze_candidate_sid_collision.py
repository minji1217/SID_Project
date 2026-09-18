import argparse
import json

from pathlib import Path
from typing import Any

import polars as pl

from src import config


# ============================================================
# STEP 13. Candidate SID 충돌 분석
#
# 목적:
# Transformer 입력용으로 만든 sequences parquet에서
# 같은 impression 안의 candidate들이
# SID만으로 서로 구분되는지 확인한다.
#
# 왜 필요한가:
# Transformer는 candidate를 article_id가 아니라 SID로 본다.
# 그래서 negative의 SID가 positive의 SID와 같으면
# 모델 입력만 보고는 두 candidate를 구분할 수 없고,
# 그 negative는 학습에서 사실상 label noise가 된다.
#
# 한 impression에 positive가 여러 개일 수 있으므로
# 세 가지 관점으로 나눠서 본다.
#
#   1. negative 기준
#      negative 중 positive와 SID가 겹치는 비율
#
#   2. positive 기준
#      positive 하나하나가 몇 개의 negative와 겹치는지
#      (positive가 여러 개면 오염 정도가 서로 다르다)
#
#   3. candidate 전체 기준
#      candidate list 자체가 SID로 얼마나 구분되는지
#
# 입력:
#   train_sequences.parquet
#   validation_sequences.parquet
#   (candidate_article_ids / candidate_c1~c4 / candidate_labels 필요)
# ============================================================


# 한 번에 메모리에 올릴 impression 수.
# candidate list를 펼치면 행 수가 수십 배로 늘어나므로
# 파일 전체를 올리지 않고 이 단위로 나눠서 누적한다.
DEFAULT_CHUNK_SIZE = 100_000


CANDIDATE_COLUMNS = [
    "candidate_article_ids",
    "candidate_c1",
    "candidate_c2",
    "candidate_c3",
    "candidate_c4",
    "candidate_labels",
]


# prefix level 정의
# (c1,c2,c3)가 기본 분석 대상이고,
# 나머지는 비교용으로 같이 계산한다.
PREFIX_LEVELS: list[tuple[str, list[str]]] = [
    ("c1", ["candidate_c1"]),
    ("c1c2", ["candidate_c1", "candidate_c2"]),
    ("c1c2c3", ["candidate_c1", "candidate_c2", "candidate_c3"]),
    (
        "c1c2c3c4",
        [
            "candidate_c1",
            "candidate_c2",
            "candidate_c3",
            "candidate_c4",
        ],
    ),
]


# positive별 충돌 개수 분포를 자세히 볼 level
PRIMARY_LEVEL = "c1c2c3"


# positive 하나가 몇 개의 negative와 겹치는지에 대한 분포 구간
COLLISION_BUCKETS = [
    ("0", 0, 0),
    ("1", 1, 1),
    ("2-4", 2, 4),
    ("5-9", 5, 9),
    ("10+", 10, None),
]


def _validate_candidate_columns(
    column_names: list[str],
) -> None:
    """
    candidate 분석에 필요한 컬럼이 모두 있는지 검사한다.
    """

    missing_columns = [
        column_name
        for column_name in CANDIDATE_COLUMNS
        if column_name not in column_names
    ]

    if missing_columns:
        raise ValueError(
            "candidate 컬럼이 없습니다: "
            + ", ".join(missing_columns)
        )


def _explode_candidates(
    sequence_df: pl.DataFrame,
) -> pl.DataFrame:
    """
    impression 단위 list 컬럼을 candidate 단위 long format으로 편다.

    한 행 = 하나의 candidate
    row_index = 원래 impression 식별자
    """

    _validate_candidate_columns(sequence_df.columns)

    return (
        sequence_df
        .select(CANDIDATE_COLUMNS)
        .with_row_index("row_index")
        .explode(CANDIDATE_COLUMNS)
    )


def _new_level_counter() -> dict[str, Any]:
    """
    prefix level 하나에 대한 누적 counter를 만든다.

    chunk 단위로 값을 더해도 결과가 같도록
    비율이 아니라 원시 count만 누적한다.
    """

    return {
        # --- negative 기준 ---
        "negative_count": 0,
        "collided_negative_count": 0,
        "row_with_negative_count": 0,
        "row_with_collision_count": 0,
        # impression별 충돌 negative 비율의 합 (macro 평균용)
        "row_ratio_sum": 0.0,

        # --- positive 기준 ---
        "positive_count": 0,
        # 충돌 negative를 하나라도 가진 positive 수
        "positive_with_collision_count": 0,
        # (positive, 충돌 negative) 쌍의 총 개수
        "positive_negative_pair_count": 0,
        # positive 하나가 겪는 최대 충돌 negative 수
        "positive_collision_max": 0,
        # positive별 (충돌 negative / 그 행의 negative 총수) 합
        "positive_ratio_sum": 0.0,
        # 다른 positive와 SID가 같아진 positive 수
        "positive_in_positive_collision_count": 0,
        # positive별 충돌 개수 분포
        "collision_histogram": {
            bucket_name: 0
            for bucket_name, _, _ in COLLISION_BUCKETS
        },

        # --- candidate 전체 기준 ---
        "candidate_count": 0,
        # 같은 SID를 가진 candidate가 2개 이상인 그룹에 속한 candidate 수
        "ambiguous_candidate_count": 0,
        # 서로 다른 SID의 개수 (candidate list의 SID 해상도)
        "distinct_sid_count": 0,
    }


def _bucket_expression() -> pl.Expr:
    """
    충돌 negative 개수를 분포 구간 이름으로 바꾸는 식.
    """

    expression = pl.when(pl.col("negative_count") == 0).then(
        pl.lit("0")
    )

    for bucket_name, lower_bound, upper_bound in COLLISION_BUCKETS[1:]:
        if upper_bound is None:
            expression = expression.when(
                pl.col("negative_count") >= lower_bound
            ).then(pl.lit(bucket_name))
        else:
            expression = expression.when(
                pl.col("negative_count") <= upper_bound
            ).then(pl.lit(bucket_name))

    return expression.otherwise(pl.lit("10+"))


def _accumulate_level(
    counter: dict[str, Any],
    candidate_df: pl.DataFrame,
    sid_columns: list[str],
) -> None:
    """
    chunk 하나의 충돌 결과를 counter에 더한다.

    핵심 아이디어:
    (impression, SID prefix) 단위로 묶으면
    한 그룹 안의 positive 수와 negative 수만으로
    negative 기준 / positive 기준 통계를 모두 만들 수 있다.

    예) 한 그룹에 positive 2개, negative 3개가 있으면
        - 충돌 negative는 3개 (positive가 몇 개든 중복으로 세지 않는다)
        - positive 2개는 각각 negative 3개와 충돌한다
        - 그 positive 2개는 서로도 SID가 같다

    chunk는 impression 단위로 자르므로
    한 impression의 candidate가 두 chunk로 쪼개지지 않는다.
    """

    # STEP 13-1. (impression, SID) 그룹 집계
    group_df = (
        candidate_df
        .group_by(["row_index"] + sid_columns)
        .agg([
            (pl.col("candidate_labels") == 1)
            .sum()
            .alias("positive_count"),

            (pl.col("candidate_labels") == 0)
            .sum()
            .alias("negative_count"),
        ])
        .with_columns(
            (
                pl.col("positive_count")
                + pl.col("negative_count")
            ).alias("candidate_count")
        )
    )

    # STEP 13-2. impression 단위 집계
    # 충돌 negative는 positive가 있는 그룹의 negative 전부다.
    row_df = (
        group_df
        .group_by("row_index")
        .agg([
            pl.col("negative_count")
            .sum()
            .alias("row_negative_count"),

            pl.col("negative_count")
            .filter(pl.col("positive_count") > 0)
            .sum()
            .alias("row_collided_negative_count"),
        ])
        .with_columns(
            pl.col("row_collided_negative_count").fill_null(0)
        )
    )

    row_with_negative_df = row_df.filter(
        pl.col("row_negative_count") > 0
    )

    counter["negative_count"] += int(
        row_df.get_column("row_negative_count").sum()
    )

    counter["collided_negative_count"] += int(
        row_df.get_column("row_collided_negative_count").sum()
    )

    counter["row_with_negative_count"] += row_with_negative_df.height

    counter["row_with_collision_count"] += (
        row_df
        .filter(pl.col("row_collided_negative_count") > 0)
        .height
    )

    counter["row_ratio_sum"] += float(
        row_with_negative_df
        .select(
            (
                pl.col("row_collided_negative_count")
                / pl.col("row_negative_count")
            ).sum()
        )
        .item()
    )

    # STEP 13-3. positive 기준 집계
    # 그룹의 negative 수가 곧 그 그룹에 속한 positive 각각의 충돌 개수다.
    positive_group_df = (
        group_df
        .filter(pl.col("positive_count") > 0)
        .join(
            row_df.select(["row_index", "row_negative_count"]),
            on="row_index",
            how="left",
        )
    )

    counter["positive_count"] += int(
        positive_group_df.get_column("positive_count").sum()
    )

    counter["positive_negative_pair_count"] += int(
        positive_group_df
        .select(
            (
                pl.col("positive_count")
                * pl.col("negative_count")
            ).sum()
        )
        .item()
    )

    counter["positive_with_collision_count"] += int(
        positive_group_df
        .filter(pl.col("negative_count") > 0)
        .get_column("positive_count")
        .sum()
    )

    chunk_collision_max = (
        positive_group_df
        .get_column("negative_count")
        .max()
    )

    if chunk_collision_max is not None:
        counter["positive_collision_max"] = max(
            counter["positive_collision_max"],
            int(chunk_collision_max),
        )

    # positive 하나가 그 행의 negative 중 몇 %와 겹치는지
    counter["positive_ratio_sum"] += float(
        positive_group_df
        .filter(pl.col("row_negative_count") > 0)
        .select(
            (
                pl.col("positive_count")
                * pl.col("negative_count")
                / pl.col("row_negative_count")
            ).sum()
        )
        .item()
        or 0.0
    )

    # positive끼리 SID가 같아진 경우
    counter["positive_in_positive_collision_count"] += int(
        positive_group_df
        .filter(pl.col("positive_count") >= 2)
        .get_column("positive_count")
        .sum()
    )

    # STEP 13-4. positive별 충돌 개수 분포
    histogram_df = (
        positive_group_df
        .group_by(_bucket_expression().alias("bucket"))
        .agg(
            pl.col("positive_count").sum().alias("count")
        )
    )

    for bucket_name, bucket_count in histogram_df.iter_rows():
        counter["collision_histogram"][bucket_name] += int(
            bucket_count
        )

    # STEP 13-5. candidate 전체 기준 집계
    counter["candidate_count"] += candidate_df.height

    counter["ambiguous_candidate_count"] += int(
        group_df
        .filter(pl.col("candidate_count") >= 2)
        .get_column("candidate_count")
        .sum()
    )

    counter["distinct_sid_count"] += group_df.height


def _safe_ratio(
    numerator: float,
    denominator: float,
) -> float:
    if denominator <= 0:
        return 0.0

    return numerator / denominator


def _finalize_level(
    counter: dict[str, Any],
) -> dict[str, Any]:
    """
    누적 counter를 최종 비율로 바꾼다.
    """

    negative_count = int(counter["negative_count"])
    positive_count = int(counter["positive_count"])
    candidate_count = int(counter["candidate_count"])

    return {
        # --- negative 기준 ---
        "negative_count": negative_count,
        "collided_negative_count": int(
            counter["collided_negative_count"]
        ),
        # micro: negative 전체를 한 덩어리로 보고 계산한 비율
        "collided_negative_ratio_micro": _safe_ratio(
            counter["collided_negative_count"],
            negative_count,
        ),
        # macro: impression마다 비율을 구한 뒤 평균
        "collided_negative_ratio_macro": _safe_ratio(
            counter["row_ratio_sum"],
            counter["row_with_negative_count"],
        ),
        "row_with_negative_count": int(
            counter["row_with_negative_count"]
        ),
        "row_with_collision_count": int(
            counter["row_with_collision_count"]
        ),
        "row_with_collision_ratio": _safe_ratio(
            counter["row_with_collision_count"],
            counter["row_with_negative_count"],
        ),

        # --- positive 기준 ---
        "positive_count": positive_count,
        "positive_with_collision_count": int(
            counter["positive_with_collision_count"]
        ),
        # 충돌 negative를 하나라도 가진 positive 비율
        "positive_with_collision_ratio": _safe_ratio(
            counter["positive_with_collision_count"],
            positive_count,
        ),
        "positive_negative_pair_count": int(
            counter["positive_negative_pair_count"]
        ),
        # positive 하나당 평균 충돌 negative 개수
        "collision_per_positive_mean": _safe_ratio(
            counter["positive_negative_pair_count"],
            positive_count,
        ),
        "collision_per_positive_max": int(
            counter["positive_collision_max"]
        ),
        # positive 하나가 같은 행 negative 중 평균 몇 %와 겹치는지
        "collision_ratio_per_positive_mean": _safe_ratio(
            counter["positive_ratio_sum"],
            positive_count,
        ),
        "positive_in_positive_collision_count": int(
            counter["positive_in_positive_collision_count"]
        ),
        "positive_in_positive_collision_ratio": _safe_ratio(
            counter["positive_in_positive_collision_count"],
            positive_count,
        ),
        "collision_histogram": dict(
            counter["collision_histogram"]
        ),

        # --- candidate 전체 기준 ---
        "candidate_count": candidate_count,
        "ambiguous_candidate_count": int(
            counter["ambiguous_candidate_count"]
        ),
        # SID가 같은 candidate가 2개 이상인 그룹에 속한 candidate 비율
        "ambiguous_candidate_ratio": _safe_ratio(
            counter["ambiguous_candidate_count"],
            candidate_count,
        ),
        "distinct_sid_count": int(counter["distinct_sid_count"]),
        # candidate 대비 고유 SID 비율 (1.0이면 완전히 구분됨)
        "distinct_sid_ratio": _safe_ratio(
            counter["distinct_sid_count"],
            candidate_count,
        ),
    }


def analyze_candidate_sid_collision(
    sequences_path: Path,
    split_name: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    limit: int | None = None,
) -> dict[str, Any]:
    """
    sequences parquet 하나를 읽어서
    prefix level별 candidate SID 충돌 통계를 만든다.

    파일 전체를 한 번에 올리지 않고
    impression chunk 단위로 읽어서 누적하므로
    candidate 수가 많아도 메모리 사용량이 일정하다.
    """

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size는 1 이상의 정수여야 합니다."
        )

    lazy_frame = pl.scan_parquet(sequences_path)

    _validate_candidate_columns(lazy_frame.collect_schema().names())

    total_row_count = (
        lazy_frame
        .select(pl.len())
        .collect()
        .item()
    )

    if limit is not None:
        total_row_count = min(total_row_count, limit)

    impression_count = 0
    total_candidate_count = 0
    positive_count = 0
    negative_count = 0
    same_article_negative_count = 0
    multi_positive_row_count = 0

    level_counters = {
        level_name: _new_level_counter()
        for level_name, _ in PREFIX_LEVELS
    }

    for offset in range(0, total_row_count, chunk_size):
        current_chunk_size = min(
            chunk_size,
            total_row_count - offset,
        )

        chunk_df = (
            lazy_frame
            .slice(offset, current_chunk_size)
            .select(CANDIDATE_COLUMNS)
            .collect()
        )

        candidate_df = _explode_candidates(chunk_df)

        impression_count += chunk_df.height
        total_candidate_count += candidate_df.height

        positive_df = candidate_df.filter(
            pl.col("candidate_labels") == 1
        )

        negative_df = candidate_df.filter(
            pl.col("candidate_labels") == 0
        )

        positive_count += positive_df.height
        negative_count += negative_df.height

        # positive가 2개 이상인 impression 수
        multi_positive_row_count += (
            chunk_df
            .filter(pl.col("candidate_labels").list.sum() >= 2)
            .height
        )

        # STEP 13-6. article_id 자체가 겹치는 경우
        # SID 해상도 문제가 아니라 candidate 생성 문제이므로 따로 센다.
        same_article_negative_count += (
            negative_df
            .select(["row_index", "candidate_article_ids"])
            .join(
                positive_df
                .select(["row_index", "candidate_article_ids"])
                .unique(),
                on=["row_index", "candidate_article_ids"],
                how="semi",
            )
            .height
        )

        # STEP 13-7. prefix level별 충돌 누적
        for level_name, sid_columns in PREFIX_LEVELS:
            _accumulate_level(
                level_counters[level_name],
                candidate_df,
                sid_columns,
            )

    return {
        "split_name": split_name,
        "sequences_path": str(sequences_path),
        "impression_count": impression_count,
        "multi_positive_row_count": multi_positive_row_count,
        "multi_positive_row_ratio": _safe_ratio(
            multi_positive_row_count,
            impression_count,
        ),
        "total_candidate_count": total_candidate_count,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "positive_per_row_mean": _safe_ratio(
            positive_count,
            impression_count,
        ),
        "same_article_negative_count": same_article_negative_count,
        "levels": {
            level_name: _finalize_level(level_counters[level_name])
            for level_name, _ in PREFIX_LEVELS
        },
    }


def _print_result(
    result: dict[str, Any],
) -> None:
    """
    분석 결과를 터미널에서 보기 쉽게 출력한다.
    """

    print()
    print("=" * 78)
    print(f"[{result['split_name']}] candidate SID 충돌 분석")
    print("=" * 78)

    print(f"파일                 : {result['sequences_path']}")
    print(f"impression 수        : {result['impression_count']:,}")
    print(f"candidate 총 개수    : {result['total_candidate_count']:,}")
    print(
        f"positive 개수        : {result['positive_count']:,} "
        f"(impression당 평균 {result['positive_per_row_mean']:.2f}개)"
    )
    print(
        f"positive 2개 이상 행 : {result['multi_positive_row_count']:,} "
        f"({result['multi_positive_row_ratio']:.2%})"
    )
    print(f"negative 개수        : {result['negative_count']:,}")
    print(
        "positive와 article_id가 같은 negative : "
        f"{result['same_article_negative_count']:,}"
    )

    # --- negative 기준 ---
    print()
    print("[1] negative 기준 : negative 중 positive와 SID가 겹치는 비율")
    print(
        f"{'level':<10}{'충돌 negative':>15}{'micro':>11}"
        f"{'macro':>11}{'충돌 포함 행':>15}"
    )
    print("-" * 62)

    for level_name, _ in PREFIX_LEVELS:
        level = result["levels"][level_name]

        print(
            f"{level_name:<10}"
            f"{level['collided_negative_count']:>15,}"
            f"{level['collided_negative_ratio_micro']:>11.4%}"
            f"{level['collided_negative_ratio_macro']:>11.4%}"
            f"{level['row_with_collision_ratio']:>15.4%}"
        )

    # --- positive 기준 ---
    print()
    print("[2] positive 기준 : positive 하나하나가 얼마나 오염됐는지")
    print(
        f"{'level':<10}{'오염 positive':>15}{'비율':>10}"
        f"{'평균 충돌수':>13}{'최대':>8}{'평균 충돌비율':>15}"
    )
    print("-" * 71)

    for level_name, _ in PREFIX_LEVELS:
        level = result["levels"][level_name]

        print(
            f"{level_name:<10}"
            f"{level['positive_with_collision_count']:>15,}"
            f"{level['positive_with_collision_ratio']:>10.4%}"
            f"{level['collision_per_positive_mean']:>13.3f}"
            f"{level['collision_per_positive_max']:>8,}"
            f"{level['collision_ratio_per_positive_mean']:>15.4%}"
        )

    print()
    print(
        "  * 오염 positive : 같은 SID를 가진 negative가 "
        "1개 이상인 positive"
    )
    print(
        "  * 평균 충돌수   : positive 하나당 겹치는 negative 개수"
    )
    print(
        "  * 평균 충돌비율 : positive 하나가 그 행의 negative 중 "
        "몇 %와 겹치는지"
    )

    # --- candidate 전체 기준 ---
    print()
    print("[3] candidate 전체 기준 : candidate list가 SID로 구분되는 정도")
    print(
        f"{'level':<10}{'구분불가 candidate':>22}{'비율':>10}"
        f"{'고유 SID 비율':>16}{'positive끼리 충돌':>20}"
    )
    print("-" * 78)

    for level_name, _ in PREFIX_LEVELS:
        level = result["levels"][level_name]

        print(
            f"{level_name:<10}"
            f"{level['ambiguous_candidate_count']:>22,}"
            f"{level['ambiguous_candidate_ratio']:>10.4%}"
            f"{level['distinct_sid_ratio']:>16.4%}"
            f"{level['positive_in_positive_collision_count']:>20,}"
        )

    print()
    print(
        "  * 구분불가 candidate : 같은 SID를 가진 candidate가 "
        "2개 이상인 그룹에 속한 candidate (label 무관)"
    )
    print(
        "  * 고유 SID 비율      : 서로 다른 SID 수 / candidate 수 "
        "(100%면 완전히 구분됨)"
    )

    # --- positive별 충돌 분포 ---
    primary_level = result["levels"][PRIMARY_LEVEL]

    print()
    print(f"[4] positive별 충돌 negative 개수 분포 ({PRIMARY_LEVEL})")

    positive_count = primary_level["positive_count"]

    for bucket_name, _, _ in COLLISION_BUCKETS:
        bucket_count = primary_level["collision_histogram"][bucket_name]

        print(
            f"  충돌 {bucket_name:<5} : "
            f"{bucket_count:>12,} "
            f"({_safe_ratio(bucket_count, positive_count):>8.4%})"
        )

    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Transformer 입력 sequences parquet에서 "
            "positive와 SID가 완전히 겹치는 negative 비율을 계산한다."
        )
    )

    parser.add_argument(
        "--path",
        type=Path,
        action="append",
        default=None,
        help=(
            "분석할 sequences parquet 경로. "
            "여러 번 지정 가능. "
            "생략하면 config의 train/validation sequences를 사용한다."
        ),
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=(
            "한 번에 처리할 impression 수. "
            "메모리가 부족하면 줄인다. "
            f"기본값 {DEFAULT_CHUNK_SIZE:,}"
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "앞에서부터 이 개수의 impression만 분석한다. "
            "빠른 확인용."
        ),
    )

    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="결과를 저장할 JSON 경로. 생략하면 저장하지 않는다.",
    )

    args = parser.parse_args()

    if args.path:
        targets = [
            (path.stem, path)
            for path in args.path
        ]
    else:
        targets = [
            ("train", config.TRAIN_SEQUENCES_PATH),
            ("validation", config.VALIDATION_SEQUENCES_PATH),
        ]

    results = []

    for split_name, sequences_path in targets:
        result = analyze_candidate_sid_collision(
            sequences_path,
            split_name,
            chunk_size=args.chunk_size,
            limit=args.limit,
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
