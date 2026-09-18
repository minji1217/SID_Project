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
# 같은 impression 안의 negative candidate(label=0)가
# positive candidate(label=1)와 SID가 완전히 같아지는 비율을 구한다.
#
# 왜 필요한가:
# Transformer는 candidate를 article_id가 아니라 SID로 본다.
# 그래서 negative의 (c1,c2,c3)가 positive의 (c1,c2,c3)와 같으면
# 모델 입력만 보고는 두 candidate를 구분할 수 없고,
# 그 negative는 학습에서 사실상 label noise가 된다.
#
# 입력:
#   train_sequences.parquet
#   validation_sequences.parquet
#   (candidate_article_ids / candidate_c1~c4 / candidate_labels 필요)
#
# 출력:
#   prefix level별 충돌 통계 (c1 / c1c2 / c1c2c3 / c1c2c3c4)
#   - micro 비율  : 전체 negative 중 충돌 negative 비율
#   - macro 비율  : impression별 충돌 비율의 평균
#   - row 비율    : 충돌 negative를 하나라도 가진 impression 비율
# ============================================================


# 한 번에 메모리에 올릴 impression 수.
# candidate list를 펼치면 행 수가 수십 배로 늘어나므로
# 파일 전체를 올리지 않고 이 단위로 나눠서 누적한다.
DEFAULT_CHUNK_SIZE = 200_000


CANDIDATE_COLUMNS = [
    "candidate_article_ids",
    "candidate_c1",
    "candidate_c2",
    "candidate_c3",
    "candidate_c4",
    "candidate_labels",
]


# prefix level 정의
# (c1,c2,c3)가 사용자가 요청한 기본 분석 대상이고,
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


def _new_level_counter() -> dict[str, float]:
    """
    prefix level 하나에 대한 누적 counter를 만든다.

    chunk 단위로 값을 더해도 결과가 같도록
    비율이 아니라 원시 count만 누적한다.
    """

    return {
        "negative_count": 0,
        "collided_negative_count": 0,
        "row_with_negative_count": 0,
        "row_with_collision_count": 0,
        # impression별 충돌 비율의 합 (macro 평균 계산용)
        "ratio_sum": 0.0,
    }


def _accumulate_level(
    counter: dict[str, float],
    candidate_df: pl.DataFrame,
    sid_columns: list[str],
) -> None:
    """
    chunk 하나의 충돌 결과를 counter에 더한다.

    충돌 정의:
    같은 impression 안에서
    negative candidate의 SID prefix가
    positive candidate의 SID prefix 중 하나와 완전히 같은 경우.

    chunk는 impression 단위로 자르므로
    한 impression의 candidate가 두 chunk로 쪼개지지 않는다.
    """

    # STEP 13-1. positive SID 집합
    # 한 impression에 positive가 여러 개일 수 있으므로 unique로 줄인다.
    positive_sid_df = (
        candidate_df
        .filter(pl.col("candidate_labels") == 1)
        .select(["row_index"] + sid_columns)
        .unique()
    )

    negative_df = (
        candidate_df
        .filter(pl.col("candidate_labels") == 0)
        .select(["row_index"] + sid_columns)
    )

    # STEP 13-2. negative에 충돌 여부 플래그 부여
    flagged_df = (
        negative_df
        .join(
            positive_sid_df.with_columns(
                pl.lit(True).alias("is_collided")
            ),
            on=["row_index"] + sid_columns,
            how="left",
        )
        .with_columns(
            pl.col("is_collided").fill_null(False)
        )
    )

    # STEP 13-3. impression(row)별 비율
    # negative가 0개인 impression은 비율을 정의할 수 없으므로 제외된다.
    per_row_df = (
        flagged_df
        .group_by("row_index")
        .agg([
            pl.len().alias("negative_count"),
            pl.col("is_collided").sum().alias("collided_count"),
        ])
        .with_columns(
            (
                pl.col("collided_count")
                / pl.col("negative_count")
            ).alias("collided_ratio")
        )
    )

    counter["negative_count"] += negative_df.height

    counter["collided_negative_count"] += int(
        flagged_df.get_column("is_collided").sum()
    )

    counter["row_with_negative_count"] += per_row_df.height

    counter["row_with_collision_count"] += (
        per_row_df
        .filter(pl.col("collided_count") > 0)
        .height
    )

    counter["ratio_sum"] += float(
        per_row_df.get_column("collided_ratio").sum()
    )


def _finalize_level(
    counter: dict[str, float],
) -> dict[str, Any]:
    """
    누적 counter를 최종 비율로 바꾼다.
    """

    negative_count = int(counter["negative_count"])
    collided_count = int(counter["collided_negative_count"])
    row_with_negative_count = int(counter["row_with_negative_count"])
    row_with_collision_count = int(counter["row_with_collision_count"])

    return {
        "negative_count": negative_count,
        "collided_negative_count": collided_count,
        # micro: negative 전체를 한 덩어리로 보고 계산한 비율
        "collided_negative_ratio_micro": (
            collided_count / negative_count
            if negative_count > 0
            else 0.0
        ),
        # macro: impression마다 비율을 구한 뒤 평균
        "collided_negative_ratio_macro": (
            counter["ratio_sum"] / row_with_negative_count
            if row_with_negative_count > 0
            else 0.0
        ),
        "row_with_negative_count": row_with_negative_count,
        "row_with_collision_count": row_with_collision_count,
        "row_with_collision_ratio": (
            row_with_collision_count / row_with_negative_count
            if row_with_negative_count > 0
            else 0.0
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

    # STEP 13-4. 누적 counter 준비
    impression_count = 0
    total_candidate_count = 0
    positive_count = 0
    negative_count = 0
    same_article_negative_count = 0

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

        positive_count += (
            candidate_df
            .filter(pl.col("candidate_labels") == 1)
            .height
        )

        negative_count += (
            candidate_df
            .filter(pl.col("candidate_labels") == 0)
            .height
        )

        # STEP 13-5. article_id 자체가 겹치는 경우
        # SID 해상도 문제가 아니라 candidate 생성 문제이므로 따로 센다.
        positive_article_df = (
            candidate_df
            .filter(pl.col("candidate_labels") == 1)
            .select(["row_index", "candidate_article_ids"])
            .unique()
        )

        same_article_negative_count += (
            candidate_df
            .filter(pl.col("candidate_labels") == 0)
            .select(["row_index", "candidate_article_ids"])
            .join(
                positive_article_df,
                on=["row_index", "candidate_article_ids"],
                how="semi",
            )
            .height
        )

        # STEP 13-6. prefix level별 충돌 누적
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
        "total_candidate_count": total_candidate_count,
        "positive_count": positive_count,
        "negative_count": negative_count,
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
    print("=" * 70)
    print(f"[{result['split_name']}] candidate SID 충돌 분석")
    print("=" * 70)

    print(f"파일                 : {result['sequences_path']}")
    print(f"impression 수        : {result['impression_count']:,}")
    print(f"candidate 총 개수    : {result['total_candidate_count']:,}")
    print(f"positive 개수        : {result['positive_count']:,}")
    print(f"negative 개수        : {result['negative_count']:,}")
    print(
        "positive와 article_id가 같은 negative 개수 : "
        f"{result['same_article_negative_count']:,}"
    )

    print()
    print(
        f"{'level':<10}"
        f"{'충돌 negative':>16}"
        f"{'micro 비율':>14}"
        f"{'macro 비율':>14}"
        f"{'충돌 포함 행 비율':>20}"
    )
    print("-" * 74)

    for level_name, _ in PREFIX_LEVELS:
        level_result = result["levels"][level_name]

        print(
            f"{level_name:<10}"
            f"{level_result['collided_negative_count']:>16,}"
            f"{level_result['collided_negative_ratio_micro']:>14.4%}"
            f"{level_result['collided_negative_ratio_macro']:>14.4%}"
            f"{level_result['row_with_collision_ratio']:>20.4%}"
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
