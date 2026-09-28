"""1pos4neg parquet의 candidate 순서만 섞는다.

왜 필요한가
    build_1pos4neg_sequences.py의 STEP 15-10이 positive를 항상 맨 앞에 붙이고
    label을 [1, 0, 0, 0, 0] 리터럴로 만든다. 그래서 positive가 예외 없이
    index 0에 온다.

    평가의 Top-1은 np.argmax / torch.argmax를 쓰는데 동점이면 가장 낮은
    index를 고른다. positive가 항상 0번이면 동점이 난 자리가 전부 정답으로
    처리되어 Top-1이 실제 변별력보다 높게 나온다.

무엇을 바꾸는가
    candidate와 병렬인 컬럼의 "원소 순서"만 바꾼다. 재샘플링이 아니다.

    바꾸는 것
        candidate_article_ids
        candidate_c1, candidate_c2, candidate_c3, candidate_c4
        candidate_labels

    건드리지 않는 것
        positive 기사 선택, negative 4개 샘플링, negative candidate set,
        history_*, target_*, SID 값, split, impression 수, 행 순서, 행 개수

    target_* 는 builder가 candidate_*.list.first() 로 만든 값이라
    positive 자체를 담고 있다. candidate 순서와 무관하므로 섞지 않는다.

row 단위 deterministic seed
    sha256(seed | split_name | impression_id | positive_article_id | row_uid)

    전역 RNG 상태에 의존하지 않는다. 행 순서를 바꾸거나 일부만 다시 만들어도
    같은 permutation이 나온다.

사용법
    # 먼저 10행만 미리보기 (파일을 쓰지 않는다)
    python -m src.shuffle_candidate_order \
        --input-dir data/output/experiments/normalize_v2_uni_lu005_m05_candidate_1pos4neg_v1/post_rqvae \
        --preview 10

    # 확인 후 생성
    python -m src.shuffle_candidate_order \
        --input-dir data/output/experiments/normalize_v2_uni_lu005_m05_candidate_1pos4neg_v1/post_rqvae \
        --output-dir data/output/experiments/normalize_v2_uni_lu005_m05_candidate_1pos4neg_shuffled_v2/post_rqvae \
        --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl


DEFAULT_SEED = 42

# 순서를 함께 바꿔야 하는 컬럼.
# build_1pos4neg_sequences.py의 CANDIDATE_COLUMNS와 같다.
CANDIDATE_COLUMNS = [
    "candidate_article_ids",
    "candidate_c1",
    "candidate_c2",
    "candidate_c3",
    "candidate_c4",
    "candidate_labels",
]

# 절대 섞으면 안 되는 컬럼.
NEVER_SHUFFLE_PREFIXES = ("history_", "target_")

# 기본 처리 대상 파일과 split 이름.
DEFAULT_FILES = [
    ("train", "train_sequences_1pos4neg.parquet"),
    ("validation", "validation_sequences_1pos4neg.parquet"),
]


def section(title: str) -> None:
    print()
    print("=" * 74)
    print(title)
    print("=" * 74)


def row_permutation(
    seed: int,
    split_name: str,
    impression_id: Any,
    positive_article_id: Any,
    row_uid: int,
    length: int,
) -> list[int]:
    """row 하나의 permutation. 같은 입력이면 항상 같은 값이 나온다."""

    key = "|".join([
        str(seed),
        str(split_name),
        str(impression_id),
        str(positive_article_id),
        str(row_uid),
    ])

    digest = hashlib.sha256(key.encode("utf-8")).digest()
    row_seed = int.from_bytes(digest[:8], "big", signed=False)

    return np.random.default_rng(row_seed).permutation(length).tolist()


def positive_position(labels: list[int]) -> int:
    positions = [index for index, value in enumerate(labels) if value == 1]

    if len(positions) != 1:
        raise ValueError(f"row의 positive 수가 1이 아닙니다: {len(positions)}개")

    return positions[0]


def position_distribution(df: pl.DataFrame) -> dict[str, Any]:
    label_lists = df["candidate_labels"].to_list()
    width = len(label_lists[0]) if label_lists else 0

    counts = [0] * width

    for labels in label_lists:
        counts[positive_position(list(labels))] += 1

    total = max(sum(counts), 1)
    uniform = 1.0 / width if width else 0.0

    return {
        "counts": counts,
        "ratios": [c / total for c in counts],
        "total": sum(counts),
        "max_deviation_from_uniform": max(
            (abs(c / total - uniform) for c in counts), default=0.0
        ),
    }


def print_distribution(title: str, dist: dict[str, Any]) -> None:
    print()
    print(f"  {title}")
    print(f"  {'index':<9}{'count':>12}{'%':>10}")
    print("  " + "-" * 31)

    for index, (count, ratio) in enumerate(zip(dist["counts"], dist["ratios"])):
        print(f"  index {index:<3}{count:>12,}{ratio:>10.2%}")

    print(f"  균등 대비 최대 편차 : {dist['max_deviation_from_uniform']:.2%}")


def check_columns(df: pl.DataFrame) -> tuple[list[str], list[str]]:
    """섞을 컬럼과 그대로 둘 컬럼을 가른다."""

    present = [c for c in CANDIDATE_COLUMNS if c in df.columns]
    missing = [c for c in CANDIDATE_COLUMNS if c not in df.columns]

    if "candidate_labels" not in present:
        raise ValueError("candidate_labels 컬럼이 없습니다.")

    width = len(df["candidate_labels"].to_list()[0])

    for column in present:
        lengths = {len(v) for v in df[column].head(2000).to_list()}

        if lengths != {width}:
            raise ValueError(
                f"{column}의 길이가 일정하지 않습니다: {sorted(lengths)}"
            )

    untouched = [c for c in df.columns if c not in present]

    for column in untouched:
        if column.startswith("candidate_"):
            raise ValueError(
                f"{column}이 candidate_로 시작하는데 대상에서 빠졌습니다. "
                "CANDIDATE_COLUMNS를 확인하세요."
            )

    if missing:
        print(f"  주의: 파일에 없는 candidate 컬럼 {missing}")

    return present, untouched


def shuffle_frame(
    df: pl.DataFrame,
    columns: list[str],
    split_name: str,
    seed: int,
) -> tuple[pl.DataFrame, list[list[int]]]:
    """candidate 컬럼에 같은 permutation을 적용한다."""

    label_lists = df["candidate_labels"].to_list()
    article_lists = df["candidate_article_ids"].to_list()

    positive_positions = [positive_position(list(v)) for v in label_lists]
    positive_ids = [
        article_lists[i][positive_positions[i]] for i in range(df.height)
    ]

    impression_ids = (
        df["impression_id"].to_list()
        if "impression_id" in df.columns
        else list(range(df.height))
    )

    # 같은 (impression_id, positive)가 여러 row로 나올 경우를 위한 순번
    seen: dict[tuple[Any, Any], int] = {}
    row_uids: list[int] = []

    for impression_id, positive_id in zip(impression_ids, positive_ids):
        key = (impression_id, positive_id)
        row_uids.append(seen.get(key, 0))
        seen[key] = seen.get(key, 0) + 1

    permutations = [
        row_permutation(
            seed=seed,
            split_name=split_name,
            impression_id=impression_ids[i],
            positive_article_id=positive_ids[i],
            row_uid=row_uids[i],
            length=len(label_lists[i]),
        )
        for i in range(df.height)
    ]

    new_series = []

    for column in columns:
        values = df[column].to_list()

        new_series.append(
            pl.Series(
                name=column,
                values=[
                    [row[index] for index in permutation]
                    for row, permutation in zip(values, permutations)
                ],
                dtype=df[column].dtype,
            )
        )

    return df.with_columns(new_series), permutations


def print_preview(
    original: pl.DataFrame,
    shuffled: pl.DataFrame,
    permutations: list[list[int]],
    n: int,
) -> None:
    section(f"미리보기 {n}행")

    for i in range(min(n, original.height)):
        before_ids = list(original["candidate_article_ids"].to_list()[i])
        after_ids = list(shuffled["candidate_article_ids"].to_list()[i])
        before_labels = list(original["candidate_labels"].to_list()[i])
        after_labels = list(shuffled["candidate_labels"].to_list()[i])

        before_sids = [
            tuple(
                int(original[f"candidate_c{level}"].to_list()[i][k])
                for level in (1, 2, 3, 4)
            )
            for k in range(len(before_ids))
        ]
        after_sids = [
            tuple(
                int(shuffled[f"candidate_c{level}"].to_list()[i][k])
                for level in (1, 2, 3, 4)
            )
            for k in range(len(after_ids))
        ]

        impression = (
            original["impression_id"].to_list()[i]
            if "impression_id" in original.columns
            else i
        )

        print()
        print(f"  --- row {i}  impression_id={impression}"
              f"  permutation={permutations[i]}")
        print(f"    before ids    : {before_ids}")
        print(f"    after  ids    : {after_ids}")
        print(f"    before sids   : {before_sids}")
        print(f"    after  sids   : {after_sids}")
        print(f"    before labels : {before_labels}"
              f"   positive index = {positive_position(before_labels)}")
        print(f"    after  labels : {after_labels}"
              f"   positive index = {positive_position(after_labels)}")
        print(f"    target        : {original['target_article_ids'].to_list()[i]}"
              f"   (섞지 않음)")


def verify(
    original: pl.DataFrame,
    shuffled: pl.DataFrame,
    columns: list[str],
    untouched: list[str],
) -> dict[str, Any]:
    """기존 파일과 새 파일을 row별로 비교한다."""

    section("검증")

    checks: list[tuple[str, bool, str]] = []

    checks.append((
        "row count 동일",
        original.height == shuffled.height,
        f"{original.height:,} -> {shuffled.height:,}",
    ))

    checks.append((
        "컬럼 구성 동일",
        original.columns == shuffled.columns,
        f"{len(original.columns)}개",
    ))

    dtype_ok = all(
        original[c].dtype == shuffled[c].dtype for c in original.columns
    )
    checks.append(("dtype 동일", dtype_ok, "List(Int32/Int64) 유지"))

    untouched_ok = all(
        original[c].to_list() == shuffled[c].to_list() for c in untouched
    )
    checks.append((
        f"candidate 외 {len(untouched)}개 컬럼 변경 없음",
        untouched_ok,
        "history_*, target_*, 메타 포함",
    ))

    before_labels = original["candidate_labels"].to_list()
    after_labels = shuffled["candidate_labels"].to_list()
    before_ids = original["candidate_article_ids"].to_list()
    after_ids = shuffled["candidate_article_ids"].to_list()

    width = len(before_labels[0])

    counts_ok = all(len(v) == width for v in after_labels)
    checks.append((f"row당 candidate 수 = {width}", counts_ok, ""))

    positives_ok = all(sum(v) == 1 for v in after_labels)
    checks.append(("row당 positive 정확히 1개", positives_ok, ""))

    set_ok = True
    positive_ok = True
    negative_ok = True
    order_changed = 0
    dup_ok = True

    sid_before = [original[c].to_list() for c in
                  ("candidate_c1", "candidate_c2", "candidate_c3", "candidate_c4")]
    sid_after = [shuffled[c].to_list() for c in
                 ("candidate_c1", "candidate_c2", "candidate_c3", "candidate_c4")]

    for i in range(original.height):
        a_ids, b_ids = list(before_ids[i]), list(after_ids[i])
        a_lab, b_lab = list(before_labels[i]), list(after_labels[i])

        if sorted(a_ids) != sorted(b_ids):
            set_ok = False
            break

        if len(set(b_ids)) != width:
            dup_ok = False
            break

        a_pos = a_ids[positive_position(a_lab)]
        b_pos = b_ids[positive_position(b_lab)]

        if a_pos != b_pos:
            positive_ok = False
            break

        a_neg = sorted(x for x, l in zip(a_ids, a_lab) if l == 0)
        b_neg = sorted(x for x, l in zip(b_ids, b_lab) if l == 0)

        if a_neg != b_neg:
            negative_ok = False
            break

        # SID도 같은 기사를 따라 움직였는지
        a_map = {
            a_ids[k]: tuple(int(sid_before[l][i][k]) for l in range(4))
            for k in range(width)
        }
        b_map = {
            b_ids[k]: tuple(int(sid_after[l][i][k]) for l in range(4))
            for k in range(width)
        }

        if a_map != b_map:
            set_ok = False
            break

        if a_ids != b_ids:
            order_changed += 1

    checks.append(("candidate set 동일 (순서 무시)", set_ok, "SID 매핑 포함"))
    checks.append(("positive article 동일", positive_ok, ""))
    checks.append(("negative 4개 set 동일", negative_ok, ""))
    checks.append(("row 안 candidate 중복 없음", dup_ok, ""))
    checks.append((
        "candidate order만 변경",
        order_changed > 0,
        f"{order_changed:,} / {original.height:,} row에서 순서가 바뀜",
    ))

    print()
    for name, ok, note in checks:
        print(f"  [{'OK  ' if ok else 'FAIL'}] {name}"
              + (f"   ({note})" if note else ""))

    all_ok = all(ok for _, ok, _ in checks)

    print()
    print("  => " + ("모든 검증 통과" if all_ok else "검증 실패"))

    return {
        "all_passed": all_ok,
        "order_changed_rows": order_changed,
        "checks": [
            {"name": n, "passed": bool(o), "note": note} for n, o, note in checks
        ],
    }


def process_file(
    input_path: Path,
    output_path: Path | None,
    split_name: str,
    seed: int,
    preview: int,
) -> dict[str, Any] | None:
    section(f"[{split_name}] {input_path.name}")
    print(f"  입력 : {input_path}")

    if not input_path.exists():
        print("  파일이 없습니다. 건너뜁니다.")
        return None

    df = pl.read_parquet(input_path)
    print(f"  {df.height:,} rows, {len(df.columns)} columns")

    columns, untouched = check_columns(df)

    print()
    print(f"  permutation 적용 ({len(columns)}개)")
    for c in columns:
        print(f"    {c:<24} {df[c].dtype}")

    print()
    print(f"  그대로 두는 컬럼 ({len(untouched)}개)")
    for c in untouched:
        print(f"    {c}")

    before = position_distribution(df)
    print_distribution(f"{split_name} — shuffle 전 positive_position_distribution", before)

    if before["max_deviation_from_uniform"] > 0.05:
        print("  >>> position bias 확인됨")

    work = df.head(preview) if preview else df
    shuffled, permutations = shuffle_frame(work, columns, split_name, seed)

    if preview:
        print_preview(work, shuffled, permutations, preview)
        return {"preview": True}

    after = position_distribution(shuffled)
    print_distribution(f"{split_name} — shuffle 후 positive_position_distribution", after)

    if output_path is None:
        print()
        print("  출력 경로가 없어 파일을 쓰지 않습니다.")
        return None

    if output_path.exists():
        print()
        print(f"  이미 존재합니다. 덮어쓰지 않습니다: {output_path}")
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    shuffled.write_parquet(output_path)

    # 기존 파일과 새 파일을 다시 읽어 비교한다
    written = pl.read_parquet(output_path)
    verification = verify(df, written, columns, untouched)

    if not verification["all_passed"]:
        print()
        print(f"  검증에 실패했습니다. 생성된 파일을 확인하세요: {output_path}")

    print()
    print(f"  출력 : {output_path}")

    return {
        "split_name": split_name,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "num_rows": int(df.height),
        "seed": seed,
        "columns_shuffled": columns,
        "columns_untouched": untouched,
        "positive_position_distribution_before": before,
        "positive_position_distribution_after": after,
        "verification": verification,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "기존 1pos4neg parquet의 candidate 순서만 섞는다. "
            "재샘플링하지 않는다."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--file", action="append", default=None,
        metavar="SPLIT:PATH",
        help="개별 파일 지정. 예 test:path/to/test.parquet",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--preview", type=int, default=0,
        help="N행만 미리 보고 파일을 쓰지 않는다.",
    )
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    targets: list[tuple[str, Path, Path | None]] = []

    if args.input_dir:
        for split_name, filename in DEFAULT_FILES:
            out = (args.output_dir / filename) if args.output_dir else None
            targets.append((split_name, args.input_dir / filename, out))

    # --file은 --input-dir과 함께 쓸 수 있다. test를 추가할 때 쓴다.
    for item in args.file or []:
        split_name, _, raw_path = item.partition(":")
        path = Path(raw_path)
        out = (args.output_dir / path.name) if args.output_dir else None
        targets.append((split_name, path, out))

    if not targets:
        parser.error("--input-dir 또는 --file 중 하나가 필요합니다.")

    section("candidate order shuffle")
    print(f"  seed : {args.seed}")
    print()
    print("  바꾸는 것   : candidate 관련 컬럼의 원소 순서")
    print("  안 바꾸는 것: positive 선택, negative 샘플링, candidate set,")
    print("                history, target, SID 값, split, 행 수, 행 순서")

    results = []

    for split_name, input_path, output_path in targets:
        result = process_file(
            input_path=input_path,
            output_path=output_path,
            split_name=split_name,
            seed=args.seed,
            preview=args.preview,
        )

        if result and not result.get("preview"):
            results.append(result)

    if args.preview:
        section("미리보기만 했습니다. 파일을 쓰지 않았습니다.")
        print("  확인 후 --preview 없이 --output-dir을 주고 다시 실행하세요.")
        return 0

    if args.report and results:
        report = {
            "tool": "shuffle_candidate_order",
            "candidate_order_shuffled": True,
            "resampled": False,
            "seed": args.seed,
            "row_seed_formula": (
                "sha256(seed | split_name | impression_id | "
                "positive_article_id | row_uid)"
            ),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "splits": results,
        }

        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        print()
        print(f"리포트 : {args.report}")

    return 0 if all(
        r["verification"]["all_passed"] for r in results
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
