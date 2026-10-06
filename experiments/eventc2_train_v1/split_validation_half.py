"""validation 1pos4neg를 impression 시간순 50:50으로 validation-half / test로 나눈다.

Transformer/split_validation.py와 같은 규칙이다.
  - unique impression을 (impression_time, impression_id)로 정렬
  - 앞 절반 impression -> validation half, 나머지 -> test
  - 두 파일의 row는 (impression_time, impression_id)로 정렬
원본 스크립트는 입력/출력 경로가 Transformer/datasets/ebnerd로 고정되어 있어
공용 데이터를 덮어쓸 수 있으므로, 경로만 인자로 받도록 분리했다.
출력 파일이 이미 있으면 중단한다.

    python experiments/eventc2_train_v1/split_validation_half.py \
        --input  <exp>/candidate_1pos4neg_shuffled/validation_sequences_1pos4neg.parquet \
        --out-dir <exp>/transformer_datasets
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl


def split_validation_half(input_path: Path, out_dir: Path) -> dict:
    validation_output_path = out_dir / "validation_sequences_1pos4neg_half.parquet"
    test_output_path = out_dir / "test_sequences_1pos4neg.parquet"

    for path in (validation_output_path, test_output_path):
        if path.exists():
            raise FileExistsError(f"이미 존재합니다. 덮어쓰지 않습니다: {path}")

    df = pl.read_parquet(input_path)

    for column in ("impression_id", "impression_time"):
        if column not in df.columns:
            raise ValueError(f"{column} column이 없습니다.")

    impressions = (
        df.select(["impression_id", "impression_time"])
        .unique(subset=["impression_id"], keep="first")
        .sort(["impression_time", "impression_id"])
    )

    split_idx = impressions.height // 2
    validation_ids = impressions.slice(0, split_idx).select("impression_id")
    test_ids = impressions.slice(split_idx, impressions.height - split_idx).select("impression_id")

    validation_df = (
        df.join(validation_ids, on="impression_id", how="semi")
        .sort(["impression_time", "impression_id"])
    )
    test_df = (
        df.join(test_ids, on="impression_id", how="semi")
        .sort(["impression_time", "impression_id"])
    )

    overlap = (
        validation_df.select("impression_id").unique()
        .join(test_df.select("impression_id").unique(), on="impression_id", how="inner")
    )
    if overlap.height > 0:
        raise ValueError(f"Validation/Test impression overlap: {overlap.height}")

    out_dir.mkdir(parents=True, exist_ok=True)
    validation_df.write_parquet(validation_output_path, compression="zstd")
    test_df.write_parquet(test_output_path, compression="zstd")

    return {
        "input_path": str(input_path),
        "original_rows": df.height,
        "unique_impressions": impressions.height,
        "validation_half_path": str(validation_output_path),
        "validation_half_rows": validation_df.height,
        "validation_half_impressions": validation_df["impression_id"].n_unique(),
        "test_path": str(test_output_path),
        "test_rows": test_df.height,
        "test_impressions": test_df["impression_id"].n_unique(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    result = split_validation_half(args.input, args.out_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    (args.out_dir / "split_report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
