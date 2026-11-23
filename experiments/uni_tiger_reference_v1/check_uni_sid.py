"""UNI SID 일치 확인 (읽기 전용, CPU). 아무 파일도 쓰지 않는다.

sequence parquet(1P4N 또는 1P4N 이전)마다 history/candidate의 (article_id -> c1,c2,c3,c4)를 모아
  1) 한 파일 안에서 기사 하나에 SID가 하나인지
  2) 파일끼리, 그리고 --sid 파일(article_id,c1..c4)과 기사마다 같은지
를 확인한다. 하나라도 다르면 exit code 1.

  python experiments/uni_tiger_reference_v1/check_uni_sid.py \
      --sid /home/ubuntu/article_semantic_ids_final.parquet \
      --seq ~/shared/datasets/ebnerd/validation_sequences_1pos4neg_half.parquet \
      --seq ~/shared/datasets/ebnerd/test_sequences_1pos4neg.parquet \
      --seq ~/shared/datasets/ebnerd/train_sequences_1pos4neg.parquet

--subset A B : A의 impression_id가 모두 B에 있는지 (예: A = ~/shared 1P4N train, B = 확보한 1P4N 이전 train)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

LEVELS = ["c1", "c2", "c3", "c4"]


def sid_map_from_sequences(path: Path, chunk_rows: int) -> pl.DataFrame:
    """history + candidate에서 (article_id, c1..c4) 고유 조합. 메모리를 아끼려고 row chunk로 나눠 읽는다."""
    n = pl.scan_parquet(path).select(pl.len()).collect().item()
    schema = pl.scan_parquet(path).collect_schema().names()
    parts = []
    for prefix in ("history", "candidate"):
        cols = [f"{prefix}_article_ids"] + [f"{prefix}_{l}" for l in LEVELS]
        missing = [c for c in cols if c not in schema]
        if missing:
            raise SystemExit(f"{path}: 컬럼 없음 {missing}")
        for s in range(0, n, chunk_rows):
            part = (pl.scan_parquet(path).select(cols).slice(s, chunk_rows)
                    .explode(cols)
                    .rename({cols[0]: "article_id", **{f"{prefix}_{l}": l for l in LEVELS}})
                    .drop_nulls()
                    .with_columns([pl.col("article_id").cast(pl.Int64)] + [pl.col(l).cast(pl.Int64) for l in LEVELS])
                    .unique().collect())
            parts.append(part)
    return pl.concat(parts).unique()


def describe(name: str, m: pl.DataFrame) -> bool:
    n_art = m["article_id"].n_unique()
    conflict = m.height - n_art
    mx = {l: int(m[l].max()) for l in LEVELS}
    print(f"[{name}] 기사 {n_art:,} | SID가 2개 이상인 기사 {conflict:,} | "
          f"max " + " ".join(f"{l} {v}" for l, v in mx.items()) +
          f" (UNI 기대: c1<25, c2<128, c3<512)")
    return conflict == 0


def compare(name_a: str, a: pl.DataFrame, name_b: str, b: pl.DataFrame) -> bool:
    j = a.join(b, on="article_id", how="inner", suffix="_b")
    only_a = a["article_id"].n_unique() - j["article_id"].n_unique()
    diff = {l: int((j[l] != j[f"{l}_b"]).sum()) for l in LEVELS}
    all_same = j.filter(pl.all_horizontal([pl.col(l) == pl.col(f"{l}_b") for l in LEVELS])).height
    print(f"  {name_a} vs {name_b}: 공통 기사 {j.height:,} | c1..c4 모두 같음 {all_same:,} | "
          f"다름 " + " ".join(f"{l} {v:,}" for l, v in diff.items()) +
          f" | {name_a}에만 있는 기사 {only_a:,}")
    if all_same != j.height:
        print(j.filter(~pl.all_horizontal([pl.col(l) == pl.col(f"{l}_b") for l in LEVELS])).head(5))
    return all_same == j.height


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sid", type=Path, default=None, help="article_id,c1,c2,c3,c4 parquet")
    p.add_argument("--seq", type=Path, action="append", default=[])
    p.add_argument("--subset", type=Path, nargs=2, metavar=("A", "B"), default=None)
    p.add_argument("--chunk-rows", type=int, default=20_000)
    a = p.parse_args()

    ok = True
    maps = {}
    for path in a.seq:
        path = path.expanduser()
        maps[path.name] = sid_map_from_sequences(path, a.chunk_rows)
        ok &= describe(path.name, maps[path.name])

    if a.sid is not None:
        sid = (pl.read_parquet(a.sid.expanduser(), columns=["article_id"] + LEVELS)
               .with_columns([pl.col(c).cast(pl.Int64) for c in ["article_id"] + LEVELS]))
        ok &= describe(f"SID {a.sid.name}", sid)
        print(f"  SID 파일 행 {sid.height:,}")
        for name, m in maps.items():
            ok &= compare(name, m, "SID", sid)

    names = list(maps)
    for i in range(len(names)):
        for k in range(i + 1, len(names)):
            ok &= compare(names[i], maps[names[i]], names[k], maps[names[k]])

    if a.subset:
        sa = pl.scan_parquet(a.subset[0].expanduser()).select("impression_id").unique().collect()
        sb = pl.scan_parquet(a.subset[1].expanduser()).select("impression_id").unique().collect()
        miss = sa.join(sb, on="impression_id", how="anti").height
        print(f"[subset] {a.subset[0].name} impression {sa.height:,} 중 {a.subset[1].name}에 없는 것 {miss:,} "
              f"({a.subset[1].name} impression {sb.height:,})")
        ok &= miss == 0

    print("\n결과:", "모두 일치" if ok else "불일치 있음")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
