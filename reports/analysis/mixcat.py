import sys, polars as pl
sys.stdout.reconfigure(encoding="utf-8")
df = pl.read_parquet(sys.argv[1]).filter(pl.col("split") == "train")
ev = df.group_by("event_id").agg(pl.len().alias("n"), pl.col("c1").n_unique().alias("k"))
m = ev.filter(pl.col("n") >= 2)
print(f"train 사건 {ev.height:,} / 기사 2개+ 사건 {m.height:,}")
print(f"카테고리 섞인 사건: {(m['k'] > 1).sum():,} ({(m['k'] > 1).mean():.1%} of multi), 그 사건 기사 수 {m.filter(pl.col('k') > 1)['n'].sum():,} / train 기사 {df.height:,}")
