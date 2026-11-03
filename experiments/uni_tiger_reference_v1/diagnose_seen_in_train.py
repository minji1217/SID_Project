"""진단: validation/test 후보가 train에서 target(클릭)으로 나온 적이 있는지 (positive vs negative). 읽기 전용, CPU.

eventc2_tiger_generative_v1/diagnose_seen_in_train.py와 같은 집계를 경로만 인자로 받게 한 UNI용 사본.

    python experiments/uni_tiger_reference_v1/diagnose_seen_in_train.py \
        --train <1P4N 이전 UNI train_sequences.parquet> \
        --eval ~/shared/datasets/ebnerd/validation_sequences_1pos4neg_half.parquet \
        --eval ~/shared/datasets/ebnerd/test_sequences_1pos4neg.parquet
"""
import argparse
from pathlib import Path

import polars as pl

COLS = ["candidate_article_ids", "candidate_labels", "candidate_c1", "candidate_c2", "candidate_c3"]

p = argparse.ArgumentParser()
p.add_argument("--train", type=Path, required=True)
p.add_argument("--eval", type=Path, action="append", required=True)
a = p.parse_args()

train = pl.scan_parquet(a.train.expanduser()).select(COLS).explode(COLS) \
    .filter(pl.col("candidate_labels") == 1).collect()
art_count = train.group_by("candidate_article_ids").len().rename({"candidate_article_ids": "aid", "len": "art_clicks"})
sid_count = train.group_by(["candidate_c1", "candidate_c2", "candidate_c3"]).len().rename({"len": "sid_clicks"})
c12_count = train.group_by(["candidate_c1", "candidate_c2"]).len().rename({"len": "c12_clicks"})
print(f"train positive {train.height:,} / 고유 기사 {art_count.height:,}")

for path in a.eval:
    d = pl.read_parquet(path.expanduser(), columns=COLS).explode(COLS).rename({"candidate_article_ids": "aid"})
    d = d.join(art_count, on="aid", how="left").join(sid_count, on=["candidate_c1", "candidate_c2", "candidate_c3"], how="left") \
         .join(c12_count, on=["candidate_c1", "candidate_c2"], how="left").fill_null(0)
    s = d.group_by("candidate_labels").agg([
        (pl.col("art_clicks") > 0).mean().alias("기사가 train target에 있음"),
        (pl.col("sid_clicks") > 0).mean().alias("c1c2c3가 train target에 있음"),
        (pl.col("c12_clicks") > 0).mean().alias("c1c2가 train target에 있음"),
        pl.col("sid_clicks").median().alias("c1c2c3 train 클릭 수 (중앙값)"),
        pl.col("c12_clicks").median().alias("c1c2 train 클릭 수 (중앙값)"),
    ]).sort("candidate_labels", descending=True)
    print(f"\n== {path.name} (label 1 = positive, 0 = negative)")
    with pl.Config(tbl_cols=-1, tbl_width_chars=200):
        print(s)
