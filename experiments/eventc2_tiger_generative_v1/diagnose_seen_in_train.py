"""진단: validation/test 후보가 train에서 target으로 나온 적이 있는지 (positive vs negative). 읽기 전용, CPU.

생성형 모델은 train에서 자주 클릭된 SID에 높은 확률을 준다. validation 기간의 positive(클릭된 기사)가
새 기사이고 negative가 train 때부터 있던 기사라면, 생성 확률이 negative 쪽으로 쏠려 AUC가 0.5 아래로 갈 수 있다.

    python experiments/eventc2_tiger_generative_v1/diagnose_seen_in_train.py
"""
from pathlib import Path
import polars as pl

EXP = Path("data/output/experiments/normalize_v2_uni_lu005_m05_eventc2_retrain_v1")
train = pl.scan_parquet(EXP / "post_rqvae/train_sequences.parquet").select(
    ["candidate_article_ids", "candidate_labels", "candidate_c1", "candidate_c2", "candidate_c3"]
).explode(["candidate_article_ids", "candidate_labels", "candidate_c1", "candidate_c2", "candidate_c3"]) \
 .filter(pl.col("candidate_labels") == 1).collect()

art_count = train.group_by("candidate_article_ids").len().rename({"candidate_article_ids": "aid", "len": "art_clicks"})
sid_count = train.group_by(["candidate_c1", "candidate_c2", "candidate_c3"]).len().rename({"len": "sid_clicks"})
c12_count = train.group_by(["candidate_c1", "candidate_c2"]).len().rename({"len": "c12_clicks"})
print(f"train positive {train.height:,} / 고유 기사 {art_count.height:,}")

for name in ("validation_sequences_1pos4neg_half", "test_sequences_1pos4neg"):
    d = pl.read_parquet(EXP / f"transformer_datasets/{name}.parquet",
                        columns=["candidate_article_ids", "candidate_labels", "candidate_c1", "candidate_c2", "candidate_c3"])
    d = d.explode(["candidate_article_ids", "candidate_labels", "candidate_c1", "candidate_c2", "candidate_c3"]) \
         .rename({"candidate_article_ids": "aid"})
    d = d.join(art_count, on="aid", how="left").join(sid_count, on=["candidate_c1", "candidate_c2", "candidate_c3"], how="left") \
         .join(c12_count, on=["candidate_c1", "candidate_c2"], how="left").fill_null(0)
    s = d.group_by("candidate_labels").agg([
        (pl.col("art_clicks") > 0).mean().alias("기사가 train target에 있음"),
        (pl.col("sid_clicks") > 0).mean().alias("c1c2c3가 train target에 있음"),
        (pl.col("c12_clicks") > 0).mean().alias("c1c2가 train target에 있음"),
        pl.col("sid_clicks").median().alias("c1c2c3 train 클릭 수 (중앙값)"),
        pl.col("c12_clicks").median().alias("c1c2 train 클릭 수 (중앙값)"),
    ]).sort("candidate_labels", descending=True)
    print(f"\n== {name} (label 1 = positive, 0 = negative)")
    with pl.Config(tbl_cols=-1, tbl_width_chars=200):
        print(s)
