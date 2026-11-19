"""코드북 사용량·희소성: 코드를 늘리면 무슨 일이 생기는지 보기 위한 측정."""
import sys
import polars as pl

sys.stdout.reconfigure(encoding="utf-8")

for path in sys.argv[1:]:
    df = pl.read_parquet(path)
    tr = df.filter(pl.col("split") == "train")
    va = df.filter(pl.col("split") != "train")
    print(f"\n### {path}")
    print(f"컬럼: {df.columns}")
    print(f"train {tr.height:,} / 그 외 {va.height:,}")
    for col in ["c1", "c2", "c3"]:
        print(f"  {col}: train 사용 코드 수 {tr[col].n_unique()}, 최대값 {tr[col].max()}")
    for keys, lab in [(["c1", "c2"], "c1c2"), (["c1", "c2", "c3"], "c1c2c3")]:
        g = tr.group_by(keys).len()
        print(f"  {lab} prefix: {g.height:,}개, prefix당 train 기사 평균 {g['len'].mean():.2f}, 중앙값 {g['len'].median():.0f}, 기사 1개짜리 비율 {(g['len'] == 1).mean():.3f}")
        if va.height:
            seen = va.join(g.select(keys), on=keys, how="semi").height
            print(f"    비-train 기사 중 train에 있는 {lab} prefix를 받은 비율: {seen / va.height:.3f}")
    # c3 토큰(값) 자체의 train 등장 횟수
    c3 = tr.group_by("c3").len()
    print(f"  c3 값별 train 기사 수: 평균 {c3['len'].mean():.1f}, 최소 {c3['len'].min()}")
    if "c4" in df.columns:
        print(f"  train c4>0 (충돌) 비율: {(tr['c4'] > 0).mean():.3f}, c4 최대 {tr['c4'].max()}")
