"""Train 사건 중 기사가 2개 이상인 사건만 골라 c2 일관성을 잰다.

사용: python c2_event_consistency.py <article_semantic_ids.parquet> [...]
필요 컬럼: article_id, event_id, split, c1, c2  (split == "train" 만 사용)
"""
import sys
import random
import polars as pl

sys.stdout.reconfigure(encoding="utf-8")


def pair_agree(groups):
    """같은 그룹 안 모든 기사 쌍 중 c2가 같은 쌍 비율."""
    same = total = 0
    for codes in groups:
        n = len(codes)
        if n < 2:
            continue
        cnt = {}
        for c in codes:
            cnt[c] = cnt.get(c, 0) + 1
        same += sum(k * (k - 1) // 2 for k in cnt.values())
        total += n * (n - 1) // 2
    return same / total if total else float("nan"), total


def run(path):
    df = pl.read_parquet(path).filter(pl.col("split") == "train")
    sizes = df.group_by("event_id").len()
    multi = sizes.filter(pl.col("len") >= 2)
    m = df.join(multi.select("event_id"), on="event_id", how="inner")
    print(f"\n### {path}")
    print(f"train 기사 {df.height:,} · train 사건 {sizes.height:,} · 기사 2개 이상 사건 {multi.height:,} (기사 {m.height:,})")

    ev = m.group_by("event_id").agg(pl.col("c2"), pl.col("c1"), pl.len().alias("n"))
    # 사건 안 최빈 c2 비율 (기사 가중)
    maj = ev.with_columns(
        pl.col("c2").list.eval(pl.element().value_counts(sort=True).first().struct.field("count")).list.first().alias("maj")
    )
    purity = maj["maj"].sum() / maj["n"].sum()
    all_same = (maj["maj"] == maj["n"]).mean()
    agree, npairs = pair_agree(ev["c2"].to_list())

    # 기준선: 같은 c1(카테고리) 안에서 무작위로 뽑은 기사 쌍의 c2 일치율
    rnd = random.Random(42)
    by_c1 = {}
    for c1, c2 in zip(df["c1"].to_list(), df["c2"].to_list()):
        by_c1.setdefault(c1, []).append(c2)
    same = tot = 0
    weights = [(k, len(v)) for k, v in by_c1.items() if len(v) >= 2]
    keys = [k for k, _ in weights]
    w = [n * (n - 1) for _, n in weights]
    for _ in range(200000):
        k = rnd.choices(keys, weights=w)[0]
        a, b = rnd.sample(by_c1[k], 2)
        same += a == b
        tot += 1
    base_c1 = same / tot
    # 기준선 2: 전체 무작위 쌍
    allc = df["c2"].to_list()
    same = sum(rnd.choice(allc) == rnd.choice(allc) for _ in range(200000))
    base_all = same / 200000

    print(f"  사건 안 최빈 c2 비율 (기사 기준) : {purity:.3f}")
    print(f"  모든 기사가 같은 c2인 사건 비율  : {all_same:.3f}")
    print(f"  같은 사건 기사 쌍의 c2 일치율    : {agree:.3f}  (쌍 {npairs:,})")
    print(f"  기준선: 같은 카테고리 무작위 쌍   : {base_c1:.3f}")
    print(f"  기준선: 전체 무작위 쌍            : {base_all:.3f}")

    # 사건 크기별
    rows = []
    for lo, hi, lab in [(2, 2, "2개"), (3, 4, "3–4개"), (5, 9, "5–9개"), (10, 10**9, "10개+")]:
        sub = ev.filter((pl.col("n") >= lo) & (pl.col("n") <= hi))
        if sub.height == 0:
            continue
        a, p = pair_agree(sub["c2"].to_list())
        s = maj.filter((pl.col("n") >= lo) & (pl.col("n") <= hi))
        rows.append((lab, sub.height, int(sub["n"].sum()), a, (s["maj"] == s["n"]).mean()))
    print("  크기별: 크기 | 사건 수 | 기사 수 | 쌍 일치율 | 전부 같은 c2 비율")
    for r in rows:
        print(f"    {r[0]:>5} | {r[1]:>5,} | {r[2]:>6,} | {r[3]:.3f} | {r[4]:.3f}")
    # 사건이 몇 개의 서로 다른 c2로 쪼개지나
    nuniq = ev.with_columns(pl.col("c2").list.n_unique().alias("k"))["k"]
    print(f"  사건당 서로 다른 c2 수: 평균 {nuniq.mean():.2f}, 중앙값 {nuniq.median():.0f}, 최대 {nuniq.max()}")


for p in sys.argv[1:]:
    run(p)
