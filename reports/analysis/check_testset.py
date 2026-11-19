"""small articles.parquet 의 train/val 미참조 기사가 test set 에서 쓰이는지 대조."""
import hashlib, os, sys, zipfile
import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
DL, LOCAL = sys.argv[1], sys.argv[2]  # 다운로드 폴더, 로컬 raw 폴더


def extract(zname, out):
    if not os.path.isdir(out):
        with zipfile.ZipFile(os.path.join(DL, zname)) as z:
            for i in z.infolist():
                print("  ", zname, i.filename, i.file_size)
            z.extractall(out)
    paths = {}
    for root, _, files in os.walk(out):
        for f in files:
            paths[os.path.relpath(os.path.join(root, f), out).replace("\\", "/")] = os.path.join(root, f)
    return paths


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


small = extract("ebnerd_small.zip", os.path.join(DL, "small_x"))
test = extract("ebnerd_testset.zip", os.path.join(DL, "test_x"))
print("small files:", sorted(small))
print("test files:", sorted(test))


def find(paths, suffix):
    c = [p for k, p in paths.items() if k.endswith(suffix)]
    return c[0] if c else None


# 1) 로컬 small 과 동일한지
print("\n[1] 로컬 파일과 다운로드한 small 비교 (md5)")
for rel in ["articles.parquet", "train/behaviors.parquet", "train/history.parquet",
            "validation/behaviors.parquet", "validation/history.parquet"]:
    a, b = os.path.join(LOCAL, rel), find(small, rel)
    print(f"  {rel:32s} local={'있음' if os.path.exists(a) else '없음'}  same={os.path.exists(a) and b and md5(a) == md5(b)}")

# 2) 미참조 기사 집합
A = pl.read_parquet(find(small, "articles.parquet")).select("article_id", "published_time")
ref, users = set(), set()
for sp in ["train", "validation"]:
    H = pl.read_parquet(find(small, f"{sp}/history.parquet"))
    ref |= set(H["article_id_fixed"].explode().drop_nulls().to_list())
    users |= set(H["user_id"].to_list())
    B = pl.read_parquet(find(small, f"{sp}/behaviors.parquet"))
    ref |= set(B["article_id"].drop_nulls().to_list())
    for c in ["article_ids_inview", "article_ids_clicked"]:
        ref |= set(B[c].explode().drop_nulls().to_list())
    users |= set(B["user_id"].to_list())
unref = set(A["article_id"].to_list()) - ref
print(f"\n[2] small 기사 {A.height:,}, train/val 참조 {len(ref & set(A['article_id'].to_list())):,}, 미참조 {len(unref):,}, small 사용자 {len(users):,}")

# 3) test set
TA = pl.read_parquet(find(test, "articles.parquet")).select("article_id", "published_time")
print(f"\n[3] test articles.parquet: {TA.height:,}개, 발행 {TA['published_time'].min()} ~ {TA['published_time'].max()}")
tb = find(test, "test/behaviors.parquet")
th = find(test, "test/history.parquet")
print("  test behaviors schema:", pl.scan_parquet(tb).collect_schema())
bl = pl.scan_parquet(tb)
hl = pl.scan_parquet(th)
print("  test 노출 수:", bl.select(pl.len()).collect().item(), " 기간:", bl.select(pl.col("impression_time").min(), pl.col("impression_time").max()).collect().row(0))
inv_all = set(bl.select(pl.col("article_ids_inview").explode()).unique().collect().to_series().drop_nulls().to_list())
hist_all = set(hl.select(pl.col("article_id_fixed").explode()).unique().collect().to_series().drop_nulls().to_list())
ul = list(users)
inv_small = set(bl.filter(pl.col("user_id").is_in(ul)).select(pl.col("article_ids_inview").explode()).unique().collect().to_series().drop_nulls().to_list())
hist_small = set(hl.filter(pl.col("user_id").is_in(ul)).select(pl.col("article_id_fixed").explode()).unique().collect().to_series().drop_nulls().to_list())
cur_small = set(bl.filter(pl.col("user_id").is_in(ul)).select("article_id").collect().to_series().drop_nulls().to_list()) if "article_id" in bl.collect_schema() else set()
n_small_test_users = bl.filter(pl.col("user_id").is_in(ul)).select(pl.col("user_id").n_unique()).collect().item()
print(f"  small 사용자 중 test 에 나오는 사용자: {n_small_test_users:,}")

ta_ids = set(TA["article_id"].to_list())
test_small_any = inv_small | hist_small | cur_small
test_any = inv_all | hist_all
print("\n[4] 미참조 7,570개 대조")
for name, s in [("test articles.parquet 에 있음", ta_ids), ("test 전체 inview", inv_all), ("test 전체 history", hist_all),
                ("test 전체 (inview+history)", test_any), ("small 사용자의 test inview", inv_small),
                ("small 사용자의 test history", hist_small), ("small 사용자의 test 어디든", test_small_any)]:
    print(f"  {name:30s}: {len(unref & s):,} / {len(unref):,}")
left = unref - test_any
print(f"  test 어디에도 없음: {len(left):,}")
if left:
    L = A.filter(pl.col("article_id").is_in(list(left)))
    print("   그 기사들 발행 연도:", L.group_by(pl.col("published_time").dt.year().alias("y")).len().sort("y").rows())
# 참고: small 전체 기사 중 small 사용자 기준 train/val/test 어딘가 등장
allsmall = ref | test_small_any
print(f"\n[5] small 기사 중 small 사용자의 train/val/test 어딘가에 등장: {len(set(A['article_id'].to_list()) & allsmall):,} / {A.height:,}")
print(f"    small 사용자 데이터에 나오는데 small articles.parquet 에 없는 기사: {len(allsmall - set(A['article_id'].to_list())):,}")
