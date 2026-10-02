"""Step 2 — 정성평가 group sampling.

입력 parquet은 읽기 전용. seed 42 고정. 출력은 analysis/rqvae_sid_evaluation/step2/.

사용자 지정 구성
  prefix_3 collision group : size 2~3 15 / 4~5 10 / 6~8 5  = 30 group
  prefix_2 group           : 기사 4~5 5 / 6~7 5 / 8~10 5   = 15 group
  semantic_different       : 30 group-pair (c1,c2,c3 모두 다름) + 보조 10 pair (c4까지 다름)

선택된 group 안의 article은 전부 포함한다 (일부 pair만 뽑지 않는다).
각 size 구간 안에서 동일 c1 / (c1,c2)에 몰리지 않도록 greedy diversity 샘플링.
"""
from __future__ import annotations
import json, sys, unicodedata, re
from pathlib import Path
import numpy as np
import pandas as pd

sys.stdout.reconfigure(line_buffering=True)

SEED = 42
U = Path("/root/.claude/uploads/f5109944-18fd-59ef-9d42-2e3b01694dcf")
SID_PARQUET = U / "57f50fe6-article_semantic_ids.parquet"
ART_PARQUET = U / "f0158e4f-articles.parquet"
OUT = Path("/home/user/SID_Project/analysis/rqvae_sid_evaluation/step2")

WS = re.compile(r"\s+")
PUNCT = re.compile(r"[^\w\s]", flags=re.UNICODE)


def n_tokens(text: str) -> int:
    if not isinstance(text, str):
        return 0
    t = PUNCT.sub(" ", unicodedata.normalize("NFC", text).lower())
    return len([x for x in WS.split(t) if x])


# ------------------------------------------------------------------ 입력 (read-only)
def load_population() -> pd.DataFrame:
    """train split + body 비어있지 않은 기사 = 9,338 (Step 1의 main analysis 모집단)."""
    sid = pd.read_parquet(SID_PARQUET)
    art = pd.read_parquet(ART_PARQUET)
    sid = sid.copy()
    art = art.copy()
    sid["aid"] = sid["article_id"].astype(str).str.strip()
    art["aid"] = art["article_id"].astype(str).str.strip()
    m = sid.merge(
        art[["aid", "title", "subtitle", "body", "category_str"]],
        on="aid", how="left", validate="1:1",
    )
    if m[["title", "body"]].isna().any().any():
        raise ValueError("join 후 title/body에 NaN이 있습니다.")
    tr = m[m["split"] == "train"].reset_index(drop=True)
    body = tr["body"].fillna("").astype(str).str.strip()
    pop = tr[body != ""].reset_index(drop=True)
    return pop


# ------------------------------------------------------------------ diversity 샘플링
def diverse_pick(groups: list[dict], k: int, rng: np.random.Generator) -> list[dict]:
    """c1, (c1,c2)가 한쪽에 몰리지 않도록 라운드 방식으로 k개 고른다.

    cap을 1부터 올리며, 아직 c1 사용 횟수가 cap 미만이고 (c1,c2)가 미사용인 group을
    우선 뽑는다. 후보가 마르면 (c1,c2) 조건을 풀고, 그 다음 cap을 올린다.
    같은 cap 안에서의 순서는 seed 42 permutation으로 정한다.
    """
    if k >= len(groups):
        return list(groups)
    order = list(rng.permutation(len(groups)))
    pool = [groups[i] for i in order]
    picked: list[dict] = []
    c1_used: dict[int, int] = {}
    c12_used: set[tuple[int, int]] = set()
    cap = 1
    while len(picked) < k:
        progressed = False
        for strict_c12 in (True, False):
            for g in list(pool):
                if len(picked) >= k:
                    break
                if c1_used.get(g["c1"], 0) >= cap:
                    continue
                if strict_c12 and (g["c1"], g["c2"]) in c12_used:
                    continue
                picked.append(g)
                pool.remove(g)
                c1_used[g["c1"]] = c1_used.get(g["c1"], 0) + 1
                c12_used.add((g["c1"], g["c2"]))
                progressed = True
            if len(picked) >= k:
                break
        if len(picked) >= k:
            break
        if not progressed:
            cap += 1
            if cap > len(groups) + 1:   # 안전장치
                picked.extend(pool[: k - len(picked)])
                break
    return picked[:k]


def band_of(value: int, bands: list[tuple[int, int, str]]) -> str | None:
    for lo, hi, name in bands:
        if lo <= value <= hi:
            return name
    return None


# ------------------------------------------------------------------ prefix_3
P3_BANDS = [(2, 3, "size_2_3"), (4, 5, "size_4_5"), (6, 8, "size_6_8")]
P3_QUOTA = {"size_2_3": 15, "size_4_5": 10, "size_6_8": 5}


def sample_prefix3(pop: pd.DataFrame, rng: np.random.Generator):
    grouped = pop.groupby(["c1", "c2", "c3"], sort=True).indices
    cand: dict[str, list[dict]] = {b[2]: [] for b in P3_BANDS}
    size_hist: dict[int, int] = {}
    for (c1, c2, c3), rows in grouped.items():
        size = len(rows)
        if size < 2:
            continue
        size_hist[size] = size_hist.get(size, 0) + 1
        band = band_of(size, P3_BANDS)
        if band is None:
            continue
        cand[band].append(
            {"c1": int(c1), "c2": int(c2), "c3": int(c3), "size": size,
             "rows": np.sort(rows)}
        )
    out_rows, group_meta = [], []
    gid = 0
    for lo, hi, band in P3_BANDS:
        pick = diverse_pick(cand[band], P3_QUOTA[band], rng)
        pick = sorted(pick, key=lambda g: (-g["size"], g["c1"], g["c2"], g["c3"]))
        for g in pick:
            gid += 1
            qid = f"P3-{gid:02d}"
            group_meta.append(
                {"qualitative_group_id": qid, "size_band": band,
                 "c1": g["c1"], "c2": g["c2"], "c3": g["c3"], "group_size": g["size"],
                 "available_groups_in_band": len(cand[band])}
            )
            for r in g["rows"]:
                a = pop.iloc[int(r)]
                out_rows.append({
                    "qualitative_group_id": qid, "size_band": band,
                    "c1": g["c1"], "c2": g["c2"], "c3": g["c3"], "group_size": g["size"],
                    "article_id": a["aid"], "c4": int(a["c4"]),
                    "category_str": a["category_str"],
                    "title": a["title"], "subtitle": a["subtitle"], "body": a["body"],
                    "title_tokens": n_tokens(a["title"]),
                    "body_tokens": n_tokens(a["body"]),
                })
    return pd.DataFrame(out_rows), pd.DataFrame(group_meta), cand, size_hist


# ------------------------------------------------------------------ prefix_2
P2_BANDS = [(4, 5, "count_4_5"), (6, 7, "count_6_7"), (8, 10, "count_8_10")]
P2_QUOTA = {"count_4_5": 5, "count_6_7": 5, "count_8_10": 5}


def sample_prefix2(pop: pd.DataFrame, rng: np.random.Generator):
    grouped = pop.groupby(["c1", "c2"], sort=True).indices
    cand: dict[str, list[dict]] = {b[2]: [] for b in P2_BANDS}
    for (c1, c2), rows in grouped.items():
        size = len(rows)
        if size < 4 or size > 10:
            continue
        c3n = int(pop.iloc[rows]["c3"].nunique())
        if c3n < 2:
            continue
        band = band_of(size, P2_BANDS)
        if band is None:
            continue
        cand[band].append(
            {"c1": int(c1), "c2": int(c2), "size": size, "c3_nunique": c3n,
             "rows": np.sort(rows)}
        )
    out_rows, group_meta = [], []
    gid = 0
    for lo, hi, band in P2_BANDS:
        pick = diverse_pick(cand[band], P2_QUOTA[band], rng)
        pick = sorted(pick, key=lambda g: (-g["size"], g["c1"], g["c2"]))
        for g in pick:
            gid += 1
            qid = f"P2-{gid:02d}"
            group_meta.append(
                {"qualitative_group_id": qid, "count_band": band,
                 "c1": g["c1"], "c2": g["c2"], "group_size": g["size"],
                 "c3_nunique": g["c3_nunique"],
                 "available_groups_in_band": len(cand[band])}
            )
            for r in g["rows"]:
                a = pop.iloc[int(r)]
                out_rows.append({
                    "qualitative_group_id": qid, "count_band": band,
                    "c1": g["c1"], "c2": g["c2"], "group_size": g["size"],
                    "c3_nunique": g["c3_nunique"],
                    "article_id": a["aid"], "c3": int(a["c3"]), "c4": int(a["c4"]),
                    "category_str": a["category_str"],
                    "title": a["title"], "subtitle": a["subtitle"], "body": a["body"],
                    "title_tokens": n_tokens(a["title"]),
                    "body_tokens": n_tokens(a["body"]),
                })
    return pd.DataFrame(out_rows), pd.DataFrame(group_meta), cand


# ------------------------------------------------------------------ semantic_different
def sample_semantic_different(pop: pd.DataFrame, rng: np.random.Generator,
                              n_main: int = 30, n_supp: int = 10):
    """(c1,c2,c3) group 두 개를 고르고 각 group에서 대표 기사 1개씩.

    main  : c1, c2, c3가 모두 다른 group-pair
    보조   : 위에 더해 c4까지 다른 pair (main과 별도 파일, 본 분석에서 제외)

    greedy: seed 42 permutation 순서로 A를 고정하고, 조건을 만족하는 첫 B를 붙인다.
    c1이 한쪽에 몰리지 않도록 c1 사용 횟수에 cap을 둔다 (막히면 cap을 1 올린다).
    """
    keys = list(pop.groupby(["c1", "c2", "c3"], sort=True).indices.items())
    reps = []
    for (c1, c2, c3), rows in keys:
        rows = np.sort(rows)
        r = int(rows[rng.integers(len(rows))])
        reps.append({"c1": int(c1), "c2": int(c2), "c3": int(c3),
                     "row": r, "group_size": len(rows)})

    order = [int(i) for i in rng.permutation(len(reps))]
    n_c1 = len({g["c1"] for g in reps})

    def build(n: int, require_c4: bool, used: set[int]):
        cap = max(1, -(-2 * n // max(n_c1, 1)))
        out: list[tuple[dict, dict]] = []
        c1_used: dict[int, int] = {}
        while len(out) < n:
            progressed = False
            for ia in order:
                if len(out) >= n:
                    break
                if ia in used:
                    continue
                A = reps[ia]
                if c1_used.get(A["c1"], 0) >= cap:
                    continue
                for ib in order:
                    if ib in used or ib == ia:
                        continue
                    B = reps[ib]
                    if A["c1"] == B["c1"] or A["c2"] == B["c2"] or A["c3"] == B["c3"]:
                        continue
                    if c1_used.get(B["c1"], 0) >= cap:
                        continue
                    if require_c4 and int(pop.iloc[A["row"]]["c4"]) == int(pop.iloc[B["row"]]["c4"]):
                        continue
                    out.append((A, B))
                    used.add(ia); used.add(ib)
                    c1_used[A["c1"]] = c1_used.get(A["c1"], 0) + 1
                    c1_used[B["c1"]] = c1_used.get(B["c1"], 0) + 1
                    progressed = True
                    break
            if len(out) >= n:
                break
            if not progressed:
                cap += 1
                if cap > 2 * n + 2:      # 안전장치
                    break
        return out

    used: set[int] = set()
    main = build(n_main, False, used)
    supp = build(n_supp, True, used)
    if len(main) < n_main or len(supp) < n_supp:
        raise ValueError(f"pair 부족: main {len(main)}/{n_main}, supp {len(supp)}/{n_supp}")

    def rows_of(pairs, tag):
        recs = []
        for k, (A, B) in enumerate(pairs, 1):
            pid = f"{tag}-{k:02d}"
            for side, G in (("A", A), ("B", B)):
                a = pop.iloc[G["row"]]
                recs.append({
                    "qualitative_group_id": pid, "side": side,
                    "c1": G["c1"], "c2": G["c2"], "c3": G["c3"], "c4": int(a["c4"]),
                    "source_group_size": G["group_size"],
                    "article_id": a["aid"], "category_str": a["category_str"],
                    "title": a["title"], "subtitle": a["subtitle"], "body": a["body"],
                    "title_tokens": n_tokens(a["title"]),
                    "body_tokens": n_tokens(a["body"]),
                })
        return pd.DataFrame(recs)

    return rows_of(main, "SD"), rows_of(supp, "SD4"), len(reps)


# ------------------------------------------------------------------ main
def save(df: pd.DataFrame, stem: str):
    df.to_parquet(OUT / f"{stem}.parquet", index=False)
    df.to_csv(OUT / f"{stem}.csv", index=False, encoding="utf-8-sig")
    print(f"  저장 {stem}  rows={len(df):,}")


def main():
    if OUT.exists() and any(OUT.iterdir()):
        raise SystemExit(f"출력 폴더가 비어있지 않습니다: {OUT} (기존 결과 보호)")
    OUT.mkdir(parents=True, exist_ok=True)

    print("모집단 로드 (read-only) ...")
    pop = load_population()
    print(f"  train + body 유효 기사 = {len(pop):,}")

    rng = np.random.default_rng(SEED)

    print("\nprefix_3 collision group 샘플링 ...")
    p3, p3_meta, p3_cand, p3_size_hist = sample_prefix3(pop, rng)
    print("\nprefix_2 group 샘플링 ...")
    p2, p2_meta, p2_cand = sample_prefix2(pop, rng)
    print("\nsemantic_different group-pair 샘플링 ...")
    sd, sd4, n_reps = sample_semantic_different(pop, rng)

    print("\n파일 저장 ...")
    save(p3, "qualitative_prefix3_groups")
    save(p3_meta, "qualitative_prefix3_group_index")
    save(p2, "qualitative_prefix2_groups")
    save(p2_meta, "qualitative_prefix2_group_index")
    save(sd, "qualitative_semantic_different")
    save(sd4, "qualitative_semantic_different_allcodes_supplementary")

    def band_summary(meta, band_col, art_df):
        rows = []
        for b in sorted(meta[band_col].unique()):
            sub = meta[meta[band_col] == b]
            arts = art_df[art_df[band_col] == b]
            rows.append({
                "band": b, "groups_selected": len(sub),
                "articles": len(arts),
                "available_groups_in_population": int(sub["available_groups_in_band"].iloc[0]),
                "distinct_c1": int(sub["c1"].nunique()),
                "distinct_c1_c2": int(sub.groupby(["c1", "c2"]).ngroups),
                "max_groups_sharing_one_c1": int(sub["c1"].value_counts().max()),
            })
        return pd.DataFrame(rows)

    p3_sum = band_summary(p3_meta, "size_band", p3)
    p2_sum = band_summary(p2_meta, "count_band", p2)
    save(p3_sum, "qualitative_prefix3_band_summary")
    save(p2_sum, "qualitative_prefix2_band_summary")

    total_elig_groups = sum(len(v) for v in p3_cand.values())
    real_dist = {
        "eligible_groups_size_2_3": len(p3_cand["size_2_3"]),
        "eligible_groups_size_4_5": len(p3_cand["size_4_5"]),
        "eligible_groups_size_6_8": len(p3_cand["size_6_8"]),
        "share_size_2_3_of_eligible": round(len(p3_cand["size_2_3"]) / total_elig_groups, 4),
    }
    manifest = {
        "seed": SEED,
        "population": {
            "description": "train split, body 비어있지 않은 기사 (Step 1 main analysis 모집단)",
            "n_articles": int(len(pop)),
            "input_files_read_only": [SID_PARQUET.name, ART_PARQUET.name],
        },
        "prefix_3": {
            "quota": P3_QUOTA,
            "groups_selected": int(len(p3_meta)),
            "articles_selected": int(len(p3)),
            "all_articles_of_each_group_included": True,
            "distinct_c1": int(p3_meta["c1"].nunique()),
            "distinct_c1_c2": int(p3_meta.groupby(["c1", "c2"]).ngroups),
            "max_groups_sharing_one_c1": int(p3_meta["c1"].value_counts().max()),
            "real_population_distribution": real_dist,
            "sampling_note": (
                "전체 분포에서 size 2~3 group이 eligible group의 "
                f"{real_dist['share_size_2_3_of_eligible']*100:.0f}%를 차지하지만, "
                "큰 collision에서 의미적 일관성이 무너지는 사례를 확인하기 위한 진단 목적으로 "
                "size 4~8 group을 의도적으로 oversampling했다. "
                "따라서 이 표본은 분포 비율 추정용 통계표본이 아니다."
            ),
            "group_size_histogram_population": {str(k): v for k, v in sorted(p3_size_hist.items())},
        },
        "prefix_2": {
            "eligibility": "기사 수 4~10 AND c3 종류 >= 2",
            "quota": P2_QUOTA,
            "groups_selected": int(len(p2_meta)),
            "articles_selected": int(len(p2)),
            "all_articles_of_each_group_included": True,
            "distinct_c1": int(p2_meta["c1"].nunique()),
            "distinct_c1_c2": int(p2_meta.groupby(["c1", "c2"]).ngroups),
            "max_groups_sharing_one_c1": int(p2_meta["c1"].value_counts().max()),
        },
        "semantic_different": {
            "definition": "c1, c2, c3가 모두 다른 (c1,c2,c3) group 두 개에서 대표 기사 1개씩",
            "source_groups_available": int(n_reps),
            "pairs_main": int(len(sd) // 2),
            "distinct_c1_main": int(sd["c1"].nunique()),
            "supplementary_allcodes_different": {
                "definition": "c1,c2,c3 에 더해 c4까지 모두 다른 pair",
                "pairs": int(len(sd4) // 2),
                "excluded_from_main_analysis": True,
                "note": "c4는 collision 구분용 suffix이므로 본 분석의 pair 조건으로 쓰지 않는다.",
            },
        },
        "files": sorted(p.name for p in OUT.iterdir()),
    }
    (OUT / "step2_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== prefix_3 band summary ===")
    print(p3_sum.to_string(index=False))
    print("\n=== prefix_2 band summary ===")
    print(p2_sum.to_string(index=False))
    print("\n=== semantic_different ===")
    print(f"  main pair {len(sd)//2}  (distinct c1 {sd['c1'].nunique()})")
    print(f"  보조(c4까지 다름) pair {len(sd4)//2}  — 본 분석 제외")
    print("\n=== 선택된 prefix_3 group ===")
    print(p3_meta[["qualitative_group_id", "size_band", "c1", "c2", "c3", "group_size"]].to_string(index=False))
    print("\n=== 선택된 prefix_2 group ===")
    print(p2_meta[["qualitative_group_id", "count_band", "c1", "c2", "group_size", "c3_nunique"]].to_string(index=False))
    print(f"\n완료. 출력 → {OUT}")


if __name__ == "__main__":
    main()
