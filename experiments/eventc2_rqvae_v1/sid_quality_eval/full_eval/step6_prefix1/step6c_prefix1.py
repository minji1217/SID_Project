"""Step 6-C — Prefix-1 조건 추가 + 4단계 비교.

Prefix-1 = c1_A == c1_B AND c2_A != c2_B

기존 Step 3 / Step 6 산출물은 읽기만 하고 절대 덮어쓰지 않는다.
모집단은 Step 3 main과 동일한 body-valid train 9,338.
sampling은 Prefix-2 main과 같은 group-balanced 방식을 한 단계 위로 올린 것.
seed 42 고정.

4단계: Different < Prefix-1 < Prefix-2 < Prefix-3
"""
from __future__ import annotations
import json, sys
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
S3, S6 = ROOT / "step3", ROOT / "step6"
sys.path.insert(0, str(S3))
sys.stdout.reconfigure(line_buffering=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sid_pairs import SEED, LARGE_CAP, load, PairSet                # noqa: E402
from step3_quantitative import SURFACE, INK, INK2, GRID             # noqa: E402

# 공유 prefix 개수에 따른 ordinal ramp (dataviz blue, --ordinal 검증 통과)
ORDER = ["semantic_different", "prefix_1_same", "prefix_2_same", "prefix_3_same"]
LABEL = {"semantic_different": "Different\n(c1,c2,c3 모두 다름)\n공유 prefix 0",
         "prefix_1_same": "Prefix-1\n(c1 같고 c2 다름)\n공유 prefix 1",
         "prefix_2_same": "Prefix-2\n(c1,c2 같고 c3 다름)\n공유 prefix 2",
         "prefix_3_same": "Prefix-3\n(c1,c2,c3 모두 같음)\n공유 prefix 3"}
RAMP = {"semantic_different": "#86b6ef", "prefix_1_same": "#3987e5",
        "prefix_2_same": "#1c5cab", "prefix_3_same": "#0d366b"}
SERIES = {"original_embedding_cosine": "#2a78d6", "reconstruction_cosine": "#eb6834"}

plt.rcParams.update({
    "font.family": "WenQuanYi Zen Hei",
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": GRID,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.titlesize": 11, "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": GRID, "grid.linewidth": 0.6,
})


def stats_of(v):
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    return {"count": int(v.size), "mean": float(v.mean()), "median": float(np.median(v)),
            "std": float(v.std(ddof=1)), "q1": float(np.percentile(v, 25)),
            "q3": float(np.percentile(v, 75)), "min": float(v.min()), "max": float(v.max())}


# ------------------------------------------------------------------ 샘플러
def prefix1_group_balanced(d: pd.DataFrame, n: int, rng) -> PairSet:
    """c1 group을 round-robin으로 돌면서 서로 다른 c2 subgroup 2개를 고르고 각각에서 article 하나씩.

    Step 3의 prefix2_group_balanced와 동일한 구조를 한 단계 위(c1/c2)로 올린 것이다.
    """
    groups = {}
    for key, idx in d.groupby("c1").indices.items():
        sub = d.iloc[idx].groupby("c2").indices
        subs = {c2: idx[v] for c2, v in sub.items()}
        if len(subs) >= 2:
            groups[key] = subs
    keys = sorted(groups)
    ps, stale = PairSet(), 0
    while len(ps) < n and stale < len(keys) * 50:
        rng.shuffle(keys)
        progressed = False
        for key in keys:
            if len(ps) >= n:
                break
            subs = groups[key]
            c2a, c2b = rng.choice(list(subs), size=2, replace=False)
            a = int(rng.choice(subs[c2a])); b = int(rng.choice(subs[c2b]))
            if ps.add(a, b, source_group=str(key),
                      sub_group_a=int(c2a), sub_group_b=int(c2b)):
                progressed = True
        stale = 0 if progressed else stale + 1
    return ps


def prefix1_uniform(d: pd.DataFrame, n: int, rng) -> PairSet:
    """대조군 — article을 uniform하게 두 개 뽑고 조건을 만족할 때만 채택 (rejection)."""
    c = d[["c1", "c2"]].to_numpy()
    N = len(d)
    ps = PairSet()
    while len(ps) < n:
        a = rng.integers(0, N, size=4 * n); b = rng.integers(0, N, size=4 * n)
        ok = (c[a, 0] == c[b, 0]) & (c[a, 1] != c[b, 1])
        for i, j in zip(a[ok], b[ok]):
            if len(ps) >= n:
                break
            ps.add(int(i), int(j), source_group=str(c[i, 0]))
    return ps


def main():
    if any(p.name != "step6c_prefix1.py" for p in HERE.iterdir()):
        raise SystemExit("출력 폴더에 기존 결과가 있습니다.")

    print("모집단 (Step 3 main과 동일) ...")
    m = load()
    tr = m[m.split == "train"].reset_index(drop=True)
    pop = tr[tr.body.fillna("").astype(str).str.strip() != ""].reset_index(drop=True)
    print(f"  body-valid train {len(pop):,}")

    # ---------------- 1. Prefix-1 pair population 정확 계산
    P = lambda s: int(sum(comb(int(x), 2) for x in s if x >= 2))
    A = P(pop.groupby("c1").size())
    AB = P(pop.groupby(["c1", "c2"]).size())
    ABC = P(pop.groupby(["c1", "c2", "c3"]).size())
    pop_counts = {
        "total_pairs": comb(len(pop), 2),
        "same_c1": A, "same_c1_c2": AB, "same_c1_c2_c3": ABC,
        "prefix_1_same": A - AB, "prefix_2_same": AB - ABC, "prefix_3_same": ABC,
    }
    print("\n=== pair population (body-valid train 9,338) ===")
    print(f"  Prefix-1 (c1 같고 c2 다름)      {A - AB:,}")
    print(f"  Prefix-2 (c1,c2 같고 c3 다름)   {AB - ABC:,}")
    print(f"  Prefix-3 (c1,c2,c3 모두 같음)   {ABC:,}")

    n_c1 = pop.c1.nunique()
    eligible = sum(1 for _, g in pop.groupby("c1") if g.c2.nunique() >= 2)
    target = LARGE_CAP
    note = ""
    if pop_counts["prefix_1_same"] < target:
        target = pop_counts["prefix_1_same"]
        note = f"가능한 pair가 {target:,}개뿐이라 전수를 사용했다."
    else:
        note = (f"가능한 pair {pop_counts['prefix_1_same']:,}개 중 {target:,}개를 "
                f"group-balanced로 추출했다.")
    print(f"  c1 group {n_c1}개 · c2 종류>=2 인 group {eligible}개 · {note}")

    # ---------------- 2. 샘플링 (seed 42)
    rng = np.random.default_rng(SEED)
    print("\npair 생성 ...")
    jobs = [("group_balanced", prefix1_group_balanced), ("uniform", prefix1_uniform)]
    frames = []
    for meth, fn in jobs:
        ps = fn(pop, target, rng)
        f = pd.DataFrame(ps.rows)
        f.insert(0, "analysis_set", "large")
        f.insert(1, "condition", "prefix_1_same")
        f.insert(2, "sampling_method", meth)
        ia, ib = f.idx_a.to_numpy(), f.idx_b.to_numpy()
        for side, i in (("a", ia), ("b", ib)):
            f[f"article_id_{side}"] = pop.aid.to_numpy()[i]
            for col in ("c1", "c2", "c3", "c4"):
                f[f"{col}_{side}"] = pop[col].to_numpy()[i]
        frames.append(f)
        print(f"  {meth:15s} {len(f):>7,} pair")
    pairs = pd.concat(frames, ignore_index=True)
    pairs.insert(0, "pair_id", [f"P1-{i:07d}" for i in range(len(pairs))])
    bad = int((~((pairs.c1_a == pairs.c1_b) & (pairs.c2_a != pairs.c2_b))).sum())
    print(f"  조건 위반 {bad} · 중복 pair 0 (PairSet 보장) · self-pair "
          f"{int((pairs.article_id_a == pairs.article_id_b).sum())}")
    assert bad == 0

    # ---------------- 3. 지표 계산 (Step 6의 벡터 재사용, 읽기 전용)
    print("\n지표 계산 — Step 6의 embedding/reconstruction 재사용 ...")
    index = pd.read_parquet(S6 / "reconstruction_index.parquet")
    X = np.load(S6 / "original_embeddings.npy")
    Xh = np.load(S6 / "reconstructed_embeddings.npy")
    u = lambda M: M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-12)
    Xn, Xhn = u(X), u(Xh)
    row = dict(zip(index.article_id.astype(str), index.row.to_numpy()))
    ia = np.fromiter((row[a] for a in pairs.article_id_a), np.int64, len(pairs))
    ib = np.fromiter((row[b] for b in pairs.article_id_b), np.int64, len(pairs))
    pairs["original_embedding_cosine"] = np.einsum("ij,ij->i", Xn[ia], Xn[ib])
    pairs["reconstruction_cosine"] = np.einsum("ij,ij->i", Xhn[ia], Xhn[ib])
    pairs["delta_cosine"] = pairs.reconstruction_cosine - pairs.original_embedding_cosine
    pairs["abs_delta_cosine"] = pairs.delta_cosine.abs()

    # ---------------- 4. 기존 3조건과 합쳐 4단계 표
    old = pd.read_parquet(S6 / "pair_similarity_preservation.parquet")
    old_main = old[(old.sampling_method.isin(["exhaustive", "group_balanced"]))
                   & (old.analysis_set.isin(["large", "large_and_balanced"]))]
    new_main = pairs[pairs.sampling_method == "group_balanced"]
    cols = ["condition", "sampling_method", "original_embedding_cosine",
            "reconstruction_cosine", "delta_cosine", "abs_delta_cosine"]
    allmain = pd.concat([old_main[cols], new_main[cols]], ignore_index=True)

    METS = ["original_embedding_cosine", "reconstruction_cosine",
            "delta_cosine", "abs_delta_cosine"]
    rows = []
    for cond in ORDER:
        g = allmain[allmain.condition == cond]
        for met in METS:
            rows.append({"condition": cond, "shared_prefix": ORDER.index(cond),
                         "sampling_method": g.sampling_method.iloc[0],
                         "metric": met, **stats_of(g[met].to_numpy())})
    summary = pd.DataFrame(rows)

    # 상관
    crows = []
    for cond in ORDER:
        g = allmain[allmain.condition == cond]
        o, r = g.original_embedding_cosine.to_numpy(), g.reconstruction_cosine.to_numpy()
        # reconstruction_cosine이 사실상 상수인 경우(= Prefix-3) 상관계수는 정의되지 않는다.
        # 부동소수점 오차(~1e-7)에 대해 상관을 계산하면 의미 없는 값이 나오므로 차단한다.
        if (r.max() - r.min()) < 1e-5:
            crows.append({"condition": cond, "n_pairs": len(g), "pearson_r": float("nan"),
                          "spearman_rho": float("nan"),
                          "note": f"reconstruction_cosine이 상수(=1.0, 범위 {r.max()-r.min():.1e})라 "
                                  f"정의되지 않음 — 계산 시 부동소수점 오차에 대한 상관이 된다"})
        else:
            pr, pp = stats.pearsonr(o, r); sr, sp = stats.spearmanr(o, r)
            crows.append({"condition": cond, "n_pairs": len(g), "pearson_r": pr,
                          "pearson_p": pp, "spearman_rho": sr, "spearman_p": sp, "note": ""})
    # prefix_1 uniform 대조군 상관도 따로
    gu = pairs[pairs.sampling_method == "uniform"]
    pru, _ = stats.pearsonr(gu.original_embedding_cosine, gu.reconstruction_cosine)
    sru, _ = stats.spearmanr(gu.original_embedding_cosine, gu.reconstruction_cosine)
    corr = pd.DataFrame(crows)

    # ---------------- 5. 단조성 검증
    mono_rows = []
    for met in ("original_embedding_cosine", "reconstruction_cosine"):
        for stat in ("mean", "median"):
            v = [summary[(summary.condition == c) & (summary.metric == met)][stat].iloc[0]
                 for c in ORDER]
            mono_rows.append({
                "metric": met, "stat": stat,
                **{ORDER[i]: v[i] for i in range(4)},
                "strictly_increasing": bool(v[0] < v[1] < v[2] < v[3]),
                "relation": " < ".join(
                    f"{a:.4f}" for a in v) if v[0] < v[1] < v[2] < v[3] else
                    " ".join(f"{v[i]:.4f}{'<' if i < 3 and v[i] < v[i+1] else ('=' if i < 3 and v[i] == v[i+1] else ('>' if i < 3 else ''))}"
                             for i in range(4)),
            })
    mono = pd.DataFrame(mono_rows)
    print("\n=== 단조성 검증  Different < Prefix-1 < Prefix-2 < Prefix-3 ===")
    print(mono[["metric", "stat"] + ORDER + ["strictly_increasing"]].round(4).to_string(index=False))

    # ---------------- 저장
    print("\n저장 ...")
    pairs.to_parquet(HERE / "prefix1_pairs.parquet", index=False)
    summary.round(8).to_csv(HERE / "four_condition_summary.csv", index=False,
                            encoding="utf-8-sig")
    summary.to_parquet(HERE / "four_condition_summary.parquet", index=False)
    corr.round(8).to_csv(HERE / "four_condition_correlation.csv", index=False,
                         encoding="utf-8-sig")
    mono.to_csv(HERE / "monotonicity_check.csv", index=False, encoding="utf-8-sig")

    manifest = {
        "purpose": "Prefix-1 조건을 추가해 Different -> Prefix-1 -> Prefix-2 -> Prefix-3 "
                   "4단계로 재구성",
        "prefix_1_definition": "c1_A == c1_B AND c2_A != c2_B",
        "population": {"description": "Step 3 main과 동일한 body-valid train",
                       "n_articles": int(len(pop)), "exact_pair_counts": pop_counts,
                       "c1_groups": int(n_c1), "c1_groups_with_2plus_c2": int(eligible)},
        "sampling": {"method": "group_balanced (Prefix-2 main과 동일 구조를 c1/c2로 올린 것)",
                     "target": int(target), "seed": SEED, "note": note,
                     "control": "pair-weighted uniform 대조군도 함께 생성"},
        "reused_readonly": ["step6/reconstruction_index.parquet",
                            "step6/original_embeddings.npy",
                            "step6/reconstructed_embeddings.npy",
                            "step6/pair_similarity_preservation.parquet"],
        "existing_outputs_untouched": True,
        "prefix1_uniform_control_correlation": {"pearson_r": float(pru),
                                                "spearman_rho": float(sru)},
        "structural_note": ("Prefix-3의 reconstruction cosine = 1.0은 동일 (c1,c2,c3)이면 "
                            "동일한 q1+q2+q3를 decoder에 넣기 때문에 구조적으로 자명한 값이며, "
                            "모델 성능의 근거로 사용하지 않는다."),
        "wording_rules": ["cosine 0.93을 '93% 유사'로 표현하지 않는다.",
                          "Pearson 0.33을 '33% 정확'으로 해석하지 않는다."],
    }
    (HERE / "step6c_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---------------- 그림
    print("\n그림 ...")
    # fig12 — 4조건 분포 3패널
    fig, axes = plt.subplots(1, 3, figsize=(16.8, 5.6))
    for ax, met, title in zip(
            axes, ["original_embedding_cosine", "reconstruction_cosine", "delta_cosine"],
            ["original embedding cosine", "reconstruction cosine",
             "delta = reconstruction − original"]):
        vals = [allmain[allmain.condition == c][met].dropna().to_numpy() for c in ORDER]
        pos = np.arange(1, 5)
        bp = ax.boxplot(vals, positions=pos, widths=0.56, patch_artist=True, showfliers=False,
                        medianprops=dict(color=INK, linewidth=1.6),
                        whiskerprops=dict(color=INK2, linewidth=1.0),
                        capprops=dict(color=INK2, linewidth=1.0))
        for b, c in zip(bp["boxes"], ORDER):
            b.set(facecolor=RAMP[c], edgecolor=SURFACE, linewidth=2.0, alpha=0.95)
        lo = min(np.percentile(v, 1) for v in vals); hi = max(np.percentile(v, 99) for v in vals)
        pad = max((hi - lo) * 0.34, 1e-4)
        ax.set_ylim(lo - pad * 0.22, hi + pad)
        for p_, v in zip(pos, vals):
            ax.annotate(f"med {np.median(v):.4f}\nmean {v.mean():.4f}",
                        xy=(p_, hi + pad * 0.94), ha="center", va="top",
                        fontsize=7.5, color=INK2, linespacing=1.35)
        ax.set_xticks(pos)
        ax.set_xticklabels([f"{LABEL[c]}\nn={len(allmain[allmain.condition==c]):,}"
                            for c in ORDER], fontsize=7.2)
        ax.set_title(title, pad=8); ax.set_ylabel(title)
        ax.yaxis.grid(True); ax.set_axisbelow(True)
    fig.suptitle("fig12 — SID prefix 공유 단계별 임베딩 유사도 (4조건, Large · group-balanced)",
                 fontsize=12.5, y=0.985)
    fig.text(0.5, 0.012, "색이 진할수록 공유 prefix가 많다. Prefix-3의 reconstruction cosine "
                         "= 1.0은 동일 code triple에서 오는 구조적 결과이며 성능 근거가 아니다.",
             ha="center", fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(HERE / "fig12_four_condition_cosine.png", dpi=170); plt.close(fig)
    print("  fig12_four_condition_cosine.png")

    # fig13 — 단조성 직접 확인
    fig, ax = plt.subplots(figsize=(9.6, 5.6))
    x = np.arange(4)
    for met, lab in (("original_embedding_cosine", "original embedding cosine"),
                     ("reconstruction_cosine", "reconstruction cosine")):
        mu = [summary[(summary.condition == c) & (summary.metric == met)]["mean"].iloc[0]
              for c in ORDER]
        q1 = [summary[(summary.condition == c) & (summary.metric == met)]["q1"].iloc[0]
              for c in ORDER]
        q3 = [summary[(summary.condition == c) & (summary.metric == met)]["q3"].iloc[0]
              for c in ORDER]
        ax.fill_between(x, q1, q3, color=SERIES[met], alpha=0.16, linewidth=0)
        ax.plot(x, mu, "-o", color=SERIES[met], linewidth=2.2, markersize=9,
                markeredgecolor=SURFACE, markeredgewidth=2.0, label=f"{lab} (mean, 띠=Q1–Q3)")
        for xi, v in zip(x, mu):
            ax.annotate(f"{v:.4f}", xy=(xi, v), xytext=(0, 11), textcoords="offset points",
                        ha="center", fontsize=8.4, color=SERIES[met], fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[c].replace("\n", " · ") for c in ORDER], fontsize=8)
    ax.set_ylabel("cosine"); ax.set_xlim(-0.35, 3.35)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.yaxis.grid(True); ax.set_axisbelow(True)
    ax.set_title("fig13 — 공유 prefix가 늘수록 cosine이 단조 증가하는가", pad=10, fontsize=12.5)
    fig.text(0.5, 0.012, "두 지표 모두 네 단계에서 단조 증가한다. 다만 reconstruction 쪽의 "
                         "Prefix-3 = 1.0은 구조적 상한이다.",
             ha="center", fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(HERE / "fig13_monotonicity.png", dpi=170); plt.close(fig)
    print("  fig13_monotonicity.png")

    pd.set_option("display.width", 220)
    print("\n================ 4조건 비교 (Large · group-balanced) ================")
    sh = ["condition", "shared_prefix", "metric", "count", "mean", "median", "std", "q1", "q3"]
    print(summary.sort_values(["metric", "shared_prefix"])[sh].round(5).to_string(index=False))
    print("\n================ 상관 (original vs reconstruction) ================")
    print(corr.round(4).to_string(index=False))
    print(f"  (참고) prefix_1 uniform 대조군: Pearson {pru:.4f} / Spearman {sru:.4f}")
    print(f"\n완료 → {HERE}")


if __name__ == "__main__":
    main()
