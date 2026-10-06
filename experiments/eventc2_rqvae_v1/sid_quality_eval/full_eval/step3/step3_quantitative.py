"""Step 3 — RQ-VAE SID 정량 분석 (Jaccard / TF-IDF cosine).

모집단: body-valid train 9,338개 전체 (Step 2의 정성 표본과 무관한 별도 pair sample).
입력 parquet은 읽기 전용. seed 42 고정.

pair sample
  Large-sample      semantic_different 50,000 / prefix_2_same 50,000 / prefix_3_same 1,534(전체)
  Balanced          세 조건 모두 1,534
  대조군            pair-weighted uniform random (두 scale 모두)

metric
  title_jaccard      title token Jaccard
  body_jaccard       body token Jaccard
  body_tfidf_cosine  body TF-IDF cosine (9,338 corpus에 1회 fit, 전 article 동일 space로 transform)

sensitivity
  전체 train 9,738개 (body 빈 기사 포함) · title Jaccard only
"""
from __future__ import annotations
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.stdout.reconfigure(line_buffering=True)

from sid_pairs import (                                     # noqa: E402
    SEED, LARGE_CAP, load, exact_counts, tokenize, jaccard, build_tfidf,
    TFIDF_SETTINGS, PairSet, all_prefix3_pairs,
    prefix2_group_balanced, prefix2_uniform,
    semantic_different_group_balanced, semantic_different_uniform,
)

OUT = HERE
METRICS = ["title_jaccard", "body_jaccard", "body_tfidf_cosine"]
METRIC_LABEL = {
    "title_jaccard": "Title token Jaccard",
    "body_jaccard": "Body token Jaccard",
    "body_tfidf_cosine": "Body TF-IDF cosine",
}
CONDITIONS = ["prefix_3_same", "prefix_2_same", "semantic_different"]
COND_LABEL = {
    "prefix_3_same": "prefix_3\n(c1,c2,c3 동일)",
    "prefix_2_same": "prefix_2\n(c1,c2 동일, c3 다름)",
    "semantic_different": "semantic_different\n(c1,c2,c3 모두 다름)",
}
# dataviz 기본 categorical palette slot 1~3 (light). 3 slot all-pairs 검증 통과.
COND_COLOR = {"prefix_3_same": "#2a78d6", "prefix_2_same": "#eb6834",
              "semantic_different": "#1baf7a"}
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#d9d8d4"

plt.rcParams.update({
    "font.family": "WenQuanYi Zen Hei",
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": GRID,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.titlesize": 11, "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": GRID, "grid.linewidth": 0.6,
})


# ------------------------------------------------------------------ 통계
def stats_of(values: np.ndarray, n_pairs: int) -> dict:
    v = values[~np.isnan(values)]
    if v.size == 0:
        return {"n_pairs": n_pairs, "count": 0, "n_nan": n_pairs,
                **{k: float("nan") for k in
                   ("mean", "median", "std", "q1", "q3", "min", "max")}}
    return {
        "n_pairs": n_pairs, "count": int(v.size), "n_nan": int(n_pairs - v.size),
        "mean": float(v.mean()), "median": float(np.median(v)),
        "std": float(v.std(ddof=1)) if v.size > 1 else float("nan"),
        "q1": float(np.percentile(v, 25)), "q3": float(np.percentile(v, 75)),
        "min": float(v.min()), "max": float(v.max()),
    }


def summarize(pairs: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows = []
    keys = ["analysis_set", "condition", "sampling_method"]
    for key, g in pairs.groupby(keys, sort=False):
        for m in metrics:
            rows.append(dict(zip(keys, key), metric=m,
                             **stats_of(g[m].to_numpy(dtype=float), len(g))))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ pair 생성
def pair_frame(ps: PairSet, pop: pd.DataFrame, title_sets, body_sets, X,
               condition: str, method: str, analysis_set: str) -> pd.DataFrame:
    f = pd.DataFrame(ps.rows)
    ia, ib = f.idx_a.to_numpy(), f.idx_b.to_numpy()
    f["title_jaccard"] = [jaccard(title_sets[a], title_sets[b]) for a, b in zip(ia, ib)]
    f["body_jaccard"] = [jaccard(body_sets[a], body_sets[b]) for a, b in zip(ia, ib)]
    if X is not None:
        f["body_tfidf_cosine"] = np.asarray(X[ia].multiply(X[ib]).sum(axis=1)).ravel()
    f.insert(0, "analysis_set", analysis_set)
    f.insert(1, "condition", condition)
    f.insert(2, "sampling_method", method)
    for side, idx in (("a", ia), ("b", ib)):
        f[f"article_id_{side}"] = pop.aid.to_numpy()[idx]
        for c in ("c1", "c2", "c3", "c4"):
            f[f"{c}_{side}"] = pop[c].to_numpy()[idx]
    return f


# ------------------------------------------------------------------ 그림
def panel(ax, data: dict[str, np.ndarray], metric: str, kind: str, title: str):
    conds = [c for c in CONDITIONS if c in data and data[c].size]
    vals = [data[c] for c in conds]
    pos = np.arange(1, len(conds) + 1)
    if kind == "box":
        bp = ax.boxplot(vals, positions=pos, widths=0.5, patch_artist=True,
                        showfliers=False, medianprops=dict(color=INK, linewidth=1.6),
                        whiskerprops=dict(color=INK2, linewidth=1.0),
                        capprops=dict(color=INK2, linewidth=1.0))
        for b, c in zip(bp["boxes"], conds):
            b.set(facecolor=COND_COLOR[c], edgecolor=SURFACE, linewidth=2.0, alpha=0.92)
    else:
        vp = ax.violinplot(vals, positions=pos, widths=0.72,
                           showmedians=True, showextrema=False)
        for body, c in zip(vp["bodies"], conds):
            body.set(facecolor=COND_COLOR[c], edgecolor=SURFACE, linewidth=2.0, alpha=0.88)
        vp["cmedians"].set(color=INK, linewidth=1.6)
    # relief rule: 직접 레이블 (median / mean)
    top = max(np.percentile(v, 97.5) for v in vals)
    top = max(top, 1e-6)
    ax.set_ylim(-0.03 * top, top * 1.34)
    for p, v in zip(pos, vals):
        ax.annotate(f"med {np.median(v):.3f}\nmean {v.mean():.3f}",
                    xy=(p, top * 1.30), ha="center", va="top",
                    fontsize=7.6, color=INK2, linespacing=1.35)
    ax.set_xticks(pos)
    ax.set_xticklabels([f"{COND_LABEL[c]}\nn={len(data[c]):,}" for c in conds],
                       fontsize=7.6)
    ax.set_title(title, pad=8)
    ax.set_ylabel(METRIC_LABEL[metric])
    ax.yaxis.grid(True); ax.set_axisbelow(True)


def figure(pairs: pd.DataFrame, metrics: list[str], kind: str,
           sel: dict, suptitle: str, note: str, fname: str,
           width: float | None = None):
    sub = pairs
    for k, v in sel.items():
        sub = sub[sub[k].isin(v) if isinstance(v, (list, tuple)) else sub[k] == v]
    w = width if width is not None else 4.6 * len(metrics)
    fig, axes = plt.subplots(1, len(metrics), figsize=(w, 4.9))
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, metrics):
        data = {c: sub[sub.condition == c][m].dropna().to_numpy()
                for c in CONDITIONS}
        data = {k: v for k, v in data.items() if v.size}
        panel(ax, data, m, kind, METRIC_LABEL[m])
    fig.suptitle(suptitle, fontsize=12.5, y=0.985)
    fig.text(0.5, 0.012, note, ha="center", fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    p = OUT / fname
    fig.savefig(p, dpi=170)
    plt.close(fig)
    print(f"  그림 {fname}")
    return p


def method_figure(pairs: pd.DataFrame, fname: str):
    """group-balanced vs pair-weighted uniform 대조."""
    conds = ["prefix_2_same", "semantic_different"]
    fig, axes = plt.subplots(len(conds), len(METRICS),
                             figsize=(4.4 * len(METRICS), 3.9 * len(conds)))
    for r, cond in enumerate(conds):
        for c, m in enumerate(METRICS):
            ax = axes[r, c]
            vals, labels, colors = [], [], []
            for meth, hatch in (("group_balanced", None), ("uniform", "//")):
                s = pairs[(pairs.analysis_set == "large") & (pairs.condition == cond)
                          & (pairs.sampling_method == meth)][m].dropna().to_numpy()
                if s.size:
                    vals.append(s); labels.append(f"{meth}\nn={s.size:,}")
                    colors.append((COND_COLOR[cond], hatch))
            pos = np.arange(1, len(vals) + 1)
            bp = ax.boxplot(vals, positions=pos, widths=0.46, patch_artist=True,
                            showfliers=False, medianprops=dict(color=INK, linewidth=1.6),
                            whiskerprops=dict(color=INK2, linewidth=1.0),
                            capprops=dict(color=INK2, linewidth=1.0))
            for b, (col, hatch) in zip(bp["boxes"], colors):
                b.set(facecolor=col, edgecolor=SURFACE, linewidth=2.0, alpha=0.92)
                if hatch:
                    b.set(hatch=hatch, edgecolor=SURFACE)
            top = max(max(np.percentile(v, 97.5) for v in vals), 1e-6)
            ax.set_ylim(-0.03 * top, top * 1.38)
            for p_, v in zip(pos, vals):
                ax.annotate(f"med {np.median(v):.3f}\nmean {v.mean():.3f}",
                            xy=(p_, top * 1.34), ha="center", va="top",
                            fontsize=7.4, color=INK2, linespacing=1.35)
            ax.set_xticks(pos); ax.set_xticklabels(labels, fontsize=7.6)
            ax.set_ylabel(METRIC_LABEL[m] if c == 0 else "")
            ax.set_title(f"{cond} · {METRIC_LABEL[m]}", fontsize=9.5, pad=7)
            ax.yaxis.grid(True); ax.set_axisbelow(True)
    fig.suptitle("샘플링 방식 대조 — group-balanced vs pair-weighted uniform (Large-sample 50,000)",
                 fontsize=12.5, y=0.99)
    fig.text(0.5, 0.008, "사선 채움 = uniform 대조군. prefix_3_same은 전체 1,534 pair를 "
                         "모두 쓰므로 두 방식이 동일하여 제외.", ha="center", fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 0.965))
    fig.savefig(OUT / fname, dpi=170)
    plt.close(fig)
    print(f"  그림 {fname}")


# ------------------------------------------------------------------ main
def main():
    existing = {p.name for p in OUT.iterdir()} - {"sid_pairs.py", "step3_quantitative.py",
                                                  "__pycache__"}
    if existing:
        raise SystemExit(f"출력 폴더에 기존 결과가 있습니다: {sorted(existing)}")

    print("입력 로드 (read-only) ...")
    m = load()
    tr_all = m[m.split == "train"].reset_index(drop=True)
    body_all = tr_all.body.fillna("").astype(str).str.strip()
    pop = tr_all[body_all != ""].reset_index(drop=True)
    print(f"  전체 train SID 기사 {len(tr_all):,}  ·  body-valid {len(pop):,}")

    counts_pop = exact_counts(pop)
    counts_all = exact_counts(tr_all)
    P3_N = counts_pop["prefix_3_same"]
    print(f"  모집단 전체 pair 수: prefix_3 {P3_N:,} · prefix_2 "
          f"{counts_pop['prefix_2_same']:,} · semantic_different {counts_pop['semantic_different']:,}")

    print("\n토큰화 ...")
    title_sets = [set(tokenize(t)) for t in pop.title]
    body_sets = [set(tokenize(b)) for b in pop.body]
    body_tok = [tokenize(b) for b in pop.body]

    print("TF-IDF fit (9,338 body corpus 전체에 1회) ...")
    vec, X = build_tfidf(body_tok)
    print(f"  문서 {X.shape[0]:,}  vocab {X.shape[1]:,}  nnz {X.nnz:,}  norm=l2")

    rng = np.random.default_rng(SEED)
    jobs = [
        ("prefix_3_same",      "exhaustive",     "large_and_balanced", P3_N,
         lambda n: all_prefix3_pairs(pop)),
        ("prefix_2_same",      "group_balanced", "large",     LARGE_CAP,
         lambda n: prefix2_group_balanced(pop, n, rng)),
        ("semantic_different", "group_balanced", "large",     LARGE_CAP,
         lambda n: semantic_different_group_balanced(pop, n, rng)),
        ("prefix_2_same",      "uniform",        "large",     LARGE_CAP,
         lambda n: prefix2_uniform(pop, n, rng)),
        ("semantic_different", "uniform",        "large",     LARGE_CAP,
         lambda n: semantic_different_uniform(pop, n, rng)),
        ("prefix_2_same",      "group_balanced", "balanced",  P3_N,
         lambda n: prefix2_group_balanced(pop, n, rng)),
        ("semantic_different", "group_balanced", "balanced",  P3_N,
         lambda n: semantic_different_group_balanced(pop, n, rng)),
        ("prefix_2_same",      "uniform",        "balanced",  P3_N,
         lambda n: prefix2_uniform(pop, n, rng)),
        ("semantic_different", "uniform",        "balanced",  P3_N,
         lambda n: semantic_different_uniform(pop, n, rng)),
    ]
    print("\npair 생성 + metric 계산 ...")
    frames = []
    for cond, meth, aset, n, fn in jobs:
        ps = fn(n)
        f = pair_frame(ps, pop, title_sets, body_sets, X, cond, meth, aset)
        frames.append(f)
        print(f"  {aset:18s} {cond:19s} {meth:15s} {len(f):>7,} pair "
              f"(목표 {n:,})")
    pairs = pd.concat(frames, ignore_index=True)
    pairs.insert(0, "pair_id", [f"S3-{i:07d}" for i in range(len(pairs))])

    print("\nsensitivity — 전체 train 9,738, title Jaccard only ...")
    rng_s = np.random.default_rng(SEED)
    t_sets_all = [set(tokenize(t)) for t in tr_all.title]
    b_sets_all = [set(tokenize(b)) for b in tr_all.body]
    P3_ALL = counts_all["prefix_3_same"]
    s_jobs = [
        ("prefix_3_same",      "exhaustive",     "large_and_balanced", P3_ALL,
         lambda n: all_prefix3_pairs(tr_all)),
        ("prefix_2_same",      "group_balanced", "large",    LARGE_CAP,
         lambda n: prefix2_group_balanced(tr_all, n, rng_s)),
        ("semantic_different", "group_balanced", "large",    LARGE_CAP,
         lambda n: semantic_different_group_balanced(tr_all, n, rng_s)),
        ("prefix_2_same",      "group_balanced", "balanced", P3_ALL,
         lambda n: prefix2_group_balanced(tr_all, n, rng_s)),
        ("semantic_different", "group_balanced", "balanced", P3_ALL,
         lambda n: semantic_different_group_balanced(tr_all, n, rng_s)),
    ]
    s_frames = []
    for cond, meth, aset, n, fn in s_jobs:
        ps = fn(n)
        f = pair_frame(ps, tr_all, t_sets_all, b_sets_all, None, cond, meth, aset)
        f = f.drop(columns=["body_jaccard"])
        s_frames.append(f)
        print(f"  {aset:18s} {cond:19s} {len(f):>7,} pair")
    sens = pd.concat(s_frames, ignore_index=True)
    sens.insert(0, "pair_id", [f"S3T-{i:07d}" for i in range(len(sens))])

    print("\nsummary ...")
    summary = summarize(pairs, METRICS)
    sens_summary = summarize(sens, ["title_jaccard"])

    print("저장 ...")
    for df, stem in [(pairs, "quantitative_pairs"), (summary, "quantitative_summary"),
                     (sens, "sensitivity_title_only_pairs"),
                     (sens_summary, "sensitivity_title_only_summary")]:
        df.to_parquet(OUT / f"{stem}.parquet", index=False)
        if "pairs" not in stem:
            df.to_csv(OUT / f"{stem}.csv", index=False, encoding="utf-8-sig")
        print(f"  {stem}  rows={len(df):,}")
    summary.round(6).to_csv(OUT / "quantitative_summary.csv", index=False,
                            encoding="utf-8-sig")

    print("\n그림 ...")
    figure(pairs, METRICS, "box",
           {"sampling_method": ["exhaustive", "group_balanced"],
            "analysis_set": ["large", "large_and_balanced"]},
           "Large-sample · group-balanced — 조건별 유사도 분포 (boxplot)",
           "semantic_different 50,000 · prefix_2_same 50,000 · prefix_3_same 1,534(전체). "
           "outlier 점은 생략, whisker는 1.5 IQR.",
           "fig1_large_groupbalanced_box.png")
    figure(pairs, METRICS, "violin",
           {"sampling_method": ["exhaustive", "group_balanced"],
            "analysis_set": ["large", "large_and_balanced"]},
           "Large-sample · group-balanced — 조건별 유사도 분포 (violin)",
           "검은 선 = median. 분포 꼬리까지 포함한 커널 밀도.",
           "fig2_large_groupbalanced_violin.png")
    figure(pairs, METRICS, "box",
           {"sampling_method": ["exhaustive", "group_balanced"],
            "analysis_set": ["balanced", "large_and_balanced"]},
           "Balanced comparison · 세 조건 모두 1,534 pair (boxplot)",
           "표본 수를 prefix_3_same의 전체 pair 수(1,534)로 맞춘 비교.",
           "fig3_balanced_box.png")
    figure(pairs, METRICS, "violin",
           {"sampling_method": ["exhaustive", "group_balanced"],
            "analysis_set": ["balanced", "large_and_balanced"]},
           "Balanced comparison · 세 조건 모두 1,534 pair (violin)",
           "검은 선 = median.",
           "fig4_balanced_violin.png")
    method_figure(pairs, "fig5_sampling_method_control.png")
    figure(sens, ["title_jaccard"], "box",
           {"sampling_method": ["exhaustive", "group_balanced"],
            "analysis_set": ["large", "large_and_balanced"]},
           "Sensitivity — 전체 train 9,738개 (body 빈 기사 포함) · Title Jaccard only",
           f"body가 비어 분석에서 제외했던 {len(tr_all) - len(pop)}개 기사를 포함. "
           f"prefix_3_same 전체 pair {P3_ALL:,}.",
           "fig6_sensitivity_title_only.png", width=9.6)

    meta = {
        "seed": SEED,
        "population_main": {
            "description": "train split + body 비어있지 않은 기사",
            "n_articles": int(len(pop)),
            "exact_pair_counts": counts_pop,
        },
        "population_sensitivity": {
            "description": "train split 전체 (body 빈 기사 포함)",
            "n_articles": int(len(tr_all)),
            "exact_pair_counts": counts_all,
            "metric": "title_jaccard only",
            "empty_body_articles_added": int(len(tr_all) - len(pop)),
        },
        "input_files_read_only": ["57f50fe6-article_semantic_ids.parquet",
                                  "f0158e4f-articles.parquet"],
        "tfidf": {**{k: (str(v) if callable(v) else v) for k, v in TFIDF_SETTINGS.items()},
                  "vocabulary_size": int(X.shape[1]),
                  "fit_corpus": "body-valid train 9,338 전체, 1회 fit",
                  "transform": "모든 article을 동일 TF-IDF space로 transform 후 pair cosine",
                  "note": "norm='l2'이므로 cosine = dot product"},
        "sampling": {
            "large_sample": {"semantic_different": LARGE_CAP, "prefix_2_same": LARGE_CAP,
                             "prefix_3_same": int(P3_N)},
            "balanced": {c: int(P3_N) for c in CONDITIONS},
            "methods": ["group_balanced", "uniform(pair-weighted 대조군)",
                        "exhaustive(prefix_3_same)"],
            "note": "prefix_3_same은 전체 pair를 모두 사용하므로 group-balanced와 "
                    "uniform이 동일하다 (analysis_set=large_and_balanced).",
        },
        "metrics": METRICS,
        "nan_rule": "title/body 양쪽 token이 모두 비면 Jaccard를 0이 아니라 결측으로 둔다.",
        "figures": ["fig1_large_groupbalanced_box.png", "fig2_large_groupbalanced_violin.png",
                    "fig3_balanced_box.png", "fig4_balanced_violin.png",
                    "fig5_sampling_method_control.png", "fig6_sensitivity_title_only.png"],
    }
    (OUT / "step3_manifest.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    pd.set_option("display.width", 200)
    show = ["analysis_set", "condition", "sampling_method", "metric",
            "count", "mean", "median", "std", "q1", "q3", "min", "max"]
    print("\n================ quantitative summary ================")
    print(summary[show].round(4).to_string(index=False))
    print("\n================ sensitivity (title only, 9,738) ================")
    print(sens_summary[show].round(4).to_string(index=False))
    print(f"\n완료 → {OUT}")


if __name__ == "__main__":
    main()
