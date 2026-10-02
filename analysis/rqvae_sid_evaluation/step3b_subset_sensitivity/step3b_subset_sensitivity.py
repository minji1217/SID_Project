"""Step 3b — prefix_3_same 고분산의 원인 진단 (subset sensitivity).

질문: prefix_3_same의 높은 TF-IDF/Jaccard가 실제 semantic similarity인가,
      아니면 완전 중복 기사 / 반복성·템플릿 기사군 때문에 부풀려진 것인가.

Step 3의 main analysis 결과(9,338 body-valid train)는 그대로 둔다.
pair를 다시 뽑지 않고 Step 3이 만든 **동일한 pair set에 exclusion filter만** 적용한다.

subset
  main                  필터 없음 (Step 3 main analysis와 동일)
  no_exact_dup          두 article의 body 문자열이 완전히 같은 pair 제외
  no_repetitive         반복성/템플릿 기사(category_str == 'side9')가 한쪽이라도 낀 pair 제외
  no_both               위 둘 다 제외
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
STEP3 = HERE.parent / "step3"
sys.path.insert(0, str(STEP3))
sys.stdout.reconfigure(line_buffering=True)

from sid_pairs import load, tokenize                        # noqa: E402
from step3_quantitative import (                            # noqa: E402
    METRICS, METRIC_LABEL, CONDITIONS, COND_COLOR, SURFACE, INK, INK2, GRID,
    stats_of,
)

OUT = HERE
REPETITIVE_CATEGORY = "side9"
SUBSETS = ["main", "no_exact_dup", "no_repetitive", "no_both"]
SUBSET_LABEL = {
    "main": "main\n(필터 없음)",
    "no_exact_dup": "no_exact_dup\n(본문 완전중복 제외)",
    "no_repetitive": "no_repetitive\n(side9 제외)",
    "no_both": "no_both\n(둘 다 제외)",
}
COND_SHORT = {"prefix_3_same": "prefix_3", "prefix_2_same": "prefix_2",
              "semantic_different": "semantic_different"}

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


def _relation(v: dict) -> str:
    """세 조건의 대소관계를 '<' / '=' 로 표기. 동률과 역전을 구분한다."""
    def op(x, y):
        return "<" if x < y else (">" if x > y else "=")
    return (f"semantic_different {op(v['semantic_different'], v['prefix_2_same'])} "
            f"prefix_2_same {op(v['prefix_2_same'], v['prefix_3_same'])} prefix_3_same")


def main():
    existing = {p.name for p in OUT.iterdir()} - {"step3b_subset_sensitivity.py", "__pycache__"}
    if existing:
        raise SystemExit(f"출력 폴더에 기존 결과가 있습니다: {sorted(existing)}")

    print("모집단 로드 (read-only) ...")
    m = load()
    tr = m[m.split == "train"].reset_index(drop=True)
    pop = tr[tr.body.fillna("").astype(str).str.strip() != ""].reset_index(drop=True)
    art = pd.read_parquet(
        "/root/.claude/uploads/f5109944-18fd-59ef-9d42-2e3b01694dcf/f0158e4f-articles.parquet")
    art["aid"] = art.article_id.astype(str).str.strip()
    pop = pop.merge(art[["aid", "article_type"]], on="aid", how="left", validate="1:1")
    print(f"  body-valid train {len(pop):,}")

    # ---------------- 반복성 subset 정의 근거
    rep_mask = pop.category_str == REPETITIVE_CATEGORY
    rep = pop[rep_mask]
    btok = np.array([len(tokenize(b)) for b in pop.body])
    rep_def = {
        "definition": f"category_str == '{REPETITIVE_CATEGORY}'",
        "n_articles": int(rep_mask.sum()),
        "share_of_population_pct": round(100 * rep_mask.mean(), 2),
        "evidence": {
            "body_token_median_in_subset": int(np.median(btok[rep_mask.to_numpy()])),
            "body_token_median_rest": int(np.median(btok[~rep_mask.to_numpy()])),
            "article_type_counts": rep.article_type.value_counts().to_dict(),
            "article_page_nine_girl_all_inside_side9": bool(
                (pop[pop.article_type == "article_page_nine_girl"].category_str
                 == REPETITIVE_CATEGORY).all()),
            "title_duplicate_pct_in_subset": round(100 * (1 - rep.title.nunique() / len(rep)), 1),
            "title_duplicate_pct_rest": round(
                100 * (1 - pop[~rep_mask].title.nunique() / int((~rep_mask).sum())), 1),
            "body_duplicate_pct_in_subset": round(100 * (1 - rep.body.nunique() / len(rep)), 1),
            "side9_article_default_body_token_median": int(np.median(
                [len(tokenize(b)) for b in
                 pop[(pop.category_str == REPETITIVE_CATEGORY)
                     & (pop.article_type == "article_default")].body])),
        },
        "conclusion": ("핀업 갤러리 섹션. 데이터셋 자체의 article_type "
                       "'article_page_nine_girl'(192개)이 전부 이 섹션 안에 있고, "
                       "제목이 '<이름>, <나이> år og fra <도시>' 템플릿이며, "
                       "본문이 전체 중앙값의 1/3 수준(side9 안의 article_default는 중앙값 4 token). "
                       "반복성·템플릿 article subset으로 확인되어 sensitivity subset으로 분리한다."),
    }
    print(f"  반복성 subset '{REPETITIVE_CATEGORY}' = {rep_def['n_articles']}개 "
          f"({rep_def['share_of_population_pct']}%)")

    # ---------------- pair에 플래그 붙이기 (동일 pair set, 필터만 적용)
    print("\nStep 3 pair set 로드 (재샘플링 없음) ...")
    pairs = pd.read_parquet(STEP3 / "quantitative_pairs.parquet")
    print(f"  {len(pairs):,} pair")

    body_of = dict(zip(pop.aid, pop.body))
    is_rep = dict(zip(pop.aid, rep_mask.to_numpy()))
    a, b = pairs.article_id_a.to_numpy(), pairs.article_id_b.to_numpy()
    pairs["exact_body_dup"] = [body_of[x] == body_of[y] for x, y in zip(a, b)]
    pairs["touches_repetitive"] = [bool(is_rep[x] or is_rep[y]) for x, y in zip(a, b)]

    # ---------------- exact duplicate count 보고
    print("\n=== exact-body-duplicate pair 수 ===")
    dup_rows = []
    for (aset, cond, meth), g in pairs.groupby(["analysis_set", "condition", "sampling_method"],
                                               sort=False):
        dup_rows.append({"analysis_set": aset, "condition": cond, "sampling_method": meth,
                         "n_pairs": len(g),
                         "exact_body_dup": int(g.exact_body_dup.sum()),
                         "exact_body_dup_pct": round(100 * g.exact_body_dup.mean(), 3),
                         "touches_repetitive": int(g.touches_repetitive.sum()),
                         "touches_repetitive_pct": round(100 * g.touches_repetitive.mean(), 2)})
    dup = pd.DataFrame(dup_rows)
    print(dup.to_string(index=False))
    p3 = dup[dup.condition == "prefix_3_same"].iloc[0]
    print(f"\n>> prefix_3_same 전체 {int(p3.n_pairs):,} pair 중 "
          f"exact body duplicate {int(p3.exact_body_dup)} pair ({p3.exact_body_dup_pct}%), "
          f"side9가 낀 pair {int(p3.touches_repetitive)} pair ({p3.touches_repetitive_pct}%)")

    # ---------------- subset별 summary
    filters = {
        "main": lambda d: d,
        "no_exact_dup": lambda d: d[~d.exact_body_dup],
        "no_repetitive": lambda d: d[~d.touches_repetitive],
        "no_both": lambda d: d[~d.exact_body_dup & ~d.touches_repetitive],
    }
    rows = []
    for sub, fn in filters.items():
        d = fn(pairs)
        for (aset, cond, meth), g in d.groupby(["analysis_set", "condition", "sampling_method"],
                                               sort=False):
            for met in METRICS:
                rows.append({"subset": sub, "analysis_set": aset, "condition": cond,
                             "sampling_method": meth, "metric": met,
                             **stats_of(g[met].to_numpy(dtype=float), len(g))})
    summary = pd.DataFrame(rows)

    # ---------------- 순서 검증
    head = summary[(summary.sampling_method.isin(["exhaustive", "group_balanced"]))
                   & (summary.analysis_set.isin(["large", "large_and_balanced"]))]
    order_rows = []
    for sub in SUBSETS:
        for met in METRICS:
            s = head[(head.subset == sub) & (head.metric == met)].set_index("condition")
            for stat in ("mean", "median"):
                v = {c: s.loc[c, stat] for c in CONDITIONS}
                order_rows.append({
                    "subset": sub, "metric": met, "stat": stat,
                    "semantic_different": v["semantic_different"],
                    "prefix_2_same": v["prefix_2_same"], "prefix_3_same": v["prefix_3_same"],
                    "order_holds": bool(v["semantic_different"] < v["prefix_2_same"]
                                        < v["prefix_3_same"]),
                    "no_inversion": bool(v["semantic_different"] <= v["prefix_2_same"]
                                         <= v["prefix_3_same"]),
                    "relation": _relation(v),
                })
    order = pd.DataFrame(order_rows)

    print("\n=== 순서 검증  semantic_different < prefix_2_same < prefix_3_same ===")
    print(order.round(4).to_string(index=False))
    print(f"\n>> 엄격한 부등호 성립 {int(order.order_holds.sum())}/{len(order)} · "
          f"역전 없음(동률 허용) {int(order.no_inversion.sum())}/{len(order)}")
    if (~order.order_holds).any():
        print("   엄격 부등호가 깨진 조합 (동률 여부 포함):")
        print(order[~order.order_holds][["subset","metric","stat","semantic_different",
                                         "prefix_2_same","prefix_3_same","relation"]]
              .round(4).to_string(index=False))

    # ---------------- 저장
    for df, stem in [(dup, "subset_pair_counts"), (summary, "subset_summary"),
                     (order, "subset_order_check")]:
        df.to_parquet(OUT / f"{stem}.parquet", index=False)
        df.round(6).to_csv(OUT / f"{stem}.csv", index=False, encoding="utf-8-sig")
        print(f"  저장 {stem}  rows={len(df):,}")
    pairs[["pair_id", "analysis_set", "condition", "sampling_method",
           "article_id_a", "article_id_b", "exact_body_dup",
           "touches_repetitive"]].to_parquet(OUT / "pair_exclusion_flags.parquet", index=False)
    print("  저장 pair_exclusion_flags  (동일 pair set에 붙인 플래그만)")

    # ---------------- fig7 : subset을 가로지르는 mean/median 변화
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 5.2))
    x = np.arange(len(SUBSETS))
    for ax, met in zip(axes, METRICS):
        for cond in CONDITIONS:
            mu = [head[(head.subset == s) & (head.metric == met)
                       & (head.condition == cond)]["mean"].iloc[0] for s in SUBSETS]
            md = [head[(head.subset == s) & (head.metric == met)
                       & (head.condition == cond)]["median"].iloc[0] for s in SUBSETS]
            ax.plot(x, mu, "-o", color=COND_COLOR[cond], linewidth=2.0, markersize=8,
                    markeredgecolor=SURFACE, markeredgewidth=2.0,
                    label=f"{COND_SHORT[cond]} (mean)", zorder=3)
            ax.plot(x, md, "--s", color=COND_COLOR[cond], linewidth=1.4, markersize=6,
                    markeredgecolor=SURFACE, markeredgewidth=1.6, alpha=0.75,
                    label=f"{COND_SHORT[cond]} (median)", zorder=2)
            ax.annotate(f"{mu[0]:.3f}", xy=(0, mu[0]), xytext=(-6, 7),
                        textcoords="offset points", fontsize=7.8, color=INK2, ha="right")
            ax.annotate(f"{mu[-1]:.3f}", xy=(len(SUBSETS) - 1, mu[-1]), xytext=(6, 7),
                        textcoords="offset points", fontsize=7.8, color=INK2, ha="left")
        ax.set_xticks(x)
        ax.set_xticklabels([SUBSET_LABEL[s] for s in SUBSETS], fontsize=7.6)
        ax.set_xlim(-0.55, len(SUBSETS) - 0.45)
        ax.set_ylim(bottom=0)
        ax.set_title(METRIC_LABEL[met], pad=8)
        ax.set_ylabel(METRIC_LABEL[met])
        ax.yaxis.grid(True); ax.set_axisbelow(True)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, frameon=False, fontsize=8.2,
               bbox_to_anchor=(0.5, 0.035))
    fig.suptitle("fig7 — prefix_3_same 고분산 진단: exclusion subset별 유사도 변화 "
                 "(Large-sample · group-balanced)", fontsize=12.5, y=0.985)
    fig.text(0.5, 0.005, "실선·원 = mean, 점선·사각 = median. "
                         "pair를 다시 뽑지 않고 Step 3의 동일 pair set에 exclusion filter만 적용했다.",
             ha="center", fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.11, 1, 0.95))
    fig.savefig(OUT / "fig7_subset_sensitivity.png", dpi=170)
    plt.close(fig)
    print("  그림 fig7_subset_sensitivity.png")

    # ---------------- fig8 : subset별 분포 (boxplot)
    fig, axes = plt.subplots(1, 3, figsize=(17.4, 5.4))
    for ax, met in zip(axes, METRICS):
        pos, ticks, tlab = [], [], []
        for si, sub in enumerate(SUBSETS):
            d = filters[sub](pairs)
            d = d[d.sampling_method.isin(["exhaustive", "group_balanced"])
                  & d.analysis_set.isin(["large", "large_and_balanced"])]
            base = si * 4.0
            for ci, cond in enumerate(CONDITIONS):
                v = d[d.condition == cond][met].dropna().to_numpy()
                p = base + ci * 0.95
                bp = ax.boxplot([v], positions=[p], widths=0.78, patch_artist=True,
                                showfliers=False,
                                medianprops=dict(color=INK, linewidth=1.5),
                                whiskerprops=dict(color=INK2, linewidth=0.9),
                                capprops=dict(color=INK2, linewidth=0.9))
                bp["boxes"][0].set(facecolor=COND_COLOR[cond], edgecolor=SURFACE,
                                   linewidth=2.0, alpha=0.92)
                pos.append(p)
            ticks.append(base + 0.95); tlab.append(SUBSET_LABEL[sub])
        ax.set_xticks(ticks); ax.set_xticklabels(tlab, fontsize=7.8)
        ax.set_title(METRIC_LABEL[met], pad=8)
        ax.set_ylabel(METRIC_LABEL[met])
        ax.yaxis.grid(True); ax.set_axisbelow(True)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=COND_COLOR[c], edgecolor=SURFACE)
               for c in CONDITIONS]
    fig.legend(handles, [COND_SHORT[c] for c in CONDITIONS], loc="lower center",
               ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.035))
    fig.suptitle("fig8 — exclusion subset별 분포 (Large-sample · group-balanced, outlier 생략)",
                 fontsize=12.5, y=0.985)
    fig.text(0.5, 0.005, "각 subset 안에서 왼쪽부터 prefix_3 / prefix_2 / semantic_different.",
             ha="center", fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.11, 1, 0.95))
    fig.savefig(OUT / "fig8_subset_distributions.png", dpi=170)
    plt.close(fig)
    print("  그림 fig8_subset_distributions.png")

    meta = {
        "purpose": ("prefix_3_same의 높은 TF-IDF/Jaccard가 실제 semantic similarity 때문인지, "
                    "완전 중복 기사나 반복성/템플릿 기사군 때문에 부풀려진 것인지 확인"),
        "main_analysis_preserved": ("Step 3의 9,338 body-valid train 결과를 main analysis로 유지. "
                                    "pair를 재샘플링하지 않고 동일 pair set에 filter만 적용."),
        "exact_duplicate_definition": "두 article의 body 문자열이 완전히 동일(==)한 pair",
        "repetitive_subset": rep_def,
        "subsets": SUBSETS,
        "order_check": {"rule": "semantic_different < prefix_2_same < prefix_3_same",
                        "combinations": int(len(order)),
                        "strict_holds": int(order.order_holds.sum()),
                        "no_inversion_ties_allowed": int(order.no_inversion.sum()),
                        "note": ("엄격 부등호가 깨지는 경우는 모두 title_jaccard의 median에서 "
                                 "semantic_different와 prefix_2_same이 둘 다 정확히 0.000인 "
                                 "동률이며, 역전은 한 건도 없다.")},
        "figures": ["fig7_subset_sensitivity.png", "fig8_subset_distributions.png"],
    }
    (OUT / "step3b_manifest.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    pd.set_option("display.width", 220)
    show = ["subset", "condition", "metric", "count", "mean", "median", "std", "q1", "q3"]
    print("\n================ subset summary (Large-sample · group-balanced) ================")
    print(head.sort_values(["metric", "subset"])[show].round(4).to_string(index=False))
    print(f"\n완료 → {OUT}")


if __name__ == "__main__":
    main()
