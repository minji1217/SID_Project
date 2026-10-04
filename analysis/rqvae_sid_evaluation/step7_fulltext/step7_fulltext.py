"""Step 7 — full_text(title + subtitle + body) 기반 추가 텍스트 정량분석.

왜 하는가
---------
RQ-VAE의 article embedding은 `src/build_train.py:158-190`의 model_text,
즉 **title + "\\n" + subtitle**로 만들어졌다. body는 임베딩 입력이 아니다.
Step 3의 Body Jaccard / Body TF-IDF cosine은 그래서 "모델이 보지 않은 텍스트로
SID를 외부 검증한 것"이었다.

이번 분석은 정성평가와 동일한 텍스트 범위(title + subtitle + body)로 같은 검증을
한 번 더 수행한다. 즉 **title+subtitle로 만든 SID가, 모델이 입력으로 쓰지 않은
body까지 포함한 기사 전체 내용과도 일관되는가**를 본다.
full-text 결과가 좋더라도 "RQ-VAE가 body를 학습했다"는 뜻이 아니다.

보존 원칙
---------
step3 / step6 / step6_prefix1 / final_report는 읽기 전용이다.
기존 Title Jaccard / Body Jaccard / Body TF-IDF cosine 수치는 교체하지 않고
나란히 둔다. pair는 재샘플링하지 않고 Step 3의 pair_id로 1:1 대응시킨다.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
S3 = ROOT / "step3"
sys.path.insert(0, str(S3))
sys.stdout.reconfigure(line_buffering=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer          # noqa: E402
from sid_pairs import SEED, load, tokenize, jaccard, TFIDF_SETTINGS  # noqa: E402
from step3_quantitative import SURFACE, INK, INK2, GRID              # noqa: E402

CONDITIONS = ["semantic_different", "prefix_2_same", "prefix_3_same"]
NAME = {"semantic_different": "Different", "prefix_2_same": "Prefix-2",
        "prefix_3_same": "Prefix-3"}
LABEL = {"semantic_different": "Different\n(c1,c2,c3 모두 다름)",
         "prefix_2_same": "Prefix-2\n(c1,c2 같고 c3 다름)",
         "prefix_3_same": "Prefix-3\n(c1,c2,c3 모두 같음)"}
# 공유 prefix 수에 대한 ordinal ramp (dataviz blue, --ordinal 검증 통과)
RAMP = {"semantic_different": "#86b6ef", "prefix_2_same": "#1c5cab",
        "prefix_3_same": "#0d366b"}

NEW_METRICS = ["fulltext_jaccard", "fulltext_tfidf_cosine"]
ALL_METRICS = ["title_jaccard", "body_jaccard", "body_tfidf_cosine"] + NEW_METRICS
SCOPE = {"title_jaccard": "title-only", "body_jaccard": "body-only",
         "body_tfidf_cosine": "body-only", "fulltext_jaccard": "title+subtitle+body",
         "fulltext_tfidf_cosine": "title+subtitle+body"}
MLAB = {"title_jaccard": "Title Jaccard", "body_jaccard": "Body Jaccard",
        "body_tfidf_cosine": "Body TF-IDF cosine",
        "fulltext_jaccard": "Full-text Jaccard",
        "fulltext_tfidf_cosine": "Full-text TF-IDF cosine"}

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


def stats_of(v) -> dict:
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    if v.size == 0:
        return {k: float("nan") for k in
                ("count", "mean", "median", "std", "q1", "q3", "min", "max")}
    return {"count": int(v.size), "mean": float(v.mean()), "median": float(np.median(v)),
            "std": float(v.std(ddof=1)) if v.size > 1 else float("nan"),
            "q1": float(np.percentile(v, 25)), "q3": float(np.percentile(v, 75)),
            "min": float(v.min()), "max": float(v.max())}


def main():
    if any(p.name != "step7_fulltext.py" for p in HERE.iterdir()):
        raise SystemExit("출력 폴더에 기존 결과가 있습니다.")

    # ---------------- 모집단 (Step 3 main과 동일)
    print("모집단 로드 (read-only) ...")
    m = load()
    tr = m[m.split == "train"].reset_index(drop=True)
    pop = tr[tr.body.fillna("").astype(str).str.strip() != ""].reset_index(drop=True)
    print(f"  body-valid train {len(pop):,}")

    # ---------------- full_text 구성
    # title + " " + subtitle + " " + body. subtitle이 없으면 빈 문자열로 둔다.
    # 덴마크어 원문만 사용하며 번역·가공하지 않는다.
    def clean(col):
        return pop[col].fillna("").astype(str).str.strip()
    title, subtitle, body = clean("title"), clean("subtitle"), clean("body")
    full_text = (title + " " + subtitle + " " + body).str.strip()
    n_sub_empty = int((subtitle == "").sum())
    print(f"  full_text 구성 완료 · subtitle이 빈 기사 {n_sub_empty:,}건 "
          f"(해당 기사는 title + body만 이어붙여진다)")

    # ---------------- 토큰화 (기존 규칙 재사용)
    print("\n토큰화 (Step 3과 동일한 tokenize 규칙) ...")
    ft_tok = [tokenize(t) for t in full_text]
    ft_set = [set(t) for t in ft_tok]
    lens = np.array([len(t) for t in ft_tok])
    print(f"  full_text token  median {int(np.median(lens))}  mean {lens.mean():.0f}  "
          f"max {lens.max():,}")

    # ---------------- TF-IDF (full_text corpus에 새로 1회 fit)
    print("\nTF-IDF fit — full_text corpus 9,338에 새로 1회 ...")
    vec = TfidfVectorizer(**TFIDF_SETTINGS)
    X = vec.fit_transform(ft_tok)
    print(f"  문서 {X.shape[0]:,}  vocab {X.shape[1]:,}  nnz {X.nnz:,}  "
          f"(body corpus vocab 59,068과 별개로 새로 fit)")

    # ---------------- Step 3 pair set 재사용 (재샘플링 없음)
    print("\nStep 3 pair set 로드 — 재샘플링 없음 ...")
    pairs = pd.read_parquet(S3 / "quantitative_pairs.parquet")
    keep = pairs[pairs.condition.isin(CONDITIONS)].copy()
    print(f"  전체 {len(pairs):,} pair 중 3조건 {len(keep):,} pair 사용 "
          f"(Prefix-1은 이번 분석에 포함하지 않음)")

    aid2row = {a: i for i, a in enumerate(pop.aid)}
    ia = np.fromiter((aid2row[a] for a in keep.article_id_a), np.int64, len(keep))
    ib = np.fromiter((aid2row[b] for b in keep.article_id_b), np.int64, len(keep))

    print("\n신규 지표 계산 ...")
    keep["fulltext_jaccard"] = [jaccard(ft_set[a], ft_set[b]) for a, b in zip(ia, ib)]
    keep["fulltext_tfidf_cosine"] = np.asarray(
        X[ia].multiply(X[ib]).sum(axis=1)).ravel()
    print(f"  {len(keep):,} pair 완료")

    out_cols = ["pair_id", "analysis_set", "condition", "sampling_method",
                "article_id_a", "article_id_b",
                "title_jaccard", "body_jaccard", "body_tfidf_cosine"] + NEW_METRICS
    out = keep[out_cols].copy()

    # ---------------- summary
    def build_summary(df, label):
        rows = []
        for cond in CONDITIONS:
            g = df[df.condition == cond]
            if not len(g):
                continue
            for met in ALL_METRICS:
                rows.append({"analysis_set": label, "condition": cond,
                             "text_scope": SCOPE[met], "metric": met,
                             "is_new": met in NEW_METRICS,
                             **stats_of(g[met].to_numpy())})
        return pd.DataFrame(rows)

    MAIN = out[(out.sampling_method.isin(["exhaustive", "group_balanced"]))
               & (out.analysis_set.isin(["large", "large_and_balanced"]))]
    BAL = out[(out.sampling_method.isin(["exhaustive", "group_balanced"]))
              & (out.analysis_set.isin(["balanced", "large_and_balanced"]))]
    summary = build_summary(MAIN, "large_group_balanced")
    bal_summary = build_summary(BAL, "balanced_1534")

    # ---------------- 순서 검증
    def order_check(df, label):
        rows = []
        for met in ALL_METRICS:
            for stat in ("mean", "median"):
                v = [df[(df.condition == c) & (df.metric == met)][stat].iloc[0]
                     for c in CONDITIONS]
                rows.append({"analysis_set": label, "metric": met,
                             "text_scope": SCOPE[met], "stat": stat,
                             "Different": v[0], "Prefix_2": v[1], "Prefix_3": v[2],
                             "strictly_increasing": bool(v[0] < v[1] < v[2]),
                             "no_inversion": bool(v[0] <= v[1] <= v[2])})
        return pd.DataFrame(rows)
    order = pd.concat([order_check(summary, "large_group_balanced"),
                       order_check(bal_summary, "balanced_1534")], ignore_index=True)

    # ---------------- 저장
    print("\n저장 ...")
    out.to_parquet(HERE / "fulltext_pair_metrics.parquet", index=False)
    summary.round(8).to_csv(HERE / "fulltext_summary.csv", index=False, encoding="utf-8-sig")
    summary.to_parquet(HERE / "fulltext_summary.parquet", index=False)
    bal_summary.round(8).to_csv(HERE / "fulltext_balanced_summary.csv", index=False,
                                encoding="utf-8-sig")
    order.round(8).to_csv(HERE / "fulltext_order_check.csv", index=False, encoding="utf-8-sig")

    manifest = {
        "purpose": ("정성평가와 동일한 텍스트 범위(title+subtitle+body)로 수행한 추가 "
                    "텍스트 정량분석. 기존 Title/Body 지표를 교체하지 않고 병렬로 둔다."),
        "why_this_is_external_validation": (
            "RQ-VAE의 article embedding은 src/build_train.py:158-190의 model_text, 즉 "
            "title + '\\n' + subtitle로 만들어졌고 body는 임베딩 입력이 아니다. "
            "따라서 이 분석은 'title+subtitle로 생성된 SID가 모델이 직접 입력으로 "
            "사용하지 않은 body까지 포함한 기사 전체 내용과도 일관되는가'를 보는 "
            "외부 semantic consistency 검증이다. full-text 결과가 좋더라도 "
            "'RQ-VAE가 body를 학습했다'는 해석은 성립하지 않는다."),
        "full_text_definition": "title + ' ' + subtitle + ' ' + body (strip 후 연결)",
        "subtitle_empty_handling": "빈 문자열로 처리 (해당 기사는 title + body만 연결)",
        "subtitle_empty_count": n_sub_empty,
        "language": "덴마크어 원문만 사용. 번역문은 정량계산에 일절 사용하지 않았다.",
        "population": {"description": "Step 3 main과 동일한 body-valid train",
                       "n_articles": int(len(pop))},
        "pair_set": {
            "source": "step3/quantitative_pairs.parquet (재샘플링 없음)",
            "conditions": CONDITIONS,
            "prefix_1_excluded": "Prefix-1은 embedding/reconstruction hierarchy 분석용으로만 "
                                 "유지하며 텍스트 정량에는 추가하지 않는다.",
            "n_pairs_used": int(len(out)),
            "one_to_one_key": "pair_id는 Step 3의 pair_id와 동일하다.",
        },
        "tokenization": "Step 3의 tokenize()를 그대로 재사용 (lowercase -> 구두점 제거 -> 공백 분할)",
        "tfidf": {**{k: (str(v) if callable(v) else v) for k, v in TFIDF_SETTINGS.items()},
                  "vocabulary_size": int(X.shape[1]),
                  "fit_corpus": "full_text corpus 9,338에 새로 1회 fit",
                  "note": "body용 기존 vectorizer를 재사용하지 않았다."},
        "metrics_new": NEW_METRICS,
        "metrics_preserved": ["title_jaccard", "body_jaccard", "body_tfidf_cosine"],
        "readonly_inputs": ["step3/", "step6/", "step6_prefix1/", "final_report/"],
        "seed": SEED,
    }
    (HERE / "fulltext_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---------------- 그림
    print("\n그림 ...")
    jac = ["title_jaccard", "body_jaccard", "fulltext_jaccard"]
    tfi = ["body_tfidf_cosine", "fulltext_tfidf_cosine"]
    fig, axes = plt.subplots(1, 2, figsize=(16.2, 5.8),
                             gridspec_kw={"width_ratios": [3, 2]})
    for ax, mets, title_txt in ((axes[0], jac, "Jaccard — 텍스트 범위별"),
                                (axes[1], tfi, "TF-IDF cosine — 텍스트 범위별")):
        ticks, labs = [], []
        for si, met in enumerate(mets):
            base = si * 4.0
            vals = [MAIN[MAIN.condition == c][met].dropna().to_numpy() for c in CONDITIONS]
            for ci, (c, v) in enumerate(zip(CONDITIONS, vals)):
                p = base + ci * 0.95
                bp = ax.boxplot([v], positions=[p], widths=0.8, patch_artist=True,
                                showfliers=False,
                                medianprops=dict(color=INK, linewidth=1.5),
                                whiskerprops=dict(color=INK2, linewidth=0.9),
                                capprops=dict(color=INK2, linewidth=0.9))
                bp["boxes"][0].set(facecolor=RAMP[c], edgecolor=SURFACE,
                                   linewidth=2.0, alpha=0.95)
                ax.annotate(f"{v.mean():.3f}", xy=(p, np.percentile(v, 75)),
                            xytext=(0, 6), textcoords="offset points", ha="center",
                            fontsize=7.4, color=INK2)
            ticks.append(base + 0.95)
            labs.append(f"{MLAB[met]}\n({SCOPE[met]})")
        ax.set_xticks(ticks); ax.set_xticklabels(labs, fontsize=8.2)
        ax.set_ylabel("유사도"); ax.set_title(title_txt, pad=8)
        ax.yaxis.grid(True); ax.set_axisbelow(True)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=RAMP[c], edgecolor=SURFACE)
               for c in CONDITIONS]
    fig.legend(handles, [LABEL[c].replace("\n", " ") for c in CONDITIONS],
               loc="lower center", ncol=3, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 0.025))
    fig.suptitle("fig14 — 텍스트 범위별 비교: title-only / body-only / title+subtitle+body "
                 "(Large · group-balanced, outlier 생략)", fontsize=12.5, y=0.985)
    fig.text(0.5, 0.004, "색이 진할수록 공유 prefix가 많다. 숫자는 각 상자의 mean. "
                         "임베딩 입력은 title+subtitle이며 body는 모델이 보지 않은 텍스트다.",
             ha="center", fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.10, 1, 0.95))
    fig.savefig(HERE / "fig14_fulltext_comparison.png", dpi=170)
    plt.close(fig)
    print("  fig14_fulltext_comparison.png")

    # ---------------- 출력
    pd.set_option("display.width", 220)
    sh = ["condition", "text_scope", "metric", "count", "mean", "median", "std", "q1", "q3"]
    print("\n============ Large · group-balanced ============")
    print(summary.sort_values(["metric", "condition"])[sh].round(5).to_string(index=False))
    print("\n============ Balanced n=1,534 ============")
    print(bal_summary.sort_values(["metric", "condition"])[sh].round(5).to_string(index=False))
    print("\n============ 순서 검증  Different < Prefix-2 < Prefix-3 ============")
    print(order[["analysis_set", "metric", "stat", "Different", "Prefix_2", "Prefix_3",
                 "strictly_increasing"]].round(5).to_string(index=False))
    nk = int(order.strictly_increasing.sum())
    print(f"\n>> 엄격한 부등호 성립 {nk}/{len(order)} · "
          f"역전 없음 {int(order.no_inversion.sum())}/{len(order)}")
    print(f"\n완료 → {HERE}")


if __name__ == "__main__":
    main()
