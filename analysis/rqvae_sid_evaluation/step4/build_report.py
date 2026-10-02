"""Step 4 — sid_evaluation_report.html 생성.

교수님 피드백 #2 (정성평가 group-first 설계 + 텍스트 정량평가)와
#3 (reconstruction vector cosine, Step 5/6 예정)을 명확히 구분한다.
덴마크어 원문은 번역하지 않고 그대로 싣는다.
"""
from __future__ import annotations
import base64, html, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
S2, S3, S3B = ROOT / "step2", ROOT / "step3", ROOT / "step3b_subset_sensitivity"
sys.stdout.reconfigure(line_buffering=True)

E = html.escape
FIGS = {
    "fig1": (S3 / "fig1_large_groupbalanced_box.png", "Large-sample · group-balanced (boxplot)"),
    "fig2": (S3 / "fig2_large_groupbalanced_violin.png", "Large-sample · group-balanced (violin)"),
    "fig3": (S3 / "fig3_balanced_box.png", "Balanced 1,534 × 3 (boxplot)"),
    "fig4": (S3 / "fig4_balanced_violin.png", "Balanced 1,534 × 3 (violin)"),
    "fig5": (S3 / "fig5_sampling_method_control.png", "group-balanced vs pair-weighted uniform"),
    "fig6": (S3 / "fig6_sensitivity_title_only.png", "전체 train 9,738 · Title Jaccard only"),
    "fig7": (S3B / "fig7_subset_sensitivity.png", "exclusion subset별 mean/median 변화"),
    "fig8": (S3B / "fig8_subset_distributions.png", "exclusion subset별 분포"),
}


def img(tag: str) -> str:
    p, cap = FIGS[tag]
    b64 = base64.b64encode(p.read_bytes()).decode()
    return (f'<figure><img src="data:image/png;base64,{b64}" alt="{E(cap)}">'
            f'<figcaption><b>{tag}</b> — {E(cap)} '
            f'<span class="path">{E(p.relative_to(ROOT).as_posix())}</span></figcaption></figure>')


def table(df: pd.DataFrame, cols: list[str], heads: list[str],
          fmt: dict | None = None, cls: str = "") -> str:
    fmt = fmt or {}
    th = "".join("<th>%s</th>" % E(h) for h in heads)
    rows = []
    for _, r in df.iterrows():
        tds = []
        for c in cols:
            v = r[c]
            f = fmt.get(c)
            if f:
                s = f(v)
            elif isinstance(v, float):
                s = "–" if pd.isna(v) else ("%.4f" % v)
            elif isinstance(v, (int, np.integer)):
                s = "{:,}".format(int(v))
            else:
                s = str(v)
            num = "num" if isinstance(v, (int, float, np.integer, np.floating)) else ""
            tds.append(f'<td class="{num}">{E(s)}</td>')
        rows.append("<tr>%s</tr>" % "".join(tds))
    return (f'<div class="tw"><table class="{cls}"><thead><tr>{th}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>')


# ------------------------------------------------------------------ 데이터
def build_payload():
    gm = pd.read_parquet(HERE / "qualitative_group_metrics.parquet").set_index(
        "qualitative_group_id")
    files = {
        "prefix_3": ("qualitative_prefix3_groups", ["c1", "c2", "c3"], "size_band"),
        "prefix_2": ("qualitative_prefix2_groups", ["c1", "c2"], "count_band"),
        "semantic_different": ("qualitative_semantic_different", ["c1", "c2", "c3"], None),
        "semantic_different_supp": (
            "qualitative_semantic_different_allcodes_supplementary", ["c1", "c2", "c3"], None),
    }
    out = {}
    for kind, (stem, keys, band) in files.items():
        df = pd.read_parquet(S2 / f"{stem}.parquet")
        groups = []
        for gid, g in df.groupby("qualitative_group_id", sort=True):
            first = g.iloc[0]
            arts = []
            for _, a in g.iterrows():
                arts.append({
                    "id": str(a.article_id),
                    "c": [int(a.get("c1", first.c1)), int(a.get("c2", first.c2)),
                          int(a["c3"]) if "c3" in a and not pd.isna(a["c3"]) else None,
                          int(a.c4)],
                    "cat": str(a.category_str),
                    "t": str(a.title), "s": "" if pd.isna(a.subtitle) else str(a.subtitle),
                    "b": str(a.body), "bt": int(a.body_tokens),
                    "side": str(a.side) if "side" in a else "",
                })
            mm = gm.loc[gid] if gid in gm.index else None
            groups.append({
                "id": gid, "kind": kind,
                "band": str(first[band]) if band else "",
                "key": "-".join(str(int(first[k])) for k in keys if k in first),
                "n": len(g),
                "c3n": int(first.c3_nunique) if "c3_nunique" in first else None,
                "m": None if mm is None else {
                    "tj": float(mm.title_jaccard_mean), "bj": float(mm.body_jaccard_mean),
                    "tf": float(mm.body_tfidf_cosine_mean),
                    "tfmin": float(mm.body_tfidf_cosine_min),
                    "tfmax": float(mm.body_tfidf_cosine_max)},
                "arts": arts,
            })
        out[kind] = groups
    return out


def main():
    existing = {p.name for p in HERE.iterdir()} - {
        "build_report.py", "build_group_metrics.py", "__pycache__",
        "qualitative_group_metrics.parquet", "qualitative_group_metrics.csv",
        "qualitative_group_pair_metrics.parquet"}
    if existing:
        raise SystemExit(f"기존 결과가 있습니다: {sorted(existing)}")

    print("데이터 로드 ...")
    s3 = pd.read_parquet(S3 / "quantitative_summary.parquet")
    sens = pd.read_parquet(S3 / "sensitivity_title_only_summary.parquet")
    sub = pd.read_parquet(S3B / "subset_summary.parquet")
    cnt = pd.read_parquet(S3B / "subset_pair_counts.parquet")
    order = pd.read_parquet(S3B / "subset_order_check.parquet")
    p3b = pd.read_parquet(S2 / "qualitative_prefix3_band_summary.parquet")
    p2b = pd.read_parquet(S2 / "qualitative_prefix2_band_summary.parquet")
    s2man = json.loads((S2 / "step2_manifest.json").read_text())
    s3man = json.loads((S3 / "step3_manifest.json").read_text())
    s3bman = json.loads((S3B / "step3b_manifest.json").read_text())
    payload = build_payload()

    MAIN = s3[(s3.sampling_method.isin(["exhaustive", "group_balanced"]))
              & (s3.analysis_set.isin(["large", "large_and_balanced"]))]
    BAL = s3[(s3.sampling_method.isin(["exhaustive", "group_balanced"]))
             & (s3.analysis_set.isin(["balanced", "large_and_balanced"]))]
    HEAD = sub[(sub.sampling_method.isin(["exhaustive", "group_balanced"]))
               & (sub.analysis_set.isin(["large", "large_and_balanced"]))]
    CN = {"prefix_3_same": "prefix_3_same", "prefix_2_same": "prefix_2_same",
          "semantic_different": "semantic_different"}
    ORDER_C = ["prefix_3_same", "prefix_2_same", "semantic_different"]
    MET = ["title_jaccard", "body_jaccard", "body_tfidf_cosine"]
    MET_L = {"title_jaccard": "Title Jaccard", "body_jaccard": "Body Jaccard",
             "body_tfidf_cosine": "Body TF-IDF cosine"}

    def wide(df, extra=()):
        rows = []
        for c in ORDER_C:
            r = {"condition": CN[c]}
            n = df[(df.condition == c) & (df.metric == "title_jaccard")]
            r["n"] = int(n["count"].iloc[0]) if len(n) else 0
            for met in MET:
                s = df[(df.condition == c) & (df.metric == met)]
                if not len(s):
                    continue
                s = s.iloc[0]
                r[f"{met}_mean"] = s["mean"]; r[f"{met}_median"] = s["median"]
                r[f"{met}_std"] = s["std"]; r[f"{met}_q1"] = s["q1"]; r[f"{met}_q3"] = s["q3"]
            rows.append(r)
        return pd.DataFrame(rows)

    main_w, bal_w = wide(MAIN), wide(BAL)
    cols = ["condition", "n"] + [f"{m}_{k}" for m in MET for k in ("mean", "median", "std")]
    heads = ["condition", "n"] + [f"{MET_L[m]} {k}" for m in MET
                                  for k in ("mean", "median", "std")]

    # subset 비교 wide
    sub_rows = []
    for s in ["main", "no_exact_dup", "no_repetitive", "no_both"]:
        for c in ORDER_C:
            r = {"subset": s, "condition": CN[c]}
            g = HEAD[(HEAD.subset == s) & (HEAD.condition == c)]
            r["n"] = int(g[g.metric == "title_jaccard"]["count"].iloc[0])
            for met in MET:
                x = g[g.metric == met].iloc[0]
                r[f"{met}_mean"] = x["mean"]; r[f"{met}_median"] = x["median"]
                r[f"{met}_std"] = x["std"]
            sub_rows.append(r)
    sub_w = pd.DataFrame(sub_rows)

    n_strict = int(order.order_holds.sum()); n_noinv = int(order.no_inversion.sum())
    rep = s3bman["repetitive_subset"]
    p3cnt = cnt[cnt.condition == "prefix_3_same"].iloc[0]

    def f3(v): return "–" if pd.isna(v) else "%.3f" % v

    print("HTML 조립 ...")
    H = []
    A = H.append
    # ---------------------------------------------------------------- head
    A(f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>EB-NeRD RQ-VAE SID 평가 보고서</title>
<style>
:root{{--s1:#fcfcfb;--s2:#f3f2ef;--s3:#eae9e4;--ink:#0b0b0b;--ink2:#52514e;--ink3:#86847d;
--line:#dedcd6;--c1:#2a78d6;--c2:#eb6834;--c3:#1baf7a;--warn:#b4690e;--code:#f0efec;}}
@media (prefers-color-scheme:dark){{:root:not([data-theme=light]){{
--s1:#1a1a19;--s2:#232321;--s3:#2d2d2a;--ink:#fff;--ink2:#c3c2b7;--ink3:#8d8c83;
--line:#3a3a36;--c1:#3987e5;--c2:#d95926;--c3:#199e70;--warn:#e0a14a;--code:#232321;}}}}
:root[data-theme=dark]{{--s1:#1a1a19;--s2:#232321;--s3:#2d2d2a;--ink:#fff;--ink2:#c3c2b7;
--ink3:#8d8c83;--line:#3a3a36;--c1:#3987e5;--c2:#d95926;--c3:#199e70;--warn:#e0a14a;--code:#232321;}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--s1);color:var(--ink);
font:15px/1.72 -apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans KR",sans-serif;}}
.wrap{{max-width:1120px;margin:0 auto;padding:0 16px 96px}}
header{{padding:52px 0 26px;border-bottom:1px solid var(--line);margin-bottom:30px}}
h1{{font-size:28px;line-height:1.3;margin:0 0 10px;letter-spacing:-.01em}}
.sub{{color:var(--ink2);font-size:14.5px;margin:0}}
.meta{{display:flex;flex-wrap:wrap;gap:8px;margin-top:18px}}
.chip{{background:var(--s2);border:1px solid var(--line);border-radius:999px;
padding:4px 12px;font-size:12.5px;color:var(--ink2)}}
h2{{font-size:21px;margin:52px 0 6px;letter-spacing:-.01em;scroll-margin-top:16px}}
h2 .no{{color:var(--ink3);font-weight:600;margin-right:9px}}
h3{{font-size:16px;margin:30px 0 8px}}
h2+.lead{{color:var(--ink2);margin:0 0 18px;font-size:14.5px}}
p{{margin:12px 0}}
.fb{{display:inline-block;font-size:11.5px;font-weight:700;letter-spacing:.04em;
padding:3px 9px;border-radius:5px;vertical-align:middle;margin-left:10px}}
.fb2{{background:color-mix(in srgb,var(--c1) 16%,transparent);color:var(--c1);
border:1px solid color-mix(in srgb,var(--c1) 40%,transparent)}}
.fb3{{background:color-mix(in srgb,var(--c2) 16%,transparent);color:var(--c2);
border:1px solid color-mix(in srgb,var(--c2) 40%,transparent)}}
nav{{background:var(--s2);border:1px solid var(--line);border-radius:12px;padding:16px 20px;
margin-bottom:12px}}
nav ol{{margin:0;padding-left:20px;columns:2;column-gap:36px}}
nav li{{margin:3px 0;break-inside:avoid}}
nav a{{color:var(--ink);text-decoration:none;border-bottom:1px solid transparent}}
nav a:hover{{border-bottom-color:var(--ink3)}}
.tw{{overflow-x:auto;margin:16px 0;border:1px solid var(--line);border-radius:10px}}
table{{border-collapse:collapse;width:100%;font-size:13px}}
th,td{{padding:8px 11px;text-align:left;border-bottom:1px solid var(--line);white-space:nowrap}}
th{{background:var(--s2);font-weight:650;color:var(--ink2);font-size:12.2px;
position:sticky;top:0}}
td.num{{text-align:right;font-variant-numeric:tabular-nums}}
tbody tr:last-child td{{border-bottom:none}}
tbody tr:hover td{{background:var(--s2)}}
figure{{margin:22px 0}}
figure img{{width:100%;height:auto;border:1px solid var(--line);border-radius:10px;
background:#fcfcfb}}
figcaption{{color:var(--ink2);font-size:12.5px;margin-top:8px}}
.path{{color:var(--ink3);font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11.5px}}
code{{background:var(--code);padding:1.5px 5px;border-radius:4px;
font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12.8px}}
.box{{border:1px solid var(--line);border-left:3px solid var(--c1);background:var(--s2);
border-radius:0 10px 10px 0;padding:14px 18px;margin:20px 0}}
.box.warn{{border-left-color:var(--warn)}}
.box.key{{border-left-color:var(--c3)}}
.box h4{{margin:0 0 7px;font-size:14px}}
.box p:last-child,.box ul:last-child{{margin-bottom:0}}
ul{{margin:11px 0;padding-left:21px}} li{{margin:5px 0}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;
margin:20px 0}}
.kpi{{background:var(--s2);border:1px solid var(--line);border-radius:11px;padding:14px 16px}}
.kpi .v{{font-size:25px;font-weight:680;letter-spacing:-.02em;font-variant-numeric:tabular-nums}}
.kpi .l{{font-size:12.3px;color:var(--ink2);margin-top:3px;line-height:1.45}}
.ctl{{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:18px 0 14px}}
.ctl button,.ctl select{{background:var(--s2);color:var(--ink);border:1px solid var(--line);
border-radius:8px;padding:6px 13px;font-size:13px;cursor:pointer;font-family:inherit}}
.ctl button.on{{background:var(--c1);border-color:var(--c1);color:#fff}}
.glist{{display:grid;gap:10px}}
.g{{border:1px solid var(--line);border-radius:11px;background:var(--s2);overflow:hidden}}
.gh{{display:flex;flex-wrap:wrap;gap:12px;align-items:center;padding:12px 16px;cursor:pointer;
user-select:none}}
.gh:hover{{background:var(--s3)}}
.gid{{font-weight:700;font-size:13.5px;font-family:ui-monospace,Menlo,monospace}}
.sid{{font-family:ui-monospace,Menlo,monospace;font-size:12.5px;color:var(--ink2);
background:var(--s3);padding:2px 8px;border-radius:5px}}
.gm{{margin-left:auto;display:flex;gap:14px;font-size:12.3px;color:var(--ink2);
font-variant-numeric:tabular-nums}}
.gm b{{color:var(--ink);font-weight:650}}
.gb{{display:none;padding:2px 16px 16px;border-top:1px solid var(--line)}}
.g.open .gb{{display:block}}
.art{{border-top:1px solid var(--line);padding:13px 0}}
.art:first-child{{border-top:none}}
.ah{{display:flex;flex-wrap:wrap;gap:9px;align-items:baseline;margin-bottom:5px}}
.aid{{font-family:ui-monospace,Menlo,monospace;font-size:11.8px;color:var(--ink3)}}
.tag{{font-size:11.3px;background:var(--s3);border:1px solid var(--line);border-radius:5px;
padding:1px 7px;color:var(--ink2)}}
.at{{font-weight:650;font-size:14.6px;line-height:1.45}}
.as{{color:var(--ink2);font-size:13.4px;margin-top:3px}}
details{{margin-top:7px}}
summary{{cursor:pointer;font-size:12.5px;color:var(--ink2);outline:none}}
.body{{white-space:pre-wrap;font-size:13.2px;line-height:1.68;color:var(--ink2);
margin-top:8px;padding:11px 14px;background:var(--s1);border:1px solid var(--line);
border-radius:8px;max-height:380px;overflow-y:auto}}
.dk{{font-size:11.6px;color:var(--ink3);margin-top:5px}}
footer{{margin-top:70px;padding-top:22px;border-top:1px solid var(--line);
color:var(--ink3);font-size:12.6px}}
@media(max-width:720px){{nav ol{{columns:1}} h1{{font-size:23px}} .gm{{margin-left:0;width:100%}}}}
</style></head><body><div class="wrap">""")

    # ---------------------------------------------------------------- header
    A(f"""<header>
<h1>EB-NeRD RQ-VAE Semantic ID 평가 보고서</h1>
<p class="sub">교수님 피드백 #2 — 정성평가 group-first 설계와 텍스트 기반 정량평가</p>
<div class="meta">
<span class="chip">모집단 body-valid train <b>9,338</b></span>
<span class="chip">정성 group 30 / 15 / 30</span>
<span class="chip">정량 pair <b>207,670</b></span>
<span class="chip">seed 42</span>
<span class="chip">입력 parquet read-only</span>
<span class="chip">덴마크어 원문 무번역</span>
</div></header>

<nav><ol>
<li><a href="#s1">정성평가 설계 — group-first sampling</a></li>
<li><a href="#s2">텍스트 정량평가 — 세 지표</a></li>
<li><a href="#s3">Balanced comparison</a></li>
<li><a href="#s4">group-balanced vs uniform</a></li>
<li><a href="#s5">body-empty Title-only sensitivity</a></li>
<li><a href="#s6">subset sensitivity — 중복·템플릿 제거</a></li>
<li><a href="#s7">중요한 caveat</a></li>
<li><a href="#s8">정성·정량의 역할 연결 + 실제 group</a></li>
<li><a href="#s9">Step 5/6 예고 — reconstruction cosine</a></li>
</ol></nav>

<div class="box key"><h4>이 보고서의 결론 한 줄</h4>
<p><b>SID prefix를 많이 공유할수록 텍스트 유사도가 높아지는 일관된 경향을 확인했다.</b>
세 지표 모두에서 <code>semantic_different &lt; prefix_2_same &lt; prefix_3_same</code> 순서가
성립하고, 표본 수를 맞추거나 중복·템플릿 기사를 제거해도 역전은 한 건도 없다.
다만 <b>prefix_3 collision group 내부의 분산은 여전히 크다</b> — 같은 SID를 받은 기사가
모두 비슷하다는 뜻은 아니며, 이 이질성은 §8의 정성평가에서 실제 기사로 확인한다.
본 보고서는 “RQ-VAE가 잘 학습되었다”는 결론을 주장하지 않는다.</p></div>""")

    # ---------------------------------------------------------------- §1
    p3q = s2man["prefix_3"]; p2q = s2man["prefix_2"]
    A(f"""<h2 id="s1"><span class="no">1</span>정성평가 설계 — group-first sampling<span class="fb fb2">교수님 피드백 #2</span></h2>
<p class="lead">무작위 pair가 아니라 <b>group을 먼저 뽑고 그 group의 기사를 전부 포함</b>하는 방식으로 표본을 구성했다.</p>

<h3>왜 pair가 아니라 group-first인가</h3>
<ul>
<li><b>collision은 group 현상이다.</b> 같은 <code>(c1,c2,c3)</code>를 받은 기사 6개를 pair로 흩어 뽑으면
“이 6개가 하나로 묶인 게 타당한가”라는 질문에 답할 수 없다. 한 group을 통째로 봐야
무엇이 묶였고 무엇이 어긋났는지 판단할 수 있다.</li>
<li><b>pair 단위로 뽑으면 큰 group이 자동으로 과대 대표된다.</b> 크기 <i>k</i>인 group은
pair를 <i>k(k−1)/2</i>개 만들어내므로, pair를 균등 추출하면 큰 group의 pair가 제곱으로 많이 뽑힌다
(이 효과는 §4에서 정량적으로 확인했다).</li>
<li><b>실패 사례는 큰 group에서 나타난다.</b> size 2 group은 대개 비슷하지만, size 7~8 group에서
의미가 무너지는지를 봐야 한다. 그래서 size 구간별 할당을 두었다.</li>
<li>선택된 group 안의 기사는 <b>일부 pair만 고르지 않고 전부 포함</b>했다.</li>
</ul>

<h3>prefix_3 collision group — 30 group / 기사 {p3q["articles_selected"]}개</h3>
{table(p3b, ["band","groups_selected","articles","available_groups_in_population",
             "distinct_c1","distinct_c1_c2"],
       ["size 구간","선택 group","기사","모집단 내 group","distinct c1","distinct (c1,c2)"])}
<div class="box warn"><h4>표본 구성에 관한 명시 — 이 표본은 분포 추정용이 아니다</h4>
<ul>
<li>실제 분포에서 size 2~3 group이 eligible group의
<b>{p3q["real_population_distribution"]["share_size_2_3_of_eligible"]*100:.1f}%</b>
({p3q["real_population_distribution"]["eligible_groups_size_2_3"]}/772)를 차지한다.</li>
<li>그럼에도 <b>size 4~8 group을 의도적으로 oversampling</b>했다. 큰 collision에서 의미적
일관성이 무너지는 사례를 확인하기 위한 <b>진단 목적</b>의 표본이기 때문이다.</li>
<li>따라서 이 30개 group의 관찰 결과를 전체 collision 분포의 비율로 환산해서는 안 된다.</li>
<li><b>EB-NeRD에는 size 9 이상 collision group이 없다</b> (최대 8). MIND 참고자료의
18/8/4 구성(일반 2~3 / 중간 4~8 / 대형 9+)은 이 데이터에서 재현할 수 없어 15/10/5로 조정했다.</li>
</ul></div>

<h3>prefix_2 group — 15 group / 기사 {p2q["articles_selected"]}개</h3>
<p>조건: <code>(c1,c2)</code>가 같고 기사 수 4~10, <code>c3</code> 종류가 2개 이상.
같은 상위 2단계 아래에서 c3가 갈라질 때 실제로 다른 내용인지를 본다.</p>
{table(p2b, ["band","groups_selected","articles","available_groups_in_population",
             "distinct_c1","distinct_c1_c2"],
       ["기사 수 구간","선택 group","기사","모집단 내 group","distinct c1","distinct (c1,c2)"])}

<h3>semantic_different — 30 group-pair</h3>
<p><code>c1, c2, c3</code>가 모두 다른 두 <code>(c1,c2,c3)</code> group에서 대표 기사를 하나씩 뽑아
30쌍을 구성했다 (기사 60개, distinct c1 21). 대조군으로서 “SID가 완전히 다르면 내용도 다른가”를 본다.</p>
<p><code>c4</code>까지 모두 다른 10쌍은 <b>별도 보조 파일</b>로 분리했고 본 분석에서 제외했다.
<code>c4</code>는 의미 단계가 아니라 <b>collision 구분용 suffix</b>이므로 pair 조건으로 쓰지 않는다.</p>
<p class="dk">샘플링 다양성: 각 size 구간 안에서 동일 <code>c1</code>·<code>(c1,c2)</code>에 몰리지 않도록
greedy diversity 샘플링(seed 42)을 적용했다. prefix_3 30 group의 distinct <code>(c1,c2)</code>는 29/30이며,
중복 1건은 같은 <code>(c1,c2)</code> 아래 서로 다른 <code>c3</code> collision이라 비교 사례로 남겼다.</p>""")

    # ---------------------------------------------------------------- §2
    A(f"""<h2 id="s2"><span class="no">2</span>텍스트 정량평가 — 세 지표<span class="fb fb2">교수님 피드백 #2</span></h2>
<p class="lead">정성 표본과 <b>무관하게</b> 9,338 모집단에서 별도로 뽑은 pair에 대해 측정했다.</p>
<ul>
<li><b>Title token Jaccard</b> — 제목 토큰 집합의 Jaccard</li>
<li><b>Body token Jaccard</b> — 본문 토큰 집합의 Jaccard</li>
<li><b>Body TF-IDF cosine</b> — 9,338 body corpus에 <b>단 한 번만 fit</b>한 TF-IDF
(vocab {s3man["tfidf"]["vocabulary_size"]:,}, <code>min_df=2</code>, <code>max_df=0.6</code>,
<code>sublinear_tf</code>, <code>norm='l2'</code>)로 모든 기사를 같은 공간에 transform한 뒤 cosine.
pair마다 vectorizer를 다시 fit하지 않았다.</li>
</ul>
<p>표본: semantic_different 50,000 · prefix_2_same 50,000 · prefix_3_same은 모집단의 전체 pair
1,534개. <b>main 결과는 group-balanced sampling</b>을 사용한다 (근거는 §4).
토큰이 양쪽 모두 비는 경우 Jaccard를 0이 아니라 결측으로 두었다.</p>
{table(main_w, cols, heads, cls="wide")}
{img("fig1")}{img("fig2")}
<div class="box key"><h4>순서</h4>
<p>세 지표 모두 <b>semantic_different &lt; prefix_2_same &lt; prefix_3_same</b>.
대비가 가장 큰 지표는 TF-IDF cosine(0.029 → 0.080 → 0.248, <b>8.5배</b>),
가장 약한 지표는 Body Jaccard(0.098 → 0.134 → 0.217, 2.2배)다.</p></div>""")

    # ---------------------------------------------------------------- §3
    A(f"""<h2 id="s3"><span class="no">3</span>Balanced comparison — 세 조건 모두 1,534 pair</h2>
<p class="lead">표본 수 차이가 결과를 만든 게 아님을 확인하기 위해 세 조건을 prefix_3_same의 전체 pair 수에 맞췄다.</p>
{table(bal_w, cols, heads, cls="wide")}
{img("fig3")}{img("fig4")}
<p>표본 수를 맞춰도 순서와 크기가 사실상 그대로다 (prefix_2가 Large 대비 미세하게 낮다).
즉 §2의 결과는 50,000 대 1,534라는 표본 수 불균형의 산물이 아니다.</p>""")

    # ---------------------------------------------------------------- §4
    mg = lambda c, m: s3[(s3.analysis_set == "large") & (s3.condition == c)
                         & (s3.sampling_method == "group_balanced") & (s3.metric == m)]["mean"].iloc[0]
    mu = lambda c, m: s3[(s3.analysis_set == "large") & (s3.condition == c)
                         & (s3.sampling_method == "uniform") & (s3.metric == m)]["mean"].iloc[0]
    rows = []
    for c in ["prefix_2_same", "semantic_different"]:
        for m in MET:
            g_, u_ = mg(c, m), mu(c, m)
            rows.append({"condition": c, "metric": MET_L[m], "group_balanced": g_,
                         "uniform": u_, "diff_pct": 100 * (u_ - g_) / g_})
    A(f"""<h2 id="s4"><span class="no">4</span>group-balanced vs pair-weighted uniform</h2>
<p class="lead">main 결과에 어느 샘플링을 쓸지 정하기 위해 두 방식을 모두 유지하고 비교했다.</p>
{table(pd.DataFrame(rows), ["condition","metric","group_balanced","uniform","diff_pct"],
       ["condition","metric","group-balanced","uniform","차이 %"],
       fmt={"diff_pct": lambda v: "%+.1f%%" % v})}
{img("fig5")}
<div class="box warn"><h4>prefix_2에서만 uniform이 부풀려지는 이유</h4>
<p>pair-weighted uniform은 <b>가능한 모든 pair를 똑같은 확률로</b> 뽑는다. 그런데 기사가
<i>k</i>개인 <code>(c1,c2)</code> group은 pair를 <i>k(k−1)/2</i>개 만들어낸다. 기사 수가
두 배인 group은 pair를 약 <b>네 배</b> 만들어내므로, 소수의 거대 cluster가 표본을 지배한다.
prefix_2_same에서 uniform이 TF-IDF cosine을 <b>+26%</b>, Title Jaccard를 <b>+29%</b>
끌어올리는 것이 그 결과다.</p>
<p>semantic_different는 <code>(c1,c2,c3)</code> group이 8,276개로 잘게 쪼개져 있어 한 group이
지배하지 못하므로 두 방식 차이가 ±4% 이내다. prefix_3_same은 전체 pair를 모두 쓰므로
두 방식이 애초에 동일하다.</p>
<p><b>따라서 main 결과는 group-balanced를 사용하고, uniform은 대조군으로만 제시한다.</b>
group-balanced는 각 group에 균등한 기회를 주므로 “SID group이 의미를 묶는가”라는
질문에 더 맞는 추정량이다.</p></div>""")

    # ---------------------------------------------------------------- §5
    sw = []
    for c in ORDER_C:
        r = sens[(sens.condition == c) & (sens.analysis_set.isin(["large", "large_and_balanced"]))
                 & (sens.sampling_method.isin(["exhaustive", "group_balanced"]))].iloc[0]
        m0 = MAIN[(MAIN.condition == c) & (MAIN.metric == "title_jaccard")].iloc[0]
        sw.append({"condition": CN[c], "n_9338": int(m0["count"]), "mean_9338": m0["mean"],
                   "n_9738": int(r["count"]), "mean_9738": r["mean"],
                   "median_9738": r["median"], "std_9738": r["std"]})
    A(f"""<h2 id="s5"><span class="no">5</span>body-empty 기사 Title-only sensitivity</h2>
<p class="lead">본문이 비어 main analysis에서 제외한 400개 기사를 다시 넣고,
본문이 없으므로 Title Jaccard만 측정했다 (전체 train 9,738).</p>
{table(pd.DataFrame(sw), ["condition","n_9338","mean_9338","n_9738","mean_9738",
                          "median_9738","std_9738"],
       ["condition","n (9,338)","Title Jaccard mean (9,338)","n (9,738)",
        "Title Jaccard mean (9,738)","median","std"])}
{img("fig6")}
<div class="box"><h4>해석 — Title Jaccard에 한정</h4>
<p>body-empty 기사를 포함하면 prefix_3의 Title Jaccard 평균이
<b>0.2059 → 0.2587</b>로 올라간다 (prefix_2·semantic_different는 거의 불변).
즉 <b>body-empty 기사를 제외한 것이 prefix_3의 title similarity를 인위적으로 높인 것은 아니며,
오히려 제외 후 값이 낮아졌다.</b></p>
<p>이 해석은 <b>Title Jaccard에만 한정</b>한다. 본문이 없는 기사이므로 Body Jaccard나
TF-IDF cosine에 대해서는 이 비교로 아무것도 말할 수 없고, 모든 지표에 걸친 일반적인
‘보수적 선택’이라는 주장으로 확대하지 않는다.</p></div>""")

    # ---------------------------------------------------------------- §6
    scols = ["subset","condition","n"] + [f"{m}_{k}" for m in MET for k in ("mean","median","std")]
    sheads = ["subset","condition","n"] + [f"{MET_L[m]} {k}" for m in MET
                                           for k in ("mean","median","std")]
    A(f"""<h2 id="s6"><span class="no">6</span>subset sensitivity — 중복·템플릿 기사 제거</h2>
<p class="lead">질문: prefix_3_same의 높은 수치가 실제 내용 유사성인가, 아니면
완전 중복 기사나 반복성·템플릿 기사군 때문에 부풀려진 것인가.</p>
<p>main analysis 결과는 그대로 두고, <b>pair를 다시 뽑지 않고 §2와 동일한 pair set에
exclusion filter만</b> 적용해 비교 가능성을 유지했다.</p>

<h3>6-1. 완전 중복 본문</h3>
{table(cnt[cnt.analysis_set.isin(["large","large_and_balanced"])
           & cnt.sampling_method.isin(["exhaustive","group_balanced"])],
       ["condition","n_pairs","exact_body_dup","exact_body_dup_pct",
        "touches_repetitive","touches_repetitive_pct"],
       ["condition","n pairs","완전중복 pair","%","side9 낀 pair","%"])}
<div class="kpis">
<div class="kpi"><div class="v">7 / 1,534</div><div class="l">prefix_3의 완전 중복 본문 pair
(<b>{p3cnt.exact_body_dup_pct}%</b>)</div></div>
<div class="kpi"><div class="v">0</div><div class="l">prefix_2·semantic_different의
완전 중복 pair</div></div>
<div class="kpi"><div class="v">17.0%</div><div class="l">prefix_3 pair 중 side9가
한쪽이라도 낀 비율</div></div>
</div>
<p>완전 중복 본문은 prefix_3 1,534쌍 중 <b>7쌍(0.46%)</b>뿐이고 다른 조건에는 한 건도 없다.
<b>중복 기사 때문에 결과가 만들어진 것이 아니다.</b></p>

<h3>6-2. 반복성/템플릿 기사군의 정의 — <code>side9</code></h3>
<p>임의로 고르지 않고 데이터에서 확인한 뒤 정의했다.
<b>정의: <code>category_str == 'side9'</code></b> — {rep["n_articles"]}개, 모집단의
{rep["share_of_population_pct"]}%.</p>
<ul>
<li>데이터셋 자체의 <code>article_type</code> <b>'article_page_nine_girl'</b> 192개가
<b>전부 이 섹션 안에</b> 있고 섹션 밖에는 0개다. EB가 이 섹션을 별도 article type으로 관리한다.
추가로 <code>article_fullscreen_gallery</code>(갤러리) 18개가 포함된다.</li>
<li>본문 token 중앙값 <b>{rep["evidence"]["body_token_median_in_subset"]}</b> vs 나머지
{rep["evidence"]["body_token_median_rest"]} — 약 1/3 수준.
side9 안의 <code>article_default</code> 167개는 본문 token 중앙값이
<b>{rep["evidence"]["side9_article_default_body_token_median"]}</b>인 갤러리 캡션 stub이다.</li>
<li>제목이 <code>&lt;이름&gt;, &lt;나이&gt; år og fra &lt;도시&gt;</code> 템플릿. 이 패턴의 제목
180개 중 179개가 side9다. 나머지는 <code>Galleri med …</code> 형태의 갤러리 제목.</li>
<li>제목 중복률 <b>{rep["evidence"]["title_duplicate_pct_in_subset"]}%</b> vs 나머지
{rep["evidence"]["title_duplicate_pct_rest"]}%.</li>
</ul>
<p>→ 날짜별 반복 기사·갤러리·템플릿 기사군이 맞으므로 별도 sensitivity subset으로 분리했다.</p>

<h3>6-3. 네 subset 비교</h3>
{table(sub_w, scols, sheads, cls="wide")}
{img("fig7")}{img("fig8")}
<div class="box key"><h4>지표별로 방향이 다르다</h4>
<ul>
<li><b>Title Jaccard는 실제로 부풀려져 있었다.</b> side9 제거 시 prefix_3 평균이
<b>0.206 → 0.144 (−30%)</b>. 템플릿 제목이 토큰을 기계적으로 공유하기 때문이다.
prefix_3 대 semantic_different 비율도 19.8배 → 13.7배로 줄어든다.</li>
<li><b>TF-IDF cosine은 대체로 버틴다.</b> <b>0.248 → 0.230 (−7%)</b>에 그치고,
prefix_3 대 semantic_different 비율은 8.5배 → 7.8배로 거의 유지된다.
IDF 가중이 템플릿 공통어를 눌러준다.</li>
<li><b>Body Jaccard는 오히려 올라간다</b> (0.217 → 0.225). side9 기사는 본문이 너무 짧고
서로 다른 인물 이름이 들어가 겹침이 적었다. 즉 side9가 본문 유사도를 <b>높인 것이 아니다</b>.</li>
</ul></div>
<div class="box key"><h4>순서 검증 — 역전 0건</h4>
<p>4 subset × 3 metric × (mean, median) = <b>{len(order)}개 조합</b>에서
<code>semantic_different &lt; prefix_2_same &lt; prefix_3_same</code>의
<b>역전은 한 건도 없다</b> ({n_noinv}/{len(order)}).
엄격한 부등호는 {n_strict}/{len(order)}이며, 깨지는 {len(order)-n_strict}건은 모두
Title Jaccard의 <b>median</b>에서 semantic_different와 prefix_2_same이
<b>둘 다 정확히 0.000인 동률</b>이다(바닥 효과). prefix_3는 네 subset 모두에서 확실히 크다.</p></div>
<div class="box"><h4>이 절의 결론</h4>
<p>side9와 exact duplicate를 제거한 뒤에도 prefix_3의 TF-IDF/Body Jaccard 우위가 유지되어,
해당 결과가 템플릿/중복 기사만으로 설명되지는 않는다. 이는 실제 기사 내용 유사성과
일관된 신호이며, 최종 semantic consistency는 §8(Step 2)의 qualitative evaluation과 함께 해석한다.</p></div>""")

    # ---------------------------------------------------------------- §7
    A(f"""<h2 id="s7"><span class="no">7</span>중요한 caveat</h2>
<div class="box warn"><h4>1. Title Jaccard를 주 지표로 쓰지 않는다</h4>
<p>§6에서 확인했듯 템플릿 기사군을 제거하면 prefix_3의 Title Jaccard가 30% 떨어진다.
제목은 짧아 토큰이 적고(중앙값 한 자릿수) 편집 관행의 영향을 크게 받는다.
Title Jaccard는 보조 지표로만 제시한다.</p></div>
<div class="box warn"><h4>2. Body Jaccard의 절대값을 해석하지 않는다</h4>
<p>불용어를 제거하지 않았으므로 <code>og / i / er / til</code> 같은 덴마크어 기능어가
어떤 두 기사에나 공통으로 들어간다. semantic_different의 Body Jaccard 바닥값이
<b>0.098</b>인 것은 “약 10%의 의미가 겹친다”는 뜻이 <b>아니다</b>.
조건 간 <b>상대 비교</b>로만 쓴다.</p></div>
<div class="box key"><h4>3. Body TF-IDF cosine을 텍스트 정량평가의 주 지표로 사용한다</h4>
<p>IDF 가중이 기능어와 템플릿 공통어를 눌러주므로 semantic_different의 바닥값이
0.029까지 내려가고, 템플릿 제거에도 −7%로 안정적이며, 조건 간 대비(8.5배)가 가장 크다.</p></div>
<div class="box warn"><h4>4. “모든 collision group이 잘 묶였다”고 결론내리지 않는다</h4>
<p>side9와 중복을 모두 제거한 뒤에도 prefix_3의 TF-IDF cosine 표준편차는 <b>0.174</b>로
평균 0.230에 육박한다. 즉 collision group 내부의 이질성은 템플릿 기사로 설명되지 않고 남아 있다.
실제로 §8의 정성평가 30개 group에서 group 평균 TF-IDF cosine은
<b>0.000에서 0.714까지</b> 퍼져 있다.</p></div>
<div class="box warn"><h4>5. 지표의 성격 자체의 한계</h4>
<p>Jaccard와 TF-IDF cosine은 모두 <b>어휘·텍스트 기반</b> 지표다. 어휘가 겹치지 않으면서
의미가 같은 경우(동의어, 다른 표현)를 잡지 못하고, 어휘가 겹치면서 의미가 다른 경우를
걸러내지 못한다. 따라서 진짜 semantic consistency는 §8의 title/body 정성평가와
§9의 reconstruction cosine을 함께 봐야 판단할 수 있다.</p></div>""")

    # ---------------------------------------------------------------- §8
    A(f"""<h2 id="s8"><span class="no">8</span>정성평가와 정량평가의 역할 연결<span class="fb fb2">교수님 피드백 #2</span></h2>
<p class="lead">두 평가는 서로 다른 질문에 답한다.</p>
<ul>
<li><b>정량평가 (§2~§6)</b> — 모집단 전체에서 <b>경향</b>을 확인한다. SID prefix를 많이
공유할수록 텍스트 유사도가 증가하고, 그 순서가 표본 구성·샘플링 방식·중복/템플릿 제거에
흔들리지 않는다는 것까지 말할 수 있다. 하지만 평균값은 <b>어떤 기사가 왜 묶였는지</b>를
말해주지 않는다.</li>
<li><b>정성평가 (아래)</b> — prefix_3 내부의 큰 분산이 구체적으로 어떤 모습인지,
<b>실패 사례</b>가 어떻게 생겼는지를 실제 title/body로 확인한다. 평균이 0.230이어도
group 평균이 0.000인 group과 0.714인 group이 공존한다는 사실은 정성평가에서만 보인다.</li>
</ul>
<p>아래 뷰어에서 group을 클릭하면 그 group에 묶인 <b>모든 기사</b>의 제목·부제·본문을 볼 수 있다.
각 group의 TF-IDF cosine 평균을 함께 표시했고, <b>낮은 순으로 정렬</b>하면 실패 사례가 먼저 나온다.
덴마크어 원문은 번역 없이 그대로 싣는다.</p>
<div class="ctl" id="ctl">
<button data-kind="prefix_3" class="on">prefix_3 collision (30)</button>
<button data-kind="prefix_2">prefix_2 (15)</button>
<button data-kind="semantic_different">semantic_different (30)</button>
<button data-kind="semantic_different_supp">보조: c4까지 다름 (10) — 본 분석 제외</button>
<select id="sort">
<option value="cos_asc">TF-IDF cosine 낮은 순 (실패 사례 먼저)</option>
<option value="cos_desc">TF-IDF cosine 높은 순</option>
<option value="id">group ID 순</option>
<option value="size_desc">group 크기 큰 순</option>
</select>
<button id="exp">모두 펼치기</button><button id="col">모두 접기</button>
</div>
<div class="glist" id="glist"></div>
<script id="data" type="application/json">{json.dumps(payload, ensure_ascii=False)}</script>""")

    # ---------------------------------------------------------------- §9
    A(f"""<h2 id="s9"><span class="no">9</span>Step 5/6 예고 — reconstruction vector cosine<span class="fb fb3">교수님 피드백 #3</span></h2>
<p class="lead">본 보고서가 다루는 것은 교수님 피드백 <b>#2</b>까지다. 피드백 <b>#3</b>은 아직 수행하지 않았다.</p>
<p>지금까지의 평가는 모두 <b>텍스트</b>(제목·본문 토큰)를 근거로 했다. 그런데 RQ-VAE가 실제로
양자화한 대상은 텍스트가 아니라 <b>기사 임베딩 벡터</b>다. 따라서 다음 단계에서는 모델 내부에서
직접 확인한다.</p>
<ul>
<li><b>Step 5</b> — RQ-VAE checkpoint 경로, 최종 config, 임베딩 입력 위치, forward 흐름
(encoder → quantizer → decoder → <code>x̂</code>), reconstruction 차원을 코드에서 확인하고,
train 9,738 + validation 3,122 전체의 reconstruction을 생성할 수 있는지 판단한다.
<b>SID 정수에서 역산하지 않고 실제 checkpoint와 forward path를 사용</b>한다.</li>
<li><b>Step 6</b> — 같은 pair 조건(50,000 / 50,000 / 1,886)에 대해
<code>reconstruction_cosine</code>과 <code>original_embedding_cosine</code>을 계산하고
그래프로 저장한다. 텍스트 기반 지표와 임베딩 기반 지표가 같은 방향을 가리키는지가
핵심 확인 사항이다.</li>
</ul>
<p>이후 validation split에 대해 동일한 코드를 재사용한 분석을 추가한다.</p>

<footer>
<p>생성 스크립트 <span class="path">analysis/rqvae_sid_evaluation/step4/build_report.py</span> ·
seed 42 · 입력 parquet은 전 과정에서 읽기 전용으로만 사용했고 수정하지 않았다 ·
덴마크어 원문(title/subtitle/body)은 번역·가공 없이 원본 그대로 수록했다.</p>
<p>Step 2 표본(30/15/30)과 Step 3 main analysis 결과는 고정되어 있으며 이후 단계에서 변경하지 않는다.</p>
</footer>
</div>

<script>
const DATA = JSON.parse(document.getElementById("data").textContent);
const list = document.getElementById("glist");
const esc = s => s.replace(/[&<>"]/g, c => ({{"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}})[c]);
let kind = "prefix_3";

function sortGroups(gs) {{
  const v = document.getElementById("sort").value, a = gs.slice();
  if (v === "cos_asc")  a.sort((x,y) => (x.m?.tf ?? 9) - (y.m?.tf ?? 9));
  if (v === "cos_desc") a.sort((x,y) => (y.m?.tf ?? -1) - (x.m?.tf ?? -1));
  if (v === "id")       a.sort((x,y) => x.id.localeCompare(y.id));
  if (v === "size_desc")a.sort((x,y) => y.n - x.n || x.id.localeCompare(y.id));
  return a;
}}
function artHTML(a) {{
  const code = a.c.map((x,i) => x === null ? "" : "c"+(i+1)+"="+x).filter(Boolean).join(" · ");
  const side = a.side ? `<span class="tag">${{esc(a.side)}}</span>` : "";
  const sub = a.s ? `<div class="as">${{esc(a.s)}}</div>` : "";
  return `<div class="art"><div class="ah">${{side}}
    <span class="sid">${{esc(code)}}</span>
    <span class="tag">${{esc(a.cat)}}</span>
    <span class="tag">body ${{a.bt}} token</span>
    <span class="aid">${{esc(a.id)}}</span></div>
    <div class="at">${{esc(a.t)}}</div>${{sub}}
    <details><summary>본문 보기 (덴마크어 원문)</summary>
    <div class="body">${{esc(a.b)}}</div></details></div>`;
}}
function render() {{
  const gs = sortGroups(DATA[kind] || []);
  list.innerHTML = gs.map(g => {{
    const m = g.m;
    const met = m ? `<div class="gm">
        <span>TF-IDF <b>${{m.tf.toFixed(3)}}</b></span>
        <span>Body J <b>${{m.bj.toFixed(3)}}</b></span>
        <span>Title J <b>${{m.tj.toFixed(3)}}</b></span>
        <span>pair ${{g.n*(g.n-1)/2}}</span></div>` : "";
    const band = g.band ? `<span class="tag">${{esc(g.band)}}</span>` : "";
    const c3n = g.c3n ? `<span class="tag">c3 종류 ${{g.c3n}}</span>` : "";
    return `<div class="g"><div class="gh">
      <span class="gid">${{esc(g.id)}}</span>
      <span class="sid">${{esc(g.key)}}</span>
      ${{band}}<span class="tag">기사 ${{g.n}}</span>${{c3n}}${{met}}
      </div><div class="gb">${{g.arts.map(artHTML).join("")}}</div></div>`;
  }}).join("");
  list.querySelectorAll(".gh").forEach(h =>
    h.onclick = () => h.parentElement.classList.toggle("open"));
}}
document.querySelectorAll("#ctl button[data-kind]").forEach(b => b.onclick = () => {{
  document.querySelectorAll("#ctl button[data-kind]").forEach(x => x.classList.remove("on"));
  b.classList.add("on"); kind = b.dataset.kind; render();
}});
document.getElementById("sort").onchange = render;
document.getElementById("exp").onclick = () =>
  list.querySelectorAll(".g").forEach(g => g.classList.add("open"));
document.getElementById("col").onclick = () =>
  list.querySelectorAll(".g").forEach(g => g.classList.remove("open"));
render();
</script></body></html>""")

    p = HERE / "sid_evaluation_report.html"
    p.write_text("".join(H), encoding="utf-8")
    print(f"\n완료 → {p}  ({p.stat().st_size/1024/1024:.2f} MB)")


if __name__ == "__main__":
    main()
