"""최종 통합 보고서 — 교수님 피드백 #2(정성 group-first + 텍스트 정량) + #3(reconstruction).

step4/sid_evaluation_report.html(v1)은 보존하고 여기에 갱신본을 만든다.
덴마크어 원문은 번역하지 않는다.
"""
from __future__ import annotations
import base64, html, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
S2, S3, S3B, S4 = ROOT / "step2", ROOT / "step3", ROOT / "step3b_subset_sensitivity", ROOT / "step4"
S6, S6P = ROOT / "step6", ROOT / "step6_prefix1"
S7 = ROOT / "step7_fulltext"
sys.stdout.reconfigure(line_buffering=True)

E = html.escape
FIGS = {
    "fig1": (S3 / "fig1_large_groupbalanced_box.png", "텍스트 유사도 — Large · group-balanced"),
    "fig2": (S3 / "fig2_large_groupbalanced_violin.png", "텍스트 유사도 — violin"),
    "fig3": (S3 / "fig3_balanced_box.png", "Balanced 1,534 × 3"),
    "fig4": (S3 / "fig4_balanced_violin.png", "Balanced — violin"),
    "fig5": (S3 / "fig5_sampling_method_control.png", "group-balanced vs uniform"),
    "fig6": (S3 / "fig6_sensitivity_title_only.png", "전체 train 9,738 · Title Jaccard only"),
    "fig7": (S3B / "fig7_subset_sensitivity.png", "exclusion subset별 변화"),
    "fig8": (S3B / "fig8_subset_distributions.png", "exclusion subset별 분포"),
    "fig9": (S6 / "fig9_embedding_cosine.png", "임베딩 공간 유사도 (3조건)"),
    "fig10": (S6 / "fig10_similarity_preservation.png", "원본 관계의 보존 정도"),
    "fig11": (S6 / "fig11_self_reconstruction.png", "cos(x, x̂) — 개별 기사 복원"),
    "fig12": (S6P / "fig12_four_condition_cosine.png", "4단계 분포 — original / reconstruction / delta"),
    "fig13": (S6P / "fig13_monotonicity.png", "4단계 단조 증가 확인"),
    "fig14": (S7 / "fig14_fulltext_comparison.png",
              "텍스트 범위별 비교 — title-only / body-only / title+subtitle+body"),
}


def img(tag, cls=""):
    p, cap = FIGS[tag]
    b64 = base64.b64encode(p.read_bytes()).decode()
    return (f'<figure class="{cls}"><img src="data:image/png;base64,{b64}" alt="{E(cap)}">'
            f'<figcaption><b>{tag}</b> — {E(cap)} '
            f'<span class="path">{E(p.relative_to(ROOT).as_posix())}</span></figcaption></figure>')


def table(df, cols, heads, fmt=None, cls=""):
    fmt = fmt or {}
    th = "".join(f"<th>{E(h)}</th>" for h in heads)
    rows = []
    for _, r in df.iterrows():
        tds = []
        for c in cols:
            v = r[c]
            if c in fmt:
                s = fmt[c](v)
            elif isinstance(v, float):
                s = "–" if pd.isna(v) else f"{v:.4f}"
            elif isinstance(v, (int, np.integer)):
                s = f"{int(v):,}"
            else:
                s = str(v)
            num = "num" if isinstance(v, (int, float, np.integer, np.floating)) else ""
            tds.append(f'<td class="{num}">{E(s)}</td>')
        rows.append("<tr>" + "".join(tds) + "</tr>")
    return (f'<div class="tw"><table class="{cls}"><thead><tr>{th}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>')


def main():
    if any(p.name not in ("build_final_report_v2.py", "__pycache__") for p in HERE.iterdir()):
        raise SystemExit("출력 폴더에 기존 결과가 있습니다.")

    print("데이터 로드 ...")
    s3 = pd.read_parquet(S3 / "quantitative_summary.parquet")
    sens = pd.read_parquet(S3 / "sensitivity_title_only_summary.parquet")
    sub = pd.read_parquet(S3B / "subset_summary.parquet")
    cnt = pd.read_parquet(S3B / "subset_pair_counts.parquet")
    order = pd.read_parquet(S3B / "subset_order_check.parquet")
    four = pd.read_parquet(S6P / "four_condition_summary.parquet")
    fcorr = pd.read_csv(S6P / "four_condition_correlation.csv")
    mono = pd.read_csv(S6P / "monotonicity_check.csv")
    selfr = pd.read_csv(S6 / "self_reconstruction_summary.csv")
    p3b = pd.read_parquet(S2 / "qualitative_prefix3_band_summary.parquet")
    p2b = pd.read_parquet(S2 / "qualitative_prefix2_band_summary.parquet")
    s2man = json.loads((S2 / "step2_manifest.json").read_text())
    s3man = json.loads((S3 / "step3_manifest.json").read_text())
    s3bman = json.loads((S3B / "step3b_manifest.json").read_text())
    s6man = json.loads((S6 / "step6b_manifest.json").read_text())
    s6pman = json.loads((S6P / "step6c_manifest.json").read_text())
    ft = pd.read_parquet(S7 / "fulltext_summary.parquet")
    ftbal = pd.read_csv(S7 / "fulltext_balanced_summary.csv")
    ftorder = pd.read_csv(S7 / "fulltext_order_check.csv")
    ftman = json.loads((S7 / "fulltext_manifest.json").read_text())

    # step4의 정성평가 payload 재사용 (덴마크어 원문 그대로)
    v1 = (S4 / "sid_evaluation_report.html").read_text(encoding="utf-8")
    import re
    payload_json = re.search(r'<script id="data" type="application/json">(.*?)</script>',
                             v1, re.S).group(1)
    print(f"  정성 payload {len(payload_json)/1024:.0f} KB 재사용")

    ORDER4 = ["semantic_different", "prefix_1_same", "prefix_2_same", "prefix_3_same"]
    NAME4 = {"semantic_different": "Different", "prefix_1_same": "Prefix-1",
             "prefix_2_same": "Prefix-2", "prefix_3_same": "Prefix-3"}
    MAIN3 = s3[(s3.sampling_method.isin(["exhaustive", "group_balanced"]))
               & (s3.analysis_set.isin(["large", "large_and_balanced"]))]
    BAL3 = s3[(s3.sampling_method.isin(["exhaustive", "group_balanced"]))
              & (s3.analysis_set.isin(["balanced", "large_and_balanced"]))]
    HEADSUB = sub[(sub.sampling_method.isin(["exhaustive", "group_balanced"]))
                  & (sub.analysis_set.isin(["large", "large_and_balanced"]))]
    MET3 = ["title_jaccard", "body_jaccard", "body_tfidf_cosine"]
    ML = {"title_jaccard": "Title Jaccard", "body_jaccard": "Body Jaccard",
          "body_tfidf_cosine": "Body TF-IDF cosine"}
    C3 = ["prefix_3_same", "prefix_2_same", "semantic_different"]

    def wide3(df):
        rows = []
        for c in C3:
            r = {"condition": c}
            n = df[(df.condition == c) & (df.metric == "title_jaccard")]
            r["n"] = int(n["count"].iloc[0]) if len(n) else 0
            for m in MET3:
                s = df[(df.condition == c) & (df.metric == m)]
                if len(s):
                    s = s.iloc[0]
                    r[f"{m}_mean"] = s["mean"]; r[f"{m}_median"] = s["median"]
                    r[f"{m}_std"] = s["std"]
            rows.append(r)
        return pd.DataFrame(rows)

    # Step 7 비교표 — 텍스트 범위 x 지표 x 조건 (mean)
    FT_ROWS = [
        ("title-only", "title_jaccard", "Title Jaccard"),
        ("body-only", "body_jaccard", "Body Jaccard"),
        ("body-only", "body_tfidf_cosine", "Body TF-IDF cosine"),
        ("full-text", "fulltext_jaccard", "Full-text Jaccard"),
        ("full-text", "fulltext_tfidf_cosine", "Full-text TF-IDF cosine"),
    ]
    ft_rows = []
    for scope, met, lab in FT_ROWS:
        r = {"scope": scope, "metric": lab}
        for c in ("semantic_different", "prefix_2_same", "prefix_3_same"):
            sel = ft[(ft.condition == c) & (ft.metric == met)]
            r[c] = float(sel["mean"].iloc[0])
            selb = ftbal[(ftbal.condition == c) & (ftbal.metric == met)]
            r[f"{c}_bal"] = float(selb["mean"].iloc[0])
        ft_rows.append(r)
    ft_tbl = pd.DataFrame(ft_rows)
    ftable = table(
        ft_tbl,
        ["scope", "metric", "semantic_different", "prefix_2_same", "prefix_3_same",
         "semantic_different_bal", "prefix_2_same_bal", "prefix_3_same_bal"],
        ["text scope", "metric", "Different", "Prefix-2", "Prefix-3",
         "Different (bal)", "Prefix-2 (bal)", "Prefix-3 (bal)"],
        cls="wide")

    cols3 = ["condition", "n"] + [f"{m}_{k}" for m in MET3 for k in ("mean", "median", "std")]
    heads3 = ["condition", "n"] + [f"{ML[m]} {k}" for m in MET3 for k in ("mean", "median", "std")]

    # 4단계 wide
    f4 = []
    for c in ORDER4:
        r = {"stage": f"{ORDER4.index(c)}", "condition": NAME4[c]}
        n = four[(four.condition == c) & (four.metric == "original_embedding_cosine")].iloc[0]
        r["n"] = int(n["count"])
        for m, k in (("original_embedding_cosine", "orig"), ("reconstruction_cosine", "recon"),
                     ("delta_cosine", "delta")):
            s = four[(four.condition == c) & (four.metric == m)].iloc[0]
            r[f"{k}_mean"] = s["mean"]; r[f"{k}_median"] = s["median"]; r[f"{k}_std"] = s["std"]
        rows_c = fcorr[fcorr.condition == c].iloc[0]
        r["pearson"] = rows_c.pearson_r; r["spearman"] = rows_c.spearman_rho
        f4.append(r)
    f4 = pd.DataFrame(f4)
    nan = lambda v: "정의되지 않음" if pd.isna(v) else f"{v:.3f}"

    ck = s6man["checkpoint"]; mc = ck["model_config"]
    p3q, p2q = s2man["prefix_3"], s2man["prefix_2"]
    rep = s3bman["repetitive_subset"]
    p3cnt = cnt[cnt.condition == "prefix_3_same"].iloc[0]
    n_strict = int(order.order_holds.sum()); n_noinv = int(order.no_inversion.sum())
    pop4 = s6pman["population"]["exact_pair_counts"]

    print("HTML 조립 ...")
    H = []; A = H.append
    A(f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>EB-NeRD RQ-VAE SID 평가 보고서</title>
<style>
:root{{--s1:#fcfcfb;--s2:#f3f2ef;--s3:#eae9e4;--ink:#0b0b0b;--ink2:#52514e;--ink3:#86847d;
--line:#dedcd6;--c1:#2a78d6;--c2:#eb6834;--c3:#1baf7a;--warn:#b4690e;--code:#f0efec;
--r0:#86b6ef;--r1:#3987e5;--r2:#1c5cab;--r3:#0d366b;}}
@media (prefers-color-scheme:dark){{:root:not([data-theme=light]){{
--s1:#1a1a19;--s2:#232321;--s3:#2d2d2a;--ink:#fff;--ink2:#c3c2b7;--ink3:#8d8c83;
--line:#3a3a36;--c1:#3987e5;--c2:#d95926;--c3:#199e70;--warn:#e0a14a;--code:#232321;}}}}
:root[data-theme=dark]{{--s1:#1a1a19;--s2:#232321;--s3:#2d2d2a;--ink:#fff;--ink2:#c3c2b7;
--ink3:#8d8c83;--line:#3a3a36;--c1:#3987e5;--c2:#d95926;--c3:#199e70;--warn:#e0a14a;--code:#232321;}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--s1);color:var(--ink);
font:15px/1.72 -apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans KR",sans-serif;}}
.wrap{{max-width:1140px;margin:0 auto;padding:0 16px 96px}}
header{{padding:52px 0 26px;border-bottom:1px solid var(--line);margin-bottom:28px}}
h1{{font-size:28px;line-height:1.3;margin:0 0 10px;letter-spacing:-.01em}}
.sub{{color:var(--ink2);font-size:14.5px;margin:0}}
.meta{{display:flex;flex-wrap:wrap;gap:8px;margin-top:18px}}
.chip{{background:var(--s2);border:1px solid var(--line);border-radius:999px;
padding:4px 12px;font-size:12.5px;color:var(--ink2)}}
h2{{font-size:21px;margin:54px 0 6px;letter-spacing:-.01em;scroll-margin-top:16px}}
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
nav{{background:var(--s2);border:1px solid var(--line);border-radius:12px;padding:16px 20px}}
nav ol{{margin:0;padding-left:20px;columns:2;column-gap:36px}}
nav li{{margin:3px 0;break-inside:avoid}}
nav a{{color:var(--ink);text-decoration:none;border-bottom:1px solid transparent}}
nav a:hover{{border-bottom-color:var(--ink3)}}
.tw{{overflow-x:auto;margin:16px 0;border:1px solid var(--line);border-radius:10px}}
table{{border-collapse:collapse;width:100%;font-size:13px}}
th,td{{padding:8px 11px;text-align:left;border-bottom:1px solid var(--line);white-space:nowrap}}
th{{background:var(--s2);font-weight:650;color:var(--ink2);font-size:12.2px;position:sticky;top:0}}
td.num{{text-align:right;font-variant-numeric:tabular-nums}}
tbody tr:last-child td{{border-bottom:none}}
tbody tr:hover td{{background:var(--s2)}}
figure{{margin:22px 0}}
figure img{{width:100%;height:auto;border:1px solid var(--line);border-radius:10px;background:#fcfcfb}}
figcaption{{color:var(--ink2);font-size:12.5px;margin-top:8px}}
.path{{color:var(--ink3);font-family:ui-monospace,Menlo,monospace;font-size:11.5px}}
code{{background:var(--code);padding:1.5px 5px;border-radius:4px;
font-family:ui-monospace,Menlo,monospace;font-size:12.8px}}
.box{{border:1px solid var(--line);border-left:3px solid var(--c1);background:var(--s2);
border-radius:0 10px 10px 0;padding:14px 18px;margin:20px 0}}
.box.warn{{border-left-color:var(--warn)}} .box.key{{border-left-color:var(--c3)}}
.box h4{{margin:0 0 7px;font-size:14px}}
.box p:last-child,.box ul:last-child{{margin-bottom:0}}
ul{{margin:11px 0;padding-left:21px}} li{{margin:5px 0}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(168px,1fr));gap:12px;margin:20px 0}}
.kpi{{background:var(--s2);border:1px solid var(--line);border-radius:11px;padding:14px 16px}}
.kpi .v{{font-size:25px;font-weight:680;letter-spacing:-.02em;font-variant-numeric:tabular-nums}}
.kpi .l{{font-size:12.3px;color:var(--ink2);margin-top:3px;line-height:1.45}}
.stair{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:22px 0}}
.st{{border:1px solid var(--line);border-radius:11px;padding:13px 15px;color:#fff}}
.st:nth-child(1){{background:var(--r0);color:#0b0b0b}} .st:nth-child(2){{background:var(--r1)}}
.st:nth-child(3){{background:var(--r2)}} .st:nth-child(4){{background:var(--r3)}}
.st .n{{font-size:11.5px;opacity:.9;font-weight:650;letter-spacing:.04em}}
.st .t{{font-size:15px;font-weight:700;margin:3px 0 2px}}
.st .d{{font-size:11.8px;opacity:.92;line-height:1.5;font-family:ui-monospace,Menlo,monospace}}
.ctl{{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:18px 0 14px}}
.ctl button,.ctl select{{background:var(--s2);color:var(--ink);border:1px solid var(--line);
border-radius:8px;padding:6px 13px;font-size:13px;cursor:pointer;font-family:inherit}}
.ctl button.on{{background:var(--c1);border-color:var(--c1);color:#fff}}
.glist{{display:grid;gap:10px}}
.g{{border:1px solid var(--line);border-radius:11px;background:var(--s2);overflow:hidden}}
.gh{{display:flex;flex-wrap:wrap;gap:12px;align-items:center;padding:12px 16px;cursor:pointer;user-select:none}}
.gh:hover{{background:var(--s3)}}
.gid{{font-weight:700;font-size:13.5px;font-family:ui-monospace,Menlo,monospace}}
.sid{{font-family:ui-monospace,Menlo,monospace;font-size:12.5px;color:var(--ink2);
background:var(--s3);padding:2px 8px;border-radius:5px}}
.gm{{margin-left:auto;display:flex;gap:14px;font-size:12.3px;color:var(--ink2);
font-variant-numeric:tabular-nums}}
.gm b{{color:var(--ink);font-weight:650}}
.gb{{display:none;padding:2px 16px 16px;border-top:1px solid var(--line)}}
.g.open .gb{{display:block}}
.art{{border-top:1px solid var(--line);padding:13px 0}} .art:first-child{{border-top:none}}
.ah{{display:flex;flex-wrap:wrap;gap:9px;align-items:baseline;margin-bottom:5px}}
.aid{{font-family:ui-monospace,Menlo,monospace;font-size:11.8px;color:var(--ink3)}}
.tag{{font-size:11.3px;background:var(--s3);border:1px solid var(--line);border-radius:5px;
padding:1px 7px;color:var(--ink2)}}
.at{{font-weight:650;font-size:14.6px;line-height:1.45}}
.as{{color:var(--ink2);font-size:13.4px;margin-top:3px}}
details{{margin-top:7px}} summary{{cursor:pointer;font-size:12.5px;color:var(--ink2);outline:none}}
.body{{white-space:pre-wrap;font-size:13.2px;line-height:1.68;color:var(--ink2);margin-top:8px;
padding:11px 14px;background:var(--s1);border:1px solid var(--line);border-radius:8px;
max-height:380px;overflow-y:auto}}
.dk{{font-size:11.6px;color:var(--ink3);margin-top:5px}}
footer{{margin-top:70px;padding-top:22px;border-top:1px solid var(--line);
color:var(--ink3);font-size:12.6px}}
@media(max-width:860px){{.stair{{grid-template-columns:repeat(2,1fr)}}}}
@media(max-width:720px){{nav ol{{columns:1}} h1{{font-size:23px}} .gm{{margin-left:0;width:100%}}}}
</style></head><body><div class="wrap">

<header>
<h1>EB-NeRD RQ-VAE Semantic ID 평가 보고서</h1>
<p class="sub">교수님 피드백 #2 (정성평가 group-first 설계 + 텍스트 정량평가) ·
#3 (reconstruction vector cosine)</p>
<div class="meta">
<span class="chip">모델 <b>WD-001-UNI-lu0.05-m0.5</b></span>
<span class="chip">body-valid train <b>9,338</b></span>
<span class="chip">정성 group 30 / 15 / 30</span>
<span class="chip">Main 4-condition pair <b>151,534</b></span>
<span class="chip">Total evaluated pair rows <b>307,670</b></span>
<span class="chip">seed 42</span>
<span class="chip">입력 read-only</span>
<span class="chip">덴마크어 원문 무번역</span>
</div></header>

<nav><ol>
<li><a href="#s1">최종 사용 모델 provenance</a></li>
<li><a href="#s2">SID 유사도 4단계 정의</a></li>
<li><a href="#s3">정성평가 설계 — group-first</a></li>
<li><a href="#s4">텍스트 정량평가 — Jaccard / TF-IDF</a></li>
<li><a href="#s5">Balanced comparison</a></li>
<li><a href="#s6">group-balanced vs uniform</a></li>
<li><a href="#s7">body-empty Title-only sensitivity</a></li>
<li><a href="#s8">중복·템플릿 subset sensitivity</a></li>
<li><a href="#s9">임베딩 정량평가 — 4단계</a></li>
<li><a href="#s10">self-reconstruction과 거리 압축</a></li>
<li><a href="#s11">caveat</a></li>
<li><a href="#s12">정성·정량 연결 + 실제 group</a></li>
<li><a href="#s13">남은 작업 및 추가 분석</a></li>
</ol></nav>

<div class="box key"><h4>이 보고서의 결론</h4>
<p><b>더 긴 SID prefix를 공유하는 기사 쌍일수록 유사도가 더 높게 나타났다.</b>
두 종류의 분석이 서로 다른 조건 수로 수행되었으므로 아래와 같이 구분해 서술한다.</p>
<ul>
<li><b>텍스트 기반 지표</b> —
<b style="font-family:ui-monospace,Menlo,monospace">Different &lt; Prefix-2 &lt; Prefix-3</b>
순서가 <b>title-only, body-only, 그리고 title+subtitle+body 전체 텍스트 범위에서도
일관되게 관찰되었다.</b> 표본 수를 맞추거나 중복·템플릿 기사를 제거해도 순서 역전은
없었다. <b>Prefix-1의 텍스트 지표는 계산하지 않았으므로 텍스트 쪽을 4단계로 서술하지
않는다.</b></li>
<li><b>원본 embedding과 reconstruction</b> —
<b style="font-family:ui-monospace,Menlo,monospace">Different &lt; Prefix-1 &lt; Prefix-2 &lt; Prefix-3</b>
의 <b>4단계 단조 증가</b>가 평균과 중앙값 모두에서 관찰되었다.</li>
</ul>
<p>두 분석은 포함한 조건 수가 다르지만, <b>“더 긴 SID prefix 공유가 더 높은 유사도와
연관된다”는 동일한 방향의 근거를 제공한다.</b></p>
<p>다만 <b>Prefix-3의 reconstruction cosine = 1.0은 모델 구조에서 자동으로 따라오는 값</b>이므로
성능 근거로 쓰지 않는다. 그 값을 제외하고 Different → Prefix-1 → Prefix-2만 보아도
<b>0.9294 &lt; 0.9522 &lt; 0.9810</b>으로 증가한다는 점이 핵심 근거다.
또한 <b>collision group 내부의 분산은 여전히 크다</b>. 본 보고서는 RQ-VAE가 잘 학습되었다는
결론을 주장하지 않는다.</p></div>""")

    # ---------------------------------------------------------------- §1
    vs = ck.get("validation_state", {})
    A(f"""<h2 id="s1"><span class="no">1</span>최종 사용 모델 provenance</h2>
<p class="lead">본 보고서의 모든 수치는 아래 단일 checkpoint에서 나온 SID와 reconstruction에 기반한다.</p>
<div class="tw"><table><tbody>
<tr><th>experiment</th><td><code>WD-001-UNI-lu0.05-m0.5</code></td></tr>
<tr><th>checkpoint</th><td><code>checkpoint_best_rec.pt</code> · variant <b>best_rec</b></td></tr>
<tr><th>epoch</th><td><b>{int(ck["epoch"]) + 1}</b> (0-index {ck["epoch"]}, global_step {ck["global_step"]})</td></tr>
<tr><th>SHA256</th><td class="path">{E(ck["sha256"])}</td></tr>
<tr><th>loss</th><td>lambda_rec {mc["lambda_rec"]} · lambda_cb {mc["lambda_cb"]} ·
lambda_com {mc["lambda_com"]} · <b>lambda_uniq {mc.get("lambda_uniq","–")}</b> ·
<b>uniqueness_margin {mc.get("uniqueness_margin","–")}</b></td></tr>
<tr><th>구조</th><td>input_dim {mc["input_dim"]} · hidden {mc["hidden_dims"]} ·
embed_dim {mc["embed_dim"]} · c1 {mc["num_categories"]} · c2 {mc["c2_codebook_size"]} ·
c3 {mc["c3_codebook_size"]} · {mc["codebook_mode"]} · normalize {mc["codebook_normalize"]}</td></tr>
</tbody></table></div>

<div class="box key"><h4>SID 재현 검증 — 이 checkpoint가 본 보고서의 SID를 실제로 만들었음을 확인</h4>
<ul>
<li><b>Train SID 재현 9,738 / 9,738 완전 일치.</b>
<code>generate_semantic_ids.py</code>의 train 경로는 event-level C2로 덮어쓰지 않으므로,
같은 모델이면 c1·c2·c3가 100% 재현되어야 하고 실제로 그러했다.</li>
<li>Validation은 2,243 / 3,122 일치. 나머지는 코드가 event-level <code>fixed_c2_ids</code>를
강제하는 경로를 쓰기 때문이며 오류가 아니다.</li>
<li>checkpoint에 기록된 <code>best_valid_rec_loss</code> {vs.get("best_valid_rec_loss", 0):.7f}를
사용 임베딩으로 재계산하니 {0.1524576:.7f} (차 4.3e-06). 임베딩도 이 모델의 학습·검증에
쓰인 파일이 맞다.</li>
<li>동일 <code>(c1,c2,c3)</code> 기사들의 <code>x̂</code> 최대 절대차 <b>0.000e+00</b> (1,229 group).</li>
</ul>
<p class="dk">uniqueness loss는 <code>forward()</code>에 loss 항만 추가하고
encoder/decoder/quantizer 구조를 바꾸지 않는다 (<code>load_state_dict</code>의
missing·unexpected 모두 0). 따라서 reconstruction 경로는 base 모델과 동일하다.</p></div>""")

    # ---------------------------------------------------------------- §2
    A(f"""<h2 id="s2"><span class="no">2</span>SID 유사도 4단계 정의</h2>
<p class="lead">두 기사가 Semantic ID의 앞부분을 얼마나 공유하는지로 네 단계를 나눈다.
색이 진할수록 공유 prefix가 많다.</p>
<div class="stair">
<div class="st"><div class="n">공유 prefix 0</div><div class="t">Different</div>
<div class="d">c1 ≠ c1<br>c2 ≠ c2<br>c3 ≠ c3</div></div>
<div class="st"><div class="n">공유 prefix 1</div><div class="t">Prefix-1</div>
<div class="d">c1 = c1<br><b>c2 ≠ c2</b></div></div>
<div class="st"><div class="n">공유 prefix 2</div><div class="t">Prefix-2</div>
<div class="d">c1 = c1<br>c2 = c2<br><b>c3 ≠ c3</b></div></div>
<div class="st"><div class="n">공유 prefix 3</div><div class="t">Prefix-3</div>
<div class="d">c1 = c1<br>c2 = c2<br>c3 = c3</div></div>
</div>
<p><code>c4</code>는 의미 단계가 아니라 <b>동일 <code>(c1,c2,c3)</code>를 받은 기사를 구분하기 위한
suffix</b>이므로 어떤 조건에도 쓰지 않는다.</p>
<div class="box"><h4>조건별 분석 범위 — 텍스트와 임베딩의 조건 수가 다르다</h4>
<div class="tw"><table><thead><tr><th>조건</th>
<th>텍스트 지표 (Jaccard · TF-IDF)</th><th>임베딩 지표 (original · reconstruction)</th>
</tr></thead><tbody>
<tr><td>Different</td><td>○ 계산함</td><td>○ 계산함</td></tr>
<tr><td><b>Prefix-1</b></td><td><b>계산하지 않음</b></td><td>○ 계산함</td></tr>
<tr><td>Prefix-2</td><td>○ 계산함</td><td>○ 계산함</td></tr>
<tr><td>Prefix-3</td><td>○ 계산함</td><td>○ 계산함</td></tr>
</tbody></table></div>
<p>따라서 <b>§4~§8의 텍스트 정량평가는 Different / Prefix-2 / Prefix-3의 3조건</b>이고,
<b>§9~§10의 임베딩 정량평가는 Prefix-1을 포함한 4단계</b>다. 두 절의 결과를 읽을 때
조건 수를 혼동하지 않도록 주의한다.</p></div>

<h3>모집단 내 전체 pair 수 (body-valid train 9,338, 포함배제로 정확 계산)</h3>
<div class="tw"><table><thead><tr><th>조건</th><th>전체 pair 수</th><th>본 분석 표본</th>
<th>추출 방식</th></tr></thead><tbody>
<tr><td>Different</td>
<td class="num">{s3man["population_main"]["exact_pair_counts"]["semantic_different"]:,}</td>
<td class="num">50,000</td><td>group-balanced</td></tr>
<tr><td>Prefix-1</td><td class="num">{pop4["prefix_1_same"]:,}</td>
<td class="num">50,000</td><td>group-balanced</td></tr>
<tr><td>Prefix-2</td><td class="num">{pop4["prefix_2_same"]:,}</td>
<td class="num">50,000</td><td>group-balanced</td></tr>
<tr><td>Prefix-3</td><td class="num">{pop4["prefix_3_same"]:,}</td>
<td class="num">1,534</td><td>전수</td></tr>
</tbody></table></div>
<p class="dk">Different는 c1·c2·c3가 <b>각각</b> 모두 다른 조건이므로 단순 뺄셈이 아니라
포함배제로 계산했다 (전체 {pop4["total_pairs"]:,} pair 기준). Prefix-3만 전수이고,
나머지 세 조건은 가능한 pair가 5만을 넘어 group-balanced로 추출했다.</p>""")

    # ---------------------------------------------------------------- §3
    A(f"""<h2 id="s3"><span class="no">3</span>정성평가 설계 — group-first sampling<span class="fb fb2">피드백 #2</span></h2>
<p class="lead">무작위 pair가 아니라 <b>group을 먼저 뽑고 그 group의 기사를 전부 포함</b>했다.</p>
<ul>
<li><b>collision은 group 현상이다.</b> 같은 <code>(c1,c2,c3)</code>를 받은 기사 6개를 pair로
흩어 뽑으면 “이 6개가 하나로 묶인 게 타당한가”에 답할 수 없다.</li>
<li><b>pair 단위로 뽑으면 큰 group이 자동으로 과대 대표된다.</b> 크기 <i>k</i>인 group은
pair를 <i>k(k−1)/2</i>개 만들므로 균등 추출 시 큰 group이 제곱으로 많이 뽑힌다 (§6에서 정량 확인).</li>
<li>선택된 group의 기사는 <b>일부가 아니라 전부</b> 포함했다.</li>
</ul>
<h3>prefix_3 collision group — 30 group / 기사 {p3q["articles_selected"]}개</h3>
{table(p3b, ["band","groups_selected","articles","available_groups_in_population","distinct_c1","distinct_c1_c2"],
       ["size 구간","선택 group","기사","모집단 내 group","distinct c1","distinct (c1,c2)"])}
<div class="box warn"><h4>이 표본은 분포 추정용이 아니다</h4>
<ul>
<li>실제 분포에서 size 2~3 group이 eligible group의
<b>{p3q["real_population_distribution"]["share_size_2_3_of_eligible"]*100:.1f}%</b>를 차지한다.</li>
<li>그럼에도 <b>size 4~8 group을 의도적으로 oversampling</b>했다. 큰 collision에서 의미적
일관성이 무너지는 사례를 보기 위한 <b>진단 목적</b>이다.</li>
<li>이 모델의 EB-NeRD SID에는 <b>size 9 이상 collision group이 없다</b> (최대 8).
MIND 참고자료의 18/8/4 구성을 재현할 수 없어 15/10/5로 조정했다.</li>
</ul></div>
<h3>prefix_2 group — 15 group / 기사 {p2q["articles_selected"]}개</h3>
{table(p2b, ["band","groups_selected","articles","available_groups_in_population","distinct_c1","distinct_c1_c2"],
       ["기사 수 구간","선택 group","기사","모집단 내 group","distinct c1","distinct (c1,c2)"])}
<h3>semantic_different — 30 group-pair</h3>
<p><code>c1,c2,c3</code>가 모두 다른 두 group에서 대표 기사를 하나씩 뽑아 30쌍(기사 60개, distinct c1 21).
<code>c4</code>까지 다른 10쌍은 별도 보조 파일로 분리하고 본 분석에서 제외했다.</p>
<p class="dk">각 구간 안에서 동일 <code>c1</code>·<code>(c1,c2)</code>에 몰리지 않도록 greedy diversity
샘플링(seed 42)을 적용했다. prefix_3 30 group의 distinct <code>(c1,c2)</code>는 29/30이다.</p>""")

    # ---------------------------------------------------------------- §4~§8
    A(f"""<h2 id="s4"><span class="no">4</span>텍스트 정량평가 — Jaccard / TF-IDF<span class="fb fb2">피드백 #2</span></h2>
<p class="lead">정성 표본과 <b>무관하게</b> 9,338 모집단에서 별도로 뽑은 pair에 측정했다.
TF-IDF는 9,338 body corpus에 <b>단 한 번만 fit</b>(vocab {s3man["tfidf"]["vocabulary_size"]:,},
<code>norm='l2'</code>)한 뒤 전 기사를 같은 공간으로 transform했다.</p>
{table(wide3(MAIN3), cols3, heads3)}
{img("fig1")}{img("fig2")}
<div class="box key"><h4>순서</h4>
<p>세 지표 모두 <b>semantic_different &lt; prefix_2_same &lt; prefix_3_same</b>.</p></div>

<h3>4-2. 텍스트 범위 세 가지</h3>
<p class="lead">어떤 텍스트를 기준으로 유사도를 재느냐에 따라 의미가 달라지므로
범위를 셋으로 나누어 함께 제시한다.</p>
<div class="stair" style="grid-template-columns:repeat(3,1fr)">
<div class="st" style="background:var(--r0);color:#0b0b0b"><div class="n">범위 1</div>
<div class="t">title-only</div><div class="d">제목만<br>모델 입력의 일부</div></div>
<div class="st" style="background:var(--r2)"><div class="n">범위 2</div>
<div class="t">body-only</div><div class="d">본문만<br><b>모델 입력이 아님</b></div></div>
<div class="st" style="background:var(--r3)"><div class="n">범위 3</div>
<div class="t">title + subtitle + body</div><div class="d">기사 전체<br>입력 + 입력 밖</div></div>
</div>
{ftable}
{img("fig14")}
<p class="dk">full-text Jaccard·TF-IDF는 Step 3과 <b>동일한 pair set</b>(재샘플링 없음,
pair_id 1:1 대응)에 계산했다. tokenize 규칙은 동일하며, TF-IDF는 같은 설정
(<code>min_df=2</code>, <code>max_df=0.6</code>, <code>sublinear_tf</code>,
<code>norm='l2'</code>)으로 full_text corpus 9,338에 <b>새로 1회 fit</b>했다
(vocab {ftman["tfidf"]["vocabulary_size"]:,}). subtitle이 빈
{ftman["subtitle_empty_count"]}건은 빈 문자열로 처리했다. 덴마크어 원문만 사용했다.</p>

<div class="box"><h4>body-only 결과를 어떻게 읽는가</h4>
<ul>
<li><b>body는 article embedding 생성에 사용되지 않았다.</b> 임베딩 입력은
<code>src/build_train.py:158-190</code>의 <code>model_text</code>,
즉 <code>title + "\n" + subtitle</code>이다.</li>
<li>따라서 body-only 결과는 <b>모델 입력 밖의 텍스트에서도 SID 공유 수준과 내용 유사도의
관계가 나타나는지 확인하는 외부 검증</b>이다.</li>
</ul></div>

<div class="box"><h4>full-text 결과를 어떻게 읽는가</h4>
<ul>
<li>full-text는 title + subtitle + body이므로 <b>모델이 본 title/subtitle과 모델이 보지 않은
body가 함께 포함된다.</b></li>
<li>따라서 full-text 전체를 <b>“순수한 외부 검증”이라고 표현하지 않는다.</b>
<b>기사 전체 내용 수준의 consistency 검증</b>으로 본다.</li>
</ul></div>

<div class="box key"><h4>핵심 해석</h4>
<p><b>body-only 분석에서는 모델이 직접 입력으로 사용하지 않은 본문에서도 SID 공유 수준에
따른 유사도 순서가 관찰되었고, title+subtitle+body 전체 텍스트로 범위를 확장해도 동일한
순서가 유지되었다.</b></p>
<p>신규 두 지표 모두 <b>Different &lt; Prefix-2 &lt; Prefix-3</b> 순서가
<b>mean과 median에서 유지</b>되었으며, <b>balanced n=1,534에서도 같은 결론</b>이다
(<code>fulltext_order_check.csv</code>).</p>
<p>정확한 해석의 범위는 다음까지다 —
<b>title+subtitle 기반으로 생성된 SID가 모델이 직접 보지 않은 body 내용과도 일관된 관계를
보였다.</b></p></div>

<h2 id="s5"><span class="no">5</span>Balanced comparison — 세 조건 모두 1,534 pair</h2>
<p class="lead">표본 수 차이가 결과를 만든 게 아님을 확인하기 위해 세 조건을 맞췄다.</p>
{table(wide3(BAL3), cols3, heads3)}
{img("fig3")}{img("fig4")}
<p>표본 수를 맞춰도 순서와 크기가 사실상 그대로다.</p>

<h2 id="s6"><span class="no">6</span>group-balanced vs pair-weighted uniform</h2>
{img("fig5")}
<div class="box warn"><h4>prefix_2에서만 uniform이 부풀려진다</h4>
<p>기사가 <i>k</i>개인 <code>(c1,c2)</code> group은 pair를 <i>k(k−1)/2</i>개 만든다.
기사 수가 두 배인 group은 pair를 약 <b>네 배</b> 만들므로 소수의 거대 cluster가 표본을 지배한다.
prefix_2_same에서 uniform이 TF-IDF cosine을 <b>+26%</b>, Title Jaccard를 <b>+29%</b> 끌어올린다.
semantic_different는 group이 8,276개로 잘게 쪼개져 차이가 ±4% 이내다.</p>
<p><b>따라서 본 보고서의 main 수치는 group-balanced이고 uniform은 대조군으로만 쓴다.</b></p></div>

<h2 id="s7"><span class="no">7</span>body-empty 기사 Title-only sensitivity</h2>
{img("fig6")}
<div class="box"><h4>해석 — Title Jaccard에 한정</h4>
<p>본문이 비어 제외했던 400개 기사를 포함하면 prefix_3의 Title Jaccard 평균이
<b>0.2059 → 0.2587</b>로 올라간다. 즉 <b>body-empty 기사를 제외한 것이 prefix_3의
title similarity를 인위적으로 높인 것은 아니며, 오히려 제외 후 값이 낮아졌다.</b></p>
<p>이 해석은 <b>Title Jaccard에만 한정</b>한다. 본문이 없는 기사이므로 Body Jaccard나
TF-IDF cosine에 대해서는 이 비교로 아무것도 말할 수 없다.</p></div>

<h2 id="s8"><span class="no">8</span>중복·템플릿 subset sensitivity</h2>
<p class="lead">prefix_3의 높은 수치가 완전 중복 기사나 반복성 기사군 때문에 부풀려진 것은 아닌지
확인했다. pair를 다시 뽑지 않고 §4와 <b>동일한 pair set에 filter만</b> 적용했다.</p>
<div class="kpis">
<div class="kpi"><div class="v">7 / 1,534</div><div class="l">prefix_3의 완전 중복 본문 pair
(<b>{p3cnt.exact_body_dup_pct}%</b>)</div></div>
<div class="kpi"><div class="v">0</div><div class="l">prefix_2·semantic_different의 완전 중복 pair</div></div>
<div class="kpi"><div class="v">17.0%</div><div class="l">prefix_3 pair 중 side9가 낀 비율</div></div>
</div>
<p><b>반복성 subset 정의</b>: <code>category_str == 'side9'</code> ({rep["n_articles"]}개,
{rep["share_of_population_pct"]}%). 임의 지정이 아니라 데이터로 확인했다 —
데이터셋 자체의 <code>article_type</code> <b>'article_page_nine_girl'</b> 192개가 전부 이 섹션 안에
있고, 제목이 <code>&lt;이름&gt;, &lt;나이&gt; år og fra &lt;도시&gt;</code> 템플릿이며, 본문 token
중앙값이 {rep["evidence"]["body_token_median_in_subset"]}으로 나머지
{rep["evidence"]["body_token_median_rest"]}의 1/3 수준이다.</p>
{img("fig7")}{img("fig8")}
<div class="box key"><h4>지표별로 방향이 다르다</h4>
<ul>
<li><b>Title Jaccard는 실제로 부풀려져 있었다.</b> side9 제거 시 prefix_3 평균이
<b>0.206 → 0.144 (−30%)</b>.</li>
<li><b>TF-IDF cosine은 대체로 버틴다.</b> <b>0.248 → 0.230 (−7%)</b>.</li>
<li><b>Body Jaccard는 오히려 올라간다</b> (0.217 → 0.225). side9가 본문 유사도를 높인 게 아니다.</li>
</ul></div>
<div class="box key"><h4>순서 검증 — 역전 0건</h4>
<p>4 subset × 3 metric × (mean, median) = <b>{len(order)}개 조합</b>에서 역전이 한 건도 없다
({n_noinv}/{len(order)}). 엄격한 부등호는 {n_strict}/{len(order)}이며, 깨지는
{len(order)-n_strict}건은 모두 Title Jaccard의 median에서 두 조건이 <b>정확히 0.000인 동률</b>이다.</p>
<p>즉 side9와 exact duplicate를 제거한 뒤에도 prefix_3의 TF-IDF/Body Jaccard 우위가 유지되어,
해당 결과가 템플릿/중복 기사만으로 설명되지는 않는다. 이는 실제 기사 내용 유사성과 일관된
신호이며, 최종 semantic consistency는 §12의 qualitative evaluation과 함께 해석한다.</p></div>""")

    # ---------------------------------------------------------------- §9
    cols4 = ["stage", "condition", "n", "orig_mean", "orig_median", "orig_std",
             "recon_mean", "recon_median", "recon_std", "delta_mean", "pearson", "spearman"]
    heads4 = ["공유 prefix", "조건", "n", "original mean", "median", "std",
              "reconstruction mean", "median", "std", "delta mean", "Pearson", "Spearman"]
    A(f"""<h2 id="s9"><span class="no">9</span>임베딩 정량평가 — 4단계<span class="fb fb3">피드백 #3</span></h2>
<p class="lead">§4~§8은 텍스트를 근거로 한 <b>3조건</b>(Different / Prefix-2 / Prefix-3) 분석이었다.
여기서는 RQ-VAE가 실제로 양자화한 대상인 <b>기사 임베딩</b>과 그것을 decoder로 복원한
<b>reconstructed vector</b>를 직접 비교하며, <b>Prefix-1을 포함한 4단계</b>로 수행한다.</p>
<p><code>x̂ = decoder(Q1[c1] + Q2[c2] + Q3[c3])</code>로 기사별 reconstruction을 만들고,
Step의 동일 pair set에 두 cosine을 계산했다 (재샘플링 없음).</p>
{table(f4, cols4, heads4, fmt={"pearson": nan, "spearman": nan})}
{img("fig13")}
<div class="box key"><h4>핵심 결과 — 임베딩 지표에서 네 단계 모두 단조 증가</h4>
<p style="font-family:ui-monospace,Menlo,monospace;font-size:13.5px;line-height:2">
original&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0.7986 &lt; 0.8180 &lt; 0.8470 &lt; 0.9070<br>
reconstruction&nbsp;0.9294 &lt; 0.9522 &lt; 0.9810 &lt; 1.0000</p>
<p>평균과 중앙값 모두에서 성립한다 (4/4 조합, <code>monotonicity_check.csv</code>).
이 4단계 단조 증가는 <b>임베딩 지표에 한정된 결과</b>이며, 텍스트 지표는 Prefix-1을
포함하지 않은 3조건으로 측정했다 (§4).</p></div>
<div class="box warn"><h4>Prefix-3의 reconstruction cosine = 1.0은 성능 근거가 아니다</h4>
<p>두 기사가 <code>(c1,c2,c3)</code>를 공유하면 <code>q1, q2, q3</code>가 같고 → decoder 입력이
같고 → <code>x̂</code>가 비트 단위로 동일하다. 따라서 <b>1.0은 모델 구조에서 자동으로 따라오는
값</b>이며, 실제로 1.0에서의 최대 편차는 <b>3.6e-07</b>로 부동소수점 오차 수준이다.
이 값을 학습이 잘 되었다는 근거로 쓰면 순환 논증이 된다.</p>
<p><b>그래서 Prefix-3을 빼고 보아야 한다.</b> Different → Prefix-1 → Prefix-2 세 단계만 보아도
<b style="font-family:ui-monospace,Menlo,monospace">0.9294 &lt; 0.9522 &lt; 0.9810</b>으로 증가한다.
이 세 단계에는 구조적 자명함이 없으며, 모델이 서로 다른 code를 배정한 쌍들 사이에서 나타난
차이다. <b>이것이 피드백 #3에 대한 실질적 근거다.</b></p></div>
{img("fig12")}
<p>단계 간 간격도 균일하지 않다. original 기준으로 Different → Prefix-1은 +0.019인 반면
Prefix-2 → Prefix-3은 +0.060이다. 상위 단계만 공유하는 것보다 <code>c3</code>까지 일치할 때
유사도 차이가 더 크게 나타났다.</p>
{img("fig9")}""")

    # ---------------------------------------------------------------- §10
    sa = selfr[selfr.split == "all"].iloc[0]
    st = selfr[selfr.split == "train"].iloc[0]
    sv = selfr[selfr.split == "validation"].iloc[0]
    A(f"""<h2 id="s10"><span class="no">10</span>self-reconstruction과 거리 압축<span class="fb fb3">피드백 #3</span></h2>
<h3>10-1. 개별 기사 복원 — <code>cos(x, x̂)</code></h3>
<div class="kpis">
<div class="kpi"><div class="v">{sa["mean"]:.4f}</div><div class="l">전체 12,860</div></div>
<div class="kpi"><div class="v">{st["mean"]:.4f}</div><div class="l">train 9,738</div></div>
<div class="kpi"><div class="v">{sv["mean"]:.4f}</div><div class="l">validation 3,122</div></div>
</div>
{table(selfr, ["split","count","mean","median","std","q1","q3","min","max"],
       ["split","n","mean","median","std","Q1","Q3","min","max"])}
{img("fig11")}
<p>768차원 임베딩을 {mc["embed_dim"]}차원 code 세 개로 양자화한 뒤 복원한 결과다.
train과 validation의 차이가 {st["mean"]-sv["mean"]:.4f}로 작아, 복원 성능이 학습 데이터에만
치우쳐 있다는 징후는 뚜렷하지 않다.</p>
<p class="dk">cosine은 두 벡터가 이루는 각도의 코사인이다. 0.93을 “93% 유사”처럼 비율로
읽어서는 안 된다.</p>

<h3>10-2. reconstruction이 모든 쌍을 서로 가깝게 만든다</h3>
<div class="box warn"><h4>delta가 전 pair에서 양수다</h4>
<p><code>delta = reconstruction cosine − original cosine</code>이
<b>307,670 pair 전부에서 양수</b>다. 예외가 하나도 없다.
<code>abs_delta</code>와 <code>delta</code>의 값이 모든 조건에서 동일한 것이 그 결과다.</p>
<ul>
<li>Different &nbsp; 0.7986 → 0.9294 &nbsp;(delta +0.1308)</li>
<li>Prefix-1 &nbsp;&nbsp; 0.8180 → 0.9522 &nbsp;(delta +0.1342)</li>
<li>Prefix-2 &nbsp;&nbsp; 0.8470 → 0.9810 &nbsp;(delta +0.1340)</li>
<li>Prefix-3 &nbsp;&nbsp; 0.9070 → 1.0000 &nbsp;(delta +0.0930)</li>
</ul>
<p>원본에서 cosine 0.80이던 무관한 기사 쌍도 reconstruction에서는 0.93이 된다.
{mc["embed_dim"]}차원 code 세 개의 합이 만들 수 있는 조합이 유한하므로 복원 벡터가
좁은 영역에 모이기 때문이다. 조건 간 <b>순서</b>는 보존되지만 <b>거리의 크기</b>는
압축되어, 기사 간 미세한 거리 정보가 상당 부분 사라진다.</p>
<p>특히 Prefix-3에서는 이 손실이 완전하다. 같은 group의 원본 cosine은 평균 0.907,
<b>최솟값 0.789</b>인데 reconstruction에서는 모두 정확히 1.0이 된다. 즉 원래 존재하던
차이가 그 group 안에서는 남지 않는다.</p></div>

<h3>10-3. 보조 결과 — 원본 관계의 보존 정도</h3>
{img("fig10")}
{table(fcorr, ["condition","n_pairs","pearson_r","spearman_rho","note"],
       ["조건","n","Pearson r","Spearman ρ","비고"],
       fmt={"pearson_r": nan, "spearman_rho": nan})}
<div class="box"><h4>해석에 주의</h4>
<p>이 상관계수는 <b>보조 결과</b>다. “원본에서 더 비슷했던 쌍이 reconstruction에서도 더
비슷한 순위를 유지하는가”를 보는 값이며, <b>정확도나 백분율로 해석해서는 안 된다.</b>
Pearson 0.33을 “33% 정확”처럼 읽는 것은 잘못이다.</p>
<p>Prefix-3은 reconstruction cosine이 상수 1.0이라 상관계수가 정의되지 않는다.
값의 실제 범위 7.2e-07은 부동소수점 오차이므로, 여기에 상관을 계산하면 오차에 대한
상관이 되어 의미가 없다.</p></div>""")

    # ---------------------------------------------------------------- §11
    A(f"""<h2 id="s11"><span class="no">11</span>중요한 caveat</h2>
<div class="box warn"><h4>1. Title Jaccard를 주 지표로 쓰지 않는다</h4>
<p>템플릿 기사군을 제거하면 prefix_3의 Title Jaccard가 30% 떨어진다 (§8).
제목은 토큰이 적고 편집 관행의 영향을 크게 받아 보조 지표로만 제시한다.</p></div>
<div class="box warn"><h4>2. Body Jaccard의 절대값을 해석하지 않는다</h4>
<p>불용어를 제거하지 않았으므로 <code>og / i / er / til</code> 같은 덴마크어 기능어가
어느 두 기사에나 공통으로 들어간다. semantic_different의 바닥값 0.098은
“약 10%의 의미가 겹친다”는 뜻이 아니다. 조건 간 상대 비교로만 쓴다.</p></div>
<div class="box warn"><h4>3. 임베딩 cosine의 절대값도 마찬가지다</h4>
<p>Different 조건의 original cosine이 <b>0.799</b>로 이미 높다. 밀집 문장 임베딩에서 흔히
나타나는 성질이며, “무관한 기사도 80% 비슷하다”는 뜻이 아니다. 여기서도 조건 간
상대 비교만 유효하다.</p></div>
<div class="box warn"><h4>4. body는 RQ-VAE의 직접 입력이 아니다</h4>
<ul>
<li><b>article embedding은 title + subtitle 기반이다</b>
(<code>src/build_train.py:158-190</code>의 <code>model_text</code>).</li>
<li><b>body는 RQ-VAE의 직접 입력이 아니다.</b></li>
<li>따라서 <b>body-only는 입력 밖 텍스트를 이용한 외부 consistency 검증</b>이다.</li>
<li><b>full-text는 모델 입력(title/subtitle)과 입력 밖 텍스트(body)를 동시에 포함하므로
순수한 external-only metric은 아니다.</b></li>
</ul></div>
<div class="box key"><h4>5. 주 지표</h4>
<p>텍스트 쪽은 <b>Body TF-IDF cosine</b>, 임베딩 쪽은 <b>original embedding cosine</b>과
<b>Prefix-3을 제외한 reconstruction cosine</b>을 주 지표로 삼는다.</p></div>
<div class="box warn"><h4>6. “모든 collision group이 잘 묶였다”고 결론내리지 않는다</h4>
<p>side9와 중복을 모두 제거한 뒤에도 prefix_3의 TF-IDF cosine 표준편차는 <b>0.174</b>로
평균 0.230에 육박한다. §12의 정성평가 30개 group에서 group 평균 TF-IDF cosine은
<b>0.000에서 0.714까지</b> 퍼져 있다.</p></div>
<div class="box warn"><h4>7. 지표의 성격 자체의 한계</h4>
<p>Jaccard와 TF-IDF cosine은 어휘 기반이라 어휘가 겹치지 않으면서 의미가 같은 경우를 잡지
못한다. 임베딩 cosine은 어휘를 넘어서지만 임베딩 모델 자체의 편향을 그대로 물려받는다.
서로 다른 성질의 지표가 같은 방향을 가리킨다는 점이 근거를 보강하지만, 어느 하나도
semantic consistency를 직접 측정하지는 않는다.</p></div>""")

    # ---------------------------------------------------------------- §12
    A(f"""<h2 id="s12"><span class="no">12</span>정성평가와 정량평가의 역할 연결<span class="fb fb2">피드백 #2</span></h2>
<ul>
<li><b>정량평가 (§4~§10)</b> — 모집단 전체에서 <b>경향</b>을 확인한다.
두 갈래의 결과를 조건 수까지 구분해 정리하면 다음과 같다.
<ul>
<li><b>텍스트 지표 (§4~§8, 3조건)</b> — <code>Different &lt; Prefix-2 &lt; Prefix-3</code>의
순서가 관찰되었고, 표본 수를 맞추거나 샘플링 방식을 바꾸거나 중복·템플릿 기사를
제거해도 순서 역전이 없었다. <b>Prefix-1의 텍스트 지표는 계산하지 않았으므로
이 갈래를 4단계로 서술해서는 안 된다.</b></li>
<li><b>임베딩 지표 (§9~§10, 4단계)</b> — <code>Different &lt; Prefix-1 &lt; Prefix-2 &lt;
Prefix-3</code>의 단조 증가가 평균과 중앙값 모두에서 관찰되었다.</li>
</ul>
포함한 조건 수는 다르지만 두 갈래 모두 <b>더 긴 SID prefix 공유가 더 높은 유사도와
연관된다</b>는 같은 방향을 가리킨다. 다만 평균값은 <b>어떤 기사가 왜 묶였는지</b>를
말해주지 않는다.</li>
<li><b>정성평가 (아래)</b> — prefix_3 내부의 큰 분산이 구체적으로 어떤 모습인지, <b>실패 사례</b>가
어떻게 생겼는지를 실제 title/body로 확인한다. 평균이 0.230이어도 group 평균이 0.000인 group과
0.714인 group이 공존한다는 사실은 정성평가에서만 보인다.</li>
</ul>
<div class="box warn"><h4>진행 상태 — 사람의 판정은 아직 수행되지 않았다</h4>
<p>아래 뷰어는 정성평가를 위해 <b>표본을 구성하고 자료를 제시한 단계</b>까지다.
각 group에 대한 <b>높음 / 중간 / 낮음 판정은 아직 이루어지지 않았다.</b>
따라서 이 절의 내용을 완료된 human evaluation 결과로 인용해서는 안 되며,
본 보고서의 결론은 모두 정량 지표에 근거한다.</p></div>
<p>group을 클릭하면 그 group의 <b>모든 기사</b>의 제목·부제·본문을 볼 수 있다.
각 group의 TF-IDF cosine 평균을 함께 표시했고, <b>낮은 순으로 정렬</b>하면 실패 후보가 먼저 나온다.
덴마크어 원문은 번역 없이 그대로 싣는다.</p>
<div class="ctl" id="ctl">
<button data-kind="prefix_3" class="on">prefix_3 collision (30)</button>
<button data-kind="prefix_2">prefix_2 (15)</button>
<button data-kind="semantic_different">semantic_different (30)</button>
<button data-kind="semantic_different_supp">보조: c4까지 다름 (10) — 본 분석 제외</button>
<select id="sort">
<option value="cos_asc">TF-IDF cosine 낮은 순 (실패 후보 먼저)</option>
<option value="cos_desc">TF-IDF cosine 높은 순</option>
<option value="id">group ID 순</option>
<option value="size_desc">group 크기 큰 순</option>
</select>
<button id="exp">모두 펼치기</button><button id="col">모두 접기</button>
</div>
<div class="glist" id="glist"></div>
<script id="data" type="application/json">{payload_json}</script>

<h2 id="s13"><span class="no">13</span>남은 작업 및 추가 분석</h2>
<p class="lead">교수님 요청 범위 기준으로 <b>정량평가는 완료되었다.</b> 아래는 필수로 남은 작업과,
현재 요청 범위 밖의 추가 분석을 구분한 것이다.</p>

<h3>필수 남은 작업</h3>
<ul>
<li><b>정성 판정</b> — §12의 30 / 15 / 30 group에 대한 <b>높음 / 중간 / 낮음</b> 판정.
판정이 끝나면 group 평균 TF-IDF cosine과 사람 판정의 일치도를 볼 수 있다.</li>
</ul>
<p><b>이 정성 판정이 완료되면 현재 교수님이 요청한 정성·정량 검증 범위는 마무리된다.</b></p>

<h3>추가 분석 — 현재 교수님 요청 범위 밖</h3>
<ul>
<li><b>validation split 분석</b> — 동일 코드로 validation 3,122개에 대한 분석.</li>
<li><b>uniqueness loss 유무 baseline 비교</b> —
<code>WD-001-UNI-lu0.05-m0.5</code> vs uniqueness loss 없는 baseline.</li>
</ul>

<footer>
<p>모델 <code>WD-001-UNI-lu0.05-m0.5</code> · <code>checkpoint_best_rec</code> epoch
{int(ck["epoch"]) + 1} · Train SID 9,738/9,738 재현 확인 · seed 42 ·
입력 parquet·npy·checkpoint는 전 과정에서 읽기 전용으로만 사용했다 ·
덴마크어 원문(title/subtitle/body)은 번역·가공 없이 원본 그대로 수록했다.</p>
<p>이전 버전 보고서는 그대로 보존되어 있다 —
<span class="path">analysis/rqvae_sid_evaluation/step4/sid_evaluation_report.html</span> (v1),
<span class="path">analysis/rqvae_sid_evaluation/final_report/sid_evaluation_report.html</span> (v2).
이 문서는 Step 7(full-text 텍스트 범위 비교)을 추가한 v3이다.</p>
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
    <span class="sid">${{esc(code)}}</span><span class="tag">${{esc(a.cat)}}</span>
    <span class="tag">body ${{a.bt}} token</span><span class="aid">${{esc(a.id)}}</span></div>
    <div class="at">${{esc(a.t)}}</div>${{sub}}
    <details><summary>본문 보기 (덴마크어 원문)</summary>
    <div class="body">${{esc(a.b)}}</div></details></div>`;
}}
function render() {{
  const gs = sortGroups(DATA[kind] || []);
  list.innerHTML = gs.map(g => {{
    const m = g.m;
    const met = m ? `<div class="gm"><span>TF-IDF <b>${{m.tf.toFixed(3)}}</b></span>
        <span>Body J <b>${{m.bj.toFixed(3)}}</b></span>
        <span>Title J <b>${{m.tj.toFixed(3)}}</b></span>
        <span>pair ${{g.n*(g.n-1)/2}}</span></div>` : "";
    const band = g.band ? `<span class="tag">${{esc(g.band)}}</span>` : "";
    const c3n = g.c3n ? `<span class="tag">c3 종류 ${{g.c3n}}</span>` : "";
    return `<div class="g"><div class="gh"><span class="gid">${{esc(g.id)}}</span>
      <span class="sid">${{esc(g.key)}}</span>${{band}}
      <span class="tag">기사 ${{g.n}}</span>${{c3n}}${{met}}</div>
      <div class="gb">${{g.arts.map(artHTML).join("")}}</div></div>`;
  }}).join("");
  list.querySelectorAll(".gh").forEach(h =>
    h.onclick = () => h.parentElement.classList.toggle("open"));
}}
document.querySelectorAll("#ctl button[data-kind]").forEach(b => b.onclick = () => {{
  document.querySelectorAll("#ctl button[data-kind]").forEach(x => x.classList.remove("on"));
  b.classList.add("on"); kind = b.dataset.kind; render();
}});
document.getElementById("sort").onchange = render;
document.getElementById("exp").onclick = () => list.querySelectorAll(".g").forEach(g => g.classList.add("open"));
document.getElementById("col").onclick = () => list.querySelectorAll(".g").forEach(g => g.classList.remove("open"));
render();
</script></body></html>""")

    p = HERE / "sid_evaluation_report.html"
    p.write_text("".join(H), encoding="utf-8")
    print(f"\n완료 → {p}  ({p.stat().st_size/1024/1024:.2f} MB)")


if __name__ == "__main__":
    main()
