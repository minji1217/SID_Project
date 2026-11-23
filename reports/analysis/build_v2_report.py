"""v2 재실험 보고서 HTML 생성 (seed 42). 숫자는 AWS 실행 결과를 그대로 옮긴 것.

    python reports/analysis/build_v2_report.py   →  reports/v2_재실험_결과_seed42.html

출처 (AWS, ~/minji/SID_Project/data/output/fixedcluster_v1/v2/)
  rqvae/*/summary.json, sid/{h,r}/check_sid.txt, transformer/*/test_metrics.json,
  analysis/test_breakdown_seed42.json, analysis/popularity_fusion_seed42.json
"""
from pathlib import Path
from html import escape

OUT = Path(__file__).resolve().parents[1] / "v2_재실험_결과_seed42.html"
COL = {"V1": "var(--s-v1)", "H": "var(--s-h)", "R": "var(--s-r)", "인기만": "var(--s-pop)"}

# ---------------- 데이터 ----------------
TEST = {  # 모델만
    "V1": dict(top1=26.16, auc=0.5686, mrr=0.5101, ndcg=0.6308),
    "H": dict(top1=27.51, auc=0.5836, mrr=0.5223, ndcg=0.6401),
    "R": dict(top1=27.67, auc=0.5888, mrr=0.5249, ndcg=0.6422),
}
FUSION = {  # test Top-1: 모델만 / 인기만 / 결합(λ=5) / 결합+나이(λ=5, μ=-1)
    "V1": (26.16, 42.94, 43.23, 43.41),
    "H": (27.51, 42.93, 43.96, 44.21),
    "R": (27.67, 42.97, 44.10, 44.23),
}
FUSION_FULL = {  # 결합+나이 test: auc, mrr, ndcg
    "V1": (0.7394, 0.6546, 0.7413), "H": (0.7418, 0.6589, 0.7445), "R": (0.7438, 0.6599, 0.7454),
}
LAMS = ["0", "0.05", "0.1", "0.2", "0.3", "0.5", "0.75", "1", "1.5", "2", "3", "5", "10"]
VAL_LAM = {
    "V1": [25.91, 27.17, 28.32, 30.58, 32.57, 35.60, 38.02, 39.63, 41.46, 42.33, 42.90, 43.06, 42.88],
    "H": [27.58, 28.84, 30.01, 32.16, 33.89, 36.65, 38.93, 40.33, 41.91, 42.67, 43.14, 43.27, 43.00],
    "R": [27.11, 28.21, 29.28, 31.31, 33.13, 35.92, 38.20, 39.78, 41.60, 42.56, 43.32, 43.52, 43.25],
}
C2_USE = [("H s42", "H", 128), ("H s123", "H", 128), ("H s2026", "H", 128), ("R s42", "R", 59), ("R s123", "R", 93)]
COS_BINS = ["< 0.25", "0.25~0.35", "0.35~0.45", "≥ 0.45"]
COS_TOP1 = {"H": [4.42, 19.76, 27.37, 31.92], "R": [1.36, 23.38, 26.77, 34.85]}
COS_N = {"H": [679, 18557, 57857, 42847], "R": [1320, 31762, 53346, 33433]}


# ---------------- 그림 함수 ----------------
def bars(cats, series, ymax, unit="%", ref=None, ref_label="", h=260, fmt="{:.2f}", ns=None):
    """묶은 세로 막대. series: [(이름, 값 리스트)]"""
    W, L, B, T, RM = 640, 44, 36, 18, 74
    pw, ph = W - L - RM, h - B - T
    ng, ns_ = len(cats), len(series)
    gw = pw / ng
    bw = min(46, (gw - 18) / ns_ - 2)
    y = lambda v: T + ph - v / ymax * ph
    s = [f'<svg viewBox="0 0 {W} {h}" class="chart" role="img">']
    for k in range(5):
        v = ymax * k / 4
        s.append(f'<line x1="{L}" x2="{W-RM}" y1="{y(v):.1f}" y2="{y(v):.1f}" class="grid"/>'
                 f'<text x="{L-6}" y="{y(v)+4:.1f}" class="tick" text-anchor="end">{v:g}</text>')
    if ref is not None:
        s.append(f'<line x1="{L}" x2="{W-RM}" y1="{y(ref):.1f}" y2="{y(ref):.1f}" class="ref"/>'
                 f'<text x="{W-RM+6}" y="{y(ref)+4:.1f}" class="tick">{escape(ref_label)}</text>')
    for gi, c in enumerate(cats):
        gx = L + gi * gw + (gw - ns_ * (bw + 2)) / 2
        for si, (name, vals) in enumerate(series):
            v = vals[gi]
            x = gx + si * (bw + 2)
            yy = y(v)
            extra = f" · {ns[name][gi]:,}행" if ns else ""
            s.append(f'<g class="mark"><title>{escape(name)} · {escape(c)}: {fmt.format(v)}{unit}{extra}</title>'
                     f'<path d="M{x:.1f},{T+ph} V{yy+4:.1f} q0,-4 4,-4 h{bw-8:.1f} q4,0 4,4 V{T+ph} Z" fill="{COL[name]}"/>'
                     f'<rect x="{x-1:.1f}" y="{T}" width="{bw+2:.1f}" height="{ph}" fill="transparent"/></g>'
                     f'<text x="{x+bw/2:.1f}" y="{yy-5:.1f}" class="val" text-anchor="middle">{fmt.format(v)}</text>')
        s.append(f'<text x="{L+gi*gw+gw/2:.1f}" y="{h-12}" class="cat" text-anchor="middle">{escape(c)}</text>')
    s.append(f'<line x1="{L}" x2="{W-RM}" y1="{T+ph}" y2="{T+ph}" class="axis"/></svg>')
    return "".join(s)


def legend(names):
    return '<div class="legend">' + "".join(
        f'<span><i style="background:{COL[n]}"></i>{escape(n)}</span>' for n in names) + "</div>"


def lines(xs, series, ymin, ymax, h=270):
    W, L, B, T = 640, 44, 36, 14
    pw, ph = W - L - 16, h - B - T
    X = lambda i: L + i / (len(xs) - 1) * pw
    Y = lambda v: T + ph - (v - ymin) / (ymax - ymin) * ph
    s = [f'<svg viewBox="0 0 {W} {h}" class="chart" role="img">']
    for v in range(int(ymin), int(ymax) + 1, 5):
        s.append(f'<line x1="{L}" x2="{W-16}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" class="grid"/>'
                 f'<text x="{L-6}" y="{Y(v)+4:.1f}" class="tick" text-anchor="end">{v}</text>')
    for i, xl in enumerate(xs):
        s.append(f'<text x="{X(i):.1f}" y="{h-14}" class="tick" text-anchor="middle">{xl}</text>')
    for name, vals in series:
        pts = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(vals))
        s.append(f'<polyline points="{pts}" fill="none" stroke="{COL[name]}" stroke-width="2"/>')
        for i, v in enumerate(vals):
            s.append(f'<g class="mark"><title>{escape(name)} · λ={xs[i]}: validation Top-1 {v:.2f}%</title>'
                     f'<circle cx="{X(i):.1f}" cy="{Y(v):.1f}" r="4" fill="{COL[name]}" stroke="var(--surface)" stroke-width="2"/>'
                     f'<circle cx="{X(i):.1f}" cy="{Y(v):.1f}" r="11" fill="transparent"/></g>')
    s.append(f'<text x="{(L+W-16)/2:.0f}" y="{h-1}" class="tick" text-anchor="middle">λ (인기 가중치)</text>')
    s.append(f'<line x1="{L}" x2="{W-16}" y1="{T+ph}" y2="{T+ph}" class="axis"/></svg>')
    return "".join(s)


def hbars(rows, xmax, h_per=34):
    """가로 막대 (c2 코드 사용 수)."""
    W, L = 640, 80
    h = len(rows) * h_per + 26
    X = lambda v: L + v / xmax * (W - L - 50)
    s = [f'<svg viewBox="0 0 {W} {h}" class="chart" role="img">']
    for v in (0, 32, 64, 96, 128):
        s.append(f'<line x1="{X(v):.1f}" x2="{X(v):.1f}" y1="4" y2="{h-22}" class="grid"/>'
                 f'<text x="{X(v):.1f}" y="{h-6}" class="tick" text-anchor="middle">{v}</text>')
    for i, (lab, ser, v) in enumerate(rows):
        yy = 8 + i * h_per
        s.append(f'<text x="{L-8}" y="{yy+15}" class="cat" text-anchor="end">{escape(lab)}</text>'
                 f'<g class="mark"><title>{escape(lab)}: c2 코드 {v} / 128 사용</title>'
                 f'<path d="M{L},{yy+2} H{X(v)-4:.1f} q4,0 4,4 v10 q0,4 -4,4 H{L} Z" fill="{COL[ser]}"/></g>'
                 f'<text x="{X(v)+6:.1f}" y="{yy+16}" class="val">{v}</text>')
    s.append("</svg>")
    return "".join(s)


def table(head, rows, num_from=1):
    th = "".join(f'<th{" class=n" if i >= num_from else ""}>{escape(c)}</th>' for i, c in enumerate(head))
    body = ""
    for r in rows:
        cls = ' class="hl"' if str(r[0]).startswith("★") else ""
        body += f"<tr{cls}>" + "".join(f'<td{" class=n" if i >= num_from else ""}>{c}</td>' for i, c in enumerate(r)) + "</tr>"
    return f'<div class="tbl"><table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>'


# ---------------- 본문 ----------------
fig_main = bars(["모델만 (SID)", "인기만", "결합 + 나이"],
                [(n, [FUSION[n][0], FUSION[n][1], FUSION[n][3]]) for n in ["V1", "H", "R"]],
                50, ref=20, ref_label="무작위 20%")
fig_c2 = hbars(C2_USE, 128)
fig_cos = bars(COS_BINS, [("H", COS_TOP1["H"]), ("R", COS_TOP1["R"])], 40, ref=20, ref_label="무작위 20%", ns=COS_N)
fig_lam = lines(LAMS, [(n, VAL_LAM[n]) for n in ["V1", "H", "R"]], 25, 45)

html = f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>v2 재실험 결과</title>
<style>
:root {{ --bg:#ffffff; --surface:#fcfcfb; --text:#0b0b0b; --muted:#52514e; --line:#e4e3df; --chip:#f3f3f1;
  --s-h:#2a78d6; --s-r:#eb6834; --s-v1:#1baf7a; --s-pop:#6250d6; --hl:#eef4ff; --hl-line:#9fb8ec; --warn-bg:#fff6e6; --warn-line:#e9c27a; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#141413; --surface:#1a1a19; --text:#f0efec; --muted:#c3c2b7; --line:#34332f; --chip:#24231f;
  --s-h:#3987e5; --s-r:#d95926; --s-v1:#199e70; --s-pop:#9085e9; --hl:#1d2740; --hl-line:#3c5694; --warn-bg:#2e2615; --warn-line:#7a6329; }} }}
:root[data-theme="dark"] {{ --bg:#141413; --surface:#1a1a19; --text:#f0efec; --muted:#c3c2b7; --line:#34332f; --chip:#24231f;
  --s-h:#3987e5; --s-r:#d95926; --s-v1:#199e70; --s-pop:#9085e9; --hl:#1d2740; --hl-line:#3c5694; --warn-bg:#2e2615; --warn-line:#7a6329; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--text); font-family:system-ui,-apple-system,"Apple SD Gothic Neo","Malgun Gothic","Segoe UI",sans-serif; font-size:15px; line-height:1.65; }}
main {{ max-width:960px; margin:0 auto; padding:32px 16px 72px; }}
h1 {{ font-size:24px; margin:0 0 4px; }} .sub {{ color:var(--muted); margin:0 0 20px; font-size:14px; }}
h2 {{ font-size:19px; margin:40px 0 10px; padding-bottom:6px; border-bottom:2px solid var(--line); }}
h3 {{ font-size:16px; margin:22px 0 8px; }}
.box {{ border:1.5px solid var(--hl-line); background:var(--hl); border-radius:12px; padding:14px 18px; }}
.box ul, .note ul {{ margin:6px 0 0; padding-left:20px; }}
.note {{ border-left:4px solid var(--warn-line); background:var(--warn-bg); border-radius:0 8px 8px 0; padding:10px 14px; margin:12px 0; font-size:14px; }}
.fig {{ background:var(--surface); border:1px solid var(--line); border-radius:12px; padding:12px 12px 6px; margin:10px 0; }}
.chart {{ width:100%; height:auto; display:block; }}
.chart text {{ font-family:inherit; }} .tick {{ fill:var(--muted); font-size:11px; }} .cat {{ fill:var(--text); font-size:12.5px; }}
.val {{ fill:var(--text); font-size:11.5px; font-weight:600; }}
.grid {{ stroke:var(--line); stroke-width:1; }} .axis {{ stroke:var(--muted); stroke-width:1; }} .ref {{ stroke:var(--muted); stroke-dasharray:4 4; }}
.mark:hover path, .mark:hover circle:first-of-type {{ opacity:.8; }}
.legend {{ display:flex; gap:16px; flex-wrap:wrap; font-size:13px; color:var(--muted); margin:2px 4px 6px; }}
.legend i {{ display:inline-block; width:12px; height:12px; border-radius:3px; margin-right:6px; vertical-align:-1px; }}
.cap {{ color:var(--muted); font-size:13.5px; margin:6px 0 0; }} .cap b {{ color:var(--text); }}
.tbl {{ overflow-x:auto; border:1px solid var(--line); border-radius:10px; margin:10px 0; }}
table {{ width:100%; border-collapse:collapse; font-size:14px; }}
th, td {{ padding:7px 10px; border-bottom:1px solid var(--line); text-align:left; white-space:nowrap; }}
th {{ background:var(--chip); font-weight:600; }} td.n, th.n {{ text-align:right; font-variant-numeric:tabular-nums; }}
tr.hl td {{ background:var(--hl); font-weight:600; }}
.grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:12px; }} .grid2 > * {{ min-width:0; }}
@media (max-width:720px) {{ .grid2 {{ grid-template-columns:1fr; }} }}
</style></head><body><main>

<h1>v2 재실험 결과 — 새 SID · 실험 H vs R · 인기 결합</h1>
<p class="sub">EB-NeRD small · 여민 코드(RQ-VAE·SID 생성·Transformer) + 우리 설정(RQ-VAE UNI, Transformer V1) · Transformer seed 42 · 2026-11-23</p>

<div class="box"><b>한 줄 요약</b>
<ul>
<li><b>새 전처리 + UNI 재학습으로 만든 SID(H, R)가 예전 SID(V1)보다 좋아요.</b> 같은 코드·같은 조건에서 test Top-1 26.16% → 27.51%(H) · 27.67%(R).</li>
<li><b>H와 R은 Transformer 성능이 거의 같아요.</b> 다만 R은 Q2 코드를 128개 중 59~93개만 써요(H는 128개 모두).</li>
<li><b>병목은 두 곳이에요.</b> ① test 정답의 98%가 새 기사인데, 사건에 엉뚱하게 붙은 새 기사는 거의 못 맞혀요. ② 인기 정보가 없어요. 직전 24시간 클릭 수만으로 42.9%를 맞혀요.</li>
<li><b>재학습 없이 인기·나이를 점수에 더하면 44.2%.</b> SID는 인기 위에 +1.3%p를 더 보태고, 예전 SID(+0.5%p)보다 2.7배 많이 보태요.</li>
</ul></div>

<h2>1. 실험 구성</h2>
{table(["실험", "SID", "Q2 초기화", "validation-only 사건 C2", "돌린 단계"],
 [["V1 재실행 (기준)", "예전 UNI SID", "—", "—", "Transformer만"],
  ["H", "새 전처리 + UNI 재학습", "사건별 h 평균", "사건 h 평균 → 가장 가까운 Q2", "RQ-VAE부터 전부"],
  ["R", "새 전처리 + UNI 재학습", "사건별 r1 평균", "사건 r1 평균 → 가장 가까운 Q2", "RQ-VAE부터 전부"]], num_from=99)}
<p class="cap">공통: 여민 코드 <b>22028ba</b>(우리 UNI 학습 코드와 한 글자도 다르지 않음을 확인), 갱신 정책 B, 사건 붙이기는 IDF 가중 Jaccard ≥ 0.3 → h 코사인, Transformer V1 설정(history 50, d256 · 8 head · 2층, lr 5e-5, dropout 0) fp32, <b>후보 순서를 섞은 1pos4neg</b>. SID는 RQ-VAE seed 42 모델로 만들었어요.</p>
<div class="note"><b>V1 숫자가 예전 기록(27.07%)과 다른 이유</b> — 예전 숫자는 후보를 섞기 전 데이터(정답이 항상 첫 자리)와 AMP(bf16)로 잰 값이라, 점수 동점이 정답으로 처리되며 부풀려졌어요. 이번엔 섞은 데이터 + fp32라서 V1 기준은 <b>26.16%</b>로 새로 잡았어요.</div>

<h2>2. RQ-VAE — R은 Q2 코드북을 절반 정도만 써요</h2>
<div class="grid2">
<div>
{table(["모델", "best_rec (validation 복원 손실)", "best epoch", "끝난 epoch"],
 [["H s42", "0.150854", "190", "390"], ["R s42", "0.152190", "300", "500"], ["R s123", "0.152124", "470", "670"]])}
<p class="cap">H s42는 여민 코드로 돌렸는데 우리 ver2 s42와 <b>소수점 여섯째 자리까지 같아요</b>(코드 이식 검증). R은 복원 손실이 약 0.9% 높고 늦게 수렴해요.</p>
</div>
<div class="fig">{legend(["H", "R"])}{fig_c2}
<p class="cap">Train 기사의 c2 코드 종류 (128개 중). H s123 · s2026은 기존 ver2 모델.</p></div>
</div>
{table(["", "H s42", "R s42"],
 [["Train 기사 c2 종류", "128", "59"], ["(c1, c2) 조합 수", "1,048", "659"],
  ["SID가 다른 기사와 겹치는 기사 (c4 &gt; 0)", "15.1%", "24.4%"], ["Train 사건이 받은 C2 종류", "128", "59"]])}
<p class="cap"><b>원인으로 보이는 것:</b> R은 학습 <b>전</b>에 r1 = h − Q1[카테고리]의 사건 평균으로 Q2를 KMeans 초기화해요. 그 시점 Q1은 랜덤인데 크기가 h보다 약 2.5배 커서(평균 노름 0.26 vs 0.10), 초기 중심이 카테고리 쪽으로 몰리고 많은 코드가 끝까지 안 쓰여요. seed 두 개 모두 같은 방향이라 R 방식의 특성으로 봐요.</p>

<h2>3. SID 생성 — 새 기사는 대부분 코사인으로 사건에 붙어요</h2>
{table(["", "H", "R"],
 [["기사 수 / SID 중복", "13,160 / 0", "13,160 / 0"],
  ["Jaccard ≥ 0.3으로 붙음", "34", "35"],
  ["72h 안 코사인으로 붙음", "2,439 (89%)", "2,434"],
  ["72h 안 사건 없음 → 전체 코사인", "268", "272"],
  ["72h 코사인 값 (25% / 중앙값 / 75%)", "0.250 / 0.363 / 0.443", "0.233 / 0.336 / 0.419"],
  ["validation-only가 물려받은 C2 종류", "127", "57"]])}
<p class="cap">R1 코드에 H checkpoint를 넣어 돌리면 H와 같은 사건에 <b>100%</b> 붙어요(코사인을 둘 다 h로 하는지 확인). H와 R은 모델이 달라 실제로는 55%만 같은 사건에 붙어요.</p>

<h2>4. Transformer 결과 (모델만, test)</h2>
{table(["", "Top-1", "AUC", "MRR", "nDCG@5", "best epoch (validation Top-1)"],
 [["V1 재실행", f"{TEST['V1']['top1']:.2f}%", f"{TEST['V1']['auc']:.4f}", f"{TEST['V1']['mrr']:.4f}", f"{TEST['V1']['ndcg']:.4f}", "1 (25.91%)"],
  ["★ H", f"{TEST['H']['top1']:.2f}%", f"{TEST['H']['auc']:.4f}", f"{TEST['H']['mrr']:.4f}", f"{TEST['H']['ndcg']:.4f}", "1 (27.58%)"],
  ["★ R", f"{TEST['R']['top1']:.2f}%", f"{TEST['R']['auc']:.4f}", f"{TEST['R']['mrr']:.4f}", f"{TEST['R']['ndcg']:.4f}", "2 (27.11%)"]])}
<p class="cap">test 122,573~122,680행 · seed 42 하나. V1 seed 사이 Top-1 흔들림이 약 0.4%p라, H · R의 +1.4%p는 그보다 커요. H와 R의 차이(0.16%p)는 그 안이에요. 모든 실험이 epoch 1~2에서 정점을 찍고 내려가요(빠른 과적합).</p>

<h2>5. 어디서 막히나 — 진단</h2>
<h3>5-1. test 정답의 98%가 새 기사(validation-only)</h3>
{table(["", "정답이 새 기사", "정답이 Train 기사", "후보 중 새 기사"],
 [["V1", "26.27% (119,930행)", "21.66% (2,738행)", "86%"], ["H", "27.69% (119,940행)", "19.60% (2,740행)", "86%"], ["R", "27.85% (119,861행)", "19.73% (2,712행)", "86%"]])}
<p class="cap">test는 사실상 <b>처음 보는 새 기사를 맞히는 문제</b>예요. 새 기사의 SID는 Train 사건에 붙어 물려받은 C2라서, 사건에 얼마나 잘 붙었는지가 성능을 좌우해요.</p>

<h3>5-2. 사건에 잘 붙은 새 기사일수록 잘 맞혀요</h3>
<div class="fig">{legend(["H", "R"])}{fig_cos}
<p class="cap">새 기사가 정답인 행의 Top-1을, 그 기사가 사건에 붙을 때의 코사인 크기로 나눴어요. 점선 = 무작위 20%. H에서 Jaccard로 붙은 기사(1,262행)는 <b>48.97%</b>까지 맞혀요.</p></div>
{table(["코사인", "H Top-1 (행)", "R Top-1 (행)"],
 [[c, f"{COS_TOP1['H'][i]:.2f}% ({COS_N['H'][i]:,})", f"{COS_TOP1['R'][i]:.2f}% ({COS_N['R'][i]:,})"] for i, c in enumerate(COS_BINS)])}
<p class="cap">코사인 0.35 미만으로 억지로 붙은 기사는 무작위 수준이거나 그보다 못해요. 다만 코사인이 낮은 기사가 원래 특이해서 어려운 것일 수도 있어, 인과관계까지는 단정하지 않아요.</p>

<h3>5-3. 인기 정보가 없어요</h3>
{table(["학습 없는 기준선 (같은 test 행)", "V1", "H", "R"],
 [["인기: 직전 24시간 클릭 수가 가장 많은 후보", "42.94%", "42.93%", "42.97%"],
  ["최신: 가장 최근에 발행된 후보", "18.61%", "18.59%", "18.57%"], ["무작위", "20.00%", "20.00%", "20.00%"]])}
<p class="cap">클릭 수는 노출 시각 <b>이전</b>만 셌어요(미래 정보 없음). 모델(27%대)보다 인기만 쓴 쪽이 15%p 높아요. SID에는 "무슨 내용인지"만 있고 "지금 얼마나 읽히는지"는 없어요.</p>

<h2>6. 인기·나이 결합 (재학습 없음)</h2>
<p>최종 점수 = 모델 점수 + λ · log(1 + 직전 24시간 클릭 수) + μ · log(1 + 기사 나이(시간)). λ, μ는 <b>validation 앞 절반</b>에서 고르고(λ 13개 × μ 10개) test에는 한 번만 적용했어요. 세 실험 모두 <b>λ = 5, μ = −1</b>을 골랐어요.</p>
<div class="fig">{legend(["V1", "H", "R"])}{fig_lam}
<p class="cap">validation Top-1 vs λ (μ = 0). 꼭대기가 λ = 3~10 사이에서 평평해요.</p></div>
<div class="fig">{legend(["V1", "H", "R"])}{fig_main}
<p class="cap">test Top-1. 점선 = 무작위 20%.</p></div>
{table(["test Top-1", "모델만", "인기만", "결합 (λ=5)", "결합 + 나이", "결합 + 나이 − 인기만"],
 [[n, f"{FUSION[n][0]:.2f}%", f"{FUSION[n][1]:.2f}%", f"{FUSION[n][2]:.2f}%", f"<b>{FUSION[n][3]:.2f}%</b>", f"<b>+{FUSION[n][3]-FUSION[n][1]:.2f}%p</b>"] for n in ["V1", "H", "R"]])}
{table(["결합 + 나이 (test)", "AUC", "MRR", "nDCG@5"],
 [[n, f"{FUSION_FULL[n][0]:.4f}", f"{FUSION_FULL[n][1]:.4f}", f"{FUSION_FULL[n][2]:.4f}"] for n in ["V1", "H", "R"]])}
<p class="cap"><b>읽는 법:</b> 크게 오른 건 인기 자체의 힘이에요. 봐야 할 건 마지막 열 — SID가 인기 위에 더 보태는 정보예요. 새 SID(H, R)는 +1.3%p, 예전 SID(V1)는 +0.5%p. 나이는 +0.1~0.25%p로 작지만 방향은 일관돼요.</p>

<h2>7. 다음 단계</h2>
{table(["", "내용", "상태"],
 [["seed 123 · 2026", "V1 · H · R Transformer를 seed 2개 더 → 차이가 우연이 아닌지 확인", "진행 중"],
  ["인기·나이 함께 학습", "후보 점수 = SID 점수 + w1·logpop + w2·logage, w를 같이 학습(시작값 5, −1). Transformer가 인기로 설명 안 되는 개인 취향에 집중하도록", "여민 코드 수정 요청함, 입력 파일 준비 완료"],
  ["새 기사 SID 배정 개선", "코사인이 기준값보다 낮으면 사건 C2를 물려받지 않고 기사 자기 r1로 C2를 정하기", "제안"],
  ["R 초기화 보완", "Q2 초기화를 몇 epoch 학습 뒤에 하거나, r1을 정규화해 카테고리 쏠림 줄이기", "제안"]], num_from=99)}
<div class="note"><b>주의</b><ul>
<li>Transformer seed 42 하나의 결과예요. seed 123 · 2026 결과로 보완할 예정이에요.</li>
<li>validation 앞 절반을 best epoch 고르기와 λ · μ 고르기에 두 번 썼어요. 고르는 값이 둘뿐이고 세 실험이 같은 값을 골라 영향은 작다고 봐요.</li>
<li>클릭 기록이 train 노출 기간(5/18)부터라 train 첫 24시간 행의 클릭 수는 적게 세져요. test에는 해당 없어요.</li>
</ul></div>
</main></body></html>
"""
OUT.write_text(html, encoding="utf-8")
print(OUT)
