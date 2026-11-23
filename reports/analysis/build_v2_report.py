"""v2 재실험 보고서 HTML 생성 (Transformer seed 42 · 123 · 2026). 숫자는 AWS 실행 결과를 그대로 옮긴 것.

    python reports/analysis/build_v2_report.py   →  reports/v2_재실험_결과_3seed.html
    (seed 42 하나로 쓴 이전 판은 reports/v2_재실험_결과_seed42.html 에 그대로 둔다)

출처 (AWS, ~/minji/SID_Project/data/output/fixedcluster_v1/v2/)
  rqvae/*/summary.json, sid/{h,r}/check_sid.txt, transformer/*/test_metrics.json,
  analysis/test_breakdown_seed42.json, analysis/popularity_fusion_seed{42,123,2026}.json
  진단(5장)과 λ 곡선(6장 그림)은 seed 42 결과
"""
from statistics import mean, stdev
from pathlib import Path
from html import escape

OUT = Path(__file__).resolve().parents[1] / "v2_재실험_결과_3seed.html"
COL = {"V1": "var(--s-v1)", "H": "var(--s-h)", "R": "var(--s-r)", "인기만": "var(--s-pop)"}

# ---------------- 데이터 ----------------
SEEDS = [42, 123, 2026]
RUNS = ["V1", "H", "R"]
TEST_S = {  # 모델만 test: seed → (top1, auc, mrr, ndcg)   (collect_results.py)
    "V1": {42: (26.16, 0.5686, 0.5101, 0.6308), 123: (26.79, 0.5831, 0.5187, 0.6375), 2026: (26.76, 0.5789, 0.5168, 0.6360)},
    "H": {42: (27.51, 0.5836, 0.5223, 0.6401), 123: (27.81, 0.5810, 0.5224, 0.6401), 2026: (27.70, 0.5889, 0.5252, 0.6424)},
    "R": {42: (27.67, 0.5888, 0.5249, 0.6422), 123: (27.17, 0.5810, 0.5197, 0.6381), 2026: (27.69, 0.5852, 0.5237, 0.6412)},
}
FUSION_S = {  # test Top-1: seed → (모델만, 인기만, 결합 λ=5, 결합+나이, 고른 μ)   (popularity_fusion.py --with-age)
    "V1": {42: (26.16, 42.94, 43.23, 43.41, -1), 123: (26.79, 42.94, 43.51, 43.73, -1), 2026: (26.76, 42.94, 43.59, 43.84, -2)},
    "H": {42: (27.51, 42.93, 43.96, 44.21, -1), 123: (27.81, 42.93, 43.80, 44.14, -1), 2026: (27.70, 42.93, 44.14, 44.37, -1)},
    "R": {42: (27.67, 42.97, 44.10, 44.23, -1), 123: (27.17, 42.97, 44.02, 44.35, -1), 2026: (27.69, 42.97, 43.91, 44.09, -2)},
}
FUSION_FULL_S = {  # 결합+나이 test: seed → (auc, mrr, ndcg)
    "V1": {42: (0.7394, 0.6546, 0.7413), 123: (0.7407, 0.6565, 0.7428), 2026: (0.7398, 0.6566, 0.7428)},
    "H": {42: (0.7418, 0.6589, 0.7445), 123: (0.7419, 0.6588, 0.7445), 2026: (0.7431, 0.6602, 0.7455)},
    "R": {42: (0.7438, 0.6599, 0.7454), 123: (0.7436, 0.6604, 0.7457), 2026: (0.7421, 0.6587, 0.7444)},
}


def ms(vals, f="{:.2f}"):
    """평균 ± 표준편차(seed 3개, 표본 표준편차) 문자열."""
    return f"{f.format(mean(vals))} ± {f.format(stdev(vals))}"


TEST = {n: [mean(TEST_S[n][s][k] for s in SEEDS) for k in range(4)] for n in RUNS}           # 평균
FUSION = {n: [mean(FUSION_S[n][s][k] for s in SEEDS) for k in range(4)] for n in RUNS}       # 평균
GAIN = {n: [FUSION_S[n][s][3] - FUSION_S[n][s][1] for s in SEEDS] for n in RUNS}            # 결합+나이 − 인기만
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
# 인기·나이 함께 학습 (transformer_feat/<run>_seed42_flr0, 가중치 5 · −1 고정): test (top1, auc, mrr, ndcg), validation best
JOINT = {"V1": (42.50, 0.7342, 0.6485, 0.7368, 42.66), "H": (42.92, 0.7366, 0.6513, 0.7389, 42.68), "R": (43.07, 0.7363, 0.6519, 0.7392, 42.59)}
JOINT_H_LEARN = [41.96, 41.21, 41.10]   # H, 가중치도 학습: 학습 전 / epoch 1 / epoch 2 validation (가중치 5 → 0.78, −1 → −0.09)
JOINT_H_FIXED = [41.96, 42.68, 42.24, 42.20, 42.07, 41.68, 41.67]   # H, 가중치 고정: 학습 전, epoch 1~6 validation
# 공개 TIGER 재현 구현 (tiger/<run>_seed42): test (top1, mrr, ndcg, auc), c1 만으로 채점한 Top-1, 정답·오답 평균 log P
TIGER = {"V1": (16.69, 0.4332, 0.5720, 0.4784, 23.12, -17.05, -16.66), "H": (17.30, 0.4378, 0.5754, 0.4829, 22.05, -17.12, -16.82),
         "R": (17.77, 0.4371, 0.5746, 0.4750, 22.43, -15.81, -15.53)}


# ---------------- 그림 함수 ----------------
def bars(cats, series, ymax, unit="%", ref=None, ref_label="", h=260, fmt="{:.2f}", ns=None, pts=None):
    """묶은 세로 막대. series: [(이름, 값 리스트)]. pts: {이름: [[seed별 값], ...]} 이면 막대 위에 seed 점을 찍는다."""
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
            if pts:
                for sv in pts[name][gi]:
                    s.append(f'<circle cx="{x+bw/2:.1f}" cy="{y(sv):.1f}" r="2.6" class="seed"/>')
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
                [(n, [FUSION[n][0], FUSION[n][1], FUSION[n][3]]) for n in RUNS],
                50, ref=20, ref_label="무작위 20%")
fig_model = bars(["seed 42", "seed 123", "seed 2026", "평균"],
                 [(n, [TEST_S[n][s][0] for s in SEEDS] + [TEST[n][0]]) for n in RUNS], 30, h=240)
fig_joint = bars(["인기만", "함께 학습 (고정)", "나중에 더하기"],
                 [(n, [FUSION_S[n][42][1], JOINT[n][0], FUSION_S[n][42][3]]) for n in RUNS], 50, ref=20, ref_label="무작위 20%")
fig_tiger = bars(["공개 TIGER (생성 학습)", "우리 Transformer (후보 고르기)", "+ 인기·나이"],
                 [(n, [TIGER[n][0], TEST_S[n][42][0], FUSION_S[n][42][3]]) for n in RUNS], 50, ref=20, ref_label="무작위 20%")
fig_gain = bars(["seed 42", "seed 123", "seed 2026", "평균"],
                [(n, GAIN[n] + [mean(GAIN[n])]) for n in RUNS], 1.6, unit="%p", h=240)
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
.seed {{ fill:var(--surface); stroke:var(--text); stroke-width:1.4; }}
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
<p class="sub">EB-NeRD small · 여민 코드(RQ-VAE·SID 생성·Transformer) + 우리 설정(RQ-VAE UNI, Transformer V1) · Transformer seed 42 · 123 · 2026 · 2026-11-23</p>

<div class="box"><b>한 줄 요약</b>
<ul>
<li><b>새 전처리 + UNI 재학습으로 만든 SID(H, R)가 예전 SID(V1)보다 좋아요.</b> Transformer seed 3개 평균 test Top-1 {ms([TEST_S["V1"][s][0] for s in SEEDS])}% → {ms([TEST_S["H"][s][0] for s in SEEDS])}%(H) · {ms([TEST_S["R"][s][0] for s in SEEDS])}%(R). 차이(약 1%p)가 seed 사이 흔들림(표준편차 0.15~0.36%p)보다 커요.</li>
<li><b>H와 R은 Transformer 성능이 같다고 봐야 해요.</b> 평균 차이 0.16%p로 흔들림 안이고, seed마다 앞서는 쪽이 바뀌어요. 다만 R은 Q2 코드를 128개 중 59~93개만 써요(H는 128개 모두).</li>
<li><b>병목은 두 곳이에요.</b> ① test에서 맞혀야 할 정답의 98%가 validation-only 기사인데, SID 생성 때 비슷하지 않은 사건에 억지로 붙은 기사는 거의 못 맞혀요. ② 인기 정보가 없어요. 직전 24시간 클릭 수만으로 42.9%를 맞혀요.</li>
<li><b>재학습 없이 인기·나이를 점수에 더하면 44.2%</b>(H·R, seed 3개 평균). SID가 인기 위에 보태는 몫은 H +{mean(GAIN["H"]):.2f}%p · R +{mean(GAIN["R"]):.2f}%p로 예전 SID(+{mean(GAIN["V1"]):.2f}%p)의 약 1.8배이고, seed 3개 모두 H·R이 V1보다 커요(범위가 겹치지 않음).</li>
<li><b>인기·나이를 학습 때 함께 넣으면 오히려 나빠요</b>(42.5~43.1%, 7장). 인기·나이는 나중에 더하는 지금 방식을 유지해요.</li>
<li><b>공개 TIGER 재현 구현(생성형 학습만)은 같은 SID로 16.7~17.8%</b>로 무작위보다 낮아요(8장). 새 기사가 대부분인 이 데이터에선 후보 중 정답을 고르는 학습이 필요해요.</li>
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

<h2>4. Transformer 결과 (모델만, test · seed 3개)</h2>
{table(["seed 3개 평균 ± 표준편차", "Top-1 (%)", "AUC", "MRR", "nDCG@5"],
 [[("★ " if n != "V1" else "") + ("V1 재실행" if n == "V1" else n),
   ms([TEST_S[n][s][0] for s in SEEDS]), ms([TEST_S[n][s][1] for s in SEEDS], "{:.4f}"),
   ms([TEST_S[n][s][2] for s in SEEDS], "{:.4f}"), ms([TEST_S[n][s][3] for s in SEEDS], "{:.4f}")] for n in RUNS])}
<div class="fig">{legend(RUNS)}{fig_model}
<p class="cap">test Top-1 (모델만). 세 seed 모두 H · R이 V1보다 높아요(H − V1 = +1.35 · +1.02 · +0.94%p).</p></div>
{table(["test Top-1", "seed 42", "seed 123", "seed 2026"],
 [[n] + [f"{TEST_S[n][s][0]:.2f}%" for s in SEEDS] for n in RUNS])}
<p class="cap">test 122,573~122,680행. SID는 셋 다 RQ-VAE seed 42 모델로 만든 같은 파일이고, seed는 Transformer만 바꿨어요. H · R과 V1의 차이(약 1%p)는 표준편차보다 커서 우연으로 보기 어려워요. H와 R의 차이(0.16%p)는 표준편차 안이에요. seed 42에서는 모든 실험이 epoch 1~2에서 정점을 찍고 내려갔어요(빠른 과적합).</p>

<h2>5. 어디서 막히나 — 진단</h2>
<p class="cap">이 장은 Transformer seed 42 결과로 나눴어요.</p>
<p>4번의 Top-1을 잰 <b>test 행</b>(validation 기간 노출의 뒤 절반, 약 12만 2천 행)을 다시 나눠 봤어요. 행마다 후보 5개 중 정답 기사가 하나 있는데, <b>그 정답 기사가 어떤 기사인지</b>에 따라 Transformer가 얼마나 맞혔는지를 봐요. 기사의 SID는 그보다 앞선 SID 생성 단계에서 이미 정해진 값이에요.</p>
<h3>5-1. test 행의 정답 기사는 98%가 validation-only 기사</h3>
{table(["", "정답이 validation-only 기사일 때 Top-1", "정답이 Train 기사일 때 Top-1", "후보 중 validation-only 기사"],
 [["V1", "26.27% (119,930행)", "21.66% (2,738행)", "86%"], ["H", "27.69% (119,940행)", "19.60% (2,740행)", "86%"], ["R", "27.85% (119,861행)", "19.73% (2,712행)", "86%"]])}
<p class="cap">Transformer가 test에서 맞혀야 하는 정답은 거의 다 <b>validation-only 기사</b>(Train 기간에 안 나온 새 기사)예요. 이 기사들의 c2는 SID 생성 단계에서 <b>Train 사건에 붙어 물려받은 C2</b>라서, 그때 사건에 얼마나 잘 붙었는지가 성능을 좌우할 수 있어요.</p>

<h3>5-2. 정답 기사가 비슷한 사건에 붙었을수록 잘 맞혀요</h3>
<p>validation-only 기사는 SID 생성 때 <b>전부</b> 어떤 Train 사건에 붙어요(72시간 안에서 가장 비슷한 사건, 기준값 없음). 다만 얼마나 비슷한 사건에 붙었는지는 기사마다 달라요. 정답이 validation-only 기사인 test 행을, 그 정답 기사가 <b>사건에 붙을 때의 코사인</b>으로 나눠 Top-1을 봤어요.</p>
<div class="fig">{legend(["H", "R"])}{fig_cos}
<p class="cap">test 행 중 정답이 validation-only 기사인 행의 Top-1 (가로축 = 그 정답 기사가 사건에 붙을 때의 코사인). 점선 = 무작위 20%. H에서 개체가 겹치는 사건에 Jaccard로 붙은 기사가 정답인 행(1,262행)은 <b>48.97%</b>까지 맞혀요.</p></div>
{table(["정답 기사가 붙을 때 코사인", "H Top-1 (test 행 수)", "R Top-1 (test 행 수)"],
 [[c, f"{COS_TOP1['H'][i]:.2f}% ({COS_N['H'][i]:,})", f"{COS_TOP1['R'][i]:.2f}% ({COS_N['R'][i]:,})"] for i, c in enumerate(COS_BINS)])}
<p class="cap">비슷한 사건이 없어 코사인 0.35 미만으로 억지로 붙은 기사가 정답이면, Transformer는 무작위 수준이거나 그보다 못 맞혀요(물려받은 C2가 기사 내용과 맞지 않기 때문으로 보여요). 다만 코사인이 낮은 기사가 원래 특이해서 어려운 것일 수도 있어, 인과관계까지는 단정하지 않아요.</p>

<h3>5-3. 인기 정보가 없어요</h3>
{table(["학습 없는 기준선 (같은 test 행)", "V1", "H", "R"],
 [["인기: 직전 24시간 클릭 수가 가장 많은 후보", "42.94%", "42.93%", "42.97%"],
  ["최신: 가장 최근에 발행된 후보", "18.61%", "18.59%", "18.57%"], ["무작위", "20.00%", "20.00%", "20.00%"]])}
<p class="cap">클릭 수는 노출 시각 <b>이전</b>만 셌어요(미래 정보 없음). 모델(27%대)보다 인기만 쓴 쪽이 15%p 높아요. SID에는 "무슨 내용인지"만 있고 "지금 얼마나 읽히는지"는 없어요.</p>

<h2>6. 인기·나이 결합 (재학습 없음)</h2>
<p>최종 점수 = 모델 점수 + λ · log(1 + 직전 24시간 클릭 수) + μ · log(1 + 기사 나이(시간)). λ, μ는 <b>validation 앞 절반</b>에서 고르고(λ 13개 × μ 10개) test에는 한 번만 적용했어요. 9번(실험 3개 × seed 3개) 모두 <b>λ = 5</b>를 골랐고, μ는 −1(7번) 또는 −2(V1 · R seed 2026)였어요.</p>
<div class="fig">{legend(["V1", "H", "R"])}{fig_lam}
<p class="cap">validation Top-1 vs λ (μ = 0, seed 42). 꼭대기가 λ = 3~10 사이에서 평평해요. seed 123 · 2026도 모양이 같아요.</p></div>
<div class="fig">{legend(["V1", "H", "R"])}{fig_main}
<p class="cap">test Top-1, seed 3개 평균. 점선 = 무작위 20%. seed 사이 차이는 0.5%p 안쪽이라 이 축에서는 안 보여서, 아래 표와 그림에 seed별로 따로 적었어요.</p></div>
{table(["test Top-1 (seed 3개 평균)", "모델만", "인기만", "결합 (λ=5)", "결합 + 나이", "결합 + 나이 − 인기만"],
 [[n, f"{FUSION[n][0]:.2f}%", f"{FUSION[n][1]:.2f}%", f"{FUSION[n][2]:.2f}%", f"<b>{FUSION[n][3]:.2f}%</b>", f"<b>+{mean(GAIN[n]):.2f}%p</b>"] for n in RUNS])}
<h3>SID가 인기 위에 보태는 몫 (결합 + 나이 − 인기만)</h3>
<div class="fig">{legend(RUNS)}{fig_gain}
<p class="cap">seed별 값과 평균. V1은 가장 높은 seed도 +{max(GAIN["V1"]):.2f}%p, H · R은 가장 낮은 seed도 +{min(GAIN["H"] + GAIN["R"]):.2f}%p라 범위가 겹치지 않아요.</p></div>
{table(["결합 + 나이 − 인기만", "seed 42", "seed 123", "seed 2026", "평균 ± 표준편차"],
 [[n] + [f"+{g:.2f}%p" for g in GAIN[n]] + [f"+{ms(GAIN[n])}%p"] for n in RUNS])}
{table(["결합 + 나이 (test, seed 3개 평균 ± 표준편차)", "AUC", "MRR", "nDCG@5"],
 [[n] + [ms([FUSION_FULL_S[n][s][k] for s in SEEDS], "{:.4f}") for k in range(3)] for n in RUNS])}
<p class="cap"><b>읽는 법:</b> 크게 오른 건 인기 자체의 힘이에요. 봐야 할 건 SID가 인기 위에 더 보태는 몫이에요. 새 SID(H, R)는 +1.3%p, 예전 SID(V1)는 +0.7%p. H와 R은 여기서도 차이가 없어요. 나이는 +0.1~0.35%p로 작지만 9번 모두 같은 방향이에요.</p>

<h2>7. 인기·나이를 학습 때 함께 넣으면? — 더 나빠져요</h2>
<p>후보 점수 = SID 점수 + w1 · log(1 + 직전 24시간 클릭 수) + w2 · log(1 + 기사 나이)로 두고 Transformer를 처음부터 학습했어요(여민 코드에 우리 패치 적용, seed 42). 손실은 그대로 후보 5개 softmax → cross entropy예요.</p>
{table(["H, validation Top-1", "학습 전", "epoch 1", "epoch 2", "epoch 3", "epoch 4", "epoch 5", "epoch 6"],
 [["w도 같이 학습 (시작 5, −1)"] + [f"{v:.2f}%" for v in JOINT_H_LEARN] + ["—"] * 4,
  ["w를 5, −1로 고정"] + [f"{v:.2f}%" for v in JOINT_H_FIXED]])}
<p class="cap">w를 같이 학습하면 epoch 1 만에 인기 가중치가 5 → 0.78, 나이 −1 → −0.09로 줄고 성능이 학습 전보다 떨어졌어요(중단). 고정해도 epoch 1이 정점(42.68%)이고, 나중에 더하기의 validation(43.27% 이상)에 못 미쳐요.</p>
<div class="fig">{legend(RUNS)}{fig_joint}
<p class="cap">test Top-1, seed 42. 점선 = 무작위 20%.</p></div>
{table(["test (seed 42)", "인기만", "함께 학습 (고정) Top-1", "AUC", "MRR", "nDCG@5", "나중에 더하기 Top-1"],
 [[n, f"{FUSION_S[n][42][1]:.2f}%", f"<b>{JOINT[n][0]:.2f}%</b> ({JOINT[n][0]-FUSION_S[n][42][1]:+.2f}%p)", f"{JOINT[n][1]:.4f}", f"{JOINT[n][2]:.4f}", f"{JOINT[n][3]:.4f}",
   f"<b>{FUSION_S[n][42][3]:.2f}%</b> ({FUSION_S[n][42][3]-FUSION_S[n][42][1]:+.2f}%p)"] for n in RUNS])}
<p class="cap"><b>읽는 법:</b> 괄호는 인기만 대비 SID가 보탠 몫이에요. 함께 학습하면 그 몫이 −0.4 ~ +0.1%p로 사라지고, 나중에 더하면 +0.5 ~ +1.3%p가 남아요. <b>원인으로 보이는 것:</b> SID만 학습하면 "어떤 사건·주제가 읽히나" 같은 넓은 패턴을 배우고, 이게 새 기사에도 통해요. 인기를 같이 넣으면 그 넓은 패턴은 인기가 대신 설명하고, SID에는 train 기사를 외우는 일만 남아요. 그런데 test 정답은 98%가 처음 보는 기사라 외운 게 쓸모가 없어요. 같은 이유로 w를 train에서 학습하면 인기 비중을 작게 배워요. 그래서 <b>w는 test와 비슷한 validation에서 정하는 지금 방식(나중에 더하기)</b>을 유지해요.</p>

<h2>8. 공개 TIGER 재현 구현과 비교</h2>
<p>공개된 비공식 TIGER 재현 구현(EdoardoBotta/RQ-VAE-Recommender @ 957d32b, MIT)의 모델 · 손실 · 학습 설정(d384 · 6 head · 4층, AdamW 1e-3, batch 640, history 20)을 수정 없이 쓰고, 데이터 로더와 평가만 우리 데이터에 맞췄어요. 사용자 history로 <b>다음 클릭 기사의 SID를 생성</b>하도록 학습하고(오답 없음), test는 우리와 같이 후보 5개를 log P(c1) + log P(c2|c1) + log P(c3|c1,c2)로 채점해요. 저자들의 공식 코드는 아니에요.</p>
<div class="fig">{legend(RUNS)}{fig_tiger}
<p class="cap">test Top-1, seed 42. 점선 = 무작위 20%.</p></div>
{table(["공개 TIGER (test, seed 42)", "Top-1", "AUC", "MRR", "nDCG@5", "c1만으로 채점 Top-1", "정답 / 오답 평균 log P"],
 [[n, f"<b>{TIGER[n][0]:.2f}%</b>", f"{TIGER[n][3]:.4f}", f"{TIGER[n][1]:.4f}", f"{TIGER[n][2]:.4f}", f"{TIGER[n][4]:.2f}%", f"{TIGER[n][5]:.2f} / {TIGER[n][6]:.2f}"] for n in RUNS])}
<p class="cap"><b>읽는 법:</b> 세 SID 모두 무작위(20%)보다 낮고 AUC가 0.5 아래예요. 정답의 평균 log P가 오답보다 낮아요. TIGER는 train에서 클릭된 기사의 SID를 생성하도록 배우는데, test 정답은 98%가 처음 보는 기사라서 오히려 오답 쪽에 높은 확률을 주는 것으로 보여요(오답에는 train에서 본 기사가 정답보다 많이 섞여 있어요). c1만으로 채점하면 22~23%로 무작위보다 낫지만, c3까지 더하면 떨어져요. 같은 SID로 <b>후보 중 정답 고르기</b>를 학습한 우리 Transformer는 26.2~27.7%예요. 새 기사가 대부분인 이 데이터에서는 생성형 학습만으로는 부족하고, 오답과 비교하는 학습이 필요하다는 근거예요.</p>

<h2>9. 다음 단계</h2>
{table(["", "내용", "상태"],
 [["seed 123 · 2026", "V1 · H · R Transformer를 seed 2개 더 → 차이가 우연이 아닌지 확인", "완료 (이 보고서)"],
  ["인기·나이 함께 학습", "점수에 더해서 함께 학습 → 나중에 더하기보다 낮음 (7장)", "완료"],
  ["공개 TIGER 비교", "생성형 학습만 하는 공개 재현 구현 → 무작위보다 낮음 (8장)", "완료"],
  ["인기를 Transformer 입력으로", "점수에 더하는 대신 후보 인기 · 나이를 임베딩으로 넣어, 사용자마다 다른 인기 민감도를 배우게 하기", "제안"],
  ["새 기사 SID 배정 개선", "코사인이 기준값보다 낮으면 사건 C2를 물려받지 않고 기사 자기 r1로 C2를 정하기", "제안"],
  ["R 초기화 보완", "Q2 초기화를 몇 epoch 학습 뒤에 하거나, r1을 정규화해 카테고리 쏠림 줄이기", "제안"]], num_from=99)}
<div class="note"><b>주의</b><ul>
<li>seed는 Transformer만 바꿨어요. SID는 RQ-VAE seed 42 모델 하나로 만든 것이라, RQ-VAE seed에 따른 흔들림은 이 보고서에 들어 있지 않아요.</li>
<li>5장 진단과 6장 λ 곡선은 seed 42 결과예요.</li>
<li>validation 앞 절반을 best epoch 고르기와 λ · μ 고르기에 두 번 썼어요. 고르는 값이 둘뿐이고 λ는 9번 모두 같은 값(5)을 골라 영향은 작다고 봐요.</li>
<li>클릭 기록이 train 노출 기간(5/18)부터라 train 첫 24시간 행의 클릭 수는 적게 세져요. test에는 해당 없어요.</li>
</ul></div>
</main></body></html>
"""
OUT.write_text(html, encoding="utf-8")
print(OUT)
