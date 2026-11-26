"""인기도 · 최신성을 어떻게 계산하고 점수에 넣었는지 정리한 HTML (그림 포함).

    python reports/analysis/build_popularity_doc.py   →  reports/인기도_최신성_정리.html

숫자 출처
  - 장부 · 예시 노출: data/raw (EB-NeRD small) 를 experiments/v2_yeomin_rerun/analyze_test.load_clicks 로 직접 계산
    (예시 노출 impression_id 184583062, 정답 9788095 의 클릭 시각은 아래 CLICK_MIN 에 그대로 넣음)
  - λ 곡선 · 결과: AWS popularity_fusion.py / collect_results.py / 8_transformer_features.sh 결과 (v2 보고서와 같은 값)
  - SID 점수 s 는 학습된 모델이 AWS 에만 있어서 예시 값 (문서에 '예시'라고 표시)
"""
from html import escape
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "인기도_최신성_정리.html"
COL = {"V1": "#1baf7a", "H": "#2a78d6", "R": "#eb6834"}
BLUE, GRAY, RED, INK, MUTED, LINE = "#2a78d6", "#9aa3b2", "#d45656", "#14181f", "#5b6577", "#e3e6eb"

# T = 2023-05-30 10:40:07 (예시 노출 184583062). 아래 두 목록은 T 기준 분
CLICK_MIN = [-47.23, -46.58, -45.63, -45.35, -44.98, -44.67, -44.25, -43.85, -43.52, -42.07, -41.65, -40.13, -39.48, -38.33, -35.15, -31.03, -30.47, -30.35, -29.78, -28.2, -28.12, -28.12, -27.12, -26.72, -25.98, -25.67, -25.27, -24.8, -23.93, -23.52, -23.47, -21.63, -21.55, -21.02, -20.42, -20.38, -19.6, -19.17, -19.05, -18.0, -17.85, -17.8, -17.08, -15.55, -15.5, -15.5, -15.22, -15.17, -14.87, -14.53, -14.47, -12.57, -11.85, -11.85, -11.28, -9.82, -8.88, -8.73, -8.5, -7.82, -7.08, -7.07, -6.8, -5.82, -5.72, -5.6, -5.43, -4.92, -4.18, -3.05, -2.42, -2.23, -2.15, -0.63, -0.15]
AFTER_MIN = [0.18, 1.13, 1.72, 1.98, 2.8, 3.13, 3.97, 4.97, 5.33, 6.03, 6.63, 6.9, 7.25, 7.68, 8.25, 8.48, 8.58, 8.73, 10.95, 11.83, 12.73, 13.73, 14.62, 15.62, 16.67, 18.52, 18.75, 19.42, 19.63, 20.02, 20.78, 22.47, 23.05, 23.65, 24.13, 25.2, 25.77, 27.28, 27.6, 29.0, 29.55, 30.4, 30.53, 31.17, 31.53, 32.18, 32.58, 33.43, 34.6, 34.73, 35.52, 35.88, 36.35, 37.88, 38.07, 39.22, 39.32, 42.1, 44.73, 46.05, 46.88, 47.8, 48.18, 49.72, 51.22, 55.77, 56.63, 56.7, 56.97, 58.4, 59.02, 59.7, 61.33, 61.68, 62.08, 62.47, 62.63, 64.78, 66.17, 66.45, 67.08, 67.5, 67.7, 73.43, 75.68, 77.32, 80.03, 80.87, 80.92, 81.48, 85.55, 86.15, 86.82, 88.0, 88.05, 90.77, 91.5, 94.33, 96.0, 96.27, 97.93, 98.45, 99.87, 100.32, 104.83, 108.25, 112.73, 114.35, 117.13]
# 정답 9788095 의 실제 장부 기록 (T 기준 분): 창 안 75건, T와 같은 시각 1건, T 이후 2시간 109건
AFTER_N, AT_T_N, INSIDE_N = 109, 1, 75

CANDS = [  # id, 정답, 24시간 클릭, logpop, 발행 후 (글), logage, s(예시)
    ("8558684", False, 0, 0.00, "약 2년", 9.74, -9.5),
    ("9788095", True, 75, 4.33, "1시간", 0.70, -10.5),
    ("9428643", False, 51, 3.95, "약 8개월", 8.64, -8.8),
    ("9529776", False, 12, 2.56, "약 6개월", 8.38, -9.9),
    ("9080070", False, 4, 1.61, "약 1.3년", 9.37, -11.2),
]
LAM, MU = 5, -1
LAMS = ["0", "0.05", "0.1", "0.2", "0.3", "0.5", "0.75", "1", "1.5", "2", "3", "5", "10"]
VAL_H = [27.58, 28.84, 30.01, 32.16, 33.89, 36.65, 38.93, 40.33, 41.91, 42.67, 43.14, 43.27, 43.00]
RESULT = {"V1": (26.57, 42.94, 43.66), "H": (27.67, 42.93, 44.24), "R": (27.51, 42.97, 44.22)}   # seed 3개 평균
JOINT = {"V1": (42.94, 42.50, 43.41), "H": (42.93, 42.92, 44.21), "R": (42.97, 43.07, 44.23)}    # seed 42: 인기만/함께/나중에


def svg(w, h, body, label):
    return f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" aria-label="{escape(label)}">{body}</svg>'


def t(x, y, s, cls="lbl", anchor="start", extra=""):
    return f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" text-anchor="{anchor}" {extra}>{escape(str(s))}</text>'


# ── 그림 1. 장부 만들기 ──────────────────────────────────────
def fig_ledger():
    b = []
    rows = [("사용자 X", "05-29 21:05", "클릭: 9788095 (예)"),
            ("사용자 Y", "05-30 09:58", "클릭: 9788095, 9428643"),
            ("사용자 U", "05-30 10:40", "클릭: 9788095")]
    b.append(t(20, 22, "behaviors (모든 노출 · 모든 사용자)", "head"))
    for i, (u, tm, c) in enumerate(rows):
        y = 40 + i * 62
        b.append(f'<rect x="20" y="{y}" width="290" height="50" rx="8" class="box"/>')
        b.append(t(34, y + 21, f"{u} · {tm}", "lbl b"))
        b.append(t(34, y + 40, c, "lbl m"))
    b.append(f'<path d="M322 125 H392" class="arr" marker-end="url(#ah)"/>')
    b.append(t(357, 112, "클릭만 뽑기", "lbl m", "middle"))
    b.append(t(410, 22, "클릭 기록 장부 (기사, 노출 시각)", "head"))
    recs = ["(9788095, 05-29 21:05)", "(9788095, 05-30 09:58)", "(9428643, 05-30 09:58)", "(9788095, 05-30 10:40)", "…  총 480,566건"]
    for i, r in enumerate(recs):
        y = 40 + i * 37
        cls = "box hl" if i < 4 else "box ghost"
        b.append(f'<rect x="410" y="{y}" width="250" height="30" rx="6" class="{cls}"/>')
        b.append(t(424, y + 20, r, "lbl"))
    b.append(t(20, 238, "노출만 되고 클릭 안 된 기사는 적지 않는다 · 클릭 시각은 그 노출의 시각으로 대신한다", "lbl m"))
    return svg(680, 250, b and "".join(b), "behaviors에서 클릭만 뽑아 장부를 만드는 그림")


# ── 그림 2. T 직전 24시간 창 + 실제 클릭 ─────────────────────
def fig_window():
    W, L, R_ = 760, 40, 720
    b = []
    # 윗줄: 개념
    y0 = 60
    X = lambda h: L + (h + 26) / 29 * (R_ - L)      # −26h ~ +3h
    b.append(t(L, 22, "① 세는 구간: 노출 시각 T 직전 24시간", "head"))
    b.append(f'<rect x="{X(-24):.1f}" y="{y0 - 18}" width="{X(0) - X(-24):.1f}" height="36" rx="6" class="band"/>')
    b.append(t((X(-24) + X(0)) / 2, y0 + 5, "[T − 24시간, T) 안의 클릭만 센다", "lbl b blue", "middle"))
    b.append(f'<line x1="{L}" x2="{R_}" y1="{y0 + 30}" y2="{y0 + 30}" class="axis"/>')
    for h, lab in [(-24, "T − 24시간\n05-29 10:40"), (0, "T = 노출 시각\n05-30 10:40")]:
        b.append(f'<line x1="{X(h):.1f}" x2="{X(h):.1f}" y1="{y0 + 24}" y2="{y0 + 38}" class="axis"/>')
        for k, part in enumerate(lab.split("\n")):
            b.append(t(X(h), y0 + 54 + k * 16, part, "lbl m", "middle"))
    b.append(f'<rect x="{X(0) + 2:.1f}" y="{y0 - 18}" width="{X(3) - X(0) - 2:.1f}" height="36" rx="6" class="band red"/>')
    b.append(t(X(1.5), y0 + 5, "제외", "lbl b red", "middle"))
    # 아랫줄: 실제 클릭 (확대)
    y1 = 226
    Z = lambda m: L + (m + 75) / 201 * (R_ - L)     # −75분 ~ +126분
    b.append(t(L, y1 - 66, "② 실제: 정답 기사 9788095 의 장부 기록 (T 주변 확대)", "head"))
    b.append(f'<rect x="{Z(-75):.1f}" y="{y1 - 34}" width="{Z(0) - Z(-75):.1f}" height="68" rx="6" class="band"/>')
    b.append(f'<rect x="{Z(0) + 2:.1f}" y="{y1 - 34}" width="{Z(126) - Z(0) - 2:.1f}" height="68" rx="6" class="band red"/>')
    for i, m in enumerate(sorted(CLICK_MIN)):
        jy = ((i * 37) % 9 - 4) * 6
        b.append(f'<circle cx="{Z(m):.1f}" cy="{y1 + jy:.1f}" r="3.2" fill="{BLUE}"/>')
    for i, m in enumerate(AFTER_MIN):
        jy = ((i * 53) % 9 - 4) * 6
        b.append(f'<circle cx="{Z(m):.1f}" cy="{y1 + jy:.1f}" r="3.2" fill="{GRAY}"/>')
    b.append(f'<circle cx="{Z(0):.1f}" cy="{y1:.1f}" r="7" fill="none" stroke="{RED}" stroke-width="2.5"/>')
    b.append(f'<line x1="{L}" x2="{R_}" y1="{y1 + 44}" y2="{y1 + 44}" class="axis"/>')
    for m, lab in [(-60, "09:40"), (-47.2, ""), (0, "T 10:40"), (60, "11:40"), (120, "12:40")]:
        if lab:
            b.append(f'<line x1="{Z(m):.1f}" x2="{Z(m):.1f}" y1="{y1 + 40}" y2="{y1 + 48}" class="axis"/>')
            b.append(t(Z(m), y1 + 64, lab, "lbl m", "middle"))
    b.append(f'<line x1="{Z(-62):.1f}" x2="{Z(-62):.1f}" y1="{y1 - 34}" y2="{y1 + 34}" stroke="{INK}" stroke-dasharray="3 3"/>')
    b.append(t(Z(-62), y1 - 40, "발행 09:38", "lbl m", "middle"))
    b.append(t(L, y1 + 92, f"● 센다: {INSIDE_N}건", "lbl b blue"))
    b.append(t(L + 200, y1 + 92, f"○ 지금 이 노출의 클릭: {AT_T_N}건 제외", "lbl b red"))
    b.append(t(L + 480, y1 + 92, f"● 미래(2시간): {AFTER_N}건 제외", "lbl b gray"))
    return svg(W, 336, "".join(b), "노출 직전 24시간 창과 정답 기사의 실제 클릭 기록")


# ── 그림 3. 후보별 인기 · 나이 (실제 값) ─────────────────────
def fig_features():
    W = 760
    b = []
    PX0, PW = 120, 250        # logpop 패널
    AX0, AW = 440, 185        # logage 패널
    b.append(t(PX0, 22, "인기도  logpop = ln(1 + 24시간 클릭)", "head"))
    b.append(t(AX0, 22, "나이  logage = ln(1 + 발행 후 시간)", "head"))
    for i, (aid, ok, n, lp, agetxt, la, _) in enumerate(CANDS):
        y = 44 + i * 44
        col = BLUE if ok else GRAY
        b.append(t(PX0 - 10, y + 18, aid + (" ✓" if ok else ""), "lbl b" if ok else "lbl", "end"))
        b.append(f'<rect x="{PX0}" y="{y + 4}" width="{max(PW * lp / 10, 1.5):.1f}" height="20" rx="3" fill="{col}"/>')
        b.append(t(PX0 + PW * lp / 10 + 8, y + 19, f"{lp:.2f}  ({n}번)", "lbl"))
        b.append(f'<rect x="{AX0}" y="{y + 4}" width="{AW * la / 10:.1f}" height="20" rx="3" fill="{col}" opacity=".55"/>')
        b.append(t(AX0 + AW * la / 10 + 8, y + 19, f"{la:.2f}  ({agetxt})", "lbl"))
    b.append(t(PX0, 280, "정답(✓)은 클릭이 가장 많고, 가장 최근에 나온 기사", "lbl m"))
    return svg(W, 292, "".join(b), "후보 5개의 인기도와 나이")


# ── 그림 4. 점수 합치기 (s → 최종) ───────────────────────────
def fig_score():
    W, Z0, S = 760, 440, 14      # 0 위치, 1점당 px
    b = []
    b.append(t(20, 22, "회색 = SID 점수 s (예시) · 파랑 = 최종 점수 s + 5·logpop − 1·logage", "head"))
    b.append(f'<line x1="{Z0}" x2="{Z0}" y1="36" y2="276" stroke="{INK}" stroke-width="1"/>')
    b.append(t(Z0, 292, "0", "lbl m", "middle"))
    for i, (aid, ok, n, lp, agetxt, la, s) in enumerate(CANDS):
        f = s + LAM * lp + MU * la
        y = 44 + i * 46
        b.append(t(98, y + 22, aid + (" ✓" if ok else ""), "lbl b" if ok else "lbl", "end"))
        x1, x2 = (Z0 + s * S, Z0) if s < 0 else (Z0, Z0 + s * S)
        b.append(f'<rect x="{x1:.1f}" y="{y + 2}" width="{x2 - x1:.1f}" height="14" rx="3" fill="{GRAY}"/>')
        x1, x2 = (Z0 + f * S, Z0) if f < 0 else (Z0, Z0 + f * S)
        b.append(f'<rect x="{x1:.1f}" y="{y + 18}" width="{x2 - x1:.1f}" height="16" rx="3" fill="{BLUE if ok else "#8fb4e6"}"/>')
        lx = x2 + 8 if f >= 0 else x1 - 8
        b.append(t(lx, y + 31, f"{f:.2f}", "lbl b", "start" if f >= 0 else "end"))
        b.append(t(Z0 + s * S - 8, y + 13, f"{s}", "lbl m", "end"))
    b.append(t(20, 318, "SID만 쓰면 9428643(오답)이 1등 → 합치면 정답 9788095가 1등", "lbl b blue"))
    return svg(W, 330, "".join(b), "SID 점수와 최종 점수 비교")


# ── 그림 5. 나중에 더하기 흐름 ───────────────────────────────
def fig_flow():
    steps = [("1", "SID만으로 학습", "softmax → CE", [("train", "tr")]),
             ("2", "best epoch", "Top-1 최고", [("validation", "va")]),
             ("3", "점수 s 저장", "후보마다", [("validation", "va"), ("test", "te")]),
             ("4", "λ · μ 고르기", "130개 조합", [("validation", "va")]),
             ("5", "test 한 번", "4개 지표", [("test", "te")])]
    b, W, bw, gap = [], 780, 138, 22
    for i, (no, h, d, tags) in enumerate(steps):
        x = 8 + i * (bw + gap)
        b.append(f'<rect x="{x}" y="14" width="{bw}" height="112" rx="10" class="box"/>')
        b.append(t(x + 12, 40, f"{no}. {h}", "lbl b blue"))
        b.append(t(x + 12, 62, d, "lbl m"))
        tx = x + 12
        for lab, k in tags:
            w = 10 + len(lab) * 7.2
            b.append(f'<rect x="{tx}" y="86" width="{w:.0f}" height="24" rx="12" class="tag {k}"/>')
            b.append(t(tx + w / 2, 103, lab, "tagt", "middle"))
            tx += w + 6
        if i < 4:
            b.append(f'<path d="M{x + bw + 3} 70 H{x + bw + gap - 3}" class="arr" marker-end="url(#ah)"/>')
    return svg(W, 140, "".join(b), "나중에 더하기의 다섯 단계")


# ── 그림 6. λ 곡선 ──────────────────────────────────────────
def fig_lambda():
    W, H, L, B, T_ = 760, 300, 54, 50, 20
    pw, ph = W - L - 20, H - B - T_
    X = lambda i: L + i / (len(LAMS) - 1) * pw
    Y = lambda v: T_ + ph - (v - 25) / 20 * ph
    b = []
    for v in (25, 30, 35, 40, 45):
        b.append(f'<line x1="{L}" x2="{L + pw}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" class="grid"/>')
        b.append(t(L - 8, Y(v) + 4, v, "lbl m", "end"))
    for i, lab in enumerate(LAMS):
        b.append(t(X(i), H - B + 20, lab, "lbl m", "middle"))
    b.append(t(L + pw / 2, H - 8, "λ (인기 가중치) — 0 = SID만, 클수록 인기만에 가까움", "lbl m", "middle"))
    pts = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(VAL_H))
    b.append(f'<polyline points="{pts}" fill="none" stroke="{BLUE}" stroke-width="2.5"/>')
    for i, v in enumerate(VAL_H):
        big = i == 11
        b.append(f'<g class="mark"><title>λ={LAMS[i]}: validation Top-1 {v:.2f}%</title>'
                 f'<circle cx="{X(i):.1f}" cy="{Y(v):.1f}" r="{7 if big else 4}" fill="{BLUE if big else "#fff"}" stroke="{BLUE}" stroke-width="2"/></g>')
    b.append(t(X(11), Y(VAL_H[11]) - 14, "λ = 5 → 43.27%", "lbl b blue", "middle"))
    b.append(t(X(0) + 12, Y(VAL_H[0]) + 20, "SID만 27.58%", "lbl m"))
    return svg(W, H, "".join(b), "λ별 validation Top-1 곡선")


# ── 그림 7. 결과 막대 ───────────────────────────────────────
def fig_result():
    W, H, L, B, T_ = 760, 300, 46, 46, 24
    pw, ph = W - L - 90, H - B - T_
    Y = lambda v: T_ + ph - v / 50 * ph
    b = []
    for v in (0, 10, 20, 30, 40, 50):
        b.append(f'<line x1="{L}" x2="{L + pw}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" class="grid"/>')
        b.append(t(L - 8, Y(v) + 4, v, "lbl m", "end"))
    b.append(f'<line x1="{L}" x2="{L + pw + 6}" y1="{Y(20):.1f}" y2="{Y(20):.1f}" stroke="{MUTED}" stroke-dasharray="4 4"/>')
    b.append(t(L + pw + 10, Y(20) + 4, "무작위 20%", "lbl m"))
    cats = ["SID만", "인기만", "SID + 인기 + 나이"]
    gw = pw / 3
    for gi, c in enumerate(cats):
        for si, n in enumerate(["V1", "H", "R"]):
            v = RESULT[n][gi]
            x = L + gi * gw + gw / 2 - 3 * 52 / 2 + si * 52 + 4
            b.append(f'<g class="mark"><title>{n} · {c}: {v:.2f}%</title>'
                     f'<rect x="{x:.1f}" y="{Y(v):.1f}" width="44" height="{Y(0) - Y(v):.1f}" rx="4" fill="{COL[n]}"/></g>')
            b.append(t(x + 22, Y(v) - 6, f"{v:.1f}", "lbl b", "middle"))
        b.append(t(L + gi * gw + gw / 2, H - B + 22, c, "lbl b", "middle"))
    return svg(W, H, "".join(b), "SID만, 인기만, 결합 결과 막대")


# ── 그림 8. 학습에 넣으면: softmax 두 경우 ───────────────────
def fig_joint():
    W = 760
    b = []
    finals = [s + LAM * lp + MU * la for (_, _, _, lp, _, la, s) in CANDS]
    import math
    mx = max(finals)
    ex = [math.exp(f - mx) for f in finals]
    z = sum(ex)
    probs = [e / z for e in ex]
    cases = [("경우 1: 정답이 9788095 (인기 기사)", 1), ("경우 2: 정답이 9080070 (클릭 4번 · 옛 기사)이었다면", 4)]
    for ci, (title, k) in enumerate(cases):
        x0 = 20 + ci * 375
        b.append(t(x0, 22, title, "head"))
        for i, (aid, *_rest) in enumerate(CANDS):
            y = 40 + i * 36
            ok = i == k
            b.append(t(x0 + 70, y + 17, aid, "lbl b" if ok else "lbl", "end"))
            w = 180 * probs[i]
            b.append(f'<rect x="{x0 + 80}" y="{y + 4}" width="{max(w, 1.5):.1f}" height="18" rx="3" fill="{BLUE if ok else GRAY}"/>')
            b.append(t(x0 + 86 + max(w, 1.5), y + 18, f"{probs[i]:.4f}" + ("  ← 정답" if ok else ""), "lbl b" if ok else "lbl"))
        loss = -math.log(max(probs[k], 1e-300)) if probs[k] > 0 else float("inf")
        loss = -math.log(probs[k])
        txt = f"정답 확률 {probs[k]:.4f} → loss {loss:.4f}" if ci == 0 else f"정답 확률 ≈ 0 → loss ≈ {loss:.0f}"
        b.append(t(x0, 238, txt, "lbl b " + ("blue" if ci == 0 else "red")))
        b.append(t(x0, 258, "배울 게 거의 없음" if ci == 0 else "그 기사 SID 점수를 20 넘게 올려야 → 외우기", "lbl m"))
    return svg(W, 270, "".join(b), "학습 중 softmax 확률 두 경우")


def fig_datastart():
    """장부가 5/18 07:00부터라 train 첫날 행만 24시간을 다 못 센다."""
    W, L, R_ = 760, 150, 740
    X = lambda h: L + h / 96 * (R_ - L)          # 5/17 00:00 = 0시간 ~ 5/21 00:00 = 96시간
    S = 31                                        # 5/18 07:00
    b = [t(20, 22, "장부(클릭 기록)는 5/18 07:00부터만 있다 — 세는 구간이 그 앞으로 넘어가면 그만큼 못 센다", "head")]
    b.append(f'<rect x="{X(0):.1f}" y="40" width="{X(S) - X(0):.1f}" height="196" fill="#eceef2"/>')
    b.append(t((X(0) + X(S)) / 2, 58, "데이터 없음", "lbl b gray", "middle"))
    b.append(f'<line x1="{X(S):.1f}" x2="{X(S):.1f}" y1="40" y2="236" stroke="{INK}" stroke-width="2"/>')
    b.append(t(X(S) + 6, 58, "장부 시작 5/18 07:00", "lbl b"))
    rows = [("train 행 ①", 36, "T = 5/18 12:00 (첫날)"), ("train 행 ②", 87, "T = 5/20 15:00")]
    for i, (name, T, lab) in enumerate(rows):
        y = 92 + i * 74
        b.append(t(L - 10, y + 16, name, "lbl b", "end"))
        lo = T - 24
        if lo < S:
            b.append(f'<rect x="{X(lo):.1f}" y="{y}" width="{X(S) - X(lo):.1f}" height="24" rx="4" class="band red"/>')
            b.append(t((X(lo) + X(S)) / 2, y + 17, "기록 없음 → 못 셈", "lbl b red", "middle"))
            b.append(f'<rect x="{X(S):.1f}" y="{y}" width="{X(T) - X(S):.1f}" height="24" rx="4" fill="{BLUE}"/>')
            b.append(t(X(T) + 8, y + 17, "5시간치만 셈 → 인기가 작게 나옴", "lbl b blue"))
        else:
            b.append(f'<rect x="{X(lo):.1f}" y="{y}" width="{X(T) - X(lo):.1f}" height="24" rx="4" fill="{BLUE}"/>')
            b.append(t((X(lo) + X(T)) / 2, y + 17, "24시간 다 셈", "lbl b", "middle", 'style="fill:#fff"'))
        b.append(f'<line x1="{X(T):.1f}" x2="{X(T):.1f}" y1="{y - 6}" y2="{y + 30}" stroke="{INK}" stroke-width="2"/>')
        b.append(t(X(lo), y + 44, "T − 24시간", "lbl m", "middle"))
        b.append(t(X(T) + 6 if lo < S else X(T), y + 44, lab, "lbl m", "start" if lo < S else "middle"))
    b.append(f'<line x1="{L}" x2="{R_}" y1="236" y2="236" class="axis"/>')
    for h, lab in [(0, "5/17"), (24, "5/18"), (48, "5/19"), (72, "5/20"), (96, "5/21")]:
        b.append(f'<line x1="{X(h):.1f}" x2="{X(h):.1f}" y1="232" y2="240" class="axis"/>')
        b.append(t(X(h), 256, lab, "lbl m", "middle"))
    b.append(t(20, 284, "validation · test 행은 T가 5/25 07:00 이후 → 직전 24시간이 항상 장부 안 → 항상 다 센다", "lbl b"))
    b.append(t(20, 304, "기사의 발행 시각과는 상관없다. 그 행의 노출 시각 T가 5/19 07:00 이전인지만 중요하다", "lbl m"))
    return svg(W, 316, "".join(b), "데이터 시작 시점 때문에 train 첫날 행만 인기가 작게 세지는 그림")


DEFS = f'''<svg width="0" height="0" style="position:absolute"><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L10 5 L0 10 z" fill="{MUTED}"/></marker></defs></svg>'''

html = f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>인기도 · 최신성 정리</title>
<style>
:root {{ --bg:#fff; --text:#14181f; --muted:#5b6577; --line:#e3e6eb; --surface:#f8f9fb; --hl:#eef4ff; --hl-line:#9fb8ec; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--text); font-family:system-ui,-apple-system,"Apple SD Gothic Neo","Malgun Gothic","Segoe UI",sans-serif; font-size:15.5px; line-height:1.7; word-break:keep-all; }}
main {{ max-width:880px; margin:0 auto; padding:32px 16px 72px; }}
h1 {{ font-size:25px; margin:0 0 4px; }} .sub {{ color:var(--muted); margin:0 0 20px; font-size:14px; }}
h2 {{ font-size:20px; margin:44px 0 10px; padding-bottom:6px; border-bottom:2px solid var(--line); }}
h3 {{ font-size:16.5px; margin:24px 0 6px; }}
.box1 {{ border:1.5px solid var(--hl-line); background:var(--hl); border-radius:12px; padding:14px 18px; }}
.fig {{ background:var(--surface); border:1px solid var(--line); border-radius:12px; padding:12px; margin:12px 0; }}
.chart {{ width:100%; height:auto; display:block; font-family:inherit; }}
.cap {{ color:var(--muted); font-size:13.5px; margin:6px 2px 0; }} .cap b {{ color:var(--text); }}
.formula {{ font-size:19px; text-align:center; background:#fff; border:1px solid var(--line); border-radius:10px; padding:12px; margin:12px 0; }}
.formula i {{ font-family:"Times New Roman",serif; }}
.tbl {{ overflow-x:auto; border:1px solid var(--line); border-radius:10px; margin:10px 0; }}
table {{ width:100%; border-collapse:collapse; font-size:14.5px; }}
th, td {{ padding:7px 10px; border-bottom:1px solid var(--line); text-align:left; white-space:nowrap; }}
th {{ background:#f3f4f7; font-weight:600; color:var(--muted); }} td.n, th.n {{ text-align:right; font-variant-numeric:tabular-nums; }}
tr.hl td {{ background:var(--hl); font-weight:600; }}
ol li, ul li {{ margin:3px 0; }}
.note {{ border-left:4px solid #e9c27a; background:#fff6e6; border-radius:0 8px 8px 0; padding:10px 14px; margin:12px 0; font-size:14.5px; }}
code {{ font-family:ui-monospace,Menlo,Consolas,monospace; font-size:13px; background:#f3f4f7; padding:1px 5px; border-radius:4px; overflow-wrap:anywhere; }}
/* SVG */
.head {{ font-size:15px; font-weight:700; fill:{INK}; }} .lbl {{ font-size:13.5px; fill:{INK}; }} .lbl.m {{ fill:{MUTED}; }}
.lbl.b {{ font-weight:700; }} .blue {{ fill:{BLUE}; }} .red {{ fill:{RED}; }} .gray {{ fill:#7d8594; }}
.box {{ fill:#fff; stroke:#cfd5de; stroke-width:1.2; }} .box.hl {{ fill:#eaf1ff; stroke:#9fb8ec; }} .box.ghost {{ fill:#f3f4f7; stroke:#e3e6eb; }}
.band {{ fill:#dce6fd; }} .band.red {{ fill:#fdecec; }}
.axis {{ stroke:{MUTED}; stroke-width:1.5; }} .grid {{ stroke:{LINE}; stroke-width:1; }}
.arr {{ stroke:{MUTED}; stroke-width:2.5; fill:none; }}
.tag.tr {{ fill:#e3f5ea; }} .tag.va {{ fill:#dce6fd; }} .tag.te {{ fill:#fff1d6; }} .tagt {{ font-size:12.5px; font-weight:700; fill:{INK}; }}
.mark:hover circle, .mark:hover rect {{ opacity:.8; }}
</style></head><body>{DEFS}<main>

<h1>인기도 · 최신성을 어떻게 넣었나</h1>
<p class="sub">EB-NeRD v2 재실험 · 실제 test 노출로 따라가는 정리 · 2026-11-25</p>

<div class="box1"><b>한 줄 요약</b> — Transformer는 SID만으로 학습하고, 채점할 때 후보마다 <b>노출 직전 24시간 클릭 수(인기도)</b>와 <b>발행 후 지난 시간(최신성)</b>을 점수에 더했다. 비율은 validation에서 골랐다. test Top-1 <b>27.7% → 44.2%</b> (H, seed 3개 평균).</div>

<h2>1. 왜 넣었나</h2>
<ul>
<li>Transformer(SID만)가 test Top-1 27%대에서 오르지 않아 원인을 보다가, <b>학습 없이 "직전 24시간 클릭이 가장 많은 후보"만 골라도 42.9%</b>라는 것을 발견했다.</li>
<li>SID에는 "무슨 내용인가"만 있고 <b>"지금 얼마나 읽히나"</b>는 없다. 뉴스 클릭은 지금 뜨는 기사인지에 크게 좌우된다.</li>
<li>test 정답의 98%는 처음 보는 기사라 개인화 신호가 SID뿐이다. 인기와 최신성은 <b>새 기사에도 노출 직전에 바로 알 수 있는</b> 정보라 효과가 크다.</li>
</ul>

<h2>2. 클릭 기록 장부 (한 번만)</h2>
<div class="fig">{fig_ledger()}</div>
<p class="cap">train + validation behaviors의 <b>모든 노출 · 모든 사용자</b>에서, 클릭한 기사마다 (기사, 그 노출 시각)을 적는다. 1pos4neg와는 상관없이 원본 전체로 만든다 — <b>480,566건, 클릭된 기사 3,080개</b>.</p>

<h2>3. 인기도: 노출 직전 24시간 클릭 수</h2>
<p>채점할 1pos4neg 행마다 <b>T = 그 행의 노출 시각</b>을 잡고, 후보마다 장부에서 <b>[T − 24시간, T)</b> 안의 기록 수를 센다. 아래는 실제 test 노출 <b>184583062</b> (사용자 2141956, T = 2023-05-30 10:40:07)의 정답 기사다.</p>
<div class="fig">{fig_window()}</div>
<p class="cap">정답 기사는 09:38에 발행돼 09:52부터 클릭이 쌓였다. T 직전 24시간 기록 <b>75건만</b> 센다. T와 같은 시각 1건은 <b>지금 이 노출의 클릭</b>이라 빼고, T 이후는 미래라 뺀다. 그래서 실제 서비스에서도 그 순간 알 수 있는 값이다.</p>
<ul>
<li><b>logpop = ln(1 + 클릭 수)</b> — 0~수천인 클릭 수를 log로 눌러, "10배 많다"가 같은 크기의 차이가 되게 한다.</li>
<li>다른 사용자 · 같은 사용자의 예전 클릭 모두 센다. 기간 경계도 넘는다 (5/25 아침 행이면 전날 train 기간 클릭도 센다).</li>
<li>T가 행마다 다르므로, 같은 기사라도 노출마다 클릭 수가 다르다.</li>
</ul>

<h2>4. 최신성: 발행 후 지난 시간</h2>
<ul>
<li><b>나이 = T − 발행 시각</b> (시간 단위, articles의 발행 시각). 발행이 T보다 늦으면 0.</li>
<li><b>logage = ln(1 + 나이)</b>. 점수에서는 <b>빼서</b> 오래된 기사를 불리하게 한다.</li>
</ul>
<div class="fig">{fig_features()}</div>
<p class="cap">같은 노출의 후보 5개 (실제 값). EB-NeRD 화면에는 몇 달~몇 년 된 기사도 섞여 뜬다. 이 노출에서는 인기도와 최신성이 모두 정답을 가리킨다.</p>

<h2>5. 점수에 더하기 (최종 방식, 재학습 없음)</h2>
<div class="formula">최종 점수 = <i>s</i> + <i>λ</i> · logpop + <i>μ</i> · logage &nbsp;&nbsp;(<i>s</i> = log P(c1) + log P(c2|c1) + log P(c3|c1,c2))</div>
<div class="fig">{fig_flow()}</div>
<p class="cap">초록 = train, 파랑 = validation 앞 절반, 노랑 = test. Transformer 가중치는 바꾸지 않고, 숫자 두 개(λ, μ)만 validation에서 정해 test에 한 번 적용한다.</p>

<h3>5-1. 예시 노출에 적용 (λ = 5, μ = −1)</h3>
<div class="fig">{fig_score()}</div>
<p class="cap">SID 점수 <i>s</i>는 학습된 모델이 AWS에 있어서 <b>예시 값</b>이다. 클릭 수와 나이는 실제 값.</p>

<h3>5-2. λ · μ 고르기</h3>
<div class="tbl"><table><thead><tr><th></th><th>값</th></tr></thead><tbody>
<tr><td>λ 후보 (13개)</td><td>0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1, 1.5, 2, 3, 5, 10</td></tr>
<tr><td>μ 후보 (10개)</td><td>−5, −3, −2, −1, −0.5, −0.2, 0, 0.2, 0.5, 1 (음수 = 최근 기사 우대)</td></tr>
<tr><td>고르는 데이터</td><td>validation 앞 절반 (H 122,560행) — 학습에 안 썼고 test처럼 새 기사가 대부분</td></tr>
<tr><td>기준</td><td>130개 조합마다 전체 행을 채점해 Top-1 최대. 같으면 |λ|, |μ|가 작은 쪽</td></tr>
<tr class="hl"><td>고른 값</td><td>9번(V1 · H · R × seed 3개) 모두 λ = 5 · μ = −1 (7번) / −2 (2번)</td></tr>
<tr><td>적용</td><td>test (H 122,680행)에 한 번만 — test로는 아무것도 고르지 않음</td></tr>
</tbody></table></div>
<div class="fig">{fig_lambda()}</div>
<p class="cap">H seed 42, μ = 0일 때 λ별 validation Top-1 (실제 값). λ = 3~10에서 평평해서 5 근처면 결과가 거의 같다. μ까지 고르면 43.47% (seed 123).</p>

<h3>5-3. 결과 (test, seed 3개 평균)</h3>
<div class="fig">{fig_result()}</div>
<div class="tbl"><table><thead><tr><th>test</th><th class=n>SID만</th><th class=n>인기만</th><th class=n>SID + 인기 + 나이</th><th class=n>SID가 보탠 몫</th></tr></thead><tbody>
<tr><td>V1 (예전 SID)</td><td class=n>26.57%</td><td class=n>42.94%</td><td class=n>43.66%</td><td class=n>+0.72%p</td></tr>
<tr class="hl"><td>H</td><td class=n>27.67%</td><td class=n>42.93%</td><td class=n>44.24%</td><td class=n>+1.31%p</td></tr>
<tr><td>R</td><td class=n>27.51%</td><td class=n>42.97%</td><td class=n>44.22%</td><td class=n>+1.25%p</td></tr>
</tbody></table></div>
<p class="cap">크게 오른 건 인기 자체의 힘이고, 봐야 할 건 <b>SID가 인기 위에 보탠 몫</b>이다. AUC · MRR · nDCG@5도 같은 방향 (H: AUC 0.584 → 0.742).</p>

<h2>6. 학습에 넣어 보기 (시도했지만 채택 안 함)</h2>
<p>같은 식(<i>s</i> + w1 · logpop + w2 · logage)을 <b>학습 중에</b> 쓰고 softmax → cross entropy로 Transformer를 처음부터 학습했다. train 행에도 같은 장부 · 같은 규칙으로 인기도 · 최신성을 붙였다.</p>
<div class="fig">{fig_joint()}</div>
<p class="cap">같은 예시 노출의 최종 점수로 softmax 확률을 계산했다. <b>경우 1</b>: 인기가 이미 맞혀서 loss가 거의 0 → Transformer가 배울 게 없다. <b>경우 2</b>: 인기로 못 맞히는 행은 loss가 매우 커서, 그 기사의 SID 점수를 크게 올리려고 <b>train 기사를 외운다</b>. test 정답은 98%가 처음 보는 기사라 외운 게 쓸모없다.</p>
<div class="tbl"><table><thead><tr><th>test Top-1 (seed 42)</th><th class=n>인기만</th><th class=n>학습에 넣기 (w = 5, −1 고정)</th><th class=n>나중에 더하기</th></tr></thead><tbody>
""" + "".join(f'<tr{" class=hl" if n == "H" else ""}><td>{n}</td><td class=n>{a:.2f}%</td><td class=n>{b_:.2f}%</td><td class=n><b>{c:.2f}%</b></td></tr>'
              for n, (a, b_, c) in JOINT.items()) + f"""
</tbody></table></div>
<p class="cap">w도 같이 학습하게 두면 epoch 1에 인기 가중치가 5 → 0.78로 줄며 더 나빠졌다 (train에서는 SID가 외워서 맞히므로 인기에 덜 기댄다). 그래서 <b>비율은 test와 비슷한 validation에서 정하는 지금 방식</b>을 쓴다.</p>

<h3>6-1. train 첫날 행의 인기 값 (데이터 시작 시점)</h3>
<div class="fig">{fig_datastart()}</div>
<p class="cap">학습에 넣기만 train 행을 쓰므로 이 실험에만 해당한다 (train 1주 중 첫 24시간). 학습에 넣기가 안 된 <b>주된 이유는 위의 "기사 외우기"</b>이고, 이건 작은 부수 한계다. 나중에 더하기는 validation · test 행만 써서 상관없다.</p>

<h2>7. 정리</h2>
<div class="tbl"><table><thead><tr><th></th><th>나중에 더하기 (최종)</th><th>학습에 넣기</th></tr></thead><tbody>
<tr><td>인기도 · 최신성 계산</td><td colspan=2>같음: 장부에서 T 직전 24시간 클릭 수 → logpop, 발행 후 시간 → logage</td></tr>
<tr><td>Transformer가 보는 점수</td><td>SID 점수만</td><td>SID 점수 + 인기 · 나이</td></tr>
<tr><td>SID가 배우는 것</td><td>사용자 취향 · 사건 · 주제 패턴 (새 기사에도 통함)</td><td>인기로 안 풀리는 train 기사 외우기</td></tr>
<tr><td>비율을 정하는 데이터</td><td><b>validation</b> (새 기사가 대부분)</td><td>train (본 기사가 대부분) 또는 고정</td></tr>
<tr class="hl"><td>H test Top-1</td><td>44.21% (seed 3개 평균 44.24%)</td><td>42.92%</td></tr>
</tbody></table></div>
<div class="note"><b>주의</b><ul>
<li>클릭 시각은 데이터에 없어서 그 노출의 시각으로 대신했다.</li>
<li>train 첫 24시간 행은 장부 시작(5/18 07:00) 전 클릭이 없어서 인기도가 적게 세진다. 학습에 넣기 실험에만 영향이 있다.</li>
<li>validation 앞 절반을 best epoch 고르기와 λ · μ 고르기에 두 번 썼다. 고르는 값이 둘뿐이고 9번 모두 λ = 5라 영향은 작다고 본다.</li>
<li>그림 2의 클릭 점은 장부에서 뽑은 실제 시각이다 (점이 겹치지 않게 세로로만 흩뜨림).</li>
</ul></div>
<p class="cap">코드: <code>experiments/v2_yeomin_rerun/</code> — 장부 <code>analyze_test.py</code>(load_clicks), 나중에 더하기 <code>popularity_fusion.py --with-age</code>, validation 점수 <code>5_predict_val.sh</code>, 학습용 특징 <code>add_candidate_features.py</code>, 학습에 넣기 <code>8_transformer_features.sh</code> + <code>transformer_features.patch</code>. 이 문서: <code>reports/analysis/build_popularity_doc.py</code></p>
</main></body></html>
"""
OUT.write_text(html, encoding="utf-8")
print(OUT)
