"""SURGE (SIGIR 2021) 전체 흐름도 — 랏코 16:9 덱 (2장).

1장: 입력 → 네 모듈(A 관심 그래프 · B 관심 융합 · C 관심 추출 · D 예측) → 출력, 학습 되돌이.
2장: 예시 독자(클릭 6개, 후보 1개)로 본 단계마다의 데이터 모양 (기사 6개 → 덩어리 3개).
숫자는 reports/SURGE_예시로_따라가기.html 의 기본 설정(후보 = 연정 협상 시작) 값이다.
실행: python reports/slides/build_surge_flow_deck.py  →  reports/slides/SURGE_전체흐름.pptx
필요: python-pptx, Pillow, Pretendard OTF (~/.fonts/pretendard), pdflatex, pdftocairo
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rakko import (KIND, MARGIN, WARNINGS, Deck, arrow, box, check, chip, formula, rect, text)

OUT = Path(__file__).resolve().parent / "SURGE_전체흐름.pptx"
TOTAL = 2
FOOT = "출처: Chang et al., SURGE (SIGIR 2021) · 예시 숫자: SURGE 예시로 따라가기 (후보 = 연정 협상 시작)"
CH = "SURGE 전체 흐름"

d = Deck(TOTAL, FOOT)


def kbox(s, name, x, y, w, h, paras, kind="white", radius=12, line_w=1.5, **kw):
    return box(s, name, x, y, w, h, paras, fill=f"k-{kind}-fill", line=f"k-{kind}-line", radius=radius,
               line_w=line_w, color=f"k-{kind}-text", **kw)


def mix(t: float, hi: str = "4C7BF4") -> str:
    """0 → 흰색, 1 → hi 색."""
    t = max(0.0, min(1.0, t))
    a = [255, 255, 255]
    b = [int(hi[i:i + 2], 16) for i in (0, 2, 4)]
    return "".join(f"{round(a[k] + (b[k] - a[k]) * t):02X}" for k in range(3))


def grid(s, name, x, y, M, cell=22, vmax=None, hi="4C7BF4"):
    """행렬 M을 칸마다 색 농도로 그린다 (값이 클수록 진함). 반환: (w, h)."""
    vmax = vmax or max(max(abs(v) for v in r) for r in M) or 1
    for r, row in enumerate(M):
        for c, v in enumerate(row):
            rect(s, f"{name} {r+1},{c+1}", x + c * cell, y + r * cell, cell, cell,
                 fill=mix(abs(v) / vmax, hi), line="hairline", line_w=1)
    return len(M[0]) * cell, len(M) * cell


# ── 예시 값 (예시로 따라가기 기본 설정) ──────────────────────────
VEC = [[1.0, 0.10], [0.15, 1.0], [0.95, 0.25], [0.6, 0.55], [0.05, 1.0], [0.25, 0.95]]
HEADS = [[1, 1], [1, 0.5]]


def wcos(a, b):
    def cos(u, v):
        return sum(p * q for p, q in zip(u, v)) / (math.hypot(*u) * math.hypot(*v))
    return sum(cos([w[0] * a[0], w[1] * a[1]], [w[0] * b[0], w[1] * b[1]]) for w in HEADS) / len(HEADS)


M = [[wcos(a, b) for b in VEC] for a in VEC]
EDGES = [(0, 2), (1, 4), (1, 5), (4, 5)]
A = [[1 if i == j or (i, j) in EDGES or (j, i) in EDGES else 0 for j in range(6)] for i in range(6)]
H4 = [[1.97, 0.29, 1.48, 0.31], [0.30, 1.98, 0.22, 1.99], [1.92, 0.44, 1.43, 0.46],
      [1.20, 1.10, 0.90, 1.10], [0.20, 1.98, 0.12, 1.99], [0.40, 1.93, 0.32, 1.94]]
BETA = [0.60, 3.05, 1.04, 1.83, 3.02, 2.93]
S = [[0.99, 0.01, 0], [0, 0, 1], [0.99, 0.01, 0], [0.26, 0.43, 0.31], [0, 0, 1], [0, 0, 1]]
HSTAR = [[4.15, 1.00, 3.11, 1.04], [0.56, 0.49, 0.42, 0.49], [1.27, 6.25, 0.94, 6.25]]
GSTAR = [0.087, 0.039, 0.874]
HS = [0.63, 0.88, 0.52, 0.88]

# ════════ 1. 전체 흐름 (모듈 네 개) ════════
s = d.new(notes="입력: 사용자 클릭 기록 n개(시간순) + 후보 기사 1개. 출력: 클릭 확률 y_hat.\n"
                "A 관심 그래프 만들기(1–3단계): 가중 코사인 유사도, 상위 epsilon 비율 쌍만 선 -> 그래프 A.\n"
                "B 관심 융합(4단계): E_ij = softmax_j(alpha_i + beta_j), h'_i = sigma(W_a sum_j E_ij h_j + h_i), 헤드 이어 붙임.\n"
                "C 관심 추출(5단계): S_i = softmax(W_p sum_j A_ij h'_j), h* = S^T h', gamma* = S^T gamma, A* = S^T A S.\n"
                "D 예측(6–7단계): h_s = AUGRU(h*_1..h*_m, gamma*), h_g = sum_i gamma_i h'_i, y_hat = MLP(h_s, h_g, h_t, h_g*h_t).\n"
                "8단계 학습: 교차 엔트로피 + L2 + L_M + L_A + L_P 를 줄이도록 모든 가중치 수정.")
d.frame(s, "01", CH, "SURGE는 클릭 기록을 관심 덩어리로 묶어 읽는다",
        "입력은 클릭 기록과 후보 기사 하나, 출력은 이 후보를 클릭할 확률 하나")

ws = [196, 258, 258, 258, 258, 196]
gap = 54
x0 = MARGIN + (1728 - (sum(ws) + gap * 5)) / 2
xs = [x0 + sum(ws[:i]) + gap * i for i in range(6)]
by, bh = 450, 172
mods = [
    ("입력", None, ["클릭 기록 n개", "(시간순)"], "gray"),
    ("A · 1–3단계", "관심 그래프 만들기", ["비슷한 기사끼리", "선 긋기"], "blue"),
    ("B · 4단계", "관심 융합 (GNN)", ["후보 관련 이웃의", "값을 크게 섞기"], "blue"),
    ("C · 5단계", "관심 추출 (풀링)", ["기사 n개를", "덩어리 m개로"], "blue"),
    ("D · 6–7단계", "예측", ["덩어리를 시간순으로", "읽고 후보와 비교"], "blue"),
    ("출력", None, ["클릭 확률", "(0~1 숫자 하나)"], "green"),
]
for i, (tag_, title_, lines, kind) in enumerate(mods):
    paras = [(tag_, {"size": 18, "weight": 600, "color": "slate" if kind != "gray" else "k-gray-text"})]
    if title_:
        paras.append((title_, {"size": 24, "weight": 700, "gap": 6}))
    paras += [(lines[0], {"size": 19, "weight": 400, "gap": 10 if title_ else 14}), (lines[1], {"size": 19, "weight": 400})]
    kbox(s, f"Module {i+1}", xs[i], by, ws[i], bh, paras, kind, radius=14, line_w=2 if kind == "blue" else 1.5,
         style="caption", pad=(12, 10, 12, 10))
    if i < 5:
        arrow(s, f"Flow arrow {i+1}", [(xs[i] + ws[i] + 6, by + bh / 2), (xs[i + 1] - 8, by + bh / 2)], width_pt=3)

# 모듈 출력 (말 + 수식)
outs = [(1, "기사 그래프", r"\mathbf{A}\ (n\times n)"),
        (2, "섞인 기사 값 · 후보 관련도", r"\mathbf{h}'_i,\ \beta_j"),
        (3, "덩어리 값 · 덩어리 중요도", r"\mathbf{h}^{*}_k,\ \gamma^{*}_k"),
        (4, "관심 흐름 → 확률", r"\mathbf{h}_s,\ \mathbf{h}_g\rightarrow\hat{y}")]
for i, words, tex in outs:
    cx = xs[i] + ws[i] / 2
    text(s, f"Output label {i}", xs[i] - 14, by + bh + 18, ws[i] + 28, 28, words, "caption", "slate", align="c",
         size=19, one_line=True)
    formula(s, f"Output formula {i}", cx, by + bh + 76, tex, alt=f"{words}: {tex}", scale=1.0, anchor="mc",
            max_w=ws[i] + 20)
formula(s, "Output y-hat", xs[5] + ws[5] / 2, by + bh + 34, r"\hat{y}", alt="클릭 확률 y hat", scale=1.3,
        anchor="mc", color="success-text")

# 후보 기사 줄: 입력 위에서 B와 D로
cy = 352
kbox(s, "Candidate", xs[0], cy - 26, ws[0], 52, [("★ 후보 기사 1개", {"size": 20, "weight": 700})], "amber",
     radius=26, style="caption", pad=(8, 0, 8, 0))
lane_end = xs[4] + ws[4] / 2
arrow(s, "Candidate lane", [(xs[0] + ws[0] + 6, cy), (lane_end, cy), (lane_end, by - 8)], color="k-amber-line",
      width_pt=2.5)
arrow(s, "Candidate to B", [(xs[2] + ws[2] / 2, cy), (xs[2] + ws[2] / 2, by - 8)], color="k-amber-line", width_pt=2.5)
text(s, "Candidate note B", xs[2] + ws[2] / 2 + 12, cy + 14, 260, 28, "β: 이웃이 후보와 관련 있나", "footer",
     "k-amber-text", weight=600, one_line=True)
text(s, "Candidate note D", lane_end + 12, cy + 14, 250, 28, "후보 벡터와 비교", "footer", "k-amber-text",
     weight=600, one_line=True)

# 학습 되돌이 (점선)
ry = 790
ox = xs[5] + ws[5] / 2
arrow(s, "Training loop", [(ox, by + bh + 70), (ox, ry), (xs[1] + ws[1] / 2, ry), (xs[1] + ws[1] / 2, ry - 30)],
      color="amber", width_pt=2.5, dash="sysDot")
text(s, "Training label", xs[1] + ws[1] / 2 + 24, ry + 12, 1180, 32,
     "**8단계 학습** · ŷ와 정답 y(클릭했나)를 비교한 손실 + 덩어리 규칙 벌점 3개로 모든 가중치를 거꾸로 고친다",
     "caption", "charcoal", size=20, one_line=True)

# 핵심 한 줄
text(s, "Takeaway", MARGIN + 24, 888, 1680, 36,
     "[[핵심]]  시간 대신 비슷함으로 묶고, 후보 관점으로 덩어리를 고르고, 덩어리를 시간순으로 읽는다",
     "body", "charcoal", size=24, one_line=True)

# ════════ 2. 데이터 모양 ════════
s = d.new(notes="예시 독자: 클릭 6개(1·3 스포츠, 2·5·6 정치, 4 날씨), 후보 = 연정 협상 시작(정치).\n"
                "1 기사 벡터 h_i (6x2) -> 2 유사도 M (6x6, 헤드 평균 가중 코사인) -> 3 그래프 A (6x6, 선 1-3, 2-5, 2-6, 5-6)\n"
                "-> 4 h' (6x4: 헤드 2개 이어 붙임), beta (6) -> 5 S (6x3) -> h* = S^T h' (3x4), gamma* = S^T gamma (3)\n"
                "-> 6 h_s (4) = [0.63, 0.88, 0.52, 0.88] -> 7 y_hat = 0.89. 옆길: h_g = sum gamma_i h'_i (4) = [0.48, 1.79, 0.36, 1.79].\n"
                "색이 진할수록 값이 크다. 덩어리 열 순서: 스포츠, 날씨, 정치.")
d.frame(s, "01", CH, "기사 6개가 덩어리 3개로 줄어드는 과정",
        "예시 독자(클릭 6개, 후보 1개)로 본 단계마다의 데이터 모양 · 칸 하나 = 숫자 하나, 진할수록 큰 값")

cw, cg = 180, 41
cx0 = MARGIN + (1728 - (8 * cw + 7 * cg)) / 2
cols = [cx0 + i * (cw + cg) for i in range(8)]
gy_top, gy_mid = 372, 470
heads_ = ["1 점으로", "2 유사도", "3 선 긋기", "4 이웃과 섞기", "5 덩어리 배정", "5 덩어리 값", "6 순서대로 읽기", "7 점수"]
for i, h_ in enumerate(heads_):
    text(s, f"Step head {i+1}", cols[i] - 10, 318, cw + 20, 32, h_, "card-title", "brand-blue-deep", align="c",
         size=21, weight=700, one_line=True)


def place(i, mats, cell=22, sep=12):
    """칸 i 가운데에 행렬 여러 개를 나란히. mats: [(name, M, vmax, hi)]"""
    sizes = [(len(m[1][0]) * cell, len(m[1]) * cell) for m in mats]
    tw = sum(w for w, _ in sizes) + sep * (len(mats) - 1)
    x = cols[i] + (cw - tw) / 2
    for (nm, m, vmax, hi), (w, h) in zip(mats, sizes):
        grid(s, nm, x, gy_mid - h / 2, m, cell, vmax, hi)
        x += w + sep


place(0, [("Grid h", VEC, 1.0, "4C7BF4")])
place(1, [("Grid M", M, 1.0, "4C7BF4")])
place(2, [("Grid A", A, 1.0, "1F3F99")])
place(3, [("Grid h'", H4, 2.0, "4C7BF4"), ("Grid beta", [[b] for b in BETA], 3.1, "F2B33D")])
place(4, [("Grid S", S, 1.0, "6E56CF")])
place(5, [("Grid h*", HSTAR, 6.3, "4C7BF4"), ("Grid gamma*", [[g] for g in GSTAR], 0.9, "F2B33D")])
place(6, [("Grid h_s", [HS], 1.0, "4C7BF4")])
kbox(s, "y-hat value", cols[7] + (cw - 96) / 2, gy_mid - 28, 96, 56, [("0.89", {"size": 24, "weight": 800})], "green",
     radius=10, style="body", pad=(4, 0, 4, 0))
for i in range(7):
    arrow(s, f"Step arrow {i+1}", [(cols[i] + cw + 4, gy_mid), (cols[i + 1] - 6, gy_mid)], width_pt=2.5)

labels = [r"\mathbf{h}_i\ (6\times 2)", r"\mathbf{M}\ (6\times 6)", r"\mathbf{A}\ (6\times 6)",
          r"\mathbf{h}'\ (6\times 4),\ \beta", r"\mathbf{S}\ (6\times 3)", r"\mathbf{h}^{*}\ (3\times 4),\ \gamma^{*}",
          r"\mathbf{h}_s\ (4)", r"\hat{y}"]
for i, tex in enumerate(labels):
    formula(s, f"Shape label {i+1}", cols[i] + cw / 2, 598, tex, alt=tex, scale=0.95, anchor="mc", max_w=cw + 30)
descs = [["기사 6개,", "숫자 2개씩"], ["둘씩 가중 코사인,", "헤드 평균"], ["상위 쌍만 1", "(선 4개)"],
         ["β 비율로 이웃 섞기,", "헤드 2개 이어 붙임"], ["기사 → 덩어리", "소속 비율 (행 합 1)"],
         ["S로 모은 내용과", "후보 관련도"], ["덩어리 3개를", "시간순으로 (AUGRU)"], ["요약 셋과 후보를", "MLP로 비교"]]
for i, (l1, l2) in enumerate(descs):
    text(s, f"Step desc {i+1}", cols[i] - 18, 630, cw + 36, 60, [l1, l2], "footer", "slate", align="c", size=18)

# 옆길: h_g (4단계 h' → 7단계)
hy = 736
xa, xb = cols[3] + cw / 2, cols[7] + cw / 2
arrow(s, "h_g lane", [(xa, 694), (xa, hy), (xb, hy), (xb, 694)], color="brand-blue", width_pt=2.2)
formula(s, "h_g formula", (xa + xb) / 2 - 150, hy + 24, r"\mathbf{h}_g=\textstyle\sum_i \gamma_i\,\mathbf{h}'_i",
        alt="h_g = sum_i gamma_i h'_i", scale=0.95, anchor="tl")
text(s, "h_g label", (xa + xb) / 2 + 60, hy + 20, 520, 32, "풀링 전 기사 6개의 전체 요약 (식 15)", "footer",
     "brand-blue-deep", weight=600, one_line=True)

# 학습 되돌이
ty = 858
arrow(s, "Training loop", [(xb + 46, 700), (xb + 46, ty), (cols[0] + cw / 2, ty),
                           (cols[0] + cw / 2, ty - 26)], color="amber", width_pt=2.5, dash="sysDot")
text(s, "Training label", cols[0] + cw / 2 + 20, ty + 12, 1300, 32,
     "**8 학습** · 손실 = 교차 엔트로피(ŷ, 정답 y) + 벌점 L_M · L_A · L_P (S가 규칙을 지키나) → 1–7단계의 가중치를 모두 고친다",
     "footer", "charcoal", size=19, one_line=True)

d.save(OUT)
probs = check(OUT)
print("saved", OUT)
print("check:", probs or "ok")
print("warnings:", WARNINGS or "none")
