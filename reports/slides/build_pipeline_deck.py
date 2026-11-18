"""EB-NeRD SID 파이프라인 전체 흐름 — 랏코 16:9 덱 (모듈마다 한 장, 9장).

원본: reports/파이프라인_전체흐름.html (11/18 팀미팅 반영본). 도식은 HTML 그림의 구조·색을 그대로 옮긴다.
실행: python reports/slides/build_pipeline_deck.py  →  reports/slides/파이프라인_전체흐름.pptx
필요: python-pptx, Pillow, Pretendard OTF (~/.fonts/pretendard), pdflatex, pdftocairo
"""
from __future__ import annotations

import sys
from pathlib import Path

from pptx.enum.shapes import MSO_SHAPE

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rakko import (KIND, MARGIN, WARNINGS, Deck, arrow, box, chain, check, chip, curve, dot, formula, hline,
                   pill, polygon, rect, rgb, table, text, text_width)

OUT = Path(__file__).resolve().parent / "파이프라인_전체흐름.pptx"
TOTAL = 9
FOOT = "기준: 여민 main bcd1168 · 민지 claude/preprocess-fixed-cluster · 11/18 팀미팅 반영"
CH = {"01": "전체 흐름", "02": "전처리", "03": "RQ-VAE", "04": "SID 부여", "05": "시퀀스·후보",
      "06": "Transformer", "07": "정리"}
OWNER = {"민지": "purple", "여민": "blue", "data": "gray"}   # HTML과 같은 담당 색

d = Deck(TOTAL, FOOT)


def sub(s, name, x, y, w, t, color="ink", size=24):
    """HTML 소제목 (굵은 검정)."""
    return text(s, name, x, y, w, 36, t, "body-strong", color, size=size, one_line=True)


def kbox(s, name, x, y, w, h, paras, kind="white", radius=12, line_w=1.5, **kw):
    f, l, t = KIND[kind]
    return box(s, name, x, y, w, h, paras, fill=f"k-{kind}-fill", line=f"k-{kind}-line", radius=radius,
               line_w=line_w, color=f"k-{kind}-text", **kw)


def tag(s, name, x, y, label, kind):
    """작은 상태 배지 (확인 / 수정 / 실제 용도 …)."""
    return chip(s, name, x, y, label, kind, h=30, size=18, weight=600, padx=14, radius=15)


# ════════ 1. 표지 ════════
s = d.new()
text(s, "Deck label", MARGIN, 300, 1200, 32, "SID Project · 파이프라인 정리", "chapter", "brand-blue-deep")
text(s, "Cover title", MARGIN, 352, 1500, 260, ["EB-NeRD SID 파이프라인", "전체 흐름"], "cover-title", "ink")
text(s, "Cover subtitle", MARGIN, 636, 1500, 44,
     "전처리 · RQ-VAE · SID 부여 · 시퀀스 · Transformer — 모듈마다 한 장", "body-lg", "steel", one_line=True)
for i, (lab, val) in enumerate([("작성", "민지"), ("날짜", "2026. 11. 18"),
                                ("기준", "여민 main bcd1168 · 11/18 팀미팅 반영")]):
    x = MARGIN + [0, 220, 500][i]
    text(s, f"Meta label {i+1}", x, 880, 600, 26, lab, "footer", "muted")
    text(s, f"Meta value {i+1}", x, 910, 900, 32, val, "table", "ink", weight=600)

# ════════ 2. 전체 흐름 ════════
s = d.new()
d.frame(s, "01", CH["01"], "기사를 SID로 바꾸고 다음 클릭의 SID를 맞힌다",
        "①~⑨ 단계와 담당 — 핵심은 c2가 \"사건\"을 담게 하는 것")
steps = [("①", "원천 데이터", ["EB-NeRD", "기사·노출·히스토리"], "data", None),
         ("②", "전처리", ["임베딩·c1·개체", "train 사건 묶기"], "민지", ("확인", "amber")),
         ("③", "RQ-VAE 학습", ["train 기사만", "인코더 + Q1·Q2·Q3"], "여민", None),
         ("④", "SID 부여", ["기사 → c1·c2·c3·c4", "새 기사 → 사건 배정"], "여민", ("수정", "red")),
         ("⑤", "시퀀스 생성", ["사용자 history를", "SID 순서로"], "민지", None),
         ("⑥", "1pos4neg", ["정답 1 + 오답 4", "같은 앞 3자리 제외"], "민지", None),
         ("⑦", "split", ["노출 시각 순", "앞 절반 val · 뒤 test"], "여민", None),
         ("⑧", "Transformer", ["history 읽고", "c1→c2→c3 예측"], "여민", ("설정 수정", "red")),
         ("⑨", "예측·평가", ["후보 점수로 순위", "AUC·MRR·nDCG"], "여민", None)]
cw, chh, gx = 290, 128, 52
x0 = MARGIN + (1728 - (5 * cw + 4 * gx)) / 2
pos = {}
for i, (no, name, desc, owner, badge) in enumerate(steps):
    col = i if i < 5 else 9 - i          # 둘째 줄은 오른쪽 → 왼쪽 (HTML과 같음)
    x, y = x0 + col * (cw + gx), [306, 496][0 if i < 5 else 1]
    pos[i] = (x, y)
    rect(s, f"Step {no} card", x, y, cw, chh, fill="canvas", line="hairline", radius=10)
    rect(s, f"Step {no} bar", x, y, 8, chh, fill=f"k-{OWNER[owner]}-line")
    text(s, f"Step {no} text", x + 26, y + 14, cw - 40, chh - 20,
         [(f"{no} {name}", {"size": 22, "weight": 700, "color": "ink"}),
          (desc[0], {"size": 18, "color": "slate", "gap": 6}), (desc[1], {"size": 18, "color": "slate"})], "body")
    if badge:
        _, bw = tag(s, f"Step {no} badge", 0, 0, *badge)
        s.shapes._spTree.remove(s.shapes[-1]._element)
        tag(s, f"Step {no} badge", x + cw - bw - 16, y - 15, *badge)
for i in range(8):
    (xa, ya), (xb, yb) = pos[i], pos[i + 1]
    if i == 4:
        arrow(s, "Flow arrow 5-6", [(xa + cw / 2, ya + chh + 4), (xb + cw / 2, yb - 8)], width_pt=2)
    elif i < 4:
        arrow(s, f"Flow arrow {i+1}-{i+2}", [(xa + cw + 6, ya + chh / 2), (xb - 8, yb + chh / 2)], width_pt=2)
    else:
        arrow(s, f"Flow arrow {i+1}-{i+2}", [(xa - 6, ya + chh / 2), (xb + cw + 8, yb + chh / 2)], width_pt=2)
lx, ly = x0, 500
text(s, "Legend title", lx, ly, 280, 30, "누가 맡나", "caption", "ink", weight=700)
for k, (who, lab) in enumerate([("민지", "민지 (전처리·시퀀스)"), ("여민", "여민 (RQ-VAE·Transformer)"), ("data", "원천 데이터")]):
    rect(s, f"Legend swatch {k+1}", lx, ly + 44 + k * 30, 18, 18, fill=f"k-{OWNER[who]}-line", radius=3)
    text(s, f"Legend label {k+1}", lx + 28, ly + 38 + k * 30, 260, 28, lab, "footer", "slate")
# SID 4자리 읽는 법
sub(s, "SID heading", x0, 668, 600, "SID 4자리 읽는 법")
text(s, "SID label", x0, 760, 200, 50, "기사의 SID =", "body-strong", "ink", anchor="m")
digits = [("c1", "3", "gray", "카테고리", "예: 스포츠", None), ("c2", "41", "amber", "사건 ★", "예: A선수 이적 사건", None),
          ("c3", "120", "green", "사건 안 세부", "예: 계약 조건 기사", None),
          ("c4", "0", "white", "충돌 번호", "앞 3자리 같을 때 0,1,2…", "점수엔 안 씀")]
bx = x0 + 210
for i, (k_, v, kind, lab, ex, ex2) in enumerate(digits):
    x = bx + i * 200
    text(s, f"Digit {k_} head", x, 708, 170, 28, k_, "caption", "slate", align="c", weight=700)
    kbox(s, f"Digit {k_}", x, 740, 170, 90, [(v, {"size": 44, "weight": 800})], kind, radius=12, line_w=2,
         dash="dash" if kind == "white" else None)
    rect(s, f"Digit {k_} stem v", x + 84, 830, 2, 24, fill="stone")
    text(s, f"Digit {k_} label", x - 15, 858, 200, 32, lab, "body-strong", "k-amber-text" if kind == "amber" else "ink",
         align="c", size=22)
    text(s, f"Digit {k_} example", x - 25, 890, 220, 56, [ex] + ([ex2] if ex2 else []), "footer", "slate", align="c")
kbox(s, "Summary", bx + 4 * 200 + 40, 744, 1824 - (bx + 4 * 200 + 40) - 24, 150,
     ["**한 줄 요약**", "기사 → 4자리 SID, history의 SID로 다음 클릭 예측",
      "핵심은 **c2가 사건을 담게** 하는 것",
      "새 기사는 **train 사건의 c2를 물려받는다**"],
     "blue", radius=16, style="caption", align="l", pad=(24, 12, 20, 12))

# ════════ 3. 전처리 ════════
s = d.new(notes="IDF 가중 Jaccard(A,B) = Σ 겹친 개체 idf ÷ Σ 합친 개체 idf. 예: 2.1 ÷ (2.1+1.5+0.9+1.8) = 0.33 ≥ 0.3.")
d.frame(s, "02", CH["02"], "② 전처리 — train 기사로만 기준을 고정한다",
        "임베딩·c1·개체를 만들고, train 사건을 묶고, validation-only 기사에 평가용 event_id를 붙인다")
L = 120
for i, (lab, items) in enumerate([("임베딩", [("기사 텍스트", "gray"), ("E5", "blue"), ("768차원 벡터 x", "green")]),
                                  ("카테고리", [("카테고리 이름", "gray"), ("c1 번호", "green")]),
                                  ("개체", [("NER 개체", "gray"), ("train으로 IDF · 흔한 개체 제외", "amber"),
                                           ("기사별 개체", "green")])]):
    y = 290 + i * 58
    text(s, f"Row {i+1} label", L, y, 140, 44, lab, "body-strong", "ink", anchor="m", size=22)
    chain(s, f"Row {i+1} chain", L + 150, y, items, h=42, gap=36, size=20)
sub(s, "Event heading", L, 470, 900, "train 사건 묶는 방법 — 72h 안 그리고 IDF 가중 Jaccard ≥ 0.3")
ox, oy, kx, ky = L - 42, 498, 1.05, 1.05
P = lambda x, y: (ox + x * kx, oy + y * ky)


def grp(name, x, y, w, h, kind, lab):
    gx_, gy_ = P(x, y)
    rect(s, name, gx_, gy_, w * kx, h * ky, fill=f"k-{kind}-fill", line=f"k-{kind}-line", radius=14)
    text(s, name + " label", gx_ + 14, gy_ + 4, 120, 26, lab, "footer", f"k-{kind}-text", weight=700)


grp("Event 1", 40, 50, 340, 88, "purple", "사건 1")
grp("Event 2", 205, 152, 330, 88, "blue", "사건 2")
grp("Event 3", 722, 50, 130, 88, "purple", "사건 3")
NY = {1: 100, 2: 202}
for nm, (x1, x2, row), lab in [("Link a1-a2", (97, 191, 1), "J=0.42 ✓"), ("Link a2-a3", (217, 335, 1), "J=0.33 ✓"),
                               ("Link a4-a5", (265, 479, 2), "J=0.61 ✓")]:
    arrow(s, nm, [P(x1, NY[row]), P(x2, NY[row])], color="success-text", width_pt=2.2, head=False)
    cx, cy = P((x1 + x2) / 2, NY[row])
    text(s, nm + " score", cx - 60, cy + 8, 120, 26, lab, "footer", "success-text", align="c", weight=600)
arrow(s, "No link a3-a6", [P(361, 100), P(767, 100)], color="error", width_pt=1.6, head=False, dash="dash")
cx, cy = P(560, 100)
text(s, "No link label", cx - 230, cy - 32, 460, 26, "개체는 같지만 90h 차이 > 72h → 안 이음", "footer", "error", align="c")
for nm, x, row, kind in [("a1", 84, 1, "purple"), ("a2", 204, 1, "purple"), ("a3", 348, 1, "purple"),
                         ("a6", 780, 1, "purple"), ("a4", 252, 2, "blue"), ("a5", 492, 2, "blue")]:
    cx, cy = P(x, NY[row])
    box(s, f"Node {nm}", cx - 18, cy - 18, 36, 36, nm, fill="canvas", line=f"k-{kind}-line", line_w=2.5,
        style="footer", color="ink", pad=(0, 0, 0, 0), shape=MSO_SHAPE.OVAL, wrap=False)
    tx_, ty_ = P(x, 262)
    dot(s, f"Tick {nm}", tx_, ty_, 5, fill=f"k-{kind}-line")
arrow(s, "Time axis", [P(60, 262), P(870, 262)], color="stone", width_pt=2)
for lab, x in [("0h", 60), ("72h", 406), ("144h", 751)]:
    cx, cy = P(x, 262)
    text(s, f"Axis {lab}", cx - 40, cy + 6, 80, 24, lab, "footer", "slate", align="c")
# Jaccard 계산 예
jy = 840
text(s, "J label A", L, jy, 70, 40, "기사 A", "footer", "slate", anchor="m")
xx = L + 74
for lab, kind in [("A선수 2.1", "green"), ("B구단 1.5", "gray"), ("런던 0.9", "gray")]:
    _, w = chip(s, f"J A {lab}", xx, jy, lab, kind, h=40, size=19); xx += w + 8
xx += 24
text(s, "J label B", xx, jy, 70, 40, "기사 B", "footer", "slate", anchor="m"); xx += 74
for lab, kind in [("A선수 2.1", "green"), ("C구단 1.8", "gray")]:
    _, w = chip(s, f"J B {lab}", xx, jy, lab, kind, h=40, size=19); xx += w + 8
chip(s, "J result", L, jy + 54, "2.1 ÷ (2.1+1.5+0.9+1.8) = 0.33 ≥ 0.3 ✓", "amber", h=40, size=19)
text(s, "J rule", L + 470, jy + 54, 500, 40, "IDF 가중 Jaccard = Σ겹친 idf ÷ Σ합친 idf", "footer", "slate", anchor="m")
# 오른쪽: validation-only event_id
R = 1110
sub(s, "V heading", R, 290, 714, "validation-only 기사의 event_id")
cx = R + 230
kbox(s, "V start", cx - 190, 336, 380, 50, [("validation-only 기사 a", {"weight": 700})], "purple", radius=25,
     style="caption")
arrow(s, "V arrow 1", [(cx, 388), (cx, 412)], width_pt=2)
polygon(s, "V diamond", [(cx, 414), (cx + 220, 470), (cx, 526), (cx - 220, 470)], "canvas", line="k-gray-line",
        label="72h 안 · J ≥ 0.3 사건?", size=19, weight=700)
kbox(s, "V yes", cx + 260, 438, 214 + 10, 66, [("점수 최고 사건", {"weight": 700, "size": 20}),
                                               ("동점 → id 작은 사건", {"size": 18})], "green", style="caption")
arrow(s, "V arrow yes", [(cx + 222, 470), (cx + 254, 470)], width_pt=2)
text(s, "V yes label", cx + 222, 438, 40, 24, "예", "footer", "success-text", weight=600)
kbox(s, "V no", cx - 120, 558, 240, 48, [("새 사건 만들기", {"weight": 700})], "purple", style="caption")
arrow(s, "V arrow no", [(cx, 528), (cx, 552)], width_pt=2)
text(s, "V no label", cx + 10, 526, 70, 24, "아니오", "footer", "slate")
kbox(s, "V result", R, 636, 704, 46, [("기사마다 event_id 하나", {"weight": 700})], "white", style="caption",
     line_w=2)
s.shapes[-1].line.color.rgb = rgb("k-purple-line")
arrow(s, "V arrow r1", [(cx, 608), (cx, 630)], width_pt=2)
arrow(s, "V arrow r2", [(cx + 382, 506), (cx + 382, 630)], width_pt=2)
sub(s, "U heading", R, 708, 714, "이 event_id는 어디에 쓰이나", size=22)
kbox(s, "U source", R, 790, 190, 76, [("validation-only", {"weight": 700, "size": 19}),
                                      ("event_id", {"weight": 700, "size": 19})], "purple", style="caption")
uses = [("RQ-VAE 평가 — c2 일관성", "실제 용도", "green", "success-text"),
        ("RQ-VAE 로더 — 컬럼만", "형식상 필요", "gray", "stone"),
        ("④ SID 부여 — 안 읽게 바꿈", "결정 · 제거", "red", "error")]
for i, (lab, tg, kind, col) in enumerate(uses):
    y = 752 + i * 64
    kbox(s, f"U row {i+1}", R + 260, y, 444, 54, [(lab, {"weight": 700, "size": 19})], kind, style="caption",
         align="l", pad=(16, 0, 16, 0))
    tw_ = text_width(tg, 18, 600) + 36
    chip(s, f"U tag {i+1}", R + 260 + 444 - tw_ - 12, y + 12, tg, "white", h=30, size=18, padx=14, radius=15)
    s.shapes[-1].line.color.rgb = rgb(f"k-{kind}-line")
    curve(s, f"U curve {i+1}", (R + 192, 828), (R + 254, y + 27), color=col, width_pt=2)

# ════════ 4. RQ-VAE ════════
s = d.new(notes="r1 = h − q1(c1), r2 = r1 − q2(c2). Q2 128개, Q3 512개(UNI 설정).")
d.frame(s, "03", CH["03"], "③ RQ-VAE — 카테고리를 빼고 남은 잔차로 c2·c3를 고른다",
        "train 기사만으로 학습 · 인코더 + 코드북 Q1·Q2·Q3 · Q2 시작점은 train 사건별 평균 KMeans")
yc = 470                      # 가운데 줄
kbox(s, "x", 140, yc - 32, 120, 64, [("x", {"weight": 700, "size": 24})], "white")
arrow(s, "A x-enc", [(264, yc), (304, yc)], width_pt=2)
polygon(s, "Encoder", [(310, yc - 44), (450, yc - 26), (450, yc + 26), (310, yc + 44)], "k-blue-fill",
        line="k-blue-line", label="인코더", size=20, color="k-blue-text")
arrow(s, "A enc-h", [(454, yc), (494, yc)], width_pt=2)
kbox(s, "h", 500, yc - 32, 110, 64, [("h", {"weight": 700, "size": 24})], "white")
arrow(s, "A h-m1", [(614, yc), (656, yc)], width_pt=2)
dot(s, "Minus 1", 662 + 22, yc - 22 + 22, 22, fill="canvas", line="ink", line_w=2)
rect(s, "Minus 1 bar", 662 + 12, yc - 22 + 21, 20, 3, fill="ink")
kbox(s, "Q1", 604, 330, 160, 60, [("Q1 카테고리", {"weight": 700, "size": 20})], "gray")
arrow(s, "A q1-m1", [(684, 394), (684, yc - 26)], width_pt=2)
arrow(s, "A q1-c1", [(768, 360), (812, 360)], width_pt=2)
chip(s, "c1", 818, 340, "c1", "green", h=40, size=22, w=84)
arrow(s, "A m1-r1", [(710, yc), (746, yc)], width_pt=2)
kbox(s, "r1", 752, yc - 34, 110, 68, [("r1", {"weight": 700, "size": 22}), ("h − q1", {"size": 18, "color": "slate"})],
     "white")
arrow(s, "A r1-q2", [(866, yc), (906, yc)], width_pt=2)
kbox(s, "Q2", 912, yc - 34, 180, 68, [("Q2 코드북", {"weight": 700, "size": 20}), ("128개", {"size": 18})], "amber")
arrow(s, "A q2-c2", [(1096, yc), (1136, yc)], width_pt=2)
chip(s, "c2", 1142, yc - 20, "c2", "green", h=40, size=22, w=84)
y2 = 600
dot(s, "Minus 2", 980 + 22, y2 - 22 + 22, 22, fill="canvas", line="ink", line_w=2)
rect(s, "Minus 2 bar", 980 + 12, y2 - 22 + 21, 20, 3, fill="ink")
arrow(s, "A q2-m2", [(1002, yc + 38), (1002, y2 - 26)], width_pt=2)
arrow(s, "A r1-m2", [(807, yc + 38), (807, y2), (974, y2)], width_pt=2)
arrow(s, "A m2-r2", [(1028, y2), (1066, y2)], width_pt=2)
kbox(s, "r2", 1072, y2 - 34, 110, 68, [("r2", {"weight": 700, "size": 22}), ("r1 − q2", {"size": 18, "color": "slate"})],
     "white")
arrow(s, "A r2-q3", [(1186, y2), (1226, y2)], width_pt=2)
kbox(s, "Q3", 1232, y2 - 34, 180, 68, [("Q3 코드북", {"weight": 700, "size": 20}), ("512개 (UNI)", {"size": 18})], "green")
arrow(s, "A q3-c3", [(1416, y2), (1456, y2)], width_pt=2)
chip(s, "c3", 1462, y2 - 20, "c3", "green", h=40, size=22, w=84)
text(s, "Diagram caption", 140, 664, 1400, 30, "카테고리를 빼고 남은 **r1**로 c2를, 또 빼고 남은 **r2**로 c3를 고름",
     "caption", "slate")
# z(E) 실험 격자
sub(s, "Exp heading", 140, 716, 1100, "z(E) 공간 실험 — 두 곳을 같은 벡터로 맞춰 쌍으로 (R / H는 다음 미팅에서 결정)", size=22)
cols_x = [140, 400, 660, 920]
for j, h_ in enumerate(["", "Q2 KMeans 초기화", "사건 C2", "어느 사건에 붙일지"]):
    if h_:
        text(s, f"Exp col {j}", cols_x[j], 758, 240, 28, h_, "footer", "ink", align="c", weight=700)
for i, (lab, sub_, kind, a, b) in enumerate([("실험 R", "(잔차, 잔차)", "amber", "평균 r1", "Q2(평균 r1)"),
                                             ("실험 H", "(인코더, 인코더)", "blue", "평균 h", "Q2(평균 h)")]):
    y = 792 + i * 76
    kbox(s, f"Exp {lab}", cols_x[0], y, 230, 64, [(lab, {"weight": 700, "size": 20}), (sub_, {"size": 18})], kind)
    kbox(s, f"Exp {lab} init", cols_x[1], y + 4, 240, 56, [(a, {"weight": 700, "size": 20})], "white")
    s.shapes[-1].line.color.rgb = rgb(f"k-{kind}-line")
    kbox(s, f"Exp {lab} c2", cols_x[2], y + 4, 240, 56, [(b, {"weight": 700, "size": 20})], "white")
    s.shapes[-1].line.color.rgb = rgb(f"k-{kind}-line")
kbox(s, "Exp match", cols_x[3], 792, 240, 140, [("h 코사인", {"weight": 700, "size": 22}), ("두 실험 공통", {"size": 18})],
     "green", line_w=2)
# 오른쪽 메모
R = 1500
sub(s, "Rule heading", R, 330, 320, "학습 규칙", size=22)
text(s, "Rule list", R, 372, 320, 300, ["train 기사만 사용", "c2 = 기사마다 **r1 최근접** (교수님)",
                                        "Q2는 학습 전 한 번 **사건별 평균**으로 KMeans (k 128)",
                                        "UNI: c2 128 · c3 512 · λuniq 0.1"], "caption", "charcoal", bullets=True, gap=8)
text(s, "Exp note", 1180, 800, 640, 140, ["바뀌는 건 **Q2 시작점**과 **물려주는 사건 C2**뿐",
                                          "기사가 어느 사건에 붙는지는 두 실험이 같다",
                                          "실험 H = 지금 코드 그대로 · 실험 R = 새로 구현, **③ 재학습부터**"],
     "caption", "charcoal", bullets=True, gap=6)

# ════════ 5. SID 부여 ════════
s = d.new()
d.frame(s, "04", CH["04"], "④ SID 부여 — 새 기사는 train 사건의 C2를 물려받는다",
        "train 사건 정보(z(E)·C2·개체·시각)를 먼저 준비하고, validation-only 기사는 발행 시각 순으로 train 사건에만 붙인다")
cards = [("A. train 기사", "blue", ["c1 = 카테고리", "**c2 = 기사마다 r1 최근접**", "c3 = r2 최근접",
                                    "c4 = 같은 c1·c2·c3끼리 0, 1, 2…",
                                    "  train은 article_id 순 → 새 기사는 발행 순"]),
         ("B. train 사건 정보 (시작 때 준비)", "amber", None),
         ("C. validation-only 기사", "green", ["발행 시각 순으로 하나씩", "기존 train 사건에 붙어",
                                              "**그 사건의 C2를 물려받음**", "새 사건은 만들지 않음"])]
cwid, cy0, chh = 520, 286, 226
for i, (h_, kind, lines) in enumerate(cards):
    x = 120 + i * (cwid + 64)
    rect(s, f"Card {i+1}", x, cy0, cwid, chh, fill="canvas", line="hairline", radius=12)
    rect(s, f"Card {i+1} bar", x, cy0, cwid, 6, fill=f"k-{kind}-line", radius=3)
    text(s, f"Card {i+1} title", x + 22, cy0 + 18, cwid - 40, 32, h_, "body-strong", "ink", size=22, one_line=True)
    if lines:
        text(s, f"Card {i+1} body", x + 22, cy0 + 56, cwid - 40, 160, lines, "caption", "slate", gap=0, lh=1.32)
    if i < 2:
        arrow(s, f"Card arrow {i+1}", [(x + cwid + 14, cy0 + chh / 2), (x + cwid + 50, cy0 + chh / 2)], width_pt=2)
tb = table(s, "Event summary table", 120 + cwid + 64 + 22, cy0 + 58, [70, 86, 64, 126, 130], 34,
           [["사건", "z(E)", "C2", "개체", "마지막 시각"], ["E7", "평균 h", "41", "A선수…", "3/14"],
            ["E8", "평균 h", "12", "D회사…", "3/15"]], style="footer", head_style="footer", align=["c"] * 5,
           cell_colors={(0, j): "k-amber-fill" for j in range(5)})
text(s, "Card 2 note", 120 + cwid + 64 + 22, cy0 + 168, cwid - 40, 54,
     ["**z(E)·C2** train 기사 h 평균 → Q2 최근접 · 고정",
      "**개체** article_entities 합집합 · **시각** train 최댓값 · 갱신"], "footer", "slate", one_line=True)
# 흐름도 (HTML과 같은 마름모)
sub(s, "Flow heading", 120, 524, 800, "C. 새 기사 하나가 처리되는 순서", size=22)
cx = 430
# HTML과 같은 모양: 마름모 2개 → 결과 상자 하나(아래, 넓게)로 세 갈래가 모두 모인다
kbox(s, "F start", cx - 180, 566, 360, 42, [("validation-only 기사 a", {"weight": 700})], "blue", radius=21,
     style="caption")
arrow(s, "F a1", [(cx, 610), (cx, 624)], width_pt=2)
polygon(s, "F d1", [(cx, 626), (cx + 220, 680), (cx, 734), (cx - 220, 680)], "canvas", line="k-gray-line")
text(s, "F d1 text", cx - 150, 652, 300, 56, ["최근 72h 안 사건 중", ("IDF Jaccard ≥ 0.3 있나?", {"weight": 700})],
     "footer", "ink", align="c", anchor="m", lh=1.2)
kbox(s, "F yes1", cx + 290, 650, 270, 60, [("점수 최고 사건 E", {"weight": 700, "size": 20}),
                                           ("동점 → id 작은 사건", {"size": 18})], "green", style="caption")
arrow(s, "F y1", [(cx + 222, 680), (cx + 284, 680)], width_pt=2)
text(s, "F y1 label", cx + 232, 648, 40, 24, "예", "footer", "success-text", weight=600)
arrow(s, "F n1", [(cx, 736), (cx, 750)], width_pt=2)
text(s, "F n1 label", cx + 12, 730, 70, 22, "아니오", "footer", "slate")
polygon(s, "F d2", [(cx, 752), (cx + 190, 798), (cx, 844), (cx - 190, 798)], "canvas", line="k-gray-line")
text(s, "F d2 text", cx - 120, 774, 240, 50, ["최근 72h 안에", ("사건이 있나?", {"weight": 700})],
     "footer", "ink", align="c", anchor="m", lh=1.2)
kbox(s, "F yes2", cx + 290, 772, 270, 52, [("창 안 h 코사인 최고", {"weight": 700, "size": 20})], "amber", style="caption")
arrow(s, "F y2", [(cx + 192, 798), (cx + 284, 798)], width_pt=2)
text(s, "F y2 label", cx + 232, 766, 40, 24, "예", "footer", "success-text", weight=600)
arrow(s, "F n2", [(cx, 846), (cx, 858)], width_pt=2)
text(s, "F n2 label", cx - 84, 834, 72, 22, "아니오", "footer", "slate", align="r")
kbox(s, "F all", cx - 150, 860, 300, 40, [("전체 사건 h 코사인 최고", {"weight": 700, "size": 19})], "amber",
     style="caption")
rx0, rx1 = cx - 250, cx + 610
kbox(s, "F result", rx0, 912, rx1 - rx0, 44,
     [("**c2 = 사건 E의 C2 물려받기**  ·  그다음 E의 개체·마지막 시각 갱신", {"size": 19})], "white", line_w=2,
     style="caption")
s.shapes[-1].line.color.rgb = rgb("k-blue-line")
arrow(s, "F r-all", [(cx, 902), (cx, 906)], width_pt=2)
arrow(s, "F r-y2", [(cx + 425, 826), (cx + 425, 906)], width_pt=2)
arrow(s, "F r-y1", [(cx + 562, 680), (cx + 590, 680), (cx + 590, 906)], width_pt=2)
# 오른쪽: 갱신 정책
R = 1080
sub(s, "P heading", R, 524, 744, "사건 시각을 갱신하면 뭐가 달라지나", size=22)
ox, oy, kx, ky = R - 6, 520, 0.86, 0.92
P = lambda x, y: (ox + x * kx, oy + y * ky)
for row, (lab, pol, col, ok) in enumerate([("시각 갱신 O", "정책 A · B", "success-text", True),
                                           ("시각 갱신 X", "정책 C", "error", False)]):
    yy = [70, 146][row]
    lx_, ly_ = P(10, yy)
    text(s, f"P row {row+1} label", lx_, ly_ - 4, 150, 26, lab, "footer", col, weight=700)
    text(s, f"P row {row+1} policy", lx_, ly_ + 20, 150, 24, pol, "footer", "slate")
    x1, y1 = P(170, yy)
    rect(s, f"P row {row+1} window", x1, y1, 245 * kx, 26 * ky, fill="k-blue-fill", radius=6)
    if ok:
        x2, _ = P(340, yy)
        rect(s, "P extended window", x2, y1, 245 * kx, 26 * ky, fill="k-green-fill", radius=6)
    sx, sy = P(163, yy + 6)
    rect(s, f"P row {row+1} event", sx, sy, 13, 13, fill="brand-purple")
    ax, ay = P(340, yy + 12)
    dot(s, f"P row {row+1} a", ax, ay, 9, fill="canvas", line="k-blue-line")
    text(s, f"P row {row+1} a label", ax - 80, ay + 14, 160, 24, "a (50h) 붙음", "footer", "slate", align="c")
    bx_, by_ = P(510, yy + 12)
    dot(s, f"P row {row+1} b", bx_, by_, 9, fill=col)
    text(s, f"P row {row+1} b label", bx_ - 90, by_ + 14, 180, 24, "b (100h) 창 안 ✓" if ok else "b (100h) 창 밖 ✗",
         "footer", col, align="c", weight=700)
a1, a2 = P(170, 222), P(800, 222)
arrow(s, "P axis", [a1, a2], color="stone", width_pt=1.6)
for lab, x in [("0h", 170), ("72h", 415), ("144h", 660)]:
    cx_, cy_ = P(x, 222)
    text(s, f"P axis {lab}", cx_ - 36, cy_ + 2, 72, 22, lab, "footer", "slate", align="c")
table(s, "Policy table", R, 762, [120, 160, 190, 274], 47,
      [["정책", "개체로 붙으면", "코사인으로 붙으면", "한 줄 특징"], ["A", "✓ / ✓", "✓ / ✓", "가장 빨리 커짐 · 섞일 위험"],
       ["B  추천", "✓ / ✓", "✗ / ✓", "사건 유지, 개체는 확실할 때만"], ["C", "✓ / ✓", "✗ / ✗", "가장 보수적 · 빨리 밀려남"]],
      style="footer", head_style="footer", bold_cols=(0,), highlight_rows=(2,), align=["l", "c", "c", "l"])

# ════════ 6. 시퀀스·후보 ════════
s = d.new()
d.frame(s, "05", CH["05"], "⑤⑥ 시퀀스·후보 — history를 SID로, 정답 1 + 오답 4",
        "train / validation을 따로 만든다 · 노출 때 읽던 기사도 history 끝에 붙는다")
L = 130
sub(s, "Seq heading", L, 288, 900, "⑤ 시퀀스 생성", size=22)


def tok(name, x, y, v, kind="gray", w=58):
    chip(s, name, x, y, str(v), kind, h=42, size=20, w=w)
    return x + w + 8


def group(name, x, y, label, vals):
    """점선 묶음 (입력 / 후보)."""
    w = 56 + len(vals) * 66 + 8
    rect(s, name, x, y - 8, w, 58, fill=None, line="hairline", radius=10, dash="dash")
    text(s, name + " label", x + 10, y, 50, 42, label, "footer", "slate", anchor="m")
    xx = x + 56
    for v, kind in vals:
        xx = tok(f"{name} {v}", xx, y, v, kind)
    return x + w


rows = [("시작", "history.parquet", None), ("노출 1", "", (["10", "20", "30"], [("40", "green"), "50", "60", "70", "80"], "정답 40")),
        ("노출 2", "40이 history에 추가", (["10", "20", "30", ("40", "blue")], ["55", ("65", "green"), "75", "85", "95"], "정답 65")),
        ("마지막", "id → SID", None)]
for i, (lab, note, data) in enumerate(rows):
    y = 338 + i * 82
    text(s, f"Seq {i+1} label", L, y - 6, 220, 30, lab, "body-strong", "ink", size=22)
    if note:
        text(s, f"Seq {i+1} note", L, y + 24, 220, 24, note, "footer", "slate")
    if i == 0:
        xx = L + 240
        for v in ["10", "20", "30"]:
            xx = tok(f"Start {v}", xx, y, v)
    elif i == 3:
        xx = tok("Last 40", L + 240, y, "40")
        arrow(s, "Last arrow", [(xx + 4, y + 21), (xx + 40, y + 21)], width_pt=2)
        chip(s, "Last SID", xx + 46, y, "3 · 41 · 120 · 0", "purple", h=42, size=20)
    else:
        inp, cand, ans = data
        norm = lambda vs: [(v, "gray") if isinstance(v, str) else v for v in vs]
        xe = group(f"Seq {i+1} input", L + 240, y, "입력", norm(inp))
        xe = group(f"Seq {i+1} cand", xe + 20, y, "후보", norm(cand))
        arrow(s, f"Seq {i+1} arrow", [(xe + 10, y + 21), (xe + 46, y + 21)], width_pt=2)
        chip(s, f"Seq {i+1} answer", xe + 52, y, ans, "green", h=42, size=20)
    if i < 3:
        hline(s, f"Seq rule {i+1}", L, y + 64, 1660, color="hairline-soft")
sub(s, "Neg heading", L, 676, 900, "⑥ 1pos4neg 후보 만들기", size=22)
cands = [("40", "3·41·120", "정답", "green"), ("70", "3·41·120", "정답과 같음 → 제외", "red"),
         ("50", "2·7·33", "오답", "white"), ("60", "3·12·88", "오답", "white"), ("80", "5·90·14", "오답", "white"),
         ("90", "3·41·77", "오답", "white")]
for i, (aid, sid, st, kind) in enumerate(cands):
    x = L + i * (266 + 13)
    kbox(s, f"Cand {aid}", x, 722, 266, 116, [(aid, {"weight": 800, "size": 26}), (sid, {"size": 20}),
                                              (st, {"weight": 700, "size": 19})], kind, line_w=1.5, style="caption")
    if kind == "red":
        hline(s, "Strike 70", x + 112, 752, 42, color="error", width_px=2)
text(s, "Neg note", L, 856, 1660, 32,
     "정답 1 + 오답 4. **정답과 (c1, c2, c3)가 똑같은 오답은 뺌** — 점수에 c4를 안 써서 구분이 안 되기 때문 (교수님 9/18)",
     "caption", "slate", one_line=True)

# ════════ 7. Transformer ════════
s = d.new(notes="후보 점수 = log p(c1) + log p(c2|c1) + log p(c3|c1,c2)")
d.frame(s, "06", CH["06"], "⑦⑧⑨ split · Transformer · 평가 — 확률을 더해 순위를 매긴다",
        "validation을 노출 시각 순으로 반 나누고, T5 방식 모델이 c1→c2→c3를 낸다 · c4는 점수에 안 씀")
L = 130
sub(s, "Split heading", L, 288, 900, "⑦ validation / test 나누기", size=22)
kbox(s, "Split val", L, 334, 820, 48, [("validation (앞 절반)", {"weight": 700, "size": 20})], "blue", radius=8)
kbox(s, "Split test", L + 820, 334, 820, 48, [("test (뒤 절반)", {"weight": 700, "size": 20})], "green", radius=8)
arrow(s, "Split axis", [(L, 402), (L + 1640, 402)], color="stone", width_pt=1.6)
text(s, "Split axis label", L + 1440, 408, 200, 24, "노출 시각", "footer", "slate", align="r")
text(s, "Split note", L, 430, 1400, 30, "1pos4neg를 노출 시각 순으로 정렬해 **노출(impression) 단위**로 반 나눔 — 같은 노출이 양쪽에 섞이지 않음",
     "caption", "slate", one_line=True)
sub(s, "Model heading", L, 488, 900, "⑧ Transformer 학습 → ⑨ 예측·평가", size=22)
text(s, "Tokens label", L, 532, 600, 26, "history (기사마다 SID 토큰)", "footer", "ink", weight=700)
xx = L
for gi, trip in enumerate([("3", "41", "120"), ("5", "12", "77"), ("3", "41", "98")]):
    for v, kind in zip(trip, ["gray", "amber", "green"]):
        chip(s, f"Tok {gi}-{v}", xx, 564, v, kind, h=40, size=19, w=58)
        xx += 64
    xx += 22
arrow(s, "Tok arrow", [(L + 300, 610), (L + 300, 638)], width_pt=2)
kbox(s, "Encoder", L, 642, 640, 52, [("인코더: 사용자 history 읽기", {"weight": 700, "size": 20})], "blue")
arrow(s, "Enc arrow", [(L + 300, 698), (L + 300, 724)], width_pt=2)
kbox(s, "Decoder", L, 728, 640, 52, [("디코더: p(c1) → p(c2|c1) → p(c3|c1,c2)", {"weight": 700, "size": 20})], "blue")
arrow(s, "Dec arrow", [(L + 646, 754), (L + 720, 754)], width_pt=2)
kbox(s, "Fix", L, 820, 640, 52, ["**고칠 것** — c1·c3·c4 크기를 SID 파일과 맞추기 · 경로 맞추기"], "red", style="footer",
     align="l", pad=(18, 0, 18, 0))
R = 900
text(s, "Score label", R, 532, 900, 26, "후보 점수", "footer", "ink", weight=700)
formula(s, "Score formula", R + 110, 528, r"=\log p(c_1)+\log p(c_2\mid c_1)+\log p(c_3\mid c_1,c_2)",
        "후보 점수 = log p(c1) + log p(c2|c1) + log p(c3|c1,c2)", scale=0.9)
for i, (lab, sc) in enumerate([("1위  후보 40 (정답)", 1.2), ("2위  후보 90", 2.5), ("3위  후보 60", 3.1),
                               ("4위  후보 50", 3.8), ("5위  후보 80", 4.4)]):
    y = 600 + i * 54
    text(s, f"Rank {i+1} label", R, y, 260, 36, lab, "caption", "ink" if i == 0 else "charcoal", anchor="m",
         weight=700 if i == 0 else 400)
    bw = 110 + (4.4 - sc) / 3.2 * 390
    rect(s, f"Rank {i+1} bar", R + 270, y + 6, bw, 26, fill="success-text" if i == 0 else "hairline", radius=3)
    text(s, f"Rank {i+1} score", R + 280 + bw, y, 90, 36, f"−{sc}", "footer", "slate", anchor="m")
text(s, "Metrics", R, 880, 920, 30, "순위로 **AUC · MRR · nDCG@5/10 · Top-1** 계산 (c4는 점수에 안 씀)", "caption", "slate")

# ════════ 8. 다시 돌릴 범위 ════════
s = d.new()
d.frame(s, "07", CH["07"], "무엇이 바뀌면 어디부터 다시 돌리나",
        "SID가 한 자리라도 바뀌면 시퀀스·1pos4neg·split 파일이 옛 SID를 들고 있다 — 꼭 새로 만들기")
nums = ["①", "②", "③", "④", "⑤", "⑥", "⑦", "⑧", "⑨"]
cases = [("SID 부여 로직", "정책 A/B/C · 실험 H", 4), ("z(E) 공간 실험", "실험 R — KMeans 입력이 바뀜", 3),
         ("전처리", "임베딩·개체·사건", 2), ("Transformer 설정만", "", 8)]
gx0, cw_, rh = 560, 136, 64
for j, n_ in enumerate(nums):
    text(s, f"Col {n_}", gx0 + j * cw_, 322, cw_ - 10, 44, n_, "body-strong", "slate", align="c", size=34)
for i, (lab, sub_, start) in enumerate(cases):
    y = 380 + i * 96
    text(s, f"Case {i+1} label", 130, y + (0 if sub_ else 16), 420, 34, lab, "body-strong", "ink", size=22)
    if sub_:
        text(s, f"Case {i+1} sub", 130, y + 34, 420, 28, sub_, "footer", "slate")
    for j in range(9):
        st = j + 1
        fill = "error" if st == start else ("brand-blue" if st > start else "k-gray-fill")
        rect(s, f"Case {i+1} cell {nums[j]}", gx0 + j * cw_, y, cw_ - 10, rh, fill=fill, radius=6)
ly = 380 + 4 * 96 + 12
rect(s, "Legend changed", 130, ly + 6, 22, 22, fill="error", radius=4)
text(s, "Legend changed label", 162, ly, 140, 34, "바뀐 곳", "caption", "charcoal", anchor="m")
rect(s, "Legend rerun", 290, ly + 6, 22, 22, fill="brand-blue", radius=4)
text(s, "Legend rerun label", 322, ly, 200, 34, "다시 돌릴 곳", "caption", "charcoal", anchor="m")
text(s, "Note", 130, ly + 52, 1660, 70, ["전처리가 바뀌면 **RQ-VAE 재학습부터**. 실험 H는 KMeans 입력이 지금과 같아서 지금 체크포인트로 **④부터** 돌릴 수 있다."],
     "caption", "slate")

# ════════ 9. 정리 ════════
s = d.new()
d.frame(s, "07", CH["07"], "정한 것과 다음 미팅에서 정할 것",
        "11/18 팀미팅까지 정한 규칙 · 남은 결정 · 담당별 할 일")
L = 130
sub(s, "Done heading", L, 296, 900, "이미 정한 것", size=24)
done = ["Train C2 = 기사마다 r1 최근접 (교수님 결정)",
        "validation-only 기사는 새 사건 없이 **train 사건에만** 배정",
        "1차 개체(72h · J ≥ 0.3) → 2차 코사인 — **코사인은 항상 h**",
        "동점이면 **event_id 숫자가 작은 사건** (전처리와 같음)",
        "validation event_id는 **c2 일관성 채점 전용** (컬럼 유지)",
        "개체는 train·validation 모두 **article_entities**에서 (흔한 개체 제외)",
        "1pos4neg: 정답과 (c1,c2,c3) 같은 오답 제외 (교수님 9/18)",
        "시간 창: EB-NeRD 72h / MIND 24h"]
for i, t_ in enumerate(done):
    y = 346 + i * 70
    text(s, f"Done {i+1} check", L, y, 30, 40, "✓", "body-strong", "success-text", anchor="m")
    text(s, f"Done {i+1}", L + 40, y, 900, 40, t_, "body", "charcoal", anchor="m", one_line=True)
    hline(s, f"Done rule {i+1}", L, y + 52, 940, color="hairline-soft")
R = 1150
sub(s, "Todo heading", R, 296, 680, "다음 미팅에서 정할 것", size=24)
for i, t_ in enumerate(["사건 C2·Q2 KMeans 초기화 공간 — **실험 R / H**", "갱신 정책 **A / B / C** (추천 B)",
                        "비교: c2 일관성 · 코사인으로 붙는 비율 · 추천 성능"]):
    y = 346 + i * 64
    rect(s, f"Todo {i+1} box", R, y + 10, 22, 22, fill="canvas", line="k-amber-line", radius=4, line_w=2)
    text(s, f"Todo {i+1}", R + 36, y, 640, 40, t_, "caption", "charcoal", anchor="m", one_line=True)
sub(s, "Work heading", R, 568, 680, "할 일", size=24)
for i, (who, items) in enumerate([("여민", "개체를 article_entities에서 읽기 (흔한 개체 제외) · IDF는 entity_idf · 숫자 정렬 · 정책·z(E) 옵션 · c1/c3/c4 크기"),
                                  ("민지", "하드코딩 값 정리 · c2 일관성 측정 (AWS) · 개체 합집합 = event_master 확인")]):
    y = 618 + i * 130
    tag(s, f"Work {who}", R, y, who, OWNER[who])
    text(s, f"Work {who} items", R, y + 40, 660, 80, items, "caption", "charcoal")

# ════════ 저장·점검 ════════
d.save(OUT)
probs = check(OUT)
print(f"저장: {OUT}  ({d.n}장)")
for w_ in WARNINGS + probs:
    print("  !", w_)
if d.n != TOTAL:
    print(f"  ! 장 수 {d.n} ≠ TOTAL {TOTAL}")
