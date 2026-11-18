"""EB-NeRD SID 파이프라인 전체 흐름 — 랏코 16:9 덱 (모듈마다 한 장, 9장).

원본: reports/파이프라인_전체흐름.html (11/18 팀미팅 반영본)
실행: python reports/slides/build_pipeline_deck.py  →  reports/slides/파이프라인_전체흐름.pptx
필요: python-pptx, Pillow, Pretendard OTF (~/.fonts/pretendard), pdflatex, pdftocairo
"""
from __future__ import annotations

import sys
from pathlib import Path

from pptx.enum.shapes import MSO_SHAPE

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rakko import (C, MARGIN, W, WARNINGS, Deck, arrow, box, check, dot, formula, hline, pill, rect, table,
                   text, text_width)

OUT = Path(__file__).resolve().parent / "파이프라인_전체흐름.pptx"
TOTAL = 9
FOOT = "기준: 여민 main bcd1168 · 민지 claude/preprocess-fixed-cluster · 11/18 팀미팅 반영"

CH = {"01": "전체 흐름", "02": "전처리", "03": "RQ-VAE", "04": "SID 부여", "05": "시퀀스·후보", "06": "Transformer", "07": "정리"}
MINJI, YEOMIN = "brand-blue-200", "brand-lavender"     # 담당 색 (덱 전체에서 고정)
MINJI_T, YEOMIN_T = "brand-blue-deep", "brand-purple"

d = Deck(TOTAL, FOOT)


def heading(s, name, x, y, w, t):
    return text(s, name, x, y, w, 40, t, "card-title", "brand-blue-deep", one_line=True)


def bullets(s, name, x, y, w, h, items, style="body", gap=10):
    return text(s, name, x, y, w, h, items, style, "charcoal", bullets=True, gap=gap)


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
        "①~⑨ 단계와 담당 — 민지는 데이터 쪽, 여민은 모델 쪽")
steps = [("①", "원천 데이터", "EB-NeRD 기사·노출·history", "data", ""),
         ("②", "전처리", "임베딩·c1·개체, train 사건", MINJI, "확인"),
         ("③", "RQ-VAE 학습", "train 기사만, Q1·Q2·Q3", YEOMIN, ""),
         ("④", "SID 부여", "기사 → c1·c2·c3·c4", YEOMIN, "수정 필요"),
         ("⑤", "시퀀스", "history를 SID 순서로", MINJI, ""),
         ("⑥", "1pos4neg", "정답 1 + 오답 4", MINJI, ""),
         ("⑦", "split", "노출 시각 순 val / test", YEOMIN, ""),
         ("⑧", "Transformer", "c1→c2→c3 생성", YEOMIN, ""),
         ("⑨", "예측·평가", "AUC·MRR·nDCG", YEOMIN, "설정 수정")]
cw, ch_, gx = 280, 130, 56
x0 = MARGIN + (1728 - (5 * cw + 4 * gx)) / 2
pos = {}
for i, (no, name, desc, owner, flag) in enumerate(steps):
    r_, c_ = (0, i) if i < 5 else (1, i - 5)
    x, y = x0 + c_ * (cw + gx), [316, 520][r_]
    pos[i] = (x, y)
    fill = {"data": "surface"}.get(owner, owner)
    tcol = {"data": "ink", MINJI: MINJI_T, YEOMIN: "ink"}[owner]
    box(s, f"Step {no}", x, y, cw, ch_, [(f"{no} {name}", {"weight": 700, "size": 26, "color": tcol}),
                                         (desc, {"size": 20, "color": "charcoal", "gap": 6})],
        fill=fill, radius=16, align="l", anchor="m", pad=(24, 10, 20, 10))
    if flag:
        _, fw = pill(s, f"Step {no} flag", 0, 0, flag)  # 폭 재기용
        s.shapes._spTree.remove(s.shapes[-1]._element)
        pill(s, f"Step {no} flag", x + cw - fw - 14, y - 16, flag,
             fill="canvas" if flag == "확인" else "brand-blue-deep",
             color="brand-blue-deep" if flag == "확인" else "on-primary")
for i in range(8):
    (xa, ya), (xb, yb) = pos[i], pos[i + 1]
    if i == 4:
        arrow(s, "Flow arrow 5-6", [(xa + cw / 2, ya + ch_ + 4), (xa + cw / 2, 488), (xb + cw / 2, 488), (xb + cw / 2, yb - 6)])
    else:
        arrow(s, f"Flow arrow {i+1}-{i+2}", [(xa + cw + 6, ya + ch_ / 2), (xb - 8, yb + ch_ / 2)])
lx = pos[8][0] + cw + 96
for k, (fill, lab) in enumerate([(MINJI, "민지 — 전처리·시퀀스·후보"), (YEOMIN, "여민 — RQ-VAE·SID·Transformer")]):
    rect(s, f"Legend swatch {k+1}", lx, 540 + k * 52, 28, 28, fill=fill, radius=6)
    text(s, f"Legend label {k+1}", lx + 44, 536 + k * 52, 420, 36, lab, "caption", "charcoal", anchor="m")
heading(s, "SID heading", x0, 714, 900, "SID 4자리 — 예: 3 · 41 · 120 · 0")
for i, (k_, v, lab, hero) in enumerate([("c1", "3", "카테고리", False), ("c2", "41", "사건", True),
                                        ("c3", "120", "사건 안 세부", False), ("c4", "0", "충돌 번호", False)]):
    x = x0 + i * (220 + 20)
    box(s, f"Digit {k_}", x, 766, 220, 110,
        [(f"{k_} · {lab}", {"size": 20, "weight": 600, "color": "on-primary" if hero else "steel"}),
         (v, {"size": 44, "weight": 800, "color": "on-primary" if hero else "ink", "lh": 1.1})],
        fill="brand-blue" if hero else "canvas", line=None if hero else "hairline", radius=32 if hero else 16)
bullets(s, "SID notes", x0 + 4 * 240 + 40, 770, 640, 120, [
    "**c2가 사건을 담게** 하는 것이 핵심",
    "새 기사는 train 사건에 붙어 **그 c2를 물려받음**",
    "c4는 앞 3자리가 같을 때 순번 — 점수엔 안 씀"], style="caption", gap=6)

# ════════ 3. 전처리 ════════
s = d.new(notes="IDF(e) = log((N+1)/(df(e)+1)) + 1, N = train 기사 수.\n"
                "J(A,B) = Σ_{e∈A∩B} idf(e) / Σ_{e∈A∪B} idf(e).")
d.frame(s, "02", CH["02"], "② 전처리는 train 기사로만 기준을 고정한다",
        "기사마다 임베딩·c1·개체를 만들고, train 사건을 묶고, validation-only 기사에 평가용 event_id를 붙인다")
L = 120
heading(s, "S1 heading", L, 296, 760, "1. 기사마다 만드는 것")
bullets(s, "S1 list", L, 342, 760, 140, ["**임베딩** 기사 텍스트 → E5 → 768차원 x",
                                         "**c1** 카테고리 이름 → 번호 (train 기준 매핑)",
                                         "**개체** TYPE::name 통일 · 흔한 개체(df ≥ 1%) 제외"], gap=6)
heading(s, "S2 heading", L, 512, 760, "2. train 사건 — 72h 안 · J ≥ 0.3이면 잇는다")
ox, oy, kx, ky = L - 30, 512, 0.86, 1.4
P = lambda x, y: (ox + x * kx, oy + y * ky)


def grp(name, x, y, w, h, fill, lab, labcol):
    gx_, gy_ = P(x, y)
    rect(s, name, gx_, gy_, w * kx, h * ky, fill=fill, radius=16)
    text(s, name + " label", gx_ + 14, gy_ + 8, 120, 26, lab, "footer", labcol, weight=700)


grp("Event 1", 40, 50, 340, 88, YEOMIN, "사건 1", "brand-purple")
grp("Event 2", 205, 152, 330, 88, MINJI, "사건 2", "brand-blue-deep")
grp("Event 3", 722, 50, 130, 88, YEOMIN, "사건 3", "brand-purple")
NODE_Y = {1: 112, 2: 212}
for nm, (x1, x2, row), lab in [("Link a1-a2", (97, 191, 1), "J=0.42"), ("Link a2-a3", (217, 335, 1), "J=0.33"),
                               ("Link a4-a5", (265, 479, 2), "J=0.61")]:
    y_ = NODE_Y[row]
    arrow(s, nm, [P(x1, y_), P(x2, y_)], color="success-text", width_pt=2.2, head=False)
    cx, cy = P((x1 + x2) / 2, y_)
    text(s, nm + " score", cx - 50, cy - 34, 100, 26, lab, "footer", "success-text", align="c", weight=600)
arrow(s, "No link a3-a6", [P(361, 112), P(767, 112)], color="error", width_pt=1.6, head=False, dash="dash")
cx, cy = P(560, 112)
text(s, "No link label", cx - 180, cy - 34, 360, 26, "90h 차이 > 72h → 안 이음", "footer", "error", align="c", weight=600)
for nm, x, row, col in [("a1", 84, 1, "brand-purple"), ("a2", 204, 1, "brand-purple"), ("a3", 348, 1, "brand-purple"),
                        ("a6", 780, 1, "brand-purple"), ("a4", 252, 2, "brand-blue"), ("a5", 492, 2, "brand-blue")]:
    cx, cy = P(x, NODE_Y[row])
    box(s, f"Node {nm}", cx - 19, cy - 19, 38, 38, nm, fill="canvas", line=col, line_w=2, radius=0,
        style="footer", color="ink", pad=(0, 0, 0, 0), shape=MSO_SHAPE.OVAL, wrap=False)
    tx_, ty_ = P(x, 266)
    dot(s, f"Tick {nm}", tx_, ty_, 5, fill=col)
arrow(s, "Time axis", [P(60, 266), P(860, 266)], color="stone", width_pt=2)
for lab, x in [("0h", 60), ("72h", 406), ("144h", 751)]:
    cx, cy = P(x, 266)
    text(s, f"Axis {lab}", cx - 40, cy + 8, 80, 26, lab, "footer", "steel", align="c")
R = 980
heading(s, "S3 heading", R, 296, 840, "3. 개체 가중치와 유사도 (train 기사로만)")
formula(s, "Formula IDF", R, 352, r"\mathrm{idf}(e)=\log\frac{N+1}{\mathrm{df}(e)+1}+1", "IDF 공식", scale=0.95)
formula(s, "Formula Jaccard", R + 400, 344,
        r"J(A,B)=\frac{\sum_{e\in A\cap B}\mathrm{idf}(e)}{\sum_{e\in A\cup B}\mathrm{idf}(e)}", "IDF 가중 Jaccard", scale=0.95)
heading(s, "S4 heading", R, 512, 840, "4. validation-only 기사의 event_id (발행 시각 순)")
box(s, "V start", R, 572, 200, 84, ["기사 a"], fill="surface", radius=16, style="body", color="ink")
box(s, "V question", R + 260, 572, 280, 84, ["72h 안 · J ≥ 0.3", "사건이 있나?"], fill="canvas", line="brand-blue",
    line_w=2, style="caption", color="ink")
box(s, "V yes", R + 600, 572, 240, 84, [("점수 최고 사건", {"weight": 700}),
                                        ("동점 → 작은 id", {"size": 18, "color": "brand-blue-deep", "weight": 600})],
    fill=MINJI, style="caption", color="ink")
box(s, "V no", R + 600, 692, 240, 60, "새 사건 만들기", fill="surface", style="caption", color="ink")
arrow(s, "V arrow 1", [(R + 204, 614), (R + 254, 614)])
arrow(s, "V arrow yes", [(R + 544, 614), (R + 594, 614)])
arrow(s, "V arrow no", [(R + 400, 660), (R + 400, 722), (R + 594, 722)])
text(s, "V label yes", R + 548, 578, 50, 26, "예", "footer", "steel", weight=600)
text(s, "V label no", R + 412, 690, 80, 26, "아니오", "footer", "steel", weight=600)
bullets(s, "V notes", R, 790, 840, 130, ["이 event_id는 **c2 일관성 채점 전용** — 학습·SID·Transformer엔 안 씀",
                                         "④가 지금 읽는 부분은 개체를 파일에서 읽게 고치면 사라짐"],
        style="caption", gap=6)

# ════════ 4. RQ-VAE ════════
s = d.new(notes="r1 = h − q1(c1), r2 = r1 − q2(c2). Q2 128개, Q3 512개(UNI 설정).")
d.frame(s, "03", CH["03"], "③ RQ-VAE는 카테고리를 빼고 남은 잔차로 c2·c3를 고른다",
        "train 기사만으로 학습 — 인코더 + 코드북 Q1·Q2·Q3, Q2 시작점은 사건별 평균 KMeans")
y0 = 316
chain = [("x", "E5 임베딩", 140, "surface"), ("인코더", "", 160, "surface"), ("h", "", 90, "canvas"),
         ("Q1", "카테고리", 150, MINJI), ("r1", "h − q1", 140, "canvas"), ("Q2", "128개", 150, YEOMIN),
         ("r2", "r1 − q2", 140, "canvas"), ("Q3", "512개", 150, YEOMIN)]
gap = 52
x = MARGIN + (1728 - (sum(c[2] for c in chain) + gap * (len(chain) - 1))) / 2
cx_ = {}
for i, (lab, sub, w_, fill) in enumerate(chain):
    paras = [(lab, {"weight": 700, "size": 26, "color": "ink"})] + ([(sub, {"size": 18, "color": "steel"})] if sub else [])
    box(s, f"Chain {lab}", x, y0, w_, 100, paras, fill=fill, line="hairline" if fill == "canvas" else None,
        radius=16, pad=(8, 6, 8, 6))
    cx_[lab] = (x, w_)
    if i:
        arrow(s, f"Chain arrow {i}", [(x - gap + 6, y0 + 50), (x - 8, y0 + 50)])
    x += w_ + gap
for lab, code in [("Q1", "c1"), ("Q2", "c2"), ("Q3", "c3")]:
    xx, ww = cx_[lab]
    arrow(s, f"Code arrow {code}", [(xx + ww / 2, y0 + 104), (xx + ww / 2, y0 + 146)])
    box(s, f"Code {code}", xx + ww / 2 - 46, y0 + 152, 92, 48, code, fill="brand-blue" if code == "c2" else "primary",
        radius=24, style="card-title", color="on-primary", pad=(4, 0, 4, 0))
L = 120
heading(s, "Train heading", L, 590, 760, "1. 학습 규칙")
bullets(s, "Train list", L, 636, 760, 260, [
    "**c2 = 기사마다 r1 최근접 Q2 코드** (교수님 결정, 학습·SID 부여 같음)",
    "Q2는 학습 전 한 번 **train 사건별 평균으로 KMeans** (k 128, seed 42)",
    "UNI 설정: embed 128 · c2 128 · c3 512 · λuniq 0.1 · margin 0.5"], gap=10)
R = 960
heading(s, "Exp heading", R, 590, 860, "2. z(E) 공간 실험 — R / H는 다음 미팅에서 결정")
table(s, "Experiment table", R, 640, [330, 250, 260], 60,
      [["", "Q2 KMeans 초기화", "사건 C2 (물려줄 c2)"],
       ["실험 R (잔차, 잔차)", "사건별 평균 r1", "Q2(평균 r1)"],
       ["실험 H (인코더, 인코더)", "사건별 평균 h", "Q2(평균 h)"]], bold_cols=(0,))
bullets(s, "Exp notes", R, 836, 860, 90, ["사건 매칭은 둘 다 **h 코사인** · R은 **③ 재학습부터**, H는 **④부터**"],
        style="caption")

# ════════ 5. SID 부여 ════════
s = d.new()
d.frame(s, "04", CH["04"], "④ 새 기사는 train 사건에 붙어 그 C2를 물려받는다",
        "train 기사는 기사마다 최근접 코드, validation-only 기사는 발행 시각 순으로 train 사건에만 배정")
text(s, "Train rule", 120, 290, 1700, 40,
     "[[train 기사]]  c1 = 카테고리 · c2 = 기사마다 r1 최근접 · c3 = r2 최근접 — validation 행의 train 기사는 그대로",
     "body", "charcoal", one_line=True)
fx = 120
box(s, "Flow start", fx, 358, 340, 56, "validation-only 기사 a", fill="surface", radius=28, style="body", color="ink",
    wrap=False)
box(s, "Q1", fx, 454, 340, 90, ["최근 72h 안 사건 중", "J ≥ 0.3 있나?"], fill="canvas", line="brand-blue", line_w=2,
    style="caption", color="ink")
arrow(s, "Arrow start-Q1", [(fx + 170, 418), (fx + 170, 448)])
box(s, "R1", fx + 420, 454, 360, 90, [("점수 최고 사건 E", {"weight": 700}),
                                      ("동점 → event_id 작은 사건", {"size": 18, "color": "brand-blue-deep", "weight": 600})],
    fill=MINJI, style="caption", color="ink")
arrow(s, "Arrow Q1 yes", [(fx + 346, 499), (fx + 414, 499)])
text(s, "Q1 yes", fx + 356, 462, 50, 26, "예", "footer", "steel", weight=600)
box(s, "Q2", fx, 598, 340, 90, ["최근 72h 안에", "사건이 있나?"], fill="canvas", line="brand-blue", line_w=2,
    style="caption", color="ink")
arrow(s, "Arrow Q1 no", [(fx + 170, 550), (fx + 170, 592)])
text(s, "Q1 no", fx + 182, 556, 80, 26, "아니오", "footer", "steel", weight=600)
box(s, "R2", fx + 420, 598, 360, 90, "창 안 h 코사인 최고", fill=MINJI, style="caption", color="ink")
arrow(s, "Arrow Q2 yes", [(fx + 346, 643), (fx + 414, 643)])
text(s, "Q2 yes", fx + 356, 606, 50, 26, "예", "footer", "steel", weight=600)
box(s, "R3", fx, 742, 340, 76, "전체 사건 h 코사인 최고", fill=MINJI, style="caption", color="ink")
arrow(s, "Arrow Q2 no", [(fx + 170, 694), (fx + 170, 736)])
text(s, "Q2 no", fx + 182, 700, 80, 26, "아니오", "footer", "steel", weight=600)
box(s, "Result", fx + 420, 742, 360, 76, [("c2 = 사건 E의 C2", {"weight": 700, "color": "on-primary"})],
    fill="brand-blue-deep", radius=16, style="body")
arrow(s, "Arrow R3-result", [(fx + 346, 780), (fx + 414, 780)])
arrow(s, "Arrow R1-result", [(fx + 786, 499), (fx + 820, 499), (fx + 820, 780), (fx + 786, 780)])
arrow(s, "Arrow R2-result", [(fx + 786, 643), (fx + 820, 643)], head=False)
text(s, "After", fx + 420, 832, 460, 28, "→ 그다음 E의 개체·마지막 시각 갱신", "footer", "steel")
R = 1000
heading(s, "Summary heading", R, 358, 820, "사건 요약표 — train 기사로 한 번 계산")
table(s, "Event summary table", R, 404, [100, 150, 90, 300, 180], 48,
      [["사건", "매칭용", "C2", "개체", "마지막 시각"], ["E7", "평균 h", "41", "A선수 · …", "3/14"],
       ["E8", "평균 h", "12", "D회사 · …", "3/15"]], align=["c", "c", "c", "l", "c"])
text(s, "Summary note", R, 552, 820, 28, "평균 h·C2는 고정, 개체·마지막 시각은 새 기사가 붙을 때 갱신", "caption", "steel",
     one_line=True)
heading(s, "Policy heading", R, 610, 820, "갱신 정책 — 코사인으로 붙었을 때 (개체로 붙으면 둘 다 갱신)")
table(s, "Policy table", R, 656, [130, 190, 500], 48,
      [["정책", "개체 / 시각", "특징"], ["A", "✓ / ✓", "가장 빨리 커짐 — 엉뚱한 개체 섞일 위험"],
       ["B  추천", "✗ / ✓", "사건은 살아 있고, 개체는 확실할 때만"],
       ["C", "✗ / ✗", "가장 보수적 — 창 밖으로 빨리 밀려남"]],
      bold_cols=(0,), highlight_rows=(2,), align=["l", "c", "l"])
text(s, "C4 rule", R, 868, 820, 32, "[[c4]]  앞 3자리가 같으면 순번 (train 먼저) — 3·41·120·0, ·1, ·2",
     "caption", "charcoal", one_line=True)

# ════════ 6. 시퀀스 + 1pos4neg ════════
s = d.new()
d.frame(s, "05", CH["05"], "⑤⑥ history를 SID로 바꾸고 정답 1·오답 4로 묻는다",
        "⑤ 시퀀스와 ⑥ 1pos4neg — train과 validation을 따로 만든다")
lx = MARGIN + 24
heading(s, "Seq heading", lx, 352, 760, "1. 시퀀스 생성 — 노출마다 history가 자란다")


def tokens(prefix, x, y, ids, fill, col="ink", w_=64):
    for j, t_ in enumerate(ids):
        box(s, f"{prefix} {t_}", x + j * (w_ + 8), y, w_, 44, str(t_), fill=fill, radius=8, style="table", color=col,
            pad=(2, 0, 2, 0))


rows_ = [("노출 1", [10, 20, 30], [40, 50, 60, 70, 80], 40), ("노출 2", [10, 20, 30, 40], [55, 65, 75, 85, 95], 65)]
for i, (lab, hist, cand, ans) in enumerate(rows_):
    y = 420 + i * 150
    text(s, f"Seq {i+1} label", lx, y + 6, 100, 34, lab, "body-strong", "ink")
    text(s, f"Seq {i+1} input label", lx + 110, y - 30, 200, 26, "입력 history", "footer", "steel")
    tokens(f"Seq {i+1} hist", lx + 110, y, hist, "surface")
    text(s, f"Seq {i+1} cand label", lx + 110, y + 54, 200, 26, "후보", "footer", "steel")
    for j, c_ in enumerate(cand):
        box(s, f"Seq {i+1} cand {c_}", lx + 160 + j * 72, y + 50, 64, 40, str(c_),
            fill=MINJI if c_ == ans else "canvas", line=None if c_ == ans else "hairline", radius=8, style="caption",
            color="brand-blue-deep" if c_ == ans else "charcoal", pad=(2, 0, 2, 0))
text(s, "Seq note", lx, 728, 760, 70, ["정답(파란 칸)은 다음 노출의 history 끝에 붙는다",
                                       "마지막에 기사 id를 SID로 바꿔 저장 — 40 → 3 · 41 · 120 · 0"], "caption", "charcoal", gap=4)
rx = 1040
heading(s, "Neg heading", rx, 352, 780, "2. 1pos4neg — 앞 3자리가 정답과 같은 오답은 뺀다")
cands = [("40", "3 · 41 · 120", "정답", "brand-blue-deep"), ("70", "3 · 41 · 120", "정답과 같음 → 제외", "error"),
         ("50", "2 · 7 · 33", "오답", "charcoal"), ("60", "3 · 12 · 88", "오답", "charcoal"),
         ("80", "5 · 90 · 14", "오답", "charcoal"), ("90", "3 · 41 · 77", "오답", "charcoal")]
for i, (aid, sid, st, col) in enumerate(cands):
    y = 412 + i * 56
    if i == 0:
        rect(s, "Answer highlight", rx - 12, y - 6, 720, 52, fill=MINJI, radius=8)
    text(s, f"Cand {aid} id", rx, y, 80, 40, aid, "body-strong", "ink", anchor="m")
    text(s, f"Cand {aid} sid", rx + 100, y, 260, 40, sid, "body", "ink" if i < 2 else "charcoal", anchor="m")
    text(s, f"Cand {aid} status", rx + 380, y, 320, 40, st, "caption", col, anchor="m", weight=600 if i < 2 else 400)
    if i == 1:
        hline(s, "Strike 70", rx, y + 20, 340, color="error", width_px=2)
text(s, "Neg note", rx, 760, 780, 70, ["점수에 c4를 안 써서 앞 3자리가 같으면 구분이 안 된다",
                                       "정답 1 + 오답 4로 채운다 (교수님 9/18)"], "caption", "charcoal", gap=4)

# ════════ 7. Transformer ════════
s = d.new(notes="score = log p(c1) + log p(c2|c1) + log p(c3|c1,c2)")
d.frame(s, "06", CH["06"], "⑦⑧⑨ 후보마다 c1→c2→c3 확률을 더해 순위를 매긴다",
        "⑦ split · ⑧ Transformer(T5 방식) · ⑨ 평가 — c4는 점수에 쓰지 않는다")
lx = MARGIN + 24
heading(s, "Split heading", lx, 352, 640, "1. validation / test 나누기")
rect(s, "Split val", lx, 408, 300, 56, fill=MINJI, radius=8)
text(s, "Split val label", lx, 408, 300, 56, "validation (앞 절반)", "caption", "brand-blue-deep", align="c", anchor="m", weight=600)
rect(s, "Split test", lx + 308, 408, 300, 56, fill=YEOMIN, radius=8)
text(s, "Split test label", lx + 308, 408, 300, 56, "test (뒤 절반)", "caption", "brand-purple", align="c", anchor="m", weight=600)
arrow(s, "Split axis", [(lx, 488), (lx + 620, 488)], color="stone", width_pt=2)
text(s, "Split axis label", lx + 420, 496, 200, 26, "노출 시각", "footer", "steel", align="r")
text(s, "Split note", lx, 540, 640, 70, ["노출 시각 순으로 정렬해 **노출 단위**로 반 나눔",
                                         "같은 노출이 양쪽에 섞이지 않는다"], "caption", "charcoal", gap=4)
heading(s, "Score heading", lx, 660, 640, "2. 점수")
formula(s, "Score formula", lx, 712, r"\mathrm{score}=\log p(c_1)+\log p(c_2\mid c_1)+\log p(c_3\mid c_1,c_2)",
        "score = log p(c1) + log p(c2|c1) + log p(c3|c1,c2)", scale=1.0, max_w=660)
text(s, "Score note", lx, 790, 660, 34, "인코더가 history의 SID 토큰을 읽고, 디코더가 c1→c2→c3를 낸다", "caption", "charcoal")
rx = 960
heading(s, "Rank heading", rx, 352, 860, "3. 예시 — 후보 5개의 순위")
table(s, "Rank table", rx, 404, [110, 260, 200, 230], 60,
      [["순위", "후보", "SID (앞 3자리)", "score"], ["1", "40 (정답)", "3 · 41 · 120", "−1.2"], ["2", "90", "3 · 41 · 77", "−2.5"],
       ["3", "60", "3 · 12 · 88", "−3.1"], ["4", "50", "2 · 7 · 33", "−3.8"], ["5", "80", "5 · 90 · 14", "−4.4"]],
      highlight_rows=(1,), align=["c", "l", "c", "r"])
text(s, "Metrics", rx, 790, 860, 36, "[[평가]]  이 순위로 AUC · MRR · nDCG@5/10 · Top-1 계산", "body", "charcoal")

# ════════ 8. 다시 돌릴 범위 ════════
s = d.new()
d.frame(s, "07", CH["07"], "바뀐 단계부터 ⑨까지 다시 돌린다",
        "SID가 한 자리라도 바뀌면 ⑤ 시퀀스·⑥ 1pos4neg·⑦ split 파일이 옛 SID를 들고 있다")
nums = ["①", "②", "③", "④", "⑤", "⑥", "⑦", "⑧", "⑨"]
cases = [("전처리 수정 (임베딩·개체·사건)", 2), ("z(E) 실험 R (KMeans 입력이 바뀜)", 3),
         ("z(E) 실험 H · SID 부여 로직 (정책 A/B/C 등)", 4), ("Transformer 설정만", 8)]
gx0, gy0, cw_, rh = 820, 372, 104, 84
for j, n_ in enumerate(nums):
    text(s, f"Col {n_}", gx0 + j * cw_, gy0 - 44, cw_ - 8, 36, n_, "card-title", "steel", align="c")
for i, (lab, start) in enumerate(cases):
    y = gy0 + i * (rh + 16)
    text(s, f"Case {i+1} label", MARGIN + 24, y, 680, rh, lab, "body", "ink", anchor="m", weight=600, one_line=True)
    for j in range(9):
        st = j + 1
        fill = "brand-blue-deep" if st == start else (MINJI if st > start else "surface-soft")
        rect(s, f"Case {i+1} cell {nums[j]}", gx0 + j * cw_, y + 8, cw_ - 8, rh - 16, fill=fill, radius=8)
lgy = gy0 + 4 * (rh + 16) + 20
for k_, (fill, lab) in enumerate([("brand-blue-deep", "바뀐 곳"), (MINJI, "다시 돌릴 곳")]):
    rect(s, f"Legend {k_+1}", gx0 + k_ * 220, lgy, 28, 28, fill=fill, radius=6)
    text(s, f"Legend {k_+1} label", gx0 + k_ * 220 + 40, lgy - 4, 170, 36, lab, "caption", "charcoal")
text(s, "Takeaway", MARGIN + 24, 880, 1680, 40,
     "[[핵심]]  실험 H는 KMeans 입력이 지금과 같아서 지금 체크포인트로 ④부터 돌릴 수 있다", "body", "charcoal", one_line=True)

# ════════ 9. 정리 ════════
s = d.new()
d.frame(s, "07", CH["07"], "정한 것과 다음 미팅에서 정할 것",
        "11/18 팀미팅까지 정한 규칙, 남은 결정, 담당별 할 일")
cols = [("1. 정한 것", "brand-blue", [
            "Train c2 = 기사마다 r1 최근접 (교수님)",
            "새 기사는 **train 사건에만** 배정",
            "사건 매칭 코사인은 **항상 h**",
            "동점이면 **event_id가 작은 사건**",
            "validation event_id는 **채점 전용**",
            "정답과 앞 3자리 같은 오답 제외"]),
        ("2. 다음 미팅에서 정할 것", "hairline", [
            "사건 C2·Q2 KMeans 초기화 공간 — **실험 R / H**",
            "갱신 정책 **A / B / C** (추천 B)",
            "비교 지표: c2 일관성 · 코사인으로 붙는 비율 · 추천 성능"]),
        ("3. 할 일", "hairline", [
            "[[여민]] 개체를 파일에서 읽기 · IDF Jaccard · 숫자 정렬 · 정책·z(E) 옵션 · c1/c3/c4 크기",
            "[[민지]] main.py에 개체 생성 연결 · 하드코딩 값 정리 · c2 일관성 측정 (AWS)"])]
for i, (h_, rule, items) in enumerate(cols):
    x = MARGIN + i * (552 + 36)
    hline(s, f"Column {i+1} rule", x, 352, 552, color=rule, width_px=3 if i == 0 else 1)
    heading(s, f"Column {i+1} heading", x, 376, 552, h_)
    bullets(s, f"Column {i+1} list", x, 432, 552, 470, items, style="body", gap=14)

# ════════ 저장·점검 ════════
d.save(OUT)
probs = check(OUT)
print(f"저장: {OUT}  ({d.n}장)")
for w_ in WARNINGS + probs:
    print("  !", w_)
if d.n != TOTAL:
    print(f"  ! 장 수 {d.n} ≠ TOTAL {TOTAL}")
