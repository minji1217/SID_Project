"""v2 재실험 결과 — 랏코 16:9 덱 (14장).

원본: reports/v2_재실험_결과_3seed.html (reports/analysis/build_v2_report.py). 숫자는 그 보고서와 같다.
실행: python reports/slides/build_v2_deck.py  →  reports/slides/v2_재실험_결과.pptx
필요: python-pptx, Pillow, Pretendard OTF (~/.fonts/pretendard), pdflatex, pdftocairo
"""
from __future__ import annotations

import sys
from pathlib import Path
from statistics import mean, stdev

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rakko import (KIND, MARGIN, WARNINGS, Deck, arrow, box, check, chip, dot, formula, hline, pill, rect,
                   table, text, text_width)

OUT = Path(__file__).resolve().parent / "v2_재실험_결과.pptx"
TOTAL = 14
FOOT = "기준: 여민 코드 22028ba + 우리 설정 · Transformer seed 42 · 123 · 2026 · 보고서 v2_재실험_결과_3seed.html"
CH = {"01": "실험 구성", "02": "RQ-VAE · SID", "03": "Transformer 결과", "04": "진단", "05": "인기 · 나이",
      "06": "TIGER 비교", "07": "정리"}
COL = {"V1": "stone", "H": "brand-blue", "R": "brand-purple"}   # 실험마다 한 색 (덱 전체 같음)
RUNS = ["V1", "H", "R"]
SEEDS = [42, 123, 2026]

# ── 데이터 (보고서와 같은 값) ─────────────────────────────────
TOP1 = {"V1": [26.16, 26.79, 26.76], "H": [27.51, 27.81, 27.70], "R": [27.67, 27.17, 27.69]}
AUC = {"V1": [0.5686, 0.5831, 0.5789], "H": [0.5836, 0.5810, 0.5889], "R": [0.5888, 0.5810, 0.5852]}
MRR = {"V1": [0.5101, 0.5187, 0.5168], "H": [0.5223, 0.5224, 0.5252], "R": [0.5249, 0.5197, 0.5237]}
NDCG = {"V1": [0.6308, 0.6375, 0.6360], "H": [0.6401, 0.6401, 0.6424], "R": [0.6422, 0.6381, 0.6412]}
POP = {"V1": 42.94, "H": 42.93, "R": 42.97}
FUSED = {"V1": [43.41, 43.73, 43.84], "H": [44.21, 44.14, 44.37], "R": [44.23, 44.35, 44.09]}
GAIN = {n: [f - POP[n] for f in FUSED[n]] for n in RUNS}
JOINT = {"V1": 42.50, "H": 42.92, "R": 43.07}       # 함께 학습, 가중치 고정, seed 42
TIGER = {"V1": 16.69, "H": 17.30, "R": 17.77}       # 공개 TIGER, seed 42
COS_BINS = ["< 0.25", "0.25~0.35", "0.35~0.45", "≥ 0.45"]
COS_TOP1 = {"H": [4.42, 19.76, 27.37, 31.92], "R": [1.36, 23.38, 26.77, 34.85]}

d = Deck(TOTAL, FOOT)


def ms(v, f="{:.2f}"):
    return f"{f.format(mean(v))} ± {f.format(stdev(v))}"


def sub(s, name, x, y, w, t, color="brand-blue-deep", size=26):
    return text(s, name, x, y, w, 40, t, "card-title", color, size=size, one_line=True)


def bullets(s, name, x, y, w, h, items, style="body", gap=14):
    return text(s, name, x, y, w, h, items, style, "charcoal", bullets=True, gap=gap)


def legend(s, name, x, y, names, size=20):
    for i, n in enumerate(names):
        rect(s, f"{name} swatch {n}", x, y + 6, 18, 18, fill=COL[n], radius=4)
        text(s, f"{name} label {n}", x + 26, y, 120, 30, n, "caption", "charcoal", size=size, weight=600, anchor="m",
             lh=1.0)
        x += 26 + text_width(n, size, 600) + 36


def vbars(s, name, x, y, w, h, cats, series, ymax, ticks, ref=None, ref_label="", fmt="{:.2f}", bw_max=64,
          cat_size=20):
    """세로 묶음 막대. series: [(이름, 값 목록)]. x,y = 그림 왼쪽 위 (축 숫자 포함)."""
    L = 52                              # 축 숫자 자리
    R = 120 if ref is not None else 0   # 오른쪽 = 기준선 라벨 자리
    px, pw, ph = x + L, w - L - R, h - 40   # 아래 40px = 범주 이름
    base = y + ph
    Y = lambda v: base - v / ymax * ph
    for t in ticks:
        hline(s, f"{name} grid {t}", px, Y(t), pw, color="hairline-soft")
        text(s, f"{name} tick {t}", x, Y(t) - 14, L - 10, 28, f"{t:g}", "footer", "muted", align="r")
    if ref is not None:
        rect(s, f"{name} ref line", px, Y(ref) - 1, pw + 8, 2, fill="steel")
        text(s, f"{name} ref label", px + pw + 14, Y(ref) - 16, R - 14, 32, ref_label, "footer", "steel",
             anchor="m", lh=1.0)
    ng, ns = len(cats), len(series)
    gw = pw / ng
    bw = min(bw_max, (gw - 36) / ns - 6)
    for gi, c in enumerate(cats):
        gx = px + gi * gw + (gw - ns * (bw + 6) + 6) / 2
        for si, (n, vals) in enumerate(series):
            v = vals[gi]
            bx = gx + si * (bw + 6)
            rect(s, f"{name} bar {c} {n}", bx, Y(v), bw, base - Y(v), fill=COL[n])
            text(s, f"{name} value {c} {n}", bx - 24, Y(v) - 30, bw + 48, 28, fmt.format(v), "footer", "ink",
                 align="c", weight=600)
        text(s, f"{name} category {c}", px + gi * gw, base + 8, gw, 32, c, "caption", "charcoal", align="c",
             size=cat_size, weight=600)
    hline(s, f"{name} axis", px, base, pw, color="steel", width_px=2)


def stepcard(s, name, x, y, w, h, no, title, desc, tags):
    """번호 단계 카드 (흰 카드 + 데이터 꼬리표)."""
    rect(s, f"{name} card", x, y, w, h, fill="canvas", line="hairline", radius=16)
    text(s, f"{name} title", x + 24, y + 20, w - 48, 36, f"{no}. {title}", "card-title", "brand-blue-deep",
         size=24, weight=700, one_line=True)
    text(s, f"{name} desc", x + 24, y + 62, w - 48, h - 120, desc, "footer", "slate", size=19)
    tx = x + 24
    for t, kind in tags:
        _, tw = chip(s, f"{name} tag {t}", tx, y + h - 50, t, kind, h=32, size=18, padx=14, radius=16)
        tx += tw + 8


DATA = {"train": "green", "validation": "blue", "test": "amber"}

# ════════ 1. 표지 ════════
s = d.new("v2 재실험 결과 보고. 숫자는 reports/v2_재실험_결과_3seed.html 과 같다.")
text(s, "Deck label", MARGIN, 300, 1200, 32, "SID Project · v2 재실험 결과", "chapter", "brand-blue-deep")
text(s, "Cover title", MARGIN, 352, 1600, 260, ["새 SID · 실험 H vs R", "인기 · 나이 결합과 TIGER 비교"],
     "cover-title", "ink")
text(s, "Cover subtitle", MARGIN, 664, 1600, 44,
     "여민 코드 + 우리 설정으로 RQ-VAE부터 Transformer까지 다시 돌린 결과 (Transformer seed 3개)",
     "body-lg", "steel", one_line=True)
for i, (lab, val) in enumerate([("작성", "민지"), ("날짜", "2026. 11. 24"),
                                ("기준", "여민 22028ba · RQ-VAE UNI · Transformer V1 설정 · fp32")]):
    x = MARGIN + [0, 220, 500][i]
    text(s, f"Meta label {i+1}", x, 880, 600, 26, lab, "footer", "muted")
    text(s, f"Meta value {i+1}", x, 910, 1100, 32, val, "table", "ink", weight=600)

# ════════ 2. 목차 ════════
s = d.new()
text(s, "Title", MARGIN, 112, 1728, 72, "목차", "title", "ink", anchor="m")
text(s, "Subtitle", MARGIN, 192, 1728, 40, "실험 구성부터 결론까지 일곱 부분", "subtitle", "steel", anchor="m")
hline(s, "Header rule", MARGIN, 256, 1728)
agenda = [("01", "실험 구성", "V1 · H · R은 무엇이 다른가", "3–4"),
          ("02", "RQ-VAE · SID", "코드북 사용과 새 기사 붙이기", "5–6"),
          ("03", "Transformer 결과", "seed 3개 test 성능", "7"),
          ("04", "진단", "어디서 막히나", "8"),
          ("05", "인기 · 나이", "나중에 더하기 vs 함께 학습", "9–11"),
          ("06", "TIGER 비교", "공개 재현 구현과 비교", "12"),
          ("07", "정리", "결론과 다음 단계", "13")]
for i, (no, name, q, pg) in enumerate(agenda):
    col, row = (0, i) if i < 4 else (1, i - 4)
    x, y = [132, 945][col], 330 + row * 125
    text(s, f"Agenda {no} number", x, y, 110, 70, no, "stat-lg", "brand-blue", size=56, weight=800)
    text(s, f"Agenda {no} name", x + 120, y + 2, 560, 40, name, "heading-md", "ink", one_line=True)
    text(s, f"Agenda {no} question", x + 120, y + 46, 560, 32, f"{q} · p.{pg}", "caption", "steel", one_line=True)
d.footer(s)

# ════════ 3. 실험 구성 ════════
s = d.new("V1 27.07% → 26.16%: 예전 숫자는 후보를 섞기 전 데이터(정답이 항상 첫 자리)와 AMP(bf16) 동점 처리로 부풀려졌다.")
d.frame(s, "01", CH["01"], "세 실험은 SID만 다르고 Transformer 조건은 같다",
        "V1은 예전 SID 그대로, H · R은 새 전처리로 RQ-VAE부터 다시 만들었다")
table(s, "Setup table", MARGIN, 336, [220, 440, 330, 488, 250], 80,
      [["실험", "SID", "Q2 초기화", "validation-only 사건 C2", "돌린 단계"],
       ["V1 (기준)", "예전 전처리 + UNI\n(예전에 만든 SID 그대로)", "사건별 h 평균\n(예전 사건)", "대부분 기사 자기 사건의\nh 평균 → 가장 가까운 Q2", "Transformer만"],
       ["H", "새 전처리 + UNI 재학습", "사건별 h 평균", "붙은 사건 h 평균\n→ 가장 가까운 Q2", "RQ-VAE부터 전부"],
       ["R", "새 전처리 + UNI 재학습", "사건별 r1 평균", "붙은 사건 r1 평균\n→ 가장 가까운 Q2", "RQ-VAE부터 전부"]],
      highlight_rows=(2, 3), align=["l", "l", "l", "l", "l"], bold_cols=(0,))
bullets(s, "Common list", MARGIN, 690, 1600, 150,
        ["**세 실험 공통** — 여민 Transformer + V1 설정(history 50, d256 · 8 head · 2층, lr 5e-5, dropout 0), fp32, 후보 순서를 섞은 1pos4neg",
         "**H · R 공통** — 여민 코드 22028ba, 새 기사는 Jaccard ≥ 0.3 → h 코사인으로 Train 사건에 붙임, 갱신 정책 B, RQ-VAE seed 42"],
        "body", gap=16)
text(s, "Note", MARGIN, 870, 1728, 40,
     "[[참고]]  V1 기준 27.07% → 26.16%: 예전 숫자는 후보를 안 섞은 데이터와 AMP 동점 처리로 부풀려져 있었다",
     "body", "charcoal", one_line=True)

# ════════ 4. V1 vs H ════════
s = d.new("숫자는 예전 normalize_v2 결과 파일(experiment/entity-linking-event 브랜치)과 새 전처리 결과를 직접 세어 확인.")
d.frame(s, "01", CH["01"], "V1과 H는 전처리와 새 기사 규칙이 다르다",
        "SID 만드는 방식(Q2 초기화 · 사건 C2 = h 평균)은 같고, 넣은 사건 정보가 다르다")
table(s, "Diff table", MARGIN, 316, [320, 704, 704], 70,
      [["", "V1 (예전 전처리 normalize_v2)", "H (새 전처리)"],
       ["Train 기사 정의", "train history + 현재 기사 + 클릭 기사", "여기에 [[노출 목록(inview)]] 추가"],
       ["학습 기사 / Train 사건", "9,738 / 9,145", "10,419 / 9,794"],
       ["validation-only 기사", "3,122", "2,741 (노출 기사가 Train으로)"],
       ["새 기사가 붙는 곳", "Train 사건에 붙은 기사 32개(1%)\n나머지는 새 사건 3,001개 (대부분 기사 1개)", "[[모두 Train 사건]]: Jaccard ≥ 0.3 (72h)\n→ h 코사인 최대 (72h) → 전체 사건"],
       ["새 기사의 C2", "사실상 기사 자기 h → 가장 가까운 Q2", "붙은 Train 사건의 C2를 물려받음"]],
      align=["l", "l", "l"], bold_cols=(0,))
text(s, "Takeaway", MARGIN, 860, 1728, 40,
     "[[핵심]]  V1 → H의 약 +1%p는 노출 기사를 Train에 넣은 것과 새 기사를 Train 사건에 붙인 것을 합친 효과 (따로 떼어 보지 않음)",
     "body", "charcoal", one_line=True)

# ════════ 5. RQ-VAE ════════
s = d.new("R의 Q2 코드 사용: seed 42에서 59개, seed 123에서 93개 (128개 중). H s123·s2026은 기존 ver2 모델.")
d.frame(s, "02", CH["02"], "R은 Q2 코드 128개 중 절반 정도만 쓴다",
        "H는 128개 모두 사용 · 복원 손실은 R이 약 0.9% 높고 늦게 수렴한다")
sub(s, "Chart caption", MARGIN, 330, 800, "1. Train 기사의 c2 코드 종류 (128개 중)")
bars = [("H s42", "H", 128), ("H s123", "H", 128), ("H s2026", "H", 128), ("R s42", "R", 59), ("R s123", "R", 93)]
X0, XW = 250, 560
for i, (lab, n, v) in enumerate(bars):
    y = 400 + i * 74
    text(s, f"C2 use label {lab}", MARGIN, y, 140, 44, lab, "table", "charcoal", anchor="m", weight=600)
    rect(s, f"C2 use bar {lab}", X0, y + 6, XW * v / 128, 32, fill=COL[n], radius=6)
    text(s, f"C2 use value {lab}", X0 + XW * v / 128 + 12, y, 80, 44, str(v), "table", "ink", anchor="m", weight=700)
for t in (0, 32, 64, 96, 128):
    text(s, f"C2 use tick {t}", X0 + XW * t / 128 - 30, 776, 60, 28, str(t), "footer", "muted", align="c")
hline(s, "C2 use axis", X0, 770, XW, color="steel", width_px=2)
text(s, "C2 use note", MARGIN, 814, 800, 28, "H s123 · s2026은 기존 ver2 모델", "footer", "muted")
RX = 1010
sub(s, "Table caption", RX, 330, 814, "2. RQ-VAE 학습 (validation 복원 손실 최소 checkpoint)")
table(s, "RQ table", RX, 392, [200, 260, 170, 184], 60,
      [["모델", "best_rec", "best epoch", "끝난 epoch"],
       ["H s42", "0.150854", "190", "390"], ["R s42", "0.152190", "300", "500"], ["R s123", "0.152124", "470", "670"]],
      align=["l", "r", "r", "r"], bold_cols=(0,))
sub(s, "Cause heading", RX, 666, 814, "원인으로 보이는 것", color="ink", size=24)
bullets(s, "Cause list", RX, 712, 814, 200,
        ["학습 전 **r1 = h − Q1[카테고리]**의 사건 평균으로 Q2를 KMeans 초기화",
         "그때 Q1은 랜덤이고 h보다 약 2.5배 커서 중심이 카테고리 쪽으로 몰림 → 많은 코드가 끝까지 안 쓰임"],
        "caption", gap=12)

# ════════ 6. SID 생성 ════════
s = d.new("R1 코드에 H checkpoint를 넣으면 H와 같은 사건에 100% 붙음 → 코사인은 둘 다 h로 계산됨을 확인. H와 R은 모델이 달라 실제로는 55%만 같은 사건.")
d.frame(s, "02", CH["02"], "새 기사는 89%가 h 코사인으로 사건에 붙는다",
        "validation-only 기사 2,741개가 Train 사건에 붙는 순서와 건수 (H · R)")
cards = [("1", "Jaccard ≥ 0.3", "72시간 안, 개체가 겹치는 사건", "H 34 · R 35개 (1%)", "white"),
         ("2", "h 코사인 최대", "72시간 안, 1이 없을 때", "H 2,439 · R 2,434개 (89%)", "blue"),
         ("3", "전체 사건 코사인", "72시간 안에 사건이 없을 때", "H 268 · R 272개", "white")]
CW, GAP = 470, 159
cx0 = MARGIN + (1728 - (3 * CW + 2 * GAP)) / 2
for i, (no, t, cond, cnt, kind) in enumerate(cards):
    x = cx0 + i * (CW + GAP)
    f, l, tc = KIND[kind]
    rect(s, f"Rule {no} card", x, 340, CW, 170, fill=f"k-{kind}-fill", line=f"k-{kind}-line", radius=16,
         line_w=2 if kind == "blue" else 1)
    text(s, f"Rule {no} title", x + 28, 362, CW - 56, 40, f"{no}. {t}", "card-title", "brand-blue-deep", size=26,
         weight=700, one_line=True)
    text(s, f"Rule {no} condition", x + 28, 408, CW - 56, 32, cond, "caption", "slate", one_line=True)
    text(s, f"Rule {no} count", x + 28, 450, CW - 56, 40, cnt, "body-strong", "ink", size=26, one_line=True)
    if i < 2:
        arrow(s, f"Rule arrow {i+1}", [(x + CW + 14, 425), (x + CW + GAP - 14, 425)], width_pt=3)
        text(s, f"Rule arrow label {i+1}", x + CW, 384, GAP, 28, "없으면", "footer", "muted", align="c")
table(s, "SID table", MARGIN + 180, 586, [620, 374, 374], 62,
      [["", "H", "R"],
       ["72h 코사인 값 (25% / 중앙 / 75%)", "0.250 / 0.363 / 0.443", "0.233 / 0.336 / 0.419"],
       ["새 기사가 물려받은 C2 종류", "127", "57"],
       ["기사 수 / SID 중복", "13,160 / 0", "13,160 / 0"]],
      align=["l", "c", "c"], bold_cols=(0,))
text(s, "SID note", MARGIN + 180, 846, 1400, 30,
     "R1 코드에 H checkpoint를 넣으면 H와 같은 사건에 100% 붙음 → 코사인은 둘 다 h로 계산", "caption", "steel")

# ════════ 7. Transformer ════════
s = d.new("test Top-1 seed 42/123/2026: V1 26.16/26.79/26.76, H 27.51/27.81/27.70, R 27.67/27.17/27.69. 표준편차는 표본 표준편차.")
d.frame(s, "03", CH["03"], "새 SID가 예전 SID보다 약 1%p 높다",
        "Transformer만 (인기 없음), test 약 12만 2천 행, seed 3개 평균 ± 표준편차")
table(s, "Mean table", MARGIN, 360, [150, 200, 214, 214, 214], 66,
      [["", "Top-1 (%)", "AUC", "MRR", "nDCG@5"]] +
      [[n + (" (기준)" if n == "V1" else ""), ms(TOP1[n]), ms(AUC[n], "{:.4f}"), ms(MRR[n], "{:.4f}"), ms(NDCG[n], "{:.4f}")]
       for n in RUNS],
      highlight_rows=(2, 3), align=["l", "r", "r", "r", "r"], bold_cols=(0,))
bullets(s, "Result list", MARGIN, 666, 990, 220,
        ["세 seed 모두 **H · R > V1** (H − V1 = +1.35 · +1.02 · +0.94%p)",
         "**H와 R은 같다고 봐야 함**: 차이 0.16%p로 표준편차 안, seed마다 앞서는 쪽이 바뀜",
         "seed는 Transformer만 바꿈 (SID는 RQ-VAE seed 42 하나)"], "body", gap=14)
# 점 그림 (seed별 ○, 평균 ●)
PX0, PX1, AX0, AX1 = 1250, 1790, 25.5, 28.5
XP = lambda v: PX0 + (v - AX0) / (AX1 - AX0) * (PX1 - PX0)
sub(s, "Dot caption", 1170, 340, 660, "seed별 test Top-1", color="ink", size=24)
for i, n in enumerate(RUNS):
    y = 440 + i * 110
    text(s, f"Dot row {n}", 1170, y - 22, 70, 44, n, "body-strong", "ink", anchor="m")
    hline(s, f"Dot row rule {n}", PX0, y, PX1 - PX0, color="hairline-soft")
    for v in TOP1[n]:
        dot(s, f"Dot {n} seed {v}", XP(v), y, 9, fill="canvas", line=COL[n], line_w=3)
    m = mean(TOP1[n])
    dot(s, f"Dot {n} mean", XP(m), y, 13, fill=COL[n])
    text(s, f"Dot {n} mean label", XP(m) - 60, y - 52, 120, 28, f"{m:.2f}", "footer", "ink", align="c", weight=700)
for t in (26, 27, 28):
    text(s, f"Dot tick {t}", XP(t) - 30, 788, 60, 28, str(t), "footer", "muted", align="c")
    rect(s, f"Dot tick mark {t}", XP(t) - 1, 772, 2, 10, fill="steel")
hline(s, "Dot axis", PX0, 772, PX1 - PX0, color="steel", width_px=2)
text(s, "Dot legend", PX0, 826, 560, 28, "○ seed별 값 · ● 평균 · 가로축 test Top-1 (%)", "footer", "steel")

# ════════ 8. 진단 ════════
s = d.new("정답이 Jaccard로 붙은 기사인 행(1,262행)은 H 48.97%. 코사인이 낮은 기사가 원래 특이해서 어려운 것일 수도 있어 인과는 단정하지 않음.")
d.frame(s, "04", CH["04"], "비슷하지 않은 사건에 붙은 새 기사는 거의 못 맞힌다",
        "test 정답 기사가 사건에 붙을 때의 코사인별 Transformer Top-1 (seed 42)")
text(s, "Hero number", MARGIN, 360, 640, 170, "98%", "stat-giant", "brand-blue-deep")
text(s, "Hero label", MARGIN, 540, 640, 40, "test 정답 중 validation-only(새) 기사", "body-strong", "ink", one_line=True)
bullets(s, "Hero list", MARGIN, 610, 640, 260,
        ["새 기사의 c2는 SID 생성 때 붙은 **Train 사건에서 물려받음**",
         "개체가 겹쳐(Jaccard) 붙은 기사가 정답이면 H **48.97%**",
         "코사인 0.25 미만으로 억지로 붙으면 **4% 이하**"], "body", gap=14)
legend(s, "Cos legend", 860, 330, ["H", "R"])
vbars(s, "Cos chart", 860, 380, 964, 480, COS_BINS, [("H", COS_TOP1["H"]), ("R", COS_TOP1["R"])], 40,
      [0, 10, 20, 30, 40], ref=20, ref_label="무작위 20%")
text(s, "Cos caption", 912, 878, 900, 28, "가로축 = 정답 기사가 사건에 붙을 때의 h 코사인", "footer", "steel")

# ════════ 9. 결합 방법 ════════
s = d.new("score = s + λ·log(1 + n_24h) + μ·log(1 + age). s = log P(c1) + log P(c2|c1) + log P(c3|c1,c2). "
          "n_24h = 노출 직전 24시간 동안 그 후보 기사의 전체 사용자 클릭 수 [T−24h, T). age = 노출 시각 − 발행 시각(시간).")
d.frame(s, "05", CH["05"], "학습된 점수에 인기 · 나이를 더한다 (재학습 없음)",
        "λ · μ는 validation 앞 절반에서 고르고 test에는 한 번만 적용한다")
_, fw, fh = formula(s, "Fusion formula", 960, 380,
                    r"\mathrm{score} = s + \lambda\,\log(1+n_{24h}) + \mu\,\log(1+\mathrm{age})",
                    "score = s + lambda log(1 + n_24h) + mu log(1 + age)", scale=1.7, anchor="mc")
text(s, "Formula legend", MARGIN + 160, 440, 1408, 32,
     "s = Transformer의 SID 점수 · n = 노출 직전 24시간 클릭 수 (전체 사용자, 노출 이전만) · age = 발행 후 지난 시간",
     "caption", "steel", align="c", one_line=True)
steps = [("SID만 학습", "후보 5개 softmax → CE", [("train", "green")]),
         ("best epoch", "epoch마다 Top-1, 최고 checkpoint", [("validation", "blue")]),
         ("점수 저장", "후보마다 s 계산", [("validation", "blue"), ("test", "amber")]),
         ("λ · μ 고르기", "13 × 10 조합 중 Top-1 최대", [("validation", "blue")]),
         ("test 한 번", "Top-1 · AUC · MRR · nDCG@5", [("test", "amber")])]
SW, SG = 316, 37
for i, (t, desc, tags) in enumerate(steps):
    x = MARGIN + i * (SW + SG)
    stepcard(s, f"Step {i+1}", x, 530, SW, 190, i + 1, t, desc, tags)
    if i < 4:
        arrow(s, f"Step arrow {i+1}", [(x + SW + 6, 625), (x + SW + SG - 6, 625)], width_pt=2.5)
text(s, "Choice line", MARGIN, 790, 1728, 40,
     "[[고른 값]]  9번(실험 3 × seed 3) 모두 λ = 5 · μ는 −1(7번) 또는 −2(2번) — Transformer 가중치는 하나도 바꾸지 않는다",
     "body", "charcoal", one_line=True)

# ════════ 10. 결합 결과 ════════
s = d.new("결합+나이 − 인기만, seed 42/123/2026: V1 +0.47/+0.79/+0.90, H +1.28/+1.21/+1.44, R +1.26/+1.38/+1.12.")
d.frame(s, "05", CH["05"], "인기 · 나이를 더하면 44.2%, SID 몫은 새 SID가 약 2배",
        "test Top-1, seed 3개 평균 · 인기만으로도 42.9%를 맞힌다")
legend(s, "Fusion legend", MARGIN, 320, RUNS)
vbars(s, "Fusion chart", MARGIN, 370, 900, 500, ["모델만", "인기만", "결합 + 나이"],
      [(n, [mean(TOP1[n]), POP[n], mean(FUSED[n])]) for n in RUNS], 50, [0, 10, 20, 30, 40, 50],
      ref=20, ref_label="무작위 20%", bw_max=70)
GX = 1100
sub(s, "Gain heading", GX, 330, 724, "SID가 인기 위에 보태는 몫 (결합 + 나이 − 인기만)", color="ink", size=24)
table(s, "Gain table", GX, 390, [110, 140, 140, 140, 194], 62,
      [["", "seed 42", "seed 123", "seed 2026", "평균"]] +
      [[n] + [f"+{g:.2f}" for g in GAIN[n]] + [f"+{mean(GAIN[n]):.2f}%p"] for n in RUNS],
      highlight_rows=(2, 3), align=["l", "r", "r", "r", "r"], bold_cols=(0, 4))
bullets(s, "Gain list", GX, 672, 724, 220,
        ["V1 최고 seed(+0.90)와 H · R 최저 seed(+1.12)가 **겹치지 않음**",
         "크게 오른 건 인기 자체의 힘, 봐야 할 건 **SID가 보태는 몫**",
         "나이는 +0.1~0.35%p로 작지만 9번 모두 같은 방향"], "caption", gap=12)

# ════════ 11. 함께 학습 ════════
s = d.new("score = s + w1·logpop + w2·logage 로 Transformer를 처음부터 학습 (여민 코드 + 우리 패치). "
          "w도 같이 학습: epoch 1에 5 → 0.78, −1 → −0.09로 줄고 validation 41.96 → 41.21 → 41.10%로 하락(중단). "
          "w 고정(5, −1): validation 42.68%(epoch 1)가 최고, 이후 하락. 나중에 더하기 validation 43.27% 이상.")
d.frame(s, "05", CH["05"], "인기 · 나이를 학습 때 넣으면 오히려 낮아진다",
        "같은 점수 식으로 Transformer를 처음부터 학습 (seed 42, test Top-1)")
formula(s, "Joint formula", MARGIN, 350, r"\mathrm{score} = s + w_1\,\mathrm{logpop} + w_2\,\mathrm{logage}",
        "score = s + w1 logpop + w2 logage", scale=1.4)
bullets(s, "Joint list", MARGIN, 440, 760, 330,
        ["**w도 같이 학습**: epoch 1에 인기 가중치 5 → 0.78로 줄며 학습 전보다 하락 (중단)",
         "**w를 5, −1로 고정**: epoch 1이 정점, 나중에 더하기보다 낮음",
         "**원인으로 보이는 것**: train 정답은 대부분 본 기사라 SID가 외워서 맞히고, 인기를 덜 씀 → test 정답은 98%가 새 기사"],
        "body", gap=18)
table(s, "Joint table", 960, 370, [150, 210, 250, 254], 70,
      [["", "인기만", "함께 학습 (고정)", "나중에 더하기"]] +
      [[n, f"{POP[n]:.2f}%", f"{JOINT[n]:.2f}%  ({JOINT[n] - POP[n]:+.2f})", f"**{FUSED[n][0]:.2f}%**  ({FUSED[n][0] - POP[n]:+.2f})"]
       for n in RUNS],
      align=["l", "r", "r", "r"], bold_cols=(0,))
text(s, "Joint table note", 960, 666, 864, 28, "괄호 = 인기만 대비 SID가 보탠 몫 (%p)", "footer", "steel")
text(s, "Takeaway", 960, 740, 864, 120,
     ["[[결론]]  w는 test와 비슷한 **validation에서** 정한다",
      "→ SID만 학습하고 인기 · 나이는 나중에 더하는 방식을 유지"], "body", "charcoal", gap=8)

# ════════ 12. TIGER ════════
s = d.new("공개 TIGER test (seed 42): V1 16.69% / AUC 0.478, H 17.30% / 0.483, R 17.77% / 0.475. 정답 평균 log P가 오답보다 낮음. "
          "c1만으로 채점하면 22~23%. 출처: Rajput 외, Recommender Systems with Generative Retrieval, NeurIPS 2023 (arXiv:2305.05065), "
          "구현 EdoardoBotta/RQ-VAE-Recommender @ 957d32b (MIT, 비공식 재현).")
d.frame(s, "06", CH["06"], "생성만 학습한 공개 TIGER는 무작위보다 낮다",
        "같은 SID · 같은 test로 비교 (seed 42) — 공개 비공식 재현 구현의 모델 · 손실 · 설정을 그대로 사용",
        footer_left="TIGER: Rajput 외, NeurIPS 2023 · 구현: EdoardoBotta/RQ-VAE-Recommender @ 957d32b (MIT, 비공식 재현)")
sub(s, "Setup heading", MARGIN, 330, 760, "어떻게 돌렸나", color="ink", size=24)
bullets(s, "Setup list", MARGIN, 378, 760, 300,
        ["history 최근 20개 SID → **다음 클릭 기사 SID를 생성** (오답 없이 정답만)",
         "d384 · 6 head · 4층, AdamW 1e-3, batch 640 — **공개 구현 그대로**",
         "test는 우리와 같이 후보 5개를 log P(c1..c3) 합으로 채점"], "caption", gap=12)
sub(s, "Why heading", MARGIN, 640, 760, "왜 낮나", color="ink", size=24)
bullets(s, "Why list", MARGIN, 688, 760, 200,
        ["AUC 0.48: 정답보다 **오답에 더 높은 확률**",
         "본 기사의 SID만 생성하도록 배웠는데 test 정답은 98%가 새 기사"], "caption", gap=12)
legend(s, "Tiger legend", 940, 330, RUNS)
vbars(s, "Tiger chart", 940, 380, 884, 490, ["공개 TIGER", "우리 Transformer", "+ 인기 · 나이"],
      [(n, [TIGER[n], TOP1[n][0], FUSED[n][0]]) for n in RUNS], 50, [0, 10, 20, 30, 40, 50],
      ref=20, ref_label="무작위 20%", bw_max=64)

# ════════ 13. 정리 ════════
s = d.new()
d.frame(s, "07", CH["07"], "새 SID가 낫고, 인기 · 나이는 나중에 더한다",
        "결론 세 가지와 다음 단계")
cols = [("1. 새 SID가 낫다", ["seed 3개 모두 H · R > V1, 약 **+1%p**",
                            "인기 위에 보태는 몫도 **+1.3 vs +0.7%p**",
                            "H와 R은 성능 차이 없음"]),
        ("2. 병목은 새 기사 배정", ["test 정답의 **98%가 새 기사**",
                               "비슷하지 않은 사건에 붙으면 **4% 이하**",
                               "생성형 TIGER는 무작위보다 낮음"]),
        ("3. 인기 · 나이는 나중에", ["더하면 **44.2%** (λ = 5, μ = −1)",
                                "학습 때 넣으면 오히려 낮음",
                                "가중치는 validation에서 정함"])]
for i, (h, items) in enumerate(cols):
    x = MARGIN + i * 592
    hline(s, f"Takeaway rule {i+1}", x, 330, 544, color="brand-blue" if i == 0 else "hairline", width_px=3)
    text(s, f"Takeaway heading {i+1}", x, 352, 544, 40, h, "heading-md", "brand-blue-deep", size=30, one_line=True)
    bullets(s, f"Takeaway list {i+1}", x, 410, 544, 200, items, "body", gap=10)
sub(s, "Next heading", MARGIN, 650, 900, "다음 단계", color="ink", size=26)
for i, (t, st) in enumerate([("새 기사 SID 배정: 코사인이 낮으면 사건 C2 대신 기사 자기 r1로 C2", "제안 · 가장 큰 병목"),
                             ("인기를 점수 대신 Transformer 입력으로: 사용자마다 다른 인기 민감도", "제안"),
                             ("MIND에서도 인기 · 나이 결합과 공개 TIGER를 같은 방식으로", "요청함"),
                             ("R 초기화 보완 (Q2를 학습 뒤 초기화 · r1 정규화)", "우선순위 낮음")]):
    y = 704 + i * 58
    rect(s, f"Next {i+1} box", MARGIN, y + 11, 22, 22, fill="canvas", line="k-amber-line", radius=4, line_w=2)
    text(s, f"Next {i+1}", MARGIN + 38, y, 1300, 44, t, "body", "charcoal", anchor="m", one_line=True)
    text(s, f"Next {i+1} status", 1500, y, 324, 44, st, "caption", "steel", anchor="m", align="r", one_line=True)

# ════════ 14. 감사 ════════
s = d.new()
text(s, "Thanks title", MARGIN, 420, 1728, 130, "들어주셔서 감사합니다", "section-title", "ink")
text(s, "Thanks subtitle", MARGIN, 570, 1728, 44, "v2 재실험 결과 — 새 SID · 실험 H vs R · 인기 · 나이 결합 · TIGER 비교",
     "body-lg", "steel", one_line=True)
d.footer(s, "코드: github.com/minji1217/SID_Project (claude/preprocess-fixed-cluster) · 여민 baekyeomin/SID_Project 22028ba")

# ════════ 저장·점검 ════════
d.save(OUT)
probs = check(OUT)
print(f"저장: {OUT}  ({d.n}장)")
for w_ in WARNINGS + probs:
    print("  !", w_)
if d.n != TOTAL:
    print(f"  ! 장 수 {d.n} ≠ TOTAL {TOTAL}")
