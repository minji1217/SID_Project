"""SURGE 논문 다시 읽기 (논문 이해 전용) HTML 생성. 저장소 루트에서: python reports/analysis/build_surge_reread.py
스타일은 reports/SURGE_논문_해설.html 의 <style> 을 그대로 가져온다."""
import math
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'reports' / 'SURGE_논문_다시읽기.html'
_h = (ROOT / 'reports' / 'SURGE_논문_해설.html').read_text(encoding='utf-8')
STYLE = _h[_h.index('<style>'):_h.index('</style>') + 8]
STYLE = STYLE.replace('</style>', '''
  .abs { display: grid; grid-template-columns: 34px minmax(0, 1fr) minmax(0, 1fr); gap: 0; border: 1px solid var(--line); border-radius: 12px; overflow: hidden; margin: 14px 0; font-size: 14px; }
  .abs > div { padding: 9px 12px; border-top: 1px solid var(--line); min-width: 0; }
  .abs > div:nth-child(-n+3) { border-top: 0; background: var(--panel); font-weight: 650; font-size: 13px; }
  .abs .k { color: var(--accent); font-weight: 700; font-variant-numeric: tabular-nums; }
  .abs .en { color: var(--muted); font-size: 13px; font-style: italic; }
  @media (max-width: 720px) { .abs { grid-template-columns: 28px minmax(0, 1fr); } .abs .en { grid-column: 2; padding-top: 0; border-top: 0; } .abs > div:nth-child(2) { grid-column: 2; } .abs > div:nth-child(3) { display: none; } }
  .pill { display: inline-block; font-size: 12px; font-weight: 650; padding: 1px 9px; border-radius: 999px; background: var(--accent-bg); color: var(--accent); margin-right: 4px; }
  .q { font-size: 17px; font-weight: 650; border-left: 4px solid var(--accent); padding: 6px 14px; margin: 14px 0; background: var(--accent-bg); border-radius: 0 8px 8px 0; }
</style>''')

# ---------------------------------------------------------------- svg helpers
def svg(vb_w, vb_h, body, cls='wide', label=''):
    return (f'<div class="scroll"><svg class="{cls}" viewBox="0 0 {vb_w} {vb_h}" role="img" aria-label="{label}">'
            '<defs><marker id="ah" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="arrowhead"/></marker>'
            '<marker id="aha" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="arrowhead-a"/></marker></defs>'
            + body + '</svg></div>')

def T(x, y, s, size=12, w=400, cls='', anchor='start', extra=''):
    c = f' class="{cls}"' if cls else ''
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{w}" text-anchor="{anchor}"{c} {extra}>{s}</text>'

def Rc(x, y, w, h, cls='box', rx=6, extra=''):
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" class="{cls}" {extra}/>'

def C(x, y, r, cls, extra=''):
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" class="{cls}" {extra}/>'

def L(x1, y1, x2, y2, cls='ln', extra=''):
    return f'<path d="M{x1:.1f} {y1:.1f} L{x2:.1f} {y2:.1f}" class="{cls}" {extra}/>'

def A(x1, y1, x2, y2, acc=False, extra=''):
    return f'<path d="M{x1:.1f} {y1:.1f} L{x2:.1f} {y2:.1f}" class="{"ln-a" if acc else "ln"}" marker-end="url(#{"aha" if acc else "ah"})" {extra}/>'

def fig(body, cap):
    return f'<div class="fig">{body}<div class="cap">{cap}</div></div>'

G = {'s': 'g0', 'p': 'g1', 'c': 'g2', 'n': 'gn'}     # 스포츠, 정치, 요리, 잡음
NAME = {'s': '스포츠', 'p': '정치', 'c': '요리', 'n': '잡음'}
SEQ = ['s', 'p', 's', 'n', 'p', 'c', 'p', 'p']        # 예시 기록 8개 (시간순)

# ---------------------------------------------------------------- 그림 0: 전체 흐름 (논문 Figure 1 다시 그리기)
def fig_overview():
    b = ''
    # A: 시퀀스
    b += Rc(10, 40, 200, 230, 'box', 12)
    b += T(110, 62, 'A. 관심 그래프 만들기', 13, 700, 'tx-a', 'middle')
    b += T(110, 82, '느슨한 시퀀스 → 촘촘한 그래프', 11.5, 400, 'tx-m', 'middle')
    for k, g in enumerate(SEQ):
        b += C(30 + k * 23, 110, 8, G[g])
    b += T(30, 132, '시간 →', 10.5, 400, 'tx-m')
    b += A(110, 138, 110, 160)
    b += T(118, 148, '유사도 학습', 10.5, 400, 'tx-m')
    b += T(118, 162, '+ 상위 ε', 10.5, 400, 'tx-m')
    # 작은 그래프
    pos = {0: (55, 200), 2: (80, 240), 1: (150, 190), 4: (175, 225), 6: (140, 250), 7: (185, 255), 5: (60, 262), 3: (115, 215)}
    edges = [(0, 2), (1, 4), (1, 6), (4, 7), (6, 7), (1, 7), (4, 6)]
    for i, j in edges: b += L(*pos[i], *pos[j], 'edge')
    for i, (x, y) in pos.items(): b += C(x, y, 8, G[SEQ[i]] + ' ring')
    # B
    b += A(214, 155, 236, 155)
    b += Rc(240, 40, 200, 230, 'box', 12)
    b += T(340, 62, 'B. 관심 융합 (GNN)', 13, 700, 'tx-a', 'middle')
    b += T(340, 82, '약한 신호를 이웃과 섞어 키움', 11.5, 400, 'tx-m', 'middle')
    pos2 = {k: (x + 230, y - 60) for k, (x, y) in pos.items()}
    for i, j in edges: b += L(*pos2[i], *pos2[j], 'edge', 'stroke-width="3.5"')
    for i, (x, y) in pos2.items():
        r = 11 if SEQ[i] == 'p' else (9 if SEQ[i] == 's' else 6)
        b += C(x, y, r, G[SEQ[i]] + ' ring')
    b += T(340, 236, '핵심 관심(촘촘한 곳)은 커지고', 11, 400, 'tx-m', 'middle')
    b += T(340, 252, '외톨이(잡음)는 작아짐', 11, 400, 'tx-m', 'middle')
    # C
    b += A(444, 155, 466, 155)
    b += Rc(470, 40, 200, 230, 'box', 12)
    b += T(570, 62, 'C. 관심 추출 (풀링)', 13, 700, 'tx-a', 'middle')
    b += T(570, 82, '노드 n개 → 덩어리 m개', 11.5, 400, 'tx-m', 'middle')
    for k, (g, r, x, y) in enumerate([('s', 20, 515, 150), ('p', 28, 590, 130), ('c', 14, 625, 200)]):
        b += C(x, y, r, G[g] + ' ring')
    b += L(535, 150, 562, 135, 'edge'); b += L(605, 155, 618, 188, 'edge')
    b += T(570, 236, '덩어리마다 값 h*와', 11, 400, 'tx-m', 'middle')
    b += T(570, 252, '중요도 γ*', 11, 400, 'tx-m', 'middle')
    # D
    b += A(674, 155, 696, 155)
    b += Rc(700, 40, 190, 230, 'box-a', 12)
    b += T(795, 62, 'D. 예측', 13, 700, 'tx-a', 'middle')
    b += T(795, 82, '덩어리를 시간순으로 펴서 읽기', 11.5, 400, 'tx-m', 'middle')
    for k, g in enumerate(['s', 'p', 'c']):
        b += C(735 + k * 30, 115, 11, G[g] + ' ring')
        if k < 2: b += A(747 + k * 30, 115, 753 + k * 30, 115)
    b += T(795, 145, 'AUGRU → 관심 변화 h_s', 11.5, 400, '', 'middle')
    b += T(795, 168, '+ 후보 아이템 h_t', 11.5, 400, '', 'middle')
    b += A(795, 178, 795, 200)
    b += Rc(745, 204, 100, 28, 'box', 6)
    b += T(795, 223, 'MLP → ŷ', 12.5, 700, '', 'middle')
    b += T(795, 252, '클릭할 확률', 11, 400, 'tx-m', 'middle')
    # legend
    for k, g in enumerate(['s', 'p', 'c', 'n']):
        b += C(20 + k * 92, 18, 6, G[g]); b += T(30 + k * 92, 22, NAME[g] + (' 관심' if g != 'n' else ' (한 번 클릭)'), 11.5)
    return svg(900, 280, b, 'wide', 'SURGE 전체 흐름')

# ---------------------------------------------------------------- 그림 1: 두 문제
def fig_problems():
    b = ''
    b += T(10, 20, '한 사용자의 클릭 기록 (시간순)', 13, 700)
    seq = ['s', 's', 'n', 's', 'p', 'n', 's', 'p', 'p', 'c', 'p', 'n', 'p', 'p', 'c', 'p']
    x0, dx = 30, 52
    b += L(20, 70, 880, 70, 'ln')
    for k, g in enumerate(seq):
        x = x0 + k * dx
        b += C(x, 70, 11 if g != 'n' else 8, G[g] + ' ring')
        if g == 'n':
            b += T(x, 100, '?', 13, 700, 'tx-m', 'middle')
    b += T(880, 92, '지금 →', 11, 400, 'tx-m', 'end')
    # 문제 1
    b += Rc(10, 118, 430, 92, 'box', 10)
    b += T(24, 140, '① 클릭은 암묵적이고 잡음이 많다', 13, 700, 'tx-a')
    b += T(24, 162, '회색 점처럼 관심 없이 눌렀거나 다시 찾지 않는 클릭이', 12)
    b += T(24, 180, '섞여 있어요. 클릭 하나만으로는 진짜 선호를 알 수 없어요.', 12)
    b += T(24, 198, '→ 비슷한 클릭이 많이 모인 곳 = 진짜 관심', 12, 650)
    # 문제 2
    b += Rc(460, 118, 430, 92, 'box', 10)
    b += T(474, 140, '② 관심은 시간에 따라 빠르게 바뀐다', 13, 700, 'tx-a')
    b += T(474, 162, '앞쪽은 스포츠, 뒤쪽은 정치가 활발해요. 지금 시점에', 12)
    b += T(474, 180, '켜져 있는 관심과 꺼진 관심이 섞여 있어요.', 12)
    b += T(474, 198, '→ 지금 활성화된 관심을 골라야 함', 12, 650)
    # 활성 띠
    b += T(10, 238, '시점별로 켜져 있는 관심', 12, 650, 'tx-m')
    b += Rc(30, 248, 330, 16, 'g0', 4, 'opacity=".85"')
    b += Rc(230, 268, 650, 16, 'g1', 4, 'opacity=".85"')
    b += Rc(490, 288, 360, 16, 'g2', 4, 'opacity=".6"')
    b += T(36, 260, '스포츠', 11, 700, '', 'start', 'style="fill:#fff"')
    b += T(236, 280, '정치', 11, 700, '', 'start', 'style="fill:#fff"')
    b += T(496, 300, '요리', 11, 700, '', 'start', 'style="fill:#fff"')
    return svg(900, 312, b, 'wide', '순차 추천의 두 문제')

# ---------------------------------------------------------------- 3.1 그래프 만들기 (실제 계산)
EMB = {'s': (1.0, 0.1, 0.0), 'p': (0.05, 1.0, 0.1), 'c': (0.1, 0.15, 1.0), 'n': (0.35, -0.3, -0.5)}
JIT = [(0.0, 0.05, 0.0), (0.05, 0.0, 0.08), (0.08, -0.06, 0.02), (0, 0, 0), (-0.04, 0.02, -0.05), (0.0, 0.05, 0.0), (0.06, 0.0, -0.03), (-0.05, 0.04, 0.06)]
H = [tuple(a + b for a, b in zip(EMB[g], JIT[k])) for k, g in enumerate(SEQ)]
def cos(a, b):
    return sum(x * y for x, y in zip(a, b)) / math.sqrt(sum(x * x for x in a) * sum(y * y for y in b))
M = [[cos(H[i], H[j]) for j in range(8)] for i in range(8)]
EPS = 0.34
flat = sorted((v for r in M for v in r), reverse=True)
THR = flat[int(EPS * 64) - 1]
AD = [[1 if M[i][j] >= THR else 0 for j in range(8)] for i in range(8)]

def fig_graph():
    b = ''
    # (가) 기록
    b += T(10, 20, '(가) 기록 8개 = 노드 8개', 13, 700, 'tx-a')
    for k, g in enumerate(SEQ):
        b += C(30 + k * 30, 48, 10, G[g] + ' ring'); b += T(30 + k * 30, 52, str(k + 1), 10.5, 700, '', 'middle', 'style="fill:#fff"')
    b += T(10, 80, '각 노드는 임베딩 h_i ∈ ℝ^d', 11.5, 400, 'tx-m')
    # (나) 유사도 행렬
    x0, y0, cw = 300, 40, 26
    b += T(300, 20, '(나) 유사도 M_ij  식 (1)(2)', 13, 700, 'tx-a')
    for i in range(8):
        b += T(x0 - 8, y0 + i * cw + 17, str(i + 1), 10.5, 400, 'tx-m', 'end')
        b += T(x0 + i * cw + cw / 2, y0 - 4, str(i + 1), 10.5, 400, 'tx-m', 'middle')
        for j in range(8):
            v = M[i][j]
            op = max(0.06, (v + 0.2) / 1.2) if v > -0.2 else 0.06
            b += Rc(x0 + j * cw, y0 + i * cw, cw - 2, cw - 2, 'g1' if AD[i][j] else 'gn', 3, f'opacity="{op:.2f}"')
            b += T(x0 + j * cw + cw / 2 - 1, y0 + i * cw + 16, f'{v:.1f}'.replace('-0.', '-.').replace('0.', '.'), 9, 400, '', 'middle')
    b += T(x0, y0 + 8 * cw + 18, '색이 진할수록 비슷함. 주황 테두리 칸 = 상위 ε 안', 11, 400, 'tx-m')
    for i in range(8):
        for j in range(8):
            if AD[i][j]: b += Rc(x0 + j * cw, y0 + i * cw, cw - 2, cw - 2, 'box', 3, 'style="fill:none;stroke:var(--s2);stroke-width:1.6"')
    # (다) 그래프
    b += T(560, 20, f'(다) 상위 ε = {EPS} 만 1 → 그래프 A  식 (3)', 13, 700, 'tx-a')
    pos = {0: (600, 90), 2: (630, 150), 1: (760, 80), 4: (820, 120), 6: (750, 170), 7: (830, 190), 5: (680, 230), 3: (590, 220)}
    for i in range(8):
        for j in range(i + 1, 8):
            if AD[i][j]: b += L(*pos[i], *pos[j], 'edge')
    for i, (x, y) in pos.items():
        b += C(x, y, 13, G[SEQ[i]] + ' ring'); b += T(x, y + 4, str(i + 1), 11, 700, '', 'middle', 'style="fill:#fff"')
    b += T(560, 262, '정치 4개는 촘촘한 덩어리(핵심 관심),', 11.5, 400, 'tx-m')
    b += T(560, 279, '요리 1개와 잡음 1개는 외톨이(주변 관심).', 11.5, 400, 'tx-m')
    # 왼쪽 아래 설명
    b += T(10, 120, 'w: 학습되는 가중치 벡터', 11.5, 650)
    b += T(10, 138, '(임베딩 차원마다 볼륨 손잡이)', 11.5, 400, 'tx-m')
    b += T(10, 162, '헤드 φ개가 각자 w로 M을 만들고', 11.5, 650)
    b += T(10, 180, '평균을 냄 (식 2)', 11.5, 400, 'tx-m')
    b += T(10, 204, 'ε: 전체 n² 칸 중 남길 비율', 11.5, 650)
    b += T(10, 222, '(대각선 8칸도 포함해서 셈)', 11.5, 400, 'tx-m')
    return svg(900, 292, b, 'wide', '관심 그래프 만들기')

# ---------------------------------------------------------------- 3.2 융합
def fig_fusion():
    b = ''
    # 그래프 조각: 노드 i(5번 정치) + 이웃
    b += T(10, 20, '노드 i = 5번(정치)이 이웃에게서 정보를 받는 장면', 13, 700, 'tx-a')
    ci = (170, 150)
    nb = {'2': (70, 80, 'p'), '7': (275, 85, 'p'), '8': (270, 225, 'p')}
    for k, (x, y, g) in nb.items():
        w = {'2': 3, '7': 5, '8': 6}[k]
        b += f'<path d="M{x} {y} L{ci[0]} {ci[1]}" class="ln-a" stroke-width="{w}" marker-end="url(#aha)" opacity=".85"/>'
        b += C(x, y, 15, G[g] + ' ring'); b += T(x, y + 4, k, 11.5, 700, '', 'middle', 'style="fill:#fff"')
    b += f'<path d="M{ci[0]-30} {ci[1]+30} C {ci[0]-70} {ci[1]+70}, {ci[0]+10} {ci[1]+80}, {ci[0]-4} {ci[1]+22}" class="ln-a" stroke-width="2.5" marker-end="url(#aha)" opacity=".85"/>'
    b += C(*ci, 20, 'g1 ring'); b += T(ci[0], ci[1] + 5, '5 (i)', 12, 700, '', 'middle', 'style="fill:#fff"')
    b += T(120, 270, '자기 자신도 이웃에 포함 (self-loop)', 11, 400, 'tx-m', 'middle')
    b += T(170, 292, '화살표 굵기 = 섞는 비율 E_ij', 11.5, 650, '', 'middle')
    # 오른쪽: α, β, E
    x = 370
    b += Rc(x, 40, 250, 120, 'box', 10)
    b += T(x + 12, 62, 'α_i: 군집 인식 (식 6)', 13, 700)
    b += T(x + 12, 84, '받는 쪽 i가 동네의 중심인가?', 12)
    b += T(x + 12, 104, 'h_i 와 동네 평균 h_i^c 를 비교', 11.5, 400, 'tx-m')
    b += T(x + 12, 122, '(동네 = k-hop 이웃)', 11.5, 400, 'tx-m')
    b += T(x + 12, 146, '2층 FFN(LeakyReLU)이 점수를 냄', 11.5, 400, 'tx-m')
    b += Rc(x + 270, 40, 250, 120, 'box', 10)
    b += T(x + 282, 62, 'β_j: 질의 인식 (식 7)', 13, 700)
    b += T(x + 282, 84, '보내는 쪽 j가 후보와 관련 있나?', 12)
    b += T(x + 282, 104, 'h_j 와 후보 아이템 h_t 를 비교', 11.5, 400, 'tx-m')
    b += T(x + 282, 122, '관련 없는 정보는 덜 보냄', 11.5, 400, 'tx-m')
    b += T(x + 282, 146, '2층 FFN(LeakyReLU)이 점수를 냄', 11.5, 400, 'tx-m')
    b += A(x + 125, 162, x + 230, 196); b += A(x + 395, 162, x + 290, 196)
    b += Rc(x + 90, 198, 340, 40, 'box-a', 10)
    b += T(x + 260, 223, 'E_ij = softmax_j ( α_i + β_j )   식 (8)', 13, 700, '', 'middle')
    b += A(x + 260, 240, x + 260, 258)
    b += Rc(x + 60, 260, 400, 40, 'box-a', 10)
    b += T(x + 260, 285, "h′_i = σ( W_a · Σ_j E_ij h_j + h_i )   식 (4)", 13, 700, '', 'middle')
    b += T(x + 260, 318, '헤드 φ개를 이어 붙여 h′_i 는 φd′ 차원 (식 5)', 11.5, 400, 'tx-m', 'middle')
    return svg(900, 330, b, 'wide', '관심 융합')

# ---------------------------------------------------------------- 3.3 풀링
S = [[.92, .05, .03], [.04, .90, .06], [.88, .08, .04], [.40, .32, .28], [.03, .93, .04], [.05, .15, .80], [.02, .92, .06], [.03, .88, .09]]
CL = ['덩어리 1', '덩어리 2', '덩어리 3']
def fig_pool():
    b = ''
    b += T(10, 20, '배정표 S (n × m): 각 노드가 각 덩어리에 속하는 비율 (행 합 1)  식 (11)', 13, 700, 'tx-a')
    x0, y0, cw, ch = 70, 52, 58, 24
    for k, c in enumerate(CL): b += T(x0 + k * cw + cw / 2, y0 - 8, c, 11, 650, 'tx-m', 'middle')
    for i, row in enumerate(S):
        b += C(x0 - 40, y0 + i * ch + 11, 8, G[SEQ[i]]); b += T(x0 - 10, y0 + i * ch + 16, f'{i+1}', 11, 400, 'tx-m', 'end')
        for k, v in enumerate(row):
            b += Rc(x0 + k * cw, y0 + i * ch, cw - 3, ch - 3, 'g1', 3, f'opacity="{0.08 + 0.8 * v:.2f}"')
            b += T(x0 + k * cw + cw / 2 - 1, y0 + i * ch + 15, f'{v:.2f}', 10.5, 400, '', 'middle')
    b += T(x0 - 44, y0 + 8 * ch + 18, '4번(잡음)은 비율이 퍼져 있음', 11, 400, 'tx-m')
    b += T(x0 - 44, y0 + 8 * ch + 34, '→ 단일 소속 벌점 L_A가 줄이려는 것', 11, 400, 'tx-m')
    # 화살표 → 결과
    b += A(260, 140, 300, 140)
    rx = 310
    b += Rc(rx, 40, 270, 100, 'box-a', 10)
    b += T(rx + 12, 62, 'h* = Sᵀ h′   (m 개)  식 (9)', 13, 700)
    b += T(rx + 12, 84, '덩어리 값: 담긴 비율만큼 노드 벡터를 더함', 11.5)
    for k, (g, r) in enumerate([('s', 14), ('p', 22), ('c', 11)]):
        b += C(rx + 40 + k * 60, 115, r, G[g] + ' ring')
    b += Rc(rx, 150, 270, 100, 'box-a', 10)
    b += T(rx + 12, 172, 'γ* = Sᵀ γ   (m 개)  식 (10)', 13, 700)
    b += T(rx + 12, 194, '덩어리 중요도: γ_i = softmax(β_i) 를 모음', 11.5)
    for k, (g, v) in enumerate([('s', .08), ('p', .85), ('c', .07)]):
        b += Rc(rx + 20 + k * 80, 236 - v * 30, 40, v * 30, G[g], 3); b += T(rx + 40 + k * 80, 246, f'{v:.2f}', 10.5, 400, '', 'middle')
    b += Rc(rx, 260, 270, 60, 'box', 10)
    b += T(rx + 12, 282, 'A* = Sᵀ A S   (m × m)', 13, 700)
    b += T(rx + 12, 302, '덩어리끼리의 연결. 여러 층 풀링에 씀', 11.5, 400, 'tx-m')
    # readout
    b += Rc(600, 40, 290, 130, 'box', 10)
    b += T(612, 62, '그래프 읽기 h_g  식 (15)', 13, 700)
    b += T(612, 84, '풀링 전 노드 전체를 γ_i 로 가중합', 11.5)
    b += T(612, 104, 'h_g = Σ_i γ_i · h′_i', 12.5, 650)
    b += T(612, 128, '"후보와 관련된 노드 위주의', 11.5, 400, 'tx-m')
    b += T(612, 146, '기록 전체 요약" → 예측 층으로', 11.5, 400, 'tx-m')
    b += Rc(600, 180, 290, 140, 'box', 10)
    b += T(612, 202, 'm 은 미리 정하는 값 (m < n)', 13, 700)
    b += T(612, 224, 'CNN의 풀링처럼 그래프를 줄여요', 11.5)
    b += T(612, 246, '풀링 길이 탐색 범위', 11.5, 400, 'tx-m')
    b += T(612, 264, 'Taobao [10…50], Kuaishou [50…250]', 11.5, 400, 'tx-m')
    b += T(612, 288, 'S 는 GNN(A, h′)과 softmax 로 만듦', 11.5, 400, 'tx-m')
    b += T(612, 306, '→ 사용자마다, 후보마다 달라요', 11.5, 400, 'tx-m')
    return svg(900, 330, b, 'wide', '관심 추출 풀링')

def fig_regs():
    b = ''
    w = 290
    titles = [('L_M 같은 배정 (식 12)', '‖A − SSᵀ‖_F', '연결된 노드는 같은 덩어리로'),
              ('L_A 단일 소속 (식 13)', '행마다 엔트로피 평균', '한 노드는 한 덩어리에 몰리게'),
              ('L_P 상대 위치 (식 14)', '‖P_n S − P_m‖₂', '앞쪽 노드는 앞쪽 덩어리로')]
    for k, (t, f, d) in enumerate(titles):
        x = 10 + k * (w + 15)
        b += Rc(x, 10, w, 220, 'box', 10)
        b += T(x + 12, 32, t, 13, 700, 'tx-a'); b += T(x + 12, 52, f, 12.5, 650); b += T(x + 12, 72, d, 11.5, 400, 'tx-m')
    # L_M: 두 작은 행렬
    def mini(x, y, mat, lab):
        s = ''
        for i, r in enumerate(mat):
            for j, v in enumerate(r):
                s += Rc(x + j * 14, y + i * 14, 12, 12, 'g1', 2, f'opacity="{0.08 + 0.85 * v:.2f}"')
        return s + T(x + len(mat) * 7, y + len(mat) * 14 + 14, lab, 11, 400, 'tx-m', 'middle')
    sub = [0, 1, 2, 4]
    Am = [[AD[i][j] for j in sub] for i in sub]
    SS = [[sum(S[i][k] * S[j][k] for k in range(3)) for j in sub] for i in sub]
    b += mini(40, 100, Am, 'A (연결)') + T(120, 135, '≈', 18, 400, '', 'middle') + mini(150, 100, SS, 'SSᵀ (같은 덩어리 확률)')
    b += T(22, 200, '두 행렬이 비슷할수록 벌점이 작아요', 11.5)
    b += T(22, 218, '(노드 1·2·3·5 부분만 그림)', 11, 400, 'tx-m')
    # L_A: 뾰족 vs 평평
    x = 10 + w + 15
    for k, (vals, lab) in enumerate([([.92, .05, .03], '1번: 엔트로피 낮음'), ([.40, .32, .28], '4번: 엔트로피 높음')]):
        for j, v in enumerate(vals):
            b += Rc(x + 30 + k * 130 + j * 26, 170 - v * 70, 20, v * 70, 'g1', 3)
        b += T(x + 63 + k * 130, 190, lab, 11, 400, 'tx-m', 'middle')
    b += T(x + 12, 214, '0에 가까울수록 소속이 분명해요', 11.5)
    # L_P: 대각선
    x = 10 + 2 * (w + 15)
    for i in range(8):
        for j in range(3):
            on = (j == 0 and i < 3) or (j == 1 and 2 <= i < 6) or (j == 2 and i >= 5)
            b += Rc(x + 40 + j * 22, 88 + i * 13, 20, 11, 'g1' if on else 'gn', 2, f'opacity="{0.85 if on else 0.15}"')
    b += T(x + 40, 208, '노드 1→8 (위→아래), 덩어리 1→3', 11, 400, 'tx-m')
    b += T(x + 150, 110, '덩어리 순서가', 11.5); b += T(x + 150, 128, '시간 순서를', 11.5); b += T(x + 150, 146, '따르게 해서', 11.5)
    b += T(x + 150, 164, '6단계가 시간순으로', 11.5); b += T(x + 150, 182, '읽을 수 있게', 11.5)
    b += T(x + 12, 224, '덩어리 인덱스 교환은 미분이 안 돼서 벌점으로', 10.5, 400, 'tx-m')
    return svg(900, 240, b, 'wide', '배정 정규화 세 개')

# ---------------------------------------------------------------- 3.4 예측
def fig_pred():
    b = ''
    b += T(10, 20, '덩어리 시퀀스를 AUGRU로 읽고, 요약들을 후보와 이어 붙여 MLP로 예측', 13, 700, 'tx-a')
    gam = [.08, .85, .07]
    for k, (g, v) in enumerate(zip(['s', 'p', 'c'], gam)):
        x = 60 + k * 170
        b += C(x, 70, 16, G[g] + ' ring'); b += T(x + 24, 75, f'h*{k+1}', 12, 650)
        b += Rc(x - 36, 120, 72, 44, 'box-a', 8); b += T(x, 140, 'AUGRU', 12, 700, '', 'middle'); b += T(x, 156, f'문 × γ*={v:.2f}', 10.5, 400, 'tx-m', 'middle')
        b += A(x, 88, x, 118)
        if k < 2: b += A(x + 38, 142, x + 132, 142)
    b += T(290, 186, '상태 = (1 − γ*·u) × 옛 상태 + γ*·u × 새 후보 상태', 11.5, 400, 'tx-m', 'middle')
    b += T(290, 204, 'γ* 가 작은 덩어리는 기억을 거의 바꾸지 못함 (식 16)', 11.5, 400, 'tx-m', 'middle')
    b += A(436, 142, 520, 142)
    # concat
    parts = [('h_s', '관심 변화'), ('h_g', '그래프 요약'), ('h_t', '후보'), ('h_g ⊙ h_t', '요약×후보')]
    for k, (p, d) in enumerate(parts):
        y = 50 + k * 42
        b += Rc(524, y, 110, 34, 'box', 6); b += T(579, y + 16, p, 12, 700, '', 'middle'); b += T(579, y + 29, d, 10, 400, 'tx-m', 'middle')
    b += A(638, 133, 680, 133)
    b += Rc(684, 110, 90, 46, 'box-a', 8); b += T(729, 130, 'MLP', 13, 700, '', 'middle'); b += T(729, 147, '[100, 64]', 10.5, 400, 'tx-m', 'middle')
    b += A(778, 133, 812, 133)
    b += C(842, 133, 24, 'box-a'); b += T(842, 138, 'ŷ', 15, 700, '', 'middle')
    b += T(579, 230, 'ŷ = Predict(h_s ‖ h_g ‖ h_t ‖ h_g ⊙ h_t)  식 (17)', 12, 650, '', 'middle')
    # loss
    b += Rc(10, 250, 880, 58, 'box', 10)
    b += T(24, 272, '손실 (식 18): 교차 엔트로피 −[y log ŷ + (1−y) log(1−ŷ)] 평균 + λ‖Θ‖²  + 3.3.2절의 벌점 세 개 (L_M, L_A, L_P)', 12.5, 650)
    b += T(24, 294, '모든 비교 모델이 같은 예측 층(2층 FFN [100, 64])과 같은 손실을 써요. 한 인스턴스 = (기록, 후보 하나, 클릭했나 0/1)', 11.5, 400, 'tx-m')
    return svg(900, 318, b, 'wide', '예측 층과 손실')

# ---------------------------------------------------------------- 결과 차트
T2 = [('NCF', .7128, .7221, .1446, .0829, .5559, .5531, .7734, .8327, '비순차'),
      ('DIN', .7637, .8524, .3091, .2352, .6160, .7483, .8863, .9160, '비순차'),
      ('LightGCN', .7483, .7513, .1669, .1012, .6403, .6407, .8175, .8653, '비순차'),
      ('Caser', .8312, .8499, .3508, .2890, .7795, .8097, .9100, .9336, '순차'),
      ('GRU4REC', .8635, .8680, .3993, .3422, .8156, .8333, .9174, .9391, '순차'),
      ('DIEN', .8477, .8745, .4011, .3404, .7037, .7800, .9030, .9284, '순차'),
      ('SLi-Rec', .8664, .8669, .3617, .2971, .7978, .8128, .9075, .9318, '순차'),
      ('SURGE', .8906, .8888, .4228, .3625, .8525, .8610, .9316, .9495, '제안')]

def hbar(title, rows, lo, hi, ticks, fmt, w=430, lab_w=86, hl='SURGE', note=''):
    b = T(0, 16, title, 13, 700)
    if note: b += T(0, 33, note, 11, 400, 'tx-m')
    y0 = 46; bh = 18; gap = 6; x0 = lab_w; W = w - lab_w - 50
    X = lambda v: x0 + (v - lo) / (hi - lo) * W
    for t in ticks:
        b += L(X(t), y0 - 4, X(t), y0 + len(rows) * (bh + gap), 'gridl', 'style="stroke:var(--grid)"')
        b += T(X(t), y0 + len(rows) * (bh + gap) + 14, fmt(t), 10.5, 400, 'tx-m', 'middle')
    for k, (name, v) in enumerate(rows):
        y = y0 + k * (bh + gap)
        b += T(x0 - 8, y + 13, name, 11.5, 700 if name == hl else 400, '', 'end')
        b += Rc(X(lo), y, max(1, X(v) - X(lo)), bh, 'g0' if name == hl else 'gn', 3, '' if name == hl else 'opacity=".55"')
        b += T(X(v) + 5, y + 13, fmt(v), 11, 700 if name == hl else 400)
    return b, y0 + len(rows) * (bh + gap) + 24

def two_charts(title_l, rows_l, title_r, rows_r, lo, hi, ticks, fmt, label, note=''):
    bl, hl_ = hbar(title_l, rows_l, lo, hi, ticks, fmt, note=note)
    br, hr_ = hbar(title_r, rows_r, lo, hi, ticks, fmt, note=note)
    Hh = max(hl_, hr_)
    return svg(900, Hh, f'<g transform="translate(10,0)">{bl}</g><g transform="translate(465,0)">{br}</g>', 'wide', label)

def fig_auc():
    f3 = lambda v: f'{v:.2f}' if v in (.5, .6, .7, .8, .9) else f'{v:.4f}'
    return two_charts('Taobao (평균 기록 길이 39.85)', [(r[0], r[1]) for r in T2], 'Kuaishou (평균 기록 길이 245.88)', [(r[0], r[5]) for r in T2],
                      .5, .9, [.5, .6, .7, .8, .9], f3, 'Table 2 AUC', '가로축은 0.5(무작위)부터')

def fig_time():
    T3 = [('DIN', 22.65, 20.59), ('Caser', 23.66, 120.26), ('GRU4REC', 26.78, 73.35), ('DIEN', 18.74, 28.47), ('SLi-Rec', 27.82, 28.84), ('SURGE', 14.96, 22.86)]
    fm = lambda v: f'{v:g}분' if v in (0, 30, 60, 90, 120) else f'{v:.2f}분'
    return two_charts('Taobao: 수렴까지 학습 시간', [(r[0], r[1]) for r in T3], 'Kuaishou: 수렴까지 학습 시간', [(r[0], r[2]) for r in T3],
                      0, 130, [0, 30, 60, 90, 120], fm, 'Table 3 학습 시간')

def fig_ablation():
    T4 = [('w/o Fusion', .8307, .7149), ('w/o Query-aware', .8720, .7641), ('w/o Cluster-aware', .8764, .8213),
          ('w/o Extraction', .8513, .8240), ('w/o Readout', .8578, .8422), ('w/o Regularization', .8815, .8487), ('SURGE (전체)', .8906, .8525)]
    f3 = lambda v: f'{v:.2f}' if v in (.7, .75, .8, .85, .9) else f'{v:.4f}'
    bl, hl_ = hbar('Taobao AUC', [(r[0], r[1]) for r in T4], .7, .9, [.7, .75, .8, .85, .9], f3, lab_w=128, hl='SURGE (전체)', note='가로축은 0.70부터')
    br, hr_ = hbar('Kuaishou AUC', [(r[0], r[2]) for r in T4], .7, .9, [.7, .75, .8, .85, .9], f3, lab_w=128, hl='SURGE (전체)', note='가로축은 0.70부터')
    return svg(900, max(hl_, hr_), f'<g transform="translate(10,0)">{bl}</g><g transform="translate(465,0)">{br}</g>', 'wide', 'Table 4 ablation')

def fig_length():
    b = ''
    b += T(10, 20, '기록 길이별 성능 (논문 Figure 2를 말로 옮긴 모양)', 13, 700, 'tx-a')
    b += L(60, 200, 860, 200, 'ln'); b += L(60, 40, 60, 200, 'ln')
    b += T(860, 220, '기록 길이 →', 11, 400, 'tx-m', 'end'); b += T(66, 46, 'GAUC', 11, 400, 'tx-m')
    xs = [140, 300, 460, 620, 780]
    base = [120, 80, 70, 95, 130]
    surge = [110, 70, 60, 62, 66]
    pb = ' '.join(f'{"M" if k == 0 else "L"}{x} {y}' for k, (x, y) in enumerate(zip(xs, base)))
    ps = ' '.join(f'{"M" if k == 0 else "L"}{x} {y}' for k, (x, y) in enumerate(zip(xs, surge)))
    b += f'<path d="{pb}" class="ln" stroke-width="2.5" style="stroke:var(--s0)"/>'
    b += f'<path d="{ps}" class="ln" stroke-width="3" style="stroke:var(--s1)"/>'
    for x, y in zip(xs, base): b += C(x, y, 5, 'gn')
    for x, y in zip(xs, surge): b += C(x, y, 5, 'g0')
    b += T(140, 230, '짧음', 11, 400, 'tx-m', 'middle'); b += T(780, 230, '김', 11, 400, 'tx-m', 'middle')
    b += T(790, 125, '대부분의 비교 모델', 11.5, 650, 'tx-m'); b += T(790, 60, 'SURGE', 12, 700, 'tx-a')
    b += T(140, 150, '짧으면 데이터가', 11, 400, 'tx-m', 'middle'); b += T(140, 166, '부족해 모두 낮음', 11, 400, 'tx-m', 'middle')
    b += T(620, 140, '길어지면 잡음 때문에', 11, 400, 'tx-m', 'middle'); b += T(620, 156, '비교 모델은 떨어짐', 11, 400, 'tx-m', 'middle')
    return svg(900, 242, b, 'wide', '기록 길이별 성능 경향')

# ---------------------------------------------------------------- 본문
ABS = [
 ('Sequential recommendation aims to leverage users’ historical behaviors to predict their next interaction.',
  '순차 추천은 사용자의 과거 행동으로 다음 상호작용을 예측하는 과제예요.'),
 ('Existing works have not yet addressed two main challenges in sequential recommendation.',
  '기존 연구가 아직 풀지 못한 문제가 두 가지 있다고 주장해요.'),
 ('First, user behaviors in their rich historical sequences are often implicit and noisy preference signals, they cannot sufficiently reflect users’ actual preferences.',
  '<b>문제 ①</b> 긴 기록 속 행동은 암묵적(implicit)이고 잡음이 많은 신호라서, 실제 선호를 충분히 드러내지 못해요.'),
 ('In addition, users’ dynamic preferences often change rapidly over time, and hence it is difficult to capture user patterns in their historical sequences.',
  '<b>문제 ②</b> 선호가 시간에 따라 빠르게 바뀌어서, 기록에서 패턴을 잡기 어려워요.'),
 ('In this work, we propose a graph neural network model called SURGE (short for SeqUential Recommendation with Graph neural nEtworks) to address these two issues.',
  '이 두 문제를 풀려고 그래프 신경망 모델 SURGE를 제안해요.'),
 ('Specifically, SURGE integrates different types of preferences in long-term user behaviors into clusters in the graph by re-constructing loose item sequences into tight item-item interest graphs based on metric learning.',
  '<b>방법 ①</b> 느슨한 아이템 <b>시퀀스</b>를 metric learning으로 촘촘한 아이템-아이템 <b>관심 그래프</b>로 다시 만들어, 긴 기록의 여러 선호를 그래프 안의 <b>덩어리(cluster)</b>로 모아요.'),
 ('This helps explicitly distinguish users’ core interests, by forming dense clusters in the interest graph.',
  '그러면 촘촘한 덩어리가 생겨서 <b>핵심 관심</b>이 분명히 구별돼요. (문제 ①에 대한 답)'),
 ('Then, we perform cluster-aware and query-aware graph convolutional propagation and graph pooling on the constructed graph.',
  '<b>방법 ②</b> 그 그래프 위에서 <b>군집 인식·질의(후보) 인식</b> 그래프 합성곱 전파와 <b>그래프 풀링</b>을 해요.'),
 ('It dynamically fuses and extracts users’ current activated core interests from noisy user behavior sequences.',
  '이로써 잡음 많은 기록에서 <b>지금 활성화된 핵심 관심</b>을 동적으로 융합하고 추출해요. (문제 ②에 대한 답)'),
 ('We conduct extensive experiments on both public and proprietary industrial datasets. Experimental results demonstrate significant performance gains of our proposed method compared to state-of-the-art methods.',
  '공개 데이터(Taobao)와 사내 데이터(Kuaishou)에서 최신 방법들보다 크게 좋았어요.'),
 ('Further studies on sequence length confirm that our method can model long behavioral sequences effectively and efficiently.',
  '시퀀스 길이 실험으로 <b>긴 기록을 효과적이고 효율적으로</b> 다룬다는 걸 확인했어요.'),
]
abs_html = '<div class="abs"><div>#</div><div>한국어 풀이</div><div>원문</div>' + ''.join(
    f'<div class="k">{k+1}</div><div>{ko}</div><div class="en">{en}</div>' for k, (en, ko) in enumerate(ABS)) + '</div>'

HL = ' class="hl"'
t2_rows = ''.join(f'<tr{HL if r[0] == "SURGE" else ""}><td>{r[0]}</td><td>{r[9]}</td>' + ''.join(f'<td class="n">{v:.4f}</td>' for v in r[1:9]) + '</tr>' for r in T2)

html = f'''<title>SURGE 논문 다시 읽기</title>
<meta name="claude:viewport-fit-sides" content="cover">
{STYLE}
<main>
<div class="eyebrow">SIGIR 2021 · Chang et al. · Tsinghua University · Kuaishou</div>
<h1>SURGE 논문 다시 읽기</h1>
<p class="meta">Sequential Recommendation with Graph Neural Networks. Abstract부터 결론까지 논문 순서대로 읽어요. 식 번호는 논문 그대로이고, 그림 속 기록과 숫자는 이해를 돕는 예시예요(실험 결과 표의 숫자만 논문 값).</p>

<div class="summary">
<p class="lead"><b>한 줄 요약.</b> 길고 잡음 많은 클릭 기록을 <b>비슷한 것끼리 이은 그래프</b>로 바꾸고, 그래프 위에서 신호를 모아 <b>관심 덩어리 몇 개로 압축</b>한 뒤, 그 덩어리를 시간순으로 읽어 다음 클릭을 예측해요.</p>
</div>

<nav class="toc">
<a href="#s1"><span class="no">1</span>Abstract 한 문장씩</a>
<a href="#s2"><span class="no">2</span>Introduction: 문제와 기존 연구</a>
<a href="#s3"><span class="no">3</span>문제 정의</a>
<a href="#s4"><span class="no">4</span>전체 구조 (Figure 1)</a>
<a href="#s5"><span class="no">5</span>A. 관심 그래프 만들기 (3.1)</a>
<a href="#s6"><span class="no">6</span>B. 관심 융합 (3.2)</a>
<a href="#s7"><span class="no">7</span>C. 관심 추출 풀링 (3.3)</a>
<a href="#s8"><span class="no">8</span>D. 예측과 손실 (3.4)</a>
<a href="#s9"><span class="no">9</span>실험 설정 (4.1)</a>
<a href="#s10"><span class="no">10</span>실험 결과 (4.2–4.4)</a>
<a href="#s11"><span class="no">11</span>관련 연구와 결론 (5–6)</a>
<a href="#s12"><span class="no">12</span>읽을 때 주의할 점</a>
</nav>

<h2 id="s1"><span class="no">1</span>Abstract 한 문장씩</h2>
<p>Abstract는 <b>문제 두 개 → 방법 두 개 → 결과</b> 순서로 짜여 있어요. 방법 ①이 문제 ①(잡음)을, 방법 ②가 문제 ②(관심 변화)를 맡는 구조예요.</p>
{abs_html}
<div class="q">핵심 연구 질문: 길고 잡음 많은 클릭 기록에서, 지금 활성화된 핵심 관심만 골라내 다음 클릭을 예측할 수 있는가?</div>
<p>Abstract에 나온 용어:</p>
<div class="tbl"><table>
<tr><th>용어</th><th>뜻</th></tr>
<tr><td>implicit feedback (암묵적 피드백)</td><td>클릭·시청처럼 사용자가 "좋다"고 말하지 않은 행동. 좋아요·즐겨찾기 같은 explicit feedback과 달리 하나만으로는 선호를 알 수 없어요</td></tr>
<tr><td>loose sequence → tight graph</td><td>시간순으로 늘어놓은 기록(느슨함)을, 비슷한 아이템끼리 선으로 이은 그래프(촘촘함)로 바꾸기. 시간상 멀리 떨어진 같은 관심이 이웃이 돼요</td></tr>
<tr><td>metric learning</td><td>"무엇을 비슷하다고 볼지"를 재는 자(척도)를 데이터로 학습하는 것. 여기서는 가중 코사인의 가중치 w를 학습해요</td></tr>
<tr><td>core interest (핵심 관심)</td><td>그래프에서 촘촘한 덩어리. 비슷한 클릭이 여러 번 모인 곳이에요</td></tr>
<tr><td>cluster-aware / query-aware</td><td>정보를 섞을 때 "받는 노드가 덩어리 중심인가"와 "보내는 노드가 후보(query)와 관련 있나"를 함께 보는 것</td></tr>
<tr><td>graph pooling</td><td>노드 n개짜리 그래프를 덩어리 m개(m &lt; n)로 줄이는 것. CNN의 풀링처럼 크기를 줄여요</td></tr>
<tr><td>activated interest</td><td>지금 시점에 켜져 있는 관심. 예전엔 있었지만 지금은 꺼진(deactivated) 관심과 구별해요</td></tr>
</table></div>

<h2 id="s2"><span class="no">2</span>Introduction: 문제와 기존 연구</h2>
<p>순차 추천은 고정된 선호가 아니라 <b>변하는 선호</b>를 잡으려는 과제예요. 논문의 예: 월드컵 기간에만 축구 뉴스를 보는 사용자(단기 선호).</p>
<h3>기존 연구의 세 갈래와 공통 한계</h3>
<div class="tbl"><table>
<tr><th>갈래</th><th>대표 모델</th><th>방식</th><th>한계 (논문 주장)</th></tr>
<tr><td>규칙·어텐션으로 시간 가중치</td><td>FPMC, DIN</td><td>사람이 정한 규칙이나 어텐션으로 과거 아이템에 시간 감쇠 가중치</td><td rowspan="3">공통적으로 <b>최근 행동에 집중</b>하고, 오래된 기록을 충분히 캐내지 못해 지금 관심을 정확히 추정하지 못함</td></tr>
<tr><td>순환망(RNN)으로 요약</td><td>GRU4Rec, DIEN</td><td>기록을 차례로 읽어 상태 하나로 요약</td></tr>
<tr><td>장기·단기 함께</td><td>SLi-Rec</td><td>장기와 단기 관심을 따로 모델링해 합침</td></tr>
</table></div>
<p>그래서 아직 잘 풀리지 않은 문제가 두 가지라고 말해요.</p>
{fig(fig_problems(), '<b>그림 1.</b> 논문이 말하는 두 문제. ① 클릭 중에는 관심 없이 누른 것(회색)이 섞여 있어, 비슷한 클릭이 많이 모인 곳을 진짜 관심으로 봐야 해요. ② 시점마다 켜져 있는 관심이 달라서(아래 띠), 지금 켜진 관심을 골라야 해요. 기록은 설명용 예시예요.')}
<h3>제안과 기여 (논문이 정리한 세 가지)</h3>
<ol>
<li>암묵적 신호와 빠르게 바뀌는 선호를 함께 고려하는 <b>새 관점</b>으로 순차 추천에 접근해요.</li>
<li>아이템-아이템 관심 그래프 위에서 GNN으로 <b>암묵적 신호를 명시적 신호로 모으고</b>, <b>동적 풀링</b>으로 활성화된 핵심 선호만 남겨요.</li>
<li>대규모 실제 데이터 두 개에서 최신 방법보다 좋고, <b>긴 기록을 효과적·효율적으로</b> 다룬다는 걸 보여요.</li>
</ol>

<h2 id="s3"><span class="no">3</span>문제 정의</h2>
<div class="eq"><span>입력: 사용자의 상호작용 기록 {{x<sub>1</sub>, x<sub>2</sub>, …, x<sub>n</sub>}}</span><span class="n">(시간순, n = 기록 길이)</span></div>
<div class="eq"><span>출력: 후보 x<sub>t</sub> 를 (n+1)번째에 상호작용할 확률을 내는 모델</span><span class="n">(클릭 예측, CTR 방식)</span></div>
<p>즉 "다음 아이템 하나를 고르기"가 아니라 <b>(기록, 후보 하나) → 클릭 확률</b>을 내는 이진 분류로 풀어요. 후보가 여러 개면 후보마다 따로 계산해요.</p>

<h2 id="s4"><span class="no">4</span>전체 구조 (Figure 1)</h2>
{fig(fig_overview(), '<b>그림 2.</b> 논문 Figure 1을 다시 그린 것. A에서 시퀀스를 그래프로 바꾸고, B에서 이웃끼리 정보를 섞어 핵심 관심을 키우고, C에서 덩어리 m개로 압축한 뒤, D에서 덩어리를 시간순으로 펴서(flatten) 읽고 후보와 함께 클릭 확률을 내요.')}
<div class="tbl"><table>
<tr><th>부분</th><th>절</th><th>입력 → 출력</th><th>푸는 문제</th></tr>
<tr><td>A. 관심 그래프 만들기</td><td>3.1</td><td>아이템 임베딩 n개 → 인접 행렬 A (n × n)</td><td>시간상 흩어진 같은 관심을 이웃으로 만들기</td></tr>
<tr><td>B. 관심 융합 GNN</td><td>3.2</td><td>A, h → 고친 노드 벡터 h′ (n개)</td><td>약한 신호를 강한 신호로 (잡음 약화)</td></tr>
<tr><td>C. 관심 추출 풀링</td><td>3.3</td><td>A, h′ → 덩어리 값 h* (m개), 중요도 γ*</td><td>지금 활성화된 핵심 관심만 남기기</td></tr>
<tr><td>D. 예측</td><td>3.4</td><td>h*, γ*, h_g, 후보 h_t → ŷ</td><td>관심의 변화(진화)를 반영해 예측</td></tr>
</table></div>

<h2 id="s5"><span class="no">5</span>A. 관심 그래프 만들기 (3.1)</h2>
<p>사용자 한 명의 기록마다 무방향 그래프를 하나씩 만들어요. <b>노드 = 기록의 아이템 하나</b>(n개), 각 노드는 임베딩 h<sub>i</sub> ∈ ℝ<sup>d</sup>를 가져요. 목표는 인접 행렬 A를 정하는 거예요.</p>
<p><b>왜 그래프인가.</b> 그래프로 보면 핵심 관심과 주변 관심을 가르기 쉬워요. 핵심 관심 노드는 비슷한 노드와 많이 이어져 <b>차수(degree)가 높고</b>, 비슷한 관심이 자주 나오면 <b>크고 촘촘한 부분 그래프</b>가 돼요. "이웃은 비슷하다, 촘촘한 부분 그래프가 핵심 관심이다"를 사전 가정(prior)으로 둬요.</p>
<p><b>왜 동시 출현이 아니라 유사도 학습인가.</b> 두 아이템이 같이 나온 횟수로 잇는 것도 합리적이지만, 너무 희소해서 사용자마다 연결된 그래프가 잘 안 만들어져요. 그래서 그래프 만들기를 <b>노드 유사도 척도 학습(metric learning)</b> 문제로 바꾸고, 추천 과제와 함께 학습해요. 함수라서 테스트 때 새 아이템에도 바로 쓸 수 있다고 해요(귀납적 학습).</p>
<div class="eq"><span>M<sub>ij</sub> = cos( w ⊙ h<sub>i</sub>, w ⊙ h<sub>j</sub> )</span><span class="n">식 (1) 가중 코사인, ⊙ = 원소별 곱, w = 학습되는 가중치 벡터</span></div>
<div class="eq"><span>M<sup>δ</sup><sub>ij</sub> = cos( w<sub>δ</sub> ⊙ h<sub>i</sub>, w<sub>δ</sub> ⊙ h<sub>j</sub> ),&nbsp; M<sub>ij</sub> = (1/φ) Σ<sub>δ</sub> M<sup>δ</sup><sub>ij</sub></span><span class="n">식 (2) 헤드 φ개의 평균</span></div>
<div class="eq"><span>A<sub>ij</sub> = 1 (M<sub>ij</sub> ≥ Rank<sub>εn²</sub>(M)),&nbsp; 0 (그 외)</span><span class="n">식 (3) 전체에서 상위 ε 비율만 남김</span></div>
{fig(fig_graph(), f'<b>그림 3.</b> 예시 기록 8개(스포츠 2, 정치 4, 요리 1, 잡음 1)로 실제로 계산한 유사도 행렬과 그래프. ε = {EPS}이면 64칸 중 상위 {int(EPS*64)}칸(대각선 8칸 포함)이 1이 돼서, 정치 4개는 서로 다 이어진 촘촘한 덩어리가 되고 요리와 잡음은 외톨이가 돼요. 임베딩은 설명용으로 정한 3차원 값이고 w = 1로 뒀어요.')}
<h3>희소화를 왜 "전체 상대 순위"로 하나</h3>
<p>코사인은 −1~1이라 그대로 쓰면 모든 쌍이 이어진 완전 그래프가 돼요. 계산이 무겁고, 중요하지 않은 선(잡음)이 섞이고, 뒤의 합성곱이 중요한 부분에 집중하지 못해요. 그래서 일부만 남기는데, 남기는 방식이 셋 있어요.</p>
<div class="tbl"><table>
<tr><th>방식</th><th>규칙</th><th>문제</th></tr>
<tr><td>전체 절대 임계값</td><td>M ≥ 0.9 이면 연결</td><td>임베딩이 학습되며 값의 분포가 움직이면 선이 하나도 없거나 모두 이어져요</td></tr>
<tr><td>노드별 상위 k개</td><td>노드마다 가장 비슷한 k개</td><td>모든 노드의 차수가 같아져서 "어디가 촘촘한지" 정보가 사라져요</td></tr>
<tr class="hl"><td>전체 상대 순위 (SURGE)</td><td>n² 칸 전체에서 상위 ε 비율</td><td>비율만 정하니 분포가 움직여도 밀도가 일정하고, 촘촘한 곳과 성긴 곳의 차이가 남아요</td></tr>
</table></div>

<h2 id="s6"><span class="no">6</span>B. 관심 융합: 군집·질의 인식 그래프 어텐션 합성곱 (3.2)</h2>
<p>그래프가 생기면 핵심 관심은 큰 덩어리, 주변 관심은 작은 덩어리가 돼요. 이제 이웃끼리 정보를 모아(aggregate) <b>약한 신호를 강한 신호로</b> 만들어요. 각 노드 i는 이웃 j에게서 비율 E<sub>ij</sub>만큼 받아 자기 벡터를 고쳐요.</p>
<div class="eq"><span>h′<sub>i</sub> = σ( W<sub>a</sub> · Aggregate( E<sub>ij</sub> · h<sub>j</sub> | j ∈ N<sub>i</sub> ) + h<sub>i</sub> )</span><span class="n">식 (4) Aggregate = 합, + h_i = 잔차 연결</span></div>
<div class="eq"><span>h′<sub>i</sub> = ∥<sub>δ=1..φ</sub> σ( W<sub>a</sub><sup>δ</sup> · Aggregate( E<sup>δ</sup><sub>ij</sub> · h<sub>j</sub> ) + h<sub>i</sub> )</span><span class="n">식 (5) 헤드 φ개를 이어 붙임 → φd′ 차원</span></div>
<p>섞는 비율 E<sub>ij</sub>는 두 가지를 함께 봐요.</p>
<div class="cards">
<div class="card"><h4>α<sub>i</sub>: 군집 인식 (cluster-aware)</h4><p>받는 노드 i를 덩어리의 중심(medoid) 후보로 보고, i의 k-hop 이웃 평균 h<sub>i</sub><sup>c</sup>와 비교해 "i가 덩어리 중심인가"를 점수로 내요.</p><p class="m">α<sub>i</sub> = Attention<sub>c</sub>( W<sub>c</sub>h<sub>i</sub> ∥ h<sub>i</sub><sup>c</sup> ∥ W<sub>c</sub>h<sub>i</sub> ⊙ h<sub>i</sub><sup>c</sup> ) (6)</p></div>
<div class="card"><h4>β<sub>j</sub>: 질의 인식 (query-aware)</h4><p>보내는 노드 j가 후보 아이템 h<sub>t</sub>와 얼마나 관련 있는지 점수로 내요. 관련 있는 정보만 남기고 관련 없는 정보는 버리려는 거예요.</p><p class="m">β<sub>j</sub> = Attention<sub>q</sub>( W<sub>q</sub>h<sub>j</sub> ∥ h<sub>t</sub> ∥ W<sub>q</sub>h<sub>j</sub> ⊙ h<sub>t</sub> ) (7)</p></div>
</div>
<div class="eq"><span>E<sub>ij</sub> = softmax<sub>j</sub>( α<sub>i</sub> + β<sub>j</sub> ) = exp(α<sub>i</sub> + β<sub>j</sub>) / Σ<sub>k∈N<sub>i</sub></sub> exp(α<sub>i</sub> + β<sub>k</sub>)</span><span class="n">식 (8) 덧셈형 어텐션, N_i 는 i 자신 포함</span></div>
{fig(fig_fusion(), '<b>그림 4.</b> 정치 노드 5번이 이웃(2·7·8번)과 자기 자신에게서 정보를 받는 장면. 섞는 비율 E는 받는 쪽의 "중심다움" α와 보내는 쪽의 "후보 관련도" β를 더해 softmax로 정해요. 두 Attention은 모두 2층 FFN(LeakyReLU)이에요. 화살표 굵기는 설명용이에요.')}
<p>논문 설명: α<sub>i</sub>는 받는 노드가 정보를 <b>얼마나 받을지</b>, β<sub>j</sub>는 보내는 노드가 <b>얼마나 보낼지</b>를 정해요. 그 결과 핵심 관심은 강해지고 잡음은 약해져요. 후보가 β로 들어가므로 이 단계부터 결과가 <b>후보마다 달라요</b>.</p>

<h2 id="s7"><span class="no">7</span>C. 관심 추출: 그래프 풀링 (3.3)</h2>
<p>융합으로 신호를 키운 뒤, CNN의 풀링처럼 그래프를 <b>줄여서</b> 느슨한 관심을 촘촘한 관심으로 바꿔요. 이때 관심의 분포는 유지해요.</p>
<h3>배정표 S와 덩어리 값</h3>
<div class="eq"><span>S<sub>i:</sub> = softmax( W<sub>p</sub> · Aggregate( A<sub>ij</sub> · h′<sub>j</sub> | j ∈ N<sub>i</sub> ) )</span><span class="n">식 (11) S ∈ ℝ^(n×m), W_p 출력 = 덩어리 수 m</span></div>
<div class="eq"><span>{{h*<sub>1</sub>, …, h*<sub>m</sub>}} = Sᵀ {{h′<sub>1</sub>, …, h′<sub>n</sub>}},&nbsp; {{γ*<sub>1</sub>, …, γ*<sub>m</sub>}} = Sᵀ {{γ<sub>1</sub>, …, γ<sub>n</sub>}}</span><span class="n">식 (9)(10) γ_i = β_i 의 softmax</span></div>
{fig(fig_pool(), '<b>그림 5.</b> S의 행은 노드, 열은 덩어리이고, 한 행은 그 노드가 덩어리마다 속하는 비율(합 1)이에요. Sᵀ를 곱하면 노드 값이 덩어리 값으로 모여요: 벡터 h′를 모으면 덩어리 값 h*, 노드 중요도 γ를 모으면 덩어리 중요도 γ*. SᵀAS는 덩어리끼리의 연결이라 같은 계산을 반복하면 여러 층으로 압축할 수 있어요. 숫자는 설명용이에요.')}
<h3>배정 정규화 세 개 (3.3.2)</h3>
<p>추천 손실의 기울기만으로는 S를 학습시키기 어렵대요. 비볼록 최적화라 초반에 나쁜 국소해에 빠지기 쉽고, 풀링 뒤 덩어리 사이의 <b>시간 순서</b>도 보장되지 않아요. 그래서 벌점 셋을 손실에 더해요.</p>
<div class="eq"><span>L<sub>M</sub> = ‖A − SSᵀ‖<sub>F</sub>,&nbsp; L<sub>A</sub> = (1/n) Σ<sub>i</sub> H(S<sub>i:</sub>),&nbsp; L<sub>P</sub> = ‖P<sub>n</sub>S − P<sub>m</sub>‖<sub>2</sub></span><span class="n">식 (12)(13)(14), P_n = (1…n), P_m = (1…m)</span></div>
{fig(fig_regs(), '<b>그림 6.</b> 세 벌점이 S에 바라는 모양. L<sub>M</sub>: 연결 강도 A와 "같은 덩어리일 확률" SSᵀ가 비슷하게. L<sub>A</sub>: 각 행이 원-핫에 가깝게(엔트로피 0). L<sub>P</sub>: 앞쪽 노드가 앞쪽 덩어리에 배정되도록 S의 0 아닌 칸을 주대각선 가까이. 덩어리 순서를 직접 바꾸는 건 미분이 안 돼서 벌점으로 대신해요.')}
<h3>그래프 읽기 (3.3.3)</h3>
<div class="eq"><span>h<sub>g</sub> = Readout( {{ γ<sub>i</sub> · h′<sub>i</sub>, i ∈ G }} )</span><span class="n">식 (15) Readout = 합, 풀링 전 그래프 전체</span></div>
<p>압축된 그래프와 별도로, 풀링 <b>전</b> 노드 전체를 중요도 γ<sub>i</sub>로 가중합해 그래프 수준 요약 h<sub>g</sub>를 만들어 예측 층에 넣어요. 노드마다의 중요도를 제약해서 풀링 층이 덩어리 정보를 더 잘 뽑게 하려는 목적이에요.</p>

<h2 id="s8"><span class="no">8</span>D. 예측과 손실 (3.4)</h2>
<p>관심은 외부 환경과 내부 인식의 영향으로 계속 변해요(예: 한동안 여러 스포츠, 다른 때는 책). 그래서 덩어리들의 <b>시간 관계</b>를 반영해야 해요. L<sub>P</sub> 덕분에 덩어리 순서가 시간 순서를 따르므로, 풀링된 그래프를 <b>펴면(flatten)</b> 관심 신호가 강화된 짧은 시퀀스가 돼요. 이 시퀀스에는 어떤 순차 모델을 써도 되고, 논문은 AUGRU를 써요.</p>
<div class="eq"><span>h<sub>s</sub> = AUGRU( h*<sub>1</sub>, h*<sub>2</sub>, …, h*<sub>m</sub> )</span><span class="n">식 (16) 갱신 문 u 의 모든 차원에 γ*_k 를 곱함</span></div>
<div class="eq"><span>ŷ = Predict( h<sub>s</sub> ∥ h<sub>g</sub> ∥ h<sub>t</sub> ∥ h<sub>g</sub> ⊙ h<sub>t</sub> )</span><span class="n">식 (17) 2층 FFN</span></div>
<div class="eq"><span>L = −(1/|O|) Σ<sub>o</sub> [ y<sub>o</sub> log ŷ<sub>o</sub> + (1−y<sub>o</sub>) log(1−ŷ<sub>o</sub>) ] + λ‖Θ‖²</span><span class="n">식 (18) + 벌점 L_M, L_A, L_P</span></div>
{fig(fig_pred(), '<b>그림 7.</b> 덩어리 m개를 시간순으로 AUGRU에 넣으면, 중요도 γ*가 작은 덩어리는 기억을 거의 바꾸지 못해서 관심 표류(interest drifting)의 방해가 줄어요. 마지막 상태 h<sub>s</sub>, 그래프 요약 h<sub>g</sub>, 후보 h<sub>t</sub>, h<sub>g</sub>⊙h<sub>t</sub>를 이어 붙여 MLP로 클릭 확률 ŷ를 내요.')}

<h2 id="s9"><span class="no">9</span>실험 설정 (4.1)</h2>
<p>연구 질문 세 개: <span class="pill">RQ1</span>최신 방법보다 좋은가 <span class="pill">RQ2</span>길이가 다른 시퀀스를 효과적·효율적으로 다루나 <span class="pill">RQ3</span>각 구성 요소의 효과는</p>
<div class="tbl"><table>
<tr><th>데이터</th><th class="n">사용자</th><th class="n">아이템</th><th class="n">인스턴스</th><th class="n">평균 기록 길이</th><th>기간과 분할</th></tr>
<tr><td>Taobao (공개, 전자상거래)</td><td class="n">36,915</td><td class="n">64,138</td><td class="n">1,471,155</td><td class="n">39.85</td><td>2017-11-25~12-03 클릭, 상호작용 10개 미만 사용자 제외. 앞 7일 학습, 8일째 검증, 마지막 날 테스트</td></tr>
<tr><td>Kuaishou (사내, 짧은 동영상)</td><td class="n">60,813</td><td class="n">292,286</td><td class="n">14,952,659</td><td class="n">245.88</td><td>2020-10-22~28 클릭, 10-core 필터. 앞 6일 학습, 마지막 날 정오 전 검증, 정오 후 테스트</td></tr>
</table></div>
<div class="cards">
<div class="card"><h4>지표</h4><p><b>AUC</b>: 정답 점수가 오답보다 높을 확률. <b>GAUC</b>: 사용자별 AUC를 클릭 수로 가중 평균. <b>MRR</b>: 첫 정답 순위의 역수 평균. <b>NDCG@2</b>: 상위 2개 안의 순위 가중 적중.</p></div>
<div class="card"><h4>비교 모델</h4><p><b>비순차</b>: NCF, DIN, LightGCN. <b>순차</b>: Caser(CNN), GRU4REC(GRU), DIEN(2층 GRU + AUGRU), SLi-Rec(장·단기 + 시간 인식 LSTM).</p></div>
<div class="card"><h4>하이퍼파라미터</h4><p>Microsoft Recommenders(TensorFlow), Adam 학습률 0.001, 배치 500, 임베딩 40, Xavier 초기화, 예측망 [100, 64] 공통. 최대 길이 50 / 250. 정규화 계수 {{1e−7, 1e−5, 1e−3}}, 풀링 길이 [10…50] / [50…250]에서 탐색.</p></div>
</div>
<p class="small muted">세션 기반 추천의 그래프 모델(SR-GNN 등)은 짧은 세션 안의 반복 전이를 그래프로 잡아요. 긴 기간에는 같은 행동이 잘 반복되지 않아서 순차 추천에는 바로 쓰기 어렵다고 해서 비교에서 뺐어요.</p>

<h2 id="s10"><span class="no">10</span>실험 결과 (4.2–4.4)</h2>
<h3>RQ1. 전체 성능 (Table 2)</h3>
{fig(fig_auc(), '<b>그림 8.</b> Table 2의 AUC. SURGE가 두 데이터 모두 가장 높고, 기록이 긴 Kuaishou에서 차이가 더 커요(AUC +0.04, p &lt; 0.001). 숫자는 논문 표 그대로예요.')}
<div class="tbl"><table>
<tr><th rowspan="2">모델</th><th rowspan="2">종류</th><th colspan="4">Taobao</th><th colspan="4">Kuaishou</th></tr>
<tr><th class="n">AUC</th><th class="n">GAUC</th><th class="n">MRR</th><th class="n">NDCG@2</th><th class="n">AUC</th><th class="n">GAUC</th><th class="n">MRR</th><th class="n">NDCG@2</th></tr>
{t2_rows}
</table></div>
<p>논문이 읽어 낸 세 가지:</p>
<ol>
<li><b>SURGE가 일관되게 가장 좋다.</b> AUC가 Taobao에서 약 0.03, Kuaishou에서 약 0.04 올랐고, 긴 기록(Kuaishou)에서 더 뚜렷해요.</li>
<li><b>순차 모델은 효과적이지만 단기 병목이 있다.</b> 순차 모델(Caser, DIEN, GRU4Rec)이 비순차 모델보다 낫지만, 긴 기록에서는 RNN이 오래된 관심을 잊어서 Caser와 비슷해지고, DIEN은 잡음에 흔들려요.</li>
<li><b>장·단기를 합친다고 늘 좋아지지는 않는다.</b> SLi-Rec은 Taobao에서 AUC는 좋지만 순위 지표가 약하고, 긴 Kuaishou에서는 GRU4Rec보다 못해요. SURGE는 시간 정보를 따로 모델링하지 않고 압축만으로 더 나아요.</li>
</ol>
<h3>RQ2. 기록 길이와 효율 (Figure 2·3, Table 3)</h3>
{fig(fig_length(), '<b>그림 9.</b> 논문 Figure 2의 경향을 옮긴 모양(선의 높이는 설명용). 짧은 기록은 데이터가 부족해 모두 낮고, 길어지면 대부분 올랐다가 잡음 때문에 다시 떨어져요(DIN, DIEN이 가장 크게). SURGE는 가장 긴 그룹에서도 GAUC 0.8919(Taobao), 0.8502(Kuaishou)를 유지해서, 길수록 다른 모델과의 차이가 커져요.')}
{fig(fig_time(), '<b>그림 10.</b> Table 3의 수렴까지 학습 시간. 풀링으로 시퀀스를 줄인 뒤 RNN에 넣으니 순환 단계가 줄고, 잡음이 걸러져 수렴도 빨라요. Kuaishou에서 비교 모델 대비 20% 넘게 빠르다고 해요(DIN 제외). DIN은 긴 기록에서 핵심에 집중하지 못해 학습 곡선이 계속 흔들려요(Figure 3).')}
<h3>RQ3. 구성 요소 효과 (Table 4, Figure 4)</h3>
{fig(fig_ablation(), '<b>그림 11.</b> Table 4의 AUC. 융합(Fusion)을 빼면 가장 크게 떨어지고, 풀링(Extraction)을 빼도 크게 떨어져요. 융합 안에서는 질의 인식과 군집 인식이, 추출 안에서는 그래프 읽기와 정규화가 각각 보탬이 돼요.')}
<ul>
<li><b>관심 융합:</b> 전파 없음 &lt; 군집 인식만 · 질의 인식만 &lt; 둘 다. 핵심 관심 강화와 후보 관심 활성화가 각각 보탬이 돼요.</li>
<li><b>관심 추출:</b> 풀링 없음, 정규화 없음, 가중 읽기 없음보다 전체가 좋아요. 추출이 무관한 잡음을 걸러 모델이 핵심에 집중하게 해요.</li>
<li><b>예측 층 선택 (Figure 4):</b> 풀링된 시퀀스 위에 Attention(DIN), GRU, AUGRU, Time4LSTM 어느 것을 써도 원래 모델보다 좋아져요. DIN 외에는 차이가 작고, γ*를 쓰는 AUGRU가 조금 더 좋아요. 즉 이득은 주로 <b>풀링(압축)</b>에서 와요.</li>
</ul>

<h2 id="s11"><span class="no">11</span>관련 연구와 결론 (5–6)</h2>
<div class="cards">
<div class="card"><h4>순차 추천</h4><p>마르코프 체인(FPMC) → RNN·어텐션 기반 딥러닝. 대부분 최근 행동에 집중해요. 장·단기를 합친 연구(SLi-Rec 등)도 있지만, 관심을 장기·단기로 거칠게 나누는 건 합리적이지 않다고 봐요. SURGE의 관점은 "행동은 약한 선호 신호이고, 어떤 선호는 특정 시점에 꺼져 있다"예요.</p></div>
<div class="card"><h4>추천을 위한 GNN</h4><p>PinSage, NGCF, LightGCN 등이 사용자-아이템 그래프에서 성과를 냈어요. 세션 추천의 GNN은 짧은 세션의 반복 전이를 잡는 것이라 순차 추천과 달라요. SURGE는 그래프 합성곱으로 약한 신호를 모으고, 그래프 풀링으로 긴 기록에서 활성화된 핵심 선호를 뽑는다는 점이 다르다고 해요.</p></div>
<div class="card"><h4>결론과 향후 과제</h4><p>느슨한 시퀀스를 촘촘한 관심 그래프로 바꿔, 잡음 많은 기록에서 활성 핵심 관심을 융합·추출하는 방법을 제안했어요. 향후에는 온라인 A/B 테스트와, 클릭·즐겨찾기 같은 여러 행동 유형을 함께 쓰는 연구를 하겠다고 해요.</p></div>
</div>

<h2 id="s12"><span class="no">12</span>읽을 때 주의할 점</h2>
<p>논문을 정확히 이해하려고 확인한 것들이에요.</p>
<div class="tbl"><table>
<tr><th>위치</th><th>내용</th></tr>
<tr><td>식 (8)</td><td>softmax가 j에 대해 돌고 α<sub>i</sub>는 j와 무관해서, 수식대로면 α<sub>i</sub>는 분자·분모에서 약분돼 E에 영향을 주지 않아요. 그런데 Table 4에서 군집 인식을 빼면 성능이 떨어져요. 공개 코드는 군집 점수를 보내는 쪽에, 질의 점수를 받는 쪽에 붙여 반대로 계산해요.</td></tr>
<tr><td>식 (3)</td><td>계단 함수라 기울기가 없어요. w가 추천 손실로 어떻게 학습되는지는 본문에 없어요. 순위를 셀 때 대각선(자기 자신, 값 1) n칸도 포함돼요.</td></tr>
<tr><td>3.3</td><td>공개 코드의 풀링은 소프트 배정 S와 정규화 셋 대신, 점수 상위 노드만 남기는 방식이에요.</td></tr>
<tr><td>하이퍼파라미터</td><td>헤드 수 φ, 출력 차원 d′, 덩어리 수 m의 최종값은 논문에 없어요(m은 탐색 범위만).</td></tr>
<tr><td>Table 4</td><td>Taobao의 w/o Fusion(MRR 0.0317)과 w/o Regularization(MRR 0.0343)은 AUC에 비해 MRR이 지나치게 낮은데, 설명이 없어요.</td></tr>
<tr><td>임베딩</td><td>아이템 표현은 학습되는 ID 임베딩(공개 코드: 아이템 32 + 카테고리 8 = 40차원)이에요. 내용(텍스트) 정보는 쓰지 않아요.</td></tr>
</table></div>
<p class="small muted">근거: Chang et al., "Sequential Recommendation with Graph Neural Networks", SIGIR '21, pp. 378–387, 그리고 공개 코드(tsinghua-fib-lab/SIGIR21-SURGE). 그림 속 기록·임베딩·S·화살표 굵기는 설명용이고, 표와 막대그래프의 수치는 논문 Table 1–4에서 옮겼어요.</p>
</main>
'''
open(OUT, 'w', encoding='utf-8').write(html)
print('ok', len(html), 'THR', round(THR, 3), 'edges', sum(map(sum, AD)))
