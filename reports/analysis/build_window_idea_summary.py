"""11/26 저녁 정리 문서 → reports/인기도창_아이디어_정리.html

    python reports/analysis/build_window_idea_summary.py

숫자는 AWS 실행 결과(로그 화면)와 이 저장소의 원본 데이터 분석(build_click_lifecycle_doc.py)을 옮겨 적은 것이다.
"""
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "reports/인기도창_아이디어_정리.html"
BLUE, ORANGE, MUTED, GRID = "#2a78d6", "#d9792b", "#5b6577", "#e3e6eb"

# popularity_window.py (AWS, seed 42·123·2026 평균): Transformer + 인기 + 나이 (validation, test)
WIN = [0.5, 1, 2, 3, 6, 12, 24, 48, 72]
TPA = {"H": [(38.84, 39.44), (40.07, 40.77), (41.47, 42.26), (42.43, 43.12), (43.77, 44.12), (44.29, 44.78), (43.58, 44.24), (42.28, 43.36), (42.07, 43.35)],
       "R": [(38.88, 39.68), (40.05, 40.98), (41.46, 42.35), (42.52, 43.21), (43.90, 44.04), (44.41, 44.86), (43.73, 44.23), (42.47, 43.09), (42.26, 43.09)]}
PA_H = [(37.72, 38.11), (38.93, 39.38), (40.49, 40.89), (41.51, 41.99), (42.96, 43.01), (43.32, 43.88), (42.71, 43.38), (41.66, 42.39), (41.57, 42.32)]
P_H = [(37.33, 37.95), (38.54, 39.25), (40.25, 40.70), (41.34, 41.80), (42.82, 43.02), (43.29, 43.92), (42.62, 42.93), (40.04, 40.50), (39.32, 39.80)]
ZERO_H = [(4.74, 20.91), (3.53, 17.31), (2.66, 13.99), (2.22, 12.33), (1.72, 9.81), (1.27, 7.55), (0.95, 5.70), (0.75, 4.41), (0.68, 3.74)]


def line_chart(run):
    w, h, L, R, T, B = 370, 250, 44, 12, 24, 40
    pw, ph = w - L - R, h - T - B
    lo, hi = 38, 45.5
    X = lambda i: L + i / (len(WIN) - 1) * pw
    Y = lambda v: T + (hi - v) / (hi - lo) * ph
    g = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" aria-label="{run}: 인기도 창별 Top-1">',
         f'<text x="{L}" y="14" class="lbl b">{run}</text>']
    for v in [38, 40, 42, 44]:
        g.append(f'<line x1="{L}" y1="{Y(v):.1f}" x2="{w - R}" y2="{Y(v):.1f}" stroke="{GRID}"/>')
        g.append(f'<text x="{L - 6}" y="{Y(v) + 4:.1f}" class="lbl sm m" text-anchor="end">{v}%</text>')
    i12 = WIN.index(12)
    g.append(f'<rect x="{X(i12) - 14:.1f}" y="{T}" width="28" height="{ph}" fill="#eef4ff"/>')
    for k, (col, name) in enumerate([(BLUE, "validation"), (ORANGE, "test")]):
        pts = " ".join(f"{X(i):.1f},{Y(v[k]):.1f}" for i, v in enumerate(TPA[run]))
        g.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2"/>')
        for i, v in enumerate(TPA[run]):
            g.append(f'<circle cx="{X(i):.1f}" cy="{Y(v[k]):.1f}" r="4" fill="{col}" stroke="#fff" stroke-width="2">'
                     f'<title>{run} {WIN[i]:g}시간 {name}: {v[k]:.2f}%</title></circle>')
    for i, wv in enumerate(WIN):
        g.append(f'<text x="{X(i):.1f}" y="{h - 22}" class="lbl sm m" text-anchor="middle">{wv:g}</text>')
    g.append(f'<text x="{L + pw / 2:.1f}" y="{h - 4}" class="lbl sm m" text-anchor="middle">인기도 창 (시간)</text>')
    v12 = TPA[run][i12]
    g.append(f'<text x="{X(i12):.1f}" y="{Y(v12[1]) - 10:.1f}" class="lbl sm b" text-anchor="middle">{v12[1]:.2f}%</text>')
    g.append("</svg>")
    return "\n".join(g)


FLOW = '''<svg viewBox="0 0 780 250" class="chart" role="img" aria-label="12시간 창을 고르게 된 과정">
  <g class="lbl">
  <rect x="10" y="20" width="175" height="92" rx="10" class="box"/>
  <text x="97" y="44" class="b" text-anchor="middle">① 질문</text>
  <text x="97" y="66" class="sm m" text-anchor="middle">정답 기사 클릭은</text>
  <text x="97" y="82" class="sm m" text-anchor="middle">언제 몰려 있나?</text>
  <text x="97" y="98" class="sm m" text-anchor="middle">3시간 창이면 될까?</text>
  <path d="M189 66 H207" class="arr" marker-end="url(#ah)"/>
  <rect x="211" y="20" width="175" height="92" rx="10" class="box"/>
  <text x="298" y="44" class="b" text-anchor="middle">② 원본 데이터 분석</text>
  <text x="298" y="66" class="sm m" text-anchor="middle">기사 수명이 몇 시간</text>
  <text x="298" y="82" class="sm m" text-anchor="middle">근사 test로 창 비교</text>
  <text x="298" y="98" class="sm m" text-anchor="middle">→ 12시간이 좋아 보임</text>
  <path d="M390 66 H408" class="arr" marker-end="url(#ah)"/>
  <rect x="412" y="20" width="175" height="92" rx="10" class="shared"/>
  <text x="499" y="44" class="b" text-anchor="middle">③ AWS 실제 데이터</text>
  <text x="499" y="66" class="sm m" text-anchor="middle">실제 test 행 · 오답</text>
  <text x="499" y="82" class="sm m" text-anchor="middle">Transformer 점수 3 seed</text>
  <text x="499" y="98" class="sm m" text-anchor="middle">창 9개 비교</text>
  <path d="M591 66 H609" class="arr" marker-end="url(#ah)"/>
  <rect x="613" y="20" width="157" height="92" rx="10" class="pos"/>
  <text x="691" y="44" class="b" text-anchor="middle">④ 12시간 선택</text>
  <text x="691" y="66" class="sm m" text-anchor="middle">validation 최고</text>
  <text x="691" y="82" class="sm m" text-anchor="middle">H · R, 세 점수 모두</text>
  <text x="691" y="98" class="sm m" text-anchor="middle">test +0.5~0.6%p</text>
  <rect x="10" y="140" width="760" height="92" rx="10" fill="#fff6e6" stroke="#e9c27a"/>
  <text x="26" y="164" class="b">검산과 다음 단계</text>
  <text x="26" y="188" class="sm">· 24시간 행이 기존 결과와 정확히 같음 (Transformer + 인기 + 나이 H 44.24% · R 44.23%, 인기 + 나이 H 43.38%) → 계산이 맞다</text>
  <text x="26" y="208" class="sm">· 다음: 최종 조합(Transformer + 인기 + 나이 + 아이디어 2)을 12시간 창으로 seed 3개 다시 계산</text>
  <text x="26" y="226" class="sm">· 아이디어 2를 최근 50개 + 이전 history로 나눈 E 도 함께 계산 (--split-history)</text>
  </g>
</svg>'''


def rows(vals, fmt="{:.2f}%"):
    return "".join(f'<td class="n">{fmt.format(v)}</td>' for v in vals)


win_tbl = "".join(
    f'<tr{" class=\"hl\"" if w == 12 else ""}><td>{w:g}시간</td>'
    f'<td class="n">{z[0]:.2f}% / {z[1]:.2f}%</td><td class="n">{p[0]:.2f}% · {p[1]:.2f}%</td>'
    f'<td class="n">{pa[0]:.2f}% · {pa[1]:.2f}%</td><td class="n">{h[0]:.2f}% · <b>{h[1]:.2f}%</b></td>'
    f'<td class="n">{r[0]:.2f}% · <b>{r[1]:.2f}%</b></td></tr>'
    for w, z, p, pa, h, r in zip(WIN, ZERO_H, P_H, PA_H, TPA["H"], TPA["R"]))

head = (REPO / "reports/NRMS_동작방식_적용계획.html").read_text(encoding="utf-8")
head = head[: head.index("<main>")].replace("<title>NRMS 동작 방식</title>", "<title>인기도 창과 아이디어 정리</title>")

body = f"""<main>
<h1>인기도 창 12시간, 아이디어 1 · 2, TIGER 결과 정리</h1>
<p class="sub">11/26 저녁 (집에서 이어서 한 작업) · EB-NeRD v2 · 숫자는 AWS 실행 결과와 원본 데이터 분석</p>

<div class="box1"><b>한눈에</b>
<ul style="margin:6px 0 0 0">
<li><b>공개 TIGER를 우리 설정 그대로</b> 돌리면 3-seed 평균 Top-1 H 20.19% · R 20.63% · V1 20.42%로 무작위(20%) 수준이다. 우리 Transformer(26.6~27.7%)와 약 7%p 차이는 학습 방식(정답 SID 생성 vs 후보 중 고르기)에서 온다.</li>
<li><b>아이디어 2(history 사건 매칭)</b>는 최근 50개 · 넓힌 후보로 3-seed H 44.93% · R 44.76% (+0.69 · +0.54%p, seed 3개 모두 상승). <b>아이디어 1(사건 인기도)</b>은 효과 없음.</li>
<li><b>인기도 창</b>은 24시간보다 <b>12시간</b>이 낫다. 기사는 발행 후 12시간 안에 평생 클릭의 91%를 받는다. AWS 실제 데이터에서 validation 최고, test +0.54 · +0.63%p. 3시간은 오히려 나쁘다.</li>
<li>다음: 최종 조합을 12시간 창으로 다시 계산하고, 아이디어 2를 "최근 50개 + 그 이전 history"로 나눠 Transformer가 못 보는 오래된 클릭이 도움이 되는지 본다.</li>
</ul></div>

<h2>1. 공개 TIGER, 우리 설정 그대로 (3 seed)</h2>
<p>모델 크기 · 학습 설정 · 데이터를 우리 Transformer와 똑같이 맞추고 "무엇을 배우나"(손실)만 TIGER 방식으로 두었다.</p>
<div class="tbl"><table>
<tr><th></th><th class="n">TIGER seed 42 / 123 / 2026</th><th class="n">TIGER 평균</th><th class="n">AUC 평균</th><th class="n">우리 Transformer</th><th class="n">차이</th></tr>
<tr><td>H</td>{rows([20.09, 20.09, 20.39], "{:.2f}")}<td class="n"><b>20.19%</b></td><td class="n">0.509</td><td class="n">27.67%</td><td class="n">−7.5%p</td></tr>
<tr><td>R</td>{rows([21.13, 20.41, 20.35], "{:.2f}")}<td class="n"><b>20.63%</b></td><td class="n">0.518</td><td class="n">27.51%</td><td class="n">−6.9%p</td></tr>
<tr><td>V1</td>{rows([20.89, 20.21, 20.17], "{:.2f}")}<td class="n"><b>20.42%</b></td><td class="n">0.511</td><td class="n">26.57%</td><td class="n">−6.2%p</td></tr>
</table></div>
<ul>
<li>모든 실행에서 best epoch이 1이고 6 epoch에서 멈췄다. TIGER는 학습할수록 train에서 클릭된 기사의 SID를 외우는데, test 정답은 대부분 새 기사라 도움이 안 된다. 그래서 실행 하나가 약 15분으로 예상(1시간)보다 빨랐다.</li>
<li>레벨별로는 c1만 쓸 때 25%대로 가장 좋고, c3까지 더하면 무작위로 떨어진다. 우리 Transformer의 "c3가 해롭다"와 같은 방향이다.</li>
</ul>

<h2>2. 아이디어 1 · 2 (3 seed)</h2>
<p>A = Transformer + 인기 + 나이, B = A + 아이디어 1(사건 인기도), C = A + 아이디어 2(history 사건 매칭), D = 둘 다. 가중치는 validation 앞 절반에서 고르고 test에 한 번. 인기도 창은 24시간(12시간 재계산 전).</p>
<div class="tbl"><table>
<tr><th></th><th class="n">A</th><th class="n">B</th><th class="n">C (최종)</th><th class="n">D</th></tr>
<tr><td>H 평균</td><td class="n">44.24%</td><td class="n">44.24%</td><td class="n"><b>44.93%</b></td><td class="n">44.86%</td></tr>
<tr><td>H seed별</td><td class="n">44.21 / 44.14 / 44.37</td><td class="n">같음 (λe = 0)</td><td class="n">44.88 / 44.86 / 45.05</td><td class="n">44.81 / 44.73 / 45.05</td></tr>
<tr><td>R 평균</td><td class="n">44.22%</td><td class="n">44.09%</td><td class="n"><b>44.76%</b></td><td class="n">44.76%</td></tr>
<tr><td>R seed별</td><td class="n">44.23 / 44.35 / 44.09</td><td class="n">43.96 / 44.27 / 44.03</td><td class="n">44.79 / 44.77 / 44.72</td><td class="n">같음</td></tr>
</table></div>
<p>C의 다른 지표(3-seed): H AUC 0.7458 · MRR 0.6637 · nDCG@5 0.7481, R AUC 0.7460 · MRR 0.6632 · nDCG@5 0.7478. 아이디어 2에서는 H가 seed 3개 모두 R보다 높다(c2 코드 사용 H 128개 vs R 59개, 사건 구분이 세밀). 차이는 0.17%p로 작다.</p>

<h3>아이디어 2 설정을 고른 과정 (seed 42)</h3>
<div class="tbl"><table>
<tr><th>history · 후보 범위</th><th class="n">H validation</th><th class="n">H test</th><th class="n">R validation</th><th class="n">R test</th></tr>
<tr><td>최근 50개 · 기본 후보 (τ ≤ 168h, ν ≤ 0.3, c ≤ 16)</td><td class="n">–</td><td class="n">44.69%</td><td class="n">–</td><td class="n">44.53%</td></tr>
<tr class="hl"><td><b>최근 50개 · 넓힌 후보</b> (τ 336h · 720h · ∞, ν ~1, c ~128 추가)</td><td class="n"><b>44.40%</b></td><td class="n"><b>44.88%</b></td><td class="n"><b>44.38%</b></td><td class="n"><b>44.79%</b></td></tr>
<tr><td>전체 history (평균 300개 이상) · 넓힌 후보</td><td class="n">44.04%</td><td class="n">44.47%</td><td class="n">44.11%</td><td class="n">44.39%</td></tr>
</table></div>
<ul>
<li>기본 후보에서 τ · ν · c가 모두 범위 끝으로 뽑혀 후보를 넓혔고, 실제로 바깥 값이 뽑혔다.</li>
<li>뽑힌 값 (seed 42 / 123 / 2026): H τ 720h · ∞ · ∞, R τ ∞ · 720h · ∞, ν는 6번 모두 0.2, c는 32~64. <b>시간 감쇠가 거의 없는 쪽</b>이 뽑혔다.</li>
<li>전체 history는 validation에서도 낮았다. 오래된 클릭이 섞여 최근 취향이 묻힌 것으로 보인다.</li>
</ul>

<h3>τ(타우)의 뜻</h3>
<p>τ는 "예전 클릭을 얼마나 오래 기억해 줄지"이다. 가중치 w = exp(−경과 시간 / τ). τ가 작으면 최근 클릭만 보고(요즘 관심사), 크면 예전 클릭도 똑같이 본다(꾸준한 취향).
실제 사용자 한 명의 예시는 <a href="아이디어2_타우_설명.html">아이디어 2 τ 설명</a>에 있다. 최근 클릭 50개 중 1개만 3시간 전 범죄 기사이고 49개가 7~11일 전 뉴스 · 스포츠인 사용자에게, τ = 24시간이면 범죄 기사(오답)를, τ = 720시간이면 뉴스 기사(정답)를 고른다.
validation에서 큰 τ가 뽑힌 것은 EB-NeRD 사용자가 <b>꾸준히 보던 주제</b>를 더 많이 클릭한다는 뜻이다.</p>

<h3>seed가 필요한 실험</h3>
<p>seed는 Transformer를 학습할 때만 쓰인다. 인기도 · 나이 · 아이디어 1 · 2는 데이터에서 계산하는 값이라 seed와 상관없다.
그래서 <b>인기 + 나이만</b>은 한 번이면 끝이고, <b>Transformer 점수 위에 더하는 실험</b>(A~D)은 seed 3개의 Transformer 점수마다 따로 계산해 평균을 낸다.</p>

<h3>w를 다른 식으로 바꾸는 것 (MIND까지 고려)</h3>
<div class="tbl"><table>
<tr><th>w 방법</th><th>EB-NeRD</th><th>MIND</th><th>판단</th></tr>
<tr><td>시간 감쇠 (τ 유한)</td><td>가능</td><td>불가 (MIND history에 클릭 시각 없음)</td><td>공통으로 못 씀</td></tr>
<tr><td>τ = ∞ (최근 N개의 사건별 클릭 수 비율)</td><td>가능</td><td>가능</td><td>EB-NeRD에서도 6번 중 4번 뽑힘 → 공통 정의로 적합</td></tr>
<tr><td>읽은 시간 · 스크롤</td><td>가능</td><td>불가</td><td>제외</td></tr>
<tr><td>lift (평균보다 특히 많이 보는 사건 우대)</td><td>가능</td><td>가능</td><td>후보 (인기도 항과 덜 겹침)</td></tr>
</table></div>

<h3>아이디어 1이 안 되는 이유와 개선 방향</h3>
<ul>
<li>사건 클릭 수 안에 그 기사 자신의 클릭이 들어 있어 <b>기사 인기도와 거의 같은 정보</b>이다.</li>
<li>사건 인기도가 도울 곳은 "기사는 클릭이 없는데 사건은 뜨거운" 경우인데, test 정답 중 직전 클릭 0은 1% 남짓이다.</li>
<li>모두에게 같은(비개인화) 점수라 인기도와 역할이 겹친다. 반대로 아이디어 2는 사람마다 달라서 효과가 있었다.</li>
<li>개선 후보: 자기 클릭 빼기(같은 사건 <i>다른</i> 기사들의 인기), 개인화 결합(사건 인기도 × 매칭), 사건 크기로 나누기, 창 따로 정하기(사건은 기사보다 오래가므로 12시간보다 길 수 있음). 기대 효과는 작다.</li>
</ul>

<h2>3. 데이터 분석: test 기사의 클릭 수명</h2>
<p>자세한 그림은 <a href="test기사_클릭수명_분석.html">test 기사 클릭 수명 분석</a>에 있다.</p>
<div class="tbl"><table>
<tr><th>구간</th><th>기간</th><th class="n">총 클릭</th><th class="n">하루 클릭</th><th class="n">클릭된 기사</th><th class="n">기사당 중앙값 · 90%</th></tr>
<tr><td>train</td><td>05-18 07:00 ~ 05-25 07:00 (7일)</td><td class="n">234,277</td><td class="n">3.3만</td><td class="n">1,892</td><td class="n">8 · 377</td></tr>
<tr><td>validation 절반</td><td>~ 05-28 21:21 (3.6일)</td><td class="n">123,148</td><td class="n">3.4만</td><td class="n">1,354</td><td class="n">4 · 312</td></tr>
<tr><td>test</td><td>~ 06-01 07:00 (3.4일)</td><td class="n">123,141</td><td class="n">3.6만</td><td class="n">1,312</td><td class="n">4 · 324</td></tr>
</table></div>
<ul>
<li>기사 사이 차이가 크다: 클릭 1~2개 기사가 32~42%, 상위 10% 기사가 클릭의 43~52%.</li>
<li><b>기사는 짧게 산다</b>: 첫 클릭은 발행 후 중앙값 5.5분, 반감기 중앙값 2.3시간, 발행 후 3시간에 평생 클릭의 51%, 12시간에 91%, 24시간에 97%.</li>
<li><b>test 정답은 갓 나온 기사</b>: 노출 시점 나이 중앙값 3.2시간, 마지막 클릭이 노출 중앙값 0.9분 전. 노출 전 72시간 클릭 중 직전 3시간에 52%, 12시간에 82%.</li>
<li>test 클릭의 98%는 train 기간에 한 번도 클릭되지 않은 기사, 86%는 test 기간 중 발행된 기사를 향한다.</li>
</ul>
<div class="tbl"><table>
<tr><th>노출 시점 직전 클릭 수 (test)</th><th class="n">정답 중앙값 (25~75%)</th><th class="n">오답 중앙값 (25~75%)</th><th class="n">0인 비율 정답 / 오답</th></tr>
<tr><td>3시간</td><td class="n">87 (35~170)</td><td class="n">31 (4~98)</td><td class="n">2.2% / 12.5%</td></tr>
<tr class="hl"><td>12시간</td><td class="n">140 (64~250)</td><td class="n">61 (9~166)</td><td class="n">1.3% / 7.3%</td></tr>
<tr><td>24시간</td><td class="n">157 (70~268)</td><td class="n">73 (14~187)</td><td class="n">1.0% / 5.3%</td></tr>
</table></div>

<h2>4. 인기도 창 12시간을 고른 과정</h2>
<div class="fig">{FLOW}
<p class="cap"><b>그림 1.</b> 질문에서 선택까지. 근사 분석에서 가설을 세우고, AWS의 실제 test 행과 실제 Transformer 점수로 확인했다. 창은 validation으로만 골랐다.</p></div>
<div class="fig" style="display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:8px">
{line_chart("H")}
{line_chart("R")}
<p class="cap" style="grid-column:1/-1"><span style="color:{BLUE}">●</span> validation · <span style="color:{ORANGE}">●</span> test. <b>그림 2.</b> Transformer + 인기 + 나이의 창별 Top-1 (seed 3개 평균, 실제 데이터). 두 실험 모두 12시간(파란 띠)에서 가장 높다. 점에 마우스를 올리면 값이 보인다.</p></div>
<div class="tbl"><table>
<tr><th>창</th><th class="n">H 클릭 0 정답 / 오답</th><th class="n">H 인기만 val · test</th><th class="n">H 인기+나이 val · test</th><th class="n">H T+인기+나이 val · test</th><th class="n">R T+인기+나이 val · test</th></tr>
{win_tbl}
</table></div>
<h3>왜 12시간인가: "식은 기사"</h3>
<p>식은 기사 = 발행된 지 반나절이 넘어 받을 클릭을 거의 다 받은, 지금은 거의 안 눌리는 기사. 예: 20시간 전 발행된 기사 X와 2시간 전 발행된 기사 Y가 함께 후보로 뜬 경우(숫자는 예시).</p>
<div class="tbl"><table>
<tr><th></th><th class="n">X (식은 기사)</th><th class="n">Y (지금 뜨는 기사)</th><th>결과</th></tr>
<tr><td>직전 24시간 클릭</td><td class="n">300</td><td class="n">150</td><td>X가 더 인기 있어 보여 틀림</td></tr>
<tr class="hl"><td>직전 12시간 클릭</td><td class="n">40</td><td class="n">150</td><td>X의 전성기 클릭이 창 밖으로 빠져 Y를 맞힘</td></tr>
</table></div>
<p>3시간은 지금 뜨는 기사끼리도 클릭 수가 작고 들쭉날쭉해 0 · 동점이 많아진다(오답 클릭 0이 12%). 12시간은 지금 뜨는 기사의 클릭은 충분히 담고 식은 기사의 옛 클릭은 뺀다.
12시간 창은 <b>기본 인기도 항</b>의 창이다(아이디어 1의 사건 인기도도 같은 창을 쓴다, 아이디어 2는 창 대신 τ).</p>

<h2>5. 아이디어 2: 최근 50개 + 이전 history (E)</h2>
<p>"최근 50개는 Transformer도 보니 전체 history가 더 의미 있지 않나"라는 질문을 직접 확인하는 실험이다. 전체를 하나로 섞으면 최근 취향이 묻히므로 둘로 나눈다.</p>
<pre>최종 점수 = Transformer + λ·인기 + μ·나이 + c최근 · 매칭(최근 50개) + c이전 · 매칭(51번째 이후 클릭)</pre>
<ul>
<li>λ, μ는 C의 값을 두고, τ · ν · c최근 · c이전을 validation에서 고른다. c이전 = 0이면 C와 같으므로 E의 validation은 C보다 낮을 수 없다.</li>
<li><b>c이전 &gt; 0이 뽑히고 오르면</b> Transformer가 못 보는 오래된 history에 추가 정보가 있다는 근거. <b>c이전 = 0</b>이면 지금 설정이 맞다.</li>
<li>MIND도 history 전체가 있으므로 그대로 쓸 수 있다(시각이 없으니 τ = ∞).</li>
</ul>

<h2>6. 남은 일</h2>
<div class="tbl"><table>
<tr><th>일</th><th>상태</th></tr>
<tr><td>최종 조합을 12시간 창으로 (seed 3개)</td><td>AWS 실행 · <code>--max-history 50 --wide-grid --window-hours 12</code></td></tr>
<tr><td>E: 최근 50개 + 이전 history 매칭</td><td>옵션 준비됨 · <code>--split-history</code> (이전 history 계산에 메모리가 커서 seed를 하나씩)</td></tr>
<tr><td>아이디어 1 창 따로 정하기, lift</td><td>필요하면 옵션 추가</td></tr>
<tr><td>오리지널 NRMS (Microsoft Recommenders)</td><td>교수님 답 대기</td></tr>
<tr><td>보고서 · PPT 반영 (TIGER, 아이디어 2, 인기도 창, 인기 + 나이만)</td><td>실험 끝난 뒤</td></tr>
</table></div>
</main></body></html>
"""
OUT.write_text(head + body, encoding="utf-8")
print("저장:", OUT)
