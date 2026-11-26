"""test 기간 기사들의 클릭 수명 분석 → reports/test기사_클릭수명_분석.html

    python reports/analysis/build_click_lifecycle_doc.py

입력: data/raw/{train,validation}/behaviors.parquet, data/raw/articles.parquet (원본, 읽기만)
클릭 기록 = train + validation behaviors 의 모든 클릭 (클릭 시각 = 그 노출의 impression_time) — 인기도와 같은 정의.
test = validation 노출 중 2023-05-28 21:21:57 이후 (AWS popularity_window.py 가 출력한 H 의 실제 test 시작 시각).
"""
import html
from pathlib import Path

import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "data/raw"
OUT = REPO / "reports/test기사_클릭수명_분석.html"
S = lambda x: int(np.datetime64(x, "s").astype(np.int64))
START, TRAIN_END, TEST0, END = S("2023-05-18T07:00:00"), S("2023-05-25T07:00:00"), S("2023-05-28T21:21:57"), S("2023-06-01T07:00:00")
BLUE, BLUE_L, INK, MUTED, GRID = "#2a78d6", "#9fbdec", "#14181f", "#5b6577", "#e3e6eb"


def ts(s):
    return s.cast(pl.Datetime("us")).dt.epoch("s").to_numpy()


def q(x, ps=(10, 25, 50, 75, 90)):
    return [float(v) for v in np.nanpercentile(x, ps)]


# ---------------------------------------------------------------- 계산
cols = ["impression_id", "impression_time", "article_ids_clicked"]
b = pl.concat([pl.read_parquet(RAW / s / "behaviors.parquet", columns=cols) for s in ("train", "validation")])
led = b.select(["impression_time", "article_ids_clicked"]).explode("article_ids_clicked").drop_nulls()
A = led["article_ids_clicked"].to_numpy().astype(np.int64)
T = ts(led["impression_time"])
arts = pl.read_parquet(RAW / "articles.parquet", columns=["article_id", "published_time", "category_str"])
pub = dict(zip(arts["article_id"].to_list(), ts(arts["published_time"]).tolist()))
catd = dict(zip(arts["article_id"].to_list(), arts["category_str"].to_list()))
KO = {"sport": "스포츠", "nyheder": "뉴스", "underholdning": "연예", "krimi": "범죄", "penge": "돈·경제", "musik": "음악",
      "nationen": "국내 시사", "forbrug": "소비", "sex_og_samliv": "성·연애", "biler": "자동차"}

# 날짜별 노출 수
bt = ts(b["impression_time"])
day0 = START - 7 * 3600   # 05-18 00:00
day_idx = (bt - day0) // 86400
days = [(str(np.datetime64(day0 + 86400 * i, "s"))[:10], int((day_idx == i).sum())) for i in range(int(day_idx.max()) + 1)]

# test 클릭
mt = (T >= TEST0) & (T < END)
tA, tT = A[mt], T[mt]
test_clicks, test_articles = int(mt.sum()), int(len(np.unique(tA)))
train_clicked = set(A[T < TRAIN_END].tolist())
share_seen_train = float(np.mean([x in train_clicked for x in tA]))
age_click = np.array([(t - pub.get(int(x), np.nan)) / 3600 for x, t in zip(tA, tT)])
pub_t = np.array([pub.get(int(x), np.nan) for x in tA])
pub_period = {"test 기간에 발행": float(np.nanmean(pub_t >= TEST0)),
              "validation 절반 기간에 발행": float(np.nanmean((pub_t >= TRAIN_END) & (pub_t < TEST0))),
              "train 기간에 발행": float(np.nanmean((pub_t >= START) & (pub_t < TRAIN_END))),
              "데이터 시작 전에 발행": float(np.nanmean(pub_t < START))}

# 기사별 수명 (발행 후 72시간이 데이터 안에 다 들어오고 클릭 20개 이상)
o = np.lexsort((T, A))
A2, T2 = A[o], T[o]
cut = np.flatnonzero(np.diff(A2)) + 1
st, en = np.r_[0, cut], np.r_[cut, len(A2)]
life = []
for s_, e_ in zip(st, en):
    a = int(A2[s_]); p = pub.get(a)
    if p is None or not (START <= p and p + 72 * 3600 <= END) or e_ - s_ < 20:
        continue
    h = (T2[s_:e_] - p) / 3600
    life.append((a, e_ - s_, float(np.median(h)), float(np.percentile(h, 90)), float(h[0])))
L = np.array([x[1:] for x in life])
life_ids = np.array([x[0] for x in life])
half = L[:, 1]
full_mask = np.isin(A, life_ids)
hA = (T[full_mask] - np.array([pub[int(x)] for x in A[full_mask]])) / 3600
edges = [i / 4 for i in range(25)] + list(range(7, 25))
cnt, _ = np.histogram(hA, bins=edges)
tot72 = int(((hA >= 0) & (hA < 72)).sum())
dens = cnt / tot72 / np.diff(edges)           # 시간당 비율
cum_pub = {h: float(((hA >= 0) & (hA < h)).sum() / tot72) for h in (1, 3, 6, 12, 24, 48)}
by_cat = {}
for a, hf in zip(life_ids, half):
    by_cat.setdefault(catd.get(int(a), "?"), []).append(hf)
cat_rows = sorted([(KO.get(c, c), len(v), float(np.median(v))) for c, v in by_cat.items() if len(v) >= 30], key=lambda r: r[2])

# test 노출의 정답 기사: 노출 직전 클릭이 언제 있었나 (72시간 안)
BIG = 10 ** 10
key = np.sort(A * BIG + T)
te = b.filter(pl.col("impression_time") >= np.datetime64("2023-05-28T21:21:57")).explode("article_ids_clicked").drop_nulls("article_ids_clicked")
PA, PT = te["article_ids_clicked"].to_numpy().astype(np.int64), ts(te["impression_time"])
def cntw(lo_h, hi_h):
    x = PA * BIG + PT
    return np.searchsorted(key, x - int(lo_h * 3600), "left") - np.searchsorted(key, x - int(hi_h * 3600), "left")
b_edges = [i / 4 for i in range(25)] + list(range(7, 25))
bc = np.array([cntw(lo, hi).sum() for lo, hi in zip(b_edges[:-1], b_edges[1:])], dtype=float)
all72 = cntw(0, 72).sum()
bdens = bc / all72 / np.diff(b_edges)
cum_before = {h: float(cntw(0, h).sum() / all72) for h in (1, 3, 6, 12, 24, 48)}
i_ = np.searchsorted(key, PA * BIG + PT, "left") - 1
ic = np.maximum(i_, 0)
ok = (i_ >= 0) & (key[ic] // BIG == PA)
last_min = np.where(ok, (PT - key[ic] % BIG) / 60, np.nan)
hrs = te.with_columns(pl.col("impression_time").dt.hour().alias("h")).group_by("h").len().sort("h")
by_hour = dict(zip(hrs["h"].to_list(), hrs["len"].to_list()))


# ---------------------------------------------------------------- 그림 (SVG)
def bars(xs_edges, ys, w=760, h=260, ymax=None, xticks=(), ylabel="", fmt=lambda v: f"{v:.1%}", band=None, tip=None):
    L_, R_, T_, B_ = 56, 16, 18, 34
    pw, ph = w - L_ - R_, h - T_ - B_
    x0, x1 = xs_edges[0], xs_edges[-1]
    ymax = ymax or max(ys) * 1.12
    X = lambda v: L_ + (v - x0) / (x1 - x0) * pw
    Y = lambda v: T_ + ph - v / ymax * ph
    g = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img">']
    if band:
        for lo, hi, lab in band:
            g.append(f'<rect x="{X(lo):.1f}" y="{T_}" width="{X(hi) - X(lo):.1f}" height="{ph}" fill="#f3f4f7"/>')
            g.append(f'<text x="{(X(lo) + X(hi)) / 2:.1f}" y="{T_ + 14}" class="lbl sm m" text-anchor="middle">{lab}</text>')
    for k in range(5):
        v = ymax * k / 4
        g.append(f'<line x1="{L_}" y1="{Y(v):.1f}" x2="{w - R_}" y2="{Y(v):.1f}" stroke="{GRID}"/>')
        g.append(f'<text x="{L_ - 6}" y="{Y(v) + 4:.1f}" class="lbl sm m" text-anchor="end">{fmt(v)}</text>')
    for i, v in enumerate(ys):
        a_, b_ = X(xs_edges[i]) + 1, X(xs_edges[i + 1]) - 1
        t = tip(i) if tip else fmt(v)
        g.append(f'<rect x="{a_:.1f}" y="{Y(v):.1f}" width="{max(b_ - a_, 1):.1f}" height="{Y(0) - Y(v):.1f}" rx="2" fill="{BLUE}"><title>{html.escape(t)}</title></rect>')
    g.append(f'<line x1="{L_}" y1="{Y(0):.1f}" x2="{w - R_}" y2="{Y(0):.1f}" stroke="{MUTED}"/>')
    for j, (v, lab) in enumerate(xticks):
        anc = "end" if j == len(xticks) - 1 and X(v) > w - R_ - 30 else "middle"
        g.append(f'<text x="{X(v):.1f}" y="{h - 12}" class="lbl sm m" text-anchor="{anc}">{lab}</text>')
    if ylabel:
        g.append(f'<text x="{L_}" y="{T_ - 6}" class="lbl sm m">{ylabel}</text>')
    g.append("</svg>")
    return "\n".join(g)


def hm(hh):
    return f"{int(hh * 60)}분" if hh < 1 else f"{hh:g}시간"


fig_days = bars(list(range(len(days) + 1)), [n for _, n in days], h=250, ymax=48000, ylabel="하루 노출 수",
                fmt=lambda v: f"{v / 1000:.0f}천",
                xticks=[(i + 0.5, d[5:].replace("-", "/")) for i, (d, _) in enumerate(days) if i % 2 == 0],
                band=[(0, 7, "train (7일)"), (7, 10.89, "validation 절반"), (10.89, len(days), "test (3.4일)")],
                tip=lambda i: f"{days[i][0]}: 노출 {days[i][1]:,}개")
fig_pub = bars(edges, dens, ymax=0.24, ylabel="시간당 클릭 비율 (발행 후)", xticks=[(v, f"{v:g}시간") for v in (0, 3, 6, 12, 18, 24)],
               tip=lambda i: f"발행 후 {hm(edges[i])}~{hm(edges[i + 1])}: 시간당 {dens[i]:.1%}")
fig_before = bars(b_edges, bdens, ymax=0.32, ylabel="시간당 클릭 비율 (노출 전)", xticks=[(v, f"{v:g}시간 전") for v in (0, 3, 6, 12, 18, 24)],
                  tip=lambda i: f"노출 {hm(b_edges[i])}~{hm(b_edges[i + 1])} 전: 시간당 {bdens[i]:.1%}")
hh = list(range(24))
fig_hour = bars(list(range(25)), [by_hour.get(i, 0) for i in hh], h=230, ymax=10000, ylabel="test 노출 수 (시각별)",
                fmt=lambda v: f"{v / 1000:g}천", xticks=[(i + 0.5, f"{i}시") for i in range(0, 24, 3)],
                tip=lambda i: f"{i}시: {by_hour.get(i, 0):,}개")


def qrow(name, vals, unit, f=lambda v: f"{v:.1f}"):
    return f"<tr><td>{name}</td>" + "".join(f'<td class="n">{f(v)}{unit}</td>' for v in vals) + "</tr>"


# ---------------------------------------------------------------- 문서
head = (REPO / "reports/NRMS_동작방식_적용계획.html").read_text(encoding="utf-8")
head = head[: head.index("<main>")].replace("<title>NRMS 동작 방식</title>", "<title>test 기사 클릭 수명</title>")
ld = np.array([x[4] for x in life])   # 발행 후 첫 클릭 (시간)
body = f"""<main>
<h1>test 기간 기사들의 클릭 수명</h1>
<p class="sub">EB-NeRD small 원본 behaviors · 클릭 시각 = 그 노출의 시각 · test = 2023-05-28 21:21:57 이후 노출 · 2026-11-26</p>

<div class="box1"><b>한 줄 요약</b> — EB-NeRD 기사는 <b>아주 짧게 산다</b>. 발행 후 3시간 안에 평생 클릭의 절반, 12시간 안에 91%를 받는다.
test 정답은 대부분 <b>발행된 지 몇 시간 안 된, 지금 막 클릭이 몰리는 기사</b>이다(정답 기사 나이 중앙값 {np.median(age_click[~np.isnan(age_click)]):.1f}시간, 마지막 클릭이 노출 {np.nanmedian(last_min):.1f}분 전).
그래서 인기도 창은 기사 수명에 맞춘 <b>12시간</b>이 가장 잘 맞았다(3시간은 너무 짧고, 24시간은 이미 식은 기사까지 센다).</div>

<h2>1. 기간과 노출량</h2>
<div class="fig">{fig_days}
<p class="cap"><b>그림 1.</b> 날짜별 노출 수. train 7일(05-18 07:00 ~ 05-25 07:00), validation 절반 약 3.6일(~ 05-28 21:21), test 약 3.4일(~ 06-01 07:00). 06-01은 07시까지라 적다. 막대에 마우스를 올리면 값이 보인다.</p></div>
<div class="tbl"><table>
<tr><th>test 기간</th><th class="n">값</th></tr>
<tr><td>노출에서 나온 클릭 수</td><td class="n">{test_clicks:,}</td></tr>
<tr><td>클릭된 기사 수</td><td class="n">{test_articles:,}</td></tr>
<tr><td>train 기간에 한 번이라도 클릭된 기사에 대한 클릭</td><td class="n">{share_seen_train:.1%}</td></tr>
</table></div>
<p>test 클릭의 {1 - share_seen_train:.0%}는 train 기간에 한 번도 클릭되지 않은 기사에 대한 것이다. 우리 보고서의 "test 정답 98%가 새 기사"와 같은 이야기이다.</p>
<div class="tbl"><table>
<tr><th>test 클릭이 향한 기사는 언제 발행됐나 (클릭 기준)</th><th class="n">비율</th></tr>
""" + "".join(f'<tr><td>{k}</td><td class="n">{v:.1%}</td></tr>' for k, v in pub_period.items()) + f"""
</table></div>

<h2>2. 기사는 발행 후 얼마 동안 클릭되나</h2>
<p>발행 후 72시간이 데이터 안에 모두 들어 있고 클릭이 20개 이상인 기사 {len(life):,}개로 계산했다(데이터 끝에 발행돼 수명이 잘린 기사는 뺐다).</p>
<div class="fig">{fig_pub}
<p class="cap"><b>그림 2.</b> 발행 후 시간대별 클릭 비율(0~6시간은 15분 단위, 그 뒤 1시간 단위, 높이 = 시간당 비율이라 폭이 달라도 비교 가능). 발행 후 1~2시간에 가장 많고, 그 뒤 빠르게 줄어든다.</p></div>
<div class="tbl"><table>
<tr><th>발행 후</th>""" + "".join(f'<th class="n">{h}시간</th>' for h in cum_pub) + """</tr>
<tr><td>평생(72시간) 클릭 중 누적</td>""" + "".join(f'<td class="n">{v:.0%}</td>' for v in cum_pub.values()) + f"""</tr>
</table></div>
<div class="tbl"><table>
<tr><th>기사 하나하나 (분위수)</th><th class="n">10%</th><th class="n">25%</th><th class="n">중앙값</th><th class="n">75%</th><th class="n">90%</th></tr>
{qrow("첫 클릭까지 (발행 후)", [v * 60 for v in q(ld)], "분")}
{qrow("클릭 절반을 받을 때까지 (반감기)", q(half), "시간")}
{qrow("클릭 90%를 받을 때까지", q(L[:, 2]), "시간")}
{qrow("평생 클릭 수", q(L[:, 0]), "", lambda v: f"{v:,.0f}")}
</table></div>
<h3>카테고리별 반감기</h3>
<div class="tbl"><table>
<tr><th>카테고리</th><th class="n">기사 수</th><th class="n">반감기 중앙값</th></tr>
""" + "".join(f'<tr><td>{c}</td><td class="n">{n}</td><td class="n">{v:.1f}시간</td></tr>' for c, n, v in cat_rows) + f"""
</table></div>
<p>스포츠 기사가 가장 빨리 식고(경기 결과), 음악 · 국내 시사는 조금 더 오래 간다. 그래도 모두 몇 시간 단위이다.</p>

<h2>3. test 정답 기사: 노출 직전 클릭이 언제 있었나</h2>
<div class="fig">{fig_before}
<p class="cap"><b>그림 3.</b> test 정답 기사가 노출 전 72시간 동안 받은 클릭이 노출 몇 시간 전에 있었나(시간당 비율). 노출 직전이 가장 많다.</p></div>
<div class="tbl"><table>
<tr><th>노출 전</th>""" + "".join(f'<th class="n">{h}시간</th>' for h in cum_before) + """</tr>
<tr><td>72시간 클릭 중 누적</td>""" + "".join(f'<td class="n">{v:.0%}</td>' for v in cum_before.values()) + f"""</tr>
</table></div>
<div class="tbl"><table>
<tr><th>test 정답 기사 (분위수)</th><th class="n">10%</th><th class="n">25%</th><th class="n">중앙값</th><th class="n">75%</th><th class="n">90%</th></tr>
{qrow("노출 시점의 기사 나이", q(age_click), "시간")}
{qrow("마지막 클릭이 노출 몇 분 전", q(last_min), "분")}
</table></div>

<h2>4. test 노출은 하루 중 언제</h2>
<div class="fig">{fig_hour}
<p class="cap"><b>그림 4.</b> test 노출의 시각별 분포(데이터에 적힌 시각 기준). 이른 아침(4~6시)에 가장 많고 밤(22~2시)에 적다. 인기도를 "직전 몇 시간"으로 세면 밤 사이에는 클릭 수 자체가 적어진다.</p></div>

<h2>5. 인기도 창에 주는 뜻</h2>
<div class="tbl"><table>
<tr><th>창</th><th>기사 수명에서 보면</th><th>결과 (AWS 실제 test, Transformer + 인기 + 나이, seed 평균)</th></tr>
<tr><td>3시간</td><td>평생 클릭의 절반만 담고, 클릭 수가 작아 동점 · 0이 많다</td><td>H 43.12% · R 43.21%</td></tr>
<tr class="hl"><td>12시간</td><td>평생 클릭의 91%를 담고, 하루 지난 기사는 거의 안 센다</td><td><b>H 44.78% · R 44.86%</b> (validation 최고)</td></tr>
<tr><td>24시간 (예전)</td><td>이미 식은 어제 기사까지 점수를 준다</td><td>H 44.24% · R 44.23%</td></tr>
</table></div>
<p class="sub">1~4절은 이 저장소의 원본 데이터로 계산했다(위 스크립트). 5절 숫자는 AWS의 popularity_window.py 결과이다.</p>
</main></body></html>
"""
OUT.write_text(head + body, encoding="utf-8")
print("저장:", OUT)
print({k: round(v, 3) for k, v in pub_period.items()}, cum_pub, cum_before)
