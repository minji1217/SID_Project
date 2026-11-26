"""인기도 시간 창 분석: 기사 인기도를 직전 몇 시간 클릭으로 셀 때 가장 잘 맞나 (재학습 없음).

    python experiments/v2_yeomin_rerun/popularity_window.py                       # H · R, seed 42 · 123 · 2026
    python experiments/v2_yeomin_rerun/popularity_window.py --runs h --seeds 42

지금 인기도 = ln(1 + 노출 직전 24시간 클릭 수). 이 24시간을 0.5 ~ 72시간으로 바꿔 가며 비교한다.
정의는 surge_fusion.py · popularity_fusion.py 와 같다.
  클릭 기록 = train + validation behaviors 의 모든 클릭 (클릭 시각 = 그 노출의 impression_time), 창 = [노출 시각 − w, 노출 시각)
  나이 = ln(1 + max(0, 노출 시각 − 발행 시각) 시간)
  가중치는 창마다 validation 앞 절반 Top-1 로 고르고 (같으면 작은 쪽), test 에 한 번만. 지표 = popularity_fusion.metrics (동점 기대값).

출력 (실험마다)
  - 기간: validation 절반 · test 의 첫 · 마지막 노출 시각
  - 정답 · 오답 기사 나이 분위수, 정답 기사 클릭이 노출 몇 시간 전에 있었나 (48시간 안, 누적 비율)
  - 창별: 정답 · 오답 클릭 0 비율, 인기만 Top-1, 인기 + 나이 (Transformer 없음), Transformer + 인기 + 나이 (seed별 · 평균)
입력 (읽기만): 1pos4neg split (validation 절반, test), Transformer 예측 <V2>/transformer/<run>_seed<seed>/{val,test}_candidate_scores.parquet, 원본 behaviors · articles
결과: 화면 + <V2>/analysis/popularity_window.json (이미 있으면 _2, _3 …)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_test import REPO  # noqa: E402
from popularity_fusion import LAMBDAS, MUS, metrics  # noqa: E402
from surge_fusion import load_click_log, load_split, paths, pick, to_s, top1_fast  # noqa: E402

WINDOWS = [0.5, 1, 2, 3, 6, 12, 24, 48, 72]
BIG = 10 ** 10   # article_id * BIG + 초 (초 < 1e10)


def counts(key: np.ndarray, art: np.ndarray, t: np.ndarray, w_hours: float) -> np.ndarray:
    """[t − w, t) 안 클릭 수. key = 정렬된 article_id * BIG + 클릭 초."""
    hi = art * BIG + t
    return np.searchsorted(key, hi, "left") - np.searchsorted(key, hi - int(w_hours * 3600), "left")


def best(score_fn, grid, Yv):
    g = {k: top1_fast(score_fn(*k), Yv) for k in grid}
    k = pick(g)
    return k, g[k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["h", "r"])
    ap.add_argument("--seeds", nargs="+", default=["42", "123", "2026"])
    ap.add_argument("--pre", default=str(REPO / "data/output/fixedcluster_v1"))
    ap.add_argument("--v2", default=None, help="기본 <pre>/v2")
    ap.add_argument("--raw", default=str(REPO / "data/raw"))
    ap.add_argument("--exp-prefix", default="v2_")
    ap.add_argument("--windows", nargs="+", type=float, default=WINDOWS)
    a = ap.parse_args()
    pre = Path(a.pre).expanduser()
    v2 = Path(a.v2).expanduser() if a.v2 else pre / "v2"
    raw = Path(a.raw).expanduser()

    clicks = load_click_log(raw)
    key = np.sort(clicks.a.to_numpy().astype(np.int64) * BIG + clicks.t.to_numpy().astype(np.int64))
    arts = pd.read_parquet(raw / "articles.parquet", columns=["article_id", "published_time"])
    pub = dict(zip(arts.article_id.astype(np.int64), to_s(arts.published_time)))
    grid_pa = [(l, m) for l in LAMBDAS for m in MUS if l != 0]

    res = {}
    for run in a.runs:
        D = {}   # seed → {split: p}
        for seed in a.seeds:
            P = paths(run, pre, v2, a.exp_prefix, seed)
            miss = [str(x) for x in [*P["seq"].values(), *P["pred"].values()] if not x.exists()]
            if miss:
                print(f"[{run} seed {seed}] 입력 없음 → 건너뜀: {miss}")
                continue
            D[seed] = {s: load_split(P["pred"][s], P["seq"][s])[0] for s in ("val", "test")}
        if not D:
            continue
        p0 = next(iter(D.values()))
        F = {}
        for s in ("val", "test"):
            p = p0[s]
            art, t = p.article_id.to_numpy().astype(np.int64), p.t.to_numpy().astype(np.int64)
            pt = np.array([pub.get(x, np.nan) for x in art], dtype="float64")
            age_h = np.nan_to_num(np.clip((t - pt) / 3600.0, 0, None), nan=0.0)
            F[s] = {"Y": p.label.to_numpy(), "la": np.log1p(age_h), "age_h": age_h, "art": art, "t": t,
                    "lp": {w: np.log1p(counts(key, art, t, w)) for w in a.windows}}
            for seed in D:   # 모든 seed 의 행 · 후보 순서가 같은지
                q = D[seed][s]
                if not (np.array_equal(q.article_id.to_numpy(), art) and np.array_equal(q.label.to_numpy(), F[s]["Y"])):
                    raise ValueError(f"[{run}] seed {seed} {s} 의 후보 순서가 다릅니다")
        Fv, Ft = F["val"], F["test"]
        pos_t = Ft["Y"] == 1
        r = {"기간": {s: [str(pd.to_datetime(F[s]["t"].min(), unit="s")), str(pd.to_datetime(F[s]["t"].max(), unit="s"))] for s in F},
             "행": {s: int(len(F[s]["Y"]) // 5) for s in F},
             "기사 나이 분위수 (시간, 10/25/50/75/90%)": {
                 "정답": np.percentile(Ft["age_h"][pos_t], [10, 25, 50, 75, 90]).round(1).tolist(),
                 "오답": np.percentile(Ft["age_h"][~pos_t], [10, 25, 50, 75, 90]).round(1).tolist()}}
        # 정답 기사 클릭이 노출 몇 시간 전에 있었나 (48시간 안)
        pa, ptt = Ft["art"][pos_t], Ft["t"][pos_t]
        by_h = np.array([(counts(key, pa, ptt, h + 1) - counts(key, pa, ptt, h)).sum() for h in range(48)], dtype=float)
        cum = np.cumsum(by_h) / by_h.sum()
        r["정답 클릭 누적 비율 (노출 전 1·3·6·12·24시간)"] = {h: float(cum[h - 1]) for h in (1, 3, 6, 12, 24)}

        rows = []
        for w in a.windows:
            lpv, lpt = Fv["lp"][w], Ft["lp"][w]
            row = {"창(시간)": w,
                   "정답 클릭 0": float((np.expm1(lpt[pos_t]) == 0).mean()),
                   "오답 클릭 0": float((np.expm1(lpt[~pos_t]) == 0).mean()),
                   "인기만 val": top1_fast(lpv, Fv["Y"]), "인기만 test": top1_fast(lpt, Ft["Y"])}
            (l, m), v = best(lambda l, m: l * lpv + m * Fv["la"], grid_pa, Fv["Y"])
            row["인기+나이 (λ,μ)"] = [l, m]
            row["인기+나이 val"] = v
            row["인기+나이 test"] = metrics(pd.DataFrame({"label": Ft["Y"]}), l * lpt + m * Ft["la"])
            seeds = {}
            for seed in D:
                sv, st = D[seed]["val"].candidate_score.to_numpy(), D[seed]["test"].candidate_score.to_numpy()
                (l2, m2), v2_ = best(lambda l, m: sv + l * lpv + m * Fv["la"], [(l, m) for l in LAMBDAS for m in MUS], Fv["Y"])
                seeds[seed] = {"λ,μ": [l2, m2], "val": v2_,
                               "test": metrics(pd.DataFrame({"label": Ft["Y"]}), st + l2 * lpt + m2 * Ft["la"])}
            row["Transformer+인기+나이 seed별"] = seeds
            row["Transformer+인기+나이 val 평균"] = float(np.mean([x["val"] for x in seeds.values()]))
            row["Transformer+인기+나이 test 평균"] = float(np.mean([x["test"]["top1"] for x in seeds.values()]))
            rows.append(row)
        r["창별"] = rows
        res[run.upper()] = r

        print(f"\n===== {run.upper()} (seed {', '.join(D)}) =====")
        print("기간: " + " · ".join(f"{s} {v[0]} ~ {v[1]} ({r['행'][s]:,}행)" for s, v in r["기간"].items()))
        print("기사 나이 (시간, 10/25/50/75/90%): " + " | ".join(f"{k} {v}" for k, v in r["기사 나이 분위수 (시간, 10/25/50/75/90%)"].items()))
        print("정답 기사 클릭 누적 (노출 전): " + " · ".join(f"{h}시간 {v:.0%}" for h, v in r["정답 클릭 누적 비율 (노출 전 1·3·6·12·24시간)"].items()))
        print(f"{'창':>5} | {'정답0':>6} {'오답0':>6} | {'인기만 val':>9} {'test':>7} | {'인기+나이 val':>11} {'test':>7} | {'+Transformer val':>15} {'test':>7}")
        for x in rows:
            print(f"{x['창(시간)']:>4g}h | {x['정답 클릭 0']:>6.2%} {x['오답 클릭 0']:>6.2%} | {x['인기만 val']:>9.2%} {x['인기만 test']:>7.2%} | "
                  f"{x['인기+나이 val']:>11.2%} {x['인기+나이 test']['top1']:>7.2%} | "
                  f"{x['Transformer+인기+나이 val 평균']:>15.2%} {x['Transformer+인기+나이 test 평균']:>7.2%}")
        bw = max(rows, key=lambda x: x["Transformer+인기+나이 val 평균"])
        print(f"validation 으로 고른 창 (Transformer + 인기 + 나이, seed 평균): {bw['창(시간)']:g}시간 → test {bw['Transformer+인기+나이 test 평균']:.2%}")

    if not res:
        sys.exit("입력을 찾지 못했습니다")
    out = v2 / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    f, k = out / "popularity_window.json", 2
    while f.exists():
        f, k = out / f"popularity_window_{k}.json", k + 1
    f.write_text(json.dumps(res, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n저장: {f}")


if __name__ == "__main__":
    main()
