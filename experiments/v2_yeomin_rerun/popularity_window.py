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
다른 모델 (예: NRMS) 의 점수로 보려면 --pred-template 과 --out-name. 아래 표의 'Transformer' 는 그 모델을 뜻한다.
결과: 화면 + <V2>/analysis/popularity_window/ (이미 있으면 popularity_window_2/ …) 안에
  <run>_click_timing.csv  노출 전 0~6시간 15분 단위 · 6~72시간 1시간 단위, 정답 · 오답 기사 클릭 비율과 누적 비율
  <run>_windows.csv       창별 클릭 0 비율, 인기만, 인기+나이, Transformer+인기+나이 (seed별 · 평균) validation · test 지표
  <run>_summary.csv       기간, 날짜별 노출 수, 정답 · 오답 기사 나이, 마지막 클릭이 노출 몇 분 전, 3시간/24시간 클릭 비율
  result.json             위 내용 전체
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


def counts(key: np.ndarray, art: np.ndarray, t: np.ndarray, w_hours: float, from_hours: float = 0.0) -> np.ndarray:
    """[t − w, t − from) 안 클릭 수 (기본 [t − w, t)). key = 정렬된 article_id * BIG + 클릭 초."""
    a = art * BIG + t
    return np.searchsorted(key, a - int(from_hours * 3600), "left") - np.searchsorted(key, a - int(w_hours * 3600), "left")


def last_click_minutes(key: np.ndarray, art: np.ndarray, t: np.ndarray) -> np.ndarray:
    """노출 직전 마지막 클릭이 몇 분 전인가 (없으면 nan)."""
    i = np.searchsorted(key, art * BIG + t, "left") - 1
    ic = np.maximum(i, 0)
    ok = (i >= 0) & (key[ic] // BIG == art)
    return np.where(ok, (t - key[ic] % BIG) / 60.0, np.nan)


TIMING_BINS = [(i / 4, (i + 1) / 4) for i in range(24)] + [(h, h + 1) for h in range(6, 72)]   # 시간


def windows_df(rows: list) -> pd.DataFrame:
    """창별 결과를 한 행씩 펼친 표."""
    out = []
    for x in rows:
        d = {"창(시간)": x["창(시간)"], "정답 클릭 0": x["정답 클릭 0"], "오답 클릭 0": x["오답 클릭 0"],
             "인기만 val Top-1": x["인기만 val"], "인기만 test Top-1": x["인기만 test"],
             "인기+나이 λ": x["인기+나이 (λ,μ)"][0], "인기+나이 μ": x["인기+나이 (λ,μ)"][1], "인기+나이 val Top-1": x["인기+나이 val"]}
        d.update({f"인기+나이 test {m}": v for m, v in x["인기+나이 test"].items()})
        for seed, y in x["Transformer+인기+나이 seed별"].items():
            d[f"T+인기+나이 seed{seed} λ"], d[f"T+인기+나이 seed{seed} μ"] = y["λ,μ"]
            d[f"T+인기+나이 seed{seed} val Top-1"] = y["val"]
            d.update({f"T+인기+나이 seed{seed} test {m}": v for m, v in y["test"].items()})
        d["T+인기+나이 val Top-1 평균"] = x["Transformer+인기+나이 val 평균"]
        d["T+인기+나이 test Top-1 평균"] = x["Transformer+인기+나이 test 평균"]
        out.append(d)
    return pd.DataFrame(out)


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
    ap.add_argument("--pred-template", default=None,
                    help="모델 점수 파일 위치 (기본 Transformer). 예: '{v2}/nrms/{run}/final/lr0.0001_do0.2_seed{seed}/{split}_candidate_scores.parquet'"
                         " ({split} = val | test). NRMS + 인기 + 나이 를 볼 때 씀")
    ap.add_argument("--out-name", default="popularity_window", help="결과 폴더 이름 (<V2>/analysis/<이름>/)")
    a = ap.parse_args()
    pre = Path(a.pre).expanduser()
    v2 = Path(a.v2).expanduser() if a.v2 else pre / "v2"
    raw = Path(a.raw).expanduser()

    clicks = load_click_log(raw)
    key = np.sort(clicks.a.to_numpy().astype(np.int64) * BIG + clicks.t.to_numpy().astype(np.int64))
    arts = pd.read_parquet(raw / "articles.parquet", columns=["article_id", "published_time"])
    pub = dict(zip(arts.article_id.astype(np.int64), to_s(arts.published_time)))
    grid_pa = [(l, m) for l in LAMBDAS for m in MUS if l != 0]

    res, outputs = {}, {}
    for run in a.runs:
        D = {}   # seed → {split: p}
        for seed in a.seeds:
            P = paths(run, pre, v2, a.exp_prefix, seed)
            if a.pred_template:
                P["pred"] = {s: Path(a.pred_template.format(v2=v2, run=run, seed=seed, split=s)).expanduser() for s in ("val", "test")}
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
        # 정답 · 오답 기사 클릭이 노출 몇 시간 전에 있었나 (72시간 안, 15분 / 1시간 단위)
        timing = []
        for name, m in [("정답", pos_t), ("오답", ~pos_t)]:
            aa, tt = Ft["art"][m], Ft["t"][m]
            tot = np.array([counts(key, aa, tt, hi, lo).sum() for lo, hi in TIMING_BINS], dtype=float)
            timing.append((name, tot / max(tot.sum(), 1)))
        tdf = pd.DataFrame({"노출 전 시작(분)": [lo * 60 for lo, _ in TIMING_BINS], "노출 전 끝(분)": [hi * 60 for _, hi in TIMING_BINS]})
        for name, sh in timing:
            tdf[f"{name} 클릭 비율"] = sh
            tdf[f"{name} 누적 비율"] = np.cumsum(sh)
        cum = tdf["정답 누적 비율"].to_numpy()
        ends = tdf["노출 전 끝(분)"].to_numpy()
        r["정답 클릭 누적 비율 (노출 전 1·3·6·12·24시간)"] = {h: float(cum[ends == h * 60][0]) for h in (1, 3, 6, 12, 24)}
        pa, ptt = Ft["art"][pos_t], Ft["t"][pos_t]
        lc = last_click_minutes(key, pa, ptt)
        c3, c24 = counts(key, pa, ptt, 3), counts(key, pa, ptt, 24)
        ratio = np.where(c24 > 0, c3 / np.maximum(c24, 1), np.nan)
        days = pd.Series(pd.to_datetime(Ft["t"][pos_t], unit="s").date).value_counts().sort_index()
        summ = [("기간", s, f"{v[0]} ~ {v[1]}") for s, v in r["기간"].items()]
        summ += [("행", s, v) for s, v in r["행"].items()]
        summ += [("test 날짜별 노출", str(d), int(n)) for d, n in days.items()]
        qs = [10, 25, 50, 75, 90]
        for k, v in r["기사 나이 분위수 (시간, 10/25/50/75/90%)"].items():
            summ += [(f"{k} 기사 나이(시간)", f"{q}%", x) for q, x in zip(qs, v)]
        summ += [("정답 기사 마지막 클릭(분 전)", f"{q}%", float(x)) for q, x in zip(qs, np.nanpercentile(lc, qs).round(2))]
        summ += [("정답 기사 3시간/24시간 클릭 비율", f"{q}%", float(x)) for q, x in zip(qs, np.nanpercentile(ratio, qs).round(3))]
        summ += [("정답 클릭 없음(노출 전 전체)", "비율", float(np.isnan(lc).mean()))]

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
        outputs[run] = {"timing": tdf, "summary": pd.DataFrame(summ, columns=["항목", "구분", "값"]), "windows": windows_df(rows)}

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
    out, k = v2 / "analysis" / a.out_name, 2
    while out.exists():
        out, k = v2 / "analysis" / f"{a.out_name}_{k}", k + 1
    out.mkdir(parents=True)
    for run, o in outputs.items():
        for name, df in o.items():
            df.to_csv(out / f"{run}_{'click_timing' if name == 'timing' else name}.csv", index=False, encoding="utf-8-sig")
    (out / "result.json").write_text(json.dumps(res, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n저장: {out}/ ({', '.join(sorted(p.name for p in out.iterdir()))})")


if __name__ == "__main__":
    main()
