"""test 5지선다에서 정답 vs 오답의 인기 · 나이 분포 (MIND 팀 요청 비교용).

    python experiments/v2_yeomin_rerun/popularity_stats.py                 # H · R · V1 중 있는 것
    python experiments/v2_yeomin_rerun/popularity_stats.py --runs h

입력: <V2>/features/<run>/test_sequences_1pos4neg.parquet  (add_candidate_features.py 결과, 읽기만)
      candidate_logpop = ln(1 + 노출 직전 24시간 클릭 수), candidate_logage = ln(1 + 발행 후 시간)
출력 (실험마다)
  - 정답 / 오답의 24시간 클릭 수 · logpop · 나이(시간) 평균 · 중앙값
  - 정답이 "가장 인기 많은 후보"인 비율: 단독 1등 / 공동 1등 포함 / 동점 기대값(= 인기만 Top-1)
  - 정답이 "가장 새 후보"인 비율: 같은 세 가지 (동점 기대값 = 나이만 Top-1)
  - 후보 5개 클릭이 모두 0인 행 비율
결과: 화면 + <V2>/analysis/popularity_stats.json (이미 있으면 _2, _3 …)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]


def rank_share(score: np.ndarray, y: np.ndarray) -> dict:
    """score 가 가장 큰 후보가 정답인 비율. 동점 처리 세 가지."""
    pos = (score * y).sum(1, keepdims=True)
    neg = y == 0
    greater = ((score > pos) & neg).sum(1)
    ties = ((score == pos) & neg).sum(1)
    return {"단독 1등": float(((greater == 0) & (ties == 0)).mean()),
            "공동 1등 포함": float((greater == 0).mean()),
            "동점 기대값": float(np.where(greater == 0, 1.0 / (1 + ties), 0.0).mean())}


def stats(v: np.ndarray) -> dict:
    return {"평균": float(v.mean()), "중앙값": float(np.median(v))}


def analyze(path: Path) -> dict:
    t = pq.read_table(path, columns=["candidate_labels", "candidate_logpop", "candidate_logage"])
    y = np.array(t.column("candidate_labels").to_pylist(), dtype=np.int64)
    lp = np.array(t.column("candidate_logpop").to_pylist(), dtype=np.float64)
    la = np.array(t.column("candidate_logage").to_pylist(), dtype=np.float64)
    if y.shape[1] != 5 or not (y.sum(1) == 1).all():
        raise ValueError(f"{path}: 행마다 후보 5개 · 정답 1개여야 합니다")
    clicks, hours = np.expm1(lp), np.expm1(la)
    pos, neg = y == 1, y == 0
    r = {"행": int(len(y))}
    for name, arr in [("24시간 클릭 수", clicks), ("logpop", lp), ("나이(시간)", hours), ("logage", la)]:
        r[name] = {"정답": stats(arr[pos]), "오답": stats(arr[neg])}
    r["정답이 가장 인기 (클릭 최다)"] = rank_share(lp, y)
    r["정답이 가장 새 기사 (나이 최소)"] = rank_share(-la, y)
    r["후보 5개 클릭이 모두 0인 행"] = float((clicks.max(1) == 0).mean())
    r["정답 클릭 0"] = float((clicks[pos] == 0).mean())
    r["오답 클릭 0"] = float((clicks[neg] == 0).mean())
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre", default=str(REPO / "data/output/fixedcluster_v1"))
    ap.add_argument("--v2", default=None, help="기본 <pre>/v2")
    ap.add_argument("--runs", nargs="+", default=["h", "r", "v1"])
    a = ap.parse_args()
    v2 = Path(a.v2).expanduser() if a.v2 else Path(a.pre).expanduser() / "v2"
    res = {}
    for run in a.runs:
        f = v2 / "features" / run / "test_sequences_1pos4neg.parquet"
        if not f.exists():
            print(f"[{run}] 없음: {f}")
            continue
        r = analyze(f)
        res[run.upper()] = r
        print(f"\n===== {run.upper()} test ({r['행']:,}행) =====")
        for name in ["24시간 클릭 수", "logpop", "나이(시간)", "logage"]:
            p, n = r[name]["정답"], r[name]["오답"]
            print(f"  {name:<10} 정답 평균 {p['평균']:>9.2f} · 중앙값 {p['중앙값']:>8.2f}  |  오답 평균 {n['평균']:>9.2f} · 중앙값 {n['중앙값']:>8.2f}")
        for name in ["정답이 가장 인기 (클릭 최다)", "정답이 가장 새 기사 (나이 최소)"]:
            print(f"  {name}: " + " · ".join(f"{k} {v:.2%}" for k, v in r[name].items()))
        print(f"  후보 5개 클릭이 모두 0인 행 {r['후보 5개 클릭이 모두 0인 행']:.2%} · "
              f"클릭 0 비율 정답 {r['정답 클릭 0']:.2%} / 오답 {r['오답 클릭 0']:.2%}")
    if not res:
        sys.exit("특징 파일을 찾지 못했습니다 (add_candidate_features.py 먼저)")
    out = v2 / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    f, k = out / "popularity_stats.json", 2
    while f.exists():
        f, k = out / f"popularity_stats_{k}.json", k + 1
    f.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {f}")


if __name__ == "__main__":
    main()
