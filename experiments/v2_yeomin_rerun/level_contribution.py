"""SID 자리별 기여: test Top-1을 c1만 / c1+c2 / c1+c2+c3 점수로 다시 매기고, 후보끼리 SID가 얼마나 겹치는지 본다.

    python experiments/v2_yeomin_rerun/level_contribution.py              # V1 · H · R × seed 42 · 123 · 2026 중 있는 것

H와 R은 c2 코드 사용 수가 크게 다른데(128 vs 59) Transformer 성능은 비슷하다. 그 이유를 보려는 분석이다.
  - 자리별 Top-1: 점수 = log P(c1) [+ log P(c2|c1) [+ log P(c3|c1,c2)]]. 여민 predict_sid.py가 저장한 자리별 log 확률을 그대로 쓴다
    (c1만 점수는 c2·c3를 안 본 것이지, c1만으로 따로 학습한 모델은 아니다)
  - 후보 겹침: 한 노출의 정답과 같은 c1 / (c1,c2) / (c1,c2,c3)를 가진 오답이 있는 행의 비율
    → 겹치면 그 자리로는 정답과 오답을 가를 수 없다
  - Top-1은 동점 기대값 (정답이 동점 k개 안이면 1/k), popularity_fusion.py와 같은 정의
입력: <V2>/transformer/<run>_seed<S>/test_candidate_scores.parquet (읽기만)
결과: 화면 + <V2>/analysis/level_contribution.csv (이미 있으면 _2, _3 …)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]


def top1(score: np.ndarray, y: np.ndarray) -> float:
    pos = (score * y).sum(1, keepdims=True)
    neg = y == 0
    greater = ((score > pos) & neg).sum(1)
    ties = ((score == pos) & neg).sum(1)
    return float(np.where(greater == 0, 1.0 / (1 + ties), 0.0).mean())


def analyze(path: Path) -> dict:
    p = pd.read_parquet(path, columns=["sample_index", "label", "c1", "c2", "c3",
                                       "c1_log_prob", "c2_log_prob", "c3_log_prob", "candidate_score"])
    p = p.sort_values("sample_index", kind="stable")
    k = p.groupby("sample_index").size()
    if not (k == 5).all():
        raise ValueError(f"{path}: 행마다 후보 5개가 아닙니다")
    y = p.label.to_numpy().astype(int).reshape(-1, 5)
    l1, l2, l3 = (p[f"c{i}_log_prob"].to_numpy().reshape(-1, 5) for i in (1, 2, 3))
    r = {"행": len(y),
         "Top-1 c1만": top1(l1, y), "Top-1 c1+c2": top1(l1 + l2, y), "Top-1 c1+c2+c3": top1(l1 + l2 + l3, y),
         "Top-1 저장 점수": top1(p.candidate_score.to_numpy().reshape(-1, 5), y)}
    codes = {"c1": p.c1.to_numpy().reshape(-1, 5),
             "c1c2": (p.c1.astype(np.int64) * 1000 + p.c2).to_numpy().reshape(-1, 5),
             "c1c2c3": ((p.c1.astype(np.int64) * 1000 + p.c2) * 1000 + p.c3).to_numpy().reshape(-1, 5)}
    for name, c in codes.items():
        pos = (c * y).sum(1, keepdims=True)
        r[f"정답과 같은 {name} 오답 있음"] = float((((c == pos) & (y == 0)).any(1)).mean())
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre", default=str(REPO / "data/output/fixedcluster_v1"))
    ap.add_argument("--v2", default=None, help="기본 <pre>/v2")
    ap.add_argument("--runs", nargs="+", default=["v1", "h", "r"])
    ap.add_argument("--seeds", nargs="+", default=["42", "123", "2026"])
    ap.add_argument("--suffix", default="", help="폴더 이름 뒤 (테스트용)")
    a = ap.parse_args()
    v2 = Path(a.v2).expanduser() if a.v2 else Path(a.pre).expanduser() / "v2"
    rows = []
    for run in a.runs:
        for seed in a.seeds:
            f = v2 / "transformer" / f"{run}_seed{seed}{a.suffix}" / "test_candidate_scores.parquet"
            if not f.exists():
                continue
            rows.append({"실험": run.upper(), "seed": seed, **analyze(f)})
            print(f"  {run.upper()} seed {seed} 완료", flush=True)
    if not rows:
        sys.exit("test_candidate_scores.parquet 를 찾지 못했습니다")
    t = pd.DataFrame(rows)
    show = t.copy()
    for c in show.columns[3:]:
        show[c] = (show[c] * 100).map(lambda v: f"{v:.2f}%")
    print("\n== seed별")
    print(show.to_string(index=False))
    m = t.drop(columns=["seed"]).groupby("실험", sort=False).mean(numeric_only=True)
    m = m.drop(columns=["행"])
    print("\n== seed 평균 (%)")
    print((m * 100).round(2).to_string())
    out = v2 / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    f, k = out / "level_contribution.csv", 2
    while f.exists():
        f, k = out / f"level_contribution_{k}.csv", k + 1
    t.to_csv(f, index=False)
    print(f"\n저장: {f}")


if __name__ == "__main__":
    main()
