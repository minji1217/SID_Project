"""인기 결합 (재학습 없음): 최종 점수 = Transformer 점수 + λ · log(1 + 직전 N시간 클릭 수)

    python experiments/v2_yeomin_rerun/popularity_fusion.py --runs v1 h r

- 클릭 수: 노출 시각 '이전' N시간(기본 24h) 동안 전체 사용자가 그 기사를 클릭한 횟수 (train + validation behaviors). 미래 정보 없음.
- λ: validation 앞 절반 예측(val_candidate_scores.parquet, 5_predict_val.sh 로 만듦)에서 Top-1 이 가장 높은 값을 고른다.
  test 에는 그 λ 를 한 번만 적용한다. λ=0 은 모델만, λ=∞ 는 인기만.
- 지표 (행마다 계산해 평균): Top-1, AUC, MRR, nDCG@5. 동점은 무작위로 고르는 것과 같은 기대값으로 센다
  (Top-1: 정답이 동점 k개 안이면 1/k, 순위: 동점 가운데 평균 순위). 인기만 쓸 때 동점이 많아서 이렇게 한다.

결과: 화면 + <V2>/analysis/popularity_fusion_seed<SEED>.json, .csv
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_test import REPO, load_clicks  # noqa: E402

LAMBDAS = [0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1, 1.5, 2, 3, 5, 10]


def add_pop(p: pd.DataFrame, clicks: dict, window_s: int) -> pd.DataFrame:
    t = p.impression_time.to_numpy().astype("datetime64[s]").astype(np.int64)
    pop = np.zeros(len(p), dtype=np.int64)
    for i, (a, ti) in enumerate(zip(p.article_id.to_numpy(), t)):
        arr = clicks.get(a)
        if arr is not None:
            pop[i] = np.searchsorted(arr, ti, side="left") - np.searchsorted(arr, ti - window_s, side="left")
    p["logpop"] = np.log1p(pop)
    return p


def load(path: Path, clicks: dict, window_s: int) -> pd.DataFrame:
    p = pd.read_parquet(path, columns=["sample_index", "impression_time", "article_id", "label", "candidate_score"])
    p["article_id"] = p.article_id.astype(np.int64)
    p["label"] = p.label.astype(int)
    p = add_pop(p, clicks, window_s).sort_values(["sample_index"], kind="stable").reset_index(drop=True)
    sizes = p.groupby("sample_index").size()
    assert (sizes == 5).all(), "행마다 후보 5개여야 합니다"
    return p


def metrics(p: pd.DataFrame, score: np.ndarray) -> dict:
    s = score.reshape(-1, 5)
    y = p.label.to_numpy().reshape(-1, 5)
    pos = (s * y).sum(1, keepdims=True)                      # 정답 점수
    neg = (y == 0)
    greater = ((s > pos) & neg).sum(1)
    ties = ((s == pos) & neg).sum(1)
    top1 = np.where(greater == 0, 1.0 / (1 + ties), 0.0)
    auc = (4 - greater - ties + 0.5 * ties) / 4.0
    # 동점 안에서 정답 순위는 1+greater .. 1+greater+ties 중 균등 → 평균
    ranks = [np.arange(1 + g, 2 + g + t) for g, t in zip(greater, ties)]
    mrr = np.array([np.mean(1.0 / r) for r in ranks])
    ndcg = np.array([np.mean(1.0 / np.log2(r + 1)) for r in ranks])
    return {"top1": float(top1.mean()), "auc": float(auc.mean()), "mrr": float(mrr.mean()), "ndcg@5": float(ndcg.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["v1", "h", "r"])
    ap.add_argument("--seed", default="42")
    ap.add_argument("--pre", default=str(REPO / "data/output/fixedcluster_v1"))
    ap.add_argument("--v2", default=None, help="기본 <pre>/v2")
    ap.add_argument("--raw", default=str(REPO / "data/raw"))
    ap.add_argument("--window-hours", type=float, default=24.0)
    a = ap.parse_args()
    pre = Path(a.pre).expanduser()
    v2 = Path(a.v2).expanduser() if a.v2 else pre / "v2"
    window_s = int(a.window_hours * 3600)

    print("클릭 기록 읽는 중 ...", flush=True)
    clicks = load_clicks(Path(a.raw).expanduser())

    results, rows = [], []
    for run in a.runs:
        d = v2 / "transformer" / f"{run}_seed{a.seed}"
        vf, tf = d / "val_candidate_scores.parquet", d / "test_candidate_scores.parquet"
        if not (vf.exists() and tf.exists()):
            print(f"[{run}] 예측 파일 없음 (val: {vf.exists()}, test: {tf.exists()}) → 5_predict_val.sh 먼저")
            continue
        val, test = load(vf, clicks, window_s), load(tf, clicks, window_s)

        grid = {}
        for lam in LAMBDAS:
            grid[lam] = metrics(val, val.candidate_score.to_numpy() + lam * val.logpop.to_numpy())["top1"]
        best = max(grid, key=lambda k: (grid[k], -k))       # 같으면 작은 λ
        r = {"run": run, "seed": a.seed, "window_hours": a.window_hours, "val_top1_by_lambda": grid, "best_lambda": best,
             "test": {
                 "모델만 (λ=0)": metrics(test, test.candidate_score.to_numpy()),
                 "인기만": metrics(test, test.logpop.to_numpy()),
                 f"결합 (λ={best})": metrics(test, test.candidate_score.to_numpy() + best * test.logpop.to_numpy()),
             }}
        results.append(r)
        print(f"\n===== {run.upper()} (seed {a.seed}) =====")
        print("validation Top-1 by λ: " + " · ".join(f"{k}:{v:.2%}" for k, v in grid.items()))
        print(f"선택 λ = {best} (validation Top-1 {grid[best]:.2%})")
        for name, m in r["test"].items():
            print(f"  test {name:<14} Top-1 {m['top1']:.2%} · AUC {m['auc']:.4f} · MRR {m['mrr']:.4f} · nDCG@5 {m['ndcg@5']:.4f}")
            rows.append({"run": run, "seed": a.seed, "방식": name, **m})

    if results:
        out = v2 / "analysis"
        out.mkdir(parents=True, exist_ok=True)
        (out / f"popularity_fusion_seed{a.seed}.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        pd.DataFrame(rows).to_csv(out / f"popularity_fusion_seed{a.seed}.csv", index=False)
        print(f"\n저장: {out}/popularity_fusion_seed{a.seed}.json, .csv")


if __name__ == "__main__":
    main()
