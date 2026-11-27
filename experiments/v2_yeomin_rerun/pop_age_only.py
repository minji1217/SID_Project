"""SID(Transformer) 없이 인기 + 나이만으로 채점 (재학습 없음). SID가 순수하게 보태는 몫을 재기 위한 기준선.

    python experiments/v2_yeomin_rerun/pop_age_only.py                 # H · R · V1 중 있는 것
    python experiments/v2_yeomin_rerun/pop_age_only.py --runs h

점수 = λ · logpop + μ · logage   (Transformer 점수 s 없음)
  λ, μ 후보와 고르는 규칙은 popularity_fusion.py --with-age 와 같다 (λ 13개 × μ 10개, λ = 0 은 빼서 120개).
  validation 앞 절반 Top-1 최대 (같으면 |λ|, |μ| 작은 쪽) → test 에 한 번. 지표 정의도 같다 (동점 기대값).
  SID 점수가 없으므로 λ와 μ의 비율만 의미가 있다.

입력 (읽기만)
  <V2>/features/<run>/{validation_sequences_1pos4neg_half,test_sequences_1pos4neg}.parquet  (add_candidate_features.py 결과)
  <V2>/analysis/popularity_fusion_seed{42,123,2026}.json  (있으면: SID + 인기 + 나이 와 비교해 SID의 순수한 몫 계산)
결과: 화면 + <V2>/analysis/pop_age_only.json (이미 있으면 _2, _3 …)
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from popularity_fusion import LAMBDAS, MUS, metrics  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
FILES = {"validation": "validation_sequences_1pos4neg_half.parquet", "test": "test_sequences_1pos4neg.parquet"}


def load(path: Path):
    t = pq.read_table(path, columns=["candidate_labels", "candidate_logpop", "candidate_logage"])
    y = np.array(t.column("candidate_labels").to_pylist(), dtype=np.int64)
    lp = np.array(t.column("candidate_logpop").to_pylist(), dtype=np.float64)
    la = np.array(t.column("candidate_logage").to_pylist(), dtype=np.float64)
    if y.shape[1] != 5 or not (y.sum(1) == 1).all():
        raise ValueError(f"{path}: 행마다 후보 5개 · 정답 1개여야 합니다")
    return y, lp, la


def top1(s, y):
    pos = (s * y).sum(1, keepdims=True)
    neg = y == 0
    g = ((s > pos) & neg).sum(1)
    t = ((s == pos) & neg).sum(1)
    return float(np.where(g == 0, 1.0 / (1 + t), 0.0).mean())


def full(s, y):
    return metrics(pd.DataFrame({"label": y.reshape(-1)}), s.reshape(-1))


def fusion_top1(v2: Path, run: str) -> dict:
    """popularity_fusion_seed*.json 의 SID + 인기 + 나이 test 결과 (seed별)."""
    out = {}
    for f in sorted((v2 / "analysis").glob("popularity_fusion_seed*.json")):
        seed = f.stem.replace("popularity_fusion_seed", "")
        if not seed.isdigit():
            continue
        for r in json.loads(f.read_text(encoding="utf-8")):
            if r.get("run") == run:
                for k, m in r["test"].items():
                    if k.startswith("결합+나이"):
                        out[seed] = m
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre", default=str(REPO / "data/output/fixedcluster_v1"))
    ap.add_argument("--v2", default=None, help="기본 <pre>/v2")
    ap.add_argument("--runs", nargs="+", default=["h", "r", "v1"])
    a = ap.parse_args()
    v2 = Path(a.v2).expanduser() if a.v2 else Path(a.pre).expanduser() / "v2"
    res = {}
    for run in a.runs:
        d = v2 / "features" / run
        if not all((d / f).exists() for f in FILES.values()):
            print(f"[{run}] 특징 파일 없음: {d} (add_candidate_features.py 먼저)")
            continue
        vy, vp, va = load(d / FILES["validation"])
        ty, tp, ta = load(d / FILES["test"])
        grid = {(l, m): top1(l * vp + m * va, vy) for l, m in itertools.product(LAMBDAS, MUS) if l != 0}
        bl, bm = max(grid, key=lambda k: (grid[k], -abs(k[0]), -abs(k[1])))
        r = {"고른 값": {"lambda": bl, "mu": bm, "validation_top1": grid[(bl, bm)]},
             "test": {"인기만": full(tp, ty), "나이만 (새 기사 우선)": full(-ta, ty),
                      "인기 + 나이": full(bl * tp + bm * ta, ty)},
             "test 행": int(len(ty))}
        fu = fusion_top1(v2, run)
        if fu:
            r["SID + 인기 + 나이 (seed별 Top-1)"] = {s: m["top1"] for s, m in fu.items()}
            r["SID의 순수한 몫 (seed별, %p)"] = {s: (m["top1"] - r["test"]["인기 + 나이"]["top1"]) * 100 for s, m in fu.items()}
        res[run.upper()] = r
        print(f"\n===== {run.upper()} (test {len(ty):,}행) =====")
        print(f"고른 값: λ = {bl}, μ = {bm} (validation Top-1 {grid[(bl, bm)]:.2%}, λ : μ = 1 : {bm / bl:.3g})")
        for name, m in r["test"].items():
            print(f"  test {name:<16} Top-1 {m['top1']:.2%} · AUC {m['auc']:.4f} · MRR {m['mrr']:.4f} · nDCG@5 {m['ndcg@5']:.4f}")
        if fu:
            g = r["SID의 순수한 몫 (seed별, %p)"]
            print("  SID + 인기 + 나이: " + " · ".join(f"seed {s} {m['top1']:.2%}" for s, m in fu.items()))
            print("  SID의 순수한 몫 (SID + 인기 + 나이 − 인기 + 나이): " + " · ".join(f"seed {s} {v:+.2f}%p" for s, v in g.items())
                  + f" · 평균 {np.mean(list(g.values())):+.2f}%p")
    if not res:
        sys.exit("특징 파일을 찾지 못했습니다")
    out = v2 / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    f, k = out / "pop_age_only.json", 2
    while f.exists():
        f, k = out / f"pop_age_only_{k}.json", k + 1
    f.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {f}")


if __name__ == "__main__":
    main()
