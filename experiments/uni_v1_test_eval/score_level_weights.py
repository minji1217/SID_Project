"""UNI V1 예측 파일(c1/c2/c3 log-prob)로 level weight 설정별 지표를 다시 계산한다. 학습 없음.

    python experiments/uni_v1_test_eval/score_level_weights.py \
        --scores <out>/validation/test_candidate_scores.parquet --name validation_half
    python experiments/uni_v1_test_eval/score_level_weights.py \
        --scores <out>/test/test_candidate_scores.parquet --name test

S = a1 * L1 + a2 * L2 + a3 * L3 (L = 후보의 level별 log-prob, predict_sid.py 출력)
지표 정의는 Transformer/evaluate/metrics.py + predict_sid.py와 같다 (train_tiger.ranking_metrics).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eventc2_tiger_generative_v1"))
from train_tiger import ranking_metrics  # noqa: E402

# sweep_out/ebnerd_v2/learnable_level_weights/seed42/{best_alpha,final_metrics}.json 값
CONFIGS = [
    ("V1 (1, 1, 1)", (1.0, 1.0, 1.0)),
    ("learned alpha (train 학습, best epoch 4)", (1.0431498131168673, 0.9554956763120237, 1.001354510571109)),
    ("coarse grid best (validation에서 선택)", (1.0, 0.5, 0.1)),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scores", type=Path, required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()

    df = pd.read_parquet(a.scores).sort_values("sample_index", kind="stable")
    n = df.sample_index.nunique()
    if len(df) != 5 * n:
        raise ValueError("sample마다 후보 5개가 아닙니다.")
    L = np.stack([df[f"c{k}_log_prob"].to_numpy(float) for k in (1, 2, 3)], -1).reshape(n, 5, 3)
    labels = df.label.to_numpy(float).astype(int).reshape(n, 5)

    # predict_sid의 candidate_score = L1 + L2 + L3 인지 확인
    s = df.candidate_score.to_numpy(float).reshape(n, 5)
    gap = float(np.abs(s - L.sum(-1)).max())
    print(f"[{a.name}] samples {n:,} · candidate_score - (L1+L2+L3) 최대 차 {gap:.2e}")

    rows = []
    for name, alpha in CONFIGS:
        m = ranking_metrics((L * np.array(alpha)).sum(-1), labels)
        rows.append({"split": a.name, "config": name, "alpha": alpha, **m})
        print(f"  {name:42s} Top-1 {m['top1_accuracy']:.4f}  MRR {m['mrr']:.4f}  nDCG@5 {m['ndcg@5']:.4f}  "
              f"AUC {m['auc']:.4f}  Pref {m['preference_loss']:.4f}  PosProb {m['positive_probability']:.4f}")
    print("  (Preference Loss / Positive Probability는 alpha 합이 3이 아닌 설정끼리 직접 비교하지 않는다)")
    out = a.out or a.scores.with_name(f"level_weight_metrics_{a.name}.json")
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
