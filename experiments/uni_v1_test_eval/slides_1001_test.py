"""10/1 발표(슬라이드 3~5, 9)의 validation 수치를 test에서 다시 계산한다. 학습 없음, 새 예측 없음.

입력은 run_aws.sh가 이미 만든 UNI V1 예측 파일(후보별 c1/c2/c3 log-prob)과 1pos4neg sequence 파일이다.

  슬라이드 3  L1 / L2 / L3 / L1+L2 / L1+L2+L3 단독 점수
  슬라이드 4  S = L1 + a2*L2 + a3*L3 격자 42개 (a2 7개 x a3 6개, 원래 run_level_weight_grid.py와 같은 격자)
              validation에서 고른 (1, 0.5, 0.1)의 test 성능과 V1 대비 rescued / broken / net
  슬라이드 5  Random(이론값) / Popularity(train positive의 (c1,c2,c3) 등장 횟수) / V1 equal

Top-1은 원래 분석과 같이 np.argmax(동점이면 앞 후보)이고, 동점을 1/k로 나눈 tie-aware Top-1도 같이 낸다.
validation 값이 발표 슬라이드와 같으면 같은 계산이 맞다는 확인이 된다.

    python experiments/uni_v1_test_eval/slides_1001_test.py \
        --scores-dir data/output/experiments/uni_v1_test_eval \
        --data-dir ~/minji/SID_Project-transformer/Transformer/datasets/ebnerd
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eventc2_tiger_generative_v1"))
from train_tiger import ranking_metrics  # noqa: E402

SPLITS = {"validation_half": "validation_sequences_1pos4neg_half", "test": "test_sequences_1pos4neg"}
ALPHA2_GRID = [0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50]
ALPHA3_GRID = [0.00, 0.10, 0.25, 0.50, 0.75, 1.00]
SINGLE = {"L1": (1, 0, 0), "L2": (0, 1, 0), "L3": (0, 0, 1), "L1+L2": (1, 1, 0), "L1+L2+L3": (1, 1, 1)}
CHOSEN = (1.0, 0.5, 0.1)
RADIX = 1000


def load_scores(path: Path):
    df = pd.read_parquet(path).sort_values("sample_index", kind="stable")
    n = df.sample_index.nunique()
    if len(df) != 5 * n:
        raise ValueError(f"{path}: sample마다 후보 5개가 아닙니다.")
    L = np.stack([df[f"c{k}_log_prob"].to_numpy(float) for k in (1, 2, 3)], -1).reshape(n, 5, 3)
    labels = df.label.to_numpy(float).astype(int).reshape(n, 5)
    return L, labels


def tie_aware_top1(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[np.arange(len(scores)), labels.argmax(1)][:, None]
    best = scores.max(1, keepdims=True)
    k = (scores == best).sum(1)
    return float(np.where(pos[:, 0] == best[:, 0], 1.0 / k, 0.0).mean())


def metrics(scores, labels) -> dict:
    m = ranking_metrics(scores, labels)
    m["top1_tie_aware"] = tie_aware_top1(scores, labels)
    return m


def correct(scores, labels) -> np.ndarray:
    return labels[np.arange(len(scores)), scores.argmax(1)] == 1


def popularity(train: pd.DataFrame, target: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    def keys(df):
        c = np.stack([np.stack(df[f"candidate_c{k}"].to_numpy()) for k in (1, 2, 3)], -1).astype(np.int64)
        return (c[..., 0] * RADIX + c[..., 1]) * RADIX + c[..., 2]
    tl = np.stack(train["candidate_labels"].to_numpy()).astype(int)
    tk = keys(train)
    counter = Counter(tk[np.arange(len(tk)), tl.argmax(1)].tolist())
    vk = keys(target)
    score = np.vectorize(lambda k: counter.get(int(k), 0))(vk).astype(float)
    return score, np.stack(target["candidate_labels"].to_numpy()).astype(int)


def tie_aware_rank_metrics(scores, labels) -> dict:
    """원래 diagnose_sid.score_metrics와 같은 정의 (동점은 평균 순위)."""
    rows = np.arange(len(scores))
    ps = scores[rows, labels.argmax(1)]
    greater = (scores > ps[:, None]).sum(1)
    equal = (scores == ps[:, None]).sum(1)
    rank = greater + (equal + 1) / 2.0
    return {"top1": float(np.mean((greater == 0) / equal)), "mrr": float(np.mean(1 / rank)),
            "ndcg@5": float(np.mean(1 / np.log2(rank + 1))), "auc": float(np.mean((4 - greater - (equal - 1) * 0.5) / 4))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scores-dir", type=Path, required=True, help="run_aws.sh 출력 (validation_half/, test/ 아래 test_candidate_scores.parquet)")
    p.add_argument("--data-dir", type=Path, required=True, help="train/validation half/test 1pos4neg parquet 폴더")
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()
    out = a.out or a.scores_dir / "slides_1001_test.json"
    if out.exists():
        raise SystemExit(f"{out} 가 이미 있습니다 (덮어쓰지 않음)")

    result = {}
    train = None
    for split, seq in SPLITS.items():
        L, labels = load_scores(a.scores_dir / split / "test_candidate_scores.parquet")
        r = {"samples": int(len(labels))}
        r["slide3_single"] = {k: metrics((L * np.array(w)).sum(-1), labels) for k, w in SINGLE.items()}
        grid = {}
        for a3 in ALPHA3_GRID:
            for a2 in ALPHA2_GRID:
                grid[f"{a2:.2f},{a3:.2f}"] = float(correct((L * np.array([1.0, a2, a3])).sum(-1), labels).mean())
        r["slide4_grid_top1"] = grid
        v1 = correct(L.sum(-1), labels)
        ch = correct((L * np.array(CHOSEN)).sum(-1), labels)
        r["slide4_chosen"] = {"alpha": CHOSEN, **metrics((L * np.array(CHOSEN)).sum(-1), labels),
                              "rescued": int((~v1 & ch).sum()), "broken": int((v1 & ~ch).sum()), "net": int(ch.sum() - v1.sum())}
        if train is None:
            train = pd.read_parquet(a.data_dir / "train_sequences_1pos4neg.parquet",
                                    columns=["candidate_c1", "candidate_c2", "candidate_c3", "candidate_labels"])
        seqdf = pd.read_parquet(a.data_dir / f"{seq}.parquet",
                                columns=["candidate_c1", "candidate_c2", "candidate_c3", "candidate_labels"])
        pop, plabels = popularity(train, seqdf)
        if len(plabels) != len(labels):
            raise ValueError(f"{split}: sequence 행 수 {len(plabels)} != 예측 sample 수 {len(labels)}")
        r["slide5"] = {"random_top1": 0.2, "popularity": tie_aware_rank_metrics(pop, plabels),
                       "v1_equal": metrics(L.sum(-1), labels)}
        result[split] = r

        print(f"\n===== {split} ({len(labels):,} samples) =====")
        print("[슬라이드 3] 단독 점수 Top-1 (argmax / tie-aware)")
        for k, m in r["slide3_single"].items():
            print(f"  {k:9s} {m['top1_accuracy']:.4%}  /  {m['top1_tie_aware']:.4%}   AUC {m['auc']:.4f}")
        print("[슬라이드 4] 격자 Top-1 (행 a3, 열 a2)")
        print("  a3\\a2 " + " ".join(f"{x:>6.2f}" for x in ALPHA2_GRID))
        for a3 in reversed(ALPHA3_GRID):
            print(f"  {a3:5.2f} " + " ".join(f"{grid[f'{a2:.2f},{a3:.2f}']*100:6.2f}" for a2 in ALPHA2_GRID))
        best = max(grid, key=grid.get)
        c = r["slide4_chosen"]
        print(f"  이 split 최고 (a2,a3)=({best}) {grid[best]:.4%} · (1,0.5,0.1) {c['top1_accuracy']:.4%} "
              f"· rescued {c['rescued']:,} broken {c['broken']:,} net {c['net']:+,}")
        pm = r["slide5"]["popularity"]
        print(f"[슬라이드 5] Random 20.00% · Popularity {pm['top1']:.4%} (AUC {pm['auc']:.4f}) · "
              f"V1 equal {r['slide5']['v1_equal']['top1_accuracy']:.4%}")

    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
