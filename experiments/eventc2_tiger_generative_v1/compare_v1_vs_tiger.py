"""같은 EventC2 test set에서 V1 candidate ranker vs TIGER-style generative 비교 (집계만)

    python experiments/eventc2_tiger_generative_v1/compare_v1_vs_tiger.py \
        --v1-scores <EXP>/transformer_runs/seed42/test_candidate_scores.parquet \
        --tiger-run <EXP>/eventc2_tiger_generative_v1/seed42 \
        --out <EXP>/eventc2_tiger_generative_v1/compare_v1_vs_tiger.md

두 score 파일의 test sample이 같은지 (impression_id, 후보 article 5개, positive) 먼저 확인한다.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_tiger import ranking_metrics  # noqa: E402

import re

V1_CONFIG_KEYS = [
    "NewsEncoderDecoderTransformer.d_model", "NewsEncoderDecoderTransformer.num_heads",
    "NewsEncoderDecoderTransformer.num_layers", "NewsEncoderDecoderTransformer.d_ff",
    "NewsEncoderDecoderTransformer.dropout_rate", "NewsEncoderDecoderTransformer.use_sep",
    "NewsEncoderDecoderTransformer.c4_vocab_size", "NewsSequenceDataset.max_history_length",
    "train.learning_rate", "train.batch_size", "train.weight_decay", "train.seed",
    "train.num_epochs", "train.early_stopping_patience", "train.train_path",
]


def v1_config(run_dir: Path) -> dict:
    """V1 run_summary.json의 gin_config(실제 학습에 쓰인 최종 binding)에서 읽는다."""
    summary = json.loads((run_dir / "run_summary.json").read_text())
    gin_text = re.sub(r"\\\n\s*", "", summary.get("gin_config", ""))   # gin의 줄 이어쓰기(\) 합치기
    found = {}
    for key in V1_CONFIG_KEYS:
        m = re.search(rf"^{re.escape(key)}\s*=\s*(.+)$", gin_text, re.M)
        found[key] = m.group(1).strip() if m else None
    found["best_epoch"] = summary.get("best_epoch")
    found["selection_metric"] = summary.get("selection_metric")
    return found


METRICS = ["top1_accuracy", "mrr", "ndcg@5", "auc", "preference_loss", "positive_probability"]


def load(path: Path):
    df = pd.read_parquet(path).sort_values(["sample_index"], kind="stable")
    n = df.sample_index.nunique()
    if len(df) != n * 5:
        raise ValueError(f"{path}: sample마다 후보 5개가 아닙니다.")
    scores = df.candidate_score.to_numpy(float).reshape(n, 5)
    labels = df.label.to_numpy(float).astype(int).reshape(n, 5)
    arts = df.article_id.astype(str).to_numpy().reshape(n, 5)
    imps = df.impression_id.to_numpy().reshape(n, 5)[:, 0]
    keys = Counter((str(i), tuple(sorted(a)), a[l.argmax()]) for i, a, l in zip(imps, arts, labels))
    return scores, labels, keys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--v1-scores", type=Path, required=True)
    p.add_argument("--v1-run", type=Path, required=True, help="V1 seed42 run 폴더 (run_summary.json)")
    p.add_argument("--tiger-run", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    v1_s, v1_l, v1_k = load(a.v1_scores)
    tg_s, tg_l, tg_k = load(a.tiger_run / "test_candidate_scores.parquet")
    same = v1_k == tg_k
    print(f"test sample 동일: {same} (V1 {sum(v1_k.values()):,} / TIGER {sum(tg_k.values()):,})")
    if not same:
        raise SystemExit("두 모델의 test sample 구성이 다릅니다. 비교를 중단합니다.")

    v1 = ranking_metrics(v1_s, v1_l)
    tg = ranking_metrics(tg_s, tg_l)
    t = json.loads((a.tiger_run / "test_metrics.json").read_text())
    summary = json.loads((a.tiger_run / "run_summary.json").read_text())

    lines = ["# EventC2 test: V1 candidate ranker vs TIGER-style generative", "",
             f"같은 test sample {sum(v1_k.values()):,}개 (impression·후보 5개·positive 일치 확인)", "",
             "| 지표 | V1 ranker | TIGER-style | 차이 |", "|---|---|---|---|"]
    for m in METRICS:
        lines.append(f"| {m} | {v1[m]:.6f} | {tg[m]:.6f} | {tg[m] - v1[m]:+.6f} |")
    v1cfg = v1_config(a.v1_run)
    tcfg = summary["config"]
    lines += ["", "## 실제 학습 설정 (V1: run_summary.json의 gin_config, TIGER: config.json)", "",
              "| 항목 | V1 | TIGER |", "|---|---|---|"]
    pairs = [("d_model", "NewsEncoderDecoderTransformer.d_model", "d_model"),
             ("heads", "NewsEncoderDecoderTransformer.num_heads", "num_heads"),
             ("layers", "NewsEncoderDecoderTransformer.num_layers", "num_layers"),
             ("d_ff", "NewsEncoderDecoderTransformer.d_ff", "d_ff"),
             ("dropout", "NewsEncoderDecoderTransformer.dropout_rate", "dropout"),
             ("max history", "NewsSequenceDataset.max_history_length", "max_history"),
             ("learning rate", "train.learning_rate", "lr"),
             ("batch size", "train.batch_size", "batch_size"),
             ("weight decay", "train.weight_decay", None),
             ("seed", "train.seed", "seed"),
             ("epochs / patience", None, None),
             ("c4 vocab", "NewsEncoderDecoderTransformer.c4_vocab_size", None),
             ("train data", "train.train_path", "train")]
    for name, vk, tk in pairs:
        if name == "epochs / patience":
            lines.append(f"| {name} | {v1cfg['train.num_epochs']} / {v1cfg['train.early_stopping_patience']} | "
                         f"{tcfg['epochs']} / {tcfg['patience']} |")
            continue
        tv = tcfg.get(tk) if tk else ("level " + str(tcfg["level_sizes"][3]) if name == "c4 vocab" else "-")
        lines.append(f"| {name} | {v1cfg.get(vk)} | {tv} |")
    lines.append(f"| best epoch | {v1cfg['best_epoch']} | {summary['best_epoch']} |")

    lines += ["", f"TIGER best epoch {summary['best_epoch']} / {summary['epochs_run']} epoch", "",
              "## TIGER 진단 (test)", "", "| level | CE | token accuracy |", "|---|---|---|"]
    for l in ("c1", "c2", "c3", "c4"):
        lines.append(f"| {l} | {t['level_ce'][l]:.4f} | {t['level_token_accuracy'][l]:.4f} |")
    lines += ["", f"- positive 평균 log-prob {t['positive_mean_log_prob']:.4f}",
              f"- negative 평균 log-prob {t['negative_mean_log_prob']:.4f}",
              f"- gap {t['positive_negative_score_gap']:.4f}", "",
              "| 누적 점수 | Top-1 | AUC | MRR | nDCG@5 |", "|---|---|---|---|---|"]
    for k, m in t["cumulative_levels"].items():
        lines.append(f"| {k} | {m['top1_accuracy']:.4f} | {m['auc']:.4f} | {m['mrr']:.4f} | {m['ndcg@5']:.4f} |")
    text = "\n".join(lines) + "\n"
    a.out.write_text(text, encoding="utf-8")
    (a.out.with_suffix(".json")).write_text(json.dumps({"v1": v1, "v1_config": v1cfg, "tiger": tg,
                                                       "tiger_diagnostics": t}, indent=2))
    print(text)


if __name__ == "__main__":
    main()
