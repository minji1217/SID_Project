"""이미 학습된 P2 Direct checkpoint로 validation half 재현 확인 후 test를 예측한다. 학습 없음.

10/1 발표 슬라이드 7(P2 c1234 mean / c123 mean / Regularization best)의 validation 값을 test로 바꾸기 위한 것이다.
모델 복원은 SID_Project-transformer의 analysis/repredict_direct.py와 같은 방식(run_metadata + checkpoint의 gin binding)이다.

- run 폴더(sweep_out/...)에는 아무것도 쓰지 않는다. 결과는 --out-dir에만 쓰고, 이미 있으면 멈춘다.
- validation half를 먼저 다시 예측해 best_metrics.json의 저장값과 1e-6 안에서 같을 때만 test를 예측한다.

    cd ~/minji/SID_Project
    python experiments/uni_v1_test_eval/predict_direct_test.py \
        --transformer-root ~/minji/SID_Project-transformer/Transformer \
        --out-dir data/output/experiments/uni_v1_test_eval/p2_direct
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

DEFAULT_RUNS = {
    "P2 c1234 mean": "sweep_out/ebnerd_v2/priority2_direct/bilinear/seed42",
    "P2 c123 mean": "sweep_out/ebnerd_v2/priority2_direct/bilinear_c123/seed42",
    "Regularization best (do0.05, wd0.05)": "sweep_out/ebnerd_v2/priority2_direct/regularization_sweep/do0.05_wd0.0500/seed42",
}
ALLOWED_MISSING = {"level_weight_vector"}
KEYS = ("top1_accuracy", "mrr", "ndcg@5", "auc")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--transformer-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--test", default="datasets/ebnerd/test_sequences_1pos4neg.parquet")
    p.add_argument("--run", action="append", default=None, metavar="LABEL=PATH",
                   help="기본 3개 대신 지정 (Transformer 루트 기준 상대경로 가능)")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    a = p.parse_args()

    out_dir = a.out_dir.expanduser().resolve()
    if out_dir.exists():
        print(f"{out_dir} 가 이미 있습니다 (덮어쓰지 않음)")
        return 1
    root = a.transformer_root.expanduser().resolve()
    sys.path.insert(0, str(root))
    os.chdir(root)

    import gin
    import torch
    from torch.utils.data import DataLoader
    from data.sequence import NewsSequenceDataset, collate_news_sequences
    from modules.direct_scorer import DirectScorer
    from modules.model import NewsEncoderDecoderTransformer
    from analysis.run_priority2_direct import offline_metrics, predict_validation, resolve_path

    runs = dict(r.split("=", 1) for r in a.run) if a.run else DEFAULT_RUNS
    missing = [f"{k}: {v}" for k, v in runs.items() if not (resolve_path(v) / "checkpoint_best.pt").exists()]
    if missing:
        print("checkpoint를 찾지 못한 run:\n  " + "\n  ".join(missing))
        print("ls sweep_out/ebnerd_v2/priority2_direct/*/ 로 폴더 이름을 확인하고 --run LABEL=PATH 로 지정하세요.")
        return 1

    test_path = resolve_path(a.test)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}

    def predict(model, path):
        ds = NewsSequenceDataset(parquet_path=str(path))
        loader = DataLoader(ds, batch_size=a.batch_size, shuffle=False, num_workers=a.num_workers,
                            collate_fn=collate_news_sequences, pin_memory=(device.type == "cuda"))
        return predict_validation(model, loader, device)

    for label, rel in runs.items():
        run_dir = resolve_path(rel)
        meta = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
        ckpt = torch.load(run_dir / "checkpoint_best.pt", map_location="cpu")
        print(f"\n===== {label}\n  run {run_dir} · checkpoint epoch {ckpt.get('epoch')}")

        gin.clear_config()
        gin.parse_config_file(str(resolve_path(meta["config"])), skip_unknown=True)
        bindings = ckpt.get("gin_bindings") or meta.get("gin_bindings") or []
        if bindings:
            gin.parse_config(bindings, skip_unknown=True)
        model = DirectScorer(
            backbone=NewsEncoderDecoderTransformer().to(device),
            scorer_type=meta["scorer_type"],
            history_levels=int(meta.get("history_levels", 4)),
            history_mode=meta.get("history_mode", "mean"),
            level_weights=tuple(meta.get("level_weights", (1.0, 1.0, 1.0))),
            use_c4_identity=bool(meta.get("use_c4_identity", False)),
            beta4=float(meta.get("beta4", 0.0)),
        ).to(device)
        miss, unexp = model.load_state_dict(ckpt["model_state_dict"], strict=False)
        bad = [m for m in miss if m not in ALLOWED_MISSING]
        if bad or unexp:
            print(f"  state_dict 불일치 missing={bad} unexpected={list(unexp)} → 중단")
            return 1

        val_pred = predict(model, resolve_path(meta["validation_path"]))
        val = offline_metrics(val_pred)
        stored = json.loads((run_dir / "best_metrics.json").read_text(encoding="utf-8"))["validation"]
        worst = max(abs(val[k] - stored[k]) for k in KEYS)
        print(f"  validation 재현: Top-1 {val['top1_accuracy']:.4%} (저장 {stored['top1_accuracy']:.4%}) · 최대 차 {worst:.2e}")
        if worst > 1e-6:
            print("  저장값과 다릅니다. test를 예측하지 않고 중단합니다.")
            return 1

        test_pred = predict(model, test_path)
        test = offline_metrics(test_pred)
        print(f"  TEST ({test_pred[test_pred.columns[1]].nunique():,} samples): "
              + " · ".join(f"{k} {test[k]:.4f}" for k in KEYS))
        out_dir.mkdir(parents=True, exist_ok=True)
        safe = label.split(" (")[0].replace(" ", "_")
        test_pred.to_parquet(out_dir / f"{safe}_test_predictions.parquet", index=False)
        results[label] = {"run": str(run_dir), "checkpoint_epoch": ckpt.get("epoch"),
                          "validation": {k: val[k] for k in KEYS}, "test": {k: test[k] for k in KEYS}}

    (out_dir / "p2_direct_test_metrics.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n{'run':40s} {'Val Top-1':>10s} {'Test Top-1':>11s} {'Test AUC':>9s}")
    for k, v in results.items():
        print(f"{k:40s} {v['validation']['top1_accuracy']:>10.4%} {v['test']['top1_accuracy']:>11.4%} {v['test']['auc']:>9.4f}")
    print(f"saved -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
