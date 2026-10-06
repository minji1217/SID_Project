"""새 SID 기반 1pos4neg 데이터로 Transformer를 학습하고 Test까지 평가한다.

기존 A와 같은 조건을 쓰기 위해 Transformer 레포
(minji1217/sid_project-transformer, claude/hopeful-mendel-fhc2n4 브랜치)의
analysis/rerun_final_config.py와 같은 V1 최종 config / 학습 예산을 그대로 넘긴다.
데이터 경로, save_dir, c4_vocab_size만 이 실험 값으로 바꾼다.

    python experiments/eventc2_train_v1/run_transformer.py \
        --transformer-root ~/<repo>/sid_project-transformer/Transformer \
        --dataset-dir <exp>/transformer_datasets \
        --train-path  <exp>/candidate_1pos4neg_shuffled/train_sequences_1pos4neg.parquet \
        --sid-path    <exp>/semantic_ids/article_semantic_ids.parquet \
        --out-dir     <exp>/transformer_runs \
        --seeds 42

run_dir(<out-dir>/seed<seed>)에 이미 run_summary.json이 있으면 재학습하지 않는다.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


# rerun_final_config.py의 FINAL_CONFIG와 같다. 바꾸지 않는다.
FINAL_CONFIG: Dict[str, Any] = {
    "NewsSequenceDataset.max_history_length": 50,
    "NewsEncoderDecoderTransformer.use_sep": False,
    "NewsEncoderDecoderTransformer.d_model": 256,
    "NewsEncoderDecoderTransformer.num_heads": 8,
    "NewsEncoderDecoderTransformer.num_layers": 2,
    "NewsEncoderDecoderTransformer.d_ff": 1024,
    "NewsEncoderDecoderTransformer.dropout_rate": 0.0,
    "train.learning_rate": 5e-05,
    "train.batch_size": 128,
    "train.weight_decay": 0.0,
}

NUM_EPOCHS = 30
PATIENCE = 5

# 기존 configs/transformer_ebnerd.gin의 c4_vocab_size (A의 max c4 + 1)
BASE_C4_VOCAB_SIZE = 28


def format_binding(key: str, value: Any) -> str:
    if isinstance(value, bool):
        return f"{key} = {value}"
    if isinstance(value, str):
        return f'{key} = "{value}"'
    return f"{key} = {value!r}"


def resolve_c4_vocab_size(sid_path: Path) -> int:
    max_c4 = int(pd.read_parquet(sid_path, columns=["c4"])["c4"].max())
    needed = max_c4 + 1

    if needed <= BASE_C4_VOCAB_SIZE:
        print(f"max c4={max_c4} -> 기존과 같은 c4_vocab_size={BASE_C4_VOCAB_SIZE} 사용")
        return BASE_C4_VOCAB_SIZE

    # history에 c4가 들어가므로 vocab이 부족하면 index error가 난다.
    # 크기가 바뀌면 같은 seed라도 임베딩 초기화 RNG 순서가 달라지므로 기록해 둔다.
    print(
        f"WARNING: max c4={max_c4} > 기존 vocab {BASE_C4_VOCAB_SIZE - 1}. "
        f"c4_vocab_size={needed}로 늘린다. "
        "A와 초기화가 완전히 같지 않으므로 A를 같은 vocab으로 다시 학습해 비교하는 것을 권장한다."
    )
    return needed


def model_bindings(c4_vocab_size: int) -> Dict[str, Any]:
    bindings = {
        key: value
        for key, value in FINAL_CONFIG.items()
        if not key.startswith("train.")
    }
    bindings["NewsEncoderDecoderTransformer.c4_vocab_size"] = c4_vocab_size
    return bindings


def run(command: List[str], cwd: Path, log_path: Path) -> None:
    print("$", " ".join(command))
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
        return_code = process.wait()

    if return_code != 0:
        raise SystemExit(f"command failed ({return_code}). log: {log_path}")


def train_and_test(
    args: argparse.Namespace,
    seed: int,
    c4_vocab_size: int,
) -> Dict[str, Any]:
    transformer_root = args.transformer_root.resolve()
    run_dir = (args.out_dir / f"seed{seed}").resolve()
    validation_path = (args.dataset_dir / "validation_sequences_1pos4neg_half.parquet").resolve()
    test_path = (args.dataset_dir / "test_sequences_1pos4neg.parquet").resolve()
    summary_path = run_dir / "run_summary.json"

    if summary_path.exists():
        print(f"seed {seed}: 이미 학습됨. 재학습하지 않는다 ({summary_path})")
    else:
        bindings = dict(FINAL_CONFIG)
        bindings.update(model_bindings(c4_vocab_size))
        bindings.update({
            "train.train_path": str(args.train_path.resolve()),
            "train.validation_path": str(validation_path),
            "train.num_epochs": NUM_EPOCHS,
            "train.early_stopping_patience": PATIENCE,
            "train.seed": seed,
            "train.save_every_epoch": False,
            "train.save_optimizer_state": False,
            "train.save_dir": str(run_dir),
            "train.num_workers": args.num_workers,
            "train.progress_interval": args.progress_interval,
        })

        command = [sys.executable, "-u", "train_transformer.py", "--config", args.config]
        for key in sorted(bindings):
            command += ["--gin-binding", format_binding(key, bindings[key])]

        run(command, transformer_root, run_dir / "train.log")

    prediction_path = run_dir / "test_candidate_scores.parquet"
    test_metrics_path = run_dir / "test_metrics.json"

    if not test_metrics_path.exists():
        command = [
            sys.executable, "-u", "predict_sid.py",
            "--config", args.config,
            "--checkpoint", str(run_dir / "checkpoint_best.pt"),
            "--test_path", str(test_path),
            "--output_path", str(prediction_path),
            "--batch_size", "128",
            "--num_workers", str(args.num_workers),
        ]
        for key, value in sorted(model_bindings(c4_vocab_size).items()):
            command += ["--gin-binding", format_binding(key, value)]
        run(command, transformer_root, run_dir / "predict.log")

        run(
            [
                sys.executable, "-u", "evaluate/metrics.py",
                "--prediction_path", str(prediction_path),
                "--output_path", str(test_metrics_path),
            ],
            transformer_root,
            run_dir / "metrics.log",
        )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    test_metrics = json.loads(test_metrics_path.read_text(encoding="utf-8"))

    return {
        "seed": seed,
        "run_dir": str(run_dir),
        "c4_vocab_size": c4_vocab_size,
        "best_epoch": summary.get("best_epoch"),
        "validation": summary.get("best_metrics", {}),
        "test": test_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="eventc2 SID Transformer 학습 + Test 평가")
    parser.add_argument("--transformer-root", type=Path, required=True,
                        help="sid_project-transformer/Transformer 폴더 (hopeful-mendel 브랜치)")
    parser.add_argument("--config", default="configs/transformer_ebnerd.gin",
                        help="기존 A와 같은 base gin (Transformer 폴더 기준 상대경로 가능)")
    parser.add_argument("--train-path", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True,
                        help="validation half / test parquet이 있는 폴더")
    parser.add_argument("--sid-path", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--progress-interval", type=int, default=50)
    args = parser.parse_args()

    if not (args.transformer_root / "train_transformer.py").exists():
        raise FileNotFoundError(f"train_transformer.py가 없습니다: {args.transformer_root}")

    c4_vocab_size = resolve_c4_vocab_size(args.sid_path)

    results = [train_and_test(args, seed, c4_vocab_size) for seed in args.seeds]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "transformer_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"saved: {args.out_dir / 'transformer_results.json'}")


if __name__ == "__main__":
    main()
