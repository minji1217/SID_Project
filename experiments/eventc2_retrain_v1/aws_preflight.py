"""EventC2 downstream 실행 전 AWS 점검 (읽기 전용, 아무 파일도 만들거나 바꾸지 않는다)

    cd ~/minji/SID_Project
    python experiments/eventc2_retrain_v1/aws_preflight.py

확인 항목
  A. UNI baseline 데이터: 이름에 shuffled_v2가 들어간 폴더의 parquet (행 수, sha256)
  B. UNI baseline run: run_summary.json의 train/validation 경로가 A의 파일과 같은지 (sha256),
     validation half + test가 shuffled_v2 validation을 50:50으로 나눈 것인지,
     test_metrics.json / test_candidate_scores.parquet / checkpoint_best.pt 존재 여부
  C. post-rqvae 입력: raw behaviors/history (train, validation), category_mapping
  D. SID 파일과 1pos4neg 이전 sequences: UNI / EventC2 SID와 내용이 같은지
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

# (article_id, c1, c2, c3, c4)를 article_id 순으로 정렬한 CSV의 sha256
SID_SIGNATURES = {
    "9276c1d3b4d62f32cd3413a74e174165159ea65d8e9de5e557dfb90e9216f34c": "UNI (lu0.05, m0.5)",
    "97a65313b5d597f0e6edc6ace459ebc492e4680c35eae3a598a54ee63b18676d": "EventC2 (eventc2_lu0.05_m0.5)",
}

SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", "node_modules", "wandb", ".cache"}
_sha_cache: dict[str, str] = {}


def walk(roots, max_depth=9):
    for root in roots:
        root = Path(root).expanduser()
        if not root.exists():
            continue
        base_depth = len(root.parts)
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            depth = len(Path(dirpath).parts) - base_depth
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and depth < max_depth]
            yield Path(dirpath), dirnames, filenames


def sha256(path: Path) -> str:
    key = str(path.resolve())
    if key not in _sha_cache:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        _sha_cache[key] = h.hexdigest()
    return _sha_cache[key]


def rows(path: Path):
    try:
        return pq.read_metadata(path).num_rows
    except Exception as error:  # noqa: BLE001
        return f"읽기 실패: {error}"


def sid_signature(path: Path):
    try:
        df = pd.read_parquet(path, columns=["article_id", "c1", "c2", "c3", "c4"])
    except Exception:  # noqa: BLE001
        return None
    df = df.astype({"article_id": str, "c1": "int64", "c2": "int64", "c3": "int64", "c4": "int64"})
    df = df.sort_values("article_id")
    return hashlib.sha256(df.to_csv(index=False).encode()).hexdigest()


def candidate_signature(paths):
    """(impression_id, candidate_article_ids 순서 포함) multiset. split/정렬 순서와 무관."""
    keys = []
    for path in paths:
        df = pd.read_parquet(path, columns=["impression_id", "candidate_article_ids"])
        keys.extend(
            f"{i}|{','.join(map(str, c))}"
            for i, c in zip(df["impression_id"], df["candidate_article_ids"])
        )
    return hashlib.sha256("\n".join(sorted(keys)).encode()).hexdigest(), len(keys)


def gin_value(gin_text: str, key: str):
    match = re.search(rf"^{re.escape(key)}\s*=\s*(.+)$", gin_text or "", re.M)
    return match.group(1).strip().strip("'\"") if match else None


def find_transformer_root(start: Path):
    for parent in [start, *start.parents]:
        if (parent / "train_transformer.py").exists():
            return parent
    return None


def git_head(path: Path):
    try:
        out = subprocess.run(
            ["git", "-C", str(path), "log", "-1", "--format=%h %d %s"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", nargs="+", default=["~/minji", "~/shared"])
    args = parser.parse_args()

    shuffled_dirs, run_summaries, raw_files, sid_files, seq_files = [], [], [], [], []

    for dirpath, _, filenames in walk(args.roots):
        if "shuffled_v2" in dirpath.name and any(f.endswith(".parquet") for f in filenames):
            shuffled_dirs.append(dirpath)
        for name in filenames:
            path = dirpath / name
            if name == "run_summary.json":
                run_summaries.append(path)
            elif name in ("behaviors.parquet", "history.parquet", "category_mapping.parquet", "articles.parquet"):
                raw_files.append(path)
            elif name == "article_semantic_ids.parquet":
                sid_files.append(path)
            elif name in ("train_sequences.parquet", "validation_sequences.parquet"):
                seq_files.append(path)

    # ------------------------------------------------------------------
    print("=" * 80)
    print("A. UNI baseline 데이터 (*shuffled_v2*)")
    print("=" * 80)
    shuffled_sha = {}
    for directory in sorted(shuffled_dirs):
        print(f"\n[{directory}]")
        for path in sorted(directory.glob("*.parquet")):
            digest = sha256(path)
            shuffled_sha[digest] = path
            print(f"  {path.name:45s} rows={rows(path)}  sha256={digest[:16]}")
    if not shuffled_dirs:
        print("  찾지 못함")

    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("B. Transformer run 중 shuffled_v2 데이터로 학습한 run")
    print("=" * 80)
    matched = 0
    for summary_path in sorted(run_summaries):
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        gin_text = summary.get("gin_config", "")
        run_dir = summary_path.parent
        root = find_transformer_root(run_dir) or run_dir
        train_raw = gin_value(gin_text, "train.train_path")
        valid_raw = gin_value(gin_text, "train.validation_path")
        if not train_raw:
            continue
        train_path = Path(train_raw).expanduser()
        train_path = train_path if train_path.is_absolute() else root / train_path
        valid_path = Path(valid_raw).expanduser() if valid_raw else None
        if valid_path is not None and not valid_path.is_absolute():
            valid_path = root / valid_path
        if not train_path.exists():
            continue
        train_sha = sha256(train_path)
        if train_sha not in shuffled_sha:
            continue

        matched += 1
        best = summary.get("best_metrics", {})
        print(f"\n[{run_dir}]")
        print(f"  Transformer 코드      : {git_head(root)}")
        print(f"  train_path            : {train_raw} -> {train_path.resolve()}")
        print(f"     = {shuffled_sha[train_sha]}  (sha256 {train_sha[:16]})")
        print(f"  validation_path       : {valid_raw} -> {valid_path.resolve() if valid_path else None}")
        for key in (
            "train.seed", "NewsEncoderDecoderTransformer.c4_vocab_size",
            "NewsSequenceDataset.max_history_length", "NewsEncoderDecoderTransformer.d_model",
            "NewsEncoderDecoderTransformer.num_layers", "train.learning_rate", "train.batch_size",
            "train.weight_decay", "NewsEncoderDecoderTransformer.dropout_rate", "train.num_epochs",
            "train.early_stopping_patience",
        ):
            print(f"  {key:45s}: {gin_value(gin_text, key)}")
        print(f"  best_epoch / selection : {summary.get('best_epoch')} / {summary.get('selection_metric')}")
        print(f"  best val top1          : {best.get('val_top1_accuracy')}")
        for name in ("checkpoint_best.pt", "test_metrics.json", "test_candidate_scores.parquet", "predict.log"):
            print(f"  {name:30s}: {'있음' if (run_dir / name).exists() else '없음'}")
        predict_log = run_dir / "predict.log"
        if predict_log.exists():
            for line in predict_log.read_text(errors="ignore").splitlines():
                if line.startswith("Test path:"):
                    print(f"  {line}")
        if (run_dir / "test_metrics.json").exists():
            print(f"  test_metrics: {(run_dir / 'test_metrics.json').read_text().strip()}")

        # validation half + test == shuffled_v2 validation ?
        if valid_path is not None and valid_path.exists():
            test_path = valid_path.parent / "test_sequences_1pos4neg.parquet"
            shuffled_valid = [
                p for p in shuffled_sha.values()
                if p.name.startswith("validation") and p.parent == shuffled_sha[train_sha].parent
            ]
            if test_path.exists() and shuffled_valid:
                half_test = candidate_signature([valid_path, test_path])
                original = candidate_signature(shuffled_valid[:1])
                print(f"  test 파일             : {test_path}  rows={rows(test_path)} sha256={sha256(test_path)[:16]}")
                print(
                    "  validation half + test == shuffled_v2 validation : "
                    f"{half_test[0] == original[0]} (rows {half_test[1]} vs {original[1]})"
                )
    print(f"\n  shuffled_v2 데이터로 학습한 run: {matched}개 (전체 run_summary {len(run_summaries)}개 중)")

    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("C. post-rqvae 입력 (raw behaviors/history, category_mapping)")
    print("=" * 80)
    for path in sorted(raw_files):
        print(f"  {str(path):95s} rows={rows(path)}  {path.stat().st_size / 1e6:,.1f}MB")
    if not raw_files:
        print("  찾지 못함")
    repo_raw = Path(__file__).resolve().parents[2] / "data" / "raw"
    print(f"\n  이 레포가 읽는 위치 (src/config.py RAW_DIR): {repo_raw}")
    for rel in ("train/behaviors.parquet", "train/history.parquet",
                "validation/behaviors.parquet", "validation/history.parquet"):
        target = repo_raw / rel
        print(f"    {rel:32s}: {'있음' if target.exists() else '없음'}")

    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("D. SID 파일과 1pos4neg 이전 sequences")
    print("=" * 80)
    for path in sorted(sid_files):
        signature = sid_signature(path)
        label = SID_SIGNATURES.get(signature, "다른 SID")
        print(f"  {str(path):95s} -> {label}")
    print()
    for path in sorted(seq_files):
        sibling = path.parent / "article_semantic_ids.parquet"
        label = SID_SIGNATURES.get(sid_signature(sibling), "?") if sibling.exists() else "옆에 SID 없음"
        print(f"  {str(path):95s} rows={rows(path)}  SID={label}")


if __name__ == "__main__":
    main()
