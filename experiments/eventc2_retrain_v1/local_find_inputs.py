"""로컬 PC에서 AWS로 옮길 파일을 찾고 확인한다 (읽기 전용).

    python local_find_inputs.py --search <찾을 상위 폴더> [<폴더2> ...]

1. EB-NeRD raw 4개 (train/validation × behaviors/history) 후보를 찾는다.
2. 찾은 shuffled_v2의 impression이 그 behaviors 안에 모두 들어 있는지 확인한다.
   (train 1pos4neg ⊆ train/behaviors, validation 1pos4neg ⊆ validation/behaviors)
3. UNI Transformer run 후보 (run_summary.json, checkpoint_best.pt, test_metrics.json,
   test_candidate_scores.parquet)와 validation half / test 파일을 찾는다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pyarrow.parquet as pq

SKIP = {".git", ".venv", "venv", "__pycache__", "node_modules", "wandb", "AppData"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def impression_ids(path: Path) -> set:
    return set(pq.read_table(path, columns=["impression_id"]).column(0).to_pylist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--search", nargs="+", required=True)
    args = parser.parse_args()

    behaviors, histories, shuffled, runs, splits = [], [], [], [], []
    for root in args.search:
        for dirpath, dirnames, filenames in os.walk(Path(root).expanduser()):
            dirnames[:] = [d for d in dirnames if d not in SKIP]
            d = Path(dirpath)
            for name in filenames:
                p = d / name
                if name == "behaviors.parquet":
                    behaviors.append(p)
                elif name == "history.parquet":
                    histories.append(p)
                elif name == "run_summary.json":
                    runs.append(p)
                elif name in ("validation_sequences_1pos4neg_half.parquet", "test_sequences_1pos4neg.parquet"):
                    splits.append(p)
                elif "shuffled_v2" in str(d) and name.endswith("_1pos4neg.parquet"):
                    shuffled.append(p)

    print("=" * 80)
    print("1. raw 후보 (behaviors / history)")
    print("=" * 80)
    for p in sorted(behaviors + histories):
        print(f"  {p}  rows={pq.read_metadata(p).num_rows:,}  {p.stat().st_size / 1e6:,.1f}MB  split={p.parent.name}")

    print("\n" + "=" * 80)
    print("2. shuffled_v2 impression이 behaviors 안에 있는지")
    print("=" * 80)
    for s in sorted(shuffled):
        split = "train" if s.name.startswith("train") else "validation"
        ids = impression_ids(s)
        print(f"\n  [{s}] rows={pq.read_metadata(s).num_rows:,} impressions={len(ids):,} sha256={sha256(s)[:16]}")
        for b in sorted(p for p in behaviors if p.parent.name == split):
            covered = len(ids & impression_ids(b))
            print(f"    {b}: {covered:,}/{len(ids):,} 포함 {'<- 사용 가능' if covered == len(ids) else ''}")

    print("\n" + "=" * 80)
    print("3. Transformer run / validation half / test 후보")
    print("=" * 80)
    for r in sorted(runs):
        try:
            summary = json.loads(r.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        gin_text = summary.get("gin_config", "")
        train_line = next((l for l in gin_text.splitlines() if l.startswith("train.train_path")), "")
        seed_line = next((l for l in gin_text.splitlines() if l.startswith("train.seed")), "")
        files = {n: (r.parent / n).exists() for n in
                 ("checkpoint_best.pt", "test_metrics.json", "test_candidate_scores.parquet")}
        print(f"  {r.parent}\n    {train_line} | {seed_line} | best_epoch={summary.get('best_epoch')}\n    {files}")
    for p in sorted(splits):
        print(f"  {p}  rows={pq.read_metadata(p).num_rows:,}  sha256={sha256(p)[:16]}")


if __name__ == "__main__":
    main()
