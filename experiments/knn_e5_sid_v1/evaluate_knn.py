#!/usr/bin/env python3
"""E5 KNN 평가: 원본 E5 임베딩에서 가까운 기사가 SID prefix도 공유하는가 (발표 8장).

방법
  1. 대상 기사(기본: Train 9,738개)의 E5 임베딩을 L2 정규화한다.
  2. 기사 A마다 나머지 기사 전체와 cosine을 계산해 자기 자신을 뺀 Top-k(기본 10)를 고른다.
  3. 이웃이 A와 same c1 / same (c1,c2) / same (c1,c2,c3)인지 센다.
  4. 모든 기사에 반복한 (기사 수 × k)개 anchor-neighbor 쌍에서 비율을 낸다.
  5. Random baseline = 대상 기사 중 서로 다른 두 기사를 무작위로 골랐을 때 prefix가 같을 확률
     = Σ_g n_g(n_g−1) / (N(N−1))  (g = 같은 prefix 그룹)

입력은 읽기만 하고, 출력 폴더가 이미 있으면 덮어쓰지 않고 멈춘다.

예)
  python evaluate_knn.py \
    --sid article_semantic_ids.parquet \
    --embeddings article_embeddings.npy \
    --split train --k 10 --output-dir knn_out/uni_train_k10
  (임베딩이 여러 조각이면 --embeddings part0.npy part1.npy ... 순서대로)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

LEVELS = {"same_c1": ["c1"], "same_c12": ["c1", "c2"], "same_c123": ["c1", "c2", "c3"]}


def load_embeddings(paths: list[str]) -> np.ndarray:
    parts = [np.load(p, mmap_mode="r") for p in paths]
    return np.concatenate([np.asarray(p, dtype=np.float32) for p in parts], axis=0)


def random_baseline(df: pl.DataFrame, cols: list[str]) -> float:
    n = df.height
    sizes = df.group_by(cols).len()["len"].to_numpy().astype(np.float64)
    return float((sizes * (sizes - 1)).sum() / (n * (n - 1)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sid", required=True, help="article_semantic_ids.parquet (article_id, embedding_row, split, c1, c2, c3)")
    ap.add_argument("--embeddings", required=True, nargs="+", help="E5 npy (조각이면 순서대로 여러 개)")
    ap.add_argument("--split", default="train", help="train / validation / all")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    out = Path(args.output_dir)
    if out.exists():
        raise SystemExit(f"출력 폴더가 이미 있습니다: {out} (덮어쓰지 않음)")

    sid = pl.read_parquet(args.sid)
    if args.split != "all":
        sid = sid.filter(pl.col("split") == args.split)
    sid = sid.sort("embedding_row")
    emb_all = load_embeddings(args.embeddings)
    rows = sid["embedding_row"].to_numpy()
    if rows.max() >= emb_all.shape[0]:
        raise SystemExit(f"embedding_row {rows.max()} >= 임베딩 행 수 {emb_all.shape[0]}")

    x = emb_all[rows]
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, k = x.shape[0], args.k
    codes = sid.select(["c1", "c2", "c3"]).to_numpy()

    neighbors = np.empty((n, k), dtype=np.int64)
    neighbor_cos = np.empty((n, k), dtype=np.float32)
    for s in range(0, n, args.batch):
        sim = x[s:s + args.batch] @ x.T
        idx = np.arange(s, min(s + args.batch, n))
        sim[np.arange(len(idx)), idx] = -np.inf  # 자기 자신 제외
        top = np.argpartition(-sim, k, axis=1)[:, :k]
        top_sim = np.take_along_axis(sim, top, axis=1)
        order = np.argsort(-top_sim, axis=1)
        neighbors[s:s + len(idx)] = np.take_along_axis(top, order, axis=1)
        neighbor_cos[s:s + len(idx)] = np.take_along_axis(top_sim, order, axis=1)

    anchor = codes[:, None, :]
    nb = codes[neighbors]
    match = {
        "same_c1": (nb[..., 0] == anchor[..., 0]),
        "same_c12": (nb[..., :2] == anchor[..., :2]).all(-1),
        "same_c123": (nb == anchor).all(-1),
    }

    summary = {
        "sid": args.sid, "embeddings": args.embeddings, "split": args.split, "k": k,
        "anchors": int(n), "anchor_neighbor_pairs": int(n * k),
        "mean_neighbor_cosine": float(neighbor_cos.mean()),
        "levels": {},
    }
    for name, cols in LEVELS.items():
        rate = float(match[name].mean())
        base = random_baseline(sid, cols)
        summary["levels"][name] = {
            "knn_rate": rate,
            "matched_pairs": int(match[name].sum()),
            "random_baseline": base,
            "lift": rate / base if base > 0 else None,
            "anchors_with_any_match": float(match[name].any(1).mean()),
        }

    out.mkdir(parents=True)
    (out / "knn_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    ids = sid["article_id"].to_numpy()
    pl.DataFrame({
        "anchor_article_id": np.repeat(ids, k),
        "rank": np.tile(np.arange(1, k + 1), n),
        "neighbor_article_id": ids[neighbors].reshape(-1),
        "cosine": neighbor_cos.reshape(-1),
        **{name: m.reshape(-1) for name, m in match.items()},
    }).write_parquet(out / "knn_pairs.parquet")

    print(f"anchors {n:,} × k {k} = {n * k:,} pairs · 평균 이웃 cosine {summary['mean_neighbor_cosine']:.4f}")
    print(f"{'level':<10} {'KNN':>9} {'random':>10} {'lift':>9}")
    for name, v in summary["levels"].items():
        print(f"{name:<10} {v['knn_rate']:>8.2%} {v['random_baseline']:>10.5%} {v['lift']:>8.2f}x")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
