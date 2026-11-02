#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EB-NeRD controlled E5-vs-SID NRMS-style comparison.

Purpose
-------
Keep the recommendation model / data / loss identical and change only the
article representation:
  --representation e5 : original frozen E5 article embedding (768D)
  --representation sid: final RQ-VAE Semantic ID (c1,c2,c3)

Important
---------
This is an NRMS-style article-vector user encoder:
article vector -> multi-head self-attention over history -> attention pooling
-> user vector -> dot-product candidate ranking.

The MIND evaluation PDF only states that NRMS input was replaced by E5 vs SID;
it does not specify the exact internal SID vectorization. Therefore this script
is a controlled EB-NeRD implementation for the same research question, not a
claim of byte-for-byte reproduction of the MIND code.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    import polars as pl
except ImportError as e:
    raise SystemExit(
        "polars가 필요합니다. AWS venv에서 `pip install polars` 후 다시 실행하세요."
    ) from e


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


@dataclass
class ModelConfig:
    representation: str
    d_model: int = 256
    num_heads: int = 8
    sid_emb_dim: int = 128
    dropout: float = 0.0
    max_history: int = 50


class ArticleCatalog:
    """article_id -> E5 row / SID triple lookup."""

    def __init__(
        self,
        article_master_path: str,
        article_embeddings_path: str,
        semantic_ids_path: str,
    ) -> None:
        master = pl.read_parquet(article_master_path).select(
            ["article_id", "embedding_row"]
        )
        master = master.filter(pl.col("embedding_row").is_not_null())

        article_ids = master["article_id"].cast(pl.Int64).to_list()
        embedding_rows = master["embedding_row"].cast(pl.Int64).to_list()

        # contiguous internal index: 0 is PAD
        self.article_to_idx: dict[int, int] = {
            int(aid): i + 1 for i, aid in enumerate(article_ids)
        }
        self.num_articles = len(article_ids)

        raw_e5 = np.load(article_embeddings_path, mmap_mode="r")
        if raw_e5.ndim != 2:
            raise ValueError(f"E5 npy must be 2D, got {raw_e5.shape}")
        self.e5_dim = int(raw_e5.shape[1])

        e5 = np.zeros((self.num_articles + 1, self.e5_dim), dtype=np.float32)
        for i, row in enumerate(embedding_rows, start=1):
            if row < 0 or row >= raw_e5.shape[0]:
                raise IndexError(f"embedding_row out of range: {row}")
            e5[i] = raw_e5[row]
        self.e5 = torch.from_numpy(e5)

        sid_df = pl.read_parquet(semantic_ids_path)
        required = {"article_id", "c1", "c2", "c3"}
        missing = required - set(sid_df.columns)
        if missing:
            raise ValueError(
                f"semantic ids parquet missing columns: {sorted(missing)}; "
                f"found={sid_df.columns}"
            )

        sid_map = {
            int(r["article_id"]): (int(r["c1"]), int(r["c2"]), int(r["c3"]))
            for r in sid_df.select(["article_id", "c1", "c2", "c3"]).iter_rows(named=True)
        }

        # reserve 0 for padding, therefore actual code index = code + 1
        sid = np.zeros((self.num_articles + 1, 3), dtype=np.int64)
        sid_present = np.zeros(self.num_articles + 1, dtype=np.bool_)
        max_codes = [0, 0, 0]

        for aid, idx in self.article_to_idx.items():
            triple = sid_map.get(aid)
            if triple is None:
                continue
            vals = [int(x) + 1 for x in triple]
            sid[idx] = vals
            sid_present[idx] = True
            for j, v in enumerate(vals):
                max_codes[j] = max(max_codes[j], v)

        self.sid = torch.from_numpy(sid)
        self.sid_present = torch.from_numpy(sid_present)
        # +1 because nn.Embedding num_embeddings is exclusive upper bound.
        self.c1_vocab = max_codes[0] + 1
        self.c2_vocab = max_codes[1] + 1
        self.c3_vocab = max_codes[2] + 1

    def ids_to_indices(self, article_ids: list[int], require_sid: bool) -> list[int]:
        out = []
        for aid in article_ids:
            idx = self.article_to_idx.get(int(aid))
            if idx is None:
                raise KeyError(f"article_id={aid} not found in article_master")
            if require_sid and not bool(self.sid_present[idx]):
                raise KeyError(
                    f"article_id={aid} has no (c1,c2,c3) in article_semantic_ids"
                )
            out.append(idx)
        return out


class SequenceDataset(Dataset):
    """
    Uses only:
      - history_article_ids
      - candidate_article_ids
      - candidate_labels

    candidate position is NEVER assumed. Positive index is read from labels.
    """

    def __init__(
        self,
        parquet_path: str,
        catalog: ArticleCatalog,
        representation: str,
        max_history: int = 50,
    ) -> None:
        cols = ["history_article_ids", "candidate_article_ids", "candidate_labels"]
        df = pl.read_parquet(parquet_path, columns=cols)

        histories = df["history_article_ids"].to_list()
        candidates = df["candidate_article_ids"].to_list()
        labels = df["candidate_labels"].to_list()

        self.samples: list[tuple[list[int], list[int], int]] = []
        dropped_empty = 0

        require_sid = representation == "sid"

        for h, c, y in zip(histories, candidates, labels):
            h = [] if h is None else [int(x) for x in h if x is not None]
            c = [] if c is None else [int(x) for x in c if x is not None]
            y = [] if y is None else [int(x) for x in y]

            if not h:
                dropped_empty += 1
                continue
            if len(c) != 5 or len(y) != 5:
                raise ValueError(
                    f"1P4N row must have exactly 5 candidates/labels; "
                    f"got candidates={len(c)}, labels={len(y)}"
                )
            pos = [i for i, v in enumerate(y) if v == 1]
            if len(pos) != 1:
                raise ValueError(f"candidate_labels must contain exactly one positive: {y}")

            # most recent max_history items
            h = h[-max_history:]

            h_idx = catalog.ids_to_indices(h, require_sid=require_sid)
            c_idx = catalog.ids_to_indices(c, require_sid=require_sid)
            self.samples.append((h_idx, c_idx, pos[0]))

        print(
            f"[SequenceDataset] {parquet_path}\n"
            f"  original rows : {df.height:,}\n"
            f"  usable rows   : {len(self.samples):,}\n"
            f"  empty history : {dropped_empty:,}\n"
            f"  max_history   : {max_history}\n"
            f"  candidates    : 5 (1 positive + 4 negative)"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


def collate_batch(batch):
    histories, candidates, positives = zip(*batch)
    max_h = max(len(x) for x in histories)
    bsz = len(batch)

    history_idx = torch.zeros((bsz, max_h), dtype=torch.long)
    history_mask = torch.zeros((bsz, max_h), dtype=torch.bool)
    candidate_idx = torch.tensor(candidates, dtype=torch.long)
    positive_idx = torch.tensor(positives, dtype=torch.long)

    for i, h in enumerate(histories):
        n = len(h)
        history_idx[i, :n] = torch.tensor(h, dtype=torch.long)
        history_mask[i, :n] = True

    return {
        "history_idx": history_idx,
        "history_mask": history_mask,
        "candidate_idx": candidate_idx,
        "positive_idx": positive_idx,
    }


class AdditiveAttentionPooling(nn.Module):
    def __init__(self, d_model: int, hidden: int = 200):
        super().__init__()
        self.proj = nn.Linear(d_model, hidden)
        self.query = nn.Linear(hidden, 1, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x [B,H,D], mask [B,H] True=valid
        score = self.query(torch.tanh(self.proj(x))).squeeze(-1)
        score = score.masked_fill(~mask, -1e9)
        weight = torch.softmax(score, dim=-1)
        return torch.sum(x * weight.unsqueeze(-1), dim=1)


class ArticleEncoder(nn.Module):
    def __init__(
        self,
        cfg: ModelConfig,
        catalog: ArticleCatalog,
    ) -> None:
        super().__init__()
        self.representation = cfg.representation
        self.d_model = cfg.d_model

        if cfg.representation == "e5":
            # frozen source embedding, trainable projection only
            self.register_buffer("e5_table", catalog.e5, persistent=False)
            self.e5_proj = nn.Linear(catalog.e5_dim, cfg.d_model)
        elif cfg.representation == "sid":
            self.register_buffer("sid_table", catalog.sid, persistent=False)
            self.c1_emb = nn.Embedding(
                catalog.c1_vocab, cfg.sid_emb_dim, padding_idx=0
            )
            self.c2_emb = nn.Embedding(
                catalog.c2_vocab, cfg.sid_emb_dim, padding_idx=0
            )
            self.c3_emb = nn.Embedding(
                catalog.c3_vocab, cfg.sid_emb_dim, padding_idx=0
            )
            self.sid_proj = nn.Linear(cfg.sid_emb_dim * 3, cfg.d_model)
        else:
            raise ValueError(cfg.representation)

    def forward(self, article_idx: torch.Tensor) -> torch.Tensor:
        if self.representation == "e5":
            x = self.e5_table[article_idx]
            return self.e5_proj(x)

        sid = self.sid_table[article_idx]
        c1 = self.c1_emb(sid[..., 0])
        c2 = self.c2_emb(sid[..., 1])
        c3 = self.c3_emb(sid[..., 2])
        return self.sid_proj(torch.cat([c1, c2, c3], dim=-1))


class NRMSRepresentationModel(nn.Module):
    """
    Same recommendation backbone for both E5 and SID.
    Only ArticleEncoder changes.
    """

    def __init__(self, cfg: ModelConfig, catalog: ArticleCatalog):
        super().__init__()
        self.cfg = cfg
        self.article_encoder = ArticleEncoder(cfg, catalog)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=cfg.d_model,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(cfg.d_model)
        self.user_pool = AdditiveAttentionPooling(cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        history_idx: torch.Tensor,
        history_mask: torch.Tensor,
        candidate_idx: torch.Tensor,
    ) -> torch.Tensor:
        h = self.article_encoder(history_idx)  # [B,H,D]

        attn_out, _ = self.self_attn(
            h, h, h,
            key_padding_mask=~history_mask,
            need_weights=False,
        )
        h = self.attn_norm(h + self.dropout(attn_out))
        user = self.user_pool(h, history_mask)  # [B,D]

        cand = self.article_encoder(candidate_idx)  # [B,5,D]

        # same scorer for E5 and SID
        score = torch.einsum("bd,bkd->bk", user, cand) / math.sqrt(self.cfg.d_model)
        return score


@torch.no_grad()
def ranking_metrics(scores: torch.Tensor, positive_idx: torch.Tensor) -> dict[str, float]:
    # scores [B,5]
    order = torch.argsort(scores, dim=1, descending=True)
    pos = positive_idx.unsqueeze(1)
    rank = (order == pos).nonzero(as_tuple=False)[:, 1] + 1  # 1-based
    rank = rank.float()

    top1 = (rank == 1).float().mean().item()
    mrr = (1.0 / rank).mean().item()
    ndcg5 = (1.0 / torch.log2(rank + 1.0)).mean().item()

    pos_score = scores.gather(1, positive_idx.unsqueeze(1)).squeeze(1)
    mask = torch.ones_like(scores, dtype=torch.bool)
    mask.scatter_(1, positive_idx.unsqueeze(1), False)
    neg = scores[mask].view(scores.size(0), -1)

    greater = (pos_score.unsqueeze(1) > neg).float()
    tie = (pos_score.unsqueeze(1) == neg).float() * 0.5
    auc = (greater + tie).mean().item()

    return {"top1": top1, "mrr": mrr, "ndcg5": ndcg5, "auc": auc}


@torch.no_grad()
def evaluate_loader(model, loader, device) -> dict[str, float]:
    model.eval()
    total = 0
    sums = {"top1": 0.0, "mrr": 0.0, "ndcg5": 0.0, "auc": 0.0, "loss": 0.0}
    ce = nn.CrossEntropyLoss(reduction="sum")

    for batch in loader:
        history_idx = batch["history_idx"].to(device, non_blocking=True)
        history_mask = batch["history_mask"].to(device, non_blocking=True)
        candidate_idx = batch["candidate_idx"].to(device, non_blocking=True)
        positive_idx = batch["positive_idx"].to(device, non_blocking=True)

        scores = model(history_idx, history_mask, candidate_idx)
        bsz = scores.size(0)
        metrics = ranking_metrics(scores, positive_idx)

        for k in ("top1", "mrr", "ndcg5", "auc"):
            sums[k] += metrics[k] * bsz
        sums["loss"] += ce(scores, positive_idx).item()
        total += bsz

    return {k: v / total for k, v in sums.items()}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--representation", choices=["e5", "sid"], required=True)
    p.add_argument("--train", required=True)
    p.add_argument("--validation", required=True)
    p.add_argument("--article-master", required=True)
    p.add_argument("--article-embeddings", required=True)
    p.add_argument("--semantic-ids", required=True)
    p.add_argument("--output-dir", required=True)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-history", type=int, default=50)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--sid-emb-dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.0)

    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--learning-rate", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--num-workers", type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    catalog = ArticleCatalog(
        args.article_master,
        args.article_embeddings,
        args.semantic_ids,
    )
    print(
        f"[ArticleCatalog] articles={catalog.num_articles:,}, "
        f"E5 dim={catalog.e5_dim}, "
        f"SID vocab(c1/c2/c3)="
        f"{catalog.c1_vocab-1}/{catalog.c2_vocab-1}/{catalog.c3_vocab-1}"
    )

    cfg = ModelConfig(
        representation=args.representation,
        d_model=args.d_model,
        num_heads=args.num_heads,
        sid_emb_dim=args.sid_emb_dim,
        dropout=args.dropout,
        max_history=args.max_history,
    )

    train_ds = SequenceDataset(
        args.train, catalog, args.representation, args.max_history
    )
    val_ds = SequenceDataset(
        args.validation, catalog, args.representation, args.max_history
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_batch,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_batch,
        persistent_workers=args.num_workers > 0,
    )

    model = NRMSRepresentationModel(cfg, catalog).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    ce = nn.CrossEntropyLoss()

    best_top1 = -1.0
    best_mrr = -1.0
    bad_epochs = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_rows = 0

        for step, batch in enumerate(train_loader, start=1):
            history_idx = batch["history_idx"].to(device, non_blocking=True)
            history_mask = batch["history_mask"].to(device, non_blocking=True)
            candidate_idx = batch["candidate_idx"].to(device, non_blocking=True)
            positive_idx = batch["positive_idx"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            scores = model(history_idx, history_mask, candidate_idx)
            loss = ce(scores, positive_idx)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            bsz = scores.size(0)
            total_loss += loss.item() * bsz
            total_rows += bsz

            if step % 200 == 0:
                print(
                    f"epoch={epoch:02d} step={step:04d}/{len(train_loader)} "
                    f"train_loss={total_loss/total_rows:.5f}"
                )

        train_loss = total_loss / total_rows
        val = evaluate_loader(model, val_loader, device)

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            **{f"val_{k}": v for k, v in val.items()},
        }
        history.append(record)

        print(
            f"[Epoch {epoch:02d}] "
            f"train_loss={train_loss:.5f} | "
            f"val_top1={val['top1']:.4%} "
            f"MRR={val['mrr']:.5f} "
            f"nDCG@5={val['ndcg5']:.5f} "
            f"AUC={val['auc']:.5f} "
            f"loss={val['loss']:.5f}"
        )

        # Primary selection = Validation Top-1; MRR breaks exact ties.
        improved = (
            val["top1"] > best_top1 + 1e-12
            or (
                abs(val["top1"] - best_top1) <= 1e-12
                and val["mrr"] > best_mrr
            )
        )

        if improved:
            best_top1 = val["top1"]
            best_mrr = val["mrr"]
            bad_epochs = 0

            ckpt = {
                "model_state": model.state_dict(),
                "model_config": asdict(cfg),
                "seed": args.seed,
                "epoch": epoch,
                "validation_metrics": val,
                "train_args": vars(args),
                "catalog_meta": {
                    "e5_dim": catalog.e5_dim,
                    "c1_vocab": catalog.c1_vocab,
                    "c2_vocab": catalog.c2_vocab,
                    "c3_vocab": catalog.c3_vocab,
                },
            }
            torch.save(ckpt, outdir / "checkpoint_best.pt")
            print(f"  -> saved BEST checkpoint ({best_top1:.4%})")
        else:
            bad_epochs += 1

        with open(outdir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

        if bad_epochs >= args.patience:
            print(f"Early stopping: no Validation Top-1 improvement for {args.patience} epochs.")
            break

    summary = {
        "representation": args.representation,
        "seed": args.seed,
        "best_validation_top1": best_top1,
        "best_validation_mrr": best_mrr,
        "checkpoint": str(outdir / "checkpoint_best.pt"),
        "random_top1_baseline": 0.2,
        "note": "Test split was not used during training/model selection.",
    }
    with open(outdir / "train_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
