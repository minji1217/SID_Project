"""TIGER-style generative Transformer 학습 + Validation/Test 후보 채점 (EventC2 SID, seed 42)

    python experiments/eventc2_tiger_generative_v1/train_tiger.py \
        --train <EXP>/candidate_1pos4neg_shuffled/train_sequences_1pos4neg.parquet \
        --val   <EXP>/transformer_datasets/validation_sequences_1pos4neg_half.parquet \
        --test  <EXP>/transformer_datasets/test_sequences_1pos4neg.parquet \
        --out-dir <EXP>/eventc2_tiger_generative_v1/seed42

학습: 각 row의 target(실제 클릭 기사) SID를 teacher forcing으로 생성 (후보 negative는 학습에 쓰지 않음)
채점: score(a|H) = sum_k log P(a_k | H, a_<k)  (k = c1..c4), 후보 5개 비교
지표: Top-1 / MRR / nDCG@5 / AUC / Preference Loss / Positive Probability
     (Transformer/evaluate/metrics.py, predict_sid.py와 같은 정의)
진단: level별 CE / token accuracy, positive·negative 평균 log-prob와 gap,
     c1 / c1+c2 / c1+c2+c3 / c1+..+c4 누적 점수의 지표
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tiger_model import NUM_LEVELS, TigerGenerative  # noqa: E402

LEVELS = ["c1", "c2", "c3", "c4"]
DEFAULT_LEVEL_SIZES = [25, 128, 512]   # c1, c2, c3 (c4는 데이터 max + 1)


def log(msg: str) -> None:
    print(msg, flush=True)


# ============================================================ data
def read_truncated(path: Path, cols: list, max_history: int, limit: int | None) -> pl.DataFrame:
    """history list는 읽으면서 바로 최근 max_history개로 자른다 (긴 history 전체를 메모리에 두지 않음)."""
    lf = pl.scan_parquet(path).select(cols)
    if limit:
        lf = lf.head(limit)
    lf = lf.with_columns([pl.col(f"history_{l}").list.tail(max_history) for l in LEVELS])
    return lf.collect()


def pad_history(df: pl.DataFrame, max_history: int) -> tuple[np.ndarray, np.ndarray]:
    n = df.height
    lengths = df["history_c1"].list.len().to_numpy().astype(np.int64)
    hist = np.zeros((n, max_history, NUM_LEVELS), dtype=np.int64)
    rows = np.repeat(np.arange(n), lengths)
    pos = np.arange(lengths.sum()) - np.repeat(np.cumsum(lengths) - lengths, lengths)
    for k, l in enumerate(LEVELS):
        hist[rows, pos, k] = df[f"history_{l}"].explode().to_numpy()
    mask = np.arange(max_history)[None, :] < lengths[:, None]
    return hist, mask


def load_split(path: Path, max_history: int, limit: int | None, need_candidates: bool) -> dict:
    cols = ["impression_id", "user_id", "impression_time"]
    cols += [f"history_{l}" for l in LEVELS] + [f"target_{l}" for l in LEVELS]
    if need_candidates:
        cols += ["candidate_article_ids", "candidate_labels"] + [f"candidate_{l}" for l in LEVELS]
    df = read_truncated(path, cols, max_history, limit)            # 최근 max_history개, 시간순 유지
    n_raw = df.height
    # V1(NewsSequenceDataset drop_empty_history=True)과 같이 history가 빈 row는 제외
    df = df.filter(pl.col("history_c1").list.len() > 0)
    n = df.height
    hist, mask = pad_history(df, max_history)

    target = np.stack([df[f"target_{l}"].list.first().to_numpy() for l in LEVELS], axis=1).astype(np.int64)
    out = {
        "n_raw": n_raw, "n": n,
        "history_sids": torch.from_numpy(hist),
        "history_mask": torch.from_numpy(mask),
        "user_bucket": torch.from_numpy((df["user_id"].to_numpy().astype(np.int64) % 2000)),
        "target_sids": torch.from_numpy(target),
        "impression_id": df["impression_id"].to_numpy(),
        "user_id": df["user_id"].to_numpy(),
        "impression_time": df["impression_time"].to_list(),
    }
    if need_candidates:
        cand = np.stack([np.stack(df[f"candidate_{l}"].to_list()) for l in LEVELS], axis=-1).astype(np.int64)
        labels = np.stack(df["candidate_labels"].to_list()).astype(np.int64)
        if cand.shape[1] != 5 or not (labels.sum(1) == 1).all():
            raise ValueError(f"{path}: 후보 5개 / positive 1개가 아닙니다.")
        out["candidate_sids"] = torch.from_numpy(cand)
        out["candidate_labels"] = labels
        out["candidate_article_ids"] = np.stack(df["candidate_article_ids"].to_list())
    return out


def load_train_targets(path: Path, max_history: int, limit: int | None) -> dict:
    """post-RQ-VAE train_sequences.parquet (1pos4neg 이전)에서 학습 예시를 만든다.

    impression의 candidate 중 label == 1인 기사 하나하나가 target (1pos4neg와 같은 positive 정의).
    negative 개수나 SID 중복 때문에 1pos4neg에서 빠진 impression도 그대로 학습에 쓴다.
    """
    cols = ["impression_id", "user_id", "candidate_labels"]
    cols += [f"history_{l}" for l in LEVELS] + [f"candidate_{l}" for l in LEVELS]
    df = read_truncated(path, cols, max_history, limit)             # history를 먼저 자른 뒤 positive마다 펼친다
    n_impressions = df.height
    df = (df.explode(["candidate_labels"] + [f"candidate_{l}" for l in LEVELS])
            .filter(pl.col("candidate_labels") == 1))
    n_positive = df.height
    df = df.filter(pl.col("history_c1").list.len() > 0)      # V1과 같이 history가 빈 예시 제외
    n = df.height
    hist, mask = pad_history(df, max_history)
    target = np.stack([df[f"candidate_{l}"].to_numpy() for l in LEVELS], axis=1).astype(np.int64)
    return {
        "n_raw": n_positive, "n": n, "n_impressions": n_impressions,
        "history_sids": torch.from_numpy(hist),
        "history_mask": torch.from_numpy(mask),
        "user_bucket": torch.from_numpy(df["user_id"].to_numpy().astype(np.int64) % 2000),
        "target_sids": torch.from_numpy(target),
    }


def batch_of(data: dict, idx, device, keys) -> dict:
    return {k: data[k][idx].to(device, non_blocking=True) for k in keys}


# ============================================================ metrics
def ranking_metrics(scores: np.ndarray, labels: np.ndarray) -> dict:
    """scores/labels [N,5]. evaluate/metrics.py + predict_sid.py와 같은 정의."""
    n = scores.shape[0]
    pos = labels.argmax(1)
    order = np.argsort(-scores, axis=1, kind="stable")
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.tile(np.arange(1, 6), (n, 1)), axis=1)
    pos_rank = ranks[np.arange(n), pos]
    pos_score = scores[np.arange(n), pos][:, None]
    neg = labels == 0
    wins = ((pos_score > scores) & neg).sum(1)
    ties = ((pos_score == scores) & neg).sum(1)
    shifted = scores - scores.max(1, keepdims=True)
    log_p = shifted - np.log(np.exp(shifted).sum(1, keepdims=True))
    pos_log_p = log_p[np.arange(n), pos]
    return {
        "num_samples": int(n),
        "top1_accuracy": float((scores.argmax(1) == pos).mean()),
        "mrr": float((1.0 / pos_rank).mean()),
        "ndcg@5": float((1.0 / np.log2(pos_rank + 1)).mean()),
        "auc": float(((wins + 0.5 * ties) / neg.sum(1)).mean()),
        "preference_loss": float((-pos_log_p).mean()),
        "positive_probability": float(np.exp(pos_log_p).mean()),
    }


@torch.no_grad()
def evaluate(model, data: dict, device, batch_size: int, amp: bool) -> tuple[dict, np.ndarray, np.ndarray]:
    model.eval()
    keys = ["history_sids", "history_mask", "user_bucket", "candidate_sids"]
    lps, corrects = [], []
    for s in range(0, data["n"], batch_size):
        idx = slice(s, s + batch_size)
        b = batch_of(data, idx, device, keys)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            lp, correct = model.candidate_level_log_probs(b)
        lps.append(lp.float().cpu().numpy())
        corrects.append(correct.cpu().numpy())
    lp = np.concatenate(lps)                     # [N,5,4]
    correct = np.concatenate(corrects)           # [N,5,4]
    labels = data["candidate_labels"]
    pos_mask = labels == 1

    cum = np.cumsum(lp, axis=-1)
    result = ranking_metrics(cum[..., -1], labels)
    pos_lp = lp[pos_mask]                        # [N,4]
    neg_lp = lp[~pos_mask]                       # [4N,4]
    result["level_ce"] = {l: float(-pos_lp[:, k].mean()) for k, l in enumerate(LEVELS)}
    result["level_token_accuracy"] = {l: float(correct[pos_mask][:, k].mean()) for k, l in enumerate(LEVELS)}
    result["target_ce_mean"] = float(-pos_lp.mean())
    result["positive_mean_log_prob"] = float(pos_lp.sum(1).mean())
    result["negative_mean_log_prob"] = float(neg_lp.sum(1).mean())
    result["positive_negative_score_gap"] = result["positive_mean_log_prob"] - result["negative_mean_log_prob"]
    result["cumulative_levels"] = {
        "+".join(LEVELS[: k + 1]): {m: v for m, v in ranking_metrics(cum[..., k], labels).items()
                                   if m in ("top1_accuracy", "auc", "mrr", "ndcg@5")}
        for k in range(NUM_LEVELS)
    }
    return result, lp, cum[..., -1]


def fmt(per_level: dict) -> str:
    return " ".join("%.3f" % per_level[l] for l in LEVELS)


def lr_at(step: int, base: float, warmup: int) -> float:
    """TIGER: 처음 warmup step은 상수, 그 뒤 inverse square root decay"""
    return base if step < warmup else base * math.sqrt(warmup / step)


# ============================================================ main
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=Path, required=True,
                   help="post-RQ-VAE train_sequences.parquet (1pos4neg 이전)")
    p.add_argument("--val", type=Path, required=True)
    p.add_argument("--test", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--eval-batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=10000)
    p.add_argument("--max-history", type=int, default=20)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--num-heads", type=int, default=6)
    p.add_argument("--d-kv", type=int, default=64)
    p.add_argument("--d-ff", type=int, default=1024)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--progress-every", type=int, default=50)
    p.add_argument("--limit", type=int, default=None, help="smoke test용 row 제한")
    args = p.parse_args()

    out = args.out_dir
    if (out / "run_summary.json").exists():
        raise SystemExit(f"이미 학습된 run입니다. 덮어쓰지 않습니다: {out}")
    out.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    log(f"device {device}  amp(bf16) {amp}")

    log("데이터 로드 ...")
    train = load_train_targets(args.train, args.max_history, args.limit)
    log(f"  train            impression {train['n_impressions']:,} -> positive target {train['n_raw']:,}")
    val = load_split(args.val, args.max_history, args.limit, need_candidates=True)
    test = load_split(args.test, args.max_history, args.limit, need_candidates=True)
    for name, d in (("train", train), ("validation_half", val), ("test", test)):
        log(f"  {name:16s} rows {d['n_raw']:,} -> history 있는 row {d['n']:,}")

    max_code = [max(int(d[k][..., i].max()) for d in (train, val, test)
                    for k in ("history_sids", "target_sids", "candidate_sids") if k in d)
                for i in range(NUM_LEVELS)]
    level_sizes = [max(DEFAULT_LEVEL_SIZES[i], max_code[i] + 1) for i in range(3)] + [max_code[3] + 1]
    log(f"  level vocabulary c1..c4 = {level_sizes}")

    model = TigerGenerative(level_sizes, d_model=args.d_model, num_heads=args.num_heads, d_kv=args.d_kv,
                            d_ff=args.d_ff, num_layers=args.num_layers, dropout_rate=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"  parameters {n_params:,}")

    from transformers.optimization import Adafactor
    opt = Adafactor(model.parameters(), lr=args.lr, scale_parameter=False, relative_step=False, warmup_init=False)

    config = {**{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
              "level_sizes": level_sizes, "num_parameters": n_params, "optimizer": "Adafactor",
              "selection_metric": "val_top1_accuracy",
              "train_source": "post-RQ-VAE train_sequences (candidate_labels == 1 -> target)",
              "train_impressions": train["n_impressions"],
              "rows": {k: {"raw": d["n_raw"], "used": d["n"]} for k, d in
                       (("train", train), ("validation_half", val), ("test", test))}}
    (out / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False))

    history_path = out / "epoch_history.csv"
    fields = ["epoch", "train_loss", "train_ce_c1", "train_ce_c2", "train_ce_c3", "train_ce_c4",
              "val_top1_accuracy", "val_mrr", "val_ndcg@5", "val_auc", "val_preference_loss",
              "val_positive_probability", "val_target_ce", "lr", "seconds"]
    with open(history_path, "w", newline="") as f:
        csv.writer(f).writerow(fields)

    def record(metrics: dict, prefix: str) -> dict:
        return {f"{prefix}_{k}": v for k, v in metrics.items() if not isinstance(v, dict)}

    log("\n학습 전 Validation ...")
    v0, _, _ = evaluate(model, val, device, args.eval_batch_size, amp)
    log(f"  [epoch 0] val Top-1 {v0['top1_accuracy']:.4%}  AUC {v0['auc']:.4f}  target CE {v0['target_ce_mean']:.4f}")

    keys = ["history_sids", "history_mask", "user_bucket", "target_sids"]
    step, best, best_epoch, bad, epochs_run = 0, -1.0, 0, 0, 0
    n_batches = math.ceil(train["n"] / args.batch_size)
    gen = torch.Generator().manual_seed(args.seed)
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        perm = torch.randperm(train["n"], generator=gen)
        loss_sum, ce_sum, seen = 0.0, torch.zeros(NUM_LEVELS), 0
        for bi in range(n_batches):
            idx = perm[bi * args.batch_size:(bi + 1) * args.batch_size]
            b = batch_of(train, idx, device, keys)
            step += 1
            for g in opt.param_groups:
                g["lr"] = lr_at(step, args.lr, args.warmup_steps)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                o = model(b)
            opt.zero_grad(set_to_none=True)
            o["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            bs = idx.numel()
            loss_sum += float(o["loss"]) * bs; ce_sum += o["level_ce"].float().cpu() * bs; seen += bs
            if (bi + 1) % args.progress_every == 0 or bi + 1 == n_batches:
                el = time.time() - t0
                log(f"  [Train {epoch}/{args.epochs}] | {bi + 1:,}/{n_batches:,} batch | loss {loss_sum / seen:.4f} | "
                    f"CE c1..c4 {' '.join(f'{x:.3f}' for x in (ce_sum / seen).tolist())} | "
                    f"경과 {el:.0f}s | 남은 {el / (bi + 1) * (n_batches - bi - 1):.0f}s")
        epochs_run = epoch

        v, _, _ = evaluate(model, val, device, args.eval_batch_size, amp)
        secs = time.time() - t0
        tr_ce = (ce_sum / seen).tolist()
        log(f"Epoch {epoch} | train loss {loss_sum / seen:.4f} | val Top-1 {v['top1_accuracy']:.4%} | "
            f"MRR {v['mrr']:.4f} | nDCG@5 {v['ndcg@5']:.4f} | AUC {v['auc']:.4f} | "
            f"Pref {v['preference_loss']:.4f} | PosProb {v['positive_probability']:.4f} | "
            f"level CE {fmt(v['level_ce'])} | level acc {fmt(v['level_token_accuracy'])} | {secs:.0f}s")
        log("  cumulative " + " | ".join(f"{k}: Top-1 {m['top1_accuracy']:.4f} AUC {m['auc']:.4f}"
                                          for k, m in v["cumulative_levels"].items()))
        with open(history_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, loss_sum / seen, *tr_ce, v["top1_accuracy"], v["mrr"], v["ndcg@5"],
                                    v["auc"], v["preference_loss"], v["positive_probability"],
                                    v["target_ce_mean"], lr_at(step, args.lr, args.warmup_steps), secs])

        if v["top1_accuracy"] > best:
            best, best_epoch, bad = v["top1_accuracy"], epoch, 0
            torch.save({"model": model.state_dict(), "config": config, "epoch": epoch, "val": v},
                       out / "checkpoint_best.pt")
            (out / "best_validation_metrics.json").write_text(json.dumps(v, indent=2))
            log(f"  ✓ Best checkpoint updated (val Top-1 {best:.4%})")
        else:
            bad += 1
            if bad >= args.patience:
                log(f"Early stopping (patience {args.patience}) at epoch {epoch}")
                break

    log("\nbest checkpoint로 Test 평가 ...")
    ck = torch.load(out / "checkpoint_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    vbest, _, _ = evaluate(model, val, device, args.eval_batch_size, amp)
    t, lp, score = evaluate(model, test, device, args.eval_batch_size, amp)

    n = test["n"]
    probs = np.exp(score - score.max(1, keepdims=True)); probs /= probs.sum(1, keepdims=True)
    rows = {
        "sample_index": np.repeat(np.arange(n), 5),
        "impression_id": np.repeat(test["impression_id"], 5),
        "user_id": np.repeat(test["user_id"], 5),
        "article_id": test["candidate_article_ids"].reshape(-1),
        "label": test["candidate_labels"].reshape(-1).astype(float),
        "candidate_score": score.reshape(-1),
        "candidate_probability": probs.reshape(-1),
    }
    for k, l in enumerate(LEVELS):
        rows[l] = test["candidate_sids"][..., k].numpy().reshape(-1)
        rows[f"{l}_log_prob"] = lp[..., k].reshape(-1)
    pd.DataFrame(rows).to_parquet(out / "test_candidate_scores.parquet", index=False)
    (out / "test_metrics.json").write_text(json.dumps(t, indent=2))

    summary = {"config": config, "best_epoch": best_epoch, "epochs_run": epochs_run,
               "validation_before_training": record(v0, "val"),
               "validation_best": vbest, "test": t}
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    log(f"\n=== Test (best epoch {best_epoch}) ===")
    for m in ("top1_accuracy", "mrr", "ndcg@5", "auc", "preference_loss", "positive_probability"):
        log(f"  {m:22s} {t[m]:.6f}")
    log("  level CE        " + " ".join(f"{l} {t['level_ce'][l]:.4f}" for l in LEVELS))
    log("  level token acc " + " ".join(f"{l} {t['level_token_accuracy'][l]:.4f}" for l in LEVELS))
    log(f"  pos/neg mean log-prob {t['positive_mean_log_prob']:.4f} / {t['negative_mean_log_prob']:.4f}  "
        f"gap {t['positive_negative_score_gap']:.4f}")
    for k, m in t["cumulative_levels"].items():
        log(f"  cumulative {k:12s} Top-1 {m['top1_accuracy']:.4f}  AUC {m['auc']:.4f}  MRR {m['mrr']:.4f}")
    log(f"saved -> {out}")


if __name__ == "__main__":
    main()
