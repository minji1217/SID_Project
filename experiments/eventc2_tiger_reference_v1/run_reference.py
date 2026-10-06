"""공개 TIGER 재현 구현의 모델을 수정 없이 EventC2 데이터에 적용 (seed 42)

reference/ = EdoardoBotta/RQ-VAE-Recommender @ 957d32b (MIT) 의
  modules/model.py (EncoderDecoderRetrievalModel), data/schemas.py, modules/scheduler/inv_sqrt.py 원본 복사.
이 프로젝트 RQVAE 코드의 원본 저장소이기도 하다.

그대로 둔 것 (공개 구현 / configs/decoder_amazon.gin)
  - 모델 클래스, forward(loss = level별 CE 합), SEP token, per-level head, level offset embedding
  - t5_d_model 384, heads 6, d_ff 1024, layers 4, should_add_sep_token True, user token 없음(num_user_bins None)
  - AdamW lr 1e-3, weight_decay 1e-4, batch 640, InverseSquareRootScheduler(warmup 10000), grad clip 없음, fp32
  - max history 20 (Amazon 설정), SID = RQ 3 level + dedup 1 column. dedup column(c4)은 모델이 입력·target에서 뺀다

우리 데이터에 맞춘 것 (모델 코드 밖, adapter)
  - 공개 구현의 RQ-VAE tokenizer 대신 EventC2 SID를 그대로 [c1,c2,c3,c4(dedup)] token으로 넣음
  - num_embeddings_per_hierarchy = 512 (공개 구현은 level마다 같은 codebook 크기. 우리 최대 level = c3 512)
  - 학습 예시: post-RQ-VAE train sequence의 positive 전부 (train_tiger.py와 같은 loader)
  - 평가: 공개 구현은 beam search Recall@K. 여기서는 기존 1pos4neg 후보 5개를
    sum_{h=1..3} log P(c_h | H, c_<h) (같은 decoder_mlp head, level별 log_softmax)로 채점
  - 학습 길이와 checkpoint 선택은 V1과 같게: 최대 30 epoch, patience 5, validation Top-1
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
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "reference"))
sys.path.insert(1, str(HERE.parent / "eventc2_tiger_generative_v1"))

from data.schemas import TokenizedSeqBatch                       # noqa: E402  (공개 구현 원본)
from modules.model import EncoderDecoderRetrievalModel           # noqa: E402  (공개 구현 원본)
from modules.scheduler.inv_sqrt import InverseSquareRootScheduler  # noqa: E402  (공개 구현 원본)
from train_tiger import load_split, load_train_targets, ranking_metrics  # noqa: E402

N_LAYERS = 3                 # RQ level (c1,c2,c3). c4 = dedup column
CODEBOOK = 512
LEVELS = ["c1", "c2", "c3"]


def log(msg: str) -> None:
    print(msg, flush=True)


def to_batch(d: dict, idx, device, target_key="target_sids") -> TokenizedSeqBatch:
    hist = d["history_sids"][idx].to(device)                     # [B,H,4]
    mask = d["history_mask"][idx].to(device)                     # [B,H]
    B, H, _ = hist.shape
    sem_ids = hist.reshape(B, H * 4)
    seq_mask = mask.unsqueeze(-1).expand(-1, -1, 4).reshape(B, H * 4)
    fut = d[target_key][idx].to(device) if target_key else torch.zeros(B, 4, dtype=torch.long, device=device)
    return TokenizedSeqBatch(
        user_ids=d["user_bucket"][idx].to(device).unsqueeze(1),
        sem_ids=sem_ids, sem_ids_fut=fut, seq_mask=seq_mask,
        token_type_ids=torch.zeros_like(sem_ids), token_type_ids_fut=torch.zeros_like(fut),
    )


@torch.no_grad()
def candidate_log_probs(model, d: dict, idx, device):
    """후보 [B,5,4] -> level별 log P [B,5,3], argmax 여부 [B,5,3] (공개 구현 forward와 같은 경로)"""
    batch = to_batch(d, idx, device, target_key=None)
    sem_ids_dim = model.num_hierarchies + 1
    from modules.model import _strip_dedup_col
    input_ids = _strip_dedup_col(batch.sem_ids, sem_ids_dim, model.num_hierarchies)
    attn = _strip_dedup_col(batch.seq_mask.long(), sem_ids_dim, model.num_hierarchies)
    enc, enc_mask = model.encoder_forward_pass(attention_mask=attn, input_ids=input_ids, user_id=batch.user_ids)
    cand = d["candidate_sids"][idx].to(device)[..., :N_LAYERS]   # [B,5,3]
    B, C, _ = cand.shape
    enc_r = enc.unsqueeze(1).expand(-1, C, -1, -1).reshape(B * C, enc.shape[1], enc.shape[2])
    mask_r = enc_mask.unsqueeze(1).expand(-1, C, -1).reshape(B * C, enc_mask.shape[1])
    flat = cand.reshape(B * C, N_LAYERS)
    dec = model.decoder_forward_pass(future_ids=flat, encoder_output=enc_r,
                                     attention_mask_for_encoder=mask_r, use_cache=False)[:, :-1]
    lps, correct = [], []
    for h in range(N_LAYERS):
        lp = F.log_softmax(model.decoder_mlp[h](dec[:, h]).float(), dim=-1)
        lps.append(lp.gather(-1, flat[:, h:h + 1]).squeeze(-1))
        correct.append(lp.argmax(-1) == flat[:, h])
    return (torch.stack(lps, -1).reshape(B, C, N_LAYERS).cpu().numpy(),
            torch.stack(correct, -1).reshape(B, C, N_LAYERS).cpu().numpy())


def evaluate(model, d: dict, device, batch_size: int):
    model.eval()
    lps, cors = [], []
    for s in range(0, d["n"], batch_size):
        lp, cor = candidate_log_probs(model, d, slice(s, s + batch_size), device)
        lps.append(lp); cors.append(cor)
    lp, cor = np.concatenate(lps), np.concatenate(cors)
    labels = d["candidate_labels"]
    pos = labels == 1
    cum = np.cumsum(lp, -1)
    r = ranking_metrics(cum[..., -1], labels)
    r["level_ce"] = {l: float(-lp[pos][:, k].mean()) for k, l in enumerate(LEVELS)}
    r["level_token_accuracy"] = {l: float(cor[pos][:, k].mean()) for k, l in enumerate(LEVELS)}
    r["positive_mean_log_prob"] = float(lp[pos].sum(1).mean())
    r["negative_mean_log_prob"] = float(lp[~pos].sum(1).mean())
    r["positive_negative_score_gap"] = r["positive_mean_log_prob"] - r["negative_mean_log_prob"]
    r["cumulative_levels"] = {"+".join(LEVELS[:k + 1]): {m: v for m, v in ranking_metrics(cum[..., k], labels).items()
                                                        if m in ("top1_accuracy", "auc", "mrr", "ndcg@5")}
                              for k in range(N_LAYERS)}
    return r, lp, cum[..., -1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--val", type=Path, required=True)
    p.add_argument("--test", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=640)
    p.add_argument("--eval-batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-steps", type=int, default=10000)
    p.add_argument("--max-history", type=int, default=20)
    p.add_argument("--progress-every", type=int, default=50)
    p.add_argument("--limit", type=int, default=None)
    a = p.parse_args()

    out = a.out_dir
    if (out / "run_summary.json").exists():
        raise SystemExit(f"이미 학습된 run입니다. 덮어쓰지 않습니다: {out}")
    out.mkdir(parents=True, exist_ok=True)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device {device} (fp32, 공개 구현 기본 amp=False)")

    log("데이터 로드 ...")
    train = load_train_targets(a.train, a.max_history, a.limit)
    val = load_split(a.val, a.max_history, a.limit, need_candidates=True)
    test = load_split(a.test, a.max_history, a.limit, need_candidates=True)
    log(f"  train impression {train['n_impressions']:,} -> target {train['n']:,} | val {val['n']:,} | test {test['n']:,}")
    for d in (train, val, test):
        for k in ("history_sids", "target_sids", "candidate_sids"):
            if k in d and int(d[k][..., :N_LAYERS].max()) >= CODEBOOK:
                raise ValueError("c1..c3 code가 512 이상입니다.")

    corpus = torch.unique(torch.cat([train["target_sids"][:, :N_LAYERS],
                                     test["candidate_sids"].reshape(-1, 4)[:, :N_LAYERS]]), dim=0)
    model = EncoderDecoderRetrievalModel(
        codebooks=corpus, num_hierarchies=N_LAYERS, num_embeddings_per_hierarchy=CODEBOOK,
        t5_d_model=384, t5_num_heads=6, t5_d_ff=1024, t5_num_layers=4,
        top_k_for_generation=10, should_add_sep_token=True, num_user_bins=None,
    ).to(device)
    n_params = sum(x.numel() for x in model.parameters())
    log(f"  parameters {n_params:,}")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    sched = InverseSquareRootScheduler(optimizer=opt, warmup_steps=a.warmup_steps)

    config = {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
              "reference": "EdoardoBotta/RQ-VAE-Recommender@957d32b modules/model.py (unchanged)",
              "t5_d_model": 384, "t5_num_heads": 6, "t5_d_ff": 1024, "t5_num_layers": 4,
              "should_add_sep_token": True, "num_user_bins": None, "num_hierarchies": N_LAYERS,
              "num_embeddings_per_hierarchy": CODEBOOK, "num_parameters": n_params,
              "selection_metric": "val_top1_accuracy",
              # compare_v1_vs_tiger.py 설정 표용 별칭
              "d_model": 384, "num_heads": 6, "num_layers": 4, "d_ff": 1024,
              "dropout": "0.1 (T5Config 기본)", "max_history": a.max_history,
              "level_sizes": [CODEBOOK, CODEBOOK, CODEBOOK, "dedup(미사용)"]}
    (out / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False))
    hist_path = out / "epoch_history.csv"
    with open(hist_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "ce_c1", "ce_c2", "ce_c3", "val_top1", "val_mrr",
                                "val_ndcg5", "val_auc", "val_pref_loss", "val_pos_prob", "lr", "seconds"])

    v0, _, _ = evaluate(model, val, device, a.eval_batch_size)
    log(f"[epoch 0] val Top-1 {v0['top1_accuracy']:.4%}  AUC {v0['auc']:.4f}")

    keys_n = train["n"]; n_batches = math.ceil(keys_n / a.batch_size)
    gen = torch.Generator().manual_seed(a.seed)
    best, best_epoch, bad, epochs_run = -1.0, 0, 0, 0
    for epoch in range(1, a.epochs + 1):
        model.train(); t0 = time.time()
        perm = torch.randperm(keys_n, generator=gen)
        loss_sum, ce_sum, seen = 0.0, torch.zeros(N_LAYERS), 0
        for bi in range(n_batches):
            idx = perm[bi * a.batch_size:(bi + 1) * a.batch_size]
            o = model(to_batch(train, idx, device))
            opt.zero_grad()
            o.loss.backward()
            opt.step(); sched.step()
            bs = idx.numel(); loss_sum += float(o.loss) * bs; ce_sum += o.loss_d.float().cpu() * bs; seen += bs
            if (bi + 1) % a.progress_every == 0 or bi + 1 == n_batches:
                el = time.time() - t0
                log(f"  [Train {epoch}/{a.epochs}] | {bi + 1:,}/{n_batches:,} batch | loss(sum) {loss_sum / seen:.4f} | "
                    f"CE c1..c3 {' '.join('%.3f' % x for x in (ce_sum / seen).tolist())} | "
                    f"경과 {el:.0f}s | 남은 {el / (bi + 1) * (n_batches - bi - 1):.0f}s")
        epochs_run = epoch
        v, _, _ = evaluate(model, val, device, a.eval_batch_size)
        secs = time.time() - t0
        log(f"Epoch {epoch} | train loss {loss_sum / seen:.4f} | val Top-1 {v['top1_accuracy']:.4%} | MRR {v['mrr']:.4f} | "
            f"nDCG@5 {v['ndcg@5']:.4f} | AUC {v['auc']:.4f} | Pref {v['preference_loss']:.4f} | "
            f"PosProb {v['positive_probability']:.4f} | level CE {' '.join('%.3f' % v['level_ce'][l] for l in LEVELS)} | "
            f"level acc {' '.join('%.3f' % v['level_token_accuracy'][l] for l in LEVELS)} | {secs:.0f}s")
        log("  cumulative " + " | ".join(f"{k}: Top-1 {m['top1_accuracy']:.4f} AUC {m['auc']:.4f}"
                                          for k, m in v["cumulative_levels"].items()))
        with open(hist_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, loss_sum / seen, *(ce_sum / seen).tolist(), v["top1_accuracy"], v["mrr"],
                                    v["ndcg@5"], v["auc"], v["preference_loss"], v["positive_probability"],
                                    opt.param_groups[0]["lr"], secs])
        if v["top1_accuracy"] > best:
            best, best_epoch, bad = v["top1_accuracy"], epoch, 0
            torch.save({"model": model.state_dict(), "config": config, "epoch": epoch}, out / "checkpoint_best.pt")
            log(f"  ✓ Best checkpoint updated (val Top-1 {best:.4%})")
        else:
            bad += 1
            if bad >= a.patience:
                log(f"Early stopping (patience {a.patience}) at epoch {epoch}")
                break

    log("\nbest checkpoint로 Test 평가 ...")
    model.load_state_dict(torch.load(out / "checkpoint_best.pt", map_location=device, weights_only=False)["model"])
    vbest, _, _ = evaluate(model, val, device, a.eval_batch_size)
    t, lp, score = evaluate(model, test, device, a.eval_batch_size)
    n = test["n"]
    probs = np.exp(score - score.max(1, keepdims=True)); probs /= probs.sum(1, keepdims=True)
    rows = {"sample_index": np.repeat(np.arange(n), 5), "impression_id": np.repeat(test["impression_id"], 5),
            "user_id": np.repeat(test["user_id"], 5), "article_id": test["candidate_article_ids"].reshape(-1),
            "label": test["candidate_labels"].reshape(-1).astype(float),
            "candidate_score": score.reshape(-1), "candidate_probability": probs.reshape(-1)}
    for k, l in enumerate(LEVELS):
        rows[f"{l}_log_prob"] = lp[..., k].reshape(-1)
    pd.DataFrame(rows).to_parquet(out / "test_candidate_scores.parquet", index=False)
    (out / "test_metrics.json").write_text(json.dumps(t, indent=2))
    (out / "run_summary.json").write_text(json.dumps({"config": config, "best_epoch": best_epoch,
                                                      "epochs_run": epochs_run, "validation_best": vbest,
                                                      "test": t}, indent=2, ensure_ascii=False))
    log(f"\n=== Test (best epoch {best_epoch}) ===")
    for m in ("top1_accuracy", "mrr", "ndcg@5", "auc", "preference_loss", "positive_probability"):
        log(f"  {m:22s} {t[m]:.6f}")
    log("  level CE        " + " ".join(f"{l} {t['level_ce'][l]:.4f}" for l in LEVELS))
    log("  level token acc " + " ".join(f"{l} {t['level_token_accuracy'][l]:.4f}" for l in LEVELS))
    for k, m in t["cumulative_levels"].items():
        log(f"  cumulative {k:9s} Top-1 {m['top1_accuracy']:.4f}  AUC {m['auc']:.4f}")
    log(f"saved -> {out}")


if __name__ == "__main__":
    main()
