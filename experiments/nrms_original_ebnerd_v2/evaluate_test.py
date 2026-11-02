"""train.py가 저장한 best checkpoint(nrms_{mode}{tag}_best.pt)로 Top-1 / MRR / nDCG@5 / AUC를 계산한다.

train.py는 Top-1만 출력하므로, 보고용 지표를 같은 checkpoint·같은 데이터 순서(dataset.py의 fixed seed 123 셔플)로 다시 낸다.
학습하지 않는다. Top-1은 train.py의 evaluate와 같이 argmax로 계산하므로 train.py 로그 값과 같아야 한다.

    python evaluate_test.py --input_mode sid --tag _s0 --tables_dir <tables> --e5_path <npy> \
        --val_path <validation half> --test_path <test> --out metrics_sid_s0.json
"""
import argparse
import json
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import NRMSDataset, collate_fn
from model import NRMS
import train as T


@torch.no_grad()
def scores_of(model, loader, device):
    model.eval()
    S, Y = [], []
    for b in loader:
        S.append(model(b["history_rows"].to(device), b["history_mask"].to(device), b["candidate_rows"].to(device)).float().cpu())
        Y.append(b["label"])
    return torch.cat(S).numpy(), torch.cat(Y).numpy()


def metrics(scores, pos):
    n = len(pos)
    ps = scores[np.arange(n), pos][:, None]
    greater = (scores > ps).sum(1)
    ties = (scores == ps).sum(1) - 1
    rank = 1 + greater  # train.py evaluate와 같은 argmax 기준이 아니라 순위 지표용 (동점은 아래 AUC에서 0.5)
    k = scores.shape[1] - 1
    return {
        "samples": int(n),
        "top1_accuracy": float((scores.argmax(1) == pos).mean()),
        "mrr": float((1.0 / rank).mean()),
        "ndcg@5": float((1.0 / np.log2(rank + 1)).mean()),
        "auc": float(((k - greater - ties) + 0.5 * ties).mean() / k),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_mode", required=True, choices=["title", "e5", "sid", "sid_shuffled", "sid_c1", "sid_c12"])
    ap.add_argument("--tag", default="")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--tables_dir", default=".")
    ap.add_argument("--e5_path", default="article_embeddings.npy")
    ap.add_argument("--val_path", required=True)
    ap.add_argument("--test_path", required=True)
    ap.add_argument("--embed_dim", type=int, default=300)
    ap.add_argument("--num_heads", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    T.DIR = a.tables_dir
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    table = T.build_table(a.input_mode, a)
    N = table.shape[0]
    vocab_size = len(pickle.load(open(f"{a.tables_dir}/vocab.pkl", "rb"))) if a.input_mode == "title" else 0
    model = NRMS(a.input_mode, table, vocab_size=vocab_size, embed_dim=a.embed_dim, num_heads=a.num_heads).to(device)
    ckpt = a.checkpoint or f"nrms_{a.input_mode}{a.tag}_best.pt"
    model.load_state_dict(torch.load(ckpt, map_location=device))

    id2row = f"{a.tables_dir}/article_id2row.pkl"
    out = {"input_mode": a.input_mode, "tag": a.tag, "checkpoint": ckpt}
    for name, path in (("validation", a.val_path), ("test", a.test_path)):
        ds = NRMSDataset(path, id2row, N, shuffle="fixed", seed=123)
        dl = DataLoader(ds, batch_size=a.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)
        s, y = scores_of(model, dl, device)
        out[name] = metrics(s, y)
        m = out[name]
        print(f"[{a.input_mode}{a.tag}] {name:10s} Top-1 {m['top1_accuracy']:.4%}  MRR {m['mrr']:.4f}  "
              f"nDCG@5 {m['ndcg@5']:.4f}  AUC {m['auc']:.4f}  ({m['samples']:,})")
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
