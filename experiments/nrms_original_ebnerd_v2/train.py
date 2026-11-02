"""
NRMS 학습/평가. 기사 입력 종류(--input_mode)만 바꿔서 같은 모델, 같은 데이터, 같은 5지선다 Top-1로 비교한다.

  title        제목 단어 NRMS (기본 baseline)
  e5           원본 e5 임베딩 768차원 (압축 안 한 상한선)
  sid          RQ-VAE 시맨틱 ID (c1,c2,c3)
  sid_shuffled 시맨틱 ID를 기사들 사이에 섞은 대조군 (그룹 크기 구조는 동일, 의미만 제거)

준비:
  python3 build_tables.py --uploads_dir <parquet 폴더> --out_dir .
사용:
  python3 train.py --input_mode sid --smoke_test
  python3 train.py --input_mode sid --epochs 5 --batch_size 64 --e5_path article_embeddings.npy
"""
import argparse
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from dataset import NRMSDataset, collate_fn
from model import NRMS

DIR = "."  # --tables_dir 로 바꿀 수 있음


def build_table(mode, args):
    if mode == "title":
        return torch.from_numpy(np.load(f"{DIR}/title_tokens.npy"))
    if mode == "e5":
        return torch.from_numpy(np.load(args.e5_path).astype(np.float32))
    if mode == "sid":
        return torch.from_numpy(np.load(f"{DIR}/sid_codes.npy"))
    if mode == "sid_shuffled":
        return torch.from_numpy(np.load(f"{DIR}/sid_shuffled.npy"))
    if mode in ("sid_c1", "sid_c12"):
        codes = np.load(f"{DIR}/sid_codes.npy").copy()
        # -1 로 두면 모델에서 +1 되어 0 = pad(영벡터)가 되므로 해당 코드는 쓰이지 않는다
        codes[:, 1:] = -1 if mode == "sid_c1" else codes[:, 1:]
        if mode == "sid_c12":
            codes[:, 2] = -1
        return torch.from_numpy(codes)
    raise ValueError(mode)


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for b in loader:
            scores = model(b["history_rows"].to(device), b["history_mask"].to(device), b["candidate_rows"].to(device))
            correct += (scores.argmax(-1).cpu() == b["label"]).sum().item()
            total += b["label"].size(0)
    return correct / total if total else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_mode", default="title", choices=["title", "e5", "sid", "sid_shuffled", "sid_c1", "sid_c12"])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--embed_dim", type=int, default=300)
    ap.add_argument("--num_heads", type=int, default=20)
    ap.add_argument("--smoke_test", action="store_true")
    ap.add_argument("--train_path", default="mind_1pos4neg/train_sequences_1pos4neg.parquet")
    ap.add_argument("--val_path", default="mind_1pos4neg/validation_sequences_1pos4neg.parquet")
    ap.add_argument("--e5_path", default="article_embeddings.npy")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tables_dir", default=".")
    ap.add_argument("--test_path", default=None, help="있으면 val 최고 epoch에서의 test Top-1도 출력")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    global DIR
    DIR = args.tables_dir
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device={device}  input_mode={args.input_mode}")

    table = build_table(args.input_mode, args)
    N = table.shape[0]
    vocab_size = len(pickle.load(open(f"{DIR}/vocab.pkl", "rb"))) if args.input_mode == "title" else 0

    id2row = f"{DIR}/article_id2row.pkl"
    train_ds = NRMSDataset(args.train_path, id2row, N, shuffle="random")
    val_ds = NRMSDataset(args.val_path, id2row, N, shuffle="fixed", seed=123)
    if args.smoke_test:
        train_ds, val_ds = Subset(train_ds, range(200)), Subset(val_ds, range(200))
        args.epochs = 1
        print("[train] SMOKE TEST: 200 samples, 1 epoch")

    test_loader = None
    if args.test_path:
        test_ds = NRMSDataset(args.test_path, id2row, N, shuffle="fixed", seed=123)
        if args.smoke_test:
            test_ds = Subset(test_ds, range(200))
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn,
                                 num_workers=0 if args.smoke_test else args.num_workers)

    nw = 0 if args.smoke_test else args.num_workers
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=nw)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=nw)

    model = NRMS(args.input_mode, table, vocab_size=vocab_size, embed_dim=args.embed_dim,
                 num_heads=args.num_heads).to(device)
    print(f"[train] 학습 파라미터 수 = {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()

    best, best_test = 0.0, None
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, tot = time.time(), 0.0
        for step, b in enumerate(train_loader):
            scores = model(b["history_rows"].to(device), b["history_mask"].to(device), b["candidate_rows"].to(device))
            loss = crit(scores, b["label"].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            if args.smoke_test and step >= 5:
                break
        acc = evaluate(model, val_loader, device)
        if acc > best:
            best = acc
            if test_loader is not None:
                best_test = evaluate(model, test_loader, device)
            if not args.smoke_test:
                torch.save(model.state_dict(), f"nrms_{args.input_mode}{args.tag}_best.pt")
        print(f"[epoch {ep}] loss={tot/(step+1):.4f}  val_top1={acc*100:.2f}%  ({time.time()-t0:.1f}s)")

    print(f"[train] DONE input_mode={args.input_mode}  best val Top-1 = {best*100:.2f}%  (무작위 기준 20%)"
          + (f"  | test Top-1 @best-val = {best_test*100:.2f}%" if best_test is not None else ""))
    print("[train] 참고 - 팀 V1: 27.06% (fixed weight 29.31%)")


if __name__ == "__main__":
    main()
