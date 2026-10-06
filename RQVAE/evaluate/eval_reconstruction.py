from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from common import resolve_master_paths

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load_model(checkpoint_path: str | Path, device):
    from modules.rqvae import RqVae

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    if "model_config" not in checkpoint:
        raise KeyError("checkpoint에 model_config가 없습니다.")
    if "model" not in checkpoint:
        raise KeyError("checkpoint에 model state가 없습니다.")

    model = RqVae(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    for p in model.parameters():
        p.requires_grad_(False)

    # event-level C2 checkpoint: c2_mode / event_code_table (Train EventCode)
    c2_mode = checkpoint.get("c2_mode", "article")
    event_code_table = {
        int(k): int(v)
        for k, v in (checkpoint.get("event_code_table") or {}).items()
    }

    return model, c2_mode, event_code_table


@torch.inference_mode()
def compute_reconstruction_loss(
    model,
    master_path,
    embeddings_path,
    device,
    batch_size=512,
    gumbel_t=0.001,
    c2_mode="article",
    event_code_table=None,
):
    master = pd.read_parquet(master_path).reset_index(drop=True)
    embeddings = np.load(embeddings_path, mmap_mode="r")

    required = {"embedding_row", "model_category_id"}
    missing = required - set(master.columns)
    if missing:
        raise KeyError(f"{master_path}에 필요한 column이 없습니다: {sorted(missing)}")

    encoder_param = next(model.encoder.parameters())
    model_dtype = encoder_param.dtype

    total_loss = 0.0
    total_n = 0

    if c2_mode == "event":
        # 학습 중 validation과 같은 규칙:
        #   complete-event batch -> z(E) = mean h(a) -> EventCode -> 같은 event는 같은 q2
        #   event_code_table을 주면 그 event는 표의 code를 상속 (Validation의 기존 Train event)
        from data.event_batch_sampler import EventBatchSampler

        if "event_id" not in master.columns:
            raise KeyError(f"{master_path}에 event_id column이 없습니다.")

        batches = list(EventBatchSampler(
            event_ids=master["event_id"].to_numpy(),
            batch_size=batch_size,
            shuffle=False,
        ))
    else:
        batches = [
            np.arange(start, min(start + batch_size, len(master)))
            for start in range(0, len(master), batch_size)
        ]

    c2_by_event = {}

    for batch_index in batches:
        batch_df = master.iloc[batch_index]
        rows = batch_df["embedding_row"].astype(np.int64).to_numpy()

        x_np = np.asarray(embeddings[rows], dtype=np.float32)
        x = torch.from_numpy(x_np).to(device=device, dtype=model_dtype)

        category_ids = torch.tensor(
            batch_df["model_category_id"].astype(np.int64).to_numpy(),
            dtype=torch.long,
            device=device,
        )

        if c2_mode == "event":
            event_ids = torch.tensor(
                batch_df["event_id"].astype(np.int64).to_numpy(),
                dtype=torch.long,
                device=device,
            )
            override = None
            if event_code_table is not None:
                override = torch.tensor(
                    [event_code_table.get(int(e), -1) for e in event_ids.tolist()],
                    dtype=torch.long,
                    device=device,
                )

            output = model(
                x=x,
                category_ids=category_ids,
                gumbel_t=gumbel_t,
                event_ids=event_ids,
                event_code_override=override,
            )

            # 같은 event = 같은 c2 확인
            for e, c in zip(event_ids.tolist(), output.sem_ids[:, 1].tolist()):
                if c2_by_event.setdefault(e, c) != c:
                    raise AssertionError(f"event {e} received multiple c2")
        else:
            try:
                output = model(
                    x=x,
                    category_ids=category_ids,
                    gumbel_t=gumbel_t,
                )
            except TypeError:
                output = model(x, category_ids, gumbel_t)

        if not hasattr(output, "reconstruction_loss"):
            raise AttributeError(
                "model output에 reconstruction_loss가 없습니다. "
                "현재 modules/rqvae.py의 output field 이름을 확인해주세요."
            )

        rec = output.reconstruction_loss.detach().float()
        n = len(batch_df)

        if rec.ndim == 0:
            total_loss += rec.item() * n
        else:
            total_loss += rec.sum().item()

        total_n += n

    if c2_mode == "event":
        return total_loss / total_n, c2_by_event

    return total_loss / total_n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--gumbel-t", type=float, default=0.001)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    train_path, valid_path, embeddings_path = resolve_master_paths(data_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model, c2_mode, event_code_table = load_model(args.checkpoint, device)
    print("C2 mode:", c2_mode)

    common_args = dict(
        embeddings_path=embeddings_path,
        device=device,
        batch_size=args.batch_size,
        gumbel_t=args.gumbel_t,
    )

    if c2_mode != "event":
        train_loss = compute_reconstruction_loss(model, train_path, **common_args)
        valid_loss = compute_reconstruction_loss(model, valid_path, **common_args)

        print(f"Train Rec Loss : {train_loss:.6f}")
        print(f"Valid Rec Loss : {valid_loss:.6f}")
        return

    # Train: 현재 encoder의 in-batch z(E) -> EventCode (checkpoint 표와 일치해야 함)
    train_loss, train_codes = compute_reconstruction_loss(
        model, train_path, c2_mode="event", event_code_table=None, **common_args,
    )
    agreement = np.mean([
        event_code_table.get(e) == c for e, c in train_codes.items()
    ]) if event_code_table else float("nan")

    # Validation: 기존 Train event는 checkpoint EventCode 상속, 신규 event는 mean(h) -> Q2 nearest
    valid_loss, _ = compute_reconstruction_loss(
        model, valid_path, c2_mode="event",
        event_code_table=event_code_table, **common_args,
    )

    print(f"Train EventCode vs checkpoint table: {agreement:.6f}")
    print(f"Train Rec Loss : {train_loss:.6f}")
    print(f"Valid Rec Loss : {valid_loss:.6f}")

    # 참고: 기사 단위 c2 (q2 = Q2[argmin ||r1 - Q2||])로 계산한 값. Event-C2 모델의 평가값 아님
    article_train = compute_reconstruction_loss(model, train_path, **common_args)
    article_valid = compute_reconstruction_loss(model, valid_path, **common_args)
    print(f"(reference, article-level c2) Train Rec Loss : {article_train:.6f}")
    print(f"(reference, article-level c2) Valid Rec Loss : {article_valid:.6f}")


if __name__ == "__main__":
    main()
