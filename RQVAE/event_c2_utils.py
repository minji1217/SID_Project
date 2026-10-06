import json
import math

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader

from data.event_batch_sampler import EventBatchSampler


# ============================================================
# event-level C2 학습용 assertion / snapshot / 통계
# ============================================================

def event_size_lookup(dataset) -> Dict[int, int]:
    return (
        dataset.article_master["event_id"]
        .astype(np.int64)
        .value_counts()
        .to_dict()
    )


def assert_complete_events(
    event_ids: torch.Tensor,
    event_sizes: Dict[int, int],
) -> None:
    """batch 안의 event가 모든 기사를 다 담고 있는지 (batch 간 분할 없음)."""

    unique_ids, counts = torch.unique(
        event_ids.detach().cpu(),
        return_counts=True,
    )

    for event_id, count in zip(
        unique_ids.tolist(),
        counts.tolist(),
    ):
        expected = event_sizes.get(int(event_id))

        if expected != count:
            raise AssertionError(
                "Event split across batches: "
                f"event_id={event_id}, in_batch={count}, "
                f"event_size={expected}"
            )


def assert_one_c2_per_event(
    event_ids: torch.Tensor,
    c2: torch.Tensor,
) -> None:
    """batch 안에서 같은 event의 c2 unique 수 = 1."""

    frame = pd.DataFrame({
        "event_id": event_ids.detach().cpu().numpy(),
        "c2": c2.detach().cpu().numpy(),
    })

    c2_per_event = frame.groupby("event_id")["c2"].nunique()

    if (c2_per_event != 1).any():
        bad = c2_per_event[c2_per_event != 1]
        raise AssertionError(
            "Same event received multiple c2 in one batch: "
            f"{bad.head(10).to_dict()}"
        )


def batch_event_codes(
    event_ids: torch.Tensor,
    c2: torch.Tensor,
) -> Dict[int, int]:
    """batch에서 event별로 배정된 c2 (assert_one_c2_per_event 이후 호출)."""

    return dict(zip(
        event_ids.detach().cpu().tolist(),
        c2.detach().cpu().tolist(),
    ))


def event_dataloader(
    dataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
    pin_memory: bool,
):
    sampler = EventBatchSampler(
        event_ids=dataset.article_master["event_id"].to_numpy(),
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
    )

    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return loader, sampler


def code_override_for_batch(
    event_ids: torch.Tensor,
    train_code_table: Optional[Dict[int, int]],
) -> Optional[torch.Tensor]:
    """
    Validation: 기존 Train event는 현재 시점 Train EventCode를 상속(code),
    Validation 신규 event는 -1 (현재 encoder mean(h) -> 현재 Q2 nearest).
    """

    if train_code_table is None:
        return None

    return torch.tensor(
        [
            train_code_table.get(int(event_id), -1)
            for event_id in event_ids.detach().cpu().tolist()
        ],
        dtype=torch.long,
        device=event_ids.device,
    )


@torch.no_grad()
def build_sid_snapshot(
    model,
    dataset,
    device,
    batch_size: int,
    train_code_table: Optional[Dict[int, int]] = None,
) -> pd.DataFrame:
    """
    현재 파라미터(eval 모드)로 dataset 전체의 (c1, c2, c3)와 EventCode를 계산한다.
    train_code_table을 주면 그 event는 해당 code를 상속한다 (Validation 규칙).
    """

    was_training = model.training
    model.eval()

    loader, _ = event_dataloader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        seed=0,
        num_workers=0,
        pin_memory=False,
    )

    article_ids = []
    event_ids_all = []
    sem_ids_all = []

    for batch in loader:
        x = batch["x"].to(device)
        category_ids = batch["category_id"].to(
            device=device,
            dtype=torch.long,
        )
        event_ids = torch.as_tensor(
            batch["event_id"],
            dtype=torch.long,
        ).to(device)

        output = model.get_semantic_ids(
            x=x,
            category_ids=category_ids,
            event_ids=event_ids,
            event_code_override=code_override_for_batch(
                event_ids,
                train_code_table,
            ),
        )

        batch_article_ids = batch["article_id"]
        if torch.is_tensor(batch_article_ids):
            batch_article_ids = batch_article_ids.tolist()
        article_ids.extend([str(a) for a in batch_article_ids])
        event_ids_all.extend(event_ids.cpu().tolist())
        sem_ids_all.append(output.sem_ids.cpu())

    if was_training:
        model.train()

    sem_ids = torch.cat(sem_ids_all).numpy()

    return pd.DataFrame({
        "article_id": article_ids,
        "event_id": event_ids_all,
        "c1": sem_ids[:, 0],
        "c2": sem_ids[:, 1],
        "c3": sem_ids[:, 2],
    })


def event_code_table(snapshot: pd.DataFrame) -> Dict[int, int]:
    first = snapshot.drop_duplicates("event_id")
    return dict(zip(
        first["event_id"].astype(int),
        first["c2"].astype(int),
    ))


def same_event_c2_consistency(snapshot: pd.DataFrame) -> Optional[float]:
    per_event = snapshot.groupby("event_id")["c2"].agg(["size", "nunique"])
    multi = per_event[per_event["size"] >= 2]

    if len(multi) == 0:
        return None

    return float((multi["nunique"] == 1).mean())


def sid_statistics(snapshot: pd.DataFrame) -> dict:
    unique_c123 = snapshot[["c1", "c2", "c3"]].drop_duplicates().shape[0]
    c4 = snapshot.groupby(["c1", "c2", "c3"], sort=False).cumcount()

    return {
        "articles": int(len(snapshot)),
        "unique_c123": int(unique_c123),
        "c123_collision_rate": float(1 - unique_c123 / len(snapshot)),
        "c123_collision_article_ratio": float(
            snapshot.duplicated(["c1", "c2", "c3"], keep=False).mean()
        ),
        "max_c4": int(c4.max()),
        "same_event_c2_consistency": same_event_c2_consistency(snapshot),
    }


def q2_code_usage(
    snapshot: pd.DataFrame,
    num_codes: int,
) -> dict:
    table = event_code_table(snapshot)
    event_codes = np.fromiter(table.values(), dtype=np.int64)
    article_counts = np.bincount(
        snapshot["c2"].to_numpy(),
        minlength=num_codes,
    )
    probs = article_counts[article_counts > 0] / article_counts.sum()

    return {
        "q2_codes_used_by_events": int(len(np.unique(event_codes))),
        "q2_codes_used_by_articles": int((article_counts > 0).sum()),
        "q2_dead_codes": int(num_codes - (article_counts > 0).sum()),
        "q2_article_entropy_bits": float(-(probs * np.log2(probs)).sum()),
        "q2_top10_article_share": float(
            np.sort(article_counts)[::-1][:10].sum() / article_counts.sum()
        ),
    }


def event_code_churn(
    previous: Optional[Dict[int, int]],
    current: Dict[int, int],
    event_sizes: Dict[int, int],
) -> dict:
    if previous is None:
        return {
            "event_code_churn": None,
            "event_code_churn_article_weighted": None,
        }

    common = [e for e in current if e in previous]
    changed = [e for e in common if previous[e] != current[e]]
    total_articles = sum(event_sizes.get(e, 0) for e in common)

    return {
        "event_code_churn": (
            len(changed) / len(common) if common else None
        ),
        "event_code_churn_article_weighted": (
            sum(event_sizes.get(e, 0) for e in changed) / total_articles
            if total_articles else None
        ),
    }


def in_epoch_vs_snapshot_agreement(
    assigned: Dict[int, int],
    snapshot_table: Dict[int, int],
) -> Optional[float]:
    """학습 step에서 배정된 code와 epoch 끝 snapshot code의 일치율."""

    common = [e for e in assigned if e in snapshot_table]

    if not common:
        return None

    return float(
        np.mean([assigned[e] == snapshot_table[e] for e in common])
    )


def assert_snapshot_consistency(snapshot: pd.DataFrame, name: str) -> None:
    consistency = same_event_c2_consistency(snapshot)

    if consistency is not None and not math.isclose(consistency, 1.0):
        raise AssertionError(
            f"{name} same-event C2 consistency is {consistency:.6f}, "
            "expected 1.0"
        )


class EventC2Logger:

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: dict) -> None:
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(record, ensure_ascii=False, default=float)
                + "\n"
            )
