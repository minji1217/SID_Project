import numpy as np
import torch

from torch.utils.data import Sampler


# ============================================================
# Event-centric batch sampler
#
# event-level C2 학습에서는 같은 event의 기사가 한 학습 시점에
# 같은 EventCode(c2)를 공유해야 한다.
#
#   z(E) = mean{h(a) | a in E}
#
# 를 현재 encoder로 batch 안에서 계산하려면 event의 모든 기사가
# 같은 batch에 있어야 하므로, event를 쪼개지 않고 batch를 만든다.
# ============================================================

class EventBatchSampler(Sampler):

    def __init__(
        self,
        event_ids,
        batch_size: int,
        shuffle: bool,
        seed: int = 42,
    ) -> None:
        """
        event_ids:
            dataset index 순서의 event_id 배열
        batch_size:
            batch에 담을 최대 기사 수.
            event 하나가 batch_size보다 크면 그 event만으로 batch를 만든다.
        shuffle:
            True면 매 epoch마다 seed + epoch로 event 순서를 섞는다.
        """

        event_ids = np.asarray(event_ids).astype(np.int64)

        if batch_size <= 0:
            raise ValueError("batch_size must be > 0.")

        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        self.num_articles = len(event_ids)

        # event_id -> 기사 index 목록 (dataset 순서 유지)
        order = np.argsort(event_ids, kind="stable")
        sorted_ids = event_ids[order]
        boundaries = np.flatnonzero(np.diff(sorted_ids)) + 1

        self.event_ids = sorted_ids[
            np.concatenate([[0], boundaries])
        ] if len(sorted_ids) > 0 else sorted_ids
        self.event_members = (
            np.split(order, boundaries)
            if len(order) > 0
            else []
        )
        self.event_sizes = {
            int(event_id): len(members)
            for event_id, members in zip(
                self.event_ids,
                self.event_members,
            )
        }

        self._num_batches = len(self._build_batches(0))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _build_batches(self, epoch: int) -> list:
        event_order = np.arange(len(self.event_members))

        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed + epoch)
            event_order = torch.randperm(
                len(self.event_members),
                generator=generator,
            ).numpy()

        batches = []
        current = []

        for event_index in event_order:
            members = self.event_members[event_index]

            if (
                current
                and len(current) + len(members) > self.batch_size
            ):
                batches.append(current)
                current = []

            current.extend(members.tolist())

        if current:
            batches.append(current)

        # event가 정확히 한 batch에만 들어갔는지 확인
        covered = sum(len(batch) for batch in batches)

        if covered != self.num_articles:
            raise RuntimeError(
                "EventBatchSampler did not cover every article once. "
                f"articles={self.num_articles}, covered={covered}"
            )

        return batches

    def __iter__(self):
        batches = self._build_batches(self.epoch)
        self._num_batches = len(batches)
        return iter(batches)

    def __len__(self) -> int:
        return self._num_batches
