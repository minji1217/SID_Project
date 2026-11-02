"""
*_1pos4neg.parquet -> NRMS 학습/평가용 Dataset.

기사는 "행 번호(row)"로만 다룬다. 실제 입력(제목 토큰 / e5 벡터 / 시맨틱 ID)은 모델 안의 테이블에서
행 번호로 꺼내므로, 입력 종류를 바꿔도 이 Dataset은 그대로다.
테이블에 없는 기사는 pad 행(= 테이블 크기 N)으로 매핑된다.

*** 라벨 누수 방지 ***
저장된 파일에서 positive가 항상 candidate의 0번째에 고정돼 있어서(전수 확인), 순서를 섞지 않으면
"0번째 고르기"만으로 Top-1 100%가 나온다. 그래서
  - 학습(shuffle='random'): 매번 무작위로 섞음
  - 평가(shuffle='fixed') : 샘플 번호로 시드를 고정해 항상 같은 순서로 섞음(재현 가능)
"""
import pickle

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

MAX_HIST_LEN = 50  # 팀 V1 설정(History length=50)과 동일


class NRMSDataset(Dataset):
    def __init__(self, parquet_path, id2row_path, num_articles, max_hist_len=MAX_HIST_LEN,
                 shuffle="random", seed=123):
        df = pd.read_parquet(parquet_path, columns=["history_article_ids", "candidate_article_ids", "candidate_labels"])
        with open(id2row_path, "rb") as f:
            id2row = pickle.load(f)
        self.pad = num_articles
        self.max_hist_len = max_hist_len
        self.shuffle = shuffle
        self.seed = seed

        def to_rows(ids):
            return np.fromiter((id2row.get(int(x), self.pad) for x in ids), dtype=np.int64, count=len(ids))

        self.hist = [to_rows(list(h)[-max_hist_len:]) for h in df["history_article_ids"]]
        self.cand = [to_rows(list(c)) for c in df["candidate_article_ids"]]
        self.labels = [np.asarray(l) for l in df["candidate_labels"]]
        n_c = sum(len(c) for c in self.cand)
        n_miss = sum(int((c == self.pad).sum()) for c in self.cand)
        print(f"[dataset] {parquet_path}: {len(self.cand)} samples, candidate 조회 실패 {n_miss}/{n_c}")

    def __len__(self):
        return len(self.cand)

    def __getitem__(self, idx):
        cand, lab = self.cand[idx], self.labels[idx]
        k = len(cand)
        if self.shuffle == "random":
            order = torch.randperm(k).numpy()
        elif self.shuffle == "fixed":
            order = np.random.default_rng([self.seed, idx]).permutation(k)
        else:
            order = np.arange(k)
        cand, lab = cand[order], lab[order]

        hist = np.full(self.max_hist_len, self.pad, dtype=np.int64)
        mask = np.zeros(self.max_hist_len, dtype=np.float32)
        h = self.hist[idx]
        hist[:len(h)] = h
        mask[:len(h)] = 1.0

        return {
            "history_rows": torch.from_numpy(hist),
            "history_mask": torch.from_numpy(mask),
            "candidate_rows": torch.from_numpy(cand.astype(np.int64)),
            "label": torch.tensor(int(np.argmax(lab)), dtype=torch.long),
        }


def collate_fn(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}
