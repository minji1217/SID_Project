"""실험 R: UNI 학습 코드 그대로, Q2 K-means 초기화 입력만 사건별 h(a) 평균 → r1(a) 평균으로 바꿔 학습한다.

    r1(a) = h(a) - Q1[c1(a)]    (모델의 get_c2_residual 과 같음. Q2 가 실제로 양자화하는 값)

UNI 코드(uniqueness-loss-experiment @ 127ae7e)의 RQVAE/ 폴더 안에서 실행한다. run_aws.sh 가 VARIANT=r1 일 때 부른다.

    cd <UNI_DIR>/RQVAE
    python <이 파일> <gin 파일>

train_rqvae.py 는 고치지 않는다. initialize_c2_codebook 이 부르는 encode_all_train_articles 만 r1 버전으로 바꿔 끼운다.
나머지(모델, 손실, uniqueness loss, early stopping, best_rec 저장, 평가)는 H(지금 ver2)와 같다.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

import train_rqvae as T  # noqa: E402
from modules.utils import parse_config  # noqa: E402

CALLED = {"n": 0}


@torch.no_grad()
def encode_all_train_articles_r1(model, train_dataset, device, batch_size=512, num_workers=0):
    """원래 함수와 같은 모양 (값, event_id) 을 돌려준다. 값만 h(a) 대신 r1(a)."""
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
    was_training = model.training
    model.eval()
    r1_list, event_id_list = [], []
    for batch in loader:
        x, category_ids, event_ids = T.unpack_batch(batch)
        r1 = model.get_c2_residual(x=x, category_ids=category_ids)
        r1_list.append(r1.detach().cpu())
        event_id_list.append(event_ids.detach().cpu())
    if was_training:
        model.train()
    CALLED["n"] += 1
    r1_all = torch.cat(r1_list, dim=0)
    print(f"[R] Q2 K-means 입력 = 사건별 r1(a) 평균 · 기사 {r1_all.shape[0]:,} · 차원 {r1_all.shape[1]}", flush=True)
    return r1_all, torch.cat(event_id_list, dim=0)


def main():
    T.encode_all_train_articles = encode_all_train_articles_r1
    parse_config()
    print("[R] 실험 R: Q2 초기화를 r1(a) = h(a) - Q1[c1] 사건 평균으로 (나머지는 UNI 와 같음)", flush=True)
    T.train()
    if CALLED["n"] != 1:
        raise SystemExit(f"[R] r1 초기화가 {CALLED['n']}번 불렸습니다 (1번이어야 함). pretrained_rqvae_path 로 재개하면 초기화를 건너뜁니다.")


if __name__ == "__main__":
    main()
