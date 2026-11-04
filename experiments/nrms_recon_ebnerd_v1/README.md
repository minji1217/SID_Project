# NRMS + RQ-VAE recon 입력 (교수님 제안)

> NRMS의 뉴스 입력을 랜덤 초기화된 임베딩 대신 **RQ-VAE 출력(recon)**으로 넣어 돌려 보라.

`nrms_original_ebnerd_v2`(정민교 NRMS 코드, 원 논문 구조)의 `e5` 모드를 **코드 수정 없이** 쓴다.
기사별 768차원 표만 `article_embeddings.npy`(원본 E5) 대신 `recon_embeddings.npy`로 바꾼다.

| 모드 | 기사 입력 | 출처 |
|---|---|---|
| E5 | 원본 E5 (RQ-VAE 입력) | v2 |
| **recon** | **decoder(Q1[c1] + Q2[c2] + Q3[c3])**, UNI best_rec(epoch 180) | **이 폴더** |
| SID | SID 번호로 만든 새 랜덤 번호표 (RQ-VAE 벡터 안 씀) | v2 |

- SID는 v2 `tables/sid_codes.npy`(= SID 모드가 쓴 코드)를 그대로 쓴다. 같은 (c1,c2,c3)이면 recon도 같다(서로 다른 벡터 10,952개).
- 데이터 · 설정 · seed는 v2와 같다: UNI 1P4N train / validation half / test, 5 epoch, batch 64, seed 42 · 123 · 2026.
- 결과는 `data/output/experiments/nrms_recon_ebnerd_v1/`에만 쓴다. 원본과 v2 tables는 읽기만 한다.

## 로컬 검증 (2026-11-04)

- `build_recon_table.py` 결과가 정량평가 3(step6)의 `reconstructed_embeddings.npy`와 12,860행 모두 **차이 0.0**
- 자기복원 cosine 평균 0.9275 (정량평가 3 UNI 값과 같음)
- `train.py` / `evaluate_test.py` e5 모드 + recon 표로 smoke 학습 · 평가 끝까지 실행됨

## AWS 실행

UNI checkpoint(sha256 `dcb276f8…`, `checkpoint_best_rec.pt`)가 AWS에 있어야 한다. 위치 찾기:

```bash
find ~ -name "checkpoint_best_rec*.pt" 2>/dev/null | xargs -r sha256sum | grep dcb276f8
```

```bash
cd ~/minji/SID_Project && CKPT=<위 경로> CUDA_VISIBLE_DEVICES=0 \
  nohup bash experiments/nrms_recon_ebnerd_v1/run_aws.sh > data/output/experiments/nrms_recon_ebnerd_v1.log 2>&1 &
tail -f data/output/experiments/nrms_recon_ebnerd_v1.log
```

- RQ-VAE 코드는 `gin`, `huggingface_hub`가 필요하다. NRMS python에 없으면 `RQVAE_PYTHON=<RQ-VAE 학습 때 쓴 python>`을 함께 지정한다.
- 예상 시간: recon 표 1분 이내 + seed당 약 4분(v2 E5 기준 epoch 약 36초 × 5 + 평가) → 총 15분 안팎
- 로그에서 `candidate 조회 실패 0/...`와 `자기복원 cosine 평균 0.9275`를 확인한다.
