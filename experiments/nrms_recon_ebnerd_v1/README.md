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
- 로그에서 `candidate 조회 실패 0/...`와 자기복원 cosine을 확인한다.

### AWS 실행 기록 (2026-11-04)

- E5: `experiment/entity-linking-event:data/output/model_inputs/article_embeddings.npy`를 `~/minji/e5_input/`에 꺼내 사용(sha256 `89f8046d…`). `RAW_DIR=~/minji/e5_input`
- 로그의 자기복원 cosine 0.9256, 고유 (c1,c2,c3) 10,771은 로컬(0.9275, 10,952)과 다르다. 원인: v2 `sid_codes.npy`는 1P4N 파일에 나온 기사에서만 SID를 모으므로, **1P4N에 한 번도 안 나오는 288개 기사**는 코드가 (0,0,0)으로 남는다.
  - 288개를 빼면 고유 SID 10,770 · cosine **0.9275**로 로컬과 같다.
  - 이 288개는 학습 · 평가에 쓰이지 않으므로 결과에 영향이 없다 (조회 실패 0).

## 결과 (AWS 실행 완료, test 122,668행, seed 42 / 123 / 2026 평균)

| 모드 | Val Top-1 | Test Top-1 (seed 표준편차) | MRR | nDCG@5 | AUC |
|---|---|---|---|---|---|
| e5 (v2) | 28.82% | 29.92% (1.02) | 0.5430 | 0.6560 | 0.6092 |
| **recon** | **29.46%** | **29.99% (0.62)** | 0.5442 | 0.6569 | 0.6115 |
| sid (v2) | 25.92% | 26.09% (0.10) | 0.5107 | 0.6313 | 0.5711 |
| sid_shuffled (v2) | 20.99% | 20.71% (0.40) | 0.4647 | 0.5959 | 0.5128 |

- seed별 test Top-1: recon 42 = 29.77 / 123 = 29.51 / 2026 = 30.69
- best epoch: recon 123 = 5, 2026 = 5 (seed 42 로그 미확인)
- recon − SID = **+3.90%p**, recon − E5 = +0.07%p (seed 편차보다 작음 → E5와 동등)
- 해석: 같은 SID 정보(서로 다른 벡터 10,952개)라도 RQ-VAE 코드북 · decoder의 의미를 거치면 원본 E5와 같은 추천 성능.
  SID 모드의 차이는 SID를 이름표로만 쓰고 번호표를 랜덤에서 학습하기 때문으로 보인다.
- 주의: 세 모드 모두 최대 5 epoch이고 E5 · recon은 5 epoch에서도 오르는 중이다. recon은 고정 입력(Linear만 학습)이며, 학습 중 recon 자체를 업데이트하는 변형은 하지 않았다.
