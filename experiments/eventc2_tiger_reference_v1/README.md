# eventc2_tiger_reference_v1: 공개 TIGER 재현 모델을 수정 없이 EventC2 데이터에 적용

`reference/`는 [EdoardoBotta/RQ-VAE-Recommender](https://github.com/EdoardoBotta/RQ-VAE-Recommender) @ `957d32b` (MIT)의
`modules/model.py`, `data/schemas.py`, `modules/scheduler/` 원본 복사본이다 (byte 단위 동일 확인). 이 프로젝트 RQVAE 코드의 원본 저장소다.

## 공개 코드를 "그대로" 돌리지 못하는 부분과 처리

| 공개 코드 | 문제 | 처리 |
|---|---|---|
| `train_decoder.py`는 Amazon 데이터만 지원 (`Dataset currently not supported`) | EB-NeRD loader 없음 | 우리 loader (`train_tiger.py`와 같은 것)로 같은 batch 형식 `TokenizedSeqBatch`를 만든다 |
| 자체 RQ-VAE tokenizer로 SID 생성 | SID가 바뀜 (금지) | EventC2 SID를 [c1, c2, c3, c4(dedup)]로 그대로 넣는다 |
| level마다 같은 codebook 크기 (256) | 우리는 25 / 128 / 512 | `num_embeddings_per_hierarchy = 512` |
| 평가 = beam search Recall@K (전체 corpus) | 우리 평가는 1pos4neg | 후보 5개를 같은 head의 sum log P(c1..c3)로 채점 |

모델 클래스, loss(level CE 합), SEP token, level별 head, AdamW lr 1e-3 / wd 1e-4, batch 640,
InverseSquareRoot(warmup 10k), grad clip 없음, fp32, d384 / 6 head / FFN 1024 / 4층 (`configs/decoder_amazon.gin`),
history 20, user token 없음은 공개 코드 그대로다. 공개 구현에서 c4(dedup column)는 모델 입력·target에서 빠진다.

학습 길이와 선택은 V1과 같게: 최대 30 epoch, patience 5, validation Top-1 best.

## 실행 (AWS, `eventc2_tiger_generative_v1`이 끝난 뒤)

```bash
cd ~/minji/SID_Project && git pull origin claude/zealous-cerf-c1onwi
CUDA_VISIBLE_DEVICES=0 nohup bash experiments/eventc2_tiger_reference_v1/run_aws.sh \
  > data/output/experiments/eventc2_tiger_reference_v1.log 2>&1 &
tail -f data/output/experiments/eventc2_tiger_reference_v1.log
```

출력: `…/eventc2_tiger_reference_v1/seed42/`, 비교표 `…/eventc2_tiger_reference_v1/compare_v1_vs_reference.md`
