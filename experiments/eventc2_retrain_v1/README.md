# eventc2_retrain_v1: EventC2 RQ-VAE SID로 Transformer 재학습, UNI baseline과 비교

RQ-VAE SID만 UNI(lu0.05, m0.5) → EventC2(같은 설정 + c2_mode event)로 바꾸고 downstream 조건은 V1과 같게 둔다.

| 항목 | UNI baseline | EventC2 |
|---|---|---|
| SID | UNI checkpoint (sha256 dcb276f8…) | EventC2 `checkpoint_best_rec.pt` (epoch 210), `--train_c2_policy event` |
| 1pos4neg / shuffle / split | `normalize_v2_uni_lu005_m05_candidate_1pos4neg_shuffled_v2` (기존 그대로) | 새 SID로 처음부터 다시 생성 (같은 코드, seed 42) |
| Transformer | 기존 seed42 V1 run (그대로) | V1 최종 config, seed 42 |
| c4_vocab_size | 28 | **32** (새 SID의 max c4가 31. 튜닝이 아니라 필요한 변경) |

## 순서

1. `aws_preflight.py` (읽기 전용): UNI baseline run과 shuffled_v2 provenance, post-rqvae 입력 존재 여부
2. `run_pipeline.sh`: SID → post-rqvae → 1pos4neg → shuffle → split → Transformer → 비교
   - UNI baseline 파일과 run 폴더는 읽기만 한다. 출력은 `data/output/experiments/normalize_v2_uni_lu005_m05_eventc2_retrain_v1/`
   - baseline run에 `test_candidate_scores.parquet`이 없으면 `predict_uni_baseline.py`로 추론만 다시 해서
     `reports/uni_baseline_inference/`에 둔다 (baseline `test_metrics.json`과 일치 확인)

## 보고 항목 (`reports/uni_vs_eventc2/`)

- Test 6개 지표 (Top-1, MRR, nDCG@5, AUC, Preference Loss, Positive Probability)
  - 각자 전체 Test / 공통 impression / 공통 + candidate article_id 구성 동일
- Test impression: 공통, UNI-only, EventC2-only, 공통 중 candidate 구성 동일 수/비율
- split별 rows / impressions (train, validation half, test)
- 1pos4neg 단계별 감소량 ((positive, negative) 쌍 기준)
  - P-N: positive와 c123이 같아 제거된 negative
  - N-N: negative끼리 c123 중복으로 제거된 negative
  - unique negative < 4로 drop된 positive
  - UNI는 `--stats-only`로 parquet을 쓰지 않고 다시 집계 (baseline rows와 일치 확인)
