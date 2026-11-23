# eventc2_tiger_generative_v1: 같은 EventC2 데이터에서 TIGER-style generative Transformer

목표: RQ-VAE·SID·후보는 그대로 두고 downstream objective만 바꿔, V1(Top-1 약 26%, AUC 약 0.58)의 병목이 Transformer objective인지 확인한다.
EventC2 SID / post-RQ-VAE sequence / 1pos4neg / shuffled validation·test는 읽기만 한다.

## Audit: 현재 V1 vs TIGER

| 항목 | 현재 V1 (`Transformer/modules/model.py`) | TIGER (논문 + 공개 재현 구현) | 이번 구현 |
|---|---|---|---|
| 모델 | T5 encoder + T5 decoder. EventC2 V1 run은 `run_transformer.py` FINAL_CONFIG: d256, 8 head, 2층, d_ff 1024, dropout 0 (base gin의 d384/4층은 이 값으로 덮어써짐) | T5 encoder-decoder (논문: 4층씩, 6 head × 64, MLP 1024, d128, dropout 0.1) | TIGER 설정 |
| history 입력 | 기사마다 c1 c2 c3 c4 (+SEP)를 시간순으로 펼침, history 50 | 기사 SID token을 시간순으로 펼침 + 맨 앞 user ID token (hashing 2,000), history 최대 20 | TIGER와 같게 (user token, history 20, SEP 없음) |
| token vocabulary | level마다 별도 embedding table | level offset을 둔 하나의 vocabulary (level별로 다른 token) | 하나의 vocabulary, level offset |
| decoder 입력 / target | [BOS, c1, c2] → c1, c2, c3 (c4 없음) | BOS부터 SID 전체를 autoregressive 생성 (collision token 포함) | [BOS, c1, c2, c3] → c1, c2, c3, c4 |
| 출력 head | level별 head 3개 | 하나의 softmax (재현 구현은 level별 head) | SID vocabulary 전체 하나의 softmax |
| loss | 후보 5개 점수(log P 합)에 listwise CE | 실제 다음 아이템 SID token의 cross-entropy (teacher forcing) | target token CE의 평균 (negative 사용 안 함) |
| optimizer | AdamW lr 5e-5, wd 0, batch 128 | 논문: lr 0.01 처음 10k step 상수 후 inverse sqrt decay, batch 256 (T5X 기본 Adafactor) | Adafactor, lr 0.01, 10k step 후 inverse sqrt, batch 256 |
| 추론 | 후보 5개 log-prob 합 | beam search로 corpus 전체에서 생성 (유효하지 않은 ID 제거) | 후보 5개를 같은 생성 likelihood로 채점 (1pos4neg 평가 유지) |

학습 데이터: V1은 1pos4neg train의 positive만 target으로 쓴다. TIGER는 negative가 필요 없으므로
**1pos4neg 이전** `post_rqvae/train_sequences.parquet`에서 `candidate_labels == 1`인 기사 전부를 target으로 쓴다
(1pos4neg에서 negative 부족 등으로 빠진 impression도 학습에 포함).
Validation/Test는 V1과 같은 1pos4neg 후보 5개를 쓴다.

V1의 실제 설정은 비교 리포트가 V1 `run_summary.json`의 `gin_config`(실제 학습에 쓰인 최종 binding)에서 읽어 표로 남긴다.

학습 예산은 V1과 같게 둔다: 최대 30 epoch, early stopping patience 5, validation Top-1 기준 best checkpoint, seed 42.
history가 빈 row는 V1(`drop_empty_history=True`)처럼 제외한다.

## 점수와 지표

```
score(a | H) = log P(c1|H) + log P(c2|H,c1) + log P(c3|H,c1,c2) + log P(c4|H,c1,c2,c3)
```

Top-1 / MRR / nDCG@5 / AUC는 `Transformer/evaluate/metrics.py`와 같은 정의 (스모크 데이터로 소수점까지 일치 확인).
Preference Loss = 후보 5개 score에 대한 cross-entropy, Positive Probability = 후보 5개 softmax에서 positive 확률 (`predict_sid.py`와 같음).

진단: level별 CE / token accuracy (positive target, teacher forcing), positive·negative 평균 log-prob와 gap,
c1 / c1+c2 / c1+c2+c3 / c1+..+c4 누적 점수의 Top-1·AUC·MRR·nDCG@5 (재학습 없음).

## 실행 (AWS, single GPU)

```bash
cd ~/minji/SID_Project && git pull origin claude/zealous-cerf-c1onwi
CUDA_VISIBLE_DEVICES=0 nohup bash experiments/eventc2_tiger_generative_v1/run_aws.sh \
  > data/output/experiments/eventc2_tiger_generative_v1.log 2>&1 &
tail -f data/output/experiments/eventc2_tiger_generative_v1.log
```

출력: `data/output/experiments/normalize_v2_uni_lu005_m05_eventc2_retrain_v1/eventc2_tiger_generative_v1/seed42/`
(`epoch_history.csv`, `checkpoint_best.pt`, `test_candidate_scores.parquet`, `test_metrics.json`, `run_summary.json`)
그리고 같은 폴더 위의 `compare_v1_vs_tiger.md` (V1 test score 파일이 있을 때).

## 파일

- `tiger_model.py`: 모델 (V1 코드와 독립, `transformers`의 T5Stack만 사용)
- `train_tiger.py`: 데이터 로드, 학습, Validation/Test 채점, 진단
- `compare_v1_vs_tiger.py`: 같은 test sample인지 확인 후 6개 지표 비교
- `run_aws.sh`: 위 세 단계를 순서대로 실행

## 결과와 결정 (2026-10-07): TIGER loss 적용 철회

같은 EventC2 test 122,670 sample (seed 42, 두 모델 모두 best epoch 1)

| 지표 | V1 ranker | TIGER-style |
|---|---|---|
| Top-1 | 0.2978 | 0.1677 |
| AUC | 0.6080 | 0.4694 |
| MRR | 0.5419 | 0.4303 |
| nDCG@5 | 0.6552 | 0.5695 |
| Preference Loss | 1.6526 | 5.8346 |
| Positive Probability | 0.2537 | 0.1700 |

TIGER 누적 level: c1 AUC 0.543 → c1+c2 0.534 → c1+c2+c3 0.494 → +c4 0.469.
level CE (test): c1 1.85 / c2 4.43 / c3 8.67 / c4 4.04. positive 평균 log-prob −18.99 < negative −18.58.

`diagnose_seen_in_train.py`: train에서 클릭된 고유 기사는 1,858개뿐이고,
후보 기사가 train target에 있었던 비율은 positive 7.8% / negative 21.3% (validation),
1.7% / 12.1% (test). 생성형 loss는 train에서 본 SID에 높은 확률을 주므로
새 기사가 클릭되는 뉴스 데이터에서는 c3 이하에서 순위가 뒤집힌다.

결정: 이 과제(같은 impression의 후보 5개 ranking)에는 TIGER 생성형 loss를 쓰지 않는다.
학습 목표는 V1 listwise를 유지한다. `eventc2_tiger_reference_v1`(공개 구현 그대로)은 실행하지 않고 보존만 한다.
코드와 결과는 기록으로 남긴다.
