# uni_tiger_reference_v1 결과 (seed 42, AWS 2026-11-03, 채팅에 붙여 넣은 로그에서 옮김)

공개 TIGER 재현 모델(EdoardoBotta/RQ-VAE-Recommender@957d32b, 수정 없음) + UNI SID.
학습 입력: 1P4N 이전 UNI train (`normalize_v2_uni_lu005_m05_candidate_train_v1/post_rqvae/train_sequences.parquet`,
`check_uni_sid.py`로 SID·`~/shared` 1P4N과 일치 확인, 1P4N train impression 232,084개 모두 포함).
train impression 232,292 -> target 233,161 | val 122,549 | test 122,668 | parameters 15,738,240.

## Test (best epoch 1 / 6 epoch, patience 5로 종료). V1과 같은 test sample 122,668개 확인

| 지표 | V1 ranker | 공개 TIGER | 차이 |
|---|---|---|---|
| Top-1 | 0.270666 | 0.173582 | -0.097083 |
| MRR | 0.518065 | 0.439600 | -0.078465 |
| nDCG@5 | 0.636875 | 0.576881 | -0.059994 |
| AUC | 0.578739 | 0.486598 | -0.092141 |
| Preference loss | 1.744624 | 4.888992 | +3.144368 |
| Positive prob | 0.243484 | 0.174126 | -0.069358 |

무작위 기준: Top-1 0.20, AUC 0.50. (참고: 우리 TIGER-style + EventC2 SID는 Top-1 16.77% / AUC 0.469, SID·test가 달라 직접 비교 불가)

## level별 (test)

| level | CE | token acc | 누적 점수 Top-1 | 누적 AUC |
|---|---|---|---|---|
| c1 | 1.8779 | 0.3021 | 0.2299 | 0.5297 |
| c2 | 4.9350 | 0.0623 | 0.2372 (c1+c2) | 0.5322 |
| c3 | 10.3152 | 0.0050 | 0.1736 (c1+c2+c3) | 0.4866 |

positive 평균 log-prob -17.1281, negative -16.7928, gap -0.3353 (오답 쪽이 더 높음).
c3 CE 10.3 > ln 512 = 6.24: 균등 분포보다도 나쁨 (train에서 본 c3에 확신을 몰아줌).

## validation 추이

| epoch | train loss | val Top-1 | val AUC | val c3 CE |
|---|---|---|---|---|
| 0 | - | 20.00% | 0.5020 | - |
| 1 | 7.3589 | 16.38% | 0.4580 | 9.541 |
| 2 | 6.7141 | 15.53% | 0.4515 | 10.075 |
| 3 | 6.6316 | 15.48% | 0.4460 | 10.481 |
| 4 | 6.6620 | 15.37% | 0.4423 | 10.588 |
| 5 | 6.6586 | 15.66% | 0.4459 | 10.752 |
| 6 | 6.6650 | 15.00% | 0.4412 | 10.903 |

## 정답·오답이 train에서 클릭된 적 있는지 (train positive 233,161 / 고유 기사 1,858)

| split | label | 기사 | c1c2c3 | c1c2 | c1c2 클릭 수 중앙값 |
|---|---|---|---|---|---|
| validation half | 정답 | 7.8% | 13.5% | 85.6% | 582 |
| validation half | 오답 | 21.4% | 26.6% | 88.0% | 555 |
| test | 정답 | 1.7% | 6.5% | 86.3% | 790 |
| test | 오답 | 12.2% | 17.4% | 88.5% | 638 |

## 해석
- 공개 원본 모델도 무작위보다 낮다 (Top-1 17.4% < 20%, AUC 0.487 < 0.5). 우리 TIGER-style 실패와 같은 양상이라 구현 문제가 아니다.
- 원인: test 정답 기사 중 train에서 클릭된 적 있는 것은 1.7%, 오답은 12.2%. 생성형 모델은 train에서 본 기사(SID)에 확률을 주므로 오답을 더 높게 본다.
  학습이 진행될수록(train loss 감소) val Top-1·AUC가 떨어지고 c3 CE가 커지는 것도 같은 현상.
- c1, c1+c2까지만 쓰면 무작위보다 조금 낫다 (test Top-1 23.7%, AUC 0.532): 카테고리 수준 선호는 일반화되고, 기사 수준(c3)은 새 기사에 일반화되지 않는다.
