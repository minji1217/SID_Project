# Candidate SID 충돌 분석 결과

실험: `normalize_v2_uni_lu005_m05_candidate_train_v1`
분석 스크립트: `src/analyze_candidate_sid_collision.py`

RQ-VAE 설정: Input 768 / Hidden [512] / Latent 128 / **Q2 128** / **Q3 512** /
STE / L2 / Q2 Init = Event K-means / λrec 1.0 · λcb 0.25 · λcom 0.1 · λuniq 0.05 /
Margin 0.5 / Batch 256 / LR 2e-4 / Seed 42 / Checkpoint best_rec

---

## 1. 요약

| 확인 항목 | 결과 | 판정 |
|---|---|---|
| codebook 활용률 | c2 128/128, c3 512/512 | **붕괴 없음** |
| 후보 내 pos-neg SID 충돌 | train 0.0045% / val 0.0137% | **무시 가능** |
| 후보 내 neg-neg SID 중복 | train 0.177% / val 0.420% | 무시 가능 |
| candidate 생성 오류 | pos와 article_id 같은 neg 0건 | 이상 없음 |
| positive끼리 SID 충돌 | 0건 | 이상 없음 |

**결론: `(c1, c2, c3)`만으로 후보를 구분하는 데 문제가 없다.**
SID 충돌로 인한 label noise는 학습에 영향을 줄 수 없는 수준이며,
codebook 크기(Q2 128 / Q3 512)나 λuniq를 조정할 근거가 없다.

---

## 2. 데이터 규모

| | train | validation |
|---|---|---|
| impression 수 | 232,292 | 244,647 |
| candidate 총 개수 | 2,575,608 | 2,928,942 |
| positive 개수 | 233,161 | 245,622 |
| negative 개수 | 2,342,447 | 2,683,320 |
| impression당 candidate | 11.088 | 11.972 |
| positive 2개 이상인 행 | 653 (0.28%) | 748 (0.31%) |

positive는 사실상 impression당 1개다(평균 1.004). 따라서 positive 단위 지표와
impression 단위 지표가 거의 일치한다.

---

## 3. SID 사용 현황 (article 단위)

`article_semantic_ids.parquet` 기준이며 candidate 구성과 무관하다.

| 항목 | 값 |
|---|---|
| 기사 수 | 12,860 |
| 사용된 c1 코드 수 | 25 (= 카테고리 수) |
| **사용된 c2 코드 수** | **128 / 128** |
| **사용된 c3 코드 수** | **512 / 512** |
| 서로 다른 `(c1,c2,c3)` 수 | 10,952 |
| SID를 공유하는 기사 수 | 3,137 (24.39%) |
| SID당 평균 / 최대 기사 수 | 1.174 / 17 |

**codebook collapse가 없다.** c2와 c3 모두 할당된 코드를 전부 사용한다.
λuniq = 0.05가 의도대로 동작했으며, RQ-VAE의 대표적 실패 모드를 피했다.

다만 **"전부 사용"이 "균등 사용"을 뜻하지는 않는다.**
이론적 SID 공간은 `25 × 128 × 512 = 1,638,400`이고 기사는 12,860개뿐이므로,
균등 분포라면 SID가 겹치는 기사가 약 50개 나와야 한다.
실제 여분은 1,908개(= 12,860 − 10,952)로 **약 38배 집중**되어 있다.

이는 결함이 아니라 residual quantizer가 의미적으로 유사한 기사를
같은 코드로 묶고 있다는 증거이며, 설계 의도와 일치한다.

---

## 4. impression별 candidate 구성 분포

### train

| 항목 | 평균 | 표준편차 | 최소 | p25 | p50 | p75 | p90 | p99 | 최대 |
|---|---|---|---|---|---|---|---|---|---|
| candidate 수 | 11.088 | 7.664 | 5 | 6 | 8 | 13 | 20 | 41 | 79 |
| positive 수 | 1.004 | 0.082 | 1 | 1 | 1 | 1 | 1 | 1 | 7 |
| negative 수 | 10.084 | 7.659 | 3 | 5 | 7 | 12 | 19 | 40 | 78 |

### validation

| 항목 | 평균 | 표준편차 | 최소 | p25 | p50 | p75 | p90 | p99 | 최대 |
|---|---|---|---|---|---|---|---|---|---|
| candidate 수 | 11.972 | 9.239 | 5 | 6 | 9 | 14 | 23 | 50 | 99 |
| positive 수 | 1.004 | 0.088 | 1 | 1 | 1 | 1 | 1 | 1 | 9 |
| negative 수 | 10.968 | 9.235 | 3 | 5 | 8 | 13 | 22 | 49 | 98 |

관찰:

- **candidate 수가 0인 impression은 없다.** negative가 0인 행도 없으므로
  모든 impression이 ranking 학습·평가에 사용 가능하다.
- **최소 candidate 수가 5**이고, 가장 흔한 값도 5개다
  (train 40,800건 17.56%, validation 40,154건 16.41%).
  6개 13.9%, 7개 10.5%로 이어지는 롱테일 분포다.
- 분포가 오른쪽으로 길다. 평균 11개지만 p99가 41(train) / 50(validation),
  최대 79 / 99다. **패딩 기반 배치 처리 시 p99와 최대값이 메모리 상한을 결정한다.**

---

## 5. pos-neg SID 충돌

같은 impression에서 negative가 positive와 SID가 완전히 같은 경우.
모델 입력이 동일한데 label만 다르므로 순수한 label noise다.

### train

| level | 충돌 negative | micro | macro | 충돌 포함 행 |
|---|---|---|---|---|
| c1 | 484,108 | 20.667% | 20.374% | 71.748% |
| c1c2 | 13,198 | 0.5634% | 0.6013% | 4.5520% |
| **c1c2c3** | **106** | **0.0045%** | **0.0048%** | **0.0418%** |
| c1c2c3c4 | 0 | 0% | 0% | 0% |

### validation

| level | 충돌 negative | micro | macro | 충돌 포함 행 |
|---|---|---|---|---|
| c1 | 519,403 | 19.357% | 19.377% | 70.811% |
| c1c2 | 15,421 | 0.5747% | 0.6127% | 5.0465% |
| **c1c2c3** | **367** | **0.0137%** | **0.0183%** | **0.1435%** |
| c1c2c3c4 | 0 | 0% | 0% | 0% |

**train 234만 negative 중 106개, validation 268만 중 367개.**
영향받는 impression은 각각 97건(0.042%), 351건(0.144%)이며,
전체 impression의 99.96% / 99.86%가 완전히 깨끗하다.

positive 단위로 보아도 동일하다 — 오염된 positive는 train 97개(0.042%),
validation 351개(0.143%)이고, positive 하나가 겪는 최대 충돌은 3개 / 7개다.
positive끼리 SID가 겹친 경우는 **양쪽 모두 0건**이므로,
ranking loss에서 순위를 정할 수 없는 positive 쌍은 존재하지 않는다.

> **주의: `c1c2c3c4` 행은 해석하지 말 것.**
> c4는 `assign_disambiguation_c4()`가 같은 `(c1,c2,c3)`를 가진 기사에
> `groupby(...).cumcount()`로 0부터 순차 부여한 값이다.
> train/validation을 합친 뒤 article 단위로 한 번에 배정하므로
> `(c1,c2,c3,c4)`는 정의상 article_id와 1:1이며, 충돌 0은 항등식이다.

---

## 6. neg-neg SID 중복

같은 impression에서 negative끼리 SID가 같은 경우.
label이 둘 다 0이라 모순은 아니며, loss에서 해당 SID 영역이
중복 계상되는 가중치 문제다.

| | train | validation |
|---|---|---|
| 중복 negative (c1c2c3) | 4,139 (0.1767%) | 11,257 (0.4195%) |
| macro 비율 | 0.1563% | 0.3429% |
| 묶음마다 1개만 남길 때 제거되는 수 | 2,090 (0.0892%) | 5,977 (0.2227%) |
| 중복 묶음 수 | 2,049 | 5,280 |
| 한 묶음의 최대 negative 수 | 5 | 9 |
| 중복을 포함한 impression 비율 | 0.8567% | 1.9718% |

묶음 크기 분포 (c1c2c3):

| 묶음 크기 | train | validation |
|---|---|---|
| 2개 | 2,025 (98.83%) | 4,732 (89.62%) |
| 3-4개 | 19 (0.93%) | 539 (10.21%) |
| 5-9개 | 5 (0.24%) | 9 (0.17%) |
| 10+ | 0 | 0 |

**중복은 거의 전부 2개짜리 쌍이다.** 10개 이상 뭉친 묶음이 하나도 없으므로,
특정 SID에 후보가 몰리는 현상은 관찰되지 않는다.
article 단위로는 SID당 최대 17개까지 몰려 있지만(3절),
그 기사들이 같은 impression에 동시 노출되는 일은 드물다.

neg-neg 중복은 pos-neg 충돌보다 **train 39배, validation 31배** 크다.
positive는 impression당 1개인데 negative는 평균 10개이므로,
쌍의 개수가 대략 `C(10,2) / 10 ≈ 4.5배` 많고 여기에
클릭된 기사가 후보 중 의미적으로 독특한 경향이 더해진 결과로 보인다.

---

## 7. candidate 전체 기준

label과 무관하게 candidate list가 SID로 얼마나 구분되는지 본다.

| level | 구분불가 candidate (train) | 고유 SID 비율 (train) | 구분불가 (val) | 고유 SID 비율 (val) |
|---|---|---|---|---|
| c1 | 2,055,887 (79.82%) | 42.73% | 2,331,864 (79.61%) | 42.24% |
| c1c2 | 213,659 (8.30%) | 95.51% | 276,377 (9.44%) | 94.88% |
| **c1c2c3** | **4,326 (0.168%)** | **99.915%** | **11,948 (0.408%)** | **99.784%** |

레벨별 분리력 (충돌 감소 배수):

| 단계 | train (pos-neg) | train (neg-neg) | codebook |
|---|---|---|---|
| c1 → c1c2 | 36.7x | 9.4x | 128 |
| c1c2 → c1c2c3 | 124.5x | 46.9x | 512 |

c1 단독으로는 79.8%가 구분되지 않는다. c1은 학습된 코드가 아니라
카테고리 직접 매핑(25종)이므로 당연한 결과다.
c2(event K-means 초기화)가 한 자릿수 %로 줄이고, c3가 0.2% 이하로 떨어뜨린다.

---

## 8. train / validation 격차

| 지표 | train | validation | 배수 |
|---|---|---|---|
| pos-neg 충돌 micro | 0.0045% | 0.0137% | **3.02x** |
| neg-neg 중복 micro | 0.1767% | 0.4195% | **2.37x** |
| positive 하나의 최대 충돌 | 3 | 7 | — |
| neg 묶음 최대 크기 | 5 | 9 | — |

후보 리스트 길이 차이(11.97 vs 11.09)로 설명되는 부분은
충돌 확률이 후보 수의 제곱에 비례한다고 볼 때 **1.17배**에 불과하다.
나머지 2.6배 / 2.0배는 **일반화 격차**로 보는 것이 타당하다.
validation-only 기사는 frozen inference로 SID를 받으므로
train 기사만큼 잘 분리되지 않는다.

절대값이 작아 현재는 무해하지만, **codebook 크기나 λuniq를 바꾸는
후속 실험에서 이 격차가 벌어지는지 추적할 지표**다.

---

## 9. 권고

1. **아이템 표현은 `(c1, c2, c3)` 3-tuple로 한다.**
   c4는 의미 없는 순번이므로 모델 입력에 넣으면 존재하지 않는 규칙을
   학습하게 된다. 단, article_id 역매핑에는 필요하므로 매핑 테이블에는 유지한다.

2. **충돌 negative를 삭제하거나 마스킹하지 않는다.**
   0.0045%는 조치 대상이 아니라 데이터 품질 검증 통과를 뜻한다.
   또한 충돌률 자체가 SID 품질의 측정치이므로, 이를 제거하면
   RQ-VAE 설정 간 비교에서 좋은 SID와 나쁜 SID를 구분하는 신호를 잃는다.

3. **validation에는 어떤 후보 필터링도 적용하지 않는다.**
   어느 후보가 positive인지 알아야 필터링이 가능하므로,
   label 정보를 이용해 어려운 케이스를 평가에서 제외하는 것이 되어
   지표가 부풀려지고 baseline과 비교가 불가능해진다.

4. **codebook 설정을 유지한다.**
   c2 128/128, c3 512/512로 활용률이 포화이므로 크기를 줄일 이유가 없고,
   충돌이 무시 가능하므로 키울 이유도 없다.

5. **후속 실험에서는 `c1c2c3` micro 비율을 게이트로 쓴다.**
   1% 이상으로 올라가면 그때 train 한정 loss 마스킹을 검토한다.
   그 아래면 무조치가 맞다.

---

## 10. 재현 방법

```bash
python -m src.analyze_candidate_sid_collision \
  --semantic-ids data/output/experiments/<실험>/post_rqvae \
  --report data/output/experiments/<실험>/reports/candidate_sid_collision.json
```

옵션:

| 옵션 | 설명 |
|---|---|
| `--path` | sequences parquet 직접 지정 (반복 가능) |
| `--semantic-ids` | 실험 폴더 또는 `article_semantic_ids.parquet` 경로 |
| `--chunk-size` | 한 번에 처리할 impression 수 (기본 100,000) |
| `--limit` | 앞에서부터 N개 impression만 (빠른 확인용) |
| `--report` | 결과 JSON 저장 경로 |

구현상 보장:

- 모든 충돌 판정은 `(impression, SID)` 단위이므로 다른 impression과 섞이지 않는다.
- chunk는 impression 경계에서만 잘리므로 전량 처리와 결과가 동일하다.
- 백분위는 보간 없이 nearest-rank로 계산한다.
- 메모리는 파일 크기가 아니라 `--chunk-size`에만 비례한다.
