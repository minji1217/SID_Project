# E5 KNN 평가: 원본 E5 이웃이 SID prefix도 공유하는가 (발표 8장)

발표 8장 KNN은 당시 학습 코드 없이 사후 평가로만 계산했고 코드가 남아 있지 않았다.
같은 방법을 `evaluate_knn.py`로 다시 만들어 UNI SID로 실행했고, 발표 값이 소수점까지 그대로 재현됐다.

## 방법
1. Train 기사 9,738개의 원본 E5 임베딩(L2 정규화)
2. 기사마다 나머지 9,737개와 cosine → 자기 자신을 뺀 Top-10
3. 이웃이 같은 c1 / (c1,c2) / (c1,c2,c3)인지 확인
4. 9,738 × 10 = 97,380쌍에서 비율 계산
5. Random baseline = Train 기사 중 서로 다른 두 기사가 같은 prefix일 확률 Σ n_g(n_g−1) / (N(N−1))

## 결과 (UNI SID, `results/uni_train_k10_summary.json`)
| 기준 | E5 Top-10 일치 | Random baseline | 배수 |
|---|---|---|---|
| same c1 | 70.85% | 15.75% | 4.50× |
| same c1,c2 | 28.20% | 0.299% | 94.17× |
| same c1,c2,c3 | 2.76% | 0.00398% | 693.88× |

이웃 평균 cosine 0.8820. 입력 SID는 UNI(WD-001-UNI-lu0.05-m0.5 best_rec, max c4 16) `article_semantic_ids.parquet`,
임베딩은 `article_embeddings.npy`(6조각으로 받은 것을 순서대로 이어 사용).

## 실행 (AWS 예)
```bash
python experiments/knn_e5_sid_v1/evaluate_knn.py \
  --sid <article_semantic_ids.parquet> \
  --embeddings <article_embeddings.npy> \
  --split train --k 10 --output-dir <새 출력 폴더>
```
출력: `knn_summary.json`(지표), `knn_pairs.parquet`(anchor-neighbor 97,380쌍). 출력 폴더가 있으면 덮어쓰지 않고 멈춘다.
EventC2 등 다른 SID도 `--sid`만 바꿔 같은 방식으로 비교할 수 있다.

## 추가: Validation-only 기사 (2026-11-03, 같은 스크립트 `--split validation` / `--split all`)
이웃은 같은 split 안에서만 찾는다. Validation은 기사 수(3,122)가 Train(9,738)보다 적어 이웃 후보가 적으므로,
Train에서 3,122개를 무작위로 뽑아(seed 42 / 123 / 2026 평균) 같은 크기로 맞춘 비교도 함께 적는다 (이 비교는 스크립트 밖에서 같은 계산으로 수행).

| 기준 | Train 9,738 | Train 3,122 표본 (3-seed) | **Validation 3,122** | 전체 12,860 |
|---|---|---|---|---|
| same c1 | 70.85% (4.50×) | 65.41% (4.2×) | **65.41% (4.53×)** | 71.33% (4.67×) |
| same c1,c2 | 28.20% (94.2×) | 22.37% (73.1×) | **21.04% (43.6×)** | 27.86% (90.7×) |
| same c1,c2,c3 | 2.76% (694×) | 1.13% (262×) | **1.82% (201×)** | 3.20% (771×) |
| 이웃 평균 cosine | 0.8820 | 0.871 | 0.8726 | 0.8843 |

괄호는 random baseline 대비 배수. Validation random baseline은 c1,c2 0.483% / c1,c2,c3 0.00903%로 Train보다 높다
(validation 기사가 더 적은 (c1,c2) 그룹에 몰림: 사건 c2 상속). 그래서 배수만 보면 낮아 보이지만, 같은 크기 Train 표본과 비교한
일치율 자체는 c1 같음, c1,c2 약간 낮음(−1.3%p), c1,c2,c3 높음(+0.7%p)이다.
결과: `results/uni_validation_k10_summary.json`, `results/uni_all_k10_summary.json`.
