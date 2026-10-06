# EventC2 SID 정성·정량 평가 (UNI 평가와 같은 코드, 같은 모집단)

기존 UNI 평가: `origin/claude/hopeful-mendel-fhc2n4:analysis/rqvae_sid_evaluation` (UNI = WD-001-UNI-lu0.05-m0.5)
여기서는 그 코드를 복사해 입력 경로만 EventC2로 바꿨다 (seed 42, body-valid train 9,338, 같은 TF-IDF 설정).
복원 단계(step6b)만 EventC2에 맞게 수정: SID 재현 확인과 forward rec loss 교차검증을 event_ids로 실행.

입력: EventC2 SID (`eventc2_lu0.05_m0.5`), EB-NeRD articles.parquet, EventC2 checkpoint_best_rec.pt (epoch 210)

## 정량 1 — 실제 텍스트 (Large · group-balanced, mean)

| 지표 | 조건 | UNI | EventC2 |
|---|---|---|---|
| Body TF-IDF | Different | 0.0291 | 0.0297 |
| | Prefix-2 | 0.0799 | 0.0766 |
| | Prefix-3 | **0.2484** | **0.1936** |
| Full-text TF-IDF | Different / P2 / P3 | 0.0283 / 0.0823 / 0.2713 | 0.0288 / 0.0790 / 0.2035 |
| Title Jaccard | Prefix-3 | 0.2059 | 0.1426 |
| Prefix-3 pair 수 | | 1,534 | 8,699 |

Body TF-IDF median: UNI 0.0259 / 0.0564 / 0.1879, EventC2 0.0266 / 0.0553 / 0.1294.
EventC2도 subset(중복/반복 템플릿 제거)과 full-text에서 순서 역전 0건.

## 정량 2 — embedding (shared prefix 0→1→2→3, mean)

| | UNI | EventC2 |
|---|---|---|
| Original | 0.7986 → 0.8180 → 0.8470 → 0.9070 | 0.7995 → 0.8130 → 0.8448 → 0.8808 |
| Reconstruction | 0.9294 → 0.9522 → 0.9810 → 1.0000 | 0.9328 → 0.9455 → 0.9826 → 1.0000 |

## 정량 3 — self-reconstruction cos(x, x_hat)

| | UNI | EventC2 |
|---|---|---|
| 전체 12,860 | 0.9275 | 0.9257 |
| Train | 0.9297 | 0.9275 |
| Validation | 0.9205 | 0.9202 |

검증: SID 재현 train 9,735/9,738, validation 3,117/3,122 (CPU 재계산의 소수점 차이),
같은 SID → 같은 x_hat (최대 차 0), forward rec loss와 저장 SID x_hat rec loss 차 1.5e-7.

## 정성 (step2/out, 30 Prefix-3 group / 15 Prefix-2 group)

- 잘 묶인 예 P3-20 (auto, 4개): 기업 실적 자동기사 (Hoppeloppeland·Alk-Abelló·Norden·Rockwool "가치 수백만 크로네 상승"). Body TF-IDF 0.417
- 잘 묶인 예 P3-28 (forbrug, 6개): 통신·전기 요금 절약 기사 6개. 0.298
- TF-IDF는 낮지만 주제는 통하는 예 P3-19 (underholdning, 5개): 유명인과 자녀 이야기 (Lionel Richie, Daniel Radcliffe, De Niro, Stallone 등). 0.033
- 형식만 같은 예 P3-29 (side9, 6개): Side 9 인물 갤러리 템플릿, 서로 다른 인물. 0.043 (UNI의 P3-11과 같은 유형)
