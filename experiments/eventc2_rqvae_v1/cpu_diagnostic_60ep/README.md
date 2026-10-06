# CPU 진단 실행 (60 epoch, event vs article)

공식 결과가 아니다. 구현 검증과 학습 초기 동태 확인용이다.
- CPU, AMP 끔, 60 epoch, eval 10 epoch마다, early stopping 끔
- 나머지 하이퍼파라미터는 A와 같음 (`run_*.gin`)
- 데이터: 업로드된 EB-NeRD Train 9,738 / Validation 3,122

## epoch 60 비교 (같은 seed, 같은 데이터)

| 항목 | article (A 방식) | event |
|---|---|---|
| Validation reconstruction loss | 0.159 | 0.164 |
| Q2 code 사용 | 79 | 53 |
| Q3 code 사용 | 505 | **83** |
| unique c123 / c123 collision rate | 10,631 / 17.3% | 5,217 / **59.4%** |
| max c4 | 17 | 104 |
| Train same-event C2 consistency | 28.8% | 100% |
| cos(r2, −q1) | −0.06 | **0.59** |
| category가 설명하는 r2 분산 | 0.4% | 10.4% |

event 모드의 Q2 사용 code 추이 (epoch 1→60): 1, 6(10), 7(20), 15(30), 25(40), 38(50), 53(60).
c123 collision: 99.7% → 98.0 → 91.9 → 83.8 → 75.1 → 64.2 → 55.3% (Train snapshot).

## 해석

z(E) = mean(h)는 category 성분(q1 방향)을 포함한 h 공간의 값이라, q2(E)가 h 공간 중심을 근사한다.
그래서 r2 = h − q1 − q2(E)에 −q1 성분이 남고(cos 0.59), Q3가 기사별 차이보다
category 보정에 code를 쓰면서 c3 다양성이 무너진다.
