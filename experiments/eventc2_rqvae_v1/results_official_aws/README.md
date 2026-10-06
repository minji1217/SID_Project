# eventc2 공식 재학습 결과 (AWS single GPU)

- 학습: `configs/rqvae_ebnerd_eventc2_aws.gin`, early stop epoch 410, best_rec = 0.154025 @ epoch 210
- SID: `checkpoint_best_rec.pt` + `--train_c2_policy event` (EventCode 표 일치율 1.0, max_c4 31)
- 평가: `evaluate/evaluate_all.py` (발표 표와 같은 코드). `eval_all.txt`, `eval_validation.txt`
- checkpoint는 AWS `~/minji/SID_Project/RQVAE/out/rqvae/ebnerd/eventc2_lu0.05_m0.5/`에만 있음

## 비교 (WD-001 / UNI = 발표 값, UNI는 이 세션 재평가와 일치)

| 지표 | 범위 | WD-001 | UNI (0.05, 0.5) | EventC2 |
|---|---|---|---|---|
| Valid Rec Loss | | 0.149850 | 0.152458 | 0.154025 * |
| C2 Event Consistency | 전체 | 0.750742 | 0.741546 | **1.000000** |
| Q3 Used Codes | 전체 | 512 | 512 | **260 (50.8%)** |
| Collision Rate | 전체 | 31.56% | 24.39% | **52.40%** |
| ΔC2 | 전체 | 0.025198 | 0.023748 | 0.028625 |
| ΔC3 | 전체 | 0.048229 | 0.051026 | **0.028028** |
| max_c4 | 전체 | - | 16 | 31 |
| Q2 Used Codes | Validation | 126 | 128 | 128 |
| Q3 Used Codes | Validation | 487 | 504 | **258** |
| Collision Rate | Validation | 20.82% | 14.38% | **32.64%** |
| ΔC2 | Validation | 0.030399 | 0.023016 | 0.029519 |
| ΔC3 | Validation | 0.051005 | 0.060713 | **0.029552** |

\* `eval_reconstruction.py`는 event_ids 없이 forward(기사 단위 c2)라 EventC2 모델에서는 0.157125로 나온다.
event 모드 Validation rec는 학습 로그의 best_rec 0.154025.

## 학습 추이 (event_c2_log.jsonl, Train)

| epoch | Q2 used | Q3 used | collision(기사 비율) | max_c4 |
|---|---|---|---|---|
| 0 (init) | 128 | 78 | 92.5% | 156 |
| 1 | **1** | 5 | 100% | 2429 |
| 27 | 52 | 25 | 93.9% | 411 |
| 100 | 100 | 156 | 57.3% | 79 |
| 172 | 128 | 236 | 48.6% | 31 |
| 210 (best_rec) | 128 | 259 | 47.9% | 30 |
| 373 | 128 | 361 | 46.7% | 32 |

- epoch 1에 Q2가 code 1개로 붕괴했다가 epoch ~172에 128개로 회복.
- Q3 사용 코드는 early stop 시점까지 계속 증가 (best_rec epoch 210에서는 259).
