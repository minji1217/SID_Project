# UNI test: V1 candidate ranker vs 공개 TIGER (reference)

같은 test sample 122,668개 (impression·후보 5개·positive 일치 확인)

| 지표 | V1 ranker | TIGER-style | 차이 |
|---|---|---|---|
| top1_accuracy | 0.270666 | 0.173582 | -0.097083 |
| mrr | 0.518065 | 0.439600 | -0.078465 |
| ndcg@5 | 0.636875 | 0.576881 | -0.059994 |
| auc | 0.578739 | 0.486598 | -0.092141 |
| preference_loss | 1.744624 | 4.888992 | +3.144368 |
| positive_probability | 0.243484 | 0.174126 | -0.069358 |

## 실제 학습 설정 (V1: run_summary.json의 gin_config, TIGER: config.json)

| 항목 | V1 | TIGER |
|---|---|---|
| d_model | 256 | 384 |
| heads | 8 | 6 |
| layers | 2 | 4 |
| d_ff | 1024 | 1024 |
| dropout | 0.0 | 0.1 (T5Config 기본) |
| max history | 50 | 20 |
| learning rate | 5e-05 | 0.001 |
| batch size | 128 | 640 |
| weight decay | 0.0 | - |
| seed | 42 | 42 |
| epochs / patience | 30 / 5 | 30 / 5 |
| c4 vocab | 28 | level dedup(미사용) |
| train data | 'datasets/ebnerd/train_sequences_1pos4neg.parquet' | /home/ubuntu/minji/uni_post_rqvae/train_sequences.parquet |
| best epoch | 1 | 1 |

TIGER best epoch 1 / 6 epoch

## TIGER 진단 (test)

| level | CE | token accuracy |
|---|---|---|
| c1 | 1.8779 | 0.3021 |
| c2 | 4.9350 | 0.0623 |
| c3 | 10.3152 | 0.0050 |

- positive 평균 log-prob -17.1281
- negative 평균 log-prob -16.7928
- gap -0.3353

| 누적 점수 | Top-1 | AUC | MRR | nDCG@5 |
|---|---|---|---|---|
| c1 | 0.2299 | 0.5297 | 0.4811 | 0.6084 |
| c1+c2 | 0.2372 | 0.5322 | 0.4863 | 0.6123 |
| c1+c2+c3 | 0.1736 | 0.4866 | 0.4396 | 0.5769 |
