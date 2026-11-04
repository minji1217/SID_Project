# NRMS 20 epoch 재실행 (E5 · recon · SID · 섞은 SID)

5 epoch(정민교 님 MIND 설정)에서 E5와 recon은 best epoch이 대부분 5라서 아직 오르는 중이었다.
같은 코드 · 데이터 · 설정 · seed로 `--epochs 20`만 바꿔 다시 돌린다. checkpoint는 그대로 validation Top-1 최고 epoch.

| 모드 | 기사 입력 | 5 epoch test Top-1 |
|---|---|---|
| e5 | 원본 E5 (`~/minji/e5_input/article_embeddings.npy`) | 29.92% |
| recon | `nrms_recon_ebnerd_v1`의 `recon_embeddings.npy` | 29.99% |
| sid | v2 `tables/sid_codes.npy` | 26.09% |
| sid_shuffled | v2 `tables/sid_shuffled.npy` | 20.71% |

Title(원 논문)은 epoch당 약 8분이라(20 epoch × 3 seed ≈ 8시간) 기본 목록에서 뺐다. 필요하면 `MODES=title`.

## 실행 (AWS)

모드마다 결과 폴더가 따로라서 4개를 동시에 띄운다.

```bash
cd ~/minji/SID_Project
for m in e5 recon sid sid_shuffled; do
  MODES=$m RAW_DIR=~/minji/e5_input CUDA_VISIBLE_DEVICES=0 \
    nohup bash experiments/nrms_epochs20_ebnerd_v1/run_aws.sh > data/output/experiments/nrms_epochs20_$m.log 2>&1 &
done
tail -f data/output/experiments/nrms_epochs20_*.log
```

끝난 뒤(`== DONE mode=...` 4개):

```bash
python experiments/nrms_epochs20_ebnerd_v1/summarize.py
```

예상 시간: run 하나 = 20 epoch × 약 35초 + 평가 ≈ 13분, 모드당 seed 3개 ≈ 40분.
4개를 동시에 돌리면 GPU · CPU를 나눠 써서 epoch이 느려질 수 있어 전체 1 ~ 1.5시간 안팎(순서대로 돌리면 약 2.6시간).

## 결과 (AWS 실행 완료, test 122,668행, seed 42 / 123 / 2026 평균, `summarize.py` 출력)

| 모드 | Val Top-1 | **Test Top-1** | seed 표준편차 | MRR | nDCG@5 | AUC | best epoch (seed:epoch/20) | 5 epoch test |
|---|---|---|---|---|---|---|---|---|
| e5 | 28.89% | **29.76%** | 1.01 | 0.5426 | 0.6558 | 0.6102 | 42:9 · 123:5 · 2026:4 | 29.92% |
| **recon** | 30.02% | **30.28%** | 0.24 | 0.5458 | 0.6582 | 0.6124 | 42:10 · 123:7 · 2026:19 | 29.99% |
| sid | 25.94% | **26.09%** | 0.10 | 0.5107 | 0.6313 | 0.5711 | 42:2 · 123:1 · 2026:3 | 26.09% |
| sid_shuffled | 21.03% | **21.13%** | 0.34 | 0.4676 | 0.5981 | 0.5156 | 42:9 · 123:1 · 2026:1 | 20.71% |

- seed별 test Top-1: e5 29.65 / 28.81 / 30.82, recon 30.51 / 30.30 / 30.04, sid 26.20 / 26.01 / 26.05, sid_shuffled 21.51 / 20.86 / 21.02 (42 / 123 / 2026)
- nDCG@5 평균은 로그의 seed별 값으로 계산했다 (e5 seed 123은 best epoch 5로 5 epoch run과 같은 checkpoint → v2 `metrics_e5_s123.json` 값과 test Top-1 28.81%까지 일치).
- recon − E5 = +0.52%p (E5 seed 편차 1.01보다 작음 → 동등 이상), recon − SID = **+4.19%p**, SID ÷ E5 = 87.7%
- SID는 세 seed 모두 best epoch 1 ~ 3이라 5 epoch과 같은 checkpoint → epoch을 늘려도 그대로. recon − SID 차이는 학습 부족 때문이 아니다.
- E5는 epoch을 늘려도 평균이 오르지 않았다(29.92 → 29.76). seed 123 · 2026은 5 epoch 안에서 이미 최고.
- epoch 1 ~ 5의 val 값이 5 epoch run과 같아 재현이 정확하다.
