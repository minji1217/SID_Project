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
