# ⚠️ 이 폴더의 결과는 사용하지 마세요

업로드된 `article_semantic_ids.parquet`의 `(c1,c2,c3)`와
`checkpoint_best_rec.pt` (WD-001-7d3a10)의 codebook·decoder를 섞어서 계산한
결과입니다. **두 파일이 서로 다른 모델에서 나왔기 때문에 해석할 수 없습니다.**

## 불일치 증거

`generate_semantic_ids.py:616-623`의 train 경로는 `get_semantic_ids(x, category_ids)`를
그대로 호출하고 event C2로 덮어쓰지 않는다. 따라서 같은 모델이라면 train 9,738건의
c2/c3가 100% 재현되어야 한다. 실제로는:

| 코드 | 재현 일치율 | 무작위 기대값 |
|---|---|---|
| c1 | 100.00% | 4.00% (= 1/25) |
| c2 | **1.85%** | 0.78% (= 1/128) |
| c3 | **0.22%** | 0.20% (= 1/512) |

c1은 모델이 선택하지 않고 `model_category_id`를 그대로 쓰므로(rqvae.py:381-388,
quantize.py:270) 100%가 당연하다. 모델이 실제로 선택하는 c2/c3는 무작위 수준이다.

복원 품질 (전체 12,860, squared error):

| | rec loss |
|---|---|
| 이 checkpoint 자체 양자화 x_hat | 0.1355 |
| 저장된 SID로 만든 x_hat | 0.2266 |

## 체크포인트와 임베딩 자체는 정상 (둘 다 검증 통과)

- `article_embeddings.npy`: checkpoint에 기록된 `best_valid_rec_loss = 0.1498452`를
  이 임베딩으로 재계산 → **0.1498500** (차 4.8e-6). 학습에 쓰인 바로 그 파일이 맞다.
- checkpoint: WD-001-7d3a10, epoch 210, embed_dim 128, c2 128, c3 512.
  `gin_config`의 `save_dir_root`가 `/content/drive/MyDrive/SID_Project_Colab/results/
  ebnerd/stage_weight_decay/WD-001-7d3a10/rqvae/` — Colab 실행분.

## 결론

업로드된 SID 파일은 이 checkpoint가 생성한 것이 아니다. `best_valid_total_epoch = 20`
이므로 `checkpoint_best_total.pt`(epoch 20)로 생성했거나, 다른 EXP 폴더의 산물일 수 있다.

올바른 Step 6을 위해서는 **그 `article_semantic_ids.parquet`와 같은 EXP 폴더 안의
`rqvae/` checkpoint**가 필요하다.

(무효한 39MB `reconstructed_embeddings.npy`는 저장소 용량을 위해 삭제했다. 스크립트로 재생성 가능.)

## 이 폴더와 무관하게 유효한 값

checkpoint 자체 양자화 기준 `self_reconstruction_cosine = cos(x, x_hat)`는
저장된 SID와 무관하므로 유효하다 (`../self_reconstruction_own_quantization.csv`).
