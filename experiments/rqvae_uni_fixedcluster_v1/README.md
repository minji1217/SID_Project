# RQ-VAE UNI 재학습 (고정 클러스터 전처리 입력)

UNI(`WD-001-UNI-lu0.05-m0.5`)와 **같은 코드·같은 하이퍼파라미터**로 RQ-VAE를 다시 학습한다.
입력만 새 전처리 결과(`data/output/exports/rqvae_inputs/`)로 바꾼다.

모든 코드는 `minji1217/SID_Project` 에서 가져온다.
- 전처리 + 이 스크립트: `claude/preprocess-fixed-cluster`
- 학습 코드: `uniqueness-loss-experiment` @ `127ae7e`. UNI를 만든 코드이며, UNI gin 내용이 처음 들어간 커밋이다. `train_rqvae.py`, `modules/`, `data/` 는 브랜치 최신과 같다.

## 다시 학습해야 하는 이유

| 학습에서 쓰는 것 | 기존 UNI | 이번 |
|---|---|---|
| 학습 기사 (`article_master`) | 9,738 | 10,419 |
| Q2 코드북 K-means 초기화용 Train 사건 (`event_id`) | 9,145 | 9,794 |
| best checkpoint를 고르는 validation 기사 | 3,122 | 2,741 |

## 설정 (A checkpoint 의 gin_config 와 같음)

| 항목 | 값 |
|---|---|
| hidden / latent | [512] / 128 |
| c1 / c2 / c3 | 25 / 128 / 512 |
| lr / weight_decay | 2e-4 / 0.0 |
| λrec / λcb / λcom | 1.0 / 0.25 / 0.1 |
| λuniq / margin | **0.05** / 0.5 (원본 gin 파일에는 0.1로 적혀 있어 0.05로 덮어씀) |
| early stopping | 최소 200 epoch, 최대 1283, patience 20 (10 epoch마다 평가), min_delta 0.001 |
| checkpoint | `checkpoint_best_rec.pt` (validation 복원 손실 최소) |
| seed | 42 |

`make_gin.py` 는 원본 gin을 읽어 위 값과 같은지 먼저 확인한다. 같으면 `dataset_folder`, `lambda_uniq`, `save_dir_root`, `seed` 만 바꿔 쓴다.

## AWS 실행

```bash
cd ~/minji/SID_Project
git fetch origin claude/preprocess-fixed-cluster && git checkout claude/preprocess-fixed-cluster

# 1) 전처리 (원본 data/raw 5개 필요. 없으면 experiment/entity-linking-event 브랜치에서 꺼냄)
git fetch origin experiment/entity-linking-event
mkdir -p data/raw/train data/raw/validation
for f in articles.parquet train/behaviors.parquet train/history.parquet validation/behaviors.parquet validation/history.parquet; do
  [ -f data/raw/$f ] || git cat-file -p origin/experiment/entity-linking-event:data/raw/$f > data/raw/$f
done
python -m src.main preprocess > data/output/preprocess.log 2>&1
tail -5 data/output/preprocess.log     # PREPROCESS_COMPLETE

# 2) 패키지 (torch 는 서버 CUDA 빌드가 이미 있으면 그대로)
pip install gin-config einops accelerate scikit-learn huggingface-hub wandb

# 3) (선택) 1 epoch 스모크 테스트, 몇 분
SMOKE=1 CUDA_VISIBLE_DEVICES=0 bash experiments/rqvae_uni_fixedcluster_v1/run_aws.sh

# 4) 본 학습
SEED=42 CUDA_VISIBLE_DEVICES=0 nohup bash experiments/rqvae_uni_fixedcluster_v1/run_aws.sh \
    > data/output/rqvae_uni_fixedcluster_s42.log 2>&1 &
tail -f data/output/rqvae_uni_fixedcluster_s42.log
grep -a "Validation epoch\|EarlyStopping" data/output/rqvae/uni_fixedcluster_lu0.05_m0.5_s42/train.log | tail
```

- 학습 코드는 `~/minji/rqvae_uni_code_127ae7e/` 에 worktree로 꺼낸다. 지금 checkout은 바뀌지 않는다.
- 결과는 `data/output/rqvae/uni_fixedcluster_lu0.05_m0.5_s<SEED>/` 에만 쓴다. 폴더가 이미 있으면 멈춘다.
  - 결과 파일: `checkpoint_best_rec.pt`, `train.log`, `rqvae_uni.gin`, `operative_config.gin`, `provenance.txt`, `inputs.sha256`, `inputs_manifest.json`
- GPU는 1장만 쓴다 (`CUDA_VISIBLE_DEVICES`). 다른 사람이 쓰는 GPU는 피한다.
- 시작할 때 입력을 검사한다. 기대값은 train 10,419 / validation-only 2,741 / train 사건 9,794 이고, 다르면 경고를 출력한다.

## 확인한 것 (이 세션, CPU)

`SMOKE=1 ALLOW_CPU=1` 로 1 epoch을 돌려 다음을 확인했다.
- 입력 연결: 학습 기사 10,419, validation 2,741
- Q2 초기화: Train 사건 9,794로 K-means
- uniqueness loss 계산
- `checkpoint_best_rec.pt` 저장
- `operative_config.gin` 값이 위 표와 같음

## 다음

`checkpoint_best_rec.pt` 로 SID를 생성한다.
- Train 기사: 기사별 nearest code
- Validation-only 기사: Train 사건에만 배정 (timeupdate 수정본)
- 생성한 SID로 `python -m src.main post-rqvae` 를 돌리고 Transformer를 학습한다.
