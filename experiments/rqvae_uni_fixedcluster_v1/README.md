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

기존 결과를 덮어쓰지 않도록 전처리 출력은 새 폴더(`SID_OUTPUT_DIR`)에 둔다.

```bash
cd ~/minji/SID_Project
git status --short                     # 수정된 파일이 없어야 함
git fetch origin claude/preprocess-fixed-cluster experiment/entity-linking-event uniqueness-loss-experiment
git checkout claude/preprocess-fixed-cluster

# 1) 원본 5개 (없는 것만 git 에서 꺼냄) + 해시 확인
mkdir -p data/raw/train data/raw/validation
for f in articles.parquet train/behaviors.parquet train/history.parquet validation/behaviors.parquet validation/history.parquet; do
  [ -f data/raw/$f ] || git cat-file -p origin/experiment/entity-linking-event:data/raw/$f > data/raw/$f
done
(cd data/raw && sha256sum articles.parquet train/*.parquet validation/*.parquet)

# 2) 패키지
pip install gin-config einops accelerate scikit-learn huggingface-hub wandb rich

# 3) 전처리 → data/output/fixedcluster_v1/
export SID_OUTPUT_DIR=~/minji/SID_Project/data/output/fixedcluster_v1
CUDA_VISIBLE_DEVICES=0 python -m src.main preprocess > ~/minji/preprocess_fixedcluster_v1.log 2>&1
tail -3 ~/minji/preprocess_fixedcluster_v1.log      # PREPROCESS_COMPLETE

# 4) 기존 UNI checkpoint 찾기 (sha256 dcb276f8…)
find ~ -name "*.pt" -size +5M 2>/dev/null | xargs -r sha256sum | grep dcb276f8

# 5) (선택) 1 epoch 스모크 테스트
INPUT_DIR=$SID_OUTPUT_DIR/exports/rqvae_inputs SMOKE=1 CUDA_VISIBLE_DEVICES=0 \
  bash experiments/rqvae_uni_fixedcluster_v1/run_aws.sh

# 6) 본 학습
INPUT_DIR=$SID_OUTPUT_DIR/exports/rqvae_inputs A_CKPT=<4번 경로> SEED=42 CUDA_VISIBLE_DEVICES=0 \
  nohup bash experiments/rqvae_uni_fixedcluster_v1/run_aws.sh > ~/minji/rqvae_uni_fixedcluster_s42.log 2>&1 &
tail -f ~/minji/rqvae_uni_fixedcluster_s42.log
```

원본 해시 기대값:

| 파일 | sha256 |
|---|---|
| `articles.parquet` | `b1cea52ee4c152086ff6b70962fe543a43de50c684c1ec0735a4f061bded43d2` |
| `train/behaviors.parquet` | `eeedd1b5cb42e952b7688f3199c5eb97da5596e6245a52714083eea3cb8e3736` |
| `train/history.parquet` | `3f5ac2216f5807677c6225909cef80f1a84d08126c21e3cfeedacb54e3dd323c` |
| `validation/behaviors.parquet` | `7a212113f427d218f3f36b3f52de876fb2868844e9657474b4be07872674c61d` |
| `validation/history.parquet` | `4471f22424011192c892fecf3bb4fa3dcf769ebefd8e9fff05b625c3518ac861` |

- `A_CKPT` 를 주면 기존 UNI checkpoint 안의 `gin_config` 와 이번 gin 을 항목별로 비교한다. 다르면 학습하지 않고 멈춘다. 경로 3개(dataset_folder, save_dir_root, pretrained_rqvae_path)만 비교에서 뺀다.
- 학습 코드는 `~/minji/rqvae_uni_code_127ae7e/` 에 worktree로 꺼낸다. 지금 checkout은 바뀌지 않는다.
- 결과는 `data/output/rqvae/uni_fixedcluster_lu0.05_m0.5_s<SEED>/` 에만 쓴다. 폴더가 이미 있으면 멈춘다.

## 저장되는 것 (결과 폴더)

| 파일 | 언제 | 내용 |
|---|---|---|
| `train.log` | 계속 | 전체 로그 (매 epoch 학습 지표, 10 epoch마다 validation 지표, early stopping 상태) |
| `metrics_train.csv` | 학습 후 (중간에도 `parse_log.py` 로 생성 가능) | **매 epoch**: loss, rec, cb, com, uniq, rqvae, t, global_step |
| `metrics_validation.csv` | 〃 | **평가한 epoch마다** (10, 20, …): loss, rec, cb, com, uniq, rqvae |
| `early_stopping.csv` | 〃 | 평가마다: 개선 여부, counter, 그 시점 best_total / best_rec 와 epoch |
| `summary.json` | 〃 | best_rec / best_total 값과 epoch, 마지막 epoch, 조기 종료 여부 |
| `checkpoint_best_rec.pt` | validation 복원 손실이 좋아질 때마다 덮어씀 | **SID 생성에 쓸 모델** (A 와 같은 기준) |
| `checkpoint_best_total.pt` | validation total loss 가 좋아질 때마다 | 비교용 |
| `checkpoint_epoch_<N>.pt` | 50 epoch마다 | 중간 모델 (재개·분석용) |
| `checkpoint_final.pt` | 끝날 때 | 마지막 epoch 모델 |
| `operative_config.gin`, `rqvae_uni.gin` | 시작 | 실제 적용된 설정 / 입력한 설정 |
| `provenance.txt` | 시작·끝 | 전처리 커밋, 학습 코드 커밋, 입력 폴더, seed, 시작·끝 시각, 종료 코드, (A_CKPT 를 줬으면) A checkpoint 해시 |
| `inputs.sha256`, `inputs_manifest.json` | 시작 | 학습 입력 파일 해시와 전처리 검사 결과 |
| `check_against_a.txt` | 시작 (A_CKPT 줬을 때) | A 와 설정 비교 결과 |

모든 checkpoint 에는 모델·optimizer·epoch·`model_config`·`gin_config`·loss 가중치가 들어 있다.

validation 은 A 와 같게 **10 epoch마다** 평가한다. 매 epoch로 바꾸면 early stopping(patience = 평가 20번)의 의미가 달라져 A 와 다른 학습이 된다. 그래서 바꾸지 않았다. 학습 지표는 매 epoch 남는다.

학습 중 진행 표 보기:
```bash
python experiments/rqvae_uni_fixedcluster_v1/parse_log.py data/output/rqvae/uni_fixedcluster_lu0.05_m0.5_s42
```
- GPU는 1장만 쓴다 (`CUDA_VISIBLE_DEVICES`). 다른 사람이 쓰는 GPU는 피한다.
- 시작할 때 입력을 검사한다. 기대값은 train 10,419 / validation-only 2,741 / train 사건 9,794 이고, 다르면 경고를 출력한다.

## 확인한 것 (이 세션, CPU)

`SMOKE=1 ALLOW_CPU=1` 로 1 epoch을 돌려 다음을 확인했다. `parse_log.py` 와 `check_against_a.py` 도 이 결과로 시험했다 (같은 설정이면 통과, 다른 설정이면 다른 항목만 표시).
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
