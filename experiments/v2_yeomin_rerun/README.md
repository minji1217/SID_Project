# v2 재실험: 여민 코드 + 우리 설정 (실험 V1 재실행 · H · R)

여민 코드(`baekyeomin/SID_Project` @ `22028ba`, 2026-11-23)로 RQ-VAE부터 Transformer까지 다시 돌린다. 설정은 우리 것:
- RQ-VAE: UNI(잠재 128, c3 512, λuniq 0.05, margin 0.5, early stopping, best_rec)
- Transformer: V1(history 50, SEP 없음, d256, 8 head, 2층, lr 5e-5, wd 0, dropout 0, fp32)
- 갱신 정책 B, SID는 seed 42 RQ-VAE 모델로

## 실험

| 실험 | Q2 초기화 | validation-only 사건 C2 | 사건 붙이기 | 돌리는 단계 |
|---|---|---|---|---|
| V1 재실행 (기준) | — | — | — | ⑤만 (예전 UNI SID, 공유 폴더 1pos4neg) |
| H | 사건별 h 평균 | 사건 h 평균 → 가장 가까운 Q2 | Jaccard 0.3·72h → h 코사인 | ①~⑤ |
| R | 사건별 r1 평균 | 사건 r1 평균 → 가장 가까운 Q2 | 같음 (h 코사인) | ①~⑤ |

H와 R은 따로 학습한 모델이라 h도 달라서, 코사인으로 붙는 사건이 같지는 않다. 같은 방식으로 고를 뿐이다.

## 스크립트

| 파일 | 단계 | 코드 |
|---|---|---|
| `0_setup.sh` | 여민 코드 꺼내기, 입력 확인 | — |
| `1_train_rqvae.sh` | ① RQ-VAE 학습 (`VARIANT`, `SEED`) | 여민 학습 코드 + UNI gin (`make_gin.py`) |
| `2_gen_sid.sh` | ② SID 생성 (`VARIANT`) + `check_sid.py` | 여민 SID 코드 (hver / r1ver) |
| `3_build_candidates.sh` | ③ 시퀀스 → 1pos4neg → 섞기 → ④ split (`VARIANT`) + `check_candidates.py` | 우리 `src/` + 여민 `split_validation.py` |
| `4_transformer.sh` | ⑤ 학습 → test 예측 → 지표 (`RUN`, `SEED`) | 여민 Transformer + 우리 V1 gin |
| `collect_results.py` | 결과 표 | — |

공통: 결과 폴더가 이미 있으면 멈춘다(덮어쓰지 않음). 폴더마다 `provenance.txt`(코드 커밋, 입력, 시작·끝 시각, 종료 코드)를 남긴다.

## 결과 폴더 (`PRE = ~/minji/SID_Project/data/output/fixedcluster_v1`)

```
$PRE/v2/rqvae/{h,r}_s{42,123,2026}/    checkpoint_best_rec.pt, train.log, metrics_*.csv, summary.json, rqvae.gin
$PRE/v2/sid/{h,r}/                     article_semantic_ids.parquet, 매칭 결과, check_sid.txt
$PRE/v2/sid/_check_r1code_with_h_ckpt/ R1 코드를 H checkpoint로 돌린 확인용
$PRE/experiments/v2_{h,r}/             post_rqvae/, candidate_1pos4neg/, candidate_1pos4neg_shuffled/, split/
$PRE/v2/transformer/{v1,h,r}_seed{42,123,2026}/   transformer.gin, checkpoint_best.pt, test_candidate_scores.parquet, test_metrics.json
```

## AWS 명령 (한 줄씩)

GPU는 `nvidia-smi`로 빈 번호를 확인해 `CUDA_VISIBLE_DEVICES`에 넣는다. 긴 작업은 `nohup ... &`로 돌리고 `tail -f`로 본다.

**0. 준비**
```bash
cd ~/minji/SID_Project
git pull
bash experiments/v2_yeomin_rerun/0_setup.sh
```

**① RQ-VAE (6번: H·R × seed 3개)**. H seed 42를 먼저 돌려 우리 ver2 s42와 비교한다.
```bash
VARIANT=h SEED=42 A_CKPT=$HOME/minji/checkpoint_best_rec.pt CUDA_VISIBLE_DEVICES=0 nohup bash experiments/v2_yeomin_rerun/1_train_rqvae.sh > ~/minji/v2_rqvae_h_s42.log 2>&1 &
tail -f ~/minji/v2_rqvae_h_s42.log
cat data/output/fixedcluster_v1/v2/rqvae/h_s42/summary.json
cat data/output/rqvae/uni_fixedcluster_lu0.05_m0.5_s42/summary.json
```
두 `best_rec`·`best_rec_epoch`가 거의 같으면 여민 코드가 맞게 옮겨진 것. 그다음 나머지:
```bash
VARIANT=r SEED=42 A_CKPT=$HOME/minji/checkpoint_best_rec.pt CUDA_VISIBLE_DEVICES=1 nohup bash experiments/v2_yeomin_rerun/1_train_rqvae.sh > ~/minji/v2_rqvae_r_s42.log 2>&1 &
VARIANT=h SEED=123 CUDA_VISIBLE_DEVICES=0 nohup bash experiments/v2_yeomin_rerun/1_train_rqvae.sh > ~/minji/v2_rqvae_h_s123.log 2>&1 &
VARIANT=r SEED=123 CUDA_VISIBLE_DEVICES=1 nohup bash experiments/v2_yeomin_rerun/1_train_rqvae.sh > ~/minji/v2_rqvae_r_s123.log 2>&1 &
VARIANT=h SEED=2026 CUDA_VISIBLE_DEVICES=0 nohup bash experiments/v2_yeomin_rerun/1_train_rqvae.sh > ~/minji/v2_rqvae_h_s2026.log 2>&1 &
VARIANT=r SEED=2026 CUDA_VISIBLE_DEVICES=1 nohup bash experiments/v2_yeomin_rerun/1_train_rqvae.sh > ~/minji/v2_rqvae_r_s2026.log 2>&1 &
```

**② SID 생성** (H·R seed 42 학습이 끝난 뒤. H 먼저)
```bash
VARIANT=h CUDA_VISIBLE_DEVICES=0 bash experiments/v2_yeomin_rerun/2_gen_sid.sh > ~/minji/v2_sid_h.log 2>&1
VARIANT=r CUDA_VISIBLE_DEVICES=0 bash experiments/v2_yeomin_rerun/2_gen_sid.sh > ~/minji/v2_sid_r.log 2>&1
cat data/output/fixedcluster_v1/v2/sid/r/check_sid.txt
```
확인: 기사 13,160 · SID 중복 0 / `코드 확인 ... 같은 사건에 붙은 비율 1.0000` / 매칭 방법별 건수와 `match_cosine` 분포 / 사건 C2 종류와 가장 많은 C2 비율 (한 C2가 절반을 넘으면 주의).

**③④ 시퀀스 → 1pos4neg → 섞기 → split**
```bash
VARIANT=h bash experiments/v2_yeomin_rerun/3_build_candidates.sh > ~/minji/v2_cand_h.log 2>&1
VARIANT=r bash experiments/v2_yeomin_rerun/3_build_candidates.sh > ~/minji/v2_cand_r.log 2>&1
tail -n 5 ~/minji/v2_cand_h.log
```
확인: 세 파일 모두 후보 5개·정답 1개 True, 정답 첫 자리 비율 약 0.2.

**⑤ Transformer (V1 재실행 · H · R × seed 3개 = 9번)**
```bash
RUN=v1 SEED=42 CUDA_VISIBLE_DEVICES=0 nohup bash experiments/v2_yeomin_rerun/4_transformer.sh > ~/minji/v2_tf_v1_s42.log 2>&1 &
RUN=h SEED=42 CUDA_VISIBLE_DEVICES=1 nohup bash experiments/v2_yeomin_rerun/4_transformer.sh > ~/minji/v2_tf_h_s42.log 2>&1 &
RUN=r SEED=42 CUDA_VISIBLE_DEVICES=2 nohup bash experiments/v2_yeomin_rerun/4_transformer.sh > ~/minji/v2_tf_r_s42.log 2>&1 &
```
seed 123, 2026도 `SEED`와 로그 이름만 바꿔서 같은 방식으로.

**결과 표**
```bash
python experiments/v2_yeomin_rerun/collect_results.py data/output/fixedcluster_v1/v2
```

## 이 컨테이너에서 확인한 것 (2026-11-23, CPU 시험)

`PRE_DIR=data/output`(로컬 전처리), 여민 코드 `22028ba`로 ①~⑤를 끝까지 돌림.
- ① H·R 1 epoch: 종료 0, `lambda_uniq 0.05`, R은 r1 초기화 로그, `checkpoint_best_rec.pt` 생성
- ② H·R: 13,160개·SID 중복 0·72h 위반 0. R1 코드 + H checkpoint = H와 같은 사건 100%
- ③④ H: train 121,209 / validation 앞 절반 72,197 / test 72,298, 정답 첫 자리 비율 0.200~0.201
- ⑤ H (데이터 일부·1 epoch): 학습·예측·지표까지 정상, history 50, 어휘 크기 자동
- 1 epoch 모델이라 SID와 지표 값 자체는 의미 없음. 시험용 H 모델에서는 Train 사건 9,794개가 모두 같은 C2 하나를 받았다 → 실제 모델에서 `check_sid.txt`의 "사건 C2 종류"를 꼭 확인

## 인기·나이 함께 학습 (여민 코드 + 우리 패치)

여민에게 요청한 수정을 우리가 `transformer_features.patch`로 구현했다. 여민 저장소와 `~/minji/yeomin_code_22028ba`는 건드리지 않고, `8_transformer_features.sh`가 여민 코드를 `~/minji/yeomin_code_22028ba_features`에 한 번 더 복사해 패치를 적용한다.

- 후보 점수 = SID 점수 + w1·`candidate_logpop` + w2·`candidate_logage`. w는 같이 학습(시작 5, −1 = 재학습 없는 결합에서 validation으로 고른 값, lr 0.01 따로, weight decay 0, gradient clip에서 제외).
- 패치가 바꾸는 파일: `data/sequence.py`(특징 컬럼 읽기), `modules/model.py`(가중치·점수), `train_transformer.py`(특징 전달·optimizer 그룹·epoch마다 가중치 출력), `predict_sid.py`(특징 전달·`sid_score` 저장).
- 옵션을 안 켜면 기존과 같다: 같은 데이터·seed로 원래 코드와 loss·validation Top-1·test 점수가 비트 단위로 같았다(2026-11-23, CPU 시험).

```bash
python experiments/v2_yeomin_rerun/add_candidate_features.py --run h        # 특징 파일 (v1, r 도)
RUN=h SEED=42 CUDA_VISIBLE_DEVICES=0 nohup bash experiments/v2_yeomin_rerun/8_transformer_features.sh > ~/minji/v2_tffeat_h_s42.log 2>&1 &
```
결과: `$V2/transformer_feat/<RUN>_seed<SEED>/test_metrics.json`. 재학습 없는 결합(`popularity_fusion.py --with-age`)의 "결합+나이"와 비교한다.
