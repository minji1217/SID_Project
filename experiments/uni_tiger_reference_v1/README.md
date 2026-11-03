# uni_tiger_reference_v1: 공개 TIGER 재현 모델을 수정 없이 UNI SID로

배경과 결정 사항은 `HANDOFF.md`. 모델·학습·채점 코드는 `../eventc2_tiger_reference_v1/run_reference.py`를 **그대로** 호출한다
(공개 구현 `EdoardoBotta/RQ-VAE-Recommender@957d32b`의 `modules/model.py` 원본, 설정 표는 그 폴더 README).

## UNI에 그대로 써도 되는지 확인한 것 (`run_reference.py`)
- 하드코딩: `N_LAYERS = 3`, `CODEBOOK = 512`. UNI c1 25 / c2 128 / c3 512 → 모두 512 안에 들어간다 (넘으면 스크립트가 멈춤).
- c4(dedup)는 공개 모델이 입력·target에서 빼므로(`_strip_dedup_col`) c4 vocab(UNI max c4 = 16)은 영향 없음.
- EventC2 전용 경로·값은 `run_aws.sh`에만 있었다 → 이 폴더의 `run_aws.sh`가 UNI 경로를 넘긴다.
- `compare_v1_vs_tiger.py` 출력 제목의 "EventC2"는 이 폴더 출력 파일에서만 "UNI"로 바꾼다 (원본 스크립트는 수정 안 함).

## 파일
| 파일 | 내용 |
|---|---|
| `check_uni_sid.py` | sequence 파일들과 SID 파일의 기사별 c1..c4가 같은지 확인 (읽기 전용) |
| `run_aws.sh` | SID 확인 → 공개 TIGER 학습·Test → V1 비교 → seen-in-train 진단 |
| `diagnose_seen_in_train.py` | EventC2 폴더의 같은 진단을 경로 인자로 받게 한 사본 |

## 1단계: 1P4N 이전 UNI train sequences 확보 (HANDOFF 1단계)
(b) 경우 먼저 SID 후보 파일이 `~/shared` UNI 1P4N과 같은지:
```bash
cd ~/minji/SID_Project && python experiments/uni_tiger_reference_v1/check_uni_sid.py \
  --sid /home/ubuntu/article_semantic_ids_final.parquet \
  --seq ~/shared/datasets/ebnerd/validation_sequences_1pos4neg_half.parquet \
  --seq ~/shared/datasets/ebnerd/test_sequences_1pos4neg.parquet \
  --seq ~/shared/datasets/ebnerd/train_sequences_1pos4neg.parquet
```
확보한 train 파일 확인 (SID 일치 + `~/shared` 1P4N train의 impression이 모두 들어 있는지):
```bash
python experiments/uni_tiger_reference_v1/check_uni_sid.py \
  --seq <확보한 train_sequences.parquet> --seq ~/shared/datasets/ebnerd/test_sequences_1pos4neg.parquet \
  --subset ~/shared/datasets/ebnerd/train_sequences_1pos4neg.parquet <확보한 train_sequences.parquet>
```

## 2단계: 실행
```bash
cd ~/minji/SID_Project && UNI_TRAIN=<확보한 train_sequences.parquet> CUDA_VISIBLE_DEVICES=0 \
  nohup bash experiments/uni_tiger_reference_v1/run_aws.sh > data/output/experiments/uni_tiger_reference_v1.log 2>&1 &
```
출력: `data/output/experiments/uni_tiger_reference_v1/seed42/` (run_summary.json, test_metrics.json, test_candidate_scores.parquet,
epoch_history.csv, config.json, checkpoint_best.pt), `compare_v1_vs_reference_seed42.md/.json`, `diagnose_seen_in_train.txt`.
출력 폴더가 이미 있으면 시작하지 않는다. 다른 seed는 `SEED=123`.

## 상태
코드 준비 (합성 데이터로 동작 확인). AWS 미실행.
