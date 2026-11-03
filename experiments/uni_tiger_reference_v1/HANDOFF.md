# 인수인계: 공개 TIGER 모델을 UNI SID(EB-NeRD)로 돌리기

새 채팅에서 이 문서를 먼저 읽고 진행한다. 사용자(정민지)는 한국어로 대화하고, AWS 명령은 사용자가 직접 실행한다(채팅 세션은 AWS에 접근할 수 없음). 화면 결과는 사용자가 붙여 넣는다.

## 목표
공개 TIGER 재현 구현 [EdoardoBotta/RQ-VAE-Recommender](https://github.com/EdoardoBotta/RQ-VAE-Recommender) @ `957d32b`(MIT)의 **모델을 수정 없이** UNI SID로 학습하고,
같은 test(122,668행, 1P4N)에서 V1과 NRMS와 비교한다. RQ-VAE와 SID는 건드리지 않는다.

| 비교 대상 (UNI SID, test 122,668행) | Top-1 | AUC | 비고 |
|---|---|---|---|
| Transformer V1 | 27.07% | 0.579 | seed 42 |
| V1 + level 가중치 (1, 0.5, 0.1) | 29.46% | 0.603 | seed 42, 재학습 없음 |
| NRMS 원 논문 구조 + SID | 26.09% | 0.571 | seed 42/123/2026 평균 |
| NRMS 원 논문 구조 + E5 | 29.92% | 0.609 | 3-seed |
| (참고) 우리 TIGER-style, **EventC2** SID | 16.77% | 0.469 | 철회. SID·test 세트가 달라 직접 비교 불가 |

예상: 우리 TIGER-style이 실패한 원인은 구현이 아니라 뉴스의 성질(test 정답의 1.7%만 train에서 클릭된 적 있음, 오답은 12.1%)이었다.
공개 구현으로도 비슷할 가능성이 높지만, “공개 원본 모델로도 같다”를 확인하는 것이 이번 목적이다.

## 이미 있는 코드 (저장소 `minji1217/SID_Project`, 브랜치 `claude/zealous-cerf-c1onwi`)
- `experiments/eventc2_tiger_reference_v1/`: 공개 구현을 EventC2 데이터에 쓰도록 준비만 한 것 (미실행)
  - `reference/`: 공개 저장소 `modules/model.py`, `data/schemas.py`, `modules/scheduler/` 원본 복사본 (byte 동일)
  - `run_reference.py`: 우리 loader로 `TokenizedSeqBatch`를 만들어 공개 모델 학습, 후보 5개를 sum log P(c1..c3)로 채점, validation Top-1 best → test
  - `run_aws.sh`: EventC2 경로 기준 (`EXP_ROOT` 환경변수로 바뀜)
  - README에 공개 코드를 그대로 못 쓰는 부분과 처리 방식 정리 (codebook 크기 512로 통일, 1P4N 채점 등)
- `experiments/eventc2_tiger_generative_v1/`: 우리 TIGER-style 실험 (`train_tiger.py`의 loader, `compare_v1_vs_tiger.py`, `diagnose_seen_in_train.py`)
- 이번 작업 폴더: `experiments/uni_tiger_reference_v1/` (이 문서만 있음). EventC2 폴더는 수정하지 말고 여기에 UNI용 실행 스크립트를 만든다.

## 데이터 (AWS)
| 용도 | 경로 | 상태 |
|---|---|---|
| validation half / test (UNI 1P4N, 후보 순서 섞음) | `~/shared/datasets/ebnerd/validation_sequences_1pos4neg_half.parquet`, `test_sequences_1pos4neg.parquet` | 있음 (읽기 전용) |
| train 1P4N | `~/shared/datasets/ebnerd/train_sequences_1pos4neg.parquet` | 있음. 단 TIGER 학습 입력으로는 **쓰지 않기로 결정** |
| **TIGER 학습 입력: 1P4N 이전 UNI train sequences** | 없음 | **먼저 확보해야 함** (아래 1단계) |
| UNI V1 test 예측 | `~/minji/SID_Project/data/output/experiments/uni_v1_test_eval/test/test_candidate_scores.parquet` | 있음 |
| UNI V1 run (설정) | `~/minji/SID_Project-transformer/Transformer/sweep_out/ebnerd_v2/shuffle_check/seed42/` (`run_summary.json`의 `gin_config`) | 있음 |
| 원본 클릭 기록 | `~/minji/SID_Project/data/raw/{train,validation}/{behaviors,history}.parquet` | 있음 (EventC2 때 사용) |
| UNI SID 후보 | `/home/ubuntu/article_semantic_ids_final.parquet` (9장 NRMS가 UNI 데이터와 함께 씀) | 있음. UNI인지 확인 필요 |

**왜 1P4N train을 안 쓰나**: 1P4N은 SID 기준으로 정답과 같은 c1·c2·c3 오답을 지우고, 오답이 4개 미만이면 그 클릭을 버린다. TIGER는 “다음 클릭 기사”를 정답으로 배우므로 거르기 전 데이터에서 정답을 가져온다(사용자 결정, EventC2 TIGER 실험과 같은 방식).

## 진행 순서
1. **1P4N 이전 UNI train sequences 확보**
   - (a) 사용자 PC에서 찾기: cmd에서 `dir /s /b C:\Users\samsung\train_sequences.parquet` → 경로에 `uni`/`normalize_v2`가 있으면 scp로 AWS `~/minji/`에 업로드
     (`scp -i "C:\Users\samsung\Downloads\agent-key.pem" <파일> ubuntu@<서버IP>:~/minji/...`, 서버 IP는 접속할 때마다 바뀜)
   - (b) PC에도 없으면 AWS에서 다시 만들기: 먼저 `/home/ubuntu/article_semantic_ids_final.parquet`의 c1·c2·c3가 `~/shared` UNI 1P4N 파일의 history/candidate c1~c3와 기사마다 일치하는지 확인하는 스크립트를 만들어 실행. 일치하면
     `SID_OUTPUT_DIR=<새 출력 폴더> python -m src.main post-rqvae --sid-path <UNI SID> --experiment <새 실험 이름>`
     (`experiments/eventc2_retrain_v1/run_pipeline.sh` 2단계와 같은 명령). 기존 폴더에 쓰지 않도록 새 실험 이름 사용.
   - 확보한 파일로 1P4N을 다시 만들었을 때 `~/shared` 1P4N과 행 수(232,874)가 같은지 확인하면 더 확실하다(선택).
2. `experiments/uni_tiger_reference_v1/run_aws.sh` 작성: `eventc2_tiger_reference_v1/run_reference.py`를 그대로 호출하되 경로만 UNI로
   (`--train` = 1단계 파일, `--val`/`--test` = `~/shared` UNI 1P4N, 출력은 `~/minji/SID_Project/data/output/experiments/uni_tiger_reference_v1/seed42/`).
   비교는 `eventc2_tiger_generative_v1/compare_v1_vs_tiger.py --v1-scores <UNI V1 test 예측> --v1-run <UNI V1 run> --tiger-run <출력>`.
   `run_reference.py`가 c4 vocab 등 EventC2 전용 값을 하드코딩했는지 먼저 확인할 것 (UNI max c4 = 16, c1 25 / c2 128 / c3 512).
3. 로컬에서 합성 데이터로 동작 확인 후 커밋·푸시 → 사용자가 AWS에서 `nohup … &` + `tail -f`로 실행 (사용자는 tmux 대신 nohup을 선호).
4. 결과: test Top-1 / MRR / nDCG@5 / AUC, V1과 같은 sample인지 확인, `diagnose_seen_in_train.py`로 정답·오답의 “train에서 본 적 있음” 비율.
5. 결과 파일을 사용자가 PC로 받아(scp) 채팅에 올리면 `experiments/uni_tiger_reference_v1/results_aws/`에 커밋.

## 지켜야 할 것
- 기존 결과·파일·실험 폴더는 수정하거나 덮어쓰지 않는다. 출력 폴더가 이미 있으면 멈추게 만든다.
- `~/shared`, `~/yeomin`, `~/minkyo`에는 쓰지 않는다. RQ-VAE·SID는 바꾸지 않는다.
- 커밋·푸시는 `claude/zealous-cerf-c1onwi` 브랜치에만. PR은 만들지 않는다.
- 성능은 test로 보고하고, validation은 checkpoint 선택에만 쓴다. seed는 우선 42 (필요하면 123, 2026).
- 실행 중에는 AWS에서 `git pull`하지 않는다 (bash가 실행 중인 스크립트를 이어 읽음).
