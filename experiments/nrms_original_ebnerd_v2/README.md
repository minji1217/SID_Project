# 원 논문 구조 NRMS(정민교 코드)를 EB-NeRD에 적용

9장 EB-NeRD NRMS(`experiments/nrms_ebnerd_v1`, `train_nrms.py`)는 V1 설정에 맞춘 변형이라
원 논문 NRMS 구조와 다르다. MIND와 같은 조건으로 비교하려고 정민교 MIND 코드(AWS `~/minkyo/nrms/`)를 그대로 쓴다.

| 파일 | 출처 |
|---|---|
| `model.py`, `dataset.py`, `train.py` | `~/minkyo/nrms/` 원본 그대로 (md5 831ad4dc…, 3f771074…, b819dbba…) |
| `build_tables_ebnerd.py` | 원본 `build_tables.py`(md5 6bc488c4…)에서 데이터 읽기만 변경: 토큰 규칙 `\w+`(덴마크어), embedding_row 비연속 허용, sequence에서 SID 모을 때 나눠 읽기(AWS 메모리 부족 해결, 결과는 원본과 같음을 확인) |
| `evaluate_test.py` | 새로 작성. train.py의 best checkpoint로 validation/test Top-1·MRR·nDCG@5·AUC 계산 (학습 없음) |
| `run_aws.sh` | 테이블 생성 → title / e5 / sid / sid_shuffled × seed 42,123,2026 학습(5 epoch, batch 64) → 평가 → seed 평균 요약 |

데이터: UNI SID 1pos4neg (`~/shared/datasets/ebnerd/`, V1·9장과 같은 train / validation half / test),
기사 테이블: `~/shared/raw/ebnerd/` (경로가 다르면 `RAW_DIR`, `SEQ_DIR` 환경변수로 지정). 결과는 `data/output/experiments/nrms_original_ebnerd_v2/`에만 쓴다.

## 결과 (AWS 실행 완료, `results_aws/`)

test 122,668행, seed 42 / 123 / 2026 평균. 12개 run 모두 `metrics_*.json`이 학습 로그(`log_*.txt`)의 best val / test Top-1과 일치.
checkpoint(`nrms_*_best.pt`)는 AWS `~/minji/SID_Project/data/output/experiments/nrms_original_ebnerd_v2/`에만 있다.

| 모드 | Val Top-1 | Test Top-1 (seed 표준편차) | MRR | nDCG@5 | AUC |
|---|---|---|---|---|---|
| e5 | 28.82% | **29.92%** (1.02) | 0.5430 | 0.6560 | 0.6092 |
| sid | 25.92% | **26.09%** (0.10) | 0.5107 | 0.6313 | 0.5711 |
| title (원 논문 NRMS) | 24.99% | **26.06%** (0.22) | 0.5119 | 0.6322 | 0.5743 |
| sid_shuffled (대조군) | 20.99% | **20.71%** (0.40) | 0.4647 | 0.5959 | 0.5128 |

- seed별 test Top-1: e5 30.13 / 28.81 / 30.82, sid 26.20 / 26.01 / 26.06, title 26.14 / 25.81 / 26.22, sid_shuffled 20.26 / 20.86 / 21.02
- best epoch: e5 5 / 5 / 4 (5 epoch 끝까지 오르는 중), sid 2 / 1 / 3, title 2 / 5 / 3, sid_shuffled 전부 1
- SID − 섞은 SID = +5.38%p, SID ÷ E5 = 87.2%, SID ≈ title(원 논문 NRMS)
- 비교: 9장 NRMS(`nrms_ebnerd_v1`, V1 설정 변형) E5 29.24% / SID 28.20% (96.5%), MIND 원본 NRMS(validation) E5 38.93% / SID 38.12% (97.9%)
