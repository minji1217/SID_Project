# 원 논문 구조 NRMS(정민교 코드)를 EB-NeRD에 적용 (준비만, 아직 실행 전)

9장 EB-NeRD NRMS(`experiments/nrms_ebnerd_v1`, `train_nrms.py`)는 V1 설정에 맞춘 변형이라
원 논문 NRMS 구조와 다르다. MIND와 같은 조건으로 비교하려고 정민교 MIND 코드(AWS `~/minkyo/nrms/`)를 그대로 쓴다.

| 파일 | 출처 |
|---|---|
| `model.py`, `dataset.py`, `train.py` | `~/minkyo/nrms/` 원본 그대로 (md5 831ad4dc…, 3f771074…, b819dbba…) |
| `build_tables_ebnerd.py` | 원본 `build_tables.py`(md5 6bc488c4…)에서 두 곳만 변경: 토큰 규칙 `\w+`(덴마크어), embedding_row 비연속 허용 |
| `evaluate_test.py` | 새로 작성. train.py의 best checkpoint로 validation/test Top-1·MRR·nDCG@5·AUC 계산 (학습 없음) |
| `run_aws.sh` | 테이블 생성 → title / e5 / sid / sid_shuffled × seed 0,1,2 학습(5 epoch, batch 64) → 평가 → seed 평균 요약 |

데이터: UNI SID 1pos4neg (`~/shared/datasets/ebnerd/`, V1·9장과 같은 train / validation half / test),
기사 테이블: `~/shared/raw/ebnerd/` (경로가 다르면 `RAW_DIR`, `SEQ_DIR` 환경변수로 지정). 결과는 `data/output/experiments/nrms_original_ebnerd_v2/`에만 쓴다.

상태: 문법 확인만 했다. 로컬 동작 테스트는 중간에 멈췄고 AWS 실행 전이다.
