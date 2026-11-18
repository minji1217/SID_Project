# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 프로젝트 개요

EB-NeRD small 뉴스 추천 데이터로 기사마다 Semantic ID(SID = c1, c2, c3, c4)를 만들고, 이 SID를 생성하는 Transformer(T5 방식)로 다음 클릭을 예측한다.

파이프라인과 담당:
① 원본 → ② 전처리 [민지, 이 저장소 `src/`] → ③ RQ-VAE [여민] → ④ SID 생성 [여민] → ⑤ 시퀀스 [민지] → ⑥ 1pos4neg [민지] → ⑦ split → ⑧ Transformer → ⑨ 예측·평가 [여민]

- 이 저장소(`minji1217/SID_Project`)는 민지 쪽 저장소다. 지금 작업 브랜치는 `claude/preprocess-fixed-cluster`이고, `main`에는 push하지 않는다.
- 여민의 RQ-VAE·SID 생성·Transformer 코드는 `baekyeomin/SID_Project`에 있다. **읽기·검토 전용**이며, 그 저장소에는 push하지 않는다. 이 저장소의 `RQVAE/`는 예전에 복사해 둔 것이다. 지금 학습 코드는 `uniqueness-loss-experiment` 브랜치 @ `127ae7e`를 기준으로 한다.

## 명령

```bash
pip install -r requirements.txt                 # polars, numpy, torch, transformers
pip install gin-config einops accelerate scikit-learn huggingface-hub wandb rich   # RQ-VAE 쪽

python -m src.main preprocess     # 원본 → data/output/exports/rqvae_inputs/
python -m src.main post-rqvae     # article_semantic_ids.parquet → Transformer 입력
```

- 두 단계는 일부러 나눠 두었다. 새 전처리 결과와 예전 SID가 섞이지 않게 하려는 것이니 자동으로 이어 붙이지 않는다.
- 출력 위치는 `SID_OUTPUT_DIR` 환경변수로 바꾼다. 실험마다 새 폴더를 쓰고 기존 결과는 덮어쓰지 않는다.
- 사건 파라미터는 환경변수로 바꿀 수 있다. `SID_EVENT_SIMILARITY_THRESHOLD`(0.3), `SID_EVENT_TIME_WINDOW_HOURS`(72), `SID_EVENT_MAX_ENTITY_DF_RATIO`(0.01).
- RQ-VAE 학습은 `RQVAE/` 안에서 `python train_rqvae.py <config.gin>`으로 한다. AWS 재학습 절차(입력 해시, A checkpoint와 설정 비교, 결과 폴더)는 `experiments/rqvae_uni_fixedcluster_v1/README.md`와 `run_aws.sh`에 있다.
- 테스트 스위트는 없다. 검증은 `src/validate_data.py`(원본 검사)와 `export_inputs.py`의 export 직전 검사(manifest.json)로 한다.
- 데이터(`data/`)는 git에 없다. 원본 parquet 5개는 `experiment/entity-linking-event` 브랜치에서 꺼낼 수 있다. 최신 fixedcluster·UNI SID는 AWS에만 있다.

## 구조 (여러 파일을 읽어야 보이는 것)

- `main.py`(루트)는 예전 사본이다. 진입점은 `src/main.py`이고 `python -m src.main`으로 실행한다.
- 경로와 상수는 모두 `src/config.py` 한 곳에 있다. `configs/*.yaml`, `check.yaml`은 원본 검사용 컬럼 정의다.
- **Train 기준 고정 원칙:** 카테고리 매핑, 개체 정규화, IDF, high-DF 제외, Train 사건, RQ-VAE 학습은 모두 Train 기사로만 계산한다. Validation으로 다시 계산하지 않는다.
  - Train 기사 = train history ∪ behaviors의 현재 기사·클릭·**노출 목록(inview)**.
- **개체 키** 형식은 `TYPE::surface`이다(`src/entity_processing.py`). `event_entity_linking/`의 `WD::` 키는 별도 실험용이라 preprocess 산출물에 섞으면 export 검사에서 막힌다.
- **Train 사건**(`build_train.build_article_events`): 72시간 안에서 IDF 가중 Jaccard ≥ 0.3이면 union-find로 묶는다. IDF = log((N+1)/(df+1))+1.
- **Validation 사건**(`build_validation._assign_validation_events`): 발행 시각, 그다음 article_id 순으로 처리한다. 동점이면 event_id 숫자가 작은 사건을 고른다.
  - 이 event_id는 **평가 라벨 전용**이다(c2 일관성의 정답 묶음). SID 생성 입력이 아니다.
  - 다만 RQ-VAE 코드가 `event_id` 컬럼을 필수로 요구하므로 컬럼 이름은 유지한다.
- **RQ-VAE 입력 폴더** `exports/rqvae_inputs/`가 전처리와 RQ-VAE 사이의 계약이다. 파일과 검사 항목은 `src/README.md`에 있다.
- **RQ-VAE** 구조: h → r1 = h − Q1[카테고리] → Q2 → c2 → r2 → Q3 → c3. c4는 (c1,c2,c3) 충돌 순번이다.
  - Train 기사의 c2는 기사마다 r1 최근접이다(교수님 결정).
  - Q2는 학습 전 Train 사건별 평균으로 KMeans 초기화한다.
  - Validation-only 기사는 Train 사건에 붙어 그 사건의 C2를 물려받는다.
- 보조 폴더:
  - `event_entity_linking/`, `event_parameter_sweep/`: 사건 정의 실험
  - `experiments/`: AWS 재학습·비교 스크립트
  - `reports/`: 검토 문서(HTML)와 분석 스크립트. 최신 결정 사항은 `reports/review_2026-11-18.md`

## 작업 원칙

- 답변과 문서는 한국어로 쓴다. 문서는 밝은 테마 HTML에 도식을 넣어 `reports/`에 둔다. 그림 안에는 파일명·줄번호 같은 잔글씨를 넣지 않는다.
- 커밋과 push는 사용자가 요청할 때만 하고, 하기 전에 변경 내용을 보여 준다.
- 여민 코드의 새 커밋은 `git fetch` 후 로그로만 확인한다.
- 미정 사항(2026-11-18 기준): 사건 C2와 Q2 KMeans 초기화 공간(실험 R = r1 평균 / 실험 H = h 평균), 갱신 정책 A/B/C. 결정되면 `reports/review_2026-11-18.md`와 v2 수정요청서를 고친다.
