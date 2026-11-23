# EB-NeRD 전처리 (`python -m src.main preprocess`)

원본 `data/raw/` → RQ-VAE 입력 폴더 `data/output/exports/rqvae_inputs/` 까지 한 번에 만든다.
중간 산출물은 `data/output/model_inputs/` 에 남는다 (`SID_OUTPUT_DIR` 로 출력 위치 변경 가능).

## 실행 순서

| STEP | 함수 | 산출물 (`model_inputs/`) |
|---|---|---|
| 1 | `validate_data.run_validation` | `reports/*.json` |
| 2 | `build_train.build_valid_articles` | `articles_base.parquet` |
| 3 | `build_train.collect_train_used_article_ids` | `train_used_article_ids.parquet` |
| 4 | `build_train.build_category_mapping` | `category_mapping.parquet` |
| 5 | `build_train.apply_category_mapping_to_articles` | `articles_with_category.parquet` |
| 5-1 | `entity_processing.build_article_entities` | `article_entities.parquet`, `entity_normalization_map.parquet` |
| 6 | `build_article_embedding_input`, `generate_article_embeddings` | `article_embedding_input.parquet`, `article_embeddings.npy` |
| 7 | `build_train.build_article_events` | `article_events.parquet`, `event_master.parquet`, `entity_idf.parquet`, `entity_idf_meta.json` |
| 8 | `build_article_master.build_train_article_master` | `article_master.parquet` |
| 10 | `build_validation.build_validation` | `validation_*_article_ids.parquet`, `validation_article_events.parquet`, `event_master_with_validation.parquet`, `validation_article_master.parquet` |
| 12 | `export_inputs.export_rqvae_inputs` | `exports/rqvae_inputs/` |

## Train 기사 (STEP 3)

`train/history.parquet` 의 `article_id_fixed`
∪ `train/behaviors.parquet` 의 현재 기사 `article_id`, 클릭 `article_ids_clicked`, **노출 목록 `article_ids_inview`**
∩ 유효 기사.

노출 목록을 넣는 이유는 두 가지다.
- Transformer는 Train에서도 후보 5개를 비교한다. `build_sequences.py` 는 SID가 없는 후보가 하나라도 있으면 impression을 버린다.
- 노출 목록이 빠지면 Train 기간에 노출된 기사가 validation-only 로 잘못 분류된다.

클릭 여부(label)는 기사 집합을 정하는 데만 쓰인다. 사건·IDF·SID 계산에는 쓰이지 않는다.

카테고리 매핑, 개체 정규화 mapping, IDF / high-DF, Train 사건은 모두 **Train 기사로만** 계산하고 고정한다.
Validation 데이터로 다시 계산하지 않는다.

## 유효 기사 (STEP 2)

다음 중 하나라도 해당하면 **기사 자체를 제외**한다. 사건 계산에서만 빼는 것이 아니다.
- `article_id` null 또는 중복
- 제목·부제가 모두 빔
- `published_time` null
- `ner_clusters` / `entity_groups` 길이 불일치

발행 시각이 없으면 시간순 처리(사건, Validation SID 배정)가 불가능하다.
NER 길이가 다르면 개체와 TYPE 의 대응을 믿을 수 없다.

## 사건 (STEP 7)

Train 기사만 사용한다. 기준은 72시간 창, IDF 가중 Jaccard ≥ 0.3, df ≥ 1% 개체 제외, union-find 다.
값은 `src/config.py` 의 `SID_EVENT_SIMILARITY_THRESHOLD` / `SID_EVENT_TIME_WINDOW_HOURS` / `SID_EVENT_MAX_ENTITY_DF_RATIO` 로 바꿀 수 있다.
개체 키는 `src/entity_processing.py` 의 `TYPE::surface` 형식이다 (예: `PER::mette frederiksen`).

`event_entity_linking/` (GPT/Wikidata 연결, `WD::` 키)은 별도 실험용이다. preprocess 산출물에는 쓰지 않는다.

## Validation 사건 배정 = 평가 전용 라벨

STEP 10의 Validation 사건 배정(validation-origin 사건 생성 포함), `event_master_with_validation.parquet`,
`validation_article_master.parquet` 의 `event_id` 는 **평가·분석용 라벨**이다.
Validation-only 기사의 SID 생성 입력으로 쓰지 않는다.

`event_id` 컬럼 이름은 RQ-VAE 코드가 필수로 요구하므로 바꾸지 않는다.

Validation-only 기사의 SID는 RQ-VAE 학습 후 SID 생성 단계(`generate_semantic_ids_timeupdate`)에서 만든다.
발행 시각 순으로 처리하며 기존 Train 사건에만 배정한다.
1. 72시간 창 안 + IDF 가중 Jaccard ≥ 0.3
2. 1이 실패하면 h(a) 와 Train 사건 평균 z(E) 의 코사인

이 단계는 아래 Train 전용 파일만 읽는다.

## RQ-VAE 입력 폴더 `exports/rqvae_inputs/` (= dataset_folder)

| 파일 | 내용 |
|---|---|
| `article_master.parquet` | Train 기사. `article_id, embedding_row, model_category_id, event_id, published_time, model_text`. RQ-VAE 학습에 쓰는 유일한 파일 |
| `validation_article_master.parquet` | Validation-only 기사. 같은 컬럼이며 `event_id` 는 평가 전용 |
| `article_embeddings.npy` | 전체 유효 기사 E5 임베딩 (`embedding_row` 로 조회) |
| `event_master.parquet` | Train 사건만 (`event_entities`, `event_last_added_time` 등). Validation 반영본 아님 |
| `entity_idf.parquet` | `entity, document_frequency, document_frequency_ratio, idf, is_high_df` |
| `entity_idf_meta.json` | `{"train_article_count": N, "unseen_entity_idf": log(N+1)+1, ...}` |
| `article_entities.parquet` | `(article_id, canonical_entity_key)` 고유 쌍. Train + Validation-only 기사 |
| `manifest.json` | 파일 목록과 export 직전 검사 결과 |

export 직전 검사 항목은 다음과 같다.
- `article_embeddings.npy` 행 수 == 임베딩 입력 기사 수
- Train / Validation-only 기사가 겹치지 않음
- `event_master` 가 Train 사건만 가짐
- 개체 키가 모두 `TYPE::surface` 형식 (`WD::` 혼용 금지)
- Train 기사 개체 키 집합 == `entity_idf.entity`
- `idf` 가 `entity_idf_meta.train_article_count` 로 다시 계산한 값과 같음

## SID를 받은 뒤: 시퀀스 → 1pos4neg → 후보 순서 섞기

RQ-VAE·SID 생성 결과(`article_semantic_ids.parquet`)를 받으면 실험 폴더 하나에 차례로 만든다. 기존 결과는 덮어쓰지 않는다.

```bash
EXP=<실험 이름>          # 예: uni_fixedcluster_h_s42
E=data/output/experiments/$EXP

# 1) 시퀀스 (train / validation)
python -m src.main post-rqvae --experiment $EXP --sid-path <article_semantic_ids.parquet>

# 2) 1pos4neg: 정답마다 한 행, 오답은 label 0만, 정답과 (c1,c2,c3) 같은 오답 제거, 오답끼리 SID 중복 제거, 4개 미만이면 버림, seed 42 해시로 4개
python -m src.build_1pos4neg_sequences --out-dir $E/candidate_1pos4neg \
  --path $E/post_rqvae/train_sequences.parquet --path $E/post_rqvae/validation_sequences.parquet \
  --report $E/candidate_1pos4neg/report.json

# 3) 후보 순서만 섞기 (정답이 항상 첫 자리면 argmax 동점이 정답으로 처리되어 Top-1이 부풀기 때문)
python -m src.shuffle_candidate_order --input-dir $E/candidate_1pos4neg \
  --output-dir $E/candidate_1pos4neg_shuffled --report $E/candidate_1pos4neg_shuffled/report.json
```

- `candidate_1pos4neg_shuffled/`의 train / validation 파일을 Transformer에 넘긴다. validation을 앞 절반(모델 고르기)과 test로 나누는 건 Transformer 쪽 `split_validation.py`.
- `analyze_candidate_sid_collision.py`는 1pos4neg가 불러 쓰는 SID 로딩·충돌 분석 함수다.
- 세 스크립트는 `claude/festive-bell-a5sfbx`(정책 2026-09-18 확정)에서 그대로 가져왔다. 2026-11-23 여민 SID 생성 코드의 시험 출력으로 1)→3)이 끝까지 돌고 섞기 검증이 모두 통과하는 것을 확인했다.
