# EB-NeRD 전처리: c1 / c2 그룹 생성

RQ-VAE 학습 **이전** 전처리(`src/`) 단계에서 Semantic ID의 c1, c2 코드북 그룹 기준을 만드는 방식을 정리한 문서이다.
RQ-VAE 내부(코드북 초기화, SID 생성)는 다루지 않는다.

| 코드 | 전처리 결과 컬럼 | 그룹 기준 |
|---|---|---|
| c1 | `model_category_id` | 원본 `category_str` (train 기준 정수 매핑) |
| c2 | `event_id` | Entity 기반 사건 클러스터 (IDF-weighted Jaccard + 시간창 + Union-Find) |

## 0. 실행 순서

`src/run_article_build.py`의 `main()`:

```
build_valid_articles()               # STEP 2  유효 기사 선별 → articles_base.parquet
collect_train_used_article_ids()     # STEP 3  train 사용 기사 ID
build_category_mapping()             # STEP 4  ┐ c1
apply_category_mapping_to_articles() # STEP 5  ┘
build_article_embedding_input()      # STEP 6
generate_article_embeddings()        # STEP 6  (E5, 768d)
build_article_events()               # STEP 7  c2 (train event)
build_train_article_master()         # STEP 8
```

이후 `src/main.py`에서 `build_validation()`(validation event 배정)과 `export_rqvae_inputs()`를 호출한다.

**공통 원칙**: c1·c2 매핑은 모두 **train에서 실제 사용하는 기사만으로 fit**하고, validation에는 적용만 한다.

> 선행 조건: c2에서 사용하는 `article_entities.parquet`은 `build_article_entities()`가 만든다.
> 이 함수는 `run_article_build`에 포함되어 있지 않으므로 `src/run_entity_experiment.py`로 먼저 생성해야 한다.

### Train 사용 기사 (`collect_train_used_article_ids`, `src/build_train.py:276`)

- train history의 `article_id_fixed`
- 사용 가능한 클릭 behavior(7개 조건 통과)의 현재 기사와 클릭 target 기사
- 위 합집합 ∩ `articles_base`의 유효 article_id

---

## 1. c1: 카테고리 그룹 (`model_category_id`)

### 1-1. 매핑 생성 — `build_category_mapping()` (`src/build_train.py:601`)

1. train 사용 기사에 `category_str` 연결
2. 정리: `cast(Utf8)` → `fill_null("")` → `strip` (`" sport "`와 `"sport"`는 같은 카테고리)
3. 비어 있지 않은 카테고리만 `unique` 후 **문자열 기준 정렬**
4. 정렬 순서대로 **1부터 연속 ID** 부여, **`<UNK>` = 0** 예약
   - 원본에 `<UNK>` 문자열이 있으면 `ValueError`
5. 저장: `category_mapping.parquet`

```
category_str | model_category_id
<UNK>        | 0
culture      | 1
economy      | 2
...
```

### 1-2. 전체 기사 적용 — `apply_category_mapping_to_articles()` (`src/build_train.py:805`)

- 전체 유효 기사의 정리된 `category_str` 기준으로 left join
- 매칭되지 않으면 **0** (빈 카테고리, train에 없던 카테고리)
- 리포트: 빈 카테고리 기사 수, train에 없던 카테고리의 기사 수·종류 수·예시(최대 10개)
- validation도 같은 frozen 매핑을 사용한다 (`src/main.py:390`).

카테고리 병합이나 희소 카테고리 통합은 하지 않는다. 원본 카테고리 1개 = c1 그룹 1개.

---

## 2. c2: 사건 그룹 (`event_id`)

### 2-1. Entity 표현 (`src/entity_processing.py`)

설정 (`src/config.py:218`):

```python
ENTITY_PROCESSING_MODE = "normalize_only"
ENTITY_NORMALIZATION_VERSION = "v2"
```

사건 클러스터링은 `load_canonical_entity_lookup()`으로 `article_id → set[canonical_entity_key]`를 읽어 사용한다.

| 단계 | 처리 | 예 |
|---|---|---|
| baseline | NFKC → 공백 축약 → lowercase, `TYPE::text` 키 | `PER::vladimir putin` |
| v1 safe possessive | 끝 `-s` 제거해 base로 병합 | `ORG::ekstra bladets` → `ORG::ekstra bladet` |
| v2 safe hyphen | hyphen을 공백으로 바꿔 병합 | `ORG::jyllands - posten` → `ORG::jyllands posten` |

v1 guardrail:
- PER / ORG / LOC만 대상
- base가 같은 TYPE의 train vocabulary에 실제로 존재
- base 길이 ≥ 3, base df ≥ 2, base df ≥ variant df
- (`Andreas → Andrea`, `Sky Sports → Sky Sport` 같은 오병합 방지)

v2 guardrail:
- 같은 TYPE 내에서만 매핑
- 변환된 target이 같은 TYPE의 train vocabulary에 존재
- `midt -`처럼 시작이나 끝에 매달린 hyphen은 제외
- 끝 구두점(`allan j.`)은 처리하지 않음

normalization mapping은 train 사용 기사 mention으로만 fit하고 전체 유효 기사에 적용한다.

### 2-2. Train event 생성 — `build_article_events()` (`src/build_train.py:1884`)

하이퍼파라미터 (`src/config.py:139`, 환경변수로 덮어쓰기 가능):

| 파라미터 | 환경변수 | 기본값 |
|---|---|---|
| 유사도 임계값 θ | `SID_EVENT_SIMILARITY_THRESHOLD` | 0.3 |
| 시간창 | `SID_EVENT_TIME_WINDOW_HOURS` | 72 |
| high-df 제외 비율 | `SID_EVENT_MAX_ENTITY_DF_RATIO` | 0.01 |

절차:

1. train 사용 기사를 `(published_time, article_id)` 순으로 정렬
2. **Entity IDF** (`_build_train_entity_idf`)
   - `idf(e) = log((N+1) / (df(e)+1)) + 1`
   - train에 없던 entity: `unseen_idf = log(N+1) + 1`
3. **high-df entity 제외**: train 기사의 1% 이상에 등장한 entity(예: `ORG::ekstra bladet`)를 뺀 `clustering_entity_set` 생성 (원본 entity set은 보존)
4. **Edge 생성**: 정렬된 기사를 순회하며 72시간 이내 쌍만 비교하고, 시간차가 72시간을 넘으면 `break`
   - 유사도: **IDF-weighted Jaccard** (`_idf_weighted_jaccard`)

     ```
     sim(A, B) = Σ_{e ∈ A∩B} idf(e) / Σ_{e ∈ A∪B} idf(e)
     ```

   - 한쪽이 비어 있거나 교집합이 없으면 0
   - `sim ≥ θ`이면 Union-Find로 union
5. **Event 확정**: connected component 1개 = event 1개
   - 연결되지 않은 기사(entity 없는 기사 포함)는 단독 event
   - 연결이 전이되므로 A–B, B–C만 연결돼도 A·C가 같은 event가 되고, event 지속 기간이 72시간을 넘을 수 있다
6. **ID 부여**: component를 `(event_start_time, first_article_id)` 순으로 정렬해 `0, 1, 2, …`
   - event 상태: entity 합집합, 시작 시각, 마지막 기사 시각, 기사 수
7. 출력
   - `article_events.parquet`
   - `event_master.parquet`
   - `entity_idf.parquet` (`entity, document_frequency, document_frequency_ratio, idf, is_high_df`)

### 2-3. Validation event 배정 — `_assign_validation_events()` (`src/build_validation.py:370`)

train event는 재클러스터링하지 않는다. validation-only 기사를 `(published_time, article_id)` 순으로 하나씩 처리한다.

1. entity는 같은 canonical key, high-df 목록과 IDF는 train 결과를 재사용
2. 후보 event: `0 ≤ article_time − event_last_added_time ≤ 72h` (미래 상태의 event 사용 금지)
3. 후보 event의 entity 합집합과 IDF-weighted Jaccard 계산
4. `sim ≥ θ`인 후보 중 **최고 유사도 event**에 배정 (동점이면 작은 `event_id`)
   - 배정 후 event의 entity 합집합, `event_last_added_time`, 기사 수를 갱신 → 이후 기사의 매칭에 영향
5. 매칭되는 event가 없으면 **validation-origin 신규 event** 생성
6. 출력
   - `validation_article_events.parquet`
   - `event_master_with_validation.parquet` (train용 `event_master.parquet`은 덮어쓰지 않음)
   - `validation_article_master.parquet`

### 2-4. Entity Linking 실험 변형 (`event_entity_linking/`)

entity 표현만 GPT/Wikidata entity linking 결과(`article_linked_entities.parquet`)로 바꾸고,
같은 event 정책(0.3 / 72h / 0.01)으로 event를 다시 만드는 실험용 패키지이다. 자세한 내용은 `event_entity_linking/README.md` 참고.

---

## 3. RQ-VAE 전달 형태

`build_train_article_master()` / `build_validation_article_master()` → `export_rqvae_inputs()`
(필수 컬럼, ID 중복·null 검증 포함)

```
article_id | embedding_row | model_category_id (c1) | event_id (c2)
```

RQ-VAE 쪽 `RQVAE/data/news.py`의 `NewsArticleDataset`이 이 네 컬럼을 필수로 요구한다.

---

## 4. 확인이 필요한 점

1. **전이적 연결(chaining)**: 직접 비교하지 않은 기사도 같은 event가 되므로 거대 event가 생길 수 있다. event 크기 분포 확인 필요.
2. **train / validation 방식 차이**: train은 기사–기사 연결, validation은 기사–event 합집합 비교 + 최고 1개 배정. event entity 합집합이 커질수록 Jaccard 분모가 커져 큰 event에는 신규 기사가 붙기 어려워질 수 있다.
3. **entity 없는 기사**는 모두 단독 event가 된다. 단독 event 비율을 리포트로 확인하는 것이 좋다.
4. **카테고리 원본 그대로 사용**: 희소 카테고리 병합이 없다. RQ-VAE의 `vae_num_categories`(현재 gin 25)가 `max(model_category_id) + 1` 이상인지 확인 필요.
