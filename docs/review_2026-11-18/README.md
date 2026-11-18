# 코드 검토 자료 (2026-11-18)

여민 저장소(`baekyeomin/SID_Project`, main `bcd1168`) 검토와 11/18 팀미팅 결정을 정리한 자료.

## 문서
| 파일 | 내용 |
|---|---|
| `파이프라인_전체흐름.html` | 9단계 파이프라인 도식 + validation-only event_id 생성·사용처 + z(E) 실험 R/H + 체크리스트 |
| `EB-NeRD_데이터분석.html` | EB-NeRD small 기간·발행 시각·겹침 분석 |
| `SID생성_Transformer_수정요청_v2.html` | 여민에게 보낸 수정요청서 v2 (4번 동점·5번 사건 C2는 11/18 결정 반영 전) |
| `수정요청_timeupdate_transformer.html` | 수정요청서 v1 |
| `EBNeRD_전처리_산출물_정리.html`, `EBNeRD_데이터분석_해설.html`, `RQVAE_ver1_ver2_비교.html`, `mind와eb-nerd.png` | 기존 정리 자료 |
| 발표자료 | https://claude.ai/artifact/BU3XwcsQkF9Zu5xvAdj1Nf (비공개, 14장) |

## 11/18 팀미팅 결정
1. **z(E) 공간 쌍 실험** — Q2 KMeans 초기화 입력과 validation-only 기사가 물려받는 사건 C2(= Q2 최근접(z(E)))를 같은 공간으로:
   실험 R (잔차, 잔차) = 사건별 평균 r1 / 실험 H (인코더, 인코더) = 사건별 평균 h (지금 코드).
2. **사건 매칭의 2차 코사인은 두 실험 모두 h** — cos(h(a), 사건별 평균 h). 매칭 결과가 두 실험에서 같도록.
3. **동점이면 event_id 숫자가 작은 사건** (전처리 `build_validation.py`와 동일). 여민 코드 `generate_semantic_ids_timeupdate.py` 675줄 `sorted(..., key=str)` → 숫자 정렬로 수정.
4. **validation event_id는 RQ-VAE 평가(c2 일관성)의 정답 묶음으로만** 사용. 학습·SID·Transformer엔 영향 없음.
   지금 timeupdate 733–741줄이 event_id로 사건 개체를 꺼내 쓰지만, 개체를 `article_entities.parquet`에서 읽게 고치면 사라짐.

## 남은 할 일
- 여민: 기사 개체 파일에서 읽기 · 점수 IDF 가중 Jaccard · `--update_policy A/B/C` · 사건 id 숫자 정렬 · z(E) 공간 옵션 · Transformer c3/c4 크기 · 경로
- 민지: `main.py`에 `build_article_entities` 연결 · `event_entity_linking/config.py` 하드코딩(0.3/72/0.01) 정리 · event_id 컬럼 유지
- 재실행: 실험 R ③ 재학습→⑨ / 실험 H ④→⑨ / 정책 A/B/C ④→⑨
- (선택) v2 수정요청서에 11/18 결정 반영

## 분석 스크립트 (`analysis/`, polars 필요)
- `c2_event_consistency.py <article_semantic_ids.parquet>` — 기사 2개+ train 사건의 c2 일치율. 예전 SID(normalize_v2): 같은 사건 쌍 일치 0.50 vs 같은 카테고리 무작위 0.02, 전부 같은 c2인 사건 31~37%.
- `code_usage.py` — 코드 사용량·prefix 희소성 (c1c2c3 조합의 88~90%가 기사 1개).
- `mixcat.py` — 카테고리 섞인 train 사건 비율 (기사 2개+ 사건 중 10%).
- `check_testset.py` — small `articles.parquet`의 train/val 미참조 7,570개를 test set과 대조 (test set 다운로드 필요, 미실행).

데이터(parquet)는 저장소에 없음. 로컬 경로: `data/raw/`, `data/final_sid/`.
