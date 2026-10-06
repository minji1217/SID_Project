# EventC2 SID 품질 평가 (UNI 평가 리포트와 같은 방법)

기존 UNI 평가 코드: `origin/claude/hopeful-mendel-fhc2n4:analysis/rqvae_sid_evaluation`
(샘플러 `step3/sid_pairs.py`, `step6_prefix1/step6c_prefix1.py`를 그대로 import / 복사해 사용, seed 42)

## 1차: original embedding cosine (텍스트·checkpoint 없이 가능한 부분)

`python emb_compare.py <scratchpad>` → `emb_compare.json`

모집단: train 9,738 (기존 리포트는 body-valid 9,338. 기사 본문 파일이 없어 UNI·EventC2 모두 9,738로 맞춤)

| 조건 | UNI pairs | UNI cos | EventC2 pairs | EventC2 cos |
|---|---|---|---|---|
| Different | 39,625,898 | 0.7989 | 39,681,571 | 0.7997 |
| Prefix-1 (c1 같고 c2 다름) | 7,323,941 | 0.8185 | 7,229,729 | 0.8135 |
| Prefix-2 (c1,c2 같고 c3 다름) | 140,097 | 0.8473 | 226,279 | 0.8453 |
| Prefix-3 (c1,c2,c3 같음) | 1,886 | 0.9154 | 9,916 | 0.8847 |

(UNI 9,738 값은 기존 리포트의 9,338 값 0.7986 / 0.8180 / 0.8470 / 0.9070과 거의 같다)

| Prefix-3 group | UNI | EventC2 |
|---|---|---|
| 크기 2 이상 group 수 | 836 | 1,443 |
| 그 안의 기사 비율 | 20.9% | 47.9% |
| 평균 / 최대 크기 | 2.44 / 12 | 3.23 / 31 |
| (c1,c2) 조합 수 | 1,062 | 731 |

## 남은 항목 (입력 필요)

- 텍스트 정량(Jaccard, Body TF-IDF)과 정성 사례: EB-NeRD `articles.parquet` (title, subtitle, body, category_str)
- reconstruction / self-reconstruction: EventC2 `checkpoint_best_rec.pt`
  - x_hat = decoder(Q1[c1] + Q2[c2] + Q3[c3]) — 저장된 SID 사용이라 event 규칙이 그대로 반영된다
  - 기존 step6b의 "SID 재현" 확인은 기사 단위 get_semantic_ids를 쓰므로 EventC2에서는 event_ids를 넘겨야 한다
