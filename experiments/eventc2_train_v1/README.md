# eventc2_train_v1: Train c2를 event-level로 고정한 SID 실험

| variant | 의미 | 옵션 |
|---|---|---|
| A | 기존: article-level Q2 nearest (r1 = h − q1) | (기본값) |
| B | 교수님 설계: event-level Q2(mean(h)) | `--train_c2_policy event` |
| B-r1 | 원인 분리 진단용: event-level Q2(mean(h − q1)) | `--train_c2_policy event --event_repr mean_r1` |

B-r1에서는 기사 1개짜리 event의 c2/c3가 A와 같다. 따라서 A 대비 변화는
"같은 event를 같은 c2로 묶은 효과"만 남는다. 단 event representation은
Validation 새 event에도 같이 적용되므로 Validation-only 기사의 c2도 r1 공간 기준으로 바뀐다.

RQ-VAE는 재학습하지 않는다. 기존 final checkpoint를 그대로 쓰고,
최종 Train SID를 만드는 방식만 바꾼 뒤 downstream 전체를 새 실험 폴더에 다시 만든다.

## 무엇이 바뀌는가

`RQVAE/generate_semantic_ids.py --train_c2_policy {article,event}`

| | article (기본값, 기존 A) | event (B) |
|---|---|---|
| Train c2 | article-level Q2 nearest (r1 기준) | 자기 event의 Train EventCode (`train_event_c2_mapping`) |
| Train c3 | r2 = h − q1 − Q2[c2]에서 Q3 nearest | 같은 식, 단 q2 = Q2[event_c2] (fixed c2 기준으로 다시 계산) |
| c4 | Train+Validation 전체 c123 기준 | 같음 (새 c123 기준으로 다시 부여) |
| Validation-only 기사 | event_c2 고정 (기존과 동일) | 기존과 동일 |
| Validation의 재사용 Train 기사 | A의 Train SID | B의 Train SID (같은 event면 같은 c2) |

`--train_c2_policy`를 주지 않으면 출력이 이전 코드와 byte 단위로 같다.
event 모드에서는 Train `c2 == event_c2`가 100%가 아니면 중단하고,
`--output_dir`에 이미 `article_semantic_ids.parquet`가 있으면 덮어쓰지 않고 중단한다.
두 모드 모두 `sid_generation_meta.json`에 정책과 same-event C2 일관성을 남긴다.

## 실행

SID_Project 레포 루트에서:

```bash
RQVAE_DATA_DIR=<A SID를 만들 때 쓴 data_dir> \
CKPT=<A SID를 만들 때 쓴 final checkpoint.pt> \
TRANSFORMER_ROOT=<sid_project-transformer/Transformer, claude/hopeful-mendel-fhc2n4 브랜치> \
A_SID_DIR=<A의 generate_semantic_ids output_dir> \
A_RUN_DIR=<A의 Transformer seed42 run 폴더 (run_summary.json 있는 곳)> \
bash experiments/eventc2_train_v1/run_pipeline.sh
```

B-r1은 같은 명령에 `EVENT_REPR=mean_r1`을 붙인다
(출력: `normalize_v2_uni_lu005_m05_eventc2r1_train_v1/`).

A의 post-rqvae를 `SID_OUTPUT_DIR`를 지정해서 돌렸다면 같은 값을 함께 준다.

## 출력 (모두 `${SID_OUTPUT_DIR:-data/output}/experiments/normalize_v2_uni_lu005_m05_eventc2_train_v1/`)

| 폴더 | 내용 |
|---|---|
| `semantic_ids/` | 새 SID (`article_semantic_ids.parquet`, event mapping, `sid_generation_meta.json`) |
| `post_rqvae/`, `exports/` | 새 SID 기반 전체 train/validation sequences (`src.main post-rqvae --experiment`) |
| `candidate_1pos4neg/` | `src.build_1pos4neg_sequences` (seed 42, c1c2c3, negative 4) |
| `candidate_1pos4neg_shuffled/` | `src.shuffle_candidate_order` (seed 42) |
| `transformer_datasets/` | validation half / test (impression 시간순 50:50) |
| `transformer_runs/seed42/` | checkpoint, `run_summary.json`, `test_candidate_scores.parquet`, `test_metrics.json` |
| `reports/` | 단계별 로그, `sid_ab/`, `transformer_ab/` 비교 리포트 |

각 단계는 결과가 있으면 건너뛰므로 중간에 끊겨도 같은 명령으로 이어서 돌릴 수 있다.

## 해석할 때 주의

- 1pos4neg sampling은 (c1,c2,c3)가 겹치는 negative를 제거하므로 SID가 바뀌면 남는 negative와 탈락 row도 바뀐다.
  A와 B의 impression 수, `reports/03_1pos4neg_report.json`의 탈락 비율을 함께 볼 것.
- B의 max c4 + 1이 기존 `c4_vocab_size=28`보다 크면 vocab을 늘린다(로그에 WARNING).
  이 경우 같은 seed라도 초기화가 A와 완전히 같지 않다.
- event_c2는 mean(h)로 Q2를 찾지만 Q2는 학습 중 r1 = h − q1을 입력으로 받았다.
  B의 변화에는 event 일관성 효과와 이 검색 공간 차이가 함께 들어 있다
  (`reports/sid_ab`의 `train_c2_equals_event_c2.singleton_events`, reconstruction 항목 참고).
