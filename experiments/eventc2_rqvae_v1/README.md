# eventc2_rqvae_v1: event-level C2로 RQ-VAE 재학습

교수님 설계대로 C2를 학습 중에도 event 단위로 정하는 RQ-VAE를 처음부터 다시 학습한다.
기존 A(`checkpoint_best_rec.pt`, WD-001-UNI-lu0.05-m0.5)는 baseline으로 보존한다.

## 학습 구조

```
h(a) = Encoder(x(a))
c1 = CategoryMap[category(a)],  q1 = Q1[c1]
z(E) = mean{h(a) | a in E}                       현재 encoder output
EventCode(E) = argmin_k ||z(E) - Q2[k]||          현재 Q2, 매 step 갱신
c2 = EventCode[event(a)],  q2 = Q2[c2]            같은 event의 기사는 같은 q2
r2(a) = h(a) - q1(a) - q2(E)
c3 = argmin_k ||r2(a) - Q3[k]||                   기사 단위, STE
x_hat(a) = Decoder(q1 + q2 + q3)                  기사 단위 reconstruction
```

- **Q2 초기화**: A와 같다. 초기 encoder로 Train 전체 h → event별 z(E) → K-means → Q2 centroid. Q3는 random.
- **complete-event batch**: `RQVAE/data/event_batch_sampler.py`. event를 쪼개지 않고 최대 256개 기사까지 담는다. 매 epoch마다 seed + epoch로 event 순서를 섞는다.
- **Q2 loss (event 단위)**
  - `L_cb2 = mean_E ||sg(z(E)) - q2(E)||²`
  - `L_com2 = mean_E ||z(E) - sg(q2(E))||²`
  - λ_cb, λ_com은 A와 같다 (0.25, 0.1).
- **전체 loss**: `mean_a[rec + λcb(cb1+cb3) + λcom(com1+com3)] + λuniq·uniq + λcb·L_cb2 + λcom·L_com2`
- **STE**: 학습 중 `q2(a) = r1(a) + sg(q2(E) - r1(a))`. forward 값은 q2(E)이고, 기존 A의 Q2 STE와 같은 gradient 경로다.
  - reconstruction은 codebook으로 가지 않는다.
  - Q3 commitment는 encoder / Q1 / Q2로 가지 않는다.
  - 실제 gradient 비교와 `tests/check_event_c2.py`로 확인했다.
- **Validation**: 현재 시점 Train EventCode를 기존 Train event에 상속한다. 신규 event는 현재 encoder의 mean(h) → 현재 Q2 nearest로 정한다.
- **하이퍼파라미터**: A checkpoint 안의 `gin_config`와 같다 (`lambda_uniq=0.05` 포함). 다른 것은 `train.c2_mode = "event"`와 저장 경로뿐이다.
- **실행 환경**: single GPU만 지원한다 (`accelerator.num_processes == 1` assert).

## 실행 (single GPU, 예: Colab)

```bash
cd RQVAE
# configs/rqvae_ebnerd_eventc2.gin의 train.dataset_folder / train.save_dir_root를 환경에 맞게 확인
python train_rqvae.py configs/rqvae_ebnerd_eventc2.gin
```

학습 후 SID 생성. event checkpoint는 반드시 `--train_c2_policy event`로 만든다 (아니면 에러로 중단).

```bash
python generate_semantic_ids.py \
  --data_dir <dataset_folder> \
  --checkpoint <save_dir_root>/checkpoint_best_rec.pt \
  --output_dir <새 SID 폴더> \
  --train_c2_policy event
```

`sid_generation_meta.json`의 `checkpoint_event_code_table_agreement`는 checkpoint에 저장된 EventCode 표와 다시 계산한 표의 일치율이다.

## 로그 (`<save_dir_root>/event_c2_log.jsonl`, epoch마다 한 줄)

| 항목 | key |
|---|---|
| 한 batch에서 같은 event의 c2 unique count = 1 | 매 step assert (실패 시 중단) |
| event가 batch 간 분할되지 않음 | 매 step assert |
| 전체 Train same-event C2 consistency = 100% | 매 epoch snapshot assert, `train.same_event_c2_consistency` |
| epoch별 EventCode churn | `event_code_churn`, `event_code_churn_article_weighted`, `in_epoch_vs_snapshot_code_agreement` |
| Q2 code usage / dead code | `q2_codes_used_by_events`, `q2_codes_used_by_articles`, `q2_dead_codes`, `q2_article_entropy_bits`, `q2_top10_article_share` |
| Q3 code usage / dead code | `q3_codes_used_by_articles`, `q3_dead_codes`, `q3_article_entropy_bits`, `q3_top10_article_share` |
| reconstruction / codebook / commitment / uniqueness loss | `train_loss` (`codebook_q1/q2/q3`, `commitment_q1/q2/q3` 포함), `validation_loss` |
| c123 collision, c4 최대값 | `train`, `validation`, `train_validation`의 `c123_collision_rate`, `max_c4` |

checkpoint에는 저장 시점의 `event_code_table`과 `c2_mode`가 함께 들어간다.

## CPU 진단 실행

`cpu_diagnostic_60ep/`는 구조 확인용 CPU 60 epoch 진단 결과다 (공식 결과 아님).
Q3 code usage 감소, c123 collision 증가, r2와 −q1 방향 정렬이 관찰되었고,
이는 현재 설계(z(E)=mean(h))의 진단 결과로만 보존한다. Q2 공간 변경 등 대안은
공식 재학습 결과를 교수님과 공유한 뒤 결정한다.

## 검증

```bash
cd RQVAE
python tests/check_event_c2.py
```
