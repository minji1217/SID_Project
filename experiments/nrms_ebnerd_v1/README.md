# EB-NeRD NRMS: E5 vs SID 기사 표현 비교 (발표 9장)

AWS `~/train_nrms.py`, `~/evaluate_nrms.py`(2026-10-12)와 결과 `~/nrms_out/`을 그대로 옮겨 온 것이다.
코드는 수정하지 않았다. checkpoint(`checkpoint_best.pt`)는 AWS `~/nrms_out/<run>/`에만 있다.
`~/minkyo/nrms/`는 정민교 MIND 실험이라 여기 포함하지 않았다.

## 무엇을 비교하나

추천 모델·데이터·loss·하이퍼파라미터를 고정하고 **기사 표현만** 바꾼다.

| `--representation` | 기사 벡터 |
|---|---|
| `e5` | 고정된 E5 768차원 임베딩 → Linear(768→256) |
| `sid` | c1·c2·c3 각각 Embedding(128차원) → 이어 붙여 Linear(384→256). c4는 쓰지 않음 |

## 모델 (NRMS 방식, 원 논문 그대로는 아님)

- 원 NRMS의 제목 단어 self-attention 뉴스 인코더 대신 위 기사 벡터를 쓴다.
- 사용자 인코더: history 기사 벡터 → multi-head self-attention 1층(8 head) + residual + LayerNorm → additive attention pooling → 사용자 벡터
- 점수: 사용자 벡터 · 후보 기사 벡터 / √256, 후보 5개에 cross entropy
- 설정은 Transformer V1 최종 설정과 맞췄다: d_model 256, head 8, history 50(최근), dropout 0, batch 128, AdamW lr 5e-5, weight decay 0, 최대 30 epoch, patience 5, gradient clip 5.0
- checkpoint 선택: validation Top-1 (동점이면 MRR). test는 학습 뒤 `evaluate_nrms.py`로 한 번만 평가.
- history가 빈 행은 제외. 후보 위치는 가정하지 않고 label에서 positive를 찾는다.

## 실행 형식

데이터 경로는 실행 인자로 받으며 실제 사용한 경로는 기록되어 있지 않다.
test 행 수 122,668은 UNI SID 기준 1pos4neg test와 같다.

```bash
python train_nrms.py --representation {e5|sid} --seed {42|123|2026} \
  --train <train 1pos4neg parquet> --validation <validation half parquet> \
  --article-master <article_master parquet> --article-embeddings <article_embeddings.npy> \
  --semantic-ids <article_semantic_ids.parquet> --output-dir nrms_out/<rep>_seed<seed>
python evaluate_nrms.py --checkpoint nrms_out/<run>/checkpoint_best.pt --test <test 1pos4neg parquet> \
  --article-master ... --article-embeddings ... --semantic-ids ... --output-json nrms_out/<run>/test_metrics.json
```

## 결과 (test 122,668행, `results_aws/*/test_metrics.json`)

| 표현 | seed | best epoch | Val Top-1 | Test Top-1 | MRR | nDCG@5 | AUC |
|---|---|---|---|---|---|---|---|
| E5 | 42 | 1 | 27.89% | 29.37% | 0.5377 | 0.6519 | 0.6019 |
| E5 | 123 | 2 | 27.76% | 29.19% | 0.5360 | 0.6506 | 0.5998 |
| E5 | 2026 | 2 | 27.85% | 29.15% | 0.5359 | 0.6505 | 0.5998 |
| **E5 평균** | | | 27.83% | **29.24%** | 0.5366 | 0.6510 | **0.6005** |
| SID | 42 | 4 | 28.48% | 27.78% | 0.5247 | 0.6420 | 0.5872 |
| SID | 123 | 1 | 27.13% | 28.28% | 0.5288 | 0.6451 | 0.5923 |
| SID | 2026 | 1 | 27.79% | 28.55% | 0.5321 | 0.6477 | 0.5974 |
| **SID 평균** | | | 27.80% | **28.20%** | 0.5285 | 0.6449 | **0.5923** |

SID는 E5 Top-1의 96.5%를 유지한다. validation 평균은 두 표현이 거의 같다(27.83% vs 27.80%).
