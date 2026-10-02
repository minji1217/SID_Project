# Step 5 — RQ-VAE reconstruction 생성 경로 코드 조사

목적: Step 6(reconstruction cosine)을 실행하기 전에, **실제 프로젝트 코드에서**
reconstruction을 올바르게 생성할 수 있는 경로를 확정한다. 계산은 수행하지 않았다.
입력 parquet은 읽기 전용으로만 열었다.

조사 대상 리비전: 브랜치 `claude/hopeful-mendel-fhc2n4`, 저장소 `/home/user/SID_Project`.

---

## 1. 최종 사용된 RQ-VAE checkpoint

### 1-1. 학습이 저장하는 checkpoint 3종

`RQVAE/train_rqvae.py`

| checkpoint | 저장 위치 | line |
|---|---|---|
| `checkpoint_best_total.pt` | `{save_dir_root}/` | 1793–1796 |
| `checkpoint_best_rec.pt` | `{save_dir_root}/` | 1803–1807 |
| `checkpoint_final.pt` | `{save_dir_root}/` | 2043–2074 |
| `checkpoint_{epoch}.pt` (주기 저장) | `{save_dir_root}/` | 1898–1905 |

모든 checkpoint에 `model_config`(train_rqvae.py:699)와 `model`(state_dict)이 들어간다.

### 1-2. 어떤 checkpoint가 SID를 만들었나

`run_experiment.py`가 EXP 단위로 다음을 수행한다.

- 1285–1300 — `best_total`, `best_rec`이 있으면 둘 다 후보로 등록. 둘 다 없을 때만 `final` fallback.
- 1311–1360 — **후보 checkpoint마다 각각** `generate_semantic_ids.py`를 실행해
  `semantic_ids/{variant}/article_semantic_ids.parquet` 생성.
- 1414–1436 — `--sid-checkpoint`가 지정한 variant의 결과를
  `semantic_ids/article_semantic_ids.parquet`로 **복사**.
- 1665–1673 — `--sid-checkpoint`의 **기본값은 `best_rec`**.
- 1438–1450 — 실제 선택 결과를 `semantic_ids/selected_checkpoint.json`에
  `selected_checkpoint_variant` / `selected_checkpoint_path`로 기록.

주석(1166–1167)도 명시한다: "best_total과 best_rec 중 어느 쪽이 진짜 최종 모델인지는
자동으로 확정하지 않는다."

### 1-3. 결론

- **기본 경로상 `checkpoint_best_rec.pt`가 대표 SID를 만든다** (`--sid-checkpoint` 기본값).
- 경로 패턴: `{experiment_root}/stage_*/{EXP-ID}/rqvae/checkpoint_best_rec.pt`
- 대응 config: 같은 EXP 폴더의 `config.gin` (run_experiment.py:1117–1119),
  메타데이터는 `metadata.json`(1121–1135).
- **단정할 수 없는 부분**: 실행 시 `--sid-checkpoint best_total`을 주었을 가능성이 있다.
  확정 근거는 **학습 머신의 `semantic_ids/selected_checkpoint.json`** 하나뿐이다.
  Step 6 전에 이 파일을 반드시 확인해야 한다.

### 1-4. 저장소에 checkpoint가 없다

```
.gitignore
  RQVAE/datasets/
  RQVAE/out/
```

`*.pt / *.pth / *.ckpt` 검색 결과 **0건**. `metadata.json`, `selected_checkpoint.json`,
`all_results.csv`, `checkpoint_candidates.csv`도 저장소에 없다.
checkpoint와 실험 기록은 전부 학습 머신에만 있다.

---

## 2. 최종 RQ-VAE config

### 2-1. 저장소의 gin (`RQVAE/configs/rqvae_ebnerd.gin`)

| 항목 | 값 |
|---|---|
| `vae_input_dim` | 768 |
| `vae_hidden_dims` | `[512]` |
| `vae_embed_dim` | 256 |
| `vae_num_categories` | 25 |
| `vae_c2_codebook_size` | 128 |
| `vae_c3_codebook_size` | **256** |
| `vae_codebook_normalize` | False |
| `vae_sim_vq` | False |
| `vae_codebook_mode` | `QuantizeForwardMode.STE` |
| `lambda_rec / cb / com` | 1.0 / 1.0 / 0.25 |
| `seed` | 42 |

### 2-2. 이 gin은 최종 실험과 일치하지 않는다

업로드된 `article_semantic_ids.parquet`의 실제 값:

```
c1  distinct 25   (0~24)   -> num_categories 25 와 일치
c2  distinct 128  (0~127)  -> c2_codebook_size 128 과 일치
c3  distinct 512  (0~511)  -> c3_codebook_size 256 과 불일치
     c3 >= 256 인 기사 6,458개, 그 중 distinct 256개
```

`run_experiment.py:558–564`의 `q2q3` sweep이 `train.vae_c3_codebook_size`를
`[128, 256, 512, 1024]`로 덮어쓴다. 즉 **최종 모델은 base gin이 아니라
c3=512로 override된 sweep EXP의 산물**이다.

> **권위 있는 config는 gin 파일이 아니라 checkpoint 안의 `model_config`다.**
> `generate_semantic_ids.py:361`가 `RqVae(**checkpoint["model_config"])`로 모델을 만든다.
> Step 6도 동일하게 `model_config`를 그대로 써야 한다.

### 2-3. 코드에 고정되어 override 불가능한 설정

- **distance**: `RqVae.__init__`(rqvae.py:137–176)이 `Quantize(...)`에 `distance_mode`를
  전혀 넘기지 않는다. 따라서 항상 `QuantizeDistance.L2`(quantize.py:68)다.
  cosine distance는 구현되어 있으나(quantize.py:210–218) 이 프로젝트에서는 쓰이지 않는다.
- **decoder hidden dims**: `hidden_dims[-1::-1]` (rqvae.py:122) — encoder hidden을 뒤집어 쓴다.
- **bias 없음**: `MLP`의 모든 Linear가 `bias=False` (encoder.py:27).
- **마지막 레이어 뒤 활성함수 없음**: 마지막 Linear 뒤에는 ReLU가 붙지 않고
  `L2NormalizationLayer` 또는 `Identity`만 온다 (encoder.py:28–33).
  decoder는 `normalize=False`로 생성되므로 (rqvae.py:124) **Identity** — 즉 `x_hat`은 선형 출력이다.

---

## 3. 원본 article embedding

### 3-1. 파일

| 파일 | 내용 | 참조 |
|---|---|---|
| `{data_dir}/article_embeddings.npy` | `[num_articles, 768]` float, `mmap_mode="r"`로 로드 | generate_semantic_ids.py:158–172 |
| `{data_dir}/article_master.parquet` | train 마스터 | evaluate/common.py:12 |
| `{data_dir}/validation_article_master.parquet` | validation 마스터 | evaluate/common.py:16 |

**train과 validation이 같은 `article_embeddings.npy` 하나를 공유한다**
(`resolve_master_paths`가 embeddings 경로를 split과 무관하게 한 개만 반환, common.py:20).

### 3-2. article_id ↔ embedding row 연결

마스터 필수 컬럼 (generate_semantic_ids.py:94–99):
`article_id`, `embedding_row`, `model_category_id`, `event_id`

`SemanticIdDataset.__getitem__`(219–281):
```python
embedding_row = int(row["embedding_row"])
embedding     = np.asarray(self.embeddings[embedding_row], dtype=np.float32).copy()
x             = torch.from_numpy(embedding)
category_id   = int(row["model_category_id"])   # -> Q1의 fixed_ids = c1
```
즉 **`embedding_row`가 npy의 행 인덱스**다. 범위 검증은 174–209, 차원 검증은 436–459.

### 3-3. 업로드된 SID 파일만으로도 행 매핑이 된다

`article_semantic_ids.parquet` 컬럼:
`article_id, embedding_row, event_id, split, c1, c2, c3, c4`

- rows 12,860 / `article_id` unique 12,860 / `embedding_row` **중복 0**
- `embedding_row` 범위 0 ~ 19,906
- train 9,738 · validation 3,122, **두 split의 article_id 교집합 0**

→ npy만 확보하면 **마스터 파일 없이도** SID 파일의 `embedding_row`로 원본 x를 꺼낼 수 있다.
다만 `model_category_id`는 SID 파일에 없다. 단 Q1은 `fixed_ids=category_ids`로 동작하므로
**`c1` 자체가 `model_category_id`**다 (rqvae.py:381–388, quantize.py:270).

### 3-4. 동일 embedding 재사용 가능 여부

가능하다. train 9,738과 validation 3,122 모두 같은 npy의 서로 다른 행을 가리키며
중복이 없다. 단 **npy 파일이 SID 생성 당시와 동일한 파일이어야 한다** —
`embedding_row`는 그 npy에 대한 위치 인덱스이므로 임베딩을 다시 만들면 깨진다.

---

## 4. 실제 forward path

### 4-1. 흐름 추적

| 단계 | 구현 | 위치 |
|---|---|---|
| `x` 입력 | `SemanticIdDataset.__getitem__` | generate_semantic_ids.py:247–256 |
| `x → h` (encoder) | `RqVae.encode` → `MLP.forward` | rqvae.py:211–216 → encoder.py:35–39 |
| `q1 = Q1[c1]` | `quantizer_1(x=h, fixed_ids=category_ids)` | rqvae.py:381–388 |
| | `Quantize._forward_fixed_ids` (거리계산·argmin 없음) | quantize.py:229–287 |
| `r1 = h − q1` | | rqvae.py:396 |
| `q2`, `c2` | `quantizer_2(x=r1, fixed_ids=fixed_c2_ids)` | rqvae.py:410–417 |
| `r2 = r1 − q2` | | rqvae.py:426 |
| `q3`, `c3` | `quantizer_3(x=r2)` | rqvae.py:434–440 |
| | `Quantize._forward_nearest` (L2 argmin) | quantize.py:291–388 |
| `embeddings` stack | `torch.stack([q1,q2,q3], dim=-1)` | rqvae.py:465–472 |
| **decoder 입력** | `quantized.embeddings.sum(dim=-1)` | **rqvae.py:544–548** |
| `x_hat` | `self.decode(quantized_embedding)` | **rqvae.py:558–560** |

### 4-2. decoder input이 정확히 무엇인가

```python
quantized_embedding = quantized.embeddings.sum(dim=-1)   # rqvae.py:544-548
x_hat = self.decode(quantized_embedding)                 # rqvae.py:558-560
```

**`q1 + q2 + q3`의 단순 합**이다. concat이 아니라 **덧셈**이며, 가중치도 projection도 없다.
`c4`는 decoder에 전혀 들어가지 않는다 (c4는 generate_semantic_ids.py:1982–2160에서
SID 생성이 끝난 뒤 중복 해소용으로 붙이는 suffix).

`embeddings`가 `[B, embed_dim, 3]`으로 stack되어 있으므로 `sum(dim=-1)`은
마지막 축(3개 quantizer)에 대한 합 → `[B, embed_dim]`.

### 4-3. Q2/Q3 selected code vector를 어떻게 합치는가

residual quantization이므로 **코드 벡터를 그대로 더한다**:

```
h   ≈ q1 + q2 + q3
r1  = h  − q1      (Q2의 입력)
r2  = r1 − q2      (Q3의 입력)
```

`eval()` 모드에서는 STE도 Gumbel도 적용되지 않고 선택된 code vector가 그대로 나온다
(quantize.py:373–377: `emb_out = emb`). 즉 **추론은 완전히 결정적(deterministic)**이며
`gumbel_t`는 결과에 영향을 주지 않는다.

### 4-4. forward()가 reconstruction을 반환하는가 — **아니다**

`RqVae.forward`는 `x_hat`을 **내부에서 만들고 버린다**. 반환 타입은
`RqVaeComputedLosses`(rqvae.py:34–41, 670–686)로 loss와 로깅 지표만 담는다.
**`x_hat`을 돌려주는 public 메서드가 코드에 없다.**

기존 `RQVAE/evaluate/eval_reconstruction.py`도 `model(...)`을 호출해
`output.reconstruction_loss` 스칼라만 집계한다(:87–97). article-level `x_hat`을
저장하지 않는다. → **Step 6에는 새 추출 코드가 필요하다.**

### 4-5. validation SID 생성 경로와 reconstruction forward가 동일한가 — **동일하지 않다**

| | train SID 생성 | validation SID 생성 | `forward()` (= eval_reconstruction) |
|---|---|---|---|
| 호출 | `get_semantic_ids(x, category_ids)` | `get_semantic_ids(x, category_ids, fixed_c2_ids=...)` | `get_semantic_ids(..., fixed_c2_ids=None)` |
| 위치 | generate_semantic_ids.py:616–623 | generate_semantic_ids.py:1696–1721 | rqvae.py:533–538 |
| c2 결정 | r1에서 Q2 argmin | **event-level로 미리 정한 C2를 강제** | r1에서 Q2 argmin |

validation의 `fixed_c2_ids`는 `validation_event_c2_mapping.parquet`에서 온다
(generate_semantic_ids.py:1472–1486, 2722–2725). 기존 train event는 train EventCode를
상속하고(:1180, 1329), 새 validation event는 `mean(h)`를 frozen Q2에 통과시켜 정한다(:1266).

**따라서 validation 기사에 대해 `forward()`나 `get_semantic_ids(fixed_c2_ids=None)`를
그냥 돌리면, 저장된 `article_semantic_ids.parquet`의 c2와 다른 c2가 나올 수 있다.**

추가로 validation에는 "train에서 이미 본 article은 forward 없이 train SID를 그대로 재사용"하는
분기가 있다(generate_semantic_ids.py:1585–1622). 다만 최종 파일에서 두 split의
article_id 교집합이 0이므로, **이번 3,122건은 전부 inference 경로를 탄 것**으로 보인다
(validation 마스터가 validation-only 기사로 구성됨).

---

## 5. reconstruction dimension

`model_config`의 `input_dim = 768`, `embed_dim = 256`, `hidden_dims = [512]` 기준
(c3=512 override는 아래 shape에 영향 없음).

| 기호 | 코드상 변수 | shape | 근거 |
|---|---|---|---|
| `x` | `x` | `[B, 768]` | input_dim, encoder.py:36 assert |
| latent `z` | `h = self.encode(x)` | `[B, 256]` | embed_dim, rqvae.py:372 |
| `q1`,`q2`,`q3` | `q1_out.embeddings` 등 | 각 `[B, 256]` | quantize.py:409–418 shape 검사 |
| stack | `quantized.embeddings` | `[B, 256, 3]` | rqvae.py:465–472 |
| `z_q` | `embeddings.sum(dim=-1)` | `[B, 256]` | rqvae.py:544–548 |
| `x_hat` | `self.decode(z_q)` | `[B, 768]` | decoder out_dim=input_dim, rqvae.py:120–125 |

레이어 구성:
```
encoder : Linear(768→512, bias=False) → ReLU → Linear(512→256, bias=False) → Identity
decoder : Linear(256→512, bias=False) → ReLU → Linear(512→768, bias=False) → Identity
codebooks: Q1 [25,256]  Q2 [128,256]  Q3 [512,256]
```

---

## 6. `article_semantic_ids.parquet`만으로 reconstruction이 가능한가

### 6-1. 불가능하다

`x_hat = decoder(Q1[c1] + Q2[c2] + Q3[c3])`이므로 정수 `(c1,c2,c3)`는 **인덱스일 뿐**이다.
실제로 필요한 학습된 파라미터:

| 필요한 것 | state_dict key | shape |
|---|---|---|
| Q1 codebook | `quantizer_1.embedding.weight` | `[25, 256]` |
| Q2 codebook | `quantizer_2.embedding.weight` | `[128, 256]` |
| Q3 codebook | `quantizer_3.embedding.weight` | `[512, 256]` |
| decoder | `decoder.mlp.0.weight`, `decoder.mlp.2.weight` | `[512,256]`, `[768,512]` |

이들은 **checkpoint의 `model` state_dict에만** 존재한다. SID parquet에는 없다.
→ **사용자 예상이 맞다.** codebook weights + decoder weights가 든 checkpoint가 반드시 필요하다.

### 6-2. 다만 — encoder와 원본 x는 x_hat 계산에 필요하지 않다

decoder 입력이 오직 `Q1[c1]+Q2[c2]+Q3[c3]`이므로, **checkpoint만 있으면
저장된 `(c1,c2,c3)` 정수에서 x_hat을 그대로 만들 수 있다.** encoder를 통과시킬 필요가 없다.
원본 임베딩 `x`는 `original_embedding_cosine` 계산과 SID 재현 검증에만 필요하다.

### 6-3. ⚠️ Step 6 설계에 영향을 주는 중요한 귀결

`x_hat`이 `(c1,c2,c3)`만의 함수라는 사실에서 다음이 **수학적으로 따라온다**:

> **`prefix_3_same` pair의 `reconstruction_cosine`은 예외 없이 정확히 1.000이다.**

두 기사가 `(c1,c2,c3)`를 공유하면 `q1,q2,q3`가 동일 → decoder 입력 동일 → `x_hat` 동일.
`prefix_2_same`은 `q3`만 다르고, `semantic_different`는 셋 다 다르다.

즉 `reconstruction_cosine`은 **기사 내용의 지표가 아니라 codebook·decoder 기하구조의 지표**다.
조건별 비교를 그대로 하면 prefix_3 = 1.000이라는 자명한 결과가 나오고,
이를 "SID가 의미를 잘 묶는다"는 근거로 쓰면 **순환 논증**이 된다.

→ Step 6에서는 다음을 구분해서 보고해야 한다.
- `original_embedding_cosine` — **기사 단위 지표.** 조건 간 비교의 실질적 근거.
- `reconstruction_cosine` — codebook 기하 지표. prefix_3가 1.000인 것은 설계상 당연함을 명시.
- 추가 제안: `x` vs `x_hat`의 **기사별 self-reconstruction cosine**
  (`cos(x_i, x̂_i)`). 이것이 "RQ-VAE가 이 기사를 얼마나 잘 복원하는가"에 대한
  진짜 질문이며, collision group 내부 이질성과 연결해서 볼 수 있다.

---

## 7. Step 6 실행 가능 여부

### 7-1. 현재 상태로는 실행 불가

이 컨테이너/저장소에 없는 것:

| 필요 파일 | 상태 |
|---|---|
| `checkpoint_best_rec.pt` (또는 selected variant) | **없음** (`.gitignore: RQVAE/out/`) |
| `article_embeddings.npy` | **없음** (`.gitignore: RQVAE/datasets/`) |
| `article_master.parquet` / `validation_article_master.parquet` | **없음** |
| `selected_checkpoint.json` (어느 variant인지 확정용) | **없음** |
| `validation_event_c2_mapping.parquet` (SID 재현 검증용) | **없음** |
| `article_semantic_ids.parquet` | 있음 (업로드본, read-only) |

학습 머신에서 위 파일을 가져오면 **train 9,738 · validation 3,122 모두 생성 가능**하다.
최소 요구는 **checkpoint + article_embeddings.npy** 두 개다
(마스터는 SID 파일의 `embedding_row` / `c1`로 대체 가능).

전송 용량 추정: `article_embeddings.npy`는 19,907행 × 768 × 4B ≈ **58 MB**,
checkpoint는 파라미터 약 1.07M × 4B ≈ **4 MB** + optimizer state 포함 시 더 큼.

### 7-2. 실행 계획

**(a) 모델 로드** — `generate_semantic_ids.py:288–429`의 `load_rqvae`를 그대로 재사용.
`RqVae(**checkpoint["model_config"])` → `load_state_dict` → `.eval()` → `requires_grad_(False)`.
`model_config`를 그대로 출력해 c3=512 등 실제 config를 기록한다.

**(b) x_hat 생성** — `forward()`는 x_hat을 반환하지 않으므로 다음 3줄을 직접 호출한다.
SID 파일의 정수를 그대로 쓰므로 train/validation의 `fixed_c2_ids` 차이 문제가 **원천적으로 사라진다**.

```python
q1 = model.quantizer_1.get_item_embeddings(c1)   # [B, 256]
q2 = model.quantizer_2.get_item_embeddings(c2)
q3 = model.quantizer_3.get_item_embeddings(c3)
x_hat = model.decode(q1 + q2 + q3)               # [B, 768]
```

**(c) 검증 (필수)** — 위 경로가 실제 모델과 일치하는지 독립 확인:
1. `x`를 encoder에 통과시켜 `get_semantic_ids(x, category_ids=c1)`로 SID를 재계산.
   - train 9,738: 저장된 `(c1,c2,c3)`와 **완전 일치해야 한다**.
   - validation 3,122: `fixed_c2_ids`를 주지 않으면 c2가 달라질 수 있다.
     `validation_event_c2_mapping.parquet`을 받아 넣고 일치를 확인하거나,
     불일치 건수를 그대로 보고한다.
2. `model.forward(x, c1)`의 `reconstruction_loss`와,
   (b)로 만든 `x_hat`에 `ReconstructionLoss`를 적용한 값이 일치하는지 대조.
3. 무작위 50건에 대해 `(c1,c2,c3)`가 같은 기사들의 `x_hat`이 bit 단위로 동일한지 확인
   (6-3의 귀결이 코드에서도 성립하는지).

**(d) 지표 계산** — Step 3과 **동일한 pair set**(`quantitative_pairs.parquet`)에
`original_embedding_cosine`과 `reconstruction_cosine`을 컬럼으로 추가.
pair를 다시 뽑지 않아 Step 3/3.5와 직접 비교 가능하다. 여기에 기사별
`self_reconstruction_cosine`을 별도 산출물로 추가.

**(e) 보고** — `reconstruction_cosine`의 prefix_3 = 1.000이 설계상 자명함을 명시하고,
조건 간 실질 비교는 `original_embedding_cosine`으로 수행.

### 7-3. 출력 schema 제안

768차원 벡터를 pair parquet의 셀에 직접 넣으면 207,670행 × 768 × 2벡터가 되어
비효율적이고 중복이 크다. **벡터는 npy, 메타는 parquet**로 분리한다.

```
analysis/rqvae_sid_evaluation/step6/
├── reconstruction_index.parquet        # 12,860행
│     article_id, split, embedding_row, row,         # row = 아래 npy의 행 번호
│     c1, c2, c3, c4, model_category_id,
│     sid_reproduced (bool), self_reconstruction_cosine (float)
├── original_embeddings.npy             # float32 [12860, 768]  (npy에서 추출·정렬한 사본)
├── reconstructed_embeddings.npy        # float32 [12860, 768]
├── pair_embedding_cosine.parquet       # Step 3 pair set + 2개 컬럼
│     pair_id, analysis_set, condition, sampling_method,
│     article_id_a, article_id_b,
│     original_embedding_cosine, reconstruction_cosine
├── embedding_cosine_summary.csv        # 조건×방식×scale×metric별 count/mean/median/std/Q1/Q3/min/max
├── fig9_embedding_cosine.png
├── fig10_text_vs_embedding.png         # TF-IDF cosine vs original embedding cosine
└── step6_manifest.json                 # checkpoint 경로·sha256, model_config 전문, 검증 결과
```

`reconstruction_index.parquet`의 `row`가 두 npy의 행 인덱스이며, 원본 npy의
`embedding_row`와는 별도로 둔다(원본 npy는 19,907행, 여기서는 12,860행만 추출).

벡터를 꼭 parquet 한 파일로 원하시면 `pa.list_(pa.float32())` 컬럼으로 넣을 수 있으나
(12,860 × 768 × 4B × 2 ≈ 79 MB) 읽기·조인 비용이 커서 npy 분리를 권한다.

---

## 8. 요약 — 확정된 것과 확인이 남은 것

**확정**
- `x_hat = decoder(Q1[c1] + Q2[c2] + Q3[c3])`, decoder 입력은 세 code vector의 **단순 합**.
- `forward()`는 x_hat을 반환하지 않으며, 기존 `eval_reconstruction.py`도 스칼라 loss만 낸다.
- 추론은 `eval()`에서 완전히 결정적이며 `gumbel_t`는 무관.
- distance는 코드상 **항상 L2**, `normalize=False`, `sim_vq=False`, mode `STE`.
- SID 정수만으로는 reconstruction 불가 — codebook + decoder weight가 든 checkpoint 필요.
- shape: `x [B,768]` → `h [B,256]` → `embeddings [B,256,3]` → `z_q [B,256]` → `x_hat [B,768]`.
- 저장소의 `rqvae_ebnerd.gin`은 **c3=256**이지만 실제 SID는 **c3=512**. 권위는 checkpoint의 `model_config`.
- train/validation 모두 같은 `article_embeddings.npy`를 쓰고 `embedding_row`로 연결되며,
  두 split의 article_id 교집합은 0.

**확인이 남은 것 (학습 머신 접근 필요)**
1. `semantic_ids/selected_checkpoint.json` — 대표 checkpoint가 `best_rec`인지 `best_total`인지.
2. 해당 EXP의 `config.gin` / `metadata.json` — 최종 하이퍼파라미터 전문.
3. `checkpoint_*.pt` 와 `article_embeddings.npy` 전송.
4. (선택) `validation_event_c2_mapping.parquet` — validation SID 재현 검증용.
