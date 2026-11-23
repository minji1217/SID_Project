# V1 Transformer 설정 (여민 Transformer 코드용)

새 SID로 Transformer를 다시 학습할 때 V1(test Top-1 27.07%)과 같은 조건으로 돌리기 위한 gin 파일.

- `transformer_ebnerd_v1.gin`: 여민 저장소(`baekyeomin/SID_Project` main)의 `Transformer/train_transformer.py`에서 바로 읽히는 형식.
- 값의 출처: `minji1217/sid_project-transformer` @ `claude/hopeful-mendel-fhc2n4`
  - `Transformer/configs/transformer_ebnerd.gin` (기본값)
  - `Transformer/analysis/rerun_final_config.py`의 `FINAL_CONFIG` (설정 탐색으로 고른 값, 기본값을 덮어씀)

## 여민 기본값과 다른 것

| 항목 | V1 | 여민 기본 |
|---|---|---|
| max_history_length | 50 | 20 |
| use_sep | False | True |
| d_model / num_heads / num_layers | 256 / 8 / 2 | 384 / 6 / 4 |
| dropout_rate | 0.0 | 0.1 |
| learning_rate / weight_decay | 5e-5 / 0.0 | 1e-4 / 0.01 |

같은 것: d_ff 1024, batch 128, 최대 30 epoch, patience 5, gradient_clip_norm 1.0, seed 42, best = validation 앞 절반 Top-1 최고 epoch.

## 실행

```bash
cd Transformer
python train_transformer.py \
  --config <이 폴더>/transformer_ebnerd_v1.gin \
  --sid-path <새 article_semantic_ids.parquet>
```

- `train.train_path`, `train.validation_path`, `train.save_dir`는 실험마다 바꾼다 (gin 파일 수정 또는 복사본).
- 어휘 크기(c1~c4)는 `--sid-path`의 SID 파일에서 (최댓값 + 1)로 자동으로 들어간다. 그래서 gin에 적지 않는다.

## V1과 완전히 같지 않은 점

모델·loss·평가 식·split 방식은 V1 코드와 같다. 조건(fp32, num_workers 0)을 맞추면 V1 코드와 epoch마다 loss·Top-1이 비트 단위로 같았다 (2026-11-23, 합성 데이터 CPU 실측). 다른 것은 셋:

- AMP: V1은 bfloat16(autocast)으로 학습·validation 평가. 여민 코드는 fp32만 있다.
- DataLoader: V1은 `num_workers=4` + persistent workers라 epoch 2부터 섞는 순서가 달라진다. 여민 코드는 이 옵션이 없다.
- 어휘 크기: "SID 파일 최댓값 + 1"로 자동. V1은 25 / 128 / 512 / 28 고정. 예전 V1 SID 파일은 c4 최댓값이 16이라 c4 어휘가 17이 된다.

그래서 V1(기존 SID)도 이 코드와 이 gin으로 다시 돌려 새 기준으로 쓴다. seed 사이 Top-1 표준편차가 약 0.4%p라 그 정도 차이는 예상한다.

## 실행할 때 주의

- 예측(`predict_sid.py`)에도 **이 gin**을 `--config`로 주고 `--checkpoint <save_dir>/checkpoint_best.pt`를 명시한다. 기본 경로는 학습 저장 위치와 다르고, 다른 gin을 주면 history 길이(50)가 경고 없이 바뀐다.
- 학습 때 `--sid-path`를 항상 명시한다. 빼면 `train_path` 옆의 `article_semantic_ids.parquet`를 자동으로 골라 다른 SID가 잡힐 수 있다.
- `split_validation.py`는 입력·출력 폴더가 `datasets/ebnerd/`로 고정이다. 실험(V1 / H / R)마다 폴더를 나눠 덮어쓰지 않게 한다. `save_dir`도 실험마다 새 폴더.
- 입력 1pos4neg가 후보 순서를 섞은 버전인지 먼저 확인한다 (정답이 첫 자리인 비율 약 0.2). 코드가 검사하지 않아서 섞지 않은 파일이면 Top-1이 부풀어도 경고가 없다.
- 학습 로그에는 Top-1만 나온다. AUC · MRR · nDCG는 예측 결과로 `evaluate/metrics.py`를 돌려 구한다.
