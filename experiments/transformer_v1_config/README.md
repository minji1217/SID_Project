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

- V1은 AMP(bfloat16)로 학습했다. 여민 코드에는 AMP 설정이 없어 fp32로 학습한다. 수치가 아주 조금 다를 수 있다.
- 어휘 크기가 "SID 파일 최댓값 + 1"이라 한 번도 안 쓰인 끝 번호가 있으면 V1(25 / 128 / 512)보다 작아질 수 있다. 학습은 되지만 출력층 크기가 달라진다.
- 엄밀한 비교가 필요하면 V1(기존 SID)도 이 코드와 이 gin으로 다시 돌려 기준을 맞춘다.
