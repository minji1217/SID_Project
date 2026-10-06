# AWS 실행 가이드: event-level C2 RQ-VAE 재학습

대상 브랜치: `claude/zealous-cerf-c1onwi`

서버 구조 (공용 서버):

```
/home/ubuntu/
├── minji/
│   ├── SID_Project-transformer/   (기존 Transformer, 그대로 둠)
│   └── SID_Project/               ← 이번에 새로 clone (RQ-VAE 재학습)
├── yeomin/                        (다른 분 작업 폴더, 건드리지 않음)
└── shared/datasets/ebnerd/        (A 기준 Transformer 입력, 공용. 쓰지 않음)
```

모든 작업은 `~/minji/SID_Project/` 안에서만 한다. `~/shared/`에는 아무것도 쓰지 않는다.

---

## 0. 무엇을 학습하나

```
A (baseline, 보존)   c2 = argmin_k || r1(a) - Q2[k] ||          기사마다 따로
이번 재학습           z(E) = mean{h(a) | a ∈ E}
                     EventCode(E) = argmin_k || z(E) - Q2[k] ||
                     c2 = EventCode[event(a)]                   같은 event면 같은 c2
```

하이퍼파라미터, seed, early stopping, best_rec 기준은 A와 같다.
다른 것은 `train.c2_mode = "event"`와 저장 경로뿐이다.
A는 Colab GPU에서 학습했으므로 AWS GPU 결과와는 소수점 수준 차이가 생길 수 있다.

## 1. 서버 확인

```bash
nvidia-smi          # 비어 있는 GPU 번호와 CUDA 버전 확인 (다른 분이 쓰는 GPU 피하기)
python3 --version   # 3.10+
```

## 2. 코드 받기

```bash
cd ~/minji
git clone -b claude/zealous-cerf-c1onwi https://github.com/minji1217/SID_Project.git
cd SID_Project
git log --oneline -1
```

## 3. 패키지 설치

torch를 먼저 서버 CUDA에 맞는 빌드로 설치한다.
(`requirements_rqvae.txt`에 `torch`가 들어 있어서 순서를 바꾸면 CPU 빌드가 깔릴 수 있다.)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install torch --index-url https://download.pytorch.org/whl/cu121   # nvidia-smi의 CUDA 버전에 맞게
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # True 확인
pip install -r RQVAE/requirements_rqvae.txt
```

## 4. 입력 데이터 3개 ★가장 틀리기 쉬움

같은 저장소의 `experiment/entity-linking-event` 브랜치에서 꺼낸다.
반드시 `data/output/model_inputs/` 경로를 쓴다.

```bash
cd ~/minji/SID_Project
git fetch origin experiment/entity-linking-event
mkdir -p RQVAE/datasets/ebnerd
for f in article_embeddings.npy article_master.parquet validation_article_master.parquet; do
  git cat-file -p origin/experiment/entity-linking-event:data/output/model_inputs/$f \
    > RQVAE/datasets/ebnerd/$f
done
sha256sum RQVAE/datasets/ebnerd/*
```

기대값:

| 파일 | 크기 | SHA256 |
|---|---|---|
| `article_embeddings.npy` | 63,648,896 | `89f8046d21df11342157b0ad1ecf32afe69ca74e5f7a7e5fc64d005f3b7b8e42` |
| `article_master.parquet` | 750,468 | `4b44f6fe118063190b654a9d6a39fdcb7e792f3844556f6c51c1a2cbe9a85c8c` |
| `validation_article_master.parquet` | 243,457 | `738c167d66d583a7fd57e88956c81b09103d026b01e0c0115f6a68e5661bd4e7` |

이 3개는 A SID를 byte 단위로 재현한 입력과 같다 (기사 12,860개 event_id, embedding_row 전부 일치).

### 다른 경로를 쓰면 안 되는 이유

같은 브랜치에 master 변종이 여러 개 있고 `event_id`가 다르다.
이번 학습은 `event_id`가 C2에 직접 쓰이므로, 잘못된 경로를 쓰면 에러 없이 다른 모델이 학습된다.

| 경로 | A와 Train event_id 일치 |
|---|---|
| **`data/output/model_inputs/`** | **9,738 / 9,738** |
| `data/output/experiments/normalize_v2/model_inputs/` | 9,738 / 9,738 (같은 내용) |
| `data/output/exports/rqvae_*_inputs/` | 2,012 / 9,738 |
| `data/output/experiments/baseline/model_inputs/` | 2,012 / 9,738 |
| `data/output/experiments/normalize_only/model_inputs/` | 2,489 / 9,738 |

## 5. 학습 전 검증 (수 초)

```bash
cd ~/minji/SID_Project/RQVAE
python tests/check_event_c2.py
```

마지막 줄에 `all checks passed`가 나와야 한다. 실패하면 학습을 시작하지 말고 에러 메시지를 그대로 전달한다.

## 6. 학습 실행

```bash
cd ~/minji/SID_Project/RQVAE
mkdir -p out/rqvae/ebnerd/eventc2_lu0.05_m0.5

CUDA_VISIBLE_DEVICES=0 nohup python train_rqvae.py configs/rqvae_ebnerd_eventc2_aws.gin \
  > out/rqvae/ebnerd/eventc2_lu0.05_m0.5/train.log 2>&1 &
echo $!    # PID
```

- `CUDA_VISIBLE_DEVICES`는 1단계에서 확인한 비어 있는 GPU 번호로 바꾼다. GPU는 1장만 쓴다 (여러 프로세스면 assert로 중단).
- `nohup ... &`라서 ssh가 끊겨도 계속 돈다.
- 학습 명령의 출력을 `| head`로 파이프하지 않는다. 로그 파일을 `grep`, `tail`, `head`로 읽는 것은 학습에 영향이 없다.
- 최소 200 epoch, 최대 1283 epoch (early stopping: patience 20, min_delta 0.001, eval 10 epoch마다).

### 진행 확인

```bash
cd ~/minji/SID_Project/RQVAE
grep -a "EventC2 epoch\|Validation epoch" out/rqvae/ebnerd/eventc2_lu0.05_m0.5/train.log | tail
ps aux | grep train_rqvae | grep -v grep
tail -1 out/rqvae/ebnerd/eventc2_lu0.05_m0.5/event_c2_log.jsonl | python -m json.tool
```

`event_c2_log.jsonl`에는 epoch마다 same-event C2 consistency(100%여야 함), EventCode churn,
Q2/Q3 code usage와 dead code, loss 분해, c123 collision rate, max_c4가 들어간다.

## 7. 학습이 끝나면 SID 생성

```bash
cd ~/minji/SID_Project/RQVAE
python generate_semantic_ids.py \
  --data_dir datasets/ebnerd \
  --checkpoint out/rqvae/ebnerd/eventc2_lu0.05_m0.5/checkpoint_best_rec.pt \
  --output_dir out/semantic_ids/eventc2_lu0.05_m0.5 \
  --train_c2_policy event
```

- `--train_c2_policy event`를 빼면 에러로 중단된다 (event checkpoint 보호).
- `sid_generation_meta.json`의 `checkpoint_event_code_table_agreement`가 checkpoint에 저장된 EventCode 표와 재계산 표의 일치율이다.

## 8. 결과 전달

```
~/minji/SID_Project/RQVAE/out/rqvae/ebnerd/eventc2_lu0.05_m0.5/
    event_c2_log.jsonl
    train.log
    checkpoint_best_rec.pt
~/minji/SID_Project/RQVAE/out/semantic_ids/eventc2_lu0.05_m0.5/   (폴더 통째로)
```

`out/`은 gitignore라 git에 올라가지 않는다. scp로 내려받는다.

```bash
# 로컬에서
scp -i <키.pem> -r ubuntu@<서버IP>:~/minji/SID_Project/RQVAE/out/semantic_ids/eventc2_lu0.05_m0.5 .
scp -i <키.pem> ubuntu@<서버IP>:~/minji/SID_Project/RQVAE/out/rqvae/ebnerd/eventc2_lu0.05_m0.5/{event_c2_log.jsonl,train.log,checkpoint_best_rec.pt} .
```

비교 스크립트: `experiments/eventc2_train_v1/compare_sid_ab.py`.

## 9. 다음 단계(Transformer) 주의

새 SID로 1pos4neg 데이터를 다시 만들 때 `~/shared/datasets/ebnerd/`에 덮어쓰지 않는다.
그 파일들은 A 기준이고 다른 분도 함께 쓴다. 새 데이터는 `~/minji/` 아래 별도 폴더에 두고
Transformer gin의 데이터 경로만 바꾼다.

## 10. 막혔을 때

| 증상 | 확인할 것 |
|---|---|
| `check_event_c2.py` 실패 | 학습 시작하지 말고 전체 에러 메시지 전달 |
| `torch.cuda.is_available()` False | torch를 CUDA 빌드로 다시 설치 |
| 시작하자마자 assert로 멈춤 | `CUDA_VISIBLE_DEVICES`로 GPU 1장만 지정했는지 |
| 입력 파일 못 찾음 | `ls RQVAE/datasets/ebnerd/`에 3개 다 있는지 |
| SID 생성 시 중단 | `--train_c2_policy event`가 빠졌는지 |
| same_event_c2_consistency ≠ 100% | 학습이 assert로 멈춤. `event_c2_log.jsonl` 전달 |
| 로그가 멈춘 것처럼 보임 | `nvidia-smi`로 GPU 사용 확인 (eval은 10 epoch마다) |
