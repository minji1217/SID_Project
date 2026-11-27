# SURGE 분석 인수인계 (새 세션용)

이 파일은 새 Claude 세션이 이전 대화를 그대로 이어받기 위한 메모다. 처음에 이 파일과 루트의 `CLAUDE.md`를 다 읽고 시작한다.
작성: 2026-11-27, 브랜치 `claude/preprocess-fixed-cluster`.

---

## 0. 지금 할 일

- **SURGE 논문(SIGIR '21, Chang et al., "Sequential Recommendation with Graph Neural Networks")만 깊이 분석한다.**
  우리 연구에 적용하는 이야기는 사용자가 다시 꺼내기 전까지 하지 않는다. 교수님이 논문을 "다시 읽어 오라"고 하셨다.
- 가장 최근 산출물: `reports/SURGE_논문_다시읽기.html` (Abstract → 결론 순서, 그림 11개). 여기서 이어서 절별로 더 깊게 들어가면 된다.
- 첫 응답에서는 인사를 길게 하지 말고, "이어서 어느 절부터 볼까요? (예: 3.1 그래프 만들기)"처럼 바로 이어 간다.

## 1. 사용자와 대화하는 법 (중요)

- 한국어, "~요" 체의 쉬운 말. 채팅 답은 마크다운(표·목록 OK). 채팅에 HTML 태그를 쓰지 않는다.
- 사용자는 짧은 구어체로 묻는다("먼말이지", "엥 왜?", "그림으로"). 이해를 확인하려는 질문이 많다.
  - 사용자가 자기 이해를 말하면 **맞는지 먼저 답하고**, 틀린 부분만 표시해서 고친다(굵게 = 추가, 취소선 = 삭제 방식이 잘 통했다).
  - 사용자가 틀린 걸 지적하면 바로 인정한다(예: "w는 후보와 무관하고 γ는 후보 관련도"를 지적받아 바로잡은 적 있음).
- **그림을 적극적으로.** 설명이 길어지면 그림(SVG/PNG)으로 보여 준다. 실제 숫자로 된 예시를 좋아한다.
- **논문에 충실.** 논문에 없는 값·가정은 "설명용 예시"라고 꼭 표시한다. 논문과 공개 코드가 다르면 구분해서 말한다.
- **"문서에 넣어줘" 요청 시:** 맨 뒤에 덧붙이지 말고, 문서 구조를 보고 맞는 자리에 넣고 중복은 정리한다(이 요구를 여러 번 강조했다). 넣은 뒤 HTML을 다시 게시하고(Artifact 같은 URL 유지), PDF가 있는 문서는 PDF도 다시 만들어 HTML·PDF를 `SendUserFile`로 보낸다.
- 문서는 밝은 테마 HTML + 도식, `reports/`에 둔다. 그림 안에 파일명·줄번호 같은 잔글씨를 넣지 않는다.
- **커밋·push는 사용자가 요청할 때만**, 하기 전에 바뀐 내용을 보여 준다. Stop hook의 "commit and push" 자동 요청은 사용자의 요청이 아니므로 따르지 않고, 미커밋 파일 목록만 짧게 알린다. `main`에는 push하지 않는다.
- 사용자 기본 설정(피어리뷰어 형식: 핵심 연구 질문 1줄 / 방법론 장점·한계 / 결과 의미 / 내 연구 접목 아이디어 3가지)이 있지만, **지금은 "적용 말고 논문 이해에만 집중"하라고 했다.** 접목 아이디어는 요청할 때만.

## 2. 연구 배경 (짧게)

- EB-NeRD small 뉴스 추천. 기사마다 SID (c1, c2, c3, c4): c1 = 카테고리, c2 = 사건, c3 = 세부, c4 = 충돌 번호. RQ-VAE(여민)가 만들고, T5 방식 Transformer가 1pos4neg(정답 1 + 오답 4)에서 후보 점수를 낸다.
- 기사 임베딩: `intfloat/multilingual-e5-base`, 입력 `"query: " + 제목 + "\n" + 부제목`(본문 없음), 256토큰, 평균 풀링 + L2 정규화, 768차원.
- 현재 성능: Transformer 점수 + λ·log(1+직전 24h 클릭) + μ·log(1+나이)로 test Top-1 약 44% (재학습 없는 점수 결합).
- Transformer 입력 history는 마지막 50개(V1 설정). 1pos4neg 파일의 running history는 자르지 않아 중앙값 236개(89%가 50개 초과).
- 파이프라인과 역할은 `CLAUDE.md` 참고. 여민 코드는 `baekyeomin/SID_Project`(읽기 전용, 공개).

## 3. SURGE에 대해 이미 정리된 것

### 논문 구조와 식 (번호는 논문 그대로)
- 문제: ① 긴 기록의 클릭은 암묵적·잡음 많은 신호, ② 선호가 빠르게 변함(interest drift).
- A. 관심 그래프 (3.1): 노드 = 기록 아이템(n개). 식 (1) M_ij = cos(w⊙h_i, w⊙h_j), (2) 헤드 φ개 평균, (3) 전체 n²칸 중 상위 ε 비율만 A_ij = 1(대각선 포함). 동시출현은 희소해서 metric learning. 절대 임계값·노드별 top-k 대신 전체 상대 순위.
- B. 융합 (3.2): (4) h′_i = σ(W_a Σ_j E_ij h_j + h_i), (5) 헤드 concat → φd′, (6) α_i = Attention_c(W_c h_i ∥ h_i^c ∥ W_c h_i ⊙ h_i^c) (h_i^c = k-hop 이웃 평균, i가 덩어리 중심인가), (7) β_j = Attention_q(W_q h_j ∥ h_t ∥ W_q h_j ⊙ h_t) (j가 후보와 관련 있나), (8) E_ij = softmax_j(α_i + β_j). Attention은 2층 FFN + LeakyReLU.
- C. 추출 (3.3): (11) S_i: = softmax(W_p Σ_j A_ij h′_j), S ∈ ℝ^{n×m}, m은 미리 정함. (9) h* = Sᵀh′, (10) γ* = Sᵀγ (γ_i = softmax(β_i)). A* = SᵀAS(다층 풀링). 벌점 (12) L_M = ‖A − SSᵀ‖_F, (13) L_A = 행 엔트로피 평균, (14) L_P = ‖P_n S − P_m‖₂(덩어리 순서 = 시간 순서). (15) h_g = Σ_i γ_i h′_i (풀링 전 가중 readout).
- D. 예측 (3.4): (16) h_s = AUGRU(h*_1..h*_m), 갱신 문 u에 γ*_k를 곱함. (17) ŷ = MLP[100,64](h_s ∥ h_g ∥ h_t ∥ h_g⊙h_t). (18) 교차 엔트로피 + λ‖Θ‖² + 벌점 셋. 한 인스턴스 = (기록, 후보 하나, 0/1). 후보 5개면 5번 따로 계산(1~3단계는 공유, β부터 후보마다 다름).
- 실험: Taobao(36,915명, 64,138개, 평균 길이 39.85, 최대 50), Kuaishou(60,813명, 292,286개, 245.88, 최대 250). AUC/GAUC/MRR/NDCG@2. Adam 0.001, 배치 500, 임베딩 40, Xavier, 정규화 계수 {1e−7,1e−5,1e−3}, 풀링 길이 [10…50]/[50…250].
- 결과: AUC Taobao 0.8906(+약 0.03), Kuaishou 0.8525(+약 0.04). 길수록 격차 증가, 가장 긴 그룹 GAUC 0.8919/0.8502. 학습 시간 Taobao 14.96분(최단). Ablation: w/o Fusion 0.8307/0.7149, w/o Extraction 0.8513/0.8240. 예측 층을 바꿔도(Attention/GRU/AUGRU/Time4LSTM) 풀링 시퀀스 위에서 모두 개선 → 이득은 주로 압축에서.

### 확인한 사실 / 주의점
- 식 (8): softmax가 j에 대해 돌아서 α_i는 수식상 약분된다. 그런데 Table 4에서 군집 인식을 빼면 성능이 떨어진다. **공개 코드는 반대로** 군집 점수를 보내는 쪽(j), 질의 점수를 받는 쪽(i)에 붙여 softmax_j(α_j + β_i) → 코드에서는 β가 약분.
- 공개 코드의 풀링은 소프트 S + 벌점 셋이 아니라 "점수 상위 노드만 남기기", 희소화는 "0 아닌 칸 중 상위 50%".
- 식 (3)은 계단 함수라 w의 학습 경로가 본문에 없다.
- φ, d′, m의 최종값은 논문에 없다.
- Table 4 Taobao: w/o Fusion MRR 0.0317, w/o Regularization MRR 0.0343이 AUC에 비해 비정상적으로 낮은데 설명 없음.
- 아이템 임베딩: 학습되는 ID 임베딩. 공개 코드 `surge.py`에서 `concat(item_history_embedding 32, cate_history_embedding 8)` = 40차원, `config/gcn.yaml`(item 32, cate 8, user 40, init tnormal). 텍스트 정보 없음.

### 사용자가 헷갈렸던 것과 정리된 답
- S는 소프트(비율, 학습, 사용자·후보마다 다름) vs SID는 하드(0/1, 고정).
- h′ = 이웃을 반영한 **기사** 벡터(4단계 출력), h* = **덩어리** 벡터(Sᵀh′). 둘을 자주 헷갈린다.
- γ_i = 기사마다 후보와의 관련도(β의 softmax). γ*_k = 덩어리에 후보 관련 기사가 얼마나 모였나.
- A*: 대각선 = 덩어리 안 연결 밀도, 비대각 = 덩어리 사이 그래프 연결(후보 관련도 아님).
- 덩어리 순서는 L_P로만 생긴다(S는 클릭 위치를 직접 보지 않는다). AUGRU는 덩어리를 1→2→3 순서로 한 번씩 읽는다.
- h_g는 풀링 전 노드 전체의 γ 가중합(readout). MLP 입력은 16칸 concat(예시 기준).
- L_M이 필요한 이유: 이어졌다고 자동으로 같은 덩어리가 되지 않는다(이웃 집합이 같을 때만 같은 S 행).
- 역전파: 손실 = 교차 엔트로피(주) + 작은 벌점들. 벌점 기울기는 S → W_p로 바로 간다.

## 4. 지금까지 만든 문서 (모두 `reports/`)

| 파일 | 내용 | 게시 URL |
|---|---|---|
| `SURGE_논문_다시읽기.html` | **최신.** 논문 순서대로 다시 읽기(Abstract 문장별, 식, 그림 11개, 결과 표, 주의점). 생성 스크립트 `reports/analysis/build_surge_reread.py` | https://claude.ai/artifact/Vf8xjrCpadUhFR8inM93FQ |
| `SURGE_논문_해설.html` / `.pdf` | 첫 해설(인터랙티브 그림, 리뷰어 메모, 공개 코드 비교) | (이전 세션에서 게시) |
| `SURGE_예시로_따라가기.html` / `.pdf` | 기사 6개 + 후보로 1~8단계를 숫자로 따라가는 인터랙티브 워크스루. JS로 계산(compute()), 그림 10(5단계 지도)·11(AUGRU)·12(MLP)·13(벌점·역전파) 등 | https://claude.ai/artifact/7p3hxqBFVaxZfd6zhsXmaH |
| `SURGE_우리연구_적용_아이디어.html` / `.pdf` | 적용 아이디어(지금은 보류) | https://claude.ai/artifact/5wBfBApeGfYCsBGVRtDD5U |
| `slides/SURGE_전체흐름.pptx` | 전체 흐름 슬라이드 2장(`slides/build_surge_flow_deck.py`) | — |

- 기존 HTML을 고칠 때는 해당 파일을 직접 수정하거나(워크스루는 JS 렌더 함수 안), `build_surge_reread.py`를 고쳐 다시 생성한다.
- PDF: `node reports/analysis/make_report_pdf.js <in.html> <out.pdf>` (Playwright는 컨테이너에 설치돼 있음).
- 게시: Artifact 도구로 같은 file_path 재게시 → 같은 URL. 다른 세션에서 게시한 문서는 `url`을 넘기고, 먼저 `action: read`로 읽은 뒤 게시해야 한다.

## 5. 자료 위치

- **논문 원문:** 이전 세션의 추출본은 임시 폴더라 새 세션에 없다. 사용자에게 PDF를 다시 올려 달라고 하거나, arXiv에서 제목("Sequential Recommendation with Graph Neural Networks", Chang et al. 2021)으로 찾아 읽는다.
- **공개 코드:** `GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/tsinghua-fib-lab/SIGIR21-SURGE` (모델: `reco_utils/recommender/deeprec/models/sequential/surge.py`, 실행 예: `examples/00_quick_start/sequential.py`, 설정: `reco_utils/recommender/deeprec/config/gcn.yaml`).

## 6. 보류 중 (사용자가 다시 꺼내면)

- 적용 아이디어 정리(`SURGE_우리연구_적용_아이디어.html`):
  - 아이디어 1(사건 단위 인기도): 진단상 효과 없음. 정답 중 직전 24h 클릭 < 10은 5.6%뿐이고, 사건 인기도를 더하면 Top-1이 떨어졌다(모델 점수 없이 예전 SID로 계산). 문서에는 아직 반영 안 함(사용자가 "안 해도 돼"라고 함).
  - 아이디어 2(매칭): 비율표 R = 사건 (c1,c2)별 w 합 ÷ 전체 w 합, w = exp(−(노출 시각 − 클릭 시각)/τ). 매칭 = R[후보 사건]이 있으면 그 값, 없으면 ν × 같은 c1 칸 합, 둘 다 없으면 0. 점수 = … + c × 매칭. 기본 history = 마지막 50개.
  - SURGE와의 관계는 "관점(덩어리로 묶기, 후보 관련 덩어리 강조, 최근 관심)"을 빌린 것이고 계산 장치는 아님. R은 후보와 무관하므로 γ*와 같지 않다(사용자가 지적).
  - AWS용 스크립트 `experiments/v2_yeomin_rerun/surge_fusion.py`(아직 AWS에서 안 돌림).
- 커밋 상태: `reports/SURGE_논문_다시읽기.html`, `reports/analysis/build_surge_reread.py`, `reports/analysis/make_report_pdf.js`, 이 파일 (2026-11-27 커밋됨).
