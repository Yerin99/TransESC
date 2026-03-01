# TransESC 재현 실험 가이드

> **논문**: TransESC: Smoothing Emotional Support Conversation via Turn-Level State Transition
> **학회**: ACL Findings 2023
> **저자**: Weixiang Zhao, Yanyan Zhao, Shilong Wang, Bing Qin (Harbin Institute of Technology)
> **코드**: https://github.com/circle-hit/TransESC
> **데이터**: https://drive.google.com/drive/folders/1rNpa5vDuB7KssQuVBAhkKtmAuqPa0Qgf

---

## 1. 핵심 아이디어

ESC를 턴 단위 상태 전이(turn-level state transition) 관점에서 모델링한다. 세 가지 전이를 동시에 포착:

1. **Semantics Transition** — 대화가 진행되면서 seeker가 말하는 내용의 의미 변화
2. **Strategy Transition** — supporter가 사용하는 전략 간의 의존성/순서 패턴
3. **Emotion Transition** — seeker의 감정 상태 변화 추적

---

## 2. 전체 아키텍처 (3개 모듈)

```
[Context Encoder] → [Turn-Level State Transition Module] → [Transition-Aware Decoder]
     (§4.1)                    (§4.2, §4.3)                       (§4.4)
```

### 2.1 Context Encoder (§4.1)

- **백본**: BlenderBot-small (90M) 의 Transformer Encoder
- 대화를 하나의 단어 시퀀스로 flatten
- 각 발화 앞에 `[CLS]` 토큰 추가 + 다음 응답을 위한 `[CLS]` 하나 더 추가
- 출력: `H_c ∈ R^{N × d_h}` (N = 발화 수)

### 2.2 Turn-Level State Transition Module (§4.2)

#### 2.2.1 State Transition Graph 구조

**노드 (Node)**:
- 각 대화 턴이 하나의 노드
- **Supporter 턴 노드**: Semantics State + Strategy State (2개 state)
- **Seeker 턴 노드**: Semantics State + Emotion State (2개 state)

**엣지 (Edge)** — 총 7종류, 2그룹:

| 그룹 | 엣지 타입 | 설명 |
|------|----------|------|
| **Transition Edges (T)** | Sem-Sem | 같은 유형 state 간 전이 (의미 → 의미) |
| | Stra-Stra | 전략 → 전략 |
| | Emo-Emo | 감정 → 감정 |
| **Interaction Edges (I)** | Sem-Emo | 의미 ↔ 감정 상호작용 |
| | Sem-Stra | 의미 ↔ 전략 상호작용 |
| | Emo-Stra | 감정 ↔ 전략 상호작용 |
| | Stra-Emo | 전략 ↔ 감정 상호작용 |

**연결 방식**:
- 현재 턴과 이전 모든 턴 간 연결 (window 내)
- 4가지 role 조합: Seeker→Seeker, Seeker→Supporter, Supporter→Supporter, Supporter→Seeker

**Transition Window**: 고정 크기 `w` 내에서만 전이 수행
- 현재 supporter 응답 턴 `u_e`를 끝점으로
- `w`번째 이전 supporter 발화 `u_s`를 시작점으로
- 이 사이의 모든 발화가 transition window 구성
- **최적 window size: w = 2**

#### 2.2.2 Graph Initialization

| State | 초기화 방법 |
|-------|-----------|
| **Semantics State** | 해당 발화의 `[CLS_i]` 토큰 임베딩 |
| **Strategy State** | 해당 발화의 `[CLS_i]` 토큰 임베딩 |
| **Emotion State** | `[CLS_i]` + COMET 감정 지식 `csk_i` |

**COMET 감정 지식 획득**:
- COMET 모델: **BART-based** variation, **ATOMIC-2020** 데이터셋으로 학습된 것
- 관계 타입: **xReact** ("PersonX는 이벤트 후 어떻게 느끼는가?")
- 입력 형식: `(X_i, xReact, [GEN])` → seeker의 각 발화 X_i에 대해
- COMET 마지막 레이어의 hidden state를 감정 지식 표현 `csk_i`로 사용
- 최종 Emotion State = `[CLS_i]` + `csk_i`

#### 2.2.3 Transit-Then-Interact (TTI) — 2단계 업데이트

**핵심 메커니즘**: Relation-Enhanced Multi-Head Attention (R-MHA)

일반 MHA:
```
v̂_i = MHA_{j∈N}(q_i, k_j, v_j)
```

R-MHA는 엣지 타입 임베딩 `r_ij`를 query와 key에 더함:

**Step 1 — Transit** (같은 타입 state 간 전이):
```
s'_i = R-MHA_{e_ij ∈ T}(s_i + r_ij, s_j + r_ij, s_j)
```

**Step 2 — Interact** (다른 타입 state 간 상호작용):
```
s''_i = R-MHA_{e_ij ∈ I}(s'_i + r_ij, s'_j + r_ij, s'_j)
```

**Dynamic Fusion** (Transit + Interact 결합):
```
ŝ_i = g^tti ⊙ s'_i + (1 - g^tti) ⊙ s''_i
g^tti = σ([s'_i; s''_i] W^tti + b^tti)
```
- `W^tti ∈ R^{2d_h × d_h}`, `b^tti ∈ R^{d_h}`
- Strategy State `ŝt_i`, Emotion State `ê_i`도 동일한 방식으로 업데이트

### 2.3 State Prediction (§4.3) — 턴 레벨 감독 신호

#### 2.3.1 Semantic Keyword Prediction (Bag-of-Words Loss)

TTI 전후의 차이 벡터 활용:
```
Δ_i = ŝ_i - s_i   (전이 후 - 전이 전)
```
Keyword set `K_i = [k_1, k_2, ..., k_k]` 예측 (비자기회귀):
```
f = softmax(W^sem Δ_i + b^sem)     W^sem ∈ R^{d_h × |V|}
```
```
L_SEM = -Σ_{i=1}^{N} Σ_{j=1}^{k} log f_{k_j^i}
```

#### 2.3.2 Strategy Prediction (Cross-Entropy)

```
ŷ_str = softmax(W^str ŝt_i + b^str)     W^str ∈ R^{d_h × n_s}, n_s = 8
```
```
L_STR = -(1/N) Σ_{i=1}^{N} Σ_{j=1}^{n_s} ŷ^j_{str,i} · log(y^j_{str,i})
```

#### 2.3.3 Emotion Prediction (Cross-Entropy)

```
ŷ_emo = softmax(W^emo ê_i + b^emo)     W^emo ∈ R^{d_h × n_e}, n_e = 6
```
6개 감정 카테고리: **joy, anger, sadness, fear, disgust, neutral**
```
L_EMO = -(1/N) Σ_{i=1}^{N} Σ_{j=1}^{n_e} ŷ^j_{emo,i} · log(y^j_{emo,i})
```

### 2.4 Transition-Aware Decoder (§4.4)

기본 Transformer Decoder에 3가지 전이 정보 주입:

#### (1) Strategy Injection (입력 단계)

마지막 Strategy State `ŝt`를 디코더 입력 임베딩과 동적 융합:
```
Ê_i = g^str ⊙ E_i + (1 - g^str) ⊙ ŝt
g^str = σ([E_i; ŝt] W_1 + b_1)
```

#### (2) Emotion Injection (Cross-Attention 단계)

Seeker의 감정 상태 시퀀스 `H_emo` 구성:
- Seeker 턴: 감정 상태 `e_i`
- Supporter 턴: COMET의 **oReact** 관계로 생성한 감정 효과 지식 `e^oR_i`

```
Ĥ_emo = Cross-Att(H_c, H_emo)
Ĥ = g^emo ⊙ H_c + (1 - g^emo) ⊙ Ĥ_emo
g^emo = σ([H_c; Ĥ_emo] W_2 + b_2)
```

#### (3) Semantics Injection (출력 단계)

마지막 semantics 차이 벡터 `Δ_i`를 디코더 hidden state와 융합:
```
ĥ = g^sem ⊙ h_t + (1 - g^sem) ⊙ Δ_i
g^sem = σ([h_t; Δ_i] W_3 + b_3)    (논문에서 W_sem이라 표기했지만 W_3)
```

최종 토큰 분포:
```
P(y_t | y_{<t}, D) = softmax(W ĥ + b)
```

### 2.5 총 Loss Function (Multi-Task Learning)

```
L = γ₁ L_GEN + γ₂ L_SEM + γ₃ L_STR + γ₄ L_EMO
```

| 하이퍼파라미터 | 값 | 설명 |
|-------------|---|------|
| γ₁ | 1.0 | 응답 생성 Loss 가중치 |
| γ₂ | 0.2 | Semantic Keyword Loss 가중치 |
| γ₃ | 1.0 | Strategy Prediction Loss 가중치 |
| γ₄ | 1.0 | Emotion Prediction Loss 가중치 |

---

## 3. 데이터 전처리

### 3.1 ESConv 데이터셋

- 1,300 대화, 평균 29.8 턴/대화
- 8개 전략: Question, Restatement or Paraphrasing, Reflection of Feelings, Self-disclosure, Affirmation and Reassurance, Providing Suggestions, Information, Others
- 공식 train/dev/test split 사용

### 3.2 추가 어노테이션 (원본 ESConv에 없는 것)

ESConv에는 **키워드**, **감정 라벨**이 없으므로 외부 도구로 자동 어노테이션:

| 어노테이션 | 방법 | 비고 |
|-----------|------|------|
| **키워드 (Keywords)** | 각 발화에서 top-k 키워드 추출 | 구체적 도구는 GitHub 데이터 참조 |
| **감정 라벨 (Emotion)** | 자동 감정 분류기 | 6 카테고리: joy, anger, sadness, fear, disgust, neutral |
| **COMET 지식** | BART-based COMET (ATOMIC-2020) | xReact, oReact 관계 사용 |

**중요**: GitHub에서 전처리된 데이터를 제공함
- 전처리된 데이터셋
- 어노테이션된 감정/키워드
- 생성된 commonsense knowledge
→ 모두 `/dataset` 디렉토리에 배치

### 3.3 COMET 상세

| 항목 | 설정 |
|------|------|
| 모델 | BART-based COMET |
| 학습 데이터 | ATOMIC-2020 |
| 사용 관계 (Emotion State 초기화) | **xReact** — "사건 후 PersonX는 어떻게 느끼나?" |
| 사용 관계 (Decoder Emotion Injection) | **oReact** — "사건 후 다른 사람은 어떻게 느끼나?" |
| 입력 형식 | `(X_i, relation, [GEN])` |
| 출력 | 마지막 레이어 hidden state 표현 |

ATOMIC-2020의 9개 관계 타입 참고:
- xIntent, xNeed, xAttr, xEffect, xWant, **xReact**, **oReact**, oWant, oEffect

---

## 4. Implementation Details

### 4.1 모델 하이퍼파라미터

| 항목 | 값 |
|------|---|
| **백본 모델** | `facebook/blenderbot_small-90M` |
| **모델 파라미터** | 90M |
| **Hidden dimension (d_h)** | 300 |
| **Vocabulary & hidden size** | BlenderBot-small 기본값 사용 |
| **Transition window size (w)** | 2 |
| **R-MHA attention heads** | 16 |
| **Emotion-aware attention heads** | 4 |
| **전략 수 (n_s)** | 8 |
| **감정 카테고리 수 (n_e)** | 6 |

### 4.2 학습 설정

| 항목 | 값 |
|------|---|
| **Optimizer** | AdamW (β₁=0.9, β₂=0.999) |
| **Initial learning rate** | 2e-5 |
| **LR Schedule** | Linear warmup |
| **Warmup steps** | 120 |
| **Batch size** | 20 |
| **GPU** | 1× NVIDIA Tesla A100 |
| **Loss weights** | γ₁=1, γ₂=0.2, γ₃=1, γ₄=1 |

### 4.3 디코딩 설정

| 항목 | 값 |
|------|---|
| **디코딩 방식** | Top-p + Top-k Sampling |
| **Top-p** | 0.3 |
| **Top-k** | 30 |
| **Temperature (τ)** | 0.7 |
| **Repetition penalty** | 1.03 |

※ Liu et al. (2021)의 디코딩 설정을 따름

---

## 5. 평가 메트릭

### 5.1 자동 평가

| 메트릭 | 측정 대상 |
|--------|----------|
| **ACC** | 전략 예측 정확도 (8-class) |
| **PPL** | Perplexity — 생성 품질 |
| **B-1, B-2, B-3, B-4** | BLEU — 어휘/의미 유사도 |
| **R-L** | ROUGE-L — 최장 공통 부분 시퀀스 |
| **D-1, D-2** | Distinct — 응답 다양성 (unique n-gram 비율) |

### 5.2 주요 결과 (재현 타겟)

| Model | ACC | PPL | D-1 | D-2 | B-1 | B-2 | B-3 | B-4 | R-L |
|-------|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| BlenderBot-Joint | 17.69 | 17.39 | 2.96 | 17.87 | 18.78 | 7.02 | 3.20 | 1.63 | 14.92 |
| MISC | 31.67 | 16.27 | 4.62 | 20.17 | 16.31 | 6.57 | 3.26 | 1.83 | 17.24 |
| **TransESC** | **34.71** | **15.85** | **4.73** | **20.48** | **17.92** | **7.64** | **4.01** | **2.43** | **17.51** |

### 5.3 Ablation 결과 (각 전이 제거 시)

| Model | D-1 | B-2 | B-4 | R-L |
|-------|-----|-----|-----|-----|
| TransESC (full) | 4.73 | 7.64 | 2.43 | 17.51 |
| w/o Sem. Trans | 4.55 | 7.04 | 2.13 | 17.37 |
| w/o Stra. Trans | 4.29 | 6.68 | 2.01 | 17.15 |
| w/o Emo. Trans | 4.82 | 7.14 | 2.22 | 17.45 |
| w/o T-L. Trans (전부) | 4.19 | 6.35 | 1.94 | 16.88 |

→ **Strategy Transition 제거가 가장 큰 성능 하락** (B-2: 7.64 → 6.68)

---

## 6. 재현 시 주의사항

### 6.1 데이터 관련
- ESConv 공식 split 사용 (MISC가 re-split한 것과 다름)
- 키워드/감정 어노테이션은 GitHub 전처리 데이터 그대로 사용 권장
- COMET 지식도 미리 생성된 것 제공됨

### 6.2 모델 관련
- BlenderBot-small의 vocab/hidden size를 **기본값** 그대로 사용
- d_h=300은 state transition graph의 hidden dimension (BlenderBot의 hidden과 별개일 수 있음 — 코드 확인 필요)
- R-MHA에서 엣지 타입 임베딩 `r_ij`의 차원 = d_h

### 6.3 Transition Window
- w=2가 최적 (Table 5에서 확인)
- w=1은 너무 짧아서 의존성 부족
- w≥3은 redundant 정보로 성능 하락

### 6.4 COMET 사용 주의
- **xReact**: Seeker 발화에 대해 → Emotion State 초기화용
- **oReact**: Supporter 발화에 대해 → Decoder의 Emotion Injection용
- 두 관계를 혼동하지 않을 것

### 6.5 코드 실행
```bash
# 1. 코드 클론
git clone https://github.com/circle-hit/TransESC.git
cd TransESC

# 2. 데이터 다운로드 (Google Drive 링크에서)
# → /dataset 디렉토리에 배치

# 3. 의존성 설치 (requirements.txt 확인)

# 4. 학습 실행 (구체적 명령어는 README 참조)
```
