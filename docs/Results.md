# TransESC 재현 실험 결과

## 1. 실험 개요

TransESC의 원본 데이터 분할(8:1:1)은 utterance-pair 단위 random split으로 인해 **data leakage** 문제가 존재한다.
같은 대화의 여러 sliding window 샘플이 train/dev/test에 분산되어, 모델이 test 시 이미 본 대화 맥락을 활용할 수 있다.

이를 해결하기 위해 **대화 단위(conversation-level) 70:15:15 split**을 수행하고,
동일 조건에서 재학습 및 추론하여 data leakage가 성능에 미치는 영향을 분석한다.

## 2. 데이터 분할

### 2.1 원본 (8:1:1, Leaky)

| Split | Samples |
|-------|---------|
| Train | 14,116  |
| Dev   | 1,763   |
| Test  | 1,763   |
| **Total** | **17,642** |

- Utterance-pair 단위 random split
- 같은 대화의 샘플이 train/test에 동시 존재 (data leakage)

### 2.2 Leak-Free (70:15:15, Conversation-Level)

| Split | Conversations | Samples |
|-------|---------------|---------|
| Train | 910           | 12,233  |
| Dev   | 195           | 2,616   |
| Test  | 195           | 2,793   |
| **Total** | **1,300** | **17,642** |

- `codes_zcj` (Blenderbot-Joint)와 동일한 대화 분할 사용 (seed=13)
- `(emotion_label, situation_text)` 기준 대화 그룹핑 + ESConv.json 대조 검증
- 3개 collision pair (동일 key를 공유하는 서로 다른 대화)는 dialog content로 disambiguation
- Train/dev/test 간 대화 중복 없음 검증 완료

## 3. 학습 설정

두 실험 모두 동일한 하이퍼파라미터를 사용한다.

| 항목 | 값 |
|------|---|
| 백본 모델 | `facebook/blenderbot_small-90M` (로컬) |
| Optimizer | AdamW (lr=2e-5) |
| Warmup steps | 120 |
| Batch size | 20 |
| Epochs | 8 |
| GPU | NVIDIA A100 |
| Best checkpoint 선정 | Dev set `eval_strategy_predict_accuracy` 기준 |

| 실험 | Best dev ACC | Best dev PPL |
|------|-------------|-------------|
| 8:1:1 재현 | 35.22% | 15.77 |
| Leak-free 70:15:15 | 32.22% | 16.13 |

## 4. 결과 비교

### 4.1 메인 결과

| Model | ACC | PPL | D-1 | D-2 | B-1 | B-2 | B-4 | R-L |
|-------|-----|-----|-----|-----|-----|-----|-----|-----|
| TransESC (논문, 8:1:1) | **34.71** | **15.85** | 4.73 | 20.48 | **17.92** | **7.64** | **2.43** | **17.51** |
| TransESC (재현, 8:1:1) | 34.26 | 15.85 | **4.72** | **20.79** | 16.92 | 7.14 | 2.25 | 17.45 |
| TransESC (leak-free 70:15:15) | 31.15 | 16.92 | 3.34 | 15.52 | 16.27 | 6.51 | 1.89 | 15.94 |

### 4.2 전체 메트릭

| Metric | 논문 (8:1:1) | 재현 (8:1:1) | Leak-free (70:15:15) |
|--------|-------------|-------------|---------------------|
| Strategy Accuracy (ACC) | 34.71% | 34.26% | 31.15% |
| Perplexity (PPL) | 15.85 | 15.85 | 16.92 |
| Distinct-1 | 4.73 | 4.72 | 3.34 |
| Distinct-2 | 20.48 | 20.79 | 15.52 |
| Distinct-3 | — | 37.25 | 29.73 |
| BLEU-1 | 17.92 | 16.92 | 16.27 |
| BLEU-2 | 7.64 | 7.14 | 6.51 |
| BLEU-3 | — | 3.74 | 3.27 |
| BLEU-4 | 2.43 | 2.25 | 1.89 |
| ROUGE-L | 17.51 | 17.45 | 15.94 |
| F1 | — | 20.62 | 18.83 |
| Avg. Length | — | 16.47 | 16.00 |

### 4.3 재현성 검증 (논문 vs 재현, 8:1:1)

| Metric | 논문 | 재현 | 차이 |
|--------|------|------|------|
| ACC    | 34.71 | 34.26 | -0.45 |
| PPL    | 15.85 | 15.85 | 0.00 |
| D-1    | 4.73  | 4.72  | -0.01 |
| D-2    | 20.48 | 20.79 | +0.31 |
| B-1    | 17.92 | 16.92 | -1.00 |
| B-2    | 7.64  | 7.14  | -0.50 |
| B-4    | 2.43  | 2.25  | -0.18 |
| R-L    | 17.51 | 17.45 | -0.06 |

PPL, Distinct, ROUGE-L은 거의 동일하게 재현됨. BLEU와 ACC의 소폭 차이는 top-p/top-k sampling의 확률적 특성과 환경 차이(GPU, 라이브러리 버전)에 기인.

### 4.4 Data Leakage 영향 (재현 8:1:1 vs Leak-free)

| Metric | 재현 (8:1:1, leaky) | Leak-free (70:15:15) | 차이 | 변화율 |
|--------|-------------------|---------------------|------|--------|
| ACC    | 34.26 | 31.15 | -3.11 | -9.1% |
| PPL    | 15.85 | 16.92 | +1.07 | +6.8% |
| D-1    | 4.72  | 3.34  | -1.38 | -29.2% |
| D-2    | 20.79 | 15.52 | -5.27 | -25.3% |
| B-1    | 16.92 | 16.27 | -0.65 | -3.8% |
| B-2    | 7.14  | 6.51  | -0.63 | -8.8% |
| B-4    | 2.25  | 1.89  | -0.36 | -16.0% |
| R-L    | 17.45 | 15.94 | -1.51 | -8.7% |

## 5. 분석

### 5.1 재현성

8:1:1 재현 실험은 논문 결과를 충실히 재현한다. PPL은 15.85로 완벽히 일치하고, Distinct-1/2, ROUGE-L도 거의 동일하다. BLEU의 소폭 차이(-0.18~1.00)는 top-p/top-k sampling의 확률적 특성에 기인하며, 재현 환경(GPU, 라이브러리 버전)에 따른 정상적인 범위이다.

### 5.2 Data Leakage의 영향

**동일 코드·동일 하이퍼파라미터**에서 데이터 분할만 변경했을 때, 모든 메트릭에서 leak-free 결과가 하락한다. 이는 8:1:1 split에서 같은 대화의 sliding window 샘플이 train/test에 동시 존재하여, 모델이 test 시 이미 학습한 대화 맥락을 활용했음을 직접적으로 보여준다.

- **Distinct-1/2**: -29.2%/-25.3% 하락 (가장 큰 변화). Leaky 환경에서는 test 대화의 어휘/표현을 이미 학습하여 다양한 표현이 가능했으나, leak-free에서는 unseen 대화에 대해 더 보수적인 생성을 함.
- **Strategy Accuracy**: -9.1% 하락. 같은 대화의 이전 턴에서 전략 패턴을 학습한 효과가 사라짐.
- **BLEU-4**: -16.0% 하락. Reference와의 고차 n-gram overlap 감소.
- **ROUGE-L**: -8.7% 하락. 최장 공통 부분 시퀀스 일치도 감소.
- **Perplexity**: +6.8% 상승. Unseen 대화에 대한 예측 불확실성 증가.

### 5.3 비교 기준선 (Blenderbot-Joint, 논문 보고)

| Model | ACC | PPL | D-1 | D-2 | B-1 | B-2 | B-4 | R-L |
|-------|-----|-----|-----|-----|-----|-----|-----|-----|
| BlenderBot-Joint (논문) | 17.69 | 17.39 | 2.96 | 17.87 | 18.78 | 7.02 | 1.63 | 14.92 |
| TransESC (재현, 8:1:1)  | 34.26 | 15.85 | 4.72 | 20.79 | 16.92 | 7.14 | 2.25 | 17.45 |
| TransESC (leak-free)    | 31.15 | 16.92 | 3.34 | 15.52 | 16.27 | 6.51 | 1.89 | 15.94 |

Leak-free TransESC는 여전히 BlenderBot-Joint 대비 ACC(+13.46), PPL(-0.47), B-4(+0.26), R-L(+1.02)에서 우위를 보인다. 다만 B-1(-2.51), B-2(-0.51), D-2(-2.35)에서는 열세이다.

**참고**: BlenderBot-Joint의 논문 결과도 동일한 leaky split 기반일 가능성이 있어, 공정한 비교를 위해서는 BlenderBot-Joint 역시 leak-free split으로 재평가가 필요하다.

### 5.4 요약

- **재현성 확인**: 8:1:1 split으로 논문 결과를 충실히 재현함 (PPL 완벽 일치, 기타 메트릭 소폭 차이).
- **Leakage 영향 확인**: 동일 조건에서 데이터 분할만 변경했을 때 모든 메트릭이 하락하며, 특히 Distinct (-25~29%)에서 영향이 가장 큼.
- **모델 가치 유효**: Leak-free 환경에서도 BlenderBot-Joint 대비 ACC, PPL, B-4, R-L에서 우위를 보여, turn-level state transition의 기여는 유효함.
- **공정 비교 필요**: 논문에서 보고된 8:1:1 기반 수치는 data leakage로 인해 과대평가되었으므로, 다른 모델과의 공정한 비교를 위해 leak-free split 기준 재평가가 필요함.

## 6. 재현 환경

| 항목 | 경로 |
|------|------|
| 대화 단위 분할 스크립트 | `scripts/conversational_split.py` |
| 8:1:1 학습 로그 | `logs/train_log2.txt` |
| 8:1:1 추론 로그 | `logs/eval_log.txt` |
| Leak-free 학습 로그 | `logs/train_leakfree_70_15_15.log` |
| Leak-free 추론 로그 | `logs/infer_leakfree_70_15_15.log` |
| 생성 결과 | `generated_data/` |
| Conda 환경 | `environment.yml` |
| 원본 데이터 백업 | `leaky_data/8_1_1/` |
