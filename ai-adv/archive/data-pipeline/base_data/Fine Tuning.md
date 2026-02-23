***

## 1. Training vs Fine-tuning

- LLM Training
  - 처음부터 초대량 코스 학습, 언어 패턴 전반 이해 → Base/Pretrained 모델.
  - 돈・시간 많이 듦, 지식・언어의 "토양"을 만드는 단계.

- Fine-tuning 수행 목적 4가지
  - 지식: 최신/도메인 지식 주입.
  - 능력: 요약・번역・수학풀이 등 특정 태스크 성능 향상.
  - 형식: 말투・길이・업무 포맷 통일.
  - 안전: 유해 요청 거절, 유익한 응답 유도.

- Fine-tuning 주의할 점
  - Catastrophic Forgetting (추가 학습을 진행하자 모델의 기존 능력이 감소하는 현현)
  - Overfitting(과적함 - Ex. 의료 상담 데이터로 과적합된 모델)

- Fine-tuning 파이프라인의 5가지 Step (준비물 : Base/Instruct Model, Fine-tuning 데이터)
  - Continuous PreTraining으로 도메인 지식 주입하기
  - Instruction Tuning으로 지시사항/질의응답 학습하기
  - Rejection Sampling + SFT으로 출력 성능 개선하기
  - DPO등(PRO, GRPO, ...)으로 선호 패턴 학습하기
  - Alignment 데이터로 모델 안정성 높이기

***

## 2. Base / Instruct / Quantized.

- Base Model
  - Pretraining만 한 모델, 다음 토큰 예측만 잘함.
  - 질의응답/지시 이해는 약함, 위키・뉴스 등 웹 코퍼스로 학습.

- Instruct Model
  - QA/지시사항 템플릿(시스템・유저・어시스턴트 포맷)으로 SFT한 모델.
  - "질문-답변" 형식을 이해해서 챗봇처럼 동작.

- Quantized Model
  - Instruct 모델을 양자화해 크기・메모리 감소.
  - 예: Qwen2.5-7B-Instruct-GPTQ-Int4/Int8.

***

## 3. Fine-tuning 종류: CPT vs SFT vs RL

- CPT(Continuous Pre-Training)
  - Pretraining과 같은 방식으로 "새 코퍼스"로 추가 학습.
  - 언어/도메인 능력 크게 바꾸지만 데이터 많이 필요, 망각 위험 큼.
  - Instruct 모델에는 잘 안 씀(Instruction 능력 손상 연구).

- SFT(Supervised Fine-tuning)
  - 정답(y)을 다음 토큰 예측 형태로 지도학습.
  - Instruction Tuning, RAFT, Reasoning SFT 등이 여기에 포함.
  - 일반화된 지식 : Corpus 필요 (지식 주입은 Pretrained 모델에서 수행 가능)
  - 질의응답 : QA Data 필요 (템플릿 필요, 지식 주입 능력은 제한적이나 시도해볼 수 있음)
  - 도메인 지식을 넣은 채팅 모델 생성(Base Model → Continuous Pre-Training → Instruction Tuning / 가장 복잡하지만 이론적으로 확실)
  - 최근 Reasoning 모델 : Reasoning IT만으로 해결할 수 있음

- RL(Reinforcement Learing - 강화학습) 기반 파인튜닝
  - RLHF, DPO, GRPO, RLVR 등 보상 기반 최적화.
  - SFT로 QA 능력 만든 뒤, 보상 함수로 "선호・안전・추론 품질" 조정.
  - 지식 일반화를 위해 필요 (SFT는 질문과 답변이 정해져 있어서 일반화가 어려움)
  - 확률 분포 기반 예측에 보상(Reward) 함수를 적용하여 학습(SFT는 레이블 기반으로 정답을 100% 암기하도록 학습)

***

## 4. Instruction Tuning(IT) 핵심

- 정의
  - "질문+답변"(+시스템)"을 일정 템플릿으로 묶어 다음 토큰 예측으로 학습.
  - Instruction 패턴・지시 이행 능력 학습이 목적(지식 자체는 주로 Pretrain).

- 대표 포맷
  - Llama3 템플릿, Alpace 포맷(Instruction / Question / Answer).
  - Reasoning data: 질문 + Thinking(과정) + Answer 학습(Open Thought, LIMO 등).

- 역할 변화
  - 2023 LIMA: 지식은 주로 CPT, Instruction은 QA 패턴 학습.
  - 2024: 범위 좁으면 IT만으로도 지식 향상 가능(하지만 일반화 한계).
  - 2025: Reasoning 과정 포함 SFT로 지식+추론 향상 시도.

- Instruct Model + SFT 활용 예시
  - 오류 탐지 강화 (Ex. Review Data 입력 시 스팸 여부 판별)
  - 특정 Task 성능 강화
  - Domain 특화 작업 성능 향상
  - 출력 형식 제어
  - 행동 스타일 조정
  - 복잡한 프롬프트 엔지니어링 필요성 감소 (긴 프롬프트 축소 - Ex. RAFT)

***

## 5. RAFT & RAG 관련 포인트

- RAFT(Retrieval-Augmented Fine-Tuning : 오픈 북 시험 공부) 개념
  - RAG 포맷(Q, Positive/Negative 문서, 근거・인용 포함 CoT)을 SFT로 학습.
  - "문서를 기반으로 푸는 방식・형식"을 모델에 각인 → 환각 감소.

- 특징
  - Positive + Negative 섞어도 올바른 문서 고르도록 튜닝.
  - CoT-RAG 패턴(질문 → 판단 → 근거 제시 → 문서 인용 → 해석 병행 → 최종 결론)을 데이터셋으로 생성하여 Fine-Tuning 진행, 해당 고정 형식으로 학습.

***

## 6. RL: RLHF, DPO, GRPO, RLVR

- 보상 모델 & RLHF(Reinforcement Learning with Human Feedback - 아첨모델 탄생)
  - (프롬프트+응답)→Score 분류기처럼 학습, 좋은/나쁜 답 상대 비교로 학슴.
  - PPO(Proximal Policy Optimization - 보상 모델에 의한 최적화)로 "보상↑ + 원본모델과 KL 거리 제한" → 보상 해킹・붕괴 방지.
  - 장점: 친절・안전・정렬된 응답, 단점: 아첨(sycophantic)・3LM 오버헤드.

- DPO(Direct Preference Optimization) : RL의 훌륭한 대체
  - 별도 보상 모델 없이 "좋은/나쁜 답 쌍"으로 직접 최적화.
  - 구현 단순, 데이터만 있으면 됨.

- 기각 샘플링(Rejection Sampling) + SFT
  - 좋은 데이터만 선별하여 추가 학습하는 방법
  - 보상 모델이나 기준으로 통과하지 못하면 기각하는 방식
  - DPO의 확장 방식으로 모델 출력을 올려 품질을 향상
  - 파인 튜닝 파이프라인에서 성능을 높이는 기법

- RLVR(Reinforcement Learing with Verifiable Reward)
  - 정확한 검증이 간능한 문제 기반 강화학습 (Ex. 코딩 테스트 케이스 실행)
  - 보상함수 단수화를 통한 강화학습, DeepSeek-R1, Tulu3 등에서 핵심 개념.

- GRPO(Group Relative Policy Optimization)
  - 상대 평가를 통한 선호 최적화 (DeepSeek-Math에서 처음 제안)
  - 보상함수 사용 (PPO : 보상함수에 따라 학습 / DPO : 보상함수 없이 데이터로 학습)
  - 한 질문에 LLM이 여러 답 생성 → 답변을 보상함수에 통과시켜 Reward 계산 → Reward들을 평균편차로 나눠서 정규화 → 그룹 내에서 높은 Reward를 받는 출력을 자주 생성하도록 학습   
  - 점수 낮아도 상대적으로 좋으면 반영, 점수가 높아도 표준화로 조정 가능

***

## 7. DeepSeek R1 계열 흐름

- R1-Zero
  - DeepSeek-V3 Base + Rule-based Reward로 GRPO 학습, SFT 없이 시작.
  - <think>추론</think>, <answer>답</answer> 형식, 학습 중 Self-reflection/Aha moment 등장.

- 문제점 & 해결
  - R1-Zero 가동성 낮고 코드・언어 혼재.
  - 해결 방법 : 일반적인 Instruction 학습 방식에 GRPO를 결합한 파이프라인을 다시 학습
  - 4단계 파이프라인 : Cold-start SFT(R1-Zero를 이용해 답변 데이터 세트 생성) → GRPO → Rejection Sampling + SFT → All-Scenario RL

- Distill
  - 3단계 학습인 Rejection Sampling에 사용했던 800k개의 데이터를 이용(600K의 Resoning Dataset + DeepSeek-V3-Instruct 학습 때 사용했던 200k의 Non-Reasoning 데이터 추가)
  - SFT만으로도 추론 성능을 끌어올림 (추론 모델은 SFT와 RL의 합작)
  - 결론: "강력한 Base + SFT + RL" 조합이 중요. (베이스 모델 성능 중요 + SFT와 RL을 함께)
  
***

## 8. RL vs SFT 논점

- RL의 장점
  - 불필요한 반복 줄이고, 더 짧고 효율적인 추론 경로 선택.
  - 중간 지식 오류를 줄이고, 더 정확한 지식・추론 경로로 유도.

- RL 비판("RL 뮤용론" 논문)
  - RL이 Coverage(다양한 풀이 시도)를 줄이고, 비슷한 풀이만 반복.
  - pass@k에서 오히려 성능↓, 창의성・탐색 범위 감소.

- 종합 결론
  - 외부 지식 학습 없이 RL만으로는 근본적 항상 한계.
  - SFT로 지식・형식, RL로 추론 경로・효율 보정 → 둘의 결합이 핵심.

***

## 9. 데이터 구축 & 합성 데이터 

- 수집
  - 도메인 문서 크롤링, 기존 코퍼스 활용.
  - "풀고 싶은 문제 분포"를 충분히 표현해야 함.

- 생성/증강(가공)
  - 고성능 LLM으로 QA・Reasoning 데이터 생성(지식 종류).
  - 함성 데이터 + 실제 데이터 결합이 중요, 모델별 이용 규정 확인 필요.

- Instruction vs Corpus
  - 다양한 Instruction 데이터 확보는 어렵지만, 코퍼스로 QA 세트 생성 가능.
  - "어떤 주제의 이해를 넣으려면 코퍼스 기반 CPT/QA 생성이 더 적합".

***

## 10. PEFT & LoRA 핵심 암기

- PEFT(Parameter Efficient Fine Tuning) 정의
  - Fine Tuning의 병목(Bottleneck) 해결 가능 (Fine Tuning 후 특정 지식만 Unlearning 가능)
  - 원래 파라미터를 거의 그대로 두고, 소량 추가 파라미터(어댑터)만 학습.
  - 메모리・GPU 적게, Unlearning 쉬움, 기존 모델 보존.
  - 어댑터(Adapter) : 모델 성능의 방향을 조정(튜닝), 어댑터의 가중치만 학습시키는 방향으로 Fine Tuning 진행

- LoRA(Low-Rank Adaptation Fine Tuning) 메커니즘(가장 대표적인 PEFT 알고리즘)
  - Pretrained Weights는 그대로 두고, 새로운 A,B를 적용하여 결과 변경

- LoRA 장점
  - 파라미터 <1%만 변경, 메모리・연산량 크게 절약. (매우 적은 파라미터로 파인 튜닝 가능)
  - 원본 파라미터 불변 → 탈락 가능, Multi-LoRA(수학/번역/법률) 조합 가능(하나의 모델에 여러 개의 LoRA)
  - Full Fine Tuning보다 Catastrophic forgetting, 망각이 적게 발생.

- LoRA 단점
  - "LoRA learns less & forgets less": 목표 도메인 학습량은 Full Fine Tuning 보다 적을 수 있음.
  - CPT(토양 자체 변경)나 매우 어려운 문제에서는 성능 열세.
  - 논문: Full Fine Tuning와 동급이라는 인식은 착시, 복잡 문제에서 일반화 약화.

***

## 11. Catastrophic Forgetting & 부작용

- Catastrophic Forgetting
  - 추가 학습 도메인 성능↑, 기존 도메인 성능↓ Trade-off.
  - 예: 코딩 학습 후 언어 능력↓, 한국어 학습 후 영어 능력↓.

- Overfitting
  - LLM에도 여전히 Train/Test, Generalization 개념 그대로 적용.
  - 특정 데이터(예 의료 상담)에 과도 적합 시, 다른 도메인・문제에서 비상식적 답변.

*** 
