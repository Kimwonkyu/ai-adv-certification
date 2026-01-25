📘 [학습 노트] 교재 6. LLM Fine Tuning (파인튜닝)
1. Fine-tuning 개요 및 목적
• 정의: 이미 학습된(Pre-trained) 모델에 새로운 데이터를 추가 학습시켜 성능을 향상시키는 방법입니다. 프롬프트 엔지니어링이 모델 외부에서 접근하는 것이라면, 파인튜닝은 모델 자체의 파라미터(가중치)를 조정하거나 사고 회로를 변경하는 것입니다.
• Training vs Fine-tuning 비교:
    ◦ Training: 백지상태에서 언어 패턴 전반을 학습(Base Model), 비용과 시간이 매우 많이 듭니다,.
    ◦ Fine-tuning: 이미 만들어진 모델에 목적을 갖춘 추가 학습을 수행하며, 적은 데이터로 빠르게 수렴합니다.
• 수행 목적 4가지 (핵심),:
    1. 지식 (Knowledge): 최신 정보나 특정 도메인(의료, 법률 등) 지식을 주입.
    2. 능력 (Ability): 요약, 번역, 논리 추론(Reasoning) 등 특정 태스크 성능 향상.
    3. 형식 (Format): JSON 출력, 특정 말투, 업무 포맷 준수 등 출력 스타일 통일.
    4. 안전 (Safety): 유해한 요청 거절, 윤리적 응답 유도 (Alignment).
2. 모델의 유형과 학습 파이프라인
• 모델 유형:
    ◦ Base Model: 다음 토큰 예측만 잘하며, 질의응답 능력은 부족합니다 (예: Qwen2.5-7B),.
    ◦ Instruct Model: Base 모델에 질의응답(QA) 템플릿을 학습(SFT)시켜 챗봇처럼 동작하게 만든 모델입니다,.
• Fine-tuning 파이프라인 5단계,:
    1. CPT (Continuous Pre-Training): 새로운 코퍼스로 도메인 지식 주입 (언어의 '토양' 바꾸기).
    2. SFT (Instruction Tuning): 지시사항 이행 및 질의응답 패턴 학습.
    3. Rejection Sampling + SFT: 모델이 생성한 여러 답변 중 우수한 것만 선별해 재학습.
    4. DPO/RL (선호 최적화): 인간이 선호하는 답변을 하도록 보상 기반 학습.
    5. Alignment: 안전성 및 윤리적 기준 학습.
3. 핵심 학습 기법: SFT, RL, CPT
• SFT (Supervised Fine-Tuning),:
    ◦ "질문+답변" 정답 데이터셋을 통해 지도 학습을 수행합니다.
    ◦ Instruction Tuning (IT): 다양한 지시사항 템플릿(Alpaca, Llama3 포맷 등)을 학습하여 지시 이행 능력을 강화합니다,.
• RL (Reinforcement Learning, 강화학습),,:
    ◦ SFT로 만든 모델이 더 좋은 답변을 하도록 보상(Reward)을 주어 최적화합니다.
    ◦ RLHF: 보상 모델(Reward Model)을 따로 만들어 PPO 알고리즘 등으로 학습. (단점: 비용 높음, 아첨 모델 가능성),.
    ◦ DPO (Direct Preference Optimization): 별도의 보상 모델 없이 "좋은 답 vs 나쁜 답" 데이터 쌍만으로 직접 최적화하는 효율적 방식입니다,.
    ◦ GRPO: DeepSeek 등에서 사용하는 방식으로, 그룹 내 답변들의 상대적 평가를 통해 최적화합니다 (비용 절감),.
• CPT (Continuous Pre-Training),:
    ◦ Base 모델에 전문 서적, 논문 등 대량의 텍스트(Corpus)를 추가로 읽혀 도메인 이해도를 근본적으로 높이는 과정입니다.
4. 최신 트렌드: Reasoning & RAFT
• Reasoning (추론) 모델 학습,:
    ◦ DeepSeek R1 사례: <think> 태그 내에서 추론 과정을 스스로 생성하고 수정(Self-Correction)하도록 학습합니다.
    ◦ RLVR: 수학/코딩처럼 정답 검증이 가능한 문제에 대해 강화학습을 수행하여 추론 능력을 극대화합니다,.
• RAFT (Retrieval-Augmented Fine-Tuning),:
    ◦ RAG(검색 증강 생성) 시스템에 최적화된 모델을 만드는 기법입니다.
    ◦ "관련 문서(Positive)와 무관한 문서(Negative)가 섞여 있어도 정답을 찾아내는 능력"을 학습시켜 환각(Hallucination)을 줄입니다.
5. 효율적 학습: PEFT & LoRA
• PEFT (Parameter Efficient Fine-Tuning),:
    ◦ 모델 전체를 재학습(Full Fine-tuning)하는 대신, 파라미터의 극히 일부만 학습하여 비용과 VRAM 소모를 줄이는 기술입니다.
• LoRA (Low-Rank Adaptation),:
    ◦ 가장 대표적인 PEFT 기법으로, 기존 가중치(W)는 고정하고 별도의 작은 행렬(A, B)만 학습시킵니다.
    ◦ 장점: 파라미터를 1% 이하로 줄여 메모리 절약, 학습 후 원본 모델 훼손 없이 어댑터(Adapter)만 교체 가능 (Multi-LoRA).
    ◦ 단점: 전체 파인튜닝에 비해 복잡한 문제 해결 능력이나 지식 주입 효과는 다소 떨어질 수 있습니다 ("Learns less & forgets less").
6. 주요 리스크 관리
• Catastrophic Forgetting (파멸적 망각),,:
    ◦ 새로운 지식을 배우면서 기존에 알던 지식(예: 영어 능력, 일반 상식)을 잊어버리는 현상입니다. LoRA를 사용하거나 다양한 데이터를 섞어서 학습하여 방지합니다.
• Overfitting (과적합),,:
    ◦ 특정 데이터(예: 특정 말투)에 너무 익숙해져서 일반적인 질문에 이상하게 답하는 현상입니다.