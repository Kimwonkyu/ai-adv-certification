📘 [학습 노트] 교재 3. LLM 기초 및 Fine-Tuning
1. LLM의 정의와 핵심 구조 (Transformer)
• LLM의 정의: 언어의 패턴을 학습하여 주어진 입력 다음에 올 "다음 토큰(Next Token)"을 예측하며 문장을 생성하는 모델입니다.
• 핵심 기술:
    ◦ 임베딩(Embedding): 단어를 고차원 벡터(숫자)로 변환하여 의미를 표현합니다. 과거 Word2Vec의 한계(동음이의어 처리 불가 등)를 극복하고 문맥(Context)을 학습합니다.
    ◦ 토큰(Token): 모델이 데이터를 처리하는 기본 단위입니다. 토큰 수가 많을수록 처리 가능한 컨텍스트(맥락) 길이가 늘어납니다.
• Transformer 구조:
    ◦ Encoder: 입력 문장의 의미와 맥락을 병렬로 계산합니다 (예: BERT).
    ◦ Decoder: 지금까지 출력된 토큰을 바탕으로 다음 토큰을 순차적으로 생성합니다 (예: GPT).
    ◦ Attention 메커니즘: 문장 내의 단어들이 서로 어떤 관계가 있는지(어디에 집중해야 하는지) 계산하여 맥락을 파악합니다.
2. 모델의 발전과 종류
• BERT vs GPT:
    ◦ BERT: Encoder 기반의 양방향 모델로, 빈칸 채우기나 문장 의미 파악에 강합니다.
    ◦ GPT: Decoder 기반의 일방향 모델로, 텍스트 생성에 특화되어 있으며 현재 대부분 LLM의 표준이 되었습니다.
• 상용 vs 오픈 모델:
    ◦ 상용(Closed): GPT(OpenAI), Claude(Anthropic) 등 성능이 최상위이나 가중치가 비공개입니다.
    ◦ 오픈(Open): Llama(Meta), Qwen(Alibaba), Mistral, Gemma(Google) 등 가중치가 공개되어 로컬 실행 및 파인튜닝이 가능합니다.
• Reasoning 모델 (DeepSeek R1 등): 단순 답변 생성이 아니라, Thinking(사고) 과정을 거쳐 수학이나 코딩 등 복잡한 문제를 풀도록 설계된 최신 모델입니다.
3. Fine-Tuning (파인튜닝) 개요
• 개념: Pre-training(사전 학습)된 Base 모델에 특정한 지식, 말투, 형식, 안전성 등을 주입하여 목적에 맞게 최적화하는 과정입니다.
• 모델 유형:
    ◦ Base Model: 사전 학습만 된 상태. 다음 단어 예측은 잘하지만 질문에 대한 답변 능력은 약합니다.
    ◦ Instruct Model: QA(질문-답변) 쌍으로 학습(SFT)되어 챗봇처럼 지시를 따르는 모델입니다.
• 주요 리스크:
    ◦ Catastrophic Forgetting (파멸적 망각): 새로운 지식을 배우면서 기존의 언어 능력이나 지식을 잊어버리는 현상입니다,.
    ◦ Overfitting (과적합): 특정 데이터(예: 의료 상담)에만 너무 익숙해져 일반적인 대화 능력이 떨어지는 현상입니다.
4. 파인튜닝 핵심 기법 (SFT & RL)
• SFT (Supervised Fine-Tuning):
    ◦ 사람이 만든 정답(질문+답변) 데이터셋을 이용해 지도 학습을 수행합니다. 지시사항 이행(Instruction Tuning) 능력을 키우는 데 필수적입니다,.
• RL (Reinforcement Learning, 강화학습):
    ◦ RLHF: 보상 모델(Reward Model)을 만들어 인간의 선호도에 맞는 답변을 하도록 PPO 알고리즘 등으로 최적화합니다.
    ◦ DPO (Direct Preference Optimization): 별도의 보상 모델 없이 "좋은 답변 vs 나쁜 답변" 데이터 쌍만으로 직접 최적화하는 효율적인 방식입니다.
    ◦ GRPO: DeepSeek 등에서 사용하는 방식으로, 그룹 내 답변들의 상대적 평가를 통해 최적화합니다.
• 결론: SFT로 기본적인 형식과 지식을 잡고, RL로 추론 경로와 효율을 보정하는 "SFT + RL" 결합이 성능 향상의 핵심입니다.
5. 효율적인 학습 (PEFT & Quantization)
• PEFT (Parameter Efficient Fine-Tuning):
    ◦ 모델의 전체 파라미터를 수정하는 대신, **어댑터(Adapter)**라고 불리는 소량의 파라미터만 추가하여 학습합니다.
    ◦ LoRA (Low-Rank Adaptation): 가장 대표적인 PEFT 기법으로, 메모리 사용량을 크게 줄이면서도 전체 파인튜닝에 준하는 성능을 냅니다.
• 양자화 (Quantization):
    ◦ 모델의 가중치를 표현하는 비트(bit) 수를 줄여(예: 16bit → 4bit) 모델 크기와 VRAM 소모를 줄이는 기술입니다.
    ◦ **QAT(Quantization Aware Training)**나 GGUF 포맷 등을 통해 로컬 장비에서도 고성능 모델을 돌릴 수 있게 합니다,.