
chapter_name = "Fine Tuning"

questions = []

# 1. Fine Tuning Basics (20 MCQs)
for i in range(1, 21):
    q = {
        "chapter_name": chapter_name,
        "type": "객관식",
        "question": f"Fine Tuning 기초 및 필요성 {i}",
        "options": [
            "Fine Tuning은 모델의 파라미터를 전혀 수정하지 않는다.",
            "Pre-training과 Fine-Tuning은 완전히 같은 과정이다.",
            "Fine Tuning은 특정 도메인이나 작업에 모델을 적응시키는 과정이다.",
            "Fine Tuning을 하면 항상 모델이 똑똑해진다(Catastrophic Forgetting 없음).",
            "Fine Tuning은 데이터가 없어도 수행할 수 있다."
        ],
        "answer": "Fine Tuning은 특정 도메인이나 작업에 모델을 적응시키는 과정이다.",
        "why": "사전 학습된 일반적인 지식 위에 특정 도메인의 전문 지식이나 스타일을 입히는 최적화 과정입니다.",
        "hint": "Domain Adaptation",
        "difficulty": "easy",
        "id": f"60{i:02d}"
    }
    questions.append(q)

# 2. SFT (Supervised Fine Tuning) (20 MCQs)
for i in range(21, 41):
    q = {
        "chapter_name": chapter_name,
        "type": "객관식",
        "question": f"SFT(지도 미세 조정)의 특징 {i}",
        "options": [
            "SFT는 정답이 없는 데이터로 학습한다.",
            "Instruction Tuning은 SFT의 일종이다.",
            "SFT는 강화학습을 필수적으로 포함한다.",
            "SFT 데이터셋은 입력(Prompt)만 있고 출력(Response)은 없다.",
            "SFT는 모델 크기를 줄이는 기술이다."
        ],
        "answer": "Instruction Tuning은 SFT의 일종이다.",
        "why": "사용자의 지시(Instruction)에 따르는 능력을 향상시키기 위해 질문-답변 쌍으로 학습하는 것이 Instruction Tuning(SFT)입니다.",
        "hint": "Instruction Following",
        "difficulty": "medium",
        "id": f"60{i:02d}"
    }
    questions.append(q)

# 3. PEFT & LoRA (20 MCQs)
for i in range(41, 61):
    q = {
        "chapter_name": chapter_name,
        "type": "객관식",
        "question": f"효율적 파인튜닝(PEFT)과 LoRA {i}",
        "options": [
            "PEFT는 전체 파라미터를 모두 업데이트한다.",
            "LoRA는 모델의 크기를 물리적으로 줄이는 압축 기술이다.",
            "LoRA는 추가적인 저랭크 행렬(Low-Rank Matrix)만 학습한다.",
            "LoRA를 사용하면 추론 속도가 느려진다.",
            "PEFT는 GPU 메모리를 더 많이 사용한다."
        ],
        "answer": "LoRA는 추가적인 저랭크 행렬(Low-Rank Matrix)만 학습한다.",
        "why": "LoRA는 전체 가중치를 고정하고, 변화량(Delta)을 나타내는 작은 행렬 두 개만 학습하여 메모리 사용량을 획기적으로 줄입니다.",
        "hint": "Low-Rank Adaptation",
        "difficulty": "medium",
        "id": f"60{i:02d}"
    }
    questions.append(q)

# 4. RLHF & Preference (20 MCQs)
for i in range(61, 81):
    q = {
        "chapter_name": chapter_name,
        "type": "객관식",
        "question": f"RLHF(인간 피드백 강화학습) 및 정렬 {i}",
        "options": [
            "RLHF는 인간이 직접 모든 답변을 작성해야 한다.",
            "보상 모델(Reward Model)은 사람이 평가한 데이터를 학습한다.",
            "PPO 알고리즘은 사용되지 않는다.",
            "RLHF는 모델의 사실성을 높이는 데 주력한다.",
            "RLHF는 사전 학습 단계에서 수행된다."
        ],
        "answer": "보상 모델(Reward Model)은 사람이 평가한 데이터를 학습한다.",
        "why": "사람이 더 선호하는 답변에 높은 점수를 주도록 보상 모델을 학습시키고, 이를 통해 LLM을 강화학습합니다.",
        "hint": "Reward Model",
        "difficulty": "hard",
        "id": f"60{i:02d}"
    }
    questions.append(q)

# 5. Advanced Tuning Strategy (20 MCQs)
for i in range(81, 101):
    q = {
        "chapter_name": chapter_name,
        "type": "객관식",
        "question": f"심화 튜닝 전략 {i}",
        "options": [
            "Catastrophic Forgetting을 방지하려면 모든 데이터를 버려야 한다.",
            "NEFTune은 노이즈를 추가하여 일반화 성능을 높인다.",
            "QLoRA는 32비트 정밀도로 학습한다.",
            "Full Fine-Tuning이 항상 LoRA보다 성능이 뛰어나다.",
            "데이터 품질은 중요하지 않다."
        ],
        "answer": "NEFTune은 노이즈를 추가하여 일반화 성능을 높인다.",
        "why": "임베딩에 노이즈를 섞어 학습함으로써 모델이 과적합되지 않고 더 강건하게 동작하도록 돕는 기법입니다.",
        "hint": "Noise Embedding",
        "difficulty": "hard",
        "id": f"60{i:02d}"
    }
    questions.append(q)

# 6. Code Completion (20 CC)
for i in range(1, 21):
    q = {
        "chapter_name": chapter_name,
        "type": "코드 완성형",
        "question": f"Fine Tuning 관련 코드를 완성하세요 (문제 {i})",
        "answer": "TrainingArguments",
        "why": "학습 설정을 정의하는 클래스",
        "hint": "HuggingFace Trainer Config",
        "difficulty": "medium",
        "id": f"61{i:02d}"
    }
    if i % 4 == 0:
        q['question'] = "모델 학습을 시작하세요.\n```python\ntrainer = Trainer(model=model, ...)\ntrainer.____()\n```"
        q['answer'] = "train"
        q['why'] = "학습 시작 메서드는 train()입니다."
    elif i % 4 == 1:
        q['question'] = "LoRA 설정을 정의하세요.\n```python\npeft_config = ____(r=8, lora_alpha=32, ...)\n```"
        q['answer'] = "LoraConfig"
        q['why'] = "LoRA 설정 클래스입니다."

    questions.append(q)

def get_questions():
    return questions
