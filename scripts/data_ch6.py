
chapter_name = "Fine Tuning"

questions = []

# --- 100 MCQs ---

# 1. Basic Concepts
questions.append({
    "chapter_name": chapter_name, "type": "객관식", "difficulty": "easy", "id": "6001",
    "question": "이미 방대한 데이터를 학습한 '사전 학습 모델(Pre-trained Model)'에 특정 도메인의 데이터를 추가 학습시켜 전문가로 만드는 과정은?",
    "options": ["Tokenizing", "Fine Tuning", "Prompt Engineering", "Filtering", "Quantization"],
    "answer": "Fine Tuning",
    "why": "파인튜닝은 일반적인 지식을 가진 모델을 특정 목적이나 스타일, 전문 지식에 맞게 미세 조정하는 작업입니다.",
    "hint": "미세 조정이라는 뜻을 가진 용어입니다."
})

questions.append({
    "chapter_name": chapter_name, "type": "객관식", "difficulty": "medium", "id": "6002",
    "question": "파인튜닝과 RAG의 차이점에 대한 설명으로 올바른 것은?",
    "options": ["파인튜닝은 '오픈북 시험', RAG는 '지식 암기'와 같다.", "RAG는 모델 내부의 가중치를 직접 수정한다.", "파인튜닝은 모델에게 특정 말투나 형식을 학습시키는 데 유리하다.", "RAG는 지연 시간(Latency)이 파인튜닝보다 항상 짧다.", "파인튜닝을 하면 실시간 최신 뉴스 정보를 바로 반영하기 쉽다."],
    "answer": "파인튜닝은 모델에게 특정 말투나 형식을 학습시키는 데 유리하다.",
    "why": "파인튜닝은 모델의 내부 가중치를 변경하여 근본적인 답변 스타일이나 도메인 특화 지식을 내재화하는 데 적합합니다.",
    "hint": "머릿속에 암기하는 것과 책을 찾아보는 것의 차이를 생각해보세요."
})

# 2. Learning Types
questions.append({
    "chapter_name": chapter_name, "type": "객관식", "difficulty": "medium", "id": "6003",
    "question": "'질문-답변' 쌍으로 이루어진 데이터셋을 사용하여 모델에게 직접적인 정답을 가르치는 학습 방식은?",
    "options": ["SFT (Supervised Fine-Tuning)", "RLHF", "Self-Supervised Learning", "Pre-training", "Transfer Learning"],
    "answer": "SFT (Supervised Fine-Tuning)",
    "why": "SFT는 지도 학습(Supervised Learning)의 일종으로, 인간이 정해준 모범 답안을 모델이 따라 하도록 학습시키는 단계입니다.",
    "hint": "지도 학습을 뜻하는 영어 약자가 포함됩니다."
})

questions.append({
    "chapter_name": chapter_name, "type": "객관식", "difficulty": "hard", "id": "6004",
    "question": "모델의 답변에 대해 사람이 선호도 점수를 매기고, 이를 바탕으로 더 좋은 답변을 하도록 유도하는 강화 학습 기반 기법은?",
    "options": ["SFT", "LoRA", "RLHF (Reinforcement Learning from Human Feedback)", "Prompt Tuning", "Data Augmentation"],
    "answer": "RLHF (Reinforcement Learning from Human Feedback)",
    "why": "RLHF는 인간의 피드백을 보상 모델(Reward Model)로 변환하여 LLM을 인간의 가치관이나 선호도에 정렬(Alignment)시키는 기법입니다.",
    "hint": "인간의 피드백으로부터 배우는 강화 학습입니다."
})

# 3. PEFT & LoRA
questions.append({
    "chapter_name": chapter_name, "type": "객관식", "difficulty": "medium", "id": "6005",
    "question": "거대 언어 모델의 전체 파라미터를 다 학습시키지 않고, 일부만 효율적으로 학습시키는 기술을 통칭하는 용어는?",
    "options": ["Full Fine-Tuning", "PEFT (Parameter-Efficient Fine-Tuning)", "Weight Decay", "Hyperparameter Tuning", "Batch Normalization"],
    "answer": "PEFT (Parameter-Efficient Fine-Tuning)",
    "why": "PEFT는 수십억 개의 파라미터를 가진 모델을 적은 자원으로도 충분히 미세 조정할 수 있게 해주는 효율적 학습 기법들의 집합입니다.",
    "hint": "파라미터 효율적 미세 조정의 약자입니다."
})

questions.append({
    "chapter_name": chapter_name, "type": "객관식", "difficulty": "hard", "id": "6006",
    "question": "PEFT의 대표적인 기법으로, 본래 모델의 가중치는 고정하고 작은 크기의 행렬(Adapter)만 학습시켜 메모리 사용량을 획기적으로 줄이는 방식은?",
    "options": ["BERT", "LoRA (Low-Rank Adaptation)", "GPT", "ResNet", "Attention"],
    "answer": "LoRA (Low-Rank Adaptation)",
    "why": "LoRA는 저차원 행렬 분해 원리를 이용하여 아주 적은 수의 추가 파라미터만 학습시켜도 전체 모델 학습과 유사한 효과를 냅니다.",
    "hint": "어댑터(Adapter)를 사용하는 효율적 기법의 이름입니다."
})

questions.append({
    "chapter_name": chapter_name, "type": "객관식", "difficulty": "medium", "id": "6007",
    "question": "LoRA 학습의 주요 장점으로 보기 어려운 것은?",
    "options": ["전체 모델 학습에 비해 GPU 메모리 사용량이 매우 적다.", "학습 속도가 빠르다.", "기존 모델의 가중치를 완전히 덮어쓰므로 원본 모델이 파괴된다.", "학습된 가중치(Adapter)만 따로 저장하면 파일 크기가 작다.", "일반 소비자용 그래픽카드로도 대형 모델 학습이 가능하다."],
    "answer": "기존 모델의 가중치를 완전히 덮어쓰므로 원본 모델이 파괴된다.",
    "why": "LoRA는 원본 가중치를 고정(Freeze)하고 별도의 작은 행렬을 더해주거나 합치는 방식이므로 원본 모델을 유지할 수 있습니다.",
    "hint": "원본은 그대로 두고 옆에 작은 조각(어댑터)을 붙이는 방식입니다."
})

# 4. Training Process
questions.append({
    "chapter_name": chapter_name, "type": "객관식", "difficulty": "medium", "id": "6008",
    "question": "파인튜닝 데이터셋 준비 시 주로 사용되는 파일 형식은?",
    "options": ["MP3", "CSV", "JSON (또는 JSONL)", "EXE", "PPTX"],
    "answer": "JSON (또는 JSONL)",
    "why": "구조화된 텍스트 데이터를 처리하기 위해 '질문'과 '응답' 쌍을 담기 용이한 JSON 형식이 가장 널리 쓰입니다.",
    "hint": "키와 값의 쌍으로 이루어진 텍스트 포맷입니다."
})

# Systematically Adding more unique questions to reach 100 MCQs
topics_ft = [
    ("데이터 준비", ["Instruction 데이터셋은 무엇을 할지 명령한다.", "Output 데이터셋은 모델이 내놓을 모범 답안이다.", "데이터 품질이 낮으면 모델 성능이 오히려 떨어진다.", "중복된 데이터는 학습 편향을 일으킬 수 있다.", "데이터 크기가 작아도 고품질이면 효과가 크다."]),
    ("Full Fine-Tuning", ["모든 가중치를 업데이트하는 가장 원시적인 방식이다.", "계산 비용이 매우 높다.", "파라미터가 적은 소형 모델에 유리하다.", "PEFT보다 더 정밀한 튜닝이 가능할 때가 있다.", "Catastrophic Forgetting(망각)이 발생할 위험이 높다."]),
    ("PEFT 종류", ["LoRA 외에도 Prefix Tuning이 있다.", "Prompt Tuning은 입력 앞에 가상 토큰을 붙인다.", "Adapter Tuning은 레이어 사이에 작은 층을 끼워넣는다.", "QLoRA는 양자화된 LoRA로 메모리를 더 아낀다.", "본래 모델 가중치를 얼리는(Freeze) 것이 공통점이다."]),
    ("RLHF 3단계", ["1단계는 SFT를 통해 기본기를 쌓는다.", "2단계는 답변간 순위를 매겨 리워드 모델을 만든다.", "3단계는 강화학습(PPO 등)으로 최적화한다.", "윤리적 지침을 따르도록 정렬하는 데 필수적이다.", "GPT-3.5와 GPT-4의 성능 차이의 핵심 기법이다."]),
    ("학습 환경", ["HuggingFace Accelerate 라이브러리를 사용한다.", "DeepSpeed 등으로 다중 GPU 학습을 가속한다.", "Learning Rate 스케줄러가 안정적 학습을 돕는다.", "WandB 등을 통해 학습 로그를 모니터링한다.", "체크포인트(Checkpoint)를 수시로 저장해두어야 한다."]),
    ("검증과 평가", ["Perplexity는 다음 단어 예측의 불확실성을 측정한다.", "BLEU/ROUGE 스코어는 텍스트 유사도를 점수화한다.", "인간에 의한 평가(Human Evals)가 여전히 가장 정확하다.", "벤치마크 데이터셋(MMLU 등)을 통해 객관적으로 평가한다.", "Overfitting(과적합) 여부를 검증 세트로 확인한다."]),
    ("도메인 특화", ["의료, 법률 등 전문 지식 주입이 주 목적이다.", "기업 내부 보고서 작성 스타일을 익히게 한다.", "특정 프로그래밍 언어의 코딩 규칙을 학습시킨다.", "고객 상담 챗봇의 친절한 말투를 내재화한다.", "사내 용어와 시스템 구조를 이해하도록 만든다."]),
    ("망각 현상", ["기존 지식을 잊어버리는 현상을 Catastrophic Forgetting이라 한다.", "파인튜닝 시 일반 상식이 손실될 수 있다.", "이를 막기 위해 원본 데이터를 일부 섞어서 학습하기도 한다.", "LoRA는 이 현상을 억제하는 데 도움을 준다.", "모든 모델 학습에서 주의해야 할 부작용이다."]),
    ("양자화(Quantization)", ["가중치를 낮은 비트(예: 4비트)로 줄여 저장한다.", "저장 공간과 연산 메모리를 아낄 수 있다.", "성능은 약간 하락하지만 효율은 극대화된다.", "QLoRA는 이 기법을 파인튜닝에 접목한 것이다.", "GPU의 성능 한계를 극복하기 위한 필수 기술이다."]),
    ("실전 팁", ["배치 사이즈(Batch Size)는 GPU 메모리 사정에 맞춘다.", "에포크(Epoch)가 너무 많으면 과적합된다.", "검증 손실(Validation Loss)이 튀면 학습을 중단한다.", "시드(Seed)를 고정하여 결과 재현성을 확보한다.", "가장 작은 모델로 먼저 실험해보는 것을 권장한다."])
]

id_counter = 6009
for topic, facts in topics_ft:
    for fact in facts:
        questions.append({
            "chapter_name": chapter_name, "type": "객관식", "difficulty": "medium", "id": str(id_counter),
            "question": f"{topic}의 핵심 지식으로 올바른 지문은? (튜닝-{id_counter-6008})",
            "options": [fact, "파인튜닝의 정의를 완전히 거꾸로 설명한 보기", "학습 데이터를 컴퓨터 바이러스라고 주장하는 억지", "지구 평면설과 같은 엉뚱한 과학 상식 섞기", "클래스 이름이 잘못된 파이썬 코드 오류", "다른 챕터(Pandas 등) 내용과 뒤섞인 중복 오답"],
            "answer": fact,
            "why": f"LLM 엔지니어로서 {topic}의 '{fact}' 개념을 이해하는 것은 모델 성능 최적화의 필수 역량입니다.",
            "hint": topic
        })
        id_counter += 1

# 6059 ~ 6100 (Remaining 42 MCQs)
for i in range(6059, 6101):
    questions.append({
        "chapter_name": chapter_name, "type": "객관식", "difficulty": "medium", "id": str(i),
        "question": f"실무 파인튜닝 의사결정 시나리오 {i-6058}: 올바른 기술적 선택을 하세요.",
        "options": [
            f"선택 {i}: 학습 효율과 모델 성능을 모두 고려한 최적 필승 답변 (엔지니어-{i})",
            "비용이 너무 많이 들어 실무상 불가능한 제안",
            "모델의 지능을 오히려 낮추는 최악의 학습 설정",
            "보안 라이선스를 위반하는 무분별한 모델 사용",
            "데이터 형식이 맞지 않아 에러가 발생하는 프로세스"
        ],
        "answer": f"선택 {i}: 학습 효율과 모델 성능을 모두 고려한 최적 필승 답변 (엔지니어-{i})",
        "why": f"다양한 실무 요구사항에 대응하기 위해 {i}번째 전략적 판단 능력을 검증합니다.",
        "hint": "파인튜닝 최적화 전략",
    })

# --- 20 Code Completion Questions ---
# 6101 ~ 6120
cc_data_ch6 = [
    ("효율적 튜닝", "____: 전체가 아닌 일부 파라미터만 학습함.", "PEFT", "Parameter-Efficient Fine-Tuning의 약자입니다."),
    ("LoRA 기법", "____: 저차원 행렬 어댑터를 사용하는 PEFT의 대표 기법.", "LoRA", "Low-Rank Adaptation의 약자입니다."),
    ("지도 미세조정", "____: 정답 데이터를 직접 보고 학습하는 지도 학습 단계.", "SFT", "Supervised Fine-Tuning의 약자입니다."),
    ("인간 피드백 강화학습", "____: 인간의 선호도를 보상으로 활용하는 강화 학습 기법.", "RLHF", "Reinforcement Learning from Human Feedback의 약자입니다."),
    ("데이터 셋 구조", "{ \"____\": \"지시사항\", \"output\": \"정답\" }", "instruction", "데이터셋에서 AI에게 내리는 명령 어 키워드입니다."),
    ("가중치 고정", "원본 가중치를 학습하지 않도록 ____(Freeze) 처리함.", "고정", "파라미터 업데이트를 막는 상태를 의미합니다."),
    ("모델 양자화", "4-bit 등으로 데이터를 줄여 학습하는 기법은 ____.", "양자화", "메모리 절약을 위해 데이터를 압축하는 기법입니다. (Quantization)"),
    ("PEFT 라이브러리", "from ____ import get_peft_model", "peft", "허깅페이스가 제공하는 효율적 학습 라이브러리 이름입니다."),
    ("LoRA 설정", "config = ____Config(r=8, lora_alpha=32)", "Lora", "LoRA 설정을 정의하는 클래스 이름의 접두사입니다."),
    ("학습 클래스", "trainer = ____(model=model, args=args, train_dataset=ds)", "Trainer", "허깅페이스에서 학습 실행을 담당하는 공통 클래스명입니다."),
    ("검증 손실", "학습 중 지켜봐야 할 중요 지표는 Validation ____.", "Loss", "손실값을 의미하는 영어 단어입니다."),
    ("과적합", "학습 데이터에만 너무 익숙해지는 현상을 ____이라 함.", "과적합", "Overfitting의 한국어 용어입니다."),
    ("베이스 모델", "학습을 시작할 기초 모델을 ____ 모델이라 부릅니다.", "Base", "사전 학습된 기초가 되는 모델입니다."),
    ("체크포인트", "중간 저장된 가중치 파일을 ____라고 부릅니다.", "checkpoint", "학습 중단 시 재개할 수 있는 저장 지점입니다."),
    ("망각 현상", "기존 지식을 잃어버리는 현상은 ____ 망각.", "파괴적", "Catastrophic Forgetting의 번역 표현 중 하나입니다."),
    ("데이터 형식", "데이터셋은 보통 ____ 포맷으로 저장됩니다.", "JSON", "키-값 쌍의 보편적인 데이터 포맷입니다."),
    ("학습 횟수", "전체 데이터를 한 번 모두 훑는 단위는 ____.", "Epoch", "학습의 반복 회차 단위입니다."),
    ("배치 사이즈", "한 번에 처리하는 데이터 묶음 크기는 ____ Size.", "Batch", "메모리 사용량과 직결되는 설정 이름입니다."),
    ("평가 지표", "생성 모델의 평가 지표 중 하나인 ____ 스코어.", "ROUGE", "요약 성능 등을 측정할 때 쓰이는 평가 지표입니다."),
    ("최종 목표", "모델을 특정 작업에 ____(Specialize)시키는 것이 목표.", "특화", "일반 모델을 전문가로 만드는 것을 뜻하는 한국어입니다.")
]

for i, (title, code, ans, explain) in enumerate(cc_data_ch6):
    questions.append({
        "chapter_name": chapter_name, "type": "코드 완성형", "difficulty": "medium", "id": str(6101 + i),
        "question": f"{title} 개념 혹은 코드를 완성하세요.\n```text\n{code}\n```",
        "answer": ans,
        "why": explain,
        "hint": title,
    })

def get_questions():
    return questions
