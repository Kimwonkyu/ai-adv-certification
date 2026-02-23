import json
import random

# New, diverse questions for Fine Tuning to replace duplicates
NEW_FINE_TUNING_QUESTIONS = [
    {
        "chapter_name": "Fine Tuning",
        "type": "객관식",
        "question": "LoRA(Low-Rank Adaptation)에서 'Alpha' 파라미터가 32이고 'Rank(r)'가 32일 때, 스케일링 팩터(Alpha/r)의 값은?",
        "options": ["0.5", "1.0", "2.0", "32", "0"],
        "answer": "1.0",
        "why": "스케일링 팩터는 Alpha / Rank로 계산됩니다. 32/32 = 1이므로 어댑터의 가중치가 그대로 반영됩니다.",
        "hint": "Alpha 나누기 Rank입니다.",
        "trap_points": ["Alpha와 Rank가 같으면 보정이 없다는 뜻임"],
        "difficulty": "medium",
        "id": "0801"
    },
    {
        "chapter_name": "Fine Tuning",
        "type": "객관식",
        "question": "파인튜닝 시 전체 모델 파라미터를 고정(Freeze)하고, 특정 레이어에만 학습 가능한 파라미터를 주입하는 방식의 총칭은?",
        "options": ["PEFT (Parameter-Efficient Fine-Tuning)", "Full Fine-tuning", "Pre-training", "RLHF", "DPO"],
        "answer": "PEFT (Parameter-Efficient Fine-Tuning)",
        "why": "PEFT는 거대 모델의 대부분을 얼리고 일부만 학습하여 자원 효율성을 극대화하는 기법들의 통칭입니다.",
        "hint": "효율적(Efficient)인 파라미터 사용.",
        "trap_points": ["LoRA는 PEFT의 하위 종류 중 하나임"],
        "difficulty": "easy",
        "id": "0802"
    },
    {
        "chapter_name": "Fine Tuning",
        "type": "객관식",
        "question": "CAT(Catastrophic Forgetting) 현상을 막기 위해, 파인튜닝 시 기존 데이터셋의 일부를 섞어서 학습시키는 전략은?",
        "options": ["Replay Buffer (리플레이 버퍼)", "Dropout", "Early Stopping", "Gradient Clipping", "Batch Normalization"],
        "answer": "Replay Buffer (리플레이 버퍼)",
        "why": "과거의 데이터를 '다시 재생(Replay)'하여 모델이 이전 지식을 잊지 않도록 상기시켜 줍니다.",
        "hint": "다시 플레이(Re-play)함.",
        "trap_points": ["강화학습에서도 사용되는 용어임"],
        "difficulty": "hard",
        "id": "0803"
    },
    {
        "chapter_name": "Fine Tuning",
        "type": "객관식",
        "question": "모델이 이미 학습한 지식과 상충되는 데이터로 무리하게 파인튜닝을 시도할 때 발생하는 '지식 충돌'의 결과는?",
        "options": ["모델이 더 똑똑해진다", "할루시네이션이 증가하고 일관성이 떨어진다", "학습 속도가 빨라진다", "GPU 사용량이 줄어든다", "새로운 언어를 창조한다"],
        "answer": "할루시네이션이 증가하고 일관성이 떨어진다",
        "why": "내재된 지식과 새로운 정보가 싸우면서 모델이 혼란을 겪고, 결국 거짓 정보를 지어내게 됩니다.",
        "hint": "억지로 주입식 교육을 할 때의 부작용을 생각하세요.",
        "trap_points": ["지식 편집(Knowledge Editing)이 어려운 이유 중 하나"],
        "difficulty": "medium",
        "id": "0804"
    },
    {
        "chapter_name": "Fine Tuning",
        "type": "객관식",
        "question": "LLM의 성능 평가 벤치마크 중, 수학적 추론 능력을 집중적으로 평가하는 지표는?",
        "options": ["GSM8K", "MMLU", "HellaSwag", "HumanEval", "BLEU"],
        "answer": "GSM8K",
        "why": "초등학교 수준의 수학 문제를 단계적으로 풀게 하여 모델의 논리적 추론력을 측정합니다.",
        "hint": "수학(Math) 관련 약자.",
        "trap_points": ["MMLU는 종합 지식 평가임"],
        "difficulty": "hard",
        "id": "0805"
    },
    {
        "chapter_name": "Fine Tuning",
        "type": "객관식",
        "question": "RLHF에서 사람의 선호도 데이터를 모을 때 'A 답변이 B 답변보다 낫다'고 판단하는 방식은?",
        "options": ["Pairwise Comparison (쌍대 비교)", "Absolute Scoring", "Binary Classification", "Multi-class Labeling", "Regression"],
        "answer": "Pairwise Comparison (쌍대 비교)",
        "why": "점수를 매기는 것보다 두 개를 두고 비교하는 것이 평가자의 일관성을 유지하기 훨씬 쉽습니다.",
        "hint": "둘씩 짝(Pair)을 지어 비교함.",
        "trap_points": ["Elo Rating 시스템과 유사한 원리"],
        "difficulty": "medium",
        "id": "0806"
    },
    {
        "chapter_name": "Fine Tuning",
        "type": "객관식",
        "question": "QLoRA에서 사용하는 'Double Quantization' 기술의 핵심 목적은?",
        "options": ["추론 속도를 2배로 늘리기 위해", "양자화 상수(Quantization Constant) 자체를 또 양자화하여 메모리를 추가로 절약하기 위해", "정확도를 2배 올리기 위해", "학습 시간을 줄이기 위해", "GPU 온도를 낮추기 위해"],
        "answer": "양자화 상수(Quantization Constant) 자체를 또 양자화하여 메모리를 추가로 절약하기 위해",
        "why": "티끌 모아 태산이라고, 메타데이터인 상수값까지 압축하여 거대 모델을 소비자용 GPU에 구겨 넣습니다.",
        "hint": "두 번(Double) 양자화함.",
        "trap_points": ["극도의 메모리 효율화를 위한 기법임"],
        "difficulty": "hard",
        "id": "0807"
    },
    {
        "chapter_name": "Fine Tuning",
        "type": "객관식",
        "question": "SFT(Supervised Fine-Tuning)용 데이터셋에서 가장 중요한 품질 요소는?",
        "options": ["데이터의 양 (Quantity)", "데이터의 다양성과 정확성 (Quality & Diversity)", "데이터의 파일 포맷", "데이터의 생성 날짜", "데이터의 언어"],
        "answer": "데이터의 다양성과 정확성 (Quality & Diversity)",
        "why": "적은 양이라도 고품질의 다양한 데이터를 학습하는 것이 수만 개의 저품질 데이터보다 훨씬 효과적입니다(LIMA 논문).",
        "hint": "Less is More.",
        "trap_points": ["양이 많으면 오히려 독이 될 수도 있음"],
        "difficulty": "easy",
        "id": "0808"
    },
    {
        "chapter_name": "Fine Tuning",
        "type": "객관식",
        "question": "RAG 시스템을 위해 임베딩 모델을 파인튜닝할 때 사용하는 손실 함수(Loss Function)는?",
        "options": ["Contrastive Loss (대조 손실)", "Cross Entropy Loss", "Mean Squared Error", "Hinge Loss", "Log Loss"],
        "answer": "Contrastive Loss (대조 손실)",
        "why": "관련 있는 문서끼리는 가깝게, 없는 문서끼리는 멀게 벡터 공간을 조정해야 하기 때문입니다.",
        "hint": "관련성 유무를 대조(Contrast)합니다.",
        "trap_points": ["InfoNCE Loss라고도 불림"],
        "difficulty": "hard",
        "id": "0809"
    },
    {
        "chapter_name": "Fine Tuning",
        "type": "객관식",
        "question": "파인튜닝된 모델이 특정 프롬프트 템플릿(<|user|>, <|assistant|>)을 지키지 않으면 발생하는 문제는?",
        "options": ["EOS 토큰을 생성하지 못하고 끝없이 횡설수설한다", "모델이 침묵한다", "기존 성능이 향상된다", "토큰 비용이 줄어든다", "아무 문제 없다"],
        "answer": "EOS 토큰을 생성하지 못하고 끝없이 횡설수설한다",
        "why": "학습된 종료 패턴을 찾지 못해 문장을 끝맺지 못하고 계속 이어 말하는 현상이 발생합니다.",
        "hint": "말을 멈추는 법을 모르게 됩니다.",
        "trap_points": ["채팅 모델 튜닝 시 가장 흔한 실수임"],
        "difficulty": "medium",
        "id": "0810"
    },
    {
        "chapter_name": "Fine Tuning",
        "type": "객관식",
        "question": "파인튜닝 시 배치 크기(Batch Size)를 키우기 위해 여러 GPU에 데이터를 나누어 담는 기술은?",
        "options": ["Data Parallelism (데이터 병렬화)", "Model Parallelism", "Pipeline Parallelism", "Tensor Parallelism", "Serial Processing"],
        "answer": "Data Parallelism (데이터 병렬화)",
        "why": "같은 모델을 여러 GPU에 복사해두고, 서로 다른 데이터 조각을 먹여서 학습 속도를 배로 늘립니다.",
        "hint": "데이터를 병렬(Parallel)로 처리.",
        "trap_points": ["DDP, FSDP 등이 여기에 해당함"],
        "difficulty": "medium",
        "id": "0811"
    },
    {
        "chapter_name": "Fine Tuning",
        "type": "객관식",
        "question": "의료나 법률 같은 특수 데이터로 파인튜닝할 때, 일반 상식 능력이 떨어지는 현상을 방지하기 위한 방법은?",
        "options": ["일반 데이터를 섞어서 학습 (General Instruction Mixing)", "특수 데이터만 계속 학습", "학습률을 높임", "모델 크기를 줄임", "영어 데이터 제거"],
        "answer": "일반 데이터를 섞어서 학습 (General Instruction Mixing)",
        "why": "편식하면 건강을 해치듯, 전문 지식만 배우면 일반 대화 능력을 잃기 때문에 균형 잡힌 식단(데이터)이 필요합니다.",
        "hint": "골고루 섞어줍니다.",
        "trap_points": ["전문성만 추구하다 바보가 되는 것을 막음"],
        "difficulty": "easy",
        "id": "0812"
    },
    {
        "chapter_name": "Fine Tuning",
        "type": "객관식",
        "question": "최신 파인튜닝 트렌드인 'NEFTune' 기법이 임베딩 벡터에 노이즈를 섞는 이유는?",
        "options": ["모델의 과적합을 막고 일반화 성능을 높이기 위해", "모델을 헷갈리게 하려고", "보안을 강화하려고", "학습을 방해하려고", "데이터를 손상시키려고"],
        "answer": "모델의 과적합을 막고 일반화 성능을 높이기 위해",
        "why": "약간의 무작위 노이즈는 모델이 특정 데이터 패턴에만 집착하는 것을 방지하여 오히려 대화의 질을 높여줍니다.",
        "hint": "적절한 소음(Noise)은 면역력을 키워줍니다.",
        "trap_points": ["알파카(Alpaca) 데이터셋 실험에서 효과가 입증됨"],
        "difficulty": "hard",
        "id": "0813"
    },
    {
        "chapter_name": "Fine Tuning",
        "type": "객관식",
        "question": "파인튜닝이 완료된 LoRA 어댑터를 원본 모델과 합쳐서 하나의 파일로 만드는 과정은?",
        "options": ["Merge & Unload", "Zip", "Compile", "Quantize", "Distill"],
        "answer": "Merge & Unload",
        "why": "추론 속도를 높이기 위해 별도로 계산되던 어댑터 가중치를 원본 가중치 행렬에 더해버리고 메모리에서 내립니다.",
        "hint": "병합(Merge)하고 내보냄.",
        "trap_points": ["이후에는 어댑터를 분리할 수 없음 (비가역적)"],
        "difficulty": "medium",
        "id": "0814"
    },
    {
        "chapter_name": "Fine Tuning",
        "type": "객관식",
        "question": "생성형 AI 모델을 평가할 때 'GPT-4'를 심판(Judge)으로 사용하여 점수를 매기는 방식을 무엇이라 하나요?",
        "options": ["LLM-as-a-Judge", "Human Eval", "Code Review", "Self-Play", "Peer Review"],
        "answer": "LLM-as-a-Judge",
        "why": "사람이 평가하기엔 너무 비싸고 느려서, 고성능 모델에게 채점을 맡기는 현대적인 평가 트렌드입니다.",
        "hint": "LLM이 판사(Judge) 역할을 함.",
        "trap_points": ["MT-Bench 등이 이 방식을 사용함"],
        "difficulty": "medium",
        "id": "0815"
    }
]

def fix_content():
    try:
        with open('public/questions.json', 'r', encoding='utf-8') as f:
            questions = json.load(f)
    except FileNotFoundError:
        print("Error: public/questions.json not found.")
        return

    # 1. Deduplication for Fine Tuning
    # Identified Duplicates range roughly from id 0685 to 0700 (based on audit)
    # Strategy: Remove specific IDs identified as duplicates in the audit.
    # The audit showed 0685-0700 are duplicates of 0601-0616.
    
    duplicate_ids = [
        "0685", "0686", "0687", "0688", "0689", "0690", "0691", "0692",
        "0693", "0694", "0695", "0696", "0697", "0698", "0699", "0700"
    ]
    
    initial_count = len(questions)
    cleaned_questions = [q for q in questions if q['id'] not in duplicate_ids]
    removed_count = initial_count - len(cleaned_questions)
    
    print(f"Removed {removed_count} duplicate questions.")

    # 2. Fix 'All of the above' Question (ID 0547)
    for q in cleaned_questions:
        if q['id'] == "0547":
            q['question'] = "에이전트가 목표 달성 루프를 긴급 중단해야 하는 '안전 조건'으로 가장 적절한 것은?"
            q['options'] = [
                "답변이 너무 짧을 때",
                "설정된 최대 반복 횟수(Max Iterations)에 도달하여 무한 루프 위험이 있을 때",
                "사용자가 칭찬했을 때",
                "인터넷 연결이 너무 빠를 때",
                "배터리가 100%일 때"
            ]
            q['answer'] = "설정된 최대 반복 횟수(Max Iterations)에 도달하여 무한 루프 위험이 있을 때"
            q['why'] = "비용 폭주와 시스템 과부하를 막기 위해 에이전트는 반드시 '최대 시도 횟수'라는 안전장치를 가져야 합니다."
            print("Fixed Question 0547.")

    # 3. Add New Questions
    # Ensure IDs are unique. We started new IDs from 0801.
    final_questions = cleaned_questions + NEW_FINE_TUNING_QUESTIONS
    print(f"Added {len(NEW_FINE_TUNING_QUESTIONS)} new questions.")
    
    # 4. Save
    with open('public/questions.json', 'w', encoding='utf-8') as f:
        json.dump(final_questions, f, indent=2, ensure_ascii=False)
    
    print(f"Total questions count: {len(final_questions)}")

if __name__ == "__main__":
    fix_content()
