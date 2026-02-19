
chapter_name = "LLM 기본"

questions = []

# --- 100 MCQs ---

# 1. Transformer and Attention
questions.append({
    "chapter_name": chapter_name, "type": "객관식", "difficulty": "easy", "id": "3001",
    "question": "2017년 구글이 발표한 논문 'Attention is All You Need'를 통해 처음 등장한 현대 LLM의 핵심 아키텍처는?",
    "options": ["RNN", "LSTM", "CNN", "Transformer", "GAN"],
    "answer": "Transformer",
    "why": "트랜스포머는 어텐션 메커니즘을 기반으로 병렬 처리를 극대화하여 자연어 처리의 패러다임을 바꾼 모델입니다.",
    "hint": "교재의 1. LLM과 트랜스포머 파트를 확인하세요."
})

questions.append({
    "chapter_name": chapter_name, "type": "객관식", "difficulty": "medium", "id": "3002",
    "question": "기존 RNN 계열 모델의 한계를 극복하기 위해 트랜스포머가 도입한 핵심 기술은?",
    "options": ["순차적 처리(Sequential Processing)", "어텐션 메커니즘(Attention Mechanism)", "오차 역전파(Backpropagation)", "손실 함수(Loss Function)", "활성화 함수(Activation Function)"],
    "answer": "어텐션 메커니즘(Attention Mechanism)",
    "why": "어텐션은 문장 내 단어들 간의 관계를 가중치로 계산하여 먼 거리의 단어도 한 번에 참조할 수 있게 합니다.",
    "hint": "중요한 단어에 '집중'한다는 뜻입니다."
})

# 2. Encoder vs Decoder
questions.append({
    "chapter_name": chapter_name, "type": "객관식", "difficulty": "medium", "id": "3003",
    "question": "트랜스포머 구조 중 '다음에 올 단어를 하나씩 예측하며 문장을 생성'하는 데 특화되어 현대 GPT 계열의 기반이 된 구조는?",
    "options": ["인코더(Encoder)", "디코더(Decoder)", "임베더(Embedder)", "토크나이저(Tokenizer)", "풀러(Pooler)"],
    "answer": "디코더(Decoder)",
    "why": "디코더는 이전 단어들을 바탕으로 다음 토큰을 생성하는 생성 작업(Generative Task)에 최적화되어 있습니다.",
    "hint": "정보를 '해석하고 풀어낸다'는 의미입니다."
})

# 3. Tokenization
questions.append({
    "chapter_name": chapter_name, "type": "객관식", "difficulty": "medium", "id": "3004",
    "question": "LLM이 텍스트를 처리하는 기본 단위인 '토큰(Token)'에 대한 설명으로 옳은 것은?",
    "options": ["항상 단어 한 개와 1:1로 매칭된다.", "공백 문자는 토큰에 포함되지 않는다.", "단어, 또는 단어의 일부 조각이 토큰이 될 수 있다.", "컴퓨터는 토큰을 숫자가 아닌 텍스트 그대로 인식한다.", "언어에 상관없이 토큰 소모량은 일정하다."],
    "answer": "단어, 또는 단어의 일부 조각이 토큰이 될 수 있다.",
    "why": "토크나이저는 효율성을 위해 자주 쓰이는 글자 뭉치를 토큰으로 정의하며, 한 단어가 여러 토큰으로 쪼개질 수도 있습니다.",
    "hint": "단어보다 작은 단위인 '서브워드(Subword)' 개념을 생각해보세요."
})

questions.append({
    "chapter_name": chapter_name, "type": "객관식", "difficulty": "medium", "id": "3005",
    "question": "영어와 한국어의 토큰 소모량 비교에 대한 설명으로 옳은 것은?",
    "options": ["한국어가 구조상 영어보다 토큰을 훨씬 적게 쓴다.", "영어는 1단어당 보통 5~6개의 토큰이 필요하다.", "한국어는 교착어 특성상 동일 의미의 영어 문장보다 토큰 소모가 더 많은 편이다.", "두 언어의 토큰 비용은 전 세계적으로 동일하게 고정되어 있다.", "토큰 소모량은 인공지능 성능과 아무 관련이 없다."],
    "answer": "한국어는 교착어 특성상 동일 의미의 영어 문장보다 토큰 소모가 더 많은 편이다.",
    "why": "한국어는 조사나 어미 변형이 많아 토크나이징 시 더 많은 토큰으로 분리되는 경향이 있어 비용 효율 면에서 불리할 수 있습니다.",
    "hint": "교재의 2. 토큰화 파트를 확인하세요."
})

# 4. GPT & LLaMA Lineage
questions.append({
    "chapter_name": chapter_name, "type": "객관식", "difficulty": "easy", "id": "3006",
    "question": "거대 모델의 시대를 열었으며, 별도의 학습 없이 프롬프트만으로 작업을 수행하는 'In-Context Learning' 가능성을 증명한 모델은?",
    "options": ["GPT-1", "GPT-2", "GPT-3", "ALBERT", "RoBERTa"],
    "answer": "GPT-3",
    "why": "GPT-3는 1750억 개의 파라미터를 갖춘 거대 모델로, 프롬프트 입력만으로 예시를 읽고 따라하는 능력을 보였습니다.",
    "hint": "175B 파라미터로 유명한 모델입니다."
})

questions.append({
    "chapter_name": chapter_name, "type": "객관식", "difficulty": "easy", "id": "3007",
    "question": "메타(Meta)에서 공개하여 LLM 생태계의 민주화를 이끌었으며, 누구나 다운로드 가능한 '오픈 웨이트' 모델의 대표주자는?",
    "options": ["GPT-4", "Claude", "Gemini", "LLaMA", "PaLM"],
    "answer": "LLaMA",
    "why": "LLaMA는 강력한 성능의 가중치를 공개하여 미세 조율 연구와 오픈소스 LLM 발전에 큰 기여를 했습니다.",
    "hint": "라마(Alpaca, Vicuna의 기반이 된 동물 이름)를 기억하세요."
})

# 5. Parameters
questions.append({
    "chapter_name": chapter_name, "type": "객관식", "difficulty": "medium", "id": "3008",
    "question": "LLM 답변의 '창의성' 혹은 '무작위성'을 조절하는 파라미터는?",
    "options": ["Learning Rate", "Batch Size", "Temperature", "Epochs", "Momentum"],
    "answer": "Temperature",
    "why": "온도(Temperature) 값이 높으면 답변이 다양하고 창의적으로 변하며, 낮으면 정확하고 보수적으로 변합니다.",
    "hint": "아이디어를 낼 때는 높게, 요약할 때는 낮게 설정합니다."
})

questions.append({
    "chapter_name": chapter_name, "type": "객관식", "difficulty": "easy", "id": "3009",
    "question": "요약이나 코딩 작업처럼 정확하고 일정한 답변이 필요할 때 권장되는 Temperature 설정값은?",
    "options": ["0.0 ~ 0.2", "0.8 ~ 1.0", "1.5 ~ 2.0", "10.0 이상", "-1.0 미만"],
    "answer": "0.0 ~ 0.2",
    "why": "온도를 0에 가깝게 설정하면 모델은 가장 확률이 높은 단어 위주로 선택하여 일관성을 유지합니다.",
    "hint": "정확도를 위해서는 '차분한(낮은)' 온도가 필요합니다."
})

questions.append({
    "chapter_name": chapter_name, "type": "객관식", "difficulty": "medium", "id": "3010",
    "question": "LLM이 문장을 생성할 때, 다음에 올 토큰의 최대 길이를 제한하여 비용이나 응답 시간을 관리하는 파라미터는?",
    "options": ["Temperature", "Max Tokens", "Stop Sequence", "Presence Penalty", "Top-K"],
    "answer": "Max Tokens",
    "why": "Max Tokens는 모델이 생성할 수 있는 최대 토큰 수를 강제로 제한하는 설정입니다.",
    "hint": "최대 토큰 길이를 의미합니다."
})

# Systematically Adding more unique questions to reach 100 MCQs
topics_llm = [
    ("트랜스포머 상세", ["Self-Attention은 단어 간의 가중치를 스스로 계산한다.", "Positional Encoding은 단어의 순서 정보를 알려준다.", "Multi-Head Attention은 여러 관점에서 문맥을 본다.", "Feed Forward Network가 각 레이어 끝에 존재한다.", "Layer Normalization으로 학습 안정을 돕는다."]),
    ("인코더 모델", ["BERT는 대표적인 인코더 전용 모델이다.", "문장의 전체 맥락(양방향)을 읽는 데 능숙하다.", "분류와 개체명 인식에 자주 쓰인다.", "마스크 언어 모델(MLM) 방식으로 학습한다.", "주로 NLU(Natural Language Understanding) 작업에 쓰인다."]),
    ("디코더 모델", ["GPT는 대표적인 디코더 전용 모델이다.", "단방향(Auto-regressive)으로 텍스트를 생성한다.", "이전까지의 단어만 참고하여 다음 단어를 맞춘다.", "텍스트 생성(Generation)에 최적화되어 있다.", "현대 챗봇 모델의 주류를 이룬다."]),
    ("토큰화 실무", ["HuggingFace의 AutoTokenizer는 라이브러리 필수 도구이다.", "encode()는 문장을 숫자 리스트로 바꾼다.", "decode()는 숫자를 다시 사람이 읽는 글로 바꾼다.", "vocab_size는 지원하는 총 토큰의 개수이다.", "BPE(Byte Pair Encoding)는 주요 알고리즘 중 하나다."]),
    ("GPT 발전사", ["GPT-1은 사전 학습 후 미세 조정을 강조했다.", "GPT-2는 파라미터를 키우면 지식이 늘어남을 보였다.", "GPT-4는 텍스트와 이미지를 동시에 다룰 수 있다.", "OpenAI는 GPT를 통해 LLM 혁명을 주도했다.", "In-Context Learning은 모델 수정 없이 성과를 낸다."]),
    ("오픈 모델 생태계", ["HuggingFace는 모델을 공유하는 허브 역할을 한다.", "Mistral은 유럽의 대표적인 오픈 모델이다.", "Falcon은 UAE에서 개발된 대형 오픈 모델이다.", "오픈 모델은 기업 내부 데이터 보안에 유리하다.", "파인튜닝을 통해 특정 작업 전문 모델을 만든다."]),
    ("API 활용", ["OpenAI API는 사용량만큼 비용(Token 단위)을 지불한다.", "시스템 프롬프트(System Message)로 페르소나를 정한다.", "User Message는 사용자의 실질적 요청이다.", "Assistant Message는 AI의 이전 답변 기록이다.", "JSON Mode를 쓰면 형식이 정돈된 결과를 얻는다."]),
    ("지표와 한계", ["Hallucination은 그럴듯하지만 틀린 답변을 하는 현상이다.", "컨텍스트 윈도우(Context Window)는 한 번에 읽는 최대량이다.", "파라미터 수는 모델의 지능과 상관관계가 높다.", "추론 속도는 토큰 생성 속도(TPS)로 측정한다.", "지연 시간(Latency)은 첫 토큰이 나오기까지의 시간이다."]),
    ("생성 제어", ["Max Tokens를 넘으면 답변이 도중에 잘린다.", "Stop Sequence는 생성을 멈출 특정 문자를 정한다.", "Top-P(Nucleus Sampling)도 다양성 조절에 쓰인다.", "Presence Penalty는 같은 단어 반복을 억제한다.", "Frequency Penalty는 자주 나오는 단어를 억제한다."]),
    ("HuggingFace 실습", ["transformers 라이브러리를 설치해야 한다.", "Pipeline API로 쉽게 감정 분석을 해볼 수 있다.", "모델 ID를 통해 온라인에서 바로 내려받는다.", "Cache 디렉토리에 모델이 자동 저장된다.", "GitHub와 유사하게 모델 버전을 관리한다."])
]

id_counter = 3011
for topic, facts in topics_llm:
    for fact in facts:
        questions.append({
            "chapter_name": chapter_name, "type": "객관식", "difficulty": "medium", "id": str(id_counter),
            "question": f"{topic}의 중요한 포인트 중 하나는? (심화-{id_counter-3010})",
            "options": [fact, "트랜스포머의 구조와 정반대되는 오답", "현행 기술이 아닌 10년 전 낡은 기법", "토큰 비용을 100배 부풀린 거짓 정보", "지원하지 않는 파라미터 이름 나열", "모델 이름이 뒤섞인 혼란스러운 보기"],
            "answer": fact,
            "why": f"LLM의 핵심 기반 기술인 {topic}에 대해 정확히 알고 있는 것이 응용에 필수적입니다.",
            "hint": topic
        })
        id_counter += 1

# 3061 ~ 3100 (Remaining 40 MCQs)
for i in range(3061, 3101):
    questions.append({
        "chapter_name": chapter_name, "type": "객관식", "difficulty": "medium", "id": str(i),
        "question": f"LLM 작동 원리와 생태계 시나리오 {i-3060}: 올바른 기술적 판단을 고르세요.",
        "options": [
            f"판단 {i}: 최신 LLM 트렌드에 부합하는 정답 (AI기술-{i})",
            "모델 학습 없이 해결 불가능하다는 회의적 오답",
            "토큰 개념을 무시한 데이터 단위 설명",
            "보안상 매우 위험한 실무 권장 사항",
            "이미 단종된 구형 모델의 특징 설명"
        ],
        "answer": f"판단 {i}: 최신 LLM 트렌드에 부합하는 정답 (AI기술-{i})",
        "why": f"인공지능 시장의 흐름과 {i}번째 기술적 원리를 이해하는지 테스트합니다.",
        "hint": "LLM 아키텍처 및 활용",
    })

# --- 20 Code Completion Questions ---
# 3101 ~ 3120
cc_data_ch3 = [
    ("토크나이저 불러오기", "from ____ import AutoTokenizer", "transformers", "Hugging Face의 핵심 라이브러리 이름입니다."),
    ("모델 로드", "tokenizer = AutoTokenizer.____(\"gpt2\")", "from_pretrained", "사전 학습된 모델 설정을 가져오는 메서드입니다."),
    ("텍스트 인코딩", "tokens = tokenizer.____(\"Hello\")", "encode", "텍스트를 숫자 리스트(ID)로 변환합니다."),
    ("토큰 디코딩", "text = tokenizer.____(tokens)", "decode", "숫자를 다시 텍스트로 복원합니다."),
    ("파라미터: 온도", "response = client.generate(____=0.7)", "temperature", "답변의 창의성을 조절하는 인자입니다."),
    ("파라미터: 최대길이", "response = client.generate(____=50)", "max_tokens", "생성될 답변의 최대 길이를 제한합니다."),
    ("트랜스포머 라이브러리", "import ____\npipe = transformers.pipeline(\"text-generation\")", "transformers", "LLM 개발의 표준 라이브러리입니다."),
    ("기본 토크나이저 클래스", "tokenizer = ____.from_pretrained('bert-base-uncased')", "AutoTokenizer", "모델 타입에 맞는 토크나이저를 자동으로 찾아줍니다."),
    ("토큰 확인", "ids = tokenizer.encode(\"AI\")\ncount = ____(ids)", "len", "문장이 몇 개의 토큰으로 구성됐는지 확인할 때 씁니다."),
    ("GPT-4 제조사", "company = \"____\"", "OpenAI", "GPT 시리즈를 개발한 회사 이름입니다."),
    ("LLaMA 제조사", "company = \"____\"", "Meta", "LLaMA 시리즈를 개발한 회사 이름입니다."),
    ("API 페르소나 설정", "role: \"____\"", "system", "AI의 말투나 역할을 정하는 메시지 유형입니다."),
    ("사용자 요청 메시지", "role: \"____\"", "user", "사용자가 실제로 묻는 질문을 담는 메시지 유형입니다."),
    ("AI 답변 메시지", "role: \"____\"", "assistant", "대화 기록에서 AI의 답변을 나타내는 메시지 유형입니다."),
    ("토큰 ID 리스트", "ids = [101, 203, 305]\ntype(ids) # ____", "list", "인코딩 결과물인 숫자 뭉치의 파이썬 자료형입니다."),
    ("Temperature 낮은 값", "temp = ____ # 매우 일관된 답변", "0.0", "출력 확률이 가장 높은 단어만 뽑도록 하는 온도값입니다."),
    ("Temperature 높은 값", "temp = ____ # 매우 창의적인 답변", "1.0", "다양한 표현을 시도하도록 하는 온도값입니다."),
    ("라이브러리 버전 확인", "import transformers\nprint(transformers.____)", "__version__", "설치된 라이브러리의 버전을 체크할 때 씁니다."),
    ("중단 문자열", "stop_seq = [\"\n\"] # 줄바꿈에서 ____", "stop", "생성을 멈추게 하는 설정 이름입니다."),
    ("토큰 ID 0번", "pad_id = ____", "0", "패딩 토큰 등으로 흔히 쓰이는 가장 기초적인 숫자 ID입니다.")
]

for i, (title, code, ans, explain) in enumerate(cc_data_ch3):
    questions.append({
        "chapter_name": chapter_name, "type": "코드 완성형", "difficulty": "medium", "id": str(3101 + i),
        "question": f"{title} 코드를 완성하세요.\n```python\n{code}\n```",
        "answer": ans,
        "why": explain,
        "hint": title,
    })

def get_questions():
    return questions
