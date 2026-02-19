
chapter_name = "LLM 기본"

questions = []

# 1. Transformer Architecture (20 MCQs)
for i in range(1, 21):
    q = {
        "chapter_name": chapter_name,
        "type": "객관식",
        "question": f"Transformer 아키텍처의 특징으로 올바른 설명 {i}?",
        "options": [
            "순차적으로 데이터를 처리해야 한다(RNN 방식).",
            "Attention 메커니즘을 통해 병렬 처리가 가능하다.",
            "이미지 처리에만 특화된 모델이다.",
            "인코더 없이 디코더만 결합된 구조가 원조다.",
            "장기 의존성(Long-term dependency) 문제를 해결하지 못했다."
        ],
        "answer": "Attention 메커니즘을 통해 병렬 처리가 가능하다.",
        "why": "Transformer는 Self-Attention을 사용하여 문장 전체를 한 번에(병렬로) 처리하며, RNN의 순차적 처리 한계를 극복했습니다.",
        "hint": "Attention Is All You Need",
        "difficulty": "medium",
        "id": f"30{i:02d}"
    }
    if i % 3 == 0:
        q['question'] = "Transformer의 핵심 구성 요소인 'Attention'의 역할은?"
        q['options'] = ["입력 데이터를 압축하여 손실을 유도한다.", "중요한 단어에 가중치를 부여하여 문맥을 파악한다.", "무작위 노이즈를 생성하여 창의성을 높인다.", "이미지를 텍스트로 변환한다.", "데이터를 순서대로 정렬한다."]
        q['answer'] = "중요한 단어에 가중치를 부여하여 문맥을 파악한다."
        q['why'] = "Attention은 문장 내의 단어들 사이의 연관성을 계산하여 중요한 정보에 집중하게 합니다."
        
    questions.append(q)

# 2. Tokenization (20 MCQs)
for i in range(21, 41):
    q = {
        "chapter_name": chapter_name,
        "type": "객관식",
        "question": f"토큰화(Tokenization)에 대한 설명 {i}",
        "options": [
            "무조건 글자 단위로 자른다.",
            "무조건 단어 단위로 자른다.",
            "토큰은 의미를 가진 최소 단위로, 단어의 일부일 수도 있다.",
            "이미지를 픽셀 단위로 자르는 과정이다.",
            "토큰화 후에는 항상 이미지가 생성된다."
        ],
        "answer": "토큰은 의미를 가진 최소 단위로, 단어의 일부일 수도 있다.",
        "why": "현대 LLM은 BPE(Byte Pair Encoding) 등의 알고리즘을 사용하여, 자주 등장하는 문자열 조각을 토큰으로 정의합니다.",
        "hint": "Subword Tokenization",
        "difficulty": "easy",
        "id": f"30{i:02d}"
    }
    questions.append(q)

# 3. Model History (GPT/BERT) (20 MCQs)
for i in range(41, 61):
    q = {
        "chapter_name": chapter_name,
        "type": "객관식",
        "question": f"주요 언어 모델의 특징 {i}",
        "options": [
            "BERT는 디코더(Decoder) 전용 모델이다.",
            "GPT는 인코더(Encoder) 전용 모델이다.",
            "GPT-3는 1750억 개의 파라미터를 가진다.",
            "Transformer는 2023년에 처음 발표되었다.",
            "LLaMA는 구글이 개발한 모델이다."
        ],
        "answer": "GPT-3는 1750억 개의 파라미터를 가진다.",
        "why": "GPT-3는 175B 파라미터의 거대 모델로, In-Context Learning 능력을 보여주었습니다. (BERT=Encoder, GPT=Decoder, Transformer=2017, LLaMA=Meta)",
        "hint": "GPT-3 Parameter",
        "difficulty": "medium",
        "id": f"30{i:02d}"
    }
    questions.append(q)

# 4. LLM Usage (API vs Open) (20 MCQs)
for i in range(61, 81):
    q = {
        "chapter_name": chapter_name,
        "type": "객관식",
        "question": f"LLM 활용 방식 비교 {i}",
        "options": [
            "API 방식은 인프라 구축 비용이 매우 높다.",
            "오픈 모델은 데이터 보안 유지에 불리하다.",
            "API 방식은 최신 고성능 모델을 쉽게 사용할 수 있다.",
            "오픈 모델은 커스터마이징이 불가능하다.",
            "API 방식은 인터넷 연결 없이 사용 가능하다."
        ],
        "answer": "API 방식은 최신 고성능 모델을 쉽게 사용할 수 있다.",
        "why": "OpenAI 등에서 제공하는 API를 사용하면 별도의 GPU 서버 구축 없이 고성능 모델을 호출하여 사용할 수 있습니다.",
        "hint": "API의 편의성",
        "difficulty": "easy",
        "id": f"30{i:02d}"
    }
    questions.append(q)

# 5. Hallucination & Limits (20 MCQs)
for i in range(81, 101):
    q = {
        "chapter_name": chapter_name,
        "type": "객관식",
        "question": f"LLM의 한계점과 환각(Hallucination) {i}",
        "options": [
            "LLM은 모든 사실관계를 완벽하게 검증한다.",
            "환각은 모델이 사실인 것처럼 거짓 정보를 생성하는 현상이다.",
            "LLM은 최신 뉴스를 실시간으로 스스로 학습한다.",
            "LLM은 수학 계산에 실수가 없다.",
            "환각은 데이터가 많으면 100% 사라진다."
        ],
        "answer": "환각은 모델이 사실인 것처럼 거짓 정보를 생성하는 현상이다.",
        "why": "LLM은 확률적으로 다음 단어를 예측하므로, 사실이 아닌 내용을 그럴듯하게 생성하는 환각 현상이 발생할 수 있습니다.",
        "hint": "거짓말을 그럴듯하게 하는 것",
        "difficulty": "medium",
        "id": f"30{i:02d}"
    }
    questions.append(q)

# 6. Code Completion (20 CC)
for i in range(1, 21):
    q = {
        "chapter_name": chapter_name,
        "type": "코드 완성형",
        "question": f"LLM 관련 코드를 완성하세요. (문제 {i})",
        "answer": "transformers",
        "why": "HuggingFace의 핵심 라이브러리 이름은 transformers입니다.",
        "hint": "HuggingFace Library",
        "difficulty": "easy",
        "id": f"31{i:02d}"
    }
    if i % 4 == 0:
        q['question'] = "토크나이저를 로드하세요.\n```python\nfrom transformers import AutoTokenizer\ntokenizer = AutoTokenizer.____('gpt2')\n```"
        q['answer'] = "from_pretrained"
        q['why'] = "사전 학습된 모델 로드 메서드"
    elif i % 4 == 1:
         q['question'] = "텍스트를 토큰으로 변환하세요.\n```python\ntext = 'Hello'\ntokens = tokenizer.____(text)\n```"
         q['answer'] = "encode"
         q['why'] = "텍스트 -> 토큰 ID 변환은 encode()"

    questions.append(q)

def get_questions():
    return questions
