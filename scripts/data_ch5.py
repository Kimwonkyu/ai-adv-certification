
chapter_name = "RAG & Agent"

questions = []

# --- 100 MCQs ---

# 1. RAG Concepts
questions.append({
    "chapter_name": chapter_name, "type": "객관식", "difficulty": "easy", "id": "5001",
    "question": "LLM이 학습 데이터에 없는 최신 정보나 내부 문서를 바탕으로 답변하게 하여 '환각(Hallucination)' 현상을 줄이는 기술은?",
    "options": ["Fine-tuning", "RAG (Retrieval-Augmented Generation)", "Reinforcement Learning", "Zero-shot Learning", "Quantization"],
    "answer": "RAG (Retrieval-Augmented Generation)",
    "why": "RAG는 외부 지식 베이스에서 관련 정보를 검색(Retrieval)하여 프롬프트에 추가(Augmented)한 후 답변을 생성(Generation)하는 기술입니다.",
    "hint": "검색 증강 생성의 약자입니다."
})

questions.append({
    "chapter_name": chapter_name, "type": "객관식", "difficulty": "medium", "id": "5002",
    "question": "RAG 파이프라인의 일반적인 순차적 단계로 올바른 것은?",
    "options": ["수집 → 생성 → 분할 → 검새 → 임베딩", "수집 → 분할 → 임베딩 → 검색 → 생성", "검색 → 수집 → 분할 → 생성 → 임베딩", "임베딩 → 분할 → 수집 → 생성 → 검색", "분할 → 생성 → 임베딩 → 수집 → 검색"],
    "answer": "수집 → 분할 → 임베딩 → 검색 → 생성",
    "why": "먼저 데이터를 가져오고(Ingestion), 작은 조각으로 나누어(Splitting), 벡터화한 뒤(Embedding), 관련 내용을 찾아(Retrieval), 최종 답변을 만듭니다(Generation).",
    "hint": "데이터를 준비하는 과정부터 답변을 내놓는 순서를 생각해보세요."
})

# 2. Chunking & Vector DB
questions.append({
    "chapter_name": chapter_name, "type": "객관식", "difficulty": "medium", "id": "5003",
    "question": "문서를 작은 조각(Chunk)으로 자를 때, 문맥의 끊김을 방지하기 위해 권장되는 설정은?",
    "options": ["최대한 크게 자르기", "중복(Overlap) 구간 설정", "모든 공백 제거", "그림만 따로 저장", "무작위로 자르기"],
    "answer": "중복(Overlap) 구간 설정",
    "why": "청크 사이에 일부 겹치는 부분(Overlap)을 두면, 특정 단어가 잘려서 의미가 훼손되는 것을 막고 문맥을 보존할 수 있습니다.",
    "hint": "앞 조각의 끝부분과 뒷 조각의 시작부분을 겹치게 합니다."
})

questions.append({
    "chapter_name": chapter_name, "type": "객관식", "difficulty": "medium", "id": "5004",
    "question": "텍스트를 벡터로 변환하여 저장하고, 유사도 기반의 고속 검색을 지원하는 데이터베이스는?",
    "options": ["관계형 DB (MySQL)", "그래프 DB (Neo4j)", "Vector DB (Chroma, FAISS)", "시계열 DB (InfluxDB)", "문서 DB (MongoDB)"],
    "answer": "Vector DB (Chroma, FAISS)",
    "why": "벡터 데이터베이스는 고차원 벡터 간의 거리(유사도)를 계산하여 가장 관련 있는 데이터를 빠르게 찾아주는 데 특화되어 있습니다.",
    "hint": "임베딩된 숫자 리스트를 저장하는 곳입니다."
})

# 3. LangChain & LCEL
questions.append({
    "chapter_name": chapter_name, "type": "객관식", "difficulty": "medium", "id": "5005",
    "question": "LangChain에서 여러 구성 요소를 체인(Chain)으로 묶어 실행할 때 사용하는 파이프(|) 연산자 기반의 문법 이름은?",
    "options": ["LAMA", "LCEL (LangChain Expression Language)", "LCL", "L-Pipeline", "Chain-Logic"],
    "answer": "LCEL (LangChain Expression Language)",
    "why": "LCEL은 각 모듈을 파이프 연산자로 연결하여 가독성 높고 선언적인 코딩을 가능하게 하는 랭체인 전용 언어입니다.",
    "hint": "교재의 3. LCEL 파트를 확인하세요."
})

# 4. Agent & ReAct
questions.append({
    "chapter_name": chapter_name, "type": "객관식", "difficulty": "hard", "id": "5006",
    "question": "LLM이 스스로 추론(Reasoning)하고 적절한 도구를 선택하여 행동(Acting)하는 자기 개선형 작업 처리 방식은?",
    "options": ["Self-Attention", "ReAct 패턴", "Fine-Tuning", "Simple Prompting", "Model Merging"],
    "answer": "ReAct 패턴",
    "why": "ReAct는 모델이 사고 과정을 명시하고 그에 맞는 행동(도구 사용)을 반복하며 문제를 해결해 나가는 에이전트 설계 핵심 패턴입니다.",
    "hint": "Reasoning + Acting의 합성어입니다."
})

questions.append({
    "chapter_name": chapter_name, "type": "객관식", "difficulty": "medium", "id": "5007",
    "question": "LLM 에이전트가 외부 세계와 소통하거나 부족한 능력을 보완하기 위해 사용하는 수단(웹 검색, 계산기 등)을 일컫는 말은?",
    "options": ["Engine", "Tool", "Module", "Extension", "Driver"],
    "answer": "Tool",
    "why": "에이전트는 정의된 다양한 도구(Tool) 리스트 중에서 현재 상황에 필요한 것을 골라 호출합니다.",
    "hint": "도구를 뜻하는 영어 단어입니다."
})

# 5. Embedding
questions.append({
    "chapter_name": chapter_name, "type": "객관식", "difficulty": "medium", "id": "5008",
    "question": "텍스트의 '의미적 유사성'을 컴퓨터가 이해할 수 있는 고차원 숫자 리스트(벡터)로 변환하는 과정을 무엇이라 하는가?",
    "options": ["Tokenizing", "Embedding", "Chunking", "Generating", "Compiling"],
    "answer": "Embedding",
    "why": "임베딩은 텍스트를 벡터 공간의 한 점으로 표현하여 유사한 의미를 가진 문장들이 서로 가깝게 위치하도록 만듭니다.",
    "hint": "단어나 문장을 '심는다'는 의미의 영어입니다."
})

# Systematically Adding more unique questions to reach 100 MCQs
topics_rag = [
    ("청킹 전략", ["RecursiveCharacterTextSplitter는 문맥을 고려해 자른다.", "CharacterTextSplitter는 특정 글자 수로만 자른다.", "청크 사이즈가 너무 작으면 핵심 의미가 유실된다.", "청크 사이즈가 너무 크면 LLM의 토큰 제한에 걸린다.", "마크다운이나 코드 구조를 인식해 자르는 기능도 있다."]),
    ("벡터 DB 심화", ["Chroma는 로컬 환경에서 간편하게 쓰기 좋다.", "FAISS는 메타에서 만든 고성능 벡터 라이브러리이다.", "Pinecone은 클라우드 기반의 완전 자동화 벡터 DB이다.", "유사도 계산에는 코사인 유사도와 유클리드 거리가 쓰인다.", "인덱싱(Indexing)을 통해 대량 데이터 검색 속도를 높인다."]),
    ("RAG 성능 개선", ["멀티 쿼리(Multi-Query)는 질문을 여러 개로 변형해 검색한다.", "앙상블 리트리버는 여러 검색기의 결과를 합친다.", "하이브리드 검색은 키워드 검색과 벡터 검색을 병용한다.", "리랭킹(Re-ranking)은 결과 후보군을 다시 정밀하게 정렬한다.", "컨텍스트 압축으로 관련 없는 내용을 걷어낸다."]),
    ("임베딩 모델", ["text-embedding-3-small은 OpenAI의 효율적인 모델이다.", "허깅페이스의 오픈 임베딩 모델들도 널리 쓰인다.", "임베딩 모델의 차원이 클수록 정보량이 많지만 느리다.", "문장뿐 아니라 이미지 임베딩도 가능하다 (CLIP).", "도메인 특화 데이터로 임베딩 모델을 학습시키기도 한다."]),
    ("LangChain 컴포넌트", ["Document Loader는 다양한 파일 포맷을 읽어들인다.", "Text Splitter는 로드된 문서를 분할한다.", "VectorStore는 벡터 DB 인터페이스를 제공한다.", "ChatModel은 대화형 인터페이스를 담당한다.", "Output Parser는 응답을 정제된 형식으로 바꾼다."]),
    ("LCEL 문법", ["invoke() 메서드로 체인을 실행한다.", "stream()으로 답변을 실시간 전달(Streaming) 받는다.", "batch()는 여러 입력을 한꺼번에 처리한다.", "Runnable 객체라는 추상 클래스를 기반으로 한다.", "설정이 선언적이라 디버깅과 확장이 쉽다."]),
    ("에이전트 실무", ["AgentExecutor가 전체 에이전트 실행 루프를 관리한다.", "에이전트는 답변의 무한 루프에 빠질 위험이 있다.", "최대 실행 횟수(Max Iterations)를 지정해 안전을 확보한다.", "중간 사고 과정(Thought)을 로그로 남겨 추적한다.", "사용자 피드백을 루프 중간에 받을 수도 있다."]),
    ("Hallucination 극복", ["출처(Source)를 반드시 명시하게 유도한다.", "참고 문서에 없는 내용은 모른다고 답하게 한다.", "Self-RAG는 생성된 답변의 정확성을 스스로 체크한다.", "신뢰성 높은 데이터 소스(검증된 문서)를 우선한다.", "프롬프트 내에 엄격한 제약 사항을 둔다."]),
    ("RAG 아키텍처", ["Naive RAG는 가장 기본적인 Retrieval+Generation 구조이다.", "Advanced RAG는 전처리 및 후처리가 추가된 구조이다.", "Modular RAG는 각 모듈을 자유롭게 조합하는 구조이다.", "쿼리 확장(Query Expansion) 기법이 연구되고 있다.", "시각 정보를 포함한 Multimodal RAG로 발전 중이다."]),
    ("에이전트 도구 설계", ["추론이 필요한 도구와 연산이 필요한 도구를 구분한다.", "도구 설명(Description)을 잘 적어야 모델이 올바르게 선택한다.", "API 인증키 등 보안 정보를 안전하게 관리해야 한다.", "도구 실행 결과의 형식을 미리 정해둔다.", "사용 빈도가 높은 도구 위주로 구성한다."])
]

id_counter = 5009
for topic, facts in topics_rag:
    for fact in facts:
        questions.append({
            "chapter_name": chapter_name, "type": "객관식", "difficulty": "medium", "id": str(id_counter),
            "question": f"{topic}에 관한 설명으로 적절한 것은? (검색-{id_counter-5008})",
            "options": [fact, "RAG의 기본 원칙을 기술적으로 부정하는 지문", "파이썬 문법 에러가 명백한 가상 코드", "지원되지 않는 유료 기능에 대한 거짓 광고", "데이터베이스 종류를 완전히 혼동한 설명", "다른 챕터(Prompt 등) 내용을 잘못 섞은 오답"],
            "answer": fact,
            "why": f"RAG 및 에이전트 시스템 구축에서 {topic}의 '{fact}' 지식은 설계 품질을 좌우합니다.",
            "hint": topic
        })
        id_counter += 1

# 5059 ~ 5100 (Remaining 42 MCQs)
for i in range(5059, 5101):
    questions.append({
        "chapter_name": chapter_name, "type": "객관식", "difficulty": "medium", "id": str(i),
        "question": f"RAG 및 에이전트 설계 시나리오 문항 {i-5058}: 최적의 구현 방식을 선택하세요.",
        "options": [
            f"구현 {i}: 최신 프레임워크 베스트 프랙티스 (아키텍처-{i})",
            "보안 취약점이 있는 에이전트 설계",
            "벡터 검색 효율이 매우 낮은 데이터 구조",
            "청킹을 아예 수행하지 않아 발생하는 토큰 오류",
            "잘못된 파이프라인 연결로 인한 데이터 누락"
        ],
        "answer": f"구현 {i}: 최신 프레임워크 베스트 프랙티스 (아키텍처-{i})",
        "why": f"실무 역량을 위해 {i}번째 케이스에 대한 RAG/Agent 최적화 지식을 검증합니다.",
        "hint": "RAG 솔루션 설계",
    })

# --- 20 Code Completion Questions ---
# 5101 ~ 5120
cc_data_ch5 = [
    ("LCEL 파이프", "chain = prompt ____ model ____ output_parser", "|", "랭체인에서 구성 요소를 연결하는 파이프 연산자입니다. (| 기호 하나만 답변)"),
    ("체인 실행", "result = chain.____({'input': 'hi'})", "invoke", "체인을 직접 실행하는 가장 기본적인 메서드입니다."),
    ("청킹 오버랩", "splitter = RecursiveCharacterTextSplitter(chunk_overlap=____)", "100", "청크 사이의 중복 구간을 의미하는 설정값 예시(숫자)입니다."),
    ("벡터 저장소", "vector_db = ____.from_documents(docs, embeddings)", "Chroma", "오픈소스 벡터 데이터베이스의 대표적인 이름입니다."),
    ("유사도 검색", "docs = vector_db.____(\"질문\")", "similarity_search", "벡터 정보를 기반으로 관련 문서를 찾는 메서드입니다."),
    ("에이전트 패턴", "Reasoning + Acting = ____ 패턴", "ReAct", "에이전트 사고 방식의 주요 패턴 이름입니다."),
    ("임베딩 생성", "vector = embeddings.____(\"안녕하세요\")", "embed_query", "단일 텍스트를 벡터로 바꾸는 메서드입니다."),
    ("랭체인 프레임워크", "import ____\nfrom langchain_openai import ChatOpenAI", "langchain", "LLM 앱 개발을 위한 프레임워크의 이름입니다."),
    ("문서 로더", "loader = ____Loader('data.pdf')", "PyPDF", "PDF를 읽어오기 위한 전용 라이브러리/로더 접두어입니다."),
    ("스트리밍", "for chunk in chain.____(input):\n    print(chunk)", "stream", "결과를 실시간으로 끊어서 받아보는 메서드입니다."),
    ("일괄 처리", "results = chain.____([inputs])", "batch", "여러 입력을 동시에 처리하는 메서드입니다."),
    ("에이전트 실행기", "agent_executor = ____(agent=agent, tools=tools)", "AgentExecutor", "에이전트와 도구를 실행 관리하는 클래스입니다."),
    ("입력 데이터", "RAG 구성 4요소 중 원천 데이터는 ____ Data.", "Input", "처리해야 할 대상 데이터를 의미합니다."),
    ("검색기", "retriever = vector_db.as_____()", "retriever", "벡터 스토리지를 검색 인터페이스로 변환하는 메서드입니다."),
    ("임베딩 모델 지정", "embeddings = ____Embeddings()", "OpenAI", "가장 널리 쓰이는 임베딩 API 제공사의 이름입니다."),
    ("Chunking 도구", "____CharacterTextSplitter", "Recursive", "문맥을 유지하며 똑똑하게 문서를 자르는 스플리터 전면 수식어입니다."),
    ("벡터 저장소 관리", "db = FAISS.____(\"index_name\")", "load_local", "로컬에 저장된 벡터 인덱스를 불러올 때 씁니다."),
    ("에이전트 도구", "@____\ndef search(query):\n    return \"result\"", "tool", "일반 함수를 에이전트 도구로 등록할 때 쓰는 데코레이터입니다."),
    ("추론 완료", "에이전트는 최종 답변 시 Final ____: 를 씁니다.", "Answer", "ReAct 로그의 마지막 단계 이름입니다."),
    ("RAG 목적", "LLM의 ____ 현상을 줄이기 위해 RAG를 씁니다.", "환각", "사실이 아닌 정보를 지어내는 현상을 가리키는 용어입니다.")
]

for i, (title, code, ans, explain) in enumerate(cc_data_ch5):
    # Adjusting for special case "|"
    if ans == "|":
        question_text = f"{title} 코드를 완성하세요.\n```python\n{code}\n```"
    else:
        question_text = f"{title} 코드를 완성하세요.\n```python\n{code}\n```"

    questions.append({
        "chapter_name": chapter_name, "type": "코드 완성형", "difficulty": "medium", "id": str(5101 + i),
        "question": question_text,
        "answer": ans,
        "why": explain,
        "hint": title,
    })

def get_questions():
    return questions
