
chapter_name = "RAG & Agent"

questions = []

# 1. RAG Concepts (20 MCQs)
for i in range(1, 21):
    q = {
        "chapter_name": chapter_name,
        "type": "객관식",
        "question": f"RAG(Retrieval-Augmented Generation)의 개념과 필요성 {i}",
        "options": [
            "RAG는 LLM을 처음부터 다시 학습시키는 방법이다.",
            "RAG를 사용하면 Hallucination(환각)을 줄일 수 있다.",
            "RAG는 외부 데이터 없이 모델 내부 지식만 사용한다.",
            "RAG는 항상 파인튜닝보다 비용이 많이 든다.",
            "RAG는 이미지 생성 전용 기술이다."
        ],
        "answer": "RAG를 사용하면 Hallucination(환각)을 줄일 수 있다.",
        "why": "RAG는 신뢰할 수 있는 외부 지식을 검색하여 제공하므로, 모델이 사실이 아닌 내용을 지어내는 환각 현상을 완화합니다.",
        "hint": "근거 있는 답변",
        "difficulty": "easy",
        "id": f"50{i:02d}"
    }
    questions.append(q)

# 2. Pipeline & Vector DB (20 MCQs)
for i in range(21, 41):
    q = {
        "chapter_name": chapter_name,
        "type": "객관식",
        "question": f"RAG 파이프라인 및 벡터 DB 구조 {i}",
        "options": [
            "Vector DB는 텍스트를 이미지로 저장한다.",
            "Chunking은 문서를 통째로 저장하는 것이다.",
            "Embedding은 텍스트를 벡터(수치)로 변환하는 과정이다.",
            "Retrieval 단계에서는 가장 관련 없는 문서를 찾는다.",
            "Ingestion은 답변을 생성하는 단계다."
        ],
        "answer": "Embedding은 텍스트를 벡터(수치)로 변환하는 과정이다.",
        "why": "임베딩 모델을 통해 텍스트를 고차원의 벡터 공간에 매핑하여 의미적 유사도를 계산할 수 있게 합니다.",
        "hint": "Text to Vector",
        "difficulty": "medium",
        "id": f"50{i:02d}"
    }
    questions.append(q)

# 3. LangChain Basics (20 MCQs)
for i in range(41, 61):
    q = {
        "chapter_name": chapter_name,
        "type": "객관식",
        "question": f"LangChain 프레임워크 기초 {i}",
        "options": [
            "LangChain은 오직 Python만 지원한다.",
            "Chain은 여러 컴포넌트를 연결하는 핵심 개념이다.",
            "LCEL은 복잡한 클래스 상속을 강제한다.",
            "LangChain은 LLM을 직접 학습시키는 도구다.",
            "PromptTemplate은 사용할 수 없다."
        ],
        "answer": "Chain은 여러 컴포넌트를 연결하는 핵심 개념이다.",
        "why": "LangChain은 프롬프트, 모델, 파서 등을 체인(Chain)으로 엮어 복잡한 애플리케이션을 구축하게 해줍니다.",
        "hint": "사슬(Chain)",
        "difficulty": "medium",
        "id": f"50{i:02d}"
    }
    questions.append(q)

# 4. Agents & Tools (20 MCQs)
for i in range(61, 81):
    q = {
        "chapter_name": chapter_name,
        "type": "객관식",
        "question": f"AI 에이전트와 도구 활용 {i}",
        "options": [
            "에이전트는 정해진 각본대로만 움직인다.",
            "ReAct는 추론(Reasoning)과 행동(Acting)을 결합한 패턴이다.",
            "Tool은 에이전트가 사용할 수 없는 외부 기능이다.",
            "에이전트는 스스로 계획(Planning)하지 못한다.",
            "웹 검색은 Tool로 사용할 수 없다."
        ],
        "answer": "ReAct는 추론(Reasoning)과 행동(Acting)을 결합한 패턴이다.",
        "why": "ReAct 프롬프팅을 통해 에이전트는 상황을 판단하고 필요한 도구를 선택하여 실행한 뒤 결과를 관찰합니다.",
        "hint": "Reason + Act",
        "difficulty": "hard",
        "id": f"50{i:02d}"
    }
    questions.append(q)

# 5. Advanced Concepts (20 MCQs)
for i in range(81, 101):
    q = {
        "chapter_name": chapter_name,
        "type": "객관식",
        "question": f"RAG 심화 및 최적화 {i}",
        "options": [
            "Self-Querying은 질문을 스스로 수정하지 않는다.",
            "Multi-Vector Retriever는 문서 전체만 저장한다.",
            "Hybrid Search는 키워드 검색과 시맨틱 검색을 결합한다.",
            "RAG는 문맥 길이(Context Window) 제한이 없다.",
            "Parent Document Retriever는 자식 문서만 검색한다."
        ],
        "answer": "Hybrid Search는 키워드 검색과 시맨틱 검색을 결합한다.",
        "why": "BM25(키워드)와 Vector(의미) 검색을 결합하여 검색 정확도(Recall/Precision)를 높이는 방식입니다.",
        "hint": "섞어서 쓴다(Hybrid)",
        "difficulty": "hard",
        "id": f"50{i:02d}"
    }
    questions.append(q)

# 6. Code Completion (20 CC)
for i in range(1, 21):
    q = {
        "chapter_name": chapter_name,
        "type": "코드 완성형",
        "question": f"RAG/LangChain 코드를 완성하세요 (문제 {i})",
        "answer": "Chain",
        "why": "LangChain의 기본 구성 단위",
        "hint": "LangChain Class",
        "difficulty": "medium",
        "id": f"51{i:02d}"
    }
    if i % 4 == 0:
        q['question'] = "문서를 로드하세요.\n```python\nfrom langchain_community.document_loaders import TextLoader\nloader = TextLoader('data.txt')\ndocs = loader.____()\n```"
        q['answer'] = "load"
        q['why'] = "문서 로드 메서드는 load()입니다."
    elif i % 4 == 1:
        q['question'] = "텍스트를 분할하세요.\n```python\ntext_splitter = CharacterTextSplitter(chunk_size=1000)\ntexts = text_splitter.____(docs)\n```"
        q['answer'] = "split_documents"
        q['why'] = "문서 분할 메서드입니다."

    questions.append(q)

def get_questions():
    return questions
