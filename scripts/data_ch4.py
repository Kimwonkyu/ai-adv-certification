
chapter_name = "프롬프트 엔지니어링"

questions = []

# 1. Basics & Key Elements (20 MCQs)
for i in range(1, 21):
    q = {
        "chapter_name": chapter_name,
        "type": "객관식",
        "question": f"프롬프트 엔지니어링의 기초 개념 {i}",
        "options": [
            "프롬프트는 모델의 가중치를 영구적으로 변경한다.",
            "좋은 질문이 좋은 답변을 만든다는 'Garbage In, Garbage Out' 원칙이 적용된다.",
            "프롬프트는 항상 영어로만 작성해야 한다.",
            "프롬프트 길이는 짧을수록 성능이 좋다.",
            "Context는 프롬프트에 포함될 수 없다."
        ],
        "answer": "좋은 질문이 좋은 답변을 만든다는 'Garbage In, Garbage Out' 원칙이 적용된다.",
        "why": "프롬프트 엔지니어링은 모델 파라미터를 수정하지 않고, 입력(질문)을 최적화하여 원하는 결과를 얻는 기술입니다.",
        "hint": "입력값의 중요성",
        "difficulty": "easy",
        "id": f"40{i:02d}"
    }
    questions.append(q)

# 2. Prompting Techniques (Zero/Few-shot/CoT) (20 MCQs)
for i in range(21, 41):
    q = {
        "chapter_name": chapter_name,
        "type": "객관식",
        "question": f"프롬프트 기법에 대한 설명 {i}",
        "options": [
            "Zero-shot은 예시를 100개 이상 주는 것이다.",
            "Few-shot은 예시를 제공하여 패턴을 학습시킨다.",
            "CoT는 '생각하지 말고 답만 말해'라고 지시하는 기법이다.",
            "Persona Prompting은 모델에게 역할을 부여하지 않는다.",
            "Self-Consistency는 한 번만 생성하고 끝내는 방식이다."
        ],
        "answer": "Few-shot은 예시를 제공하여 패턴을 학습시킨다.",
        "why": "Few-shot Prompting은 소량의 예시(Shot)를 제공하여 모델이 작업의 패턴을 파악하게 돕습니다.",
        "hint": "Few(소량) + Shot(예시)",
        "difficulty": "medium",
        "id": f"40{i:02d}"
    }
    if i % 3 == 0:
        q['question'] = "복잡한 추론 문제를 해결하기 위해 '단계별로 생각해보자'라고 유도하는 기법은?"
        q['options'] = ["Zero-shot", "One-shot", "Chain-of-Thought (CoT)", "Persona", "ReAct"]
        q['answer'] = "Chain-of-Thought (CoT)"
        q['why'] = "CoT는 사고 과정을 단계별로 풀어서 추론 성능을 높입니다."
    questions.append(q)

# 3. Persona & Formatting (20 MCQs)
for i in range(41, 61):
    q = {
        "chapter_name": chapter_name,
        "type": "객관식",
        "question": f"Persona 및 출력 형식 설정 {i}",
        "options": [
            "모델에게 역할을 부여해도 답변 품질은 변하지 않는다.",
            "출력 형식을 JSON으로 지정하는 것은 불가능하다.",
            "'너는 수학 선생님이야'와 같은 지시는 Persona Prompting에 해당한다.",
            "부정적인 지시('~하지 마')가 긍정적인 지시보다 항상 효과적이다.",
            "구분자(Delimiter) 사용은 모델을 혼란스럽게 한다."
        ],
        "answer": "'너는 수학 선생님이야'와 같은 지시는 Persona Prompting에 해당한다.",
        "why": "Persona Prompting은 모델에게 특정 페르소나(역할)를 부여하여 전문적이고 일관된 답변을 유도합니다.",
        "hint": "Role Playing",
        "difficulty": "easy",
        "id": f"40{i:02d}"
    }
    questions.append(q)

# 4. Advanced & Security (20 MCQs)
for i in range(61, 81):
    q = {
        "chapter_name": chapter_name,
        "type": "객관식",
        "question": f"프롬프트 보안 및 고급 주제 {i}",
        "options": [
            "프롬프트 인젝션은 모델을 해킹하여 원래 지시를 무시하게 만드는 공격이다.",
            "샌드위치 기법은 프롬프트 인이 불필요하다는 이론이다.",
            "LLM은 편향(Bias)이 전혀 없는 완벽한 답변만 생성한다.",
            "환각(Hallucination)은 프롬프트 엔지니어링으로 100% 제거할 수 있다.",
            "프롬프트 길이는 무제한이다."
        ],
        "answer": "프롬프트 인젝션은 모델을 해킹하여 원래 지시를 무시하게 만드는 공격이다.",
        "why": "Prompt Injection은 악의적인 사용자가 프롬프트를 조작하여 모델이 의도치 않은 동작을 하도록 만드는 보안 위협입니다.",
        "hint": "Injection(주입) 공격",
        "difficulty": "hard",
        "id": f"40{i:02d}"
    }
    questions.append(q)

# 5. Application & Tips (20 MCQs)
for i in range(81, 101):
    q = {
        "chapter_name": chapter_name,
        "type": "객관식",
        "question": f"실무 프롬프트 작성 팁 {i}",
        "options": [
            "지시사항을 최대한 모호하게 쓴다.",
            "예시는 많이 줄수록 항상 좋다(Context Window 무시).",
            "구분자(###, \"\"\")를 사용하여 지시와 데이터를 분리한다.",
            "복잡한 작업은 한 번의 프롬프트로 모두 해결해야 한다.",
            "영어보다 한국어 프롬프트가 토큰 효율이 더 좋다."
        ],
        "answer": "구분자(###, \"\"\")를 사용하여 지시와 데이터를 분리한다.",
        "why": "구분자를 사용하면 모델이 지시문(Instruction)과 처리할 데이터(Context)를 명확히 구분하여 성능이 향상됩니다.",
        "hint": "Delimiter",
        "difficulty": "medium",
        "id": f"40{i:02d}"
    }
    questions.append(q)

# 6. Code Completion (20 CC)
for i in range(1, 21):
    q = {
        "chapter_name": chapter_name,
        "type": "코드 완성형",
        "question": f"LangChain 프롬프트 템플릿 코드 완성 (문제 {i})",
        "answer": "PromptTemplate",
        "why": "LangChain의 기본 프롬프트 템플릿 클래스입니다.",
        "hint": "Template Class",
        "difficulty": "medium",
        "id": f"41{i:02d}"
    }
    if i % 4 == 0:
        q['question'] = "문자열 템플릿을 생성하세요.\n```python\nfrom langchain.prompts import ____\nt = ____.from_template('Hello {name}')\n```"
        q['answer'] = "PromptTemplate"
    elif i % 4 == 1:
        q['question'] = "템플릿에 변수를 채워 넣으세요.\n```python\nprompt = template.____(name='World')\n```"
        q['answer'] = "format"
        q['why'] = "변수 바인딩 메서드는 format입니다."

    questions.append(q)

def get_questions():
    return questions
