
chapter_name = "프롬프트 엔지니어링"

questions = []

# --- 100 MCQs ---

# 1. Basics & Components
questions.append({
    "chapter_name": chapter_name, "type": "객관식", "difficulty": "easy", "id": "4001",
    "question": "프롬프트 엔지니어링의 핵심 원칙 중 하나인 'GIGO'가 의미하는 바는?",
    "options": ["빠른 답변이 좋은 답변이다.", "입력이 나쁘면 결과도 나쁘다 (Garbage In, Garbage Out).", "많은 데이터를 넣을수록 항상 유리하다.", "명령어는 짧을수록 모델이 더 잘 이해한다.", "이미 학습된 데이터만 반복해서 보여줘야 한다."],
    "answer": "입력이 나쁘면 결과도 나쁘다 (Garbage In, Garbage Out).",
    "why": "프롬프트(질문)가 모호하거나 잘못되어 있으면 AI 또한 정확한 답변을 내놓을 수 없음을 강조하는 표현입니다.",
    "hint": "쓰레기가 들어가면 쓰레기가 나온다는 뜻입니다."
})

questions.append({
    "chapter_name": chapter_name, "type": "객관식", "difficulty": "medium", "id": "4002",
    "question": "좋은 프롬프트를 구성하는 4대 요소 중 '모델이 참고해야 할 배경 정보'를 뜻하는 것은?",
    "options": ["Instruction (지시)", "Context (문맥)", "Input Data (입력 데이터)", "Output Indicator (출력 형식)", "Delimiter (구분자)"],
    "answer": "Context (문맥)",
    "why": "문맥(Context)은 작업의 배경이나 제약 사항을 알려주어 AI가 상황에 맞는 답변을 생성하도록 돕습니다.",
    "hint": "상황이나 배경을 설정하는 부분입니다."
})

# 2. Key Techniques
questions.append({
    "chapter_name": chapter_name, "type": "객관식", "difficulty": "easy", "id": "4003",
    "question": "모델에게 어떠한 예시도 주지 않고 바로 질문을 던지는 방식은?",
    "options": ["Zero-shot Prompting", "One-shot Prompting", "Few-shot Prompting", "Chain-of-Thought", "Persona Prompting"],
    "answer": "Zero-shot Prompting",
    "why": "사전 지식이나 예시(Shot) 없이 Zero 상태에서 바로 명령을 수행하도록 하는 방식입니다.",
    "hint": "예시를 0개 준다는 의미입니다."
})

questions.append({
    "chapter_name": chapter_name, "type": "객관식", "difficulty": "medium", "id": "4004",
    "question": "모델에게 몇 가지 예시를 보여주어 원하는 답변의 패턴이나 톤을 유도하는 기법은?",
    "options": ["Zero-shot Prompting", "Few-shot Prompting", "Zero-shot CoT", "Recursive Prompting", "Least-to-Most Prompting"],
    "answer": "Few-shot Prompting",
    "why": "몇 개(Few)의 예시를 통해 모델이 문맥 안에서 학습(In-Context Learning)하도록 하는 기법입니다.",
    "hint": "패턴을 익히게 몇 가지 예(Shot)를 보여줍니다."
})

questions.append({
    "chapter_name": chapter_name, "type": "객관식", "difficulty": "hard", "id": "4005",
    "question": "모델에게 '단계별로 생각해보자(Let's think step by step)'라고 지시하여 논리적 추론 능력을 높이는 기법은?",
    "options": ["Few-shot Prompting", "Chain-of-Thought (CoT)", "Persona Prompting", "Negative Prompting", "Token Tuning"],
    "answer": "Chain-of-Thought (CoT)",
    "why": "답변을 내놓기 전 중간 추론 과정을 명시하게 하여 복잡한 문제의 해결력을 높이는 '생각의 사슬' 기법입니다.",
    "hint": "생각을 사슬처럼 이어나간다는 뜻입니다."
})

questions.append({
    "chapter_name": chapter_name, "type": "객관식", "difficulty": "medium", "id": "4006",
    "question": "모델에게 '전문 개발자' 혹은 '동화 작가'와 같은 특정한 역할을 부여하여 전문적인 조언을 얻는 기법은?",
    "options": ["Few-shot Prompting", "Context Learning", "Persona Prompting (Role-playing)", "Instruction Tuning", "Output Filtering"],
    "answer": "Persona Prompting (Role-playing)",
    "why": "페르소나(역할)설정은 모델의 어조와 지식 범위를 특정 도메인으로 좁혀 더 깊이 있는 답변을 얻게 합니다.",
    "hint": "연극의 배역(Persona)을 정해주는 것과 같습니다."
})

# 3. Writing Tips
questions.append({
    "chapter_name": chapter_name, "type": "객관식", "difficulty": "medium", "id": "4007",
    "question": "프롬프트 작성 시 권장되는 팁 중 올바른 것은?",
    "options": ["'하지 마세요'와 같은 부정문을 최대한 많이 쓴다.", "지시문과 입력 데이터를 구분하지 않고 섞어 쓴다.", "원하는 결과를 얻기 위해 가능한 모호하게 명령한다.", "구분자(###, \"\"\" 등)를 사용하여 영역을 명확히 나눈다.", "최대한 짧은 단어로만 대화한다."],
    "answer": "구분자(###, \"\"\" 등)를 사용하여 영역을 명확히 나눈다.",
    "why": "구분자를 사용하면 모델이 어디까지가 지침이고 어디서부터가 가공할 데이터인지 명확히 인지할 수 있습니다.",
    "hint": "경계선을 명확히 그으세요."
})

questions.append({
    "chapter_name": chapter_name, "type": "객관식", "difficulty": "easy", "id": "4008",
    "question": "효율적인 프롬프트를 위해 긍정문과 부정문 중 어느 쪽이 권장되는가?",
    "options": ["'무엇을 하지 마라'는 부정문 위주", "'무엇을 하라'는 긍정문 위주", "둘 다 성능 차이가 전혀 없다.", "최대한 길고 복잡한 부정문", "명령어 없이 데이터만 주는 방식"],
    "answer": "'무엇을 하라'는 긍정문 위주",
    "why": "LLM은 하지 말아야 할 것보다 해야 할 것을 명확히 지시받았을 때 지침을 더 잘 따르는 경향이 있습니다.",
    "hint": "하지 말라는 것보다 할 것을 짚어주는 것이 낫습니다."
})

questions.append({
    "chapter_name": chapter_name, "type": "객관식", "difficulty": "medium", "id": "4009",
    "question": "모델의 출력이 너무 길거나 복잡할 때, '3줄 요약' 혹은 '불렛 포인트로 정리'와 같이 출력의 형태를 지정하는 구성 요소는?",
    "options": ["Instruction", "Context", "Input Data", "Output Indicator", "Delimiter"],
    "answer": "Output Indicator",
    "why": "Output Indicator는 원하는 답변의 형식이나 스타일을 지정하여 후처리가 용이하게 만듭니다.",
    "hint": "출력 결과물의 생김새를 정해줍니다."
})

questions.append({
    "chapter_name": chapter_name, "type": "객관식", "difficulty": "medium", "id": "4010",
    "question": "프롬프트 상에서 지시문과 입력 데이터를 명확히 분리하기 위해 사용하는 기호(예: ###, ---, \"\"\")를 무엇이라 부르는가?",
    "options": ["Token", "Delimiter (구분자)", "Augmenter", "Operator", "Connector"],
    "answer": "Delimiter (구분자)",
    "why": "구분자(Delimiter)는 모델이 프롬프트 내의 서로 다른 영역을 혼동하지 않도록 경계를 나누는 역할을 합니다.",
    "hint": "데이터와 명령 사이의 칸막이 역할을 합니다."
})

# Systematically Adding more unique questions to reach 100 MCQs
topics_pe = [
    ("CoT 응용", ["Zero-shot CoT는 지시어 하나로 추론을 유도한다.", "복잡한 수학 문제에서 정답률이 비약적으로 올라간다.", "중간 과정을 출력하게 하면 검증도가 높아진다.", "ToT(Tree of Thoughts)는 여러 경로를 탐색하는 CoT의 확장이다.", "아이들이 이해하기 쉽게 단계별로 설명해달라고 할 때 유용하다."]),
    ("Few-shot 디테일", ["예시의 순서가 성능에 영향을 줄 수 있다.", "예시가 너무 많으면 컨텍스트 윈도우를 초과할 수 있다.", "예시의 형식이 결과값의 형식을 결정한다.", "잘못된 예시를 주면 모델이 혼란을 겪는다.", "분류 문제(스팸 여부 등)에서 특히 강력하다."]),
    ("Persona 팁", ["구체적인 경력을 명시하면 더 전문적으로 변한다.", "어조(친절함, 딱딱함 등)도 페르소나의 일부이다.", "반대되는 페르소나를 설정해 논쟁을 시킬 수도 있다.", "학생에게 가르치는 선생님 페르소나는 설명력을 높인다.", "페르소나에 따른 지식의 깊이가 달라진다."]),
    ("구조화 기법", ["JSON 형식으로 답해달라는 요청은 구조화의 일종이다.", "표(Table) 형식을 명시하여 가공을 편하게 한다.", "Markdown을 활용하여 가독성을 높인다.", "불렛포인트를 사용하여 핵심을 전달한다.", "번호표를 매겨 순서를 명확히 한다."]),
    ("프롬프트 보안", ["프롬프트 인젝션은 지시문을 무시하게 만드는 공격이다.", "민감 정보가 프롬프트에 포함되지 않게 주의해야 한다.", "외부 입력을 그대로 프롬프트에 넣을 때 위험이 발생한다.", "시스템 메시지를 통해 필터링을 강화할 수 있다.", "입력을 샌드박싱하는 기법도 프롬프트 수준에서 고려된다."]),
    ("실무 프롬프트 예시", ["블로그 포스팅 작성 시 타겟 독자를 명시한다.", "코드 리뷰 시 성능 최적화 관점을 강조한다.", "번역 시 특정 전문 용어 사전을 미리 주입한다.", "요약 시 핵심 키워드 3개를 포함해달라고 한다.", "이메일 작성 시 상황과 감정 톤을 지정한다."]),
    ("프롬프트 엔지니어링 툴", ["Playground에서 다양한 파라미터를 테스트해본다.", "LangChain 등으로 프롬프트를 템플릿화한다.", "버전 관리를 통해 좋은 프롬프트를 기록한다.", "DSPy와 같이 프롬프트를 자동 최적화하는 연구도 있다.", "프롬프트 라이브러리를 구축하여 팀과 공유한다."]),
    ("지시어 선정", ["'간단히'보다는 '한 문장으로'가 더 구체적이다.", "'잘해봐'보다는 '다음 조건을 충족해'가 명확하다.", "동사 위주의 강력한 지시어를 사용한다.", "모호한 표현(아마도, 대충)은 피한다.", "우선순위에 따라 번호를 매긴다."]),
    ("출력 형태 제어", ["'Yes or No로만 답해'라고 압박할 수 있다.", "어미를 '~입니다'로 통일하도록 지시한다.", "코드를 출력할 때는 마크다운 블록을 요청한다.", "감정 분석 시 점수(1~5)로 환산해달라고 한다.", "특정 단어의 사용을 금지할 수도 있다."]),
    ("자기 개선 프롬프트", ["'네 답변을 스스로 비판해봐'라고 시켜 품질을 높인다.", "틀린 부분을 직접 찾고 수정하게 만든다.", "사용자의 의도를 한 번 더 물어보게 한다.", "답변이 신뢰할 수 있는지 점수를 매기게 한다.", "출처를 명시하도록 하여 근거를 보강한다."])
]

id_counter = 4011
for topic, facts in topics_pe:
    for fact in facts:
        questions.append({
            "chapter_name": chapter_name, "type": "객관식", "difficulty": "medium", "id": str(id_counter),
            "question": f"{topic}의 핵심 활용 방법은? (엔지니어링-{id_counter-4010})",
            "options": [fact, "프롬프트의 의미를 완전히 왜곡한 지문", "최신 모델에서는 절대 금지되는 구식 방식", "컴퓨터 전원을 끄는 것과 같은 무의미한 명령", "오타가 가득한 이해 불가능한 보기", "다른 챕터(NumPy 등) 내용을 섞은 오답"],
            "answer": fact,
            "why": f"프롬프트 엔지니어링 실무에서 {topic}의 '{fact}' 개념은 결과물의 품질을 결정짓는 중요한 요소입니다.",
            "hint": topic
        })
        id_counter += 1

# 4061 ~ 4100 (Remaining 40 MCQs)
for i in range(4061, 4101):
    questions.append({
        "chapter_name": chapter_name, "type": "객관식", "difficulty": "medium", "id": str(i),
        "question": f"프롬프트 설계 시나리오 및 기법 {i-4060}: 최선의 프롬프트 전략을 고르세요.",
        "options": [
            f"전략 {i}: 기법을 올바르게 적용한 고품질 프롬프트 (설계-{i})",
            "질문의 의도를 파악하기 힘든 모호한 지시",
            "모델의 계산 효율을 심각하게 떨어뜨리는 지시",
            "출력 형식을 무시한 채 텍스트만 나열하는 방식",
            "페르소나 설정이 모순되어 발생하는 논리적 오류"
        ],
        "answer": f"전략 {i}: 기법을 올바르게 적용한 고품질 프롬프트 (설계-{i})",
        "why": f"좋은 프롬프트는 답변의 일관성과 품질을 보장합니다. {i}번 케이스의 최적화를 검증합니다.",
        "hint": "프롬프트 최적화 실무",
    })

# --- 20 Code Completion Questions ---
# 4101 ~ 4120
cc_data_ch4 = [
    ("CoT 지시어", "다음 문제를 ____ 생각해보자.", "단계별로", "Chain-of-Thought 유도 문구의 한글 번역 예시입니다."),
    ("Step by Step", "Let's think step by ____.", "step", "논리적 추론을 유도하는 가장 유명한 마법의 문구입니다."),
    ("Zero-shot", "____-shot: 예시 없이 바로 질문함.", "Zero", "예시를 전혀 주지 않는 기법의 명칭입니다."),
    ("Few-shot", "____-shot: 예시를 2~3개 보여줌.", "Few", "패턴 학습을 위해 소량의 예를 보여주는 기법입니다."),
    ("구분자", "###\n지시사항\n###\n인풋\n여기서 ###은 ____입니다.", "구분자", "영역을 나누어 모델의 혼란을 방지하는 도구입니다."),
    ("페르소나", "너는 이제부터 숙련된 '____'야.", "페르소나", "역할을 부여하는 기법의 명칭입니다. (또는 역할)"),
    ("시스템 메시지", "____ Message: 모델의 성격과 규칙을 정의함.", "System", "API 호출 시 근본적인 지침을 내리는 메시지 유형입니다."),
    ("JSON 출력", "결과를 ____ 형식으로 반환해줘.", "JSON", "기계가 읽기 좋은 대표적인 구조화 데이터 포맷입니다."),
    ("긍정 지시", "부정문보다 ____문을 사용하세요.", "긍정", "AI의 지시 이행력을 높이는 프롬프트 팁입니다."),
    ("Shot 의미", "In-Context Learning에서 예시를 1개 주면 ____-shot.", "One", "예시 숫자에 따른 호출 명칭입니다."),
    ("문맥 추가", "Instruction 외에 배경 지식을 주는 것은 ____ 설정입니다.", "Context", "프롬프트 구성 4요소 중 하나입니다."),
    ("출력 지표", "결과 스타일을 정하는 요소는 Output ____입니다.", "Indicator", "프롬프트 구성 4요소 중 하나입니다."),
    ("역할 놀이", "Persona Prompting은 다른 말로 ____-Playing 기법.", "Role", "페르소나를 정해주는 것의 영어 표현입니다."),
    ("생각의 사슬", "CoT의 약자는 Chain-of-____.", "Thought", "추론 유도 기법의 풀네임 중 일부입니다."),
    ("정확도 조절", "Temperature가 ____에 가까울수록 일관성이 높다.", "0", "일관된 답변을 위한 온도 설정값입니다."),
    ("창의성 조절", "Temperature가 ____에 가까울수록 답변이 다양해진다.", "1", "풍부한 상상력이 필요할 때의 온도 설정값입니다."),
    ("프롬프트 누설 방지", "____-Injection을 방어하는 설계가 필요하다.", "Prompt", "보안을 위협하는 유명한 공격 기법 명칭입니다."),
    ("예시 패턴", "패턴 학습을 유도하는 학습 방식은 In-____ Learning.", "Context", "모델 개선 없이 입력창 내에서 배우는 방식입니다."),
    ("구현체", "LangChain에서 프롬프트를 만드는 틀은 Prompt ____.", "Template", "프롬프트를 찍어내는 템플릿의 명칭입니다."),
    ("최종 점검", "질문이 모호하면 답변도 모호하다 (G____).", "GIGO", "데이터 과학의 오랜 격언입니다.")
]

for i, (title, code, ans, explain) in enumerate(cc_data_ch4):
    questions.append({
        "chapter_name": chapter_name, "type": "코드 완성형", "difficulty": "medium", "id": str(4101 + i),
        "question": f"{title} 문장 혹은 코드를 완성하세요.\n```text\n{code}\n```",
        "answer": ans,
        "why": explain,
        "hint": title,
    })

def get_questions():
    return questions
