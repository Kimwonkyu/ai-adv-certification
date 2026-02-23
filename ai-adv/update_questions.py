import json

new_questions = [
    # Chapter 1: Python 기초
    {
        "chapter_name": "Python 기초",
        "type": "코드 완성형",
        "question": "리스트 `nums`의 각 요소를 제곱한 새로운 리스트를 리스트 컴프리헨션으로 생성하세요.\n```python\nnums = [1, 2, 3, 4]\nsquares = [x**2 _____ x in nums]\n```",
        "answer": "for",
        "why": "리스트 컴프리헨션의 기본 문법은 [표현식 for 변수 in 반복가능객체] 입니다.",
        "hint": "반복문을 시작하는 키워드입니다.",
        "trap_points": ["in 키워드와 함께 쓰입니다."],
        "difficulty": "easy",
        "id": "q-c-001"
    },
    {
        "chapter_name": "Python 기초",
        "type": "코드 완성형",
        "question": "리스트를 순회하며 인덱스와 값을 동시에 얻기 위한 함수를 채우세요.\n```python\nitems = ['a', 'b', 'c']\nfor idx, val in _____(items):\n    print(idx, val)\n```",
        "answer": "enumerate",
        "why": "enumerate() 함수는 인덱스와 요소를 튜플 형태로 반환합니다.",
        "hint": "열거하다라는 뜻의 영어 단어입니다.",
        "trap_points": ["기본 시작 인덱스는 0입니다."],
        "difficulty": "easy",
        "id": "q-c-002"
    },
    {
        "chapter_name": "Python 기초",
        "type": "코드 완성형",
        "question": "자원 해제를 자동으로 보장하는 `with` 문에서 내부적으로 호출되는 메서드를 완성하세요.\n```python\nclass MyContext:\n    def _____(self):\n        print('시작')\n        return self\n    def __exit__(self, exc_type, exc_val, exc_tb):\n        print('종료')\n```",
        "answer": "__enter__",
        "why": "컨텍스트 매니저는 __enter__와 __exit__ 메서드를 구현해야 합니다.",
        "hint": "진입을 의미하는 매직 메서드입니다.",
        "trap_points": ["언더바(_) 2개가 앞뒤로 붙습니다."],
        "difficulty": "medium",
        "id": "q-c-003"
    },
    {
        "chapter_name": "Python 기초",
        "type": "코드 완성형",
        "question": "가변 개수의 키워드 인자를 받는 매개변수 형식을 완성하세요.\n```python\ndef my_func(_____+kwargs):\n    for key, value in kwargs.items():\n        print(key, value)\n```",
        "answer": "**",
        "why": "**kwargs는 딕셔너리 형태로 모든 키워드 인자를 받습니다.",
        "hint": "별표(asterisk)의 개수를 생각하세요.",
        "trap_points": ["*args는 위치 인자, **kwargs는 키워드 인자입니다."],
        "difficulty": "medium",
        "id": "q-c-004"
    },
    {
        "chapter_name": "Python 기초",
        "type": "코드 완성형",
        "question": "중첩 함수에서 상위 함수의 변수를 수정하기 위한 키워드를 채우세요.\n```python\ndef outer():\n    count = 0\n    def inner():\n        _____ count\n        count += 1\n    inner()\n```",
        "answer": "nonlocal",
        "why": "nonlocal은 현재 스코프가 아닌 가장 가까운 상위 스코프의 변수를 가리킵니다.",
        "hint": "local이 아니라는 뜻입니다.",
        "trap_points": ["global은 전역 변수를 수정할 때 사용합니다."],
        "difficulty": "hard",
        "id": "q-c-005"
    },
    {
        "chapter_name": "Python 기초",
        "type": "코드 완성형",
        "question": "딕셔너리를 이용해 리스트의 중복을 제거하면서 순서를 유지하는 관용구를 완성하세요.\n```python\nl = [3, 1, 2, 1, 3]\nunique_l = list(dict._____(l).keys())\n```",
        "answer": "fromkeys",
        "why": "dict.fromkeys()는 반복 가능한 객체의 요소를 키로 가지는 딕셔너리를 생성합니다.",
        "hint": "키로부터(from keys) 딕셔너리를 만듭니다.",
        "trap_points": ["Python 3.7+ 부터 딕셔너리는 삽입 순서를 유지합니다."],
        "difficulty": "hard",
        "id": "q-c-006"
    },
    {
        "chapter_name": "Python 기초",
        "type": "코드 완성형",
        "question": "에러 발생 여부와 상관없이 항상 실행되는 블록을 완성하세요.\n```python\ntry:\n    res = 10 / 0\nexcept ZeroDivisionError:\n    print('error')\n_____:\n    print('정리 작업')\n```",
        "answer": "finally",
        "why": "finally 블록은 예외 발생 및 처리 여부와 관계없이 무조건 실행됩니다.",
        "hint": "마지막이라는 뜻입니다.",
        "trap_points": ["자원 반납 루틴에 주로 쓰입니다."],
        "difficulty": "easy",
        "id": "q-c-007"
    },
    {
        "chapter_name": "Python 기초",
        "type": "코드 완성형",
        "question": "부모 클래스의 메서드를 호출하기 위한 함수를 채우세요.\n```python\nclass Child(Parent):\n    def greet(self):\n        return _____().greet() + ' and child'\n```",
        "answer": "super",
        "why": "super() 함수는 자식 클래스에서 부모 클래스의 객체를 참조할 때 사용합니다.",
        "hint": "상위를 뜻하는 단어입니다.",
        "trap_points": ["self를 인자로 전달하지 않아도 됩니다."],
        "difficulty": "medium",
        "id": "q-c-008"
    },
    {
        "chapter_name": "Python 기초",
        "type": "코드 완성형",
        "question": "데코레이터를 만들기 위해 함수를 인자로 받아 내부 함수를 반환하는 구조를 완성하세요.\n```python\ndef my_decorator(func):\n    def wrapper(*args, **kwargs):\n        print('Before')\n        result = _____(+args, **kwargs)\n        print('After')\n        return result\n    return wrapper\n```",
        "answer": "func",
        "why": "전달받은 원본 함수 `func`를 내부에서 호출해야 합니다.",
        "hint": "인자로 넘어온 함수의 이름입니다.",
        "trap_points": ["wrapper 함수는 원본 함수의 인자를 그대로 전달해야 합니다."],
        "difficulty": "hard",
        "id": "q-c-009"
    },
    {
        "chapter_name": "Python 기초",
        "type": "코드 완성형",
        "question": "람다 함수를 사용하여 두 수의 합을 구하는 식을 완성하세요.\n```python\nadd = _____ x, y: x + y\nprint(add(3, 5))\n```",
        "answer": "lambda",
        "why": "익명 함수를 정의할 때 lambda 키워드를 사용합니다.",
        "hint": "그리스 문자 이름입니다.",
        "trap_points": ["한 줄의 표현식만 가질 수 있습니다."],
        "difficulty": "easy",
        "id": "q-c-010"
    },

    # Chapter 2: 데이터 분석
    {
        "chapter_name": "데이터 분석",
        "type": "코드 완성형",
        "question": "Pandas에서 데이터프레임의 상위 n개 행을 확인하는 메서드를 채우세요.\n```python\nimport pandas as pd\ndf = pd.read_csv('data.csv')\nprint(df._____())\n```",
        "answer": "head",
        "why": "head() 메서드는 기본적으로 데이터프레임의 첫 5개 행을 보여줍니다.",
        "hint": "머리라는 뜻입니다.",
        "trap_points": ["괄호 안에 숫자를 넣어 개수를 지정할 수 있습니다."],
        "difficulty": "easy",
        "id": "q-c-011"
    },
    {
        "chapter_name": "데이터 분석",
        "type": "코드 완성형",
        "question": "Numpy에서 배열의 형태(차원)를 확인하는 속성을 채우세요.\n```python\nimport numpy as np\narr = np.array([[1, 2], [3, 4]])\nprint(arr._____)\n```",
        "answer": "shape",
        "why": "shape는 배열의 각 차원 크기를 알려주는 튜플 속성입니다.",
        "hint": "모양이라는 뜻입니다.",
        "trap_points": ["속성이므로 메서드처럼 ()를 붙이지 않습니다."],
        "difficulty": "easy",
        "id": "q-c-012"
    },
    {
        "chapter_name": "데이터 분석",
        "type": "코드 완성형",
        "question": "Pandas에서 결측치(NaN)가 있는 행을 제거하는 메서드를 채우세요.\n```python\ndf_clean = df._____()\n```",
        "answer": "dropna",
        "why": "dropna()는 결측치가 포함된 행이나 열을 제거합니다.",
        "hint": "떨어뜨리다(drop)와 관련이 있습니다.",
        "trap_points": ["원본을 바꾸려면 inplace=True 옵션이 필요합니다."],
        "difficulty": "medium",
        "id": "q-c-013"
    },
    {
        "chapter_name": "데이터 분석",
        "type": "코드 완성형",
        "question": "Matplotlib에서 산점도(Scatter plot)를 그리는 함수를 채우세요.\n```python\nimport matplotlib.pyplot as plt\nplt._____(x, y)\nplt.show()\n```",
        "answer": "scatter",
        "why": "scatter() 함수는 점으로 이루어진 그래프를 그립니다.",
        "hint": "흩뿌리다라는 뜻입니다.",
        "trap_points": ["선 그래프는 plot()을 사용합니다."],
        "difficulty": "easy",
        "id": "q-c-014"
    },
    {
        "chapter_name": "데이터 분석",
        "type": "코드 완성형",
        "question": "정규표현식에서 숫자를 의미하는 패턴을 채우세요.\n```python\nimport re\ntext = \"Price: 1500\"\nprice = re.findall(r'\\_____+', text)\n```",
        "answer": "d",
        "why": "\\d는 digit(숫자)를 의미하는 정규표현식 메타 문자입니다.",
        "hint": "숫자를 뜻하는 영어 단어의 첫 글자입니다.",
        "trap_points": ["대문자 \\D는 숫자가 아닌 것을 의미합니다."],
        "difficulty": "medium",
        "id": "q-c-015"
    },
    {
        "chapter_name": "데이터 분석",
        "type": "코드 완성형",
        "question": "Pandas에서 두 데이터프레임을 특정 키 기준으로 합치는(Join) 메서드를 채우세요.\n```python\nmerged_df = df1._____(df2, on='user_id')\n```",
        "answer": "merge",
        "why": "merge() 함수는 SQL의 JOIN과 유사한 병합 기능을 수행합니다.",
        "hint": "합치다라는 뜻입니다.",
        "trap_points": ["단순히 위아래/좌우로 붙이려면 concat을 씁니다."],
        "difficulty": "medium",
        "id": "q-c-016"
    },
    {
        "chapter_name": "데이터 분석",
        "type": "코드 완성형",
        "question": "Numpy에서 모든 요소가 0인 3x3 행렬을 만드는 함수를 채우세요.\n```python\nzero_matrix = np._____((3, 3))\n```",
        "answer": "zeros",
        "why": "np.zeros()는 지정된 크기의 0으로 채워진 배열을 생성합니다.",
        "hint": "0들의 복수형입니다.",
        "trap_points": ["튜플 형태로 크기를 전달해야 합니다."],
        "difficulty": "easy",
        "id": "q-c-017"
    },
    {
        "chapter_name": "데이터 분석",
        "type": "코드 완성형",
        "question": "Pandas에서 인덱스를 일반 열로 변환하는 메서드를 채우세요.\n```python\ndf_new = df._____()\n```",
        "answer": "reset_index",
        "why": "reset_index()는 기존 인덱스를 열로 보내고 0부터 시작하는 정수 인덱스를 설정합니다.",
        "hint": "인덱스를 재설정(Reset)합니다.",
        "trap_points": ["drop=True 옵션을 주면 기존 인덱스를 버립니다."],
        "difficulty": "medium",
        "id": "q-c-018"
    },
    {
        "chapter_name": "데이터 분석",
        "type": "코드 완성형",
        "question": "Seaborn에서 데이터의 분포를 히트맵으로 시각화하는 함수를 채우세요.\n```python\nimport seaborn as sns\nsns._____(data.corr())\n```",
        "answer": "heatmap",
        "why": "heatmap()은 데이터의 수치를 색상으로 표현하는 차트를 만듭니다.",
        "hint": "열(Heat) 지도를 뜻합니다.",
        "trap_points": ["상관관계 행렬 시각화에 자주 쓰입니다."],
        "difficulty": "medium",
        "id": "q-c-019"
    },
    {
        "chapter_name": "데이터 분석",
        "type": "코드 완성형",
        "question": "Scikit-learn에서 데이터를 학습(Train)과 검증(Test) 세트로 나누는 함수를 채우세요.\n```python\nfrom sklearn.model_selection import _____\nX_train, X_test, y_train, y_test = _____(X, y)\n```",
        "answer": "train_test_split",
        "why": "train_test_split()은 데이터를 무작위로 분할하여 과적합 평가를 돕습니다.",
        "hint": "학습_테스트_분할을 영어로 적으세요.",
        "trap_points": ["random_state를 고정하면 재현 가능합니다."],
        "difficulty": "hard",
        "id": "q-c-020"
    },

    # Chapter 3: LLM 기본
    {
        "chapter_name": "LLM 기본",
        "type": "코드 완성형",
        "question": "LangChain에서 프롬프트를 만드는 클래스를 채우세요.\n```python\nfrom langchain.prompts import _____\nprompt = _____.from_template(\"Tell me a joke about {topic}\")\n```",
        "answer": "ChatPromptTemplate",
        "why": "ChatPromptTemplate은 채팅 모델용 프롬프트를 구조화하는 기본 도구입니다.",
        "hint": "채팅_프롬프트_템플릿입니다.",
        "trap_points": ["문자열 하나면 PromptTemplate도 가능하지만 채팅에는 Chat용을 권장합니다."],
        "difficulty": "easy",
        "id": "q-c-021"
    },
    {
        "chapter_name": "LLM 기본",
        "type": "코드 완성형",
        "question": "LCEL(LangChain Expression Language)에서 컴포넌트들을 연결하는 연산자를 채우세요.\n```python\nchain = prompt _____ llm _____ StrOutputParser()\n```",
        "answer": "|",
        "why": "| (파이프) 연산자를 사용하여 선언적인 체인을 구성합니다.",
        "hint": "OR 연산 기호와 같습니다.",
        "trap_points": ["Unix 파이프라인 개념에서 유래했습니다."],
        "difficulty": "easy",
        "id": "q-c-022"
    },
    {
        "chapter_name": "LLM 기본",
        "type": "코드 완성형",
        "question": "체인의 입력을 그대로 전달하기 위해 사용하는 클래스를 채우세요.\n```python\nfrom langchain.schema.runnable import _____\nsetup_and_retrieval = {\"context\": retriever, \"question\": _____()}\n```",
        "answer": "RunnablePassthrough",
        "why": "RunnablePassthrough는 데이터를 변형 없이 다음 단계로 넘깁니다.",
        "hint": "실행가능한_그대로통과입니다.",
        "trap_points": ["람다 함수 대신 선언적으로 사용 가능합니다."],
        "difficulty": "medium",
        "id": "q-c-023"
    },
    {
        "chapter_name": "LLM 기본",
        "type": "코드 완성형",
        "question": "OpenAI의 채팅 API 호출 시 모델의 창의성을 조절하는 파라미터를 채우세요.\n```python\nresponse = client.chat.completions.create(\n    model=\"gpt-4\",\n    messages=messages,\n    _____=0.7\n)\n```",
        "answer": "temperature",
        "why": "temperature는 다음 토큰의 확률 분포를 조절하여 답변의 다양성을 결정합니다.",
        "hint": "온도를 뜻합니다.",
        "trap_points": ["0에 가까울수록 결정론적인 답을 냅니다."],
        "difficulty": "easy",
        "id": "q-c-024"
    },
    {
        "chapter_name": "LLM 기본",
        "type": "코드 완성형",
        "question": "토크나이저에서 사용되는 특수 토큰 중 문장의 시작을 알리는 약어를 채우세요.\n```python\ntokenizer.bos_token  # _____ of sentence\n```",
        "answer": "Beginning",
        "why": "BOS는 Beginning Of Sentence의 약자입니다.",
        "hint": "시작이라는 뜻입니다.",
        "trap_points": ["종료는 EOS (End Of Sentence) 입니다."],
        "difficulty": "medium",
        "id": "q-c-025"
    },
    {
        "chapter_name": "LLM 기본",
        "type": "코드 완성형",
        "question": "트랜스포머 아키텍처에서 핵심적인 '주의 집중' 메커니즘을 채우세요.\n```python\ndef _____(query, key, value):\n    scores = matmul(query, key.T) / sqrt(dk)\n    return matmul(softmax(scores), value)\n```",
        "answer": "attention",
        "why": "Attention 기호는 단어 간의 관계적 중요도를 계산합니다.",
        "hint": "주의, 집중이라는 뜻입니다.",
        "trap_points": ["Scaled Dot-Product Attention이 정식 명칭입니다."],
        "difficulty": "hard",
        "id": "q-c-026"
    },
    {
        "chapter_name": "LLM 기본",
        "type": "코드 완성형",
        "question": "모델이 한 번에 처리할 수 있는 최대 토큰 범위를 무엇이라 하나요?\n```python\n# This model has a 128k _____ window\n```",
        "answer": "context",
        "why": "Context Window는 모델의 단기 기억 용량과 같습니다.",
        "hint": "문맥이라는 뜻입니다.",
        "trap_points": ["윈도우 크기를 넘어가면 이전 내용은 잊혀집니다."],
        "difficulty": "easy",
        "id": "q-c-027"
    },
    {
        "chapter_name": "LLM 기본",
        "type": "코드 완성형",
        "question": "Ollama를 통해 로컬 모델을 불러올 때 사용하는 랭체인 클래스를 완성하세요.\n```python\nfrom langchain_ollama import _____\nllm = _____(model=\"llama3\")\n```",
        "answer": "ChatOllama",
        "why": "ChatOllama 클래스는 로컬 Ollama 서버와 통신하는 래퍼입니다.",
        "hint": "채팅_올라마입니다.",
        "trap_points": ["서버가 로컬에 실행 중이어야 합니다."],
        "difficulty": "medium",
        "id": "q-c-028"
    },
    {
        "chapter_name": "LLM 기본",
        "type": "코드 완성형",
        "question": "임베딩 모델을 사용하여 텍스트를 벡터로 바꾸는 메서드를 채우세요.\n```python\nvector = embeddings._____(text)\n```",
        "answer": "embed_query",
        "why": "embed_query()는 단일 텍스트 질문을 벡터화하는 표준 메서드입니다.",
        "hint": "쿼리(질문)를_임베딩합니다.",
        "trap_points": ["문서 뭉치를 한 번에 하려면 embed_documents를 씁니다."],
        "difficulty": "medium",
        "id": "q-c-029"
    },
    {
        "chapter_name": "LLM 기본",
        "type": "코드 완성형",
        "question": "체인의 출력을 문자열로 파싱해주는 기본 파서를 채우세요.\n```python\nfrom langchain.schema.output_parser import _____\nparser = _____()\n```",
        "answer": "StrOutputParser",
        "why": "StrOutputParser는 AIMessage 객체에서 텍스트 내용만 추출합니다.",
        "hint": "문자열_출력_파서입니다.",
        "trap_points": ["가장 많이 쓰이는 기본 파서입니다."],
        "difficulty": "easy",
        "id": "q-c-030"
    },

    # Chapter 4: 프롬프트 엔지니어링
    {
        "chapter_name": "프롬프트 엔지니어링",
        "type": "코드 완성형",
        "question": "모델에게 가상의 인격을 부여하여 행동 지침을 내리는 역할을 채우세요.\n```python\nprompt = ChatPromptTemplate.from_messages([\n    (\"_____\", \"You are a helpful travel agent.\"),\n    (\"human\", \"{input}\")\n])\n```",
        "answer": "system",
        "why": "system 메시지는 모델의 정체성과 제약을 정의하는 고수준 지침입니다.",
        "hint": "시스템 계층의 지시입니다.",
        "trap_points": ["최신 모델일수록 시스템 메시지 준수율이 높습니다."],
        "difficulty": "easy",
        "id": "q-c-031"
    },
    {
        "chapter_name": "프롬프트 엔지니어링",
        "type": "코드 완성형",
        "question": "예시를 몇 가지 보여주어 학습시키는 기법을 완성하세요.\n```python\n# This is a _____-shot prompting technique\nexample = \"Question: 1+1, Answer: 2\"\n```",
        "answer": "few",
        "why": "Few-shot 기법은 텍스트 내에서 몇 개의 예시를 통해 패턴을 전달합니다.",
        "hint": "약간의, 소수의라는 뜻입니다.",
        "trap_points": ["예시가 하나면 One-shot, 없으면 Zero-shot 입니다."],
        "difficulty": "easy",
        "id": "q-c-032"
    },
    {
        "chapter_name": "프롬프트 엔지니어링",
        "type": "코드 완성형",
        "question": "'단계별로 차근차근 생각해봐'라고 유도하여 추론 능력을 높이는 기법을 채우세요.\n```python\n# Chain of _____ (CoT)\nprompt = \"Let's think step by step.\"\n```",
        "answer": "Thought",
        "why": "Chain of Thought (CoT)는 복잡한 논리 문제를 해결할 때 중간 과정을 거치게 합니다.",
        "hint": "생각, 사고라는 뜻입니다.",
        "trap_points": ["단순 지식 인출보다 추론 문제에 효과적입니다."],
        "difficulty": "medium",
        "id": "q-c-033"
    },
    {
        "chapter_name": "프롬프트 엔지니어링",
        "type": "코드 완성형",
        "question": "사용자의 입력 데이터와 지시 사항을 명확히 분리하기 위한 도구를 완성하세요.\n```python\nprompt = \"\"\"\nBelow is the user input.\nUse the _____ '###' to wrap the input.\n\n###\n{user_input}\n###\n\"\"\"\n```",
        "answer": "delimiter",
        "why": "구분자(Delimiter)를 사용하면 프롬프트 인젝션 공격 방어에 도움이 됩니다.",
        "hint": "구분자라는 뜻의 영어 단어입니다.",
        "trap_points": ["모델이 입력값을 명령어로 오해하지 않게 합니다."],
        "difficulty": "medium",
        "id": "q-c-034"
    },
    {
        "chapter_name": "프롬프트 엔지니어링",
        "type": "코드 완성형",
        "question": "모델이 스스로의 답을 검토하고 수정하게 하는 성찰(Reflection) 구조를 완성하세요.\n```python\n# _____-reflect on your previous answer\nprompt = \"Check if there are any errors in your code.\"\n```",
        "answer": "Self",
        "why": "Self-reflection은 정확도를 높이기 위한 중요한 메타인지 기법입니다.",
        "hint": "스스로를 뜻하는 접두사입니다.",
        "trap_points": ["비용이 두 배로 들지만 품질이 향상됩니다."],
        "difficulty": "medium",
        "id": "q-c-035"
    },
    {
        "chapter_name": "프롬프트 엔지니어링",
        "type": "코드 완성형",
        "question": "특정 형식(예: JSON)으로만 답하도록 강제하는 설정을 채우세요.\n```python\nresponse = client.chat.completions.create(\n    model=\"gpt-4-turbo\",\n    response_format={ \"type\": \"_____\" }\n)\n```",
        "answer": "json_object",
        "why": "response_format을 json_object로 설정하면 항상 유효한 JSON을 반환합니다.",
        "hint": "JSON_객체라는 뜻입니다.",
        "trap_points": ["시스템 프롬프트에도 JSON 관련 언급이 있어야 작동합니다."],
        "difficulty": "hard",
        "id": "q-c-036"
    },
    {
        "chapter_name": "프롬프트 엔지니어링",
        "type": "코드 완성형",
        "question": "모델의 답변 길이에 제한을 두는 파라미터를 완성하세요.\n```python\nresponse = client.chat.completions.create(\n    model=\"gpt-4\",\n    _____+tokens=100\n)\n```",
        "answer": "max",
        "why": "max_tokens (최신 API는 max_completion_tokens)는 답변의 최대 길이를 제어합니다.",
        "hint": "최대라는 뜻의 약자입니다.",
        "trap_points": ["값이 너무 작으면 답변이 중간에 끊깁니다."],
        "difficulty": "easy",
        "id": "q-c-037"
    },
    {
        "chapter_name": "프롬프트 엔지니어링",
        "type": "코드 완성형",
        "question": "프롬프트에서 변수를 채워 넣는 메서드를 완성하세요.\n```python\nfrom langchain.prompts import PromptTemplate\nprompt = PromptTemplate.from_template(\"{adjective} joke\")\nprompt._____(adjective=\"funny\")\n```",
        "answer": "format",
        "why": "format() 메서드는 템플릿의 변수 자리에 실제 값을 주입합니다.",
        "hint": "형식을 갖추다라는 뜻입니다.",
        "trap_points": ["문자열 f-string 대용으로 쓰입니다."],
        "difficulty": "easy",
        "id": "q-c-038"
    },
    {
        "chapter_name": "프롬프트 엔지니어링",
        "type": "코드 완성형",
        "question": "외부 환경 변수(API KEY 등)를 로드하기 위한 파이썬 함수를 채우세요.\n```python\nfrom dotenv import _____\n_____()\n```",
        "answer": "load_dotenv",
        "why": "load_dotenv()는 .env 파일의 내용을 환경 변수로 불러옵니다.",
        "hint": "환경(dotenv)을_불러오다(load)입니다.",
        "trap_points": ["os.getenv()와 함께 주로 사용됩니다."],
        "difficulty": "medium",
        "id": "q-c-039"
    },
    {
        "chapter_name": "프롬프트 엔지니어링",
        "type": "코드 완성형",
        "question": "모델이 이미 알고 있는 지식이 아닌 외부 데이터와의 연동을 위해 검색하는 과정을 완성하세요.\n```python\n# Retrieval Augmented Generation (_____)\n```",
        "answer": "RAG",
        "why": "RAG는 지식 검색과 생성을 결합한 최신 기술 패러다임입니다.",
        "hint": "알파벳 3글자 약어입니다.",
        "trap_points": ["환각 현상을 줄이는 가장 효과적인 방법입니다."],
        "difficulty": "easy",
        "id": "q-c-040"
    },

    # Chapter 5: RAG & Agent
    {
        "chapter_name": "RAG & Agent",
        "type": "코드 완성형",
        "question": "PDF 파일을 불러오기 위한 랭체인 로더 클래스를 완성하세요.\n```python\nfrom langchain_community.document_loaders import _____\nloader = _____(\"my_file.pdf\")\n```",
        "answer": "PyMuPDFLoader",
        "why": "PyMuPDFLoader는 PDF 문서를 읽어 랭체인 Document 객체로 변환합니다.",
        "hint": "파이_뮤_피디에프_로더입니다.",
        "trap_points": ["PyPDFLoader 등 다양한 옵션이 존재합니다."],
        "difficulty": "medium",
        "id": "q-c-041"
    },
    {
        "chapter_name": "RAG & Agent",
        "type": "코드 완성형",
        "question": "방대한 문서를 의미 있는 조각으로 나누는 클래스를 완성하세요.\n```python\nfrom langchain.text_splitter import _____\nsplitter = _____(chunk_size=1000, chunk_overlap=200)\n```",
        "answer": "RecursiveCharacterTextSplitter",
        "why": "RecursiveCharacterTextSplitter는 문맥을 유지하며 텍스트를 최적으로 분할합니다.",
        "hint": "재귀적_문자_텍스트_분할기입니다.",
        "trap_points": ["chunk_overlap은 정보의 누락을 방지합니다."],
        "difficulty": "hard",
        "id": "q-c-042"
    },
    {
        "chapter_name": "RAG & Agent",
        "type": "코드 완성형",
        "question": "벡터 저장소를 검색기(Retriever)로 변환하는 메서드를 채우세요.\n```python\nretriever = vectorstore._____(search_kwargs={\"k\": 3})\n```",
        "answer": "as_retriever",
        "why": "as_retriever()는 벡터 DB를 인터페이스화하여 검색 기능을 제공합니다.",
        "hint": "검색기로서(as retriever) 사용합니다.",
        "trap_points": ["k값은 가져올 문서의 개수입니다."],
        "difficulty": "medium",
        "id": "q-c-043"
    },
    {
        "chapter_name": "RAG & Agent",
        "type": "코드 완성형",
        "question": "하나의 질문을 여러 질문으로 재작성하여 검색 성능을 높이는 클래스를 완성하세요.\n```python\nfrom langchain.retrievers.multi_query import _____\nretriever = _____.from_llm(retriever=base, llm=llm)\n```",
        "answer": "MultiQueryRetriever",
        "why": "MultiQueryRetriever는 다양한 관점에서 쿼리를 생성해 검색 누락을 방지합니다.",
        "hint": "멀티_쿼리_리트리버입니다.",
        "trap_points": ["검색 결과들은 합쳐져서 중복이 제거됩니다."],
        "difficulty": "hard",
        "id": "q-c-044"
    },
    {
        "chapter_name": "RAG & Agent",
        "type": "코드 완성형",
        "question": "에이전트가 어떤 도구를 사용할지 결정하고 실행하는 루프를 완성하세요.\n```python\nwhile True:\n    response = llm._____(tools)\n    # Logic to execute tool and observe result\n```",
        "answer": "invoke",
        "why": "invoke()는 모델을 호출하여 다음 행동을 결정하게 합니다.",
        "hint": "호출하다라는 뜻입니다.",
        "trap_points": ["최신 API에서는 bind_tools와 함께 쓰입니다."],
        "difficulty": "easy",
        "id": "q-c-045"
    },
    {
        "chapter_name": "RAG & Agent",
        "type": "코드 완성형",
        "question": "파이썬 코드를 정의하고 에이전트 도구로 등록하기 위한 데코레이터를 완성하세요.\n```python\nfrom langchain_core.tools import _____\n\n@_____\ndef my_tool(query: str):\n    \"\"\"This is a tool description.\"\"\"\n    return \"result\"\n```",
        "answer": "tool",
        "why": "@tool 데코레이터는 함수를 에이전트가 인식할 수 있는 Tool 객체로 바꿉니다.",
        "hint": "도구라는 뜻입니다.",
        "trap_points": ["함수의 독스트링(Docstring)이 매우 중요합니다."],
        "difficulty": "easy",
        "id": "q-c-046"
    },
    {
        "chapter_name": "RAG & Agent",
        "type": "코드 완성형",
        "question": "에이전트가 웹에서 정보를 검색할 수 있게 해주는 대표적인 API 서비스를 채우세요.\n```python\nfrom langchain_community.tools.tavily_search import _____\ntool = _____()\n```",
        "answer": "TavilySearchResults",
        "why": "Tavily는 AI 친화적인 실시간 웹 검색 결과를 제공합니다.",
        "hint": "타빌리_검색_결과입니다.",
        "trap_points": ["API 키 설정이 필요합니다."],
        "difficulty": "medium",
        "id": "q-c-047"
    },
    {
        "chapter_name": "RAG & Agent",
        "type": "코드 완성형",
        "question": "검색 결과들을 하나의 문자열로 합치기 위한 함수 구조를 완성하세요.\n```python\ndef format_docs(docs):\n    return \"\\n\\n\"._____(doc.page_content for doc in docs)\n```",
        "answer": "join",
        "why": "join() 메서드는 리스트의 요소를 하나로 병합합니다.",
        "hint": "연결하다라는 뜻입니다.",
        "trap_points": ["구분자를 공백이나 줄바꿈으로 설정할 수 있습니다."],
        "difficulty": "easy",
        "id": "q-c-048"
    },
    {
        "chapter_name": "RAG & Agent",
        "type": "코드 완성형",
        "question": "Chroma 벡터 DB를 로컬 폴더에 저장하기 위한 파라미터를 완성하세요.\n```python\nvectorstore = Chroma.from_documents(\n    documents=chunks,\n    embedding=embedding,\n    _____+directory=\"./my_db\"\n)\n```",
        "answer": "persist",
        "why": "persist_directory는 세션 종료 후에도 벡터 DB를 유지할 위치를 지정합니다.",
        "hint": "지속성(Persist)과 관련이 있습니다.",
        "trap_points": ["최신 버전에서는 명칭이 바뀔 수 있음에 주의하세요."],
        "difficulty": "medium",
        "id": "q-c-049"
    },
    {
        "chapter_name": "RAG & Agent",
        "type": "코드 완성형",
        "question": "에이전트의 사고 과정을 기록하는 'Reason + Act'의 약자를 채우세요.\n```python\n# This is a _____ agent architecture.\n```",
        "answer": "ReAct",
        "why": "ReAct 프레임워크는 논리적 추론과 도구 실행을 반복하는 구조입니다.",
        "hint": "R-e-A-c-t 입니다.",
        "trap_points": ["생각(Thought)과 행동(Action)의 결합입니다."],
        "difficulty": "easy",
        "id": "q-c-050"
    },

    # Chapter 6: Fine Tuning
    {
        "chapter_name": "Fine Tuning",
        "type": "코드 완성형",
        "question": "기존 지식은 유지하며 소량의 파라미터만 학습하는 효율적 기법의 약자를 채우세요.\n```python\n# Parameter-Efficient Fine-Tuning (_____)\n```",
        "answer": "PEFT",
        "why": "PEFT는 99%의 가중치를 고정하고 일부만 학습하여 자원을 절약합니다.",
        "hint": "알파벳 4글자 약어입니다.",
        "trap_points": ["LoRA가 대표적인 PEFT 알고리즘입니다."],
        "difficulty": "easy",
        "id": "q-c-051"
    },
    {
        "chapter_name": "Fine Tuning",
        "type": "코드 완성형",
        "question": "가장 널리 쓰이는 저차원 어댑터 학습 알고리즘을 완성하세요.\n```python\n# Low-Rank Adaptation (_____)\n```",
        "answer": "LoRA",
        "why": "LoRA는 행렬 분해 원리를 이용해 학습 대상을 비약적으로 줄입니다.",
        "hint": "알파벳 4글자 로-라 입니다.",
        "trap_points": ["원본 모델 훼손 없이 어댑터만 교체 가능합니다."],
        "difficulty": "easy",
        "id": "q-c-052"
    },
    {
        "chapter_name": "Fine Tuning",
        "type": "코드 완성형",
        "question": "인간의 선호를 반영하기 위해 '좋은 것'과 '나쁜 것'을 직접 비교하는 선호 최적화 기법을 채우세요.\n```python\n# Direct Preference Optimization (_____)\n```",
        "answer": "DPO",
        "why": "DPO는 복잡한 RLHF 보상 모델 없이 직접 데이터를 통해 선호를 학습합니다.",
        "hint": "알파벳 3글자 약어입니다.",
        "trap_points": ["RLHF보다 구현이 단순하고 안정적입니다."],
        "difficulty": "medium",
        "id": "q-c-053"
    },
    {
        "chapter_name": "Fine Tuning",
        "type": "코드 완성형",
        "question": "학습량을 극단적으로 줄여 단 1~2개 행렬만 튜닝하는 LoRA의 핵심 개념을 채우세요.\n```python\n# Low-_____ matrix factorization\n```",
        "answer": "Rank",
        "why": "Rank(계수) 값을 낮게 설정하여 메모리 소모를 줄입니다.",
        "hint": "순위, 계수라는 뜻입니다.",
        "trap_points": ["보통 8, 16, 32 등의 값을 사용합니다."],
        "difficulty": "hard",
        "id": "q-c-054"
    },
    {
        "chapter_name": "Fine Tuning",
        "type": "코드 완성형",
        "question": "파인튜닝 후 모델이 기존에 학습했던 능력을 잊어버리는 현상을 무엇이라 하나요?\n```python\n# _____ Forgetting\n```",
        "answer": "Catastrophic",
        "why": "파멸적 망각(Catastrophic Forgetting)은 전이 학습의 대표적인 부작용입니다.",
        "hint": "파멸적인, 치명적인이라는 뜻입니다.",
        "trap_points": ["데이터 혼합 학습(Replay)으로 완화 가능합니다."],
        "difficulty": "medium",
        "id": "q-c-055"
    },
    {
        "chapter_name": "Fine Tuning",
        "type": "코드 완성형",
        "question": "데이터를 질의응답(Q&A) 템플릿으로 가공하여 학습시키는 단계를 완성하세요.\n```python\n# SFT (_____ Fine-Tuning)\n```",
        "answer": "Supervised",
        "why": "지도 학습(Supervised) 기반으로 모델의 대화 형식을 정제합니다.",
        "hint": "감독되다, 지도받다라는 뜻입니다.",
        "trap_points": ["RL 연구 전 단계로 주로 수행됩니다."],
        "difficulty": "easy",
        "id": "q-c-056"
    },
    {
        "chapter_name": "Fine Tuning",
        "type": "코드 완성형",
        "question": "추론 과정(Thinking process)을 포함하도록 데이터셋을 구성하는 기법을 채우세요.\n```python\n# Reasoning SFT (Cold-_____ startup)\n```",
        "answer": "start",
        "why": "Cold-start 데이터는 모델이 추론 루프를 돌기 위한 초기 지침을 제공합니다.",
        "hint": "시작이라는 뜻입니다.",
        "trap_points": ["DeepSeek-R1 학습의 핵심 단계입니다."],
        "difficulty": "hard",
        "id": "q-c-057"
    },
    {
        "chapter_name": "Fine Tuning",
        "type": "코드 완성형",
        "question": "여러 개의 출력 그룹 중 상대적으로 우수한 것에 높은 가중치를 주는 강화학습 알고리즘을 채우세요.\n```python\n# Group Relative Policy Optimization (_____)\n```",
        "answer": "GRPO",
        "why": "GRPO는 보상 모델 없이 그룹 내 편차를 이용해 효율적으로 학습합니다.",
        "hint": "알파벳 4글자 약어입니다.",
        "trap_points": ["PPO 모델의 한계를 극복하기 위해 제안되었습니다."],
        "difficulty": "hard",
        "id": "q-c-058"
    },
    {
        "chapter_name": "Fine Tuning",
        "type": "코드 완성형",
        "question": "모델의 가중치를 4비트나 8비트로 압축하여 메모리 효율을 높이는 기법을 완성하세요.\n```python\n# Model _____\n```",
        "answer": "Quantization",
        "why": "양자화(Quantization)는 계산 한계를 극복하여 소비자용 GPU에서도 모델이 돌게 해줍니다.",
        "hint": "양자화라는 단어를 적으세요.",
        "trap_points": ["비트수가 낮을수록 성능 손실(Perplexity↑)이 발생합니다."],
        "difficulty": "medium",
        "id": "q-c-059"
    },
    {
        "chapter_name": "Fine Tuning",
        "type": "코드 완성형",
        "question": "모델이 이미 도메인 지식은 갖추고 지시 이행 능력만 배우는 과정을 무엇이라 하나요?\n```python\n# Instruction _____\n```",
        "answer": "Tuning",
        "why": "Instruction Tuning은 모델을 개인 비서처럼 동작하게 만드는 과정입니다.",
        "hint": "조율하다라는 뜻입니다.",
        "trap_points": ["Chat/Instruct라는 이름이 붙는 이유입니다."],
        "difficulty": "easy",
        "id": "q-c-060"
    }
]

# Load existing questions
with open('vibe-web/public/questions.json', 'r', encoding='utf-8') as f:
    existing_questions = json.load(f)

# Filter out old "코드 완성형" (formerly "주관식") to avoid duplicates if necessary,
# but the user said "문제를 새로 만들어서 업데이트 해줘".
# I'll keep existing MCQs and replace/add the new code completion questions.

# Remove old "코드 완성형" (formerly "주관식") questions
filtered_questions = [q for q in existing_questions if q['type'] == '객관식']

# Add new questions
all_questions = filtered_questions + new_questions

# Sort by id if possible, or just write
with open('vibe-web/public/questions.json', 'w', encoding='utf-8') as f:
    json.dump(all_questions, f, ensure_ascii=False, indent=2)

print(f"Successfully updated! Total questions: {len(all_questions)}")
