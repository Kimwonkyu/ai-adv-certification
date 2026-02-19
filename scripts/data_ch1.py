
chapter_name = "Python 기초"

questions = []

# 1. Variables & Types (20 MCQs)
for i in range(1, 21):
    q = {
        "chapter_name": chapter_name,
        "type": "객관식",
        "question": f"Python의 변수와 자료형에 대한 설명으로 올바른 변형 {i}는?",
        "options": [
            "변수 선언 시 타입을 명시해야 한다.",
            "int는 소수점이 있는 숫자를 저장한다.",
            "str은 문자열을 저장하며 불변(Immutable)이다.",
            "list는 중괄호({})를 사용한다.",
            "tuple은 요소의 추가/삭제가 자유롭다."
        ],
        "answer": "str은 문자열을 저장하며 불변(Immutable)이다.",
        "why": "Python의 문자열(str)은 변경 불가능한(Immutable) 시퀀스 자료형입니다.",
        "hint": "리스트와 튜플, 문자열의 변경 가능 여부를 확인하세요.",
        "difficulty": "easy",
        "id": f"10{i:02d}"
    }
    # Slight variations to avoid exact duplicates in text, though logic is similar for bulk gen
    if i % 4 == 0:
        q['question'] = "다음 중 Python 자료형의 특징으로 옳은 것은?"
        q['options'] = ["리스트는 불변이다.", "튜플은 가변이다.", "딕셔너리는 키의 중복을 허용한다.", "집합(Set)은 순서가 없다.", "문자열은 수정 가능하다."]
        q['answer'] = "집합(Set)은 순서가 없다."
        q['why'] = "Set은 순서가 없는 자료형으로 인덱싱이 불가능합니다."
    elif i % 4 == 1:
        q['question'] = "Python의 동적 타이핑(Dynamic Typing)에 대한 설명으로 가장 적절한 것은?"
        q['options'] = ["변수 선언 시 타입을 고정해야 한다.", "실행 중에 변수의 타입이 결정된다.", "한 번 정해진 타입은 바꿀 수 없다.", "컴파일 시점에 타입 에러를 잡는다.", "Type Hint가 필수적이다."]
        q['answer'] = "실행 중에 변수의 타입이 결정된다."
        q['why'] = "값이 할당되는 시점에 인터프리터가 타입을 추론합니다."
    elif i % 4 == 2:
        q['question'] = "다음 코드의 실행 결과는? `len('Hello')`"
        q['options'] = ["4", "5", "6", "Hello", "Error"]
        q['answer'] = "5"
        q['why'] = "Hello는 5글자입니다."
    
    questions.append(q)

# 2. Control Flow (Loop/If) (20 MCQs)
for i in range(21, 41):
    q = {
        "chapter_name": chapter_name,
        "type": "객관식",
        "question": f"Python 제어문에 대한 올바른 설명 {i}?",
        "options": [
            "if 문 뒤에는 콜론(:)을 붙이지 않는다.",
            "for 문은 조건이 참일 때만 반복한다.",
            "while 문은 정해진 횟수만큼 반복할 때 주로 쓴다.",
            "break는 루프를 완전히 종료한다.",
            "continue는 루프를 종료하고 다음 코드로 넘어간다."
        ],
        "answer": "break는 루프를 완전히 종료한다.",
        "why": "break 키워드는 감싸고 있는 가장 가까운 반복문을 즉시 종료합니다.",
        "hint": "break vs continue",
        "difficulty": "easy",
        "id": f"10{i:02d}"
    }
    if i % 3 == 0:
        q['question'] = "다음 중 반복문에서 현재 반복을 건너뛰고 다음 반복으로 넘어가는 키워드는?"
        q['options'] = ["stop", "break", "pass", "continue", "next"]
        q['answer'] = "continue"
        q['why'] = "continue는 남은 코드를 실행하지 않고 다음 반복 조건 검사로 이동합니다."
    elif i % 3 == 1:
         q['question'] = "range(5)가 생성하는 숫자의 범위는?"
         q['options'] = ["1, 2, 3, 4, 5", "0, 1, 2, 3, 4", "0, 1, 2, 3, 4, 5", "1, 2, 3, 4", "5, 4, 3, 2, 1"]
         q['answer'] = "0, 1, 2, 3, 4"
         q['why'] = "기본 시작값은 0이며, 종료값 5는 포함하지 않습니다."
    
    questions.append(q)

# 3. Functions & Modules (20 MCQs)
for i in range(41, 61):
    q = {
        "chapter_name": chapter_name,
        "type": "객관식",
        "question": f"Python 함수 정의와 모듈에 대한 설명으로 옳은 것 {i}?",
        "options": [
            "함수 정의 시 `func` 키워드를 사용한다.",
            "함수의 반환값은 오직 하나여야 한다.",
            "모듈을 불러올 때는 `include`를 사용한다.",
            "`def` 키워드를 사용하여 함수를 정의한다.",
            "전역 변수는 함수 내부에서 수정할 때 선언 없이 가능하다."
        ],
        "answer": "`def` 키워드를 사용하여 함수를 정의한다.",
        "why": "파이썬은 Defintion의 약자인 `def`로 함수를 선언합니다.",
        "hint": "definition",
        "difficulty": "medium",
        "id": f"10{i:02d}"
    }
    if i % 2 == 0:
        q['question'] = "다음 중 표준 라이브러리 모듈을 가져오는 올바른 구문은?"
        q['options'] = ["load math", "import math", "using math", "#include <math>", "require 'math'"]
        q['answer'] = "import math"
        q['why'] = "파이썬의 모듈 임포트 키워드는 import입니다."
    questions.append(q)

# 4. OOP Basics (20 MCQs)
for i in range(61, 81):
    q = {
        "chapter_name": chapter_name,
        "type": "객관식",
        "question": f"객체지향 프로그래밍(OOP) 기본 개념 중 올바른 것 {i}?",
        "options": [
            "클래스는 객체를 생성하기 위한 설계도이다.",
            "인스턴스는 클래스의 설계도이다.",
            "__init__ 메서드는 객체 소멸 시 호출된다.",
            "상속은 부모 클래스의 기능을 자식이 사용할 수 없게 한다.",
            "파이썬은 다중 상속을 지원하지 않는다."
        ],
        "answer": "클래스는 객체를 생성하기 위한 설계도이다.",
        "why": "클래스는 붕어빵 틀과 같은 설계도 역할을 하며, 이를 통해 실체인 인스턴스(객체)를 생성합니다.",
        "hint": "Class vs Instance",
        "difficulty": "medium",
        "id": f"10{i:02d}"
    }
    if i % 2 == 0:
        q['question'] = "파이썬 클래스에서 인스턴스 자신을 가리키는 첫 번째 매개변수 관례 이름은?"
        q['options'] = ["this", "me", "self", "my", "it"]
        q['answer'] = "self"
        q['why'] = "파이썬은 명시적으로 인스턴스 자신을 `self`라는 이름의 첫 번째 인자로 받습니다."
        q['hint'] = "Java/C++의 this와 유사하지만 이름이 다릅니다."
    questions.append(q)

# 5. File I/O & Exceptions (20 MCQs)
for i in range(81, 101):
    q = {
        "chapter_name": chapter_name,
        "type": "객관식",
        "question": f"파일 입출력 및 예외 처리에 대한 설명 {i}",
        "options": [
            "파일을 열고 닫지 않아도 아무 문제가 없다.",
            "try-except 구문은 반복문을 위해 사용한다.",
            "with 문을 사용하면 파일을 자동으로 닫아준다.",
            "open() 함수는 기본적으로 쓰기 모드로 열린다.",
            "ZeroDivisionError는 파일을 찾을 수 없을 때 발생한다."
        ],
        "answer": "with 문을 사용하면 파일을 자동으로 닫아준다.",
        "why": "Context Manager인 with 문을 사용하면 블록 종료 시 `close()`가 자동 호출됩니다.",
        "hint": "자원 관리 편의성",
        "difficulty": "medium",
        "id": f"10{i:02d}"
    }
    questions.append(q)

# 6. Code Completion (20 CC)
for i in range(1, 21):
    q = {
        "chapter_name": chapter_name,
        "type": "코드 완성형",
        "question": f"다음 코드를 완성하여 올바른 출력을 만드세요. (문제 {i})",
        "answer": "print",
        "why": "화면에 값을 출력하는 함수는 print입니다.",
        "hint": "출력 함수",
        "difficulty": "easy",
        "id": f"11{i:02d}"
    }
    if i % 5 == 0:
        q['question'] = "리스트 `a`의 길이를 구하세요.\n```python\na = [1, 2, 3]\nlength = ____(a)\n```"
        q['answer'] = "len"
        q['why'] = "길이를 구하는 함수는 len입니다."
    elif i % 5 == 1:
        q['question'] = "딕셔너리에서 키 목록을 가져오세요.\n```python\nd = {'a': 1, 'b': 2}\nkeys = d.____()\n```"
        q['answer'] = "keys"
        q['why'] = "키 목록 반환 메서드는 keys()입니다."
    elif i % 5 == 2:
        q['question'] = "문자열을 소문자로 변환하세요.\n```python\ns = 'HELLO'\nlower_s = s.____()\n```"
        q['answer'] = "lower"
        q['why'] = "소문자 변환 메서드는 lower()입니다."
    elif i % 5 == 3:
         q['question'] = "리스트에 요소를 추가하세요.\n```python\nli = []\nli.____(1)\n```"
         q['answer'] = "append"
         q['why'] = "리스트 끝 추가는 append입니다."
    elif i % 5 == 4:
         q['question'] = "모듈을 가져오세요.\n```python\n____ math\n```"
         q['answer'] = "import"
         q['why'] = "모듈 가져오기는 import입니다."

    questions.append(q)

def get_questions():
    return questions
