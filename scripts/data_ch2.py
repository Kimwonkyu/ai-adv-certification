
chapter_name = "데이터 분석"

questions = []

# 1. NumPy Basics (20 MCQs)
for i in range(1, 21):
    q = {
        "chapter_name": chapter_name,
        "type": "객관식",
        "question": f"NumPy에 대한 설명으로 올바른 변형 {i}는?",
        "options": [
            "리스트보다 연산 속도가 느리다.",
            "동일한 자료형의 데이터만 담을 수 있다.",
            "GPU 가속을 기본적으로 지원한다.",
            "Python 기본 `math` 라이브러리와 완전히 동일하다.",
            "인덱싱이 1부터 시작한다."
        ],
        "answer": "동일한 자료형의 데이터만 담을 수 있다.",
        "why": "NumPy 배열(ndarray)은 성능 최적화를 위해 모든 요소가 동일한 데이터 타입이어야 합니다.",
        "hint": "Homogeneous array",
        "difficulty": "easy",
        "id": f"20{i:02d}"
    }
    if i % 3 == 0:
        q['question'] = "0으로 초기화된 크기 10의 배열을 생성하는 함수는?"
        q['options'] = ["np.array(10)", "np.zeros(10)", "np.empty(10)", "np.nulls(10)", "np.init(10)"]
        q['answer'] = "np.zeros(10)"
        q['why'] = "zeros 함수는 0으로 채워진 배열을 생성합니다."
    
    questions.append(q)

# 2. Pandas Basics & Series (20 MCQs)
for i in range(21, 41):
    q = {
        "chapter_name": chapter_name,
        "type": "객관식",
        "question": f"Pandas Series와 DataFrame에 대한 설명 {i}",
        "options": [
            "Series는 2차원 데이터를 다룬다.",
            "DataFrame은 행과 열이 있는 구조이다.",
            "Pandas는 수치 데이터만 처리할 수 있다.",
            "DataFrame은 엑셀 파일로 저장이 불가능하다.",
            "Index는 중복될 수 없다."
        ],
        "answer": "DataFrame은 행과 열이 있는 구조이다.",
        "why": "DataFrame은 행(Row)과 열(Column)로 구성된 2차원 테이블 형태의 데이터 구조입니다.",
        "hint": "엑셀 시트와 비슷한 구조",
        "difficulty": "easy",
        "id": f"20{i:02d}"
    }
    questions.append(q)

# 3. Data Selection & Filtering (20 MCQs)
for i in range(41, 61):
    q = {
        "chapter_name": chapter_name,
        "type": "객관식",
        "question": f"데이터 선택 및 필터링 방법 {i}",
        "options": [
            "loc은 정수 위치 기반 인덱싱이다.",
            "iloc은 라벨(이름) 기반 인덱싱이다.",
            "df['col']은 열을 선택한다.",
            "특정 조건의 행을 추출할 수 없다.",
            "슬라이싱은 지원하지 않는다."
        ],
        "answer": "df['col']은 열을 선택한다.",
        "why": "대괄호 `[]` 안에 컬럼명을 넣으면 해당 Series(열)를 선택합니다.",
        "hint": "Column Selection",
        "difficulty": "medium",
        "id": f"20{i:02d}"
    }
    if i % 2 == 0:
        q['question'] = "Pandas에서 인덱스 번호(0, 1, 2...)로 데이터에 접근하는 속성은?"
        q['options'] = ["ix", "at", "loc", "iloc", "idx"]
        q['answer'] = "iloc"
        q['why'] = "iloc stands for integer location."
        q['hint'] = "integer location"
    questions.append(q)

# 4. Preprocessing (20 MCQs)
for i in range(61, 81):
    q = {
        "chapter_name": chapter_name,
        "type": "객관식",
        "question": f"데이터 전처리에 관한 내용 {i}",
        "options": [
            "결측치는 무조건 삭제해야 한다.",
            "fillna()는 결측치를 채우는 함수다.",
            "데이터 타입 변환은 불가능하다.",
            "중복 데이터는 분석에 도움이 되므로 유지한다.",
            "이상치는 항상 평균값으로 대체한다."
        ],
        "answer": "fillna()는 결측치를 채우는 함수다.",
        "why": "fillna() 메서드를 사용하여 NaN 값을 특정 값으로 대체할 수 있습니다.",
        "hint": "채우다(Fill)",
        "difficulty": "medium",
        "id": f"20{i:02d}"
    }
    questions.append(q)

# 5. Advanced & Visualization (20 MCQs)
for i in range(81, 101):
    q = {
        "chapter_name": chapter_name,
        "type": "객관식",
        "question": f"고급 데이터 처리 및 시각화 {i}",
        "options": [
            "groupby는 데이터를 그룹별로 나눈다.",
            "merge는 데이터를 위아래로만 합친다.",
            "concat은 두 테이블을 Key 기준으로 병합한다.",
            "matplotlib은 3D 그래프를 그릴 수 없다.",
            "seaborn은 pandas와 호환되지 않는다."
        ],
        "answer": "groupby는 데이터를 그룹별로 나눈다.",
        "why": "groupby()는 특정 컬럼의 값을 기준으로 데이터를 그룹핑하여 집계 연산을 수행합니다.",
        "hint": "Group By",
        "difficulty": "hard",
        "id": f"20{i:02d}"
    }
    questions.append(q)

# 6. Code Completion (20 CC)
for i in range(1, 21):
    q = {
        "chapter_name": chapter_name,
        "type": "코드 완성형",
        "question": f"데이터 분석 코드를 완성하세요. (문제 {i})",
        "answer": "pd",
        "why": "보통 `import pandas as pd`로 사용합니다.",
        "hint": "pandas alias",
        "difficulty": "easy",
        "id": f"21{i:02d}"
    }
    if i % 4 == 0:
        q['question'] = "CSV 파일을 읽어오세요.\n```python\nimport pandas as pd\ndf = pd.____('data.csv')\n```"
        q['answer'] = "read_csv"
        q['why'] = "파일 읽기 함수는 read_csv입니다."
    elif i % 4 == 1:
         q['question'] = "처음 5개 행을 출력하세요.\n```python\ndf.____()\n```"
         q['answer'] = "head"
         q['why'] = "상위 행 출력은 head()입니다."
    elif i % 4 == 2:
         q['question'] = "결측치를 0으로 채우세요.\n```python\ndf.____(0)\n```"
         q['answer'] = "fillna"
         q['why'] = "결측치 채우기는 fillna입니다."
    elif i % 4 == 3:
         q['question'] = "기초 통계량을 확인하세요.\n```python\ndf.____()\n```"
         q['answer'] = "describe"
         q['why'] = "통계 요약은 describe()입니다."

    questions.append(q)

def get_questions():
    return questions
