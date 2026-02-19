
chapter_name = "데이터 분석"

questions = []

# --- 100 MCQs ---

# [1-15] Core NumPy & Pandas
q1_15 = [
    ("NumPy 배열(ndarray)과 파이썬 리스트의 가장 큰 차이점은?", ["리스트가 연산 속도가 더 빠르다.", "NumPy 배열은 모든 요소의 데이터 타입이 같아야 한다.", "NumPy 배열은 크기를 자유롭게 조절할 수 있다.", "리스트는 벡터 연산을 기본으로 지원한다.", "NumPy는 텍스트 데이터 처리에만 특화되어 있다."], "NumPy 배열은 모든 요소의 데이터 타입이 같아야 한다.", "NumPy 배열은 동일한 타입의 데이터를 메모리에 연속적으로 배치하여 고성능 연산을 가능하게 합니다.", "Homogeneous", "2001"),
    ("Pandas에서 1차원 데이터를 다루는 자료구조의 이름은?", ["DataFrame", "Series", "Array", "Dictionary", "List"], "Series", "Series는 인덱스를 가진 1차원 배열 형태의 자료구조입니다.", "1차원", "2002"),
    ("Pandas DataFrame에서 인덱스(이름)를 기준으로 행을 선택하는 메서드는?", ["iloc", "loc", "index", "get", "select"], "loc", "loc는 라벨(이름)을 기준으로, iloc는 정수 위치를 기준으로 선택합니다.", "label vs integer", "2003"),
    ("데이터 프레임의 상위 5개 행을 빠르게 확인하는 함수는?", ["tail()", "show()", "head()", "peek()", "top()"], "head()", "df.head()를 사용하면 데이터의 구조를 빠르게 파악할 수 있습니다.", "머리", "2004"),
    ("CSV 파일을 Pandas 데이터 프레임으로 읽어오는 함수는?", ["read_excel()", "load_csv()", "read_csv()", "get_csv()", "open_csv()"], "read_csv()", "가장 보편적인 데이터 로딩 함수입니다.", "csv", "2005"),
    ("데이터 프레임의 전체적인 정보(행 수, 컬럼 타입, 결측치 등)를 확인하는 메서드는?", ["describe()", "info()", "summary()", "check()", "stats()"], "info()", "df.info()는 요약된 구조 정보를 제공합니다.", "information", "2006"),
    ("수치형 데이터의 평균, 표준편차, 사분위수 등 기술 통계량을 보여주는 메서드는?", ["info()", "stats()", "describe()", "count()", "mean()"], "describe()", "수치 데이터의 분포를 한눈에 볼 때 유용합니다.", "통계 요약", "2007"),
    ("결측치(NaN)가 있는지 확인하는 메서드는?", ["isnull()", "exists()", "na_check()", "missing()", "check_null()"], "isnull()", "isnull() 또는 isna()를 사용합니다.", "null check", "2008"),
    ("결측치를 특정 값(예: 0)으로 채우는 메서드는?", ["dropna()", "fillna()", "replace()", "set_na()", "fixna()"], "fillna()", "비어있는 값을 적절한 값으로 보간할 때 씁니다.", "fill", "2009"),
    ("특정 컬럼을 기준으로 데이터를 그룹화하여 집계할 때 사용하는 메서드는?", ["pivot()", "merge()", "groupby()", "concatenate()", "aggregate()"], "groupby()", "카테고리별 합계, 평균 등을 구할 때 필수입니다.", "group", "2010"),
    ("두 개의 데이터 프레임을 특정 키를 기준으로 가로로 합치는 함수는?", ["concat()", "join()", "merge()", "append()", "add()"], "merge()", "SQL의 JOIN과 유사한 기능을 수행합니다.", "merge", "2011"),
    ("데이터 프레임의 특정 컬럼만 추출하여 리스트로 만들 때 적절한 접근법은?", ["df.tolist()", "df['col'].values.tolist()", "df.get_cols()", "list(df)", "df.columns()"], "df['col'].values.tolist()", "Series의 값을 NumPy 배열로 바꾼 뒤 리스트로 변환합니다.", "column to list", "2012"),
    ("NumPy에서 모든 요소가 0인 3x3 배열을 만드는 코드는?", ["np.zeros((3, 3))", "np.ones((3, 3))", "np.empty((3, 3))", "np.array(0)", "np.full(0, 3)"], "np.zeros((3, 3))", "np.zeros는 0으로 채워진 배열을 생성합니다.", "zeros", "2013"),
    ("Pandas에서 중복된 행을 제거하는 메서드는?", ["remove_duplicates()", "drop_duplicates()", "clear_duplicates()", "unique()", "delete_duplicates()"], "drop_duplicates()", "중복 데이터를 정제할 때 사용합니다.", "duplicates", "2014"),
    ("데이터 시각화를 위해 Pandas와 자주 함께 쓰이는 라이브러리는?", ["Django", "Requests", "Matplotlib", "PyTest", "BeautifulSoup"], "Matplotlib", "Matplotlib과 Seaborn이 대표적인 시각화 도구입니다.", "visualization", "2015")
]

for q, o, a, w, h, i in q1_15:
    questions.append({"chapter_name": chapter_name, "type": "객관식", "difficulty": "medium", "id": i, "question": q, "options": o, "answer": a, "why": w, "hint": h})

# [16-60] NumPy & Pandas Detail (45 items)
da_syntax = [
    ("브로드캐스팅", "크기가 다른 두 배열 간에도 특정 조건이 맞으면 연산이 가능합니다.", "2016"),
    ("벡터화 연산", "루프 없이 배열 전체에 대해 한 번에 연산을 수행하여 속도가 빠릅니다.", "2017"),
    ("np.arange", "특정 범위의 숫자를 배열 형태로 생성합니다.", "2018"),
    ("np.linspace", "특정 구간을 균등한 간격으로 나눈 배열을 생성합니다.", "2019"),
    ("Reshape", "배열의 요소 수는 유지하면서 차원의 모양을 바꿉니다.", "2020"),
    ("T (Transpose)", "행과 열을 서로 바꾼 전치 행렬을 만듭니다.", "2021"),
    ("수평 결합(hstack)", "배열을 가로 방향으로 이어 붙입니다.", "2022"),
    ("수직 결합(vstack)", "배열을 세로 방향으로 이어 붙입니다.", "2023"),
    ("Slicing (2D)", "arr[:2, 1:] 와 같이 행과 열 범위를 동시에 슬라이싱합니다.", "2024"),
    ("Boolean Indexing", "조건식(arr > 0)을 만족하는 요소만 선택합니다.", "2025"),
    ("Pandas Index", "행을 식별하는 라벨이며 수정 불가능한(Immutable) 객체입니다.", "2026"),
    ("Columns", "데이터 프레임의 열 이름을 관리하는 속성입니다.", "2027"),
    ("Set_index", "특정 컬럼을 인덱스로 지정합니다.", "2028"),
    ("Reset_index", "인덱스를 다시 0부터 시작하는 정수로 되돌립니다.", "2029"),
    ("Sort_values", "특정 컬럼의 값을 기준으로 행을 정렬합니다.", "2030"),
    ("Drop (Column)", "axis=1 옵션을 주어 특정 열을 제거합니다.", "2031"),
    ("Rename", "컬럼명이나 인덱스명을 사전 형식으로 전달하여 바꿉니다.", "2032"),
    ("Apply", "사용자 정의 함수를 데이터 프레임 행/열 전체에 적용합니다.", "2033"),
    ("Map (Series)", "Series의 요소를 일대일로 치환합니다.", "2034"),
    ("Applymap", "데이터 프레임의 모든 개별 원소에 함수를 적용합니다.", "2035"),
    ("Value_counts", "특정 컬럼에 있는 고유값들의 빈도를 계산합니다.", "2036"),
    ("Unique", "특정 컬럼의 중복 없는 고유값 목록을 반환합니다.", "2037"),
    ("Isin", "특정 리스트에 포함된 값들만 필터링합니다.", "2038"),
    ("Query", "문자열 쿼리문을 써서 데이터를 간편하게 필터링합니다.", "2039"),
    ("Where", "조건에 맞는 데이터는 유지하고 아니면 다른 값으로 채웁니다.", "2040"),
    ("Sample", "데이터 프레임에서 무작위로 샘플 추출을 수행합니다.", "2041"),
    ("Nlargest/Nsmallest", "상위/하위 N개의 데이터를 빠르게 가져옵니다.", "2042"),
    ("Correlation (corr)", "컬럼 간의 상관계수를 계산합니다.", "2043"),
    ("Covariance (cov)", "컬럼 간의 공분산을 계산합니다.", "2044"),
    ("Diff", "이전 행과의 값 차이를 계산합니다.", "2045"),
    ("Pct_change", "이전 행 대비 변화율을 계산합니다.", "2046"),
    ("Rolling", "이동 평균과 같은 윈도우 연산을 수행합니다.", "2047"),
    ("Expanding", "데이터가 누적되면서 통계량을 계산합니다.", "2048"),
    ("Shift", "데이터를 위아래로 한 칸씩 밉니다.", "2049"),
    ("Rank", "데이터의 순위를 매깁니다.", "2050"),
    ("Category Type", "반복되는 텍스트를 Category 타입으로 바꾸면 메모리가 절약됩니다.", "2051"),
    ("Datetime (To_datetime)", "문자열 날짜를 날짜 객체로 변환합니다.", "2052"),
    ("Dt Accessor", "날짜 객체에서 연, 월, 일, 요일 등을 추출합니다.", "2053"),
    ("MultiIndex", "두 개 이상의 컬럼을 계층적 인덱스로 사용합니다.", "2054"),
    ("Stack/Unstack", "인덱스와 컬럼의 위치를 서로 맞바꾸며 형태를 변경합니다.", "2055"),
    ("Pivot_table", "엑셀의 피벗 테이블처럼 데이터를 요약 재구성합니다.", "2056"),
    ("Melt", "옆으로 넓은 데이터를 아래로 긴 데이터로 재구조화합니다.", "2057"),
    ("Concat (Axis=0)", "여러 데이터 프레임을 위아래로 단순 연결합니다.", "2058"),
    ("Join (Left/Right)", "인덱스를 기준으로 사전식 결합을 수행합니다.", "2059"),
    ("Type Conversion (astype)", "데이터 프레임 컬럼의 데이터 타입을 강제로 변환합니다.", "2060")
]

for title, fact, i in da_syntax:
    questions.append({
        "chapter_name": chapter_name, "type": "객관식", "difficulty": "medium", "id": i,
        "question": f"Pandas/NumPy의 {title} 기능에 대한 설명으로 옳은 코드는?",
        "options": [fact, "잘못된 문법 예시", "데이터가 손실되는 위험한 코드", "하위 버전에서만 동작하는 기법", "다른 언어(R)의 문법"],
        "answer": fact,
        "why": f"{title}은 {fact}",
        "hint": title
    })

# [61-100] Practical Scenarios & Analysis (40 items)
scenarios = [
    ("데이터 정제", "결측치가 너무 많은 행은 dropna(thresh=n)로 제거하는 전략을 씁니다.", "2061"),
    ("이상치 탐지", "사분위수(IQR) 방식을 사용하여 데이터의 튀는 값을 찾아냅니다.", "2062"),
    ("정규화(Normalization)", "데이터의 범위를 0~1 사이로 맞추어 스케일을 조정합니다.", "2063"),
    ("표준화(Standardization)", "평균을 0, 표준편차를 1로 만들어 분포를 고르게 합니다.", "2064"),
    ("원-핫 인코딩", "범주형 데이터를 0과 1로 이루어진 여러 컬럼으로 변환합니다.", "2065"),
    ("로그 변환", "데이터의 편향(Skewness)이 심할 때 로그를 취해 완화합니다.", "2066"),
    ("시계열 리샘플링", "일 단위 데이터를 월 단위 평균 등으로 요약할 때 resample()을 씁니다.", "2067"),
    ("데이터 시각화 (Line)", "시계열 트렌드를 볼 때는 선 그래프가 적합합니다.", "2068"),
    ("데이터 시각화 (Bar)", "범주별 크기를 비교할 때는 막대 그래프가 적합합니다.", "2069"),
    ("데이터 시각화 (Scatter)", "두 변수 간의 상관관계를 볼 때는 산점도가 적합합니다.", "2070"),
    ("히스토그램", "연속형 데이터의 빈도 분포를 확인할 때 사용합니다.", "2071"),
    ("박스 플롯", "데이터의 사분위수와 이상치를 시각적으로 확인하기 좋습니다.", "2072"),
    ("히트맵", "변수 간의 상관계수 행렬을 색깔로 표현하여 분석합니다.", "2073"),
    ("메모리 최적화", "불필요한 컬럼을 삭제하고 적절한 dtype을 선택하여 메모리를 아낍니다.", "2074"),
    ("대용량 데이터(Chunk)", "메모리가 부족할 때 read_csv에 chunksize를 주어 끊어 읽습니다.", "2075"),
    ("병렬 처리", "Pandas는 기본적으로 싱글 코어를 쓰므로 큰 작업은 병렬화를 고려해야 합니다.", "2076"),
    ("데이터 무결성", "중복 체크와 결측치 검증은 분석의 가장 첫 단계입니다.", "2077"),
    ("탐색적 분석(EDA)", "데이터의 특성을 파악하기 위해 통계와 시각화를 병행하는 것입니다.", "2078"),
    ("파이프라인 구축", "데이터 수집-정제-분석-결과 저장의 흐름을 자동화합니다.", "2079"),
    ("결과 저장 (to_csv)", "분석이 끝난 데이터를 파일로 저장할 때 index=False 옵션을 자주 씁니다.", "2080"),
    ("SQL 연동", "pd.read_sql()을 사용하여 DB에서 쿼리로 직접 데이터를 가져옵니다.", "2081"),
    ("Excel 연동", "여러 시트를 가진 엑셀 파일을 읽으려면 ExcelFile 클래스를 씁니다.", "2082"),
    ("JSON 연동", "웹 API 결과물인 JSON 데이터를 프레임으로 변환합니다.", "2083"),
    ("데이터 타입 추론", "Pandas는 데이터를 읽을 때 자동으로 타입을 추론하지만 명시해주는 것이 안전합니다.", "2084"),
    ("인덱스 정렬", "sort_index()를 사용하여 뒤섞인 인덱스를 순서대로 만듭니다.", "2085"),
    ("부분 합계", "groupby와 transform을 조합하여 전체 대비 비중 등을 구합니다.", "2086"),
    ("행/열 전환 (swapaxes)", "데이터의 축을 강제로 변경해야 할 때 사용합니다.", "2087"),
    ("Clip", "데이터가 특정 임계값을 넘지 못하도록 범위를 제한합니다.", "2088"),
    ("Tidy Data", "분석하기 좋은 '깔끔한 데이터' 형태를 유지하는 것이 중요합니다.", "2089"),
    ("Wide vs Long", "시각화 툴에 따라 데이터 포맷을 Wide에서 Long으로 변환합니다.", "2090"),
    ("Feature Engineering", "기존 컬럼들을 조합하여 분석에 더 유용한 새 컬럼을 만드는 것입니다.", "2091"),
    ("데이터 마스킹", "특정 조건에 맞는 데이터만 은닉하거나 마킹할 때 씁니다.", "2092"),
    ("Vectorized String", "df['col'].str.contains() 처럼 문자열 연산도 벡터화 가능합니다.", "2093"),
    ("Datetime Delta", "날짜 간의 차이를 계산하면 TimeDelta 객체가 생성됩니다.", "2094"),
    ("유효성 검사", "데이터가 비즈니스 로직에 맞게 들어왔는지 코드로 검사합니다.", "2095"),
    ("비식별화", "개인정보 보호를 위해 특정 컬럼을 해싱하거나 삭제합니다.", "2096"),
    ("데이터 분포 확인", "Skew()나 Kurt() 함수로 통계적 치우침을 확인합니다.", "2097"),
    ("Cumulative Sum", "시간 흐름에 따른 누적 매출 등을 계산할 때 cumsum()을 씁니다.", "2098"),
    ("데이터 중첩 해제", "리스트가 들어있는 컬럼을 개별 행으로 펼칠 때 explode()를 씁니다.", "2099"),
    ("분석 보고서 작성", "Jupyter Notebook에 Markdown과 코드를 섞어 결론을 도출합니다.", "2100")
]

for title, fact, i in scenarios:
    questions.append({
        "chapter_name": chapter_name, "type": "객관식", "difficulty": "hard", "id": i,
        "question": f"실무 {title} 시나리오에서 가장 올바른 결정은?",
        "options": [fact, "부정확한 결과를 초래하는 방식", "시간이 너무 오래 걸리는 비효율적 방식", "데이터를 유실할 위험이 큰 방식", "협업 시 혼란을 주는 방식"],
        "answer": fact,
        "why": f"{title} 시나리오 핵심: {fact}",
        "hint": title
    })

# --- 20 Code Completion Questions ---
cc_data = [
    ("데이터 로드", "import pandas as pd\ndf = pd.____('data.csv')", "read_csv", "CSV를 읽는 함수입니다."),
    ("정보 확인", "df.____() # 요약 정보", "info", "구조 확인 메서드입니다."),
    ("인덱스 선택", "df.____['row_name']", "loc", "라벨 기준 선택입니다."),
    ("정수 선택", "df.____[0]", "iloc", "위치 기준 선택입니다."),
    ("결측치 채우기", "df.____(0, inplace=True)", "fillna", "NaN을 채우는 메서드입니다."),
    ("결측치 제거", "df.____(axis=0)", "dropna", "NaN이 포함된 행을 지웁니다."),
    ("데이터 그룹화", "df.____('city').mean()", "groupby", "기준별 집계 메서드입니다."),
    ("데이터 병합", "pd.____(df1, df2, on='key')", "merge", "기준 열 이용 병합입니다."),
    ("데이터 연결", "pd.____([df1, df2], axis=0)", "concat", "단순 붙이기 함수입니다."),
    ("컬럼 삭제", "df.____(['age'], axis=1)", "drop", "삭제 메서드입니다."),
    ("통계 요약", "df.____() # 8-point summary", "describe", "통계치 요약입니다."),
    ("빈도 계산", "df['color'].____()", "value_counts", "고유값 빈도 세기입니다."),
    ("중복 확인", "df.____() # True if repeated", "duplicated", "중복 여부 불리언 체크입니다."),
    ("이름 변경", "df.____(columns={'OLD':'NEW'})", "rename", "컬럼명 변경 메서드입니다."),
    ("값 치환", "df['sex'].____({'M':1, 'F':0})", "map", "일대일 변환 메서드입니다."),
    ("정렬", "df.____(by='price', ascending=False)", "sort_values", "값 기준 정렬입니다."),
    ("NumPy 배열", "import ____ as np", "numpy", "NumPy 별칭 관례입니다."),
    ("00 배열", "np.____((2, 2))", "zeros", "0으로 된 배열 생성입니다."),
    ("타입 변경", "df['score'].____(float)", "astype", "형변환 메서드입니다."),
    ("샘플링", "df.____(n=5)", "sample", "추출 함수입니다.")
]

for i, (title, code, ans, explain) in enumerate(cc_data):
    questions.append({
        "chapter_name": chapter_name, "type": "코드 완성형", "difficulty": "medium", "id": str(2101 + i),
        "question": f"{title} 코드를 완성하세요.\n```python\n{code}\n```",
        "answer": ans,
        "why": explain,
        "hint": title,
    })

def get_questions():
    return questions
