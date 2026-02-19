
chapter_name = "데이터 분석"

questions = []

# --- 100 MCQs ---
# Unique conceptual and practical questions based on 2.md

mcq_data = [
    # 1. NumPy 기초 (1-20)
    ("NumPy 라이브러리의 주요 목적이 아닌 것은?", ["대규모 다차원 배열 처리를 지원한다.", "파이썬 리스트보다 수치 연산 속도가 빠르다.", "고수준 머신러닝 알고리즘의 기반이 된다.", "데이터 프레임을 이용하여 SQL 쿼리를 직접 수행한다.", "강력한 벡터 연산(Vectorization) 기능을 제공한다."], "데이터 프레임을 이용하여 SQL 쿼리를 직접 수행한다.", "데이터 프레임과 SQL 스타일의 쿼리는 주로 Pandas의 역할입니다. NumPy는 수치 행렬 연산에 집중합니다.", "NumPy의 역할", "2001"),
    ("NumPy 배열(ndarray)의 특징으로 옳은 것은?", ["서로 다른 자료형의 데이터를 한 배열에 담을 수 있다.", "모든 요소는 반드시 동일한 자료형(dtype)이어야 한다.", "인덱싱과 슬라이싱 기능이 전혀 없다.", "파이썬 리스트보다 메모리를 더 많이 소모한다.", "데이터 수정이 불가능한 불변(Immutable) 객체이다."], "모든 요소는 반드시 동일한 자료형(dtype)이어야 한다.", "NumPy 배열은 연속된 메모리 배치를 위해 동일한 데이터 타입을 요구하며, 이를 통해 고속 연산을 가능하게 합니다.", "ndarray 특징", "2002"),
    ("파이썬 리스트를 NumPy 배열로 변환하는 함수는?", ["np.tolist()", "np.convert()", "np.array()", "np.make_array()", "np.asarray_list()"], "np.array()", "np.array() 함수는 리스트나 다른 시퀀스 데이터를 ndarray로 변환합니다.", "배열 생성", "2003"),
    ("모든 요소가 0으로 채워진 크기 10의 배열을 만드는 코드는?", ["np.zeros(10)", "np.ones(10)", "np.empty(10)", "np.full(0, 10)", "np.range(0, 10)"], "np.zeros(10)", "np.zeros()는 지정한 크기만큼 0으로 초기화된 배열을 생성합니다.", "zeros", "2004"),
    ("NumPy에서 `np.arange(0, 10, 2)`를 실행한 결과는?", ["[0, 2, 4, 6, 8]", "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]", "[2, 4, 6, 8, 10]", "[0, 2, 4, 6, 8, 10]", "[0, 1, 2]"], "[0, 2, 4, 6, 8]", "range와 유사하게 start(0)부터 end(10) 미만까지 step(2) 간격으로 생성합니다.", "arange", "2005"),
    ("배열의 각 요소에 대해 반복문 없이 연산을 적용하는 기법의 명칭은?", ["Sequential Mapping", "Vectorization (벡터화)", "Normalization", "Looping", "Direct Access"], "Vectorization (벡터화)", "벡터화 연산을 통해 코드가 짧아지고 실행 속도가 비약적으로 향상됩니다.", "벡터화", "2006"),
    ("NumPy 배열 `arr = np.array([1, 2, 3])` 에 `arr * 10`을 수행한 결과는?", ["[10, 2, 3]", "[1, 2, 3, 10]", "[10, 20, 30]", "[11, 12, 13]", "에러가 발생한다"], "[10, 20, 30]", "NumPy는 배열과 스칼라 값 간의 요소별(Element-wise) 연산을 지원합니다.", "요소별 연산", "2007"),
    ("배열의 현재 모양(차원)을 확인하는 속성은?", ["arr.size", "arr.ndim", "arr.shape", "arr.form", "arr.type"], "arr.shape", "shape 속성은 각 차원의 크기를 튜플 형태로 보여줍니다.", "shape", "2008"),
    ("배열의 데이터 타입을 확인하는 속성은?", ["arr.type", "arr.dtype", "arr.kind", "arr.class", "arr.format"], "arr.dtype", "dtype은 데이터가 int, float 등 어떤 타입인지 나타냅니다.", "dtype", "2009"),
    ("1차원 배열을 2차원(예: 2x5)으로 재구성할 때 쓰는 메서드는?", ["arr.reform(2, 5)", "arr.resize(2, 5)", "arr.reshape(2, 5)", "arr.change(2, 5)", "arr.align(2, 5)"], "arr.reshape(2, 5)", "reshape()는 전체 요소 개수가 유지되는 한도 내에서 차원을 바꿉니다.", "reshape", "2010"),
    ("NumPy에서 두 배열을 세로로(위아래로) 합치는 함수는?", ["np.hstack()", "np.vstack()", "np.concatenate(axis=1)", "np.merge()", "np.combine()"], "np.vstack()", "Vertical Stack의 약자인 vstack()은 수직 방향으로 배열을 쌓습니다.", "vstack", "2011"),
    ("NumPy 배열에서 조건에 맞는 요소만 추출하는 기법은?", ["Condition Indexing", "Boolean Indexing", "Bitwise Filtering", "Select Index", "Logical Slicing"], "Boolean Indexing", "대괄호 안에 조건식을 넣어 True인 요소만 필터링할 수 있습니다.", "불리언 인덱싱", "2012"),
    ("NumPy의 `np.mean(arr)` 함수가 계산하는 값은?", ["합계", "최댓값", "평균값", "중앙값", "표준편차"], "평균값", "mean()은 산술 평균을 구하는 통계 함수입니다.", "평균", "2013"),
    ("배열 내 모든 요소의 합을 구하는 함수는?", ["np.total()", "np.add_all()", "np.sum()", "np.plus()", "np.integrate()"], "np.sum()", "sum()은 축(axis)에 따른 합계 연산을 지원합니다.", "합계", "2014"),
    ("배열에서 가장 큰 값의 '인덱스'를 찾는 함수는?", ["np.max()", "np.argmax()", "np.find_max()", "np.top_index()", "np.high()"], "np.argmax()", "argmax()는 최댓값이 위치한 위치 번호를 반환합니다.", "argmax", "2015"),
    ("배열의 모든 요소에 동일한 상수를 더했을 때 각 요소가 모두 커지는 현상을 지원하는 기능?", ["Broadcasting (브로드캐스팅)", "Duplication", "Magnification", "Streaming", "Auto-increment"], "Broadcasting (브로드캐스팅)", "서로 모양이 다른 배열 간의 연산을 가능하게 하는 NumPy의 핵심 메커니즘입니다.", "브로드캐스팅", "2016"),
    ("NumPy에서 난수를 생성하는 서브 모듈 이름은?", ["np.random", "np.chance", "np.guess", "np.stat", "np.variable"], "np.random", "random 모듈 내에 rand, randn, randint 등의 함수가 있습니다.", "random", "2017"),
    ("행과 열을 서로 맞바꾸는(전치) 연산자 또는 메서드는?", ["arr.flip()", "arr.swap()", "arr.T", "arr.reverse()", "arr.turn()"], "arr.T", ".T(Transpose) 속성을 사용하면 간단히 행과 열이 바뀝니다.", "전치", "2018"),
    ("1차원 배열의 순서를 거꾸로 뒤집는 대표적인 슬라이싱 방식은?", ["arr[::]", "arr[::-1]", "arr[0:0]", "arr[1:0]", "arr[-1:1]"], "arr[::-1]", "step에 -1을 주면 역순으로 슬라이싱합니다.", "역순 슬라이싱", "2019"),
    ("`np.ones((2, 3))` 명령어로 만들어지는 배열의 총 요소 개수는?", ["5개", "6개", "2개", "3개", "1개"], "6개", "2행 3열의 배열이므로 2 * 3 = 6개의 요소가 생성됩니다.", "배열 크기", "2020"),

    # 2. Pandas 기초 (21-40)
    ("Pandas 라이브러리의 핵심적인 목적은?", ["GPU를 이용한 딥러닝 연산", "테이플 형태의 정형 데이터 분석과 처리", "웹 디자인 및 UI 구현", "네트워크 패킷 분석", "운영체제 리소스 모니터링"], "테이플 형태의 정형 데이터 분석과 처리", "Pandas는 엑셀이나 SQL과 같은 행/열 구조의 데이터를 다루는 데 최적화되어 있습니다.", "Pandas 목적", "2021"),
    ("Pandas에서 인덱스를 가진 1차원 배열 형태의 자료구조는?", ["DataFrame", "Series", "Record", "Vector", "Column"], "Series", "Series는 하나의 컬럼 데이터를 담을 수 있는 1차원 구조입니다.", "Series", "2022"),
    ("행과 열이 있는 2차원 표 형태의 자료구조는?", ["DataMatrix", "DataTable", "DataFrame", "DataGrid", "DataSpread"], "DataFrame", "DataFrame은 여러 개의 Series가 합쳐진 2차원 구조입니다.", "DataFrame", "2023"),
    ("Pandas에서 CSV 파일을 불러올 때 쓰는 함수는?", ["pd.open_csv()", "pd.read_csv()", "pd.load_csv()", "pd.from_csv()", "pd.get_csv()"], "pd.read_csv()", "read_csv()는 다양한 옵션을 통해 CSV 파일을 데이터프레임으로 변환합니다.", "read_csv", "2024"),
    ("데이터프레임의 상위 5개 행을 조회하는 메서드는?", ["df.top()", "df.head()", "df.first()", "df.show()", "df.peek()"], "df.head()", "데이터의 구조를 대략적으로 파악할 때 가장 먼저 사용됩니다.", "head", "2025"),
    ("데이터프레임의 전체 행 수, 컬럼명, 데이터 타입 및 결측치를 요약 확인하는 메서드는?", ["df.describe()", "df.info()", "df.summary()", "df.check()", "df.status()"], "df.info()", "info() 메서드는 데이터프레임의 메타데이터를 요약해 보여줍니다.", "info", "2026"),
    ("수치형 컬럼들의 평균, 표준편차, 최솟값, 최댓값 등 통계량을 요약해주는 메서드는?", ["df.info()", "df.stats()", "df.describe()", "df.stat_summary()", "df.math()"], "df.describe()", "기술 통계량을 한눈에 확인할 때 유용합니다.", "describe", "2027"),
    ("데이터프레임에서 'name'이라는 컬럼 하나만 선택하는 올바른 문법은?", ["df.select('name')", "df['name']", "df{name}", "df : name", "df.get_rows('name')"], "df['name']", "딕셔너리와 유사한 키 방식을 사용하여 특정 열(Series)을 추출합니다.", "열 선택", "2028"),
    ("라벨(이름)을 기준으로 데이터의 행이나 열을 선택하는 메서드는?", ["df.iloc", "df.loc", "df.idx", "df.label", "df.select"], "df.loc", "loc는 location의 약자로 행/열의 이름을 기준으로 접근합니다.", "loc", "2029"),
    ("정수 위치(Index 번호)를 기준으로 데이터를 선택하는 메서드는?", ["df.loc", "df.iloc", "df.slice", "df.pos", "df.point"], "df.iloc", "iloc는 integer location의 약자로 0부터 시작하는 숫자로 접근합니다.", "iloc", "2030"),
    ("나이가 20세 이상인 데이터만 필터링하는 파이썬 코드는?", ["df.filter(age >= 20)", "df[df['age'] >= 20]", "df.where('age' >= 20)", "df.get(age > 20)", "df{age >= 20}"], "df[df['age'] >= 20]", "불리언 마스크를 대괄호 안에 넣어 조건에 맞는 행을 추출합니다.", "필터링 문법", "2031"),
    ("Pandas에서 결측치(Null, NaN)를 확인하는 메서드는?", ["df.ismissing()", "df.isnull()", "df.none()", "df.empty()", "df.check_na()"], "df.isnull()", "df.isnull() 또는 df.isna()를 사용하여 결측 위치를 찾습니다.", "결측치 확인", "2032"),
    ("결측치가 포함된 '행'을 통째로 삭제해버리는 메서드는?", ["df.remove_na()", "df.delete_na()", "df.clear_na()", "df.dropna()", "df.cut_na()"], "df.dropna()", "데이터 정제 시 불완전한 기록을 지우고자 할 때 사용합니다.", "dropna", "2033"),
    ("결측치를 특정 값(예: 0이나 평균)으로 채워 넣는 메서드는?", ["df.refill()", "df.fixna()", "df.fillna()", "df.replace_na()", "df.put_na()"], "df.fillna()", "결측치를 다른 데이터로 보정하여 분석을 계속할 때 씁니다.", "fillna", "2034"),
    ("특정 컬럼의 데이터 타입을 강제로 바꾸는 메서드는?", ["df.convert()", "df.to_type()", "df.astype()", "df.format()", "df.change_type()"], "df.astype()", "예를 들어 문자열로 된 숫자를 숫자형으로 바꿀 때 자주 쓰입니다.", "astype", "2035"),
    ("데이터프레임의 특정 컬럼에 있는 고유값들의 빈도수를 세는 메서드는?", ["df.count()", "df.unique_count()", "df.value_counts()", "df.group_count()", "df.freq()"], "df.value_counts()", "범주형 데이터의 분포를 파악할 때 필수적입니다.", "value_counts", "2036"),
    ("컬럼명을 'old_name'에서 'new_name'으로 바꾸는 올바른 메서드는?", ["df.rename(columns={'old_name': 'new_name'})", "df.replace_column('old_name', 'new_name')", "df.columns = 'new_name'", "df.set_name('old_name', 'new_name')", "df.change_header(...)"], "df.rename(columns={'old_name': 'new_name'})", "rename() 메서드에 딕셔너리 형태로 전달하여 이름을 바꿉니다.", "rename", "2037"),
    ("데이터프레임을 특정 컬럼 기준으로 정렬하는 메서드는?", ["df.align()", "df.arrange()", "df.sort_values()", "df.order_by()", "df.reorder()"], "df.sort_values()", "by 인자에 기준 컬럼을, ascending에 정렬 순서를 지정합니다.", "sort_values", "2038"),
    ("데이터프레임에서 불필요한 행이나 열을 삭제하는 메서드는?", ["df.remove()", "df.delete()", "df.drop()", "df.cut()", "df.exclude()"], "df.drop()", "axis=0은 행, axis=1은 열을 삭제합니다.", "drop", "2039"),
    ("중복된 행을 찾아 제거하는 메서드는?", ["df.remove_duplicates()", "df.drop_duplicates()", "df.clear_duplicates()", "df.unique()", "df.filter_duplicates()"], "df.drop_duplicates()", "완전히 똑같은 행이 여러 번 등장할 때 하나만 남깁니다.", "중복 제거", "2040"),

    # 3. 데이터 전처리 및 고급 활용 (41-70)
    ("데이터프레임의 행이나 열에 사용자 정의 함수를 일괄 적용하는 메서드는?", ["df.each()", "df.apply()", "df.map()", "df.run()", "df.execute()"], "df.apply()", "복잡한 계산 로직을 컬럼 전체에 적용할 때 매우 유용합니다.", "apply", "2041"),
    ("데이터를 특정 컬럼의 값에 따라 그룹으로 묶어주는 메서드는?", ["df.pivot()", "df.groupby()", "df.aggregate()", "df.cluster()", "df.section()"], "df.groupby()", "그룹별 통계(합계, 평균 등)를 낼 때 가장 많이 쓰입니다.", "groupby", "2042"),
    ("특정 기준으로 그룹화한 뒤 평균을 구하는 올바른 코드는?", ["df.groupby('A').average()", "df.groupby('A').mean()", "df.cluster('A').mean()", "df.group('A')[mean]", "df.summarize('A', mean)"], "df.groupby('A').mean()", "groupby 객체에 바로 통계 함수를 연결하여 사용합니다.", "그룹 집계", "2043"),
    ("두 데이터프레임을 위아래 혹은 옆으로 단순 연결(붙이기)하는 함수는?", ["pd.merge()", "pd.join()", "pd.concat()", "pd.append_all()", "pd.attach()"], "pd.concat()", "Concatenate의 약자로 테이블을 단순히 이어 붙일 때 씁니다.", "concat", "2044"),
    ("두 데이터프레임을 공통 키(Key)를 기준으로 합치는(Join) 함수는?", ["pd.combine()", "pd.merge()", "pd.link()", "pd.connect()", "pd.zip_tables()"], "pd.merge()", "SQL의 JOIN 연산과 동일한 기능을 수행합니다.", "merge", "2045"),
    ("Pandas merge에서 교집합(Inner Join)이 아닌 '합집합' 결과를 얻는 방식은?", ["how='inner'", "how='left'", "how='outer'", "how='union'", "how='full'"], "how='outer'", "outer 옵션을 주면 양쪽 데이터 모두를 유지하며 합칩니다.", "outer join", "2046"),
    ("데이터프레임의 모든 문자열을 소문자로 바꾸기 위해 컬럼 `df['name']` 에 적용하는 도구는?", ["df['name'].lower()", "df['name'].str.lower()", "df['name'].to_lower()", "df['name'].apply(low)", "pd.lower(df['name'])"], "df['name'].str.lower()", ".str 접근자를 통해 문자열 전용 함수를 벡터화하여 적용합니다.", "str 접근자", "2047"),
    ("시계열 데이터 처리를 위해 문자열 날짜를 Timestamp 객체로 바꾸는 함수는?", ["pd.to_date()", "pd.to_datetime()", "pd.parse_time()", "pd.convert_time()", "pd.make_time()"], "pd.to_datetime()", "문자열 형태의 날짜 데이터를 분석 가능한 날짜 형식으로 변환합니다.", "to_datetime", "2048"),
    ("데이터프레임의 행과 열을 맞바꾸는 속성은?", ["df.reverse", "df.swap", "df.T", "df.flip", "df.rotate"], "df.T", "NumPy와 마찬가지로 .T 속성을 사용하여 전치(Transpose)합니다.", "전치", "2049"),
    ("엑셀의 피벗 테이블과 같이 데이터를 요약 재구성하는 메서드는?", ["df.summary()", "df.pivot_table()", "df.reshape()", "df.cube()", "df.cross_tab()"], "df.pivot_table()", "인덱스, 컬럼, 값, 집계 함수를 지정하여 유연하게 표를 만듭니다.", "pivot_table", "2050"),
    ("Wide 포맷 데이터를 Long 포맷으로 녹이듯이 변환하는 메서드는?", ["df.melt()", "df.freeze()", "df.boil()", "df.squeeze()", "df.expand()"], "df.melt()", "정리되지 않은(Un-tidy) 데이터를 분석하기 좋게 변환할 때 씁니다.", "melt", "2051"),
    ("특정 조건에 맞는 행을 '문자열 쿼리' 형식으로 추출하는 메서드는?", ["df.select()", "df.query()", "df.search()", "df.find()", "df.fetch()"], "df.query()", "`df.query('age > 20')` 처럼 가독성 좋은 쿼리 작성이 가능합니다.", "query", "2052"),
    ("데이터프레임에서 무작위로 n개의 샘플을 추출하는 메서드는?", ["df.random()", "df.sample()", "df.pick()", "df.extract()", "df.take()"], "df.sample()", "대용량 데이터의 일부만 미리 볼 때나 샘플링 시 사용합니다.", "sample", "2053"),
    ("결측치가 아닌 데이터의 개수만 세는 메서드는?", ["df.length()", "df.count()", "df.size()", "df.num()", "df.exist_count()"], "df.count()", "전체 행 수와 비교하여 누락된 데이터가 얼마나 되는지 가늠할 때 씁니다.", "count", "2054"),
    ("데이터프레임의 한 컬럼을 인덱스로 지정하는 메서드는?", ["df.set_label()", "df.make_index()", "df.set_index()", "df.use_index()", "df.apply_index()"], "df.set_index()", "특정 고유 ID 컬럼 등을 인덱스로 쓸 때 사용합니다.", "set_index", "2055"),
    ("인덱스로 설정된 값을 다시 일반 컬럼으로 되돌리는 메서드는?", ["df.clear_index()", "df.reset_index()", "df.unset_index()", "df.fix_index()", "df.back_index()"], "df.reset_index()", "인덱스를 0부터 시작하는 숫자로 초기화하고 기존 인덱스는 컬럼으로 보냅니다.", "reset_index", "2056"),
    ("특정 컬럼의 고유값 목록(List)만 뽑아내는 메서드는?", ["df['col'].values", "df['col'].unique()", "df['col'].list()", "df['col'].distinct()", "df['col'].only()"], "df['col'].unique()", "SQL의 DISTINCT와 유사하게 중복 없는 값들을 반환합니다.", "unique", "2057"),
    ("상위 n개 데이터를 가져오는 head()의 반대 기능(하위 n개)은?", ["df.bottom()", "df.end()", "df.tail()", "df.back()", "df.final()"], "df.tail()", "데이터의 마지막 부분(꼬리)을 확인할 때 씁니다.", "tail", "2058"),
    ("수치 데이터를 일정한 구간(Category)으로 나누는(Binning) 기술은?", ["pd.cut()", "pd.divide()", "pd.split_range()", "pd.bin()", "pd.section()"], "pd.cut()", "연속형 변수를 범주형 변수로 변환할 때(예: 점수 -> 등급) 사용합니다.", "cut", "2059"),
    ("데이터프레임에서 컬럼 이름들만 리스트 형태로 확인하는 속성은?", ["df.names", "df.headers", "df.columns", "df.fields", "df.labels"], "df.columns", "컬럼명 전체 리스트를 인덱스 객체 형태로 반환합니다.", "columns", "2060"),

    # 4. 시각화 및 실무 시나리오 (71-100)
    ("파이썬 시각화의 가장 근간이 되는 기초 라이브러리는?", ["Seaborn", "Plotly", "Matplotlib", "Bokeh", "Folium"], "Matplotlib", "대부분의 파이썬 시각화 도구는 Matplotlib을 기반으로 확장되었습니다.", "Matplotlib", "2061"),
    ("Matplotlib을 기반으로 더 세련된 디자인과 통계 기능을 제공하는 라이브러리는?", ["PyPlot", "Seaborn", "GraphViz", "VisualPy", "ArtPandas"], "Seaborn", "Pandas 데이터프레임과 호환성이 뛰어나며 통계 그래프 작성에 최적입니다.", "Seaborn", "2062"),
    ("데이터의 분포(전체적인 흐름)를 막대 형태로 보여주는 그래프는?", ["Line Plot", "Bar Chart", "Histogram", "Scatter Plot", "Pie Chart"], "Histogram", "연속형 변수의 구간별 빈도를 나타내는 데 쓰입니다.", "히스토그램", "2063"),
    ("두 수치형 변수 간의 관계(상관관계)를 점으로 찍어서 표현하는 그래프는?", ["Line Plot", "Box Plot", "Scatter Plot (산점도)", "Bar Plot", "Heatmap"], "Scatter Plot (산점도)", "두 변수가 서로 어떤 방향으로 움직이는지 파악할 때 유용합니다.", "산점도", "2064"),
    ("시간의 흐름에 따른 데이터의 변화(트렌드)를 보기 좋은 그래프는?", ["Line Plot (선 그래프)", "Pie Chart", "Histogram", "Box Plot", "Bar Plot"], "Line Plot (선 그래프)", "시계열 데이터를 분석할 때 가장 기본적으로 쓰입니다.", "선 그래프", "2065"),
    ("데이터의 중앙값, 사분위수, 이상치(Outlier)를 한눈에 식별하기 좋은 그래프는?", ["Histogram", "Violin Plot", "Box Plot", "Pie Chart", "Line Chart"], "Box Plot", "데이터의 통계적 분포와 외딴값들을 찾을 때 강력합니다.", "박스 플롯", "2066"),
    ("여러 변수 간의 상관계수를 색상으로 표현하여 시각화하는 도구는?", ["Bar Chart", "Heatmap (히트맵)", "Dot Plot", "Area Chart", "Radar Chart"], "Heatmap (히트맵)", "변수가 많을 때 어떤 것들이 서로 밀접한지 색으로 보여줍니다.", "히트맵", "2067"),
    ("데이터 분석 실무에서 전처리(Preprocessing) 단계가 차지하는 중요도는?", ["전체 과정의 10% 미만이다.", "거의 중요하지 않고 모델링이 전부다.", "가장 많은 시간과 노력이 소모되는 핵심 단계이다.", "마지막 보고서 쓸 때만 필요하다.", "컴퓨터가 알아서 해주므로 신경 쓸 필요 없다."], "가장 많은 시간과 노력이 소모되는 핵심 단계이다.", "쓰레기를 넣으면 쓰레기가 나온다는 원칙에 따라 데이터 정제가 가장 중요합니다.", "전처리의 중요성", "2068"),
    ("이상치(Outlier)를 처리하는 가장 일반적인 방법이 아닌 것은?", ["데이터에서 제거한다.", "중앙값이나 평균값으로 대체한다.", "별도의 분석 대상으로 분리한다.", "모든 데이터를 이상치에 맞춰 강제로 조정한다.", "로그 변환 등을 통해 영향력을 줄인다."], "모든 데이터를 이상치에 맞춰 강제로 조정한다.", "이상치 때문에 전체 데이터를 왜곡시키는 것은 잘못된 분석입니다.", "이상치 처리", "2069"),
    ("데이터 분석 시 '결측치'가 발생하는 주된 원인이 아닌 것은?", ["입력자의 실수나 누락", "센서 오작동", "설문 응답 거부", "의도적인 데이터 암호화", "시스템 전송 오류"], "의도적인 데이터 암호화", "암호화는 데이터 보안을 위한 것이지 시스템상 누락된 결측치와는 다릅니다.", "결측치 원인", "2070"),
    ("범주형(Categorical) 변수를 수치화할 때 자주 쓰이는 방식은?", ["Random Scaling", "One-Hot Encoding", "Binary Sum", "Value Mapping (1, 2, 3...)", "One-Hot Encoding 및 Value Mapping"], "One-Hot Encoding 및 Value Mapping", "모델이 이해할 수 있도록 범주를 숫자로 변환하는 필수 과정입니다.", "인코딩 기법", "2071"),
    ("데이터의 편향(Skewness)을 줄이기 위해 원본 데이터에 취하는 수학적 연산은?", ["더하기 100", "로그(Log) 변환", "곱하기 2", "나누기 10", "제곱근(Sqrt) 변환"], "로그(Log) 변환", "큰 값들의 차이를 좁혀주어 분포를 정규분포에 가깝게 만듭니다.", "로그 변환", "2072"),
    ("분석 결과를 타인에게 전달할 때 가장 중요한 요소는?", ["사용한 코드의 길이", "얼마나 비싼 GPU를 썼는지", "비즈니스 인사이트를 도출하는 시각화와 설명", "오직 정확도 숫자 하나", "사용한 라이브러리의 버전 목록"], "비즈니스 인사이트를 도출하는 시각화와 설명", "데이터를 통해 어떤 결정을 내려야 하는지 설득력 있게 전달해야 합니다.", "데이터 스토리텔링", "2073"),
    ("Pandas `df.apply(len)`을 전체 문자열 컬럼에 적용하면 얻는 결과는?", ["각 문자열의 첫 글자", "각 문자열의 길이 값", "문자열 내 공백 개수", "전체 행의 숫자", "에러 발생"], "각 문자열의 길이 값", "모든 행에 대해 len 함수가 실행되어 글자 수가 계산된 결과가 나옵니다.", "apply 활용", "2074"),
    ("실무에서 CSV 파일이 대중적으로 쓰이는 이유는?", ["파이썬에서만 열리기 때문에", "데이터 보안이 가장 완벽해서", "구조가 단순하고 가독성이 좋으며 호환성이 뛰어나서", "압축률이 전 세계 최고라서", "이미지 데이터 저장에 적합해서"], "구조가 단순하고 가독성이 좋으며 호환성이 뛰어나서", "쉼표로 구분된 텍스트 형식이어서 거의 모든 도구(엑셀 등)에서 열립니다.", "CSV 장점", "2075"),
    ("데이터프레임 `df.isna().sum()` 코드가 의미하는 바는?", ["결측치가 있는 행의 이름", "컬럼별 결측치의 총 개수", "전체 데이터의 평균값", "중복된 데이터의 개수", "데이터 타입의 목록"], "컬럼별 결측치의 총 개수", "isna()로 비어있는지 체크하고 sum()으로 각 열의 True 개수를 합산합니다.", "결측치 합계", "2076"),
    ("시계열 데이터에서 특정 기간(예: 7일)의 이동 평균을 구하는 메서드는?", ["df.shift()", "df.rolling(window=7).mean()", "df.expand().mean()", "df.avg_move(7)", "df.time_mean(7)"], "df.rolling(window=7).mean()", "시계열의 변동을 완만하게 보고 트렌드를 파악할 때 씁니다.", "이동 평균", "2077"),
    ("Pandas `describe()` 결과에서 50% 지점에 해당하는 통계량의 명칭은?", ["Mean (평균)", "Std (표준편차)", "Median (중앙값)", "Mode (최빈값)", "Range (범위)"], "Median (중앙값)", "데이터를 순서대로 세웠을 때 정중앙에 위치하는 값입니다.", "중앙값", "2078"),
    ("데이터프레임의 인덱스를 무작위로 섞고 싶을 때 사용하는 방법은?", ["df.mix()", "df.sample(frac=1)", "df.reverse()", "df.shuffle()", "df.random_index()"], "df.sample(frac=1)", "전체 데이터(frac=1)를 무작위로 추출하면 셔플 효과가 납니다.", "인덱스 셔플", "2079"),
    ("여러 컬럼 중 상관계수가 1에 가깝게 나온 두 변수의 관계는?", ["아무런 상관이 없다.", "서로 반대 방향으로 똑같이 움직인다.", "한 변수가 커질 때 다른 변수도 거의 똑같이 커진다.", "두 변수는 서로 독립적이다.", "데이터가 잘못 입력된 것이다."], "한 변수가 커질 때 다른 변수도 거의 똑같이 커진다.", "양의 상관계수 1은 완전한 정비례 관계를 의미합니다.", "상관계수 의미", "2080"),
    ("데이터 용량이 너무 커서 메모리에 한 번에 안 올라갈 때의 대책은?", ["데이터를 절반 잘라버린다.", "NumPy만 쓴다.", "read_csv에 chunksize를 주어 끊어 읽는다.", "모든 숫자를 정수로 바꾼다.", "분석을 포기한다."], "read_csv에 chunksize를 주어 끊어 읽는다.", "파일을 조각조각 읽어서 처리하는 방식으로 메모리 문제를 해결합니다.", "대용량 처리", "2081"),
    ("Seaborn의 히트맵(Heatmap)을 그릴 때 주로 입력값으로 주는 것은?", ["문자열 원본 리스트", "컬럼별 평균값 리스트", "상관계수 행렬 (df.corr())", "전체 데이터프레임 원본", "데이터 타입 목록"], "상관계수 행렬 (df.corr())", "격자 구조의 수치 행렬을 주면 색상으로 강도를 표시합니다.", "히트맵 입력", "2082"),
    ("데이터 분석 보고서에서 '가설 설정'의 단계는 보통 언제 이루어지는가?", ["데이터 분석이 다 끝난 후 결론 쓸 때", "분석을 시작하기 전 혹은 EDA 중간 단계", "라이브러리 import 할 때", "컴퓨터 전원 켤 때", "고객에게 보고서 제출할 때"], "분석을 시작하기 전 혹은 EDA 중간 단계", "데이터를 통해 무엇을 증명할지 미리 정하고 접근해야 효율적입니다.", "분석 프로세스", "2083"),
    ("Pandas `df.iloc[0:3, 1:4]` 가 의미하는 선택 범위는?", ["1~3행, 2~4열", "0~2번 행, 1~3번 열", "첫 번째 행부터 세 번째 행까지 전체 열", "0, 1, 2, 3번 행, 1, 2, 3, 4번 열", "에러"], "0~2번 행, 1~3번 열", "정수 위치 기반이며 슬라이싱 끝 번호는 포함되지 않습니다.", "iloc 범위", "2084"),
    ("데이터프레임에서 '성별' 컬럼의 '남'을 1, '여'를 0으로 바꾸는 가장 편리한 함수는?", ["apply(int)", "replace({'남': 1, '여': 0})", "strip()", "sum()", "astype(int)"], "replace({'남': 1, '여': 0})", "사전 형식을 전달하여 특정 값을 다른 값으로 일괄 변경합니다.", "값 치환", "2085"),
    ("Matplotlib에서 그래프에 제목을 다는 함수는?", ["plt.name()", "plt.title()", "plt.header()", "plt.tag()", "plt.subject()"], "plt.title()", "그래프 상단에 설명을 추가하는 기본 함수입니다.", "제목 달기", "2086"),
    ("Pandas에서 중복 행이 있는지 여부(Boolean)만 확인하는 메서드는?", ["duplicated()", "is_repeating()", "check_copy()", "repeat()", "exist()"], "duplicated()", "각 행이 이전 행과 중복되는지를 True/False로 반환합니다.", "중복 체크", "2087"),
    ("데이터프레임 두 개를 옆으로(가로로) 붙일 때 `pd.concat`에 주어야 할 축 옵션은?", ["axis=0", "axis=1", "direction='horizontal'", "side='right'", "join='true'"], "axis=1", "axis=0(기본값)은 위아래, axis=1은 좌우 연결입니다.", "concat 축", "2088"),
    ("데이터 분석 시 한글 폰트가 깨지지 않게 하기 위해 설정해야 하는 라이브러리는?", ["NLTK", "KoNLPy", "Matplotlib (rc 설정)", "OS", "Pandas"], "Matplotlib (rc 설정)", "Matplotlib의 기본 폰트는 영문이므로 한글 폰트 경로를 직접 잡아줘야 합니다.", "한글 폰트", "2089"),
    ("데이터프레임 저장 시 인덱스(0, 1, 2...)는 빼고 저장하고 싶을 때 옵션은?", ["drop_index=True", "index=False", "no_index=1", "save_index=0", "header=None"], "index=False", "`to_csv('file.csv', index=False)` 처럼 사용합니다.", "저장 옵션", "2090"),
    ("NumPy 배열에서 0이 아닌 요소의 위치를 찾는 함수는?", ["np.find()", "np.nonzero()", "np.where()", "np.search()", "np.nonzero() 및 np.where()"], "np.nonzero() 및 np.where()", "조건에 맞는 요소의 인덱스를 찾아내는 핵심 함수들입니다.", "위치 찾기", "2091"),
    ("데이터 분석가에게 가장 권장되는 파이썬 실행 환경은?", ["메모장", "명령 프롬프트(CMD)", "Jupyter Notebook / JupyterLab", "웹 사이트 소스 보기", "그림판"], "Jupyter Notebook / JupyterLab", "대화형 실행과 시각화, 문서화가 동시에 가능하여 분석가 표준 도구입니다.", "실행 환경", "2092"),
    ("Pandas `df['A'].shift(1)` 을 실행했을 때 일어나는 일은?", ["A 컬럼의 모든 값이 1씩 증가한다.", "A 컬럼의 값들이 아래로 한 칸씩 밀리고 첫 행은 NaN이 된다.", "A 컬럼이 리스트로 바뀐다.", "A 컬럼이 통째로 삭제된다.", "순서가 거꾸로 뒤집힌다."], "A 컬럼의 값들이 아래로 한 칸씩 밀리고 첫 행은 NaN이 된다.", "시계열 분석에서 전일 대비 증감 등을 계산할 때 필수입니다.", "shift", "2093"),
    ("데이터프레임의 인덱스명을 기준으로 정렬하고 싶을 때 쓰는 메서드는?", ["sort_values()", "sort_index()", "sort_label()", "order_index()", "align_index()"], "sort_index()", "값이 아닌 인덱스(날짜 등)를 순서대로 나열할 때 씁니다.", "sort_index", "2094"),
    ("`pd.read_csv('data.csv', nrows=10)` 코드가 수행하는 작업은?", ["모든 행의 이름을 10으로 바꾼다.", "전체 파일 중 상위 10개 행만 읽어온다.", "파일의 10번째 줄부터 읽기 시작한다.", "파일에 10개의 빈 줄을 추가한다.", "에러가 난다."], "전체 파일 중 상위 10개 행만 읽어온다.", "매우 큰 파일의 샘플만 빠르게 보고 싶을 때 유용합니다.", "nrows", "2095"),
    ("데이터프레임의 모든 원소에 대해 개별적으로 함수를 적용하는 메서드는?", ["apply()", "map()", "applymap()", "foreach()", "transform()"], "applymap()", "데이터프레임 전체 격자에 대해 원소 단위 연산을 수행합니다.", "applymap", "2096"),
    ("Numpy `np.linspace(0, 10, 5)` 가 생성하는 배열은?", ["[0, 2.5, 5, 7.5, 10]", "[0, 2, 4, 6, 8, 10]", "[0, 5, 10]", "[0, 1, 2, 3, 4, 5...]", "[2, 4, 6, 8]"], "[0, 2.5, 5, 7.5, 10]", "0부터 10 사이를 균등하게 5개의 포인트로 나눕니다.", "linspace", "2097"),
    ("시각화 도구 중 웹 기반 인터랙티브(확대, 마우스 오버 등) 그래프에 강점이 있는 것은?", ["Matplotlib", "Seaborn", "Plotly", "ggplot", "PIL"], "Plotly", "마우스 움직임에 반응하는 화려한 그래프를 만들 때 선호됩니다.", "Plotly", "2098"),
    ("데이터 전처리 시 '정규화(Normalization)'와 '표준화(Standardization)'의 공통적인 목적은?", ["데이터를 암호화하기 위해", "프로그램 실행 속도를 높이기 위해", "서로 다른 변수의 스케일을 맞춰 비교 가능하게 하기 위해", "결측치를 자동으로 채우기 위해", "파일 용량을 크게 만들기 위해"], "서로 다른 변수의 스케일을 맞춰 비교 가능하게 하기 위해", "단위가 다른 데이터(예: 키와 몸무게)를 공정한 범위에서 비교하게 돕습니다.", "스케일링 목적", "2099"),
    ("데이터 분석의 최종적인 결과물로 가장 적절한 형태는?", ["수만 줄의 소스 코드", "단순히 '정확도가 높다'라는 말", "데이터 기반의 인사이트와 실행 권고안이 담긴 리포트", "사용자가 이해할 수 없는 도표들", "아무런 설명 없는 엑셀 파일"], "데이터 기반의 인사이트와 실행 권고안이 담긴 리포트", "데이터 분석의 목적은 비즈니스 문제를 해결하고 의사결정을 돕는 것에 있습니다.", "분석의 결과", "2100")
]

for q, o, a, w, h, i in mcq_data:
    questions.append({"chapter_name": chapter_name, "type": "객관식", "difficulty": "medium", "id": i, "question": q, "options": o, "answer": a, "why": w, "hint": h})

# --- 20 Code Completion Questions ---
cc_data = [
    ("데이터 로드", "import pandas as pd\ndf = pd.____('data.csv') # 이 부분을 채우세요", "read_csv", "CSV 파일을 읽어오는 표준 함수입니다."),
    ("상위 데이터 확인", "df.____(10) # 이 부분을 채우세요 (상위 10개)", "head", "데이터의 앞부분을 확인하는 메서드입니다."),
    ("요약 정보 확인", "df.____() # 요약 정보 출력 (이 부분을 채우세요)", "info", "컬럼 정보와 결측치를 보여주는 메서드입니다."),
    ("기술 통계량", "df.____() # 수치형 요약 (이 부분을 채우세요)", "describe", "평균, 표준편차 등을 요약해주는 메서드입니다."),
    ("인덱스 기준 선택", "df.____['row1', 'col1'] # 이 부분을 채우세요", "loc", "이름 기반 데이터 접근자입니다."),
    ("정수 위치 선택", "df.____[0, 0] # 이 부분을 채우세요", "iloc", "숫자 위치 기반 데이터 접근자입니다."),
    ("결측치 채우기", "df['age'] = df['age'].____(20) # 이 부분을 채우세요", "fillna", "빈 값을 특정 값으로 채우는 메서드입니다."),
    ("결측 행 삭제", "clean_df = df.____() # 이 부분을 채우세요", "dropna", "결측치가 포함된 행을 제거합니다."),
    ("그룹화", "grouped = df.____('category') # 이 부분을 채우세요", "groupby", "데이터를 그룹별로 묶어주는 메서드입니다."),
    ("데이터 병합", "merged = pd.____(df1, df2, on='id') # 이 부분을 채우세요", "merge", "키를 기준으로 합치는 함수입니다."),
    ("단순 연결", "combined = pd.____([df1, df2], axis=0) # 이 부분을 채우세요", "concat", "리스트로 묶어 연결하는 함수입니다."),
    ("컬럼 삭제", "df_small = df.____('price', axis=1) # 이 부분을 채우세요", "drop", "특정 열을 지우는 메서드입니다."),
    ("고유값 빈도", "counts = df['city'].____() # 이 부분을 채우세요", "value_counts", "가장 많이 등장하는 값을 찾을 때 씁니다."),
    ("형변환", "df['year'] = df['year'].____(int) # 이 부분을 채우세요", "astype", "컬럼의 데이터 타입을 변환합니다."),
    ("함수 일괄 적용", "df['len'] = df['name'].____(len) # 이 부분을 채우세요", "apply", "함수를 요소마다 적용합니다."),
    ("NumPy 임포트", "import ____ as np # 이 부분을 채우세요", "numpy", "NumPy의 표준 별칭입니다."),
    ("0 초기화 배열", "arr = np.____((3, 3)) # 이 부분을 채우세요", "zeros", "0으로 가득 찬 배열을 만듭니다."),
    ("배열 모양 확인", "print(arr.____) # 이 부분을 채우세요 (모양 속성)", "shape", "배열의 차원을 알려주는 속성입니다."),
    ("배열 재구성", "new_arr = arr.____(1, 9) # 이 부분을 채우세요", "reshape", "요소 수는 같게 차원만 바꿉니다."),
    ("그래프 제목", "import matplotlib.pyplot as plt\nplt.____('Result') # 이 부분을 채우세요", "title", "그래프 상단에 제목을 노출합니다.")
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
