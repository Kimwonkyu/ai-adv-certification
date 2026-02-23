# [실습] Pandas 데이터 처리하기

## 학습목표
- Pandas DataFrame을 생성하고 조작할 수 있다
- CSV, JSON 파일을 읽고 쓸 수 있다
- 데이터를 필터링, 정렬, 그룹화할 수 있다
- 텍스트 데이터를 분석하고 통계를 추출할 수 있다
%pip install pandas 
## Pandas 기초

Pandas는 데이터 분석을 위한 파이썬의 핵심 라이브러리입니다.  
특히 텍스트 데이터 전처리와 분석에 매우 유용합니다.
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 기본 DataFrame 생성
data = {
    'name': ['김철수', '이영희', '박민수', '정수진', '최동현'],
    'department': ['AI개발팀', '데이터팀', 'AI개발팀', '데이터팀', 'AI개발팀'],
    'experience': [3, 5, 2, 7, 4],
    'skills': ['Python, ML', 'Python, SQL', 'Java, Python', 'R, Python', 'Python, DL']
}

df = pd.DataFrame(data)
print("=== 직원 데이터 ===")
print(df)
print(f"\n데이터 형태: {df.shape}")
print(f"컬럼: {df.columns.tolist()}")
df
# 데이터 타입과 기본 정보
print("=== 데이터 정보 ===")
print(df.info())
print("\n=== 통계 요약 ===")
print(df.describe())
print("\n=== 데이터 타입 ===")
print(df.dtypes)
## 데이터 로드 및 저장

실제 데이터 파일을 읽고 쓰는 방법을 알아보겠습니다.
# JSON 데이터 읽기
import json

# 샘플 뉴스 데이터 로드
try:
    with open('news_data.json', 'r', encoding='utf-8') as f:
        news_data = json.load(f)
    
    news_df = pd.DataFrame(news_data)
    print("=== 뉴스 데이터 로드 성공 ===")
    print(f"총 {len(news_df)}개 기사")
    print("\n첫 3개 기사:")
    print(news_df.head(3))
except FileNotFoundError:
    print("파일이 없어 샘플 데이터를 생성합니다...")
    # 샘플 데이터 생성
    news_df = pd.DataFrame({
        'title': [
            'AI 기술 발전으로 업무 자동화 가속화',
            '파이썬, 가장 인기 있는 프로그래밍 언어 1위',
            '머신러닝 엔지니어 수요 급증'
        ],
        'category': ['기술', '개발', '채용'],
        'date': ['2025-01-15', '2025-01-16', '2025-01-17'],
        'views': [1500, 2300, 1800]
    })
    print(news_df)
news_df.head()
## 데이터 선택과 필터링

DataFrame에서 원하는 데이터를 선택하고 필터링하는 방법을 알아보겠습니다.
# 컬럼 선택
print("=== 특정 컬럼 선택 ===")
print(df['name'])
print("\n=== 여러 컬럼 선택 ===")
print(df[['name', 'department']])

# 인덱스로 선택
print("\n=== iloc으로 행 선택 ===")
print(df.iloc[0])  # 첫 번째 행
print("\n=== 슬라이싱 ===")
print(df.iloc[1:3])  # 두 번째부터 세 번째 행
df
df[(df['department'] == 'AI개발팀') & (df['experience'] >= 3)]
# 조건 필터링
print("=== 경력 3년 이상 직원 ===")
experienced = df[df['experience'] >= 3]
print(experienced)

print("\n=== AI개발팀 직원 ===")
ai_team = df[df['department'] == 'AI개발팀']
print(ai_team)

# 복합 조건
print("\n=== AI개발팀이면서 경력 3년 이상 ===")
filtered = df[(df['department'] == 'AI개발팀') & (df['experience'] >= 3)]
print(filtered)
## 데이터 변환과 처리

데이터를 변환하고 새로운 컬럼을 생성하는 방법을 배워보겠습니다.
df
# 새로운 컬럼 추가
df['level'] = df['experience'].apply(lambda x: '시니어' if x >= 5 else ('주니어' if x < 3 else '미드'))
df['has_python'] = df['skills'].str.contains('Python')
df['has_DL'] = df['skills'].str.contains('DL')

print("=== 레벨과 Python 스킬 추가 ===")
print(df)

# 스킬 개수 계산
df['skill_count'] = df['skills'].str.split(',').str.len()
print("\n=== 스킬 개수 추가 ===")
print(df[['name', 'skills', 'skill_count']])
df[['name', 'skills', 'skill_count']]
# 문자열 처리
df['skills_lower'] = df['skills'].str.lower()
df['skills_list'] = df['skills'].str.split(', ')

print("=== 문자열 처리 결과 ===")
for idx, row in df.iterrows():
    print(f"{row['name']}: {row['skills_list']}")
## 그룹화와 집계

데이터를 그룹화하고 통계를 계산하는 방법을 알아보겠습니다.
df
# 부서별 통계
dept_stats = df.groupby('department').agg({
    'experience': ['mean', 'max', 'min'],
    'name': 'count',
    'skill_count': 'mean'
})

print("=== 부서별 통계 ===")


# 레벨별 분포
level_dist = df['level'].value_counts()
print("\n=== 레벨별 인원 분포 ===")
print(level_dist)
## 텍스트 데이터 분석 실습

실제 텍스트 데이터를 분석하는 종합 실습을 진행하겠습니다.
# 댓글 데이터 생성
comments_data = {
    'comment_id': range(1, 11),
    'user': ['user1', 'user2', 'user3', 'user4', 'user5', 
             'user1', 'user2', 'user6', 'user7', 'user8'],
    'text': [
        '정말 유용한 기능이네요! Python 최고입니다',
        '이 코드 잘 안되는데 도움 좀 주세요',
        'AI 기술 발전이 놀랍습니다. 머신러닝 공부하고 싶어요',
        '별로예요... 다른 방법이 더 좋을 것 같아요',
        'Python pandas 정말 편리하네요',
        '딥러닝 프레임워크 추천해주세요',
        '데이터 전처리가 제일 어려워요 ㅠㅠ',
        '좋은 정보 감사합니다! 많이 배웠어요',
        '코드 에러 발생... 왜 안되는지 모르겠네요',
        'AI와 머신러닝의 차이가 뭔가요?'
    ],
    'timestamp': pd.date_range('2024-01-01', periods=10, freq='h'),
    'likes': [15, 3, 12, 2, 8, 5, 4, 10, 1, 6]
}

comments_df = pd.DataFrame(comments_data)
print("=== 댓글 데이터 ===")
print(comments_df.head())
comments_df
# 텍스트 길이 분석
comments_df['text_length'] = comments_df['text'].str.len()
comments_df['word_count'] = comments_df['text'].str.split().str.len()

print("=== 텍스트 길이 분석 ===")
print(comments_df[['text', 'text_length', 'word_count']].head())

print(f"\n평균 텍스트 길이: {comments_df['text_length'].mean()}")
print(f"평균 단어 수: {comments_df['word_count'].mean()}")
# 키워드 포함 여부 확인
tech_keywords = ['Python', 'AI', '머신러닝', '딥러닝', '데이터']

for keyword in tech_keywords:
    comments_df[f'has_{keyword}'] = comments_df['text'].str.contains(keyword, case=False)
    # f-string으로 파생 변수 만들기

# 기술 관련 댓글 필터링
tech_columns = [col for col in comments_df.columns if col.startswith('has_')]
tech_columns
comments_df[tech_columns].any(axis=1)
comments_df['is_tech'] = comments_df[tech_columns].any(axis=1)

print("=== 기술 관련 댓글 ===")
tech_comments = comments_df[comments_df['is_tech']]
print(tech_comments[['user', 'text', 'likes']])
# 감정 분석 (간단한 규칙 기반)
positive_words = ['좋', '최고', '유용', '편리', '감사', '배웠']
negative_words = ['별로', '어려', '에러', '안되', '모르겠']

def analyze_sentiment(text):
    text_lower = text.lower()
    pos_score = sum(1 for word in positive_words if word in text_lower)
    neg_score = sum(1 for word in negative_words if word in text_lower)
    
    if pos_score > neg_score:
        return '긍정'
    elif neg_score > pos_score:
        return '부정'
    else:
        return '중립'

comments_df['sentiment'] = comments_df['text'].apply(analyze_sentiment)

print("=== 감정 분석 결과 ===")
print(comments_df['sentiment'].value_counts())
print("\n=== 감정별 평균 좋아요 수 ===")
print(comments_df.groupby('sentiment')['likes'].mean())
## 시계열 데이터 처리

시간 데이터를 다루는 방법을 알아보겠습니다.
comments_df
# 시간대별 분석
comments_df['hour'] = comments_df['timestamp'].dt.hour
comments_df['date'] = comments_df['timestamp'].dt.date

hourly_stats = comments_df.groupby('hour').agg({
    'comment_id': 'count',
    'likes': 'mean'
}).rename(columns={'comment_id': 'count', 'likes': 'avg_likes'})

print("=== 시간대별 통계 ===")
print(hourly_stats)

# 활동이 가장 많은 사용자
user_activity = comments_df.groupby('user').agg({
    'comment_id': 'count',
    'likes': 'sum',
    'text_length': 'mean'
}).rename(columns={
    'comment_id': 'comment_count',
    'likes': 'total_likes',
    'text_length': 'avg_text_length'
})

print("\n=== 사용자별 활동 ===")
print(user_activity.sort_values('total_likes', ascending=False))
## 데이터 병합과 조인

여러 DataFrame을 합치는 방법을 알아보겠습니다.
# 사용자 정보 데이터
users_data = {
    'user': ['user1', 'user2', 'user3', 'user4', 'user5'],
    'join_date': ['2023-01-01', '2023-03-15', '2023-06-20', '2023-09-10', '2023-11-05'],
    'user_level': ['골드', '실버', '브론즈', '실버', '골드']
}
users_df = pd.DataFrame(users_data)
users_df
comments_df

# 댓글 데이터와 병합
merged_df = comments_df.merge(users_df, on='user', how='left')
# 'user 컬럼 기준으로 결합', 
merged_df
print("=== 병합된 데이터 ===")
print(merged_df[['user', 'text', 'user_level', 'likes']].head())

# 레벨별 통계
level_stats = merged_df.groupby('user_level').agg({
    'likes': 'mean',
    'word_count': 'mean',
    'sentiment': lambda x: x.value_counts().to_dict()
})

print("\n=== 사용자 레벨별 통계 ===")
print(level_stats)
## 피벗 테이블과 크로스탭

데이터를 재구조화하는 고급 기법을 알아보겠습니다.
# 피벗 테이블 생성
pivot_table = merged_df.pivot_table(
    values='likes',
    index='sentiment',
    columns='user_level',
    aggfunc='mean',
    fill_value=0
)

print("=== 감정-레벨별 평균 좋아요 수 ===")
print(pivot_table)

pivot_table

# 크로스탭
cross_tab = pd.crosstab(
    merged_df['sentiment'],
    merged_df['is_tech'],
    margins=True
)

print("\n=== 감정-기술관련 크로스탭 ===")
print(cross_tab)

cross_tab
## 데이터 저장

처리한 데이터를 파일로 저장하는 방법을 알아보겠습니다.
# CSV로 저장
output_file = 'processed_comments.csv'
comments_df.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"CSV 파일 저장 완료: {output_file}")

# JSON으로 저장
json_output = 'comments_analysis.json'
analysis_result = {
    'total_comments': len(comments_df),
    'sentiment_distribution': comments_df['sentiment'].value_counts().to_dict(),
    'avg_likes': float(comments_df['likes'].mean()),
    'tech_comments_ratio': float(comments_df['is_tech'].mean()),
    'top_keywords': tech_keywords[:3]
}

with open(json_output, 'w', encoding='utf-8') as f:
    json.dump(analysis_result, f, ensure_ascii=False, indent=2)
print(f"\nJSON 분석 결과 저장 완료: {json_output}")
print(json.dumps(analysis_result, ensure_ascii=False, indent=2))
## 학습 내용 정리

### 핵심 Pandas 연산

| 연산 | 코드 예시 | 설명 |
|------|----------|------|
| DataFrame 생성 | `pd.DataFrame(data)` | 딕셔너리나 리스트로 생성 |
| 파일 읽기 | `pd.read_csv()`, `pd.read_json()` | 외부 파일 로드 |
| 필터링 | `df[df['col'] > value]` | 조건에 맞는 행 선택 |
| 그룹화 | `df.groupby('col').agg()` | 그룹별 집계 |
| 병합 | `df1.merge(df2, on='key')` | 두 DataFrame 조인 |
| 피벗 | `df.pivot_table()` | 데이터 재구조화 |
| 문자열 처리 | `df['col'].str.method()` | 텍스트 컬럼 처리 |
| Apply | `df['col'].apply(function)` | 함수 적용 |

### 실습 완료 체크리스트

✓ DataFrame 생성 및 기본 조작  
✓ 데이터 필터링과 선택  
✓ 텍스트 데이터 전처리  
✓ 그룹화와 집계 분석  
✓ 데이터 병합과 피벗  
✓ 파일 입출력 (CSV, JSON)