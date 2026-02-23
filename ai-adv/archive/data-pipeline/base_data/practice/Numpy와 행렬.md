# [실습] Numpy와 행렬

## 학습목표
- NumPy 배열을 생성하고 다룰 수 있다
- 벡터와 행렬 연산을 수행할 수 있다
- 임베딩 벡터 간 유사도를 계산할 수 있다
- 텍스트 데이터를 벡터로 변환하고 분석할 수 있다
%pip install numpy
## NumPy 기초

NumPy는 파이썬에서 수치 연산을 위한 핵심 라이브러리입니다.  
특히 LLM과 AI 분야에서 벡터 연산의 기본이 됩니다.
import numpy as np
# numpy.OO 대신 np.OO를 불러오는 축약어 표현

# NumPy 배열 생성
arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.array([[1, 2, 3], [4, 5, 6]])

print(f"1차원 배열: {arr1}")
print(f"형태: {arr1.shape}, 차원: {arr1.ndim}, 타입: {arr1.dtype}")
print()
print(f"2차원 배열:\n{arr2}")
print(f"형태: {arr2.shape}, 차원: {arr2.ndim}")
# 다양한 배열 생성 방법
zeros = np.zeros((3, 4))
ones = np.ones((2, 3))
random_arr = np.random.randn(3, 3)  # 표준정규분포
range_arr = np.arange(0, 10, 2)  # 0부터 10미만까지 2씩 증가

print(f"영행렬 (3x4):\n{zeros}\n")
print(f"일행렬 (2x3):\n{ones}\n")
print(f"랜덤 배열 (3x3):\n{random_arr}\n")
print(f"범위 배열: {range_arr}")
## 벡터와 행렬 연산

AI에서 가장 중요한 연산들을 살펴보겠습니다.
# 벡터 연산
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

# 기본 연산
print(f"v1 + v2 = {v1 + v2}")
print(f"v1 * 2 = {v1 * 2}")
print(f"v1 * v2 (요소별 곱) = {v1 * v2}")

# 내적 (dot product)
dot_product = np.dot(v1, v2)
print(f"v1 · v2 (내적) = {dot_product}")

# 벡터 크기 (norm)
norm_v1 = np.linalg.norm(v1)
print(f"||v1|| = {norm_v1:.4f}")
# 행렬 연산
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print(f"행렬 A:\n{A}\n")
print(f"행렬 B:\n{B}\n")

# 행렬 곱셈
matrix_product = np.matmul(A, B)
print(f"A × B:\n{matrix_product}\n")

# 전치 행렬
A_transpose = A.T
print(f"A의 전치행렬:\n{A_transpose}")
## 텍스트 임베딩 시뮬레이션

실제 LLM에서 사용하는 임베딩 벡터를 간단히 시뮬레이션해보겠습니다.
np.random.randn(5)
# 단어를 임베딩 벡터로 표현 (실제로는 학습된 값을 사용)
np.random.seed(42)  # 재현성을 위한 시드 고정

word_embeddings = {
    'python': np.random.randn(5),
    'java': np.random.randn(5),
    'AI': np.random.randn(5),
    '머신러닝': np.random.randn(5),
    '딥러닝': np.random.randn(5),
    '데이터': np.random.randn(5)
}

# 유사한 단어끼리는 벡터를 비슷하게 조정
word_embeddings['머신러닝'] = word_embeddings['AI'] + np.random.randn(5) * 0.3
word_embeddings['딥러닝'] = word_embeddings['머신러닝'] + np.random.randn(5) * 0.3

print("=== 단어 임베딩 벡터 (5차원) ===")
for word, vector in word_embeddings.items():
    print(f"{word:8s}: {vector}")
## 코사인 유사도 계산

임베딩 벡터 간의 유사도를 계산하는 가장 일반적인 방법입니다.
def cosine_similarity(v1, v2):
    """두 벡터 간의 코사인 유사도 계산"""
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    if norm_v1 == 0 or norm_v2 == 0:
        return 0
    
    return dot_product / (norm_v1 * norm_v2)

# 단어 간 유사도 계산
words = list(word_embeddings.keys())
similarity_matrix = np.zeros((len(words), len(words)))
# 6*6 zero matrix

for i, word1 in enumerate(words):
    for j, word2 in enumerate(words):
        sim = cosine_similarity(word_embeddings[word1], word_embeddings[word2])
        similarity_matrix[i, j] = sim

# 유사도 출력
print("=== 단어 간 코사인 유사도 ===")
print(f"{'':10s}", end='')
for word in words:  # 표시 공간을 위해 처음 4개만
    print(f"{word:10s}", end='')
print()

for i, word1 in enumerate(words):
    print(f"{word1:10s}", end='')
    for j in range(6):
        print(f"{similarity_matrix[i, j]:10.3f}", end='')
    print()
def find_similar_words(target_word, word_embeddings, top_k=3):
    """특정 단어와 가장 유사한 단어 찾기"""
    if target_word not in word_embeddings:
        return []
    
    target_vector = word_embeddings[target_word]
    similarities = []
    
    for word, vector in word_embeddings.items():
        if word != target_word:
            sim = cosine_similarity(target_vector, vector)
            similarities.append((word, sim))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]

# 각 단어별 유사 단어 찾기
test_words = ['AI', '머신러닝', 'python']

for word in test_words:
    similar = find_similar_words(word, word_embeddings, top_k=2)
    print(f"\n'{word}'와 유사한 단어:")
    for sim_word, score in similar:
        print(f"  - {sim_word}: {score:.3f}")
## 문장 임베딩 계산

단어 임베딩을 조합하여 문장의 임베딩을 만드는 간단한 방법을 구현해보겠습니다.
def sentence_embedding(sentence, word_embeddings):
    """문장의 임베딩을 단어 임베딩의 평균으로 계산"""
    words = sentence.lower().split()
    embeddings = []
    
    for word in words:
        if word in word_embeddings:
            embeddings.append(word_embeddings[word])
    
    if not embeddings:
        return np.zeros(5)  # 임베딩 차원과 동일
    
    return np.mean(embeddings, axis=0)

# 문장 임베딩 테스트
sentences = [
    "AI 머신러닝 딥러닝",
    "머신러닝 데이터",
    "python java",
    "AI 데이터 python"
]

sentence_vectors = {}
for sentence in sentences:
    vector = sentence_embedding(sentence, word_embeddings)
    sentence_vectors[sentence] = vector
    print(f"문장: '{sentence}'")
    print(f"  임베딩: {vector}\n")
# 문장 간 유사도 계산
print("=== 문장 간 유사도 ===")
for i, sent1 in enumerate(sentences):
    for j, sent2 in enumerate(sentences):
        if i < j:  # 중복 제거
            sim = cosine_similarity(sentence_vectors[sent1], sentence_vectors[sent2])
            print(f"'{sent1}' vs '{sent2}'")
            print(f"  유사도: {sim:.3f}\n")
## 실전 응용: 문서 검색 시스템

임베딩을 활용한 간단한 문서 검색 시스템을 만들어보겠습니다.
# 문서 데이터베이스
documents = [
    "Python은 데이터 분석에 널리 사용되는 프로그래밍 언어입니다",
    "머신러닝과 딥러닝은 AI의 핵심 기술입니다",
    "Java는 엔터프라이즈 애플리케이션 개발에 많이 사용됩니다",
    "데이터 전처리는 머신러닝 프로젝트의 중요한 단계입니다",
    "딥러닝 모델은 대량의 데이터를 필요로 합니다",
    "Python과 Java는 인기 있는 프로그래밍 언어입니다"
]

# 각 문서를 벡터로 변환
doc_vectors = []
for doc in documents:
    vector = sentence_embedding(doc, word_embeddings)
    doc_vectors.append(vector)

doc_vectors = np.array(doc_vectors)
print(f"문서 벡터 행렬 형태: {doc_vectors.shape}")
print(f"총 {len(documents)}개 문서, 각 {doc_vectors.shape[1]}차원 벡터")
def search_documents(query, documents, doc_vectors, word_embeddings, top_k=3):
    """쿼리와 가장 유사한 문서 검색"""
    query_vector = sentence_embedding(query, word_embeddings)
    
    similarities = []
    for i, doc_vector in enumerate(doc_vectors):
        sim = cosine_similarity(query_vector, doc_vector)
        similarities.append((i, sim))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    results = []
    for idx, sim in similarities[:top_k]:
        results.append({
            'document': documents[idx],
            'similarity': sim,
            'rank': len(results) + 1
        })
    
    return results

# 검색 테스트
queries = [
    "AI 딥러닝",
    "python 데이터",
    "java",
    '오늘 저녁 메뉴'
]

for query in queries:
    print(f"\n=== 검색어: '{query}' ===")
    results = search_documents(query, documents, doc_vectors, word_embeddings, top_k=2)
    
    for result in results:
        print(f"{result['rank']}. 유사도: {result['similarity']:.3f}")
        print(f"   {result['document']}")
## 학습 내용 정리
### 핵심 NumPy 연산

| 연산 | 코드 |
|------|------|
| 벡터 생성 | `np.array([1, 2, 3])` |
| 영벡터 | `np.zeros(shape)` |
| 랜덤 벡터 | `np.random.randn(shape)` |
| 내적 | `np.dot(v1, v2)` |
| 노름 | `np.linalg.norm(v)` |
| 정규화 | `v / np.linalg.norm(v)` |
| 코사인 유사도 | `np.dot(v1, v2) / (norm1 * norm2)` |
| 행렬 곱 | `np.matmul(A, B)` or `A @ B` |
| 전치 | `A.T` |

### 실습 완료 체크리스트

✓ NumPy 배열 생성 및 조작  
✓ 벡터와 행렬 연산  
✓ 임베딩 벡터 유사도 계산  
✓ 문서 검색 시스템 구현  
