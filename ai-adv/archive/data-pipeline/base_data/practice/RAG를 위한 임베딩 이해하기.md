# 🚀 RAG를 위한 임베딩(Embedding) 이해하기

## 📖 학습 목표
- 임베딩의 개념과 원리 이해
- Qwen3-Embedding 모델을 사용한 한국어 텍스트 임베딩
- 임베딩 벡터의 시각화와 유사도 계산
- RAG 시스템에서의 임베딩 활용법 이해
## 1. 환경 설정

### 1.1 필요한 라이브러리 설치
먼저 실습에 필요한 라이브러리들을 설치합니다.
!pip install -q transformers torch scikit-learn matplotlib seaborn numpy pandas accelerate setuptools koreanize-matplotlib
### 1.2 라이브러리 임포트
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

print("✅ 모든 라이브러리가 성공적으로 로드되었습니다!")
print(f"PyTorch 버전: {torch.__version__}")
print(f"GPU 사용 가능: {torch.cuda.is_available()}")
## 2. 임베딩이란?

**임베딩(Embedding)**은 텍스트를 컴퓨터가 이해할 수 있는 숫자 벡터로 변환하는 과정입니다.

### 💡 핵심 개념
- **텍스트 → 벡터**: "안녕하세요" → [0.1, -0.5, 0.3, ...]
- **의미 보존**: 비슷한 의미의 텍스트는 비슷한 벡터로 표현
- **차원**: 보통 384~1536 차원의 벡터 사용

### 🎯 RAG에서의 역할
1. **문서 인덱싱**: 문서를 벡터로 변환하여 벡터 DB에 저장
2. **유사도 검색**: 쿼리와 가장 유사한 문서 찾기
3. **의미 기반 검색**: 단순 키워드가 아닌 의미 기반 매칭
## 3. Qwen3-Embedding 모델 로드

### 📦 모델 소개
- **모델명**: Qwen3-Embedding-0.6B (https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
- **파라미터**: 6억 개 (경량 모델)
- **특징**: BFloat16으로 메모리 효율적, 다국어 지원
- **차원**: 1024차원 벡터 출력
# 모델과 토크나이저 로드
print("🔄 Qwen3-Embedding 모델을 로드하고 있습니다...")

model_name = "Qwen/Qwen3-Embedding-0.6B"

# 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# 모델 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModel.from_pretrained(
    model_name,
    torch_dtype=torch.float16, # bfloat16이 맞지만 T4 GPU를 고려해서 수정정
    trust_remote_code=True
).to(device)

model.eval()  # 평가 모드로 설정

print(f"✅ 모델이 성공적으로 로드되었습니다!")
print(f"📍 디바이스: {device}")
print(f"📐 임베딩 차원: {model.config.hidden_size}")
## 4. 임베딩 생성 함수 구현

텍스트를 임베딩 벡터로 변환하는 함수를 구현합니다.
def get_embedding(text, model, tokenizer, device):
    """
    텍스트를 임베딩 벡터로 변환하는 함수

    Args:
        text: 입력 텍스트 (str 또는 list)
        model: 임베딩 모델
        tokenizer: 토크나이저
        device: 연산 디바이스

    Returns:
        numpy array: 임베딩 벡터
    """
    # 텍스트가 리스트가 아니면 리스트로 변환
    if isinstance(text, str):
        text = [text]

    # 토크나이징
    inputs = tokenizer(text, padding=True, truncation=True,
                      max_length=512, return_tensors="pt").to(device)

    # 임베딩 생성
    with torch.no_grad():
        outputs = model(**inputs)
        # Mean pooling: 토큰 임베딩의 평균을 구함
        embeddings = outputs.last_hidden_state.mean(dim=1)

    # CPU로 이동 후 numpy 변환
    return embeddings.cpu().numpy()

# 테스트
test_text = "안녕하세요, 임베딩 테스트입니다."
test_embedding = get_embedding(test_text, model, tokenizer, device)
print(f"✅ 임베딩 생성 완료!")
print(f"📊 임베딩 shape: {test_embedding.shape}")
print(f"📈 벡터 예시 (처음 5개 차원): {test_embedding[0][:5]}")
## 5. 한국어 샘플 텍스트 준비

다양한 주제의 한국어 문장을 준비하여 임베딩의 특성을 관찰합니다.
# 카테고리별 샘플 텍스트
sample_texts = {
    "기술": [
        "인공지능은 미래 기술의 핵심입니다.",
        "머신러닝을 통해 데이터를 분석합니다.",
        "딥러닝 모델이 이미지를 인식합니다.",
        "자연어 처리 기술이 발전하고 있습니다.",
        "빅데이터 분석이 중요해지고 있습니다.",
        "클라우드 컴퓨팅은 IT 인프라의 패러다임을 바꾸고 있습니다.",
        "5G 네트워크는 초고속 통신을 가능하게 합니다.",
        "Quantum computing will redefine problem-solving in the future.",
        "Cybersecurity is becoming increasingly important in the digital era."
    ],
    "음식": [
        "김치는 한국의 대표적인 발효 음식입니다.",
        "불고기는 달콤한 간장 양념이 특징입니다.",
        "비빔밥은 여러 채소와 고기를 섞어 먹습니다.",
        "된장찌개는 구수한 맛이 일품입니다.",
        "삼겹살을 구워서 쌈을 싸먹습니다.",
        "냉면은 여름철 인기 있는 시원한 음식입니다.",
        "떡볶이는 매콤달콤한 맛으로 사랑받습니다.",
        "잡채는 잔치에서 빠질 수 없는 음식입니다.",
        "호떡은 겨울 길거리 간식의 대표주자입니다.",
        "Pizza is one of the most popular foods worldwide.",
        "Sushi combines fresh fish with delicate rice seasoning."
    ],
    "날씨": [
        "오늘은 맑고 화창한 날씨입니다.",
        "내일은 비가 올 예정입니다.",
        "겨울에는 눈이 많이 내립니다.",
        "봄에는 꽃이 아름답게 핍니다.",
        "여름은 덥고 습한 날씨가 계속됩니다.",
        "가을에는 단풍이 물들어 경치가 아름답습니다.",
        "태풍이 북상하면 강풍과 폭우가 동반됩니다.",
        "무더위 속에서는 열사병에 주의해야 합니다.",
        "The weather is unpredictable during the change of seasons.",
        "It often rains in April, making the flowers bloom beautifully."
    ],
    "운동": [
        "축구는 전 세계적으로 인기 있는 스포츠입니다.",
        "야구는 한국에서 많은 사랑을 받습니다.",
        "농구는 빠른 템포가 매력적입니다.",
        "수영은 전신 운동으로 좋습니다.",
        "요가는 몸과 마음의 균형을 잡아줍니다.",
        "등산은 자연 속에서 체력을 기를 수 있는 좋은 활동입니다.",
        "테니스는 민첩성과 집중력을 요구합니다.",
        "골프는 전략과 인내심이 필요한 운동입니다.",
        "달리기는 체력을 단련하고 스트레스를 해소합니다.",
        "탁구는 빠른 반사 신경을 필요로 합니다.",
        "Running a marathon requires months of preparation.",
        "Basketball brings people together as a team sport."
    ]
}


# 모든 텍스트와 라벨 준비
all_texts = []
all_labels = []
all_categories = []

for category, texts in sample_texts.items():
    all_texts.extend(texts)
    all_labels.extend([category] * len(texts))
    all_categories.extend([list(sample_texts.keys()).index(category)] * len(texts))

print(f"📝 총 {len(all_texts)}개의 샘플 텍스트가 준비되었습니다.")
print(f"📂 카테고리: {list(sample_texts.keys())}")

# 데이터프레임으로 정리
df_samples = pd.DataFrame({
    'text': all_texts,
    'category': all_labels
})
print("\n샘플 데이터 미리보기:")
df_samples.head()
## 6. 모든 텍스트의 임베딩 생성

준비한 샘플 텍스트들을 임베딩 벡터로 변환합니다.
print("🔄 임베딩 벡터를 생성하고 있습니다...")

# 배치 처리로 임베딩 생성 (메모리 효율적)
batch_size = 5
embeddings = []

for i in range(0, len(all_texts), batch_size):
    batch_texts = all_texts[i:i+batch_size]
    batch_embeddings = get_embedding(batch_texts, model, tokenizer, device)
    embeddings.append(batch_embeddings)
    print(f"  처리 중: {i+len(batch_texts)}/{len(all_texts)}")

# 모든 임베딩을 하나의 배열로 결합
embeddings = np.vstack(embeddings)

print(f"\n✅ 임베딩 생성 완료!")
## 7. T-SNE를 이용한 임베딩 시각화

고차원 임베딩 벡터를 2차원으로 축소하여 시각화합니다.
import koreanize_matplotlib
print("🔄 T-SNE를 사용하여 차원을 축소하고 있습니다...")

# T-SNE 수행
tsne = TSNE(n_components=2, random_state=42, perplexity=5, n_iter=1000)
embeddings_2d = tsne.fit_transform(embeddings)

print("✅ 차원 축소 완료!")

# 시각화
plt.figure(figsize=(10, 6))

# 카테고리별 색상 설정
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
markers = ['o', 's', '^', 'D', 'v']

# 점과 레이블 그리기
for i, category in enumerate(sample_texts.keys()):
    mask = np.array(all_labels) == category
    plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
               c=colors[i], label=category, s=100, alpha=0.7,
               marker=markers[i], edgecolors='black', linewidth=0.5)

# 각 점 옆에 텍스트 레이블 추가 (처음 10글자만)
for idx, (x, y) in enumerate(embeddings_2d):
    # 텍스트 처음 10글자만 표시
    label_text = all_texts[idx][:10]
    if len(all_texts[idx]) > 10:
        label_text += "..."

    # 카테고리별로 색상 매칭
    category_idx = all_categories[idx]
    text_color = colors[category_idx]

    # 텍스트 추가 (약간 오프셋 적용)
    plt.annotate(label_text,
                xy=(x, y),
                xytext=(3, 3),  # 점으로부터의 오프셋
                textcoords='offset points',
                fontsize=8,
                color='black',
                alpha=0.8,
                ha='left')

plt.xlabel('T-SNE 차원 1', fontsize=12)
plt.ylabel('T-SNE 차원 2', fontsize=12)
plt.title('한국어 텍스트 임베딩의 T-SNE 시각화', fontsize=14, fontweight='bold')
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
## 8. 코사인 유사도 계산

임베딩 벡터 간의 코사인 유사도를 계산하여 의미적 유사성을 측정합니다.
# 유사도 행렬 계산
similarity_matrix = cosine_similarity(embeddings)

# 히트맵으로 시각화
plt.figure(figsize=(14, 12))

# 카테고리별로 정렬된 인덱스 생성
sorted_indices = np.argsort(all_categories)
sorted_similarity = similarity_matrix[sorted_indices][:, sorted_indices]
sorted_labels = [all_labels[i] for i in sorted_indices]

# 히트맵 그리기
ax = sns.heatmap(sorted_similarity,
                cmap='RdYlBu_r',
                vmin=-0.2, vmax=1,
                square=True,
                cbar_kws={'label': '코사인 유사도'})

# 카테고리 경계선 추가
category_counts = [5, 5, 5, 5]  # 각 카테고리별 샘플 수
cumsum = np.cumsum([0] + category_counts)
for i in cumsum[1:-1]:
    ax.axhline(i, color='white', linewidth=2)
    ax.axvline(i, color='white', linewidth=2)

plt.title('텍스트 임베딩 간 코사인 유사도 히트맵', fontsize=14, fontweight='bold')
plt.xlabel('텍스트 인덱스', fontsize=12)
plt.ylabel('텍스트 인덱스', fontsize=12)

# 카테고리 레이블 추가
category_positions = [(cumsum[i] + cumsum[i+1]) / 2 for i in range(len(category_counts))]
ax.set_xticks(category_positions)
ax.set_xticklabels(list(sample_texts.keys()), rotation=45)
ax.set_yticks(category_positions)
ax.set_yticklabels(list(sample_texts.keys()), rotation=0)

plt.tight_layout()
plt.show()
## 9. 의미 기반 검색 실습

쿼리 텍스트와 가장 유사한 문서를 찾는 실습입니다.
def semantic_search(query, documents, embeddings, model, tokenizer, device, top_k=5):
    """
    의미 기반 검색 함수

    Args:
        query: 검색 쿼리
        documents: 문서 리스트
        embeddings: 문서 임베딩 벡터
        top_k: 반환할 상위 문서 개수

    Returns:
        검색 결과 (문서, 유사도 점수)
    """
    # 쿼리 임베딩 생성
    query_embedding = get_embedding(query, model, tokenizer, device)

    # 코사인 유사도 계산
    similarities = cosine_similarity(query_embedding, embeddings)[0]

    # 상위 k개 인덱스
    top_indices = np.argsort(similarities)[::-1][:top_k]

    # 결과 반환
    results = []
    for idx in top_indices:
        results.append({
            'text': documents[idx],
            'category': all_labels[idx],
            'similarity': similarities[idx]
        })

    return results

# 검색 쿼리 예시
queries = [
    "AI와 컴퓨터 비전에 대해 알려줘",
    "한국 전통 요리는 뭐가 있나요?",
    "주말 날씨가 어떨까요?",
    "건강을 위한 활동 추천해줘"
]

print("🔍 의미 기반 검색 실습\n")
print("="*60)

for query in queries:
    print(f"\n📝 쿼리: '{query}'")
    print("-"*40)

    results = semantic_search(query, all_texts, embeddings, model, tokenizer, device, top_k=3)

    for i, result in enumerate(results, 1):
        print(f"{i}. [{result['category']}] {result['text'][:50]}...")
        print(f"   유사도: {result['similarity']:.4f}")

    print()
## 10. 카테고리별 평균 임베딩 분석

각 카테고리의 중심 벡터(centroid)를 계산하고 카테고리 간 거리를 분석합니다.
# 카테고리별 평균 임베딩 계산
category_embeddings = {}
for category in sample_texts.keys():
    mask = np.array(all_labels) == category
    category_embeddings[category] = embeddings[mask].mean(axis=0)

# 카테고리 간 유사도 행렬
categories = list(category_embeddings.keys())
n_categories = len(categories)
category_similarity = np.zeros((n_categories, n_categories))

for i, cat1 in enumerate(categories):
    for j, cat2 in enumerate(categories):
        sim = cosine_similarity(
            category_embeddings[cat1].reshape(1, -1),
            category_embeddings[cat2].reshape(1, -1)
        )[0, 0]
        category_similarity[i, j] = sim

# 시각화
plt.figure(figsize=(10, 8))
sns.heatmap(category_similarity,
            annot=True,
            fmt='.3f',
            xticklabels=categories,
            yticklabels=categories,
            cmap='YlOrRd',
            vmin=0, vmax=1,
            square=True,
            cbar_kws={'label': '코사인 유사도'})

plt.title('카테고리 간 평균 임베딩 유사도', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# 가장 유사한 카테고리 쌍 찾기
similarity_pairs = []
for i in range(n_categories):
    for j in range(i+1, n_categories):
        similarity_pairs.append((
            categories[i],
            categories[j],
            category_similarity[i, j]
        ))

similarity_pairs.sort(key=lambda x: x[2], reverse=True)

print("\n📊 카테고리 간 유사도 순위:")
print("="*40)
for i, (cat1, cat2, sim) in enumerate(similarity_pairs, 1):
    print(f"{i}. {cat1} ↔ {cat2}: {sim:.4f}")
## 11. RAG 시스템 시뮬레이션

간단한 RAG 파이프라인을 구현하여 실제 활용 예시를 보여줍니다.
class SimpleRAG:
    def __init__(self, documents, model, tokenizer, device):
        """
        간단한 RAG 시스템
        """
        self.documents = documents
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        # 문서 임베딩 생성
        print("📚 문서 임베딩을 생성하고 있습니다...")
        self.embeddings = get_embedding(documents, model, tokenizer, device)
        print(f"✅ {len(documents)}개 문서 인덱싱 완료!\n")

    def retrieve(self, query, top_k=3):
        """
        관련 문서 검색
        """
        # 쿼리 임베딩
        query_embedding = get_embedding(query, self.model, self.tokenizer, self.device)

        # 유사도 계산
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]

        # 상위 k개 선택
        top_indices = np.argsort(similarities)[::-1][:top_k]

        retrieved_docs = []
        for idx in top_indices:
            retrieved_docs.append({
                'document': self.documents[idx],
                'similarity': similarities[idx]
            })

        return retrieved_docs

    def answer(self, query):
        """
        RAG 기반 답변 생성 (시뮬레이션)
        """
        # 1. Retrieval: 관련 문서 검색
        retrieved = self.retrieve(query, top_k=3)

        # 2. Augmentation: 컨텍스트 구성
        context = "\n".join([doc['document'] for doc in retrieved])

        # 3. Generation: 답변 생성 (시뮬레이션)
        print(f"💬 쿼리: {query}\n")
        print("📄 검색된 관련 문서:")
        for i, doc in enumerate(retrieved, 1):
            print(f"  {i}. {doc['document'][:50]}...")
            print(f"     (유사도: {doc['similarity']:.4f})")

        print(f"\n🤖 생성된 답변 (시뮬레이션):")
        print(f"  '{query}'에 대한 답변입니다.")
        print(f"  검색된 문서들을 바탕으로 다음과 같이 설명할 수 있습니다:")
        print(f"  {retrieved[0]['document']}")

        return context

# RAG 시스템 초기화
rag_system = SimpleRAG(all_texts, model, tokenizer, device)

# 테스트 쿼리
test_queries = [
    "인공지능 기술에 대해 설명해줘",
    "맛있는 한국 음식 추천해줘",
    "오늘 운동하기 좋은 날씨일까?"
]

print("🚀 RAG 시스템 데모\n")
print("="*60)

for query in test_queries:
    rag_system.answer(query)
    print("\n" + "="*60 + "\n")