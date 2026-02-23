# [실습] 랭체인의 Document Loader와 Splitter

랭체인의 기능을 사용하여, 다양한 파일을 불러오고 전처리해 보겠습니다.
필수 라이브러리를 설치합니다.
!pip install langchain==0.3.27 langchain-community==0.3.27 langchain-experimental jq langchain-openai tiktoken pypdf beautifulsoup4 lxml python-docx pandas openpyxl -q
import os
from typing import List
import json
import pandas as pd
from pprint import pprint
from dotenv import load_dotenv

load_dotenv('.env', override=True)

# LangChain Document Loaders
from langchain.document_loaders import (
    TextLoader,
    PyPDFLoader,
    WebBaseLoader,
    CSVLoader,
    JSONLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader,
    DirectoryLoader,
    NotebookLoader,
    GitLoader
)

from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter,
    MarkdownHeaderTextSplitter,
    HTMLHeaderTextSplitter,
    RecursiveJsonSplitter,
    Language
)
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

print("필수 라이브러리 임포트 완료")
청킹은 RAG 성능에 매우 중요한 영향을 미칩니다. 

- **청크 크기(Chunk Size)**: 너무 크면 검색 정확도 감소, 너무 작으면 문맥 손실
- **오버랩(Overlap)**: 청크 간 연결성 유지
- **구조 보존**: 문서의 논리적 구조 유지
- **의미적 일관성**: 관련 정보를 같은 청크에 유지
WebBaseLoader는 웹 페이지를 로드합니다.
# 웹 페이지 로더
from langchain.document_loaders import WebBaseLoader

# 여러 웹 페이지를 한번에 로드
web_loader = WebBaseLoader([
    "https://python.langchain.com/docs/get_started/introduction",
    "https://python.langchain.com/docs/integrations/providers/openai/"
])

# 웹 페이지 로드
try:
    web_documents = web_loader.load()
    print(f"로드된 웹 페이지 수: {len(web_documents)}")
    
    for i, doc in enumerate(web_documents[:2]):
        print(f"\n웹 페이지 {i+1}:")
        print(f"URL: {doc.metadata.get('source', 'Unknown')}")
        print(f"제목: {doc.metadata.get('title', 'No title')}")
        print(f"전체 길이:" )
        print(f"내용 미리보기: {doc.page_content[:200]}...")
except Exception as e:
    print(f"웹 페이지 로딩 오류: {e}")
    print("오프라인 환경이거나 웹사이트 접속이 제한된 경우입니다.")
web_documents[1]
CSVLoader는 CSV 파일을 로드합니다.    
임의의 Pandas DataFrame을 만들고 실행해 보겠습니다.
# 샘플 CSV 데이터 생성
import pandas as pd

# 샘플 데이터프레임 생성
data = {
    'product_name': ['노트북 Pro', '무선 마우스', '기계식 키보드', 'USB-C 허브', '웹캠 HD'],
    'category': ['컴퓨터', '액세서리', '액세서리', '액세서리', '액세서리'],
    'price': [1500000, 35000, 120000, 45000, 80000],
    'description': [
        '고성능 노트북으로 개발자와 디자이너에게 적합합니다.',
        '편안한 그립감과 정확한 트래킹을 제공하는 무선 마우스입니다.',
        '체리 MX 스위치를 사용한 기계식 키보드로 타이핑 감각이 뛰어납니다.',
        '다양한 포트를 지원하는 USB-C 허브입니다.',
        '1080p 해상도를 지원하는 고화질 웹캠입니다.'
    ],
    'stock': [50, 200, 100, 150, 75]
}

df = pd.DataFrame(data)
df.to_csv('products.csv', index=False, encoding='utf-8')
print("CSV 파일이 생성되었습니다.")


# CSV 로더 사용
csv_loader = CSVLoader(
    file_path='products.csv',
    encoding='utf-8'
)

csv_documents = csv_loader.load()
print(f"\n로드된 CSV 행 수: {len(csv_documents)}")

# 각 행의 내용 확인
for i, doc in enumerate(csv_documents[:3]):
    print(f"\n행 {i+1}:")
    print(f"내용: {doc.page_content}")
    print(f"메타데이터: {doc.metadata}")
JSON 데이터도 불러올 수 있습니다.
# 샘플 JSON 데이터 생성
json_data = {
    "courses": [
        {
            "id": 1,
            "title": "Python 기초",
            "instructor": "김파이썬",
            "duration": "4주",
            "level": "초급",
            "topics": ["변수", "조건문", "반복문", "함수"],
            "description": "파이썬 프로그래밍의 기초를 배우는 과정입니다."
        },
        {
            "id": 2,
            "title": "머신러닝 입문",
            "instructor": "이AI",
            "duration": "8주",
            "level": "중급",
            "topics": ["지도학습", "비지도학습", "신경망", "딥러닝 기초"],
            "description": "머신러닝의 기본 개념과 알고리즘을 학습합니다."
        },
        {
            "id": 3,
            "title": "LangChain 마스터",
            "instructor": "박체인",
            "duration": "6주",
            "level": "고급",
            "topics": ["Document Loader", "Text Splitter", "Embeddings", "Vector Store", "Chains", "Agents"],
            "description": "LangChain을 활용한 AI 애플리케이션 개발 방법을 배웁니다."
        }
    ]
}

# JSON 파일로 저장
with open('courses.json', 'w', encoding='utf-8') as f:
    json.dump(json_data, f, ensure_ascii=False, indent=2)
print("JSON 파일이 생성되었습니다.")

# JSONLoader 사용 (jq 스타일 필터링)
from langchain.document_loaders import JSONLoader

def metadata_func(record: dict, metadata: dict) -> dict:
    """메타데이터 추출 함수"""
    metadata["instructor"] = record.get("instructor", "Unknown")
    metadata["level"] = record.get("level", "Unknown")
    return metadata

# JSON 로더 설정
json_loader = JSONLoader(
    file_path='courses.json',
    jq_schema='.courses[]',  
    metadata_func=metadata_func,
    content_key='description',
)

json_documents = json_loader.load()
print(f"\n로드된 JSON 문서 수: {len(json_documents)}")

# 각 문서 확인
for i, doc in enumerate(json_documents):
    print(f"\n문서 {i+1}:")
    print(f"내용: {doc.page_content[:200]}...")
    print(f"메타데이터: {doc.metadata}")
이외에도 다양한 Document Loader를 선택할 수 있습니다.
- **TextLoader**: 일반 텍스트 파일 (.txt, .log)
- **PyPDFLoader**: PDF 문서 (페이지별 메타데이터 보존)
- **WebBaseLoader**: 웹 페이지 크롤링
- **CSVLoader**: 구조화된 표 형식 데이터
- **JSONLoader**: API 응답, 설정 파일
- **DirectoryLoader**: 대량 파일 일괄 처리
- **GitLoader**: 버전 관리된 코드베이스
불러온 데이터는 Text Splitter를 통해 분할합니다.   
# 샘플 텍스트 파일 생성
sample_text = """
RAG(Retrieval-Augmented Generation)는 2020년 Facebook AI Research 팀이 처음 제안한 개념으로, 대규모 언어 모델의 한계를 극복하기 위해 고안되었습니다.
기존의 언어 모델들이 학습 시점의 정보만을 내재화하여 최신 정보나 특정 도메인 지식에 대한 정확성이 떨어지는 문제를 해결하고자 했습니다.
외부 지식 베이스에서 관련 정보를 검색하여 언어 모델의 생성 과정에 통합함으로써, 보다 정확하고 신뢰할 수 있는 응답을 생성할 수 있게 되었습니다.
초기 RAG 시스템은 DPR(Dense Passage Retrieval)과 같은 밀집 벡터 검색 기법과 BART나 T5 같은 생성 모델을 결합한 형태였습니다.
사용자 쿼리를 벡터로 변환하고, 이와 유사한 문서들을 벡터 데이터베이스에서 검색한 후, 검색된 문서들을 컨텍스트로 활용하여 답변을 생성하는 파이프라인 구조를 가졌습니다.
이 시기의 RAG는 주로 오픈 도메인 질의응답 시스템에 적용되었으며, Wikipedia와 같은 대규모 텍스트 코퍼스를 지식 소스로 활용했습니다.
하지만 검색과 생성이 독립적으로 최적화되어 통합적인 성능 향상에 한계가 있었습니다.
2021년부터 2022년까지 RAG 시스템은 여러 방향으로 발전했습니다.
Fusion-in-Decoder와 같은 모델은 여러 개의 검색된 문서를 병렬로 처리하여 보다 효과적으로 정보를 통합할 수 있게 했습니다.
RETRO(Retrieval-Enhanced Transformer)는 사전 학습 단계부터 검색을 통합하여 모델의 파라미터 효율성을 크게 개선했습니다.
또한 검색 단계에서도 BM25와 같은 전통적인 키워드 기반 검색과 벡터 검색을 결합한 하이브리드 검색 방법이 등장했고, 재순위화(re-ranking) 기법을 통해 검색 품질을 향상시켰습니다.
이 시기에는 도메인 특화 RAG 시스템도 등장하여 의료, 법률, 금융 등 전문 분야에서의 활용이 시작되었습니다.
2023년 이후 대규모 언어 모델의 급속한 발전과 함께 RAG 기술도 크게 진화했습니다.
ChatGPT와 같은 대화형 AI의 등장으로 RAG는 단순한 질의응답을 넘어 복잡한 대화 컨텍스트를 유지하면서도 외부 지식을 효과적으로 활용할 수 있게 되었습니다.
특히 LangChain, LlamaIndex와 같은 프레임워크의 등장으로 RAG 시스템 구축이 표준화되고 접근성이 높아졌습니다.
또한 Self-RAG, CRAG(Corrective RAG)와 같은 기법들이 제안되어 검색된 정보의 관련성을 자체적으로 평가하고 필요시 재검색하는 등의 자기 수정 능력을 갖추게 되었습니다.
벡터 데이터베이스 기술도 크게 발전하여 Pinecone, Weaviate, Qdrant 등 전문화된 솔루션들이 등장했습니다.
현재 RAG는 기업의 지식 관리 시스템, 고객 지원 챗봇, 코드 생성 도구, 연구 보조 시스템 등 다양한 분야에서 활발히 적용되고 있습니다.
특히 기업 내부 문서와 데이터를 활용한 엔터프라이즈 RAG 솔루션이 주목받고 있으며, 프라이버시와 보안을 고려한 온프레미스 RAG 시스템도 증가하고 있습니다.
멀티모달 RAG의 발전으로 텍스트뿐만 아니라 이미지, 표, 그래프 등 다양한 형태의 정보를 통합적으로 처리할 수 있게 되었습니다.
또한 GraphRAG와 같이 지식 그래프를 활용한 구조화된 정보 검색과 추론을 결합한 고급 기법들도 등장했습니다.
향후 RAG 기술은 더욱 정교하고 효율적인 방향으로 발전할 것으로 예상됩니다.
에이전틱 RAG는 단순 검색을 넘어 능동적으로 정보를 탐색하고 추론하는 능력을 갖추게 될 것입니다.
또한 개인화된 RAG 시스템을 통해 사용자의 선호도와 컨텍스트를 고려한 맞춤형 정보 제공이 가능해질 것입니다.
검색 효율성 측면에서는 증분 인덱싱과 실시간 업데이트 기능이 강화되어 동적으로 변화하는 지식 베이스를 효과적으로 관리할 수 있을 것입니다.
무엇보다 RAG는 대규모 언어 모델의 환각 현상을 줄이고 신뢰성을 높이는 핵심 기술로서, AI의 실용적 적용을 위한 필수 요소로 자리잡을 것으로 전망됩니다."""

# 텍스트 파일로 저장
with open('sample_ai.txt', 'w', encoding='utf-8') as f:
    f.write(sample_text)

print("샘플 파일이 생성되었습니다.")
### RecursiveCharacterTextSplitter - 기본 분할 전략
# 텍스트 로더로 파일 로드
loader = TextLoader('sample_ai.txt', encoding='utf-8')
documents = loader.load()

len(documents[0].page_content)

# RecursiveCharacterTextSplitter 설정
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,  # 청크 크기
    chunk_overlap=30,  # 청크 간 중복
    length_function=len,
    separators=["\n\n", "\n", ".", " ", ""]  # 분할 우선순위
)

# 문서 분할
splits = text_splitter.split_documents(documents)

print(f"원본 문서 수: {len(documents)}")
print(f"분할된 청크 수: {len(splits)}\n")

# 처음 3개 청크 확인
for i, split in enumerate(splits[:3]):
    print(f"청크 {i+1}:")
    print(f"내용: {split.page_content[:80]}...")
    print(f"길이: {len(split.page_content)}\n")
토큰 기반의 분할은 아래와 같이 만들 수 있습니다.
# RecursiveCharacterTextSplitter 설정
token_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    model_name = 'gpt-5-mini',
    chunk_size=300,  # 청크 크기
    chunk_overlap=30,  # 청크 간 중복
    separators=["\n\n", "\n", ".", " ", ""]  # 분할 우선순위
)

# 문서 분할
splits = token_splitter.split_documents(documents)

print(f"원본 문서 수: {len(documents)}")
print(f"분할된 청크 수: {len(splits)}\n")

# 처음 3개 청크 확인
for i, split in enumerate(splits[:3]):
    print(f"청크 {i+1}:")
    print(f"내용: {split.page_content[:80]}...")
    print(f"길이: {len(split.page_content)}\n")
**💡 실무 팁**
- 한글 문서는 토큰 수가 영어보다 많으므로 chunk_size 조정 필요
- overlap은 보통 chunk_size의 10-20% 권장


**💡 토큰 기반 분할의 장점**
- 모델의 컨텍스트 길이 제한에 정확히 맞춤
- 다국어 텍스트에서 더 일관된 분할
- API 비용 예측 가능
단순히 글자/토큰 기반의 분할 이외에도, 다양한 기준에 맞춘 분할이 가능합니다.
# 샘플 Markdown 문서 생성
markdown_document = """
# LangChain 소개

LangChain은 대규모 언어 모델(LLM)을 활용한 애플리케이션 개발 프레임워크입니다.

## 주요 구성요소

### 1. Models

LangChain은 다양한 LLM 제공자를 지원합니다.
- OpenAI
- Anthropic Claude
- Google Generative AI
- Ollama

### 2. Prompts

프롬프트 템플릿을 통해 재사용 가능한 프롬프트를 만들 수 있습니다.
변수를 사용하여 동적 프롬프트 생성이 가능합니다.

- ChatPromptTemplate
- PromptTemplate
- FewShotPromptTemplate

## 고급 기능

### Chains

여러 컴포넌트를 연결하여 복잡한 워크플로우를 구성합니다.
체인을 연결하는 문법은 LCEL(LangChain Expression Language)라고 부릅니다.

### Agents

bind_tools() 과 ToolMessage를 활용하면
도구를 사용하여 작업을 수행하는 자율적인 에이전트를 만들 수 있습니다.
"""

# Markdown 헤더 기반 분할 설정
headers_to_split_on = [
    ("#", "제목1"),
    ("##", "제목2"),
    ("###", "제목3"),
]

markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on,
    strip_headers=False  # 헤더를 콘텐츠에 포함
)

# Markdown 문서 분할
md_header_splits = markdown_splitter.split_text(markdown_document)

print(f"분할된 섹션 수: {len(md_header_splits)}\n")

# 각 섹션의 내용과 메타데이터 확인
for i, doc in enumerate(md_header_splits[:3]):
    print(f"섹션 {i+1}:")
    print(f"메타데이터: {doc.metadata}")
    print(f"내용 미리보기: {doc.page_content[:100]}...\n")
**🔍 핵심 포인트**
- 문서의 계층 구조가 메타데이터로 보존됨
- 검색 시 특정 섹션을 타겟팅하기 용이
- 긴 문서의 구조적 탐색 가능
HTML과 같이 복잡한 구조는 메타데이터를 선택할 수 있습니다.
# 샘플 HTML 문서
html_string = """
<!DOCTYPE html>
<html>
<body>
    <h1>웹 개발 기초</h1>
    <p>웹 개발은 프론트엔드와 백엔드로 구분됩니다.</p>
    
    <h2>프론트엔드 기술</h2>
    <p>사용자 인터페이스를 담당합니다.</p>
    
    <h3>HTML</h3>
    <p>웹 페이지의 구조를 정의합니다.</p>
    <ul>
        <li>태그를 사용한 마크업</li>
        <li>시맨틱 HTML5 요소</li>
    </ul>
    
    <h3>CSS</h3>
    <p>스타일과 레이아웃을 담당합니다.</p>
    <table>
        <tr>
            <th>속성</th>
            <th>설명</th>
        </tr>
        <tr>
            <td>color</td>
            <td>텍스트 색상</td>
        </tr>
        <tr>
            <td>margin</td>
            <td>외부 여백</td>
        </tr>
    </table>
    
    <h2>백엔드 기술</h2>
    <p>서버 측 로직을 처리합니다.</p>
</body>
</html>
"""

# HTML 헤더 기반 분할
headers_to_split_on = [
    ("h1", "제목1"),
    ("h2", "제목2"),
    ("h3", "제목3"),
]

html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
html_header_splits = html_splitter.split_text(html_string)

print(f"HTML 분할 결과: {len(html_header_splits)}개 섹션\n")

# 구조화된 요소(테이블, 리스트) 보존 확인
for i, split in enumerate(html_header_splits):
    print(f"섹션 {i+1}:")
    print(f"메타데이터: {split.metadata}")
    if 'table' in split.page_content or 'ul' in split.page_content:
        print("✅ 구조화된 요소 포함")
    print(f"내용 길이: {len(split.page_content)} 글자")
    print(f"내용: {split.page_content[0:10]}\n")
소스 코드의 종류에 따른 분할도 지원합니다.
# 샘플 Python 코드
python_code = """
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class DataProcessor:
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None
        self.processed_data = None
    
    def load_data(self):
        try:
            self.data = pd.read_csv(self.data_path)
            print(f"데이터 로드 완료: {len(self.data)} 행")
            return self.data
        except Exception as e:
            print(f"에러 발생: {e}")
            return None
    
    def preprocess(self):
        if self.data is None:
            raise ValueError("데이터를 먼저 로드하세요")
        
        # 결측치 처리
        self.processed_data = self.data.fillna(0)
        
        # 정규화
        numeric_columns = self.processed_data.select_dtypes(include=[np.number]).columns
        self.processed_data[numeric_columns] = (self.processed_data[numeric_columns] - 
                                                self.processed_data[numeric_columns].mean()) / \
                                               self.processed_data[numeric_columns].std()
        return self.processed_data

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # 모델 학습 코드
    return X_train, X_test, y_train, y_test

# 메인 실행 코드
if __name__ == "__main__":
    processor = DataProcessor("data.csv")
    processor.load_data()
    processor.preprocess()
"""

# Python 코드 전용 분할기
python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=200,
    chunk_overlap=50
)

python_docs = python_splitter.create_documents([python_code])

print(f"Python 코드 분할 결과: {len(python_docs)}개 청크\n")

# 각 청크 확인
for i, doc in enumerate(python_docs[:3]):
    print(f"청크 {i+1}:")
    print(doc.page_content)
    print("-" * 50)
# JavaScript 코드 예시
js_code = """
// React 컴포넌트 예시
import React, { useState, useEffect } from 'react';
import axios from 'axios';

const UserProfile = ({ userId }) => {
    const [user, setUser] = useState(null);
    const [loading, setLoading] = useState(true);
    
    useEffect(() => {
        fetchUserData();
    }, [userId]);
    
    const fetchUserData = async () => {
        try {
            const response = await axios.get(`/api/users/${userId}`);
            setUser(response.data);
        } catch (error) {
            console.error('Error fetching user:', error);
        } finally {
            setLoading(false);
        }
    };
    
    if (loading) return <div>Loading...</div>;
    if (!user) return <div>User not found</div>;
    
    return (
        <div className="user-profile">
            <h2>{user.name}</h2>
            <p>{user.email}</p>
        </div>
    );
};

export default UserProfile;
"""

# JavaScript 코드 분할
js_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.JS,
    chunk_size=150,
    chunk_overlap=30
)

js_docs = js_splitter.create_documents([js_code])
print(f"JavaScript 코드 분할: {len(js_docs)}개 청크")

# 각 청크 확인
for i, doc in enumerate(js_docs[:3]):
    print(f"청크 {i+1}:")
    print(doc.page_content[:150])
    print("-" * 50)
**💡 코드 분할 팁**
- 함수/클래스 경계를 고려한 분할
- 언어별 구문 규칙 반영
- import 문과 함수 정의 분리 고려
### JSON 데이터 분할
# 복잡한 JSON 데이터 예시
json_data = {
    "company": "TechCorp",
    "founded": 2010,
    "departments": [
        {
            "name": "Engineering",
            "employees": 150,
            "teams": [
                {
                    "name": "Backend",
                    "members": 30,
                    "technologies": ["Python", "Java", "Go"],
                    "projects": [
                        {"name": "API Gateway", "status": "active"},
                        {"name": "Data Pipeline", "status": "planning"}
                    ]
                },
                {
                    "name": "Frontend",
                    "members": 25,
                    "technologies": ["React", "Vue", "TypeScript"],
                    "projects": [
                        {"name": "Admin Dashboard", "status": "active"},
                        {"name": "Mobile App", "status": "development"}
                    ]
                },
                {
                    "name": "DevOps",
                    "members": 15,
                    "technologies": ["Docker", "Kubernetes", "Terraform"],
                    "infrastructure": {
                        "cloud_providers": ["AWS", "GCP"],
                        "monitoring_tools": ["Prometheus", "Grafana"],
                        "ci_cd": "Jenkins"
                    }
                }
            ]
        },
        {
            "name": "Marketing",
            "employees": 45,
            "campaigns": [
                {
                    "name": "Summer Launch",
                    "budget": 50000,
                    "channels": ["Social Media", "Email", "Content Marketing"]
                }
            ]
        }
    ],
    "products": [
        {
            "name": "Product A",
            "version": "2.5.0",
            "features": ["Feature 1", "Feature 2", "Feature 3"]
        },
        {
            "name": "Product B",
            "version": "1.2.0",
            "features": ["Feature A", "Feature B"]
        }
    ]
}

# RecursiveJsonSplitter 사용
json_splitter = RecursiveJsonSplitter(max_chunk_size=300)

# JSON을 청크로 분할
json_chunks = json_splitter.split_json(json_data=json_data)

print(f"JSON 데이터 분할 결과: {len(json_chunks)}개 청크\n")

# 각 청크 확인
for i, chunk in enumerate(json_chunks[:3]):
    print(f"청크 {i+1}:")
    print(json.dumps(chunk, indent=2, ensure_ascii=False)[:300])
    print(f"청크 크기: {len(json.dumps(chunk))} 문자")
    print("-" * 50)
**🔍 JSON 분할 특징**
- 중첩된 구조 유지
- 논리적 단위로 분할
- API 응답 데이터 처리에 유용
## Semantic Chunking

시맨틱 청킹은 의미 기반으로 문서를 분할하는 방법입니다.   
가장 고도화된 청킹 방법으로 볼 수 있습니다.
# 의미적으로 구분된 긴 텍스트 예시
long_text = """
인공지능의 역사는 1950년대로 거슬러 올라갑니다. 앨런 튜링은 기계가 생각할 수 있는지를 
판단하는 튜링 테스트를 제안했습니다. 이는 AI 연구의 시작점이 되었습니다.

1960년대와 1970년대는 AI의 황금기로 불립니다. 전문가 시스템이 개발되었고, 
LISP와 같은 AI 전용 프로그래밍 언어가 만들어졌습니다.

1980년대 후반부터 1990년대 초반까지 AI 겨울이라 불리는 침체기가 있었습니다. 
과도한 기대와 실망으로 인해 연구 자금이 줄어들었습니다.

2000년대 들어 빅데이터와 컴퓨팅 파워의 발전으로 머신러닝이 부활했습니다. 
특히 딥러닝의 등장은 AI 분야에 혁명을 일으켰습니다.

최근에는 트랜스포머 아키텍처가 등장했습니다. BERT, GPT 시리즈 같은 대규모 언어 모델이 
자연어 처리 분야를 완전히 바꾸어 놓았습니다. 이들은 번역, 요약, 질의응답 등 
다양한 작업에서 인간 수준의 성능을 보여주고 있습니다.

컴퓨터 비전 분야도 급속히 발전했습니다. CNN(Convolutional Neural Network)을 통해 
이미지 인식의 정확도가 크게 향상되었습니다. 자율주행차, 의료 영상 진단 등에 활용되고 있습니다.

강화학습은 게임과 로봇공학에서 큰 성과를 거두었습니다. AlphaGo가 바둑에서 인간을 이긴 것은 
역사적인 순간이었습니다. 이제 AI는 복잡한 전략적 의사결정도 수행할 수 있습니다.
"""

embeddings = OpenAIEmbeddings()

# Semantic Chunker 생성 - percentile 방식
semantic_chunker_percentile = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=50  # 상위 50% 이상의 차이가 나면 분할
)

# 의미 기반 분할 수행
semantic_docs = semantic_chunker_percentile.create_documents([long_text])

print(f"의미 기반 분할 결과: {len(semantic_docs)}개 청크\n")

for i, doc in enumerate(semantic_docs):
    print(f"청크 {i+1} (의미적으로 연관된 내용):")
    print(doc.page_content[:200])
    print(f"청크 길이: {len(doc.page_content)} 문자\n")
    print("-" * 50)

랭체인의 Semantic Chunking은 다양한 Breakpoint 전략을 지원합니다.  

전체 문장들 간의 거리를 모두 계산한 뒤, Breakpoint를 기준으로 나눕니다.


💡 Breakpoint 전략 선택 가이드:
- Percentile: 일반적인 문서에 적합
- Standard Deviation: 일관된 스타일의 문서
- Interquartile: 이상치가 많은 문서
- Gradient: 급격한 주제 변화가 있는 문서
# Standard Deviation 방식
semantic_chunker_std = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=3  # 3 표준편차 이상 차이
)

# Interquartile 방식
semantic_chunker_iqr = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="interquartile"
)

# Gradient 방식
semantic_chunker_gradient = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="gradient",
    breakpoint_threshold_amount=95
)

# 각 방식으로 분할 수행
strategies = {
    "Percentile": semantic_chunker_percentile,
    "Standard Deviation": semantic_chunker_std,
    "Interquartile": semantic_chunker_iqr,
    "Gradient": semantic_chunker_gradient
}

print("각 전략별 분할 결과 비교:\n")
for name, chunker in strategies.items():
    docs = chunker.create_documents([long_text])
    print(f"{name}: {len(docs)}개 청크")
    avg_length = sum(len(doc.page_content) for doc in docs) / len(docs)
    print(f"  평균 청크 길이: {avg_length:.0f} 문자\n")



📋 문서 타입별 추천 Splitter:

📄 MARKDOWN    
   추천: MarkdownHeaderTextSplitter   
   이유: 헤더 기반 구조 보존   
   설정: `{'headers_to_split_on': ['#', '##', '###']}`

📄 HTML   
   추천: HTMLSemanticPreservingSplitter   
   이유: 테이블, 리스트 등 구조 보존   
   설정: `{'elements_to_preserve': ['table', 'ul', 'ol']}`   

📄 CODE   
   추천: RecursiveCharacterTextSplitter.from_language   
   이유: 언어별 구문 고려   
   설정: `{'chunk_size': 200, 'chunk_overlap': 50}`   

📄 JSON   
   추천: RecursiveJsonSplitter   
   이유: 중첩 구조 유지   
   설정: `{'max_chunk_size': 500}`   

📄 RESEARCH_PAPER   
   추천: SemanticChunker   
   이유: 의미적 연관성 기반 분할   
   설정: `{'breakpoint_threshold_type': 'percentile'}`   
## 🎯 핵심 정리 및 Best Practices

### Document Loader 선택 가이드

| 파일 타입 | 추천 Loader | 특징 |
|---------|------------|------|
| `.txt` | TextLoader | 단순 텍스트, 인코딩 지정 가능 |
| `.pdf` | PyPDFLoader | 페이지별 메타데이터 보존 |
| `.csv` | CSVLoader | 행 단위 자동 분할 |
| `.json` | JSONLoader | jq 스타일 필터링 지원 |
| `.html` | WebBaseLoader | CSS 선택자로 특정 요소 추출 |
| 디렉토리 | DirectoryLoader | 대량 파일 일괄 처리 |
| 웹페이지 | WebBaseLoader | 실시간 웹 크롤링 |

### Text Splitter 선택 가이드

| 문서 타입 | 추천 Splitter | 핵심 고려사항 |
|---------|-------------|------------|
| 일반 텍스트 | RecursiveCharacterTextSplitter | 청크 크기와 오버랩 조정 |
| 토큰 제한 | TokenTextSplitter | API 토큰 제한 준수 |
| Markdown/Docs | MarkdownHeaderTextSplitter | 헤더 레벨 기반 구조 보존 |
| HTML/Web | HTMLHeaderTextSplitter | 테이블, 리스트 구조 유지 |
| 소스 코드 | Language-specific Splitter | 구문 단위 분할 |
| JSON/API | RecursiveJsonSplitter | 중첩 구조 보존 |
| 연구 논문 | SemanticChunker | 의미적 연관성 기반 |

### 🚀 프로덕션 체크리스트

1. **로더 최적화**
   - 대량 파일 처리 시 DirectoryLoader + 멀티스레딩
   - 메모리 효율을 위한 lazy loading 고려
   - 오류 처리 및 재시도 로직 구현

2. **청킹 최적화**
   - 임베딩 모델의 토큰 제한 고려
   - 오버랩은 chunk_size의 10-20% 권장
   - 문서 타입별 맞춤 전략 적용

3. **메타데이터 관리**
   - 출처, 페이지, 섹션 정보 보존
   - 검색 시 필터링을 위한 메타데이터 활용
   - 버전 관리 및 타임스탬프 추가

4. **성능 모니터링**
   - 청크 크기 분포 분석
   - 검색 정확도 측정
   - 응답 품질 평가

### 💡 실무 팁

- **하이브리드 접근**: 구조 기반 1차 분할 → 크기 기반 2차 분할
- **동적 청킹**: 문서 타입 자동 감지 및 적응적 처리
- **캐싱 전략**: 자주 사용되는 문서 사전 처리 및 저장
- **A/B 테스트**: 다양한 분할 전략의 성능 비교 측정
- **다국어 지원**: 한글 등 non-ASCII 문자 처리 시 토큰 기반 분할 권장