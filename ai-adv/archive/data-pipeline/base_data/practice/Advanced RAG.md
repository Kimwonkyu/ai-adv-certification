# [실습] Advanced RAG

RAG의 기본 베이스 체인에서 시작하여, 다양한 기능을 추가해 보겠습니다.   
### 라이브러리 설치  

랭체인 관련 라이브러리와 벡터 데이터베이스 라이브러리를 설치합니다.   
!pip install jsonlines openai langchain==0.3.27 langchain-openai langchain-community==0.3.27 langchain-chroma tiktoken rank_bm25 pymupdf kiwipiepy
### LLM과 임베딩 모델 불러오기
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

load_dotenv('.env', override=True)
if os.environ.get('OPENAI_API_KEY'):
    print('OpenAI API 키 확인')

llm = ChatOpenAI(model="gpt-4.1-mini", temperature = 0)
sllm = ChatOpenAI(model='gpt-4.1-nano', temperature = 0.7)
reasoning_llm = ChatOpenAI(model='gpt-5-mini')

embeddings = OpenAIEmbeddings(model = 'text-embedding-3-large', chunk_size=100)
reports.zip 파일을 압축 해제합니다.
import zipfile

with zipfile.ZipFile('reports.zip', 'r') as zip_ref:
    zip_ref.extractall('.')
## 데이터 불러오기
폴더에 포함된 문서들을 pdf 로더로 불러옵니다.   
glob 라이브러리를 사용합니다.
from langchain_core.documents import Document
from glob import glob

# 모든 PDF 파일을 glob으로 찾음
pdf_files = glob("reports/*.pdf")
pdf_files
from langchain_community.document_loaders import PyMuPDFLoader

# 각 PDF 파일에서 페이지별로 내용을 불러와 하나로 합침
all_papers=[]

for i, path_paper in enumerate(pdf_files):
    loader = PyMuPDFLoader(path_paper)
    pages = loader.load()
    # 문서가 주어지면 Document List를 출력
    # Page별 저장 --> 청킹을 다시 수행

    doc = Document(page_content='', metadata = {'index':i, 'source':pages[0].metadata['source']})
    for page in pages:
        doc.page_content += page.page_content +' '

    doc.page_content = doc.page_content.replace('\n', ' ')
    for _ in range(10):
        doc.page_content = doc.page_content.replace('  ', ' ')
        doc.page_content = doc.page_content.replace('..', '.')

    all_papers.append(doc)

print(len(all_papers))
all_papers[0].page_content[:1000]
글자 수와 토큰 수를 확인해 보겠습니다.
import tiktoken

encoder = tiktoken.encoding_for_model('gpt-4.1-mini') # 4.1, 5, 4o 모두 동일
for paper in all_papers:
    print(len(paper.page_content), len(encoder.encode(paper.page_content)), paper.metadata['source'])
# 토큰 단위로 청킹하기   

tiktoken이나 huggingface를 이용해 토큰 단위 청킹을 수행합니다.
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken
token_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    model_name="gpt-4.1-mini",
    chunk_size=2000,
    chunk_overlap=400,
)
# from_huggingface_tokenizer : 허깅페이스 모델에서 토크나이저 가져오기

token_chunks = token_splitter.split_documents(all_papers)
print(len(token_chunks))
벡터 DB를 구성합니다.   
from langchain_chroma import Chroma
from tqdm import tqdm

Chroma().delete_collection()
db = Chroma(embedding_function=embeddings,
                           persist_directory="./chroma2",
                           collection_metadata={'hnsw:space':'l2'},
                           collection_name='finance',
                           )

db.add_documents(token_chunks)

retriever = db.as_retriever(search_kwargs={"k": 5})

# filter 옵션을 통해 특정 메타데이터를 가진 벡터만 검색 가능
# retriever = db.as_retriever(search_kwargs={"k": 5,"filter":{'author':'Hyungho Byun'}})
RAG 체인을 구현합니다.
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate([
    ("user", '''당신은 QA(Question-Answering)을 수행하는 Assistant입니다.
다음의 Context를 이용하여 Question에 한국어로 답변하세요.
정확한 답변을 제공하세요.
만약 모든 Context를 다 확인해도 정보가 없다면, "정보가 부족하여 답변할 수 없습니다."를 출력하세요.
---
Context: {context}
---
Question: {question}''')])
prompt.pretty_print()
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def format_docs(docs):
    return "\n---\n".join(["Content: " + doc.page_content for doc in docs])
    # join : 구분자를 기준으로 스트링 리스트를 하나의 스트링으로 연결

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}

    | prompt
    | llm
    | StrOutputParser()
)
다양한 질문을 통해 결과를 확인해 보겠습니다.
private_questions = [
    '이 회사의 사업은 크게 두 부문으로 나눠집니다. 각각은 어떤 부문입니까?',
    'IT서비스는 몇 개의 분야로 나눠집니까? 각각은 무엇입니까?',
    '2024년과 2025년에 걸쳐, 매출 비중은 어떻게 변화했나요?',
    '2024년 6월말 기준, 연결 재무제표의 유동자산은 얼마인가요?',
    'GPUaaS 서비스가 무엇입니까?',
    '회사의 연구개발 담당조직은 어떤 분야의 핵심 기술을 연구하고 있습니까?',
    '2021-2022년 사이에 있었던 합병 사례를 모두 파악하여, 표로 나타내세요.'
]
result = rag_chain.batch(private_questions)
for i, ans in enumerate(result):
    print(f"Question: {private_questions[i]}")
    print(f"Answer: {ans}")
    print('---')

# Multi-Query Retriever   
모호한 쿼리를 검색하는 대신, 다양한 관점에서 Paraphrazing한 쿼리를 사용할 수 있습니다.   
이 때, LLM의 도움을 받을 수 있습니다.
# Multi Query를 확인하기 위한 로깅
import logging

logging.basicConfig()
logging.getLogger('langchain.retrievers.multi_query').setLevel(logging.INFO)
from langchain_core.prompts import PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever

rewrite_prompt = PromptTemplate(template = """
당신은 삼성SDS의 직원들을 대상으로 하는 챗봇입니다.
기업 공시 문서에 대한 질문이 주어집니다.
'우리 회사' 등의 표현은 회사명으로 변환하세요.     

공시 문서는 '삼성SDS', '삼성에스디에스', 'Samsung SDS' 등의 표현이 혼재된 문서이므로
모든 표현을 하나씩 사용하세요.             

해당 질문에 대한 정보를 검색하기 위해, 벡터 데이터베이스에 입력할 다양한 맥락의 질문을 생성하세요.
질문은 3개 생성하고, 한 줄에 질문 하나씩 출력하세요.
질문을 다각도로 분석하여, 다양한 검색 결과가 나오도록 구성해야 합니다.
---
원본 질문: {question}

""")

multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=db.as_retriever(),
    llm=llm,
    prompt = rewrite_prompt,
)
# 질문이 주어지면, 질문을 3개로 분할
# 3*K 검색 (12)
# 합집합으로 최종 Context 만들기
len(multi_query_retriever.invoke("2021-2022년 사이에 있었던 합병 사례를 모두 파악하여, 표로 나타내세요."))
rag_chain = (
    {"context": multi_query_retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
questions = [
    '이 회사의 사업은 크게 두 부문으로 나눠집니다. 각각은 어떤 부문입니까?',
    'IT서비스는 몇 개의 분야로 나눠집니까? 각각은 무엇입니까?',
    '2024년과 2025년에 걸쳐, 매출 비중은 어떻게 변화했나요?',
    '2024년 6월말 기준, 연결 재무제표의 유동자산은 얼마인가요?',
    'GPUaaS 서비스가 무엇입니까?',
    '회사의 연구개발 담당조직은 어떤 분야의 핵심 기술을 연구하고 있습니까?',
    '2021-2022년 사이에 있었던 합병 사례를 모두 파악하여, 표로 나타내세요.'
]

result = rag_chain.batch(questions)
for i, ans in enumerate(result):
    print(f"Question: {questions[i]}")
    print(f"Answer: {ans}")
    print('---')

### Ensemble Retriever

Ensemble Retriever는 서로 다른 리트리버를 결합하여 순위를 합산합니다.   
주로 Keyword Indexing 기반 검색인 BM25 검색과 Semantic 검색을 합친 Hybrid Search를 사용합니다.
from kiwipiepy import Kiwi

kiwi = Kiwi()
def kiwi_tokenize(text):
    return [token.form for token in kiwi.tokenize(text)]

kiwi_tokenize("2021-2022년 사이에 있었던 합병 사례를 모두 파악하여, 표로 나타내세요.")
from langchain.retrievers import BM25Retriever, EnsembleRetriever

bm25_retriever = BM25Retriever.from_documents(token_chunks, preprocess_func = kiwi_tokenize)
bm25_retriever.k = 5

retriever = db.as_retriever(search_kwargs={"k": 5})

ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, retriever], weights=[0.5, 0.5]
)
rag_chain = (
    {"context": ensemble_retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
result = rag_chain.batch(questions)
for i, ans in enumerate(result):
    print(f"Question: {questions[i]}")
    print(f"Answer: {ans}")
    print('---')

# Reranker
넓은 Retriever 검색 범위를 이용한 뒤, Reranker를 통해 개수를 줄일 수 있습니다.
# GPU가 있거나, RAM이 충분한 경우에만 아래 코드를 실행해 주세요.
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers import ContextualCompressionRetriever
import torch


# Top 20
rough_retriever = db.as_retriever(search_kwargs={"k": 20})

# 다국어 리랭커 모델 사용하기
model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")

compressor = CrossEncoderReranker(model=model, top_n=5)
# Cross Encoder : 질문과 Chunk를 같이 넣고 처리하는 Transformers 계열 모델
# 점수순으로 Top 5

compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=rough_retriever)
compressed_docs = compression_retriever.invoke("2021-2022년 사이에 있었던 합병 사례를 모두 파악하여, 표로 나타내세요.")

compressed_docs


# Contextual Retrieval    

Claude가 제안한 Contextual Retrieval은 전체 Context를 활용하여     
청크별 헤더를 추가하는 방법입니다.    
# 청크 초기화
token_chunks = token_splitter.split_documents(all_papers)
print(len(token_chunks))
context_prompt = ChatPromptTemplate(
    [
        ('user', '''
기업 공시 보고서의 전체 내용과, 그 중 일부 Chunk가 주어집니다.
주어진 Document의 일부인 Chunk에 대해
간결하고 관련성 있는 짧은 설명을 생성하세요.
청크만으로는 명확하지 않으나, 전체를 참고하여 파악할 수 있는 정보를 추가하여
청크의 내용이 더 명확해지도록 하는 2~4문장 길이의 Context를 생성하면 됩니다.
아래의 가이드라인을 참고하세요.

1. 텍스트 부분에서 논의된 주요 주제나 개념을 포함하세요.
2. 문서 전체의 문맥에서 관련 정보나 비교를 언급하세요.
3. 가능한 경우, 이 정보가 문서의 전체적인 주제나 목적과 어떻게 연관되는지를 설명하세요.
4. 중요한 정보를 제공하는 주요 항목과 수치를 포함하세요.
5. 답변은 한국어로 작성합니다.

답변은 간결하게 작성하세요.

# Input Format

- [Document]: `<document> {document} </document>`
- [Chunk]: `<chunk> {chunk} </chunk>`

Context:
        ''')
    ]
)

context_chain = context_prompt | llm | StrOutputParser()

Context가 잘 생성됐는지 확인해 봅니다.
chunk = token_chunks[40]
doc = all_papers[chunk.metadata['index']].page_content
context = context_chain.invoke({'document':doc, 'chunk':chunk.page_content})
print(context)
print('========')
print(chunk.page_content)
이제 Context 추가 작업을 수행합니다.
from tqdm import tqdm

chunk_with_parents=[]

for i, chunk in enumerate(tqdm(token_chunks)):
    doc = all_papers[chunk.metadata['index']].page_content
    chunk_with_parents.append({'document':doc, 'chunk':chunk.page_content})
    # print('\n'+context)
    # print('---')

# 10개씩 실행 (API 여유시 배치 늘리기)
for i in tqdm(range(0, len(token_chunks), 10)):
    contexts = context_chain.batch(chunk_with_parents[i:min(i+10, len(token_chunks))])
    for j in range(i,min(i+10, len(token_chunks))):
        token_chunks[j].page_content = context + '\n\n' + token_chunks[j].page_content
수정된 청크를 이용해, 벡터 데이터베이스를 다시 구성합니다.
db = Chroma(embedding_function=embeddings,
                           persist_directory="./chroma_Web_ContextualRetrieval",
                           collection_metadata={'hnsw:space':'l2'},
                           collection_name='Report',
                           )

db.add_documents(token_chunks)
Contextual Header를 이용하기 위해, BM25와 Semantic Search를 결합합니다.
bm25_retriever = BM25Retriever.from_documents(token_chunks, preprocess_func = kiwi_tokenize)
bm25_retriever.k = 5

retriever = db.as_retriever(search_kwargs={"k": 5})

ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, retriever], weights=[0.5, 0.5]
)
rag_chain = (
    {"context": ensemble_retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

result = rag_chain.batch(questions)
for i, ans in enumerate(result):
    print(f"Question: {questions[i]}")
    print(f"Answer: {ans}")
    print('---')
