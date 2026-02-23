# [실습] LangChain을 이용한 오픈 RAG 만들기

RAG는 Retrieval-Augmented Generation (RAG) 의 약자로, 질문이 주어지면 관련 있는 문서를 찾아 프롬프트에 추가하는 방식의 어플리케이션입니다.   
RAG의 과정은 아래와 같이 진행됩니다.
1. Indexing : 문서를 받아 검색이 잘 되도록 저장합니다.
1. Processing : 입력 쿼리를 전처리하여 검색에 적절한 형태로 변환합니다<br>
1. Search(Retrieval) : 질문이 주어진 상황에서 가장 필요한 참고자료를 검색합니다.
1. Augmenting : Retrieval의 결과와 입력 프롬프트를 이용해 LLM에 전달할 프롬프트를 생성합니다.
1. Generation : LLM이 출력을 생성합니다.
## 라이브러리 설치  

랭체인 관련 라이브러리와 벡터 데이터베이스 라이브러리를 설치합니다.  
<br>
  
`langchain_chroma`: ChromaDB를 이용해 벡터 데이터베이스를 구성합니다.    
# LangChain + LLM
!pip install langchain-huggingface langchain==0.3.27 langchain-openai langchain-community==0.3.27
# RAG
!pip install langchain_chroma
# 기타
!pip install dotenv jsonlines beautifulsoup4 matplotlib setuptools koreanize-matplotlib
# env 파일 불러오기   
import openai
from dotenv import load_dotenv

load_dotenv('.env', override=True)

client = openai.OpenAI()
# API 키 검증하기
try: client.models.list(); print("OPENAI_API_KEY가 정상적으로 설정되어 있습니다.")
except:  print(f"OpenAI API 키가 유효하지 않습니다!")

## LLM과 임베딩 모델 구성하기     
from langchain_openai import ChatOpenAI

gpt5 = ChatOpenAI(model='gpt-5-mini',
                 temperature=1.0)

gpt41 = ChatOpenAI(model='gpt-4.1-mini',
                 temperature=0)

sllm = ChatOpenAI(model='gpt-3.5-turbo',
                 temperature=0)

test = '안녕!!!!! 너는 누구니?'
print('gpt5:', gpt5.invoke(test).content)
print('gpt41:', gpt41.invoke(test).content)
print('sllm:', sllm.invoke(test).content)
## Embedding

RAG의 구조는 검색을 수행하기 위한 임베딩 모델이 필요합니다.   
임베딩 모델은 텍스트를 벡터로 변환하며,    
이후 결과를 벡터 DB에 저장해 검색할 수 있습니다.
OpenAI의 `text-embedding-3-large` 는 빠른 속도로 연산이 가능하나, 비용이 발생하며 온라인 모델입니다.   
이에 따라, 폐쇄망/온프레미스 환경에서는 공개 임베딩 모델을 사용하여 구현해야 합니다.
from langchain_openai import OpenAIEmbeddings
openai_embeddings = OpenAIEmbeddings(model='text-embedding-3-large', 
                                     chunk_size = 100)
# Embedding의 chunk_size : 동시 처리 요청 수 (청크 크기가 아님!!!!!)
오픈 임베딩 모델에서 중요한 파라미터는 다음과 같습니다.

- 파라미터 수 : 큰 임베딩 모델의 크기는 LLM에 육박합니다. GPU를 고려하여 선택합니다.
- Max Tokens: 임베딩 모델의 최대 토큰보다 큰 데이터를 입력하면, 앞부분만을 이용해 계산하게 되므로 적절한 검색이 되지 않을 수 있습니다.
- 임베딩 차원: 큰 차원의 벡터를 생성하는 임베딩 모델은 검색 속도가 감소합니다.
현재 한국어 데이터를 임베딩하기 위해 자주 사용하는 모델은 아래와 같습니다.

- BAAI/BGE-M3 (2GB, 8194 토큰 제한)   
BGE-M3 시리즈는 BAAI의 임베딩 모델로, 현재 가장 인기가 많은 모델입니다.

- nlpai-lab/KURE-v1 (2GB, 8194 토큰 제한)    
KURE 임베딩은 고려대학교 NLP 연구실에서 만든 모델로, BGE-M3를 한국어 텍스트로 파인 튜닝한 모델입니다.

- Qwen/Qwen-3-Embedding (0.6B, 4B, 8B, 32768 토큰 제한)    
알리바바 클라우드의 Qwen 모델을 개량하여 만든 모델입니다.    
가장 최신 모델로, BGE-M3와의 성능 비교가 치열합니다.
RAG를 하기 전, 비교를 위해 LLM에게 질문해 보겠습니다.
# Test
for s in gpt5.stream("도메인 특화 언어 모델이란 무엇입니까? 어떤 예시가 있나요?"):
    print(s.content, end='')

print('\n--------\n')
for s in gpt41.stream("도메인 특화 언어 모델이란 무엇입니까? 어떤 예시가 있나요?"):
    print(s.content, end='')

print('\n--------\n')
for s in sllm.stream("도메인 특화 언어 모델이란 무엇입니까? 어떤 예시가 있나요?"):
    print(s.content, end='')
## 데이터 준비하기    
이번 실습에서는 삼성SDS의 기술 블로그 HTML 데이터를 불러와 활용해 보겠습니다.

import requests
import os

sds_urls = [
    'https://www.samsungsds.com/kr/research-blog/agentic-ai-emergence-attention.html',
    'https://www.samsungsds.com/kr/research-blog/sky-computing-future-samsungsds.html',
    'https://www.samsungsds.com/kr/research-blog/samsung-sds-research-director-interview.html',
    'https://www.samsungsds.com/kr/research-blog/samsung-sds-ai-research-team-agentic-ai-interview.html',
    'https://www.samsungsds.com/kr/research-blog/agentic-ai-sds-keynote.html',
    'https://www.samsungsds.com/kr/research-blog/post-quantum-crypto-migration.html',
    'https://www.samsungsds.com/kr/research-blog/ai-preference-consistency.html',
    'https://www.samsungsds.com/kr/research-blog/pim-inference-acceleration.html',
    'https://www.samsungsds.com/kr/insights/agentic-ai-in-mobile-and-field-operations.html',
    'https://www.samsungsds.com/kr/insights/vertical-ai-agents-part1.html',
    'https://www.samsungsds.com/kr/insights/how-to-build-agentic-ai.html',
    'https://www.samsungsds.com/kr/insights/ai-risk-management-framework.html',
    'https://www.samsungsds.com/kr/insights/deepseek-to-change-the-ai-product-market.html',
    'https://www.samsungsds.com/kr/insights/business-potential-in-small-language-models.html',
    'https://www.samsungsds.com/kr/insights/ondevice-ai-and-cloud-ai.html',
]
## LangChain Document Loaders

LangChain의 `document_loaders`는 다양한 형식의 파일을 불러올 수 있습니다.   
[https://python.langchain.com/docs/integrations/document_loaders/ ]    

Web URL로부터 페이지를 로드하는 기본 파서인 `WebBaseLoader`를 사용합니다.   
# Jupyter 분산 처리를 위한 설정
import nest_asyncio

nest_asyncio.apply()
import bs4
from langchain_community.document_loaders import WebBaseLoader

async def get_news_documents(links):
    loader = WebBaseLoader(
        web_paths=links,
        requests_per_second = 10, # 1초에 10개 요청 보내기
        show_progress = True # 진행 상황 출력
    )
    # docs = loader.load() # 기본 코드
    # return docs

    docs = []

    async for doc in loader.alazy_load():
        # 순차적 로드 대신 비동기 처리
        docs.append(doc)
    return docs

docs = await get_news_documents(sds_urls)
for doc in docs:
    print(doc.page_content[1000:1050])
    print(len(doc.page_content))
import matplotlib.pyplot as plt
import koreanize_matplotlib

document_lengths = [len(document.page_content) for document in docs]

# 히스토그램 그리기
plt.figure(figsize=(10,6))
plt.hist(document_lengths, bins=20, color='skyblue', edgecolor='black')
plt.title('문서별 page_content 길이 분포', fontsize=15)
plt.xlabel('page_content 길이', fontsize=12)
plt.ylabel('문서 개수', fontsize=12)
plt.grid(axis='y', alpha=0.5)
plt.show()

docs[0]
불필요한 내용을 전처리합니다.  

'본문 바로 가기', '\n공유하기\n', '\n인쇄하기\n' '\nloading...\n' 등의 단어를 제거하고,    
중복된 줄바꿈 기호도 하나로 연결하겠습니다.

docs[-1].page_content
import re

def preprocess(docs):
    noise_texts = [
        '본문 바로 가기',
        '\n공유하기\n',
        '\n인쇄하기\n',
        '\nloading...\n',
        '\ufeff',
    ]

    def clean_text(doc):
        text = doc.page_content
        # 노이즈 텍스트 제거
        for noise in noise_texts:
            text = text.replace(noise, '')

        # 탭과 개행문자를 공백으로 변환
        text = text.replace('\t', ' ').replace('\n', ' ')

        # 연속된 공백을 하나로 치환
        text = re.sub(r'\s+', ' ', text).strip()

        doc.page_content = text

        return doc


    preprocessed_docs = []
    for doc in docs:

        # 텍스트 정제
        doc= clean_text(doc)
        preprocessed_docs.append(doc)

    return preprocessed_docs

preprocessed_docs = preprocess(docs)

preprocessed_docs[-1].page_content
불러온 텍스트 데이터는 파일로 저장할 수 있습니다.
# 불러온 document 저장하기
import jsonlines
def save_docs_to_jsonl(documents, file_path):
    with jsonlines.open(file_path, mode="w") as writer:
        for doc in documents:
            writer.write(doc.model_dump())
save_docs_to_jsonl(preprocessed_docs, "docs.jsonl")

# jsonl 파일 불러오기
from langchain.schema import Document

def load_docs_from_jsonl(file_path):
    documents = []
    with jsonlines.open(file_path, mode="r") as reader:
        for doc in reader:
            documents.append(Document(**doc))
    return documents

## Chunking: 청크 단위로 나누기   


전처리가 완료된 docs를 chunk 단위로 분리합니다.
`chunk_size`와 `chunk_overlap`을 이용해 청크의 구성 방식을 조절할 수 있습니다.  

Chunk Size * K(검색할 청크의 수) 의 결과가 Context의 길이가 됩니다.
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# 0~1000, 800~1800, 1600~2600, 2400~3400, ...
# 구분자 기준 청킹
chunks = text_splitter.split_documents(preprocessed_docs)
print(len(chunks))
chunk_lengths = [len(chunk.page_content) for chunk in chunks]

# 히스토그램 그리기
plt.figure(figsize=(10,6))
plt.hist(chunk_lengths, bins=20, color='skyblue', edgecolor='black')
plt.title('청크별 page_content 길이 분포', fontsize=15)
plt.xlabel('page_content 길이', fontsize=12)
plt.ylabel('청크 개수', fontsize=12)
plt.grid(axis='y', alpha=0.5)
plt.show()

## Vector Database 만들기   

구성된 청크를 벡터 데이터베이스에 로드합니다.   
from langchain_chroma import Chroma
import uuid

Chroma().delete_collection() # (메모리에 저장하는 경우) 기존 데이터 삭제

uuidstr = str(uuid.uuid4())[0:6]
# 랜덤 문자열

# DB 구성하기
db = Chroma(embedding_function=openai_embeddings,
            persist_directory=f"./chroma_OpenAI_{uuidstr}",
            # 파일 시스템에 저장 (생략시 메모리에 저장)

            collection_name='Web', # 식별 이름

            collection_metadata={'hnsw:space':'l2'},
            # l2 메트릭 설정(기본값, cosine, mmr 로 변경 가능)
            )
DB에 document를 추가합니다.    
OpenAI 임베딩은 30만 토큰 동시 처리 제한이 있어, 나눠서 전달합니다.
from tqdm import tqdm

# 임베딩의 chunk_size 를 정해서 전달
# 오픈 모델의 경우, 직접 배치로 처리

db.add_documents(chunks)
db로부터 retriever를 구성합니다.
# Top 5 Search(기본값은 4)
retriever = db.as_retriever(search_kwargs={'k':10})
result = retriever.invoke("도메인 특화 언어 모델")
for doc in result:
    print(doc.page_content[0:300])
    print('-------')

## Prompting
RAG를 위한 간단한 프롬프트를 작성합니다.

from langchain.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate([
    ("system", '''당신은 QA(Question-Answering)을 수행하는 Assistant입니다.
다음의 Context를 이용하여 Question에 답변하세요.
정확한 답변을 제공하세요.
만약 모든 Context를 다 확인해도 정보가 없다면,
"정보가 부족하여 답변할 수 없습니다."를 출력하세요.'''),
('human', '''Context: {context}
---
Question: {question}''')])
prompt.pretty_print()
## Chain
RAG를 수행하기 위한 Chain을 만듭니다.
RAG Chain은 프롬프트에 context와 question을 전달해야 합니다.    
체인의 입력은 Question만 들어가므로, Context를 동시에 prompt에 넣기 위해서는 아래의 구성이 필요합니다.
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# retriever의 결과물은 List[Document] 이므로 이를 ---로 구분하는 함수
# metadata의 source를 보존하여 추가
def format_docs(docs):
    return " \n---\n".join(
        [f'Title: {doc.metadata['title']} \nURL: {doc.metadata["source"]} \nContent: {doc.page_content}\n' for doc in docs]
    )
    # result=''
    # for doc in docs:
    #     context = f'Content: {doc.page_content}\nURL:{doc.metadata["source"]}' + ' \n---\n '
    #     result += context
    # return result



    # join : 구분자를 기준으로 스트링 리스트를 하나의 스트링으로 연결

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    # retriever : question을 받아서 context 검색: document 반환
    # format_docs : document 형태를 받아서 텍스트로 변환
    # RunnablePassthrough(): 체인의 입력을 그대로 저장
    | prompt
    | gpt5
    | StrOutputParser()
)
print(format_docs(retriever.invoke("sLLM이 뭐야?")))
rag_chain.invoke("도메인 특화 언어 모델이란 무엇입니까? 어떤 예시가 있나요?")
rag_chain.invoke("인공지능의 최근 발전 방식은? 관련 링크도 보여주세요")
rag_chain.invoke("멀티모달 모델은 어떻게 발전할까요?")
rag_chain.invoke("딥시크가 뭐냐")
def rag(llm, questions):
    rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    # retriever : question을 받아서 context 검색: document 반환
    # format_docs : document 형태를 받아서 텍스트로 변환
    # RunnablePassthrough(): 체인의 입력을 그대로 저장
    | prompt
    | llm
    | StrOutputParser()
    )
    return rag_chain.batch(questions)

questions = [
    "도메인 특화 언어 모델이란 무엇입니까? 어떤 예시가 있나요?",
    "인공지능의 최근 발전 방식은? 관련 링크도 보여주세요",
    "멀티모달 모델은 어떻게 발전할까요?",
    "딥시크가 뭐냐"
]

print('\n----\n'.join(rag(llm=gpt5, questions= questions)))
print('--------')
print('\n----\n'.join(rag(llm=gpt41, questions= questions)))
print('--------')
print('\n----\n'.join(rag(llm=sllm, questions= questions)))
Context가 포함된 RAG 결과를 보고 싶다면, RunnableParallel을 사용하면 됩니다.
assign()을 이용하면, 체인의 결과를 받아 새로운 체인에 전달하고, 그 결과를 가져옵니다.
from langchain_core.runnables import RunnableParallel

rag_chain_from_docs =( 
    prompt
    | gpt5
    | StrOutputParser()
)

rag_chain_with_ref = RunnableParallel({"context": retriever, "question": RunnablePassthrough()}
                             ).assign(result = rag_chain_from_docs)   
# parallel 내부에 Dict를 넣어도 됨
rag_chain_with_ref.invoke("딥시크가 뭐야?")

이번 실습에서 만든 RAG 파이프라인은 가장 단순하게 시맨틱 검색과 Top 5 랭킹을 활용하는 구조입니다.   
다음 과정에서, RAG 파이프라인의 성능을 향상시키는 방법에 대해 알아보겠습니다!
