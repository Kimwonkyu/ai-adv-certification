# [실습] LangChain으로 첫 체인 만들기

## 라이브러리 설치  

랭체인 관련 라이브러리를 설치합니다.  

# LangChain + LLM
!pip install langchain-huggingface langchain langchain-openai langchain-community langchain-ollama langchain-google-generativeai
!pip install dotenv
## 환경 변수 설정하기

import os
from dotenv import load_dotenv
# OPENAI_API_KEY, GOOGLE_API_KEY
load_dotenv(override=True)
모델 Provide의 클래스를 불러온 뒤, LLM을 연결하면 됩니다!

## Ollama를 통해 LLM 불러오기


https://ollama.com/


올라마는 가장 간단하게 오픈 모델을 구동할 수 있는 방식입니다.   
추론 최적화/병렬 처리 등의 기능은 뛰어나지 않지만  
CPU에서도 구동이 가능하며, UI도 지원합니다.
### Ollama 설치하고 서빙하기    
터미널을 이용해 아래의 커맨드를 입력합니다.

- 설치

```curl -fsSL https://ollama.com/install.sh | sh```

- Context Window 늘리기 (기본 Context 4096)

```export OLLAMA_CONTEXT_LENGTH=16384```

- 서빙 (& 으로 백그라운드 실행)

``` ollama serve &```


올라마는 기본적으로 11434 포트에서 실행됩니다.   
Pull 을 통해 허깅페이스 모델의 주소나 Ollama 주소를 입력하면 모델을 불러와 저장합니다.
불러올 수 있는 모델의 목록은 https://ollama.com/search 에서 확인할 수 있습니다.   

기본적으로 양자화를 거친 상태로 불러오며, 16비트 Full Precision을 비롯한 다양한 버전도 확인할 수 있습니다.
터미널에서 아래 커맨드를 입력해 GPT-OSS 20B 모델을 불러옵니다.    
```
ollama pull gpt-oss:20b
```
불러온 모델은 파일 시스템에 저장되며,   
Ollama 모듈을 호출할 때 해당 모델이 GPU에 로드되는 구조입니다.
from langchain_ollama import ChatOllama
llm = ChatOllama(model='gpt-oss:20b',
                 temperature=0.1, reasoning=True)

test = '안녕!!!!!'
llm.invoke(test)
test = 'ㅠㅠ'
for s in llm.stream(test):
    print(s.content, end='')
## Embedding

RAG의 구조는 검색을 수행하기 위한 임베딩 모델이 필요합니다.   
임베딩 모델은 텍스트를 벡터로 변환하며,    
이후 결과를 벡터 DB에 저장해 검색할 수 있습니다.
OpenAI의 `text-embedding-3-large` 는 빠른 속도로 연산이 가능하나, 비용이 발생하며 온라인 모델입니다.   
이에 따라, 폐쇄망/온프레미스 환경에서는 공개 임베딩 모델을 사용하여 구현해야 합니다.
from langchain_openai import OpenAIEmbeddings
openai_embeddings = OpenAIEmbeddings(model='text-embedding-3-large')
허깅페이스에 게시된 공개 모델을 불러옵니다.   
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
from sentence_transformers import SentenceTransformer
import torch

# HuggingFace 임베딩 주소 지정하기
# intfloat/multilingual-e5-small , baai/bge-m3, 등의 주소를 입력하여 지정
# GPU에 여유가 있다면 Qwen3 Embedding의 큰 사이즈 (4B, 8B)?

model_name = 'Qwen/Qwen3-Embedding-0.6B'
#실제 주소: https://huggingface.co/BAAI/bge-m3

# CPU 설정으로 모델 불러오기
emb_model = SentenceTransformer(model_name, device='cpu',model_kwargs={'torch_dtype':torch.bfloat16})

# 로컬 폴더에 모델 저장하기
emb_model.save('./embedding')

# 모델 메모리에서 삭제
del emb_model
import gc
gc.collect()
파일 시스템에 저장한 오픈 모델은 HuggingFaceEmbeddings로 불러옵니다.
from langchain_huggingface import HuggingFaceEmbeddings
import torch

# 허깅페이스 포맷의 임베딩 모델 불러오기
open_embeddings = HuggingFaceEmbeddings(model_name= './embedding',
                                   model_kwargs={'device':'cuda', 'model_kwargs':{'torch_dtype':torch.bfloat16}})
                                                    # gpu 사용하기                               bfloat16

# GPU 로드된 것 확인
print('임베딩 모델 로드 완료!')
RAG를 하기 전, 비교를 위해 LLM에게 질문해 보겠습니다.
# Test
thinking = False
for s in llm.stream("도메인 특화 언어 모델이란 무엇입니까? 어떤 예시가 있나요?"):
    if s.additional_kwargs.get('reasoning_content'):
        if not thinking:
            print('<THINK> ', end='')
            thinking=True
        print(s.additional_kwargs['reasoning_content'], end='')
    else:
        if thinking:
            print('</THINK>\n\n', end='')
            thinking=False
        print(s.content,end='')
## 데이터 준비하기    
네이버 API를 통해, 검색어에 대한 뉴스 기사 링크를 가져오겠습니다.    

import requests
import os

def get_naver_news_links(query, num_links=100):
    """
    query와 num_links를 입력받아 네이버 검색 수행, 네이버 뉴스 URL의 기사만 수집
    """

    url = f"https://openapi.naver.com/v1/search/news.json?query={query}&display={num_links}&sort=sim"
    # 최대 100개의 결과를 표시

    client_id = os.getenv('NAVER_CLIENT_ID')
    client_secret = os.getenv('NAVER_CLIENT_SECRET')

    headers = {
        'X-Naver-Client-Id': client_id,
        'X-Naver-Client-Secret': client_secret
    }

    response = requests.get(url, headers=headers)
    result = response.json()

    # 특정 링크 형식만 필터링
    filtered_links = []
    for item in result['items']:
        link = item['link']
        if "n.news.naver.com/mnews/article/" in link:
            # 네이버 뉴스 스타일만 모으기
            filtered_links.append(link)

    # 결과 출력
    print(query, ':', len(filtered_links), 'Example:', filtered_links[0])
    # for link in filtered_links:
    #     print(link)

    return filtered_links

filtered_links = []
for topic in ['도메인 특화 언어모델', 'OpenAI', 'GPT', '구글', '가전제품', '넷플릭스']:
    filtered_links += get_naver_news_links(topic, 100)
print('Total Articles:', len(filtered_links))
print('Total Articles(Without Duplicate):',len(list(set(filtered_links))))
filtered_links = list(set(filtered_links))
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
        bs_kwargs={'parse_only':bs4.SoupStrainer(class_=("newsct", "newsct-body"))},
                                # newsct, newsct-body만 추출 : 네이버 뉴스 포맷 HTML 요소

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

docs = await get_news_documents(filtered_links)
from rich import print as rprint
rprint(docs[12])
불필요한 내용을 전처리합니다.
import re

def preprocess(docs):
    noise_texts = [
        '''구독중 구독자 0 응원수 0 더보기''',
        '''쏠쏠정보 0 흥미진진 0 공감백배 0 분석탁월 0 후속강추 0''',
        '''댓글 본문 요약봇 본문 요약봇''',
        '''도움말 자동 추출 기술로 요약된 내용입니다. 요약 기술의 특성상 본문의 주요 내용이 제외될 수 있어, 전체 맥락을 이해하기 위해서는 기사 본문 전체보기를 권장합니다. 닫기''',
        '''텍스트 음성 변환 서비스 사용하기 성별 남성 여성 말하기 속도 느림 보통 빠름''',
        '''이동 통신망을 이용하여 음성을 재생하면 별도의 데이터 통화료가 부과될 수 있습니다. 본문듣기 시작''',
        '''닫기 글자 크기 변경하기 가1단계 작게 가2단계 보통 가3단계 크게 가4단계 아주크게 가5단계 최대크게 SNS 보내기 인쇄하기''',
        'PICK 안내 언론사가 주요기사로선정한 기사입니다. 언론사별 바로가기 닫기',
        '응원 닫기',
        '구독 구독중 구독자 0 응원수 0 ',

    ]

    def clean_text(doc):
        text = doc.page_content
        # 탭과 개행문자를 공백으로 변환
        text = text.replace('\t', ' ').replace('\n', ' ')

        # 연속된 공백을 하나로 치환
        text = re.sub(r'\s+', ' ', text).strip()

        # 여러 구분자를 한번에 처리
        split_markers = [
            '구독 해지되었습니다.',
            '구독 메인에서 바로 보는 언론사 편집 뉴스 지금 바로 구독해보세요!'
        ]


        for marker in split_markers:
            parts = text.split(marker)
            if len(parts) > 1:
                if marker == '구독 해지되었습니다.':
                    text = parts[1]  # 뒷부분 사용
                else:
                    text = parts[0]  # 앞부분 사용


        # 노이즈 텍스트 제거
        for noise in noise_texts:
            text = text.replace(noise, '')

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

rprint(preprocessed_docs[12])
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

평가 데이터와의 연결을 위해, 전처리한 데이터 대신 실습 시트의 데이터를 사용하겠습니다.
preprocessed_docs = load_docs_from_jsonl("eval.jsonl")
preprocessed_docs[2]
## Chunking: 청크 단위로 나누기   


전처리가 완료된 docs를 chunk 단위로 분리합니다.
`chunk_size`와 `chunk_overlap`을 이용해 청크의 구성 방식을 조절할 수 있습니다.  

Chunk Size * K(검색할 청크의 수) 의 결과가 Context의 길이가 됩니다.
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain import hub

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# 0~1000, 800~1800, 1600~2600, ...
# 엔터, 공백, 쉼표 등의 구분자(delimiter)를 기준으로 최대한 청킹
chunks = text_splitter.split_documents(preprocessed_docs)
print(len(chunks))

# Top 5 검색이라면? --> 5 * 1000 = 5000 글자 Context
## Vector Database 만들기   

구성된 청크를 벡터 데이터베이스에 로드합니다.   
from langchain_chroma import Chroma
import uuid

Chroma().delete_collection() # (메모리에 저장하는 경우) 기존 데이터 삭제

uuidstr = str(uuid.uuid4())[0:6]
# 랜덤 식별자

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

print(len(chunks))
# 300,000 토큰 제한

# 100개씩 추가
for i in tqdm(range(0, len(chunks), 100)):
    db.add_documents(chunks[i:min(i+100, len(chunks))])
db로부터 retriever를 구성합니다.
# Top 5 Search(기본값은 4)
retriever = db.as_retriever(search_kwargs={'k':5})
retriever.invoke("도메인 특화 언어 모델")
## Prompting
RAG를 위한 간단한 프롬프트를 작성합니다.
from langchain.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate([
    ("system", '''당신은 QA(Question-Answering)을 수행하는 Assistant입니다.
다음의 Context를 이용하여 Question에 답변하세요.
정확한 답변을 제공하세요.
만약 모든 Context를 다 확인해도 정보가 없다면,
"정보가 부족하여 답변할 수 없습니다."를 출력하세요.
Question과 무관한 내용이 들어 있을 수 있으므로 신중히 판단하세요.'''),
('human','''
---
Context: {context}
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
    return " \n\n---\n\n ".join(['URL: '+ doc.metadata['source'] + '\nContent:' + doc.page_content for doc in docs])
    # join : 구분자를 기준으로 스트링 리스트를 하나의 스트링으로 연결


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    # retriever : question을 받아서 context 검색: document 반환
    # format_docs : document 형태를 받아서 텍스트로 변환
    # RunnablePassthrough(): 체인의 입력을 그대로 저장
    | prompt
    | llm
    | StrOutputParser()
)
print(format_docs(retriever.invoke("알리바바의 언어 모델 이름은?")))
rag_chain.invoke("도메인 특화 언어 모델이란 무엇입니까? 어떤 예시가 있나요?")
rag_chain.invoke("인공지능의 최근 발전 방식은? 관련 링크도 보여주세요")
rag_chain.invoke("알리바바가 개발한 언어 모델 이름은?")
만약 Context가 포함된 RAG 결과를 보고 싶다면, RunnableParallel을 사용하면 됩니다.
assign()을 이용하면, 체인의 결과를 받아 새로운 체인에 전달하고, 그 결과를 가져옵니다.
# assign : 결과를 받아서 새로운 인수 추가하고 원래 결과와 함께 전달

from langchain_core.runnables import RunnableParallel

rag_chain_from_docs = (
    prompt
    | llm
    | StrOutputParser()
)

rag_chain_with_source = RunnableParallel(
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
).assign(answer=rag_chain_from_docs)

rag_chain_with_source.invoke("인공지능의 최근 발전 방식은? 관련 링크도 보여주세요")

# retriever가 1번 실행됨
# retriever의 실행 결과를 rag_chain_from_docs 에 넘겨주기 때문에

# Runnable Quiz

runnable = RunnableParallel(
    passed=RunnablePassthrough(),
    extra=RunnablePassthrough.assign(mult=lambda x: x["num"] * 3),
    modified=lambda x: x["num"] + 1,
)

runnable.invoke({"num": 1})
# 결과 이해해보기!
이번에는 오픈 모델을 사용합니다.   
오픈 임베딩을 통해 구성한 DB와 원래 DB를 비교해 보겠습니다.
uuidstr = str(uuid.uuid4())[0:6]

open_db = Chroma(embedding_function=open_embeddings, # qwen 3 emb 0.6b
                           persist_directory=f"./chroma_open_{uuidstr}", # 별도 폴더에 저장
                           collection_name='Web', # 식별 이름
                           collection_metadata={'hnsw:space':'l2'}
                           )

# 10개씩 추가
for i in tqdm(range(0, len(chunks), 10)):
    open_db.add_documents(chunks[i:min(i+10, len(chunks))])
open_retriever = open_db.as_retriever(search_kwargs={'k':5})
open_retriever.invoke("알리바바의 언어 모델은 무엇입니까?")
rag_chain_open = (
    {"context": open_retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
rag_chain_from_docs = (
    prompt
    | llm
    | StrOutputParser()
)

rag_chain_with_source_open = RunnableParallel(
    {"context": open_retriever | format_docs, "question": RunnablePassthrough()}
).assign(answer=rag_chain_from_docs)
questions = ["도메인 특화 언어 모델이란 무엇입니까? 어떤 예시가 있나요?",
             "인공지능의 최근 발전 방식은? 관련 링크도 보여주세요",
             "알리바바의 언어 모델 이름은?"]
# Retriever 비교

result_open = open_retriever.batch(questions)

for i in range(3):
    print(f'Question: {questions[i]}')
    for j in result_open[i]:
        print('   ',j.page_content[0:50])

result_openai = retriever.batch(questions)
result_openai
for i in range(3):
    print(f'Question: {questions[i]}')
    for j in result_openai[i]:
        print('   ',j.page_content[0:50])

# 최종 결과 비교
print(rag_chain_open.batch(questions))
print(rag_chain.batch(questions))
# RAG 성능 평가하기   

구성한 RAG의 성능은 어떻게 평가할까요?   

RAGAS (https://docs.ragas.io/en/stable/)는 다양한 메트릭을 통한 RAG의 성능 평가를 지원합니다.

RAG의 평가를 위해서는 정답이 있는 Q/A 데이터가 필요합니다.   
실습 시트에서 eval.jsonl을 다운로드하여 불러옵니다.
import pandas as pd
df = pd.read_csv('./eval.csv')
eval_dataset = df.to_dict('list')
eval_dataset
questions, ground_truths = eval_dataset['questions'], eval_dataset['ground_truths']
for i in range(len(questions)):
    print(f'#{i}')
    print(f'Question: {questions[i]}\n')
    print(f'Ground Truth: {ground_truths[i]}\n')
    print('-----------')
구성된 RAG 체인을 이용해, RAGAS의 Evaluate에 필요한 데이터를 구성합니다.
dataset = []

for query,reference in zip(questions,ground_truths):

    relevant_docs = [doc.page_content for doc in retriever.invoke(query)]
    response = rag_chain.invoke(query)
    dataset.append(
        {
            "user_input":query,
            "retrieved_contexts":relevant_docs,
            "response":response,
            "reference":reference
        }
        # 질문, 검색결과, 답변, 정답(레퍼런스)
    )
from ragas import EvaluationDataset
evaluation_dataset = EvaluationDataset.from_list(dataset)
evaluation_dataset
RAGAS는 LLM을 이용해 정답과 답변을 개별 Claim(주장)으로 분할합니다.

이후, `LLMContextRecall`, `Faitufulness`, `FactualCorrectness` 등의 다양한 메트릭을 통해 RAG 파이프라인의 성능을 평가합니다.   
LLM 기반의 방법이므로 평가 LLM의 선정이 중요하며, 절대 수치보다는 상대적 비교가 효과적입니다.


- Context Recall: 정답의 Claim들이 모두 검색됐는가
- Faithfulness : 답변의 Claim이 얼마나 검색 결과에 근거했는가
- Factual Correctness : 정답과 답변의 Claim이 얼마나 일치하는가
- Bleu Score : 정답과 답변 키워드가 얼마나 일치하는가
- Semantic Sim : 정답 답변 임베딩이 얼마나 가까운가   

from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness
from ragas.metrics import BleuScore, SemanticSimilarity
from langchain_openai import ChatOpenAI

# https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/

# 평가자 LLM
evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4.1"))

# 평가자 Embedding
evaluator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model = 'text-embedding-3-large'))
semantic_scorer = SemanticSimilarity(embeddings = evaluator_embeddings)


result = evaluate(dataset=evaluation_dataset,
                  metrics=[BleuScore(), LLMContextRecall(), semantic_scorer, Faithfulness(), FactualCorrectness()],
                  llm=evaluator_llm,
                  embeddings = evaluator_embeddings
                  )
result
result.scores
result.to_pandas()
# dict로 변경하기(복잡..)
import ast

result_str = str(result)
result_dict = ast.literal_eval(result_str)
result_dict
# [실습] 오픈 임베딩 체인의 성능 평가하기   

다음의 코드를 이용해, 오픈 임베딩을 비롯한 다양한 파이프라인의 모델 성능도 평가해 보세요.   

새로운 Ollama 모델을 쓰고 싶은 경우, `ollama pull`을 통해 저장하고   
`ollama stop`을 통해 기존 모델을 해제한 뒤 실행해 주세요.
MODEL_NAME='gemma3:12b'
# 'gpt-'를 통해 gpt-oss 이외의 OpenAI 모델 불러올 수 있음

TEMPERATURE=0
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 10
embedding_method = 'QWEN' # or 'QWEN'


# CSV 파일 경로 설정
csv_path = "experiments/experiment_log.csv"
os.makedirs("experiments", exist_ok=True)


def RAG_pipeline():

    emb_map = {'QWEN':open_embeddings, 'OPENAI':openai_embeddings}

    dataset = []

    if 'gpt' in MODEL_NAME:
        llm = ChatOpenAI(model=MODEL_NAME, temperature = TEMPERATURE)
    else:
        try: # Reasoning Model에 한해 Path 분리
            llm = ChatOllama(model=MODEL_NAME, temperature = TEMPERATURE, reasoning = True)
            llm.invoke("hello!")
        except:
            llm = ChatOllama(model=MODEL_NAME, temperature = TEMPERATURE)

    print('## LLM 세팅 완료')
    Chroma().delete_collection() # 메모리 기존 데이터 삭제

    preprocessed_docs = load_docs_from_jsonl("eval.jsonl")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    print('## DB 구성 중...')

    chunks = text_splitter.split_documents(preprocessed_docs)
    print(f'## Chunks 생성 완료... 총 {len(chunks)} 개 청크')
    db = Chroma(embedding_function=emb_map[embedding_method],
                collection_metadata={'hnsw:space':'l2'},
                )

    for i in tqdm(range(0, len(chunks), 10)):
        db.add_documents(chunks[i:min(i+10, len(chunks))])

    print('## DB 구성 완료...')

    retriever = db.as_retriever(search_kwargs={'k':TOP_K})

    rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser())

    for query,reference in tqdm(zip(questions,ground_truths)):

        relevant_docs = [doc.page_content for doc in retriever.invoke(query)]
        response = rag_chain.invoke(query)
        dataset.append(
            {
                "user_input":query,
                "retrieved_contexts":relevant_docs,
                "response":response,
                "reference":reference
            }
        )

    print('## 답변 생성 완료...')
    evaluation_dataset = EvaluationDataset.from_list(dataset)

    result = evaluate(dataset=evaluation_dataset,
                    metrics=[BleuScore(), LLMContextRecall(), semantic_scorer, Faithfulness(), FactualCorrectness()],
                    llm=evaluator_llm,
                    embeddings = evaluator_embeddings
                    )

    result_str = str(result)
    result_dict = ast.literal_eval(result_str)
    return result_dict

실험 결과를 저장할 수 있습니다.
import csv
import datetime
import os

result = RAG_pipeline()

timestamp = datetime.datetime.now().isoformat()
row = {
    "timestamp": timestamp,
    "model_name": MODEL_NAME,
    "embedding_method": embedding_method,
    "temperature": TEMPERATURE,
    "chunk_size": CHUNK_SIZE,
    "chunk_overlap": CHUNK_OVERLAP,
    "top_k": TOP_K,
    **result
}

# 파일이 없으면 헤더 추가
write_header = not os.path.exists(csv_path)

with open(csv_path, "a", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=row.keys())
    if write_header:
        writer.writeheader()
    writer.writerow(row)

print(f"[{timestamp}] {MODEL_NAME} {embedding_method}| temp={TEMPERATURE}, chunk={CHUNK_SIZE}/{CHUNK_OVERLAP}, top_k={TOP_K} | results={result}")


