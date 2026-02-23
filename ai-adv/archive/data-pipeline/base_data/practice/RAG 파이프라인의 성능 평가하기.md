# [실습] RAG 파이프라인 성능 평가하기   

지금까지 RAG의 성능을 높이기 위한 다양한 방법에 대해 알아봤는데요.   
실제 RAG의 성능은 어떻게 측정해야 할까요?
이번 실습에서는 정답이 존재하는 RAG 데이터를 이용해, 성능을 평가하는 과정에 대해 알아보겠습니다.
### 라이브러리 설치  

랭체인 관련 라이브러리와 벡터 데이터베이스 라이브러리를 설치합니다.   
!pip install sacrebleu ragas dotenv langchain_huggingface jsonlines langchain==0.3.27 langchain-openai langchain-community==0.3.27 beautifulsoup4 langchain_chroma
import os
from dotenv import load_dotenv
load_dotenv('.env', override=True)

if os.environ.get('OPENAI_API_KEY'):
    print('OpenAI API 키 확인')
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4.1-nano", temperature = 0, max_tokens = 4096)
# 4o : Context 128k - 200k
# 4.1 : Context 1M
# 4o-mini > 4.1 Nano

RAG의 평가를 위해서는 정답이 있는 Q/A 데이터가 필요합니다.   
실습 시트에서 eval.jsonl을 다운로드하여 불러옵니다.
import pandas as pd

evaluation_dataset = pd.read_csv('./eval_dataset.csv', encoding='cp949')
evaluation_dataset
기존 데이터를 불러옵니다.
from glob import glob
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document

pdf_files = glob("reports/*.pdf")
pdf_files

# 각 PDF 파일에서 페이지별로 내용을 불러와 하나로 합침
all_papers=[]

for i, path_paper in enumerate(pdf_files):
    loader = PyMuPDFLoader(path_paper)
    pages = loader.load()
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

Chunking을 수행합니다.
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    model_name='gpt-4o-mini',
    chunk_size=1000, chunk_overlap=200)

chunks = text_splitter.split_documents(all_papers)
print(len(chunks))
Embedding 모델을 구성합니다.
from langchain_openai import OpenAIEmbeddings
openai_embeddings = OpenAIEmbeddings(model='text-embedding-3-large', chunk_size=100)
ChromaDB를 구성합니다.
from langchain_chroma import Chroma

Chroma().delete_collection() # (메모리에 저장하는 경우) 기존 데이터 삭제

# DB 구성하기
db = Chroma(embedding_function=openai_embeddings,
            persist_directory="./evaluation",
            collection_metadata={'hnsw:space':'l2'},
            )
DB에 document를 추가합니다.
db.add_documents(chunks)
db로부터 retriever를 구성합니다.
retriever = db.as_retriever(search_kwargs={'k':5})
# Chunk Size * K = Context 글자
# 1000      *  5 = 5000 토큰 
RAG를 위한 간단한 프롬프트를 작성합니다.
from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate([
    ("system", '''당신은 QA(Question-Answering)을 수행하는 Assistant입니다.
다음의 Context를 이용하여 Question에 답변하세요.
정확한 답변을 제공하세요.
만약 모든 Context를 다 확인해도 정보가 없다면,
"정보가 부족하여 답변할 수 없습니다."를 출력하세요.'''),
("human",'''
Context: {context}
---
Question: {question}''')])

prompt.pretty_print()
RAG를 수행하기 위한 Chain을 만듭니다.
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_community.output_parsers import StrOutputParser

def format_docs(docs):
    return " \n---\n ".join([doc.page_content+ '\n' for doc in docs])

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
rag_chain_from_docs = (
    prompt
    | llm
    | StrOutputParser()
)

rag_chain_with_ref = RunnableParallel(
    context = retriever | format_docs, question = RunnablePassthrough()
).assign(answer=rag_chain_from_docs)
# context, question, answer
# RAGAS 사용하기


RAGAS 는 다양한 메트릭을 통한 RAG의 성능 평가를 지원합니다.


각각의 메트릭은 아래의 링크에서 확인할 수 있습니다.

https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/
시트에서 평가 데이터셋을 불러옵니다.   
Question/Ground Truth의 구성입니다.
# Best: Retrieval의 Ground Truth, Generation의 Ground Truth가 모두 존재

import pandas as pd
df = pd.read_csv('./eval_dataset.csv', encoding='cp949')
eval_dataset = df.to_dict('list')
eval_dataset
구성된 RAG 체인을 이용해, RAGAS의 Evaluate 형태로 변환합니다.
questions, ground_truths = eval_dataset['questions'], eval_dataset['ground_truths']
for i in range(len(questions)):
    print(f'#{i}')
    print(f'Question: {questions[i]}\n')
    print(f'Ground Truth: {ground_truths[i]}\n')
    print('-----------')
RAG 체인을 실행해, 테스트 문제에 대한 답변을 생성합니다.
rag_chain_from_docs = (
    prompt
    | llm
    | StrOutputParser()
)

rag_chain_with_ref = RunnableParallel(
    raw_context = retriever, question = RunnablePassthrough()
).assign(context = lambda x: format_docs(x['raw_context'])).assign(answer=rag_chain_from_docs)

rag_chain_with_ref.invoke("이 보고서는 어느 회사 꺼야?")
# raw_context, question, context, answer
from tqdm import tqdm

results = rag_chain_with_ref.batch(questions)

dataset=[]

for result, reference in tqdm(zip(results,ground_truths)):
    dataset.append(
        {
            "user_input":result['question'],
            "retrieved_contexts":[doc.page_content for doc in result['raw_context']],
            "response":result['answer'],
            "reference":reference
        }
    )    
# # [질문, 검색결과, 답변, 정답(레퍼런스)]
# for query,reference in tqdm(zip(questions,ground_truths)):

#     relevant_docs = [doc.page_content for doc in retriever.invoke(query)]
#     response = rag_chain.invoke(query)
#     dataset.append(
#         {
#             "user_input":query,
#             "retrieved_contexts":relevant_docs,
#             "response":response,
#             "reference":reference
#         }
#     )

from ragas import EvaluationDataset
evaluation_dataset = EvaluationDataset.from_list(dataset)
evaluation_dataset
RAGAS는 LLM을 이용해 정답과 답변을 개별 Claim(주장)으로 분할합니다.

이후, `LLMContextRecall`, `Faithfulness`, `FactualCorrectness` 등의 다양한 메트릭을 통해 RAG 파이프라인의 성능을 평가합니다.   
LLM 기반의 방법이므로 평가 LLM의 선정이 중요하며, 절대 수치보다는 상대적 비교가 효과적입니다.


- Context Recall: 정답의 Claim들이 모두 검색됐는가
- Faithfulness : 답변의 Claim이 얼마나 검색 결과에 근거했는가
- Factual Correctness : 정답과 답변의 Claim이 얼마나 일치하는가
- Bleu Score : 정답과 답변 키워드가 얼마나 일치하는가
- Semantic Sim : 정답 답변 임베딩이 얼마나 가까운가   

1) Context Recall: 검색 결과 - 정답

낮은 값 --> Retrieval 실패

2) Faithfulness: 검색 결과 - 답변

낮은 값 --> LLM의 성능 문제

3) Factual Correctness: 정답 - 답변

F1 Score 방식

from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness
from ragas.metrics import BleuScore, SemanticSimilarity

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
# [실습] 다양한 파이프라인 파라미터 수정하기

현재 파이프라인에는 매우 다양한 조절 가능한 파라미터가 있습니다.   

각각의 파라미터를 수정하여, 성능 변화를 확인해 보세요.

Ex)
- Chunk의 크기/개수 늘리기   
- 모델 더 저렴한 모델로 바꾸기   
-
MODEL_NAME='gpt-5'
TEMPERATURE=0
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 5

# CSV 파일 경로 설정
csv_path = "experiments/experiment_log.csv"
os.makedirs("experiments", exist_ok=True)


def RAG_pipeline():

    dataset = []

    llm = ChatOpenAI(model=MODEL_NAME, temperature = TEMPERATURE, max_tokens = 4096)
    Chroma().delete_collection() # 메모리 기존 데이터 삭제
    

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    print('## DB 구성 중...')
    

    chunks = text_splitter.split_documents(all_papers)
    print(f'## Chunks 생성 완료... 총 {len(chunks)} 개 청크')




    db = Chroma(embedding_function=openai_embeddings,
                collection_metadata={'hnsw:space':'l2'},
                )
    for i in tqdm(range(0, len(chunks), 50)):
        db.add_documents(chunks[i:min(i+50, len(chunks))])

    print('## DB 구성 완료...')


    retriever = db.as_retriever(search_kwargs={'k':TOP_K})


    def format_docs(docs):
        return " \n---\n ".join([f'[출처]:{doc.metadata['source']}\n[내용]: {doc.page_content}\n' for doc in docs])
    
    rag_chain_from_docs = (
    prompt
    | llm
    | StrOutputParser()
    )


    rag_chain_with_ref = RunnableParallel(
        raw_context = retriever, question = RunnablePassthrough()
    ).assign(context = lambda x: format_docs(x['raw_context'])).assign(answer=rag_chain_from_docs)



    results = rag_chain_with_ref.batch(questions)

    dataset=[]

    for result, reference in tqdm(zip(results,ground_truths)):
        dataset.append(
            {
                "user_input":result['question'],
                "retrieved_contexts":[doc.page_content for doc in result['raw_context']],
                "response":result['answer'],
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

실험 시작 시간과 정보를 저장할 수 있습니다.
import csv
import datetime
import os
def run_experiment():
    result = RAG_pipeline()
    timestamp = datetime.datetime.now().isoformat()
    row = {
        "timestamp": timestamp,
        "model_name": MODEL_NAME,
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

    print(f"[{timestamp}] {MODEL_NAME} | temp={TEMPERATURE}, chunk={CHUNK_SIZE}/{CHUNK_OVERLAP}, top_k={TOP_K} | results={result}")

MODEL_NAME='gpt-4.1-nano'
TEMPERATURE=0
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 5

run_experiment()
MODEL_NAME='gpt-4.1-nano'
TEMPERATURE=0
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 10

run_experiment()
MODEL_NAME='gpt-4.1-nano'
TEMPERATURE=0
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 400
TOP_K = 5

run_experiment()
MODEL_NAME='gpt-4.1-nano'
TEMPERATURE=0
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 400
TOP_K = 10

run_experiment()
다양한 기법을 위 코드에서 적용하여 성능을 높일 수 있습니다.
MODEL_NAME='gpt-4.1-mini'
TEMPERATURE=0
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 400
TOP_K = 10

run_experiment()
MODEL_NAME='gpt-5-mini'
TEMPERATURE=0
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 400
TOP_K = 10

run_experiment()