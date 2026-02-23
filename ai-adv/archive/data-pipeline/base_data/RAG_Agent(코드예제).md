***

## 시험에 잘 나올 수 있는 LangChain/RAG 핵심 코드 패턴

### 1. Document Loader & Splitter

```python
from langchain.document_loaders import PyMuPDFLoader, WebBaseLoader, CSVLoader, JSONLoader
from langchain.text_splitter import RecursiveCharacterTextgSplitter

loader = PyMuPDFLoader("reports2025.03.11.pdf")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=400,
)
chunks = splitter.split_documents(docs)
```

### 2. Chroma 벡터DB + Retriever

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
db = Chroma(
    embedding_function=embeddings,
    persist_directory=".chroma2",
    collection_name="finance",
    collection_metadata={"hnsw:space":"12"},
)

db.add_documents(chunks)

retriever = db.as_retriever(
    search_kwargs={"k": 5},
)
```

### 3. 기본 RAG 체인 구성

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_messages([
    ("system", "QA Question-Answering Assistant. \nContext와 Question으로만 답하세요."),
    ("human", "Context:\n{context}\n---\nQuestion: {question}"),
])

def format_docs(docs)
    return "---".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

### 4. MultiQueryRetriever 생성

```python
from langchain_core.prompts import PromptTemplate
from langchain.retrievers.multiquery import MultiQueryRetriever

rewrite_prompt = PromptTemplate(
    template=(
        "SDS 사업 보고서를 잘 아는 전문가입니다.\n"
        "다음 질문을 3개의 서로 다른 검색 쿼리로 바꿔주세요.\n"
        "---\n"
        "{question}"
    ),
    input_variables=["question"],
)

multi_query_retriver = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    prompt=rewrite_prompt,
    llm=llm,
)
```
### 5. BM25 / EnsembleRetriever

```python
from kiwipiepy import Kiwi
from langchain.retrievers import BM25Retriever, EnsembleRetriever

kiwi = Kiwi()

def kiwi_tokenize(text: str):
    return []
