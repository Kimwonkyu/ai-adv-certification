***

## 1. LangChain + LLM 기본 패턴

시험에 자주 낼 수 있는 "틀" 위주로 정리하면 다음 형태들이 반복됩니다.

- LLM 생성 (예: OpenAI, Ollama 등)
  ```python
  from langchain_ollama import ChatOllama
  llm = ChatOllama(model="gpt-oss:20b", temperature=0.1, reasoning=True)
  ```

- 프롬프트 템플릿
  ```python
  from langchain.prompts import ChatPromptTemplate

  prompt = ChatPromptTemplate([
    ("system", "당신은 QA Assistant입니다. \nContext를 활용해 답하세요."),
    ("human","---\nContext: {context}\n---\nQuestion: {question}")
  ])
  ```

- 체인 구성 (RAG 체인 기본형)
  ```python
  from langchain.schema.runnable import RunnablePassthrough
  from langchain.schema.output_parser import StrOutputParser

  def format_docs(docs):
    return " \n\n---\n\n ".join(
        ["URL: " + d.metadata["source"] + "\nContent:" + d.page_content for d in docs]
    ) 
  
  rag_chain = (
    {"context": retriever | format_docs,
     "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser() 
  )
  ```

***  

## 2. Retriever / Vector DB 관련 자주 나올 코드

RAG 파트에서 빈칸으로 내기 좋은 부분들입니다.

- Chroma 벡터 DB 생성
  ```python
  from langchain_chroma import Chroma

  db = Chroma(
    embeding_function=openai_embeddings,
    persist_directory=f"./chroma_OpenAI_{uuidstr}",
    collection_name="Web",
    collection_metadata={"hnsw:space": "12"},
  )
  ```
- Retriever 생성 및 호출
  ```python
  retriever = db.as_retriever(search_kwargs={"k": 5})
  retriever.invoke("도메인 특화 언어 모델")
  ```  
- OpenAI 임베딩
  ```python
  from langchain_openai import OpenAIEmbeddings
  openai_embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
  ```

***

## 3. Runnable / Parallel 패턴 (퀴즈용으로 좋음)

체인 조합 쪽도 그대로 빈칸 문재로 내기 좋습니다.

- RunnableParallel + assign
  ```python
  from langchain_core.runnables import RunnableParallel
  from langchain.schema.runnable import RunnablePassthrough

  rag_chain_from_docs = (
    prompt
    | llm
    | StrOutputParser()
  )

  rag_chain_with_source = RunnableParallel(
    {"context": retriever | format_docs,
    "question": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)
    ```
- Runnable 퀴즈 예제
  ```python
  runnable = RunnableParallel(
    passed=RunnablePassthrough(),
    extra=RunnablePassthrough.assign(mult=lambda x : x["num"] * 3),
    modified=lamda x: x["num"] + 1,
  )    

  runnable.invoke({"num": 1})
  ```

***

## 4. OpenAI Python 클라이언트 기본 패턴

OpenAI API 활용 실습에서 나올 수 있는 빈칸 포인트 입니다.

- 기본 호출
  ```python
  from openao import OpenAI
  client = OpenAI()

  messages = [
    {"role": "system","content": "역할 안내...."},
    {"role": "user", "content": "질문"},
  ]

  response = client.chat.completions.create(
    model="gpt-4.1-mini",
    message=messages,
    temperature=0,
    max_completion_tokens=512
    n=1,
  )
  print(response.choices[0].message.content)
  ```
- 멀티턴 대화: messages 리스크에 이전 턴을 그대로 누적해서 다시 호출.

***  