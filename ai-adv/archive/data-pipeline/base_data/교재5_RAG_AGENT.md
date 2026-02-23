📘 [학습 노트] 교재 5. RAG & Agent (검색 증강 생성 및 에이전트)
1. RAG (Retrieval Augmented Generation) 개요
• 정의: 검색(Retrieval)과 생성(Generation)을 결합한 기술로, 외부 데이터베이스나 웹에서 관련 정보를 찾아 프롬프트에 '맥락(Context)'으로 붙여 넣고 LLM이 답을 생성하게 하는 방식입니다.
• 필요성: LLM의 **지식 단절(Knowledge Cutoff)**과 할루시네이션(환각) 문제를 해결하고, 최신 정보나 내부(Private) 데이터를 활용하기 위해 필수적입니다.
• RAG vs Fine-Tuning 비교:
    ◦ RAG: 외부 DB에 지식을 저장하므로 업데이트가 쉽고, 근거 문서를 제시하여 환각을 줄일 수 있습니다 (In-Context Learning).
    ◦ Fine-Tuning: 모델 내부 파라미터에 지식을 각인시키는 방식으로, 말투나 형식을 교정하는 데 유리하지만 지식 업데이트가 어렵고(재학습 필요) 환각이 여전히 발생할 수 있습니다.
2. RAG 파이프라인 5단계 (핵심 암기)
시험에 자주 출제되는 프로세스이므로 순서와 키워드를 익혀두어야 합니다,.
1. Indexing: 데이터를 텍스트로 준비하고 청킹(Chunking), 임베딩(Embedding), 인덱싱을 통해 DB를 구축합니다.
2. Processing: 사용자의 쿼리가 검색에 적합하도록 전처리(확장, 의도 분류 등)합니다.
3. Searching: 벡터 DB나 키워드 검색으로 질문과 유사한 청크(Top K)를 찾습니다.
4. Augmenting: 찾은 청크를 프롬프트에 Instruction + Context + Question 형태로 조합합니다.
5. Generating: LLM이 제공된 Context를 바탕으로 답변을 생성합니다.
3. 데이터 처리 및 검색 (Indexing & Searching)
• 청킹 (Chunking):
    ◦ 텍스트를 LLM이 이해 가능한 단위로 나누는 작업입니다.
    ◦ 너무 작으면 맥락이 잘리고, 너무 크면 정확도가 떨어지므로 의미 단위나 **Overlap(10~20%)**을 주어 자릅니다.
• 검색 방식 (Search Types):
    ◦ Semantic Search: 임베딩 벡터 간의 의미적 유사도(코사인 유사도 등)를 기반으로 검색합니다.
    ◦ Lexical Search: BM25, TF-IDF 등 키워드 일치 여부를 기반으로 검색하며 정확한 용어 검색에 유리합니다.
    ◦ Hybrid Search: 두 방식을 결합(Ensemble)하여 상호 보완하는 가장 성능 좋은 방식입니다,.
• 벡터 DB: Chroma, Pinecone 등에서 **ANN(근사 최근접 이웃)**이나 HNSW 알고리즘을 사용해 빠른 검색을 수행합니다.
4. RAG 성능 고도화 전략 (Advanced RAG)
단순 검색의 한계를 극복하기 위한 심화 기법들입니다.
• Multi-Querying: 사용자의 단일 질문을 LLM을 이용해 여러 관점의 질문으로 확장하여 검색 범위를 넓힙니다 (Query Reformulation),.
• HyDE (Hypothetical Document Embedding): 질문에 대한 '가상 답변'을 먼저 생성하고, 그 가상 답변과 유사한 실제 문서를 검색하여 정확도를 높입니다.
• Reranking (재순위화):
    ◦ 1차로 빠르게 검색(Bi-Encoder)한 뒤, 정밀한 모델(Cross-Encoder)로 상위 문서들의 순위를 다시 매겨 정확도를 극대화합니다.
• Small2Big (Parent-Child): 검색은 작은 청크로 정밀하게 하되, LLM에게는 그 청크가 포함된 더 큰 맥락(Parent Chunk)을 제공하는 방식입니다.
5. LangChain 구현 및 코드 패턴 (시험 대비)
실습 및 코드 문제에서 자주 등장하는 패턴입니다,,.
• 기본 체인 구조 (LCEL):
    ◦ Retriever | Prompt | LLM | OutputParser 순서로 연결됩니다,.
    ◦ RunnablePassthrough: 값을 변경 없이 다음 단계로 전달할 때 사용합니다.
• 주요 클래스:
    ◦ RecursiveCharacterTextSplitter: 문서를 쪼갤 때 사용 (chunk_size, chunk_overlap 설정).
    ◦ MultiQueryRetriever: 쿼리 확장을 위해 사용,.
    ◦ EnsembleRetriever: BM25와 Vector 검색을 결합할 때 사용.
6. Agent와 도구 사용 (Tool Calling)
• Agent: 단순한 답변 생성을 넘어, 스스로 계획을 세우고(Reasoning) 외부 도구(검색, 계산기 등)를 사용하여(Acting) 문제를 해결하는 시스템입니다.
• ReAct 프레임워크: Reasoning(생각) → Acting(행동) → Observation(관찰) 과정을 반복하며 복잡한 과업을 수행합니다,.
• Tool Calling: LLM이 함수 실행에 필요한 인자를 JSON 형태로 출력하면, 시스템이 실제 함수를 실행하고 그 결과를 다시 LLM에 돌려주는 구조입니다,.