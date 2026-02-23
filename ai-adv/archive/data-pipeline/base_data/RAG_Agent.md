## 1. RAG 기본 개념

- RAG 정의: Retrieval(검색) + Augmented(증강) + Generation(생성) 조합으로, 외부 DB/웹/내부 DB에서 관련 정보를 찾아 프롬프트에 붙여 넣고 LLM이 답을 생성하게 하는 방식이다.
- 필요 이유: LLM은 확률 기반 생성 모델이라 지식 컷오프와 할루시네이션 때문에 최신/도메인 특화/내부 데이터에 대해 틀린 답을 줄 수 있어, 이를 보완하려고 RAG를 사용한다.

## 2. RAG vs 파인튜닝

- 학습 방식: RAG는 In-Context Learning(프롬프트에 넣어 줄 때만 활용)이고, 파인튜닝은 모델 파라미터 자체를 수정하여 지식을 내장한다.
- 차이점 암기 포인트
  - 지식 저장 위치: RAG=외부 DB, 파인튜닝=모델 파라미터.
  - 변경/확장: RAG는 DB 내용만 바꾸면 되고, 파인튜닝은 GPU•시간•머신 언러닝 문제 때문에 변경이 어렵다.
  - 환각: RAG는 실제 문서를 그대를 넣어 할루시네이션을 줄이고, 파인튜닝은 잘 학습해도 환각이 남을 수 있다.

## 3. RAG 5단계 프로세스

5단계를 순서+기워드로 통으로 암기해두면 시험에 잘 나온다.

1. Indexing: 데이터를 텍스트로 준비하고, 청킹•임베딩•인덱싱을 통해 DB를 구성한다.
2. Processing: 사용자의 쿼리를 전처리(너무 길면 요약, 너무 짧으면 맥락화, 의도 분류 등)해서 검색에 적합하게 바꾼다.
3. Searching: 벡터 DB와/또는 키워드 검색으로 질의와 유사한 청크를 Top K로 찾는다.
4. Augmenting: 찾은 청크들을 프롬프트(Context)로 구성하고, 지시하상(Instruction) + Context + Quesiton 형태로 LLM에 전달한다.
5. Generating: LLM이 Context를 바탕으로 답을 생성하며, 필요 시 "답변할 수 없습니다" 같은 정책도 포함된다.

## 4.데이터 전처리와 청킹

- Context 길이: 최신 LLM은 수십만~수백만 토큰까지 지원하지만, 문서 전체를 넣는 것은 비효율적이므로 적절한 청킹이 필수다.
- 청킹(Chunking): 텍스트를 LLM이 이해 가능한 단위로 나누는 직업으로, 일반 텍스트는 문자/토큰 단위로, 소설•긴 보고서는 맥락을 고려해 단위를 잡는다.
- 청크 사이즈:
  - 너무 작으면 주변 맥락을 활용하기 어렵고, 너무 크면 임베딩 입력 제한•정확도 저하•불필요 정보 증가로 검색 성능이 떨어진다.
- 청크 오버랩: 청크 사이의 일부를 겹치게 해서 의미를 보존하며, 일반적으로 전체 크기의 10~20% 정도를 사용한다.

## 5. 임베딩•백터 DB•검색

- 임베딩 개념: 텍스트를 벡터로 바꾸고, 벡터 간 거리를 기반으로 의미적 유사도를 측정하는 것(Transformer 인코더의 토큰 벡터를 평균 등으로 사용).
- 인덱싱: 키워드 기반 구조를 만들어 Lexical(키워드) 검색을 가능하게 하는 것.
- Semantic vs Lexical 검색:
  - Semantic: 임베딩 기반 의미 검색, 일반 사용자 Q&A에 적합하지만 이름이 비슷한 경우 환각 위험.
  - Lexical: BM25/TF-IDF 같은 키워드 일치 기반, 내부 사용자/정확한 키워드에 강함.
  - 실제로는 하이브리드 검색(두 결과를 Rank Fusion 등으로 합침)이 가장 좋다.
- 벡터 DB: Pinecone, Milvus, Qdrant, Chroma, Weaviate 등이 대표적이며, 근사 최근접 이웃(ANN - 전체 벡터들을 트리로 변환하여 검색), 계층적 작은 세상 탐색(HNSW - 전체 벡터들을 그래프로 변환하여 검색) 알고리즘으로 빠르게 Top K를 찾는다.
- 비교 메트릭(Text Embedding): 코사인 커리(1-cosine similarity), 유클리드 거리(L2) 등을 사용하며, MMR은 다양성을 고려해 유사한 것만 반복해서 뽑히는 것을 막는다.

## 6. Text Similarity와 Bi-Encoder

- Cross-Encoder: 두 문장을 하나로 합쳐 BERT에 넣고 0~1 점수로 유사도를 내며, 성능은 좋지만 문장 수가 늘면 계산량이 폭발한다(예: 10문장 → 45쌍).
- Bi-Encoder(SentenceBERT): 문장 A,B를 각각 벡터로 만들고 코사인 유사도를 비교해 빠르게 대규모 검색이 가능하며, 성능도 Cross-Encoder와 비슷하게 유지한다.
- 최신 임베딩: BERT/DistilBERT 계열에서 출발해, 요즘은 LLM의 히든 레이어를 임베딩으로 쓰거나 LLM을 임베딩 전용으로 파인튜닝하는 방식이 각광받는다.

## 7. RAG 성능 고도화 핵심 키워드

암기용으로는 "청크•컨텍스트•검색•재랭킹•생성" 최적화 아이디어 위주로 잡으면 된다.

- Small2Big / Parent-Child / Sliding Window
  - Small Chunk로 검색하고, 실제 Context는 더 큰 Chunk(Parent)에 Sliding Window로 확장하여 맥락과 검색 정확도를 동시에 잡는다.
- Contextual Retrieval
  - 각 청크에 "이 문서는 어떤 보고서의 몇 년 내용인지" 같은 요약/헤더를 LLM으로 미리 붙여 검색 가능성을 높이는 방식이지만, LLM 비용이 크다.
- HyDE(Hypothetical Document Embedding)
  - 사용자의 질문에 대해 LLM이 가상의 답변/문서를 먼저 생성하고, 그 가짜 문서를 임베딩해서 실제 문서와 비교해 검색 성능을 올리는 기법이다.
- Query Reformulation / Multi-Querying / Agentic RAG
  - 하나의 질문을 여러 변형 쿼리로 만들거나, 검색 결과의 정확성•모호성•오류 여부를 평가해 재검색•분해 등으로 개선하는 에이전트형 RAG 구조.
- Lost in the Middle & Reordering
  -  Bi-Encorder 리트리버 후 Cross-Encoder나 LLM 기반 Reranker로 Top 30 → Top 10을 선별하고, 점수 기반으로 순서를 재배열하며, 경우에 따라 원문 순서를 유지한다.
-  Generation 단계 고도화
   -  LLM이 "컨택스트 인용, 불필요 청크 무시, 무관한 청크 표시" 등을 잘 하도록 파인튜닝하고, Long Context LLM에 최적화된 RAG 전략은 적용한다.
  
## 8. 멀티모달 RAG & 실습

- 멀티모달 RAG: 텍스트뿐 아니라 이미지•테이블이 포함된 문서를 Docling 같은 피서로 텍스트/구조화 데이터로 변환해 RAG에 활용한다.
- Docling: IBM 오픈소스로 PDF, PPT, DOC, HTML 등을 페이지 이미지로 만들고 이미지•테이블을 파싱해 LangChain Document Loader로 처리 가능하게 한다.

***