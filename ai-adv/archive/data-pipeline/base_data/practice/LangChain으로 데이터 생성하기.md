# [실습] LangChain으로 데이터 생성하기




이번 실습에서는 Continuous Pretrain과 Instruction Tuning을 위한 데이터 생성에 대해 알아보겠습니다.


!pip install langchain langchain-community langchain-openai langchain-ollama
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import FewShotPromptTemplate
from langchain.schema.output_parser import StrOutputParser
import os
import json
from dotenv import load_dotenv

load_dotenv('.env',override=True)
# LLM 설정하기

무료 API: Gemini를 사용하는 경우 분당 10회 제한을 고려하여 Rate Limiter를 설정합니다.
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI


# rate limiter를 LLM에 적용
llm = ChatOpenAI(
    model="gpt-4.1",
    temperature=0,
)
llm.invoke("안녕")
## 도메인 코퍼스 만들기

Continous Pretraining을 위한 Corpus를 생성합니다.   
코퍼스는 지식 학습에 매우 중요한 역할을 수행하며, 정확성이 매우 중요하기 때문에   
일반적으로는 API, 외부 자료 수집 등을 통해 구성하는 것이 효과적입니다.
try:
    disease_list = open('./disease_list.txt','r').read().strip().split(',')
except:
    search_prompt = '''
사회 취약 계층을 위한 의료 정보 책자를 만들고자 합니다.
해당 책자에는 다양한 질환의 정의와 증상, 치료 방법, 이후의 관리 방법이 포함됩니다.
자료 수집을 위해, 대표적인 10개의 질환 목록을 선정하세요.
각 단어의 한국어 명칭을 , 로 구분하여 나열해 주세요. 예시: 단어1, 단어2, ... 단어10
단어 이외의 다른 내용은 출력하지 마세요.
한글 명칭만 작성하세요.
'''
    disease_list = llm.invoke(search_prompt).content.split(', ')
    with open('disease_list.txt','w') as f:
        f.write(','.join(disease_list))
    f.close()
len(disease_list)
disease_list
생성된 질환에 대해, 세부 사항을 작성하기 위한 카테고리를 구성합니다.
from langchain.prompts import ChatPromptTemplate
from itertools import product
from typing import List, Dict
import json
from tqdm import tqdm  # 진행상황 표시용
from datetime import datetime
import time

# 카테고리 옵션 정의
category_options = {
   # 문서 주제 분류
   'document_type': [
      # 일반 소개 관련
      'Symptoms', # 증상
      'General Information', # 일반 정보
      'Frequently Asked Questions', # 자주 묻는 질문

       # 약물치료 관련
       'Drug types and mechanisms',        # 약물 종류와 작용기전
       'Administration and precautions',   # 투약 방법과 주의사항
       'Side effects management',          # 부작용 관리

       # 수술 관련
       'Surgical indications',             # 수술 적응증
       'Procedure overview',               # 수술 과정 개요
       'Post-operative care',              # 수술 후 관리

       # 생활습관 관련
       'Diet and exercise',                # 식이요법과 운동
       'Daily life management',            # 일상생활 관리
       'Prevention guidelines',            # 예방 지침

       # 진단과 치료계획 관련
       'Diagnostic process',               # 진단 과정
       'Treatment strategy',               # 치료 전략
       'Progress monitoring',              # 경과 관찰
            
   ],

   # 상세도 수준 다양화
   'level': ['Standard', 'Expert', 'For Elderly (easy to understand)']    
}
Combination의 개수를 구합니다.
combinations = list(product(disease_list, category_options['document_type'], category_options['level']))
len(combinations)
arguments = [{'disease':x[0],'document_type':x[1],'level':x[2]} for x in combinations]
arguments[0]
이제, 생성 프롬프트 체인을 만들고 배치로 실행합니다.
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

medical_prompt = ChatPromptTemplate([
   ('system', '''You are a medical professor creating training data for healthcare AI development.

Core Principle: Each document must be self-contained and complete within its specific scope.

Writing Guidelines:
- Maintain medical accuracy while emphasizing logical flow
- Include evidence-based, up-to-date medical knowledge
- When using medical terms, include explanations in parentheses
- Write output in Korean language with clear and precise explanations
- Focus on information relevant to vulnerable populations
- Write comprehensive content within your assigned scope
- Use natural medical language without artificial boundaries

Document Type Focus - Write ONLY about:

1. **Drug types and mechanisms**: Pharmacological classifications, receptor mechanisms, metabolic pathways, drug action principles
2. **Administration and precautions**: Dosing schedules, administration routes, drug interactions, contraindications, timing considerations
3. **Side effects management**: Adverse event identification, severity assessment, management protocols, when to discontinue
4. **Surgical indications**: Surgical criteria, risk-benefit analysis, timing decisions, patient selection factors
5. **Procedure overview**: Surgical techniques, anesthesia types, procedural steps, equipment and technology used
6. **Post-operative care**: Wound management, pain control, activity progression, complication monitoring
7. **Diet and exercise**: Nutritional requirements, food restrictions, exercise prescriptions, activity modifications
8. **Daily life management**: Work adaptations, home environment modifications, assistive devices, social considerations
9. **Prevention guidelines**: Risk factor modification, screening protocols, preventive medications, lifestyle interventions
10. **Diagnostic process**: Test selection, diagnostic criteria, differential diagnosis, result interpretation
11. **Treatment strategy**: Treatment algorithms, therapy selection criteria, combination approaches, escalation protocols
12. **Progress monitoring**: Follow-up intervals, monitoring parameters, outcome measurements, treatment response criteria
13. **Frequently Asked Questions**: Common misconceptions, practical concerns, cost considerations, healthcare access
14. **General Introduction**: Disease definition, epidemiology, pathophysiology, natural history, disease burden

Remember:
- Each document should be complete and informative within its designated scope
- Write for medically vulnerable populations with clear explanations
- All content must be written in Korean language
- Focus exclusively on your document type without mentioning other aspects'''),

   ('user', '''Please write comprehensive medical content about {disease}.
Document type: {document_type}
Detail level: {level}

Create a thorough, naturally flowing medical document that completely covers this specific aspect of {disease}.
Ensure the content is self-contained and professionally written.
Please write the entire response in Korean.''')
])

llm_chain = medical_prompt | llm | StrOutputParser()

generate_chain = RunnableParallel(metadata = RunnablePassthrough(), result = llm_chain)

# generated_corpus = generate_chain.batch(arguments[0:2])
import json
from datetime import datetime
import time

# 배치 크기 설정 (API 제한 고려)
BATCH_SIZE = 30  # 한 번에 처리할 개수
generated_corpus = []
failed_items = []

# 전체 조합을 배치로 나누기
batches = [combinations[i:i+BATCH_SIZE] for i in range(0, len(combinations), BATCH_SIZE)]

# 진행상황 표시하며 처리
for batch_idx, batch in enumerate(tqdm(batches, desc="Processing batches")):
    try:
        # 각 배치의 arguments 준비
        batch_args = [
            {
                'disease': disease,
                'document_type': doc_type,
                'level': level
            }
            for disease, doc_type, level in batch
        ]
        
        # 배치 처리
        results = generate_chain.batch(batch_args)
        
        # 결과에 인덱스 추가하여 저장
        for (disease, doc_type, level), result in zip(batch, results):
            generated_corpus.append({
                'index': len(generated_corpus),
                'disease': disease,
                'document_type': doc_type,
                'level': level,
                'content': result['result'],
                'metadata': result['metadata'],
                'timestamp': datetime.now().isoformat()
            })
        
        # 중간 저장 (3번 배치마다)
        if (batch_idx + 1) % 3 == 0:
            with open(f'corpus_checkpoint_{batch_idx+1}.json', 'w', encoding='utf-8') as f:
                json.dump(generated_corpus, f, ensure_ascii=False, indent=2)
            print(f"Checkpoint saved: {len(generated_corpus)} documents")
            
    except Exception as e:
        print(f"Error in batch {batch_idx}: {e}")
        failed_items.extend(batch)
        continue
    
    # API 제한 방지용 대기
    time.sleep(1)

# 최종 결과 저장
with open('medical_corpus_result.json', 'w', encoding='utf-8') as f:
    json.dump({
        'total_documents': len(generated_corpus),
        'generation_date': datetime.now().isoformat(),
        'failed_items': len(failed_items),
        'documents': generated_corpus
    }, f, ensure_ascii=False, indent=2)

print(f"완료: {len(generated_corpus)}개 생성, {len(failed_items)}개 실패")
생성된 데이터를 확인해 보겠습니다.
import json

with open ("medical_corpus_result.json", "r", encoding='utf-8') as f:
    data = json.load(f)
len(data['documents'])
data['documents'][292]
import matplotlib.pyplot as plt
content_lengths = [len(doc['content']) 
                  for doc in data['documents'] 
                  if 'content' in doc and doc['content']]

# 히스토그램 그리기
plt.figure(figsize=(10, 6))
plt.hist(content_lengths, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
plt.title('Medical Corpus - Document Length Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Document Length (characters)', fontsize=12)
plt.ylabel('Number of Documents', fontsize=12)
plt.grid(True, alpha=0.3)

# 평균선 추가
mean_length = sum(content_lengths) / len(content_lengths)
plt.axvline(mean_length, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_length:.0f}')
plt.legend()

plt.tight_layout()
plt.show()

print(f"Total documents: {len(content_lengths)}")
print(f"Average length: {mean_length:.0f} characters")
for doc in data['documents']:
    if len(doc['content']) > 2800:
        print(doc)
생성된 Corpus는 이후 Continuous Pretraining 실습에 사용할 수 있습니다.
API 호출 메타데이터를 제거하고, 파일로 저장합니다.
with open('medical_corpus.json', 'w', encoding='utf-8') as f:
    json.dump(data['documents'], f, ensure_ascii=False, indent=2)

print('medical corpus 생성 완료')