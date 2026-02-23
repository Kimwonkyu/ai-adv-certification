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
# os.environ['OPENAI_API_KEY']=''
# LLM 설정하기

무료 API: Gemini를 사용하는 경우 분당 10회 제한을 고려하여 Rate Limiter를 설정합니다.
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI


# rate limiter를 LLM에 적용
llm = ChatOpenAI(
    model="gpt-4.1",
    temperature=0,
   #  verbosity='low'
)
llm.invoke("안녕")
## Instruction Data 만들기   

도메인 지식을 Continuous Pretrain으로 학습한 모델을 실제로 사용하기 위해서는   
Instruction Tuning이 추가로 수행되어야 합니다.

간단한 의료 상담을 제공하는 질의응답 데이터를 만들어 보겠습니다.

**(대부분의 LLM은 의료, 법률 등의 상담을 실제 상황에서 직접적으로 수행하는 것에 대한 제한이 존재합니다.)**
disease_list = open('./disease_list.txt','r', encoding='cp949').read().strip().split(',')
disease_list
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import FewShotPromptTemplate
from langchain.schema.output_parser import StrOutputParser

import os
import json
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from itertools import product
from typing import List, Dict
import json
from tqdm import tqdm
from datetime import datetime
import time
import random
import re

# 카테고리 옵션 정의
category_options = {
    'question_type': [
        'medication_general',      
        'medication_side_effects', 
        'symptoms_general',        
        'symptoms_severity',       
        'diagnosis_process',       
        'test_results',           
        'treatment_options',       
        'treatment_duration',      
        'lifestyle_diet',         
        'lifestyle_exercise',      
        'prognosis_general',      
        'prevention_methods'       
    ],
    'patient_type': [
        'new_patient',            
    ],
    'age_group': [
        'child',                  
        'elderly'                 
    ],
    'verbosity': [
        '200자 이내로', '500자 이내로', '1000자 이내로'
    ]
}

# 다양성을 위한 상황 설정
PATIENT_SITUATIONS = {
    'child': [
        "5살 아들", "7살 딸", "10살 아이", "초등학생 자녀", "4살 손녀", 
        "9살 조카", "6살 아들", "11살 딸", "초등 3학년 아이", "유치원생",
        "8살 아이", "초등 1학년", "12살 아들", "3살 딸", "초등 6학년"
    ],
    'elderly': [
        "72세 어머니", "68세 아버지", "75세 시어머니", "80세 할머니", 
        "70세 장인어른", "65세 이모", "78세 할아버지", "73세 어머님",
        "69세 시아버지", "76세 친정아버지", "82세 할아버지", "71세 삼촌",
        "77세 고모", "74세 장모님", "67세 큰아버지"
    ],
    'self_age': [
        "30대 직장인", "40대 주부", "50대 자영업자", "35세 회사원",
        "42세 교사", "38세 간호사", "45세 사업가", "28세 대학원생",
        "55세 공무원", "33세 프리랜서", "48세 의사", "36세 엔지니어",
        "29세 디자이너", "52세 요리사", "41세 변호사"
    ],
    'duration': [
        "3일 전부터", "일주일째", "2주 전부터", "한 달 전부터", "최근 며칠간",
        "어제부터", "5일째", "보름 전부터", "열흘 정도", "이틀 전부터",
        "4일 전부터", "3주째", "6일 동안", "반달 전부터", "한 시간 전부터"
    ],
    'severity': [
        "조금씩", "갑자기", "점점 심해져서", "가끔씩", "계속해서",
        "반복적으로", "간헐적으로", "심하게", "약간", "자주",
        "때때로", "급격히", "서서히", "주기적으로", "매일"
    ],

}

# Q&A 생성 프롬프트
qa_prompt = ChatPromptTemplate([
    ('system', '''You are an AI assistant creating diverse doctor-patient Q&A pairs for medical AI training.

Task: Generate two DIFFERENT patient questions and doctor answers.

DIVERSITY REQUIREMENTS:
- Each Q&A must use completely different patient situations
- Vary ages: {child_examples} / {elderly_examples} / {self_examples}
- Vary durations: {duration_examples}
- Vary severity: {severity_examples}
- Make questions natural and realistic
- Answers should be professional, accurate, and helpful

Output Format (STRICT JSON):
{{
    "qa_1": {{
        "question": "구체적인 상황이 포함된 환자 질문",
        "answer": "전문적이고 친절한 의사 답변"
    }},
    "qa_2": {{
        "question": "완전히 다른 상황의 환자 질문",
        "answer": "해당 상황에 맞춘 의사 답변"
    }}
}}

Write everything in Korean. Each Q&A must be unique and different.'''),
    
    ('user', '''질병: {disease}
질문 유형: {question_type}
환자 유형: {patient_type}
연령대: {age_group}
Answer Verbosity : {verbosity}

서로 다른 상황의 Q&A 2개를 JSON 형식으로 생성하세요.''')
])

def parse_qa_response(text: str) -> dict:
    """응답을 파싱하여 Q&A 데이터 추출"""
    try:
        if hasattr(text, 'content'):
            text = text.content
        return json.loads(text)
    except:
        text = str(text).strip()
        if '```json' in text:
            text = text.split('```json')[1].split('```')[0]
        elif '```' in text:
            text = text.split('```')[1].split('```')[0]
        
        try:
            return json.loads(text)
        except:
            return None

def generate_medical_qa_dataset_batch(diseases: List[str], llm) -> List[Dict]:
    """배치 처리로 빠른 Q&A 데이터셋 생성"""
    
    qa_dataset = []
    failed_generations = []
    
    # 전체 조합 생성
    combinations = list(product(
        diseases,
        category_options['question_type'],
        category_options['patient_type'],
        category_options['age_group'],
        category_options['verbosity']
    ))
    
    print(f"📊 Total combinations: {len(combinations)}")
    print(f"🎯 Expected Q&A pairs: {len(combinations) * 2}")
    
    # 배치 크기 설정 (API 제한 고려)
    BATCH_SIZE = 80  # 한번에 80개씩 처리
    
    # LLM 체인 생성
    chain = qa_prompt | llm
    
    # 전체 입력 파라미터 준비
    all_params = []
    for disease, q_type, p_type, age, verbosity in combinations:
        params = {
            'disease': disease,
            'question_type': q_type,
            'patient_type': p_type,
            'age_group': age,
            'verbosity': verbosity,
            'child_examples': ', '.join(random.sample(PATIENT_SITUATIONS['child'], 4)),
            'elderly_examples': ', '.join(random.sample(PATIENT_SITUATIONS['elderly'], 4)),
            'self_examples': ', '.join(random.sample(PATIENT_SITUATIONS['self_age'], 4)),
            'duration_examples': ', '.join(random.sample(PATIENT_SITUATIONS['duration'], 4)),
            'severity_examples': ', '.join(random.sample(PATIENT_SITUATIONS['severity'], 4))
        }
        all_params.append(params)
    
    # 배치로 나누기
    batches = [all_params[i:i+BATCH_SIZE] for i in range(0, len(all_params), BATCH_SIZE)]
    
    print(f"🚀 Processing {len(batches)} batches of {BATCH_SIZE} items each")
    
    # 배치 처리
    for batch_idx, batch_params in enumerate(tqdm(batches, desc="Processing batches")):
        try:
            # 배치 실행
            batch_results = chain.batch(batch_params)
            
            # 결과 처리
            for params, result in zip(batch_params, batch_results):
                try:
                    # 파싱
                    qa_data = parse_qa_response(result)
                    
                    if qa_data is None:
                        raise ValueError("Failed to parse")
                    
                    # 개별 Q&A 저장
                    for i, (qa_key, qa_content) in enumerate(qa_data.items(), 1):
                        if 'question' in qa_content and 'answer' in qa_content:
                            qa_item = {
                                'id': f"{params['disease']}_{params['question_type']}_{params['age_group']}_{len(qa_dataset)}",
                                'disease': params['disease'],
                                'question_type': params['question_type'],
                                'patient_type': params['patient_type'],
                                'age_group': params['age_group'],
                                'pair_number': i,
                                'question': qa_content['question'],
                                'answer': qa_content['answer'],
                                'timestamp': datetime.now().isoformat()
                            }
                            qa_dataset.append(qa_item)
                
                except Exception as e:
                    failed_generations.append({
                        'disease': params['disease'],
                        'question_type': params['question_type'],
                        'error': str(e)[:100]
                    })
            
            # 체크포인트 저장 (2 배치마다)
            if (batch_idx + 1) % 2 == 0:
                checkpoint = {
                    'metadata': {
                        'processed_batches': batch_idx + 1,
                        'total_qa_pairs': len(qa_dataset),
                        'failed': len(failed_generations)
                    },
                    'qa_dataset': qa_dataset
                }
                with open(f'qa_checkpoint_batch_{batch_idx+1}.json', 'w', encoding='utf-8') as f:
                    json.dump(checkpoint, f, ensure_ascii=False, indent=2)
                print(f"\n💾 Checkpoint: {len(qa_dataset)} Q&As saved")
            
            # API 제한 방지 (필요시)
            time.sleep(1)
            
        except Exception as e:
            print(f"\n❌ Batch {batch_idx} failed: {str(e)[:100]}")
            for params in batch_params:
                failed_generations.append({
                    'disease': params['disease'],
                    'question_type': params['question_type'],
                    'batch_error': str(e)[:100]
                })
    
    return qa_dataset, failed_generations

def save_final_dataset(qa_dataset: List[Dict], failed: List[Dict]):
    """최종 데이터셋 JSON 저장"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 통계 정보
    stats = {
        'total_qa_pairs': len(qa_dataset),
        'unique_diseases': len(set(item['disease'] for item in qa_dataset)),
        'unique_question_types': len(set(item['question_type'] for item in qa_dataset)),
        'failed_attempts': len(failed),
        'success_rate': f"{(len(qa_dataset) / (len(qa_dataset) + len(failed)*2) * 100):.1f}%" if qa_dataset else "0%",
        'generation_date': timestamp
    }
    
    # 최종 데이터
    final_data = {
        'metadata': stats,
        'qa_dataset': qa_dataset,
        'failed_generations': failed
    }
    
    # JSON 저장
    filename = f'medical_qa_dataset.json'
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Dataset saved!")
    print(f"📊 Total Q&A pairs: {stats['total_qa_pairs']}")
    print(f"📊 Success rate: {stats['success_rate']}")
    print(f"📁 File: {filename}")
    
    return filename

# 실행 코드
if __name__ == "__main__":
    try:
        start_time = time.time()
        print("🚀 Starting BATCH Medical Q&A Generation")
        print(f"📋 {len(disease_list)} diseases × 12 types × 2 age groups = {len(disease_list)*24} combinations")
        print(f"⚡ Using batch processing for speed")
        print("=" * 50)
        
        # 배치 처리로 데이터셋 생성
        qa_dataset, failed_generations = generate_medical_qa_dataset_batch(disease_list, llm)
        
        # 최종 저장
        filename = save_final_dataset(qa_dataset, failed_generations)
        
        elapsed = (time.time() - start_time) / 60
        print(f"\n✨ Completed in {elapsed:.1f} minutes!")
        print(f"⚡ Speed: {len(qa_dataset)/elapsed:.1f} Q&As per minute")
        
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted")
        if 'qa_dataset' in locals():
            save_final_dataset(qa_dataset, failed_generations)
    
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        if 'qa_dataset' in locals():
            save_final_dataset(qa_dataset, failed_generations)
import json

with open('./medical_qa_dataset.json', 'r', encoding='utf-8') as file:
    qa_corpus = json.load(file)

qa_corpus['qa_dataset']
with open('medical_qa_data.json', 'w', encoding='utf-8') as f:
    json.dump(qa_corpus['qa_dataset'], f, ensure_ascii=False, indent=2)

print('medical QA 생성 완료')
