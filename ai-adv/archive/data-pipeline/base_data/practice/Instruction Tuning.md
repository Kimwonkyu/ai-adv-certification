# [실습] Instruction Tuning

Continuous Pretraining(CPT)을 통해 지식을 주입한 뒤에는    

Instruction Tuning을 통해 질의응답 능력을 학습시킵니다.   

CPT 과정과 마찬가지로, 학습 중 다양한 데이터를 함께 학습시켜 일반화 성능을 유지할 수 있습니다.


!pip install tensorboard transformers==4.56.0 seaborn langchain langchain-huggingface pandas accelerate datasets huggingface_hub trl bitsandbytes -q
# flash attention
# !pip install flash-attn --no-build-isolation -q
허깅페이스 토큰을 입력합니다.
import os

from huggingface_hub import login

# 허깅페이스 토큰 로그인: Llama, DeepSeek 의 경우 로그인 필수, Qwen은 로그인 필요없음
# login(token='')
모델의 주소를 입력합니다.    
이전 학습에서 그대로 이어서 진행하는 경우, 로컬 주소를 넣어도 됩니다.   
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer

model_id="./gemma-3-1b-pt-MED"
# 모델의 주소
# (로컬 주소도 가능)

model_name = model_id.split('/')[1]
print("## MODEL:", model_name)
모델과 토크나이저를 불러옵니다.
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
)
model = AutoModelForCausalLM.from_pretrained(model_id,
                                             torch_dtype='auto',
                                             device_map="auto",
                                             attn_implementation="eager",
                                            )
Chat Template 확인을 위해 tokenizer의 템플릿을 확인합니다.
print(tokenizer.chat_template)
Pretrain 모델에 해당 템플릿이 포함된 경우도 있지만, 이번 경우에는 설정이 되어 있지 않은 상황입니다.
Instruct 모델의 토크나이저를 추가로 불러와 활용하겠습니다.
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

instruct_model_id = 'unsloth/gemma-3-1b-it'


tokenizer = AutoTokenizer.from_pretrained(
    instruct_model_id,
)
print(tokenizer.chat_template[:100])
tokenizer.eos_token
학습 데이터를 불러옵니다.
## [중요]
이번에는 모델의 일반화 성능을 평가하기 위해, 모델을 Train/Test로 분리합니다.

Test 데이터는 검증 데이터에 해당하며, 학습에는 참여하지 않으나   
학습의 과적합과 종료 여부를 판단하는 과정에 활용합니다.
from datasets import load_dataset

file_path = ['med_qa_data.json']

# 1. load_dataset으로 DatasetDict 형태로 불러오기
medQA_data = load_dataset("json", data_files={"train": file_path})

# 2. 데이터 분할
# 원래는 더 많이 넣어야 하나 데이터가 부족해서 5%만 분할..
medQA_data = medQA_data["train"].train_test_split(test_size=0.05, seed=42)

medQA_data

# 학습에 사용할 train
print('Question:', medQA_data['train'][0]['question'])
print('Answer:', medQA_data['train'][0]['answer'])
# 검증에 사용할 test
print('Question:', medQA_data['test'][0]['question'])
print('Answer:', medQA_data['test'][0]['answer'])
QA 데이터만으로 학습해도 되지만, 일반 QA를 포함하여 학습시키면 기존 능력을 다소 유지시킬 수 있습니다.  

대표적인 한국어 Instruction Data인 KoAlpaca-RealQA 데이터를 합쳐 보겠습니다.

https://huggingface.co/datasets/beomi/KoAlpaca-RealQA (인증 필요)
# 인증이 필요하므로, 이번에는 실습시트의 토큰을 그대로 사용
login(token='hf_REDACTED')

realQA_data = load_dataset("beomi/KoAlpaca-RealQA")
realQA_data = realQA_data["train"].train_test_split(test_size=0.05, seed=42)

realQA_data
print('Question:', realQA_data['train'][0]['question'])
print('Answer:', realQA_data['train'][0]['answer'])
# 학습에 참여하지 않는 validation
print('Question:', realQA_data['test'][0]['question'])
print('Answer:', realQA_data['test'][0]['answer'])
두 데이터를 결합하여 전체 데이터를 구성합니다.   
비율은 학습의 결과를 보면서 결정해야 합니다.
[realQA_data, medQA_data]
from datasets import concatenate_datasets, DatasetDict

realQA_data['train'] = realQA_data['train'].shard(num_shards=3, index=0)
# 1/3만  선택하는 방법 : 6175개

# 이제 Dataset들끼리 합침
merged_train = concatenate_datasets([realQA_data['train'], medQA_data['train']])
merged_test = concatenate_datasets([realQA_data['test'], medQA_data['test']])

# DatasetDict로 재구성
data = DatasetDict({
    'train': merged_train.shuffle(seed=42),
    'test': merged_test.shuffle(seed=42)
})
data
이제 학습 데이터가 준비되었습니다!
학습 데이터는 토크나이저의 Chat 템플릿으로 변환해야 합니다.
토크나이저로부터, 템플릿을 변환하는 함수를 생성합니다.
def convert_format(question,answer=None):
    # question, answer 세트(학습시) 또는 question만 입력(실행시)

    chat = [{'role': 'user', 'content': f"{question}"}]

    if answer:
        chat.append({'role':'assistant', 'content':f"{answer}"})

        return {'text':tokenizer.apply_chat_template(chat, tokenize=False)}
        # 학습해야 하는 데이터이므로 Generation Prompt 넣지 않음

    return {'text': tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)}
    # 입력으로 전달해야 하는 데이터이므로 add_generation_prompt 넣음

`map`으로 QA 데이터세트를 전처리합니다.
data = data.map(lambda x:convert_format(x['question'], x['answer']))
이전 실습과 동일하게, 토큰 수를 세어 보겠습니다.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter

def analyze_token_distribution(dataset, text_column='text', bins=30):

    # 토큰 수 계산
    token_counts = []
    for text in dataset[text_column]:
        tokens = tokenizer.encode(text)
        token_counts.append(len(tokens))

    # 기본 통계 계산
    stats = {
        '평균 토큰 수': np.mean(token_counts),
        '중앙값': np.median(token_counts),
        '최소 토큰 수': min(token_counts),
        '최대 토큰 수': max(token_counts),
        '표준편차': np.std(token_counts),
        '90퍼센타일': np.percentile(token_counts, 90),
        '95퍼센타일': np.percentile(token_counts, 95),
        '99퍼센타일': np.percentile(token_counts, 99),
        '총 샘플 수': len(token_counts)
    }

    # 분포 시각화
    plt.figure(figsize=(12, 6))

    # 히스토그램과 KDE
    sns.histplot(data=token_counts, bins=bins, kde=True)
    plt.title(f'Token Length Distribution for {text_column}')
    plt.xlabel('Token Count')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

    # 상세 통계 출력
    print("\n=== 토큰 수 통계 ===")
    for key, value in stats.items():
        print(f"{key}: {value:.1f}")

    return stats

print(analyze_token_distribution(data['train']))
Instruction Tuning을 수행합니다.    
이번에는 길이가 짧으므로, 전처리를 추가로 하지 않습니다.
from trl import SFTTrainer, SFTConfig
from trl import DataCollatorForCompletionOnlyLM
from accelerate import Accelerator

# GPU를 적게 쓰는 Gradient Checkpointing
model.gradient_checkpointing_enable()
# Checkpointing을 수행하는 경우 use_cache = False
model.config.use_cache = False

tokenizer.pad_token = tokenizer.eos_token


sft_config = SFTConfig(
    report_to='tensorboard',

    ######## 추가된 부분 ##############
    eval_strategy="steps",
    eval_steps=100,
    ## 100 step마다 검증 데이터를 통한 로스 계산
    save_total_limit=3,          # 최대 3개 체크포인트만 유지
    load_best_model_at_end=True, # 학습 끝나면 최고 모델 로드
    metric_for_best_model="eval_loss",  # 최고 모델 기준은 Evaluation Loss 기준
    #################################

    # Callback을 통해 중간 과정을 체크할 수도 있음
    # 테스트 문제 넣고 출력 확인하거나, 다른 LLM으로 평가하거나, ...

    num_train_epochs=3,

    dataset_text_field="text",
    # dataset 'text' 필드를 사용

    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,

    max_seq_length=800,
    lr_scheduler_type='cosine',

    learning_rate=2e-5,
    # Instruction Tuning은 더 낮은 LR
    warmup_ratio=0.03,

    bf16=True,

    optim="paged_adamw_8bit",

    output_dir="outputs",
    logging_steps=100,


    save_steps=100
    # 추가: 100 step마다 체크포인트 저장

)
Data Collator는 Trainer에 넣을 데이터를 전달합니다.   
Instruction Tuning의 경우, 출력 부분부터 학습시켜도 됩니다.   

collator를 전달하지 않는 경우, CPT처럼 전체 토큰을 예측합니다.
response_template = "<start_of_turn>model"
# 답변 부분만 학습시키기

collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
# Data Collator : 데이터를 어떻게 전달할 것인가?
# Next Token Prediction의 방법은 동일하나, 어디서부터 학습시킬 것인지를 결정
# 질문 내용까지 학습하기 VS 답변만 학습하기
# DataCollator를 전달하지 않으면, 전체 텍스트를 그대로 학습
학습을 수행합니다.
Instruction Data는 데이터가 더 짧으므로, Batch를 늘려도 될 것 같습니다.   
accelerator = Accelerator()

trainer = SFTTrainer(
    model=model,
    train_dataset=data['train'],

    ## 추가된 부분
    eval_dataset=data['test'],
    ## 검증데이터 활용

    args=sft_config,
    data_collator=collator
)

with accelerator.main_process_first():
    trainer.train()

# Loss Function : Cross Entropy
# 입력 데이터(배치)에 대한 평균 예측 확률 : e^(-Loss)
# Ex) 0.2 --> 81% (e^-0.2)
Training Loss과 Validation Loss는 처음에는 함께 감소하나   
과적합이 발생하면 Validation Loss가 점점 증가하게 됩니다.   
`metric_for_best_model="eval_loss"을 통해 체크포인트 중 가장 val loss가 낮은 가중치를 로드합니다.
model.eval()
torch.cuda.empty_cache()
만든 모델을 평가합니다.
from transformers import pipeline

# 파인 튜닝 모델이므로, 공식 파리미터를 따를 필요는 없음

gen_config = dict(
    do_sample=True,
    max_new_tokens=1024,
    repetition_penalty = 1.1,
    temperature = 0.1,
    top_p = 0.95,
    top_k = 64,

)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, return_full_text=False,
                **gen_config)
만들어진 모델은, 랭체인을 이용해 실행해 봅시다.
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace

base_llm = HuggingFacePipeline(pipeline=pipe, pipeline_kwargs=gen_config)

llm = ChatHuggingFace(llm=base_llm, tokenizer=tokenizer)
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate([('user', '{question}\n')])

chain = prompt | llm | StrOutputParser()
inputs = [
    "제 1형 당뇨병은 어떻게 치료하나요?",
    "고혈압인데 먹으면 안되는 거 있나요?",
    "충치가 너무 아파요.",
    "불안장애 환자의 식이는 어떻게 해야 되나요?"
]
for i in inputs:
    prompt = convert_format(i)['text']
    print(prompt)
    for s in base_llm.stream(prompt):
        print(s,end='')
    print('\n--------------------\n')
model_name
model_save_dir = model_name+'-Instruct'

model.save_pretrained(model_save_dir, safe_serialization=False)
tokenizer.save_pretrained(model_save_dir)
from huggingface_hub import login
import locale
locale.getpreferredencoding = lambda: "UTF-8"

login('')
# Write 권한 토큰 필요
# username=''

# # # 개인 계정 주소에 업로드하기
# model.push_to_hub(f'{username}/{model_save_dir}_0904')
# tokenizer.push_to_hub(f'{username}/{model_save_dir}_0904')
현재 모델은 학습을 충분히 하지 않았으므로, 답변의 품질이 유동적입니다.
inputs = [
    "감기약 종류 좀 알려주세요.",
]*10
for i in inputs:
    print('Question:', i)
    print('Answer:', chain.invoke(i))
    print('---------')

## Catastrophic Forgetting과 일반화

의료 데이터가 아닌 데이터를 일부 학습했기 때문에, 일반적인 질문에 대해서도 답변할 수 있습니다.   
편향된 데이터만으로 풀 파인 튜닝을 하는 경우에는 기존의 지식을 잘 출력하지 못하게 됩니다.

inputs = [
    "사과는 왜 빨간가요?",
    "저는 사과 12개가 있었는데, 친구에게 절반을 나눠줬어요. 남은 사과는 몇 개인가요?",
    "곰과 사자 중 누가 더 큰가요?",
]
for i in inputs:
    print('Question:', i)
    print('Answer:', chain.invoke(i))
코랩에서 실행하시는 경우, `런타임 --> 세션 다시 시작`   
`런타임 --> 런타임 연결 해제 및 삭제`   
를 실행해야 무료 사용량을 절약할 수 있습니다.