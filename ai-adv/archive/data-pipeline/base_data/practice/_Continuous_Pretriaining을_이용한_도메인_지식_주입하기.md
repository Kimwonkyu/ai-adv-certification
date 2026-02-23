# [실습] Continuous Pretraining

Continuous Pretraining은 Pretrain 모델에 새로운 코퍼스를 학습시키는 과정입니다.   

의료 도메인의 코퍼스에 대해, sLLM을 학습시켜 보겠습니다.   
!pip install transformers==4.56.0 tensorboard seaborn langchain langchain-huggingface pandas accelerate datasets huggingface_hub trl==0.19.1 bitsandbytes -q
# !pip install flash-attn --no-build-isolation -q
허깅페이스 토큰을 입력합니다.
import os
from huggingface_hub import login

# 허깅페이스 토큰 로그인: Llama, DeepSeek 의 경우 로그인 필수, Qwen은 로그인 필요없음
login(token='hf_REDACTED')
모델의 주소를 입력합니다.
from transformers import AutoModelForCausalLM,AutoTokenizer

model_id="unsloth/gemma-3-1b-pt" # 모델의 주소
model_name = model_id.split('/')[1]
print("## MODEL:", model_name)
모델과 토크나이저를 불러옵니다.
import torch

torch.set_float32_matmul_precision('high')
# Torch 기본설정


tokenizer = AutoTokenizer.from_pretrained(
    model_id,
)
model = AutoModelForCausalLM.from_pretrained(model_id,
                                             torch_dtype='auto',
                                             device_map="auto",
                                            attn_implementation="eager",
                                            )
모델 기본 상태 확인을 위해 pipeline을 구성하고 실행해 보겠습니다.
from transformers import pipeline

gen_config = dict(
    do_sample=True,
    max_new_tokens=512,
    temperature = 0.7,
    top_p = 0.95,
    top_k = 64,

    repetition_penalty=1.05
)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, return_full_text=True,
                **gen_config)
from langchain_huggingface import HuggingFacePipeline
base_llm = HuggingFacePipeline(pipeline=pipe)
예시 데이터에 대한 출력을 수행합니다.
inputs = [
    "질문: 고혈압의 약물 치료는 어떻게 해야 하나요? \n답변:",
    "질문: 축농증의 수술이 필요한 경우는 언제인가요? \n답변:",
    "질문: 불안장애 환자의 식이의 특징은 무엇입니까? \n답변:",
    "질문: 갑상선 기능 항진증 치료에 사용하는 대표적인 약물들은? \n답변:"
]

for i in range(len(inputs)):
    # print('Question:', inputs[i])
    print(f'{inputs[i]}', end='')
    for s in base_llm.stream(inputs[i]):
        print(s,end='')

    print('\n-------')
학습에 사용할 데이터를 불러옵니다.
from datasets import load_dataset
import os

file_path = ['medical_corpus.json']

data = load_dataset("json", data_files={"train":file_path})

data = data.shuffle()
data
data['train'][0]['content']
학습 데이터의 길이와 분포를 분석합니다.  
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter

def analyze_token_distribution(dataset, text_column='content', bins=30, tokenized=False):

    # 토큰 수 계산
    token_counts = []
    if not tokenized:

        for text in dataset[text_column]:
            tokens = tokenizer.encode(text)
            token_counts.append(len(tokens))
    else:
        for tokens in dataset['input_ids']:
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
`max_seq_length`를 설정합니다.

학습 데이터를 모두 포함하도록 길이를 설정하면 GPU 소모가 증가하므로   

적절한 크기로 max_seq를 줄일 수 있습니다   

<br><br>
최대 길이보다 긴 길이의 텍스트가 많은 경우, 아래의 작업을 고려합니다:   

1) 최대 토큰 기준으로 청킹하기(오버랩 고려)   

2) 최대 토큰 이후 값은 무시하기 (전처리 필요 없음)
max_seq_length = 1000
text_field_name = "content"
overlap_size = 500

# --- 오버랩을 적용하여 텍스트를 자르는 함수 ---
def chunk_text_with_overlap(examples, tokenizer, max_seq_length, text_field, overlap):
    """
    데이터셋의 각 텍스트를 토큰화하고 오버랩을 적용하여 max_seq_length 길이의 청크로 나눕니다.
    """
    if overlap >= max_seq_length:
        raise ValueError("Overlap size must be smaller than max_seq_length")

    stride = max_seq_length - overlap # 다음 청크 시작 위치 계산

    # 입력된 배치(examples)의 텍스트 필드에서 모든 텍스트를 가져옵니다.
    texts = examples[text_field]

    # 텍스트를 토큰화합니다.
    all_token_ids = []
    for text in texts:
        tokenized_output = tokenizer(text, add_special_tokens=False, truncation=False, padding=False)
        all_token_ids.append(tokenized_output['input_ids'])

    # 결과를 저장할 리스트 초기화
    chunked_input_ids = []
    chunked_attention_mask = []

    # 각 텍스트의 토큰 ID 리스트를 순회합니다.
    for token_ids in all_token_ids:
        # stride 간격으로 토큰 ID 리스트를 순회하며 청크를 만듭니다.
        for i in range(0, len(token_ids), stride):
            # 현재 위치부터 max_seq_length 만큼 토큰 ID를 잘라 청크를 만듭니다.
            # 마지막 청크는 max_seq_length보다 짧을 수 있습니다.
            chunk = token_ids[i : i + max_seq_length]

            # 생성된 청크와 그에 맞는 attention_mask(모두 1)를 결과 리스트에 추가합니다.
            # (주의: 청크 길이가 0인 경우 방지 - 매우 짧은 원본 텍스트 처리)
            if len(chunk) > 0:
                chunked_input_ids.append(chunk)
                chunked_attention_mask.append([1] * len(chunk))

    # map 함수가 처리할 수 있도록 딕셔너리 형태로 반환합니다.
    return {
        "input_ids": chunked_input_ids,
        "attention_mask": chunked_attention_mask,
    }

chunked_dataset_overlap = data['train'].map(
    chunk_text_with_overlap,
    batched=True,
    remove_columns=data['train'].column_names, # 기존 컬럼들 제거
    fn_kwargs={
        "tokenizer": tokenizer,
        "max_seq_length": max_seq_length,
        "text_field": text_field_name,
        "overlap": overlap_size
    }
)

# --- 결과 확인 (선택 사항) ---
print(f"Original dataset size: {len(data)}")
print(f"Chunked dataset size (with overlap): {len(chunked_dataset_overlap)}")
print(analyze_token_distribution(chunked_dataset_overlap, tokenized=True))
학습 파라미터를 설정합니다.   
from trl import SFTTrainer, SFTConfig

# GPU를 적게 쓰는 Gradient Checkpointing
model.gradient_checkpointing_enable()
# Checkpointing을 수행하는 경우 use_cache = False
model.config.use_cache = False

tokenizer.pad_token = tokenizer.eos_token
# 패딩: 배치 데이터들의 길이를 맞추는 작업
# 패딩을 위한 토큰을 eos 토큰으로 선택

sft_config = SFTConfig(
    report_to='tensorboard',
    # 텐서보드를 통해 학습 확인

    # evaluation_strategy="steps",
    # # logging_step당 validation loss 계산하는 옵션

    # 학습 방식: Epochs 단위 or Step 단위
    # Epoch: 전체 데이터를 한 번 학습하는 단위
    # Step: 배치만큼의 데이터를 학습하는 단위

    # batch * max_steps / len(data) = 총 Epochs

    num_train_epochs=3,
    # max_steps= 1000,

    dataset_text_field="content",
    # dataset 'text' 필드를 사용

    per_device_train_batch_size=2,
    # GPU당 데이터 2개씩 입력
    # GPU 사용량에 매우 큰 영향

    gradient_accumulation_steps=4,
    # 실행 속도에 매우 큰 영향
    # (Batch 대용) 그래디언트를 모아서 반영하는 스텝 수
    # 배치사이즈를 키우는 것에 비해 메모리 소모가 감소하나, 속도가 느려짐
    # 실질적으로 Batch 8의 효과

    # LLM에서는 일반적으로 1~10 사이의 Epoch 수행
    # Loss는 Epoch이 바뀔 때마다 크게 떨어짐, Val loss를 통해 오버피팅 방지

    max_seq_length=max_seq_length, # 예제에서는 1800

    lr_scheduler_type='cosine',
    # 학습률을 cosine 형태로 점진적 감소
    learning_rate=5e-5,
    # 학습률: 딥 러닝에서 오차를 이용해 파라미터를 변화시키는 속도
    # LLM 기본 Pretrain 과정보다는 낮은 학습률로 천천히 학습

    warmup_ratio=0.03,  # 전체 학습의 3%를 웜업에 사용
    # 웜업이란: 학습률을 초기에 매우 천천히 증가시키는 과정

    bf16=True, # bfloat16 모델이므로 bf16 설정

    optim="paged_adamw_8bit",
    # 예제에서는 GPU 소모를 최소화하는 Optimization 방법 선택 (CPU 오프로딩 포함)

    output_dir="outputs",
    logging_steps=25, # 손실함수 출력

    # save_steps=50
    # 체크포인트 저장
)
작성한 구성에 맞게 학습을 수행합니다.   
from accelerate import Accelerator

accelerator = Accelerator()
trainer = SFTTrainer(
    model=model,
    train_dataset=chunked_dataset_overlap,
    # # eval 데이터셋을 사용하고 싶은 경우 아래 주석 해제
    # eval_dataset=data['test'],
    args=sft_config,
)

with accelerator.main_process_first():
    trainer.train()

# Loss Function : Cross Entropy
# 입력 데이터(배치)에 대한 평균 예측 확률 : e^(-Loss)
# Ex) 0.2 --> 81% (e^-0.2)
위 작업의 `report_to='tensorboard'`는 tensorboard를 통해 학습 과정을 기록하게 됩니다.
터미널에서 아래 커맨드를 실행하면 확인할 수 있습니다.  
(2025 포트: 사전 설정으로 오픈)
```
tensorboard --logdir outputs/runs --port 2025 --bind_all
```
학습이 끝나면, 모델을 evaluation 모드로 바꾸고 캐시를 비웁니다.
model.eval()
torch.cuda.empty_cache()
gen_config = dict(
    do_sample=True,
    max_new_tokens=512,
    temperature = 0.2,
    top_p = 0.95,
    top_k = 64,

    repetition_penalty=1.05
)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, return_full_text=True,
                **gen_config)

base_llm = HuggingFacePipeline(pipeline=pipe)

inputs = [
    "질문: 고혈압의 약물 치료는 어떻게 해야 하나요? \n답변:",
    "질문: 축농증의 수술이 필요한 경우는 언제인가요? \n답변:",
    "질문: 불안장애 환자의 식이의 특징은 무엇입니까? \n답변:",
    "질문: 갑상선 기능 항진증 치료에 사용하는 대표적인 약물들은? \n답변:"
]

for i in range(len(inputs)):
    # print('Question:', inputs[i])
    print(f'{inputs[i]}', end='')
    for s in base_llm.stream(inputs[i]):
        print(s,end='')

    print('\n-------')
학습을 마친 모델은 로컬에 저장하거나, HF 저장소에 업로드할 수 있습니다.
model_name
model_save_dir = model_name+'-MED'

model.save_pretrained(model_save_dir, safe_serialization=False)
tokenizer.save_pretrained(model_save_dir)
허깅페이스 저장을 위해, Write 권한 토큰을 입력합니다.
# from huggingface_hub import login
# import locale
# locale.getpreferredencoding = lambda: "UTF-8"

# login('')
# # Write 권한 토큰 필요

# username=''

# # 개인 계정 주소에 업로드하기
# model.push_to_hub(f'{username}/{model_save_dir}_0904')
# tokenizer.push_to_hub(f'{username}/{model_save_dir}_0904')
코랩에서 실행하시는 경우, `런타임 --> 세션 다시 시작`   
`런타임 --> 런타임 연결 해제 및 삭제`   
를 실행해야 무료 사용량을 절약할 수 있습니다.
