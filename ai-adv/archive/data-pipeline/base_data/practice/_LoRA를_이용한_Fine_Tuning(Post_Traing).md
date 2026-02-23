# [실습] LoRA를 이용한 Style Tuning: RAFT

이번 실습에서는 LoRA 기반의 파인 튜닝을 수행합니다.  

LoRA 학습은 주로 Instruction 형태의 데이터 학습에 적합하며,    
풀 파인 튜닝 대비 적은 파라미터로 파인 튜닝을 수행할 수 있습니다.   

또한, GPU가 매우 부족한 경우에는       
베이스모델을 양자화하여 LoRA 학습을 수행하는 QLoRA 방법을 고려할 수 있습니다.
!pip install transformers==4.56.0 peft tensorboard seaborn langchain langchain-huggingface pandas accelerate datasets huggingface_hub trl==0.19.1 bitsandbytes -q
허깅페이스 토큰을 입력합니다.
import os
from huggingface_hub import login

# login(token='')
모델의 주소를 입력합니다.    
이번에는 1B보다 큰 모델을 양자화하여 불러오겠습니다.
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig

torch.set_float32_matmul_precision('high')
# Torch 기본설정


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)


model_id="Qwen/Qwen3-4B-Instruct-2507"


model_name = model_id.split('/')[1]
print("## MODEL:", model_name)
모델과 토크나이저를 불러옵니다.
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
)
model = AutoModelForCausalLM.from_pretrained(model_id,
                                             torch_dtype='auto',
                                             quantization_config=bnb_config,
                                             device_map="auto",
                                             attn_implementation="eager",

                                            )
학습 데이터를 불러옵니다.
file_path = 'RAG_Data_full.csv'
import pandas as pd
pd.read_csv(file_path, encoding='cp949').head()
from datasets import load_dataset
import os

file_path = ['RAG_Data_full.csv', 'RAG_Data_full_neg.csv']

data = load_dataset("csv",encoding='cp949',
                    data_files={"train":file_path})

# train_test split 나누기
# data = load_dataset("csv",
#                     data_files={"train":file_path}, split='train').train_test_split(0.1)


test_context = data['train'][0]['context']
test_question = data['train'][0]['question']
test_answer = data['train'][0]['cot']

print('Context:', test_context)
print('Question:', test_question)
print('Answer:', test_answer)



data = data.shuffle(seed=1234)

# data['train'] = data['train'].shard(num_shards=3, index=0)
# 단축하여 240개만 선택하는 경우

data
학습 데이터를 수행하기 전, 프롬프트만 넣었을 때의 결과를 확인합니다.
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline

gen_config = dict(
    do_sample=True,
    max_new_tokens=1024,
    top_p = 0.95,
    top_k = 64
)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, return_full_text=True,
                **gen_config)


base_llm = HuggingFacePipeline(pipeline=pipe, pipeline_kwargs=gen_config)


def convert_format(context,question,answer=None, add_generation_prompt=False):

    chat = [
    {'role': 'system',
     'content': """너는 무척 거만한 AI야. 사용자가 물어본 [Question]에 대해 주어진 [Context]를 참고해서 반말로 대답해.
정답을 알고 있다면, 대답은 무조건 '그것도 몰라?'로 시작해야 해.
그 뒤에 [Context]에서 관련 있는 부분을 '여기' 로 인용하면서 설명해.
마지막에는 거만하고 무례하게 요약해.
모르는 경우에는 '내가 그딴 걸 어떻게 알아?'라고만 대답해.
"""},
    {'role': 'user',
     'content': f"Context: {context}\n"
     "---\n"
     f"Question: {question}"}
]
    if answer:
        chat.append(
            {'role':'assistant',
             'content':f"{answer}"}
        )
    return {'text':tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=add_generation_prompt)}

test_prompt = convert_format(test_context,test_question, add_generation_prompt=True)['text']
print(test_prompt)
response = base_llm.stream(test_prompt)

for s in response:
    print(s, end='')

이제 전체 데이터를 학습에 적합한 형태로 변환합니다.
# 프롬프트 엔지니어링만 한 경우
def convert_format_without_pe(context,question,answer=None, add_generation_prompt=False):

    chat = [
    {'role': 'system',
     'content': """당신은 Rude-RAG Bot입니다. 주어진 [Context]를 참고하여, [Question]에 거만하게 대답하세요."""},
    {'role': 'user',
     'content': f"Context: {context}\n"
     "---\n"
     f"Question: {question}"}
]
    if answer:
        chat.append(
            {'role':'assistant',
             'content':f"{answer}"}
        )
    text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=add_generation_prompt)
    text = text.replace('<think>\n\n</think>\n\n','')
    # Qwen Instruct Model은 think 미사용으로 제거하기
    return {'text':text}

data = data.map(lambda x:convert_format_without_pe(x['context'],x['question'], x['cot']))
data['train']['text'][1]
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
## PEFT(Prompt-Efficient Fine Tuning)로 학습하기   
전체 파인 튜닝을 하지 않고도, PEFT를 사용하면 파라미터의 수를 줄인 효과적인 튜닝이 가능합니다.
model
모델마다 LoRa를 적용하는 레이어가 달라질 수 있습니다.   
model의 출력 결과에서 모델의 구성 요소를 확인하여 LoRA 적용 레이어를 결정합니다.
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model

# GPU를 적게 쓰는 Gradient Checkpointing
model.gradient_checkpointing_enable()
# Checkpointing을 수행하는 경우 use_cache = False
model.config.use_cache = False
model = prepare_model_for_kbit_training(model)

config = LoraConfig(
    r = 4, # 4차원 (d-->r) : 보통은 8이나 16
    lora_alpha=8, # (보통은 r과 함께 (8,16) 또는 (16,32))
    target_modules=[
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    'up_proj',
    'down_proj',
    'gate_proj'],
    # LoRA를 어떤 모듈에 부착할 것인가? (경험적)

    # Continuous Pretraining : embed_tokens 과 lm_head까지 부착 (메모리 소모 증가)
    # Ref) https://unsloth.ai/blog/contpretraining

    lora_dropout=0,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)
PEFT 모듈이 탑재된 모델은 구조가 바뀝니다.
model
# 학습에 참여할 파라미터 수 출력
model.print_trainable_parameters()
SFTTrainer를 통해 학습을 준비합니다.
from trl import SFTTrainer, SFTConfig
from trl import DataCollatorForCompletionOnlyLM

tokenizer.pad_token = tokenizer.eos_token

sft_config = SFTConfig(
    report_to='none',

    #max_steps= 150,
    num_train_epochs = 1,

    dataset_text_field="text",

    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,

    max_seq_length=1000,
    lr_scheduler_type='cosine',

    learning_rate=1e-4,
    # LoRA 학습률은 더 높아도 됨
    warmup_ratio=0.03,

    bf16=True,

    optim="paged_adamw_8bit",
    output_dir="outputs",
    logging_steps=25,
    # 손실함수 출력

    # save_steps=50
    # 체크포인트 저장
)

response_template = "<|im_start|>assistant"
# 답변 부분만 학습시키기

collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
# 질문 내용까지 학습하기 VS 답변만 학습하기
# DataCollator를 전달하지 않으면, 전체 텍스트를 그대로 학습
학습을 진행합니다.   
형식을 바꾸는 간단한 학습이므로, 1 Epoch만 학습시켜 보겠습니다
from trl import DataCollatorForCompletionOnlyLM
from accelerate import Accelerator

accelerator = Accelerator()

trainer = SFTTrainer(
    model=model,
    train_dataset=data['train'],

    args=sft_config,
    data_collator=collator
)

with accelerator.main_process_first():
    trainer.train()

# Loss Function : Cross Entropy
# 입력 데이터(배치)에 대한 평균 예측 확률 : e^(-Loss)
# Ex) 0.2 --> 81% (e^-0.2)
## 학습 결과 확인하기
학습을 완료했으니, 어댑터를 저장합니다.
model.eval()
torch.cuda.empty_cache()
model_save_dir = model_name+'-Rude-LORA'

model.save_pretrained(model_save_dir)
tokenizer.save_pretrained(model_save_dir)
허깅페이스 계정에 어댑터만 저장할 수 있습니다.
from huggingface_hub import login
import locale
locale.getpreferredencoding = lambda: "UTF-8"

login('hf_REDACTED')
# Write 권한 토큰 필요
username='NotoriousH2'

# 개인 계정 주소에 업로드하기
model.push_to_hub(f'{username}/{model_save_dir}_Rude_LoRA')
tokenizer.push_to_hub(f'{username}/{model_save_dir}_Rude_LoRA')
# 모델 로드하기   
학습 직후의 모델을 바로 로드하거나, 양자화 이전 모델을 불러와 결합할 수 있습니다.
LoRA 모듈을 원래의 16비트 모델에 연결해 보겠습니다.
base_model = AutoModelForCausalLM.from_pretrained(model_id,
                                             torch_dtype='auto',
                                             device_map="auto",
                                             attn_implementation="eager",
                                            )
from peft import PeftModel

# LoRA 어댑터 로드
model = PeftModel.from_pretrained(
    base_model,
    model_save_dir,
    torch_dtype=torch.bfloat16
)
print("LoRA 모델 결합 완료!")

gen_config = dict(
    do_sample=True,
    max_new_tokens=1024,
    top_p = 0.95,
    top_k = 64,

    repetition_penalty = 1.10,
)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, return_full_text=True,
                **gen_config)


base_llm = HuggingFacePipeline(pipeline=pipe, pipeline_kwargs=gen_config)
test_context = '''참고래의 등 부분은 밤회색이며, 배 쪽은 하얗다.
튀어나온 두 쌍의 숨구멍이 있으며, 납작하고 넓은 주둥이를 가지고 있다.
두 개의 밝은 색 문양이 숨구멍 뒤에서 시작해 몸의 측면으로 따라가 꼬리로 이어진다.
오른쪽 턱에 하얀색 무늬가 있으며, 왼쪽은 회색 또는 검은색이다.
참고래는 턱에서 몸 밑의 중앙부까지 이어지는 56에서 100개의 주름을 지니고 있는데,
먹이를 잡을 때 목을 팽창시키기 쉽게 하기 위한 것이다.
이들의 등지느러미의 길이는 60센티미터 정도이다.
가슴지느러미는 아주 작으며, 꼬리는 넓고 V자 모양이며 끝은 뾰족한 편이다.'''

test_question = '''참고래의 주름은 어떤 용도인가요? 등지느러미는 몇 CM인가요?'''

test_prompt = convert_format_without_pe(test_context,test_question, add_generation_prompt=True)['text']

print(test_prompt)
for s in base_llm.stream(test_prompt):
    print(s, end='')
test_context = '''
문서명) 르윈 디아즈/플레이 스타일

신장 193cm의 건장한 체구에서 나오는 빠르고 강한 스윙으로 타구 스피드를 올리는 생산하는 풀스윙 히터다. 이전에 삼성에서 활약했던 다린 러프와 호세 피렐라가 장타 향상을 위해 그랬듯이 공을 띄우는 타격을 하지만, 배트 스피드가 매우 빨라서 각도가 낮은 타구도 내야를 뚫고 안타가 된다.
당기는 어퍼 스윙을 하기 때문에 몸쪽 낮은 공에 매우 강하다. 평범한 몸쪽 공을 쪼개버릴듯한 풀 스윙으로 우측 담장을 넘겨버리는 것은 디아즈의 시그니처. 특히 라이온즈 파크일 경우 디아즈 상대로 몸쪽 애매한 볼은 거의 실점과 동의어이다. 다만 기본적으로는 풀히터임에도 불구하고 밀어치는 타구 역시 적지 않으며, 좌완 투수에게 크게 약한 모습을 보여주지 않는다.

반면 선구안은 확실히 좋지 않다. 볼삼비와 타출갭이 모두 낮은 편이고, 인-아웃존 스윙 비율 역시 리그 평균보다 꽤 아래이다. 또한 이재현과 비슷한 타입의 어퍼 스윙 히터라서 김영웅, 박병호 등 못지 않게 내야 뜬공 비율이 매우 높다. 볼넷보다는 안타를 추구하는 적극적인 타격을 하는데 내야 뜬공도 많다보니 타구 스피드는 총알같음에도 불구하고 BABIP는 리그 평균 아래를 마크한다. 그래서 컨디션이 좋을 땐 홈런을 안타 치듯이 양산하지만 나쁠 땐 홈런만 치는 공갈포가 된다. 그래도 수싸움 능력이 있어서 볼넷을 고르겠다는 마음을 먹으면 골라낼 수 있는 능력도 있다. 디아즈를 상대로 볼카운트가 불리해졌다고 스트라이크 존 한복판에 과감하게 공을 꽂아넣을 수 있는 간 큰 투수는 드물고, 그런 간 큰 투수가 등장한다면 공은 담장을 넘어간다.

결론적으로 KBO 수준에서 최상급인 파워와 준수한 컨택, 나쁜 선구안을 가진 타자로 불과 몇 년전에 삼성에서 뛴 호세 피렐라와 플레이 스타일이 매우 유사하다.[1] 또한 득점권에서 상당히 강한 모습을 보여 2025 시즌 현재 타점 부문에서 압도적인 선두를 달리고 있다.

현재 디아즈의 타점은 125타점으로, 만일 남은 25경기 동안 22타점 이상 기록한다면 같은 팀의 박병호가 2015년 달성한 단일 시즌 최다 타점(146점) 기록을 깰 수 있다. 또한 산술적으로 48홈런 페이스인데, 페이스대로 48홈런을 친다면 기존의 2015년 야마이코 나바로가 가지고 있는 KBO 역대 외인 타자 단일 시즌 최다 홈런 타이 기록을 달성할 수 있다. 2
'''

test_question = '''르윈 디아즈은 어떻게 타점을 많이 생산했나요?'''

test_prompt = convert_format_without_pe(test_context,test_question, add_generation_prompt=True)['text']

print(test_prompt)
for s in base_llm.stream(test_prompt):
    print(s, end='')
# OOD (Out-of-Distribution) 데이터라면 어떨까?

test_context = '''
The pursuit of artificial general intelligence (AGI) or artificial super intelligence (ASI) has long been a go
for humanity. Recent advancements in large foundation models, e.g., GPT-4o (OpenAI, 2024), Claude
3.7 (Anthropic, 2025), Gemini 2.5 (DeepMind, 2025), DeepSeek-V3 (Liu et al., 2024a), Llama-4 (Meta-AI,
2025), and Qwen2.5 (Yang et al., 2024b), have demonstrated significant progress toward this objective.
These models are trained on vast datasets spanning trillions of tokens across diverse domains and tasks,
effectively distilling human knowledge and capabilities into their parameters. Furthermore, recent
developments in reasoning models, optimized through reinforcement learning, highlight the potential
for foundation models to enhance inference-time scaling and achieve higher levels of intelligence, e.g.,
o3 (OpenAI, 2025), DeepSeek-R1 (Guo et al., 2025). While most state-of-the-art models remain proprietary,
the rapid growth of open-source communities has substantially reduced the performance gap between
open-weight and closed-source models. Notably, an increasing number of top-tier models (Meta-AI, 2025;
Liu et al., 2024a; Guo et al., 2025; Yang et al., 2024b) are now being released as open-source, fostering
broader research and innovation in artificial intelligence.
In this work, we introduce Qwen3, the latest series in our foundation model family, Qwen. Qwen3 is
a collection of open-weight large language models (LLMs) that achieve state-of-the-art performance
across a wide variety of tasks and domains. We release both dense and Mixture-of-Experts (MoE) models,
with the number of parameters ranging from 0.6 billion to 235 billion, to meet the needs of different
downstream applications. Notably, the flagship model, Qwen3-235B-A22B, is an MoE model with a
total of 235 billion parameters and 22 billion activated ones per token. This design ensures both high
performance and efficient inference.
Qwen3 introduces several key advancements to enhance its functionality and usability. First, it integrates
two distinct operating modes, thinking mode and non-thinking mode, into a single model. This allows
users to switch between these modes without alternating between different models, e.g., switching from
Qwen2.5 to QwQ (Qwen Team, 2024). This flexibility ensures that developers and users can adapt the
model’s behavior to suit specific tasks efficiently. Additionally, Qwen3 incorporates thinking budgets, providing users with fine-grained control over the level of reasoning effort applied by the model during task
execution. This capability is crucial to the optimization of computational resources and performance, tailoring the model’s thinking behavior to meet varying complexity in real-world applications. Furthermore,
Qwen3 has been pre-trained on 36 trillion tokens covering up to 119 languages and dialects, effectively
enhancing its multilingual capabilities. This broadened language support amplifies its potential for
deployment in global use cases and international applications. These advancements together establish
Qwen3 as a cutting-edge open-source large language model family, capable of effectively addressing
complex tasks across various domains and languages.
The pre-training process for Qwen3 utilizes a large-scale dataset consisting of approximately 36 trillion
tokens, curated to ensure linguistic and domain diversity. To efficiently expand the training data, we
employ a multi-modal approach: Qwen2.5-VL (Bai et al., 2025) is finetuned to extract text from extensive
PDF documents. We also generate synthetic data using domain-specific models: Qwen2.5-Math (Yang
et al., 2024c) for mathematical content and Qwen2.5-Coder (Hui et al., 2024) for code-related data. The
pre-training process follows a three-stage strategy. In the first stage, the model is trained on about 30
trillion tokens to build a strong foundation of general knowledge. In the second stage, it is further trained
on knowledge-intensive data to enhance reasoning abilities in areas like science, technology, engineering,
and mathematics (STEM) and coding. Finally, in the third stage, the model is trained on long-context
data to increase its maximum context length from 4,096 to 32,768 tokens.
To better align foundation models with human preferences and downstream applications, we employ a
multi-stage post-training approach that empowers both thinking (reasoning) and non-thinking modes. In
the first two stages, we focus on developing strong reasoning abilities through long chain-of-thought
(CoT) cold-start finetuning and reinforcement learning focusing on mathematics and coding tasks. In the
final two stages, we combine data with and without reasoning paths into a unified dataset for further
fine-tuning, enabling the model to handle both types of input effectively, and we then apply generaldomain reinforcement learning to improve performance across a wide range of downstream tasks. For
smaller models, we use strong-to-weak distillation, leveraging both off-policy and on-policy knowledge
transfer from larger models to enhance their capabilities. Distillation from advanced teacher models
significantly outperforms reinforcement learning in performance and training efficiency.
We evaluate both pre-trained and post-trained versions of our models across a comprehensive set of
benchmarks spanning multiple tasks and domains. Experimental results show that our base pre-trained
models achieve state-of-the-art performance. The post-trained models, whether in thinking or nonthinking mode, perform competitively against leading proprietary models and large mixture-of-experts
(MoE) models such as o1, o3-mini, and DeepSeek-V3. Notably, our models excel in coding, mathematics,
and agent-related tasks. For example, the flagship model Qwen3-235B-A22B achieves 85.7 on AIME’24
2and 81.5 on AIME’25 (AIME, 2025), 70.7 on LiveCodeBench v5 (Jain et al., 2024), 2,056 on CodeForces,
and 70.8 on BFCL v3 (Yan et al., 2024). In addition, other models in the Qwen3 series also show strong
performance relative to their size. Furthermore, we observe that increasing the thinking budget for
thinking tokens leads to a consistent improvement in the model’s performance across various tasks.
In the following sections, we describe the design of the model architecture, provide details on its training
procedures, present the experimental results of pre-trained and post-trained models, and finally, conclude
this technical report by summarizing the key findings and outlining potential directions for future
research
'''

test_question = '''What is the context size of Qwen 3?'''

test_prompt = convert_format_without_pe(test_context,test_question, add_generation_prompt=True)['text']

print(test_prompt)
for s in base_llm.stream(test_prompt):
    print(s, end='')
# 시스템 프롬프트를 넣지 않는 경우, 기존 능력이 유지됨

messages=[
    {'role':'user', 'content':"LoRA 파인튜닝이 뭔가요?"}]

general_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt = True)
print(general_prompt)

for s in base_llm.stream(general_prompt):
    print(s, end='')
## 어댑터와 원래 모델 결합하기   




merge_and_unload()를 통해 원래 모델에 어댑터의 가중치를 더할 수 있습니다.  

파라미터 수가 기존 모델과 동일하게 구성됩니다.
# LoRA를 베이스 모델에 병합
model = model.merge_and_unload()

# 병합된 모델 저장
merged_model_path = model_name + '-Rude-Merged'
model.save_pretrained(merged_model_path)
tokenizer.save_pretrained(merged_model_path)

print(f"병합된 모델이 {merged_model_path}에 저장되었습니다.")
