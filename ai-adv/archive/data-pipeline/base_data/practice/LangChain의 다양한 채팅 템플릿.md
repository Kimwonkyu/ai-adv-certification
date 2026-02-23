# [실습] LangChain의 다양한 채팅 템플릿


실제 어플리케이션 환경에서는 프롬프트를 매번 다시 쓰는 대신,  
프롬프트의 템플릿을 구성하고, 입력 변수의 공간을 할당하여 일관성 있는 입력을 전달합니다.  


LangChain은 다양한 형태의 프롬프트 템플릿 기능을 지원합니다.
## 실습 환경 설정
기본 라이브러리를 설치합니다.
!pip install rich pandas langchain_community openai langchain langchain_openai langchain_google_genai -q
환경 변수 파일을 불러옵니다.
import os
from dotenv import load_dotenv
# OPENAI_API_KEY, GOOGLE_API_KEY
load_dotenv(override=True)
LLM을 정의합니다.   
API 키가 필요하므로, 실습 환경에 맞게 선택해 주세요.
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.rate_limiters import InMemoryRateLimiter

# Gemini: 무료 API 사용량 존재
# 안정적 서빙을 위해 분당 10개 설정
# 즉, 초당 약 0.167개 요청 (10/60)
# `https://aistudio.google.com/`에서 모델별 사용량 확인

rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.167,  # 분당 10개 요청
    check_every_n_seconds=0.1,  # 100ms마다 체크
    max_bucket_size=10,  # 최대 버스트 크기
)


# LLM 초기화
llm = ChatOpenAI(model='gpt-5-mini', temperature=0.3, max_tokens=8192)
llm_gemini = ChatGoogleGenerativeAI(model='gemini-2.5-flash', temperature=0.7, max_tokens=8192, 
                                rate_limiter=rate_limiter)

response = llm_gemini.invoke("안녕?")
response
from rich import print as rprint
rprint(response)
## LangChain의 Prompt

랭체인에서 LLM에 프롬프트를 입력하는 방법은 아래 3가지가 있습니다.

1. 문자열 그대로 입력하기 (이전 방법과 동일하게)
2. 메시지 클래스 리스트 입력하기
3. 프롬프트 템플릿 구성하기
### 1) 문자열 그대로 입력하기   

문자열을 `invoke()`를 통해 전달합니다.   
prompt = '''
역사상 가장 매력적이었던 악역이 나오는 영화 5개를 알려주세요.
주연 배우와 명대사, 매력적인 이유도 한 줄로 설명하세요.
출력의 마지막에는 전체 내용을 표로 표현하세요.'''

result = llm.stream(prompt)
for response in result:
    print(response.content, end='')

result = llm_gemini.stream(prompt)
for response in result:
    print(response.content, end='')
### 2) Message 클래스 전달하기   
클래스를 직접 생성하고 전달합니다.  
`HumanMessage, SystemMessage, AIMessage` 클래스의 리스트를 전달하면 됩니다.
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

messages=[]

messages.append(SystemMessage('당신은 매우 창의력이 뛰어납니다.'))
messages.append(HumanMessage('엔비디아와 XAI가 합병한다면 어떤 이름이 좋을까요?'))

result = llm.invoke(messages)
rprint(result)

AIMessage를 함께 전달하는 방식으로, 멀티-턴 대화를 수행할 수 있습니다.
messages.append(result) # AIMessage 추가
messages.append(HumanMessage('이 중에서 당신이 꼽은 가장 좋은 이름은 뭔가요?'))

result = llm.invoke(messages)
rprint(result)
### 3. Prompt Template


from langchain.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate, FewShotPromptTemplate, FewShotChatMessagePromptTemplate
PromptTemplate을 이용하여, 프롬프트의 기본적인 형태를 만들 수 있습니다.   

prompt = PromptTemplate(
    template='''
당신은 OpenAI에 경쟁적인 Google을 대표하며,
논문 설명의 전문가입니다. 주어진 논문이나 기술에 대해서 매우 유머러스하고 통찰력이 있지만 때론 비판적인
5문장 길이의 브리핑을 작성해 주세요.
이모지를 많이 추가하세요.
---

논문: {article}

'''
)

print(prompt.format(article='트랜스포머 디코더 기반의 모델인 GPT-1'))
result = llm_gemini.invoke(prompt.format(article='트랜스포머 디코더 기반의 모델인 GPT-1'))

rprint(result)

두 개의 매개변수를 받아 프롬프트를 만들어 보겠습니다.
prompt = PromptTemplate(
    template='''
주어진 문제에 대해, 문제를 체계적으로 해결하는 Step-by-Step의 단계별 과정을 설명하세요.
각 과정은 4문장에서 5문장으로 하세요.

그리고 마지막에 최종 결론을 출력하세요.

---

문제: {problem}
과정의 단계 수: {steps}

'''
)

example = {'problem':'숙면 취하기', 'steps':6}

print(prompt.format(**example))
result = llm.invoke(prompt.format(**example))
rprint(result)
일부만 먼저 포함하는 것도 가능합니다.
prompt = PromptTemplate(
    template='''
주어진 문제에 대해, 문제를 체계적으로 해결하는 Step-by-Step의 단계별 과정을 설명하세요.
각 과정은 4문장에서 5문장으로 하세요.

그리고 마지막에 최종 결론을 출력하세요.

---

문제: {problem}
과정의 단계 수: {steps}

'''
).partial(steps = 10)

example = {'problem':'건강한 아침식사 하기'}

print(prompt.format(**example))

result = llm_gemini.invoke(prompt.format(**example))

print(result.content)
LangChain의 구성 요소들은 batch()를 통해 여러 개 실행할 수도 있습니다.
messages = [
    '안녕? 한국어로 대답해줘',
    'Hello?',
    '너는 이름이 뭐니? Answer in Spanish',
]

results = llm.batch(messages)

results

## ChatPromptTemplate

ChatPromptTemplate은 유저 이외의 다른 역할을 추가합니다.  


최근에는 GPT와, Claude의 WebUI에 포함된 자체 시스템 프롬프트가 많은 관심을 받고 있습니다.   

- GPT의 시스템 프롬프트: https://chatgpt.com/share/68a13d0f-85b0-8006-9ef6-5577796f5989

- Claude의 시스템 프롬프트: https://docs.anthropic.com/en/release-notes/system-prompts#august-5-2025
 시스템 프롬프트는 주로 유저 프롬프트보다 높은 우선순위로 더 긴 Context에 영향을 줍니다.
example = llm.invoke("구체적인 시간 장소가 적혀서, 추후 메모해야 할 것 같은 회의록을 500자 이내로 생성해주세요.").content
print(example)
prompt = ChatPromptTemplate(
    [
        ('system', '당신은 주어진 정보에서 시간과 장소를 추출해야 합니다. 형식은 json으로 출력하세요.'),
        ('human','{article}')        
    ]
)

prompt.format_messages(article=example)

response = llm.invoke(prompt.format_messages(article=example))
print(response.content)


## Few-Shot Prompting
Few-Shot Prompt Template을 이용하면, 예시를 쉽게 찾을 수 있습니다.
questions = [
    '당신은 어떤 모델입니까?',
    '인공지능 배워야 하나요?',
    '파이썬은 뭐가 좋나요?'
]

prompt = ChatPromptTemplate(['다음 질문에 대해서 10자 이내로 아주 간결하고 건조하게 설명해주세요. \n{question}'])

llm.invoke(prompt.format(question=questions[2]))

examples =[

    {'question': '당신은 어떤 모델입니까?', 'answer': '언어모델'},
    {'question': '인공지능 배워야 하나요?', 'answer': '권장됨'},
    {'question': '파이썬은 뭐가 좋나요?', 'answer': '문법간결'},
]

example_prompt = PromptTemplate(template='질문: {question}\n답변: {answer}')

fewshotprompt = FewShotPromptTemplate(
    examples = examples,
    example_prompt = example_prompt,

    prefix ='주어진 예시 형식을 참고하여 답변하세요.',
    suffix = '질문: {question} \n답변:'
)

test = fewshotprompt.format(question = '원래 말투가 그래?')
print(test)

print(llm.invoke(test).content)

Chat Message에 Few Shot을 적용하는 경우, 아래와 같이 나타납니다.
from langchain.prompts.few_shot import FewShotChatMessagePromptTemplate

example_prompt = ChatPromptTemplate(
        [
        ("human", "{question}"),
        ("ai", "{answer}"),
    ]
)
few_shot_prompt = FewShotChatMessagePromptTemplate(

    example_prompt=example_prompt,
    examples=examples,
)

few_shot_prompt.format_messages()

final_prompt = ChatPromptTemplate(
    [
        ("system", "다음 예시를 참고하여 같은 형식으로 답변하세요."),
        few_shot_prompt,
        ("human", "{input}"),
    ]
)
final_prompt.format_messages(input="휴가 신청은 어떻게 해야 하나요?")
### 멀티모달 프롬프트 전달하기

멀티모달 모델의 입력은 OpenAI API와 동일합니다.
import base64
import httpx

# Test 이미지 URL
image_url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/c/c1/Fawn_pug_2.5year-old.JPG/330px-Fawn_pug_2.5year-old.JPG'

response = httpx.get(image_url)

# dog.jpg에 저장
with open('dog.jpeg', 'wb') as file:
    file.write(response.content)
prompt = ChatPromptTemplate(
    [
        ('human', [
            {'type':'image', 'image_url':'{image_url}'},
            {'type':'text', 'text':'이 그림을 자세히 묘사하세요'}
                   
                   ])
    ]
)
llm.invoke(prompt.format_messages(image_url = image_url))




# 로컬 이미지 전달
with open('./dog.jpeg', 'rb') as image_file:
    image_data = base64.b64encode(image_file.read()).decode('utf-8')


image_prompt = ChatPromptTemplate([
    ('user',[
                {"type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,{image_data}"}
                },
                {"type": "text", "text": "{question}"},
             ]
     )])

X = llm.invoke(image_prompt.format_messages(
    question='이 사진을 자세히 묘사해주세요. 강아지의 품종은 설명하지 마세요.',
    image_data=image_data))

print(X.content)