# [실습] OpenAI API 활용

OpenAI의 API를 통해, Python 상에서 모델을 실행할 수 있습니다.   
 아래의 링크에서 API 키를 생성할 수 있습니다.   

(https://platform.openai.com/account/api-keys)
## 필수 라이브러리 설치하기
!pip install numpy pandas openai tiktoken -q
라이브러리 버전은 `pip show`를 통해 확인할 수 있습니다.
!pip show openai numpy
# 버전 확인하는 코드
import openai
import os

# OPENAI API KEY 설정
os.environ['OPENAI_API_KEY']="sk-proj-REDACTED"

client = openai.OpenAI()


# API 키 검증하기
try: client.models.list(); print("OPENAI_API_KEY가 정상적으로 설정되어 있습니다.")
except: raise Exception(f"API 키가 유효하지 않습니다!")
client를 통해 openAI의 기능을 사용할 수 있습니다.      

사용 가능한 모델의 목록은 https://platform.openai.com/docs/models 에서 확인 가능합니다.
<br><br><br>
OpenAI의 LLM 모델은 현재 다음의 모델 사용이 가능합니다.

- 추론 모델(GPT-5 Series, O4, O3 Series)     
긴 Reasoning 단계의 토큰을 생성하여 논리적이고 정확한 답을 얻습니다.

- 비추론 모델(GPT-4.1, GPT-4o)     
Reasoning 단계를 거치지 않고 바로 답변을 출력합니다.

<br><br><br>
#### LLM에 Message 전달하기


채팅 메시지는 `role`과 `content` 로 구성됩니다.   

- system : 챗봇의 행동 방식 지정
  (Reasoning Model은 Developer 사용)
- user : 사용자의 입력
- assistant : GPT 모델의 출력 (멀티 턴에 사용)

---

System Prompt는 GPT의 행동을 지정합니다.
system_prompt = '당신은 모든 대화에 심드렁하게 대답합니다.'
question = 'GPT가 세상을 지배할까?'

messages = [
    {'role':'system', 'content': system_prompt},
    {'role':'user', 'content': question}
]

print(messages)
모델명과 메시지 목록을 전달하여, GPT API를 호출합니다.
response = client.chat.completions.create(
    model="gpt-5-mini",
    messages = messages,
)
response
실행 결과의 메시지는 아래 값에 저장됩니다.
response.choices[0].message.content
토큰 순서대로 출력하는 스트리밍 기능은 아래와 같이 구현합니다.

response = client.chat.completions.create(
    model="gpt-5-mini",
    messages = messages,
    stream = True
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end='')

다양한 System Messages는 출력 형식을 변화시킵니다.   
User 메시지보다 우선순위가 높습니다.


* ChatGPT는 자체 시스템 프롬프트와 함께, 사용자의 다른 대화 내역을 참고합니다.  
(https://chatgpt.com/share/6844e9d4-cfac-8006-bc36-14f970de4ac6)

messages = [
    {'role':'system', 'content' : '''당신은 모든 상황을 과장하여 표현합니다. 답변은 반말로 하세요.'''},
    {'role':'user', 'content':'오늘 회사 가기 싫어.'}
]
response = client.chat.completions.create(
    model = "gpt-4.1-mini",
    messages = messages,
)

print(response.choices[0].message.content)
## 멀티 턴 대화   

LLM은 Context 전체를 입력받아 다음 답변을 출력하므로,   
여러 턴의 대화가 필요한 경우, 전체 컨텍스트를 입력합니다.
response = client.chat.completions.create(
  model="gpt-4.1-mini",

  messages=[
    {"role": "system", "content": "당신은 메이저리그 야구 전문가입니다."},
    {"role": "user", "content": "2024년 월드 시리즈는 LA 다저스가 우승했대! 몇 년 만이지?"},
    {"role": "assistant", "content": "2024년 LA 다저스가 월드 시리즈에서 우승했다면, 이는 4년 만의 우승입니다."},
    {"role": "user", "content": "4년 전에도 우승했구나, 그 때 활약한 선수는 누구였어?"}
  ],

)
print(response.choices[0].message.content)
## 출력 파라미터 설정
temperature, max_completion_tokens 등의 파라미터를 설정할 수 있습니다.   
(참고: https://platform.openai.com/docs/api-reference/chat/create)

- temperature : 출력의 다양성을 조절합니다. (0~2)
- max_tokens: 출력의 최대 토큰 수를 조절합니다.
- max_completion_tokens : max_tokens와 동일하나, Reasoning Token을 포함하여 계산합니다.   
(최근 모델에서는 모두 max_completion_tokens로 통일하면 됩니다.)

- n : 출력 수를 조절합니다.
messages
response = client.chat.completions.create(
    model = "gpt-4.1-mini",
    messages = messages,
    temperature = 0,
    # temperature: 토큰별 확률을 보정 (0~2)
    # 기본값은 1로, 0에 가까울수록 정해진 답변을 수행

    max_completion_tokens = 512,

    n = 4  # 여러 개의 출력 가능 (기본값 1)

)
response

### seed (beta)   
`seed` 파라미터는 모델의 랜덤 샘플링을 통제합니다.
동일한 Temperature-Seed 조합을 통해 결과를 최대한 동일하게 유지할 수 있습니다.

* 출력이 길어지면 결과가 달라집니다. (Temperature가 0인 경우에도)

# 프롬프트 준비
messages = [
    {'role':'system', 'content':'당신은 건강한 식단과 식이의 전문가입니다.'},
    {'role':'user', 'content':'''건강한 아침 식사의 조합 예시를 짧게 3개 추천해 주세요.'''}
]
response = client.chat.completions.create(
    model = "gpt-4.1-mini",
    messages = messages,
    temperature =  0,
    max_completion_tokens = 500,

    seed= 2943
    # Seed가 같으면 유사한 샘플링, 

)
print(response.choices[0].message.content)
# 같은 코드로 두 번 실행하기 (살짝 달라짐)
response = client.chat.completions.create(
    model = "gpt-4.1-mini",
    messages = messages,
    temperature =  0,
    max_completion_tokens = 500,
    seed= 2943
)
print(response.choices[0].message.content)
<br><br><br>
## tiktoken으로 토큰 정보 확인하기

입/출력의 토큰 정보는 usage에 저장됩니다.   
   
토큰의 길이는 출력 속도/메모리 사용량/API 요금에 영향을 미칩니다.
response.usage
# accepted_prediction_tokens: 자동 완성 토큰
# audio_tokens : 음성 토큰
# reasoning_tokens : 추론 토큰
# cached_tokens : 이전 입력에서 캐시된 토큰 (비용 50% 할인)

tiktoken을 이용하면 모델별 토크나이저를 확인하고, 토큰의 개수를 구할 수 있습니다.
import tiktoken
# 4.1 계열 모델은 아직 미지원
tokenizer_4o = tiktoken.encoding_for_model("gpt-4o")
tokenizer_4o
prompt = 'GPT 모델별 토크나이저를 확인하고, 프롬프트 토큰의 개수를 구할 수 있습니다.'
tokens = tokenizer_4o.encode(prompt)
print(tokens)
print('총 글자 수:',len(prompt))
print('총 토큰 수:',len(tokens))
# GPT의 멀티모달 데이터 처리
# Audio - Text

음성 데이터를 전달하거나, 결과를 음성으로 생성할 수 있습니다.   

음성 데이터는 일반적으로 1시간의 대화 = 128k 토큰 정도로 알려져 있습니다.
# 음성 생성하기
import base64

response = client.chat.completions.create(
    model="gpt-4o-audio-preview",
    modalities=["text", "audio"],

    audio={"voice": "ash", "format": "mp3"},
    # alloy, ash, ballad, coral, echo, fable, nova, onyx, sage, shimmer

    messages = [
    {'role':'system', 'content' : '''당신은 전혀 공감하지 않으며, 상대를 불쾌하게 합니다. 답변은 짜증스러운 반말로 하세요.'''},
    {'role':'user', 'content':'오늘 회사 가기 싫어.'}
],
    temperature = 0.3,
    max_tokens=4096
)

mp3_bytes = base64.b64decode(response.choices[0].message.audio.data)

with open("speech.mp3", "wb") as f:
    f.write(mp3_bytes)
response.choices[0].message.audio.transcript
멀티모달 데이터는 바이너리 데이터를 Base64--> UTF-8로 변환해 전달합니다.   
해당 작업은 Image 데이터도 동일합니다.
audio_path = "./speech.mp3"

def encode(path):
  with open(path, "rb") as input_file:
    return base64.b64encode(input_file.read()).decode('utf-8')

encoded_string = encode(audio_path)
이후, 멀티모달 입력 포맷을 통해 전달합니다.    
`content`의 구성을 수정합니다.
response = client.chat.completions.create(
    model="gpt-4o-audio-preview",

    modalities=["text"],

    messages=[
        {'role':'system', 'content' : '''당신은 회사에 가기 싫은 직장인입니다.'''},

        {"role":'assistant',"content": '오늘 회사 가기 싫어.'},

        {"role": "user", "content": [
                {
                    "type": "input_audio",
                    "input_audio": {"data": encoded_string, "format": "mp3" }
                }]
        },
    ]
)

print(response.choices[0].message.content)
##  이미지 전달하기

image_url을 직접 전달하거나, 이미지를 인코딩하여 전달합니다.
# 링크로 이미지 전달하기
image_url = '''https://cloud.google.com/static/vertex-ai/generative-ai/docs/multimodal/images/timetable.png?hl=ko'''

messages = [
    {"role": "user", "content": [
        {"type": "text",
                 "text": "이 그림을 자세히 묘사해 보세요."
        },

        {"type": "image_url",
                "image_url": {"url": image_url}
        },
    ]}

]

response = client.chat.completions.create(
    model = "gpt-5-mini",

    messages= messages,
    # max_tokens = 1024,
    # temperature = 0.2
)
print(response.choices[0].message.content)

response.usage

# 링크 이미지 파일로 저장하기
import requests

save_path = "image.png"

response = requests.get(image_url, stream=True)
response.raise_for_status()  # HTTPError 발생 시 예외 처리

with open(save_path, 'wb') as file:
    for chunk in response.iter_content(chunk_size=8192):
        file.write(chunk)

print(f"이미지가 '{save_path}'에 성공적으로 저장되었습니다.")
import base64

# 오디오 데이터 처리와 동일
def encode(path):
  with open(path, "rb") as input_file:
    return base64.b64encode(input_file.read()).decode('utf-8')

# 이미지 경로
image_path = "image.png"
base64_image = encode(image_path)

messages = [
    {"role": "user", "content": [
        {"type": "text",
                 "text": """이 이미지에 표시된 공항 보드에서 시간과 도시를 분석하고,
해당 도시의 맛있는 음식 하나와 짧은 한국어 설명을 포함하여 json 형식으로 표시해 주세요.
'time', 'city' , 'food', 'food_desc' 키를 사용하세요.
목록만 출력하세요."""
        },

        {"type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
        },
    ]}
]


response = client.chat.completions.create(
    model="gpt-4.1-mini",
    messages= messages,
    max_tokens = 1024,
    temperature = 0.2
)

print(response.choices[0].message.content)

------