# [실습] LangChain으로 첫 체인 만들기

## 라이브러리 설치  

랭체인 관련 라이브러리를 설치합니다.  

# LangChain + LLM
!pip install langchain-huggingface langchain langchain-openai langchain-community langchain-ollama langchain-google-genai
!pip install dotenv
## 환경 변수 설정하기

import os
from dotenv import load_dotenv
# OPENAI_API_KEY, GOOGLE_API_KEY
load_dotenv(override=True)
# .env에서 환경 변수를 불러옴

# https://platform.openai.com/docs/overview OpenAI (유료 크레딧 필요)
# https://aistudio.google.com/prompts/new_chat 구글 (무료 API 가능)
모델 Provide의 클래스를 불러온 뒤, LLM을 연결하면 됩니다!
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

gpt = ChatOpenAI(model='gpt-5')
gemini = ChatGoogleGenerativeAI(model='gemini-2.5-flash', temperature=0.2)

question = '안녕? 너는 누구니? 한 문장으로, 진실만 말해.'


print(f'GPT: {gpt.invoke(question).content}')
print(f'Gemini: {gemini.invoke(question).content}')

## Ollama를 통해 LLM 불러오기


https://ollama.com/


올라마는 가장 간단하게 오픈 모델을 구동할 수 있는 방식입니다.   
추론 최적화/병렬 처리 등의 기능은 뛰어나지 않지만  
CPU에서도 구동이 가능하며, UI도 지원합니다.
### Ollama 설치하고 서빙하기    
터미널을 이용해 아래의 커맨드를 입력합니다.

- 설치

```curl -fsSL https://ollama.com/install.sh | sh```

- Context Window 늘리기 (기본 Context 4096)

```export OLLAMA_CONTEXT_LENGTH=16384```

- 서빙 (& 으로 백그라운드 실행)

``` ollama serve &```


올라마는 기본적으로 11434 포트에서 실행됩니다.   
Pull 을 통해 허깅페이스 모델의 주소나 Ollama 주소를 입력하면 모델을 불러와 저장합니다.
불러올 수 있는 모델의 목록은 https://ollama.com/search 에서 확인할 수 있습니다.   

기본적으로 양자화를 거친 상태로 불러오며, 16비트 Full Precision을 비롯한 다양한 버전도 확인할 수 있습니다.
from langchain_ollama import ChatOllama

qwen = ChatOllama(model='qwen3:4b-instruct', temperature=0.3,
                  max_tokens = 128)

print(f'Qwen3 4B Instruct: {qwen.invoke(question).content}')


gemma = ChatOllama(model='gemma3:4b', temperature=0.3,
                   max_tokens=128)

print(f'Qwen3 4B Instruct: {gemma.invoke(question).content}')
