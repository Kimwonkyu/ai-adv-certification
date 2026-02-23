# [실습] LangChain Tool Call
LLM은 언어 모델이지만, 특수한 형식의 출력을 통해 외부 도구와 소통할 수 있습니다.   

1. LLM에게 프롬프트를 통해 도구의 사용법을 전달합니다.   
2. LLM은 도구가 필요한 경우 파싱 가능한 출력을 생성합니다.   
3. 해당 출력을 파싱하여 실제 도구를 실행하는 과정을 자동화합니다.
4. 실행 결과를 다시 LLM에 전달하면, LLM은 결과를 참고하여 다음 출력을 생성합니다.

# 라이브러리 설정 및 LLM 불러오기
!pip install langchain==0.3.27 langchain_community==0.3.27 langgraph==0.6.8 langchain_google_genai langchain_openai langchain_community langchain_tavily -q
import os
from dotenv import load_dotenv
# OPENAI_API_KEY, GOOGLE_API_KEY
load_dotenv(override=True)
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
try:
    llm = ChatOpenAI(model='gpt-5-mini', temperature=0.3)
    llm_nonreasoning = ChatOpenAI(model='gpt-4.1-mini', temperature=0.3)
    print("✅ GPT API 사용 가능!")
except:
    print("❌ GPT API 사용 불가- API 키를 확인하세요!")
try:
    llm_gemini = ChatGoogleGenerativeAI(model='gemini-2.5-flash', temperature=0.7, 
                                rate_limiter=rate_limiter)
    llm_gemini_nonreasoning = ChatGoogleGenerativeAI(model='gemini-2.0-flash', temperature=0.3, 
                                rate_limiter=rate_limiter)
    print("✅ Gemini API 사용 가능!")
except:
    print("❌ Gemini API 사용 불가- API 키를 확인하세요!")
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

print("필수 모듈 임포트 완료")
## LLM에게 어려운 질문하기   

LLM에게 학습하지 못한 범위의 질문을 전달하면 정확도가 떨어지거나 잘못된 답변을 생성합니다.   
llm.invoke("케이팝 데몬헌터스가 뭐야?")
# 5-mini: 24.6 까지만 
llm_gemini.invoke("케이팝 데몬헌터스가 뭐야?")

Anthropic의 기술 블로그에서는, LLM의 외연을 확장하는 방법으로 검색(Retrieval)과 도구(Tool)을 제안했습니다.   
Retrieval은 이후의 RAG에서 자세히 다루도록 하고, 이번 과정에서는 Tool에 대해 이해해 보겠습니다.
## Tool
Tool은 LLM이 이해하고 사용할 수 있는 도구입니다.   
프롬프트를 통해 설명을 전달하므로, 넓은 의미에서 Context Engineering에 해당합니다.   

# OpenAI API의 도구 생성 기능

from openai import pydantic_function_tool
from pydantic import BaseModel, Field

class web_search(BaseModel):
    """웹 검색을 수행합니다."""
    query: str= Field(description="""검색 키워드""")

web_search_tool = pydantic_function_tool(web_search)

web_search_tool
위 결과를 Tool Schema라고 하며, 해당 내용이 LLM의 프롬프트에 포함되어 툴 사용 정보를 전달합니다.
## LangChain Tool 설정하기

LangChain은 Tool Calling을 쉽게 연동하기 위한 기능을 제공합니다.   


LangChain에서 자체적으로 지원하는 Tool을 사용하거나, RAG의 Retriever를 Tool로 변환하는 것도 가능합니다.     
또한, 함수를 Tool로 변환할 수도 있습니다.
가장 대표적인 built-in 툴인 Tavily Search (http://app.tavily.com/) 를 연결해 보겠습니다.

해당 도메인에 접속하여, 구글 계정으로 회원가입을 진행해 주세요.

이후, API 키를 생성하여 env 파일에 `TAVILY_API_KEY`로 추가합니다.
from langchain_tavily import TavilySearch

# 파일 업데이트 후 실행
load_dotenv(override=True)

tavily_search = TavilySearch(
    max_results=5,
    # 더 많은 옵션은 Tavily API 문서 Playground 참고
    )
tavily_search
search_docs = tavily_search.invoke("케이팝 데몬 헌터스의 의미")
search_docs
API 호출은 대부분 사용자가 원하는 정보보다 많은 내용을 전달합니다.   
불필요한 내용을 제거하는 함수를 구성합니다.
from langchain_core.tools import tool

@tool
def tavily_search(query, max_results = 5):
    """Tavily 검색을 통해 인터넷 검색 결과를 가져옵니다.
query는 검색어를 의미하며, max_results는 최대 검색 문서 수를 의미합니다.
사용자가 별도의 요청을 하지 않으면 max_results는 5를 사용하고, 최대한 검색하라고 하는 경우 20을 사용하세요."""
    tavily_search = TavilySearch(max_results=max_results)
    results = tavily_search.invoke(query)['results']

    context = ''
    for result in results:
        # result: dict
        doc_content = 'URL: '+ result.get('url') +'\n Title:'+ result.get('title') +'\n Content:' + result.get('content')
        context += doc_content + '\n\n---\n\n'

    return context

print(tavily_search.invoke({'query':'케이팝 데몬헌터스'}))




위에서 만든 함수에, @tool decorator를 붙이면 랭체인의 툴로 변환됩니다.
## LLM에 Tool 연결하기   

생성한 툴은 llm.bind_tools()를 통해 LLM에 연결할 수 있습니다.    
tools = [tavily_search]

llm_with_tools = llm.bind_tools(tools)
llm_with_tools
랭체인에서, tool 정보는 tools에 저장되는데요.   
해당 내용은 랭체인 내부에서 json Schema 형식으로 프롬프트에 포함됩니다.
llm_with_tools.kwargs['tools']
LLM은 프롬프트로 주어지는 툴 정보를 바탕으로 Tool의 사용을 결정합니다.    
Schema를 통해, 툴의 의미와 사용 방법, 형식을 이해합니다.
Tool을 탑재한 LLM을 실행해 보겠습니다.
prompt = ChatPromptTemplate(
    [
        ('system', '주어진 검색 툴을 사용해서 질문에 답변하세요.'),
        ('human', '{query}')
    ]
)
chain = prompt | llm_with_tools
tool_call_msg = chain.invoke('삼성SDS의 GPUaaS 서비스가 뭐야?')

tool_call_msg

`tool_calls`에 포함된 name을 활용하면 툴 결과를 전달할 수 있습니다.   
`name` 값은 문자열이므로, dictionary를 통해 연결합니다.
tool_call_msg.tool_calls
Tool에 tool_call을 입력해 invoke를 수행합니다.
tool_list = {'tavily_search': tavily_search}
tool_exec = tool_list[tool_call_msg.tool_calls[0]['name']]
tool_exec
tool_msg = tool_exec.invoke(tool_call_msg.tool_calls[0])
tool_msg
ToolMessage라는 새로운 형태의 메시지가 생성되었습니다.
HumanMessage, AIMessage(Tool Call), ToolMessage를 모두 전달합니다.
messages= []

messages.append(HumanMessage('삼성SDS의 GPUaaS 서비스가 뭐야?'))
# 질문이 들어있는 HumanMessage + 툴 요청이 들어있는 AIMessage + 툴 결과가 들어있는 ToolMessage
messages.append(tool_call_msg)
messages.append(tool_msg)


`Query-Tool Call-Tool`의 형식은 가장 기본적인 툴 사용 방법입니다.
result = llm_with_tools.invoke(messages)
print(result.content)






