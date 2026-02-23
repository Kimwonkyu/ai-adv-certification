# [실습] LangChain Tool Call과 Agent
Tool Calling의 템플릿을 이용하여, 툴을 연속적으로 실행하면 어떨까요?   

최근의 복잡한 에이전트 구조는 다양한 툴을 연결하고 다단 작업을 수행합니다.
# 라이브러리 설정 및 LLM 불러오기
!pip install setuptools langchain==0.3.27 langchain_community==0.3.27 langgraph==0.6.8 koreanize_matplotlib matplotlib langchain_experimental langchain_google_genai langchain_openai langchain_tavily
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
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_tavily import TavilySearch
from langchain_core.tools import tool

print("필수 모듈 임포트 완료")
tavily tool을 준비합니다.
@tool
def tavily_search(query, max_results=5):
    """Tavily API를 통해 검색 결과를 가져옵니다.
주어진 주제에 맞는 적절한 argument 값을 선정하세요.
query: 검색어
max_results : 검색 결과의 수(최소 1, 최대 20, 별도의 요청이 없으면 5로 고정)"""
    tavily_search = TavilySearch(max_results=max_results)

    search_results = tavily_search.invoke(query)['results']

    context =''
    for doc in search_results:
        doc_content = doc.get('content')

        context += 'TITLE: ' + doc.get('title','N/A') + '\nURL:' + doc.get('url')+ '\nContent:'+ doc_content
    return context
이번에는 커스텀 툴을 더 추가해 보겠습니다.
# 커스텀 툴 추가하기   

다양한 외부 API나 기능을 구현하여 툴로 구성할 수 있습니다.   
프롬프트에 포함되는 툴 이름과 설명, 입출력 형식을 이용해 툴의 사용성을 높입니다.
@tool
def multiply(x:int, y:int) -> int:
    "x와 y를 입력받아, x와 y를 곱한 결과를 반환합니다."
    return x*y

@tool
def current_date() -> str:
    "현재 날짜를 %y-%m-%d 형식으로 반환합니다."
    from datetime import datetime
    return f'현재 날짜는 {datetime.now().strftime("%Y-%m-%d")} 입니다!'

print(multiply.invoke({'x':3, 'y':4}))
print(current_date.invoke({}))

다단 Tool Calling을 수행하는 에이전트 중 가장 대표적인 것은 ReAct 에이전트입니다.
<br><br>

While문과 For문을 이용하여, ReAct 에이전트와 유사한 아래의 함수를 만들어 보겠습니다.
def react_agent(llm, question , tools = [tavily_search]):

    # 툴과 LLM 구성

    tool_list = {x.name: x for x in tools}
    # tavily_search.name = 'tavily_search' 을 이용하면
    # tool_list = {'tavily_search': tavily_search} 와 동일합니다.

    llm_with_tools = llm.bind_tools(tools)


    # 메시지 구성
    messages = [HumanMessage(question)]
    print('Query:', question)


    # LLM에 메시지 전달 (분기)
    tool_msg = llm_with_tools.invoke(question)
    # 1) 툴이 필요없으면: 응답 후 종료
    # 2) 툴이 필요하면: AIMessage (+ Tool Call)
    print('## LLM 호출 ##')
    messages.append(tool_msg)

    if tool_msg.content:
        print('LLM:', tool_msg.content)

    while tool_msg.tool_calls:
        # 툴 호출이 있을 경우: 툴 실행 후 결과를 전달 (반복)

        for tool_call in tool_msg.tool_calls:
            tool_name = tool_call['name']

            print(f"-- {tool_name} 사용 중 --")
            print(tool_call)


            tool_exec = tool_list[tool_name]

            tool_result = tool_exec.invoke(tool_call)
            messages.append(tool_result)
        tool_msg = llm_with_tools.invoke(messages)
        print('## LLM 호출 ##')
        messages.append(tool_msg)

        if tool_msg.content:
            print('LLM:', tool_msg.content)


    result = tool_msg

    return result.content
response = react_agent(llm, "Gemma 3 모델의 장단점 검색해서 알려줘. 매번 툴 요청과 함께, 어떤 툴을 어떻게 실행할 건지 활기차게 말해줘.")
# GPT-5-Mini
print(response)
response = react_agent(llm_nonreasoning, "Gemma 3 모델의 장단점 검색해서 알려줘. 매번 툴 요청과 함께, 어떤 툴을 어떻게 실행할 건지 활기차게 말해줘.")
# GPT-5-Mini
print(response)
tools = [tavily_search, multiply, current_date]
question = '오늘 날짜가 어떻게 되니?'
response = react_agent(llm_gemini_nonreasoning, question, tools= tools)
# 2.0 Flash
print(response)
# Python REPL Tool 추가하기

3개의 툴에 추가로, Python 코드를 실행하는 REPL 툴을 추가해 보겠습니다.
# 한글 출력 테스트
import matplotlib.pyplot as plt
import koreanize_matplotlib

plt.plot([1, 2, 3], [3, 2, 1])
plt.title("한글 제목 테스트")
plt.xlabel("X축")
plt.ylabel("Y축")
plt.show()
from langchain_experimental.utilities import PythonREPL
repl = PythonREPL()
PythonREPL은 임의의 코드를 전달하면 결과를 돌려줍니다.
repl.run('print(1)')
이를 활용하여, 충분한 맥락이 담긴 툴을 구성합니다.
from typing import Annotated

@tool
def python_repl_tool(
    code: Annotated[str, "The python code to execute."],
):
    """Use this to execute python code. If you want to see the output of a value,
    you MUST print it out with `print(...)`. This is visible to the user."""
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    result_str = f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"
    return result_str
tools = [tavily_search, current_date, python_repl_tool, multiply]

question = '2024년 프로야구 정규시즌 최종 순위 알려줘. 전체를 그래프로 구성하여 1.png에 저장해.'
react_agent(llm_nonreasoning, question=question, tools=tools)

tools = [tavily_search, current_date, python_repl_tool, multiply]

question = '이번 달에 발매된 주요 한국 래퍼들의 음악 소개해줘. 검색해서 알려줘. 안나오면 나올때까지 다른 검색어를 활용해서 검색해.'
react_agent(llm_gemini, question=question, tools=tools)

tools = [tavily_search, current_date, python_repl_tool, multiply]

question = '이번 달에 발매된 주요 한국 래퍼들의 음악 소개해줘. 검색은 두번만 하고, 되묻지 말고 끝까지 해.'
react_agent(llm, question=question, tools=tools)


툴의 다단계 실행이 필요한 입력을 전달하여, react_agent 함수의 작동 과정을 이해해 보세요.

우리가 만든 `react_agent` 함수는 가장 간단한 형태의 Tool Calling을 연속적으로 수행하는 방식이었는데요.   

랭그래프의 `creat_react_agent`는 이와 같은 기능을 빌트인으로 지원합니다.
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(llm, tools)
# system_prompt를 통해 시스템 프롬프트도 커스텀

agent
agent.invoke(
    {'messages':[HumanMessage('오늘 태어난 유명인 알려줘')]})
자율 Agent는 산업의 다양한 분야에서 높은 활용성을 갖지만, 아래의 개선점을 고려해야 합니다.

- Context가 계속 늘어나기만 하는 구조이므로, 작업이 길어지면 성능이 저하됩니다.
- 당연한 툴 실행에도 매번 LLM을 다중 실행하므로, 토큰 소모가 큽니다.


여러 개의 에이전트가 서로 소통하도록 하는 멀티 에이전트 구조의 경우에는 해당 문제가 더욱 중요합니다.