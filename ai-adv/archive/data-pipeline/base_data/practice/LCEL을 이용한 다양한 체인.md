# [실습] LCEL을 이용한 다양한 체인


LangChain Expression Language(LCEL)는 랭체인에서 체인을 간결하게 구성하는 문법입니다.    

단일 체인으로 다양한 모듈을 구성하며, 이 때 `|` 연산자를 사용합니다.
!pip install langchain langchain-openai langchain-community dotenv
import os
from dotenv import load_dotenv

load_dotenv(override=True)
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model='gpt-5-mini',
    temperature = 1.0, 
    max_tokens = 8192
)

llm.invoke("안녕? 너는 모델 이름이 뭐니?")
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser

print("필수 모듈 임포트 완료")
## LCEL 체인: Prompt | LLM
prompt = ChatPromptTemplate(
    [
        ('system', '당신은 LLM과 자연언어 처리의 전문가입니다. 주어진 단어를 일반인들도 이해할 수 있게 쉽게 설명해주세요.'),
        ('user', '[단어]: {term}')
    ]
)
prompt

# prompt.format_messages(term='토크나이저')
프롬프트와 LLM을 |로 연결하면, 입력 변수 전달 --> 프롬프트 --> LLM 의 과정이 한 번에 실행됩니다.
chain = prompt | llm

response = chain.invoke({"term":"토크나이저"})

print(response.content)
chain = prompt | llm
chain

response = chain.invoke("할루시네이션(언어 모델)")
# 매개변수가 1개이므로 바로 실행되는 구조

print(response.content)
from rich import print as rprint

rprint(chain)
매개변수가 2개인 체인도 동일합니다.
prompt = ChatPromptTemplate(
    [
        ('system', '당신은 {topic}의 전문가입니다. 주어진 단어를 일반인들도 이해할 수 있게 쉽게 설명해주세요.'),
        ('user', '[단어]: {term}')
    ]
)
prompt

prompt.format_messages(topic='하드웨어와 컴퓨팅', term = 'GPU')
chain2 = prompt | llm
result = chain2.invoke({'topic':'하드웨어와 컴퓨팅', 'term':'GPU'})
print(result.content)
result = chain2.invoke({'topic':'만화와 애니메이션', 'term':'슈퍼마리오'})
print(result.content)
## LLM의 구조화된 출력 생성하기

LLM은 기본적으로 텍스트만을 생성하지만, 구조화된 데이터를 생성할 수도 있습니다.   

랭체인의 기본 기능인 with_structured_output을 사용하거나, 파서를 통해 변환합니다.
from pydantic import BaseModel, Field

# Pydantic Class: 데이터 형식을 지정 (강제)
class recipe(BaseModel):
    preparation: str = Field(description='준비 재료(이름, 개수, 무게 등)')
    process: str = Field(description='준비 과정')
    note: str = Field(description='성공적인 결과를 위해 필요한 참고 내용')


X = recipe(preparation='재료', process='---', note='끝!')
X
summarizer = llm.with_structured_output(recipe)
summarizer
result = summarizer.invoke("피자 만드는 법 알려줘.")
rprint(result)
result.note
파서(Parser)는 LLM 뒤에서 출력을 변환합니다.   
이 때, LLM이 파싱할 수 있는 출력을 해야 하므로 프롬프트도 추가합니다.
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser, PydanticOutputParser

parser = StrOutputParser()
str_chain = llm | parser

str_chain
str_chain.invoke("언어 모델이 작곡을 할 수 있니?")
jsonparser  = JsonOutputParser()
# 출력을 json으로 변환 

json_chain = llm | jsonparser

result = json_chain.invoke("""LLM 모델로 유명한 한국과 해외의 기업 세 곳을 선정하여, 
대표 모델과 함께 간략하게 정리해줘.

JSON 형식으로 출력해.""")

result
result[0]['대표모델']
parser = JsonOutputParser()
# recipe 형식의 json만 파싱하는 파서

format_str = parser.get_format_instructions()
print(format_str)
parser = JsonOutputParser(pydantic_object=recipe)
# recipe 형식의 json만 파싱하는 파서

format_str = parser.get_format_instructions()
print(format_str)

prompt = ChatPromptTemplate(
    [
        ('system', '주어진 문제에 대한 답변을 제공하세요.'),
        ('human', '''사용자의 질문: {question}
---
{format_str}''')
    ]
).partial(format_str = format_str)

# prompt | llm.with_structured_output(Recipe)
# question --> Recipe Class

chain = prompt | llm | parser

result = chain.invoke("카라멜 푸라푸치노 만드는 법")
result
parser = PydanticOutputParser(pydantic_object=recipe)
# recipe 형식의 Class 파싱하는 파서

format_str = parser.get_format_instructions()

prompt = ChatPromptTemplate(
    [
        ('system', '주어진 문제에 대한 답변을 제공하세요.'),
        ('human', '''사용자의 질문: {question}
---
{format_str}''')
    ]
).partial(format_str = format_str)

# prompt | llm.with_structured_output(Recipe)
# question --> Recipe Class

chain = prompt | llm | parser

result = chain.invoke("연어 크림치즈 베이글 만들기")
result
rprint(result)
<br><br>
## Runnables

Runnables는 LCEL의 기본 단위로, 입력을 받아 출력을 생성하는 기본 단위입니다.    
llm, prompt, chain 등이 모두 Runnable 구조에 해당합니다.

이번에는, 데이터 흐름을 제어하는 특별한 Runnable인   
RunnableParallel과 RunnablePassthrough을 이용해 체인을 구성해 보겠습니다.


### RunnableParallel

RunnableParallel은 서로 다른 체인을 병렬적으로 실행합니다.
from langchain_core.runnables import RunnableParallel

prompt1 = ChatPromptTemplate(["주어진 의견에 대해 무조건적으로 반대하세요. 답변은 반말로 하세요.\n 의견:{opinion}"])
prompt2 = ChatPromptTemplate(["주어진 의견에 대해 무조건적으로 찬성하세요. 답변은 정중하게 하세요.\n 의견:{opinion}"])

chain1 = prompt1 | llm | StrOutputParser()
chain2 = prompt2 | llm | StrOutputParser()

chain3 = RunnableParallel(cons = chain1, pros = chain2)

result = chain3.invoke({'opinion':'컴퓨터공학이 세상에서 가장 중요한 학문이다.'})

result
print(result)
체인의 직렬 연결은 아래와 같이 만들 수 있습니다.
prompt3 = ChatPromptTemplate(
    [
        ('system','당신은 매우 합리적이고 논리적입니다. '),
        ('human','''
아래의 두 의견 중, 당신은 누구의 손을 들어 주겠습니까?

찬성 의견: {pros}
         

반대 의견: {cons}''')
    ]
)

chain4 = chain3 | prompt3 | llm | StrOutputParser()
# Chain3 : Dict Return --> Prompt3 입력

result = chain4.invoke("인간은 지구상에서 가장 위대한 생물이다.")

result

체인의 중간에 Dict가 붙는 경우, 이는 내부적으로 RunnableParallel로 변환되어 실행됩니다.
# 체인에는 dict가 포함될 수 있음

chain4 = chain3 | prompt3 | llm | {'Decision':StrOutputParser()}
# chain3 | prompt3 | llm | RunnableParallel(Decision=StrOutputParser())

chain4.invoke("구글이 최고의 언어 모델 기업이다.")




## RunnableParallel.Assign   

Assign을 사용하면, 직전 체인의 실행 결과를 다음 체인에 전달하고, 결과를 결합할 수 있습니다.   
assign을 붙이기 위해서는 체인의 결과물이 dict 형태여야 합니다.


chain4 = prompt3 | llm | StrOutputParser()
# chain4 = {pros, cons} --> decision 출력하는 체인
# chain3 = RunnableParallel로 {pros, cons} 생성하는 체인

chain5 = chain3.assign(Decision = chain4)
#        pros, cons +  Decision
result = chain5.invoke("인간은 미래에 인공지능을 이길 수 없다.")
result

<br><br>
### RunnablePassthrough
RunnablePassthrough는 체인의 직전 출력을 그대로 가져옵니다.
from langchain_core.runnables import RunnablePassthrough

prompt1 = ChatPromptTemplate(["주어진 단어에 대한 재미있는 삼행시를 작성하세요.\n 단어:{word}"])

chain1 = (prompt1 
          | llm
          | StrOutputParser()
          | {'result':RunnablePassthrough()}
        )

result = chain1.invoke('컴퓨터')
result

prompt1 = ChatPromptTemplate(["주어진 단어에 대한 재미있는 삼행시를 작성하세요.\n 단어:{word}"])
prompt2 = ChatPromptTemplate(['''주어진 시가 재미있는지 판단하고, 이를 개선해 주세요. 
{N} 문장으로 만들어야 합니다.
---                          

시: {poem}'''])

# TODO: 입력변수 word, N --> 2번 체인: poem, N 

chain1 = prompt1 | llm| StrOutputParser()
chain1_5 = prompt2 | llm | StrOutputParser()

chain2 = RunnablePassthrough.assign(poem = chain1) | chain1_5
#        word, N              +       poem        ---> 출력

result = chain2.invoke({'word':'컴퓨터', 'N':5})
result
chain2 = RunnablePassthrough.assign(poem = chain1).assign(edit= chain1_5)
#        word, N              +       poem          +      edit

result = chain2.invoke({'word':'컴퓨터', 'N':5})
result
prompt2 = ChatPromptTemplate(['''주어진 시가 재미있는지 판단하고, 이를 개선해 주세요. 
{word}에 대한 삼행시이므로, 규칙을 잘 지키는지 검증하고, 다시 써 주세요.                          

{N}개의 감상 포인트도 알려주세요.                              
---                          

시: {poem}'''])

chain1_5 = prompt2 | llm | StrOutputParser()


chain2 = RunnablePassthrough.assign(poem = chain1).assign(edit= chain1_5)
#        word, N              +       poem          +      edit

result = chain2.invoke({'word':'컴퓨터공학', 'N':5})
result
## 복잡한 체인 만들기
chain2에서 새로운 매개변수가 추가되는 경우는 어떻게 해야 할까요?
prompt = ChatPromptTemplate(['''영화가 주어지면, 주연 배우 1명과 감독을 json으로 출력하세요.
actor, director의 키를 사용하세요.

movie: {movie}'''])

chain = prompt | llm | JsonOutputParser()

chain.invoke("타이타닉")
prompt = ChatPromptTemplate(['''{actor}와 {director}는 어떤 영화에서 같이 만났나요? 모두 알려주세요.'''])

chain2 = prompt | llm | StrOutputParser()

chain3 = chain | chain2

chain3.invoke("인셉션")

chain3 = chain.assign(result = chain2)
chain3.invoke("더 울프 오브 월 스트리트")
<br><br>
체인을 분리하고 RunnableParallel을 이용하면 중간 과정을 모두 출력할 수 있습니다.

<br><br><br>하나의 체인에서 여러 개의 값을 생성하려면,   
JsonOutputParser를 쓰면 됩니다.

