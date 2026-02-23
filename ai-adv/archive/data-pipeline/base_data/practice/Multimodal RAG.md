# [실습] Multimodal RAG 구현하기
일반적인 RAG 구조에서, 성능이 떨어지는 대표적인 상황은 이미지/테이블 데이터가 포함된 문서를 처리하는 경우입니다.

일반적인 문서 로더를 이용해 RAG를 수행할 경우, 텍스트만 활용하게 되어 정보의 부분적 손실이 발생하는데요.    

이번 실습에서는 오픈 소스 라이브러리 파서인 Docling을 이용해 이미지/표 등이 포함된 PDF 문서를 재구성하고, 이를 통해 RAG를 수행해 보겠습니다.
## 필수 라이브러리 설치
docling 라이브러리를 설치합니다.    

GITHUB: https://github.com/DS4SD/docling   

https://docling-project.github.io/docling/
!pip install docling==2.48.0 langchain==0.3.27 langchain_huggingface sentence_transformers jsonlines langchain-openai langchain-google-genai langchain-community==0.3.27 beautifulsoup4 langchain_chroma
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from rich import print as rprint

load_dotenv('env', override=True)

reasoning = {
    "effort": "medium",  # 'low', 'medium', or 'high'
    "summary": "auto",  # 'detailed', 'auto', or None
}

llm = ChatOpenAI(model="gpt-5-mini", reasoning=reasoning)
response = llm.invoke("멀티모달 RAG 알아?")
response
print(response.text())
docling은 PDF 데이터를 마크다운으로 변환합니다.      
텍스트 이외에도, 표와 이미지를 바운딩 박스로 추출할 수 있습니다.
# 기본 코드: Image를 제외한 텍스트를 마크다운으로 변경

from docling.document_converter import DocumentConverter

source = "https://arxiv.org/pdf/2408.09869"  # PDF path or URL
converter = DocumentConverter()
result = converter.convert(source)

with open("result.md", "w") as f:
    f.write(result.document.export_to_markdown())  # output: "### Docling Technical Report[...]"

f.close()
Docling은 다음의 작업을 지원합니다.
1. 각 페이지를 이미지로 추출하기
2. 페이지에 포함된 각 이미지를 추출하기
3. 전체를 HTML/MD 형식으로 재구성하기
import logging
import time
import re
import requests
from pathlib import Path
from urllib.parse import urlparse
from docling_core.types.doc import ImageRefMode, PictureItem, TableItem
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

IMAGE_RESOLUTION_SCALE = 2.0
# 이미지 해상도 확대 스케일링

_log = logging.getLogger(__name__)

def download_pdf(url, save_dir="downloads"):
    """URL에서 PDF 파일을 다운로드하여 로컬 경로를 반환"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    response = requests.get(url, stream=True)
    if response.status_code == 200:
        filename = url.split("/")[-1]

        if not filename.endswith('pdf'):
            filename+='.pdf'
        file_path = save_dir / filename

        with open(file_path, "wb") as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)

        return str(file_path)
    else:
        raise Exception(f"Failed to download file: {url} (Status code: {response.status_code})")

def is_url(path):
    """주어진 문자열이 URL인지 확인"""
    return re.match(r'https?://', path) is not None

def parse(path, output_dir='docling_result',
capture_page_as_images = False,
capture_table_as_images = False,
capture_picture_as_images = False,
parse_with_image_refs = False):

    logging.basicConfig(level=logging.INFO)

    if is_url(path):  # URL이면 다운로드
        _log.info(f"Downloading PDF from {path}...")
        path = download_pdf(path)

    input_doc_path = Path(path)
    output_dir = Path(output_dir)

    pipeline_options = PdfPipelineOptions()
    pipeline_options.images_scale = IMAGE_RESOLUTION_SCALE
    pipeline_options.generate_page_images = True
    pipeline_options.generate_picture_images = True

    doc_converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )

    start_time = time.time()
    conv_res = doc_converter.convert(input_doc_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    doc_filename = conv_res.input.file.stem

    # 페이지 이미지 저장
    if capture_page_as_images:
        print("Capturing page images...")
        for page_no, page in conv_res.document.pages.items():
            page_image_filename = output_dir / f"{doc_filename}-{page_no}.png"
            with page_image_filename.open("wb") as fp:
                page.image.pil_image.save(fp, format="PNG")

    # 이미지/테이블 저장
    if capture_table_as_images:
        print("Capturing table images...")
        table_counter = 0
        for element, _level in conv_res.document.iterate_items():
            if isinstance(element, TableItem):
                table_counter += 1
                element_image_filename = output_dir / f"{doc_filename}-table-{table_counter}.png"
                with element_image_filename.open("wb") as fp:
                    element.get_image(conv_res.document).save(fp, "PNG")

    if capture_picture_as_images:
        print("Capturing picture images...")
        picture_counter = 0
        for element, _level in conv_res.document.iterate_items():
            if isinstance(element, PictureItem):
                picture_counter += 1
                element_image_filename = output_dir / f"{doc_filename}-picture-{picture_counter}.png"
                with element_image_filename.open("wb") as fp:
                    element.get_image(conv_res.document).save(fp, "PNG")

    # 전체 마크다운 저장 (이미지는 링크 형태로)
    if parse_with_image_refs:
        print("Saving markdown with image references...")
        md_filename = output_dir / f"{doc_filename}-with-image-refs.md"
        conv_res.document.save_as_markdown(md_filename, image_mode=ImageRefMode.REFERENCED)

    else:
        # 전체 마크다운 저장 (이미지는 utf8 형태로)
        print("Saving markdown with images...")
        md_filename = output_dir / f"{doc_filename}-with-images.md"
        conv_res.document.save_as_markdown(md_filename, image_mode=ImageRefMode.EMBEDDED)


    end_time = time.time() - start_time
    _log.info(f"Document converted and figures exported in {end_time:.2f} seconds.")
    _log.info(f"Output file: {md_filename}")
    # total number of images and tables
    _log.info(f"Total number of tables: {table_counter}")
    _log.info(f"Total number of pictures: {picture_counter}")

# 실행 예시
parse("https://arxiv.org/pdf/2503.09516", capture_page_as_images=True, capture_table_as_images=True, capture_picture_as_images=True, parse_with_image_refs=True)

`with-image-refs.md` 파일에는 이미지 파일의 경로가 마크다운 링크로 포함되어 있습니다.    

이를 이용해, 전체 문서에서 각 이미지를 캡셔닝하고, 이를 바탕으로 멀티모달 RAG를 수행합니다.
def simple_parse(path, output_dir='mm_rag'):
    logging.basicConfig(level=logging.INFO)

    if is_url(path):  # URL이면 다운로드
        _log.info(f"Downloading PDF from {path}...")
        path = download_pdf(path)

    input_doc_path = Path(path)
    output_dir = Path(output_dir)

    pipeline_options = PdfPipelineOptions()
    pipeline_options.images_scale = IMAGE_RESOLUTION_SCALE
    pipeline_options.generate_page_images = True
    pipeline_options.generate_picture_images = True

    doc_converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )

    start_time = time.time()
    conv_res = doc_converter.convert(input_doc_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    doc_filename = conv_res.input.file.stem

    # # 이미지/테이블 저장
    # table_counter = 0
    # picture_counter = 0
    # for element, _level in conv_res.document.iterate_items():
    #     if isinstance(element, TableItem):
    #         table_counter += 1
    #         element_image_filename = output_dir / f"{doc_filename}-table-{table_counter}.png"
    #         with element_image_filename.open("wb") as fp:
    #             element.get_image(conv_res.document).save(fp, "PNG")

    #     if isinstance(element, PictureItem):
    #         picture_counter += 1
    #         element_image_filename = output_dir / f"{doc_filename}-picture-{picture_counter}.png"
    #         with element_image_filename.open("wb") as fp:
    #             element.get_image(conv_res.document).save(fp, "PNG")

    # 전체 마크다운 저장(이미지는 Reference 형태로)
    md_filename = f"{output_dir}/{doc_filename}-with-image-refs.md"
    conv_res.document.save_as_markdown(md_filename, image_mode=ImageRefMode.REFERENCED)

    end_time = time.time() - start_time
    _log.info(f"Document converted and figures exported in {end_time:.2f} seconds.")
    return str(md_filename)



example_path = 'https://arxiv.org/pdf/2503.09516'
md_path = simple_parse(example_path)
마크다운 텍스트를 불러옵니다.
with open(md_path, "r") as f:
    markdown_text = f.read()

print(markdown_text[1300:2400])

import re

# get [Image] ()
image_list = re.findall(r'\[Image\]\((.*?)\)', markdown_text)
len(image_list)
image_list
마크다운 텍스트에서, 각각의 이미지는 [Image] (링크) 구조로 만들어집니다.   

해당 패턴을 모두 찾은 뒤, 직전/직후의 N글자 청크를 포함하여 캡션을 생성합니다.
import re
import requests
from io import BytesIO
from PIL import Image

def extract_image_info(markdown_text, n=500, output_path='mm_rag'):
    """
    마크다운 텍스트에서 이미지 태그 정보를 추출하여 리스트로 반환합니다.

    Args:
        markdown_text (str): 마크다운 텍스트
        n (int): 이미지 태그 직전/직후 글자 수

    Returns:
        list: [직전N글자, 직후N글자, 이미지] 형태의 이중 리스트
    """

    image_pattern = r'!\[.*?\]\((.*?)\)'
    matches = re.finditer(image_pattern, markdown_text)
    image_info_list = []

    for match in matches:
        image_path = match.group(1)
        start_index = match.start()
        end_index = match.end()

        # 직전 N글자 추출
        before_text = markdown_text[max(0, start_index - n):start_index]
        # 직후 N글자 추출
        after_text = markdown_text[end_index:min(len(markdown_text), end_index + n)]


        full_path = f"{output_path}/{image_path}" #base path와 image path합치기

        image_info_list.append([before_text, after_text, full_path])

    return image_info_list


# 이미지 정보 추출 및 출력
image_info = extract_image_info(markdown_text, n=1000)

print('전체 이미지 수:', len(image_info))
for before, after, image in image_info:
    print('----------------------------------------------')

    print(f"{before} \n [Image Goes Here] \n {after}")
    print('----------------------------------------------')
이미지를 정상적으로 추출했고, 앞뒤 캡션을 연결하는 과정을 확인했습니다.    
이제 LLM을 통해 캡션을 생성합니다.
llm에게 전체 문서의 내용을 포함하여, Image의 정보를 얻는 Contextual Retrieval을 수행합니다.
Image.open(f'mm_rag/{image_list[0]}')
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import base64

def get_caption(before, after, image_path):
    with open(image_path, 'rb') as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')
    context_prompt = ChatPromptTemplate(
        [
            ('system', '''
당신은 주어진 문서의 내용을 참고하여, 이미지에 대한 캡션을 작성해야 합니다.
전체 문서의 전반부 내용과 이미지 이전의 텍스트와 이후의 텍스트가 주어집니다.
이를 활용하여, 문맥을 고려하여 이해를 돕기 위한 캡션을 작성하세요.

캡션의 내용은 최대 5문장으로 작성하고, 원문 Document의 언어를 그대로 따르세요.
만약 문서가 영어인 경우, 영어로 작성하세요.'''),
            ('user',
            [{'type':'text', 'text':'''
[Document]: {document}

---

[Before]: {before}

---

[Image goes Here]

---
[After]: {after}'''},
            {"type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
                    }
            ]
        )])


    context_chain = context_prompt.partial(document=markdown_text) | llm | StrOutputParser()
    result = context_chain.invoke({'before':before, 'after':after})
    return result
두 함수를 바탕으로, 마크다운 텍스트를 입력으로 받아 캡션을 추가하는 함수를 구현합니다.
def extract_image_info(markdown_text, n=10, output_path='mm_rag'):
    """
    마크다운 텍스트에서 이미지 태그 정보를 추출하고 캡션을 추가합니다.

    Args:
        markdown_text (str): 마크다운 텍스트
        n (int): 이미지 태그 직전/직후 글자 수
        output_path (str): 이미지 파일의 기본 경로

    Returns:
        str: 캡션이 추가된 마크다운 텍스트
    """

    caption_list = []

    image_pattern = r'!\[.*?\]\((.*?)\)'
    matches = list(re.finditer(image_pattern, markdown_text))


    for match in matches:
        image_path = match.group(1)
        start_index = match.start()
        end_index = match.end()

        # 직전 N글자 추출
        before_text = markdown_text[max(0, start_index - n):start_index]
        # 직후 N글자 추출
        after_text = markdown_text[end_index:min(len(markdown_text), end_index + n)]

        full_path = f"{output_path}/{image_path}" #base path와 image path합치기

        caption = get_caption(before_text, after_text, full_path)
        caption_list.append(caption)
    return caption_list

print(len(markdown_text))

caption_list = extract_image_info(markdown_text,500)
print(len(caption_list))
caption_list
이미지와 텍스트를 포함하는 벡터 DB를 구성하겠습니다.
from langchain_core.documents import Document

doc = Document(markdown_text, metadata={'type': 'text'})

from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=400,
)


chunks = text_splitter.split_documents([doc])
print('Text Chunks:', len(chunks))

images= [Document(page_content = caption, metadata = {'type':'image', 'source': image_list[i]}) for i, caption in enumerate(caption_list)]

print('Image Chunks:', len(images))

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model = 'text-embedding-3-large', chunk_size=100)

Chroma().delete_collection()

db = Chroma(persist_directory='./chroma_mmrag',
            embedding_function=embeddings,
            collection_metadata={'hnsw:space':'l2'}
            )

db.add_documents(images)
db.add_documents(chunks)

retriever = db.as_retriever(search_kwargs={"k": 5})
RAG 체인을 구성합니다.   
원문 문서가 영어로 되어 있으므로, 번역 체인을 추가합니다.
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate

translate_prompt = ChatPromptTemplate([
    ("system", """You are a helpful assistant that translates Korean Question to English.
Translate the following question to English.
Just Write the translated Question and nothing else"""),
    ("user", "Korean Question:{text}")
])

translate_chain = translate_prompt | llm | StrOutputParser()

translate_chain.invoke("왜 논문 이름이 Search-R1이야?")
prompt = ChatPromptTemplate([
    ("user", '''당신은 QA(Question-Answering)을 수행하는 Assistant입니다.
다음의 Context를 이용하여 Question에 한국어로 답변하세요.
정확한 답변을 제공하세요.
만약 모든 Context를 다 확인해도 정보가 없다면, "정보가 부족하여 답변할 수 없습니다."를 출력하세요.
---
Context: {context}
---
Question: {question}''')])

prompt.pretty_print()
def format_docs(docs):
    return "\n\n---\n\n".join(['Content:' +doc.page_content for doc in docs])
    # join : 구분자를 기준으로 스트링 리스트를 하나의 스트링으로 연결

rag_chain = (
    {"context": translate_chain | retriever | format_docs, "question": RunnablePassthrough()}
    # retriever : question을 받아서 context 검색: document 반환
    # format_docs : document 형태를 받아서 텍스트로 변환
    # RunnablePassthrough(): 체인의 입력을 그대로 저장
    | prompt
    | llm
    | StrOutputParser()
)
rag_chain.invoke("왜 논문 이름이 Search-R1이야? DeepSeek 모델이야?")
rag_chain_v2 = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}

    | RunnablePassthrough().assign(answer = prompt | llm | StrOutputParser())
)
rag_chain_v2.invoke("검색 과정에 RL을 어떻게 적용해?")
Retriever의 결과를 확인하여 질문이 주어지면 관련 이미지의 링크를 함께 전달하게 만들어 보겠습니다.   
해당 링크를 화면에 띄우는 방식으로, 답변과 함께 보여주는 것도 가능할 것 같습니다.
retriever = db.as_retriever(search_kwargs={"k": 10})
def format_docs(docs):
    return "\n\n---\n\n".join(['Content:' +doc.page_content for doc in docs])
    # join : 구분자를 기준으로 스트링 리스트를 하나의 스트링으로 연결

rag_chain = (
    {"context": translate_chain | retriever | format_docs, "question": RunnablePassthrough()}
    # retriever : question을 받아서 context 검색: document 반환
    # format_docs : document 형태를 받아서 텍스트로 변환
    # RunnablePassthrough(): 체인의 입력을 그대로 저장
    | prompt
    | llm
    | StrOutputParser()
)
from typing import List
from langchain_core.documents import Document
from PIL import Image
import matplotlib.pyplot as plt
import os

def show_images_with_caption(docs: List[Document]):
    """
    랭체인 Document 객체 목록에서 이미지와 해당 캡션을 순서대로 화면에 표시합니다.
    Document의 metadata['type']이 'image'인 경우에만 처리합니다.

    각 Document 객체는 다음을 포함해야 합니다:
    - metadata['source']: 이미지 파일의 경로 (문자열).
    - metadata['type']: 문서의 타입 (예: 'image', 'text' 등).
    - page_content: 이미지에 대한 캡션 (문자열).

    Args:
        docs: 랭체인 Document 객체의 목록.
    """
    if not docs:
        print("표시할 문서가 없습니다.")
        return

    image_count = 0
    for i, doc in enumerate(docs):
        # 문서 타입이 'image'인지 확인
        if doc.metadata.get('type') != 'image':
            # print(f"--- 문서 {i+1} (건너뜀) ---")
            # print(f"이 문서는 이미지가 아닙니다 (type: {doc.metadata.get('type', 'N/A')}).")
            # print("\n")
            continue

        image_count += 1
        image_path = 'mm_rag/'+ doc.metadata.get('source')
        caption = doc.page_content

        print(f"--- 이미지 {image_count} ---")
        print(f"캡션: {caption}")

        if image_path:
            if os.path.exists(image_path):
                try:
                    img = Image.open(image_path)

                    # Matplotlib을 사용하여 이미지 표시
                    plt.figure(figsize=(8, 6)) # 이미지 표시를 위한 그림 크기 설정
                    plt.imshow(img)
                    plt.axis('off') # 축 숨기기
                    plt.title(f"Image {image_count}")
                    plt.show() # 이미지를 화면에 표시
                except Exception as e:
                    print(f"오류: '{image_path}' 경로의 이미지를 표시할 수 없습니다. 오류: {e}")
            else:
                print(f"오류: 이미지 파일이 존재하지 않습니다: '{image_path}'")
        else:
            print("오류: Document 메타데이터에 'source' (이미지 경로)가 없습니다.")
        print("\n") # 각 이미지와 캡션 사이에 공백 추가

    if image_count == 0:
        print("표시할 이미지가 없습니다.")

docs = retriever.invoke(translate_chain.invoke("검색 과정에 RL을 어떻게 적용해?"))

show_images_with_caption(docs)
result = rag_chain.invoke("검색 과정에 RL을 어떻게 적용해?")
print(result)
show_images_with_caption(retriever.invoke("question"))
