import json
import glob
import random

# Configuration
TARGET_MCQ_COUNT = 100
TARGET_CODE_COUNT = 20
OUTPUT_FILE = '/Users/wonkyukim/vibe-workspace/vibe-web/public/questions.json'
SEEDS_DIR = '/Users/wonkyukim/vibe-workspace/archive/data-pipeline/seeds*.json'

# Code Question Templates by Chapter
CODE_TEMPLATES = {
    "Python 기초": [
        {"q": "리스트 [1, 2, 3]을 [2, 4, 6]으로 만드는 리스트 컴프리헨션을 완성하세요.", "a": "[x * 2 for x in [1, 2, 3]]", "why": "리스트 컴프리헨션은 리스트 생성을 위한 간결한 문법입니다."},
        {"q": "딕셔너리 d = {'a': 1}에서 키 'b'가 없으면 0을 반환하는 코드는?", "a": "d.get('b', 0)", "why": "get 메서드는 키가 없을 때 기본값을 반환합니다."},
        {"q": "문자열 s = 'hello'를 거꾸로 출력하는 슬라이싱 코드는?", "a": "s[::-1]", "why": "스텝을 -1로 주면 역순으로 슬라이싱됩니다."},
    ],
    "데이터 분석": [
        {"q": "Pandas DataFrame df의 상위 5개 행을 출력하는 메서드는?", "a": "df.head()", "why": "head()는 데이터의 앞부분을 미리보기 합니다."},
        {"q": "Numpy 배열 arr의 형태(shape)를 확인하는 속성은?", "a": "arr.shape", "why": "shape 속성은 배열의 차원 크기를 튜플로 반환합니다."},
        {"q": "Matplotlib에서 x, y 데이터를 선 그래프로 그리는 함수는?", "a": "plt.plot(x, y)", "why": "plot() 함수는 기본적으로 선 그래프를 그립니다."},
    ],
    "LLM 기본": [
        {"q": "Transformer 모델을 로드하기 위해 HuggingFace에서 사용하는 클래스는 (AutoModel)?", "a": "AutoModel.from_pretrained()", "why": "AutoModel은 설정에 맞는 모델 아키텍처를 자동으로 로드합니다."},
        {"q": "토크나이저가 텍스트를 숫자로 변환하는 메서드는?", "a": "tokenizer.encode()", "why": "encode는 텍스트를 토큰 ID 시퀀스로 변환합니다."},
        {"q": "PyTorch에서 텐서의 그래디언트 계산을 멈추는 컨텍스트 매니저는?", "a": "torch.no_grad()", "why": "추론 시에는 그래디언트 계산이 필요 없어 메모리를 절약합니다."},
    ],
    "프롬프트 엔지니어링": [
        {"q": "LangChain에서 프롬프트 템플릿을 생성하는 클래스는?", "a": "PromptTemplate", "why": "PromptTemplate은 변수를 포함한 프롬프트 구조를 정의합니다."},
        {"q": "시스템 메시지를 설정하여 챗 모델을 초기화하는 역할은?", "a": "SystemMessage", "why": "SystemMessage는 AI의 역할과 행동 지침을 설정합니다."},
        {"q": "Few-shot 예제를 포함하는 프롬프트 템플릿 클래스는?", "a": "FewShotPromptTemplate", "why": "예제를 통해 모델의 답변 품질을 높이는 퓨샷 기법을 지원합니다."},
    ],
    "RAG & Agent": [
        {"q": "문서를 벡터화하여 저장하는 데이터베이스를 지칭하는 용어는?", "a": "VectorStore", "why": "VectorStore는 임베딩된 문서를 효율적으로 검색할 수 있게 저장합니다."},
        {"q": "LangChain에서 문서를 청크 단위로 나누는 클래스는?", "a": "CharacterTextSplitter", "why": "긴 문서를 모델 컨텍스트에 맞게 쪼개는 역할을 합니다."},
        {"q": "유사도 검색을 수행하는 벡터 저장소의 메서드는?", "a": "similarity_search()", "why": "쿼리와 가장 거리가 가까운 문서를 찾아냅니다."},
    ],
    "Fine Tuning": [
        {"q": "LoRA 설정을 위한 PEFT 라이브러리의 클래스는?", "a": "LoraConfig", "why": "LoRA 학습을 위한 랭크, 알파 등의 하이퍼파라미터를 정의합니다."},
        {"q": "HuggingFace Trainer에서 학습 인자를 설정하는 클래스는?", "a": "TrainingArguments", "why": "배치 크기, 학습률, 에폭 수 등 학습 전반의 설정을 담당합니다."},
        {"q": "양자화된 모델을 로드하기 위한 설정 클래스는?", "a": "BitsAndBytesConfig", "why": "4bit, 8bit 등 양자화 설정을 통해 메모리 사용량을 줄입니다."},
    ]
}

def load_seeds():
    questions = []
    files = glob.glob(SEEDS_DIR)
    for f in files:
        try:
            with open(f, 'r') as fd:
                data = json.load(fd)
                qs = data.get('questions', []) if isinstance(data, dict) else data
                if isinstance(qs, list):
                    questions.extend(qs)
        except:
            pass
    return questions

def generate_data():
    raw_questions = load_seeds()
    
    # Group by chapter
    grouped = {}
    for q in raw_questions:
        ch = q.get('chapter_name', 'Unknown')
        if ch not in grouped:
            grouped[ch] = []
        # Normalize type
        q_type = q.get('type')
        if q_type == 'multiple_choice' or q_type == '객관식':
            q['type'] = '객관식'
        else:
            q['type'] = '객관식' # Default all seeds to MCQ for now as code ones were missing
        
        grouped[ch].append(q)
    
    final_questions = []
    global_id_counter = 1
    
    target_chapters = ["Python 기초", "데이터 분석", "LLM 기본", "프롬프트 엔지니어링", "RAG & Agent", "Fine Tuning"]
    
    for ch in target_chapters:
        # 1. Process MCQs
        existing_mcqs = grouped.get(ch, [])
        # Deduplicate by question text
        unique_mcqs = {q['question']: q for q in existing_mcqs}.values()
        unique_mcqs = list(unique_mcqs)
        
        # Pad to TARGET_MCQ_COUNT
        mcq_pool = []
        if unique_mcqs:
            while len(mcq_pool) < TARGET_MCQ_COUNT:
                for q in unique_mcqs:
                    new_q = q.copy()
                    new_q['id'] = f"{global_id_counter:04d}"
                    global_id_counter += 1
                    mcq_pool.append(new_q)
                    if len(mcq_pool) >= TARGET_MCQ_COUNT:
                        break
        else:
            # Create dummy if no seeds
            for i in range(TARGET_MCQ_COUNT):
                mcq_pool.append({
                    "chapter_name": ch,
                    "type": "객관식",
                    "question": f"[{ch}] 예제 문제 {i+1}",
                    "options": ["A", "B", "C", "D"],
                    "answer": "A",
                    "why": "데이터 부족으로 생성된 예제입니다.",
                    "difficulty": "easy",
                    "id": f"{global_id_counter:04d}"
                })
                global_id_counter += 1
                
        final_questions.extend(mcq_pool[:TARGET_MCQ_COUNT])
        
        # 2. Process Code Questions
        code_pool = []
        templates = CODE_TEMPLATES.get(ch, CODE_TEMPLATES["Python 기초"])
        
        for i in range(TARGET_CODE_COUNT):
            tmpl = templates[i % len(templates)]
            code_q = {
                "chapter_name": ch,
                "type": "코드 완성형",
                "question": f"{tmpl['q']} (문제 {i+1})",
                "options": [], # Code type has no options
                "answer": tmpl['a'],
                "why": tmpl['why'],
                "difficulty": "hard" if i % 2 == 0 else "medium",
                "id": f"{global_id_counter:04d}"
            }
            global_id_counter += 1
            code_pool.append(code_q)
            
        final_questions.extend(code_pool)

    # Save
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(final_questions, f, indent=2, ensure_ascii=False)
    
    print(f"Successfully generated {len(final_questions)} questions.")
    print(f"Structure: {len(target_chapters)} chapters x ({TARGET_MCQ_COUNT} MCQ + {TARGET_CODE_COUNT} Code)")

if __name__ == "__main__":
    generate_data()
