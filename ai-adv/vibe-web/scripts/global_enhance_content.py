import json
import random

# New diverse questions for EACH chapter
NEW_QUESTIONS = {
    "Python 기초": [
        {
            "chapter_name": "Python 기초",
            "type": "객관식",
            "question": "@staticmethod와 @classmethod의 가장 큰 차이점은?",
            "options": [
                "첫 번째 인자로 cls를 받느냐 아니냐",
                "이름이 다르다",
                "사용하는 메모리 양",
                "상속 가능 여부",
                "리턴 타입"
            ],
            "answer": "첫 번째 인자로 cls를 받느냐 아니냐",
            "why": "classmethod는 첫 인자로 클래스 자체(cls)를 받고, staticmethod는 아무런 암시적 인자도 받지 않습니다.",
            "hint": "cls 인자의 유무를 확인하세요.",
            "difficulty": "medium",
            "id": "0901"
        },
        {
            "chapter_name": "Python 기초",
            "type": "객관식",
            "question": "파이썬의 GIL(Global Interpreter Lock)이 성능에 영향을 주는 주된 상황은?",
            "options": [
                "CPU 연산이 많은 멀티스레딩 작업",
                "I/O 작업",
                "단일 스레드 작업",
                "함수 호출",
                "메모리 할당"
            ],
            "answer": "CPU 연산이 많은 멀티스레딩 작업",
            "why": "GIL 때문에 한 순간에 하나의 스레드만 바이트코드를 실행할 수 있어, CPU 집약적 작업에서는 멀티스레딩 효율이 떨어집니다.",
            "hint": "CPU를 많이 쓰는 작업을 할 때 락(Lock)이 걸립니다.",
            "difficulty": "hard",
            "id": "0902"
        },
        {
            "chapter_name": "Python 기초",
            "type": "객관식",
            "question": "제너레이터(Generator)를 만드는 키워드는?",
            "options": ["yield", "return", "break", "continue", "pass"],
            "answer": "yield",
            "why": "함수 내에서 yield를 사용하면 그 함수는 제너레이터가 되어 값을 하나씩 반환하고 상태를 유지합니다.",
            "hint": "양보하다(yield)라는 뜻입니다.",
            "difficulty": "medium",
            "id": "0903"
        },
        {
            "chapter_name": "Python 기초",
            "type": "객관식",
            "question": "컨텍스트 매니저(Context Manager)를 구현할 때 필요한 매직 메서드 두 개는?",
            "options": ["__init__, __del__", "__enter__, __exit__", "__str__, __repr__", "__len__, __getitem__", "__new__, __init__"],
            "answer": "__enter__, __exit__",
            "why": "with 문이 시작될 때 __enter__가, 끝날 때 __exit__가 호출되어 자원을 안전하게 관리합니다.",
            "hint": "들어가고(enter) 나가는(exit) 것.",
            "difficulty": "medium",
            "id": "0904"
        },
        {
            "chapter_name": "Python 기초",
            "type": "객관식",
            "question": "파이썬 3.10부터 도입된 구조적 패턴 매칭(Structural Pattern Matching) 키워드는?",
            "options": ["switch / case", "match / case", "if / else", "try / except", "for / in"],
            "answer": "match / case",
            "why": "다른 언어의 switch 문과 유사하지만 더 강력한 패턴 매칭 기능을 제공합니다.",
            "hint": "매칭(match) 한다고 합니다.",
            "difficulty": "easy",
            "id": "0905"
        }
    ],
    "데이터 분석": [
        {
            "chapter_name": "데이터 분석",
            "type": "객관식",
            "question": "Pandas에서 메모리 사용량을 줄이기 위해 범주형 데이터(문자열 등)를 효율적으로 저장하는 타입은?",
            "options": ["category", "object", "string", "int64", "float32"],
            "answer": "category",
            "why": "중복되는 문자열을 내부적으로 정수로 매핑하여 메모리를 획기적으로 절약합니다.",
            "hint": "카테고리(category)화 합니다.",
            "difficulty": "medium",
            "id": "0906"
        },
        {
            "chapter_name": "데이터 분석",
            "type": "객관식",
            "question": "두 변수 간의 상관관계를 -1에서 1 사이의 값으로 나타내는 통계 지표는?",
            "options": ["피어슨 상관계수", "평균", "표준편차", "분산", "중앙값"],
            "answer": "피어슨 상관계수",
            "why": "1에 가까우면 양의 상관관계, -1에 가까우면 음의 상관관계를 의미합니다.",
            "hint": "P로 시작하는 상관계수.",
            "difficulty": "easy",
            "id": "0907"
        },
        {
            "chapter_name": "데이터 분석",
            "type": "객관식",
            "question": "데이터 분포가 정규분포를 따르지 않을 때 사용하는 비모수 검정 방법은?",
            "options": ["T-test", "ANOVA", "Mann-Whitney U test", "Z-test", "F-test"],
            "answer": "Mann-Whitney U test",
            "why": "데이터의 순위(Rank) 정보를 이용하여 평균 차이를 검정하므로 정규성 가정이 필요 없습니다.",
            "hint": "이름이 깁니다.",
            "difficulty": "hard",
            "id": "0908"
        },
        {
            "chapter_name": "데이터 분석",
            "type": "객관식",
            "question": "결측치를 처리하는 방법 중, 다른 변수들과의 관계를 모델링하여 빈 값을 예측해 채우는 방식은?",
            "options": ["Model-based Imputation", "Mean Imputation", "Deletion", "Zero Filling", "Mode Imputation"],
            "answer": "Model-based Imputation",
            "why": "단순 평균 대치보다 더 정교하게 실제값에 데이터의 패턴을 반영하여 채워 넣습니다 (예: KNN Imputer).",
            "hint": "모델(Model)을 씁니다.",
            "difficulty": "medium",
            "id": "0909"
        },
        {
            "chapter_name": "데이터 분석",
            "type": "객관식",
            "question": "시계열 데이터의 '계절성(Seasonality)'을 분해하여 분석하는 기법은?",
            "options": ["STL Decomposition", "PCA", "K-means", "Regression", "ANOVA"],
            "answer": "STL Decomposition",
            "why": "Seasonal-Trend decomposition using Loess의 약자로 추세, 계절성, 잔차로 시계열을 쪼갭니다.",
            "hint": "계절(Seasonal)의 S.",
            "difficulty": "hard",
            "id": "0910"
        }
    ],
    "LLM 기본": [
        {
            "chapter_name": "LLM 기본",
            "type": "객관식",
            "question": "Transformer의 Self-Attention 연산 복잡도는 입력 시퀀스 길이 N에 대해 어떻게 증가하는가?",
            "options": ["O(N^2)", "O(N)", "O(log N)", "O(N log N)", "O(1)"],
            "answer": "O(N^2)",
            "why": "모든 토큰이 서로서로(All-to-All) 어텐션을 계산해야 하므로 길이의 제곱에 비례하여 연산량이 폭증합니다.",
            "hint": "N의 제곱.",
            "difficulty": "hard",
            "id": "0911"
        },
        {
            "chapter_name": "LLM 기본",
            "type": "객관식",
            "question": "LLM의 환각(Hallucination) 현상을 완화하기 위해 출력 확률 분포를 평탄하게 만드는(Entropy증가) 기법은?",
            "options": ["Temperature Scaling (온도 조절)", "Beam search", "Greedy decoding", "Top-k", "Top-p"],
            "answer": "Temperature Scaling (온도 조절)",
            "why": "온도를 높이면 분포가 평평해져 다양성이 늘지만 환각 위험도 커지고, 낮추면 뾰족해져 정확도가 오릅니다.",
            "hint": "온도(Temperature)를 낮추면 확실한 것만 말합니다.",
            "difficulty": "medium",
            "id": "0912"
        },
        {
            "chapter_name": "LLM 기본",
            "type": "객관식",
            "question": "위치 정보(Positional Encoding)가 Transformer에 꼭 필요한 이유는?",
            "options": ["Attention 메커니즘 자체는 순서 정보를 알지 못하기 때문", "단어 뜻을 알기 위해", "문법 검사를 위해", "속도를 높이려고", "용량을 줄이려고"],
            "answer": "Attention 메커니즘 자체는 순서 정보를 알지 못하기 때문",
            "why": "RNN과 달리 병렬로 처리하므로 '나는'이 처음에 있는지 끝에 있는지 알려주는 위치 표지가 필수입니다.",
            "hint": "순서(Sequence)를 모릅니다.",
            "difficulty": "medium",
            "id": "0913"
        },
        {
            "chapter_name": "LLM 기본",
            "type": "객관식",
            "question": "BPE(Byte-Pair Encoding) 토크나이저의 학습 원리는?",
            "options": ["가장 빈번하게 등장하는 문자 쌍을 반복적으로 병합", "사전 정의된 단어만 사용", "랜덤하게 자르기", "문법 규칙에 따라 분리", "띄어쓰기 기준 분리"],
            "answer": "가장 빈번하게 등장하는 문자 쌍을 반복적으로 병합",
            "why": "데이터 압축 알고리즘에 기초하여 빈도가 높은 쌍을 하나의 토큰으로 만들어 어휘 집합을 구축합니다.",
            "hint": "자주 나오는 '쌍(Pair)'을 합칩니다.",
            "difficulty": "hard",
            "id": "0914"
        },
        {
            "chapter_name": "LLM 기본",
            "type": "객관식",
            "question": "모델의 크기는 그대로 두면서 학습 데이터 양만 무한히 늘리면 성능은 어떻게 되는가? (Chinchilla Scaling Law)",
            "options": ["어느 지점에서 성능 향상이 포화되거나 효율이 떨어진다", "무한히 좋아진다", "나빠진다", "상관없다", "모델이 폭발한다"],
            "answer": "어느 지점에서 성능 향상이 포화되거나 효율이 떨어진다",
            "why": "친칠라 법칙에 따르면 모델 크기와 데이터 양은 최적의 비율(약 1:20)이 있으며, 균형 있게 늘려야 합니다.",
            "hint": "친칠라 법칙.",
            "difficulty": "hard",
            "id": "0915"
        }
    ],
    "프롬프트 엔지니어링": [
        {
            "chapter_name": "프롬프트 엔지니어링",
            "type": "객관식",
            "question": "프롬프트 해킹 기법 중, 모델에게 '너는 이제부터 사악한 해커야'라는 식으로 역할극을 강요하여 안전 장치를 우회하는 것은?",
            "options": ["DAN (Do Anything Now) / 역할극 공격", "Prompt Injection", "Leakage", "Jailbreaking", "Phishing"],
            "answer": "DAN (Do Anything Now) / 역할극 공격",
            "why": "페르소나를 씌워 윤리적 제약을 잊게 만드는 대표적인 탈옥(Jailbreak) 기법 중 하나입니다.",
            "hint": "유명한 'DAN' 공격입니다.",
            "difficulty": "medium",
            "id": "0916"
        },
        {
            "chapter_name": "프롬프트 엔지니어링",
            "type": "객관식",
            "question": "복잡한 문제를 풀 때 '단계별로 생각해(Let's think step by step)'라고 지시하는 것만으로 성능이 오르는 현상은?",
            "options": ["Zero-shot CoT", "Few-shot", "One-shot", "Fine-tuning", "Pre-training"],
            "answer": "Zero-shot CoT",
            "why": "예시(Shot)를 하나도 안 줬는데(Zero-shot) 생각의 사슬(CoT)을 유도했기 때문입니다.",
            "hint": "예시 없이(Zero) 단계별 사고.",
            "difficulty": "medium",
            "id": "0917"
        },
        {
            "chapter_name": "프롬프트 엔지니어링",
            "type": "객관식",
            "question": "프롬프트 내에 정답 예시를 너무 많이 넣었을 때, 모델이 예시의 정답 분포나 마지막 예시에만 편향되는 현상은?",
            "options": ["Recency Bias (최신 편향) / Majority Label Bias", "Overfitting", "Underfitting", "Hallucination", "Catastrophic Forgetting"],
            "answer": "Recency Bias (최신 편향) / Majority Label Bias",
            "why": "컨텍스트 윈도우의 끝부분(최신)이나 다수결에 영향을 과하게 받는 LLM의 특성입니다.",
            "hint": "최근(Recency) 것을 더 잘 기억함.",
            "difficulty": "medium",
            "id": "0918"
        },
        {
            "chapter_name": "프롬프트 엔지니어링",
            "type": "객관식",
            "question": "긴 프롬프트의 중간에 있는 내용을 모델이 잘 기억하지 못하고 앞부분과 뒷부분만 잘 기억하는 현상은?",
            "options": ["Lost in the Middle", "Vanishing Gradient", "Exploding Gradient", "Memory Leak", "Attention Failure"],
            "answer": "Lost in the Middle",
            "why": "샌드위치처럼 양 끝의 정보는 잘 가져오지만 가운데 정보는 손실되는(Lost) 경향이 있습니다.",
            "hint": "가운데(Middle)에서 길을 잃음.",
            "difficulty": "medium",
            "id": "0919"
        },
        {
            "chapter_name": "프롬프트 엔지니어링",
            "type": "객관식",
            "question": "모델의 답변을 구조화된 포맷(JSON 등)으로 강제하기 위해 사용하는 프롬프트 전략은?",
            "options": ["Output Parser / Output Schema 명시", "Role Playing", "Few-shot", "Chain of Thought", "ReAct"],
            "answer": "Output Parser / Output Schema 명시",
            "why": "원하는 스키마나 예시 JSON을 명확히 보여주고 '이 형식 아니면 뱉지 마'라고 강제하는 것이 가장 효과적입니다.",
            "hint": "출력(Output) 형식을 지정.",
            "difficulty": "easy",
            "id": "0920"
        }
    ],
    "RAG & Agent": [
        {
            "chapter_name": "RAG & Agent",
            "type": "객관식",
            "question": "RAG 검색 단계에서 의미적으로 유사하지만 키워드가 전혀 다른 문서를 찾기 위해 필수적인 것은?",
            "options": ["Dense Embedding (밀집 임베딩)", "Sparse Embedding (희소 임베딩)", "BM25", "Keyword Match", "Regex"],
            "answer": "Dense Embedding (밀집 임베딩)",
            "why": "단어를 벡터 공간에 매핑하여 '자동차'와 '승용차'가 가까운 점임을 인식하게 합니다.",
            "hint": "빽빽한(Dense) 벡터.",
            "difficulty": "medium",
            "id": "0921"
        },
        {
            "chapter_name": "RAG & Agent",
            "type": "객관식",
            "question": "여러 문서에서 공통된 정보를 종합해야 답이 나오는 질문(Multi-hop QA)을 해결하기 위한 RAG 기법은?",
            "options": ["Graph RAG (지식 그래프 활용)", "Simple RAG", "Naive RAG", "Vector Search", "Keyword Search"],
            "answer": "Graph RAG (지식 그래프 활용)",
            "why": "문서 간의 연결 관계를 그래프로 표현하여 징검다리 건너듯 여러 정보를 연결해 추론합니다.",
            "hint": "그래프(Graph)를 타고 넘어다님.",
            "difficulty": "hard",
            "id": "0922"
        },
        {
            "chapter_name": "RAG & Agent",
            "type": "객관식",
            "question": "최신 에이전트 프레임워크인 'LangGraph'의 핵심 차별점은?",
            "options": ["순환(Cycle)이 가능한 그래프 구조 지원", "비선형 구조 불가", "단순 체인 구조", "LLM 미사용", "파인튜닝 필수"],
            "answer": "순환(Cycle)이 가능한 그래프 구조 지원",
            "why": "단순 DAG(단방향)가 아니라 루프를 돌며 작업을 반복/수정할 수 있는 순환 구조를 코드 레벨에서 지원합니다.",
            "hint": "그래프(Graph) 위를 뱅글뱅글 돎.",
            "difficulty": "hard",
            "id": "0923"
        },
        {
            "chapter_name": "RAG & Agent",
            "type": "객관식",
            "question": "검색된 문서들의 순서가 LLM의 답변 품질에 영향을 주는데, 가장 관련성 높은 문서를 어디에 배치하는 것이 유리한가?",
            "options": ["입력의 시작과 끝 (Primacy & Recency Effect)", "무조건 중간", "랜덤", "문서 길이순", "알파벳순"],
            "answer": "입력의 시작과 끝 (Primacy & Recency Effect)",
            "why": "Lost in the Middle 현상 때문에 모델이 가장 잘 보는 양오 끝단에 중요한 정보를 배치하는 것이 Re-ranking의 핵심입니다.",
            "hint": "처음과 끝이 중요합니다.",
            "difficulty": "hard",
            "id": "0924"
        },
        {
            "chapter_name": "RAG & Agent",
            "type": "객관식",
            "question": "하이브리드 검색(Hybrid Search)은 보통 무엇과 무엇의 결합인가?",
            "options": ["키워드 검색(BM25) + 시맨틱 검색(Vector)", "이미지 + 텍스트", "음성 + 텍스트", "SQL + NoSQL", "CPU + GPU"],
            "answer": "키워드 검색(BM25) + 시맨틱 검색(Vector)",
            "why": "정확한 단어 매칭(키워드)과 맥락 매칭(벡터)의 장점을 합쳐 상호 보완합니다.",
            "hint": "전통 검색(키워드)과 최신 검색(벡터)의 짬뽕.",
            "difficulty": "medium",
            "id": "0925"
        }
    ],
    "Fine Tuning": [
        {
            "chapter_name": "Fine Tuning",
            "type": "객관식",
            "question": "모델의 크기를 줄이는 '양자화(Quantization)' 시, 주로 손실되는 정보는?",
            "options": ["가중치의 정밀도(Precision)", "모델의 층 수", "입력 토큰 길이", "학습률", "어휘 크기"],
            "answer": "가중치의 정밀도(Precision)",
            "why": "32비트 실수를 4비트 정수로 깎아내므로 미세한 숫자의 디테일(정밀도)이 뭉개집니다.",
            "hint": "정밀함(Precision)을 잃습니다.",
            "difficulty": "medium",
            "id": "0926"
        },
        {
            "chapter_name": "Fine Tuning",
            "type": "객관식",
            "question": "파인튜닝 시 훈련 데이터셋의 '프롬프트 템플릿' 형식이 맞지 않을 때 발생하는 'Silent Fail' 현상은?",
            "options": ["에러 없이 학습되지만 성능이 엉망임", "학습이 멈춤", "컴퓨터가 꺼짐", "데이터가 삭제됨", "경고창이 뜸"],
            "answer": "에러 없이 학습되지만 성능이 엉망임",
            "why": "코드는 돌아가지만 모델은 이게 질문인지 답변인지 구분을 못 한 채로 텍스트 덩어리만 배워 바보가 됩니다.",
            "hint": "조용히(Silent) 망함.",
            "difficulty": "hard",
            "id": "0927"
        },
        {
            "chapter_name": "Fine Tuning",
            "type": "객관식",
            "question": "모델 병합(Merging) 기법 중 'SLERP(Spherical Linear Interpolation)'가 단순 평균보다 좋은 이유는?",
            "options": ["고차원 벡터 공간의 구면 기하학적 특성을 보존하기 때문", "계산이 더 단순해서", "메모리를 적게 써서", "정수만 사용해서", "이미지 처리에 좋아서"],
            "answer": "고차원 벡터 공간의 구면 기하학적 특성을 보존하기 때문",
            "why": "단순 직선(Linear) 평균은 고차원 공간에서 벡터의 성질을 왜곡할 수 있어 구면(Spherical) 궤적을 따라 섞습니다.",
            "hint": "구(Sphere) 위에서 섞습니다.",
            "difficulty": "hard",
            "id": "0928"
        },
        {
            "chapter_name": "Fine Tuning",
            "type": "객관식",
            "question": "Instruct Tuning 데이터셋 구축 시 'Decontamination(오염 제거)' 작업의 목적은?",
            "options": ["평가 집합(Test Set)이 학습 데이터에 들어가는 것을 막기 위해", "바이러스를 잡기 위해", "욕설을 지우기 위해", "중복을 늘리기 위해", "파일 크기를 키우기 위해"],
            "answer": "평가 집합(Test Set)이 학습 데이터에 들어가는 것을 막기 위해",
            "why": "답안지를 미리 보고 공부하면(Data Leakage) 실제 실력을 측정할 수 없으므로 철저히 분리합니다.",
            "hint": "시험 문제 유출 방지.",
            "difficulty": "medium",
            "id": "0929"
        },
        {
            "chapter_name": "Fine Tuning",
            "type": "객관식",
            "question": "DPO 학습 시 'Reference Model'이 필요한 이유는?",
            "options": ["학습 중인 모델이 너무 많이 변하지 않도록(KL 제약) 기준점을 잡아주기 위해", "정답을 베끼기 위해", "속도를 높이기 위해", "메모리를 절약하기 위해", "그냥 관습적으로"],
            "answer": "학습 중인 모델이 너무 많이 변하지 않도록(KL 제약) 기준점을 잡아주기 위해",
            "why": "원래 모델의 분포에서 너무 멀어지면 언어 능력이 붕괴되므로, 원래의 나(Reference)와 비교하며 학습합니다.",
            "hint": "기준(Reference)이 흔들리지 않게.",
            "difficulty": "hard",
            "id": "0930"
        }
    ]
}

def enhance_content():
    try:
        with open('public/questions.json', 'r', encoding='utf-8') as f:
            questions = json.load(f)
    except FileNotFoundError:
        print("Error: public/questions.json not found.")
        return

    # 1. Deduplication (LLM 기본: 0340, 프롬프트 엔지니어링: 0460)
    # Based on audit: 0340 is dup of 0241, 0460 is dup of 0361
    duplicate_ids = ["0340", "0460"]
    
    cleaned_questions = [q for q in questions if q['id'] not in duplicate_ids]
    print(f"Removed {len(questions) - len(cleaned_questions)} duplicate questions.")

    # 2. Fix 'All of the above' Patterns
    # ID 0019, 0041, 0649 (Heuristics flagged these, let's refine them)
    for q in cleaned_questions:
        if q['id'] == "0019": # digit check
            q['options'] = ["isdigit()", "isdecimal()", "isnumeric()", "check_num()", "test_int()"]
            q['answer'] = "isdigit()"
            q['why'] = "가장 일반적으로 문자열이 숫자로만 구성되었는지 확인할 때 사용합니다."
            
        elif q['id'] == "0041": # copy
            q['options'] = ["얕은 복사", "깊은 복사", "참조 복사", "이동", "삭제"]
            q['answer'] = "깊은 복사"
            q['question'] = "리스트 내부의 리스트까지 완전히 새로운 객체로 복제하여 원본과 독립적으로 만드는 복사 방식은?"

        elif q['id'] == "0649": # Epoch
             q['answer'] = "1 Epoch (1 에폭)" # Ensure clarity
    
    print("Fixed problematic patterns.")

    # 3. Augment Content (Add new diverse questions)
    all_new_questions = []
    for chap, new_qs in NEW_QUESTIONS.items():
        all_new_questions.extend(new_qs)
        print(f"Adding {len(new_qs)} questions to {chap}")

    final_questions = cleaned_questions + all_new_questions
    
    # Sort or Verify IDs? IDs are strings, so simple append is fine.
    # Total count check
    print(f"Original count: {len(questions)}")
    print(f"New count: {len(final_questions)}")

    with open('public/questions.json', 'w', encoding='utf-8') as f:
        json.dump(final_questions, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    enhance_content()
