
chapter_name = "LLM 기본"

questions = []

# --- 100 MCQs ---
# Unique conceptual and practical questions based on 3.md

mcq_data = [
    # 1. LLM과 트랜스포머 (1-25)
    ("전통적인 RNN/LSTM 모델이 긴 문장을 처리할 때 겪었던 '장기 의존성(Long-term Dependency)' 문제에 대한 설명으로 옳은 것은?", ["문장이 길어질수록 앞부분의 정보를 소실하거나 잊어버리는 현상이다.", "문장의 길이에 상관없이 항상 동일한 성능을 유지하는 특징이다.", "문장 전체의 통계적 빈도만 고려하여 문맥을 파악하지 못하는 것이다.", "컴퓨터의 메모리 용량이 부족하여 프로그램이 꺼지는 현상이다.", "단어를 무작위로 섞어서 학습하기 때문에 발생하는 오류이다."], "문장이 길어질수록 앞부분의 정보를 소실하거나 잊어버리는 현상이다.", "RNN은 순차적으로 데이터를 처리하므로 멀리 떨어진 단어 간의 관계를 학습하기 어려운 한계가 있었습니다. 이 한계를 극복하기 위해 LSTM이 등장했으나 근본적 해결책은 트랜스포머의 어텐션 메커니즘이었습니다.", "RNN 한계", "3001", "easy"),
    ("2017년 구글이 발표한 'Attention is All You Need' 논문의 핵심적인 기여는?", ["RNN의 성능을 2배 높이는 새로운 방식을 제안했다.", "이미지 처리를 위한 CNN 아키텍처를 완성했다.", "순차 처리 대신 병렬 처리가 가능한 '트랜스포머' 구조를 제시했다.", "데이터 보안을 위한 새로운 암호화 알고리즘을 발표했다.", "파이썬의 실행 속도를 높이는 인터프리터를 개발했다."], "순차 처리 대신 병렬 처리가 가능한 '트랜스포머' 구조를 제시했다.", "트랜스포머는 어텐션 메커니즘을 전면에 내세워 문맥 파악 능력을 비약적으로 높였습니다. 이 논문 이후 NLP 연구의 패러다임이 완전히 바뀌었으며, GPT와 BERT 등 모든 현대 LLM의 기반이 되었습니다.", "트랜스포머 탄생", "3002", "easy"),
    ("트랜스포머의 '어텐션(Attention)' 메커니즘이 수행하는 가장 주된 작업은?", ["단어의 글자 수를 세어 가장 긴 단어를 찾는다.", "문맥상 어떤 단어들이 서로 밀접한 관계가 있는지 가중치를 계산한다.", "파일의 용량을 줄이기 위해 텍스트를 압축한다.", "오타를 실시간으로 교정하여 문법을 맞춘다.", "단어를 다른 나라 언어로 즉시 번역한다."], "문맥상 어떤 단어들이 서로 밀접한 관계가 있는지 가중치를 계산한다.", "어텐션은 특정 단어를 이해할 때 문장 내 다른 단어들을 얼마나 참고할지 점수를 매깁니다. 예를 들어 '그것'이라는 단어가 실제로 무엇을 지칭하는지 문장의 다른 단어와의 연관성으로 파악합니다.", "어텐션 원리", "3003", "medium"),
    ("트랜스포머 아키텍처 중 '인코더(Encoder)'의 특징에 대한 설명으로 옳은 것은?", ["주로 문장을 새로 생성(Generate)하는 작업에 최적화되어 있다.", "입력 문장을 수치화하여 그 의미를 압축하고 이해하는 데 강점이 있다.", "GPT 모델의 핵심 구조로 사용된다.", "다음에 올 단어를 하나씩 예측하며 결과물을 내놓는다.", "오직 한국어 분석에만 사용 가능한 특수 구조이다."], "입력 문장을 수치화하여 그 의미를 압축하고 이해하는 데 강점이 있다.", "인코더는 문맥의 상호 의미를 파악하는 데 특화되어 있으며, BERT가 대표적인 인코더 기반 모델입니다. 분류, 감성 분석, QA 등 입력 문맥 이해가 필요한 태스크에 강합니다.", "인코더 특징", "3004", "medium"),
    ("현대 LLM(GPT 등)이 주로 채택하고 있는 '디코더 전용(Decoder-only)' 구조의 특징은?", ["문장의 의미를 이해하기만 할 뿐, 새로운 답변을 만들지는 못한다.", "앞서 생성된 단어들을 바탕으로 다음에 올 단어를 확률적으로 예측한다.", "인코더보다 항상 크기가 작고 성능이 낮다.", "반드시 이미지 데이터와 함께 학습되어야만 동작한다.", "입력 데이터의 순서를 고려하지 않고 무작위로 답변을 내놓는다."], "앞서 생성된 단어들을 바탕으로 다음에 올 단어를 확률적으로 예측한다.", "디코더는 이전 토큰들의 맥락을 유지하며 다음 토큰을 생성하는 생성 작업(Generation)의 표준입니다. Causal LM이라고도 부르며, 미래 토큰을 보지 못하게 하는 Causal Masking이 핵심입니다.", "디코더 특징", "3005", "medium"),
    ("GPT 모델이 문장을 생성할 때 단어를 하나씩 내뱉는 방식은?", ["한꺼번에 문장 전체를 사진처럼 찍듯이 생성한다.", "다음 토큰을 예측하며 순차적으로 한 단어씩 생성한다.", "문장의 마지막 단어부터 거꾸로 생성한다.", "사전에 정의된 문장 템플릿에 단어만 끼워 넣는다.", "사용자가 엔터를 칠 때까지 기다렸다가 한 번에 대답한다."], "다음 토큰을 예측하며 순차적으로 한 단어씩 생성한다.", "Auto-regressive(자기 회귀) 방식으로, 이전 결과가 다음 입력이 되어 문장을 완성해 나갑니다. 각 스텝마다 전체 어휘에 대한 확률 분포를 계산하고 샘플링합니다.", "생성 메커니즘", "3006", "medium"),
    ("트랜스포머에서 단어의 순서(위치) 정보를 모델에게 전달하기 위해 사용하는 기법은?", ["Sequence Count", "Positional Encoding", "Index Mapping", "Word Order Tagging", "Linear Alignment"], "Positional Encoding", "트랜스포머는 데이터를 한꺼번에 처리하므로, 단어의 위치 구분을 위해 사인/코사인 함수 기반 인코딩 값을 임베딩에 더해줍니다.", "위치 인코딩", "3007", "easy"),
    ("모델 아키텍처 중 BERT는 주로 ( A ) 방식이며, GPT는 주로 ( B ) 방식이다. ( )에 들어갈 적절한 조합은?", ["A: 디코더, B: 인코더", "A: 인코더, B: 디코더", "A: CNN, B: RNN", "A: 임베딩, B: 토크나이징", "A: 지도학습, B: 비지도학습"], "A: 인코더, B: 디코더", "BERT는 Masked Language Modeling으로 양방향 문맥을 이해(인코더)하며, GPT는 다음 토큰 예측으로 텍스트를 생성(디코더)합니다. 태스크 설계 시 두 구조의 차이를 명확히 구분해야 합니다.", "모델 구분", "3008", "hard"),
    ("트랜스포머 구조에서 여러 개의 어텐션을 동시에 수행하여 다양한 관점을 학습하는 기술은?", ["Single-Line Attention", "Parallel Attention", "Multi-Head Attention", "Complex Attention", "Super Attention"], "Multi-Head Attention", "여러 '헤드'를 통해 문장의 다양한 문맥적 특징을 동시에 추출합니다. 각 헤드는 독립적인 Query/Key/Value 행렬을 학습하며, 결과를 concat하여 표현력을 높입니다.", "Multi-Head", "3009", "hard"),
    ("딥러닝 모델의 층이 깊어질 때 학습이 잘 안 되는 문제를 해결하기 위해, 입력값을 뒤쪽 층에 직접 전달하는 구조는?", ["Skip Layer", "Back Link", "Residual Connection (잔차 연결)", "Fast Track", "Data Tunnel"], "Residual Connection (잔차 연결)", "입력 정보를 결과에 더해주어(x + F(x)) 기울기 소실(Vanishing Gradient) 문제를 완화합니다. ResNet에서 처음 제안된 이 구조는 트랜스포머의 각 서브레이어 뒤에도 적용됩니다.", "잔차 연결", "3010", "hard"),
    ("LLM이 처리하는 데이터의 최소 단위인 '토큰(Token)'에 대한 설명으로 틀린 것은?", ["글자 하나일 수도 있고, 단어 하나일 수도 있다.", "모델은 텍스트를 직접 읽는 것이 아니라 토큰화된 숫자를 처리한다.", "영어보다 한글이 토큰 소모량이 보통 더 적다.", "단어의 일부(서브워드) 단위로 쪼개지기도 한다.", "토큰 소모량이 많을수록 API 비용이 더 많이 발생한다."], "영어보다 한글이 토큰 소모량이 보통 더 적다.", "한글은 교착어 특성상 형태소 단위로 쪼개지면 영어보다 토큰을 더 많이 사용하는 경향이 있습니다. 같은 내용의 문장도 한국어로 쓰면 영어보다 2~3배 많은 토큰을 소비할 수 있습니다.", "토큰의 정의", "3011", "easy"),
    ("단어의 의미를 고차원 공간상의 좌표(실수 리스트)로 나타내는 과정을 무엇이라 하는가?", ["Vectorization", "Embedding (임베딩)", "Scaling", "Positioning", "Dimensioning"], "Embedding (임베딩)", "임베딩을 통해 컴퓨터는 단어 사이의 의미적 유사도를 계산할 수 있게 됩니다. '왕 - 남자 + 여자 ≈ 여왕'처럼 벡터 연산으로 의미 관계도 표현됩니다.", "임베딩", "3012", "easy"),
    ("유사한 의미를 가진 단어들은 벡터 공간상에서 어떤 특징을 갖는가?", ["서로 멀리 떨어져 있다.", "서로 수직 관계에 있다.", "서로 가까운 거리에 위치한다.", "모두 0에 수렴한다.", "아무런 상관관계가 없다."], "서로 가까운 거리에 위치한다.", "코사인 유사도 등을 통해 벡터 간의 거리가 가까울수록 의미가 유사하다고 판단합니다. 이 성질을 이용해 RAG에서 의미 기반 검색(Semantic Search)을 수행합니다.", "공간적 의미", "3013", "easy"),
    ("서브워드(Subword) 토큰화 기법 중 하나로, 자주 등장하는 문자 쌍을 반복적으로 병합하는 방식은?", ["WordPiece", "SentencePiece", "BPE (Byte Pair Encoding)", "N-gram", "Jamo Splitting"], "BPE (Byte Pair Encoding)", "BPE는 가장 빈번한 조합을 하나의 단어로 묶어 어휘 사전의 효율성을 극대화합니다. GPT 계열 모델에서 주로 사용하며, 미지의 단어도 서브워드로 분해하여 처리할 수 있습니다.", "BPE", "3014", "medium"),
    ("LLM이 한 번에 기억하고 처리할 수 있는 입력 데이터의 최대 범위는?", ["Memory Span", "Context Window (문맥 창)", "Token Buffer", "Input Horizon", "Processing Limit"], "Context Window (문맥 창)", "이 범위를 벗어난 이전 대화 내용은 모델이 망각하게 됩니다. GPT-4 Turbo는 128K 토큰, Claude 3는 200K 토큰까지 지원하여 긴 문서 처리가 가능합니다.", "문맥 창", "3015", "easy"),
    ("GPT-3 모델의 매개변수(Parameter) 개수는 약 얼마인가?", ["1.7B (17억 개)", "175B (1,750억 개)", "7B (70억 개)", "1T (1조 개)", "500M (5억 개)"], "175B (1,750억 개)", "GPT-3는 초거대 언어 모델의 시대를 연 상징적인 모델로 1,750억 개의 파라미터를 가집니다. 당시 기준으로 전례 없는 규모였으며, Few-shot Learning의 놀라운 성능을 처음으로 보여준 모델입니다.", "GPT-3 규모", "3016", "medium"),
    ("별도의 추가 학습 없이 프롬프트에 예시를 몇 개 보여주는 것만으로 모델이 방식을 익히는 현상은?", ["Fine-tuning", "In-Context Learning (Few-shot)", "Hard Coding", "Manual Training", "Meta Learning"], "In-Context Learning (Few-shot)", "모델 가중치를 고정하고 프롬프트 맥락 내에서 지식을 습득하는 능력입니다. GPT-3에서 처음으로 강력한 성능이 확인되었으며, 파인튜닝 없이도 다양한 태스크에 적응할 수 있습니다.", "퓨샷 학습", "3017", "easy"),
    ("예시를 전혀 주지 않고 바로 명령만 내리는 방식을 무엇이라 하는가?", ["One-shot", "Zero-shot", "No-shot", "Direct-shot", "Fast-shot"], "Zero-shot", "모델의 사전 지식과 지시 이행 능력에만 의존하는 방식입니다. 예시 없이도 모델이 지시를 이해하고 수행할 수 있는지 테스트할 때 사용합니다.", "제로샷", "3018", "easy"),
    ("OpenAI가 발표한 모델 중 멀티모달 능력을 갖추고 이미지 인식까지 가능해진 유료 모델 버전은?", ["GPT-2", "GPT-3", "GPT-4", "GPT-Neo", "InstructGPT"], "GPT-4", "GPT-4는 텍스트뿐만 아니라 이미지 입력을 이해할 수 있는 강력한 성능을 보여줍니다. 이전 GPT 계열과 달리 멀티모달(Vision) 능력을 갖추며 복잡한 추론 능력이 크게 향상되었습니다.", "GPT-4", "3019", "hard"),
    ("메타(Meta)가 공개하여 오픈소스 LLM 생태계를 폭발시킨 모델의 이름은?", ["Alpaca", "Claude", "LLaMA (라마)", "Gemini", "Mistral"], "LLaMA (라마)", "라마의 가중치 공개는 개인과 연구자들이 저사양으로도 LLM을 연구하게 만든 전환점이 되었습니다. LLaMA 2, 3 등 후속 버전도 계속 오픈소스로 공개되며 오픈 생태계를 선도하고 있습니다.", "LLaMA", "3020", "medium"),
    ("허깅페이스(HuggingFace)에서 모델을 다운로드하여 내 서버에서 직접 구동하는 방식의 장점은?", ["관리 인력이 아예 필요 없다.", "서버 비용이 0원이다.", "데이터 보안이 강력하고 커스텀 학습이 자유롭다.", "메모리(RAM)를 거의 쓰지 않는다.", "인터넷이 끊겨도 전 세계 데이터를 다 안다."], "데이터 보안이 강력하고 커스텀 학습이 자유롭다.", "외부 서버로 데이터를 보내지 않아 보안에 유리하며, 우리 비즈니스에 맞게 수정이 가능합니다. 의료, 금융 등 민감 데이터를 다루는 기업에서 특히 중요한 선택 기준입니다.", "오픈 모델 장점", "3021", "medium"),
    ("OpenAI API 등을 사용하여 클라우드 기반으로 모델을 쓰는 방식의 장점은?", ["가장 최신/최고 성능의 모델을 인프라 관리 없이 즉시 쓸 수 있다.", "데이터 유출 위험이 절대 없다.", "인터넷이 없어도 작동한다.", "사용료가 평생 무료이다.", "모델의 내부 코드를 마음껏 수정할 수 있다."], "가장 최신/최고 성능의 모델을 인프라 관리 없이 즉시 쓸 수 있다.", "고성능 GPU 서빙 비용과 운영 리스크를 줄이며 최상급 성능을 활용할 수 있습니다. 스타트업이나 개인 개발자가 GPU 인프라 없이도 최고 수준의 AI를 활용할 수 있게 합니다.", "상용 API 장점", "3022", "medium"),
    ("모델의 답변 스타일 중 '온도(Temperature)'를 낮게 설정하면 나타나는 결과는?", ["답변이 매우 창의적이고 돌발적으로 바뀐다.", "답변이 일관되고 결정론적이며 보수적으로 나온다.", "답변의 길이가 10배 이상 길어진다.", "답변의 속도가 훨씬 느려진다.", "틀린 글자가 더 많이 섞이게 된다."], "답변이 일관되고 결정론적이며 보수적으로 나온다.", "낮은 온도는 가장 확률이 높은 단어 위주로 선택하여 정확성을 높여줍니다. 코드 생성, 사실 질문 등 정확한 답이 필요한 태스크에 적합합니다.", "온도 낮음", "3023", "easy"),
    ("소설이나 창의적인 아이디어를 얻고 싶을 때 권장되는 'Temperature' 범위는?", ["0.0 ~ 0.2", "0.3 ~ 0.5", "0.7 ~ 1.0", "-1.0 ~ 0.0", "오직 0.0 고정"], "0.7 ~ 1.0", "높은 온도는 모델이 다양한 후보 단어를 선택하게 하여 창의적인 결과를 유도합니다. 브레인스토밍, 시나리오 작성, 마케팅 문구 등에 적합합니다.", "온도 높음", "3024", "easy"),
    ("LLM이 존재하지 않는 사실을 지어내어 말하는 '환각' 현상의 영문 명칭은?", ["Illusion", "Distortion", "Hallucination", "Confusion", "Deception"], "Hallucination", "학습되지 않은 내용에 대해 그럴싸한 거짓말을 하는 생성 모델의 한계점입니다. 특히 최신 정보, 구체적인 수치, 출처가 중요한 답변에서 자주 발생하므로 RAG 등으로 완화합니다.", "환각", "3025", "easy"),

    # 2. 토큰화 및 모델 특징 (26-50)
    ("한글 텍스트 '안녕하세요'를 GPT 토크나이저로 변환했을 때 예상되는 결과 구조는?", ["한 글자당 토큰 1개씩 총 5개", "전체를 묶어 토큰 1개", "의미와 형태소에 따라 쪼개진 여러 개의 숫자 리스트", "영어 알파벳으로 치환된 텍스트", "바이트 단위의 0과 1"], "의미와 형태소에 따라 쪼개진 여러 개의 숫자 리스트", "토크나이저는 문장을 수치화된 토큰 ID의 시퀀스로 변환합니다. GPT 계열의 tiktoken은 한글을 UTF-8 바이트 시퀀스로 분해하므로 한 글자가 여러 토큰이 될 수 있습니다.", "한글 토큰화", "3026", "medium"),
    ("OpenAI의 'tiktoken'이나 HuggingFace의 'tokenizers' 라이브러리의 역할은?", ["텍스트의 오타를 수정한다.", "텍스트를 토큰으로 분리하거나 토큰을 텍스트로 합친다.", "모델을 직접 학습시킨다.", "강력한 보안 암호화 기능을 제공한다.", "인터넷 검색 속도를 높여준다."], "텍스트를 토큰으로 분리하거나 토큰을 텍스트로 합친다.", "모델 입력 전의 전처리(encode)와 모델 출력 후의 후처리(decode)를 담당하는 핵심 도구입니다. API 호출 전 토큰 수를 미리 계산하여 비용 추정에도 활용합니다.", "토크나이저 라이브러리", "3027", "easy"),
    ("토큰(Token)과 단어(Word)의 관계에 대한 설명으로 옳은 것은?", ["항상 1토큰 = 1단어이다.", "보통 1단어는 1개 이상의 여러 토큰으로 쪼개질 수 있다.", "토큰은 단어보다 항상 긴 텍스트 단위이다.", "단어는 잊어버리고 오직 토큰만 사전에 등록된다.", "영어는 토큰을 쓰고 한글은 단어를 쓴다."], "보통 1단어는 1개 이상의 여러 토큰으로 쪼개질 수 있다.", "공통되지 않은 단어는 서브워드로 쪼개어 효율적으로 처리합니다. 영어에서도 'unbelievable'은 'un', 'believ', 'able' 등으로 쪼개질 수 있습니다.", "토큰 vs 단어", "3028", "easy"),
    ("모델의 '파라미터(Parameter)'가 많아질수록 나타나는 일반적인 특징은?", ["학습 속도가 빨라진다.", "더 정교한 추론과 지식 습득이 가능하지만 연산 비용이 증가한다.", "저장 용량이 획기적으로 줄어든다.", "모델이 훨씬 멍청해진다.", "인터넷이 없어도 동작하지 않게 된다."], "더 정교한 추론과 지식 습득이 가능하지만 연산 비용이 증가한다.", "규모의 경제(Scaling Law)에 따라 모델이 클수록 더 똑똑해지는 경향이 있습니다. 그러나 추론 시 필요한 VRAM과 연산량도 함께 증가하므로 비용과의 트레이드오프를 고려해야 합니다.", "파라미터 증량", "3029", "hard"),
    ("트랜스포머 아키텍처에서 '병렬 처리'가 가능하다는 말의 의미는?", ["여러 문장을 한 번에 번역한다는 뜻이다.", "문장 내 단어들을 동시에 한 번에 계산할 수 있다는 뜻이다.", "CPU와 GPU를 동시에 쓴다는 뜻이다.", "파이썬과 C언어를 섞어 쓴다는 뜻이다.", "사용자가 여러 명이어도 괜찮다는 뜻이다."], "문장 내 단어들을 동시에 한 번에 계산할 수 있다는 뜻이다.", "RNN처럼 앞 순서를 기다리지 않고 행렬 연산으로 한 번에 처리하여 속도가 빠릅니다. 이는 GPU의 병렬 연산 능력과 결합해 트랜스포머의 학습 효율을 극적으로 높입니다.", "병렬 처리", "3030", "medium"),
    ("다음 중 '오픈 웨이트(Open Weights)' 모델에 해당하는 것은?", ["GPT-4", "Claude 3.5", "Llama 3", "Gemini 1.5 Pro", "o1-preview"], "Llama 3", "Llama, Mistral 등은 모델의 가중치를 공개하여 로컬 실행이 가능한 오픈 모델입니다. '오픈소스'와 '오픈 웨이트'는 다른 개념으로, 가중치만 공개하고 학습 코드/데이터는 비공개인 경우도 있습니다.", "오픈 웨이트", "3031", "medium"),
    ("상용 LLM API(예: gpt-4o) 호출 시 가장 큰 비용을 차지하는 요소는?", ["사용한 API 키의 개수", "입력 및 출력에 소모된 '토큰'의 양", "접속한 인터넷의 속도", "키보드를 타이핑한 횟수", "모니터의 해상도"], "입력 및 출력에 소모된 '토큰'의 양", "대부분의 LLM 서비스는 토큰 단위로 과금을 진행합니다. 입력 토큰과 출력 토큰의 단가가 다르며, 일반적으로 출력 토큰이 더 비쌉니다.", "API 과금", "3032", "medium"),
    ("프롬프트에 '너는 친절한 상담원이야'라고 설정하는 가장 윗 단계의 입력창 이름은?", ["User Message", "System Message", "Assistant Message", "Instruction Message", "Base Prompt"], "System Message", "시스템 메시지는 모델의 정체성과 가이드라인을 정하는 최상위 지시문입니다. 전체 대화에 걸쳐 모델의 행동 방식과 역할을 일관되게 유지시키는 역할을 합니다.", "시스템 메시지", "3033", "easy"),
    ("이전 대화 내역을 모델에게 전달할 때 쓰는 메시지 유형은?", ["User/Assistant Message", "History Message", "Log Message", "Archive Message", "Backlink Message"], "User/Assistant Message", "이전의 질문과 답변 쌍을 순서대로 전달하여 문맥을 유지합니다. LLM은 무상태(Stateless)이므로 대화 히스토리를 직접 포함시켜 전달해야 합니다.", "대화 내역", "3034", "easy"),
    ("HuggingFace 모델 페이지에서 볼 수 있는 'Model Card'의 역할은?", ["모델을 유료로 결제하는 카드", "모델의 용도, 학습 데이터, 제약 사항 등을 적은 설명서", "모델의 성능을 2배 높이는 치트키", "모델의 로고 디자인", "모델 제작자의 명함"], "모델의 용도, 학습 데이터, 제약 사항 등을 적은 설명서", "모델의 윤리적 사용과 기술적 사양을 명시한 문서입니다. 라이선스, 제한 사항, 성능 지표, 평가 결과 등 모델 선택 시 반드시 확인해야 할 정보를 담고 있습니다.", "모델 카드", "3035", "medium"),
    ("모델 크기가 커져도 성능이 일정 수준에서 멈추지 않고 계속 좋아진다는 법칙은?", ["Moore's Law", "Scaling Law (척도 법칙)", "Entropy Law", "Zipf's Law", "Efficiency Law"], "Scaling Law (척도 법칙)", "데이터, 파라미터, 연산량이 늘어나면 언어 능력이 향상된다는 관찰 결과입니다. OpenAI의 Kaplan 등이 2020년 논문에서 체계화하였으며, 이는 GPT-3, GPT-4 개발의 이론적 토대가 됩니다.", "스케일링 법칙", "3036", "easy"),
    ("특정 규모 이상의 모델에서 갑자기 나타나는 논리 추론 등의 고차원 능력을 일컫는 말은?", ["Hidden Skill", "Emergent Ability (창발적 능력)", "Sudden IQ", "Jump Point", "Super Feature"], "Emergent Ability (창발적 능력)", "작은 모델에서는 불가능하던 작업이 거대 모델에서 가능해지는 현상입니다. 예를 들어 산술 연산, 논리 추론, 코드 생성 능력 등이 특정 파라미터 임계값을 넘으면 갑자기 나타납니다.", "창발적 능력", "3037", "hard"),
    ("임베딩 벡터들 간의 유사도를 측정할 때 가장 표준적으로 사용되는 계산법은?", ["덧셈과 뺄셈", "유클리드 거리", "코사인 유사도 (Cosine Similarity)", "평균값 비교", "글자 수 비교"], "코사인 유사도 (Cosine Similarity)", "방향성을 위주로 측정하여 단어 간의 의미적 유사성을 잘 잡아냅니다. 두 벡터의 내적을 각 벡터 크기의 곱으로 나눈 값으로, -1~1 사이로 정규화되어 있습니다.", "코사인 유사도", "3038", "hard"),
    ("LLM이 다음에 올 토큰의 확률 분포에서 샘플링을 할 때, 상위 P%의 누적 확률 내 단어들만 고려하는 기법은?", ["Top-K", "Nucleus Sampling (Top-P)", "Random Cut", "Greedy Search", "Softmax Filter"], "Nucleus Sampling (Top-P)", "확률이 낮은 꼬리 부분을 자르고 유의미한 상위 단어들만 후보로 삼습니다. top_p=0.9라면 누적 확률이 90%가 될 때까지의 상위 토큰들만 고려하며, 남은 확률 mass를 제외합니다.", "Top-P", "3039", "hard"),
    ("매번 가장 높은 확률을 가진 단어 하나만 100% 선택하여 생성하는 딱딱한 방식은?", ["Random Search", "Greedy Search (탐욕적 검색)", "Beam Search", "Smart Pick", "Top-N"], "Greedy Search (탐욕적 검색)", "가장 뻔한 답변이 나오기 쉽고 창의성이 낮아집니다. 각 스텝에서 지역 최적(local optimum)을 선택하므로 전체적으로 최적이 아닌 답이 나올 수 있습니다.", "그리디 서치", "3040", "hard"),
    ("트랜스포머 아키텍처 논문 제목 'Attention is All You Need'가 시사하는 바는?", ["RNN을 더 많이 써야 한다.", "어테션만으로도 충분히 강력한 모델을 만들 수 있다.", "데이터 보안이 가장 중요하다.", "인터넷 속도가 생명이다.", "사람의 관심(Attention)이 모델을 만든다."], "어테션만으로도 충분히 강력한 모델을 만들 수 있다.", "기존의 복잡한 RNN/CNN 구조를 걷어내고 어텐션만으로도 state-of-the-art 성능을 달성할 수 있음을 보여준 혁신적인 논문입니다.", "논문 제목 의미", "3041", "hard"),
    ("GPT 시리즈의 역사를 순서대로 나열한 것은?", ["GPT-3 -> GPT-2 -> GPT-1", "GPT-1 -> GPT-2 -> GPT-3", "BERT -> GPT-1 -> T5", "GPT-Open -> GPT-Closed", "Llama -> GPT -> Claude"], "GPT-1 -> GPT-2 -> GPT-3", "버전 번호가 커지며 파라미터 수와 성능이 비약적으로 증가해 왔습니다. GPT-1(117M) → GPT-2(1.5B) → GPT-3(175B)로 규모가 급격히 커졌습니다.", "GPT 역사", "3042", "medium"),
    ("라마(LLaMA) 모델이 벤치마크 점수는 높으면서도 크기를 줄일 수 있었던 비결은?", ["학습 데이터를 모두 한글로 해서", "양보다 질 좋은 방대한 양의 데이터를 학습해서", "파라미터를 0으로 만들어서", "이미지만 학습해서", "인공지능을 쓰지 않아서"], "양보다 질 좋은 방대한 양의 데이터를 학습해서", "모델 크기 대비 더 많은 양의 고품질 텍스트를 학습시킨 것이 핵심입니다. 이는 Chinchilla 스케일링 법칙에 기반한 접근으로, 적절한 모델 크기에 충분한 데이터를 맞추는 전략입니다.", "LLaMA 성공 요인", "3043", "medium"),
    ("다음 중 모델 서빙 시 메모리 사용량을 줄이기 위해 가중치의 정밀도를 낮추는 기법은?", ["Normalization", "Quantization (양자화)", "Distillation", "Pruning", "Augmentation"], "Quantization (양자화)", "16비트 모델을 4비트로 낮추면 메모리를 약 4배 아낄 수 있습니다. bitsandbytes 라이브러리의 load_in_4bit 옵션 등을 사용해 소비자 GPU에서도 대형 모델을 실행할 수 있습니다.", "양자화", "3044", "hard"),
    ("큰 모델(Teacher)의 지식을 작은 모델(Student)에게 전수하여 소형화하는 기법은?", ["Teaching", "Knowledge Distillation (지식 증류)", "Inheritance", "Copy-Paste", "Hard Training"], "Knowledge Distillation (지식 증류)", "작지만 똑똑한 모델을 만드는 데 사용되는 최적화 기법입니다. Teacher 모델의 소프트 확률 분포를 Student 모델이 모방하도록 학습시켜 성능 손실을 최소화합니다.", "지식 증류", "3045", "hard"),
    ("데이터 분석과 학습 기록을 위해 코드와 실행 결과를 한 장의 문서로 관리하는 도구는?", ["Excel", "Jupyter Notebook", "Notepad", "PowerPoint", "Slack"], "Jupyter Notebook", "LLM 학습 및 테스트 시 인터랙티브한 코딩 환경을 제공합니다. 코드, 마크다운 설명, 그래프, 실행 결과를 하나의 .ipynb 파일로 관리하여 실험 재현성에 유리합니다.", "쥬피터 노트북", "3046", "medium"),
    ("모델 학습 시 '에포크(Epoch)'의 정의로 옳은 것은?", ["데이터를 한 줄 읽었을 때", "전체 데이터를 모델이 한 번 다 훑었을 때", "1초의 시간이 흘렀을 때", "에러가 한 번 났을 때", "답변을 한 번 생성했을 때"], "전체 데이터를 모델이 한 번 다 훑었을 때", "학습의 반복 단위를 나타내는 기본 용어입니다. 일반적으로 여러 에포크를 반복하며 학습이 진행되며, 에포크가 너무 많으면 과적합이 발생할 수 있습니다.", "에포크", "3047", "medium"),
    ("LLM이 '이전의 대화 흐름'을 기억하려면 매번 질문할 때 무엇을 같이 보내야 하는가?", ["전체 대화 내역(Chat History)", "내 컴퓨터의 로그인 ID", "내 어제의 일기", "현재 날씨 정보", "인터넷 브라우저 쿠키"], "전체 대화 내역(Chat History)", "모델은 상태를 저장하지 않으므로(Stateless), 이전 내역을 매번 전부 전달해야 합니다. 대화가 길어질수록 전달해야 할 토큰도 늘어나 컨텍스트 윈도우 한계와 비용이 문제가 됩니다.", "대화 기억", "3048", "easy"),
    ("HuggingFace에서 모델을 불러올 때 사용되는 파이썬 라이브러리 명칭은?", ["torch", "transformers", "scipy", "django", "requests"], "transformers", "트랜스포머 기반 모델들을 쉽게 다루는 표준 라이브러리입니다. AutoModel, pipeline 등 다양한 고수준 API를 제공하며, PyTorch/TensorFlow 백엔드를 모두 지원합니다.", "transformers Lib", "3049", "easy"),
    ("GPT-4o 모델에서 'o'의 의미와 멀티모달의 연결이 적절한 것은?", ["Optimized: 속도가 빠름", "Open: 코드가 공개됨", "Omni: 텍스트/이미지/오디오 통합 처리", "Online: 실시간 검색 가능", "Only: 텍스트만 처리"], "Omni: 텍스트/이미지/오디오 통합 처리", "Omni는 '모든'이라는 뜻으로 다양한 미디어를 입출력함을 의미합니다. 텍스트, 이미지, 오디오를 단일 네이티브 멀티모달 모델로 처리하여 이전 방식보다 지연 시간과 정확도가 개선되었습니다.", "GPT-4o", "3050", "medium"),

    # 추가 50문제 (개념 심화 및 실무)
    ("트랜스포머에서 'Self-Attention'과 'Cross-Attention'의 차이점으로 옳은 것은?", ["Self는 자기 자신을, Cross는 다른 모델을 본다.", "Self는 입력 문장 내의 관계를, Cross는 인코더와 디코더 사이의 관계를 본다.", "Self는 영어만, Cross는 번역만 한다.", "둘은 이름만 다를 뿐 100% 동일한 연산이다.", "Self는 CPU에서, Cross는 GPU에서 수행된다."], "Self는 입력 문장 내의 관계를, Cross는 인코더와 디코더 사이의 관계를 본다.", "Cross-Attention은 번역기 등에서 소스 문장(인코더 출력)을 참고하여 디코더가 적절한 단어를 선택할 때 중요합니다. Query는 디코더에서, Key/Value는 인코더에서 옵니다.", "Self vs Cross", "3051", "hard"),
    ("생성 제어 파라미터 중 Top-K를 1로 설정하면 어떤 기법과 동일해지는가?", ["Beam Search", "Random Sampling", "Greedy Search", "Nucleus Sampling", "Penalized Search"], "Greedy Search", "가장 확률 높은 1개 단어만 후보로 두므로 탐욕적 검색과 같아집니다. 반대로 Top-K를 어휘 전체 크기로 설정하면 완전 랜덤 샘플링이 됩니다.", "Top-K 1", "3052", "hard"),
    ("학습 데이터에 편향(Bias)이 섞여 있을 때 발생하는 사회적 위험은?", ["컴퓨터가 고장 난다.", "인종, 성별 등에 대해 차별적인 답변을 내놓을 수 있다.", "전기료가 많이 나온다.", "인터넷 속도가 느려진다.", "모델이 아무 대답도 하지 못한다."], "인종, 성별 등에 대해 차별적인 답변을 내놓을 수 있다.", "공정하고 윤리적인 AI를 위해 데이터 정제가 필수적입니다. 학습 데이터의 편향은 모델의 출력에 그대로 반영되어 사회적 차별과 불평등을 강화할 수 있습니다.", "모델 편향", "3053", "medium"),
    ("거대 언어 모델이 추론(Reasoning)을 더 잘하게 만들기 위해 단계별로 생각하게 유도하는 프롬프트 기법은?", ["CoT (Chain-of-Thought)", "Few-shot", "Persona", "Output formatting", "Role playing"], "CoT (Chain-of-Thought)", "풀이 과정을 먼저 적게 함으로써 정확한 정답 도출을 돕습니다. '단계별로 생각해봐(Let's think step by step)'라는 한 문장만으로도 수학 추론 정확도가 크게 향상됩니다.", "CoT", "3054", "hard"),
    ("OpenAI API에서 'max_tokens'를 너무 작게 설정하면 발생하는 일은?", ["답변이 나오지 않는다.", "답변이 중간에 뚝 끊긴다.", "답변이 더 정확해진다.", "무료로 전활된다.", "오타가 수정된다."], "답변이 중간에 뚝 끊긴다.", "생성할 수 있는 최대 길이를 넘어서면 끊긴 채로 전달됩니다. 응답의 finish_reason이 'length'이면 토큰 한도 초과로 잘린 것이고, 'stop'이면 정상 완료입니다.", "맥스 토큰", "3055", "medium"),
    ("모델의 '가중치(Weights)'란 무엇을 의미하는가?", ["모델 파일의 실제 무게(kg)", "단어 간의 관계 강도를 나타내는 수치 값들", "모델 개발자의 직급", "데이터베이스의 용량", "서버의 전기 소모량"], "단어 간의 관계 강도를 나타내는 수치 값들", "학습 과정을 통해 최적화된 수억~수천억 개의 수치들을 말합니다. 이 가중치들이 모델 파일(.safetensors, .bin 등)에 저장되며, 모델이 '안다'는 것의 실체입니다.", "가중치", "3056", "easy"),
    ("임베딩 벡터의 차원이 보통 수백~수천 차원인 이유는?", ["컴퓨터가 보기에 멋있어 보여서", "단어의 복잡한 의미적 특징을 다각도로 담아내기 위해서", "메모리를 최대한 많이 쓰기 위해서", "해킹을 어렵게 하려고", "숫자가 클수록 무조건 좋아서"], "단어의 복잡한 의미적 특징을 다각도로 담아내기 위해서", "고차원 공간일수록 미세한 의미 차이를 분리하여 표현하기 유리합니다. text-embedding-3-small은 1536차원, text-embedding-3-large는 3072차원을 기본으로 사용합니다.", "임베딩 차원", "3057", "medium"),
    ("다음 중 OpenAI가 제공하는 가장 똑똑하지만 비싼 최상위 모델 라인업은?", ["Mini", "Ada", "Turbo", "GPT-4 / 4o", "Babbage"], "GPT-4 / 4o", "GPT-4 계열은 가장 복잡한 추론 작업을 수행하기 위한 플래그십 모델입니다. 비용이 높은 대신 복잡한 코드 생성, 다단계 추론, 멀티모달 처리 등에서 압도적 성능을 보입니다.", "모델 레벨", "3058", "medium"),
    ("실무에서 '토큰화' 비용을 줄이기 위한 가장 효과적인 방법은?", ["영어로만 대화한다.", "프롬프트를 최대한 길게 쓴다.", "질문을 명확히 하고 불필요한 컨텍스트를 제거한다.", "인터넷창을 닫는다.", "회원 가입을 다시 한다."], "질문을 명확히 하고 불필요한 컨텍스트를 제거한다.", "간결하고 핵심적인 프롬프트 구성은 비용 효율적입니다. 불필요한 예시, 중복 설명, 과도한 시스템 메시지를 제거하면 같은 품질로 비용을 크게 줄일 수 있습니다.", "비용 절감", "3059", "medium"),
    ("모델이 사용자의 위험한 질문(폭탄 제조 등)을 거부하도록 훈련된 것을 무엇이라 하는가?", ["Safety Alignment (안전 정렬)", "Hard Coding", "Blacklisting", "Firewalling", "Blocking"], "Safety Alignment (안전 정렬)", "RLHF(인간 피드백 강화 학습) 등을 통해 유해한 출력을 방지하도록 정교하게 조정됩니다. Constitutional AI, RLAIF 등 다양한 정렬 기법들이 연구되고 있습니다.", "안전 정렬", "3060", "hard"),
    ("학습에 사용되지 않은 외부 문서를 가져와 답변에 참고하는 기술의 약자는?", ["Fine-tuning", "RAG", "GAN", "RNN", "API"], "RAG", "검색 증강 생성(Retrieval-Augmented Generation)의 약자입니다. 벡터 DB에서 관련 문서를 검색해 프롬프트에 포함시켜 최신 정보와 환각 문제를 동시에 해결합니다.", "RAG 약자", "3061", "easy"),
    ("모델 서빙 도구 중 'vLLM'이나 'TGI'가 주로 해결하는 문제는?", ["모델을 더 예쁘게 시각화하기 위해", "추론 속도와 처리량(Throughput)을 극대화하기 위해", "오타를 교정하기 위해", "코딩 교육을 하기 위해", "배터리를 절약하기 위해"], "추론 속도와 처리량(Throughput)을 극대화하기 위해", "고성능 추론 엔진을 통해 대규모 동시 접속을 효율적으로 처리합니다. vLLM의 PagedAttention, continuous batching 등의 기술로 GPU 활용률을 극대화합니다.", "서빙 엔진", "3062", "medium"),
    ("트랜스포머의 '레이어 정규화(Layer Norm)'가 수행되는 위치는?", ["학습이 다 끝난 후 파일 저장 시", "각 레이어의 연산 과정 중간중간", "사용자가 질문을 날릴 때 딱 한 번", "데이터를 웹에서 가져올 때", "컴퓨터 부팅 시"], "각 레이어의 연산 과정 중간중간", "수치 안정성을 유지하여 깊은 신경망의 학습을 가능하게 합니다. 각 서브레이어(어텐션, 피드포워드) 전후에 적용되며, 활성화 값의 분포를 정규화하여 학습을 안정화합니다.", "레이어 정규화", "3063", "medium"),
    ("딥러닝 학습 시 가중치를 업데이트하는 방향을 결정하는 핵심 알고리즘은?", ["Forward Propagation", "Backpropagation (역전파)", "Encryption", "Parsing", "Sorting"], "Backpropagation (역전파)", "오차를 뒤로 전달하며 파라미터를 수정해 나가는 기본 원리입니다. Chain Rule(연쇄 법칙)을 이용해 손실 함수의 기울기를 각 파라미터에 대해 계산합니다.", "역전파", "3064", "hard"),
    ("LLM 학습을 위해 인터넷 상의 모든 텍스트를 긁어모으는 행위를 무엇이라 하는가?", ["Mining", "Scraping/Crawling", "Fishing", "Hunting", "Hoarding"], "Scraping/Crawling", "Web 데이터는 LLM 사전 학습의 가장 큰 재료입니다. CommonCrawl, C4, The Pile 등의 데이터셋이 이런 방식으로 수집된 대표적인 사전 학습 데이터셋입니다.", "데이터 수집", "3065", "medium"),
    ("거대 모델일수록 '환각' 현상이 완전히 사라진다는 주장은?", ["100% 사실이다.", "전혀 사실이 아니며 거대 모델도 환각을 일으킨다.", "이미 2023년에 해결된 문제이다.", "모델 크기와 환각은 상관이 없다.", "환각은 사람이 느끼는 착각일 뿐이다."], "전혀 사실이 아니며 거대 모델도 환각을 일으킨다.", "생성 모델의 본질적 특성상 환각은 완전히 제거하기 매우 어렵습니다. 오히려 큰 모델이 더 그럴듯하게 거짓말을 하여 탐지가 더 어려울 수 있습니다.", "환각과 규모", "3066", "medium"),
    ("Anthropic의 Claude 모델이 강조하는 'Constitutional AI'의 핵심 요소는?", ["모델에게 수만 권의 법전을 외우게 한다.", "모델이 지켜야 할 원칙(헌법)을 주고 스스로를 정렬하게 한다.", "국가 헌법 기관에 모델을 설치한다.", "모델의 이름을 대통령 이름으로 짓는다.", "오직 법률 상담만 한다."], "모델이 지켜야 할 원칙(헌법)을 주고 스스로를 정렬하게 한다.", "인간의 지속적 피드백 대신 원칙 기반의 자동 정렬을 시도하는 기술입니다. 모델 스스로 원칙에 따라 자신의 출력을 비판하고 개선하는 RLAIF 방식을 활용합니다.", "Claude 특징", "3067", "easy"),
    ("파이썬의 'list'와 'numpy array'의 차이점에 대한 복습: NumPy가 데이터 분석에 유리한 이유는?", ["파이썬 리스트는 숫자를 저장할 수 없어서", "배열 전체에 대한 벡터화 연산이 가능하여 매우 빨라서", "NumPy가 더 최신 라이브러리라서", "NumPy 배열은 크기를 줄일 수 없어서", "NumPy는 유료이기 때문"], "배열 전체에 대한 벡터화 연산이 가능하여 매우 빨라서", "행렬 연산을 순식간에 처리하는 NumPy는 AI 연산의 기초입니다. C로 구현된 내부 연산 덕분에 순수 파이썬 루프보다 수십~수백 배 빠른 속도로 대규모 행렬 연산을 수행합니다.", "NumPy 복습", "3068", "hard"),
    ("HuggingFace 모델 이름이 `meta-llama/Llama-3-8B`일 때 '8B'가 뜻하는 것은?", ["파일 용량이 8기가바이트이다.", "학습 기간이 8개월이다.", "매개변수(Parameter) 개수가 80억 개이다.", "동시 사용자 수가 8명이다.", "데이터 종류가 8가지이다."], "매개변수(Parameter) 개수가 80억 개이다.", "B는 Billion(10억)의 약자로, 모델의 지능 척도를 나타냅니다. FP16 기준 8B 모델의 실제 파일 크기는 약 16GB이며, 4비트 양자화 시 약 4.5GB로 줄어듭니다.", "8B 의미", "3069", "hard"),
    ("사용자가 '이 말을 비밀로 해줘'라고 했을 때 모델이 실제로 기억을 삭제하는가?", ["네, 즉시 서버에서 지웁니다.", "아뇨, 모델은 실시간으로 지식을 잊거나 배우는 능력이 기본적으로 없습니다.", "네, 다음 사용자는 그 비밀을 모릅니다.", "사용자가 돈을 내면 지워줍니다.", "모델이 '알겠습니다'라고 하면 진짜 지운 것입니다."], "아뇨, 모델은 실시간으로 지식을 잊거나 배우는 능력이 기본적으로 없습니다.", "모델은 학습된 시점에 고정되어 있으며 대화 내역은 일시적인 데이터일 뿐입니다. 대화 세션이 끝나면 해당 내역은 모델에 저장되지 않고, 모델 가중치는 변하지 않습니다.", "모델의 기억 실체", "3070", "medium"),
    ("GPT-4o가 소리를 실시간으로 듣고 반응할 때 사용하는 기술 흐름은?", ["소리를 텍스트로 바꾸고 답변을 다시 소리로 바꾼다.", "중간 변환 없이 소리 데이터를 직접 처리하는 단일 신경망 모델이다.", "사람이 뒤에서 몰래 타이핑해준다.", "오디오를 이미지로 찍어서 판독한다.", "소리 주파수를 수학적으로 계산만 한다."], "중간 변환 없이 소리 데이터를 직접 처리하는 단일 신경망 모델이다.", "Native Multimodal로, 지연 시간을 최소화하고 감정까지 읽을 수 있습니다. 기존 STT→LLM→TTS 파이프라인 방식보다 대기 시간이 짧고 억양, 감정 등 비언어 정보도 처리할 수 있습니다.", "오디오 처리", "3071", "hard"),
    ("LLM이 특정 전문 분야(의료, 금융 등)의 용어를 더 잘 쓰게 하려면 추천되는 방식은?", ["모델에게 응원 메시지를 보낸다.", "해당 분야 데이터로 파인튜닝(Fine-tuning)을 수행한다.", "질문할 때 '의사라고 생각하고 답해'라고 한 번 말하고 끝낸다.", "컴퓨터를 해당 병원에 비치한다.", "인터넷 게시판에 질문을 올린다."], "해당 분야 데이터로 파인튜닝(Fine-tuning)을 수행한다.", "특화된 데이터셋 학습을 통해 도메인 전문가로 만들 수 있습니다. LoRA, QLoRA 등의 파라미터 효율적 파인튜닝 기법을 활용하면 상대적으로 적은 비용으로도 전문화가 가능합니다.", "전문성 향상", "3072", "medium"),
    ("프롬프트 엔지니어링 팁 중 '구분자(Delimiter)'를 사용하라는 말의 의미는?", ["입력 데이터와 지시문을 ### 등 특수문자로 나누어 모델의 혼란을 방지한다.", "단어마다 띄어쓰기를 3번씩 한다.", "영어와 한글을 절대 섞어 쓰지 않는다.", "질문을 여러 개의 파일로 쪼개어 보낸다.", "정답을 미리 알려주고 모른 척한다."], "입력 데이터와 지시문을 ### 등 특수문자로 나누어 모델의 혼란을 방지한다.", "구조화된 입력은 모델이 작업 범위를 정확히 파악하게 돕습니다. ''', \"\"\" , ### , <tag> 등을 사용하여 지시문과 데이터 영역을 명확히 분리합니다.", "구분자 활용", "3073", "medium"),
    ("모델 서빙 중 'FP16'에서 'INT8'로 양자화하면 줄어드는 비용은?", ["전기세", "메모리 점유량과 연산 속도", "인터넷 통신료", "사무실 월세", "라이브러리 사용료"], "메모리 점유량과 연산 속도", "수치의 정밀도를 낮추어 성능 하락을 최소화하면서 자원을 아낍니다. FP16(2바이트)에서 INT8(1바이트)로 변환하면 메모리가 절반으로 줄고, 정수 연산이 빠른 하드웨어에서는 속도도 향상됩니다.", "양자화 효과", "3074", "hard"),
    ("다음 중 LLM을 활용한 서비스 개발 시 '할루시네이션(환각)'을 줄이는 가장 실질적인 방법은?", ["모델에게 '거짓말하지 마'라고 계속 입력한다.", "RAG 시스템을 도입하여 근거 문서를 기반으로 답하게 한다.", "온도(Temperature)를 2.0으로 높인다.", "답변의 길이를 최대(Max Tokens)로 설정한다.", "모든 질문을 영어로 번역해서 시킨다."], "RAG 시스템을 도입하여 근거 문서를 기반으로 답하게 한다.", "검색된 사실 정보를 프롬프트에 제공하는 것이 환각 방지의 표준입니다. 모델에게 '제공된 문서를 기반으로만 답하고, 없으면 모른다고 해'라는 지침과 함께 사용하면 효과가 극대화됩니다.", "환각 방지 실무", "3075", "hard"),
    ("Transformer 블록 내에서 텍스트 데이터가 흐르는 순서는?", ["Embedding -> Attention -> FeedForward", "FeedForward -> Attention -> Embedding", "Attention -> Embedding -> Output", "Output -> Attention -> Embedding", "수시로 바뀐다"], "Embedding -> Attention -> FeedForward", "입력이 수치화된 후 관계를 파악하고 고차원 특징을 추출하는 순서입니다. 실제로는 각 단계 전후에 잔차 연결과 레이어 정규화가 적용됩니다.", "데이터 흐름", "3076", "medium"),
    ("LLM이 답변을 생성하다가 갑자기 멈춘 경우, 다시 이어 쓰게 하려면 보통 어떤 명령을 내리는가?", ["처음부터 다시 해", "계속해서(Continue) 설명해줘", "왜 멈췄어?", "돈 줄게", "키보드 엔터 키를 누른다"], "계속해서(Continue) 설명해줘", "모델에게 이전 문맥의 마지막을 보여주며 이어서 생성하도록 유도합니다. max_tokens 한도 도달이나 stop 시퀀스 감지로 중단된 경우 이 방법으로 이어서 생성할 수 있습니다.", "생성 중단 대처", "3077", "medium"),
    ("트랜스포머 아키텍처에서 '병렬성'을 저해하는 요소가 거의 없는 이유는?", ["CPU를 안 쓰기 때문", "단어 간의 순차적 상태 전달(Hidden State)이 없기 때문", "글자 수가 적어서", "프로그램이 간단해서", "구글이 만들었기 때문"], "단어 간의 순차적 상태 전달(Hidden State)이 없기 때문", "RNN과 달리 행렬 연산으로 앞뒤 결과를 한 번에 계산할 수 있는 구조입니다. 어텐션은 모든 토큰 쌍을 동시에 계산하므로 GPU의 SIMD 병렬 연산 특성을 최대한 활용합니다.", "병렬성 극대화", "3078", "hard"),
    ("거대 언어 모델이 추론 시 사용하는 GPU의 주요 자원은?", ["코어 클럭 속도", "비디오 메모리 (VRAM)", "RGB 조명", "쿨링 팬 속도", "모니터 연결 단자"], "비디오 메모리 (VRAM)", "수천억 개의 파라미터(가중치)를 메모리에 상주시켜야 하므로 VRAM 용량이 핵심입니다. 7B 모델은 FP16 기준 약 14GB VRAM이 필요하며, 이를 줄이기 위해 양자화 기법을 사용합니다.", "GPU 자원", "3079", "hard"),
    ("최근 LLM 동향 중 'Small Language Models (SLM)'이 주목받는 이유는?", ["큰 모델보다 항상 똑똑해서", "특정 도메인에서 저비용/고효율로 동작 가능해서", "이름이 귀여워서", "무료로만 배포되기 때문", "업데이트가 안 되기 때문"], "특정 도메인에서 저비용/고효율로 동작 가능해서", "특정 작업에 최적화된 작은 모델은 실무 적용 시 가성비가 매우 높습니다. Microsoft의 Phi 시리즈, Google의 Gemma 등이 대표적이며, 모바일 기기에서도 실행 가능합니다.", "SLM 주목 이유", "3080", "medium"),
    ("`tokenizer.decode([10, 25, 40])`를 실행한 결과물은?", ["숫자 리스트 [10, 25, 40]", "해당 숫자들에 매칭되는 '문자열'", "에러 메시지", "이미지 파일", "오디오 파일"], "해당 숫자들에 매칭되는 '문자열'", "ID 숫자를 다시 사람이 읽을 수 있는 글자로 변환하는 과정입니다. 이 과정을 '디코딩'이라 하며, tokenizer.encode()의 역연산에 해당합니다.", "디코딩", "3081", "medium"),
    ("트랜스포머의 인코더가 출력하는 정보의 형태는?", ["정답 문장 하나", "각 단어의 의미가 담긴 벡터 리스트 (Contextual Embeddings)", "예/아니오 결과", "랜덤한 숫자", "파이썬 코드"], "각 단어의 의미가 담긴 벡터 리스트 (Contextual Embeddings)", "문맥 정보를 가득 담은 임베딩 값을 다음 층이나 디코더로 넘깁니다. Static 임베딩(Word2Vec)과 달리, 동일한 단어도 문맥에 따라 다른 벡터를 갖는 것이 BERT 같은 인코더의 강점입니다.", "인코더 출력", "3082", "medium"),
    ("GPT의 'Attention Mask' 설정값이 0인 부분의 의미는?", ["모델이 이 부분에 집중해야 함", "모델이 이 부분을 무시(Ignore)해야 함", "오타가 있는 부분임", "정답이 숨겨진 부분임", "가장 중요한 단어임"], "모델이 이 부분을 무시(Ignore)해야 함", "미래의 단어나 불필요한 패딩(Padding) 부분을 보지 못하게 가리는 용도입니다. 배치 처리 시 짧은 문장에 추가된 PAD 토큰 위치에 0을 넣어 모델이 그 부분을 무시하게 합니다.", "어텐션 마스크", "3083", "medium"),
    ("LLM 학습 데이터 전처리 시 중복 제거(Deduplication)를 하는 주된 목적은?", ["데이터 양을 억지로 부풀리기 위해", "모델이 특정 문장을 암기(Memorization)하는 것을 방지하기 위해", "데이터를 모두 지우기 위해", "저작권을 속이기 위해", "파일 개수를 맞추려고"], "모델이 특정 문장을 암기(Memorization)하는 것을 방지하기 위해", "중복이 많으면 모델이 편향되거나 단순 암기를 하게 되어 범용성이 떨어집니다. 또한 개인 정보가 포함된 문서가 중복되면 모델이 이를 그대로 출력할 위험이 있어 프라이버시 측면에서도 중요합니다.", "중복 제거 목적", "3084", "medium"),
    ("모델 평가 지표 중 'MMLU'는 무엇을 측정하는가?", ["모델의 생성 속도", "다양한 학문 분야에 대한 일반 지식과 문제 풀이 능력", "이미지 생성 퀄리티", "네트워크 지연 시간", "한국어 맞춤법"], "다양한 학문 분야에 대한 일반 지식과 문제 풀이 능력", "대학 수준의 지식을 얼마나 잘 알고 있는지 평가하는 대표 벤치마크입니다. 수학, 역사, 법률, 의학 등 57개 분야에서 4지선다 문제로 구성되어 있습니다.", "MMLU", "3085", "medium"),
    ("딥러닝 학습 시 'Overfitting(과적합)'이 발생했다는 것은?", ["학습 데이터는 잘 맞추지만 새로운 데이터에는 멍청해진 상태", "너무 똑똑해져서 사람을 무시하는 상태", "데이터가 너무 적어서 학습이 안 된 상태", "컴퓨터가 과열된 상태", "인터넷이 끊긴 상태"], "학습 데이터는 잘 맞추지만 새로운 데이터에는 멍청해진 상태", "학습 데이터에만 너무 맞춰져 범용적인 추론 능력이 상실된 경우입니다. Dropout, 정규화, 조기 종료(Early Stopping) 등의 기법으로 과적합을 방지합니다.", "과적합", "3086", "easy"),
    ("LLM 서비스 시 답변이 한 글자씩 나오는 'Streaming'의 장점은?", ["최종 답변이 더 정확해진다.", "사용자가 답변이 생성되는 과정을 체감하여 답답함을 줄여준다.", "토큰 비용이 저렴해진다.", "모델 개발이 더 쉬워진다.", "보안이 더 안전해진다."], "사용자가 답변이 생성되는 과정을 체감하여 답답함을 줄여준다.", "전체 생성을 기다리는 지루함을 없애 체감 속도(UX)를 높여줍니다. OpenAI API에서 stream=True로 설정하고 Server-Sent Events 방식으로 청크를 받아 처리합니다.", "스트리밍 장점", "3087", "easy"),
    ("다음 중 '멀티모달' 기능과 가장 무관한 작업은?", ["이미지를 보고 텍스트로 설명하기", "음성 명령을 듣고 그림 그리기", "텍스트를 다른 나라 언어로 번역하기", "동영상을 보고 내용 요약하기", "표를 보고 엑셀로 변환하기 (시각 정보 포함)"], "텍스트를 다른 나라 언어로 번역하기", "단순 텍스트 번역은 텍스트 입력→텍스트 출력이므로 단일 모달(Unimodal) 작업에 해당합니다. 나머지 선택지들은 모두 다른 형태의 미디어를 입출력으로 사용하는 멀티모달 작업입니다.", "멀티모달 구분", "3088", "medium"),
    ("GPT-3와 같은 LLM은 기본적으로 ( ) 방식으로 다음 토큰을 맞히며 학습된다. 빈칸은?", ["지도 학습", "비지도 학습 (Self-supervised)", "강화 학습", "전이 학습", "사후 학습"], "비지도 학습 (Self-supervised)", "별도의 정답 라벨 없이 텍스트 자체에서 다음 단어를 맞히며 스스로 공부합니다. 인터넷에 있는 방대한 텍스트를 레이블링 없이 그대로 학습 데이터로 사용할 수 있어 대규모 학습이 가능합니다.", "학습 방식", "3089", "medium"),
    ("거대 언어 모델이 인류의 안전과 이익에 부합하도록 만드는 최종 조율 단계는?", ["Pre-training", "Pre-processing", "Alignment (정렬)", "Parsing", "Compressing"], "Alignment (정렬)", "RLHF(인간 피드백 강화 학습) 등을 통해 인간의 의도와 윤리에 맞게 맞추는 핵심 단계입니다. Helpfulness, Harmlessness, Honesty(3H)를 목표로 모델 행동을 인간의 가치와 정렬합니다.", "정렬", "3090", "easy"),
    ("LLM이 특정 문법 형식을 지키도록(예: JSON) 시스템 프롬프트에 예시를 넣는 것을 무엇이라 하는가?", ["Strict Mode", "Output Constrainting", "Formatting Guide", "Constraint Prompting", "JSON Enforcement"], "Constraint Prompting", "모델의 자유도를 제한하여 구조화된 데이터를 얻는 기법입니다. 예시를 제공하거나 response_format 파라미터를 활용하여 모델이 항상 특정 형식으로 응답하게 강제할 수 있습니다.", "형식 제약", "3091", "medium"),
    ("토크나이저 사전(Vocabulary)에 없는 단어가 들어오면 보통 어떤 토큰으로 처리되는가?", ["<END>", "<UNK> (Unknown)", "<START>", "<PAD>", "에러로 중단됨"], "<UNK> (Unknown)", "모르는 단어는 특수 토큰으로 처리하여 일단 진행합니다. 그러나 BPE 기반 최신 토크나이저는 모든 단어를 서브워드로 분해할 수 있어 <UNK>가 사실상 등장하지 않습니다.", "UNK 토큰", "3092", "medium"),
    ("트랜스포머의 'FeedForward' 층이 수행하는 역할은?", ["단어 간의 관계를 찾는다.", "어텐션 결과를 바탕으로 고차원적인 비선형 특징을 추출한다.", "결과를 화면에 출력한다.", "데이터를 외부로 전송한다.", "가장 높은 확률값을 고른다."], "어텐션 결과를 바탕으로 고차원적인 비선형 특징을 추출한다.", "어텐션이 토큰 간 관계를 본다면, 피드포워드는 각 토큰의 의미를 더 깊게 가공합니다. 일반적으로 모델 차원의 4배 크기로 확장되었다가 다시 줄어드는 구조(확장-압축)를 가집니다.", "피드포워드 역할", "3093", "hard"),
    ("학습 시 데이터셋을 작은 덩어리로 나누어 GPU에 올리는 단위를 무엇이라 하는가?", ["Chunk", "Batch", "Segment", "Particle", "Piece"], "Batch", "한 번의 가중치 업데이트를 위해 묶음으로 처리하는 단위입니다. 배치 크기가 클수록 학습이 안정적이지만 메모리를 많이 사용하고, 작을수록 노이즈가 많지만 메모리 효율이 높습니다.", "배치", "3094", "medium"),
    ("LLM을 사용할 때 '할루시네이션'을 긍정적으로 활용할 수 있는 분야는?", ["의료 진단 보고서", "은행 대출 심사", "소설 창작 및 브레인스토밍", "법률 판례 분석", "정밀 부품 설계"], "소설 창작 및 브레인스토밍", "창의적 영역에서는 때로 사실이 아닌 기발한 상상이 도움이 됩니다. 반면 의료, 법률, 금융처럼 정확성이 생명인 분야에서는 환각이 치명적인 결과를 초래할 수 있습니다.", "환각의 활용", "3095", "medium"),
    ("모델의 'Softmax' 함수 출력값의 모든 합은 항상 얼마인가?", ["0", "1", "100", "무한대", "데이터마다 다름"], "1", "출력값을 확률 분포로 바꾸어 전체 합이 1(100%)이 되도록 정규화합니다. 각 토큰의 소프트맥스 출력값은 그 토큰이 다음에 나올 확률을 의미합니다.", "Softmax 특성", "3096", "medium"),
    ("API 호출 시 'stop' 파라미터는 언제 사용하는가?", ["특정 글자가 나오면 생성을 강제로 멈추고 싶을 때", "매달 결제를 멈추고 싶을 때", "키보드 작성을 멈출 때", "인터넷을 끌 때", "모델을 삭제할 때"], "특정 글자가 나오면 생성을 강제로 멈추고 싶을 때", "불필요한 답변 생성을 막아 비용과 시간을 아끼는 용도입니다. 예: stop=['\\n', '###']로 설정하면 개행이나 구분자가 나오는 즉시 생성을 중단합니다.", "Stop 옵션", "3097", "medium"),
    ("로컬에서 LLM을 돌릴 때 CPU보다 GPU가 권장되는 가장 큰 성능상의 이유는?", ["GPU가 전기료가 싸서", "수천 개의 행렬 연산을 동시에 처리하는 병렬성에 최적화되어 있어서", "GPU가 기억력이 더 좋아서", "CPU는 게임용이기 때문", "GPU가 더 예쁘게 생겨서"], "수천 개의 행렬 연산을 동시에 처리하는 병렬성에 최적화되어 있어서", "행렬 곱셈이 99%인 딥러닝 연산은 GPU의 병렬 구조에서 압도적으로 처리됩니다. A100 GPU는 CPU 대비 LLM 추론에서 수십 배 빠른 처리량을 보여줍니다.", "GPU 필요성", "3098", "medium"),
    ("오픈소스 모델 중 벤치마크 1위를 수차례 탈환한 프랑스 기반의 AI 팀 이름은?", ["OpenAI", "Anthropic", "Mistral AI", "DeepMind", "Meta"], "Mistral AI", "Mistral 7B, Mixtral 등 작지만 강력한 오픈 모델로 유명합니다. Mixture-of-Experts 아키텍처를 채택한 Mixtral 8x7B는 GPT-3.5 수준의 성능을 오픈소스로 제공했습니다.", "Mistral", "3099", "medium"),
    ("교재 3장의 내용을 바탕으로 할 때, 좋은 LLM 활용 능력을 갖추기 위해 가장 중요한 습득 사항은?", ["모델의 모든 수학적 수식을 외우는 것", "프롬프트 원리와 모델별 특징을 알고 적절히 도구화하는 것", "PC의 메모리를 1테라로 늘리는 것", "타이핑 속도를 높이는 것", "인터넷 유료 기사를 많이 읽는 것"], "프롬프트 원리와 모델별 특징을 알고 적절히 도구화하는 것", "기술적 배경을 이해하고 이를 실무에 녹여내는 능력이 인공지능 시대의 핵심 경쟁력입니다. 수식보다 언제 어떤 모델을 왜 선택하는지, 프롬프트를 어떻게 구성하는지가 실질적으로 더 중요합니다.", "학습의 목적", "3100", "medium")
]

for q, o, a, w, h, i, d in mcq_data:
    questions.append({"chapter_name": chapter_name, "type": "객관식", "difficulty": d, "id": i, "question": q, "options": o, "answer": a, "why": w, "hint": h})

# --- 20 Code Completion Questions ---
cc_data = [
    ("HuggingFace AutoTokenizer로 텍스트 인코딩",
     "from transformers import AutoTokenizer\n\ntokenizer = AutoTokenizer.from_pretrained('gpt2')\ntext = '인공지능은 미래를 바꿉니다.'\ntokens = tokenizer._____(text)\nprint(tokens)",
     "encode",
     "tokenizer.encode()는 텍스트를 토큰 ID 정수 리스트로 변환합니다. tokenizer(text)['input_ids']와 동일한 결과를 냅니다. 반대 변환은 tokenizer.decode(token_ids)로 수행합니다."),

    ("transformers pipeline으로 감정 분류",
     "from transformers import pipeline\n\nclassifier = pipeline('sentiment-analysis')\nresult = classifier('This movie was absolutely wonderful!')\nprint(result)\n# [{'label': 'POSITIVE', 'score': ___}]",
     "0.99",
     "pipeline()은 HuggingFace 모델을 가장 빠르게 사용하는 방법입니다. 'sentiment-analysis' 태스크를 지정하면 입력 텍스트의 감성 레이블과 신뢰도(score)를 반환합니다."),

    ("AutoModelForCausalLM으로 텍스트 생성",
     "from transformers import AutoTokenizer, AutoModelForCausalLM\nimport torch\n\nmodel_id = 'gpt2'\ntokenizer = AutoTokenizer.from_pretrained(model_id)\nmodel = AutoModelForCausalLM._____(model_id)\n\ninputs = tokenizer('Python is', return_tensors='pt')\noutputs = model.generate(**inputs, max_new_tokens=20)\nprint(tokenizer.decode(outputs[0], skip_special_tokens=True))",
     "from_pretrained",
     "from_pretrained()는 HuggingFace Hub 또는 로컬 경로에서 모델 가중치를 로드합니다. AutoModelForCausalLM은 텍스트 생성(Causal LM)에 특화된 모델 클래스이며, GPT 계열 모델에 사용합니다."),

    ("Temperature를 이용한 텍스트 생성",
     "from transformers import pipeline\n\ngenerator = pipeline('text-generation', model='gpt2')\nresult = generator(\n    'Once upon a time',\n    max_new_tokens=50,\n    _____=0.9,\n    do_sample=True\n)\nprint(result[0]['generated_text'])",
     "temperature",
     "temperature 파라미터는 생성의 무작위성을 조절합니다. 0에 가까울수록 항상 같은 답, 1.0은 기본값, 2.0 이상은 매우 창의적(혼란스럽)입니다. do_sample=True를 함께 설정해야 temperature가 적용됩니다."),

    ("토큰 개수 계산 및 비용 추정",
     "from transformers import AutoTokenizer\n\ntokenizer = AutoTokenizer.from_pretrained('gpt2')\ntext = 'Hello, this is a test message for token counting.'\ntokens = tokenizer._____(text)\nprint(f'토큰 수: {len(tokens)}')\nprint(f'예상 비용 (gpt-4o): ${len(tokens) * 0.000005:.6f}')",
     "tokenize",
     "tokenizer.tokenize()는 텍스트를 문자열 토큰 리스트로 반환합니다(정수 ID가 아닌 서브워드 문자열). len()으로 토큰 수를 세고 모델별 단가를 곱해 비용을 추정할 수 있습니다."),

    ("OpenAI API 기본 호출",
     "from openai import OpenAI\n\nclient = OpenAI(api_key='YOUR_API_KEY')\nresponse = client.chat.completions.create(\n    model='gpt-4o-mini',\n    messages=[{_____ : 'user', 'content': '파이썬이란 무엇인가요?'}]\n)\nprint(response.choices[0].message.content)",
     "'role'",
     "OpenAI Chat Completions API는 messages 리스트에 role과 content 쌍을 전달합니다. role 값은 'system'(시스템 지침), 'user'(사용자 입력), 'assistant'(AI 이전 답변) 세 가지입니다."),

    ("System 메시지로 역할 설정",
     "from openai import OpenAI\nclient = OpenAI()\n\nresponse = client.chat.completions.create(\n    model='gpt-4o-mini',\n    messages=[\n        {'role': _____, 'content': '당신은 전문 번역가입니다. 항상 한국어로 번역해주세요.'},\n        {'role': 'user', 'content': 'Hello world'}\n    ]\n)\nprint(response.choices[0].message.content)",
     "'system'",
     "role='system'은 모델의 행동 방식과 역할을 설정하는 최상위 지침입니다. 시스템 메시지는 전체 대화에 걸쳐 모델의 페르소나와 제약 조건을 유지시킵니다."),

    ("임베딩 벡터 생성",
     "from openai import OpenAI\nimport numpy as np\n\nclient = OpenAI()\nresponse = client.embeddings.create(\n    model='text-embedding-3-small',\n    _____=['Python is great', '파이썬은 훌륭합니다']\n)\n\nvec1 = np.array(response.data[0].embedding)\nvec2 = np.array(response.data[1].embedding)\nsimilarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))\nprint(f'코사인 유사도: {similarity:.4f}')",
     "input",
     "embeddings.create()의 input 파라미터에 텍스트 리스트를 전달하면 각각의 임베딩 벡터를 반환합니다. 코사인 유사도 = 두 벡터의 내적 / (각 벡터의 크기의 곱)으로 계산합니다."),

    ("max_tokens 설정으로 출력 제어",
     "from openai import OpenAI\nclient = OpenAI()\n\nresponse = client.chat.completions.create(\n    model='gpt-4o-mini',\n    messages=[{'role': 'user', 'content': '세계 7대 불가사의를 나열해줘'}],\n    max_tokens=50,\n    _____=0.0\n)\nprint(response.choices[0].message.content)\nprint(response.usage.total_tokens)",
     "temperature",
     "temperature=0.0은 가장 확률이 높은 토큰만 선택해 일관된 답을 냅니다. max_tokens는 출력의 최대 길이를 제한하며, response.usage로 실제 사용된 토큰 수를 확인할 수 있습니다."),

    ("HuggingFace pipeline으로 요약",
     "from transformers import pipeline\n\nsummarizer = pipeline(\n    'summarization',\n    model='facebook/bart-large-cnn'\n)\nlong_text = '''The transformer architecture was introduced in 2017\nand has revolutionized natural language processing.\nIt uses attention mechanisms to process sequences in parallel,\novercoming limitations of previous RNN-based models.'''\n\nresult = summarizer(long_text, max_length=50, _____=25)\nprint(result[0]['summary_text'])",
     "min_length",
     "HuggingFace summarization pipeline은 긴 텍스트를 요약합니다. max_length와 min_length로 출력 범위를 제어합니다. facebook/bart-large-cnn은 CNN/DailyMail 데이터셋으로 학습된 대표적인 요약 모델입니다."),

    ("토크나이저로 배치 처리",
     "from transformers import AutoTokenizer\n\ntokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')\nsentences = [\n    'Hello world',\n    'Natural language processing is fascinating',\n    'BERT understands context'\n]\nbatch = tokenizer(\n    sentences,\n    padding=True,\n    truncation=True,\n    _____='pt'\n)\nprint('input_ids shape:', batch['input_ids'].shape)",
     "return_tensors",
     "return_tensors='pt'는 결과를 PyTorch 텐서로 반환합니다. 'tf'는 TensorFlow, 'np'는 NumPy 배열입니다. padding=True는 배치 내 가장 긴 문장에 맞춰 패딩을 추가하여 모든 입력이 같은 길이를 갖게 합니다."),

    ("대화 히스토리 유지",
     "from openai import OpenAI\nclient = OpenAI()\n\nmessages = [{'role': 'system', 'content': '친절한 AI 어시스턴트입니다.'}]\n\ndef chat(user_input):\n    _____.append({'role': 'user', 'content': user_input})\n    resp = client.chat.completions.create(model='gpt-4o-mini', messages=messages)\n    assistant_msg = resp.choices[0].message.content\n    messages.append({'role': 'assistant', 'content': assistant_msg})\n    return assistant_msg\n\nprint(chat('안녕하세요!'))\nprint(chat('방금 뭐라고 했죠?'))",
     "messages",
     "LLM은 기본적으로 무상태(Stateless)입니다. 대화 맥락을 유지하려면 이전 user/assistant 메시지를 messages 리스트에 누적하여 매번 API에 전달해야 합니다."),

    ("top_p Nucleus Sampling",
     "from openai import OpenAI\nclient = OpenAI()\n\nresponse = client.chat.completions.create(\n    model='gpt-4o-mini',\n    messages=[{'role': 'user', 'content': '창의적인 소설 도입부를 써줘'}],\n    temperature=1.0,\n    _____=0.9\n)\nprint(response.choices[0].message.content)",
     "top_p",
     "top_p(Nucleus Sampling)는 누적 확률이 p에 도달할 때까지의 후보 토큰만 고려합니다. temperature와 함께 쓸 때는 보통 둘 중 하나만 조정합니다. top_p=0.9는 상위 90% 누적 확률 내 토큰에서 샘플링합니다."),

    ("Anthropic Claude API 호출",
     "import anthropic\n\nclient = anthropic.Anthropic(api_key='YOUR_KEY')\nmessage = client.messages.create(\n    model='claude-3-5-sonnet-20241022',\n    max_tokens=1024,\n    messages=[{'role': 'user', 'content': 'RAG란 무엇인지 설명해줘'}]\n)\nprint(message.content[0]._____ )",
     "text",
     "Anthropic API에서 응답 텍스트는 message.content[0].text로 접근합니다. OpenAI의 response.choices[0].message.content와 다른 응답 구조를 가집니다. 두 API의 구조 차이를 익혀두는 것이 실무에 중요합니다."),

    ("BPE 토큰화 시각화",
     "from transformers import AutoTokenizer\n\ntokenizer = AutoTokenizer.from_pretrained('gpt2')\ntext = 'Tokenization is fascinating!'\n\ntoken_strings = tokenizer._____(text)\ntoken_ids = tokenizer.encode(text)\n\nfor token, tid in zip(token_strings, token_ids):\n    print(f'  {repr(token):15} -> {tid}')",
     "tokenize",
     "tokenizer.tokenize()는 서브워드 문자열 리스트를 반환합니다. 예: 'fascinating' → ['fasc', 'inating']. BPE가 어떻게 단어를 쪼개는지 시각화할 때 사용하며, Ġ는 단어 시작에 붙는 GPT2 특수 기호입니다."),

    ("모델 응답 스트리밍",
     "from openai import OpenAI\nclient = OpenAI()\n\nstream = client.chat.completions.create(\n    model='gpt-4o-mini',\n    messages=[{'role': 'user', 'content': '파이썬의 장점 5가지를 설명해줘'}],\n    _____=True\n)\n\nfor chunk in stream:\n    if chunk.choices[0].delta.content is not None:\n        print(chunk.choices[0].delta.content, end='', flush=True)",
     "stream",
     "stream=True 옵션을 설정하면 응답이 청크(chunk) 단위로 실시간 전달됩니다. 각 청크에서 .choices[0].delta.content로 생성된 텍스트를 추출합니다. end=''와 flush=True로 줄바꿈 없이 실시간 출력합니다."),

    ("허깅페이스 모델 양자화 로드",
     "from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig\nimport torch\n\nquant_config = BitsAndBytesConfig(load_in_4bit=True)\nmodel = AutoModelForCausalLM.from_pretrained(\n    'meta-llama/Llama-2-7b-hf',\n    quantization_config=_____,\n    device_map='auto'\n)\nprint(f'모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}')",
     "quant_config",
     "BitsAndBytesConfig로 4비트 양자화를 설정하면 7B 모델을 ~4GB VRAM에서 실행할 수 있습니다. device_map='auto'는 사용 가능한 GPU에 레이어를 자동 배치합니다. 양자화는 성능 손실을 최소화하면서 메모리를 획기적으로 절감합니다."),

    ("감정 분류 배치 처리",
     "from transformers import pipeline\n\nclassifier = pipeline('zero-shot-classification',\n                      model='facebook/bart-large-mnli')\n\nsequences = ['I love this product!', 'The service was terrible']\ncandidate_labels = ['positive', 'negative', 'neutral']\n\nfor text in sequences:\n    result = classifier(text, _____)\n    print(f'{text}: {result[\"labels\"][0]}')",
     "candidate_labels",
     "zero-shot-classification은 학습 없이 임의의 라벨로 분류합니다. candidate_labels 리스트를 두 번째 인자로 전달하며, 결과의 labels[0]이 가장 확률 높은 레이블입니다. Fine-tuning 없이도 커스텀 분류가 가능합니다."),

    ("임베딩 유사도 기반 검색",
     "import numpy as np\nfrom transformers import AutoTokenizer, AutoModel\nimport torch\n\ndef get_embedding(text, tokenizer, model):\n    inputs = tokenizer(text, return_tensors='pt', padding=True)\n    with torch._____():\n        outputs = model(**inputs)\n    return outputs.last_hidden_state[:, 0, :].numpy()\n",
     "no_grad",
     "torch.no_grad()는 추론(inference) 시 그래디언트 계산을 비활성화합니다. 메모리 사용량을 줄이고 속도를 높입니다. 학습(training)이 아닌 모델 사용 시에는 항상 no_grad() 컨텍스트를 사용해야 합니다."),

    ("LLM 응답 JSON 파싱",
     "from openai import OpenAI\nimport json\n\nclient = OpenAI()\nresponse = client.chat.completions.create(\n    model='gpt-4o-mini',\n    messages=[{\n        'role': 'user',\n        'content': '이름, 나이, 직업을 JSON 형식으로 알려줘. 예시: {\"name\": \"...\"}'\n    }],\n    response_format={_____: 'json_object'}\n)\n\ndata = json.loads(response.choices[0].message.content)\nprint(data)",
     "'type'",
     "response_format={'type': 'json_object'}를 설정하면 모델이 항상 유효한 JSON을 반환합니다. 이를 JSON 모드라고 하며, json.loads()로 바로 파싱할 수 있습니다. 구조화된 데이터 추출 시 필수 옵션입니다.")
]

for i, (title, code, ans, explain) in enumerate(cc_data):
    questions.append({
        "chapter_name": chapter_name, "type": "코드 완성형", "difficulty": "medium", "id": str(3101 + i),
        "question": f"{title} 코드를 완성하세요.\n```python\n{code}\n```",
        "answer": ans,
        "why": explain,
        "hint": title,
    })

def get_questions():
    return questions
