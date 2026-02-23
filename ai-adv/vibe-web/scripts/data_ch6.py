
chapter_name = "서비스 구축 및 배포"

questions = []

# --- 100 MCQs ---
mcq_data = [
    # 1. 서비스 구축 아키텍처 및 프런트엔드 (1-30)
    ("LLM 서비스를 구축할 때 가장 먼저 고려해야 할 아키텍처 구성 요소는?", ["로드 밸런서 설정", "프런트엔드, 백엔드 서버, 그리고 LLM API/모델 엔진의 연결 구조", "CDN 설정", "데이터베이스 인덱스", "API 게이트웨이"], "프런트엔드, 백엔드 서버, 그리고 LLM API/모델 엔진의 연결 구조", "사용자의 입력을 받고 AI의 응답을 전달하는 전체적인 시스템 설계가 핵심입니다. 각 컴포넌트가 어떻게 통신하는지 정의해야 이후의 개발 방향이 결정됩니다.", "서비스 아키텍처", "6001", "easy"),
    ("사용자가 웹 브라우저를 통해 채팅 인터페이스를 보고 메시지를 입력하는 영역은?", ["백엔드 (Backend)", "프런트엔드 (Frontend)", "데이터베이스 (Database)", "GPU 서버", "네트워크 라우터"], "프런트엔드 (Frontend)", "React, Vue, Next.js 등을 사용하여 사용자 경험(UX)을 구현하는 영역입니다. 사용자가 직접 보고 상호작용하는 모든 UI 요소가 프런트엔드에 속합니다.", "프런트엔드", "6002", "easy"),
    ("LLM 서비스에서 실시간으로 글자가 하나씩 생성되는 느낌을 주기 위해 사용하는 기술은?", ["이미지 캡처", "스트리밍 (Streaming / Server-Sent Events)", "파일 다운로드", "전체 화면 캡처", "단순 페이지 새로고침"], "스트리밍 (Streaming / Server-Sent Events)", "사용자가 답변이 완료될 때까지 기다리지 않고 생성 과정을 즉시 볼 수 있게 합니다. SSE는 서버에서 클라이언트로 단방향 실시간 데이터를 전송하는 표준 방식입니다.", "스트리밍", "6003", "easy"),
    ("프런트엔드에서 사용자의 이전 대화 내역을 유지하여 보여주는 기능의 명칭은?", ["Chat History", "File Explorer", "Task Manager", "Disk Utility", "Control Panel"], "Chat History", "과거의 대화를 리스트 형태로 관리하여 사용자가 맥락을 파악하도록 돕습니다. LLM 서비스에서 멀티턴 대화가 자연스럽게 이어지게 하는 핵심 UI 기능입니다.", "채팅 히스토리", "6004", "easy"),
    ("사용자가 입력한 메시지를 백엔드로 전달할 때 주로 사용하는 HTTP 메서드는?", ["GET", "POST", "DELETE", "HEAD", "OPTIONS"], "POST", "메시지 본문(Payload)에 데이터를 담아 서버로 안전하게 전송하기 위해 쓰입니다. GET은 URL에 데이터가 노출되므로 민감한 메시지 전송에는 POST가 적합합니다.", "POST 메서드", "6005", "easy"),
    ("웹 서비스 구축 시 화면의 레이아웃과 디자인을 담당하는 언어는?", ["Python", "SQL", "CSS", "C++", "Java"], "CSS", "채팅창의 너비, 배경색, 글꼴 스타일 등을 정의하여 시각적인 완성도를 높입니다. Tailwind CSS, Bootstrap 등의 프레임워크로 더욱 빠르게 스타일링할 수 있습니다.", "CSS", "6006", "easy"),
    ("반응형 웹 디자인(Responsive Web Design)이 LLM 서비스에서 중요한 이유는?", ["코드가 예뻐서", "PC뿐만 아니라 모바일 환경에서도 최적화된 채팅 UI를 제공하기 위해", "속도가 빨라져서", "비용이 저렴해서", "보안이 강화되어서"], "PC뿐만 아니라 모바일 환경에서도 최적화된 채팅 UI를 제공하기 위해", "다양한 기기에서 인공지능과 원활하게 대화할 수 있는 접근성을 확보합니다. CSS 미디어 쿼리를 통해 화면 크기에 따라 레이아웃이 동적으로 변환됩니다.", "반응형 디자인", "6007", "hard"),
    ("프런트엔드 프레임워크 중 'Next.js'를 LLM 앱 개발에 자주 사용하는 주된 장점은?", ["게임 개발에 특화됨", "서버 사이드 렌더링(SSR)과 API Routes를 통해 풀스택 개발이 용이함", "운영체제를 직접 만듦", "포토샵 기능 내장", "엑셀과 완벽 호환"], "서버 사이드 렌더링(SSR)과 API Routes를 통해 풀스택 개발이 용이함", "백엔드 로직과 프런트엔드 UI를 하나의 프로젝트에서 효율적으로 관리할 수 있습니다. App Router를 사용하면 Server Components로 LLM 호출 로직을 서버에서 직접 처리할 수 있습니다.", "Next.js 장점", "6008", "hard"),
    ("채팅창에서 AI 답변이 생성 중임을 알리는 '로딩 애니메이션'의 UX적 효과는?", ["배터리 절약", "사용자에게 시스템이 정상적으로 응답 중임을 알려 불안감을 해소함", "인터넷 속도 향상", "자동 오타 교정", "화면 보호"], "사용자에게 시스템이 정상적으로 응답 중임을 알려 불안감을 해소함", "생성 시간이 다소 소요되는 LLM의 특성상 사용자 대기 경험을 관리하는 것이 중요합니다. 점 세 개가 순서대로 점멸하는 '타이핑 인디케이터'가 대표적인 패턴입니다.", "로딩 UI", "6009", "hard"),
    ("마크다운(Markdown) 렌더링 기능이 LLM 프런트엔드에 필수적인 이유는?", ["파일 용량을 줄이기 위해", "코드 블록, 표, 굵은 글씨 등 AI의 응답을 가독성 있게 표현하기 위해", "광고를 띄우기 위해", "한글을 영어로 바꾸기 위해", "로그인을 대신 하기 위해"], "코드 블록, 표, 굵은 글씨 등 AI의 응답을 가독성 있게 표현하기 위해", "AI가 생성한 다양한 형식의 정보를 사용자가 읽기 편한 구조로 보여줍니다. react-markdown, marked.js 등의 라이브러리로 구현하며, 코드 하이라이팅도 함께 적용합니다.", "마크다운", "6010", "hard"),

    # 2. 백엔드 및 API 개발 (31-60)
    ("사용자의 요청을 받아 LLM 모델에 전달하고 결과를 가공하여 응답하는 서버 영역은?", ["프런트엔드", "백엔드 (Backend / API Server)", "캐시 서버 구성", "메시지 큐(Message Queue)", "로드 밸런서 설정"], "백엔드 (Backend / API Server)", "비즈니스 로직 처리, 보안 인증, 외부 API 연동 등을 담당하는 핵심 두뇌입니다. 백엔드는 사용자에게 직접 노출되지 않으므로 API Key 등 민감한 정보를 안전하게 보관합니다.", "백엔드", "6011", "easy"),
    ("파이썬 기반의 빠르고 현대적인 웹 프레임워크로, LLM API 서버 구축에 많이 쓰이는 것은?", ["Flask", "Django", "FastAPI", "PHP", "JSP"], "FastAPI", "비동기(Async) 처리가 강력하고 자동 API 문서 생성(Swagger) 기능을 제공하여 효율적입니다. Pydantic으로 요청/응답 타입을 자동 검증하여 버그를 줄일 수 있습니다.", "FastAPI", "6012", "easy"),
    ("API 서버 내부에서 LLM 서비스를 제공하는 회사(OpenAI, Anthropic 등)와 통신할 때 필요한 인증 수단은?", ["ID/비밀번호", "API Key (인증 키)", "SSL 인증서", "OAuth 토큰", "IP 화이트리스트"], "API Key (인증 키)", "인가된 사용자만 모델 사용량을 소모할 수 있도록 관리하는 보안 장치입니다. API Key는 절대 클라이언트 코드에 노출해서는 안 되며 반드시 서버 측 환경 변수로 관리해야 합니다.", "API Key", "6013", "easy"),
    ("백엔드 서버에서 여러 사용자의 요청을 동시에 효율적으로 처리하기 위해 사용하는 프로그래밍 방식은?", ["순차 실행", "비동기 프로그래밍 (Asynchronous Programming)", "멀티 스레드 블로킹", "단일 프로세스 동기 처리", "배치 처리"], "비동기 프로그래밍 (Asynchronous Programming)", "하나의 요청이 처리되는 동안(예: AI 응답 대기) 다른 요청을 처리하여 효율을 극대화합니다. Python의 asyncio와 FastAPI의 async def 조합이 LLM 서버에서 표준입니다.", "비동기", "6014", "hard"),
    ("백엔드 서버에서 민감한 API Key를 코드에 직접 노출하지 않고 관리하는 방법은?", ["코드에 주석으로 남기기", "환경 변수(Environment Variables) 파일(.env) 사용", "Git 공개 저장소에 업로드", "Slack 채널에 공유", "README에 작성"], "환경 변수(Environment Variables) 파일(.env) 사용", "보안 유출을 방지하기 위해 설정값과 실행 코드를 분리하는 모범 사례입니다. .env 파일은 반드시 .gitignore에 추가하여 Git 저장소에 올라가지 않도록 해야 합니다.", "환경 변수", "6015", "easy"),
    ("API 서버가 클라이언트에게 데이터를 줄 때 가장 흔히 사용하는 가벼운 데이터 형식은?", ["XML", "JSON", "CSV", "TXT", "XLSX"], "JSON", "키-값 쌍으로 이루어져 프로그래밍 언어 간 데이터 교환에 최적화되어 있습니다. JavaScript 객체와 거의 동일한 문법 덕분에 웹 브라우저에서 별도 파싱 없이 바로 사용할 수 있습니다.", "JSON", "6016", "easy"),
    ("백엔드 서버에서 답변 생성 전, 사용자의 질문이 부적절한지 검사하는 과정을 무엇이라 하나?", ["Preprocessing / Content Filtering", "Postprocessing", "Hardening", "Formatting", "Deleting"], "Preprocessing / Content Filtering", "유해한 콘텐츠 생성을 방지하고 서비스 가이드라인을 준수하기 위한 안전 단계입니다. OpenAI Moderation API 같은 외부 서비스를 활용하거나 자체 규칙 기반 필터를 구현할 수 있습니다.", "필터링", "6017", "hard"),
    ("데이터베이스(DB)를 백엔드에 연동하는 주된 이유 중 '대화 영속성'이란?", ["속도를 높이는 것", "대화가 종료되어도 나중에 이전 내역을 다시 불러올 수 있게 저장하는 것", "글자 수를 늘리는 것", "영어로 대화하는 것", "비용을 결제하는 것"], "대화가 종료되어도 나중에 이전 내역을 다시 불러올 수 있게 저장하는 것", "사용자가 다시 접속했을 때 과거의 맥락을 이어서 대화할 수 있도록 합니다. PostgreSQL이나 MongoDB 같은 DB에 사용자 ID와 매핑하여 대화 기록을 저장합니다.", "영속성", "6018", "hard"),
    ("API 문서 자동 생성 도구인 'Swagger'를 통해 얻을 수 있는 이점은?", ["코딩을 대신 해줌", "프런트엔드 개발자가 서버 API의 사양을 쉽게 파악하고 테스트할 수 있음", "API 게이트웨이 자동 설정", "데이터베이스 인덱스 생성", "로드 밸런서 자동 구성"], "프런트엔드 개발자가 서버 API의 사양을 쉽게 파악하고 테스트할 수 있음", "협업 효율성을 높이고 API 호출 시 에러를 줄여주는 강력한 도구입니다. FastAPI는 /docs 경로에서 Swagger UI를, /redoc 경로에서 ReDoc 문서를 자동으로 제공합니다.", "Swagger", "6019", "hard"),
    ("백엔드에서 LLM 응답을 받은 후, 특정 형식에 맞춰 텍스트를 정돈하는 과정을 무엇이라 하나?", ["Ingestion", "Post-processing (후처리)", "Encoding", "Scaling", "Training"], "Post-processing (후처리)", "불필요한 공백 제거, 특수 문자 정제, 특정 포맷 변환 등을 수행합니다. JSON 형식으로 응답을 요청했을 때 LLM이 마크다운 코드블록을 포함시키는 경우 후처리로 정제합니다.", "후처리", "6020", "hard"),

    # 3. 배포 및 클라우드 인프라 (61-100)
    ("작성한 코드를 실제 인터넷 상에서 누구나 접속 가능한 상태로 만드는 과정을 무엇이라 하나?", ["Coding", "Deployment (배포)", "Debugging", "Designing", "Deleting"], "Deployment (배포)", "로컬 환경을 넘어 실제 서버 인프라에 서비스를 올리는 최종 단계입니다. Docker 컨테이너화 → CI/CD 파이프라인 → 클라우드 서버 실행의 순서로 진행되는 것이 일반적입니다.", "배포의 정의", "6021", "easy"),
    ("서버를 직접 구매하지 않고 가상의 컴퓨팅 자원을 빌려 쓰는 최신 서비스 형태는?", ["Offline Store", "Cloud Computing (클라우드 컴퓨팅)", "Hard Disk", "USB Memory", "Floppy Disk"], "Cloud Computing (클라우드 컴퓨팅)", "AWS, Google Cloud, Azure 등 유연한 자원 확장이 가능한 인프라를 의미합니다. 사용한 만큼만 비용을 내는 종량제 방식으로 초기 설비 투자 없이 서비스를 시작할 수 있습니다.", "클라우드", "6022", "easy"),
    ("코드와 실행 환경을 하나로 묶어 어디서나 동일하게 실행되도록 만드는 가상화 기술은?", ["VMware", "Docker (도커)", "캐시 서버 구성", "메시지 큐(Message Queue)", "API 게이트웨이"], "Docker (도커)", "컨테이너 기술을 통해 '내 컴퓨터에선 되는데 서버에선 안 되는' 문제를 해결합니다. Dockerfile에 실행 환경을 정의하면 OS에 무관하게 동일한 환경이 재현됩니다.", "도커", "6023", "hard"),
    ("배포 시 트래픽이 몰릴 때 서버의 개수를 자동으로 늘려주는 기능을 무엇이라 하나?", ["Auto-Save", "Auto-Scaling (오토 스케일링)", "Auto-Focus", "Auto-Complete", "Auto-Pilot"], "Auto-Scaling (오토 스케일링)", "사용자의 접속량에 따라 인프라를 유연하게 조절하여 안정적인 서비스를 유지합니다. CPU 사용률 80% 초과 시 서버를 추가하고, 낮아지면 줄이는 방식으로 비용을 최적화합니다.", "오토 스케일링", "6024", "hard"),
    ("서비스의 주소(URL)를 쉽게 기억할 수 있도록 연결해주는 시스템은?", ["CPU", "DNS (Domain Name System)", "RAM", "GPU", "SSD"], "DNS (Domain Name System)", "IP 주소 대신 'example.com' 같은 문자로 서버에 접속하게 해줍니다. 도메인을 구매한 후 DNS 레코드(A Record)에 서버 IP를 등록하면 주소로 접속할 수 있습니다.", "DNS", "6025", "hard"),
    ("전 세계 사용자에게 콘텐츠를 빠르게 전달하기 위해 서버를 분산 배치하는 기술은?", ["FTP", "CDN (Content Delivery Network)", "HTTP", "LAN", "WAN"], "CDN (Content Delivery Network)", "지리적으로 가까운 서버에서 데이터를 전송하여 응답 속도를 개선합니다. 정적 파일(이미지, JS, CSS)을 CDN에 올리면 원본 서버 부하를 크게 줄일 수 있습니다.", "CDN", "6026", "easy"),
    ("배포 후 서비스의 상태(에러 발생 여부, CPU 점유율 등)를 실시간으로 확인하는 작업은?", ["Designing", "Monitoring (모니터링)", "Planning", "Meeting", "Resting"], "Monitoring (모니터링)", "장애를 미리 예방하고 성능 병목을 파악하기 위한 필수 운영 활동입니다. Prometheus + Grafana, Datadog, CloudWatch 등의 도구로 실시간 대시보드를 구성합니다.", "모니터링", "6027", "hard"),
    ("코드 변경 사항을 자동으로 테스트하고 서버에 즉시 배포하는 자동화 파이프라인은?", ["CI/CD", "GUI", "CLI", "IDE", "USB"], "CI/CD", "지속적 통합(CI)과 지속적 배포(CD)를 통해 개발 생산성을 획기적으로 높입니다. GitHub Actions, GitLab CI, Jenkins 등의 도구로 구현하며, 테스트 실패 시 배포가 자동 중단됩니다.", "CI/CD", "6028", "hard"),
    ("인터넷 상의 위협(해킹, DDoS 등)으로부터 서버를 보호하기 위해 앞단에 두는 네트워크 보안 장치는?", ["Router", "Firewall (방화벽)", "메시지 큐(Message Queue)", "캐시 서버 구성", "API 게이트웨이"], "Firewall (방화벽)", "허용되지 않은 접근을 차단하여 소중한 데이터와 시스템을 지킵니다. AWS Security Group, GCP Firewall Rules 같은 클라우드 방화벽으로 인바운드/아웃바운드 규칙을 설정합니다.", "방화벽", "6029", "hard"),
    ("배포된 서비스에 보안 연결(HTTPS)을 적용하기 위해 필요한 인증서는?", ["졸업 증명서", "SSL/TLS 인증서", "운전 면허증", "건강 진단서", "경력 증명서"], "SSL/TLS 인증서", "데이터 전송 구간을 암호화하여 중간에서 정보를 가로채지 못하게 보호합니다. Let's Encrypt를 이용하면 무료로 SSL 인증서를 발급받고 자동 갱신까지 설정할 수 있습니다.", "SSL/TLS", "6030", "hard"),

    # 추가 70문제 (프런트/백/배포 응용)
    ("API 호출 횟수를 제한하여 특정 사용자가 서버 자원을 독점하지 못하게 하는 정책은?", ["Rate Limiting", "Open Access", "Free Pass", "Full Speed", "No Limit"], "Rate Limiting", "공정하고 안정적인 서비스 운영을 위해 필수적인 트래픽 제어 방식입니다. 분당 요청 수를 Redis에 기록하고 초과 시 429 Too Many Requests를 반환하는 방식으로 구현합니다.", "Rate Limiting", "6031", "easy"),
    ("프런트엔드에서 AI의 답변 길이를 시각적으로 제한하거나 '더보기' 버튼을 만드는 UI 설계의 목적은?", ["데이터 삭제", "답변이 너무 길어 화면을 가득 채우는 것을 막고 가독성을 유지하기 위해", "비용 청구", "영역 숨기기", "오타 유도"], "답변이 너무 길어 화면을 가득 채우는 것을 막고 가독성을 유지하기 위해", "깔끔한 인터페이스 유지를 위한 레이아웃 관리 전략입니다. CSS의 line-clamp 속성과 React 상태 관리를 결합하면 손쉽게 구현할 수 있습니다.", "UI 가독성", "6032", "easy"),
    ("백엔드에서 '로그(Log)'를 남기는 것이 중요한 이유는?", ["코드를 길게 하려고", "문제가 발생했을 때 원인을 추적하고 시스템 방문 기록을 분석하기 위해", "종이를 아끼려고", "서버를 끄려고", "데이터베이스를 지우려고"], "문제가 발생했을 때 원인을 추적하고 시스템 방문 기록을 분석하기 위해", "운영 중 발생하는 이슈를 해결하는 가장 강력한 단서가 됩니다. Python의 logging 모듈로 레벨별(DEBUG/INFO/ERROR)로 구조화된 로그를 남기는 것이 모범 사례입니다.", "Logging", "6033", "easy"),
    ("배포 환경과 로컬 개발 환경의 설정을 분리하는 가장 좋은 방법은?", ["코드를 두 번 짜기", "환경 변수를 활용하여 주소나 키값을 동적으로 로드하기", "종이에 적어두기", "전부 다 지우기", "로그인 안 하기"], "환경 변수를 활용하여 주소나 키값을 동적으로 로드하기", "하나의 소스 코드로 여러 환경에서 안정적으로 동작하게 만드는 설계 방식입니다. .env.development, .env.production 파일로 환경별 설정을 분리하는 것이 표준 관행입니다.", "환경 분리", "6034", "medium"),
    ("도커 컨테이너를 여러 개 관리하고 배포를 자동화해주는 '오케스트레이션' 도구는?", ["Kubernetes (쿠버네티스)", "메모장", "엑셀", "파워포인트", "API 게이트웨이"], "Kubernetes (쿠버네티스)", "대규모 서비스의 컨테이너 운영을 자동화하는 업계 표준 도구입니다. Pod, Deployment, Service 등의 개념으로 컨테이너의 실행, 스케일링, 자가치유를 자동화합니다.", "Kubernetes", "6035", "medium"),
    ("사용자가 채팅창에 대용량 파일을 업로드할 때 백엔드에서 고려해야 할 점은?", ["파일 이름", "파일 크기 제한(Body Size Limit) 및 안전한 저장소(S3 등) 확보", "파일의 색깔", "API 게이트웨이 설정", "로드 밸런서 구성"], "파일 크기 제한(Body Size Limit) 및 안전한 저장소(S3 등) 확보", "서버 자원 고갈을 막고 데이터를 안정적으로 관리하기 위한 설계입니다. Nginx의 client_max_body_size 설정으로 파일 크기를 제한하고, AWS S3 Presigned URL로 안전하게 저장합니다.", "파일 업로드 처리", "6036", "medium"),
    ("프런트엔드에서 '다크 모드'를 지원할 때 얻는 UX적 장점은?", ["전기세 폭탄", "사용자의 눈 피로도를 낮추고 세련된 디자인 감성을 제공함", "API 속도 2배", "AI의 지능 향상", "자동 번역"], "사용자의 눈 피로도를 낮추고 세련된 디자인 감성을 제공함", "장시간 채팅을 이용하는 사용자에게 편리한 시각적 환경을 선사합니다. CSS의 prefers-color-scheme 미디어 쿼리로 시스템 설정을 감지하고 Tailwind의 dark: 클래스로 구현합니다.", "다크 모드", "6037", "hard"),
    ("백엔드 서버 배포 시 '무중단 배포'를 하는 이유는?", ["전기를 아끼려고", "새로운 버전 업데이트 중에도 사용자가 끊김 없이 서비스를 이용하게 하려고", "캐시 서버를 교체하려고", "데이터베이스 인덱스를 재구성하려고", "API 게이트웨이를 재시작하려고"], "새로운 버전 업데이트 중에도 사용자가 끊김 없이 서비스를 이용하게 하려고", "업데이트로 인한 서비스 중지 시간을 없애 신뢰도를 유지합니다. Rolling Update, Blue-Green, Canary Deployment 등이 대표적인 무중단 배포 전략입니다.", "무중단 배포", "6038", "hard"),
    ("API 응답 시간을 측정할 때 사용하는 단위는?", ["미터(m)", "밀리초 (ms)", "킬로그램(kg)", "리터(L)", "온도(℃)"], "밀리초 (ms)", "1,000분의 1초 단위로 측정하여 서비스 반응성을 정교하게 관리합니다. LLM API는 보통 500ms~3000ms 범위이며, 이를 줄이기 위해 스트리밍과 캐싱을 함께 사용합니다.", "Latency 단위", "6039", "hard"),
    ("배포 후 'Health Check' API를 만드는 주된 목적은?", ["건강 검진 기록", "서버가 현재 정상적으로 살아있는지 주기적으로 확인하기 위해", "이름 짓기", "데이터베이스 인덱스 점검", "캐시 서버 상태 확인"], "서버가 현재 정상적으로 살아있는지 주기적으로 확인하기 위해", "모니터링 시스템이나 로드 밸런서가 서버의 생존 여부를 판단하는 척도가 됩니다. /health 엔드포인트가 200을 반환하지 않으면 로드 밸런서가 해당 서버를 자동으로 제외합니다.", "Health Check", "6040", "hard"),
    ("프런트엔드에서 '복사하기' 버튼을 구현하는 이유는?", ["종이가 없어서", "AI의 긴 답변이나 코드를 사용자가 간편하게 복사할 수 있도록 하여 편의성을 높이기 위해", "인터넷을 중단하려고", "화면을 끄려고", "파일을 지우려고"], "AI의 긴 답변이나 코드를 사용자가 간편하게 복사할 수 있도록 하여 편의성을 높이기 위해", "생산성 향상을 위한 작은 디테일이 사용자 경험을 크게 개선합니다. navigator.clipboard.writeText() API로 구현하며, 성공 시 '복사됨!' 토스트를 잠깐 보여주는 것이 관례입니다.", "복사 기능", "6041", "easy"),
    ("백엔드에서 'CORS' 에러가 발생하는 상황은?", ["API 게이트웨이가 없을 때", "서로 다른 도메인(주소) 간에 리소스를 요청할 때 보안 정책으로 차단되는 경우", "인터넷이 끊겼을 때", "데이터베이스 인덱스가 없을 때", "로드 밸런서가 없을 때"], "서로 다른 도메인(주소) 간에 리소스를 요청할 때 보안 정책으로 차단되는 경우", "보안을 위해 브라우저가 타 도메인으로의 요청을 제어하는 정책입니다. FastAPI에서는 CORSMiddleware로 허용 오리진을 명시하여 해결합니다.", "CORS", "6042", "easy"),
    ("서버 배포 시 '리전(Region)' 선택 시 가장 중요한 기준은?", ["리전 이름의 알파벳 순서", "사용자와 가장 가까운 지리적 위치(속도 향상 및 지연 감소)", "날씨가 좋은 곳", "유명 관광지 근처", "내가 가고 싶은 나라"], "사용자와 가장 가까운 지리적 위치(속도 향상 및 지연 감소)", "한국 사용자라면 서울 리전(ap-northeast-2)을 선택하는 것이 가장 빠른 성능을 냅니다. 데이터 주권 법규나 컴플라이언스 요건도 리전 선택에 영향을 미칩니다.", "리전 선택", "6043", "medium"),
    ("백엔드에서 사용되는 '데이터베이스 인덱싱'의 효과는?", ["글자 크기 키우기", "데이터 검색 속도를 비약적으로 높여 응답 시간을 단축함", "용량 늘리기", "파일 삭제", "API 게이트웨이 속도 향상"], "데이터 검색 속도를 비약적으로 높여 응답 시간을 단축함", "수백만 건의 대화 기록 속에서 원하는 내용을 순식간에 찾아내게 합니다. 자주 조회되는 user_id, created_at 컬럼에 인덱스를 걸면 전체 테이블 스캔을 피할 수 있습니다.", "DB 인덱스", "6044", "medium"),
    ("프런트엔드에서 'Toast 메시지'(짧게 떴다 사라지는 알림)의 용도는?", ["토스트 구워 먹기", "작업 완료나 에러 발생을 사용자에게 방해되지 않게 살짝 알려줌", "컴퓨터 부팅", "로그인 강제", "광고 노출"], "작업 완료나 에러 발생을 사용자에게 방해되지 않게 살짝 알려줌", "현재 작업 흐름을 깨지 않으면서 정보를 전달하는 유용한 UI 요소입니다. react-hot-toast, Sonner 등의 라이브러리로 손쉽게 구현할 수 있습니다.", "Toast UI", "6045", "medium"),
    ("백엔드 개발 시 '단위 테스트(Unit Test)'의 역할은?", ["배터리 수명 체크", "개별 기능(함수)이 의도대로 정확히 동작하는지 코드로 검증하여 버그를 방지함", "타이핑 속도 측정", "캐시 서버 성능 측정", "API 게이트웨이 응답 시간 측정"], "개별 기능(함수)이 의도대로 정확히 동작하는지 코드로 검증하여 버그를 방지함", "코드를 신뢰할 수 있게 만들고 향후 수정 시 발생하는 사이드 이펙트를 막아줍니다. pytest를 사용하면 FastAPI 엔드포인트 테스트와 비즈니스 로직 테스트를 모두 작성할 수 있습니다.", "Unit Test", "6046", "medium"),
    ("배포 시 'Git'을 사용하는 근거 중 하나인 '버전 관리'란?", ["컴퓨터를 새로 사는 것", "코드의 수정 이력을 기록하고 필요할 때 과거 상태로 되돌리는 것", "이름을 멋지게 짓는 것", "파일을 다 합치는 것", "인터넷 쇼핑"], "코드의 수정 이력을 기록하고 필요할 때 과거 상태로 되돌리는 것", "여러 개발자가 협업하고 시스템을 안정적으로 업데이트하는 기반이 됩니다. 브랜치 전략(Git Flow, GitHub Flow)을 함께 사용하면 병렬 개발이 가능합니다.", "Git 버전 관리", "6047", "medium"),
    ("프런트엔드에서 'State Management'(상태 관리)가 필요한 이유는?", ["기분이 안 좋아서", "채팅방 내용, 로그인 정보 등 실시간으로 변하는 데이터를 일관되게 관리하기 위해", "캐시 서버를 설정하려고", "API 게이트웨이를 관리하려고", "로드 밸런서를 제어하려고"], "채팅방 내용, 로그인 정보 등 실시간으로 변하는 데이터를 일관되게 관리하기 위해", "복잡한 앱의 데이터 흐름을 꼬이지 않게 잡아주는 핵심 기술입니다. React의 useState, useContext, Zustand, Redux 등의 도구로 구현합니다.", "상태 관리", "6048", "medium"),
    ("클라우드 서비스 중 'Serverless'(서버리스)의 특징은?", ["서버가 아예 없는 것", "사용자가 서버 관리를 직접 하지 않고 실행된 만큼만 비용을 지불하는 방식", "메시지 큐가 없는 것", "API 게이트웨이가 없는 것", "로드 밸런서가 없는 것"], "사용자가 서버 관리를 직접 하지 않고 실행된 만큼만 비용을 지불하는 방식", "관리가 편하고 초기 비용 부담이 적어 가벼운 AI 앱 배포에 좋습니다. AWS Lambda, Vercel Edge Functions 등이 대표적이며, 콜드 스타트 지연이 단점입니다.", "서버리스", "6049", "hard"),
    ("백엔드 서버에서 '세션(Session)'과 '쿠키(Cookie)'의 역할은?", ["과자 먹기", "사용자의 로그인 상태나 방문 정보를 유지하여 개인화된 경험을 제공함", "컴퓨터 부품 이름", "인터넷 브라우저 이름", "사이트 주소"], "사용자의 로그인 상태나 방문 정보를 유지하여 개인화된 경험을 제공함", "사용자를 식별하여 '내 대화 내역'을 안전하게 보여주는 기반이 됩니다. JWT 기반 인증이 확산되면서 Stateless 방식이 세션 대신 많이 쓰입니다.", "세션과 쿠키", "6050", "medium"),
    ("프런트엔드에서 'Skeleton Screen'을 보여주는 의도는?", ["해골 그림 그리기", "데이터 로딩 중 빈 화면 대신 레이아웃 윤곽을 미리 보여줘 체감 속도를 높임", "사이트 끄기", "로그아웃", "광고 띄우기"], "데이터 로딩 중 빈 화면 대신 레이아웃 윤곽을 미리 보여줘 체감 속도를 높임", "사용자가 콘텐츠가 곧 나올 것임을 인지하게 하여 시각적 만족감을 줍니다. 실제 로딩 시간이 같아도 Skeleton UI가 있으면 더 빠르게 느껴지는 심리적 효과가 있습니다.", "Skeleton UI", "6051", "easy"),
    ("API 서버의 'Versioning'(버전 관리)은 왜 하나?", ["숫자 공부하려고", "기존 사용자 기능을 유지하면서 새로운 기능을 안전하게 추가하고 업데이트하기 위해", "이름이 없어서", "순서를 정하려고", "비용을 늘리려고"], "기존 사용자 기능을 유지하면서 새로운 기능을 안전하게 추가하고 업데이트하기 위해", "예: /v1/chat과 /v2/chat을 구분하여 하위 호환성을 지키는 전문적인 개발 방식입니다. APIRouter에 prefix='/api/v1'을 설정하면 FastAPI에서 쉽게 버전을 분리할 수 있습니다.", "API 버전", "6052", "easy"),
    ("클라우드 인프라에서 'VPC'(가상 사설 클라우드)의 역할은?", ["가상 게임 공간", "사설 네트워크를 구축하여 독립적이고 안전한 리소스 환경을 제공함", "인터넷 카페", "파일 공유 폴더", "컴퓨터 부품"], "사설 네트워크를 구축하여 독립적이고 안전한 리소스 환경을 제공함", "기업의 데이터를 공용 인터넷과 격리하여 보안을 극대화하는 성벽 역할을 합니다. 퍼블릭 서브넷(웹 서버)과 프라이빗 서브넷(DB 서버)을 분리하는 것이 보안 모범 사례입니다.", "VPC", "6053", "medium"),
    ("백엔드 비즈니스 로직 중 'Validation'(검증)이란?", ["유통기한 확인", "입력 데이터가 형식에 맞는지, 위험한 코드가 섞여 있지 않은지 확인하는 절차", "인기 투표", "친구 찾기", "비밀번호 삭제"], "입력 데이터가 형식에 맞는지, 위험한 코드가 섞여 있지 않은지 확인하는 절차", "시스템 에러를 막고 악의적인 데이터 주입 공격을 원천 차단합니다. Pydantic Field()의 min_length, max_length, ge, le 등으로 FastAPI에서 자동 검증이 가능합니다.", "Validation", "6054", "medium"),
    ("배포 전략 중 'Blue-Green Deployment'의 특징은?", ["색깔 고르기", "기존 서버(Blue)와 새 서버(Green)를 동시에 띄워 교체하며 장애를 최소화함", "로드 밸런서를 교체함", "캐시 서버를 초기화함", "API 게이트웨이를 재시작함"], "기존 서버(Blue)와 새 서버(Green)를 동시에 띄워 교체하며 장애를 최소화함", "리스크가 적고 실패 시 즉시 이전 버전으로 되돌릴 수 있는 안정적인 배포 기법입니다. 로드 밸런서의 타겟을 Blue에서 Green으로 전환하는 방식으로 동작합니다.", "Blue-Green", "6055", "medium"),
    ("프런트엔드 빌드(Build) 과정에서 하는 일은?", ["집 짓기", "소스 코드를 브라우저가 읽기 최적화된 파일로 압축하고 변환하는 과정", "파일 하나씩 읽기", "API 게이트웨이 설정", "캐시 서버 초기화"], "소스 코드를 브라우저가 읽기 최적화된 파일로 압축하고 변환하는 과정", "용량을 줄이고 보안을 강화하여 실제 서비스 성능을 높이는 전처리입니다. Webpack, Vite, Turbopack 등이 빌드 도구로 사용되며, 코드 분할(Code Splitting)로 초기 로딩 속도를 개선합니다.", "Frontend Build", "6056", "medium"),
    ("API 서버의 'Rate Limit Exceeded' 에러를 받았을 때 프런트엔드의 대처는?", ["서버를 계속 공격함", "잠시 기다려달라는 메시지를 보여주고 재시도 버튼을 안내함", "로그아웃", "글자 다 지우기", "화면 끄기"], "잠시 기다려달라는 메시지를 보여주고 재시도 버튼을 안내함", "서버 정책을 사용자에게 친절하게 안내하여 서비스에 대한 긍정적 경험을 유지합니다. Retry-After 헤더 값을 읽어 카운트다운 타이머와 함께 안내하면 더 좋은 UX를 제공합니다.", "과부하 대응", "6057", "medium"),
    ("데이터베이스 서버를 '분리'해서 구축하는 장점은?", ["관리하기 귀찮음", "웹 서버와 데이터 저장 서버의 역할을 나누어 성능을 최적화하고 보안을 강화함", "컴퓨터 대수 줄이기", "캐시 서버 제거", "로드 밸런서 비용 절약"], "웹 서버와 데이터 저장 서버의 역할을 나누어 성능을 최적화하고 보안을 강화함", "전문적인 인프라 구성을 통해 서비스의 안정성과 확장성을 확보합니다. DB 서버는 프라이빗 서브넷에 배치하여 직접 외부 접근을 차단하는 것이 보안 모범 사례입니다.", "DB 분리", "6058", "medium"),
    ("배포 시 '환경 설정 파일(.yml, .json 등)'을 사용하는 이유는?", ["그림 그리려고", "소소한 설정 변경 시 코드 수정 없이 파일만 갈아끼워 적용하기 위해", "글자 수 채우기", "멋있어 보이려고", "비밀번호 저장소"], "소소한 설정 변경 시 코드 수정 없이 파일만 갈아끼워 적용하기 위해", "유연하고 유지보수가 쉬운 환경 관리를 가능하게 합니다. docker-compose.yml, kubernetes 매니페스트, GitHub Actions 워크플로우 파일이 대표적인 예입니다.", "Config File", "6059", "medium"),
    ("배포된 인벤토리를 관리하는 기술 용어 'IaC'(Infrastructure as Code)란?", ["인프라를 손으로 만들기", "서버 설정을 코드로 작성하여 자동화하고 버전 관리하는 기술", "컴퓨터 부품 이름", "비밀번호 분실", "인터넷 쇼핑"], "서버 설정을 코드로 작성하여 자동화하고 버전 관리하는 기술", "사람의 실수를 줄이고 수천 대의 서버를 일관되게 관리할 수 있는 현대적 기술입니다. Terraform, AWS CloudFormation, Pulumi 등의 도구로 클라우드 인프라를 코드로 정의합니다.", "IaC", "6060", "medium"),

    # 나머지 40문제 (실무 종합)
    ("프런트엔드에서 '클립보드 API'를 사용하는 예시는?", ["AI가 생성한 코드를 버튼 하나로 복사하게 함", "사진 찍기", "인터넷 끊기", "로그아웃하기", "파일 삭제하기"], "AI가 생성한 코드를 버튼 하나로 복사하게 함", "사용자 편의성을 위한 실용적인 기능 구현 사례입니다. navigator.clipboard.writeText()를 async/await로 호출하고 성공 여부를 Toast로 알려줍니다.", "클립보드", "6061", "easy"),
    ("백엔드 서버에서 API 성능을 개선하기 위한 'Redis'의 역할은?", ["컴퓨터 수리", "자주 사용되는 데이터를 메모리에 임시 저장하여 초고속으로 응답(캐싱)함", "사진첩", "음악 재생", "동영상 편집"], "자주 사용되는 데이터를 메모리에 임시 저장하여 초고속으로 응답(캐싱)함", "DB 부하를 줄이고 응답 속도를 획기적으로 향상시켜 줍니다. 동일한 LLM 프롬프트에 대한 응답을 Redis에 캐싱하면 API 비용과 응답 시간을 동시에 줄일 수 있습니다.", "Redis/Caching", "6062", "easy"),
    ("배포된 웹사이트 주소 앞에 'https://'가 붙어 있다면 무엇을 뜻하나?", ["속도가 2배 느림", "보안 인증서가 적용되어 통신 내용이 암호화되고 있음", "광고가 많음", "무료 사이트임", "로그인이 안 됨"], "보안 인증서가 적용되어 통신 내용이 암호화되고 있음", "사용자의 개인정보를 안전하게 보호하는 신뢰할 수 있는 사이트임을 의미합니다. 브라우저는 HTTP 사이트에 '주의 요함' 경고를 표시하므로 HTTPS는 SEO와 신뢰 모두에 필수입니다.", "HTTPS 의미", "6063", "medium"),
    ("프런트엔드에서 사용자가 메시지를 보낸 직후 채팅창을 맨 아래로 내리는 이유는?", ["화면을 숨기려고", "새로 생성된 답변을 사용자가 바로 확인할 수 있게 시야를 맞춰줌", "로드 밸런서를 교체하려고", "API 게이트웨이를 재설정하려고", "캐시 서버를 초기화하려고"], "새로 생성된 답변을 사용자가 바로 확인할 수 있게 시야를 맞춰줌", "자연스러운 대화 흐름(UX)을 유지하기 위한 필수적인 스크롤 처리입니다. scrollRef.current.scrollIntoView({ behavior: 'smooth' })로 매끄러운 스크롤을 구현합니다.", "Auto Scroll", "6064", "medium"),
    ("백엔드에서 'JWT(JSON Web Token)'를 사용하는 주된 목적은?", ["게임 머니", "사용자의 신원을 증명하고 권한을 안전하게 확인하는 인증 수단", "글씨체 바꾸기", "인터넷 가입", "사진 저장"], "사용자의 신원을 증명하고 권한을 안전하게 확인하는 인증 수단", "현대적인 웹 서비스에서 로그인 상태를 유지하는 표준적인 기술 중 하나입니다. 헤더에 서명이 포함되어 있어 서버에 세션을 저장하지 않아도 위변조 여부를 검증할 수 있습니다.", "JWT", "6065", "medium"),
    ("배포 시 서버의 'CPU 사용률이 100%'라면 취해야 할 조치는?", ["서버 수 늘리기(Scale out) 또는 성능 높이기(Scale up)", "컴퓨터 끄기", "캐시 서버 제거", "데이터베이스 인덱스 삭제", "API 게이트웨이 비활성화"], "서버 수 늘리기(Scale out) 또는 성능 높이기(Scale up)", "인프라 확장을 통해 서비스 중단 없이 문제를 해결하는 올바른 대응입니다. 단기적으로는 Auto Scaling으로 서버를 추가하고, 장기적으로는 병목 코드를 프로파일링하여 개선해야 합니다.", "리소스 관리", "6066", "medium"),
    ("프런트엔드 배포 플랫폼인 'Vercel'이나 'Netlify'의 장점은?", ["서버를 직접 조립해야 함", "Git과 연동되어 코드만 올리면 자동으로 빌드하고 인터넷에 배포해줌", "가격이 무조건 비쌈", "수동으로만 작동함", "오프라인 전용임"], "Git과 연동되어 코드만 올리면 자동으로 빌드하고 인터넷에 배포해줌", "개인 프로젝트나 프로토타입을 순식간에 서비스화할 수 있는 강력한 도구입니다. 브랜치별 Preview URL을 자동 생성해주어 PR 리뷰 시 실제 환경을 즉시 확인할 수 있습니다.", "Frontend Platform", "6067", "medium"),
    ("백엔드 서버에서 에러가 났을 때 클라이언트에게 알려주는 '500' 코드는?", ["정상 작동", "서버 내부 오류 (Internal Server Error)", "찾을 수 없음", "전원 꺼짐", "로그인 성공"], "서버 내부 오류 (Internal Server Error)", "서버 쪽 로직에 문제가 생겼음을 알려 디버깅의 시작점을 파악하게 해줍니다. 사용자에게는 친절한 메시지를, 개발자에게는 상세 스택 트레이스를 로그로 남기는 분리 처리가 중요합니다.", "500 에러", "6068", "hard"),
    ("배포 후 '검색 엔진 최적화(SEO)'를 하는 이유는?", ["속도를 높이려고", "네이버나 구글 검색 시 내 서비스가 더 잘 노출되게 하기 위해", "글자를 숨기려고", "광고를 보려고", "컴퓨터를 끄려고"], "네이버나 구글 검색 시 내 서비스가 더 잘 노출되게 하기 위해", "더 많은 잠재 사용자가 서비스를 발견하게 만드는 마케팅적 기술입니다. Next.js의 metadata API로 Open Graph 태그를 설정하면 SNS 공유 시에도 미리보기가 잘 표시됩니다.", "SEO", "6069", "medium"),
    ("프런트엔드에서 'Favicon'(파비콘)이란?", ["인공지능 이름", "웹 브라우저 탭에 표시되는 작은 아이콘 로고", "배경 음악", "글꼴 이름", "사이트 하단 문구"], "웹 브라우저 탭에 표시되는 작은 아이콘 로고", "서비스의 아이덴티티를 시각적으로 완성해주는 작은 디테일입니다. 32x32 또는 64x64 픽셀의 .ico/.png 파일을 <link rel='icon'>으로 HTML에 등록합니다.", "Favicon", "6070", "medium"),
    ("백엔드 서버 개발 시 'API Endpoint'란?", ["컴퓨터 전원 종료", "클라이언트가 특정 기능(채팅, 로그인 등)을 호출하기 위해 접속하는 주소 경로", "파일의 마지막 줄", "캐시 서버 주소", "메시지 큐 이름"], "클라이언트가 특정 기능(채팅, 로그인 등)을 호출하기 위해 접속하는 주소 경로", "예: /api/v1/chat 과 같이 서비스가 제공하는 기능들의 주소를 의미합니다. RESTful 설계 원칙에 따라 리소스 중심의 URL 구조를 설계하는 것이 권장됩니다.", "Endpoint", "6071", "easy"),
    ("클라우드 인프라 배포 시 'S3'(Simple Storage Service)의 용도는?", ["인터넷 채팅", "이미지, PDF, 로그 파일 등 비정형 데이터를 안전하게 저장하는 창고", "글 작성", "코딩 도구", "게임 서버"], "이미지, PDF, 로그 파일 등 비정형 데이터를 안전하게 저장하는 창고", "용량 제한 없이 파일을 무한히 저장하고 불러올 수 있는 클라우드 저장소입니다. Presigned URL을 사용하면 클라이언트가 서버를 거치지 않고 직접 S3에 파일을 업로드할 수 있습니다.", "S3 Storage", "6072", "medium"),
    ("프런트엔드에서 '애니메이션 효과'를 넣는 가장 큰 이유는?", ["전력을 소모하려고", "상태 변화를 부드럽게 보여줘서 사용자에게 즐거움과 정보 전달력을 높임", "캐시 서버 부하를 줄이려고", "API 게이트웨이 속도를 높이려고", "로드 밸런서를 재시작하려고"], "상태 변화를 부드럽게 보여줘서 사용자에게 즐거움과 정보 전달력을 높임", "생성 AI 서비스의 생동감을 불어넣는 UX 요소입니다. Framer Motion, CSS transition 등을 사용하며, 과도한 애니메이션은 성능에 영향을 줄 수 있으므로 적절히 사용해야 합니다.", "애니메이션", "6073", "medium"),
    ("백엔드 개발 시 'ORM' 라이브러리를 사용하는 이유는?", ["코드를 어렵게 하려고", "SQL 쿼리 대신 익숙한 프로그래밍 언어로 DB를 쉽고 안전하게 다루기 위해", "속도를 억지로 늦추려고", "API 게이트웨이를 제거하려고", "캐시 서버를 대체하려고"], "SQL 쿼리 대신 익숙한 프로그래밍 언어로 DB를 쉽고 안전하게 다루기 위해", "개발 생산성을 높이고 데이터베이스 접근 코드를 깔끔하게 관리하게 해줍니다. SQLAlchemy, Tortoise ORM, Prisma 등이 Python/FastAPI 생태계에서 자주 사용됩니다.", "ORM", "6074", "hard"),
    ("배포된 서비스의 'Uptime'(업타임)이란?", ["서버가 켜진 이후 현재까지 정상적으로 가동된 시간", "사용자가 잠자는 시간", "컴퓨터 사는 시간", "공부하는 시간", "비용 결제 시간"], "서버가 켜진 이후 현재까지 정상적으로 가동된 시간", "서비스의 신뢰도와 안정성을 나타내는 직접적인 지표입니다. 99.9% Uptime이면 연간 약 8.7시간의 장애를 의미하며, SLA(서비스 수준 계약)에서 보장 Uptime을 명시합니다.", "Uptime", "6075", "medium"),
    ("프런트엔드에서 'Local Storage'에 대화 내역을 저장할 때의 특징은?", ["서버에 저장됨", "사용자의 브라우저에 데이터가 저장되어 재방문 시에도 내용이 유지됨", "해킹이 불가능함", "용량이 무제한임", "영구히 삭제됨"], "사용자의 브라우저에 데이터가 저장되어 재방문 시에도 내용이 유지됨", "서버 DB 없이도 간단한 히스토리 기능을 구현할 수 있는 방법입니다. 최대 5~10MB 용량 제한이 있고, 민감한 정보는 저장하지 않는 것이 보안상 권장됩니다.", "Local Storage", "6076", "medium"),
    ("API 서버 구축 시 'HTTP 상태 코드 404'의 의미는?", ["정상", "요청한 주소(리소스)를 찾을 수 없음 (Not Found)", "권한 없음", "잘못된 요청", "서버 과부하"], "요청한 주소(리소스)를 찾을 수 없음 (Not Found)", "주소를 틀렸거나 삭제된 페이지에 접속했을 때 나타나는 표준 응답입니다. FastAPI에서는 raise HTTPException(status_code=404, detail='Not found')로 명시적으로 반환합니다.", "404 에러", "6077", "medium"),
    ("클라우드 배포 시 '로드 밸런서(Load Balancer)'의 역할은?", ["무게 재기", "여러 대의 서버에 트래픽을 골고루 나누어 서버 부하를 분산함", "돈 계산", "사진 편집", "영화 감상"], "여러 대의 서버에 트래픽을 골고루 나누어 서버 부하를 분산함", "단일 서버에 가중되는 부담을 줄여 대규모 사용자를 수용하게 돕습니다. Round Robin, Least Connections, IP Hash 등 다양한 알고리즘으로 트래픽을 분산합니다.", "로드 밸런싱", "6078", "medium"),
    ("백엔드에서 사용되는 '환경 변수' 중 PORT 번호를 바꾸는 목적은?", ["서버가 접속을 기다릴 가상의 통로 번호를 지정하기 위해", "글자 수 늘리기", "비번 바꾸기", "컴퓨터 이름 바꾸기", "인터넷 속도"], "서버가 접속을 기다릴 가상의 통로 번호를 지정하기 위해", "서버가 통신할 창구를 정하는 기본 설정입니다. 포트 충돌 방지, 방화벽 규칙 적용, 리버스 프록시(Nginx) 연동 시 포트 번호가 중요한 역할을 합니다.", "Port Number", "6079", "medium"),
    ("최종적으로 서비스를 '런칭'한 후 가장 중요하게 챙겨야 할 것은?", ["개발 중단 및 휴식", "사용자 피드백 수집과 지속적인 모니터링 및 업데이트", "사이트 삭제", "비밀번호 노출", "로그아웃"], "사용자 피드백 수집과 지속적인 모니터링 및 업데이트", "출시는 시작일 뿐, 사용자의 반응에 맞춰 진화하는 것이 진정한 서비스의 완성입니다. 에러 로그, 사용자 행동 분석, NPS 조사 등을 통해 지속적으로 개선점을 발굴합니다.", "런칭 후 관리", "6080", "medium"),

    # 남은 20문제
    ("프런트엔드 앱의 '로딩 바'가 멈춰 있다면 의심되는 원인은?", ["모델이 너무 똑똑해서", "네트워크 에러나 백엔드 서버의 무한 루프 등 비정상 종료", "캐시 서버 과부하", "API 게이트웨이 재시작", "로드 밸런서 교체"], "네트워크 에러나 백엔드 서버의 무한 루프 등 비정상 종료", "사용자에게 시스템의 장애 상태를 인지시키는 디버깅 신호입니다. setTimeout으로 일정 시간 후 에러 메시지를 표시하고 재시도 버튼을 제공하는 것이 좋은 UX입니다.", "로딩 멈춤", "6081", "medium"),
    ("배포 완료 후 '구글 분석기(Google Analytics)'를 심는 주된 이유는?", ["사용자의 비번 확인", "얼마나 많은 사람이 들어오고 어떤 기능을 자주 쓰는지 통계를 보기 위해", "그림 그리기", "음악 듣기", "게임 하기"], "얼마나 많은 사람이 들어오고 어떤 기능을 자주 쓰는지 통계를 보기 위해", "데이터에 기반해 서비스를 개선하기 위한 분석 도구입니다. 이탈률, 세션 시간, 전환율 같은 지표를 통해 어떤 기능이 사용자에게 가치 있는지 파악합니다.", "GA 심기", "6082", "medium"),
    ("백엔드 서버 배포 시 'Secrets Management'(비밀 정보 관리)가 중요한 이유는?", ["코드가 예뻐서", "DB 비번이나 API Key 유출로 인한 금전적/데이터 손실을 막기 위해", "이름 짓기", "사진첩", "글자 수"], "DB 비번이나 API Key 유출로 인한 금전적/데이터 손실을 막기 위해", "개발 보안의 가장 기초이자 필수적인 항목입니다. AWS Secrets Manager, HashiCorp Vault 같은 전용 시크릿 관리 서비스를 사용하면 키 로테이션까지 자동화할 수 있습니다.", "Secret 관리", "6083", "easy"),
    ("프런트엔드에서 '모바일 브라우저' 상단 바 색깔을 지정하는 이유는?", ["사이트의 브랜드 컬러를 UI와 통일시켜 사용자에게 몰입감을 주기 위해", "배터리 절약", "인터넷 속도", "오타 방지", "화면 보호"], "사이트의 브랜드 컬러를 UI와 통일시켜 사용자에게 몰입감을 주기 위해", "모바일 웹 UX의 완성도를 높여주는 디자인 디테일입니다. <meta name='theme-color' content='#색상코드'>로 Android Chrome 주소창 색상을 커스터마이징할 수 있습니다.", "모바일 UI 테마", "6084", "medium"),
    ("배포 과정 중 '스테이징(Staging)' 환경이란?", ["실제 배포 전, 실제 서비스와 똑같은 환경에서 마지막으로 테스트하는 연습 서버", "공연 무대", "잠자는 곳", "밥 먹는 곳", "공부하는 곳"], "실제 배포 전, 실제 서비스와 똑같은 환경에서 마지막으로 테스트하는 연습 서버", "실제 사용자에게 장애를 노출하지 않기 위한 최종 리허설 공간입니다. development → staging → production 순서로 코드가 이동하며 각 단계마다 검증을 거칩니다.", "Staging", "6085", "medium"),
    ("백엔드 서버에서 AI 모델의 '온도(Temperature)'를 설정값으로 받는 이유는?", ["방 안이 너무 더워서", "답변의 무작위성을 제어하여 상황에 맞는 응답을 얻기 위해", "서버가 뜨거워서", "전기세가 많이 나와서", "컴퓨터가 고장 나서"], "답변의 무작위성을 제어하여 상황에 맞는 응답을 얻기 위해", "서버 로직에서 AI의 창의성 정도를 결정하는 핵심 파라미터입니다. 0에 가까울수록 결정론적 응답, 1 이상이면 창의적이지만 예측 불가능한 응답이 나옵니다.", "Temperature 설정", "6086", "medium"),
    ("프런트엔드에서 '글자 수 한도'를 표시해주는 UI의 효과는?", ["글자를 못 쓰게 함", "사용자가 AI에게 보낼 메시지 양을 인지하게 하여 토큰 낭비를 예방함", "로그인", "로그아웃", "광고 띄우기"], "사용자가 AI에게 보낼 메시지 양을 인지하게 하여 토큰 낭비를 예방함", "사용자에게 제약 사항을 명확히 알려 시스템 오류를 미연에 방지합니다. textarea의 maxLength 속성과 현재 글자 수를 실시간으로 표시하는 카운터 UI가 일반적입니다.", "글자 수 표시", "6087", "medium"),
    ("백엔드 서버의 '타임아웃(Timeout)' 시간을 너무 짧게 잡았을 때의 문제는?", ["답변이 너무 빨리 나옴", "AI가 답변을 생성하는 도중에 연결이 강제로 끊겨 응답이 누락됨", "캐시 서버가 꺼짐", "API 게이트웨이가 재시작됨", "로드 밸런서가 멈춤"], "AI가 답변을 생성하는 도중에 연결이 강제로 끊겨 응답이 누락됨", "생성 시간이 필요한 LLM의 특성을 고려해 적절한 대기 시간을 유지해야 합니다. 스트리밍 응답 사용 시 keep-alive 연결 유지로 Timeout 문제를 회피할 수 있습니다.", "Timeout 문제", "6088", "medium"),
    ("배포된 인프라의 '로그 분석'을 통해 해커의 공격 시도를 발견하는 법은?", ["그림 보기", "특정 IP에서 비정상적으로 많은 요청이 들어오는지 로그 패턴을 확인함", "노래 듣기", "게임 하기", "잠자기"], "특정 IP에서 비정상적으로 많은 요청이 들어오는지 로그 패턴을 확인함", "모니터링과 보안은 서비스 운영의 두 기둥입니다. CloudWatch, ELK Stack 같은 로그 집계 도구로 비정상 패턴을 자동으로 탐지하고 알림을 받을 수 있습니다.", "로그 분석 보안", "6089", "medium"),
    ("최고의 AI 서비스를 만드는 마지막 비결은 무엇인가?", ["코드를 한 번 짜고 끝내는 것", "지속적으로 사용자와 소통하며 AI 성능과 UI/UX를 개선해 나가는 것", "컴퓨터를 끄는 것", "비밀번호를 공개하는 것", "인터넷 해지"], "지속적으로 사용자와 소통하며 AI 성능과 UI/UX를 개선해 나가는 것", "완성된 코드는 없으며, 살아 움직이며 진화하는 서비스가 최고의 서비스입니다. A/B 테스트, 사용자 인터뷰, 피드백 루프를 통해 데이터 기반의 의사결정을 이어가야 합니다.", "진정한 완성", "6090", "medium"),
    ("프런트엔드 성능 최적화 기법 중 '이미지 지연 로딩(Lazy Loading)'의 효과는?", ["배터리 폭발", "사용자가 보고 있는 화면의 이미지만 먼저 불러와 초기 로딩 속도를 높임", "사진 삭제", "글자 지우기", "로그아웃"], "사용자가 보고 있는 화면의 이미지만 먼저 불러와 초기 로딩 속도를 높임", "불필요한 네트워크 자원 소모를 줄여 쾌적한 웹 경험을 제공합니다. HTML의 loading='lazy' 속성이나 Intersection Observer API로 구현하며, Next.js의 Image 컴포넌트는 기본 적용됩니다.", "Lazy Loading", "6091", "medium"),
    ("백엔드에서 'SQL Injection' 공격을 막기 위한 가장 좋은 방법은?", ["비밀번호 없애기", "사용자 입력을 쿼리에 직접 넣지 않고 Parameterized Query(준비된 문구)를 사용하기", "API 게이트웨이 교체", "캐시 서버 초기화", "로드 밸런서 재구성"], "사용자 입력을 쿼리에 직접 넣지 않고 Parameterized Query(준비된 문구)를 사용하기", "DB 보안의 기본 중의 기본으로, 악의적인 쿼리 실행을 원천 차단합니다. ORM을 사용하면 자동으로 Parameterized Query가 적용되어 SQL Injection 공격에 강해집니다.", "SQL Injection 방어", "6092", "medium"),
    ("배포 후 'Google Search Console'을 사용하는 목적은?", ["게임 하기", "구글 검색 결과에서 내 사이트의 노출 현황을 확인하고 문제점을 고치기 위해", "사진 편집", "음악 듣기", "잠자기"], "구글 검색 결과에서 내 사이트의 노출 현황을 확인하고 문제점을 고치기 위해", "검색 유입을 늘리고 웹사이트의 건강 상태를 체크하는 전문 도구입니다. 크롤링 에러, 색인 생성 여부, Core Web Vitals 점수를 무료로 확인할 수 있습니다.", "Search Console", "6093", "medium"),
    ("프런트엔드에서 'Web Accessibility'(웹 접근성)를 준수하는 이유는?", ["법을 어기려고", "장애인이나 고령자 등 모든 사용자가 차별 없이 서비스를 이용할 수 있도록 하기 위해", "사진 숨기기", "글자 작게 하기", "로그인 금지"], "장애인이나 고령자 등 모든 사용자가 차별 없이 서비스를 이용할 수 있도록 하기 위해", "사회적 책임과 동시에 더 많은 사용자층을 확보하는 포용적인 설계입니다. ARIA 레이블, 충분한 색상 대비, 키보드 네비게이션 지원이 핵심 요소입니다.", "웹 접근성", "6094", "medium"),
    ("백엔드 서버의 'Load Average'(부하 평균) 지표를 확인하는 이유는?", ["날짜 확인", "시스템이 현재 처리해야 할 작업이 얼마나 쌓여 있는지 부하 정도를 파악하기 위해", "이름 짓기", "사진첩", "인터넷 속도"], "시스템이 현재 처리해야 할 작업이 얼마나 쌓여 있는지 부하 정도를 파악하기 위해", "서버가 과부하 상태인지 판단하여 인프라 증설 여부를 결정하는 척도가 됩니다. CPU 코어 수보다 Load Average가 높으면 서버 증설 또는 코드 최적화가 필요합니다.", "Load Average", "6095", "medium"),
    ("배포 시 'Rollback'(롤백)이란?", ["앞으로 가기", "배포 후 심각한 버그 발견 시 즉시 이전의 정상적인 상태로 되돌리는 것", "컴퓨터 끄기", "캐시 서버 재구성", "데이터베이스 인덱스 삭제"], "배포 후 심각한 버그 발견 시 즉시 이전의 정상적인 상태로 되돌리는 것", "서비스의 가용성을 지키기 위한 최후의 방어 수단입니다. Kubernetes의 kubectl rollout undo나 Git 태그 기반 이전 버전 재배포로 빠르게 복구합니다.", "Rollback", "6096", "medium"),
    ("프런트엔드에서 'SEO'를 위해 설정하는 <meta> 태그의 역할은?", ["배경 음악 재생", "사이트의 제목, 설명, 키워드를 검색 엔진에 알려주는 명함 역할", "비밀번호 저장", "파일 다운로드", "로그아웃"], "사이트의 제목, 설명, 키워드를 검색 엔진에 알려주는 명함 역할", "검색 결과 미리보기에 나타나는 텍스트를 결정하여 클릭률을 높여줍니다. og:title, og:description, og:image 같은 Open Graph 태그는 SNS 공유 시 미리보기에 영향을 미칩니다.", "Meta Tag", "6097", "hard"),
    ("백엔드 서버에서 'API 키'를 탈취당했을 때 가장 먼저 해야 할 일은?", ["사이트 폐쇄", "기존 키를 즉시 무효화(Revoke)하고 새로운 키를 발급받아 교체하기", "메시지 큐 중단", "캐시 서버 초기화", "인터넷 해지"], "기존 키를 즉시 무효화(Revoke)하고 새로운 키를 발급받아 교체하기", "추가적인 비용 발생이나 정보 유출을 막기 위한 긴급 보안 조치입니다. GitGuardian 같은 도구로 Git 커밋에 실수로 올라간 API 키를 자동 감지하고 알림을 받을 수 있습니다.", "Key Revocation", "6098", "medium"),
    ("서비스 배포 후 '사용자 행동 분석' 도구를 활용하는 목적은?", ["개인 정보 도난", "어느 버튼을 많이 누르고 어디서 이탈하는지 파악하여 UX를 개선하기 위해", "사진 감상", "음악 듣기", "게임 하기"], "어느 버튼을 많이 누르고 어디서 이탈하는지 파악하여 UX를 개선하기 위해", "사용자의 불편함을 데이터로 읽어내어 더 사랑받는 서비스를 만드는 과정입니다. Hotjar의 히트맵, Mixpanel의 퍼널 분석, FullStory의 세션 레코딩이 대표적인 도구입니다.", "행동 분석", "6099", "hard"),
    ("성공적인 LLM 서비스 배포를 위한 마인드셋은?", ["한 번 만들어두면 평생 갈 것", "기술은 계속 변하므로 끊임없이 학습하고 서비스를 고도화하려는 태도", "컴퓨터 끄기", "비밀번호 노출", "인터넷 해지"], "기술은 계속 변하므로 끊임없이 학습하고 서비스를 고도화하려는 태도", "빠르게 변하는 AI 시대에 발맞춰 성장하는 개발자의 기본 소양입니다. 새로운 모델 출시, API 변경, 보안 패치를 지속적으로 추적하고 적용하는 습관이 중요합니다.", "Growth Mindset", "6100", "medium")
]

for q, o, a, w, h, i, d in mcq_data:
    questions.append({"chapter_name": chapter_name, "type": "객관식", "difficulty": d, "id": i, "question": q, "options": o, "answer": a, "why": w, "hint": h})

# --- 20 Code Completion Questions ---
cc_data = [
    ("FastAPI POST 엔드포인트",
     "from fastapi import FastAPI\nfrom pydantic import BaseModel\n\napp = FastAPI()\n\nclass ChatRequest(BaseModel):\n    message: str\n    model: str = 'gpt-4o-mini'\n\n@app._____(('/chat')\nasync def chat(request: ChatRequest):\n    return {'response': f'Echo: {request.message}', 'model': request.model}",
     "post",
     "FastAPI에서 @app.post()는 HTTP POST 요청을 처리하는 라우트를 정의합니다. Pydantic BaseModel로 요청 바디를 타입 안전하게 파싱합니다. POST는 데이터를 서버로 전송할 때 사용하며, LLM 채팅 API에서 표준입니다."),

    ("환경변수 로드 with dotenv",
     "from dotenv import load_dotenv\nimport os\n\n_____('.env')\n\napi_key = os.getenv('OPENAI_API_KEY')\ndb_url = os.getenv('DATABASE_URL', 'sqlite:///default.db')\n\nif not api_key:\n    raise ValueError('OPENAI_API_KEY 환경변수가 설정되지 않았습니다')\n\nprint(f'API Key 앞 8자: {api_key[:8]}...')",
     "load_dotenv",
     "load_dotenv()는 .env 파일의 키-값을 환경 변수로 로드합니다. os.getenv()의 두 번째 인자는 기본값입니다. API 키를 코드에 직접 쓰지 않고 .env 파일로 관리하면 Git에 민감 정보가 올라가는 사고를 방지합니다."),

    ("FastAPI 스트리밍 응답",
     "from fastapi import FastAPI\nfrom fastapi.responses import StreamingResponse\nfrom openai import OpenAI\n\napp = FastAPI()\nclient = OpenAI()\n\nasync def generate_stream(message: str):\n    stream = client.chat.completions.create(\n        model='gpt-4o-mini',\n        messages=[{'role': 'user', 'content': message}],\n        stream=True\n    )\n    for chunk in stream:\n        if chunk.choices[0].delta.content:\n            yield chunk.choices[0].delta.content\n\n@app.get('/stream')\nasync def stream_chat(message: str):\n    return _____(generate_stream(message), media_type='text/plain')",
     "StreamingResponse",
     "StreamingResponse는 서버에서 데이터를 청크 단위로 실시간 전송합니다. 제너레이터 함수를 인자로 전달하면 yield된 값이 순서대로 클라이언트에 전달됩니다. LLM 응답의 타이핑 효과를 구현하는 표준 방식입니다."),

    ("FastAPI CORS 설정",
     "from fastapi import FastAPI\nfrom fastapi.middleware.cors import CORSMiddleware\n\napp = FastAPI()\n\napp.add_middleware(\n    _____,\n    allow_origins=['http://localhost:3000', 'https://myapp.com'],\n    allow_credentials=True,\n    allow_methods=['*'],\n    allow_headers=['*']\n)\n\n@app.get('/')\ndef root():\n    return {'message': 'CORS 설정 완료'}",
     "CORSMiddleware",
     "CORS(Cross-Origin Resource Sharing)는 다른 도메인에서의 API 접근을 허용/제한합니다. 프런트엔드(localhost:3000)와 백엔드(다른 포트)가 분리된 경우 CORS 설정이 필수입니다. allow_origins에 허용할 도메인을 명시합니다."),

    ("비동기 FastAPI with httpx",
     "from fastapi import FastAPI\nimport httpx\n\napp = FastAPI()\n\n@app.get('/weather/{city}')\nasync def get_weather(city: str):\n    async with _____.AsyncClient() as client:\n        response = await client.get(\n            f'https://api.weatherapi.com/v1/current.json',\n            params={'key': 'API_KEY', 'q': city}\n        )\n        return response.json()",
     "httpx",
     "httpx.AsyncClient()는 비동기 HTTP 클라이언트입니다. async with 컨텍스트 매니저로 사용하면 요청 완료 후 연결이 자동 해제됩니다. FastAPI의 async def 엔드포인트에서 외부 API를 호출할 때 httpx가 requests보다 권장됩니다."),

    ("FastAPI 의존성 주입 - API 키 인증",
     "from fastapi import FastAPI, Depends, HTTPException, Header\nfrom typing import Optional\n\napp = FastAPI()\nVALID_KEY = 'secret-api-key'\n\nasync def verify_api_key(x_api_key: Optional[str] = Header(None)):\n    if x_api_key != VALID_KEY:\n        raise _____(status_code=401, detail='Invalid API Key')\n    return x_api_key\n\n@app.post('/chat')\nasync def protected_chat(message: str, key: str = Depends(verify_api_key)):\n    return {'message': f'인증됨: {message}'}",
     "HTTPException",
     "HTTPException은 FastAPI에서 HTTP 오류 응답을 반환합니다. status_code=401은 Unauthorized(인증 실패)입니다. Depends()는 의존성 주입으로, verify_api_key 함수가 엔드포인트 실행 전 자동 호출됩니다."),

    ("Pydantic 필드 검증",
     "from fastapi import FastAPI\nfrom pydantic import BaseModel, Field\n\napp = FastAPI()\n\nclass LLMRequest(BaseModel):\n    prompt: str = Field(..., min_length=1, max_length=4000)\n    temperature: float = Field(default=0.7, ge=0.0, _____=2.0)\n    max_tokens: int = Field(default=1024, gt=0, le=4096)\n\n@app.post('/generate')\nasync def generate(req: LLMRequest):\n    return {'prompt_length': len(req.prompt), 'settings': req.model_dump()}",
     "le",
     "Field()의 ge(>=), le(<=), gt(>), lt(<)로 숫자 범위를 제한합니다. le=2.0은 temperature <= 2.0을 의미합니다. FastAPI는 자동으로 Pydantic 검증을 수행하고 실패 시 422 Unprocessable Entity 에러를 반환합니다."),

    ("FastAPI 백그라운드 태스크",
     "from fastapi import FastAPI, BackgroundTasks\nimport time\n\napp = FastAPI()\n\ndef send_notification(email: str, message: str):\n    time.sleep(2)\n    print(f'{email}에게 알림: {message}')\n\n@app.post('/chat')\nasync def chat(message: str, email: str, background_tasks: _____):\n    background_tasks.add_task(send_notification, email, '새 메시지 도착')\n    return {'response': f'처리됨: {message}'}",
     "BackgroundTasks",
     "BackgroundTasks는 응답을 먼저 반환하고 느린 작업(이메일, 로깅 등)을 백그라운드에서 실행합니다. add_task(함수, *인자)로 작업을 등록합니다. 사용자 대기 시간을 줄이는 실무 패턴입니다."),

    ("FastAPI 전역 예외 핸들러",
     "from fastapi import FastAPI, Request\nfrom fastapi.responses import JSONResponse\nimport openai\n\napp = FastAPI()\n\n@app.exception_handler(openai.RateLimitError)\nasync def rate_limit_handler(request: Request, exc: openai.RateLimitError):\n    return _____(\n        status_code=429,\n        content={'error': 'API 요청 한도 초과', 'retry_after': 60}\n    )\n\n@app.get('/ask')\nasync def ask(q: str):\n    pass",
     "JSONResponse",
     "exception_handler()는 특정 예외 타입을 전역으로 처리합니다. JSONResponse()로 구조화된 오류 응답을 반환합니다. status_code=429는 Too Many Requests입니다. 전역 예외 처리기로 모든 엔드포인트에 일관된 오류 응답을 적용합니다."),

    ("비동기 병렬 OpenAI 호출",
     "from openai import AsyncOpenAI\nimport asyncio\n\nclient = AsyncOpenAI()\n\nasync def get_response(question: str) -> str:\n    response = await client.chat.completions.create(\n        model='gpt-4o-mini',\n        messages=[{'role': 'user', 'content': question}]\n    )\n    return response.choices[0].message.content\n\nasync def main():\n    questions = ['Python이란?', 'FastAPI란?', 'Docker란?']\n    tasks = [get_response(q) for q in questions]\n    results = await asyncio._____(*tasks)\n    for q, r in zip(questions, results):\n        print(f'Q: {q}\\nA: {r[:50]}')\n\nasyncio.run(main())",
     "gather",
     "asyncio.gather()는 여러 코루틴을 동시에 실행합니다. AsyncOpenAI()는 비동기 클라이언트로 await와 함께 사용합니다. 3개 요청을 순차 처리하면 3배 시간이 걸리지만 gather()로 병렬 처리하면 1배 시간에 완료됩니다."),

    ("FastAPI 라우터 모듈화",
     "from fastapi import APIRouter\nfrom openai import OpenAI\n\nrouter = _____(prefix='/api/v1', tags=['chat'])\nclient = OpenAI()\n\n@router.post('/chat')\nasync def chat(message: str):\n    resp = client.chat.completions.create(\n        model='gpt-4o-mini',\n        messages=[{'role': 'user', 'content': message}]\n    )\n    return {'response': resp.choices[0].message.content}\n\n# main.py에서: app.include_router(router)",
     "APIRouter",
     "APIRouter로 라우트를 모듈별로 분리합니다. prefix='/api/v1'로 공통 경로를 지정하고, tags는 Swagger 문서 그룹화에 사용됩니다. app.include_router(router)로 메인 앱에 등록합니다. 대규모 프로젝트 구조화의 핵심입니다."),

    ("Redis 캐시로 응답 최적화",
     "import redis\nimport json\nimport hashlib\nfrom openai import OpenAI\n\nr = redis.Redis(host='localhost', port=6379)\nclient = OpenAI()\n\ndef cached_llm_call(prompt: str, ttl: int = 3600) -> str:\n    cache_key = hashlib.md5(prompt.encode()).hexdigest()\n    \n    cached = r._____(cache_key)\n    if cached:\n        return json.loads(cached)\n    \n    response = client.chat.completions.create(\n        model='gpt-4o-mini',\n        messages=[{'role': 'user', 'content': prompt}]\n    )\n    result = response.choices[0].message.content\n    r.setex(cache_key, ttl, json.dumps(result))\n    return result",
     "get",
     "redis.get()은 키가 있으면 값을 반환하고 없으면 None을 반환합니다. MD5 해시를 캐시 키로 사용하면 동일한 프롬프트를 식별합니다. setex(키, TTL초, 값)로 만료 시간을 설정합니다. LLM API 비용을 획기적으로 절감할 수 있습니다."),

    ("로깅 미들웨어",
     "from fastapi import FastAPI, Request\nimport time\nimport logging\n\nlogging.basicConfig(level=logging.INFO)\nlogger = logging.getLogger(__name__)\n\napp = FastAPI()\n\n@app.middleware('http')\nasync def log_requests(request: Request, call_next):\n    start = time.time()\n    response = await _____(request)\n    duration = time.time() - start\n    logger.info(\n        f'{request.method} {request.url.path} '\n        f'{response.status_code} {duration:.3f}s'\n    )\n    return response",
     "call_next",
     "@app.middleware('http')는 모든 요청에 적용되는 미들웨어를 정의합니다. call_next(request)로 실제 라우트 핸들러를 호출합니다. 응답 전후에 처리 시간, 상태 코드, 경로를 로깅하면 성능 모니터링에 유용합니다."),

    ("JWT 토큰 생성 및 검증",
     "import jwt\nfrom datetime import datetime, timedelta\n\nSECRET = 'my-secret-key'\n\ndef create_token(user_id: str) -> str:\n    payload = {\n        'sub': user_id,\n        'exp': datetime.utcnow() + timedelta(hours=24)\n    }\n    return jwt._____(payload, SECRET, algorithm='HS256')\n\ndef verify_token(token: str) -> dict:\n    try:\n        return jwt.decode(token, SECRET, algorithms=['HS256'])\n    except jwt.ExpiredSignatureError:\n        raise ValueError('토큰이 만료되었습니다')\n\ntoken = create_token('user123')\nprint(verify_token(token))",
     "encode",
     "jwt.encode()는 페이로드와 시크릿 키로 JWT 토큰을 생성합니다. exp 클레임으로 만료 시간을 설정합니다. jwt.decode()로 서명을 검증하고 페이로드를 추출합니다. stateless 인증에서 사용자 세션을 서버에 저장하지 않아도 됩니다."),

    ("FastAPI Lifespan 초기화",
     "from fastapi import FastAPI\nfrom contextlib import asynccontextmanager\nfrom openai import OpenAI\n\nclient = None\n\n@asynccontextmanager\nasync def lifespan(app: FastAPI):\n    global client\n    print('서버 시작 - 초기화')\n    client = _____()\n    yield\n    print('서버 종료')\n    client = None\n\napp = FastAPI(lifespan=lifespan)\n\n@app.post('/chat')\nasync def chat(message: str):\n    resp = client.chat.completions.create(\n        model='gpt-4o-mini',\n        messages=[{'role': 'user', 'content': message}]\n    )\n    return {'response': resp.choices[0].message.content}",
     "OpenAI",
     "lifespan 컨텍스트 매니저는 서버 시작(yield 전)과 종료(yield 후) 시 코드를 실행합니다. DB 연결, API 클라이언트 초기화 등 무거운 리소스를 한 번만 생성하고 재사용합니다. FastAPI 0.95+ 권장 방식입니다."),

    ("환경별 설정 관리",
     "from pydantic_settings import BaseSettings\nfrom functools import lru_cache\n\nclass Settings(BaseSettings):\n    openai_api_key: str\n    database_url: str = 'sqlite:///./app.db'\n    debug: bool = False\n    max_tokens: int = 1024\n    \n    class Config:\n        env_file = '.env'\n\n@_____\ndef get_settings() -> Settings:\n    return Settings()\n\nsettings = get_settings()\nprint(f'Debug 모드: {settings.debug}')\nprint(f'Max Tokens: {settings.max_tokens}')",
     "lru_cache",
     "@lru_cache는 함수 결과를 캐싱하여 Settings()를 한 번만 생성합니다. pydantic_settings.BaseSettings는 .env 파일과 환경 변수를 자동으로 읽어 타입 변환합니다. FastAPI의 Depends(get_settings)와 함께 사용하면 설정 의존성 주입이 가능합니다."),

    ("FastAPI WebSocket 채팅",
     "from fastapi import FastAPI, WebSocket\nimport json\n\napp = FastAPI()\n\n@app.websocket('/ws/chat')\nasync def websocket_chat(websocket: _____):\n    await websocket.accept()\n    try:\n        while True:\n            data = await websocket.receive_text()\n            message = json.loads(data)\n            await websocket.send_text(json.dumps({'response': f\"Echo: {message['text']}\"}))\n    except Exception:\n        await websocket.close()",
     "WebSocket",
     "WebSocket은 지속적인 양방향 통신을 지원합니다. websocket.accept()로 연결을 수립하고, receive_text()/send_text()로 메시지를 주고받습니다. HTTP의 요청-응답 패턴보다 실시간 채팅에 더 적합합니다."),

    ("FastAPI 파일 업로드",
     "from fastapi import FastAPI, UploadFile, File\nimport shutil\nimport os\n\napp = FastAPI()\n\n@app.post('/upload')\nasync def upload_document(file: _____ = File(...)):\n    os.makedirs('uploads', exist_ok=True)\n    file_path = f'uploads/{file.filename}'\n    \n    with open(file_path, 'wb') as buffer:\n        shutil.copyfileobj(file.file, buffer)\n    \n    return {\n        'filename': file.filename,\n        'size': os.path.getsize(file_path)\n    }",
     "UploadFile",
     "UploadFile은 FastAPI의 파일 업로드 타입입니다. File(...)은 필수 파일 파라미터를 의미합니다. shutil.copyfileobj()로 스트리밍 방식으로 파일을 저장합니다. RAG 시스템에서 사용자가 PDF를 업로드하면 이 엔드포인트로 받아 처리합니다."),

    ("Docker subprocess 관리",
     "import subprocess\n\ndef build_and_run(image_name: str = 'my-llm-app', port: int = 8000):\n    build_cmd = ['docker', 'build', '-t', image_name, '.']\n    result = subprocess.run(build_cmd, _____=True, text=True)\n    if result.returncode != 0:\n        print(f'빌드 실패:\\n{result.stderr}')\n        return False\n    \n    run_cmd = ['docker', 'run', '-p', f'{port}:{port}', '--env-file', '.env', image_name]\n    subprocess.run(run_cmd)\n    return True\n\nbuild_and_run()",
     "capture_output",
     "subprocess.run()의 capture_output=True는 stdout과 stderr를 캡처합니다. result.returncode로 성공 여부를 확인합니다. Docker 명령을 Python에서 관리하면 배포 자동화 스크립트를 작성할 수 있습니다."),

    ("Prometheus 메트릭 수집",
     "from fastapi import FastAPI\nfrom prometheus_client import Counter, Histogram, generate_latest\nfrom fastapi.responses import PlainTextResponse\nimport time\n\napp = FastAPI()\n\nREQUEST_COUNT = _(\"http_requests_total\", \"총 HTTP 요청 수\", [\"method\", \"endpoint\"])\nRESPONSE_TIME = Histogram(\"response_time_seconds\", \"응답 시간\")\n\n@app.middleware('http')\nasync def metrics_middleware(request, call_next):\n    start = time.time()\n    response = await call_next(request)\n    REQUEST_COUNT.labels(method=request.method, endpoint=request.url.path).inc()\n    RESPONSE_TIME.observe(time.time() - start)\n    return response\n\n@app.get('/metrics')\ndef metrics():\n    return PlainTextResponse(generate_latest())",
     "Counter",
     "Prometheus Counter는 단조 증가하는 카운터입니다. labels()로 다차원 메트릭을 분류하고 inc()로 증가시킵니다. Histogram은 응답 시간 분포를 측정합니다. /metrics 엔드포인트를 Prometheus가 주기적으로 수집합니다."),
]

for i, (title, code, ans, explain) in enumerate(cc_data):
    questions.append({
        "chapter_name": chapter_name, "type": "코드 완성형", "difficulty": "medium", "id": str(6101 + i),
        "question": f"{title} 코드를 완성하세요.\n```python\n{code}\n```",
        "answer": ans,
        "why": explain,
        "hint": title,
    })

def get_questions():
    return questions
