# 프로젝트 인수인계 문서 (Handoff Document)

## 현재 상태 (Current State)
- 루트 디렉토리에 있던 Next.js 프로젝트 설정, 소스 코드(`src/`), 데이터 자동화 스크립트(`scripts/`) 및 교재 파일(`public/textbooks/` 등) 들이 모두 `ai-adv/` 폴더 안으로 이동(리팩토링)되었습니다.
- 이전 파일들은 삭제되고, 대신 하위 폴더인 `ai-adv/`에서 관리되도록 구조가 변경되었습니다. 

## 미반영된 변경사항 (Git Status)
- `ai-adv/` 디렉토리가 새롭게 Untracked 상태로 존재합니다.
- 기존 파일들은 모두 삭제(`deleted`) 상태로 잡혀 있습니다.
- `.vscode/` 디렉토리가 새롭게 Untracked 상태로 존재합니다.
- `.gitignore` 파일이 수정되었습니다.

## 다음 진행을 위한 가이드 (Next Steps for AI Assistant)

이 문서를 읽은 AI 어시스턴트(Claude Code 등)는 다음을 수행해 주세요:
1. `ai-adv/` 폴더 내의 구조를 파악하고, 필요한 패키지나 Next.js 앱의 진입점을 확인합니다.
2. 현재까지 변경된 폴더 구조 이동 및 코드 변경(삭제 및 새 폴더 생성) 내역을 확인해 주세요.
3. 사용자가 요청하는 다음 작업을 이 새로운 구조(`ai-adv/` 기준)에 맞추어 진행해 주세요.
