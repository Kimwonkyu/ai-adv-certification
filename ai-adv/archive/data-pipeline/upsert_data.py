import json
import os
from supabase import create_client, Client

# 1. Supabase 설정
SUPABASE_URL = "https://zgbzkjypxxirxmvxkafu.supabase.co"
SUPABASE_KEY = "sb_publishable_Sc_5RBq0bUq8Bviyz-I8hQ_Pp2AxUeV"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def upload_data():
    try:
        # 데이터 파일 로드
        with open('seeds.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 1. Chapters ID 매핑 (이전에 생성된 Chapter 명칭 기반)
        chapters_res = supabase.table("chapters").select("id, name").execute()
        chapter_map = {item['name']: item['id'] for item in chapters_res.data}
        
        if not chapter_map:
            print("에러: 'chapters' 테이블에 데이터가 없습니다. schema.sql의 Insert문을 먼저 실행해 주세요.")
            return

        # 2. Questions 데이터 준비 및 삽입
        processed_questions = []
        for q in data['questions']:
            chapter_name = q.get('chapter_name')
            if chapter_name in chapter_map:
                processed_questions.append({
                    "chapter_id": chapter_map[chapter_name],
                    "type": q['type'],
                    "question": q['question'],
                    "options": q.get('options'),
                    "answer": q['answer'],
                    "why": q.get('why'),
                    "hint": q.get('hint'),
                    "trap_points": q.get('trap_points'),
                    "difficulty": q.get('difficulty', 'medium')
                })
        
        if processed_questions:
            # 100개씩 청크로 나누어 삽입 (안정성)
            chunk_size = 100
            for i in range(0, len(processed_questions), chunk_size):
                chunk = processed_questions[i:i + chunk_size]
                res = supabase.table("questions").insert(chunk).execute()
                print(f"{i + len(chunk)}번째 문제 삽입 완료...")
            
            print(f"총 {len(processed_questions)}개의 문제가 성공적으로 업로드되었습니다.")
        else:
            print("업로드할 유효한 데이터가 없습니다.")

    except Exception as e:
        print(f"데이터 업로드 중 오류 발생: {e}")

if __name__ == "__main__":
    if SUPABASE_URL == "YOUR_SUPABASE_URL":
        print("URL과 KEY를 스크립트에 설정하거나 환경 변수로 제공해 주세요.")
    else:
        upload_data()
