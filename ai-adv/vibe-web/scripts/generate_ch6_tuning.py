import os
import json
import time
import google.generativeai as genai
from concurrent.futures import ThreadPoolExecutor, as_completed

# Setup Gemini API key
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

model = genai.GenerativeModel('gemini-3.1-pro-preview', generation_config={"response_mime_type": "application/json"})

def generate_q(topic_line, index):
    q_type = "객관식" if index < 100 else "코드 완성형"
    
    if index % 5 == 0:
        difficulty = "easy"
    elif index % 5 == 4:
        difficulty = "hard"
    else:
        difficulty = "medium"

    prompt = f"""You are an expert AI/Python instructor creating exams.
Create a high-quality Korean exam question based strictly on this topic:
Topic: {topic_line}

Requirements:
- Chapter Name: "LLM 튜닝"
- Type: "{q_type}"
- Difficulty: "{difficulty}" (Do NOT use complex paragraph-long scenarios. Just ask a straightforward conceptual or theoretical question, but make the distractors appropriately difficult).
- Question Text: Must be short, direct, and in Korean. (e.g., "다음 중 ~에 대한 설명으로 옳은 것은?", "~의 주된 원인은?", "~ 기법은?")
- Answer/Options/Why/Hint: Must be in Korean.

For '객관식', provide exactly 5 'options' strings.
For '코드 완성형', do NOT provide 'options' field. Instead, put a valid Python code block with `_____` in the 'question' field, and the answer in 'answer'.

Respond ONLY with a valid JSON object matching this schema exactly:
{{
  "chapter_name": "LLM 튜닝",
  "type": "{q_type}",
  "difficulty": "{difficulty}",
  "id": "6{index+1:03d}",
  "question": "...",
  "options": ["...", "...", "...", "...", "..."], // Only if type is '객관식'
  "answer": "...",
  "why": "...",
  "hint": "..."
}}
"""
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            data = json.loads(response.text)
            # Enforce metadata
            data['id'] = f"6{index+1:03d}"
            data['chapter_name'] = "LLM 튜닝"
            data['type'] = q_type
            data['difficulty'] = difficulty
            
            if q_type == '객관식':
                if 'options' not in data or len(data['options']) != 5:
                    raise ValueError("Missing exactly 5 options")
            else:
                if 'options' in data:
                    del data['options']
                    
            return data
        except Exception as e:
            time.sleep((attempt + 1) * 3)
            
    print(f"Failed to generate for index {index}")
    return None

def main():
    with open('topics_ch6.txt', 'r', encoding='utf-8') as f:
        topics = [line.strip() for line in f if line.strip() and "question" in line]
    
    clean_topics = []
    for t in topics:
        start = t.find('"question":')
        if start != -1:
            q_start = t.find('"', start+11)
            q_end = t.rfind('"')
            if q_start != -1 and q_end != -1 and q_end > q_start:
                clean_topics.append(t[q_start+1:q_end])
            else:
                clean_topics.append(t)
        else:
            clean_topics.append(t)
            
    results = []
    # 3 workers to bypass rate limits smoothly
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(generate_q, clean_topics[i], i): i for i in range(120)}
        for future in as_completed(futures):
            res = future.result()
            if res:
                results.append(res)
                if len(results) % 10 == 0:
                    print(f"Generated {len(results)}/120")
                
    results.sort(key=lambda x: str(x['id']))
    
    with open('ch6_new.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        
    print("Done generating 120 questions for LLM Tuning with Gemini!")

if __name__ == "__main__":
    main()
