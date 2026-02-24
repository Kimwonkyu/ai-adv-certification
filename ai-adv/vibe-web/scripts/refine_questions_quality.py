import json
import os
import time
from traceback import print_exc
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

# Initialize the OpenAI client (expects OPENAI_API_KEY environment variable)
client = OpenAI()

INPUT_FILE = "public/questions.json"
OUTPUT_FILE = "public/questions_refined.json"

SYSTEM_PROMPT = """You are an expert technical instructor and exam creator for an advanced AI/Python certification.
Your task is to refine a given exam question to vastly improve its quality, specifically focusing on its distractors (incorrect options) and overall difficulty curve, WITHOUT changing the core topic or the correct answer conceptually.

Guidelines based on difficulty:
1. 'easy': The question should still test basic concepts, but the distractors MUST be highly plausible misconceptions. Avoid silly, obvious, or overly short (<5 chars) distractors. They should look like real technical terms or common beginner mistakes.
2. 'medium': Move away from pure definitional questions. Frame the question as a small scenario, a code snippet behavior, or a practical use-case. Distractors must be challenging and require applying knowledge, not just recalling it.
3. 'hard': The question MUST be complex. Frame it as a debugging scenario, an edge-case behavior, system architecture decision, or performance trade-off. Distractors should be highly attractive to intermediate developers who lack deep expert knowledge.
4. '코드 완성형' (Code Completion): Ensure the blank in the code requires understanding the logic, not just a trivial syntax rule. Update the description/options if necessary.

Output Format:
You MUST return ONLY valid JSON matching the exact input schema. Do not output markdown code blocks (` ```json `), just the raw JSON object.
Schema:
{
  "chapter_name": "...",
  "type": "...",
  "difficulty": "...",
  "id": "...",
  "question": "...",
  "options": ["...", "...", "...", "...", "..."], // Only for 객관식
  "answer": "...",
  "why": "...",
  "hint": "..."
}

CRITICAL RULES:
- The `id`, `chapter_name`, `type`, and `difficulty` MUST remain exactly the same.
- For '객관식', exactly 5 options must be provided.
- For '코드 완성형', do not provide 'options'. The question must contain a code block with `_____`.
- The 'why' explanation should be comprehensive and clearly explain why the answer is correct and why the distractors are wrong.
- The output MUST be valid, parsable JSON.
"""

def refine_question(question_obj):
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Refine this question:\n\n{json.dumps(question_obj, ensure_ascii=False, indent=2)}"}
                ],
                temperature=0.4,
                max_tokens=1500,
                response_format={"type": "json_object"}
            )
            
            refined_json_str = response.choices[0].message.content
            refined_obj = json.loads(refined_json_str)
            
            # Enforce keeping original metadata
            refined_obj["id"] = question_obj["id"]
            refined_obj["chapter_name"] = question_obj["chapter_name"]
            refined_obj["type"] = question_obj["type"]
            refined_obj["difficulty"] = question_obj["difficulty"]
            
            return refined_obj

        except Exception as e:
            err_msg = str(e)
            if '429' in err_msg or 'Rate limit' in err_msg:
                sleep_time = (attempt + 1) * 3
                print(f"Rate limit hit for {question_obj.get('id')}. Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                print(f"Error refining question {question_obj.get('id')}: {e}")
                return question_obj
                
    print(f"Failed to refine question {question_obj.get('id')} after {max_retries} retries.")
    return question_obj

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} questions. Starting refinement process using ThreadPoolExecutor...")
    
    refined_data = []
    
    # Using 3 workers to respect API rate limits (TPM: 30000)
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_to_q = {executor.submit(refine_question, q): q for q in data}
        
        count = 0
        for future in as_completed(future_to_q):
            result = future.result()
            refined_data.append(result)
            count += 1
            if count % 50 == 0:
                print(f"Processed {count}/{len(data)} questions...")

    # Sort back by original ID since asynchronous processing messes up the order
    refined_data.sort(key=lambda x: str(x.get('id', '')))

    # Overwrite the original file or save to a new one (safe approach first)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(refined_data, f, ensure_ascii=False, indent=2)

    print(f"\nRefinement completed. Saved {len(refined_data)} questions to {OUTPUT_FILE}.")

if __name__ == "__main__":
    main()
