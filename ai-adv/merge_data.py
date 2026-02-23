import json
import glob
import os

def merge_questions():
    pattern = "/Users/wonkyukim/vibe-workspace/archive/data-pipeline/seeds*.json"
    files = glob.glob(pattern)
    
    all_questions = []
    seen_questions = set()
    
    # Sort files to maintain some order (optional)
    files.sort()
    
    for file_path in files:
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                questions = data.get('questions', [])
                for q in questions:
                    # Map types to Korean
                    if q.get('type') == 'multiple_choice':
                        q['type'] = '객관식'
                    elif q.get('type') == 'short_answer':
                        q['type'] = '주관식'
                        
                    # Deduplicate based on question text
                    q_text = q.get('question', '')
                    if q_text not in seen_questions:
                        all_questions.append(q)
                        seen_questions.add(q_text)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    
    # Add unique IDs
    for idx, q in enumerate(all_questions):
        q['id'] = f"q-{idx+1:03}"
        
    output_dir = "/Users/wonkyukim/vibe-workspace/vibe-web/public"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "questions.json")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_questions, f, ensure_ascii=False, indent=2)
    
    print(f"Merged {len(all_questions)} questions into {output_path}")

if __name__ == "__main__":
    merge_questions()
