import json

def extract_problematic_questions():
    file_path = "/Users/wonkyukim/vibe-workspace/vibe-web/public/questions.json"
    with open(file_path, "r", encoding="utf-8") as f:
        questions = json.load(f)
    
    mcq_to_fix = []
    
    for q in questions:
        if q.get("type") == "객관식" and ("options" not in q or not q["options"]):
            mcq_to_fix.append({
                "id": q.get("id"),
                "question": q.get("question"),
                "answer": q.get("answer")
            })

    with open("mcq_to_fix.json", "w", encoding="utf-8") as f:
        json.dump(mcq_to_fix, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    extract_problematic_questions()
