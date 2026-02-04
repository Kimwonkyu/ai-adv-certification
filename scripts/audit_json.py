import json

def audit_questions():
    file_path = "/Users/wonkyukim/vibe-workspace/vibe-web/public/questions.json"
    with open(file_path, "r", encoding="utf-8") as f:
        questions = json.load(f)
    
    mcq_missing_options = []
    code_completion_issues = []
    
    for q in questions:
        q_id = q.get("id")
        q_type = q.get("type")
        
        if q_type == "객관식":
            if "options" not in q or not q["options"]:
                mcq_missing_options.append(q_id)
        elif q_type == "코드 완성형":
            # Check if it has the placeholder ____ or similar expected format
            if "____" not in q.get("question", ""):
                code_completion_issues.append(q_id)

    print(f"Total MCQs missing options: {len(mcq_missing_options)}")
    if mcq_missing_options:
        print(f"IDs: {mcq_missing_options[:20]}...")
        
    print(f"Total Code Completion questions missing placeholder: {len(code_completion_issues)}")
    if code_completion_issues:
        print(f"IDs: {code_completion_issues[:20]}...")

if __name__ == "__main__":
    audit_questions()
