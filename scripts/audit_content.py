import json
from collections import defaultdict
import difflib

def audit_content():
    try:
        with open('public/questions.json', 'r', encoding='utf-8') as f:
            questions = json.load(f)
    except FileNotFoundError:
        print("Error: public/questions.json not found.")
        return

    chapter_counts = defaultdict(int)
    fine_tuning_questions = []
    rag_agent_questions = []
    
    # 1. Basic Counts and Separation
    for q in questions:
        chapter_counts[q['chapter_name']] += 1
        if q['chapter_name'] == 'Fine Tuning':
            fine_tuning_questions.append(q)
        elif q['chapter_name'] == 'RAG & Agent':
            rag_agent_questions.append(q)

    print("=== Chapter Counts ===")
    for chap, count in chapter_counts.items():
        print(f"  {chap}: {count}")

    # 2. Check for Duplicates in ALL Chapters
    print("\n=== Duplicates/Similarities by Chapter ===")
    
    questions_by_chapter = defaultdict(list)
    for q in questions:
        questions_by_chapter[q['chapter_name']].append(q)

    total_duplicates = 0
    
    for chapter, chat_questions in questions_by_chapter.items():
        seen_questions = {}
        chapter_duplicates = []
        
        for q in chat_questions:
            q_text = q['question'].strip()
            # Simple exact match check
            if q_text in seen_questions:
                chapter_duplicates.append((q['id'], seen_questions[q_text], q_text))
            else:
                seen_questions[q_text] = q['id']
        
        if chapter_duplicates:
            print(f"\n[{chapter}] Found {len(chapter_duplicates)} duplicates:")
            for new_id, old_id, text in chapter_duplicates:
                print(f"  - ID {new_id} is duplicate of {old_id}: {text[:40]}...")
            total_duplicates += len(chapter_duplicates)
        else:
            print(f"\n[{chapter}] No exact duplicates found.")

    print(f"\nTotal Duplicates Found: {total_duplicates}")

    # 3. Check for 'All of the above' in ALL Chapters
    print("\n=== 'All of the above' Patterns in ALL Chapters ===")
    problematic_answers = ["모두", "all of the above", "위의 셋", "전부", "위의 모든"]
    found_issues = []
    
    for q in questions:
        # Check answer field
        ans_lower = q['answer'].lower()
        if any(p in ans_lower for p in problematic_answers):
             # Filter out legit sentences containing these words if necessary, 
             # but usually "모두" as a standalone or start is suspicious in this context.
             # We'll valid strict matches or short matches.
             if len(ans_lower) < 20: # Heuristic: short answers likely to be "위 셋 모두"
                 found_issues.append((q['chapter_name'], q['id'], q['answer'], "Answer pattern"))
        
        # Check options if answer matches a "All of above" option
        if q.get('options'):
            for opt in q['options']:
                 if any(p in opt.lower() for p in problematic_answers):
                     if opt == q['answer']:
                         found_issues.append((q['chapter_name'], q['id'], q['answer'], "Option pattern"))

    if found_issues:
        print(f"Found {len(found_issues)} potentially problematic answers:")
        for chap, qid, ans, reason in found_issues:
            print(f"  - [{chap}] ID {qid}: {ans} ({reason})")
    else:
        print("No obvious 'All of the above' answers found.")

if __name__ == "__main__":
    audit_content()
