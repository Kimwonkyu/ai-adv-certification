import json
import collections

ORIGINAL_FILE = "public/questions.json"
REFINED_FILE = "public/questions_refined.json"

def verify():
    with open(ORIGINAL_FILE, 'r', encoding='utf-8') as f:
        orig_data = json.load(f)
    with open(REFINED_FILE, 'r', encoding='utf-8') as f:
        ref_data = json.load(f)

    print(f"Original Count: {len(orig_data)}")
    print(f"Refined Count: {len(ref_data)}")

    if len(orig_data) != len(ref_data):
        print("ERROR: Counts do not match!")
        return False

    orig_diffs = collections.Counter([q['difficulty'] for q in orig_data])
    ref_diffs = collections.Counter([q['difficulty'] for q in ref_data])

    print(f"Original Difficulties: {orig_diffs}")
    print(f"Refined Difficulties: {ref_diffs}")

    if orig_diffs != ref_diffs:
        print("ERROR: Difficulty distribution changed!")
        return False

    orig_chapters = collections.Counter([q['chapter_name'] for q in orig_data])
    ref_chapters = collections.Counter([q['chapter_name'] for q in ref_data])

    print(f"Original Chapters: {orig_chapters}")
    print(f"Refined Chapters: {ref_chapters}")

    if orig_chapters != ref_chapters:
        print("ERROR: Chapter distribution changed!")
        return False

    # Check schema
    required_keys = {'chapter_name', 'type', 'difficulty', 'id', 'question', 'answer', 'why', 'hint'}
    for i, q in enumerate(ref_data):
        missing = required_keys - set(q.keys())
        if missing:
            print(f"ERROR: Question {q['id']} is missing keys: {missing}")
            return False
            
        if q['type'] == '객관식':
            if 'options' not in q:
                print(f"ERROR: 객관식 Question {q['id']} is missing options")
                return False
            if len(q['options']) != 5:
                print(f"ERROR: 객관식 Question {q['id']} does not have exactly 5 options. Found: {len(q['options'])}")
                return False
            
            # Check for very short options (indicating poor quality distractor)
            short_options = [opt for opt in q['options'] if len(str(opt).strip()) < 3 and not str(opt).isdigit()]
            if short_options and orig_data[i]['type'] == '객관식':
                # Just a warning, some short options might be ok depending on context
                print(f"WARNING: Question {q['id']} has short options: {short_options}")

        if str(q['id']) != str(orig_data[i]['id']):
             print(f"ERROR: ID mismatch at index {i}: {q['id']} != {orig_data[i]['id']}")
             return False

    print("SUCCESS: Structure and distributions are perfectly aligned.")
    return True

if __name__ == "__main__":
    verify()
