
import json
import os
import sys

# Import data modules
try:
    import data_ch1
    import data_ch2
    import data_ch3
    import data_ch4
    import data_ch5
    import data_ch6
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Run this script from the directory containing the data_ch*.py files or ensure they are in PYTHONPATH.")
    sys.exit(1)

def main():
    all_questions = []
    
    # Collect questions from all chapters
    modules = [data_ch1, data_ch2, data_ch3, data_ch4, data_ch5, data_ch6]
    
    for mod in modules:
        qs = mod.get_questions()
        print(f"Loaded {len(qs)} questions from {mod.chapter_name}")
        all_questions.extend(qs)
    
    print(f"Total questions: {len(all_questions)}")
    
    # Validation
    assert len(all_questions) == 720, f"Expected 720 questions, got {len(all_questions)}"
    
    # Check for duplicates or missing fields
    ids = set()
    for q in all_questions:
        if q['id'] in ids:
            print(f"Warning: Duplicate ID found: {q['id']}")
        ids.add(q['id'])
        
        required_fields = ['chapter_name', 'type', 'question', 'answer', 'why', 'difficulty', 'id']
        missing = [f for f in required_fields if f not in q]
        if missing:
             print(f"Error: Question {q['id']} missing fields: {missing}")
    
    # Write to JSON
    output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../public/questions.json'))
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_questions, f, indent=2, ensure_ascii=False)
    
    print(f"Successfully wrote {len(all_questions)} questions to {output_path}")

if __name__ == "__main__":
    main()
