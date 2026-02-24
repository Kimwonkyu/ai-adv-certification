import json
import os

def main():
    with open('public/questions.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # Remove existing chapter 6 questions (ID starts with 6)
    new_data = [q for q in data if not str(q['id']).startswith('6')]
    
    if os.path.exists('ch6_new.json'):
        with open('ch6_new.json', 'r', encoding='utf-8') as f:
            ch6_data = json.load(f)
            
        final_data = new_data + ch6_data
        final_data.sort(key=lambda x: str(x['id']))
        
        with open('public/questions.json', 'w', encoding='utf-8') as f:
            json.dump(final_data, f, ensure_ascii=False, indent=2)
            
        print(f"Merged successfully. Total questions: {len(final_data)}.")
    else:
        print("Error: ch6_new.json not found!")

if __name__ == "__main__":
    main()
