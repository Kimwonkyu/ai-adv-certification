import json
import random

def trim_content():
    try:
        with open('public/questions.json', 'r', encoding='utf-8') as f:
            questions = json.load(f)
    except FileNotFoundError:
        print("Error: public/questions.json not found.")
        return

    # Group by chapter
    grouped = {}
    for q in questions:
        c = q['chapter_name']
        if c not in grouped:
            grouped[c] = []
        grouped[c].append(q)

    final_list = []
    
    # Process each chapter
    for chap, qs in grouped.items():
        mcqs = [q for q in qs if q['type'] == '객관식']
        codes = [q for q in qs if q['type'] == '코드 완성형']
        
        target_mcq_count = 100
        current_mcq_count = len(mcqs)
        
        if current_mcq_count > target_mcq_count:
            # We need to remove some MCQs.
            # Priority to KEEP:
            # 1. New IDs starting with '08' or '09' (Fixed/Enhanced content)
            # 2. Specifically fixed IDs '0547', '0019', '0041', '0649' 
            # (though these likely fall into MCQs)
            
            must_keep = []
            removable = []
            
            # IDs to protect
            protected_prefixes = ['08', '09']
            protected_exact = ['0547', '0019', '0041', '0649']
            
            for q in mcqs:
                qid = q['id']
                is_protected = False
                
                # Check prefixes
                for p in protected_prefixes:
                    if qid.startswith(p):
                        is_protected = True
                        break
                
                # Check exact matches
                if qid in protected_exact:
                    is_protected = True
                
                if is_protected:
                    must_keep.append(q)
                else:
                    removable.append(q)
            
            # Calculate how many we still need to reach 100
            slots_needed = target_mcq_count - len(must_keep)
            
            if slots_needed > 0:
                # We need to take 'slots_needed' from 'removable'
                # To maintain some consistency/order, we can just sort by ID and take the first ones 
                # or random. User wants "diversity". Random removal of OLD questions is safer 
                # to avoid removing a contiguous block of concepts.
                
                # Sort removable by ID first to be deterministic, then maybe shuffle?
                # Actually, simple ID sort is better for reproducibility.
                # But removing the *first* ones (0001, 0002...) might be removing basics.
                # Removing the *last* old ones might be better?
                # Let's verify IDs. Old IDs are like 0001~0700.
                # Let's just keep the *last* N of the removable list (assuming later ID = slightly more advanced or just arbitrary).
                # Actually, let's keep the *first* N to preserve "Basic" questions?
                # Usually Chapter 1 basics are important.
                # Let's keep a mix. 
                # Best strategy: Remove from the *middle* of removable to preserve Intro and Advanced?
                # Let's simply take the first N (lowest IDs) because they are foundational.
                removable.sort(key=lambda x: x['id'])
                selected_removable = removable[:slots_needed]
                
                kept_mcqs = must_keep + selected_removable
            else:
                # We have so many protected questions that they exceed 100? 
                # Unlikely (only ~30 added). But if so, we keep all protected and cutoff.
                kept_mcqs = must_keep[:target_mcq_count]
            
            # Sort final kept list by ID for tidiness
            kept_mcqs.sort(key=lambda x: x['id'])
            
            print(f"[{chap}] Trimmed MCQs from {current_mcq_count} to {len(kept_mcqs)}. (Kept {len(codes)} Code Completion)")
            final_list.extend(kept_mcqs)
            final_list.extend(codes)
            
        else:
            print(f"[{chap}] Count OK ({current_mcq_count} MCQs).")
            final_list.extend(mcqs)
            final_list.extend(codes)

    # Save
    with open('public/questions.json', 'w', encoding='utf-8') as f:
        json.dump(final_list, f, indent=2, ensure_ascii=False)
    
    print(f"Total Final Count: {len(final_list)}")

if __name__ == "__main__":
    trim_content()
