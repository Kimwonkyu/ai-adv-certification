import json
from collections import defaultdict

with open('public/questions.json', 'r') as f:
    data = json.load(f)

stats = defaultdict(lambda: {'객관식': 0, '코드 완성형': 0})
for q in data:
    stats[q['chapter_name']][q['type']] += 1

for chap, counts in stats.items():
    print(f"{chap}: 객관식 {counts['객관식']}, 코드 완성형 {counts['코드 완성형']}")
