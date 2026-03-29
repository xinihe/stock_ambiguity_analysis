import re
with open('code/Adaptive/GeoUncertainty/outputs/results/geopoliticalAmb04_article.tex', 'r') as f:
    text = f.read()

matches = re.findall(r'\\textcolor\{red\}\{(.*?)\}', text, re.DOTALL)
for i, m in enumerate(matches):
    print(f"--- Match {i} ---")
    print(m)
