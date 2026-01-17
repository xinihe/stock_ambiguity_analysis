input_file = "/Users/tlxy/Library/Mobile Documents/com~apple~CloudDocs/Research/Projects/Ambiguity/stock_ambiguity_analysis/doc/draft/paper_overleaf/QuantAmbi2.tex"

with open(input_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

print(f"Total lines: {len(lines)}")
for i in range(68, 76):
    if i < len(lines):
        print(f"Line {i+1}: {repr(lines[i])}")
