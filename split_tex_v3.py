import os

input_file = "/Users/tlxy/Library/Mobile Documents/com~apple~CloudDocs/Research/Projects/Ambiguity/stock_ambiguity_analysis/doc/draft/paper_overleaf/QuantAmbi2.tex"
title_page_file = "/Users/tlxy/Library/Mobile Documents/com~apple~CloudDocs/Research/Projects/Ambiguity/stock_ambiguity_analysis/doc/draft/paper_overleaf/QuantAmbi2_TitlePage.tex"
anonymous_file = "/Users/tlxy/Library/Mobile Documents/com~apple~CloudDocs/Research/Projects/Ambiguity/stock_ambiguity_analysis/doc/draft/paper_overleaf/QuantAmbi2_Anonymous.tex"

print(f"Reading from: {input_file}")

with open(input_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Identify sections
begin_document = -1
begin_frontmatter = -1
title_line = -1
author_start = -1
abstract_start = -1
end_frontmatter = -1

for i, line in enumerate(lines):
    if "\\begin{document}" in line:
        begin_document = i
    if "\\begin{frontmatter}" in line:
        begin_frontmatter = i
    if "\\title{" in line:
        title_line = i
    if "\\author[" in line and author_start == -1:
        author_start = i
    if "\\begin{abstract}" in line:
        abstract_start = i
    if "\\end{frontmatter}" in line:
        end_frontmatter = i

print(f"Indices found:")
print(f"begin_document: {begin_document}")
print(f"begin_frontmatter: {begin_frontmatter}")
print(f"title_line: {title_line}")
print(f"author_start: {author_start}")
print(f"abstract_start: {abstract_start}")
print(f"end_frontmatter: {end_frontmatter}")

if begin_frontmatter == -1 or abstract_start == -1:
    print("Critical markers not found!")
    exit(1)

# Title Page
print("Writing Title Page...")
with open(title_page_file, 'w', encoding='utf-8') as f:
    # Everything up to and including \begin{frontmatter}
    f.writelines(lines[:begin_frontmatter+1])
    
    # Title and Authors (from after \begin{frontmatter} up to BEFORE \begin{abstract})
    # We include everything until abstract_start
    f.writelines(lines[begin_frontmatter+1:abstract_start])
    
    # Close
    f.write("\\end{frontmatter}\n")
    f.write("\\end{document}\n")

# Anonymous Article
print("Writing Anonymous Article...")
with open(anonymous_file, 'w', encoding='utf-8') as f:
    # Everything up to and including \begin{frontmatter}
    f.writelines(lines[:begin_frontmatter+1])
    
    # Title only.
    # We write lines from begin_frontmatter+1 up to author_start
    if author_start != -1:
        f.writelines(lines[begin_frontmatter+1:author_start])
    else:
        print("Warning: \\author[ not found. Writing up to abstract...")
        f.writelines(lines[begin_frontmatter+1:abstract_start])
        
    # Skip authors, write from abstract to end
    f.writelines(lines[abstract_start:])

print("Done.")
