import os

input_file = "/Users/tlxy/Library/Mobile Documents/com~apple~CloudDocs/Research/Projects/Ambiguity/stock_ambiguity_analysis/doc/draft/paper_overleaf/QuantAmbi2.tex"
title_page_file = "/Users/tlxy/Library/Mobile Documents/com~apple~CloudDocs/Research/Projects/Ambiguity/stock_ambiguity_analysis/doc/draft/paper_overleaf/QuantAmbi2_TitlePage.tex"
anonymous_file = "/Users/tlxy/Library/Mobile Documents/com~apple~CloudDocs/Research/Projects/Ambiguity/stock_ambiguity_analysis/doc/draft/paper_overleaf/QuantAmbi2_Anonymous.tex"

with open(input_file, 'r') as f:
    lines = f.readlines()

# Identify sections
preamble_end = 0
begin_document = 0
begin_frontmatter = 0
title_line = 0
author_start = 0
author_end = 0
abstract_start = 0
end_frontmatter = 0
end_document = len(lines)

for i, line in enumerate(lines):
    if "\begin{document}" in line:
        begin_document = i
    if "\begin{frontmatter}" in line:
        begin_frontmatter = i
    if "\title{" in line:
        title_line = i
    if "\author[" in line and author_start == 0:
        author_start = i
    if "\begin{abstract}" in line:
        abstract_start = i
    if "\end{frontmatter}" in line:
        end_frontmatter = i

# Refine author_end (it should be before abstract)
author_end = abstract_start

# Construct Title Page
# Preamble + begin document + frontmatter + title + authors + end frontmatter + end document
with open(title_page_file, 'w') as f:
    # Preamble + begin document + frontmatter
    f.writelines(lines[:begin_frontmatter+1])
    
    # Title (include comments around it if any, basically from title_line up to author_start)
    # Actually, let's include everything from begin_frontmatter+1 to abstract_start
    # But wait, we want to STOP before abstract.
    
    # Let's take lines from begin_frontmatter+1 up to abstract_start
    # This covers title and authors.
    f.writelines(lines[begin_frontmatter+1:abstract_start])
    
    # Close it off
    f.write("\end{frontmatter}\n")
    f.write("\end{document}\n")

# Construct Anonymous Article
# Preamble + begin document + frontmatter + title + (SKIP AUTHORS) + abstract + ...
with open(anonymous_file, 'w') as f:
    # Preamble + begin document + frontmatter
    f.writelines(lines[:begin_frontmatter+1])
    
    # Find where title ends. Title is usually one line or a few. 
    # But simpler: take from begin_frontmatter+1 up to author_start.
    # This includes the title.
    # CAREFUL: There might be comments or spacing between title and author.
    
    # Let's verify author_start.
    # In the file, there are comments between title and author.
    # We want to keep the title, but remove authors.
    
    # Strategy:
    # 1. Write everything up to title_line (inclusive)
    # 2. Check if title spans multiple lines? (Usually \title{...})
    #    Let's assume it's one line or we include a buffer. 
    #    Actually, let's look at the file content again.
    #    Line 100: \title{Quantifying...}
    #    Line 120: \author[inst1]{He Ni}
    #    So lines 100-119 are title + comments.
    
    # We want to keep lines up to author_start, BUT exclude the actual author commands.
    # The comments in lines 102-117 explain how to use author/affiliation.
    # We can keep them or drop them. It doesn't matter much for the PDF.
    # But strictly, we should cut from the first \author command.
    
    f.writelines(lines[begin_frontmatter+1:author_start])
    
    # Skip authors (author_start to abstract_start)
    # Write from abstract_start to end
    f.writelines(lines[abstract_start:])

print("Files created.")
