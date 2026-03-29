import re

filepath = "/Users/tlxy/Library/Mobile Documents/com~apple~CloudDocs/Research/Projects/Ambiguity/stock_ambiguity_analysis/code/Adaptive/GeoUncertainty/outputs/results/resubmission/respond_3.tex"

with open(filepath, 'r') as f:
    content = f.read()

# 1. Add tcolorbox package and highlightbox definition to preamble
preamble_addition = r"""
\usepackage[skins,breakable]{tcolorbox}
\newtcolorbox{highlightbox}{
    colback=highlight,
    colframe=highlight,
    boxrule=0pt,
    arc=0pt,
    outer arc=0pt,
    top=1ex, bottom=1ex, left=1ex, right=1ex,
    breakable,
    before upper={\parskip=0.5\baselineskip}
}
"""

# Insert before \title or \hypersetup
if r'\title{' in content:
    content = content.replace(r'\title{', preamble_addition + '\n' + r'\title{')
elif r'\begin{document}' in content:
    content = content.replace(r'\begin{document}', preamble_addition + '\n' + r'\begin{document}')

# 2. Replace \colorbox{highlight}{\parbox{\dimexpr\linewidth-2\fboxsep}{% with \begin{highlightbox}
content = content.replace(r'\colorbox{highlight}{\parbox{\dimexpr\linewidth-2\fboxsep}{%', r'\begin{highlightbox}')

# 3. We need to replace the closing '}}' for each of these boxes. 
# It's a bit tricky because '}}' might appear elsewhere, but looking at the grep output, 
# it's on its own line like '}}' or '}}' at the end of the block.
# Actually, the grep shows '}}' is often on a line by itself. Let's do a regex replacement.
# A safe way is to replace '\n}}' with '\n\end{highlightbox}' ONLY IF it corresponds to the box.
# Let's count \begin{highlightbox}
count_begin = content.count(r'\begin{highlightbox}')
print(f"Found {count_begin} highlightbox beginnings.")

# Since '}}' on its own line or followed by newline is what we used in the previous script to wrap.
# Let's do a more robust approach: find \begin{highlightbox}, then find the next '}}' that closes it.
# Actually, looking at the previous python script that generated these boxes:
# wrapped_response = '\\colorbox{highlight}{\\parbox{\\dimexpr\\linewidth-2\\fboxsep}{%\n' + response_text + '\n}}\n\n'
# So the closing is strictly '\n}}\n\n' or '\n}}'
content = content.replace('\n}}\n', '\n\\end{highlightbox}\n')
# Also handle EOF if needed
if content.endswith('\n}}'):
    content = content[:-3] + '\n\\end{highlightbox}'

# Double check the count of \end{highlightbox}
count_end = content.count(r'\end{highlightbox}')
print(f"Found {count_end} highlightbox endings.")

with open(filepath, 'w') as f:
    f.write(content)
