import re

filepath = '/Users/tlxy/Library/Mobile Documents/com~apple~CloudDocs/Research/Projects/Ambiguity/stock_ambiguity_analysis/code/Adaptive/GeoUncertainty/outputs/results/resubmission/respond_3.tex'
with open(filepath, 'r') as f:
    content = f.read()

# Replace \textit{\textcolor{red}{ ... }} with {\color{red}\textit{ ... }}
# This is a bit tricky due to nested braces. But we can do a simpler replacement.
# Since the \textcolor{red} is applied to large blocks that break across pages, using \color{red} instead is much safer.
# We can find \textit{\textcolor{red}{ and replace it with {\color{red}\textit{
# And then the matching closing brace of \textcolor{red}{ needs to be removed? Wait, no.
# If we replace `\textit{\textcolor{red}{` with `{\color{red}\textit{`, the structure is:
# Before: \textit{\textcolor{red}{ ... }}
# After:  {\color{red}\textit{ ... }}
# The number of opening and closing braces remains EXACTLY the same!

content = content.replace(r'\textit{\textcolor{red}{', r'{\color{red}\textit{')

# Also handle any cases of just \textcolor{red}{...}
# Actually, the user has \textcolor{red}{ around a large block. Wait, let's look at the document.
# Example:
# \textit{\textcolor{red}{"However, while the Baron-Kenny framework ... definitive causal proof."}}
# \textit{\textcolor{red}{"While the 'Non-Asian GPR' instrument ... appropriate caution."}}
# \textit{\textcolor{red}{"We use a contemporaneous alignment ... timing assumptions."}}
# \textit{\textcolor{red}{"By doing so, we bridge ... in Appendix B."}}

# Wait, in Comment 9:
# \textit{\textcolor{red}{
# \begin{center}
# \centering
# \textbf{Table 6: Summary of Main Effects Across Model Specifications}
# ...
# }}
# Replacing \textit{\textcolor{red}{ with {\color{red}\textit{ will work perfectly for all these cases!

with open(filepath, 'w') as f:
    f.write(content)

print("Replacement done.")
