with open("code/Adaptive/GeoUncertainty/outputs/results/review/respond_3.tex", "r") as f:
    text = f.read()

text = text.replace(r"\begin{table}[htbp]", r"\begin{center}")
text = text.replace(r"\end{table}", r"\end{center}")
text = text.replace(r"\caption{", r"\textbf{Table 5: ")

with open("code/Adaptive/GeoUncertainty/outputs/results/review/respond_3.tex", "w") as f:
    f.write(text)
