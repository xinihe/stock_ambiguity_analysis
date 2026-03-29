with open("code/Adaptive/GeoUncertainty/outputs/results/review/respond_3.tex", "r") as f:
    text = f.read()

text = text.replace(r"\citet{Baker2016}", "Baker et al. (2016)")
text = text.replace(r"\citet{Caldara2022}", "Caldara and Iacoviello (2022)")
text = text.replace(r"\citet{Brenner2018}", "Brenner and Izhakian (2018)")
text = text.replace(r"\begin{table}[H]", r"\begin{table}[htbp]")

with open("code/Adaptive/GeoUncertainty/outputs/results/review/respond_3.tex", "w") as f:
    f.write(text)
