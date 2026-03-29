import re

def enrich_manuscript():
    with open('code/Adaptive/GeoUncertainty/outputs/results/geopoliticalAmb04_article.tex', 'r') as f:
        content = f.read()

    # Comment 1: Ambiguity and Volatility Language
    old_intro1 = r"\\textcolor\{red\}\{\(iii\) models ignoring ambiguity misattribute a portion of ambiguity's impact to volatility, illustrating the complementary nature of these two uncertainty channels\. By treating ambiguity and volatility as dual, coexisting factors, we align our narrative with foundational theories that posit ambiguity aversion as a distinct preference from risk aversion\.\}"
    new_intro1 = r"\\textcolor{red}{(iii) models ignoring ambiguity misattribute a portion of ambiguity's impact to volatility, illustrating the complementary nature of these two uncertainty channels. The distinction between risk and Knightian uncertainty is not a competition for explanatory supremacy; rather, both capture fuimport re

def enrich_manuscript():
    with open(ss