import re
import sys

def remove_textcolor(text):
    while True:
        idx = text.find('\\textcolor{red}{')
        if idx == -1:
            break
        # Find matching closing brace
        brace_count = 1
        i = idx + len('\\textcolor{red}{')
        while i < len(text) and brace_count > 0:
            if text[i] == '{':
                brace_count += 1
            elif text[i] == '}':
                brace_count -= 1
            i += 1
        if brace_count == 0:
            content = text[idx + len('\\textcolor{red}{'):i-1]
            text = text[:idx] + content + text[i:]
        else:
            break
    return text

def remove_color(text):
    while True:
        idx = text.find('{\\color{red}')
        if idx == -1:
            break
        # Find matching closing brace
        brace_count = 1
        i = idx + len('{\\color{red}')
        while i < len(text) and brace_count > 0:
            if text[i] == '{':
                brace_count += 1
            elif text[i] == '}':
                brace_count -= 1
            i += 1
        if brace_count == 0:
            content = text[idx + len('{\\color{red}'):i-1]
            text = text[:idx] + content + text[i:]
        else:
            break
    return text

if __name__ == "__main__":
    with open('code/Adaptive/GeoUncertainty/outputs/results/geopoliticalAmb03_article.tex', 'r') as f:
        content = f.read()
    
    content = remove_textcolor(content)
    content = remove_color(content)
    
    with open('code/Adaptive/GeoUncertainty/outputs/results/geopoliticalAmb04_article.tex', 'w') as f:
        f.write(content)
