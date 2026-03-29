import re
import sys

def process_file():
    filepath = "/Users/tlxy/Library/Mobile Documents/com~apple~CloudDocs/Research/Projects/Ambiguity/stock_ambiguity_analysis/code/Adaptive/GeoUncertainty/outputs/results/resubmission/respond_3.tex"
    with open(filepath, 'r') as f:
        content = f.read()

    # We will split by \subsection*{Comment
    parts = content.split('\\subsection*{Comment ')
    
    new_parts = [parts[0]]
    for part in parts[1:]:
        # Find where the Response starts
        if '\\textbf{Response:}' in part:
            resp_start = part.find('\\textbf{Response:}')
            
            # Find where the response ends (either at the next colorbox for Revisions, or at \vspace{0.5cm} for the last one)
            if '\\colorbox{highlight}' in part[resp_start:]:
                resp_end = part.find('\\colorbox{highlight}', resp_start)
            elif '\\vspace{0.5cm}' in part[resp_start:]:
                resp_end = part.find('\\vspace{0.5cm}', resp_start)
            else:
                resp_end = len(part)
                
            # Extract the response text, stripping trailing whitespace
            response_text = part[resp_start:resp_end].strip()
            
            # Wrap it
            wrapped_response = '\\colorbox{highlight}{\\parbox{\\dimexpr\\linewidth-2\\fboxsep}{%\n' + response_text + '\n}}\n\n'
            
            # Reconstruct the part
            new_part = part[:resp_start] + wrapped_response + part[resp_end:]
            new_parts.append(new_part)
        else:
            new_parts.append(part)
            
    new_content = '\\subsection*{Comment '.join(new_parts)
    
    with open(filepath, 'w') as f:
        f.write(new_content)
        
    print("Done processing.")

if __name__ == '__main__':
    process_file()
