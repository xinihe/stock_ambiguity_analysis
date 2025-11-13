#!/bin/bash

# Strict mode: exit on error, unset variables, and pipe failure
set -euo pipefail

# Check if an argument is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <filename_without_extension>"
    exit 1
fi

# Get the base filename from the first argument
FILENAME=$1

# --- Compilation Workflow ---
echo "Starting compilation for ${FILENAME}.tex..."

# First pdflatex pass to generate .aux file
echo "--> Running pdflatex (pass 1)..."
pdflatex "${FILENAME}.tex"

# Run bibtex to process the bibliography
echo "--> Running bibtex..."
bibtex "${FILENAME}"

# Second pdflatex pass to incorporate the bibliography
echo "--> Running pdflatex (pass 2)..."
pdflatex "${FILENAME}.tex"

# Third pdflatex pass to resolve all cross-references
echo "--> Running pdflatex (pass 3)..."
pdflatex "${FILENAME}.tex"

# --- File Cleanup ---
echo "--> Cleaning up auxiliary files..."
rm -f "${FILENAME}.aux" "${FILENAME}.bbl" "${FILENAME}.blg" "${FILENAME}.log" "${FILENAME}.out" "${FILENAME}.fdb_latexmk" "${FILENAME}.fls" "${FILENAME}.spl" "${FILENAME}.synctex.gz"

echo ""
echo "Success! ${FILENAME}.pdf has been generated."