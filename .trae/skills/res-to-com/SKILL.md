---
name: "res-to-com"
description: "Builds a referee-response workflow for paper revisions. Invoke when revising a manuscript from comments and producing a response letter, marked TeX revision, and PDFs."
---

# Res-to-Com (Response to Comments)

Use this skill when a paper revision requires all of the following:

- reading referee, reviewer, or associate editor comments
- revising a LaTeX manuscript
- drafting a point-by-point response letter
- marking manuscript changes in red
- recompiling the updated TeX files into PDFs

## Core Workflow

1. Read the comments file and split it into discrete numbered issues.
2. Read the current manuscript and map each comment to exact sections, tables, appendices, or claims that must change.
3. Draft a response strategy for each comment before editing:
   - acknowledge the concern
   - explain the issue
   - state the action taken
   - quote the revised manuscript language
4. Create or update the response letter `.tex`:
   - one subsection per comment
   - exact quoted comment
   - prose response block
   - separate manuscript-revision block
5. Create or update the marked manuscript `.tex`:
   - preserve the original structure
   - revise the affected text directly
   - highlight inserted or changed text in red
6. Synchronize the response letter and manuscript:
   - every claimed edit in the response letter must exist in the revised manuscript
   - all section numbers, table numbers, and statistics must match
7. Compile both outputs to PDF.
8. Clean auxiliary files if appropriate.

## Required Standards

- Do not make claims beyond the evidence.
- If the comment asks for moderation, soften both interpretation and tone.
- If the comment asks for more literature context, add new comparative discussion without duplicating existing text.
- If the comment requests methodological caveats, state the identification limits explicitly.
- If the comment refers to tables or numeric results, verify every value against the revised manuscript.

## Response Letter Template

```tex
\subsection*{Comment N: [short title]}
\textit{"[exact comment here]"}

\begin{highlightbox}
\textbf{Response:} [explanatory response]
\end{highlightbox}

\begin{highlightbox}
\textbf{Revisions in the Manuscript:}

In the [section], we revised the text to state:
\vspace{0.2cm}
\begin{quote}
\color{red}\itshape "[quoted revised manuscript text]"
\end{quote}
\end{highlightbox}
```

## Marked Revision Template

```tex
Original sentence. \textcolor{red}{Revised sentence inserted here.}
```

## Important Formatting Rule

For long quoted revisions inside the response letter, prefer:

```tex
\begin{quote}
\color{red}\itshape "..."
\end{quote}
```

This is safer than wrapping the entire long passage inside `\textcolor{red}{...}` when the quote may flow across pages.

## Round Versioning Convention

Increment the response letter (`respond_n.tex`) and comment file (`comments_n.md`) every round, regardless of whether the manuscript base changes.

Increment the manuscript revision filename only when the base manuscript changes (e.g., after a reject-and-resubmit):

| Round | Comment file | Response letter | Marked manuscript |
|-------|-------------|-----------------|-------------------|
| 1 | `comments1.md` | `respond_1.tex` | `rev_marked_v1.tex` |
| 2 | `comments2.md` | `respond_2.tex` | `rev_marked_v2.tex` |
| 3 | `comments3.md` | `respond_3.tex` | `rev_marked_v2.tex` |

Always maintain a `COMMENT_TRACKING.md` file in the resubmission root. After each round, archive the prior-round files to `history/round{N}/`.

## Compilation Rule

Compile the response letter and the marked manuscript after all edits. If bibliography regeneration is required, run the full LaTeX cycle; otherwise run `pdflatex` directly.

## Multi-Round Journal Revision Workflow

### Version Naming Convention

The manuscript filename only increments when the base manuscript changes (e.g., after a reject-and-resubmit). The response letter and comment file always increment per round, even if the manuscript base stays the same.

### Round-Tracking Comment Table

Maintain a master comment table across all rounds in `COMMENT_TRACKING.md`. Add a new column for each round.

```markdown
| C# | Section | Issue | Round 1 | Round 2 | Round 3 | Status |
|----|---------|-------|---------|---------|---------|--------|
| 1  | Abstract | framing | ✓ softened | — | — | Closed |
| 2  | §2.2 | proxy | ✓ Appendix added | — | — | Closed |
```

Update this table each round:
- **Closed**: the referee confirmed the issue was addressed
- **Carried Forward**: the comment was not fully addressed; add the new round number
- **New**: a brand-new comment; assign the next available number

### Starting a New Revision Round

```text
Round N inputs:
  - prior response letter:   respond_{N-1}.tex
  - prior marked manuscript: rev_marked_v{M}.tex
  - prior comment file:      comments_{N-1}.md
  - referee comments:        comments_{N}.md
  - comment-tracking table:  COMMENT_TRACKING.md

Round N outputs:
  - respond_{N}.tex
  - rev_marked_v{M}.tex (if unchanged) OR rev_marked_v{M+1}.tex (if changed)
  - COMMENT_TRACKING.md (updated)
  - history/round{N-1}/ (archived prior-round files)
```