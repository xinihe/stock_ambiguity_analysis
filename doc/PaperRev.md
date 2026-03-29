---
name: "PaperRev"
description: "Builds a referee-response package from manuscript comments to revised TeX and PDF outputs. Invoke when revising an academic paper and preparing a response letter."
---

# PaperRev

## Purpose

PaperRev is a detailed workflow for revising an academic manuscript from referee or associate editor comments, drafting a point-by-point response letter, implementing marked changes in the manuscript, and recompiling the deliverables into PDF files.

This workflow is designed for the revision pattern used in this project:

- Input manuscript: `geopoliticalAmb03.tex`
- Comment file: `comments3.md`
- Response letter output: `respond_3.tex`
- Marked revision output: `rev_marked_v2.tex`

The same workflow can be reused for later rounds by changing the source manuscript, comment file, and output filenames.

## When to Invoke

Invoke this workflow when:

- a journal, referee, reviewer, or associate editor provides revision comments
- an original or prior-round manuscript exists in LaTeX
- a detailed response letter must be drafted comment by comment
- the revised manuscript must visibly mark changes in red
- the response letter and marked manuscript must both be compiled to PDF
- consistency between the response letter and the revised manuscript must be verified before submission

Do not invoke this workflow for a simple proofreading-only task that does not require a response letter, marked revision, or comment-by-comment tracking.

## Required Inputs

At minimum, collect the following:

1. Original or prior-round manuscript `.tex`
2. Referee or associate editor comments file
3. Bibliography file(s), style file(s), and class file(s) needed for compilation
4. Target filenames for:
   - response letter `.tex`
   - revised marked manuscript `.tex`
   - output PDF files
5. Any prior response letters or revision history that help preserve continuity

For this project, the core inputs are:

- Manuscript: `/Users/tlxy/Library/Mobile Documents/com~apple~CloudDocs/Research/Projects/Ambiguity/stock_ambiguity_analysis/code/Adaptive/GeoUncertainty/outputs/results/geopoliticalAmb03.tex`
- Comments: `/Users/tlxy/Library/Mobile Documents/com~apple~CloudDocs/Research/Projects/Ambiguity/stock_ambiguity_analysis/code/Adaptive/GeoUncertainty/outputs/results/resubmission/comments3.md`
- Response letter target: `/Users/tlxy/Library/Mobile Documents/com~apple~CloudDocs/Research/Projects/Ambiguity/stock_ambiguity_analysis/code/Adaptive/GeoUncertainty/outputs/results/resubmission/respond_3.tex`
- Marked revision target: `/Users/tlxy/Library/Mobile Documents/com~apple~CloudDocs/Research/Projects/Ambiguity/stock_ambiguity_analysis/code/Adaptive/GeoUncertainty/outputs/results/resubmission/rev_marked_v2.tex`

## Expected Outputs

This workflow must produce:

1. A complete response letter in TeX
2. A marked revision manuscript in TeX
3. PDFs for both files
4. Internal alignment between:
   - each comment
   - each claimed response
   - each manuscript revision
   - each highlighted insertion in the revised manuscript

## Non-Negotiable Rules

Follow these rules throughout the revision process:

1. Every substantive comment must map to at least one explicit manuscript action.
2. Every manuscript action mentioned in the response letter must exist in the revised manuscript.
3. Do not make claims beyond the evidence. If the evidence is associational, do not present it as causal.
4. If a comment asks for moderation of claims, soften wording in both the manuscript and the response letter.
5. If a comment asks for literature positioning, add new comparative discussion without duplicating text already present.
6. If a table or statistic is quoted in the response letter, verify it against the manuscript before finalizing.
7. Mark manuscript revisions in red.
8. In the response letter, keep the response narrative readable and clearly separated from the quoted revised text.
9. Recompile after the edits are complete.
10. Clean auxiliary files before handoff unless they are intentionally retained.

## End-to-End Workflow

### Step 1: Read and Decompose the Comments

Read the full comments file and split it into discrete numbered issues.

For each issue, create a working record with:

- comment number
- short title
- exact comment text
- type of request
- affected manuscript location
- required response strategy

Recommended request types:

- interpretive moderation
- literature expansion
- methodological clarification
- empirical robustness
- table simplification
- wording and style cleanup

### Step 2: Audit the Manuscript Against Each Comment

Open the source manuscript and locate every passage relevant to each comment.

For each comment, identify:

- where the current wording creates the problem
- what revision is needed
- whether the issue appears in more than one section
- whether a new section, appendix, table, or footnote is needed

Build a revision map like this:

| Comment | Problem | Manuscript location | Revision action |
|---|---|---|---|
| 1 | ambiguity framed as dominant | Abstract, Introduction, Literature Review, Conclusion | soften and present ambiguity as complementary to volatility |
| 2 | insufficient comparison to alternative ambiguity proxies | Section 2.2 and new appendix | add comparative discussion and Appendix B |
| 3 | magnitude interpretation too strong | Introduction, results, conclusion | remove annualized claims and moderate portfolio language |

### Step 3: Draft the Response Strategy Before Editing

For each comment, decide the response structure before touching the text:

1. Acknowledge the concern directly
2. State agreement or partial agreement
3. Explain the conceptual issue
4. State what was changed
5. Identify manuscript sections changed
6. Quote the revised text

This prevents vague responses and keeps the response letter tied to actual edits.

### Step 4: Create the Response Letter Skeleton

Start the response letter with:

- title
- opening gratitude paragraph
- one subsection per comment
- the full quoted comment
- a prose response block
- a manuscript-revision block showing the new language

The response letter should separate two different functions:

1. The explanatory response
2. The literal revised text inserted into the manuscript

That separation is essential because the response letter must explain the reasoning, not only show the replacement text.

### Step 5: Create the Marked Revision Skeleton

Duplicate the source manuscript into a new marked-revision file.

Then:

1. Preserve the original document class, packages, bibliography, and structure
2. Add `xcolor` if not already present
3. Apply revisions directly in the manuscript
4. Highlight each inserted or revised phrase in red
5. Avoid introducing layout-breaking macros unless necessary

For manuscript highlighting, the standard approach is:

```tex
\textcolor{red}{revised text here}
```

For longer revised passages inside environments that may break or that are copied into the response letter, prefer a declarative color scope when necessary:

```tex
\color{red}\itshape revised quoted text here
```

### Step 6: Revise the Manuscript Comment by Comment

Work sequentially through the comments. For each one:

1. Edit the manuscript first
2. Confirm the revised wording compiles logically in context
3. Copy the final revised text into the response letter
4. Explain where the change was made and why

Never draft a response that promises a revision that has not yet been implemented.

### Step 7: Cross-Check Response Letter Against Manuscript

Before compiling, verify:

- every quoted revision in `respond_3.tex` appears in `rev_marked_v2.tex`
- section references are accurate
- table numbers match
- terminology is consistent
- the addressee label is correct across the letter

### Step 8: Compile Both Files

Compile:

- the response letter
- the revised marked manuscript

If bibliography changes are involved, use the full LaTeX cycle:

```bash
pdflatex file.tex
bibtex file
pdflatex file.tex
pdflatex file.tex
```

If the `.bbl` is already available and unchanged, a direct `pdflatex` pass may be enough.

### Step 9: Visual Quality Control

Inspect the PDFs for:

- page breaks inside response blocks
- red text continuity across pages
- spacing between paragraphs
- overfull boxes severe enough to require manual fixes
- broken quotes, braces, or environments
- tables that extend outside the page

### Step 10: Cleanup and Delivery

Remove temporary scripts and auxiliary compilation files unless intentionally needed:

- `.aux`
- `.log`
- `.out`
- `.spl`
- `.blg`
- `.fls`
- `.xml`
- scratch helper scripts used only for one-off replacement tasks

Deliver the final `.tex` and `.pdf` outputs together.

## Comment-by-Comment Execution Pattern

Use the following mini-template for every comment.

### Comment Record Template

```text
Comment N: [short title]
Comment text:
[exact quoted comment]

Diagnosis:
- what the referee is concerned about
- why the current draft is vulnerable

Response plan:
- what stance to take
- what evidence or clarification to add
- what exact manuscript sections to revise

Manuscript actions:
- section(s) changed
- table(s) changed
- appendix added if needed

Response-letter content:
- explanation paragraph(s)
- quoted revised text

Validation:
- confirm each claimed change exists in the manuscript
- confirm numbers and labels are consistent
```

## Project-Specific Mapping for the Current Revision Round

This section records how the current comment set should be handled one by one.

### Comment 1: Ambiguity and Volatility Language

**Concern**

The comment says the manuscript still implicitly frames ambiguity as dominating volatility.

**Required editorial stance**

Present ambiguity and volatility as complementary uncertainty channels rather than rival mechanisms.

**Manuscript actions**

- revise Abstract
- revise Introduction contribution statements
- revise literature framing if it implies hierarchy
- revise conclusion language

**Revision logic**

- remove phrases such as `primary channel`, `dominant`, `more than volatility`, `inflating volatility's role by 47%`
- replace them with language about joint explanatory power, complementary channels, and distinct dimensions of uncertainty

**Response-letter logic**

- explain why the old phrasing implied a zero-sum comparison
- explain that uncertainty in Knight's sense and risk are conceptually distinct
- show the revised wording

### Comment 2: Alternative Ambiguity Proxies

**Concern**

The comment requests stronger empirical justification for the cross-entropy ambiguity measure through comparison with alternative proxies.

**Required editorial stance**

Position the proposed measure relative to forecast dispersion and variance-of-variance without repeating the manuscript's pre-existing discussion.

**Manuscript actions**

- expand Section 2.2
- add Appendix B
- explain why the cross-entropy measure captures model uncertainty rather than only second-moment risk

**Revision logic**

- contrast analyst-based forecast dispersion with market-wide high-frequency price data
- contrast variance-of-variance with true uncertainty in Knight's sense
- stress market coverage, frequency, objectivity, and theoretical grounding

**Response-letter logic**

- explain why the comparison matters
- state that Appendix B was added
- quote both the new literature-review sentence and the appendix framing

### Comment 3: Economic Magnitudes

**Concern**

The comment says annualized interpretations and portfolio implications are too strong.

**Required editorial stance**

Moderate the empirical claims and keep interpretation at the daily or associational level.

**Manuscript actions**

- remove annualized effects
- revise portfolio implication language
- soften conclusion language

**Revision logic**

- use daily basis-point language
- avoid deterministic portfolio claims
- explicitly state that the analysis is associational

### Comment 4: Mediation Limitations

**Concern**

The Baron-Kenny mediation framework needs clearer identification caveats in asset-pricing data.

**Required editorial stance**

Present mediation as statistical decomposition rather than causal proof.

**Manuscript actions**

- add an explicit limitations paragraph in the mediation section

**Revision logic**

- mention simultaneity
- mention contemporaneous omitted shocks
- explain that isolated transmission paths are difficult to identify in daily financial data

### Comment 5: IV Strategy and Exclusion Restriction

**Concern**

The comment says the exclusion restriction for the Non-Asian GPR instrument needs stronger discussion.

**Required editorial stance**

Acknowledge the instrument's usefulness while explicitly recognizing alternative channels.

**Manuscript actions**

- revise IV robustness discussion

**Revision logic**

- mention commodity-price channels
- mention safe-haven currency flows
- mention global liquidity or credit conditions
- caution interpretation

### Comment 6: Timing Assumptions

**Concern**

The comment asks for clearer timing alignment and possibly added robustness.

**Required editorial stance**

Clarify the daily timing convention and add a lagged robustness specification.

**Manuscript actions**

- add timestamp explanation
- explain contemporaneous `(t, t)` alignment
- add lagged `(t-1, t)` robustness check
- summarize new results in the robustness table if relevant

### Comment 7: Broader Literature Contribution

**Concern**

The contribution to uncertainty, ambiguity, and GPR literatures needs clearer positioning.

**Required editorial stance**

Explain precisely how the paper bridges macro uncertainty and ambiguity pricing.

**Manuscript actions**

- revise Introduction
- revise Section 2.2
- reinforce connection in Appendix B if used

**Revision logic**

- say prior work often treats uncertainty as monolithic
- state that the paper decomposes uncertainty into risk and ambiguity
- show what is genuinely new

### Comment 8: Regression Models and Controls

**Concern**

The controls discussion is too verbose.

**Required editorial stance**

Streamline rather than expand.

**Manuscript actions**

- condense the control-variable discussion
- remove distracting justification where possible

**Revision logic**

- keep only the essential role of each control
- improve readability without dropping methodological clarity

### Comment 9: Simplification of Empirical Results

**Concern**

The results presentation is too dense.

**Required editorial stance**

Create a summary table and simplify the narrative.

**Manuscript actions**

- add a summary table of key coefficients
- ensure the response letter reports the correct table number and values

**Revision logic**

- use a compact cross-specification summary
- verify all coefficients are sourced from the manuscript

### Comment 10: Grammar and Style

**Concern**

Minor inconsistencies remain.

**Required editorial stance**

State that a full proofread was conducted and standardize terminology.

**Manuscript actions**

- clean grammar
- standardize recurring terms

**Revision logic**

- identify inconsistent terminology such as `entropy-based ambiguity` versus `cross-entropy-based ambiguity`
- ensure the response does not overstate invisible edits that cannot be verified

## Response Letter Template

Use the following template for a response letter that allows highlighted blocks to break across pages while preserving readability.

```tex
\documentclass[12pt]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{xcolor}
\usepackage{geometry}
\usepackage{hyperref}
\usepackage[skins,breakable]{tcolorbox}

\geometry{a4paper, margin=1in}

\definecolor{highlight}{RGB}{255,255,200}
\definecolor{darkblue}{RGB}{0,0,139}

\hypersetup{
    colorlinks=true,
    linkcolor=darkblue,
    urlcolor=darkblue
}

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

\title{Response to Associate Editor Comments}
\author{}
\date{}

\begin{document}
\maketitle

\section*{Response to Associate Editor Comments}

We sincerely thank the associate editor for the careful reading of our revised manuscript. Below we provide a point-by-point response and indicate the exact changes implemented in the manuscript.

\subsection*{Comment 1: [short title]}
\textit{"[paste exact comment here]"}

\begin{highlightbox}
\textbf{Response:} [write the explanatory response here]
\end{highlightbox}

\begin{highlightbox}
\textbf{Revisions in the Manuscript:}

In the [section name], we revised the text to state:
\vspace{0.2cm}
\begin{quote}
\color{red}\itshape "[paste revised text here]"
\end{quote}
\end{highlightbox}

\end{document}
```

## Response Letter Formatting Notes

Use these formatting rules:

1. Quote the exact comment verbatim.
2. Use one subsection per comment.
3. Keep prose explanation inside a breakable `highlightbox`.
4. Keep revised manuscript excerpts in a separate `highlightbox`.
5. For long quoted revisions inside the response letter, use:

```tex
\begin{quote}
\color{red}\itshape "..."
\end{quote}
```

This is safer across broken pages than wrapping the entire passage in `\textcolor{red}{...}`.

6. Keep editorial response text in black.
7. Keep quoted manuscript revisions in red.
8. If the document is addressed to an associate editor rather than a reviewer, use that label consistently throughout.

## Marked Revision Template

Use the manuscript itself as the template. Duplicate the source file and mark only the revised text in red.

```tex
\documentclass[preprint,12pt,authoryear]{elsarticle}

\usepackage{amssymb}
\usepackage{amsmath}
\usepackage{mathrsfs}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{CJKutf8}
\usepackage{booktabs}
\usepackage{threeparttable}
\usepackage{graphicx}
\usepackage{subcaption}
\usepackage{rotating}
\usepackage{tabularx}
\usepackage{array}
\usepackage{float}
\usepackage{xcolor}
\usepackage{hyperref}

\begin{document}

\begin{frontmatter}
\title{[paper title]}
\begin{abstract}
Original sentence. \textcolor{red}{Inserted revised sentence in red.}
\end{abstract}
\end{frontmatter}

\section{Introduction}
Original text. \textcolor{red}{Revised text in red.}

\end{document}
```

## Marked Revision Rules

Apply these rules in the revised manuscript:

1. Prefer minimal red insertions rather than repainting unchanged material.
2. If replacing a phrase, preserve the surrounding sentence unless a full rewrite is cleaner.
3. Keep section numbering, labels, citations, and table labels intact unless the comment requires structural change.
4. If a new appendix is added, reference it from the main text where relevant.
5. If a new robustness table is added, ensure the response letter cites the correct table number.
6. Do not allow the response letter to mention a section or table number that differs from the actual manuscript.

## Synchronization Checklist

Before compiling, verify all of the following:

- each comment has a response subsection
- each response subsection has at least one concrete manuscript action
- each quoted revised sentence appears in the manuscript
- section references are correct
- table numbers are correct
- coefficients and statistics match the manuscript exactly
- terminology is consistent
- references to reviewer versus associate editor are consistent
- any new appendix is referenced in both manuscript and response letter

## Compilation Procedure

### Response Letter

From the output directory:

```bash
pdflatex -interaction=nonstopmode respond_3.tex
```

### Marked Revision

If bibliography files need refreshing:

```bash
pdflatex -interaction=nonstopmode rev_marked_v2.tex
bibtex rev_marked_v2
pdflatex -interaction=nonstopmode rev_marked_v2.tex
pdflatex -interaction=nonstopmode rev_marked_v2.tex
```

If the bibliography output is already present and unchanged, a direct `pdflatex` pass is often enough.

### Cleanup

After successful compilation:

```bash
find . -maxdepth 1 -type f \( -name "*.aux" -o -name "*.log" -o -name "*.out" -o -name "*.spl" -o -name "*.blg" -o -name "*.fls" -o -name "*.xml" -o -name "*~" \) -delete
```

## Failure Modes and Fixes

### 1. Response Letter Quotes Lose Red Color on the Next Page

**Cause**

Using `\textcolor{red}{...}` or nested grouped formatting inside long quote blocks that cross a page break.

**Fix**

Use:

```tex
\begin{quote}
\color{red}\itshape "..."
\end{quote}
```

instead of grouping the entire paragraph inside `\textcolor{red}{...}`.

### 2. Highlight Boxes Leave Large Blank Areas

**Cause**

A non-breakable environment is wrapping the response text.

**Fix**

Use breakable `tcolorbox` blocks:

```tex
\usepackage[skins,breakable]{tcolorbox}
```

and define a breakable box.

### 3. Paragraphs Inside Response Blocks Are Too Dense

**Cause**

No paragraph spacing is set inside the box.

**Fix**

Set:

```tex
before upper={\parskip=0.5\baselineskip}
```

in the `tcolorbox` definition.

### 4. Table Numbers Differ Between Response Letter and Manuscript

**Cause**

The response letter was drafted before the manuscript numbering stabilized.

**Fix**

Always verify the final table number in the revised manuscript before finalizing the response letter.

### 5. The Response Claims a Revision That Does Not Exist

**Cause**

The response was drafted ahead of implementation.

**Fix**

Always implement first, then quote the exact revised text from the manuscript into the response letter.

## Quality-Control Checklist Before Submission

Use this final checklist:

- manuscript tone is balanced and not overstated
- literature additions are new and not repetitive
- empirical caveats are explicit where needed
- new appendix or table is introduced cleanly
- response letter is respectful, direct, and specific
- all revised quotations are present in the manuscript
- both PDFs compile successfully
- temporary helper files are removed

## Deliverables for This Workflow

The final deliverables are:

- `respond_3.tex` and `respond_3.pdf`
- `rev_marked_v2.tex` and `rev_marked_v2.pdf`
- this workflow file: `PaperRev.md`

---

## Multi-Round Journal Revision Workflow

When a journal returns a manuscript after a revision round, use the following system to track progress across multiple rounds.

### Version Naming Convention

Use a round number suffix for every output file. The convention is:

| Round | Comment file | Response letter | Marked manuscript | Manuscript PDF |
|-------|-------------|-----------------|-------------------|----------------|
| 1 | `comments1.md` | `respond_1.tex` | `rev_marked_v1.tex` | `rev_marked_v1.pdf` |
| 2 | `comments2.md` | `respond_2.tex` | `rev_marked_v2.tex` | `rev_marked_v2.pdf` |
| 3 | `comments3.md` | `respond_3.tex` | `rev_marked_v2.tex` | `rev_marked_v2.pdf` |

The manuscript filename only increments when the base manuscript changes (e.g., after a reject-and-resubmit). The response letter and comment file always increment per round, even if the manuscript base stays the same.

### Round-Tracking Comment Table

Maintain a master comment table across all rounds. Add a new column for each round.

```
| C# | Section | Issue | Round 1 | Round 2 | Round 3 | Status |
|----|---------|-------|---------|---------|---------|--------|
| 1  | Abstract | "dominant" framing | ✓ softened | — | — | Closed |
| 2  | §2.2 | proxy comparison | ✓ Appendix B added | — | — | Closed |
| 3  | §3.2.1 | annualized claims | ✓ removed | — | — | Closed |
```

Update this table each round:

- **Open** → **Closed**: the referee confirmed the issue was addressed
- **Open** → **Carried Forward**: the comment was not fully addressed or was only partially addressed; add the new round number
- **New**: a brand-new comment that did not appear in prior rounds; assign the next available number

### Per-Round Workflow

For each new revision round:

1. **Before starting** — increment the comment file: `comments2.md` → `comments3.md`, `comments3.md` → `comments4.md`
2. **Copy the prior response letter** to the new round: `respond_2.tex` → `respond_3.tex` and update the round number in the title
3. **Copy the prior tracking table** and mark each prior issue as either Closed or Carried Forward
4. **Audit the new referee comments** against the tracking table
5. **Identify which issues are carried forward** versus genuinely new
6. **Revise the manuscript** for carried-forward and new issues, incrementing the revision version only if the base manuscript has changed
7. **Add new response sections** to the new response letter
8. **Close or carry each row** in the tracking table
9. **Compile and verify** both the response letter and the revised manuscript
10. **Archive** the old `comments`, `respond`, and `rev` files in a `history/round1/` subdirectory

### Starting a New Revision Round

```
Round N inputs:
  - prior response letter:   respond_{N-1}.tex
  - prior marked manuscript: rev_marked_v{M}.tex  (M = last manuscript version)
  - prior comment file:      comments_{N-1}.md
  - referee comments:        comments_{N}.md  (new file from journal)
  - comment-tracking table:   COMMENT_TRACKING.md

Round N outputs:
  - respond_{N}.tex  (response to round N comments only)
  - rev_marked_v{M}.tex  (if manuscript base unchanged) OR rev_marked_v{M+1}.tex  (if manuscript base changed)
  - COMMENT_TRACKING.md  (updated)
  - history/round{N-1}/  (archived prior-round files)
```

### Keeping the Comment-Tracking Table

Create a `COMMENT_TRACKING.md` file in the resubmission directory. It lives there throughout all rounds.

Example:

```markdown
# Comment Tracking Across Revision Rounds

| C# | Section | Issue | Round 1 | Round 2 | Round 3 | Status |
|----|---------|-------|---------|---------|---------|--------|
| 1 | Abstract | ambiguity framed as dominant | softened in Abstract | — | — | Closed |
| 2 | §2.2 | proxy comparison insufficient | added Appendix B | — | — | Closed |
| 3 | §3.2.1 | annualized claims too strong | removed | — | — | Closed |
| 4 | §3.2.2 | mediation limitations unclear | added caveat para | — | — | Closed |
| 5 | §3.2.4 | IV exclusion restriction | expanded discussion | — | — | Closed |
| 6 | §3.1.4 | timing assumptions vague | clarified + lagged spec | — | — | Closed |
| 7 | §1 + §2.2 | contribution positioning | revised framing | — | — | Closed |
| 8 | §3.1.3 | controls too verbose | streamlined | — | — | Closed |
| 9 | §3.2.4 | results too dense | added Table 6 | — | — | Closed |
|10 | all | grammar inconsistencies | proofread | — | — | Closed |
```

When a new round arrives, copy the table, update the Round N column, and change the Status column.

### Closing a Round

A comment is **Closed** when the referee does not raise it again. A comment that is raised again becomes **Carried Forward** with the new round noted.

### Archive Structure

After completing Round N, archive the prior round's files:

```
resubmission/
  history/
    round1/
      comments1.md
      respond_1.tex
      rev_marked_v1.tex
    round2/
      comments2.md
      respond_2.tex
      rev_marked_v2.tex
  COMMENT_TRACKING.md   ← always kept at resubmission root
  comments3.md          ← current round
  respond_3.tex         ← current round
  rev_marked_v2.tex     ← current manuscript base
```

This structure makes it easy to reconstruct what changed between rounds and what was agreed in prior rounds.

This file is the canonical detailed workflow reference for future paper-revision rounds in this repository.
