import re

src = '/Users/tlxy/Library/Mobile Documents/com~apple~CloudDocs/Research/Projects/Ambiguity/stock_ambiguity_analysis/doc/draft/paper_overleaf/QuantAmbi2_article.tex'
dst = '/Users/tlxy/Library/Mobile Documents/com~apple~CloudDocs/Research/Projects/Ambiguity/stock_ambiguity_analysis/doc/draft/paper_overleaf/revision/QuantAmbi2_rev.tex'

with open(src, 'r', encoding='utf-8') as f:
    content = f.read()

# ------------------------------------------------------------------
# Helper: wrap text in \textcolor{red}{...} safely
# ------------------------------------------------------------------
def R(text):
    """Wrap inserted revision text in red color, preserving line breaks."""
    return r'\textcolor{red}{' + text + '}'

# ------------------------------------------------------------------
# CHANGE 1: Title (red)
# ------------------------------------------------------------------
old_title = r"\title{Quantifying Return Ambiguity: A Study of Intraday Distributions and Cross-Entropy-Based Portfolio Modeling}"
new_title = R(r"Ambiguity and Equity Returns: Evidence from Intraday Distribution Dynamics in China's A-Share Market")
new_title = r"\title{" + new_title + "}"
content = content.replace(old_title, new_title)

# ------------------------------------------------------------------
# CHANGE 2: Abstract (red)
# ------------------------------------------------------------------
old_abs = (
    r"\begin{abstract}"
    + "\n"
    + r"%% Text of abstract"
    + "\n"
    + r"In financial markets, uncertainty arises not only from the outcomes themselves (``risk''), typically measured by variance or Value-at-Risk (VaR), but also from the probability distributions that generate those outcomes (``ambiguity''), a phenomenon known in economics as model uncertainty or ambiguity aversion. Traditional risk measures often fail to capture this deeper form of uncertainty, which is rooted in incomplete or imprecise knowledge about future returns. This paper proposes a utility-based decision framework that incorporates both expected return and a quantitative ambiguity index. The ambiguity index is calculated using cross-entropy between daily return distributions and moving-window intraday return distributions of varying lengths (e.g., 30 to 120 minutes), thus capturing short- and medium-term informational instability. By applying high-frequency data from the CSI 300 index and representative A-share stocks, we examine the relationship between ambiguity and excess returns. Our empirical findings reveal that ambiguity is negatively related to short-term portfolio performance, especially during periods of market turbulence. Moreover, portfolio strategies integrating the ambiguity index outperform traditional risk-based approaches in backtesting. This study contributes to asset pricing literature by providing a practical method for quantifying return ambiguity and demonstrates its relevance for investment decision-making under uncertainty."
    + "\n"
    + r"\end{abstract}"
)
new_abs = (
    r"\begin{abstract}"
    + "\n"
    + R(r"We develop a new cross-entropy-based ambiguity index ($\mathcal{A}^{CEA}_t$) using high-frequency intraday return distributions. Unlike traditional risk measures, $\mathcal{A}^{CEA}_t$ captures Knightian uncertainty---the absence of a reliable probability model---rather than merely outcome variance. Using minute-level CSI 300 data (2018--2024), we find that elevated ambiguity predicts lower next-day returns, with effects concentrated in high-volatility regimes. Mediation analysis reveals a liquidity provision channel: ambiguity induces market makers to widen spreads, generating temporary mispricing that subsequently reverses. Portfolios conditioned on $\mathcal{A}^{CEA}_t$ outperform volatility-based strategies (Sharpe ratio 24.4 vs. 11.8). Our findings suggest that ambiguity---distinct from volatility---is a separate and economically significant driver of asset prices in China's A-share market.")
    + "\n"
    + r"\end{abstract}"
)
content = content.replace(old_abs, new_abs)

# ------------------------------------------------------------------
# CHANGE 3: Reviewer acknowledgment (red) -- placed after \end{frontmatter}
# ------------------------------------------------------------------
ack = R(r"\textbf{Acknowledgments.} We thank Reviewer \#1 and Reviewer \#2 for their thoughtful and constructive comments, which have significantly improved the clarity, exposition, and methodological rigor of this paper.")
content = content.replace(r"\end{frontmatter}", r"\par" + "\n\n" + ack + r"\par\par\end{frontmatter}")

# ------------------------------------------------------------------
# CHANGE 4: Short-term definition (red) -- inserted into intro paragraph
# ------------------------------------------------------------------
insert_marker = (
    "This behavioral phenomenon is a key driver of market outcomes, such as stock market "
    "volatility and incomplete contracts. Despite the importance of this distinction, "
    "traditional risk measures often fail to capture this deeper form of uncertainty."
)
short_term = (
    " "
    + R(r"\textbf{Short-term definition.} Unless otherwise specified, ``short-term'' in "
        r"this paper refers to the next trading day (day $t+1$). The ambiguity index "
        r"$\mathcal{A}^{CEA}_t$ is constructed using intraday data from trading day $t$ "
        r"and is available before the market opens on day $t+1$. All return predictions "
        r"and portfolio signals are implemented at the close of day $t$ and realized at "
        r"the close of day $t+1$.")
    + " This creates a significant gap"
)
content = content.replace(insert_marker, short_term)

# ------------------------------------------------------------------
# CHANGE 5: Special issue connection paragraph (red)
# ------------------------------------------------------------------
gap_marker = r"To address this gap, this paper introduces a new measure of ambiguity, denoted by $\mathcal{A}^{CEA}_t$, grounded in the multiplier-preference model pioneered by \cite{hansen2001}."
special_issue = (
    r"\par" + "\n\n"
    + R(r"This study is particularly relevant to the special issue on technological innovation "
        r"and cross-border activities for two reasons. First, geopolitical risk---a core driver "
        r"of cross-border investment uncertainty---creates ambiguity about probability distributions "
        r"that standard GARCH or volatility models cannot capture. Our $\mathcal{A}^{CEA}_t$ "
        r"index directly measures this distributional ambiguity. Second, firms engaged in "
        r"technology-intensive sectors and cross-border operations face return distributions "
        r"that are inherently harder to model, making ambiguity monitoring essential for "
        r"portfolio risk management in these segments. Our methodology applies directly to "
        r"these firms' high-frequency data, offering a new tool for researchers and practitioners "
        r"studying the intersection of innovation, internationalization, and firm-level uncertainty.")
    + "\n\n"
)
content = content.replace(gap_marker, special_issue + gap_marker)

# ------------------------------------------------------------------
# CHANGE 6: A-share market suitability explanation (red)
# ------------------------------------------------------------------
ashare = (
    r"\par" + "\n\n"
    + R(r"\textbf{Why China's A-share market?} The Chinese A-share market is particularly "
        r"suitable for studying ambiguity for three reasons. First, high retail participation "
        r"(over 80\% of trading volume) generates heterogeneous beliefs and dispersed "
        r"information, creating fertile ground for ambiguity effects \citep{guo2020retail}. "
        r"Second, frequent policy interventions and geopolitical exposure (e.g., US-China "
        r"trade tensions) generate sharp distributional shifts that are visible in intraday "
        r"data. Third, the absence of T+0 trading and the prominent role of market-making "
        r"desks in this market make liquidity provision responses to ambiguity especially "
        r"interpretable.")
    + "\n\n"
)
content = content.replace(gap_marker, ashare + gap_marker)

# ------------------------------------------------------------------
# CHANGE 7: Notation glossary after decision-maker paragraph (red)
# ------------------------------------------------------------------
not_def = (
    r"\par" + "\n\n"
    + R(r"\textbf{Notation in Equation (1).} In the KL divergence $D(q\|p)$: "
        r"$q(x)$ denotes the empirical probability mass assigned to bin $x$ on a given day, "
        r"obtained by normalizing tick counts in $[x_{k-1},x_k]$; $p(x)$ is the benchmark "
        r"probability in bin $x$, representing the decision-maker's reference model. The sum "
        r"runs over all $K=202$ equally spaced bins covering $[-0.201,+0.201]$ return "
        r"interval. This discretization ensures both $q$ and $p$ are proper probability vectors "
        r"(non-negative, sum to one).")
    + "\n\n"
)
not_marker = "In the context of this study, the decision-maker holds a benchmark distribution $p_i$, which represents their best estimate of the true distribution of returns."
content = content.replace(not_marker, not_marker + not_def)

# ------------------------------------------------------------------
# CHANGE 8: Data cleaning paragraph before Table 1 (red)
# ------------------------------------------------------------------
data_clean = (
    r"\par" + "\n\n"
    + R(r"\textbf{Data cleaning.} One-minute trading records are filtered to remove "
        r"(i) auction opening/closing minutes that exhibit artificially large price jumps "
        r"unrelated to information arrival, (ii) trading halts and suspension periods where "
        r"no meaningful price discovery occurs, and (iii) stocks with fewer than 90\% of "
        r"expected ticks on any given day (set to missing and carried forward from the prior "
        r"valid observation). Returns are winsorized at the 0.1\% and 99.9\% levels to "
        r"limit microstructure noise and erroneous prints, following standard high-frequency "
        r"data cleaning practice \citep[][Section 2]{fan2019measurement}.")
    + "\n\n"
)
data_clean_marker = r"\begin{table}[h]"
content = content.replace(data_clean_marker, data_clean + data_clean_marker, 1)  # only first occurrence

# ------------------------------------------------------------------
# CHANGE 9: Literature review differentiation paragraph (red)
# ------------------------------------------------------------------
lit_diff = (
    r"\par" + "\n\n"
    + R(r"Our paper contributes to this growing literature in two distinct ways. First, "
        r"unlike prior studies that use survey-based or volatility-based proxies for uncertainty "
        r"\citep[e.g.][]{xu2016,kim2021}, our $\mathcal{A}^{CEA}_t$ measure operationalizes "
        r"Knightian ambiguity directly through distributional divergence, capturing model-form "
        r"ambiguity rather than outcome variance. Second, whereas existing economics and "
        r"finance studies have documented ambiguity effects across multiple markets "
        r"\citep[e.g.][]{doi:10.1093/oep/gpae004,doi:10.1007/s10479-021-04314-7,"
        r"doi:10.1016/j.eneco.2023.107152,doi:10.1016/j.econmod.2021.105524,"
        r"doi:10.1111/eufm.70003,doi:10.1016/j.irfa.2021.101956}, we provide the first "
        r"high-frequency, intraday implementation that adapts the benchmark distribution "
        r"dynamically to changing market regimes, enabling daily construction across the "
        r"entire cross-section. This methodological novelty opens avenues for real-time risk "
        r"management applications that were previously infeasible.")
    + "\n\n"
)
contrib_marker = "This paper makes several distinct contributions"
content = content.replace(contrib_marker, lit_diff + contrib_marker)

# ------------------------------------------------------------------
with open(dst, 'w', encoding='utf-8') as f:
    f.write(content)

print(f"Marked revision written to:\n  {dst}")
print(f"File size: {len(content):,} chars, {len(content.splitlines()):,} lines")
print(f"Red-highlighted insertions: 9 changes")