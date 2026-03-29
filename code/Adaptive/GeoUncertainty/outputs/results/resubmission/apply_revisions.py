import re

with open('code/Adaptive/GeoUncertainty/outputs/results/resubmission/revise_unmarked_article.tex', 'r') as f:
    content = f.read()

# 1. Abstract
old_abs = r"When ambiguity is included alongside traditional risk, models explain more variation in returns and the role of volatility-based risk is often reduced."
new_abs = r"\textcolor{red}{When ambiguity is included alongside traditional risk, models explain more variation in returns, illustrating that ambiguity captures a distinct dimension of Knightian uncertainty without invalidating the role of traditional risk. Rather than competing for explanatory supremacy, volatility and ambiguity function as dual, coexisting channels that jointly capture investor distress during geopolitical shocks.}"
content = content.replace(old_abs, new_abs)

# 2. Introduction
old_intro1 = r"(iii) models ignoring ambiguity misattribute ambiguity's impact to volatility, inflating volatility's role by 47\%."
new_intro1 = r"\textcolor{red}{(iii) models ignoring ambiguity misattribute a portion of ambiguity's impact to volatility, illustrating the complementary nature of these two uncertainty channels.}"
content = content.replace(old_intro1, new_intro1)

# 3. Alternative Proxies
old_proxies = r"However, these proxies can be indirect and may not be available at a high frequency."
new_proxies = r"\textcolor{red}{Unlike forecast dispersion, which heavily relies on subjective analyst coverage and biases samples toward large firms, or variance-of-variance, which fundamentally captures second-moment uncertainty and is sensitive to high-frequency measurement errors, our cross-entropy measure directly quantifies the divergence from historical reference models. By doing so, it captures the essence of Knightian uncertainty---where the probability distributions themselves are unknown---and can be constructed daily for the entire universe of liquid stocks.}"
content = content.replace(old_proxies, new_proxies)

# 4. Economic Magnitudes (Intro)
old_intro2 = r"\textcolor{black}{Our findings have immediate practical implications. Portfolio managers using volatility-only hedging strategies remain exposed to ambiguity risk. During high-GPR periods, minimum-variance portfolios underperform by 67 bps monthly relative to ambiguity-aware portfolios. Policy interventions focusing solely on volatility stabilization (e.g., circuit breakers) may miss the larger ambiguity transmission channel.}"
new_intro2 = r"\textcolor{red}{Our findings have practical implications. The daily associations suggest that portfolios optimized solely for minimum variance may leave investors exposed to uncompensated ambiguity risks during periods of heightened geopolitical tension, highlighting the importance of ambiguity-aware risk management. However, given the episodic and mean-reverting nature of geopolitical shocks, these associations should be interpreted cautiously as daily impacts rather than deterministic long-term outcomes.}"
content = content.replace(old_intro2, new_intro2)

# 4. Economic Magnitudes (Section 3.2.1)
content = content.replace(
    r"\textcolor{black}{Table \ref{tab:baseline_ts} reports the results. We observe a negative and significant relationship between ambiguity and returns. In terms of economic significance, a one-standard-deviation increase in ambiguity reduces returns by 11.3 bps daily---equivalent to 28.3\% annualized. Comparing magnitudes, ambiguity's impact is 1.4x larger than volatility's, despite having lower variance, underscoring why the ambiguity channel is non-negligible.}",
    r"\textcolor{black}{Table \ref{tab:baseline_ts} reports the results. We observe a negative and significant relationship between ambiguity and returns.} \textcolor{red}{In terms of economic significance, a one-standard-deviation increase in ambiguity is associated with a daily return reduction of 11.3 bps. We focus on this daily impact, which is more appropriate than annualized figures given the high-frequency nature of our data and the transient nature of the shocks we study.}"
)

# 5. Limitations of Mediation Analysis
old_med = r"which then influence end-of-day returns---supports this causal pathway.}"
new_med = r"which then influence end-of-day returns---supports this causal pathway.} \textcolor{red}{However, while the Baron-Kenny framework provides a useful statistical decomposition, we caution against strict causal interpretations. In high-frequency financial data, unobserved contemporaneous shocks (such as sudden shifts in global liquidity or simultaneous macroeconomic data releases) can affect both the mediator and the outcome simultaneously. This potential for unobserved confounding complicates the identification of isolated transmission paths, meaning our mediation results should be viewed as an associational decomposition rather than a definitive causal proof.}"
content = content.replace(old_med, new_med)

# 6. IV Strategy
old_iv = r"\textcolor{black}{ Econometric tests confirm the validity of our instrument: the first-stage $F$-statistic is 47.3"
new_iv = r"\textcolor{red}{While the 'Non-Asian GPR' instrument is temporally predetermined relative to the Chinese trading session, we acknowledge that global geopolitical shocks might influence Chinese markets through alternative contemporaneous channels, such as overnight shifts in global commodity prices or safe-haven currency flows. To the extent these channels are correlated with our instrument, the exclusion restriction may be partially violated, suggesting our IV estimates should be interpreted with appropriate caution.} \textcolor{black}{Econometric tests confirm the validity of our instrument: the first-stage $F$-statistic is 47.3"
content = content.replace(old_iv, new_iv)

# 7. Timing Assumptions
old_time = r"\textcolor{black}{We use a contemporaneous alignment $(t, t)$ for our main analysis. This is justified because the GPR news (arriving pre-open) has ample time to be incorporated into investor beliefs and trading behavior throughout the trading day, which is then reflected in the intraday-based ambiguity measure and the close-to-close returns. We verified this by testing $(t-1, t)$ specifications, which yielded qualitatively similar but slightly weaker results, supporting the view that the market reaction to geopolitical news is rapid. Robustness tests with alternative alignments confirm that our main findings are not driven by timing choices.}"
new_time = r"\textcolor{red}{We use a contemporaneous alignment $(t, t)$ for our main analysis. This is justified because the GPR news (compiled at 00:00 UTC, arriving between 5:00 AM and 8:00 AM Beijing time) has ample time to be incorporated into investor beliefs and trading behavior throughout the trading day before the 15:00 close. However, recognizing that information frictions or behavioral delays might cause market reactions to spill over, we also introduce a lagged specification $(t-1, t)$ in our robustness checks. This verifies that the negative pricing of ambiguity is robust and not merely an artifact of our specific contemporaneous timing assumptions.}"
content = content.replace(old_time, new_time)

# 8. Contribution to Broader Literature
old_lit = r"\textcolor{black}{Our work extends the emerging literature on uncertainty transmission in several critical ways. First, while \citet{Baker2016} and \citet{Caldara2022} document that uncertainty matters for markets, they treat uncertainty as monolithic. By decomposing uncertainty into risk (volatility) and ambiguity, we reveal that ambiguity accounts for the majority of GPR's pricing effect.}"
new_lit = r"\textcolor{red}{Our work extends the emerging literature on uncertainty transmission in several critical ways, bridging the gap between macroeconomic uncertainty and ambiguity pricing. First, while studies like \citet{Baker2016} and \citet{Caldara2022} document that uncertainty matters for markets, they often treat uncertainty as monolithic. By decomposing uncertainty into complementary risk (volatility) and ambiguity channels, we build on the insights of \citet{Brenner2018} and reveal that ambiguity is a distinct factor in GPR's pricing effect.}"
content = content.replace(old_lit, new_lit)

content = content.replace(r"Second, unlike traditional proxy-based ambiguity measures \citep{Brenner2018}, our entropy-based approach",
                          r"Second, unlike traditional proxy-based ambiguity measures \citep{Brenner2018}, our cross-entropy-based approach")

# 9. Regression Models and Control Variables
old_reg_pattern = r"In our time-series specifications, we incorporate a comprehensive set of control variables to isolate the impact of geopolitical ambiguity from other macroeconomic and market-wide fluctuations\. We control for the \\textbf\{Market Return \(\$MKT\$\)\}, defined as the value-weighted return of the entire A-share market, to capture systematic market movements\.\\footnote\{.*?\} To account for global risk sentiment, we include the \\textbf\{VIX Index\}, which proxies for international risk aversion\.\n\nDomestically, we control for the term structure of interest rates using the \\textbf\{Term Spread \(\$TERM\$\)\}—calculated as the difference between 10-year and 1-year Chinese government bond yields—to capture the monetary policy stance and economic growth expectations\. Additionally, we use the 3-month SHIBOR as the \\textbf\{Risk-Free Rate \(\$RF\$\)\}\."
new_reg = r"\textcolor{red}{In our time-series specifications, we incorporate a streamlined set of control variables to cleanly isolate the impact of geopolitical ambiguity. We control for the \textbf{Market Return ($MKT$)} to capture systematic market movements, the \textbf{VIX Index} for global risk sentiment, the \textbf{Term Spread ($TERM$)} for domestic monetary policy conditions, and the \textbf{Risk-Free Rate ($RF$)}. This concise set of controls ensures that the core focus remains on the interplay between geopolitical risk and ambiguity.}"
content = re.sub(old_reg_pattern, new_reg, content, flags=re.DOTALL)

# 10. Table 5 Simplification
table5 = r"""
\textcolor{red}{
\begin{table}[htbp]
\centering
\caption{Summary of Main Effects Across Model Specifications}
\label{tab:summary_effects}
\begin{threeparttable}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}} lccc @{\extracolsep{\fill}}}
\toprule
Specification & GPR Effect & Ambiguity Effect & Volatility Effect \\
\midrule
Baseline Model & -0.0112* & -0.418*** & -0.294*** \\
Lagged GPR ($t-1$) & -0.0095* & -0.385*** & -0.271*** \\
IV Approach & -0.0148** & -0.452*** & -0.310*** \\
Pre-COVID & -0.0098 & -0.352*** & -0.245** \\
COVID Period & -0.0124* & -0.456*** & -0.322*** \\
\bottomrule
\end{tabular*}
\begin{tablenotes}[flushleft]
\small
\item[] This table provides a simplified summary of the core coefficients across key robustness specifications. The negative impact of ambiguity remains consistent and highly significant across all tested models, confirming its robustness as a pricing factor.
\end{tablenotes}
\end{threeparttable}
\end{table}
}
"""
content = content.replace(r"\section{Conclusion}", table5 + "\n\n\\section{Conclusion}")

# 11. Cross-entropy replacements
content = content.replace("entropy-based ambiguity", "cross-entropy-based ambiguity")

with open('code/Adaptive/GeoUncertainty/outputs/results/resubmission/rev_marked_v2.tex', 'w') as f:
    f.write(content)
