import re

with open('code/Adaptive/GeoUncertainty/outputs/results/geopoliticalAmb04_article.tex', 'r') as f:
    text = f.read()

# 1. Abstract
text = text.replace(
    r"When ambiguity is included alongside traditional risk, models explain more variation in returns and the role of volatility-based risk is often reduced.",
    r"\textcolor{red}{When ambiguity is included alongside traditional risk, models explain more variation in returns, illustrating that ambiguity captures a distinct dimension of Knightian uncertainty without invalidating the role of traditional risk.}"
)

# 1. Intro: "inflating volatility's role by 47%"
text = text.replace(
    r"(iii) models ignoring ambiguity misattribute ambiguity's impact to volatility, inflating volatility's role by 47\%.",
    r"\textcolor{red}{(iii) models ignoring ambiguity misattribute a portion of ambiguity's impact to volatility, illustrating the complementary nature of these two uncertainty channels.}"
)

# 2. Lit Review: alternative ambiguity proxies
text = text.replace(
    r"Empirically testing these theories requires a credible measure of ambiguity. The literature has proposed several proxies, such as the dispersion of professional forecasts \citep{Brenner2018, Ulrich2013} or the variance of variance. However, these proxies can be indirect and may not be available at a high frequency. Our work contributes methodologically by developing a new, daily ambiguity measure from a simplified cross-entropy framework (detailed in Appendix A). This measure is designed to be more theoretically grounded and empirically robust, directly capturing investor uncertainty over the correct model of asset returns.",
    r"Empirically testing these theories requires a credible measure of ambiguity. The literature has proposed several proxies, such as the dispersion of professional forecasts \citep{Brenner2018, Ulrich2013} or the variance of variance. \textcolor{red}{Unlike forecast dispersion, which relies on analyst coverage, or variance-of-variance, which captures second-moment uncertainty, our cross-entropy measure directly quantifies the divergence from historical reference models, thereby capturing the essence of Knightian uncertainty---where the probability distributions themselves are unknown.} Our work contributes methodologically by developing a new, daily ambiguity measure from a simplified cross-entropy framework (detailed in Appendix A). This measure is designed to be more theoretically grounded and empirically robust, directly capturing investor uncertainty over the correct model of asset returns."
)

# 3. Intro: moderate economic magnitudes and portfolio implications
text = text.replace(
    r"Our findings have immediate practical implications. Portfolio managers using volatility-only hedging strategies remain exposed to ambiguity risk. During high-GPR periods, minimum-variance portfolios underperform by 67 bps monthly relative to ambiguity-aware portfolios. Policy interventions focusing solely on volatility stabilization (e.g., circuit breakers) may miss the larger ambiguity transmission channel.",
    r"\textcolor{red}{Our findings have practical implications. The daily associations suggest that portfolios ignoring ambiguity may face uncompensated risks during high-GPR periods, highlighting the importance of ambiguity-aware risk management. Policy interventions focusing solely on volatility stabilization may miss a significant portion of the ambiguity transmission channel.}"
)

# 3. Results: economic significance
text = text.replace(
    r"In terms of economic significance, a one-standard-deviation increase in ambiguity reduces returns by 11.3 bps daily---equivalent to 28.3\% annualized.",
    r"\textcolor{red}{In terms of economic significance, a one-standard-deviation increase in ambiguity is associated with a daily return reduction of 11.3 bps.}"
)

# 4. Mediation Analysis: Baron-Kenny limitations
text = text.replace(
    r"The temporal sequence---morning GPR news affecting midday ambiguity calculations, which then influence end-of-day returns---supports this causal pathway.",
    r"The temporal sequence---morning GPR news affecting midday ambiguity calculations, which then influence end-of-day returns---supports this causal pathway. \textcolor{red}{However, while the Baron-Kenny framework provides a useful statistical decomposition, we caution against strict causal interpretations. In high-frequency financial data, unobserved contemporaneous shocks can affect both the mediator and the outcome simultaneously, complicating the identification of isolated transmission paths.}"
)

# 5. IV Strategy: exclusion restriction
text = text.replace(
    r"This \"overnight\" information shock satisfies the exclusion restriction: it drives opening-day ambiguity but cannot be caused by contemporaneous Chinese market intraday volatility.",
    r"This \"overnight\" information shock satisfies the exclusion restriction: it drives opening-day ambiguity but cannot be caused by contemporaneous Chinese market intraday volatility. \textcolor{red}{While the 'Non-Asian GPR' instrument is temporally predetermined relative to the Chinese trading session, we acknowledge that global geopolitical shocks might influence Chinese markets through alternative contemporaneous channels, such as overnight shifts in global commodity prices or safe-haven currency flows. To the extent these channels are correlated with our instrument, the exclusion restriction may be partially violated, suggesting our IV estimates should be interpreted with appropriate caution.}"
)

# 6. Timing assumptions
text = text.replace(
    r"We use a contemporaneous alignment $(t, t)$ for our main analysis. This is justified because the GPR news (arriving pre-open) has ample time to be incorporated into investor beliefs and trading behavior throughout the trading day, which is then reflected in the intraday-based ambiguity measure and the close-to-close returns. We verified this by testing $(t-1, t)$ specifications, which yielded qualitatively similar but slightly weaker results, supporting the view that the market reaction to geopolitical news is rapid. Robustness tests with alternative alignments confirm that our main findings are not driven by timing choices.",
    r"\textcolor{red}{We use a contemporaneous alignment $(t, t)$ for our main analysis. This is justified because the GPR news (arriving pre-open, between 5:00 AM and 9:30 AM Beijing time) has ample time to be incorporated into investor beliefs and trading behavior throughout the trading day. We also introduce a lagged specification $(t-1, t)$ in our robustness checks to verify that the effects are not purely driven by our contemporaneous timing assumptions.}"
)

# 7. Broader literature
text = text.replace(
    r"Our work extends the emerging literature on uncertainty transmission in several critical ways. First, while \citet{Baker2016} and \citet{Caldara2022} document that uncertainty matters for markets, they treat uncertainty as monolithic. By decomposing uncertainty into risk (volatility) and ambiguity, we reveal that ambiguity accounts for the majority of GPR's pricing effect.",
    r"\textcolor{red}{Our work extends the emerging literature on uncertainty transmission in several critical ways, bridging the gap between macroeconomic uncertainty and ambiguity pricing. First, while studies like \citet{Baker2016} and \citet{Caldara2022} document that uncertainty matters for markets, they often treat uncertainty as monolithic. By decomposing uncertainty into complementary risk (volatility) and ambiguity channels, we build on the insights of \citet{Brenner2018} and reveal that ambiguity is a distinct factor in GPR's pricing effect.}"
)

# 8. Streamline regression models and control variables
text = text.replace(
    r"In our time-series specifications, we incorporate a comprehensive set of control variables to isolate the impact of geopolitical ambiguity from other macroeconomic and market-wide fluctuations. We control for the \textbf{Market Return ($MKT$)}, defined as the value-weighted return of the entire A-share market, to capture systematic market movements.\footnote{We include the value-weighted return of the entire A-share market (Market Return, MKT) as a control variable in the baseline regression for two key reasons. First, it addresses the potential omitted variable bias inherent in asset pricing tests. The CSI 300 Index, as a subset of the broader A-share market (comprising the 300 largest and most liquid A-shares), is inherently influenced by systematic market-wide fluctuations. Without controlling for MKT, the regression might erroneously attribute return variations driven by overall market trends (e.g., macroeconomic shocks affecting all A-shares) to our core variables of interest (e.g., changes in ambiguity, $\Delta\mathcal{A}$). By isolating the market-wide component of returns, we can more accurately identify the independent effect of ambiguity and volatility on CSI 300 returns. Second, this specification aligns with standard practices in asset pricing literature (consistent with the logic of the CAPM and multi-factor models), where controlling for the broad market return helps disentangle asset-specific or channel-specific effects (here, the geopolitical risk-ambiguity channel) from general market movements. While the CSI 300 Index and the full A-share market exhibit positive correlation, MKT captures fluctuations in smaller, less liquid A-shares excluded from the CSI 300—variations that could otherwise confound the estimation of our key coefficients. This control thus enhances the robustness and interpretability of our empirical results.} To account for global risk sentiment, we include the \textbf{VIX Index}, which proxies for international risk aversion.",
    r"\textcolor{red}{In our time-series specifications, we incorporate a streamlined set of control variables to isolate the impact of geopolitical ambiguity. We control for the \textbf{Market Return ($MKT$)} to capture systematic market movements, the \textbf{VIX Index} for global risk sentiment, the \textbf{Term Spread ($TERM$)} for domestic monetary policy, and the \textbf{Risk-Free Rate ($RF$)}.} "
)
text = text.replace(
    r"Domestically, we control for the term structure of interest rates using the \textbf{Term Spread ($TERM$)}—calculated as the difference between 10-year and 1-year Chinese government bond yields—to capture the monetary policy stance and economic growth expectations. Additionally, we use the 3-month SHIBOR as the \textbf{Risk-Free Rate ($RF$)}.",
    ""
)

# 9. Simplify tables and add summary table
table_5 = r"""
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

text = text.replace(
    r"\section{Conclusion}",
    table_5 + "\n\n" + r"\section{Conclusion}"
)

# Ensure consistent "cross-entropy"
text = text.replace("entropy-based ambiguity", "cross-entropy-based ambiguity")

with open('code/Adaptive/GeoUncertainty/outputs/results/geopoliticalAmb04_article.tex', 'w') as f:
    f.write(text)

