# Review 2

**Summary**
This manuscript proposes a cross-entropy-based ambiguity index ($\mathcal{A}^{CEA}_t$) derived from high-frequency intraday return distributions to quantify market ambiguity. The study integrates this measure into a utility-based decision framework and provides empirical evidence from the Chinese stock market (CSI 300) suggesting that ambiguity is negatively priced in the short run and that incorporating it improves portfolio performance. While the empirical results are intriguing and the focus on "distributional uncertainty" is timely, the paper faces challenges regarding the economic interpretation of the results, specifically in terms of transaction costs for such a high-frequency strategy and the identification of causal channels.

**Decision:** Major Revision

**Major Comments**
1. **Transaction Costs and Economic Significance:** The paper demonstrates that portfolio strategies integrating the ambiguity index outperform traditional approaches. However, since the measure relies on intraday data and the strategy likely involves frequent rebalancing to capture "short-term" ambiguity fluctuations, transaction costs could significantly erode these excess returns. The authors should include a discussion or a backtest scenario that accounts for realistic transaction costs (e.g., bid-ask spread, commissions) to demonstrate that the strategy remains profitable in practice.

2. **Causality vs. Predictability:** The paper uses Granger causality tests to establish predictive power. While useful, Granger causality only indicates temporal precedence, not true economic causality. To strengthen the causal argument, the authors could explore exogenous shocks or specific market events (e.g., policy announcements, earnings surprises) where ambiguity would theoretically spike, and observe the subsequent return behavior. This would provide more robust evidence than statistical lag-lead relationships alone.

3. **Sub-period Analysis (Market Regimes):** The relationship between ambiguity and returns might be state-dependent. Does the negative relationship hold equally during bull markets, bear markets, and sideways markets? The current analysis mentions "market turbulence," but a more systematic sub-period analysis (e.g., splitting the sample into crisis vs. non-crisis periods) would provide deeper insights into whether ambiguity is a constant risk factor or a regime-switching signal.

**Minor Comments**
1. **Abstract Length:** The abstract is somewhat lengthy and could be condensed to focus more sharply on the key results and less on the general background of risk vs. ambiguity.
2. **Context for International Readers:** Since the empirical analysis focuses on A-share stocks (CSI 300), it would be helpful to add a sentence explaining why this market is particularly suitable for studying ambiguity (e.g., high retail participation, policy uncertainty) to broaden the appeal to an international audience.
3. **Clarify "Liquidity Provision":** The introduction mentions investigating mediating channels like "liquidity provision." Please ensure this mechanism is clearly explained in the mechanism analysis section. How exactly does ambiguity affect liquidity provision in your framework?
4. **Variable Definitions:** In the methodology section, please define the components of the $\mathcal{A}^{CEA}_t$ formula clearly immediately after the equation is presented to improve readability.
5. **Reference Formatting:** Check the formatting of citations (e.g., "Knight (1921)" vs "(Knight, 1921)") to ensure consistency with the journal's style guide.
6. **Descriptive Statistics:** A table showing the summary statistics of the ambiguity index (mean, standard deviation, min, max) compared to other market variables would be very helpful for the reader to gauge the magnitude of the measure.
