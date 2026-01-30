# Review 1

**Summary**
This paper introduces a novel measure of return ambiguity ($\mathcal{A}^{CEA}_t$) based on the Kullback-Leibler (KL) divergence between intraday return distributions and a moving-window benchmark. The authors argue that this measure captures "second-order uncertainty" distinct from traditional risk metrics like variance and demonstrates its predictive power for short-term portfolio performance using CSI 300 high-frequency data. While the use of cross-entropy to quantify Knightian uncertainty is an interesting and potentially valuable contribution to the asset pricing literature, the paper currently lacks a sufficiently rigorous justification for the specific empirical choices made (such as window lengths) and requires a more robust comparison against established ambiguity proxies to rule out the possibility that it is merely capturing tail risk or realized volatility.

**Decision:** Major Revision

**Major Comments**
1. **Theoretical Connection to Empirical Measure:** The paper invokes the multiplier-preference model (Hansen & Sargent, 2001) as the theoretical foundation. However, the transition from this decision-theoretic framework to the specific empirical implementation of calculating KL divergence between a daily distribution and a moving-window intraday distribution is somewhat abrupt. The authors should elaborate on why the "moving-window" distribution serves as the appropriate proxy for the "reference model" in the Hansen-Sargent framework. A more detailed discussion bridging the gap between the "worst-case prior" in theory and the "intraday benchmark" in practice is needed.

2. **Sensitivity Analysis of Window Lengths:** The abstract mentions using moving windows of varying lengths (e.g., 30 to 120 minutes) to capture informational instability. The choice of these specific windows seems arbitrary. The authors should provide a sensitivity analysis or a robustness check to demonstrate that the results are not driven by the specific selection of the window size. Does the predictive power hold if the window is extended to 1 day or shortened to 5 minutes?

3. **Differentiation from Realized Volatility and Tail Risk:** A central claim is that ambiguity is distinct from risk. However, periods of high ambiguity (high KL divergence) likely coincide with periods of high volatility or fat tails. To strengthen the contribution, the authors need to show that $\mathcal{A}^{CEA}_t$ contains information orthogonal to simple Realized Volatility (RV) or higher-order moments like Realized Skewness/Kurtosis. A double-sort portfolio analysis or a regression controlling for these factors explicitly in the main results section would address this concern.

4. **Benchmarking against Alternative Ambiguity Measures:** The paper claims superiority over "traditional risk measures," but it would be more convincing if compared against other *ambiguity* measures, such as the VIX (often used as a proxy for uncertainty, though technically implied volatility) or volume-based ambiguity measures (e.g., dispersion of beliefs). Even a qualitative discussion or a simple correlation matrix with these variables would clarify the unique value added by the proposed index.

**Minor Comments**
1. **Clarification of "Short-term":** The term "short-term portfolio performance" is used frequently. Please define explicitly what "short-term" means in this context (e.g., daily, weekly, or intraday horizons?).
2. **Notation Consistency:** Please ensure that the notation for the ambiguity index ($\mathcal{A}^{CEA}_t$) is used consistently throughout the paper. In some sections, the subscripts or superscripts appear slightly different (e.g., just $A_t$).
3. **Data Cleaning Details:** Please add a brief paragraph or footnote detailing the data cleaning process for the CSI 300 high-frequency data. How were outliers or missing ticks handled? This is crucial for replication.
4. **Figure Readability:** Some figures (e.g., Figure 1 or the ambiguity dynamics plot) might benefit from clearer labeling of the axes or a more distinct color scheme to distinguish between the empirical distribution and the benchmark model.
5. **Typos:** There are a few minor typos in the introduction and literature review sections (e.g., spacing issues in citations). A careful proofreading is recommended.
