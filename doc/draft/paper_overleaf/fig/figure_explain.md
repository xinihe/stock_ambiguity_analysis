# Figure Explanations

## Figure 1: Kullback-Leibler Divergence and Distributional Differences

This figure illustrates the fundamental concept of Kullback-Leibler (KL) divergence as a measure of ambiguity, demonstrating how it quantifies the "surprise" or information gain between a reference distribution $p(x)$ and an alternative distribution $q(x)$.

**Panel (a) Different Means**: Shows the sensitivity of KL divergence to location shifts. Even with identical shapes, a shift in the central tendency (mean) results in a positive divergence.

**Panel (b) Different Variances**: Demonstrates sensitivity to dispersion. A distribution with higher variance ($q(x)$) differs from the reference ($p(x)$), capturing uncertainty related to volatility changes.

**Panel (c) Skewed Distribution**: Highlights the ability of KL divergence to capture asymmetry. A skewed distribution represents a different risk profile than a symmetric normal distribution, which KL divergence detects.

**Panel (d) Heavy Tails**: Shows the comparison between distributions with different tail behaviors (t-distributions with different degrees of freedom). KL divergence is particularly sensitive to tail events, making it suitable for capturing extreme market risks.

**Panel (e) Bimodal Distribution**: Illustrates the detection of structural changes or regime shifts where the distribution splits into two modes, representing a fundamentally different market state than a unimodal normal distribution.

**Key Properties**: The heatmap summarizes that KL divergence is non-negative, zero only when distributions are identical, and asymmetric. In finance, higher KL divergence indicates greater ambiguity or model uncertainty.

---

## Figure 2: Ambiguity Index Dynamics and Market Conditions

This figure presents the temporal evolution of the Cross-Entropy Ambiguity (CEA) index and its relationship with market performance and volatility regimes.

**Panel (a) Market Index Performance**: Displays the cumulative returns of the market over the sample period. Vertical lines mark key market events, providing context for the ambiguity analysis.

**Panel (b) Time-Varying Ambiguity Index**: Plots the $\mathcal{A}^{CEA}_t$ index. The background shading indicates different market volatility regimes (e.g., Low Vol, High Vol, Crisis). Spikes in the ambiguity index frequently coincide with transitions between regimes or during crisis periods, reflecting heightened uncertainty about the market's state.

**Panel (c) Ambiguity and Forward Returns**: A scatter plot showing the relationship between the ambiguity index (x-axis) and next-day returns (y-axis). The fitted regression line suggests a relationship between the level of ambiguity and subsequent market performance, implying a potential risk-ambiguity trade-off or premium.

---

## Figure 3: Portfolio Performance Evaluation

This figure provides a comprehensive performance comparison of ambiguity-based strategies against the market benchmark.

**Panel (a) Cumulative Portfolio Performance**: Visualizes the growth of a hypothetical $1 investment across three strategies:
- **CSI 300 Index**: The market benchmark.
- **AMBE Strategy**: A standard ambiguity-averse strategy.
- **$\mathcal{A}^{CEA}_t$ Strategy**: The proposed Cross-Entropy Ambiguity strategy.
The CEA strategy demonstrates superior long-term capital accumulation compared to the benchmark and the standard AMBE strategy.

**Panel (b) Drawdown Analysis**: Compares the depth and duration of declines from peaks. The shaded areas represent the drawdown magnitude. The ambiguity-based strategies, particularly the CEA strategy, exhibit reduced maximum drawdowns during periods of market stress, indicating better capital preservation.

**Panel (c) Rolling Sharpe Ratio**: Plots the Sharpe ratio over a rolling 252-day window. The CEA strategy maintains a more stable and generally higher risk-adjusted return profile throughout different market cycles, avoiding the deep dips seen in the market benchmark.

**Panel (d) Performance Summary Statistics**: A table presenting key metrics:
- **Annual Return**: The CEA strategy achieves the highest annualized return.
- **Annual Volatility**: Ambiguity strategies show comparable or slightly lower volatility.
- **Sharpe Ratio & Calmar Ratio**: The CEA strategy outperforms in risk-adjusted metrics, indicating it delivers more return per unit of risk and better drawdown recovery.

---

## Figure 4: Risk vs. Ambiguity: Conceptual Distinction

This figure distinguishes between traditional risk measures (moments of a single distribution) and ambiguity (uncertainty about the distribution itself).

**Panels (a-c) Traditional Risk Measures**:
- **(a) Volatility (2nd Moment)**: Measures dispersion around the mean.
- **(b) Skewness (3rd Moment)**: Measures asymmetry of returns.
- **(c) Kurtosis (4th Moment)**: Measures the thickness of tails (probability of extreme events).
These measures assume a *known* distribution and quantify its properties.

**Panel (d) Ambiguity**: Illustrated as the existence of multiple plausible distributions (e.g., Normal, Shifted, Skewed, Heavy-tail) that could fit the data. Ambiguity arises from the investor's inability to uniquely identify the "true" data-generating process among these alternatives.

**Panel (e) Correlation Matrix**: A heatmap displaying the correlations between traditional risk measures and the Ambiguity Index ($\mathcal{A}^{CEA}_t$). The low correlation values confirm that ambiguity is largely orthogonal to volatility, skewness, and kurtosis. This implies that the CEA index captures a distinct dimension of uncertainty not explained by standard risk factors.

**Information Hierarchy**: The figure elucidates that Risk deals with unknown outcomes within a known model, whereas Ambiguity deals with unknown models or parameters.

---

## Figure 5: Methodology Flowchart

This figure outlines the step-by-step computational framework for constructing the Cross-Entropy Ambiguity (CEA) index.

**Step 1: Data Collection**: Collection of high-frequency (1-minute) intraday return data.

**Step 2: Distribution Construction**: Estimation of the daily return distribution ($q_t$) from the intraday data.

**Step 3: Sliding Window Analysis**: Utilization of a 20-day historical window to capture recent market dynamics.

**Step 4: Regime Identification**: Application of K-means clustering to the historical distributions to identify distinct market regimes.

**Step 5: Benchmark Distributions**: Determination of representative benchmark distributions ($\{p_i\}$) for each identified regime.

**Step 6: KL Divergence Calculation**: Computation of the KL divergence $D_{KL}(q_t || p_i)$ between the current day's distribution and each benchmark distribution.

**Step 7: Minimum Selection**: Identification of the minimum divergence ($\min_i D_{KL}$), representing the distance to the closest known market regime.

**Step 8: Ambiguity Index**: The minimum divergence is defined as the Ambiguity Index ($\mathcal{A}^{CEA}_t$). A high value indicates the current market state is unlike any historical regime (high ambiguity).

**Output & Application**: The resulting daily ambiguity time series is utilized for portfolio optimization and risk management decisions.
- Professional academic presentation style

Visual Hierarchy:
- Main flow boxes slightly larger
- KL divergence box emphasized (central position)
- Clear separation between columns
- Arrow numbering follows flow sequence
```