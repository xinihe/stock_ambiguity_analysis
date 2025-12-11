# Figure Explanations

## Figure 1: High-Frequency Return Distribution and Ambiguity Measurement

This figure demonstrates the fundamental concept of ambiguity in financial markets using high-frequency data from the CSI 300 index. The figure is composed of two panels:

**Panel (a)** displays the intraday return distribution at 1-minute resolution, revealing the complex microstructure noise and heavy-tailed characteristics of high-frequency returns. The return distribution exhibits several key properties:
- **Heavy tails**: The distribution shows excess kurtosis compared to a normal distribution, indicating a higher probability of extreme returns
- **Central peak**: A sharp peak at zero returns reflects the bid-ask bounce and market microstructure effects
- **Asymmetry**: Slight skewness reflecting information asymmetry and order flow imbalances

**Panel (b)** illustrates the time-varying nature of ambiguity measured using our proposed CEA (Cross-Entropy Ambiguity) index. Key observations include:
- **Ambiguity spikes**: Periods of heightened ambiguity coincide with major market events, policy announcements, and economic uncertainty
- **Mean-reversion**: The ambiguity index exhibits mean-reverting behavior, suggesting that markets alternate between periods of clarity and confusion
- **Lagged effects**: High ambiguity periods often precede increased volatility and trading volume

The empirical results show that ambiguity is distinct from traditional risk measures (like volatility), capturing higher-order uncertainties about the probability distribution itself rather than just its second moment.

---

## Figure 2: Intraday Ambiguity Dynamics and Trading Activity

This figure explores the intraday patterns of ambiguity and their relationship with market activity metrics.

**Panel (a)** shows the intraday pattern of ambiguity throughout the trading day. The pattern reveals several important features:
- **U-shaped pattern**: Ambiguity tends to be higher at the market open (9:30 AM) and close (3:00 PM), corresponding to periods of information asymmetry
- **Mid-day clarity**: Ambiguity typically reaches its lowest point during midday trading when information flow is most efficient
- **Lunch effect**: A slight increase in ambiguity during the lunch hour (11:30 AM - 1:00 PM) reflects reduced liquidity

**Panel (b)** examines the relationship between ambiguity and trading volume. The scatter plot with fitted regression line demonstrates:
- **Positive correlation**: Higher ambiguity is associated with increased trading volume as investors seek to rebalance portfolios
- **Non-linearity**: The relationship is more pronounced at high ambiguity levels, suggesting threshold effects
- **Heteroskedasticity**: The variance of volume increases with ambiguity, reflecting diverse investor reactions

**Panel (c)** presents the cross-correlation between ambiguity and bid-ask spreads:
- **Lead-lag relationship**: Ambiguity tends to lead changes in bid-ask spreads by approximately 5-10 minutes
- **Information asymmetry**: This lead-lag structure suggests that ambiguity captures information asymmetry before it's reflected in market makers' quotes
- **Risk premium**: The persistence of high spreads following ambiguity peaks indicates a risk premium for uncertainty

These findings support the theoretical prediction that ambiguity affects market quality through multiple channels, including liquidity provision and price discovery.

---

## Figure 3: Portfolio Performance Evaluation

This figure provides comprehensive performance evaluation of ambiguity-based portfolio strategies compared to traditional benchmarks.

**Panel (a)** - Cumulative Performance:
- **CSI 300 Index**: The market benchmark provides a baseline return of approximately 10.5% annually
- **AMBE Strategy**: The ambiguity-averse strategy achieves cumulative returns of ~11.8% annually with similar volatility
- **CEA Strategy**: Our proposed cross-entropy ambiguity strategy delivers slightly better performance with ~12.3% annual returns while marginally reducing volatility
- **Modest outperformance**: Both ambiguity-based strategies show modest improvement over the market, with CEA providing slight additional value through better ambiguity management

**Panel (b)** - Drawdown Analysis:
- **Maximum drawdown**: CSI 300 experiences drawdowns during crisis periods (-28% during 2020 and -25% in 2022)
- **AMBE strategy**: Shows slightly improved drawdown control with maximum drawdown of -26%
- **CEA strategy**: Demonstrates modest capital preservation improvement with maximum drawdown of -24%
- **Recovery patterns**: All strategies show similar recovery patterns, with ambiguity-based strategies having marginally better drawdown management

**Panel (c)** - Rolling Sharpe Ratios (252-day window):
- **Consistency**: All strategies show similar Sharpe ratio patterns, with CEA maintaining slightly more stable ratios around 0.65-0.75
- **Market cycles**: CSI 300 Sharpe ratios are procyclical, declining during crises
- **Modest alpha**: The ambiguity strategies generate small but consistent alpha across different market regimes
- **Risk-adjusted performance**: CEA shows slight improvement in risk-adjusted metrics

**Panel (d)** - Performance Summary Statistics:
- **Annual Return**: CEA (12.3%) > AMBE (11.8%) > CSI 300 (10.5%)
- **Volatility**: CEA (18.2%) < AMBE (18.4%) < CSI 300 (18.5%)
- **Sharpe Ratio**: CEA (0.68) > AMBE (0.64) > CSI 300 (0.57)
- **Calmar Ratio**: CEA (0.51) > AMBE (0.45) > CSI 300 (0.42)

The results demonstrate that accounting for ambiguity through our CEA methodology provides modest but meaningful economic value beyond traditional risk management approaches, with slightly better risk-adjusted returns and drawdown characteristics.

---

## Figure 4: Cross-Entropy Ambiguity Calculation Framework

This figure illustrates the innovative methodology for measuring ambiguity using cross-entropy principles.

**Panel (a)** - Methodology Framework:
The framework consists of several key components:

1. **Data Collection**: High-frequency intraday returns (1-minute resolution) provide the raw input for ambiguity calculation
2. **Daily Distribution Construction**: Each day's return distribution $q_t$ is estimated from intraday data using kernel density estimation
3. **Historical Window Analysis**: A rolling 20-day window captures recent market states
4. **K-means Clustering**: Historical distributions are clustered into $k=4$ representative regimes
5. **Benchmark Distributions**: Each cluster center serves as a benchmark distribution $p_i$ representing a market regime
6. **KL Divergence Calculation**: For each day, we calculate $D_{KL}(q_t || p_i)$ for all benchmark distributions
7. **Ambiguity Index**: The minimum KL divergence across all benchmarks defines the ambiguity index: $\mathcal{A}^{CEA}_t = \min_i D_{KL}(q_t || p_i)$
8. **Portfolio Decision**: The ambiguity index drives portfolio allocation decisions

**Panel (b)** - Multiple Period Distributions:
This panel visualizes how the methodology works in practice with different market regimes:

1. **Period 1 - Normal Market**: Characterized by moderate volatility ($\sigma=1.0$) and zero mean returns, representing typical market conditions
2. **Period 2 - High Volatility**: Shows increased dispersion ($\sigma=1.5$) with zero mean, representing crisis or uncertainty periods
3. **Period 3 - Bull Market**: Exhibits positive drift ($\mu=0.5$) with reduced volatility ($\sigma=0.8$), capturing upward trends
4. **Period 4 - Bear Market**: Features negative drift ($\mu=-0.5$) with elevated volatility ($\sigma=1.2$), representing downturns

The **current distribution** $q_t$ (black dashed line) is compared against all historical regimes. The KL divergence measures the "surprise" or information gain required to update from each benchmark to the current distribution:

$$D_{KL}(q || p) = \int q(x) \log\frac{q(x)}{p(x)} dx$$

The **minimum KL divergence** determines the ambiguity level - when the current distribution closely matches a historical pattern, ambiguity is low; when it deviates significantly from all historical patterns, ambiguity is high.

**Key Advantages of the CEA Approach**:

1. **Distributional Comparison**: Unlike variance-based measures, CEA captures differences in the entire distribution shape
2. **Adaptive Benchmarking**: The methodology automatically adapts to evolving market conditions through rolling clustering
3. **Interpretability**: Each KL divergence can be interpreted as the information cost of updating beliefs
4. **Regime Awareness**: By comparing against multiple historical regimes, the measure remains robust across market states
5. **Theoretical Foundation**: Grounded in information theory, providing rigorous mathematical properties

This framework represents a significant advance in ambiguity measurement, bridging the gap between theoretical models and practical implementation in financial markets.

---

## Excalidraw Flowchart Prompt for Figure 4(a)

```
Create an Excalidraw-style flowchart for the Cross-Entropy Ambiguity (CEA) calculation methodology with the following specifications:

Main Components (use different shapes and colors):
1. Start node (rounded rectangle, light blue): "1-Minute Intraday Data"
2. Process node (diamond, light cyan): "Daily Return Distribution q_t"
3. Parallel process (rounded rectangle, light orange): "Historical Window (20 days)"
4. Process (rounded rectangle, light purple): "K-means Clustering"
5. Output (parallelogram, light red): "4 Market Regimes Identified"
6. Data storage (cylinder shape, pink): "Benchmark Distributions {p_i}"
7. Central process (hexagon, light yellow): "KL Divergence D_KL(q_t || p_i)"
8. Decision (diamond, light pink): "Minimum Selection min_i D_KL"
9. Output (ellipse, light green): "Ambiguity Index A^{CEA}_t"
10. Final process (rounded rectangle, sky blue): "Portfolio Allocation"

Flow connections (numbered arrows):
1. Arrow down from node 1 to node 2 (Process)
2. Curved arrow from node 2 to node 3 (Historical analysis)
3. Arrow down from node 3 to node 4 (Cluster)
4. Arrow from node 4 to node 5 (Identify)
5. Arrow down from node 5 to node 6 (Create)
6. Curved arrow from node 2 to node 7 (Compare)
7. Curved arrow from node 6 to node 7 (Compare)
8. Arrow down from node 7 to node 8 (Select Min)
9. Arrow down from node 8 to node 9 (Output)
10. Curved arrow from node 9 to node 10 (Inform decision)

Styling requirements:
- Hand-drawn aesthetic with slightly imperfect lines
- Drop shadows for depth (light gray offset)
- Rounded corners on rectangles
- Bold text inside shapes
- Numbered circles on flow arrows (1-10)
- Curved connection lines for better flow visualization
- Light pastel colors for shapes
- Black borders with medium thickness
- Add handwritten-style annotations:
  * "Different Market Regimes" near the benchmark distributions
  * "Portfolio Optimization Based on Ambiguity" near the allocation decision

Layout: Flow from top to bottom with branches, showing how intraday data flows through distribution creation, historical analysis, KL divergence calculation, to final portfolio decisions.
```

---

## Excalidraw Flowchart Prompt for Figure 4(a) - Revised Layout

```
Create an Excalidraw-style flowchart for the Cross-Entropy Ambiguity (CEA) calculation methodology with a clean two-column layout:

Layout Structure:
- Left column: Main data flow (vertical)
- Right column: Historical analysis branch (vertical)
- Bottom: Key Processes legend box

Main Components (use rounded rectangles with drop shadows):

Left Column (Main Flow):
1. (2, 10) - Start node: "1-Minute Intraday Data" (light blue #A5D8FF)
2. (2, 8.5) - Process: "Daily Return Distribution q_t" (light cyan #C5F6FA)
3. (2, 6.5) - Central process: "KL Divergence D_KL(q_t || p_i)" (light yellow #FFF9C4)
4. (2, 5) - Decision: "Minimum Selection min_i D_KL" (light pink #FFE0E0)
5. (2, 3.5) - Output: "Ambiguity Index A^{CEA}_t" (light green #E8F5E9)
6. (0.5, 2) - Final process: "Portfolio Allocation" (sky blue #E1F5FE)

Right Column (Historical Analysis):
1. (6.5, 9.5) - Process: "Historical Window (20d)" (light orange #FFE0B2)
2. (6.5, 8.5) - Process: "K-means Clustering" (light purple #E1BEE7)
3. (6.5, 7) - Output: "4 Market Regimes Identified" (light red #FFCDD2)
4. (6.5, 5.5) - Data storage: "Benchmark Distributions {p_i}" (pink #F8BBD9)

Arrows (numbered with circles):
1. Vertical arrow: (1→2) with label "1" on right side
2. Vertical arrow: (2→3) with label "2" on right side
3. Vertical arrow: (3→4) with label "3" on right side
4. Vertical arrow: (4→5) with label "4" on right side
5. Vertical arrow: (5→6) with label "5" on right side
6. Vertical arrow: (7→8) with label "6" on left side
7. Vertical arrow: (8→9) with label "7" on left side
8. Diagonal arrow: (5→10) with label "8"
9. Horizontal arrow: (9→3) with label "9" (from benchmarks to KL)

Key Processes Box (bottom right, white background):
Title: "Key Processes"
List items with numbered circles (①-⑨):
① Data Processing - Transform intraday returns
② Distribution Creation - Build q_t from returns
③ Historical Analysis - 20-day window clustering
④ Regime Identification - K-means finds 4 patterns
④ Benchmark Creation - Generate {p_i} distributions
⑥ KL Comparison - Measure distributional distance
⑦ Minimum Selection - Find min_i D_KL
⑧ Index Output - Generate A^{CEA}_t
⑨ Portfolio Decision - Guide allocation strategy

Styling Requirements:
- Hand-drawn aesthetic with slightly imperfect lines
- Drop shadows (light gray, 5px offset)
- Rounded corners on all rectangles
- Bold text inside shapes (10-11pt)
- Arrow labels in white circles with black borders
- Clean spacing - no overlaps
- Pastel colors as specified
- Black borders with medium thickness (1.5-2px)
- Clear vertical flow in left column
- Separate historical analysis in right column
- Professional academic presentation style

Visual Hierarchy:
- Main flow boxes slightly larger
- KL divergence box emphasized (central position)
- Clear separation between columns
- Arrow numbering follows flow sequence
```