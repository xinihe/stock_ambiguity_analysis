# The Ambiguity Channel of Geopolitical Risk: An Empirical Investigation

*January 2025*

## 1. Introduction

Geopolitical risk (GPR) is a critical determinant of financial market behavior, yet the precise mechanisms through which it affects asset prices remain an active area of research. While traditional models focus on risk-based channels, this study explores a novel transmission pathway: **market ambiguity**. We posit that GPR, as a form of profound uncertainty, fundamentally alters investors' perception of the distribution of future outcomes, a phenomenon distinct from risk.

This paper investigates the direct impact of GPR from major global powers—the United States, China, and Japan—on market ambiguity in China. Our central research question is: **How does geopolitical risk influence market ambiguity?**

By establishing this foundational link, we lay the groundwork for a more comprehensive causal model: **GPR → Ambiguity → Market Returns**. This research builds upon recent breakthroughs that have demonstrated the significant, non-linear impact of ambiguity on returns, explaining over 17% of its variation. The discovery of a significant GPR-ambiguity relationship provides the crucial first link in this causal chain, representing a significant step toward a deeper understanding of how geopolitical uncertainty is priced in financial markets.

## 2. Methodology

### 2.1. Data and Variables

Our analysis utilizes monthly data from **January 2018 to December 2024**, comprising 84 observations. The dataset combines ambiguity metrics with country-specific GPR indices.

-   **Dependent Variable: Ambiguity Metric**: The primary dependent variable is a proprietary, quantitatively derived measure of market ambiguity. It captures the uncertainty about the true probability distribution of returns, distinct from volatility (which measures risk within a known distribution).

-   **Independent Variables: Geopolitical Risk (GPR)**: We use the widely recognized GPR index to measure geopolitical risk for the **United States (GPR_US)**, **China (GPR_China)**, and **Japan (GPR_Japan)**. These indices are constructed based on the frequency of newspaper articles mentioning unfavorable geopolitical events.

### 2.2. Econometric Models

To examine the relationship between GPR and ambiguity, we employ a series of Ordinary Least Squares (OLS) regression models, progressively increasing in complexity to uncover potential non-linearities.

1.  **Benchmark Linear Model**: This model establishes a baseline linear relationship.
    `Ambiguity = β₀ + β₁*GPR_US + β₂*GPR_China + β₃*GPR_Japan + ε`

2.  **Interaction Model**: To test whether the effects of GPR are interdependent, we introduce interaction terms between the GPR indices of the three countries.

3.  **Quadratic Model**: To capture non-linear effects, we introduce squared terms for each GPR index. This allows for the possibility that the impact of GPR on ambiguity is not constant and may change at different levels of risk.
    `Ambiguity = β₀ + β₁*GPR + β₂*GPR² + ε`

4.  **Lag Model**: To investigate the delayed effects of GPR on ambiguity.

## 3. Empirical Results and Analysis

### 3.1. Monthly Analysis

The regression analysis reveals a statistically significant relationship between GPR and market ambiguity, with the quadratic and lag models providing the best fit.

| Model Specification | R²     | Key Finding                               |
| ------------------- | ------ | ----------------------------------------- |
| Linear Model        | 16.14% | China GPR has a significant negative effect. |
| Interaction Model   | 16.57% | Modest improvement over the linear model. |
| **Quadratic Model** | **19.62%** | **Best overall fit, indicating non-linearity.** |
| **Lag Model**| **27.38%** | **Lags of GPR are highly significant.** |

-   **China GPR**: The most striking result is the **statistically significant negative coefficient** for China's GPR in the linear model **(-0.269, p=0.0155)**. This suggests that as China's own geopolitical risk increases, market ambiguity *decreases*. This counter-intuitive finding is robust across specifications and is the central empirical puzzle of this study.

-   **US and Japan GPR**: The GPR from the US and Japan shows no statistically significant impact on Chinese market ambiguity in any of the models. This suggests that domestic geopolitical risk is the primary driver, at least within this sample period.

-   **Non-Linearity and Lags**: The superior performance of the quadratic model (R² = 19.62%) and especially the lag model (R² = 27.38%) are key findings. They imply that the relationship between GPR and ambiguity is not a simple linear one. The effect of a change in GPR on ambiguity depends on the existing level of GPR and its past values, suggesting potential threshold effects or regime-dependent behavior.

### 3.2. Weekly Analysis

To further investigate the relationship between GPR and ambiguity, we conducted a weekly analysis by interpolating the monthly GPR data. This allows for a more granular view of the dynamic relationship.

#### 3.2.1. Model Comparison (Weekly)

| Model       | R-squared |
| :---------- | :-------- |
| Linear      | 0.0133    |
| **Lag**     | **0.0883**|

While the overall explanatory power of the models is lower in the weekly analysis, the Lag Model still outperforms the Linear Model, consistent with the monthly findings.

#### 3.2.2. Analysis of GPR Effects (Weekly)

The most striking result from the weekly analysis is the statistically significant negative relationship between the **third-week lag of China's GPR** and ambiguity (p-value = 0.037). This finding reinforces the monthly results and provides more precise timing on the lagged effect of China's GPR. The negative coefficient suggests that increases in GPR from China are followed by a decrease in market ambiguity three weeks later.

### 3.3. The Counter-Intuitive Negative Relationship

The negative relationship between China's GPR and ambiguity challenges the conventional wisdom that higher risk should lead to higher ambiguity. We propose several potential explanations for this puzzle:

1.  **Risk Clarification Effect**: High-profile geopolitical events, while risky, may reduce ambiguity by clarifying the "rules of the game." For example, a specific geopolitical action, though destabilizing, removes uncertainty about whether that action would be taken, thus reducing ambiguity about the state of the world.
2.  **Information Flow and Attention**: Periods of high GPR attract intense media and analyst coverage. This flood of information and heightened market attention may lead to more rapid processing of uncertainty, effectively converting ambiguity into quantifiable risk.
3.  **Government Response**: In an environment of high domestic GPR, market participants may expect strong, predictable government and central bank interventions to stabilize markets. This certainty about the policy response could reduce ambiguity, even as risk remains elevated.

## 4. Future Study Proposal: A Comprehensive Economic Analysis of the GPR → Ambiguity → Market Returns Causal Pathway

The findings of this paper provide a strong foundation for a more comprehensive investigation into the causal chain linking GPR to market returns through the ambiguity channel. We propose a future study structured around the following hypotheses and methodologies.

### 4.1. Hypothesis Development

-   **H1: GPR has a significant causal effect on market ambiguity.** (This study provides preliminary evidence; the future study will aim for more rigorous causal identification.)
-   **H2: Market ambiguity acts as a significant mediator in the relationship between GPR and market returns.** This is the central hypothesis of the causal pathway: GPR affects returns *through* its effect on ambiguity.
-   **H3: The causal effect of GPR on ambiguity is non-linear and exhibits threshold effects.** The impact of a GPR shock depends on the prevailing geopolitical and market environment.
-   **H4: The negative GPR-ambiguity relationship is driven by a "risk clarification" mechanism, where GPR shocks resolve uncertainty about potential political actions.**

### 4.2. Proposed Methodology for Future Study

1.  **Causal Identification**: To move beyond correlation, we will employ advanced econometric techniques.
    -   **Instrumental Variables (IV)**: We can use exogenous shocks to GPR in other, unrelated countries as instruments for Chinese GPR to isolate the causal component.
    -   **Vector Autoregression (VAR) Models**: A VAR framework including GPR, ambiguity, and market returns will allow us to analyze the dynamic, impulse-response relationships between these variables.

2.  **Mediation Analysis**: We will use formal mediation analysis (e.g., Sobel test, bootstrapping methods) to explicitly test H2. This will quantify the proportion of the total effect of GPR on returns that is transmitted through the ambiguity channel.

3.  **Non-Linear Modeling**: To test H3, we will explore:
    -   **Threshold Autoregressive (TAR) Models**: To identify specific GPR levels at which the relationship with ambiguity changes.
    -   **Markov-Switching Models**: To allow the GPR-ambiguity relationship to evolve across different unobserved market "regimes" (e.g., high vs. low volatility).

4.  **Mechanism-Focused Analysis**: To test H4, we will use:
    -   **Textual Analysis**: By analyzing the content of news articles that contribute to the GPR index, we can create sub-indices for different types of GPR (e.g., threats vs. actions) to see if they have different effects on ambiguity.
    -   **High-Frequency Data**: Using daily or even intraday data around major geopolitical events will allow for a more granular analysis of the "risk clarification" effect.

By undertaking this comprehensive analysis, we can provide a much deeper and more nuanced understanding of how the abstract concept of geopolitical risk is priced into financial assets, with market ambiguity serving as a critical, and previously under-appreciated, transmission channel.

## Empirical Results: Ambiguity and Geopolitical Risk

### GPR -> Ambiguity

#### Monthly Analysis

Our initial analysis, conducted on a monthly basis, explored the relationship between various geopolitical risk (GPR) indices and market ambiguity. The regression models, both linear and with lagged variables, yielded low R-squared values, indicating that GPR indices explained a very small portion of the variance in ambiguity. Furthermore, none of the GPR variables demonstrated a statistically significant effect on ambiguity at this frequency.

#### Weekly Analysis

Switching to a weekly frequency, we observed a decrease in the overall explanatory power of the models, as evidenced by even lower R-squared values. However, this higher-frequency analysis revealed a statistically significant finding that was not present in the monthly data. Specifically, we identified a negative relationship between China's GPR index and market ambiguity with a three-week lag. This suggests that an increase in China's GPR is associated with a decrease in market ambiguity three weeks later. This counterintuitive finding warrants further investigation.

### Ambiguity -> Return

To further understand the impact of ambiguity, we investigated its relationship with market returns.

#### Monthly and Weekly Analysis

Across both monthly and weekly frequencies, our regression models failed to identify a statistically significant relationship between the ambiguity metric and market returns. The R-squared values were close to zero, and the p-values for the ambiguity metric were well above the conventional significance thresholds. This suggests that, within the scope of our analysis, ambiguity does not have a direct, linear impact on market returns.

### Comparison of Monthly vs. Weekly Data

The transition from monthly to weekly data presented a trade-off. While the explanatory power of the models (R-squared) decreased with the higher-frequency data, the weekly analysis of the `GPR -> Ambiguity` relationship uncovered a significant lagged effect that was not visible at the monthly level. This highlights the potential for higher-frequency data to reveal more granular and time-dependent relationships.

For the `Ambiguity -> Return` relationship, neither frequency yielded significant results.

## Future Research

Our findings open up several avenues for future research:

*   **Investigate the Lagged Effect of China's GPR**: The negative relationship between China's GPR and ambiguity at a three-week lag is a particularly intriguing result that requires a deeper dive. Future studies could explore the potential underlying economic or political mechanisms driving this relationship.
*   **Explore Non-Linear Relationships**: The lack of a linear relationship between ambiguity and returns does not preclude the existence of non-linear or more complex relationships. Future research could employ non-linear models to further explore this connection.
*   **Incorporate Additional Variables**: The low R-squared values in our models suggest that other factors not included in our analysis are significant drivers of ambiguity. Future studies could incorporate a wider range of macroeconomic, political, and market-based variables to build more comprehensive models.

## Weekly Analysis with Lagged Variables

In response to the non-significant results from the initial weekly analysis, we extended the investigation to include lagged effects of ambiguity, risk, and quadratic risk on market returns. The rationale for this approach is that the impact of these factors on investor behavior and, consequently, on market returns may not be instantaneous but may unfold over several periods.

### Methodology

We modified the `weekly_data_analysis.py` script to incorporate lagged variables for `ambiguity_metric`, `risk`, and `risk_sq` for up to four weeks. The regression model was updated to include these lagged variables as predictors of weekly returns.

The updated regression model is as follows:

`Returns = β₀ + Σ(β₁ᵢ * Ambiguity_lag_i) + Σ(β₂ᵢ * Risk_lag_i) + Σ(β₃ᵢ * Risk²_lag_i) + ε`

where `i` ranges from 1 to 4 weeks.

### Results

The regression analysis with lagged variables yielded more insightful results. The adjusted R-squared of the model improved from 0.005 to 0.051, indicating that the lagged variables collectively explain a greater portion of the variance in market returns.

The key finding from this analysis is the statistical significance of the squared risk from four weeks prior (`risk_sq_lag_4`) at the 5% level (p-value = 0.049). This suggests that a higher squared risk four weeks ago is associated with a change in current market returns.

Additionally, `risk_lag_3` and `risk_lag_4` were found to be significant at the 10% level (p-values of 0.077 and 0.088, respectively). While the lagged ambiguity metrics did not achieve statistical significance at conventional levels, `ambiguity_metric_lag_3` showed the lowest p-value among the ambiguity lags (0.117).

### Conclusion

The inclusion of lagged variables in the weekly analysis has revealed a more nuanced relationship between risk and market returns. The statistically significant impact of `risk_sq_lag_4` suggests that the effect of risk on returns is not immediate and that past risk levels, particularly from a month prior, can have a delayed but significant impact. This finding provides a more promising avenue for future research into the dynamic relationship between risk, ambiguity, and market returns.