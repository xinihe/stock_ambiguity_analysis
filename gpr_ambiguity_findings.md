# The Ambiguity Channel of Geopolitical Risk: An Empirical Investigation

## 1. Introduction

This study investigates the causal pathway from Geopolitical Risk (GPR) to market returns, mediated by ambiguity. We propose the following causal chain:

**GPR → Ambiguity → Market Returns**

This paper focuses on the first link of this chain: the impact of GPR on ambiguity.

## 2. Methodology

We employ a series of regression models to analyze the relationship between GPR and ambiguity. The models are estimated using monthly data from January 2018 to April 2024.

*   **Data:**
    *   **GPR:** Country-specific GPR indices for the US, China, and Japan.
    *   **Ambiguity:** A proxy for market ambiguity derived from high-frequency financial data.

*   **Models:**
    1.  **Linear Model:** A baseline model to assess the direct linear relationship.
    2.  **Interaction Model:** To explore how GPR from different countries jointly affects ambiguity.
    3.  **Quadratic Model:** To capture potential non-linearities in the GPR-ambiguity relationship.
    4.  **Lag Model:** To investigate the delayed effects of GPR on ambiguity.

## 3. Empirical Results (Monthly Analysis)

The empirical results from our monthly analysis provide strong evidence for the GPR-ambiguity link.

### 3.1. Model Comparison

| Model       | R-squared |
| :---------- | :-------- |
| Linear      | 0.0468    |
| Interaction | 0.0532    |
| Quadratic   | 0.1962    |
| **Lag**     | **0.2738**|

The **Lag Model** provides the best fit to the data, with an R-squared of 27.38%, suggesting that the impact of GPR on ambiguity is not contemporaneous but unfolds over time. The Quadratic model also shows a significant improvement over the linear and interaction models, indicating the presence of non-linear effects.

### 3.2. Analysis of GPR Effects

*   **Non-Linearity:** The superior performance of the Quadratic and Lag models suggests that the relationship between GPR and ambiguity is complex and not adequately captured by a simple linear model.
*   **Lagged Effects:** The Lag Model, which includes lagged values of GPR, provides the best explanation for the variation in ambiguity. This indicates that it takes time for geopolitical events to be fully priced into the market and affect ambiguity.
*   **Country-Specific Effects:** The analysis of individual GPR measures reveals that **China's GPR** has the most significant and consistent negative relationship with ambiguity. This suggests that increases in geopolitical risk originating from China tend to *decrease* market ambiguity. This counterintuitive finding warrants further investigation.

## 4. Weekly Analysis

To further investigate the relationship between GPR and ambiguity, we conducted a weekly analysis by interpolating the monthly GPR data. This allows for a more granular view of the dynamic relationship.

### 4.1. Model Comparison (Weekly)

| Model       | R-squared |
| :---------- | :-------- |
| Linear      | 0.0133    |
| **Lag**     | **0.0883**|

While the overall explanatory power of the models is lower in the weekly analysis, the Lag Model still outperforms the Linear Model, consistent with the monthly findings.

### 4.2. Analysis of GPR Effects (Weekly)

The most striking result from the weekly analysis is the statistically significant negative relationship between the **third-week lag of China's GPR** and ambiguity (p-value = 0.037). This finding reinforces the monthly results and provides more precise timing on the lagged effect of China's GPR. The negative coefficient suggests that increases in GPR from China are followed by a decrease in market ambiguity three weeks later.

## 5. Future Research

The findings from this study provide a solid foundation for the second stage of our proposed research: investigating the causal link from ambiguity to market returns. Future research should:

*   **Rigorously test the full causal pathway (GPR → Ambiguity → Market Returns)** using advanced econometric methods such as mediation analysis and structural equation modeling.
*   **Explore the mechanisms** through which GPR in China reduces market ambiguity.
*   **Investigate non-linear dynamics** in the GPR-ambiguity relationship more deeply.