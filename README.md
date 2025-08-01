# **📊 Ambiguity and Excess Returns in Financial Markets**

## **Overview**

This research project investigates the role of **ambiguity**—uncertainty about probability distributions—in influencing **excess returns** in financial markets. We explore whether ambiguity contributes to excess returns and how its impact varies across different market conditions.

## **Objectives**

1. **Define and Measure Ambiguity**: Explore various definitions and computational methods to quantify ambiguity in stock prices.
2. **Analyze Ambiguity’s Properties**: Examine statistical characteristics of ambiguity, its relationship with risk, and other influencing factors.
3. **Assess Impact on Excess Returns**: Determine how ambiguity affects excess returns under varying market scenarios.
4. **Understand Investor Behavior**: Investigate investor preferences and behaviors in response to ambiguity-induced uncertainties.

## **Methodology**

### **1. Defining and Measuring Ambiguity**

* **Multiple Definitions**: Evaluate different theoretical frameworks for ambiguity, including:
  * **Knightian Uncertainty**: Differentiating between measurable risk and unmeasurable uncertainty.
  * **Ambiguity Aversion Models**: Such as the Maxmin Expected Utility model.
* **Computational Approaches**: Implement various methods to calculate ambiguity metrics, for example:
  * **Probability Distribution Variance**: Assessing the variability in estimated probabilities over time. 
  * **Entropy-Based Measures**: Quantifying the unpredictability in return distributions.

### **2. Analyzing Ambiguity’s Properties**

* **Statistical Analysis**: Study the distribution, volatility, and autocorrelation of ambiguity measures.
* [Risk Relationship:](code/ProbVar/readme.md) Examine correlations between ambiguity and traditional risk metrics like standard deviation and beta.
* **Influencing Factors**: Identify macroeconomic and firm-specific variables that may affect or be affected by ambiguity.

### **3. Assessing Impact on Excess Returns**

* **Regression Analysis**: Incorporate ambiguity measures into asset pricing models to evaluate their explanatory power for excess returns.
* **Market Condition Segmentation**: Analyze the impact of ambiguity across different market regimes (e.g., bull vs. bear markets).
* **Time-Varying Effects**: Investigate whether the influence of ambiguity on returns changes over time.

### **4. Understanding Investor Behavior**

* **Behavioral Studies**: Review literature on investor responses to ambiguity, including ambiguity aversion and preference.
* **Survey Analysis**: If applicable, analyze survey data to gauge investor sentiment and decision-making under ambiguity.
* **Experimental Approaches**: Design experiments to observe investor choices in ambiguous scenarios.

## Data

[SSE.000300.csv](data/SSE.000300.csv)

This file contains the raw minute-level price data for the CSI 300 Index (SSE 000300), which serves as the basis for all ambiguity and risk calculations in this project.

### Column Descriptions

*datetime_nano*

* Type: Integer or string (nanosecond timestamp)
* Description: The timestamp of each recorded price, in nanoseconds since the Unix epoch (UTC). This high-resolution timestamp allows precise alignment of price data to the exact minute (or finer, if needed).
* Usage in code: Converted to a timezone-aware datetime (Asia/Shanghai), then floored to the nearest minute for analysis.

*SSE.000300.close*

* Type: Float
* Description: The closing price of the CSI 300 Index at the given timestamp.
* Usage in code: Used to compute minute-level returns, which are the foundation for all subsequent ambiguity and risk metrics.

### Typical Data Example

* The first column is the nanosecond timestamp (UTC).
* The second column is the index closing price at that timestamp.

### Processing Steps in Analysis

  Timestamp Conversion:

The datetime_nano column is converted to a pandas datetime object, localized to UTC, then converted to Asia/Shanghai time zone, and finally floored to the nearest minute.

   Return Calculation:

The SSE.000300.close column is used to calculate minute-by-minute returns, which are then aggregated and analyzed for ambiguity and risk.

*Note:* The file should be placed in the data/ directory (or the path specified in your code). Only the columns datetime_nano and SSE.000300.close are required for the analysis pipeline.*

[daily_ambiguity_var_para.csv](data/daily_ambiguity_var_para.csv)

This file contains a comprehensive panel of daily ambiguity metrics for the SSE 300 Index, calculated under a wide range of parameter settings. It is generated by systematically varying both the rolling window length and the number of bins used in the ambiguity calculation, providing a rich resource for robustness checks and further empirical analysis.

### Structure

* Rows: Each row corresponds to a unique trading day (date).
* Columns:
* date: The trading date (YYYY-MM-DD).
* Other columns: Each remaining column represents a unique combination of rolling window length and number of bins, named in the format **{window_size}d{num_bins}b**. For example, 5d30b means the ambiguity metric was calculated using a 5-day rolling window and 30 bins for the return distribution histogram.

### Parameter Grid

* Rolling Window Lengths (d): 5, 8, 11, 14, 17, 20 days
* Number of Bins (b): 20, 30, 40, 50, 60, 70, 80, 90, 100

All possible combinations of these parameters are included, resulting in a wide matrix of ambiguity metrics for each date.

### Calculation Methodology

For each date and each parameter combination:* The ambiguity metric is computed as the average standard deviation across bins of the daily return probability distributions, by the code in [cal_ambi_pv.ipynb](code/ProbVar/cal_ambi_pv.ipynb), using the specified rolling window and bin count.

* If there is insufficient historical data for a given window size on a particular date, the value is recorded as NaN.

### Usage

This dataset enables:* Robustness checks of empirical findings to different ambiguity calculation settings.

* Sensitivity analysis of how ambiguity metrics respond to methodological choices.
* Comparative studies of ambiguity’s predictive power under various parameterizations.
