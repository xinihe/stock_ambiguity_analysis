# Daily Data Analysis Report

## Abstract
This report investigates the empirical relationships between daily stock returns and measures of ambiguity, risk, and geopolitical risk. Using a combined daily dataset constructed from four sources, we estimate two ordinary least squares (OLS) models. The first model examines a non-linear risk–return specification that includes a quadratic term in risk, while the second model assesses a combined specification in which ambiguity, risk, and the Geopolitical Risk Index (GPRD) jointly explain daily returns. We find a robust negative association between ambiguity and daily returns across models. In the combined model, GPRD also exhibits a negative and statistically significant relationship with returns. R-squared values are small, which is typical for high-frequency financial returns, but the signs and significance of key coefficients are stable and economically interpretable.

## Data and Sample
- Source integration: `com_daily_data.csv` merges daily ambiguity metrics, risk measures, GPR data, and SSE 300 index prices.
- Sample period: 2018-01-09 to 2024-12-10.
- Unit of analysis: Daily observations; the dependent variable is the daily return computed from SSE 300 prices. Ambiguity metrics and risk measures are taken directly from the merged dataset; GPRD is the daily GPR index.
- Observations used in estimation: 1,679 for each regression (after listwise deletion).
 - Observations used in estimation: 1,679 for each regression (after list-wise deletion).

## Empirical Strategy
- Estimator: OLS with conventional standard errors.
- Specification 1 (Non-linear risk–return):
  `Returns = β₀ + β₁(Ambiguity) + β₂(Risk) + β₃(Risk²) + ε`
- Specification 2 (Combined model for daily returns):
  `Returns = β₀ + β₁(Ambiguity) + β₂(Risk) + β₃(GPRD) + ε`

## Results
The tables below report OLS estimates in a STATA-style presentation, with standard errors in parentheses and significance denoted by stars. Coefficients on ambiguity are negative and statistically significant in both models. In Model 1, the linear risk term is positive and marginally significant, while the quadratic term is not. In Model 2, GPRD is negative and statistically significant; the risk term loses statistical significance, suggesting that GPRD may capture overlapping variation related to risk conditions.

### Table 1. Non-Linear Risk–Return (Model 1)
```
-------------------------------------------------
                   Dependent variable: Returns    
                   OLS Estimates (STATA style)    
-------------------------------------------------
Ambiguity (ambiguity_metric_1)     -0.0369***
                                   (0.0102)

Risk (risk_1)                       0.0090*
                                   (0.0047)

Risk² (risk_1_sq)                  -0.1688
                                   (0.1159)

Constant                            0.0001
                                   (0.0001)

Observations                        1,679
R-squared                           0.0098
Adjusted R-squared                  0.0080
-------------------------------------------------
Standard errors in parentheses. * p<0.10, ** p<0.05, *** p<0.01
```

Interpretation: Ambiguity exhibits a negative association with returns that is statistically significant at the 1% level, consistent with ambiguity aversion dampening expected returns. The positive linear risk term is marginally significant, while the quadratic term is not, suggesting that within this sample and horizon, a linear risk–return trade-off is adequate.

### Table 2. Combined Model for Daily Returns (Model 2)
```
-------------------------------------------------
                   Dependent variable: Returns    
                   OLS Estimates (STATA style)    
-------------------------------------------------
Ambiguity (ambiguity_metric_1)     -0.0346***
                                   (0.0102)

Risk (risk_1)                       0.0030
                                   (0.0020)

GPRD (geopolitical risk)           -0.0000***
                                   (0.0000)

Constant                            0.0002***
                                   (0.0001)

Observations                        1,679
R-squared                           0.0133
Adjusted R-squared                  0.0116
-------------------------------------------------
Standard errors in parentheses. * p<0.10, ** p<0.05, *** p<0.01
```

Interpretation: Ambiguity remains negatively associated with returns and statistically significant. The GPRD coefficient is small in magnitude but negative and statistically significant, consistent with higher geopolitical risk being associated with lower daily returns. The risk proxy loses significance when GPRD enters, suggesting overlapping explanatory content or multi-collinearity among risk-related covariates.

## Discussion and Limitations
- Low R-squared values are expected at the daily frequency due to high idiosyncratic noise and the limited explanatory power of macro/uncertainty variables for short-horizon returns.
- Coefficient signs are stable across specifications; ambiguity and GPRD negatively load on daily returns.
 - Estimates use conventional OLS standard errors. Robust or heteroscedasticity and autocorrelation consistent (HAC) standard errors may be considered to address potential heteroscedasticity and autocorrelation at daily horizons.
- Future work could examine interactions (e.g., ambiguity × risk), alternative risk proxies, and regime-dependent specifications.

## Conclusion
Across two models, ambiguity significantly and negatively relates to daily returns, and geopolitical risk (GPRD) is also negatively associated with returns when included alongside ambiguity and risk. While explanatory power is modest—as expected for daily returns—the results are statistically robust and economically consistent with the hypothesis that uncertainty and geopolitical risk depress expected returns at high frequencies.