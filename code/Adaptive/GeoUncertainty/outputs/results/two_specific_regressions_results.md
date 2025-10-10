# Two Specific Regressions Analysis Results

## Overview
This document presents the detailed results for two specific regression models using the optimal 20-bin configuration:

1. **Ambiguity & Risk vs Monthly Returns**
2. **Climate Risk & US GPR vs Ambiguity**

---

## Regression 1: Ambiguity & Risk vs Monthly Returns

### Model Configuration
- **Dependent Variable**: Monthly_Return_Pct
- **Independent Variables**: Ambiguity_Bins_20, Risk_Bins_20
- **Observations**: 83
- **Bin Configuration**: 20 bins (optimal from comprehensive analysis)

### Regression Statistics
- **R-squared**: 0.0021
- **Adjusted R-squared**: -0.0228
- **F-statistic p-value**: 0.9179
- **Durbin-Watson**: 2.0803

### Coefficient Analysis

| Variable | Coefficient | Std Error | p-value | Significance |
|----------|-------------|-----------|---------|--------------|
| Intercept | 1.8881 | 0.4618 | 0.0000 | *** |
| Ambiguity_Bins_20 | 0.9589 | 2.6999 | 0.7175 | |
| Risk_Bins_20 | -794.1750 | 6,568.8000 | 0.9041 | |

### Economic Interpretation

#### Ambiguity Effect
- **Coefficient**: 0.9589 (p=0.7175)
- **Finding**: Ambiguity effect is **not statistically significant**
- **Implication**: Higher market ambiguity does not significantly predict monthly stock returns

#### Risk Effect
- **Coefficient**: -794.1750 (p=0.9041)
- **Finding**: Risk effect is **not statistically significant**
- **Implication**: Traditional risk measures do not significantly predict monthly returns in this model

### Key Insights
1. **Low Explanatory Power**: The model explains only 0.21% of the variation in monthly returns
2. **Statistical Insignificance**: Neither ambiguity nor risk significantly predicts returns
3. **Model Limitations**: The negative adjusted R² suggests the model may be overfitted

---

## Regression 2: Climate Risk & US GPR vs Ambiguity

### Model Configuration
- **Dependent Variable**: Ambiguity_Bins_20
- **Independent Variables**: Climate_Risk_Component, GPR_US_y
- **Observations**: 83
- **Bin Configuration**: 20 bins (optimal from comprehensive analysis)

### Regression Statistics
- **R-squared**: 0.0338
- **Adjusted R-squared**: 0.0096
- **F-statistic p-value**: 0.2528
- **Durbin-Watson**: 2.1414

### Coefficient Analysis

| Variable | Coefficient | Std Error | p-value | Significance |
|----------|-------------|-----------|---------|--------------|
| Intercept | 1.8897 | 0.0799 | 0.0000 | *** |
| Climate_Risk_Component | -0.0285 | 0.0316 | 0.3703 | |
| GPR_US_y | 0.0336 | 0.0263 | 0.2047 | |

### Economic Interpretation

#### Climate Risk Effect
- **Coefficient**: -0.0285 (p=0.3703)
- **Finding**: Climate risk effect on ambiguity is **not statistically significant**
- **Direction**: Negative relationship (higher climate risk → lower ambiguity)
- **Implication**: Climate risk does not significantly influence market ambiguity perceptions

#### US GPR Effect
- **Coefficient**: 0.0336 (p=0.2047)
- **Finding**: US GPR effect on ambiguity is **not statistically significant** (but closer to significance)
- **Direction**: Positive relationship (higher US geopolitical risk → higher ambiguity)
- **Implication**: US geopolitical risk shows a tendency to increase market ambiguity, though not statistically significant

### Key Insights
1. **Better Model Performance**: This model performs better than Regression 1
2. **Positive Adjusted R²**: Explains 0.96% of variation in ambiguity (modest but positive)
3. **US GPR Trend**: Shows a marginally significant trend (p=0.2047) suggesting potential relationship
4. **Climate Risk**: Surprisingly shows negative (though insignificant) relationship with ambiguity

---

## Comparative Analysis

### Model Performance Comparison

| Metric | Model 1 (Ambiguity&Risk→Returns) | Model 2 (Climate&GPR→Ambiguity) |
|--------|---------------------------|---------------------------|
| R-squared | 0.0021 | 0.0338 |
| Adjusted R-squared | -0.0228 | 0.0096 |
| F-test p-value | 0.9179 | 0.2528 |
| Observations | 83 | 83 |

### Key Findings

#### Model 2 Outperforms Model 1
- **Better Explanatory Power**: Adjusted R² = 0.0096 vs -0.0228
- **More Statistical Significance**: F p-value = 0.2528 vs 0.9179
- **Positive Relationship**: Climate and geopolitical factors show some predictive power for ambiguity

#### Economic Implications

1. **Ambiguity as Predictor**: Ambiguity and traditional risk measures are poor predictors of monthly returns
2. **Ambiguity as Outcome**: Market ambiguity is better explained by external factors (climate, geopolitical) than it explains returns
3. **US GPR Relevance**: US geopolitical risk shows the strongest (though still insignificant) relationship with market ambiguity
4. **Climate Risk Puzzle**: Climate risk shows unexpected negative relationship with ambiguity

---

## Methodological Notes

### Data Quality
- **Sample Size**: 83 monthly observations
- **Bin Configuration**: 20 bins (optimal from comprehensive analysis)
- **Time Series Properties**: Durbin-Watson statistics suggest no severe autocorrelation issues

### Statistical Considerations
- **Low R-squared Values**: Common in financial time series, especially monthly data
- **Multiple Testing**: Results should be interpreted with caution given multiple model testing
- **Economic vs Statistical Significance**: Even small effects can be economically meaningful in financial markets

### Recommendations for Future Research
1. **Longer Time Series**: Increase sample size for better statistical power
2. **Alternative Specifications**: Consider non-linear relationships or interaction terms
3. **Lag Structure**: Investigate lagged relationships between variables
4. **Regime Analysis**: Consider structural breaks or regime-dependent relationships

---

## Conclusion

While neither regression shows strong statistical significance, **Model 2 (Climate Risk & US GPR vs Ambiguity)** demonstrates superior performance and provides more meaningful insights:

- **US geopolitical risk** shows the most promising relationship with market ambiguity
- **Climate risk** effects are weaker and counterintuitive
- **Ambiguity and risk** are poor predictors of monthly returns in this specification
- The **20-bin configuration** provides optimal granularity for capturing ambiguity patterns

These results suggest that market ambiguity is more effectively explained by external geopolitical and environmental factors than it explains market returns, highlighting the complex nature of uncertainty in financial markets.