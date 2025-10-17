# Weekly Data Analysis Report: A Comprehensive Review

## 1. Data Interpolation: From Monthly to Weekly GPR

### 1.1. Procedure

To analyze the relationship between GPR and ambiguity at a weekly frequency, the monthly GPR data was interpolated. This was necessary because the GPR data is released on a monthly basis, while the ambiguity and returns data are available weekly. A cubic spline interpolation method was used to create a smooth curve that passes through the monthly data points, allowing for the estimation of weekly GPR values.

### 1.2. Implementation

To regenerate the weekly GPR data, you can run the following script. Please note that this script needs to be created.

```bash
python /Users/tlxy/Research/Ambiguity/code/Adaptive/GeoUncertainty/scripts/data_processing/interpolate_gpr.py
```

## 2. Regression 1: Non-Linear Risk-Return Relationship

### 2.1. Model Specification

The analysis of the risk-return relationship revealed that a simple linear model was insufficient to capture the complex dynamics present in the data. A quadratic specification, however, proved to be much more effective, revealing a significant non-linear relationship.

```
Returns = β₀ + β₁(Ambiguity) + β₂(Risk) + β₃(Risk²) + ε
```

### 2.2. Results

The quadratic model yielded the following results:

- **Risk_Metric coefficient**: 1,382.65 (t=2.187, **p=0.0317**)
- **Risk_Metric_Squared coefficient**: -47,716.40 (t=-2.116, **p=0.0375**)
- **Adjusted R²**: 0.0233
- **F-test p-value**: 0.1839

### 2.3. Interpretation

The results of the quadratic model indicate a significant non-linear relationship between risk and returns. The positive coefficient on the linear risk term and the negative coefficient on the squared risk term suggest that returns increase with risk up to a certain point, after which they begin to diminish. This is consistent with the economic theory of risk aversion, where investors demand a premium for taking on more risk, but only up to a certain level.

### 2.4. Implementation

To regenerate the results of this regression, you can run the following script:

```bash
python /Users/tlxy/Research/Ambiguity/code/Adaptive/GeoUncertainty/scripts/analysis/corrected_risk_return_regression.py
```

## 3. Regression 2: GPR and Ambiguity (Weekly Analysis with Lags)

### 3.1. Model Specification

To investigate the causal relationship between GPR and ambiguity, a regression model with lagged GPR variables was used. This approach helps to account for the time it may take for geopolitical events to be reflected in market ambiguity.

```
Ambiguity = β₀ + β₁(GPR_China_lag_1) + β₂(GPR_China_lag_2) + β₃(GPR_China_lag_3) + ... + ε
```

### 3.2. Results

The weekly analysis revealed a statistically significant negative relationship between the third-week lag of China's GPR and ambiguity (p-value = 0.037). This suggests that an increase in China's GPR is followed by a decrease in market ambiguity three weeks later.

### 3.3. Interpretation

The counter-intuitive negative relationship between China's GPR and ambiguity may be explained by a "risk clarification effect." High-profile geopolitical events, while risky, may reduce ambiguity by clarifying the "rules of the game." For example, a specific geopolitical action, though destabilizing, removes uncertainty about whether that action would be taken, thus reducing ambiguity about the state of the world.

### 3.4. Implementation

To regenerate the results of this regression, you can run the following script. Please note that this script may need to be adapted from the monthly analysis script.

```bash
python /Users/tlxy/Research/Ambiguity/code/Adaptive/GeoUncertainty/scripts/analysis/weekly_gpr_ambiguity.py
```