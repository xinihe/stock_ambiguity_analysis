# Empirical Testing Routine for Ambiguity and Global Uncertainty Analysis

## 1. Research Background and Objectives

This research examines the relationship between ambiguity metrics, global uncertainty indices, and market performance. Specifically, we investigate:

- How ambiguity metrics correlate with country-specific geopolitical risk (GPR) indices for China, Japan, and the US
- The impact of climate risk data on market volatility and ambiguity
- Causal relationships between various uncertainty measures and market outcomes
- Methods to enhance weak causal relationships between these variables

## 2. Data Sources and Processing

### 2.1 Data Sources

- **SSE300 Index**: Minute-level data converted to daily frequency
- **Ambiguity Risk Metrics**: Daily ambiguity and risk measurements
- **Combined Global Uncertainty Index**: Monthly data converted to daily frequency
- **Climate Risk Series**: Climate risk index data
- **GPR Country Data**: Geopolitical Risk indices for China, Japan, and the US

### 2.2 Data Processing Steps

1. **SSE300 Index Conversion**: Convert minute-level data to daily OHLCV and calculate daily returns
2. **Frequency Alignment**: Convert monthly global uncertainty and GPR data to daily through forward filling
3. **Data Integration**: Merge all datasets on date with appropriate handling of missing values
4. **Enhancement Variables**: Create transformed variables to enhance correlations:
   - Rolling statistics (10, 20, 30-day windows)
   - Lagged variables (1, 3, 5, 10-day lags)
   - Standardization for comparative analysis

## 3. Empirical Testing Methods

### 3.1 Descriptive Statistics

- Summary statistics for all variables
- Distribution analysis and normality testing
- Time series characteristics (trends, seasonality)

### 3.2 Correlation Analysis

- Enhanced correlation matrix including all variables
- Focused analysis on ambiguity metric correlations
- Time-lagged correlation analysis
- Visualization through heatmaps and bar charts

### 3.3 Stationarity Testing

- Augmented Dickey-Fuller (ADF) tests for all time series
- Identification of integrated processes
- Necessary transformations for stationarity

### 3.4 Granger Causality Testing

- Enhanced causality tests with expanded lag structures (up to 15 lags)
- Testing from uncertainty indices to ambiguity metrics
- Country-specific GPR causality analysis
- Visualization of p-values across different lags

### 3.5 Regression Analysis

- Multiple regression models with various specifications:
  1. Impact of ambiguity and risk on market returns
  2. Impact with uncertainty indices as additional predictors
  3. Predicting ambiguity using uncertainty and GPR variables
- Standardized coefficients for comparative analysis
- Model evaluation through R² metrics

## 4. Expected Results and Interpretations

### 4.1 Correlation Expectations
- Modest positive correlation between ambiguity metrics and GPR country indices
- Significant relationship between climate risk and global uncertainty
- Time-varying correlation patterns requiring dynamic analysis

### 4.2 Causality Interpretations
- Potential lagged effects from geopolitical events to ambiguity metrics
- Different causality patterns across countries (China, Japan, US)
- Weak direct causality but stronger indirect relationships

### 4.3 Enhancement of Causal Relationships
- Recommendations for model improvements based on initial findings
- Implementation of advanced methodologies for capturing complex relationships

## 5. Analysis Tools and Software

- **Programming Environment**: Python 3.8+
- **Key Libraries**:
  - pandas, numpy: Data manipulation and numerical operations
  - matplotlib, seaborn: Data visualization
  - statsmodels: Time series analysis and hypothesis testing
  - scikit-learn: Regression modeling and feature scaling

## 6. Output Files

All analysis results will be saved to the `analysis` directory under `GeoUncertainty`:

- `combined_data_analysis.csv`: Integrated dataset with all variables
- `correlation_heatmap.png`: Enhanced correlation matrix visualization
- `ambiguity_correlations.png`: Focused ambiguity correlation analysis
- `granger_causality.png`: Granger causality test results
- `granger_causality_heatmap.png`: Best causality test results
- `regression_analysis.png`: Enhanced regression model results
- `time_series_plots.png`: Individual time series visualizations
- `combined_time_series.png`: Combined analysis of key variables

## 7. Timeline and Process Flow

### 7.1 Data Preparation Phase
- Day 1-2: Data collection and initial cleaning
- Day 3: Frequency conversion and alignment

### 7.2 Analysis Phase
- Day 4-5: Descriptive statistics and correlation analysis
- Day 6-7: Stationarity testing and transformations
- Day 8-9: Granger causality testing
- Day 10-11: Regression modeling and enhancements

### 7.3 Results Interpretation
- Day 12-13: Interpretation of empirical findings
- Day 14: Recommendations for enhancing causal relationships
- Day 15: Final report and documentation

## 8. Enhancement Strategies for Weak Causal Relationships

When causal relationships are identified as weak, the following strategies can be implemented:

1. **Higher Frequency Data**: Utilize intraday data to capture more granular relationships
2. **Advanced Transformations**: Apply log transformations, first differences, and volatility calculations
3. **Non-linear Models**: Implement GARCH models and non-linear causality tests
4. **Expanded Variable Set**: Incorporate macroeconomic indicators and market liquidity measures
5. **Machine Learning Approaches**: Apply Random Forest, Gradient Boosting, and neural network models
6. **Wavelet Analysis**: Decompose time series to analyze relationships at different time scales
7. **Vector Autoregression**: Implement VAR models to capture system-wide dynamics

## 9. Research Limitations and Considerations

- Potential measurement error in ambiguity and uncertainty metrics
- Challenges in frequency alignment across different time series
- Endogeneity concerns in causal interpretation
- Need for robustness checks across different model specifications
- Importance of economic interpretation alongside statistical significance

## 10. Conclusion and Next Steps

This empirical testing routine provides a comprehensive framework for analyzing the relationships between ambiguity metrics, global uncertainty indices, climate risk, and country-specific GPR data. The enhanced methodology incorporates strategies to address weak causal relationships and provides a solid foundation for further research in this area.