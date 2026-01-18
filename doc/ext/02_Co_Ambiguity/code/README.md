# Co-Ambiguity Analysis - Code Documentation

## Overview

This directory contains Python implementation for Systemic Co-Ambiguity (SCA) analysis as described in the research paper. The code implements the SCA index construction, hypothesis testing for five research hypotheses, and trading strategy backtesting for out-of-sample validation.

## File Structure

```
code/
├── sca_measurement.py       # Core SCA computation module
├── hypothesis_testing.py    # Hypothesis testing implementation
├── backtest_analysis.py     # Trading strategy backtesting
├── main_pipeline.py         # Complete analysis pipeline
└── README.md                # This file
```

## Module Descriptions

### 1. sca_measurement.py

**Purpose**: Implements Systemic Co-Ambiguity Index computation

**Key Classes**:
- `SystemicCoAmbiguity`: Main class for SCA calculation

**Key Methods**:
- `compute_sca()`: Compute SCA using pairwise correlations
- `compute_sca_efficient()`: Vectorized efficient computation
- `_compute_unweighted_sca()`: Average pairwise correlation
- `_compute_weighted_sca()`: Market-cap-weighted SCA
- `compute_network_metrics()`: Additional network-based metrics

**Mathematical Foundation**:
```
SCA_t = (2 / (N*(N-1))) * Σ_{i<j} Corr_t(A_i, A_j)
```

**Usage Example**:
```python
from sca_measurement import SystemicCoAmbiguity

# Initialize
sca_calc = SystemicCoAmbiguity(corr_window=60, weighted=False)

# Compute SCA
sca_series = sca_calc.compute_sca_efficient(ambiguity_df)
```

### 2. hypothesis_testing.py

**Purpose**: Implements statistical tests for all five research hypotheses

**Key Classes**:
- `CoAmbiguityHypothesisTests`: Complete hypothesis testing suite

**Hypothesis Tests**:

#### Hypothesis 1: Leading Indicator
**Method**: `test_hypothesis_1_leading_indicator()`

**Tests**: Whether SCA exhibits statistically significant increases before financial crises

**Implementation**:
1. Identify crisis events (market drawdown > 5% in 5-day window)
2. Examine SCA behavior in [-30, +10] day windows around events
3. Test statistical significance using one-sided t-tests
4. Compare lead times with traditional indicators (VIX, correlation, CoVaR)

**Outputs**:
- t-statistics and p-values for pre-event SCA increases
- Average lead times (days)
- Comparison with traditional indicators

**Code Mapping**:
```python
# Test if SCA leads crises
h1_results = tester.test_hypothesis_1_leading_indicator()
# Expected: t_stat > 2.5, p_value < 0.01
```

#### Hypothesis 2: Incremental Power
**Method**: `test_hypothesis_2_incremental_power()`

**Tests**: Whether SCA provides incremental explanatory power beyond traditional measures

**Implementation**:
1. Construct crash indicators for horizons k = 5, 10, 20 days
2. Estimate baseline logistic regression (VIX, correlation, CoVaR)
3. Estimate augmented model (baseline + SCA)
4. Compute likelihood ratio tests and incremental pseudo-R²
5. Generate ROC curves and compare AUC

**Outputs**:
- SCA coefficient, standard error, t-statistic, p-value
- Likelihood ratio test statistics and p-values
- Incremental pseudo-R²
- AUC comparison (baseline vs. augmented)

**Code Mapping**:
```python
# Test incremental power
h2_results = tester.test_hypothesis_2_incremental_power()
# Expected: sca_p_value < 0.05, incremental_r2 > 0.10
```

#### Hypothesis 3: Liquidity Channel
**Method**: `test_hypothesis_3_liquidity_channel()`

**Tests**: Whether SCA predicts liquidity deterioration

**Implementation**:
1. Prepare liquidity metrics (spread, depth, turnover)
2. Estimate VAR models for lags L = 1, 5, 10 days
3. Compute Granger causality F-tests (SCA → liquidity)
4. Test reverse causality (liquidity → SCA)

**Outputs**:
- F-statistics and p-values for SCA → liquidity causality
- Comparison with reverse causality

**Code Mapping**:
```python
# Test liquidity channel
h3_results = tester.test_hypothesis_3_liquidity_channel()
# Expected: F-statistic > 10, p_value < 0.01 for SCA → liquidity
```

#### Hypothesis 4: Structural Change
**Method**: `test_hypothesis_4_structural_change()`

**Tests**: Whether SCA's predictive power is stronger during structural transitions

**Implementation**:
1. Classify periods as structural change vs. stable
2. Estimate interaction regression: SCA × StructuralChange
3. Test significance of interaction term
4. Compare performance across regimes (pseudo-R², AUC)

**Outputs**:
- Interaction coefficient, t-statistic, p-value
- Performance metrics for structural change vs. stable periods

**Code Mapping**:
```python
# Test structural change effects
structural_periods = [('2020-01-01', '2020-06-01')]  # Example
h4_results = tester.test_hypothesis_4_structural_change(structural_periods)
# Expected: interaction_p_value < 0.05, sc_r2 > stable_r2
```

#### Hypothesis 5: Volatility Lead
**Method**: `test_hypothesis_5_volatility_lead()`

**Tests**: Whether SCA Granger-causes volatility measures

**Implementation**:
1. Prepare volatility measures (VIX, realized volatility)
2. Estimate VAR models for lags L = 1, 5, 10, 20 days
3. Compute bidirectional Granger causality tests
4. Generate impulse response functions

**Outputs**:
- F-statistics and p-values for SCA → volatility
- Comparison with reverse causality
- Impulse response coefficients

**Code Mapping**:
```python
# Test volatility lead-lag
h5_results = tester.test_hypothesis_5_volatility_lead()
# Expected: sca_to_vol_p_value < 0.01, vol_to_sca_p_value > 0.10
```

### 3. backtest_analysis.py

**Purpose**: Implements trading strategy backtesting and validation

**Key Classes**:
- `SCABacktester`: Strategy backtesting engine

**Key Methods**:
- `generate_dynamic_threshold_signals()`: Dynamic threshold signals
- `generate_static_threshold_signals()`: Static threshold signals
- `backtest_strategy()`: Strategy performance evaluation
- `compare_strategies()`: Multi-strategy comparison
- `compute_signal_efficiency()`: ROC/AUC analysis
- `walk_forward_validation()`: Out-of-sample validation

**Performance Metrics**:
- Annualized Return
- Sharpe Ratio
- Calmar Ratio
- Sortino Ratio
- Maximum Drawdown
- Hit Rate

**Signal Efficiency Metrics**:
- Accuracy, Precision, Recall, F1 Score
- ROC curves and AUC
- False Positive Rate, False Negative Rate

**Usage Example**:
```python
from backtest_analysis import SCABacktester

# Initialize
backtester = SCABacktester(sca_series, market_returns)

# Generate signals
signals = backtester.generate_dynamic_threshold_signals(
    lookback_window=252,
    percentile_threshold=90
)

# Backtest
results = backtester.backtest_strategy(signals)
# Expected: calmar_ratio > 2.0, max_drawdown < 0.15
```

### 4. main_pipeline.py

**Purpose**: Orchestrates complete end-to-end analysis

**Key Classes**:
- `CoAmbiguityResearchPipeline`: Complete analysis pipeline

**Key Methods**:
- `load_data()`: Load or generate sample data
- `compute_sca()`: Compute SCA index
- `run_hypothesis_tests()`: Execute all five hypothesis tests
- `run_backtests()`: Perform strategy validation
- `visualize_results()`: Generate comprehensive plots
- `generate_report()`: Create text report
- `run_complete_pipeline()`: Execute full pipeline

**Usage Example**:
```python
from main_pipeline import CoAmbiguityResearchPipeline

# Initialize
pipeline = CoAmbiguityResearchPipeline(
    data_path=None,  # None for sample data
    corr_window=60,
    weighted_sca=False
)

# Run complete pipeline
results = pipeline.run_complete_pipeline()
```

## Data Requirements

### Input Data Format

1. **Ambiguity Indices** (pandas DataFrame):
   - Index: Dates (daily frequency)
   - Columns: Stock identifiers
   - Values: Individual A_CEA_t indices

2. **Returns** (pandas DataFrame):
   - Index: Dates
   - Columns: Stock identifiers
   - Values: Daily log returns

3. **Market Index** (pandas Series):
   - Index: Dates
   - Values: Market index levels or returns

4. **Liquidity Measures** (pandas DataFrame):
   - Index: Dates
   - Columns: ['Spread', 'Turnover', 'Depth']
   - Values: Daily liquidity metrics

5. **Volatility** (pandas Series):
   - Index: Dates
   - Values: Realized volatility

6. **VIX** (pandas Series):
   - Index: Dates
   - Values: Implied volatility (optional)

### Data Preprocessing

```python
# Example: Compute daily returns from prices
daily_returns = prices.pct_change().fillna(0)

# Example: Compute realized volatility
rv = returns.rolling(window=20).std()

# Example: Compute market index
market_index = (1 + returns.mean(axis=1)).cumprod()
```

## Algorithm Details

### SCA Computation

1. **Standardization**: For each stock, standardize ambiguity series to zero mean and unit variance within rolling window

2. **Correlation Matrix**: Compute pairwise correlation matrix using efficient vectorized operations

3. **Average Correlation**: Extract upper triangle (excluding diagonal) and compute average

4. **Weighted Version**: For market-cap-weighted SCA, compute: SCA = w'Rw - Σw_i²

### Computational Complexity

- **Naive approach**: O(N² × W) where N = number of stocks, W = correlation window
- **Vectorized approach**: O(N²) using matrix operations
- **Memory**: O(N²) for correlation matrix

### Optimization Techniques

1. **Vectorization**: Use NumPy broadcasting for matrix operations
2. **Rolling Windows**: Efficient pandas rolling computations
3. **Parallelization**: Potential for parallel computation across stocks
4. **Sparse Matrices**: For large universes, use sparse correlation matrices

## Output Specifications

### Hypothesis Test Results

Each hypothesis test returns a dictionary with specific outputs:

```python
# Hypothesis 1
{
    'events': [crisis_dates],
    'n_events': n,
    't_stat': t_statistic,
    'p_value': p_value,
    'comparison_with_indicators': {...}
}

# Hypothesis 2
{
    5: {  # horizon in days
        'sca_coefficient': beta,
        'sca_p_value': p_value,
        'incremental_r2': delta_r2,
        'augmented_auc': auc_augmented,
        'auc_improvement': auc_delta
    },
    ...
}

# Hypothesis 3
{
    'Spread': {
        'lag_1': {'f_statistic': F, 'p_value': p},
        'lag_5': {'f_statistic': F, 'p_value': p},
        ...
    },
    ...
}

# Hypothesis 4
{
    'interaction_coefficient': beta_int,
    'interaction_p_value': p_int,
    'sc_sca_coef': beta_sc,
    'stable_sca_coef': beta_stable
}

# Hypothesis 5
{
    'lag_1': {
        'sca_to_vol_f_stat': F_forward,
        'sca_to_vol_p_value': p_forward,
        'vol_to_sca_f_stat': F_reverse,
        'vol_to_sca_p_value': p_reverse
    },
    ...
}
```

### Backtest Results

```python
{
    'strategy_comparison': {
        'SCA_L252_P90': {
            'annualized_return': annual_ret,
            'sharpe_ratio': sharpe,
            'calmar_ratio': calmar,
            'max_drawdown': max_dd,
            ...
        },
        ...
    },
    'signal_efficiency': {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1
    }
}
```

## Dependencies

```
numpy >= 1.20.0
pandas >= 1.3.0
scipy >= 1.7.0
statsmodels >= 0.13.0
scikit-learn >= 0.24.0
matplotlib >= 3.4.0
seaborn >= 0.11.0
```

## Installation

```bash
pip install numpy pandas scipy statsmodels scikit-learn matplotlib seaborn
```

## Execution

### Run Complete Pipeline

```bash
python code/main_pipeline.py
```

### Run Individual Modules

```python
# Compute SCA only
from sca_measurement import SystemicCoAmbiguity
sca_calc = SystemicCoAmbiguity()
sca = sca_calc.compute_sca_efficient(ambiguity_df)

# Run hypothesis tests only
from hypothesis_testing import CoAmbiguityHypothesisTests
tester = CoAmbiguityHypothesisTests(...)
h1_results = tester.test_hypothesis_1_leading_indicator()

# Run backtests only
from backtest_analysis import SCABacktester
backtester = SCABacktester(...)
results = backtester.backtest_strategy(signals)
```

## Expected Results Summary

Based on the research hypotheses, the code should produce:

### Hypothesis 1: Leading Indicator
- **t-statistic**: > 2.5
- **p-value**: < 0.01
- **Lead time**: 5-20 days before crises
- **Performance**: Better than VIX, correlation, CoVaR

### Hypothesis 2: Incremental Power
- **SCA coefficient**: Positive and significant (p < 0.01)
- **Incremental pseudo-R²**: 0.10-0.20
- **AUC improvement**: 0.10-0.15 over baseline

### Hypothesis 3: Liquidity Channel
- **Granger causality F-stat**: > 10 (SCA → liquidity)
- **p-value**: < 0.01 (SCA → liquidity)
- **Reverse causality**: Weak (p > 0.10)

### Hypothesis 4: Structural Change
- **Interaction term**: Positive and significant (p < 0.05)
- **SC period pseudo-R²**: 40-60% higher than stable

### Hypothesis 5: Volatility Lead
- **Granger causality**: SCA → volatility (p < 0.01)
- **Reverse causality**: Weak (p > 0.10)
- **Lead time**: 5-15 days

### Backtest Performance
- **Calmar Ratio**: 2.3-3.1× buy-and-hold
- **Maximum Drawdown**: 45-65% reduction
- **Signal AUC**: 0.72-0.78

## Troubleshooting

### Common Issues

1. **Insufficient Data Error**:
   - **Cause**: Not enough observations for rolling window
   - **Solution**: Increase data length or reduce window size

2. **Convergence Warning in Logit**:
   - **Cause**: Perfect separation or too few events
   - **Solution**: Add regularization or increase sample size

3. **Granger Causality Error**:
   - **Cause**: Non-stationary data
   - **Solution**: Difference the series or use shorter lags

4. **Memory Error**:
   - **Cause**: Large correlation matrices
   - **Solution**: Reduce number of stocks or use sparse matrices

### Performance Optimization

```python
# Reduce memory usage
sca_calc = SystemicCoAmbiguity(corr_window=30)  # Shorter window

# Subset stocks
ambiguity_subset = ambiguity_df.iloc[:, :100]  # Fewer stocks

# Use efficient computation
sca = sca_calc.compute_sca_efficient(...)  # Not compute_sca()
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{co_ambiguity_2024,
  title={Systemic Co-Ambiguity: A Novel Early Warning Signal
         for Financial Crises Based on Uncertainty Synchronization},
  author={[Authors]},
  journal={Journal of Financial Stability},
  year={2024}
}
```

## Contact

For questions or issues, please contact [Author Information].

## License

[License Information]
