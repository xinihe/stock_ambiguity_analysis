# Ambiguity vs. Higher-Order Moments - Code Documentation

## Overview

This directory contains Python implementation for the research examining the distinction between ambiguity (model uncertainty) and higher-order moments (tail risk: skewness and kurtosis) in financial markets. The code implements orthogonality tests, interaction effects for crash prediction, and portfolio backtesting.

## File Structure

```
code/
├── moments_analysis.py      # Core analysis module
├── main_pipeline.py         # Complete analysis pipeline
└── README.md                # This file
```

## Research Hypotheses

The code tests five specific research hypotheses:

1. **H1 (Correlation Orthogonality)**: Ambiguity and moments exhibit low correlation (<0.3)
2. **H2 (Regression Orthogonality)**: Ambiguity regressed on moments yields R² < 10%
3. **H3 (Interaction Effect)**: Ambiguity × Skewness interaction improves crash prediction
4. **H4 (Factor Structure)**: PCA shows ambiguity loads on distinct factor from moments
5. **H5 (Portfolio Value)**: Double-sorted portfolios generate significant alphas

## Module Descriptions

### 1. moments_analysis.py

**Purpose**: Core module implementing all five hypothesis tests

**Key Functions**:

#### `test_hypothesis_1_correlation()`
Tests whether ambiguity and moments are uncorrelated.

**Implementation**:
- Computes daily cross-sectional correlations between ambiguity and each moment (RV, Skew, Kurt)
- Averages correlations across time
- Uses Fisher's z-transformation for confidence intervals
- Tests H0: correlation >= 0.3 (one-sided)

**Code Mapping**:
```python
# Test correlation orthogonality
h1_results = test_hypothesis_1_correlation(ambiguity_df, moments_dict)
# Expected: all correlations < 0.3, p_values < 0.05
```

**Outputs**:
- Mean correlation coefficients
- 95% confidence intervals
- P-values for orthogonality tests
- Confirmation status

#### `test_hypothesis_2_regression()`
Tests whether moments explain little variation in ambiguity.

**Implementation**:
- Estimates time-series regression for each stock:
  `Ambiguity_t = α + β₁·RV_t + β₂·Skew_t + β₃·Kurt_t + ε_t`
- Examines distribution of R² across stocks
- Wilcoxon signed-rank test: H0: median(R²) >= 0.10
- Extracts "pure ambiguity" residuals

**Code Mapping**:
```python
# Test regression orthogonality
h2_results = test_hypothesis_2_regression(ambiguity_df, moments_dict)
# Expected: median_r2 < 0.10, wilcoxon_pvalue < 0.05
```

**Outputs**:
- R² distribution (mean, median, percentiles)
- Wilcoxon test statistics
- Pure ambiguity residuals (orthogonal component)

#### `test_hypothesis_3_interaction()`
Tests whether ambiguity-skewness interaction improves crash prediction.

**Implementation**:
- Defines crash indicator: market return < -5% within 5 days
- Estimates logistic regression models:
  - Model 1 (main effects): logit(Crash) = α + β₁·Skew + β₂·Kurt + β₃·Ambiguity
  - Model 2 (with interaction): + β₄·(Skew × Ambiguity)
- Likelihood ratio test: χ²(1)
- AUC comparison using DeLong's test

**Code Mapping**:
```python
# Test interaction effect
h3_results = test_hypothesis_3_interaction(
    market_moments, market_ambiguity, market_returns
)
# Expected: interaction_p_value < 0.05, auc_improvement > 0.05
```

**Outputs**:
- Main effects and interaction model coefficients
- Likelihood ratio test statistics
- AUC for both models
- Interaction significance

#### `test_hypothesis_4_pca()`
Tests whether ambiguity loads on distinct factor from moments.

**Implementation**:
- Aggregates to market level (averages across stocks)
- Standardizes all variables to zero mean, unit variance
- Performs PCA on [Ambiguity, RV, Skew, Kurt]
- Examines factor loadings
- Computes variance inflation factors (VIF)

**Code Mapping**:
```python
# Test factor structure
h4_results = test_hypothesis_4_pca(ambiguity_df, moments_dict)
# Expected: ambiguity loads heavily (>0.5) on PC2 (distinct from PC1)
```

**Outputs**:
- Eigenvalues and explained variance ratios
- Factor loadings matrix
- Ambiguity's maximum loading and factor
- VIF values

#### `test_hypothesis_5_portfolio()`
Tests economic value through double-sorted portfolios.

**Implementation**:
- First sort: stocks into quintiles by skewness
- Second sort: within lowest skewness quintile, sort by ambiguity
- Forms portfolios: Toxic (low skew + high amb) vs. Stable (high skew + low amb)
- Computes performance metrics (Sharpe, Calmar, etc.)
- Computes Fama-French five-factor alphas

**Code Mapping**:
```python
# Test portfolio value
h5_results = test_hypothesis_5_portfolio(
    ambiguity_df, skew_df, returns_df
)
# Expected: long_short alpha significant (p < 0.05), annual alpha 8-12%
```

**Outputs**:
- Portfolio returns for all groups
- Performance metrics (Sharpe, Sortino, max DD)
- Fama-French alphas and t-statistics
- Long-short strategy results

### 2. main_pipeline.py

**Purpose**: Orchestrates complete end-to-end analysis

**Key Classes**:
- `AmbiguityMomentsResearchPipeline`: Complete analysis pipeline

**Key Methods**:
- `load_data()`: Load or generate sample data
- `run_all_hypothesis_tests()`: Execute all five tests
- `visualize_results()`: Generate comprehensive plots
- `generate_report()`: Create text report
- `run_complete_pipeline()`: Execute full pipeline

**Usage Example**:
```python
from main_pipeline import AmbiguityMomentsResearchPipeline

# Initialize
pipeline = AmbiguityMomentsResearchPipeline(data_path=None)

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
   - Values: Daily or intraday returns

3. **Moments** (can be computed from returns):
   - **RV**: Realized volatility
   - **Skew**: Skewness (third moment)
   - **Kurt**: Kurtosis (fourth moment, excess)

### Data Preprocessing

```python
# Example: Compute moments from returns
def compute_moments(returns_df):
    moments_dict = {}
    moments_dict['RV'] = returns_df.std()
    moments_dict['Skew'] = returns_df.skew()
    moments_dict['Kurt'] = returns_df.kurtosis()
    return moments_dict
```

## Algorithm Details

### Orthogonality Testing

**Correlation Analysis**:
- Compute daily cross-sectional correlation matrix
- Average correlations across time
- Fisher's z-transformation: $z = 0.5 \ln[(1+r)/(1-r)]$
- Confidence intervals: $CI = \tanh(z \pm 1.96/\sqrt{n-3})$

**Regression Analysis**:
- Time-series regression for each stock
- Collect R² values across stocks
- Wilcoxon signed-rank test for median R²
- Extract residuals as "pure ambiguity"

### Crash Prediction

**Logistic Regression with Interaction**:
$$
\text{Prob}(Crash_{t+1}=1) = \Phi(\alpha + \beta_1 Skew_t + \beta_2 Kurt_t + \beta_3 \mathcal{A}^{CEA}_t + \beta_4 (Skew_t \times \mathcal{A}^{CEA}_t))
$$

**Model Comparison**:
- Likelihood ratio test: $LR = 2(\mathcal{L}_{full} - \mathcal{L}_{restricted}) \sim \chi^2_1$
- AUC comparison using DeLong's test for correlated ROC curves

### Portfolio Double-Sorting

**Algorithm**:
1. Sort stocks into skewness quintiles (Q1-Q5)
2. Within Q1 (most negative skew), sort by ambiguity (high/low)
3. Form portfolios:
   - Toxic: Q1 + High Ambiguity
   - Stable: Q5 + Low Ambiguity
4. Compute long-short returns: Stable - Toxic
5. Calculate Fama-French alphas

### PCA Factor Analysis

**Implementation**:
1. Standardize variables to zero mean, unit variance
2. Compute covariance matrix
3. Eigenvalue decomposition
4. Extract factor loadings (eigenvectors)
5. Identify which factor has highest ambiguity loading

## Output Specifications

### Hypothesis Test Results

Each hypothesis test returns specific outputs:

```python
# H1: Correlation
{
    'Ambiguity_RV': {
        'mean_correlation': 0.15,
        'ci_lower': 0.12,
        'ci_upper': 0.18,
        'p_value': 0.001
    },
    ...
}

# H2: Regression
{
    'median_r2': 0.05,
    'wilcoxon_pvalue': 0.002,
    'orthogonality_confirmed': True,
    'pure_ambiguity_df': DataFrame
}

# H3: Interaction
{
    'interaction_coefficient': -0.25,
    'interaction_p_value': 0.003,
    'lr_pvalue': 0.001,
    'auc_improvement': 0.08
}

# H4: PCA
{
    'explained_variance_ratio': [0.45, 0.25, 0.18, 0.12],
    'ambiguity_max_loading': 0.65,
    'ambiguity_max_loading_factor': 2,
    'ambiguity_distinct_factor': True
}

# H5: Portfolio
{
    'long_short_alpha': 0.10,
    'long_short_t_stat': 3.5,
    'long_short_p_value': 0.001,
    'toxic_annual_return': -0.08,
    'stable_annual_return': 0.04
}
```

## Dependencies

```
numpy >= 1.20.0
pandas >= 1.3.0
scipy >= 1.7.0
scikit-learn >= 0.24.0
statsmodels >= 0.13.0
matplotlib >= 3.4.0
seaborn >= 0.11.0
```

## Installation

```bash
pip install numpy pandas scipy scikit-learn statsmodels matplotlib seaborn
```

## Execution

### Run Complete Pipeline

```bash
python code/main_pipeline.py
```

### Run Individual Tests

```python
from moments_analysis import (
    test_hypothesis_1_correlation,
    test_hypothesis_2_regression,
    test_hypothesis_3_interaction,
    test_hypothesis_4_pca,
    test_hypothesis_5_portfolio
)

# Test H1
h1 = test_hypothesis_1_correlation(ambiguity_df, moments_dict)

# Test H2
h2 = test_hypothesis_2_regression(ambiguity_df, moments_dict)
```

## Expected Results Summary

Based on the research hypotheses:

### H1: Correlation Orthogonality
- **Correlations**: All < 0.3 (typically 0.1-0.2)
- **P-values**: < 0.05 for H0: ρ ≥ 0.3
- **Conclusion**: Ambiguity and moments are weakly correlated

### H2: Regression Orthogonality
- **Median R²**: < 10% (typically 3-7%)
- **Wilcoxon p-value**: < 0.01
- **Conclusion**: Moments explain little variation in ambiguity

### H3: Interaction Effect
- **Interaction coefficient**: Negative and significant (p < 0.05)
- **LR test p-value**: < 0.01
- **AUC improvement**: 5-12%
- **Conclusion**: Interaction significantly improves crash prediction

### H4: Factor Structure
- **Ambiguity loading**: > 0.5 on PC2
- **Explained variance**: PC2 explains 20-30%
- **Conclusion**: Ambiguity loads on distinct factor

### H5: Portfolio Value
- **Toxic portfolio**: Underperforms by 8-12% annually
- **Long-short alpha**: 6-9% annually (t > 3, p < 0.01)
- **Sharpe ratio improvement**: 2-3× vs market
- **Conclusion**: Significant economic value

## Troubleshooting

### Common Issues

1. **Insufficient Observations Error**:
   - **Cause**: Too few time periods or stocks
   - **Solution**: Increase sample size or reduce frequency

2. **Perfect Separation Warning in Logit**:
   - **Cause**: Ambiguity perfectly separates crashes
   - **Solution**: Add regularization or use Firth logistic regression

3. **High Correlation Between Variables**:
   - **Cause**: Moments highly correlated with each other
   - **Solution**: This is expected; PCA will handle it

4. **Memory Error**:
   - **Cause**: Large correlation matrices
   - **Solution**: Process in batches or use sparse matrices

### Performance Optimization

```python
# Reduce computation time
# Use fewer stocks for testing
ambiguity_subset = ambiguity_df.iloc[:, :100]

# Use longer time periods for moment computation
# Instead of daily, use weekly moments
```

## Mathematical Foundation

### Knightian Uncertainty Framework

**Risk (Moments)**: Known probability distribution P
- Skewness: $\mathbb{E}[(X-\mu)^3] / \sigma^3$
- Kurtosis: $\mathbb{E}[(X-\mu)^4] / \sigma^4 - 3$

**Ambiguity (Entropy)**: Uncertainty about P itself
- Cross-Entropy: $H(P,Q) = -\sum_x P(x) \log Q(x)$
- KL Divergence: $D_{KL}(P\|Q) = \sum_x P(x) \log[P(x)/Q(x)]$

### Interaction Effect Theory

The interaction term captures the amplification effect:
- When skewness is negative (tail risk present), crash risk is elevated
- When ambiguity is high (model uncertainty), investors cannot assess this risk
- The combination creates a particularly dangerous environment

### Economic Interpretation

- **"Thin Ice" (Negative Skewness)**: Known fragility
- **"Fog" (High Ambiguity)**: Inability to see cracks in ice
- **Interaction**: Thin ice + fog = invisible danger → panic

## Citation

If you use this code in your research, please cite:

```bibtex
@article{ambiguity_moments_2024,
  title={Ambiguity vs. Higher-Order Moments: A Theoretical and
         Empirical Distinction with Applications to Crash Prediction},
  author={[Authors]},
  journal={Journal of Financial Economics},
  year={2024}
}
```

## Contact

For questions or issues, please contact [Author Information].

## License

[License Information]
