# Causal Analysis of Ambiguity Effects - Code Documentation

## Overview

This directory contains Python implementation for the causal analysis of ambiguity effects on asset returns as described in the research paper. The code implements the Cross-Entropy Ambiguity (A_CEA_t) measure and instrumental variable strategies for establishing causal relationships.

## File Structure

```
code/
├── ambiguity_measurement.py    # Core ambiguity measurement module
├── causal_analysis.py           # Causal inference and IV estimation
├── main_analysis.py             # Complete analysis pipeline
└── README.md                    # This file
```

## Module Descriptions

### 1. ambiguity_measurement.py

**Purpose**: Implements the Cross-Entropy Ambiguity (A_CEA_t) index

**Key Classes**:
- `AmbiguityMeasurement`: Main class for computing ambiguity measures

**Key Methods**:
- `discretize_returns()`: Convert intraday returns to histogram bins
- `compute_kl_divergence()`: Calculate KL divergence between distributions
- `fit_benchmark_distributions()`: K-means clustering for regime identification
- `select_benchmark()`: Dynamic benchmark selection
- `compute_ambiguity_for_stock()`: Compute A_CEA_t for single stock
- `compute_ambiguity_cross_section()`: Compute A_CEA_t for multiple stocks

**Usage Example**:
```python
from ambiguity_measurement import AmbiguityMeasurement

# Initialize
ambiguity_measure = AmbiguityMeasurement(
    n_bins=202,
    window_size=20,
    n_clusters=4
)

# Compute for single stock
ambiguity_series = ambiguity_measure.compute_ambiguity_for_stock(
    intraday_returns
)

# Compute for cross-section
ambiguity_df = ambiguity_measure.compute_ambiguity_cross_section(
    returns_data
)
```

### 2. causal_analysis.py

**Purpose**: Implements causal inference methods

**Key Classes**:
- `CausalAmbiguityAnalysis`: Main class for causal analysis

**Key Methods**:
- `baseline_ols()`: Baseline regression with fixed effects
- `instrumental_variables_2sls()`: Two-stage least squares estimation
- `granger_causality_test()`: Test temporal precedence
- `mediation_analysis()`: Test indirect effects through liquidity
- `heterogeneity_analysis()`: Test effects across regimes

**Usage Example**:
```python
from causal_analysis import CausalAmbiguityAnalysis

# Initialize
analysis = CausalAmbiguityAnalysis(
    ambiguity_df,
    returns_df,
    controls_df
)

# Baseline OLS
baseline_results = analysis.baseline_ols()

# IV estimation
iv_results = analysis.instrumental_variables_2sls(
    instruments=['peer_ambiguity', 'epu_interaction']
)

# Mediation analysis
mediation_results = analysis.mediation_analysis(
    mediator='Turnover'
)
```

### 3. main_analysis.py

**Purpose**: Orchestrates complete analysis pipeline

**Key Classes**:
- `CausalAnalysisPipeline`: End-to-end analysis pipeline

**Key Methods**:
- `load_data()`: Load or generate data
- `compute_ambiguity_measures()`: Compute A_CEA_t for all stocks
- `prepare_controls()`: Prepare control variables
- `generate_instruments()`: Create instrumental variables
- `run_analysis()`: Execute all causal analyses
- `visualize_results()`: Create visualizations
- `generate_report()`: Generate text report

**Usage Example**:
```python
from main_analysis import CausalAnalysisPipeline

# Initialize pipeline
pipeline = CausalAnalysisPipeline(data_path=None)

# Run complete pipeline
results = pipeline.run_pipeline()
```

## Data Requirements

### Input Data Format

1. **Intraday Returns** (pandas DataFrame):
   - Index: DatetimeIndex (minute-level)
   - Columns: Stock identifiers
   - Values: Log returns

2. **Control Variables**:
   - Realized Volatility (RV)
   - Skewness
   - Kurtosis
   - Turnover Rate
   - Bid-Ask Spread

3. **Instrumental Variables**:
   - Peer-based ambiguity (industry average)
   - EPU × Policy Sensitivity interaction
   - Filing complexity measure

### Data Preprocessing

```python
# Resample minute data to daily
daily_returns = intraday_returns.resample('D').apply(
    lambda x: np.log(x.iloc[-1] / x.iloc[0])
)

# Compute realized volatility
rv = intraday_returns.resample('D').apply(
    lambda x: np.sqrt(np.mean(x**2))
)
```

## Algorithm Details

### A_CEA_t Computation

1. **Discretization**:
   - Bin returns into 202 equally spaced bins over [-0.201, 0.201]
   - Create probability density function for each day

2. **Window-based Analysis**:
   - Partition data into 20-day windows
   - Use K-means clustering (K=4) to identify regimes
   - Compute cluster centroids as benchmarks

3. **Benchmark Selection**:
   - At window boundaries, select benchmark minimizing KL divergence
   - Use out-of-sample day for selection

4. **Daily Ambiguity**:
   - Compute KL divergence between daily PDF and selected benchmark
   - Add epsilon = 1e-10 for numerical stability

### Instrumental Variables

1. **Peer-based Ambiguity**:
   ```
   PeerAmbiguity_i,t = mean(A_CEA_j,t) for j in industry(i), j ≠ i
   ```

2. **EPU Interaction**:
   ```
   EPU_Interaction_i,t = EPU_t × PolicySensitivity_i,t
   ```

3. **Filing Complexity**:
   - Unexpected length/complexity of regulatory filings
   - Should not contain fundamental information

### 2SLS Estimation

**First Stage**:
```
A_CEA_i,t = π₀ + π₁ PeerAmbiguity + π₂ EPU_Interaction
             + π₃ FilingComplexity + Γ'Controls + η
```

**Second Stage**:
```
r_i,t+1 = α + β A_CEA_pred_i,t + γ'Controls + FixedEffects + ε
```

## Visualization

The pipeline generates four key visualizations:

1. **Ambiguity Time Series**: Cross-sectional mean over time
2. **Coefficient Comparison**: OLS vs IV estimates
3. **Granger Causality**: P-values across lags
4. **Mediation Effects**: Direct vs indirect effects

## Output

### Results Dictionary Structure

```python
{
    'baseline': {
        'coefficients': Series,
        'std_errors': Series,
        't_stats': Series,
        'p_values': Series,
        'r_squared': float
    },
    'iv': {
        'first_stage': {...},
        'second_stage': {...},
        'causal_effect': {
            'coefficient': float,
            'std_error': float,
            't_stat': float,
            'p_value': float
        }
    },
    'granger': {
        1: {...},
        2: {...},
        ...
    },
    'mediation': {
        'path_a': {...},
        'path_b': {...},
        'direct_effect': {...},
        'indirect_effect': {...},
        'proportion_mediated': float
    },
    'heterogeneity': {
        'RV': {
            'low': {...},
            'high': {...}
        }
    }
}
```

## Dependencies

```
numpy >= 1.20.0
pandas >= 1.3.0
scikit-learn >= 0.24.0
scipy >= 1.7.0
statsmodels >= 0.13.0
linearmodels >= 4.0
matplotlib >= 3.4.0
seaborn >= 0.11.0
```

## Installation

```bash
pip install numpy pandas scikit-learn scipy statsmodels linearmodels matplotlib seaborn
```

## Execution

Run the complete pipeline:

```bash
python code/main_analysis.py
```

Run individual modules:

```bash
# Compute ambiguity measures
python code/ambiguity_measurement.py

# Run causal analysis
python code/causal_analysis.py
```

## Expected Results

Based on the research hypotheses:

1. **Baseline OLS**: Positive coefficient on ambiguity (~0.007)
2. **IV Estimate**: Similar magnitude, confirms causality
3. **Granger Causality**: Significant at lags 1-5
4. **Mediation**: ~35% of effect through liquidity
5. **Heterogeneity**: Stronger effects in high RV regimes

## Troubleshooting

### Common Issues

1. **Memory Error**: Reduce window_size or n_clusters
2. **Slow Computation**: Use subset of stocks for testing
3. **Convergence Warning**: Increase n_init in K-means
4. **Zero Division**: Add epsilon to KL divergence computation

### Performance Optimization

```python
# Use smaller bins for faster computation
ambiguity_measure = AmbiguityMeasurement(n_bins=101)

# Reduce window size
ambiguity_measure = AmbiguityMeasurement(window_size=15)

# Use fewer clusters
ambiguity_measure = AmbiguityMeasurement(n_clusters=3)
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{ambiguity_causal_2024,
  title={Ambiguity as a Causal Determinant of Asset Returns:
         Evidence from High-Frequency Data and Instrumental Variables},
  author={[Authors]},
  journal={Journal of Financial Economics},
  year={2024}
}
```

## Contact

For questions or issues, please contact [Author Information].

## License

[License Information]
