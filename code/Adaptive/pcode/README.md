# Adaptive Ambiguity Parameter Estimation

This package implements the adaptive ambiguity parameter estimation procedure described in `adaptive_procedure.md`. The code estimates the time-varying parameter ρₜ using a Markov Switching Time-Varying Parameter (MS-TVP) model with Bayesian inference via Gibbs sampling.

## Overview

The implementation follows the complete procedure:

1. **Data Generation**: Synthetic financial data (price, iVIX, EPU, turnover rate)
2. **Behavioral Indicators**: Price Peak Proximity (PPP) and Price Trough Proximity (PTP)
3. **Data Preprocessing**: Standardization and lagged variable creation
4. **MS-TVP Model**: State-space representation with regime switching
5. **Gibbs Sampling**: Bayesian inference for parameter estimation

## Files

- `data_generator.py` - Generate synthetic financial data
- `behavioral_indicators.py` - Create PPP and PTP indicators
- `data_preprocessor.py` - Data preprocessing and standardization
- `ms_tvp_model.py` - MS-TVP model implementation
- `gibbs_sampler.py` - Gibbs sampling algorithm
- `main_estimation.py` - Main estimation script
- `__init__.py` - Package initialization

## Usage

### Quick Start

```python
from main_estimation import AdaptiveRhoEstimator

# Initialize estimator
estimator = AdaptiveRhoEstimator(n_regimes=2, seed=42)

# Generate data
data = estimator.generate_data(n_obs=1000)

# Prepare data for estimation
design_matrix = estimator.prepare_data()

# Estimate adaptive rho parameter
results = estimator.estimate_rho(
    n_iterations=20000,
    burn_in=10000,
    verbose=True
)

# Analyze results
analysis = estimator.analyze_results()

# Plot results
estimator.plot_results()
```

### Individual Components

```python
# Generate synthetic data
from data_generator import DataGenerator
generator = DataGenerator(n_obs=1000, seed=42)
df = generator.generate_all_data()

# Create behavioral indicators
from behavioral_indicators import BehavioralIndicators
bi = BehavioralIndicators()
behavioral_df = bi.create_behavioral_indicators(df['price'])

# Preprocess data
from data_preprocessor import DataPreprocessor
preprocessor = DataPreprocessor()
X, info = preprocessor.prepare_full_dataset(df)

# Run Gibbs sampling
from gibbs_sampler import GibbsSampler
sampler = GibbsSampler(n_regimes=2, n_features=6)
results = sampler.run_gibbs_sampling(X, rho_obs, n_iterations=20000, burn_in=10000)
```

## Model Specification

The MS-TVP model is specified as:

**Measurement Equation:**
ρₜ = Xₜ'βₜ,ₛₜ + εₜ, εₜ ~ N(0, σ²ₑ,ₛₜ)

**State Equation:**
βₜ,ₛₜ = βₜ₋₁,ₛₜ + ηₜ, ηₜ ~ N(0, Qₛₜ)

**Design Matrix:**
Xₜ = [1, iVIXₜ₋₁, EPUₜ₋₁, Turnoverₜ₋₁, PPPₜ₋₁, PTPₜ₋₁]'

## Dependencies

- numpy
- pandas
- scipy
- scikit-learn
- matplotlib
- seaborn

## Notes

- The code includes synthetic data generation for testing purposes
- For real applications, replace synthetic data with actual financial data
- MCMC settings (iterations, burn-in) can be adjusted based on convergence diagnostics
- The number of regimes can be modified, though 2 regimes are typically sufficient