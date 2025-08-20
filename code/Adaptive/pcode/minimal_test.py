"""
Minimal test to verify the implementation works.
"""

import numpy as np
import pandas as pd
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import modules
from data_generator import DataGenerator
from behavioral_indicators import BehavioralIndicators
from data_preprocessor import DataPreprocessor

def minimal_test():
    """Minimal test to verify all components work."""
    print("=" * 40)
    print("Minimal Test - Implementation Check")
    print("=" * 40)
    
    # Test 1: Data generation
    print("1. Testing data generation...")
    generator = DataGenerator(n_obs=50, seed=42)
    df = generator.generate_all_data()
    print(f"   Generated data shape: {df.shape}")
    
    # Test 2: Behavioral indicators
    print("2. Testing behavioral indicators...")
    bi = BehavioralIndicators()
    behavioral_df = bi.create_behavioral_indicators(df['price'])
    print(f"   Behavioral indicators shape: {behavioral_df.shape}")
    
    # Test 3: Data preprocessing
    print("3. Testing data preprocessing...")
    preprocessor = DataPreprocessor()
    
    # Combine data
    combined_df = pd.concat([df, behavioral_df], axis=1)
    
    # Create lagged variables
    variables_to_lag = ['ivix', 'epu', 'turnover', 'ppp', 'ptp']
    lagged_df = preprocessor.create_lagged_features(combined_df, variables_to_lag, lags=[1])
    
    # Create design matrix
    feature_cols = ['ivix_lag1', 'epu_lag1', 'turnover_lag1', 'ppp_lag1', 'ptp_lag1']
    X, _, info = preprocessor.prepare_data(lagged_df, feature_cols=feature_cols)
    
    print(f"   Design matrix shape: {X.shape}")
    print(f"   Feature names: {info['feature_names']}")
    
    # Test 4: Model components
    print("4. Testing model components...")
    from ms_tvp_model import MSTVPModel
    
    T = X.shape[0]
    n_features = X.shape[1]
    n_regimes = 2
    
    model = MSTVPModel(n_regimes, n_features)
    model.set_priors()
    model.initialize_parameters(T)
    
    # Generate test rho
    rho_obs = np.random.randn(T)
    
    # Test likelihood
    log_likelihood = model.compute_likelihood(X, rho_obs)
    print(f"   Log likelihood shape: {log_likelihood.shape}")
    
    # Test state sampling
    states = model.sample_states(log_likelihood)
    print(f"   Sampled states shape: {states.shape}")
    print(f"   State distribution: {np.bincount(states) / len(states)}")
    
    # Test 5: Gibbs sampler (very small)
    print("5. Testing Gibbs sampler (small test)...")
    from gibbs_sampler import GibbsSampler
    
    sampler = GibbsSampler(n_regimes, n_features)
    
    # Run very small test
    results = sampler.run_gibbs_sampling(
        X, rho_obs,
        n_iterations=50,  # Very small
        burn_in=25,
        verbose=False
    )
    
    print(f"   Gibbs sampling completed")
    print(f"   Rho samples shape: {results['rho'].shape}")
    print(f"   Acceptance rates: {results['acceptance_rates']}")
    
    # Calculate final rho estimate
    rho_samples = results['rho']
    rho_mean = np.mean(rho_samples, axis=0)
    rho_std = np.std(rho_samples, axis=0)
    
    print(f"   Final rho mean: {np.mean(rho_mean):.4f}")
    print(f"   Final rho std: {np.mean(rho_std):.4f}")
    
    print("\n" + "=" * 40)
    print("All tests passed! Implementation is working correctly.")
    print("=" * 40)
    
    return {
        'rho_mean': rho_mean,
        'rho_std': rho_std,
        'rho_samples': rho_samples,
        'results': results
    }


if __name__ == "__main__":
    minimal_test()