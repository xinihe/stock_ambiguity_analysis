"""
Quick test script for the adaptive rho estimation.
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

def test_data_generation():
    """Test data generation and preprocessing."""
    print("Testing data generation...")
    
    # Generate data
    generator = DataGenerator(n_obs=100, seed=42)
    df = generator.generate_all_data()
    
    print(f"Generated data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Create behavioral indicators
    bi = BehavioralIndicators()
    behavioral_df = bi.create_behavioral_indicators(df['price'])
    
    print(f"Behavioral indicators shape: {behavioral_df.shape}")
    
    # Combine data
    combined_df = pd.concat([df, behavioral_df], axis=1)
    
    # Create lagged variables
    preprocessor = DataPreprocessor()
    variables_to_lag = ['ivix', 'epu', 'turnover', 'ppp', 'ptp']
    lagged_df = preprocessor.create_lagged_features(combined_df, variables_to_lag, lags=[1])
    
    print(f"Lagged data shape: {lagged_df.shape}")
    
    # Create design matrix
    feature_cols = ['ivix_lag1', 'epu_lag1', 'turnover_lag1', 'ppp_lag1', 'ptp_lag1']
    X, _, info = preprocessor.prepare_data(lagged_df, feature_cols=feature_cols)
    
    print(f"Design matrix shape: {X.shape}")
    print(f"Feature names: {info['feature_names']}")
    
    # Check for any issues
    print(f"Any NaN values: {np.isnan(X).any()}")
    print(f"Data range: [{X.min():.3f}, {X.max():.3f}]")
    
    return X

def test_model_components():
    """Test individual model components."""
    print("\nTesting model components...")
    
    # Create test data
    T = 50
    n_features = 6
    n_regimes = 2
    
    X = np.random.randn(T, n_features)
    rho = np.random.randn(T)
    
    # Test MS-TVP model
    from ms_tvp_model import MSTVPModel
    
    model = MSTVPModel(n_regimes, n_features)
    model.set_priors()
    model.initialize_parameters(T)
    
    # Test likelihood computation
    log_likelihood = model.compute_likelihood(X, rho)
    print(f"Log likelihood shape: {log_likelihood.shape}")
    print(f"Log likelihood range: [{log_likelihood.min():.3f}, {log_likelihood.max():.3f}]")
    
    # Test state sampling
    states = model.sample_states(log_likelihood)
    print(f"Sampled states shape: {states.shape}")
    print(f"State distribution: {np.bincount(states) / len(states)}")
    
    return True

def test_gibbs_sampler():
    """Test Gibbs sampler with small dataset."""
    print("\nTesting Gibbs sampler...")
    
    # Create small test data
    T = 30
    n_features = 6
    n_regimes = 2
    
    X = np.random.randn(T, n_features)
    rho_obs = np.random.randn(T)
    
    # Test Gibbs sampler
    from gibbs_sampler import GibbsSampler
    
    sampler = GibbsSampler(n_regimes, n_features)
    
    # Run very short test
    results = sampler.run_gibbs_sampling(
        X, rho_obs,
        n_iterations=100,  # Very short test
        burn_in=50,
        verbose=True
    )
    
    print(f"Results keys: {list(results.keys())}")
    print(f"Rho samples shape: {results['rho'].shape}")
    print(f"Acceptance rates: {results['acceptance_rates']}")
    
    return True

if __name__ == "__main__":
    print("=" * 50)
    print("Adaptive Rho Estimation - Quick Test")
    print("=" * 50)
    
    try:
        # Test 1: Data generation
        X = test_data_generation()
        
        # Test 2: Model components
        test_model_components()
        
        # Test 3: Gibbs sampler
        test_gibbs_sampler()
        
        print("\n" + "=" * 50)
        print("All tests completed successfully!")
        print("=" * 50)
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()