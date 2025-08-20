"""
Simplified main estimation script for quick testing.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional
import time
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our modules
from data_generator import DataGenerator
from behavioral_indicators import BehavioralIndicators
from data_preprocessor import DataPreprocessor
from gibbs_sampler import GibbsSampler


def run_simple_estimation():
    """
    Run a simplified version of the adaptive rho estimation.
    """
    print("=" * 60)
    print("Adaptive Ambiguity Parameter Estimation - Simple Version")
    print("=" * 60)
    
    # Step 1: Generate data
    print("Generating synthetic financial data...")
    generator = DataGenerator(n_obs=200, seed=42)
    raw_data = generator.generate_all_data()
    
    # Create behavioral indicators
    bi = BehavioralIndicators()
    behavioral_data = bi.create_behavioral_indicators(raw_data['price'])
    
    # Combine data
    data = pd.concat([raw_data, behavioral_data], axis=1)
    print(f"Generated data shape: {data.shape}")
    
    # Step 2: Prepare data
    print("Preparing data for estimation...")
    preprocessor = DataPreprocessor()
    
    # Create lagged variables
    variables_to_lag = ['ivix', 'epu', 'turnover', 'ppp', 'ptp']
    lagged_data = preprocessor.create_lagged_features(data, variables_to_lag, lags=[1])
    
    # Create design matrix
    feature_cols = ['ivix_lag1', 'epu_lag1', 'turnover_lag1', 'ppp_lag1', 'ptp_lag1']
    design_matrix, _, info = preprocessor.prepare_data(lagged_data, feature_cols=feature_cols)
    
    print(f"Design matrix shape: {design_matrix.shape}")
    
    # Step 3: Estimate rho
    print(f"\nStarting adaptive rho estimation...")
    print(f"Using smaller iterations for testing...")
    
    # Initialize Gibbs sampler
    n_regimes = 2
    n_features = design_matrix.shape[1]
    sampler = GibbsSampler(n_regimes, n_features)
    
    # Generate initial rho values
    T = design_matrix.shape[0]
    rho_obs = np.random.randn(T)
    
    # Run Gibbs sampling with smaller iterations
    start_time = time.time()
    results = sampler.run_gibbs_sampling(
        design_matrix, rho_obs,
        n_iterations=1000,  # Smaller for testing
        burn_in=500,
        verbose=True
    )
    
    # Get final estimates
    rho_samples = results['rho']
    rho_mean = np.mean(rho_samples, axis=0)
    rho_std = np.std(rho_samples, axis=0)
    
    print(f"\nEstimation completed in {time.time() - start_time:.1f} seconds")
    print(f"Rho estimate shape: {rho_mean.shape}")
    print(f"Rho std shape: {rho_std.shape}")
    
    # Step 4: Basic analysis
    print(f"\nBasic analysis:")
    print(f"Rho mean: {np.mean(rho_mean):.4f}")
    print(f"Rho std: {np.std(rho_mean):.4f}")
    print(f"Rho range: [{np.min(rho_mean):.4f}, {np.max(rho_mean):.4f}]")
    
    # Step 5: Simple plots
    print("\nGenerating plots...")
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot 1: Rho estimates with uncertainty
    ax1 = axes[0]
    time_points = range(len(rho_mean))
    ax1.plot(time_points, rho_mean, label='Rho estimate', linewidth=2, color='blue')
    ax1.fill_between(time_points, 
                    rho_mean - 1.96 * rho_std,
                    rho_mean + 1.96 * rho_std,
                    alpha=0.3, color='blue', label='95% CI')
    ax1.set_title('Adaptive Ambiguity Parameter ρₜ')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('ρₜ')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Regime probabilities
    ax2 = axes[1]
    states_samples = results['states']
    n_samples, T = states_samples.shape
    
    regime_probs = np.zeros((T, n_regimes))
    for t in range(T):
        state_counts = np.bincount(states_samples[:, t], minlength=n_regimes)
        regime_probs[t, :] = state_counts / n_samples
    
    for j in range(n_regimes):
        ax2.plot(time_points, regime_probs[:, j], label=f'Regime {j+1}', linewidth=2)
    
    ax2.set_title('Regime Probabilities')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Probability')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('adaptive_rho_simple.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Step 6: Save results
    print("\nSaving results...")
    np.savez('adaptive_rho_simple_results.npz', 
             rho_mean=rho_mean,
             rho_std=rho_std,
             rho_samples=rho_samples,
             regime_probs=regime_probs)
    
    print("Results saved to 'adaptive_rho_simple_results.npz'")
    print("Plots saved to 'adaptive_rho_simple.png'")
    
    print("\n" + "=" * 60)
    print("Estimation completed successfully!")
    print("=" * 60)
    
    return {
        'rho_mean': rho_mean,
        'rho_std': rho_std,
        'rho_samples': rho_samples,
        'regime_probs': regime_probs,
        'results': results
    }


if __name__ == "__main__":
    run_simple_estimation()