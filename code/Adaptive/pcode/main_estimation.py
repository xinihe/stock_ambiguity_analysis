"""
Main estimation script for adaptive ambiguity parameter estimation.
This script implements the complete procedure described in adaptive_procedure.md.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional
import time
import os

# Import our modules
from data_generator import DataGenerator
from behavioral_indicators import BehavioralIndicators
from data_preprocessor import DataPreprocessor
from gibbs_sampler import GibbsSampler


class AdaptiveRhoEstimator:
    """
    Complete implementation of adaptive ambiguity parameter estimation.
    """
    
    def __init__(self, n_regimes: int = 2, seed: int = 42):
        """
        Initialize the estimator.
        
        Args:
            n_regimes: Number of regimes in the MS-TVP model
            seed: Random seed for reproducibility
        """
        self.n_regimes = n_regimes
        np.random.seed(seed)
        
        # Initialize components
        self.data_generator = DataGenerator(seed=seed)
        self.behavioral_indicators = BehavioralIndicators()
        self.preprocessor = DataPreprocessor()
        
        # Results storage
        self.results = {}
        self.data = None
        self.design_matrix = None
        
    def generate_data(self, n_obs: int = 1000) -> pd.DataFrame:
        """
        Generate synthetic financial data.
        
        Args:
            n_obs: Number of observations
            
        Returns:
            pd.DataFrame: Complete dataset
        """
        print("Generating synthetic financial data...")
        
        # Generate base data
        raw_data = self.data_generator.generate_all_data()
        
        # Create behavioral indicators
        behavioral_data = self.behavioral_indicators.create_behavioral_indicators(
            raw_data['price']
        )
        
        # Combine data
        self.data = pd.concat([raw_data, behavioral_data], axis=1)
        
        print(f"Generated data shape: {self.data.shape}")
        print(f"Columns: {list(self.data.columns)}")
        
        return self.data
    
    def prepare_data(self) -> np.ndarray:
        """
        Prepare data for estimation.
        
        Returns:
            np.ndarray: Design matrix
        """
        if self.data is None:
            raise ValueError("Generate data first")
        
        print("Preparing data for estimation...")
        
        # Create lagged variables
        variables_to_lag = ['ivix', 'epu', 'turnover', 'ppp', 'ptp']
        lagged_data = self.preprocessor.create_lagged_features(
            self.data, variables_to_lag, lags=[1]
        )
        
        # Create design matrix
        feature_cols = ['ivix_lag1', 'epu_lag1', 'turnover_lag1', 'ppp_lag1', 'ptp_lag1']
        self.design_matrix, _, info = self.preprocessor.prepare_data(
            lagged_data, feature_cols=feature_cols
        )
        
        print(f"Design matrix shape: {self.design_matrix.shape}")
        print(f"Feature names: {info['feature_names']}")
        
        return self.design_matrix
    
    def estimate_rho(self, n_iterations: int = 20000, burn_in: int = 10000,
                    verbose: bool = True) -> Dict[str, Any]:
        """
        Estimate the adaptive rho parameter.
        
        Args:
            n_iterations: Total MCMC iterations
            burn_in: Burn-in period
            verbose: Whether to print progress
            
        Returns:
            Dict: Estimation results
        """
        if self.design_matrix is None:
            raise ValueError("Prepare data first")
        
        print(f"\nStarting adaptive rho estimation...")
        print(f"Design matrix: {self.design_matrix.shape}")
        print(f"Number of regimes: {self.n_regimes}")
        print(f"MCMC iterations: {n_iterations} (burn-in: {burn_in})")
        
        # Initialize Gibbs sampler
        n_features = self.design_matrix.shape[1]
        sampler = GibbsSampler(self.n_regimes, n_features)
        
        # Generate initial rho values (since we don't have observed rho)
        T = self.design_matrix.shape[0]
        rho_obs = np.random.randn(T)
        
        # Run Gibbs sampling
        start_time = time.time()
        self.results = sampler.run_gibbs_sampling(
            self.design_matrix, rho_obs,
            n_iterations=n_iterations,
            burn_in=burn_in,
            verbose=verbose
        )
        
        # Get final estimates
        rho_mean, rho_std = sampler.get_rho_estimate()
        self.results['rho_estimate'] = rho_mean
        self.results['rho_std'] = rho_std
        
        # Add estimation metadata
        self.results['estimation_time'] = time.time() - start_time
        self.results['n_regimes'] = self.n_regimes
        self.results['n_features'] = n_features
        self.results['n_obs'] = T
        
        print(f"\nEstimation completed in {self.results['estimation_time']:.1f} seconds")
        print(f"Final rho estimate shape: {rho_mean.shape}")
        
        return self.results
    
    def analyze_results(self) -> Dict[str, Any]:
        """
        Analyze and summarize estimation results.
        
        Returns:
            Dict: Analysis results
        """
        if not self.results:
            raise ValueError("Run estimation first")
        
        print("\nAnalyzing results...")
        
        analysis = {}
        
        # Rho estimates
        rho_mean = self.results['rho_estimate']
        rho_std = self.results['rho_std']
        
        analysis['rho_summary'] = {
            'mean': np.mean(rho_mean),
            'std': np.std(rho_mean),
            'min': np.min(rho_mean),
            'max': np.max(rho_mean),
            'mean_uncertainty': np.mean(rho_std)
        }
        
        # Regime analysis
        states_samples = self.results['states']
        n_samples, T = states_samples.shape
        
        # Calculate regime probabilities
        regime_probs = np.zeros((T, self.n_regimes))
        for t in range(T):
            state_counts = np.bincount(states_samples[:, t], minlength=self.n_regimes)
            regime_probs[t, :] = state_counts / n_samples
        
        analysis['regime_summary'] = {
            'average_probabilities': np.mean(regime_probs, axis=0),
            'persistence': np.mean(regime_probs[1:, :] == regime_probs[:-1, :], axis=0)
        }
        
        # Transition probabilities
        P_samples = self.results['P']
        analysis['transition_probs'] = {
            'mean': np.mean(P_samples, axis=0),
            'std': np.std(P_samples, axis=0)
        }
        
        print(f"Rho summary: {analysis['rho_summary']}")
        print(f"Average regime probabilities: {analysis['regime_summary']['average_probabilities']}")
        
        return analysis
    
    def plot_results(self, save_path: Optional[str] = None):
        """
        Plot estimation results.
        
        Args:
            save_path: Path to save plots (optional)
        """
        if not self.results:
            raise ValueError("Run estimation first")
        
        print("\nGenerating plots...")
        
        # Set up plotting
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Rho estimates with uncertainty
        ax1 = axes[0, 0]
        rho_mean = self.results['rho_estimate']
        rho_std = self.results['rho_std']
        
        ax1.plot(rho_mean, label='Rho estimate', linewidth=2)
        ax1.fill_between(range(len(rho_mean)), 
                        rho_mean - 1.96 * rho_std,
                        rho_mean + 1.96 * rho_std,
                        alpha=0.3, label='95% CI')
        ax1.set_title('Adaptive Ambiguity Parameter ρₜ')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('ρₜ')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Regime probabilities
        ax2 = axes[0, 1]
        states_samples = self.results['states']
        n_samples, T = states_samples.shape
        
        regime_probs = np.zeros((T, self.n_regimes))
        for t in range(T):
            state_counts = np.bincount(states_samples[:, t], minlength=self.n_regimes)
            regime_probs[t, :] = state_counts / n_samples
        
        for j in range(self.n_regimes):
            ax2.plot(regime_probs[:, j], label=f'Regime {j+1}', linewidth=2)
        
        ax2.set_title('Regime Probabilities')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Probability')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Transition probabilities
        ax3 = axes[1, 0]
        P_samples = self.results['P']
        P_mean = np.mean(P_samples, axis=0)
        
        im = ax3.imshow(P_mean, cmap='Blues', aspect='auto')
        ax3.set_title('Average Transition Probability Matrix')
        ax3.set_xlabel('To Regime')
        ax3.set_ylabel('From Regime')
        
        # Add text annotations
        for i in range(self.n_regimes):
            for j in range(self.n_regimes):
                ax3.text(j, i, f'{P_mean[i, j]:.3f}', 
                        ha='center', va='center', fontsize=12)
        
        plt.colorbar(im, ax=ax3)
        
        # Plot 4: Rho distribution
        ax4 = axes[1, 1]
        rho_samples = self.results['rho']
        rho_flat = rho_samples.flatten()
        
        ax4.hist(rho_flat, bins=50, alpha=0.7, density=True)
        ax4.axvline(np.mean(rho_flat), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(rho_flat):.3f}')
        ax4.set_title('Distribution of Rho Estimates')
        ax4.set_xlabel('ρₜ')
        ax4.set_ylabel('Density')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plots saved to {save_path}")
        
        plt.show()
    
    def save_results(self, filepath: str):
        """
        Save results to file.
        
        Args:
            filepath: Path to save results
        """
        if not self.results:
            raise ValueError("Run estimation first")
        
        # Prepare data for saving
        save_data = {
            'rho_estimate': self.results['rho_estimate'],
            'rho_std': self.results['rho_std'],
            'estimation_time': self.results['estimation_time'],
            'n_regimes': self.results['n_regimes'],
            'n_features': self.results['n_features'],
            'n_obs': self.results['n_obs'],
            'acceptance_rates': self.results['acceptance_rates']
        }
        
        # Save as numpy arrays
        np.savez(filepath, **save_data)
        print(f"Results saved to {filepath}")


def main():
    """
    Main function to run the complete adaptive rho estimation.
    """
    print("=" * 60)
    print("Adaptive Ambiguity Parameter Estimation")
    print("=" * 60)
    
    # Initialize estimator
    estimator = AdaptiveRhoEstimator(n_regimes=2, seed=42)
    
    # Step 1: Generate data
    data = estimator.generate_data(n_obs=500)  # Smaller dataset for testing
    
    # Step 2: Prepare data
    design_matrix = estimator.prepare_data()
    
    # Step 3: Estimate rho (using smaller iterations for testing)
    results = estimator.estimate_rho(
        n_iterations=2000,  # Reduced for testing
        burn_in=1000,
        verbose=True
    )
    
    # Step 4: Analyze results
    analysis = estimator.analyze_results()
    
    # Step 5: Plot results
    estimator.plot_results()
    
    # Step 6: Save results
    estimator.save_results('adaptive_rho_results.npz')
    
    print("\n" + "=" * 60)
    print("Estimation completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()