"""
Gibbs sampling algorithm for estimating the adaptive ambiguity parameter.
Implements the full Bayesian inference procedure for the MS-TVP model.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from ms_tvp_model import MSTVPModel
import time


class GibbsSampler:
    """
    Gibbs sampling algorithm for MS-TVP model estimation.
    """
    
    def __init__(self, n_regimes: int = 2, n_features: int = 6):
        """
        Initialize the Gibbs sampler.
        
        Args:
            n_regimes: Number of regimes
            n_features: Number of features including intercept
        """
        self.model = MSTVPModel(n_regimes, n_features)
        self.n_regimes = n_regimes
        self.n_features = n_features
        
        # Storage for posterior samples
        self.posterior_samples = {}
        self.acceptance_rates = {}
        
    def initialize(self, X: np.ndarray, rho: np.ndarray):
        """
        Initialize the model and sampler.
        
        Args:
            X: Design matrix [T, n_features]
            rho: Target variable [T]
        """
        T = X.shape[0]
        
        # Set priors
        self.model.set_priors()
        
        # Initialize model parameters
        self.model.initialize_parameters(T)
        
        # Initialize storage
        self.posterior_samples = {
            'beta': np.zeros((0, T, self.n_features, self.n_regimes)),
            'Q': np.zeros((0, self.n_features, self.n_features, self.n_regimes)),
            'sigma2_eps': np.zeros((0, self.n_regimes)),
            'P': np.zeros((0, self.n_regimes, self.n_regimes)),
            'states': np.zeros((0, T), dtype=int),
            'rho': np.zeros((0, T))
        }
        
        # Initialize acceptance tracking
        self.acceptance_rates = {
            'beta': 0.0,
            'states': 0.0,
            'Q': 0.0,
            'sigma2_eps': 0.0,
            'P': 0.0
        }
        
    def run_gibbs_sampling(self, X: np.ndarray, rho: np.ndarray,
                          n_iterations: int = 20000, burn_in: int = 10000,
                          thin: int = 1, verbose: bool = True) -> Dict[str, Any]:
        """
        Run the complete Gibbs sampling algorithm.
        
        Args:
            X: Design matrix [T, n_features]
            rho: Target variable [T]
            n_iterations: Total number of iterations
            burn_in: Number of burn-in iterations
            thin: Thinning interval
            verbose: Whether to print progress
            
        Returns:
            Dict: Posterior samples and diagnostics
        """
        T = X.shape[0]
        
        # Initialize
        self.initialize(X, rho)
        
        # Storage for samples after burn-in
        n_samples = (n_iterations - burn_in) // thin
        
        # Pre-allocate storage
        beta_samples = np.zeros((n_samples, T, self.n_features, self.n_regimes))
        Q_samples = np.zeros((n_samples, self.n_features, self.n_features, self.n_regimes))
        sigma2_eps_samples = np.zeros((n_samples, self.n_regimes))
        P_samples = np.zeros((n_samples, self.n_regimes, self.n_regimes))
        states_samples = np.zeros((n_samples, T), dtype=int)
        rho_samples = np.zeros((n_samples, T))
        
        # Track acceptance rates
        beta_accepted = 0
        states_accepted = 0
        
        start_time = time.time()
        
        print(f"Starting Gibbs sampling with {n_iterations} iterations...")
        print(f"Burn-in: {burn_in}, Thinning: {thin}, Final samples: {n_samples}")
        
        sample_idx = 0
        
        for iteration in range(n_iterations):
            # Step 1: Sample time-varying coefficients beta
            self._sample_beta(X, rho)
            beta_accepted += 1
            
            # Step 2: Sample latent states
            self._sample_states(X, rho)
            states_accepted += 1
            
            # Step 3: Sample hyperparameters
            self._sample_hyperparameters(X, rho)
            
            # Step 4: Sample transition probabilities
            self._sample_transition_probs()
            
            # Store samples after burn-in
            if iteration >= burn_in and (iteration - burn_in) % thin == 0:
                beta_samples[sample_idx] = self.model.beta.copy()
                Q_samples[sample_idx] = self.model.Q.copy()
                sigma2_eps_samples[sample_idx] = self.model.sigma2_eps.copy()
                P_samples[sample_idx] = self.model.P.copy()
                states_samples[sample_idx] = self.model.states.copy()
                
                # Generate rho for this sample
                rho_samples[sample_idx] = self._generate_rho(X)
                
                sample_idx += 1
            
            # Print progress
            if verbose and iteration % 1000 == 0:
                elapsed = time.time() - start_time
                print(f"Iteration {iteration}/{n_iterations} ({elapsed:.1f}s elapsed)")
                
        # Update acceptance rates
        self.acceptance_rates['beta'] = beta_accepted / n_iterations
        self.acceptance_rates['states'] = states_accepted / n_iterations
        
        # Store results
        results = {
            'beta': beta_samples,
            'Q': Q_samples,
            'sigma2_eps': sigma2_eps_samples,
            'P': P_samples,
            'states': states_samples,
            'rho': rho_samples,
            'acceptance_rates': self.acceptance_rates,
            'n_iterations': n_iterations,
            'burn_in': burn_in,
            'thin': thin,
            'total_time': time.time() - start_time
        }
        
        self.posterior_samples = results
        
        if verbose:
            print(f"\nGibbs sampling completed in {results['total_time']:.1f} seconds")
            print(f"Final sample size: {n_samples}")
            print(f"Acceptance rates: Beta={self.acceptance_rates['beta']:.3f}, States={self.acceptance_rates['states']:.3f}")
            
        return results
    
    def _sample_beta(self, X: np.ndarray, rho: np.ndarray):
        """
        Sample time-varying coefficients using Carter-Kohn algorithm.
        
        Args:
            X: Design matrix [T, n_features]
            rho: Target variable [T]
        """
        T = X.shape[0]
        
        for j in range(self.n_regimes):
            # Get observations in regime j
            regime_mask = (self.model.states == j)
            
            if np.sum(regime_mask) > 0:
                # Sample beta path for regime j
                beta_sampled = self.model.backward_sampler(X, rho, j)
                
                # Update beta for time points in regime j
                regime_times = np.where(regime_mask)[0]
                for i, t in enumerate(regime_times):
                    self.model.beta[t, :, j] = beta_sampled[i, :]
    
    def _sample_states(self, X: np.ndarray, rho: np.ndarray):
        """
        Sample latent states using Hamilton filter and backward sampling.
        
        Args:
            X: Design matrix [T, n_features]
            rho: Target variable [T]
        """
        # Compute likelihood
        log_likelihood = self.model.compute_likelihood(X, rho)
        
        # Sample states
        self.model.states = self.model.sample_states(log_likelihood)
    
    def _sample_hyperparameters(self, X: np.ndarray, rho: np.ndarray):
        """
        Sample hyperparameters Q and sigma^2_eps.
        
        Args:
            X: Design matrix [T, n_features]
            rho: Target variable [T]
        """
        self.model.sample_hyperparameters(X, rho)
    
    def _sample_transition_probs(self):
        """
        Sample transition probability matrix.
        """
        self.model.sample_transition_probs()
    
    def _generate_rho(self, X: np.ndarray) -> np.ndarray:
        """
        Generate rho series given current parameters.
        
        Args:
            X: Design matrix [T, n_features]
            
        Returns:
            np.ndarray: Generated rho series
        """
        return self.model.generate_rho(X)
    
    def get_posterior_summary(self, parameter: str, 
                            percentiles: list = [5, 50, 95]) -> pd.DataFrame:
        """
        Get posterior summary statistics for a parameter.
        
        Args:
            parameter: Parameter name ('beta', 'rho', 'states', etc.)
            percentiles: Percentiles to compute
            
        Returns:
            pd.DataFrame: Posterior summary
        """
        if parameter not in self.posterior_samples:
            raise ValueError(f"Parameter {parameter} not found in posterior samples")
        
        samples = self.posterior_samples[parameter]
        
        if parameter == 'beta':
            # Summary for beta coefficients
            n_samples, T, n_features, n_regimes = samples.shape
            summary_data = []
            
            for t in range(T):
                for f in range(n_features):
                    for r in range(n_regimes):
                        posterior_samples = samples[:, t, f, r]
                        summary_data.append({
                            'time': t,
                            'feature': f,
                            'regime': r,
                            'mean': np.mean(posterior_samples),
                            'std': np.std(posterior_samples),
                            **{f'p{p}': np.percentile(posterior_samples, p) for p in percentiles}
                        })
            
            return pd.DataFrame(summary_data)
            
        elif parameter == 'rho':
            # Summary for rho series
            n_samples, T = samples.shape
            summary_data = []
            
            for t in range(T):
                posterior_samples = samples[:, t]
                summary_data.append({
                    'time': t,
                    'mean': np.mean(posterior_samples),
                    'std': np.std(posterior_samples),
                    **{f'p{p}': np.percentile(posterior_samples, p) for p in percentiles}
                })
            
            return pd.DataFrame(summary_data)
            
        elif parameter == 'states':
            # Summary for states (regime probabilities)
            n_samples, T = samples.shape
            summary_data = []
            
            for t in range(T):
                state_counts = np.bincount(samples[:, t], minlength=self.n_regimes)
                state_probs = state_counts / n_samples
                
                for r in range(self.n_regimes):
                    summary_data.append({
                        'time': t,
                        'regime': r,
                        'probability': state_probs[r]
                    })
            
            return pd.DataFrame(summary_data)
            
        else:
            # Default summary for other parameters
            return pd.DataFrame({
                'mean': np.mean(samples),
                'std': np.std(samples),
                **{f'p{p}': np.percentile(samples, p) for p in percentiles}
            }, index=[0])
    
    def get_rho_estimate(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the final estimate of rho and its uncertainty.
        
        Returns:
            Tuple: (rho_mean, rho_std) - mean and standard deviation of rho
        """
        if 'rho' not in self.posterior_samples:
            raise ValueError("Run Gibbs sampling first")
        
        rho_samples = self.posterior_samples['rho']
        rho_mean = np.mean(rho_samples, axis=0)
        rho_std = np.std(rho_samples, axis=0)
        
        return rho_mean, rho_std


def estimate_adaptive_rho(X: np.ndarray, rho_obs: np.ndarray = None,
                         n_iterations: int = 20000, burn_in: int = 10000,
                         n_regimes: int = 2, verbose: bool = True) -> Dict[str, Any]:
    """
    Convenience function to estimate adaptive rho parameter.
    
    Args:
        X: Design matrix [T, n_features]
        rho_obs: Observed rho values (if None, will be estimated)
        n_iterations: Total number of MCMC iterations
        burn_in: Number of burn-in iterations
        n_regimes: Number of regimes
        verbose: Whether to print progress
        
    Returns:
        Dict: Estimation results
    """
    # Initialize sampler
    n_features = X.shape[1]
    sampler = GibbsSampler(n_regimes, n_features)
    
    # If no observed rho, generate initial values
    if rho_obs is None:
        T = X.shape[0]
        rho_obs = np.random.randn(T)
    
    # Run Gibbs sampling
    results = sampler.run_gibbs_sampling(
        X, rho_obs, 
        n_iterations=n_iterations, 
        burn_in=burn_in,
        verbose=verbose
    )
    
    # Get final rho estimate
    rho_mean, rho_std = sampler.get_rho_estimate()
    
    results['rho_estimate'] = rho_mean
    results['rho_std'] = rho_std
    
    return results


if __name__ == "__main__":
    # Test the Gibbs sampler
    np.random.seed(42)
    
    # Generate test data
    T = 200
    n_features = 6
    X = np.random.randn(T, n_features)
    rho_obs = np.random.randn(T)
    
    # Run Gibbs sampling (smaller test)
    print("Running Gibbs sampler test...")
    results = estimate_adaptive_rho(
        X, rho_obs, 
        n_iterations=1000,  # Small test
        burn_in=500,
        verbose=True
    )
    
    print("\nEstimation completed!")
    print(f"Rho estimate shape: {results['rho_estimate'].shape}")
    print(f"Sample rho values: {results['rho_estimate'][:5]}")
    print(f"Rho std: {results['rho_std'][:5]}")