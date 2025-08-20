"""
Markov Switching Time-Varying Parameter (MS-TVP) model implementation.
Implements the state-space representation for the adaptive ambiguity parameter estimation.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional
from collections import defaultdict


# Manual implementations of statistical distributions
class multivariate_normal:
    @staticmethod
    def rvs(mean=None, cov=1, size=1):
        """Generate random samples from multivariate normal distribution."""
        if mean is None:
            mean = np.zeros(cov.shape[0] if isinstance(cov, np.ndarray) else 1)
        return np.random.multivariate_normal(mean, cov, size=size)
    
    @staticmethod
    def logpdf(x, mean, cov):
        """Log probability density function."""
        n = len(mean)
        det = np.linalg.det(cov)
        inv = np.linalg.inv(cov)
        diff = x - mean
        return -0.5 * (n * np.log(2 * np.pi) + np.log(det) + diff @ inv @ diff)


class invgamma:
    @staticmethod
    def rvs(a, scale=1, size=1):
        """Generate random samples from inverse gamma distribution."""
        return 1.0 / np.random.gamma(a, 1/scale, size=size)


class invwishart:
    @staticmethod
    def rvs(df, scale, size=1):
        """Generate random samples from inverse Wishart distribution."""
        # Generate from Wishart distribution and invert
        p = scale.shape[0]
        # Generate chi-square samples
        chi2_samples = np.array([np.random.chisquare(df - i + 1) for i in range(p)])
        # Create lower triangular matrix
        L = np.zeros((p, p))
        for i in range(p):
            L[i, i] = np.sqrt(chi2_samples[i])
            for j in range(i):
                L[i, j] = np.random.normal(0, 1)
        # Create Wishart sample
        A = L @ scale @ L.T
        return np.linalg.inv(A)


class dirichlet:
    @staticmethod
    def rvs(alpha, size=1):
        """Generate random samples from Dirichlet distribution."""
        if size == 1:
            gamma_samples = np.random.gamma(alpha, 1)
            return gamma_samples / gamma_samples.sum()
        else:
            gamma_samples = np.random.gamma(alpha, 1, size=(size, len(alpha)))
            return gamma_samples / gamma_samples.sum(axis=1, keepdims=True)


class MSTVPModel:
    """
    Markov Switching Time-Varying Parameter model for estimating adaptive ambiguity parameter.
    """
    
    def __init__(self, n_regimes: int = 2, n_features: int = 6):
        """
        Initialize the MS-TVP model.
        
        Args:
            n_regimes: Number of regimes (default: 2)
            n_features: Number of features including intercept
        """
        self.n_regimes = n_regimes
        self.n_features = n_features
        
        # Initialize parameters
        self.beta = None  # Time-varying coefficients [T, n_features, n_regimes]
        self.Q = None  # State innovation covariance [n_features, n_features, n_regimes]
        self.sigma2_eps = None  # Measurement error variance [n_regimes]
        self.P = None  # Transition probability matrix [n_regimes, n_regimes]
        self.states = None  # Regime states [T]
        
        # Priors
        self.prior_Q = None
        self.prior_sigma2_eps = None
        self.prior_P = None
        
    def set_priors(self, prior_Q: Dict[str, np.ndarray] = None,
                   prior_sigma2_eps: Dict[str, np.ndarray] = None,
                   prior_P: Dict[str, np.ndarray] = None):
        """
        Set prior distributions for model parameters.
        
        Args:
            prior_Q: Prior for Q matrices (Inverse-Wishart)
            prior_sigma2_eps: Prior for measurement error variances (Inverse-Gamma)
            prior_P: Prior for transition probabilities (Dirichlet)
        """
        # Default priors
        if prior_Q is None:
            # Inverse-Wishart prior for Q
            self.prior_Q = {
                'df': self.n_features + 2,
                'scale': np.eye(self.n_features) * 0.1
            }
        else:
            self.prior_Q = prior_Q
            
        if prior_sigma2_eps is None:
            # Inverse-Gamma prior for sigma^2_eps
            self.prior_sigma2_eps = {
                'alpha': 3.0,
                'beta': 1.0
            }
        else:
            self.prior_sigma2_eps = prior_sigma2_eps
            
        if prior_P is None:
            # Dirichlet prior for transition probabilities
            self.prior_P = {
                'alpha': np.ones((self.n_regimes, self.n_regimes)) * 2.0
            }
        else:
            self.prior_P = prior_P
    
    def initialize_parameters(self, T: int):
        """
        Initialize model parameters.
        
        Args:
            T: Number of time periods
        """
        # Initialize time-varying coefficients
        self.beta = np.random.randn(T, self.n_features, self.n_regimes) * 0.1
        
        # Initialize state innovation covariances
        self.Q = np.zeros((self.n_features, self.n_features, self.n_regimes))
        for j in range(self.n_regimes):
            self.Q[:, :, j] = np.eye(self.n_features) * 0.01
            
        # Initialize measurement error variances
        self.sigma2_eps = np.ones(self.n_regimes) * 0.1
        
        # Initialize transition probability matrix
        self.P = np.ones((self.n_regimes, self.n_regimes)) / self.n_regimes
        np.fill_diagonal(self.P, 0.8)  # High persistence
        
        # Initialize states
        self.states = np.random.choice(self.n_regimes, size=T)
        
    def measurement_equation(self, X: np.ndarray, t: int) -> np.ndarray:
        """
        Measurement equation: rho_t = X_t' * beta_{t,s_t} + epsilon_t
        
        Args:
            X: Design matrix [T, n_features]
            t: Time index
            
        Returns:
            np.ndarray: Predicted rho values for each regime
        """
        rho_pred = np.zeros(self.n_regimes)
        for j in range(self.n_regimes):
            rho_pred[j] = np.dot(X[t, :], self.beta[t, :, j])
        return rho_pred
    
    def state_equation(self, beta_prev: np.ndarray, regime: int) -> np.ndarray:
        """
        State equation: beta_{t,s_t} = beta_{t-1,s_t} + eta_t
        
        Args:
            beta_prev: Previous period coefficients [n_features]
            regime: Current regime
            
        Returns:
            np.ndarray: Current period coefficients
        """
        innovation = multivariate_normal.rvs(cov=self.Q[:, :, regime])
        return beta_prev + innovation
    
    def compute_likelihood(self, X: np.ndarray, rho: np.ndarray) -> np.ndarray:
        """
        Compute likelihood of observations given current parameters.
        
        Args:
            X: Design matrix [T, n_features]
            rho: Target variable [T]
            
        Returns:
            np.ndarray: Log-likelihood for each time period and regime
        """
        T = X.shape[0]
        log_likelihood = np.zeros((T, self.n_regimes))
        
        for t in range(T):
            for j in range(self.n_regimes):
                # Predict rho_t
                rho_pred = self.measurement_equation(X, t)[j]
                
                # Compute log-likelihood
                residual = rho[t] - rho_pred
                log_likelihood[t, j] = -0.5 * (np.log(2 * np.pi * self.sigma2_eps[j]) + 
                                               residual**2 / self.sigma2_eps[j])
        
        return log_likelihood
    
    def kalman_filter(self, X: np.ndarray, rho: np.ndarray, regime: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Kalman filter for a single regime.
        
        Args:
            X: Design matrix [T, n_features]
            rho: Target variable [T]
            regime: Regime index
            
        Returns:
            Tuple: (filtered_means, filtered_covariances)
        """
        T = X.shape[0]
        
        # Initialize
        beta_filtered = np.zeros((T, self.n_features))
        P_filtered = np.zeros((T, self.n_features, self.n_features))
        
        # Initial state
        beta_filtered[0, :] = self.beta[0, :, regime]
        P_filtered[0, :, :] = np.eye(self.n_features) * 0.1
        
        Q_regime = self.Q[:, :, regime]
        sigma2_regime = self.sigma2_eps[regime]
        
        for t in range(1, T):
            # Prediction step
            beta_pred = beta_filtered[t-1, :]
            P_pred = P_filtered[t-1, :, :] + Q_regime
            
            # Update step
            H = X[t, :].reshape(1, -1)
            S = H @ P_pred @ H.T + sigma2_regime
            K = P_pred @ H.T / S
            
            residual = rho[t] - H @ beta_pred
            beta_filtered[t, :] = beta_pred + K.flatten() * residual
            P_filtered[t, :, :] = P_pred - K @ H @ P_pred
            
        return beta_filtered, P_filtered
    
    def backward_sampler(self, X: np.ndarray, rho: np.ndarray, regime: int) -> np.ndarray:
        """
        Backward sampler for time-varying coefficients (Carter and Kohn, 1994).
        
        Args:
            X: Design matrix [T, n_features]
            rho: Target variable [T]
            regime: Regime index
            
        Returns:
            np.ndarray: Sampled coefficient path [T, n_features]
        """
        T = X.shape[0]
        
        # Forward pass (Kalman filter)
        beta_filtered, P_filtered = self.kalman_filter(X, rho, regime)
        
        # Backward pass
        beta_sampled = np.zeros((T, self.n_features))
        beta_sampled[-1, :] = multivariate_normal.rvs(
            mean=beta_filtered[-1, :], 
            cov=P_filtered[-1, :, :]
        )
        
        Q_regime = self.Q[:, :, regime]
        
        for t in range(T-2, -1, -1):
            # Predict next state
            beta_pred = beta_sampled[t+1, :]
            P_pred = P_filtered[t, :, :] + Q_regime
            
            # Combine with filtered estimate
            P_combined = np.linalg.inv(np.linalg.inv(P_filtered[t, :, :]) + np.linalg.inv(Q_regime))
            beta_combined = P_combined @ (np.linalg.inv(P_filtered[t, :, :]) @ beta_filtered[t, :] + 
                                         np.linalg.inv(Q_regime) @ beta_pred)
            
            # Sample
            beta_sampled[t, :] = multivariate_normal.rvs(
                mean=beta_combined, 
                cov=P_combined
            )
            
        return beta_sampled
    
    def hamilton_filter(self, log_likelihood: np.ndarray) -> np.ndarray:
        """
        Hamilton filter for regime probabilities.
        
        Args:
            log_likelihood: Log-likelihood [T, n_regimes]
            
        Returns:
            np.ndarray: Filtered probabilities [T, n_regimes]
        """
        T = log_likelihood.shape[0]
        
        # Initialize
        filtered_probs = np.zeros((T, self.n_regimes))
        filtered_probs[0, :] = 1.0 / self.n_regimes  # Equal initial probabilities
        
        # Forward filtering
        for t in range(1, T):
            # Prediction step
            pred_probs = filtered_probs[t-1, :] @ self.P
            
            # Update step
            likelihood_weights = np.exp(log_likelihood[t, :])
            filtered_probs[t, :] = pred_probs * likelihood_weights
            filtered_probs[t, :] /= filtered_probs[t, :].sum()  # Normalize
            
        return filtered_probs
    
    def sample_states(self, log_likelihood: np.ndarray) -> np.ndarray:
        """
        Sample regime states using backward sampling.
        
        Args:
            log_likelihood: Log-likelihood [T, n_regimes]
            
        Returns:
            np.ndarray: Sampled states [T]
        """
        T = log_likelihood.shape[0]
        
        # Forward filter
        filtered_probs = self.hamilton_filter(log_likelihood)
        
        # Backward sampling
        states = np.zeros(T, dtype=int)
        
        # Sample final state
        states[-1] = np.random.choice(self.n_regimes, p=filtered_probs[-1, :])
        
        # Sample backward
        for t in range(T-2, -1, -1):
            # Conditional probabilities
            cond_probs = filtered_probs[t, :] * self.P[:, states[t+1]]
            cond_probs /= cond_probs.sum()
            states[t] = np.random.choice(self.n_regimes, p=cond_probs)
            
        return states
    
    def sample_hyperparameters(self, X: np.ndarray, rho: np.ndarray):
        """
        Sample hyperparameters (Q and sigma^2_eps).
        
        Args:
            X: Design matrix [T, n_features]
            rho: Target variable [T]
        """
        T = X.shape[0]
        
        for j in range(self.n_regimes):
            # Get observations in regime j
            regime_mask = (self.states == j)
            n_regime = regime_mask.sum()
            
            if n_regime > 0:
                # Sample sigma^2_eps (Inverse-Gamma)
                residuals = np.zeros(n_regime)
                X_regime = X[regime_mask]
                rho_regime = rho[regime_mask]
                
                for i, t in enumerate(np.where(regime_mask)[0]):
                    residuals[i] = rho_regime[i] - np.dot(X_regime[i], self.beta[t, :, j])
                
                # Update Inverse-Gamma parameters
                alpha_post = self.prior_sigma2_eps['alpha'] + n_regime / 2
                beta_post = self.prior_sigma2_eps['beta'] + 0.5 * np.sum(residuals**2)
                
                self.sigma2_eps[j] = invgamma.rvs(alpha_post, scale=beta_post)
                
                # Sample Q (Inverse-Wishart)
                if n_regime > 1:
                    # Compute innovations
                    innovations = np.zeros((n_regime-1, self.n_features))
                    regime_times = np.where(regime_mask)[0]
                    
                    for i in range(1, len(regime_times)):
                        t_prev = regime_times[i-1]
                        t_curr = regime_times[i]
                        innovations[i-1, :] = self.beta[t_curr, :, j] - self.beta[t_prev, :, j]
                    
                    # Update Inverse-Wishart parameters
                    df_post = self.prior_Q['df'] + n_regime - 1
                    scale_post = self.prior_Q['scale'] + innovations.T @ innovations
                    
                    self.Q[:, :, j] = invwishart.rvs(df_post, scale_post)
    
    def sample_transition_probs(self):
        """
        Sample transition probability matrix (Dirichlet).
        """
        # Count transitions
        transition_counts = np.zeros((self.n_regimes, self.n_regimes))
        
        for t in range(1, len(self.states)):
            transition_counts[self.states[t-1], self.states[t]] += 1
        
        # Sample each row
        for i in range(self.n_regimes):
            alpha_post = self.prior_P['alpha'][i, :] + transition_counts[i, :]
            self.P[i, :] = dirichlet.rvs(alpha_post)[0]
    
    def generate_rho(self, X: np.ndarray) -> np.ndarray:
        """
        Generate rho series given current parameters.
        
        Args:
            X: Design matrix [T, n_features]
            
        Returns:
            np.ndarray: Generated rho series [T]
        """
        T = X.shape[0]
        rho = np.zeros(T)
        
        for t in range(T):
            regime = self.states[t]
            rho[t] = np.dot(X[t, :], self.beta[t, :, regime])
            rho[t] += np.random.normal(0, np.sqrt(self.sigma2_eps[regime]))
            
        return rho


if __name__ == "__main__":
    # Test the MS-TVP model
    model = MSTVPModel(n_regimes=2, n_features=6)
    model.set_priors()
    
    # Generate test data
    T = 100
    X = np.random.randn(T, 6)
    rho = np.random.randn(T)
    
    # Initialize parameters
    model.initialize_parameters(T)
    
    # Test likelihood computation
    log_likelihood = model.compute_likelihood(X, rho)
    print("Log likelihood shape:", log_likelihood.shape)
    print("Sample log likelihood:", log_likelihood[0, :])
    
    # Test state sampling
    states = model.sample_states(log_likelihood)
    print("Sampled states shape:", states.shape)
    print("State distribution:", np.bincount(states) / len(states))