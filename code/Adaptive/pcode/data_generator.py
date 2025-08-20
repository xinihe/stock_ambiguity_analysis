"""
Data generation module for financial indicators used in adaptive ambiguity parameter estimation.
Generates synthetic price data, iVIX, EPU, and turnover rate.
"""

import numpy as np
import pandas as pd
from typing import Tuple


class DataGenerator:
    """
    Generate synthetic financial data for testing the adaptive ambiguity parameter estimation.
    """
    
    def __init__(self, n_obs: int = 1000, seed: int = 42):
        """
        Initialize the data generator.
        
        Args:
            n_obs: Number of observations to generate
            seed: Random seed for reproducibility
        """
        self.n_obs = n_obs
        np.random.seed(seed)
        
    def generate_price_data(self, initial_price: float = 100.0, 
                          volatility: float = 0.02, 
                          trend: float = 0.0001) -> pd.Series:
        """
        Generate synthetic price data using geometric Brownian motion.
        
        Args:
            initial_price: Starting price level
            volatility: Price volatility
            trend: Long-term trend component
            
        Returns:
            pd.Series: Time series of prices
        """
        # Generate random returns
        returns = np.random.normal(trend, volatility, self.n_obs)
        
        # Convert to price series
        prices = [initial_price]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
            
        return pd.Series(prices, name='price')
    
    def generate_ivix(self, base_level: float = 20.0, volatility: float = 5.0) -> pd.Series:
        """
        Generate synthetic iVIX (implied volatility index) data.
        
        Args:
            base_level: Base level of iVIX
            volatility: Volatility of iVIX itself
            
        Returns:
            pd.Series: Time series of iVIX values
        """
        # Mean-reverting process for iVIX
        ivix_values = [base_level]
        for i in range(1, self.n_obs):
            # Mean reversion with random shock
            shock = np.random.normal(0, volatility)
            new_ivix = ivix_values[-1] + 0.1 * (base_level - ivix_values[-1]) + shock
            ivix_values.append(max(new_ivix, 5.0))  # Ensure positive values
            
        return pd.Series(ivix_values, name='ivix')
    
    def generate_epu(self, base_level: float = 100.0, volatility: float = 15.0) -> pd.Series:
        """
        Generate synthetic EPU (Economic Policy Uncertainty) index data.
        
        Args:
            base_level: Base level of EPU
            volatility: Volatility of EPU
            
        Returns:
            pd.Series: Time series of EPU values
        """
        # EPU tends to have spikes and slower decay
        epu_values = [base_level]
        for i in range(1, self.n_obs):
            # Occasional spikes with slow decay
            if np.random.random() < 0.05:  # 5% chance of spike
                spike = np.random.exponential(50)
                new_epu = epu_values[-1] + spike
            else:
                # Slow decay with noise
                shock = np.random.normal(0, volatility)
                new_epu = epu_values[-1] - 0.05 * (epu_values[-1] - base_level) + shock
            
            epu_values.append(max(new_epu, 20.0))  # Ensure positive values
            
        return pd.Series(epu_values, name='epu')
    
    def generate_turnover_rate(self, base_level: float = 0.02, volatility: float = 0.01) -> pd.Series:
        """
        Generate synthetic turnover rate data.
        
        Args:
            base_level: Base turnover rate
            volatility: Volatility of turnover rate
            
        Returns:
            pd.Series: Time series of turnover rates
        """
        # Turnover rate with mean reversion
        turnover_values = [base_level]
        for i in range(1, self.n_obs):
            shock = np.random.normal(0, volatility)
            new_turnover = turnover_values[-1] + 0.2 * (base_level - turnover_values[-1]) + shock
            turnover_values.append(max(new_turnover, 0.001))  # Ensure positive values
            
        return pd.Series(turnover_values, name='turnover')
    
    def generate_all_data(self) -> pd.DataFrame:
        """
        Generate all financial indicators and return as a DataFrame.
        
        Returns:
            pd.DataFrame: DataFrame containing all generated series
        """
        price = self.generate_price_data()
        ivix = self.generate_ivix()
        epu = self.generate_epu()
        turnover = self.generate_turnover_rate()
        
        df = pd.DataFrame({
            'price': price,
            'ivix': ivix,
            'epu': epu,
            'turnover': turnover
        })
        
        return df


def generate_sample_data(n_obs: int = 1000) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Convenience function to generate sample data and behavioral indicators.
    
    Args:
        n_obs: Number of observations
        
    Returns:
        Tuple: (DataFrame with all data, Series with behavioral indicators)
    """
    generator = DataGenerator(n_obs)
    df = generator.generate_all_data()
    
    # Create behavioral indicators
    from .behavioral_indicators import BehavioralIndicators
    bi = BehavioralIndicators()
    behavioral_df = bi.create_behavioral_indicators(df['price'])
    
    # Combine all data
    full_df = pd.concat([df, behavioral_df], axis=1)
    
    return full_df


if __name__ == "__main__":
    # Generate sample data
    df = generate_sample_data(1000)
    print("Generated data shape:", df.shape)
    print("\nFirst few rows:")
    print(df.head())
    print("\nData summary:")
    print(df.describe())