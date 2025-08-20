"""
Behavioral indicators module for creating Price Peak Proximity (PPP) and 
Price Trough Proximity (PTP) indicators.
"""

import numpy as np
import pandas as pd
from typing import Tuple


class BehavioralIndicators:
    """
    Create behavioral indicators based on price series.
    """
    
    def __init__(self):
        """Initialize the behavioral indicators calculator."""
        pass
    
    def calculate_expanding_high_low(self, price_series: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate expanding window all-time high and low prices.
        
        Args:
            price_series: Time series of prices
            
        Returns:
            Tuple: (expanding_high, expanding_low) series
        """
        expanding_high = price_series.expanding().max()
        expanding_low = price_series.expanding().min()
        
        return expanding_high, expanding_low
    
    def calculate_ppp_ptp(self, price_series: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate Price Peak Proximity (PPP) and Price Trough Proximity (PTP).
        
        Args:
            price_series: Time series of prices
            
        Returns:
            Tuple: (PPP, PTP) series
        """
        expanding_high, expanding_low = self.calculate_expanding_high_low(price_series)
        
        # Calculate PPP: how close current price is to all-time high
        ppp = (price_series - expanding_low) / (expanding_high - expanding_low)
        
        # Calculate PTP: how close current price is to all-time low
        ptp = (expanding_high - price_series) / (expanding_high - expanding_low)
        
        # Handle division by zero (when high = low)
        ppp = ppp.fillna(0.5)
        ptp = ptp.fillna(0.5)
        
        return ppp, ptp
    
    def create_behavioral_indicators(self, price_series: pd.Series) -> pd.DataFrame:
        """
        Create a DataFrame with PPP and PTP indicators.
        
        Args:
            price_series: Time series of prices
            
        Returns:
            pd.DataFrame: DataFrame with PPP and PTP columns
        """
        ppp, ptp = self.calculate_ppp_ptp(price_series)
        
        df = pd.DataFrame({
            'ppp': ppp,
            'ptp': ptp
        })
        
        return df
    
    def create_lagged_variables(self, df: pd.DataFrame, 
                              variables: list = None) -> pd.DataFrame:
        """
        Create lagged versions of specified variables.
        
        Args:
            df: DataFrame with original variables
            variables: List of variable names to lag (default: all columns)
            
        Returns:
            pd.DataFrame: DataFrame with lagged variables
        """
        if variables is None:
            variables = df.columns.tolist()
        
        lagged_df = df[variables].shift(1)
        lagged_df.columns = [f'{col}_lag1' for col in variables]
        
        return lagged_df


def create_behavioral_data(price_series: pd.Series) -> pd.DataFrame:
    """
    Convenience function to create behavioral indicators from price series.
    
    Args:
        price_series: Time series of prices
        
    Returns:
        pd.DataFrame: DataFrame with PPP and PTP indicators
    """
    bi = BehavioralIndicators()
    return bi.create_behavioral_indicators(price_series)


if __name__ == "__main__":
    # Test the behavioral indicators
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    prices = 100 + np.cumsum(np.random.normal(0, 1, 100))
    price_series = pd.Series(prices, index=dates, name='price')
    
    # Create indicators
    bi = BehavioralIndicators()
    behavioral_df = bi.create_behavioral_indicators(price_series)
    
    print("Behavioral Indicators:")
    print(behavioral_df.head())
    print("\nSummary:")
    print(behavioral_df.describe())