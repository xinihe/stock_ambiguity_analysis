"""
Data preprocessing module for standardizing and preparing data for the MS-TVP model.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any


class StandardScaler:
    """
    Manual implementation of StandardScaler for standardizing data.
    """
    def __init__(self):
        self.mean_ = None
        self.scale_ = None
        
    def fit(self, X):
        """Fit the scaler to the data."""
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        self.scale_[self.scale_ == 0] = 1.0  # Avoid division by zero
        return self
        
    def transform(self, X):
        """Transform the data using fitted parameters."""
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("Scaler not fitted yet")
        return (X - self.mean_) / self.scale_
        
    def fit_transform(self, X):
        """Fit and transform the data."""
        return self.fit(X).transform(X)
        
    def inverse_transform(self, X):
        """Inverse transform the data."""
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("Scaler not fitted yet")
        return X * self.scale_ + self.mean_


class DataPreprocessor:
    """
    Preprocess data for the adaptive ambiguity parameter estimation.
    """
    
    def __init__(self):
        """Initialize the data preprocessor."""
        self.scalers = {}
        self.feature_names = []
        
    def prepare_data(self, df: pd.DataFrame, 
                    target_col: str = None,
                    feature_cols: list = None) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Prepare data for model estimation.
        
        Args:
            df: Input DataFrame
            target_col: Name of target variable column
            feature_cols: List of feature column names
            
        Returns:
            Tuple: (X, y, preprocessing_info)
        """
        if feature_cols is None:
            # Default features from the paper
            feature_cols = ['ivix_lag1', 'epu_lag1', 'turnover_lag1', 'ppp_lag1', 'ptp_lag1']
            
        # Add intercept column
        X = df[feature_cols].copy()
        X['intercept'] = 1.0
        
        # Reorder to put intercept first
        X = X[['intercept'] + feature_cols]
        
        # Handle missing values
        X = X.dropna()
        
        if target_col is not None and target_col in df.columns:
            y = df.loc[X.index, target_col].values
        else:
            y = None
            
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Standardize features (except intercept)
        X_scaled = X.copy()
        if len(feature_cols) > 0:
            scaler = StandardScaler()
            X_scaled[feature_cols] = scaler.fit_transform(X[feature_cols])
            self.scalers['features'] = scaler
            
        return X_scaled.values, y, {
            'feature_names': self.feature_names,
            'index': X.index,
            'n_features': X_scaled.shape[1],
            'n_obs': X_scaled.shape[0]
        }
    
    def standardize_series(self, series: pd.Series) -> Tuple[np.ndarray, StandardScaler]:
        """
        Standardize a single time series.
        
        Args:
            series: Input series
            
        Returns:
            Tuple: (standardized_array, fitted_scaler)
        """
        scaler = StandardScaler()
        scaled_values = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()
        return scaled_values, scaler
    
    def inverse_standardize(self, scaled_values: np.ndarray, 
                          scaler: StandardScaler) -> np.ndarray:
        """
        Inverse transform standardized values.
        
        Args:
            scaled_values: Standardized values
            scaler: Fitted scaler
            
        Returns:
            np.ndarray: Original scale values
        """
        return scaler.inverse_transform(scaled_values.reshape(-1, 1)).flatten()
    
    def create_lagged_features(self, df: pd.DataFrame, 
                             variables: list, 
                             lags: list = [1]) -> pd.DataFrame:
        """
        Create lagged features for specified variables.
        
        Args:
            df: Input DataFrame
            variables: List of variable names to lag
            lags: List of lag periods
            
        Returns:
            pd.DataFrame: DataFrame with lagged features
        """
        lagged_df = df.copy()
        
        for var in variables:
            for lag in lags:
                lagged_df[f'{var}_lag{lag}'] = df[var].shift(lag)
                
        return lagged_df
    
    def prepare_full_dataset(self, raw_df: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Prepare the full dataset according to the paper's specifications.
        
        Args:
            raw_df: DataFrame with raw price, ivix, epu, turnover data
            
        Returns:
            Tuple: (prepared_data, info_dict)
        """
        # Create behavioral indicators
        from .behavioral_indicators import BehavioralIndicators
        bi = BehavioralIndicators()
        behavioral_df = bi.create_behavioral_indicators(raw_df['price'])
        
        # Combine data
        combined_df = pd.concat([raw_df, behavioral_df], axis=1)
        
        # Create lagged variables
        variables_to_lag = ['ivix', 'epu', 'turnover', 'ppp', 'ptp']
        lagged_df = self.create_lagged_features(combined_df, variables_to_lag)
        
        # Prepare final dataset
        X, _, info = self.prepare_data(lagged_df)
        
        return X, info


def create_design_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    Create design matrix with lagged features and intercept.
    
    Args:
        df: DataFrame with all variables
        
    Returns:
        np.ndarray: Design matrix X
    """
    preprocessor = DataPreprocessor()
    X, _, _ = preprocessor.prepare_full_dataset(df)
    return X


if __name__ == "__main__":
    # Test the preprocessing
    from .data_generator import DataGenerator
    
    # Generate sample data
    generator = DataGenerator(n_obs=500)
    raw_df = generator.generate_all_data()
    
    # Preprocess data
    preprocessor = DataPreprocessor()
    X, info = preprocessor.prepare_full_dataset(raw_df)
    
    print("Preprocessed data shape:", X.shape)
    print("Feature names:", info['feature_names'])
    print("\nFirst few rows of design matrix:")
    print(X[:5])
    
    # Test standardization
    test_series = raw_df['ivix']
    scaled, scaler = preprocessor.standardize_series(test_series)
    print(f"\nOriginal ivix mean: {test_series.mean():.4f}, std: {test_series.std():.4f}")
    print(f"Scaled ivix mean: {scaled.mean():.4f}, std: {scaled.std():.4f}")