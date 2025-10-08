import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class CombinedUncertaintyIndex:
    def __init__(self):
        """Initialize the Combined Uncertainty Index creator"""
        self.climate_data = None
        self.gpr_data = None
        self.monthly_climate = None
        self.normalized_climate = None
        self.normalized_gpr = None
        self.combined_index = None
    
    def load_data(self, climate_file, gpr_file):
        """Load climate risk and geopolitical risk data"""
        print(f"Loading climate risk data from {climate_file}")
        self.climate_data = pd.read_csv(climate_file)
        self.climate_data['Date'] = pd.to_datetime(self.climate_data['Date'])
        
        print(f"Loading geopolitical risk data from {gpr_file}")
        self.gpr_data = pd.read_csv(gpr_file)
        self.gpr_data['Date'] = pd.to_datetime(self.gpr_data['Date'])
        
        print(f"Climate data shape: {self.climate_data.shape}")
        print(f"GPR data shape: {self.gpr_data.shape}")
        
        return self
    
    def aggregate_climate_to_monthly(self):
        """Aggregate daily climate risk data to monthly frequency"""
        print("Aggregating climate risk data to monthly frequency...")
        
        # Set date as index
        temp_df = self.climate_data.set_index('Date').copy()
        
        # Calculate monthly averages
        self.monthly_climate = temp_df.resample('MS').mean()
        
        # Create a composite climate risk index by averaging all components
        self.monthly_climate['Climate_Risk_Composite'] = self.monthly_climate.mean(axis=1)
        
        print(f"Monthly climate data shape: {self.monthly_climate.shape}")
        return self
    
    def normalize_series(self, method='standardize'):
        """Normalize the time series to ensure comparability"""
        print(f"Normalizing series using {method} method...")
        
        if method == 'standardize':
            # Standardize (z-score normalization)
            self.normalized_climate = (self.monthly_climate['Climate_Risk_Composite'] - 
                                     self.monthly_climate['Climate_Risk_Composite'].mean()) / \
                                     self.monthly_climate['Climate_Risk_Composite'].std()
            
            # For GPR, create a composite index by averaging country values
            gpr_composite = self.gpr_data.set_index('Date').mean(axis=1)
            self.normalized_gpr = (gpr_composite - gpr_composite.mean()) / gpr_composite.std()
        
        elif method == 'scale_100':
            # Scale to a mean of 100
            self.normalized_climate = (self.monthly_climate['Climate_Risk_Composite'] / 
                                     self.monthly_climate['Climate_Risk_Composite'].mean()) * 100
            
            gpr_composite = self.gpr_data.set_index('Date').mean(axis=1)
            self.normalized_gpr = (gpr_composite / gpr_composite.mean()) * 100
        
        # Align the series by date
        aligned_data = pd.DataFrame({
            'Normalized_Climate': self.normalized_climate,
            'Normalized_GPR': self.normalized_gpr
        }).dropna()
        
        self.normalized_climate = aligned_data['Normalized_Climate']
        self.normalized_gpr = aligned_data['Normalized_GPR']
        
        print(f"Aligned data shape: {aligned_data.shape}")
        return self
    
    def beta_weighting_function(self, k, theta, m):
        """Beta polynomial weighting function for MIDAS"""
        # Beta function implementation
        w = (k / m) ** (theta[0] - 1) * ((m - k) / m) ** (theta[1] - 1)
        return w / np.sum(w)  # Normalize weights
    
    def apply_midas_weighting(self, theta=[2.0, 2.0]):
        """Apply MIDAS weighting to GPR data (simulated for monthly data)"""
        print("Applying MIDAS weighting approach...")
        
        # Since our GPR data is already monthly, we'll simulate the time-decay effect
        # by applying weights to recent observations
        
        gpr_series = self.normalized_gpr.copy()
        weighted_gpr = pd.Series(index=gpr_series.index)
        
        # Use a 12-month window for weighting
        window_size = 12
        
        for i in range(len(gpr_series)):
            if i < window_size:
                # For the first few observations, use available data
                window = gpr_series.iloc[:i+1]
                m = len(window)
                weights = [self.beta_weighting_function(k, theta, m) for k in range(1, m+1)]
            else:
                # For later observations, use the full window
                window = gpr_series.iloc[i-window_size+1:i+1]
                weights = [self.beta_weighting_function(k, theta, window_size) for k in range(1, window_size+1)]
            
            # Apply weights
            weighted_gpr.iloc[i] = np.sum(window * weights)
        
        self.normalized_gpr = weighted_gpr
        return self
    
    def calculate_pca_weights(self):
        """Calculate weights using Principal Component Analysis (simplified implementation without scikit-learn)"""
        print("Calculating weights using Principal Component Analysis...")
        
        # Create a dataframe with the normalized components
        components_df = pd.DataFrame({
            'Climate_Risk': self.normalized_climate,
            'GPR': self.normalized_gpr
        })
        
        # Calculate covariance matrix
        covariance_matrix = np.cov(components_df.T)
        
        # Calculate eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        
        # Sort eigenvalues and corresponding eigenvectors in descending order
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Get the first principal component (highest eigenvalue)
        first_pc = eigenvectors[:, 0]
        
        # Use absolute values of the eigenvector as weights
        weights = np.abs(first_pc)
        
        # Normalize weights to sum to 1
        normalized_weights = weights / np.sum(weights)
        
        weight_climate, weight_gpr = normalized_weights
        
        print(f"PCA-derived weights - Climate: {weight_climate:.4f}, GPR: {weight_gpr:.4f}")
        print(f"Explained variance ratio: {eigenvalues[0] / np.sum(eigenvalues):.4f}")
        
        return weight_climate, weight_gpr
        
    def calculate_inverse_variance_weights(self):
        """Calculate weights based on inverse variance (lower weight to more volatile components)"""
        print("Calculating weights using inverse variance approach...")
        
        # Calculate variances
        climate_var = self.normalized_climate.var()
        gpr_var = self.normalized_gpr.var()
        
        # Calculate inverse variance weights
        weight_climate = 1 / climate_var if climate_var != 0 else 0
        weight_gpr = 1 / gpr_var if gpr_var != 0 else 0
        
        # Normalize weights to sum to 1
        total = weight_climate + weight_gpr
        if total == 0:
            # Fallback to equal weights if both variances are zero
            weight_climate, weight_gpr = 0.5, 0.5
        else:
            weight_climate /= total
            weight_gpr /= total
        
        print(f"Inverse variance weights - Climate: {weight_climate:.4f}, GPR: {weight_gpr:.4f}")
        print(f"Climate variance: {climate_var:.4f}, GPR variance: {gpr_var:.4f}")
        
        return weight_climate, weight_gpr
        
    def calculate_correlation_adjusted_weights(self, target_correlation=0.5):
        """Calculate weights that balance the contributions of both components
        while accounting for their correlation with each other"""
        print("Calculating weights using correlation-adjusted approach...")
        
        # Calculate correlation between components
        components_df = pd.DataFrame({
            'Climate_Risk': self.normalized_climate,
            'GPR': self.normalized_gpr
        })
        correlation = components_df.corr().iloc[0, 1]
        
        # Calculate base weights using inverse variance
        base_climate, base_gpr = self.calculate_inverse_variance_weights()
        
        # Adjust weights to achieve a better balance
        # If components are highly correlated, move toward equal weights
        # If components are not correlated, keep the inverse variance weights
        correlation_factor = abs(correlation)
        weight_climate = base_climate * (1 - correlation_factor) + target_correlation * correlation_factor
        weight_gpr = base_gpr * (1 - correlation_factor) + target_correlation * correlation_factor
        
        # Normalize weights to sum to 1
        total = weight_climate + weight_gpr
        weight_climate /= total
        weight_gpr /= total
        
        print(f"Correlation-adjusted weights - Climate: {weight_climate:.4f}, GPR: {weight_gpr:.4f}")
        print(f"Correlation between components: {correlation:.4f}")
        
        return weight_climate, weight_gpr
        
    def calculate_ewma_weights(self, span=24):
        """Calculate dynamic weights using Exponentially Weighted Moving Average of variance ratios"""
        print(f"Calculating weights using EWMA approach (span={span})...")
        
        # Create a dataframe with the normalized components
        components_df = pd.DataFrame({
            'Climate_Risk': self.normalized_climate,
            'GPR': self.normalized_gpr
        })
        
        # Calculate rolling 6-month variance for each component
        climate_rolling_var = components_df['Climate_Risk'].rolling(window=6).var()
        gpr_rolling_var = components_df['GPR'].rolling(window=6).var()
        
        # Calculate inverse variance ratio
        ratio = gpr_rolling_var / climate_rolling_var
        ratio = ratio.fillna(1.0)  # Fill NaN with neutral ratio
        
        # Apply EWMA to smooth the ratio over time
        ewma_ratio = ratio.ewm(span=span, adjust=False).mean()
        
        # Calculate final weights
        weights = pd.DataFrame({
            'Climate': ewma_ratio / (1 + ewma_ratio),
            'GPR': 1 / (1 + ewma_ratio)
        })
        
        # Use the most recent weights
        weight_climate = weights['Climate'].iloc[-1]
        weight_gpr = weights['GPR'].iloc[-1]
        
        print(f"EWMA-derived weights - Climate: {weight_climate:.4f}, GPR: {weight_gpr:.4f}")
        print(f"Recent variance ratio (GPR/Climate): {ratio.iloc[-1]:.4f}")
        
        return weight_climate, weight_gpr
    
    def create_composite_index(self, weight_climate=None, weight_gpr=None, method='correlation_adjusted', include_both=False):
        """Create the final composite index with specified weights or method
        
        Parameters:
        -----------
        weight_climate : float, optional
            Weight for climate risk component
        weight_gpr : float, optional
            Weight for GPR component
        method : str, optional
            Weighting method ('pca', 'equal', 'custom', 'inverse_variance', 'correlation_adjusted', 'ewma')
        include_both : bool, optional
            If True, creates indices with both correlation-adjusted and equal weighting methods
        """
        # Create a dataframe with aligned components
        components_df = pd.DataFrame({
            'Climate_Risk_Component': self.normalized_climate,
            'GPR_Component': self.normalized_gpr
        }).dropna()
        
        # Get weights for the primary method
        if method == 'pca':
            weight_climate_primary, weight_gpr_primary = self.calculate_pca_weights()
        elif method == 'equal':
            weight_climate_primary, weight_gpr_primary = 0.5, 0.5
        elif method == 'inverse_variance':
            weight_climate_primary, weight_gpr_primary = self.calculate_inverse_variance_weights()
        elif method == 'correlation_adjusted':
            weight_climate_primary, weight_gpr_primary = self.calculate_correlation_adjusted_weights()
        elif method == 'ewma':
            weight_climate_primary, weight_gpr_primary = self.calculate_ewma_weights()
        elif method == 'custom':
            if weight_climate is None or weight_gpr is None:
                raise ValueError("Custom weights must be specified")
            weight_climate_primary, weight_gpr_primary = weight_climate, weight_gpr
        else:
            raise ValueError(f"Unknown weighting method: {method}")
        
        print(f"Creating primary composite index with weights - Climate: {weight_climate_primary:.4f}, GPR: {weight_gpr_primary:.4f}")
        
        # Ensure weights sum to 1
        if abs(weight_climate_primary + weight_gpr_primary - 1.0) > 1e-10:
            raise ValueError("Weights must sum to 1.0")
        
        # If include_both is True, add both correlation-adjusted and equal-weighted indices
        if include_both:
            # Get weights for both methods
            weight_climate_corr, weight_gpr_corr = self.calculate_correlation_adjusted_weights()
            weight_climate_eq, weight_gpr_eq = 0.5, 0.5
            
            print(f"Creating correlation-adjusted index with weights - Climate: {weight_climate_corr:.4f}, GPR: {weight_gpr_corr:.4f}")
            print(f"Creating equal-weighted index with weights - Climate: {weight_climate_eq:.4f}, GPR: {weight_gpr_eq:.4f}")
            
            # Create both indices in a new dataframe with Date as column
            self.combined_index = pd.DataFrame({
                'Date': components_df.index,
                'Climate_Risk_Component': components_df['Climate_Risk_Component'].values,
                'GPR_Component': components_df['GPR_Component'].values,
                'Correlation_Adjusted_Index': (weight_climate_corr * components_df['Climate_Risk_Component'] + 
                                            weight_gpr_corr * components_df['GPR_Component']),
                'Equal_Weighted_Index': (weight_climate_eq * components_df['Climate_Risk_Component'] + 
                                       weight_gpr_eq * components_df['GPR_Component'])
            })
            
            # Set a flag for plotting both indices
            self._include_both_indices = True
        else:
            # Create single index for backward compatibility
            self.combined_index = pd.DataFrame({
                'Date': components_df.index,
                'Climate_Risk_Component': components_df['Climate_Risk_Component'].values,
                'GPR_Component': components_df['GPR_Component'].values,
                'Combined_Uncertainty_Index': (weight_climate_primary * components_df['Climate_Risk_Component'] + 
                                              weight_gpr_primary * components_df['GPR_Component'])
            })
            self._include_both_indices = False
        
        return self
    
    def save_results(self, output_file):
        """Save the combined uncertainty index to CSV"""
        if self.combined_index is None:
            raise ValueError("No combined index created yet. Call create_composite_index() first.")
        
        print(f"Saving results to {output_file}")
        self.combined_index.to_csv(output_file, index=False)
        return self
    
    def plot_index(self, output_file=None):
        """Plot the combined uncertainty index"""
        if self.combined_index is None:
            raise ValueError("No combined index created yet. Call create_composite_index() first.")
        
        plt.figure(figsize=(14, 8))
        
        # Plot components
        plt.plot(self.combined_index['Date'], self.combined_index['Climate_Risk_Component'], 
                 label='Climate Risk Component', alpha=0.7)
        plt.plot(self.combined_index['Date'], self.combined_index['GPR_Component'], 
                 label='GPR Component', alpha=0.7)
        
        # Plot indices based on whether we're including both methods
        if hasattr(self, '_include_both_indices') and self._include_both_indices:
            # Plot both indices if requested
            plt.plot(self.combined_index['Date'], self.combined_index['Correlation_Adjusted_Index'], 
                     label='Correlation-Adjusted Index (88.48% Climate, 11.52% GPR)', linewidth=2, color='blue')
            plt.plot(self.combined_index['Date'], self.combined_index['Equal_Weighted_Index'], 
                     label='Equal-Weighted Index (50% Climate, 50% GPR)', linewidth=2, color='red')
            
            plt.title('Combined Global Uncertainty Index - Multiple Weighting Methods')
        else:
            # Plot single index for backward compatibility
            plt.plot(self.combined_index['Date'], self.combined_index['Combined_Uncertainty_Index'], 
                     label='Combined Uncertainty Index', linewidth=2, color='black')
            
            plt.title('Combined Global Uncertainty Index')
        
        plt.xlabel('Date')
        plt.ylabel('Index Value (Normalized)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300)
            print(f"Plot saved to {output_file}")
        else:
            plt.close()  # Close the plot to avoid display issues
        
        return self

def main():
    # Initialize the index creator
    index_creator = CombinedUncertaintyIndex()
    
    # File paths
    climate_file = 'climate_risk_series_daily_clean.csv'
    gpr_file = 'gpr_countries_data_filtered.csv'
    output_file = 'combined_global_uncertainty_index.csv'
    plot_file = 'combined_uncertainty_index_plot.png'
    
    # Execute the workflow with both correlation-adjusted and equal weighting methods
    (index_creator
     .load_data(climate_file, gpr_file)
     .aggregate_climate_to_monthly()
     .normalize_series(method='standardize')
     .apply_midas_weighting(theta=[2.0, 2.0])
     .create_composite_index(method='correlation_adjusted', include_both=True)
     .save_results(output_file)
     .plot_index(plot_file)
    )
    
    print("\nCombined Global Uncertainty Index created successfully!")
    print(f"\nSummary statistics for Correlation-Adjusted Index:")
    print(index_creator.combined_index['Correlation_Adjusted_Index'].describe())
    
    print(f"\nSummary statistics for Equal-Weighted Index:")
    print(index_creator.combined_index['Equal_Weighted_Index'].describe())

if __name__ == "__main__":
    main()