import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates

class CorrelationAnalyzer:
    def __init__(self, data_file):
        """Initialize the Correlation Analyzer with data file path"""
        self.data_file = data_file
        self.data = None
        self.rolling_correlations = None
    
    def load_data(self):
        """Load the combined uncertainty index data"""
        print(f"Loading data from {self.data_file}...")
        self.data = pd.read_csv(self.data_file)
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        self.data.set_index('Date', inplace=True)
        print(f"Data loaded. Shape: {self.data.shape}")
        print(f"Date range: {self.data.index.min()} to {self.data.index.max()}")
        return self
    
    def calculate_static_correlation(self):
        """Calculate the static correlation matrix"""
        print("\nCalculating static correlation matrix...")
        self.corr_matrix = self.data.corr()
        print("\nStatic Correlation Matrix:")
        print(self.corr_matrix)
        return self
    
    def calculate_rolling_correlations(self, window=12, min_periods=6):
        """Calculate rolling window correlations"""
        print(f"\nCalculating rolling correlations with window size {window}...")
        
        # Define pairs of series to correlate
        pairs = [
            ('Climate_Risk_Component', 'GPR_Component'),
            ('Climate_Risk_Component', 'Combined_Uncertainty_Index'),
            ('GPR_Component', 'Combined_Uncertainty_Index')
        ]
        
        # Calculate rolling correlations for each pair
        results = {}
        for series1, series2 in pairs:
            corr_key = f"{series1}_vs_{series2}"
            results[corr_key] = self.data[series1].rolling(window=window, min_periods=min_periods).corr(self.data[series2])
        
        # Create a DataFrame with all rolling correlations
        self.rolling_correlations = pd.DataFrame(results)
        print(f"Rolling correlations calculated. Shape: {self.rolling_correlations.shape}")
        return self
    
    def plot_rolling_correlations(self, output_file='rolling_correlations.png'):
        """Plot the rolling correlations"""
        if self.rolling_correlations is None:
            raise ValueError("Calculate rolling correlations first using calculate_rolling_correlations()")
        
        print("\nGenerating rolling correlation plot...")
        
        plt.figure(figsize=(14, 8))
        
        # Create a color palette
        colors = ['blue', 'green', 'red']
        labels = ['Climate vs GPR', 'Climate vs Combined', 'GPR vs Combined']
        
        # Plot each rolling correlation
        for i, (column, color, label) in enumerate(zip(self.rolling_correlations.columns, colors, labels)):
            plt.plot(self.rolling_correlations.index, self.rolling_correlations[column], 
                     label=label, color=color, alpha=0.8, linewidth=1.5)
        
        # Add a horizontal line at zero
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        
        # Format the x-axis to show years
        ax = plt.gca()
        years = mdates.YearLocator(5)  # every 5 years
        months = mdates.MonthLocator()  # every month
        years_fmt = mdates.DateFormatter('%Y')
        ax.xaxis.set_major_locator(years)
        ax.xaxis.set_major_formatter(years_fmt)
        ax.xaxis.set_minor_locator(months)
        
        # Rotate and align the x labels
        plt.gcf().autofmt_xdate()
        
        # Add labels and title
        plt.title('Rolling Correlations Between Uncertainty Components', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Correlation Coefficient', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.ylim(-1.1, 1.1)
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        print(f"Rolling correlation plot saved to {output_file}")
        
        return self
    
    def plot_heatmap(self, corr_matrix, output_file='correlation_heatmap.png'):
        """Plot a heatmap of the correlation matrix"""
        print("\nGenerating correlation heatmap...")
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, 
                    square=True, linewidths=0.5, cbar_kws={'label': 'Correlation Coefficient'})
        plt.title('Correlation Matrix of Uncertainty Components', fontsize=16)
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        print(f"Correlation heatmap saved to {output_file}")
        
        return self
    
    def plot_scatter_matrix(self, output_file='scatter_matrix.png'):
        """Plot a scatter matrix of the components"""
        print("\nGenerating scatter matrix...")
        
        # Create a scatter matrix
        scatter_fig = sns.pairplot(self.data)
        scatter_fig.fig.suptitle('Scatter Matrix of Uncertainty Components', y=1.02, fontsize=16)
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        print(f"Scatter matrix saved to {output_file}")
        
        return self
    
    def analyze_trends(self):
        """Analyze trends in the rolling correlations"""
        if self.rolling_correlations is None:
            raise ValueError("Calculate rolling correlations first using calculate_rolling_correlations()")
        
        print("\nAnalyzing correlation trends...")
        
        # Calculate descriptive statistics for each rolling correlation series
        stats = self.rolling_correlations.describe()
        print("\nDescriptive Statistics of Rolling Correlations:")
        print(stats)
        
        # Identify periods of high and low correlation
        print("\nPeriods of High Correlation (|r| > 0.7):")
        for column in self.rolling_correlations.columns:
            high_corr_periods = self.rolling_correlations[abs(self.rolling_correlations[column]) > 0.7]
            if not high_corr_periods.empty:
                print(f"\n{column}:")
                print(f"Number of high correlation periods: {len(high_corr_periods)}")
                print(f"First occurrence: {high_corr_periods.index[0].strftime('%Y-%m')}")
                print(f"Last occurrence: {high_corr_periods.index[-1].strftime('%Y-%m')}")
        
        # Calculate linear trends for each correlation series
        print("\nTrend Analysis of Rolling Correlations:")
        for column in self.rolling_correlations.columns:
            # Drop NaN values
            series = self.rolling_correlations[column].dropna()
            if len(series) > 1:
                # Convert dates to numeric values for regression
                x = np.arange(len(series))
                y = series.values
                
                # Calculate linear regression
                slope, intercept = np.polyfit(x, y, 1)
                
                # Determine trend direction
                if abs(slope) < 0.001:
                    trend = "stable"
                elif slope > 0:
                    trend = "increasing"
                else:
                    trend = "decreasing"
                
                print(f"\n{column}:")
                print(f"  Trend: {trend} (slope = {slope:.6f})")
                print(f"  Average correlation: {series.mean():.4f}")
                print(f"  Correlation range: {series.min():.4f} to {series.max():.4f}")
        
        return self
    
    def save_rolling_correlations(self, output_file='rolling_correlations.csv'):
        """Save the rolling correlations to a CSV file"""
        if self.rolling_correlations is None:
            raise ValueError("Calculate rolling correlations first using calculate_rolling_correlations()")
        
        print(f"\nSaving rolling correlations to {output_file}...")
        # Add a date column before saving
        result_df = self.rolling_correlations.reset_index()
        result_df.to_csv(output_file, index=False)
        print(f"Rolling correlations saved successfully.")
        
        return self

def main():
    # Initialize the analyzer with the combined uncertainty index file
    analyzer = CorrelationAnalyzer('combined_global_uncertainty_index.csv')
    
    # Execute the analysis workflow
    (analyzer
        .load_data()
        .calculate_static_correlation()
        .calculate_rolling_correlations(window=12, min_periods=6)
        .plot_rolling_correlations()
        .plot_heatmap(analyzer.corr_matrix)
        .plot_scatter_matrix()
        .analyze_trends()
        .save_rolling_correlations()
    )
    
    print("\nCorrelation analysis completed successfully!")

if __name__ == "__main__":
    main()