#!/usr/bin/env python3
"""
Monthly Data Analysis Script for GeoUncertainty Project

This script:
1. Converts daily data to monthly aggregation
2. Runs regression analyses
3. Creates visualizations and correlation analysis

Author: Generated for GeoUncertainty Research Project
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import warnings
import re
from datetime import datetime
import os

warnings.filterwarnings('ignore')

class MonthlyAnalyzer:
    def __init__(self, data_path):
        """Initialize the analyzer with data path"""
        self.data_path = data_path
        self.daily_data = None
        self.monthly_data = None
        self.results = {}
        
    def load_daily_data(self):
        """Load and clean the daily data"""
        print("Loading daily data...")
        self.daily_data = pd.read_csv(self.data_path)
        
        # Convert Date column to datetime
        self.daily_data['Date'] = pd.to_datetime(self.daily_data['Date'])
        
        # Clean GPR columns - extract numeric values from string representations
        gpr_columns = ['GPR_China', 'GPR_Japan', 'GPR_US']
        for col in gpr_columns:
            self.daily_data[col] = self.daily_data[col].apply(self._extract_gpr_values)
        
        print(f"Loaded {len(self.daily_data)} daily observations")
        print(f"Date range: {self.daily_data['Date'].min()} to {self.daily_data['Date'].max()}")
        
    def _extract_gpr_values(self, gpr_string):
        """Extract numeric values from GPR string representation"""
        if pd.isna(gpr_string) or gpr_string == '':
            return np.nan
        
        # Extract all numeric values from the string
        numbers = re.findall(r'[\d.]+', str(gpr_string))
        if numbers:
            # Take the mean of all numeric values found
            return np.mean([float(x) for x in numbers])
        return np.nan
    
    def convert_to_monthly(self):
        """Convert daily data to monthly aggregation"""
        print("Converting to monthly data...")
        
        # Create year-month column for grouping
        self.daily_data['YearMonth'] = self.daily_data['Date'].dt.to_period('M')
        
        # Define aggregation rules
        agg_rules = {
            'Daily_Return': 'sum',  # Accumulated monthly return
            'SSE300_Close': 'last',  # Last value of the month
            'Ambiguity_Metric': 'mean',  # Average
            'Risk_Metric': 'mean',  # Average
            'Climate_Risk_Component': 'mean',  # Average
            'Global_GPR_Component': 'mean',  # Average
            'Correlation_Adjusted_Index': 'mean',  # Average
            'Equal_Weighted_Index': 'mean',  # Average
            'GPR_China': 'mean',  # Average
            'GPR_Japan': 'mean',  # Average
            'GPR_US': 'mean'  # Average
        }
        
        # Group by year-month and aggregate
        monthly_grouped = self.daily_data.groupby('YearMonth').agg(agg_rules)
        
        # Reset index and create proper date column
        self.monthly_data = monthly_grouped.reset_index()
        self.monthly_data['Date'] = self.monthly_data['YearMonth'].dt.to_timestamp()
        
        # Calculate monthly return as percentage
        self.monthly_data['Monthly_Return_Pct'] = self.monthly_data['Daily_Return'] * 100
        
        # Drop rows with missing critical data
        self.monthly_data = self.monthly_data.dropna(subset=['Monthly_Return_Pct', 'Ambiguity_Metric', 'Risk_Metric'])
        
        print(f"Created {len(self.monthly_data)} monthly observations")
        
    def save_monthly_data(self, output_path):
        """Save monthly data to CSV"""
        print(f"Saving monthly data to {output_path}")
        
        # Select and rename columns for output
        output_columns = [
            'Date', 'Monthly_Return_Pct', 'SSE300_Close', 'Ambiguity_Metric', 
            'Risk_Metric', 'Climate_Risk_Component', 'Global_GPR_Component',
            'Correlation_Adjusted_Index', 'Equal_Weighted_Index',
            'GPR_China', 'GPR_Japan', 'GPR_US'
        ]
        
        output_data = self.monthly_data[output_columns].copy()
        output_data.to_csv(output_path, index=False)
        print(f"Monthly data saved successfully with {len(output_data)} observations")
        
    def run_regression_1(self):
        """First regression: Ambiguity and Risk vs Monthly SSE300 Returns"""
        print("\n" + "="*60)
        print("REGRESSION 1: Ambiguity and Risk vs Monthly SSE300 Returns")
        print("="*60)
        
        # Prepare data
        X = self.monthly_data[['Ambiguity_Metric', 'Risk_Metric']].values
        y = self.monthly_data['Monthly_Return_Pct'].values
        
        # Remove any remaining NaN values
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X_clean = X[mask]
        y_clean = y[mask]
        
        # Run regression
        reg1 = LinearRegression()
        reg1.fit(X_clean, y_clean)
        y_pred = reg1.predict(X_clean)
        
        # Calculate statistics
        r2 = r2_score(y_clean, y_pred)
        n = len(y_clean)
        
        # Store results
        self.results['regression_1'] = {
            'model': reg1,
            'r2': r2,
            'n_obs': n,
            'coefficients': reg1.coef_,
            'intercept': reg1.intercept_,
            'feature_names': ['Ambiguity_Metric', 'Risk_Metric']
        }
        
        # Print results
        print(f"Number of observations: {n}")
        print(f"R-squared: {r2:.4f}")
        print(f"Intercept: {reg1.intercept_:.4f}")
        print(f"Ambiguity coefficient: {reg1.coef_[0]:.4f}")
        print(f"Risk coefficient: {reg1.coef_[1]:.4f}")
        
        # Interpretation
        print("\nInterpretation:")
        print(f"- A 1-unit increase in Ambiguity is associated with {reg1.coef_[0]:.4f}% change in monthly returns")
        print(f"- A 1-unit increase in Risk is associated with {reg1.coef_[1]:.4f}% change in monthly returns")
        
    def run_regression_2(self):
        """Second regression: Climate Risk and GPR vs Ambiguity"""
        print("\n" + "="*60)
        print("REGRESSION 2: Climate Risk and GPR vs Ambiguity")
        print("="*60)
        
        # Prepare data
        X = self.monthly_data[['Climate_Risk_Component', 'Global_GPR_Component']].values
        y = self.monthly_data['Ambiguity_Metric'].values
        
        # Remove any remaining NaN values
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X_clean = X[mask]
        y_clean = y[mask]
        
        # Run regression
        reg2 = LinearRegression()
        reg2.fit(X_clean, y_clean)
        y_pred = reg2.predict(X_clean)
        
        # Calculate statistics
        r2 = r2_score(y_clean, y_pred)
        n = len(y_clean)
        
        # Store results
        self.results['regression_2'] = {
            'model': reg2,
            'r2': r2,
            'n_obs': n,
            'coefficients': reg2.coef_,
            'intercept': reg2.intercept_,
            'feature_names': ['Climate_Risk_Component', 'Global_GPR_Component']
        }
        
        # Print results
        print(f"Number of observations: {n}")
        print(f"R-squared: {r2:.4f}")
        print(f"Intercept: {reg2.intercept_:.4f}")
        print(f"Climate Risk coefficient: {reg2.coef_[0]:.4f}")
        print(f"Global GPR coefficient: {reg2.coef_[1]:.4f}")
        
        # Interpretation
        print("\nInterpretation:")
        print(f"- A 1-unit increase in Climate Risk is associated with {reg2.coef_[0]:.4f} change in Ambiguity")
        print(f"- A 1-unit increase in Global GPR is associated with {reg2.coef_[1]:.4f} change in Ambiguity")
        
    def create_visualizations(self, output_dir):
        """Create scatter plots, heatmaps, and correlation analysis"""
        print("\n" + "="*60)
        print("CREATING VISUALIZATIONS")
        print("="*60)
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Correlation Matrix Heatmap
        self._create_correlation_heatmap(output_dir)
        
        # 2. Scatter plots for Regression 1
        self._create_regression1_plots(output_dir)
        
        # 3. Scatter plots for Regression 2
        self._create_regression2_plots(output_dir)
        
        # 4. Time series plots
        self._create_time_series_plots(output_dir)
        
        print(f"All visualizations saved to {output_dir}")
        
    def _create_correlation_heatmap(self, output_dir):
        """Create correlation matrix heatmap"""
        # Select key variables for correlation analysis
        corr_vars = [
            'Monthly_Return_Pct', 'Ambiguity_Metric', 'Risk_Metric',
            'Climate_Risk_Component', 'Global_GPR_Component',
            'GPR_China', 'GPR_Japan', 'GPR_US'
        ]
        
        corr_data = self.monthly_data[corr_vars].corr()
        
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_data, dtype=bool))
        sns.heatmap(corr_data, mask=mask, annot=True, cmap='RdBu_r', center=0,
                    square=True, fmt='.3f', cbar_kws={"shrink": .8})
        plt.title('Correlation Matrix - Monthly Data', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/correlation_heatmap_monthly.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print correlation insights
        print("\nKey Correlations:")
        return_corr = corr_data['Monthly_Return_Pct'].drop('Monthly_Return_Pct').sort_values(key=abs, ascending=False)
        for var, corr in return_corr.head(5).items():
            print(f"Monthly Return vs {var}: {corr:.3f}")
            
    def _create_regression1_plots(self, output_dir):
        """Create scatter plots for regression 1"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Ambiguity vs Returns
        axes[0].scatter(self.monthly_data['Ambiguity_Metric'], 
                       self.monthly_data['Monthly_Return_Pct'], 
                       alpha=0.6, color='blue')
        axes[0].set_xlabel('Ambiguity Metric')
        axes[0].set_ylabel('Monthly Return (%)')
        axes[0].set_title('Ambiguity vs Monthly Returns')
        axes[0].grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(self.monthly_data['Ambiguity_Metric'].dropna(), 
                      self.monthly_data['Monthly_Return_Pct'].dropna(), 1)
        p = np.poly1d(z)
        axes[0].plot(self.monthly_data['Ambiguity_Metric'], 
                    p(self.monthly_data['Ambiguity_Metric']), "r--", alpha=0.8)
        
        # Risk vs Returns
        axes[1].scatter(self.monthly_data['Risk_Metric'], 
                       self.monthly_data['Monthly_Return_Pct'], 
                       alpha=0.6, color='green')
        axes[1].set_xlabel('Risk Metric')
        axes[1].set_ylabel('Monthly Return (%)')
        axes[1].set_title('Risk vs Monthly Returns')
        axes[1].grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(self.monthly_data['Risk_Metric'].dropna(), 
                      self.monthly_data['Monthly_Return_Pct'].dropna(), 1)
        p = np.poly1d(z)
        axes[1].plot(self.monthly_data['Risk_Metric'], 
                    p(self.monthly_data['Risk_Metric']), "r--", alpha=0.8)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/regression1_scatter_plots.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _create_regression2_plots(self, output_dir):
        """Create scatter plots for regression 2"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Climate Risk vs Ambiguity
        axes[0].scatter(self.monthly_data['Climate_Risk_Component'], 
                       self.monthly_data['Ambiguity_Metric'], 
                       alpha=0.6, color='red')
        axes[0].set_xlabel('Climate Risk Component')
        axes[0].set_ylabel('Ambiguity Metric')
        axes[0].set_title('Climate Risk vs Ambiguity')
        axes[0].grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(self.monthly_data['Climate_Risk_Component'].dropna(), 
                      self.monthly_data['Ambiguity_Metric'].dropna(), 1)
        p = np.poly1d(z)
        axes[0].plot(self.monthly_data['Climate_Risk_Component'], 
                    p(self.monthly_data['Climate_Risk_Component']), "r--", alpha=0.8)
        
        # Global GPR vs Ambiguity
        axes[1].scatter(self.monthly_data['Global_GPR_Component'], 
                       self.monthly_data['Ambiguity_Metric'], 
                       alpha=0.6, color='orange')
        axes[1].set_xlabel('Global GPR Component')
        axes[1].set_ylabel('Ambiguity Metric')
        axes[1].set_title('Global GPR vs Ambiguity')
        axes[1].grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(self.monthly_data['Global_GPR_Component'].dropna(), 
                      self.monthly_data['Ambiguity_Metric'].dropna(), 1)
        p = np.poly1d(z)
        axes[1].plot(self.monthly_data['Global_GPR_Component'], 
                    p(self.monthly_data['Global_GPR_Component']), "r--", alpha=0.8)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/regression2_scatter_plots.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _create_time_series_plots(self, output_dir):
        """Create time series plots"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Monthly Returns
        axes[0,0].plot(self.monthly_data['Date'], self.monthly_data['Monthly_Return_Pct'], 
                      color='blue', linewidth=1.5)
        axes[0,0].set_title('Monthly Returns Over Time')
        axes[0,0].set_ylabel('Monthly Return (%)')
        axes[0,0].grid(True, alpha=0.3)
        axes[0,0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # Ambiguity Metric
        axes[0,1].plot(self.monthly_data['Date'], self.monthly_data['Ambiguity_Metric'], 
                      color='green', linewidth=1.5)
        axes[0,1].set_title('Ambiguity Metric Over Time')
        axes[0,1].set_ylabel('Ambiguity Metric')
        axes[0,1].grid(True, alpha=0.3)
        
        # Climate Risk
        axes[1,0].plot(self.monthly_data['Date'], self.monthly_data['Climate_Risk_Component'], 
                      color='red', linewidth=1.5)
        axes[1,0].set_title('Climate Risk Component Over Time')
        axes[1,0].set_ylabel('Climate Risk Component')
        axes[1,0].grid(True, alpha=0.3)
        
        # Global GPR
        axes[1,1].plot(self.monthly_data['Date'], self.monthly_data['Global_GPR_Component'], 
                      color='orange', linewidth=1.5)
        axes[1,1].set_title('Global GPR Component Over Time')
        axes[1,1].set_ylabel('Global GPR Component')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/time_series_plots.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def generate_summary_report(self):
        """Generate a comprehensive summary report"""
        print("\n" + "="*80)
        print("COMPREHENSIVE ANALYSIS SUMMARY")
        print("="*80)
        
        print(f"\nDataset Overview:")
        print(f"- Total monthly observations: {len(self.monthly_data)}")
        print(f"- Date range: {self.monthly_data['Date'].min().strftime('%Y-%m')} to {self.monthly_data['Date'].max().strftime('%Y-%m')}")
        
        print(f"\nDescriptive Statistics:")
        key_vars = ['Monthly_Return_Pct', 'Ambiguity_Metric', 'Risk_Metric', 
                   'Climate_Risk_Component', 'Global_GPR_Component']
        desc_stats = self.monthly_data[key_vars].describe()
        print(desc_stats.round(4))
        
        print(f"\nRegression Results Summary:")
        print(f"1. Ambiguity & Risk → Monthly Returns:")
        print(f"   - R²: {self.results['regression_1']['r2']:.4f}")
        print(f"   - Ambiguity impact: {self.results['regression_1']['coefficients'][0]:.4f}")
        print(f"   - Risk impact: {self.results['regression_1']['coefficients'][1]:.4f}")
        
        print(f"\n2. Climate Risk & GPR → Ambiguity:")
        print(f"   - R²: {self.results['regression_2']['r2']:.4f}")
        print(f"   - Climate Risk impact: {self.results['regression_2']['coefficients'][0]:.4f}")
        print(f"   - GPR impact: {self.results['regression_2']['coefficients'][1]:.4f}")


def main():
    """Main execution function"""
    # Define paths
    data_path = "/Users/tlxy/Research/Ambiguity/code/Adaptive/GeoUncertainty/outputs/results/analysis/combined_data_analysis.csv"
    output_csv = "/Users/tlxy/Research/Ambiguity/code/Adaptive/GeoUncertainty/outputs/results/monthly_combined_analysis.csv"
    output_plots = "/Users/tlxy/Research/Ambiguity/code/Adaptive/GeoUncertainty/outputs/plots"
    
    # Initialize analyzer
    analyzer = MonthlyAnalyzer(data_path)
    
    # Run analysis pipeline
    analyzer.load_daily_data()
    analyzer.convert_to_monthly()
    analyzer.save_monthly_data(output_csv)
    
    # Run regressions
    analyzer.run_regression_1()
    analyzer.run_regression_2()
    
    # Create visualizations
    analyzer.create_visualizations(output_plots)
    
    # Generate summary report
    analyzer.generate_summary_report()
    
    print(f"\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print(f"Monthly data saved to: {output_csv}")
    print(f"Visualizations saved to: {output_plots}")
    print("="*80)


if __name__ == "__main__":
    main()