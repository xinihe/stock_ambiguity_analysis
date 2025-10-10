#!/usr/bin/env python3
"""
Monthly Ambiguity Analysis with Enhanced Statistical Reporting

This script:
1. Calculates monthly ambiguity metrics using window_size=20 and 40 bins
2. Merges new ambiguity metric with existing monthly data
3. Extracts and merges China GPR data
4. Performs two regressions with p-values and adjusted R²
5. Generates comprehensive statistical analysis

Author: Assistant
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
import warnings
warnings.filterwarnings('ignore')

# Import functions from the cross entropy ambiguity script
import sys
import os
sys.path.append('/Users/tlxy/Research/Ambiguity/code/Entropy')

# Import the necessary functions
from typing import Tuple, List, Union
from datetime import date, datetime
import importlib.util

class MonthlyAmbiguityAnalyzer:
    """
    Enhanced analyzer for monthly ambiguity metrics with statistical reporting
    """
    
    def __init__(self):
        self.daily_data = None
        self.monthly_data = None
        self.china_gpr_data = None
        self.results = {}
        
        # Load the cross entropy functions
        self.load_cross_entropy_functions()
    
    def load_cross_entropy_functions(self):
        """Load functions from cross entropy ambiguity.py"""
        try:
            spec = importlib.util.spec_from_file_location(
                "cross_entropy_ambiguity", 
                "/Users/tlxy/Research/Ambiguity/code/Entropy/cross entropy ambiguity.py"
            )
            cross_entropy_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cross_entropy_module)
            
            # Import the functions we need
            self.prepare_data = cross_entropy_module.prepare_data
            self.get_window_dates = cross_entropy_module.get_window_dates
            self.calculate_daily_probabilities = cross_entropy_module.calculate_daily_probabilities
            self.calculate_risk = cross_entropy_module.calculate_risk
            self.calculate_ambiguity_and_risk = cross_entropy_module.calculate_ambiguity_and_risk
            self.load_and_prepare_data = cross_entropy_module.load_and_prepare_data
            
            print("✓ Successfully loaded cross entropy functions")
            
        except Exception as e:
            print(f"Error loading cross entropy functions: {e}")
            # Fallback: define simplified versions
            self.define_fallback_functions()
    
    def define_fallback_functions(self):
        """Define simplified fallback functions if import fails"""
        print("Using fallback functions...")
        
        def prepare_data(df):
            df = df.copy()
            df['date'] = pd.to_datetime(df['Date']).dt.date
            df['close'] = df['SSE300_Close']
            df['daily_return'] = df['close'].pct_change()
            return df.dropna()
        
        def calculate_ambiguity_and_risk(df, specific_date, window_size=20, num_bins=40):
            # Simplified calculation
            df = prepare_data(df)
            target_date = pd.to_datetime(specific_date).date()
            
            # Get window data
            df_sorted = df.sort_values('date')
            target_idx = df_sorted[df_sorted['date'] == target_date].index
            if len(target_idx) == 0:
                return np.nan, np.nan
            
            target_idx = target_idx[0]
            start_idx = max(0, target_idx - window_size + 1)
            window_data = df_sorted.iloc[start_idx:target_idx + 1]
            
            if len(window_data) < window_size:
                return np.nan, np.nan
            
            # Calculate returns
            returns = window_data['daily_return'].dropna()
            if len(returns) == 0:
                return np.nan, np.nan
            
            # Simple ambiguity calculation using entropy
            hist, _ = np.histogram(returns, bins=num_bins)
            probs = hist / hist.sum()
            probs = probs[probs > 0]  # Remove zero probabilities
            ambiguity = -np.sum(probs * np.log(probs))  # Shannon entropy
            
            # Risk calculation
            risk = returns.std()
            
            return ambiguity, risk
        
        self.prepare_data = prepare_data
        self.calculate_ambiguity_and_risk = calculate_ambiguity_and_risk
    
    def load_daily_data(self, file_path):
        """Load daily combined data"""
        print(f"Loading daily data from {file_path}...")
        self.daily_data = pd.read_csv(file_path)
        print(f"✓ Loaded {len(self.daily_data)} daily observations")
        return self.daily_data
    
    def load_monthly_data(self, file_path):
        """Load existing monthly data"""
        print(f"Loading monthly data from {file_path}...")
        self.monthly_data = pd.read_csv(file_path)
        print(f"✓ Loaded {len(self.monthly_data)} monthly observations")
        return self.monthly_data
    
    def load_china_gpr_data(self, file_path):
        """Load China GPR data"""
        print(f"Loading China GPR data from {file_path}...")
        self.china_gpr_data = pd.read_csv(file_path)
        print(f"✓ Loaded {len(self.china_gpr_data)} GPR observations")
        return self.china_gpr_data
    
    def calculate_monthly_ambiguity_metrics(self, window_size=20, num_bins=40):
        """Calculate monthly ambiguity metrics with specified parameters"""
        print(f"Calculating monthly ambiguity metrics (window_size={window_size}, num_bins={num_bins})...")
        
        if self.daily_data is None:
            raise ValueError("Daily data not loaded")
        
        # Prepare daily data
        daily_df = self.daily_data.copy()
        daily_df['Date'] = pd.to_datetime(daily_df['Date'])
        daily_df = daily_df.sort_values('Date')
        
        # Get unique months from monthly data
        monthly_dates = pd.to_datetime(self.monthly_data['Date'])
        
        monthly_ambiguity = []
        monthly_risk = []
        
        for month_date in monthly_dates:
            # Find the last trading day of the month
            month_data = daily_df[
                (daily_df['Date'].dt.year == month_date.year) & 
                (daily_df['Date'].dt.month == month_date.month)
            ]
            
            if len(month_data) == 0:
                monthly_ambiguity.append(np.nan)
                monthly_risk.append(np.nan)
                continue
            
            # Use the last trading day of the month
            last_day = month_data['Date'].max()
            
            try:
                ambiguity, risk = self.calculate_ambiguity_and_risk(
                    daily_df, 
                    last_day.date(), 
                    window_size=window_size, 
                    num_bins=num_bins
                )
                monthly_ambiguity.append(ambiguity)
                monthly_risk.append(risk)
            except Exception as e:
                print(f"Error calculating for {month_date}: {e}")
                monthly_ambiguity.append(np.nan)
                monthly_risk.append(np.nan)
        
        # Create results dataframe
        results_df = pd.DataFrame({
            'Date': self.monthly_data['Date'],
            'New_Ambiguity_Metric': monthly_ambiguity,
            'New_Risk_Metric': monthly_risk
        })
        
        print(f"✓ Calculated ambiguity metrics for {len(results_df)} months")
        print(f"  Valid ambiguity values: {results_df['New_Ambiguity_Metric'].notna().sum()}")
        
        return results_df
    
    def merge_new_ambiguity_with_monthly(self, new_ambiguity_df):
        """Merge new ambiguity metrics with existing monthly data"""
        print("Merging new ambiguity metrics with monthly data...")
        
        # Merge on Date
        merged_df = self.monthly_data.merge(
            new_ambiguity_df[['Date', 'New_Ambiguity_Metric', 'New_Risk_Metric']], 
            on='Date', 
            how='left'
        )
        
        self.monthly_data = merged_df
        print(f"✓ Merged data shape: {merged_df.shape}")
        return merged_df
    
    def extract_and_merge_china_gpr(self):
        """Extract China GPR data and merge with monthly data"""
        print("Extracting and merging China GPR data...")
        
        if self.china_gpr_data is None:
            raise ValueError("China GPR data not loaded")
        
        # Prepare China GPR data
        gpr_df = self.china_gpr_data.copy()
        gpr_df['Date'] = pd.to_datetime(gpr_df['Date'])
        
        # Convert to monthly by taking the last value of each month
        gpr_monthly = gpr_df.groupby([gpr_df['Date'].dt.year, gpr_df['Date'].dt.month]).agg({
            'Date': 'max',
            'China': 'last'  # Use last value of the month
        }).reset_index(drop=True)
        
        # Rename China column to GPR_China for consistency
        gpr_monthly = gpr_monthly.rename(columns={'China': 'GPR_China'})
        print(f"  GPR monthly data shape: {gpr_monthly.shape}")
        print(f"  GPR monthly columns: {list(gpr_monthly.columns)}")
        
        # Create year-month string for matching
        gpr_monthly['YearMonth'] = gpr_monthly['Date'].dt.strftime('%Y-%m')
        
        # Create year-month for monthly data
        monthly_dates = pd.to_datetime(self.monthly_data['Date'])
        self.monthly_data['YearMonth'] = monthly_dates.dt.strftime('%Y-%m')
        
        # Check if GPR_China already exists and drop it to avoid conflicts
        if 'GPR_China' in self.monthly_data.columns:
            self.monthly_data = self.monthly_data.drop('GPR_China', axis=1)
        
        # Merge
        merged_df = self.monthly_data.merge(
            gpr_monthly[['YearMonth', 'GPR_China']], 
            on='YearMonth', 
            how='left'
        )
        
        # Drop the temporary YearMonth column
        merged_df = merged_df.drop('YearMonth', axis=1)
        
        # Clean up duplicate columns from multiple merges
        columns_to_clean = ['New_Ambiguity_Metric', 'New_Risk_Metric', 'GPR_China']
        for col in columns_to_clean:
            if f'{col}_y' in merged_df.columns:
                # Use the _y version (most recent) and drop others
                if f'{col}_x' in merged_df.columns:
                    merged_df = merged_df.drop(f'{col}_x', axis=1)
                merged_df = merged_df.rename(columns={f'{col}_y': col})
            elif f'{col}_x' in merged_df.columns and col not in merged_df.columns:
                # If only _x exists, rename it
                merged_df = merged_df.rename(columns={f'{col}_x': col})
        
        self.monthly_data = merged_df
        print(f"✓ Merged China GPR data. Shape: {merged_df.shape}")
        print(f"  Available columns: {list(merged_df.columns)}")
        
        # Check if GPR_China column exists
        if 'GPR_China' in merged_df.columns:
            print(f"  Valid China GPR values: {merged_df['GPR_China'].notna().sum()}")
        else:
            print("  Warning: GPR_China column not found after merge")
        
        return merged_df
    
    def save_updated_monthly_data(self, file_path):
        """Save updated monthly data with new metrics"""
        print(f"Saving updated monthly data to {file_path}...")
        self.monthly_data.to_csv(file_path, index=False)
        print("✓ Monthly data saved successfully")
    
    def run_regression_with_stats(self, y_col, x_cols, regression_name):
        """Run regression with comprehensive statistical reporting"""
        print(f"\n{'='*60}")
        print(f"REGRESSION ANALYSIS: {regression_name}")
        print(f"{'='*60}")
        
        # Prepare data
        data = self.monthly_data.copy()
        
        # Remove rows with missing values
        analysis_cols = [y_col] + x_cols
        data_clean = data[analysis_cols].dropna()
        
        print(f"Sample size: {len(data_clean)} observations")
        print(f"Variables: {', '.join(analysis_cols)}")
        
        if len(data_clean) < 10:
            print("⚠️  Warning: Sample size too small for reliable regression")
            return None
        
        # Prepare variables
        y = data_clean[y_col]
        X = data_clean[x_cols]
        X = sm.add_constant(X)  # Add intercept
        
        # Run regression
        model = sm.OLS(y, X).fit()
        
        # Print results
        print(f"\nDependent Variable: {y_col}")
        print(f"Independent Variables: {', '.join(x_cols)}")
        print(f"\nModel Summary:")
        print(f"R-squared: {model.rsquared:.4f}")
        print(f"Adjusted R-squared: {model.rsquared_adj:.4f}")
        print(f"F-statistic: {model.fvalue:.4f}")
        print(f"F-statistic p-value: {model.f_pvalue:.4f}")
        print(f"AIC: {model.aic:.4f}")
        print(f"BIC: {model.bic:.4f}")
        
        # Coefficient table
        print(f"\nCoefficient Estimates:")
        print("-" * 70)
        print(f"{'Variable':<20} {'Coefficient':<12} {'Std Error':<12} {'t-stat':<10} {'p-value':<10}")
        print("-" * 70)
        
        for i, var in enumerate(X.columns):
            coef = model.params[i]
            se = model.bse[i]
            t_stat = model.tvalues[i]
            p_val = model.pvalues[i]
            
            significance = ""
            if p_val < 0.001:
                significance = "***"
            elif p_val < 0.01:
                significance = "**"
            elif p_val < 0.05:
                significance = "*"
            elif p_val < 0.1:
                significance = "."
            
            print(f"{var:<20} {coef:<12.4f} {se:<12.4f} {t_stat:<10.4f} {p_val:<10.4f} {significance}")
        
        print("\nSignificance codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")
        
        # Diagnostic tests
        print(f"\nDiagnostic Tests:")
        print("-" * 40)
        
        # Durbin-Watson test for autocorrelation
        dw_stat = durbin_watson(model.resid)
        print(f"Durbin-Watson statistic: {dw_stat:.4f}")
        
        # Breusch-Pagan test for heteroscedasticity
        try:
            bp_stat, bp_pvalue, _, _ = het_breuschpagan(model.resid, X)
            print(f"Breusch-Pagan test p-value: {bp_pvalue:.4f}")
        except:
            print("Breusch-Pagan test: Could not compute")
        
        # Jarque-Bera test for normality of residuals
        jb_stat, jb_pvalue = stats.jarque_bera(model.resid)
        print(f"Jarque-Bera test p-value: {jb_pvalue:.4f}")
        
        # Store results
        self.results[regression_name] = {
            'model': model,
            'data': data_clean,
            'y_col': y_col,
            'x_cols': x_cols,
            'r_squared': model.rsquared,
            'adj_r_squared': model.rsquared_adj,
            'f_statistic': model.fvalue,
            'f_pvalue': model.f_pvalue,
            'coefficients': dict(zip(X.columns, model.params)),
            'p_values': dict(zip(X.columns, model.pvalues)),
            'std_errors': dict(zip(X.columns, model.bse))
        }
        
        return model
    
    def create_regression_plots(self, model, regression_name, y_col, x_cols):
        """Create diagnostic plots for regression"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Regression Diagnostics: {regression_name}', fontsize=16)
        
        # Residuals vs Fitted
        axes[0, 0].scatter(model.fittedvalues, model.resid, alpha=0.7)
        axes[0, 0].axhline(y=0, color='red', linestyle='--')
        axes[0, 0].set_xlabel('Fitted Values')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title('Residuals vs Fitted')
        
        # Q-Q plot
        stats.probplot(model.resid, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title('Q-Q Plot')
        
        # Histogram of residuals
        axes[1, 0].hist(model.resid, bins=20, alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('Residuals')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Distribution of Residuals')
        
        # Actual vs Predicted
        y_actual = self.results[regression_name]['data'][y_col]
        axes[1, 1].scatter(y_actual, model.fittedvalues, alpha=0.7)
        axes[1, 1].plot([y_actual.min(), y_actual.max()], 
                       [y_actual.min(), y_actual.max()], 'red', linestyle='--')
        axes[1, 1].set_xlabel('Actual Values')
        axes[1, 1].set_ylabel('Predicted Values')
        axes[1, 1].set_title('Actual vs Predicted')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = f'/Users/tlxy/Research/Ambiguity/code/Adaptive/GeoUncertainty/outputs/plots/{regression_name.lower().replace(" ", "_")}_diagnostics.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Diagnostic plots saved to {plot_path}")
    
    def generate_comprehensive_report(self):
        """Generate comprehensive analysis report"""
        print(f"\n{'='*80}")
        print("COMPREHENSIVE ANALYSIS REPORT")
        print(f"{'='*80}")
        
        # Data summary
        print(f"\nDATA SUMMARY:")
        print(f"Total monthly observations: {len(self.monthly_data)}")
        print(f"Date range: {self.monthly_data['Date'].min()} to {self.monthly_data['Date'].max()}")
        
        # New ambiguity metric summary
        new_ambiguity = self.monthly_data['New_Ambiguity_Metric'].dropna()
        if len(new_ambiguity) > 0:
            print(f"\nNEW AMBIGUITY METRIC (window_size=20, bins=40):")
            print(f"Valid observations: {len(new_ambiguity)}")
            print(f"Mean: {new_ambiguity.mean():.4f}")
            print(f"Std Dev: {new_ambiguity.std():.4f}")
            print(f"Min: {new_ambiguity.min():.4f}")
            print(f"Max: {new_ambiguity.max():.4f}")
        
        # China GPR summary
        if 'GPR_China' in self.monthly_data.columns:
            china_gpr = self.monthly_data['GPR_China'].dropna()
            if len(china_gpr) > 0:
                print(f"\nCHINA GPR DATA:")
                print(f"Valid observations: {len(china_gpr)}")
                print(f"Mean: {china_gpr.mean():.4f}")
                print(f"Std Dev: {china_gpr.std():.4f}")
                print(f"Min: {china_gpr.min():.4f}")
                print(f"Max: {china_gpr.max():.4f}")
        
        # Regression summaries
        print(f"\nREGRESSION RESULTS SUMMARY:")
        print("-" * 60)
        
        for reg_name, results in self.results.items():
            print(f"\n{reg_name}:")
            print(f"  R²: {results['r_squared']:.4f}")
            print(f"  Adjusted R²: {results['adj_r_squared']:.4f}")
            print(f"  F-statistic p-value: {results['f_pvalue']:.4f}")
            
            print(f"  Significant coefficients (p < 0.05):")
            for var, p_val in results['p_values'].items():
                if p_val < 0.05:
                    coef = results['coefficients'][var]
                    print(f"    {var}: {coef:.4f} (p = {p_val:.4f})")
        
        print(f"\n{'='*80}")
        print("ANALYSIS COMPLETE")
        print(f"{'='*80}")

def main():
    """Main execution function"""
    print("Starting Monthly Ambiguity Analysis with Enhanced Statistical Reporting")
    print("=" * 80)
    
    # Initialize analyzer
    analyzer = MonthlyAmbiguityAnalyzer()
    
    # File paths
    daily_data_path = '/Users/tlxy/Research/Ambiguity/code/Adaptive/GeoUncertainty/outputs/results/analysis/combined_data_analysis.csv'
    monthly_data_path = '/Users/tlxy/Research/Ambiguity/code/Adaptive/GeoUncertainty/outputs/results/monthly_combined_analysis.csv'
    china_gpr_path = '/Users/tlxy/Research/Ambiguity/code/Adaptive/GeoUncertainty/data/processed/gpr_countries_data_filtered.csv'
    
    try:
        # Step 1: Load data
        analyzer.load_daily_data(daily_data_path)
        analyzer.load_monthly_data(monthly_data_path)
        analyzer.load_china_gpr_data(china_gpr_path)
        
        # Step 2: Calculate new ambiguity metrics
        new_ambiguity_df = analyzer.calculate_monthly_ambiguity_metrics(window_size=20, num_bins=40)
        
        # Step 3: Merge new ambiguity metrics
        analyzer.merge_new_ambiguity_with_monthly(new_ambiguity_df)
        
        # Step 4: Extract and merge China GPR data
        analyzer.extract_and_merge_china_gpr()
        
        # Step 5: Save updated monthly data
        analyzer.save_updated_monthly_data(monthly_data_path)
        
        # Step 6: Run first regression (New Ambiguity & Risk vs Monthly Returns)
        model1 = analyzer.run_regression_with_stats(
            y_col='Monthly_Return_Pct',
            x_cols=['New_Ambiguity_Metric', 'New_Risk_Metric'],
            regression_name='New Ambiguity & Risk vs Monthly Returns'
        )
        
        if model1:
            analyzer.create_regression_plots(
                model1, 
                'New Ambiguity & Risk vs Monthly Returns',
                'Monthly_Return_Pct',
                ['New_Ambiguity_Metric', 'New_Risk_Metric']
            )
        
        # Step 7: Run second regression (Climate Risk & China GPR vs New Ambiguity)
        model2 = analyzer.run_regression_with_stats(
            y_col='New_Ambiguity_Metric',
            x_cols=['Climate_Risk_Component', 'GPR_China'],
            regression_name='Climate Risk & China GPR vs New Ambiguity'
        )
        
        if model2:
            analyzer.create_regression_plots(
                model2,
                'Climate Risk & China GPR vs New Ambiguity',
                'New_Ambiguity_Metric',
                ['Climate_Risk_Component', 'GPR_China']
            )
        
        # Step 8: Generate comprehensive report
        analyzer.generate_comprehensive_report()
        
        print("\n✓ Analysis completed successfully!")
        print(f"✓ Updated monthly data saved to: {monthly_data_path}")
        print(f"✓ Diagnostic plots saved to: /Users/tlxy/Research/Ambiguity/code/Adaptive/GeoUncertainty/outputs/plots/")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()