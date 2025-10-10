#!/usr/bin/env python3
"""
Comprehensive Bin Size Analysis for Ambiguity Metrics
=====================================================

This script systematically tests different bin sizes (5-50) for ambiguity calculation
and performs regression analysis with various configurations:
- Different bin sizes for ambiguity metrics
- Log transformation of monthly returns
- Multiple GPR countries (China, US, Japan)

Author: AI Assistant
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

# Add path for cross entropy functions
import sys
import os
sys.path.append('/Users/tlxy/Research/Ambiguity/code/Entropy')

from typing import Tuple, List, Union, Dict
from datetime import date, datetime
import importlib.util

class ComprehensiveBinAnalyzer:
    """
    Comprehensive analyzer for testing different bin sizes and regression configurations
    """
    
    def __init__(self):
        self.daily_data = None
        self.monthly_data = None
        self.gpr_data = None
        self.results = {}
        self.bin_sizes = list(range(5, 55, 5))  # [5, 10, 15, ..., 50]
        self.window_size = 20  # Keep window size constant at 20 days
        
        # Load cross entropy functions
        self.load_cross_entropy_functions()
        
    def load_cross_entropy_functions(self):
        """Load cross entropy calculation functions"""
        try:
            # Try to load from the Entropy directory
            spec = importlib.util.spec_from_file_location(
                "cross_entropy_ambiguity", 
                "/Users/tlxy/Research/Ambiguity/code/Entropy/cross entropy ambiguity.py"
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Extract the functions we need
            self.calculate_daily_probabilities = module.calculate_daily_probabilities
            self.calculate_ambiguity_and_risk = module.calculate_ambiguity_and_risk
            self.get_window_dates = module.get_window_dates
            
            print("✓ Cross entropy functions loaded successfully")
            
        except Exception as e:
            print(f"⚠️  Warning: Could not load cross entropy functions: {e}")
            print("Using fallback functions...")
            self.define_fallback_functions()
    
    def define_fallback_functions(self):
        """Define fallback functions if cross entropy module can't be loaded"""
        def calculate_daily_probabilities(returns, num_bins=20):
            """Calculate probability distribution of returns"""
            if len(returns) == 0:
                return np.array([]), np.array([])
            
            # Create histogram
            counts, bin_edges = np.histogram(returns, bins=num_bins)
            
            # Convert to probabilities
            probabilities = counts / len(returns)
            
            # Get bin centers
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            return probabilities, bin_centers
        
        def calculate_ambiguity_and_risk(returns, window_size=5, num_bins=20):
            """Calculate ambiguity and risk metrics"""
            if len(returns) < window_size:
                return np.nan, np.nan
            
            # Use the last window_size returns
            window_returns = returns[-window_size:]
            
            # Calculate probabilities
            probs, _ = calculate_daily_probabilities(window_returns, num_bins)
            
            # Remove zero probabilities for entropy calculation
            probs = probs[probs > 0]
            
            if len(probs) == 0:
                return np.nan, np.nan
            
            # Calculate entropy (ambiguity proxy)
            entropy = -np.sum(probs * np.log(probs))
            
            # Calculate risk (variance)
            risk = np.var(window_returns)
            
            return entropy, risk
        
        def get_window_dates(specific_date, window_size, date_list):
            """Get dates for the window"""
            try:
                date_idx = date_list.index(specific_date)
                start_idx = max(0, date_idx - window_size + 1)
                return date_list[start_idx:date_idx + 1]
            except ValueError:
                return []
        
        self.calculate_daily_probabilities = calculate_daily_probabilities
        self.calculate_ambiguity_and_risk = calculate_ambiguity_and_risk
        self.get_window_dates = get_window_dates
    
    def load_daily_data(self, file_path):
        """Load daily combined data"""
        self.daily_data = pd.read_csv(file_path)
        self.daily_data['Date'] = pd.to_datetime(self.daily_data['Date'])
        print(f"✓ Loaded daily data: {self.daily_data.shape}")
        return self.daily_data
    
    def load_monthly_data(self, file_path):
        """Load monthly combined data"""
        self.monthly_data = pd.read_csv(file_path)
        self.monthly_data['Date'] = pd.to_datetime(self.monthly_data['Date'])
        print(f"✓ Loaded monthly data: {self.monthly_data.shape}")
        return self.monthly_data
    
    def load_gpr_data(self, file_path):
        """Load GPR data for multiple countries"""
        self.gpr_data = pd.read_csv(file_path)
        self.gpr_data['Date'] = pd.to_datetime(self.gpr_data['Date'])
        print(f"✓ Loaded GPR data: {self.gpr_data.shape}")
        print(f"  Available countries: {[col for col in self.gpr_data.columns if col != 'Date']}")
        return self.gpr_data
    
    def calculate_ambiguity_for_all_bins(self):
        """Calculate ambiguity metrics for all bin sizes"""
        print(f"\n🔄 Calculating ambiguity metrics for bin sizes: {self.bin_sizes}")
        
        # Get date range from monthly data
        start_date = self.monthly_data['Date'].min()
        end_date = self.monthly_data['Date'].max()
        
        # Create monthly date range
        monthly_dates = pd.date_range(start=start_date, end=end_date, freq='MS')
        
        ambiguity_results = {}
        
        for num_bins in self.bin_sizes:
            print(f"  Processing bin size: {num_bins}")
            
            monthly_ambiguity = []
            monthly_risk = []
            valid_dates = []
            
            for target_date in monthly_dates:
                # Get daily data up to this month
                daily_subset = self.daily_data[self.daily_data['Date'] <= target_date].copy()
                
                if len(daily_subset) < self.window_size:
                    monthly_ambiguity.append(np.nan)
                    monthly_risk.append(np.nan)
                    continue
                
                # Calculate returns
                daily_subset = daily_subset.sort_values('Date')
                daily_subset['Returns'] = daily_subset['SSE300_Close'].pct_change()
                returns = daily_subset['Returns'].dropna().values
                
                if len(returns) < self.window_size:
                    monthly_ambiguity.append(np.nan)
                    monthly_risk.append(np.nan)
                    continue
                
                # Calculate ambiguity and risk for this month
                ambiguity, risk = self.calculate_ambiguity_and_risk(
                    returns, self.window_size, num_bins
                )
                
                monthly_ambiguity.append(ambiguity)
                monthly_risk.append(risk)
                valid_dates.append(target_date)
            
            # Create DataFrame for this bin size
            ambiguity_df = pd.DataFrame({
                'Date': monthly_dates,
                f'Ambiguity_Bins_{num_bins}': monthly_ambiguity,
                f'Risk_Bins_{num_bins}': monthly_risk
            })
            
            ambiguity_results[num_bins] = ambiguity_df
            
            valid_count = ambiguity_df[f'Ambiguity_Bins_{num_bins}'].notna().sum()
            print(f"    Valid observations: {valid_count}/{len(monthly_dates)}")
        
        self.ambiguity_results = ambiguity_results
        return ambiguity_results
    
    def merge_all_ambiguity_metrics(self):
        """Merge all ambiguity metrics with monthly data"""
        print("\n🔄 Merging all ambiguity metrics with monthly data...")
        
        # Start with monthly data
        merged_data = self.monthly_data.copy()
        
        # Add log transformation of monthly returns
        merged_data['Log_Monthly_Return'] = np.log(1 + merged_data['Monthly_Return_Pct'] / 100)
        
        # Merge each bin size result
        for num_bins in self.bin_sizes:
            ambiguity_df = self.ambiguity_results[num_bins]
            merged_data = merged_data.merge(
                ambiguity_df, on='Date', how='left'
            )
        
        # Merge GPR data for multiple countries
        if self.gpr_data is not None:
            # Aggregate GPR data to monthly
            gpr_monthly = self.gpr_data.copy()
            gpr_monthly['YearMonth'] = gpr_monthly['Date'].dt.to_period('M')
            gpr_monthly = gpr_monthly.groupby('YearMonth').agg({
                'China': 'mean',
                'US': 'mean', 
                'Japan': 'mean'
            }).reset_index()
            gpr_monthly['Date'] = gpr_monthly['YearMonth'].dt.start_time
            gpr_monthly = gpr_monthly.drop('YearMonth', axis=1)
            
            # Rename columns
            gpr_monthly = gpr_monthly.rename(columns={
                'China': 'GPR_China',
                'US': 'GPR_US',
                'Japan': 'GPR_Japan'
            })
            
            # Merge with main data
            merged_data = merged_data.merge(gpr_monthly, on='Date', how='left')
        
        self.merged_data = merged_data
        print(f"✓ Merged data shape: {merged_data.shape}")
        print(f"  Total columns: {len(merged_data.columns)}")
        
        return merged_data
    
    def run_comprehensive_regressions(self):
        """Run regressions for all bin sizes and configurations"""
        print("\n🔄 Running comprehensive regression analysis...")
        
        regression_results = {}
        
        # Define dependent variables to test
        dependent_vars = [
            'Monthly_Return_Pct',
            'Log_Monthly_Return'
        ]
        
        # Check available GPR columns
        available_gpr_cols = [col for col in self.merged_data.columns if 'GPR_' in col]
        print(f"  Available GPR columns: {available_gpr_cols}")
        
        # Use the correct GPR column names based on what's actually available
        gpr_mapping = {}
        for col in available_gpr_cols:
            if 'China' in col:
                gpr_mapping['GPR_China'] = col
            elif 'US' in col:
                gpr_mapping['GPR_US'] = col
            elif 'Japan' in col:
                gpr_mapping['GPR_Japan'] = col
        
        print(f"  GPR column mapping: {gpr_mapping}")
        
        # Define independent variable configurations using actual column names
        gpr_configs = []
        if 'GPR_China' in gpr_mapping:
            gpr_configs.append([gpr_mapping['GPR_China']])
        if 'GPR_US' in gpr_mapping:
            gpr_configs.append([gpr_mapping['GPR_US']])
        if 'GPR_Japan' in gpr_mapping:
            gpr_configs.append([gpr_mapping['GPR_Japan']])
        
        # Two-country combinations
        if 'GPR_China' in gpr_mapping and 'GPR_US' in gpr_mapping:
            gpr_configs.append([gpr_mapping['GPR_China'], gpr_mapping['GPR_US']])
        if 'GPR_China' in gpr_mapping and 'GPR_Japan' in gpr_mapping:
            gpr_configs.append([gpr_mapping['GPR_China'], gpr_mapping['GPR_Japan']])
        if 'GPR_US' in gpr_mapping and 'GPR_Japan' in gpr_mapping:
            gpr_configs.append([gpr_mapping['GPR_US'], gpr_mapping['GPR_Japan']])
        
        # Three-country combination
        if all(key in gpr_mapping for key in ['GPR_China', 'GPR_US', 'GPR_Japan']):
            gpr_configs.append([gpr_mapping['GPR_China'], gpr_mapping['GPR_US'], gpr_mapping['GPR_Japan']])
        
        print(f"  GPR configurations to test: {len(gpr_configs)}")
        
        for dep_var in dependent_vars:
            print(f"  Processing dependent variable: {dep_var}")
            regression_results[dep_var] = {}
            
            for num_bins in self.bin_sizes:
                print(f"    Processing bin size: {num_bins}")
                regression_results[dep_var][num_bins] = {}
                
                ambiguity_col = f'Ambiguity_Bins_{num_bins}'
                risk_col = f'Risk_Bins_{num_bins}'
                
                # Check if columns exist and have valid data
                if ambiguity_col not in self.merged_data.columns:
                    print(f"      Skipping - {ambiguity_col} not found")
                    continue
                
                # Test 1: Ambiguity + Risk vs Returns
                try:
                    result1 = self.run_single_regression(
                        y_col=dep_var,
                        x_cols=[ambiguity_col, risk_col],
                        regression_name=f'Ambiguity_Risk_vs_{dep_var}_Bins_{num_bins}'
                    )
                    if result1 is not None:
                        regression_results[dep_var][num_bins]['ambiguity_risk'] = result1
                        print(f"      ✓ Ambiguity+Risk regression completed (R²={result1['r_squared']:.4f})")
                except Exception as e:
                    print(f"      ❌ Error in ambiguity+risk regression: {e}")
                
                # Test 2: GPR configurations vs Ambiguity
                for i, gpr_vars in enumerate(gpr_configs):
                    try:
                        # Create a readable name for the GPR configuration
                        gpr_names = []
                        for var in gpr_vars:
                            for key, val in gpr_mapping.items():
                                if val == var:
                                    gpr_names.append(key.replace('GPR_', ''))
                                    break
                        
                        gpr_name = '_'.join(gpr_names)
                        
                        result2 = self.run_single_regression(
                            y_col=ambiguity_col,
                            x_cols=gpr_vars,
                            regression_name=f'{gpr_name}_vs_Ambiguity_Bins_{num_bins}'
                        )
                        if result2 is not None:
                            regression_results[dep_var][num_bins][f'gpr_{gpr_name}'] = result2
                            print(f"      ✓ GPR {gpr_name} regression completed (R²={result2['r_squared']:.4f})")
                    except Exception as e:
                        print(f"      ❌ Error in GPR regression {gpr_vars}: {e}")
        
        self.regression_results = regression_results
        print(f"✓ Completed {sum(len(bins.keys()) for bins in regression_results.values())} regressions")
        return regression_results
    
    def run_single_regression(self, y_col, x_cols, regression_name):
        """Run a single regression and return results"""
        # Prepare data
        analysis_cols = [y_col] + x_cols
        data_clean = self.merged_data[analysis_cols].dropna()
        
        if len(data_clean) < 10:  # Need minimum observations
            return None
        
        # Prepare variables
        y = data_clean[y_col]
        X = data_clean[x_cols]
        X = sm.add_constant(X)
        
        # Fit model
        model = sm.OLS(y, X).fit()
        
        # Calculate additional statistics
        n = len(data_clean)
        k = len(x_cols)
        
        # Diagnostic tests
        dw_stat = durbin_watson(model.resid)
        bp_test = het_breuschpagan(model.resid, X)
        jb_test = stats.jarque_bera(model.resid)
        
        return {
            'model': model,
            'data': data_clean,
            'n_obs': n,
            'r_squared': model.rsquared,
            'adj_r_squared': model.rsquared_adj,
            'f_pvalue': model.f_pvalue,
            'coefficients': model.params,
            'pvalues': model.pvalues,
            'std_errors': model.bse,
            'durbin_watson': dw_stat,
            'breusch_pagan_pvalue': bp_test[1],
            'jarque_bera_pvalue': jb_test[1],
            'regression_name': regression_name
        }
    
    def find_best_results(self):
        """Find the best regression results across all configurations"""
        print("\n🔍 Finding best regression results...")
        
        best_results = {
            'highest_r_squared': {'value': 0, 'config': None, 'result': None},
            'highest_adj_r_squared': {'value': 0, 'config': None, 'result': None},
            'most_significant': {'count': 0, 'config': None, 'result': None},
            'best_f_test': {'value': 1, 'config': None, 'result': None}
        }
        
        for dep_var in self.regression_results:
            for num_bins in self.regression_results[dep_var]:
                for reg_type in self.regression_results[dep_var][num_bins]:
                    result = self.regression_results[dep_var][num_bins][reg_type]
                    
                    if result is None:
                        continue
                    
                    config = f"{dep_var}_bins_{num_bins}_{reg_type}"
                    
                    # Check R-squared
                    if result['r_squared'] > best_results['highest_r_squared']['value']:
                        best_results['highest_r_squared'] = {
                            'value': result['r_squared'],
                            'config': config,
                            'result': result
                        }
                    
                    # Check Adjusted R-squared
                    if result['adj_r_squared'] > best_results['highest_adj_r_squared']['value']:
                        best_results['highest_adj_r_squared'] = {
                            'value': result['adj_r_squared'],
                            'config': config,
                            'result': result
                        }
                    
                    # Check significant coefficients
                    sig_count = (result['pvalues'] < 0.05).sum()
                    if sig_count > best_results['most_significant']['count']:
                        best_results['most_significant'] = {
                            'count': sig_count,
                            'config': config,
                            'result': result
                        }
                    
                    # Check F-test
                    if result['f_pvalue'] < best_results['best_f_test']['value']:
                        best_results['best_f_test'] = {
                            'value': result['f_pvalue'],
                            'config': config,
                            'result': result
                        }
        
        self.best_results = best_results
        return best_results
    
    def generate_markdown_report(self, output_file):
        """Generate comprehensive markdown report"""
        print(f"\n📝 Generating markdown report: {output_file}")
        
        with open(output_file, 'w') as f:
            f.write("# Comprehensive Bin Size Analysis Report\n\n")
            f.write("## Executive Summary\n\n")
            f.write(f"This report presents a systematic analysis of ambiguity metrics using different bin sizes ")
            f.write(f"(from {min(self.bin_sizes)} to {max(self.bin_sizes)} bins) and various regression configurations.\n\n")
            
            # Data summary
            f.write("## Data Summary\n\n")
            f.write(f"- **Analysis Period**: {self.merged_data['Date'].min().strftime('%Y-%m-%d')} to {self.merged_data['Date'].max().strftime('%Y-%m-%d')}\n")
            f.write(f"- **Total Monthly Observations**: {len(self.merged_data)}\n")
            f.write(f"- **Window Size**: {self.window_size} days\n")
            f.write(f"- **Bin Sizes Tested**: {self.bin_sizes}\n")
            f.write(f"- **GPR Countries**: China, US, Japan\n\n")
            
            # Best results summary
            f.write("## Best Results Summary\n\n")
            
            for metric, data in self.best_results.items():
                if data['result'] is not None:
                    f.write(f"### {metric.replace('_', ' ').title()}\n")
                    f.write(f"- **Configuration**: {data['config']}\n")
                    if 'value' in data:
                        f.write(f"- **Value**: {data['value']:.4f}\n")
                    elif 'count' in data:
                        f.write(f"- **Count**: {data['count']}\n")
                    
                    result = data['result']
                    f.write(f"- **R²**: {result['r_squared']:.4f}\n")
                    f.write(f"- **Adjusted R²**: {result['adj_r_squared']:.4f}\n")
                    f.write(f"- **F-test p-value**: {result['f_pvalue']:.4f}\n")
                    f.write(f"- **Observations**: {result['n_obs']}\n")
                    
                    # Significant coefficients
                    sig_coeffs = result['pvalues'][result['pvalues'] < 0.05]
                    if len(sig_coeffs) > 0:
                        f.write(f"- **Significant Coefficients**:\n")
                        for var, pval in sig_coeffs.items():
                            coeff = result['coefficients'][var]
                            f.write(f"  - {var}: {coeff:.4f} (p = {pval:.4f})\n")
                    f.write("\n")
            
            # Detailed results by bin size
            f.write("## Detailed Results by Bin Size\n\n")
            
            for dep_var in self.regression_results:
                f.write(f"### Dependent Variable: {dep_var}\n\n")
                
                # Create summary table
                f.write("| Bin Size | Best R² | Best Adj R² | Best F-test | Significant Vars |\n")
                f.write("|----------|---------|-------------|-------------|------------------|\n")
                
                for num_bins in sorted(self.regression_results[dep_var].keys()):
                    bin_results = self.regression_results[dep_var][num_bins]
                    
                    best_r2 = 0
                    best_adj_r2 = 0
                    best_f = 1
                    total_sig = 0
                    
                    for reg_type, result in bin_results.items():
                        if result is not None:
                            best_r2 = max(best_r2, result['r_squared'])
                            best_adj_r2 = max(best_adj_r2, result['adj_r_squared'])
                            best_f = min(best_f, result['f_pvalue'])
                            total_sig += (result['pvalues'] < 0.05).sum()
                    
                    f.write(f"| {num_bins} | {best_r2:.4f} | {best_adj_r2:.4f} | {best_f:.4f} | {total_sig} |\n")
                
                f.write("\n")
            
            # Detailed regression results
            f.write("## Detailed Regression Results\n\n")
            
            for dep_var in self.regression_results:
                f.write(f"### {dep_var} Results\n\n")
                
                for num_bins in sorted(self.regression_results[dep_var].keys()):
                    f.write(f"#### Bin Size: {num_bins}\n\n")
                    
                    bin_results = self.regression_results[dep_var][num_bins]
                    
                    for reg_type, result in bin_results.items():
                        if result is not None:
                            f.write(f"**{reg_type.replace('_', ' ').title()}**\n\n")
                            f.write(f"- R²: {result['r_squared']:.4f}\n")
                            f.write(f"- Adjusted R²: {result['adj_r_squared']:.4f}\n")
                            f.write(f"- F-test p-value: {result['f_pvalue']:.4f}\n")
                            f.write(f"- Observations: {result['n_obs']}\n")
                            f.write(f"- Durbin-Watson: {result['durbin_watson']:.4f}\n")
                            
                            f.write("\nCoefficients:\n")
                            for var in result['coefficients'].index:
                                coeff = result['coefficients'][var]
                                pval = result['pvalues'][var]
                                stderr = result['std_errors'][var]
                                sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
                                f.write(f"- {var}: {coeff:.4f} (SE: {stderr:.4f}, p: {pval:.4f}) {sig}\n")
                            
                            f.write("\n")
            
            # Conclusions
            f.write("## Key Findings and Conclusions\n\n")
            
            # Find optimal bin size
            bin_performance = {}
            for num_bins in self.bin_sizes:
                total_r2 = 0
                count = 0
                for dep_var in self.regression_results:
                    if num_bins in self.regression_results[dep_var]:
                        for reg_type, result in self.regression_results[dep_var][num_bins].items():
                            if result is not None:
                                total_r2 += result['adj_r_squared']
                                count += 1
                if count > 0:
                    bin_performance[num_bins] = total_r2 / count
            
            if bin_performance:
                best_bin = max(bin_performance.keys(), key=lambda x: bin_performance[x])
                f.write(f"1. **Optimal Bin Size**: {best_bin} bins (average adjusted R² = {bin_performance[best_bin]:.4f})\n")
            
            # Log transformation effectiveness
            f.write("2. **Log Transformation**: ")
            if 'Log_Monthly_Return' in self.regression_results and 'Monthly_Return_Pct' in self.regression_results:
                log_avg = np.mean([np.mean([r['adj_r_squared'] for r in bin_results.values() if r is not None]) 
                                  for bin_results in self.regression_results['Log_Monthly_Return'].values()])
                normal_avg = np.mean([np.mean([r['adj_r_squared'] for r in bin_results.values() if r is not None]) 
                                     for bin_results in self.regression_results['Monthly_Return_Pct'].values()])
                
                if log_avg > normal_avg:
                    f.write(f"Log transformation improves model performance (avg adj R²: {log_avg:.4f} vs {normal_avg:.4f})\n")
                else:
                    f.write(f"Normal returns perform better (avg adj R²: {normal_avg:.4f} vs {log_avg:.4f})\n")
            
            # GPR effectiveness
            f.write("3. **GPR Variables**: Multiple country GPR data shows varying effectiveness across different bin sizes\n")
            
            f.write("\n---\n")
            f.write(f"*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
        
        print(f"✓ Report saved to: {output_file}")

def main():
    """Main execution function"""
    try:
        # Initialize analyzer
        analyzer = ComprehensiveBinAnalyzer()
        
        # File paths
        daily_data_path = "outputs/results/analysis/combined_data_analysis.csv"
        monthly_data_path = "outputs/results/monthly_combined_analysis.csv"
        gpr_data_path = "data/processed/gpr_countries_data_filtered.csv"
        
        # Step 1: Load data
        print("=" * 60)
        print("COMPREHENSIVE BIN SIZE ANALYSIS")
        print("=" * 60)
        
        analyzer.load_daily_data(daily_data_path)
        analyzer.load_monthly_data(monthly_data_path)
        analyzer.load_gpr_data(gpr_data_path)
        
        # Step 2: Calculate ambiguity for all bin sizes
        analyzer.calculate_ambiguity_for_all_bins()
        
        # Step 3: Merge all data
        analyzer.merge_all_ambiguity_metrics()
        
        # Step 4: Run comprehensive regressions
        analyzer.run_comprehensive_regressions()
        
        # Step 5: Find best results
        analyzer.find_best_results()
        
        # Step 6: Generate report
        report_path = "outputs/results/comprehensive_bin_analysis_report.md"
        analyzer.generate_markdown_report(report_path)
        
        # Step 7: Save enhanced data
        enhanced_data_path = "outputs/results/enhanced_monthly_data_all_bins.csv"
        analyzer.merged_data.to_csv(enhanced_data_path, index=False)
        print(f"✓ Enhanced data saved to: {enhanced_data_path}")
        
        print("\n✅ Comprehensive analysis completed successfully!")
        print(f"📊 Report available at: {report_path}")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()