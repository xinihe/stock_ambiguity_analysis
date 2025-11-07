#!/usr/bin/env python3
"""
Simplified GPR-Ambiguity Relationship Analysis
Focus on key regression results without plotting
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_white
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load and prepare the data"""
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    
    # Load the monthly combined data
    data_path = '/Users/tlxy/Research/Ambiguity/code/Adaptive/GeoUncertainty/outputs/results/monthly_combined_analysis.csv'
    data = pd.read_csv(data_path)
    
    # Convert Date column
    data['Date'] = pd.to_datetime(data['Date'])
    
    print(f"Dataset shape: {data.shape}")
    print(f"Date range: {data['Date'].min()} to {data['Date'].max()}")
    
    # Check available columns
    gpr_cols = [col for col in data.columns if 'GPR' in col]
    ambiguity_cols = [col for col in data.columns if 'Ambiguity' in col]
    
    print(f"GPR columns: {gpr_cols}")
    print(f"Ambiguity columns: {ambiguity_cols}")
    
    return data

def basic_correlations(data):
    """Calculate basic correlations"""
    print("\n" + "=" * 60)
    print("CORRELATION ANALYSIS")
    print("=" * 60)
    
    # Focus on main variables
    vars_of_interest = ['Ambiguity_Metric', 'GPR_US', 'GPR_China', 'GPR_Japan']
    
    if 'New_Ambiguity_Metric' in data.columns:
        vars_of_interest.append('New_Ambiguity_Metric')
    
    corr_data = data[vars_of_interest].dropna()
    corr_matrix = corr_data.corr()
    
    print("Correlation Matrix:")
    print(corr_matrix.round(4))
    
    return corr_matrix

def benchmark_regression(data):
    """Run benchmark regression: Ambiguity = f(GPR_US, GPR_China, GPR_Japan)"""
    print("\n" + "=" * 60)
    print("BENCHMARK REGRESSION ANALYSIS")
    print("=" * 60)
    
    # Prepare regression data
    reg_vars = ['Ambiguity_Metric', 'GPR_US', 'GPR_China', 'GPR_Japan']
    reg_data = data[reg_vars].dropna()
    
    print(f"Regression sample size: {len(reg_data)}")
    
    # Model 1: Linear specification
    print("\nModel 1: Linear Specification")
    print("-" * 40)
    
    y = reg_data['Ambiguity_Metric']
    X = reg_data[['GPR_US', 'GPR_China', 'GPR_Japan']]
    X = sm.add_constant(X)
    
    model1 = sm.OLS(y, X).fit()
    print(model1.summary())
    
    # Model 2: With interaction terms
    print("\nModel 2: With Interaction Terms")
    print("-" * 40)
    
    reg_data_int = reg_data.copy()
    reg_data_int['US_China_Interaction'] = reg_data_int['GPR_US'] * reg_data_int['GPR_China']
    reg_data_int['US_Japan_Interaction'] = reg_data_int['GPR_US'] * reg_data_int['GPR_Japan']
    reg_data_int['China_Japan_Interaction'] = reg_data_int['GPR_China'] * reg_data_int['GPR_Japan']
    
    X2 = reg_data_int[['GPR_US', 'GPR_China', 'GPR_Japan',
                       'US_China_Interaction', 'US_Japan_Interaction', 'China_Japan_Interaction']]
    X2 = sm.add_constant(X2)
    
    model2 = sm.OLS(y, X2).fit()
    print(model2.summary())
    
    # Model 3: Non-linear (quadratic) specification
    print("\nModel 3: Quadratic Specification")
    print("-" * 40)
    
    reg_data_quad = reg_data.copy()
    reg_data_quad['GPR_US_Squared'] = reg_data_quad['GPR_US'] ** 2
    reg_data_quad['GPR_China_Squared'] = reg_data_quad['GPR_China'] ** 2
    reg_data_quad['GPR_Japan_Squared'] = reg_data_quad['GPR_Japan'] ** 2
    
    X3 = reg_data_quad[['GPR_US', 'GPR_China', 'GPR_Japan',
                        'GPR_US_Squared', 'GPR_China_Squared', 'GPR_Japan_Squared']]
    X3 = sm.add_constant(X3)
    
    model3 = sm.OLS(y, X3).fit()
    print(model3.summary())
    
    return model1, model2, model3

def stationarity_tests(data):
    """Test for stationarity"""
    print("\n" + "=" * 60)
    print("STATIONARITY TESTS (ADF)")
    print("=" * 60)
    
    test_vars = ['Ambiguity_Metric', 'GPR_US', 'GPR_China', 'GPR_Japan']
    
    for var in test_vars:
        if var in data.columns:
            series = data[var].dropna()
            adf_result = adfuller(series)
            print(f"\n{var}:")
            print(f"  ADF Statistic: {adf_result[0]:.4f}")
            print(f"  p-value: {adf_result[1]:.4f}")
            print(f"  Stationary: {'Yes' if adf_result[1] < 0.05 else 'No'}")

def granger_causality_analysis(data):
    """Test Granger causality"""
    print("\n" + "=" * 60)
    print("GRANGER CAUSALITY TESTS")
    print("=" * 60)
    
    gpr_vars = ['GPR_US', 'GPR_China', 'GPR_Japan']
    
    for gpr_var in gpr_vars:
        if gpr_var in data.columns:
            print(f"\nTesting: {gpr_var} → Ambiguity_Metric")
            print("-" * 40)
            
            # Prepare data for Granger test
            test_data = data[['Ambiguity_Metric', gpr_var]].dropna()
            
            if len(test_data) > 10:  # Need sufficient observations
                try:
                    # Test with 1 and 2 lags
                    for lag in [1, 2]:
                        if len(test_data) > lag + 5:
                            gc_result = grangercausalitytests(test_data, maxlag=lag, verbose=False)
                            f_stat = gc_result[lag][0]['ssr_ftest'][0]
                            p_value = gc_result[lag][0]['ssr_ftest'][1]
                            print(f"  Lag {lag}: F-stat = {f_stat:.4f}, p-value = {p_value:.4f}")
                except Exception as e:
                    print(f"  Error in Granger test: {e}")

def lag_analysis(data):
    """Run regression with lagged GPR variables"""
    print("\n" + "=" * 60)
    print("LAG ANALYSIS")
    print("=" * 60)

    # Prepare data with lags
    lag_data = data[['Ambiguity_Metric', 'GPR_US', 'GPR_China', 'GPR_Japan']].copy()
    for lag in range(1, 4):  # Lags from 1 to 3 months
        for var in ['GPR_US', 'GPR_China', 'GPR_Japan']:
            lag_data[f'{var}_L{lag}'] = lag_data[var].shift(lag)

    lag_data = lag_data.dropna()
    print(f"Lag analysis sample size: {len(lag_data)}")

    # Define independent variables including lags
    lag_vars = ['GPR_US', 'GPR_China', 'GPR_Japan'] + [f'{var}_L{lag}' for lag in range(1, 4) for var in ['GPR_US', 'GPR_China', 'GPR_Japan']]
    
    y = lag_data['Ambiguity_Metric']
    X = lag_data[lag_vars]
    X = sm.add_constant(X)

    model_lag = sm.OLS(y, X).fit()
    print(model_lag.summary())
    
    return model_lag

def alternative_ambiguity_analysis(data):
    """Test with alternative ambiguity measures"""
    print("\n" + "=" * 60)
    print("ALTERNATIVE AMBIGUITY MEASURES")
    print("=" * 60)
    
    if 'New_Ambiguity_Metric' in data.columns:
        print("Testing with New_Ambiguity_Metric")
        print("-" * 40)
        
        reg_vars = ['New_Ambiguity_Metric', 'GPR_US', 'GPR_China', 'GPR_Japan']
        reg_data = data[reg_vars].dropna()
        
        if len(reg_data) > 10:
            y = reg_data['New_Ambiguity_Metric']
            X = reg_data[['GPR_US', 'GPR_China', 'GPR_Japan']]
            X = sm.add_constant(X)
            
            model_alt = sm.OLS(y, X).fit()
            print(model_alt.summary())
        else:
            print("Insufficient data for New_Ambiguity_Metric analysis")
    else:
        print("New_Ambiguity_Metric not available")

def summary_results(model1, model2, model3, model_lag):
    """Summarize key findings"""
    print("\n" + "=" * 60)
    print("SUMMARY OF KEY FINDINGS")
    print("=" * 60)
    
    print("Model Comparison:")
    print(f"Linear Model R²: {model1.rsquared:.4f}")
    print(f"Interaction Model R²: {model2.rsquared:.4f}")
    print(f"Quadratic Model R²: {model3.rsquared:.4f}")
    print(f"Lag Model R²: {model_lag.rsquared:.4f}")
    
    print("\nSignificant GPR Effects (p < 0.05):")
    
    # Check significance in linear model
    print("\nLinear Model:")
    for var in ['GPR_US', 'GPR_China', 'GPR_Japan']:
        if var in model1.params.index:
            coef = model1.params[var]
            pval = model1.pvalues[var]
            sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
            print(f"  {var}: {coef:.6f} (p={pval:.4f}) {sig}")
    
    print(f"\nBest Model: {'Quadratic' if model3.rsquared > max(model1.rsquared, model2.rsquared) else 'Interaction' if model2.rsquared > model1.rsquared else 'Linear'}")

def main():
    """Main execution function"""
    data = load_data()
    basic_correlations(data)
    model1, model2, model3 = benchmark_regression(data)
    stationarity_tests(data)
    granger_causality_analysis(data)
    model_lag = lag_analysis(data)  # Run lag analysis
    alternative_ambiguity_analysis(data)
    summary_results(model1, model2, model3, model_lag) # Pass lag model to summary

if __name__ == "__main__":
    main()