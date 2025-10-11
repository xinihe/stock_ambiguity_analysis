#!/usr/bin/env python3
"""
GPR-Ambiguity Relationship Analysis
==================================

This script investigates the relationship between Geopolitical Risk (GPR) from 
US, China, and Japan and market ambiguity measures. This analysis establishes
the foundation for understanding the causal mechanism:
GPR → Ambiguity → Market Returns

Key Research Questions:
1. Do GPR measures from major economies significantly affect market ambiguity?
2. Which country's GPR has the strongest impact on ambiguity?
3. What are the optimal lag structures for GPR effects?
4. Are there interaction effects between different countries' GPR?

Author: Research Team
Date: December 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_prepare_data():
    """Load and merge GPR data with ambiguity measures"""
    print("=" * 60)
    print("LOADING AND PREPARING DATA")
    print("=" * 60)
    
    # Load monthly combined data (contains both ambiguity measures and GPR data)
    print("Loading monthly combined data with ambiguity measures and GPR data...")
    data_path = '/Users/tlxy/Research/Ambiguity/code/Adaptive/GeoUncertainty/outputs/results/monthly_combined_analysis.csv'
    merged_data = pd.read_csv(data_path)
    
    # Convert dates
    merged_data['Date'] = pd.to_datetime(merged_data['Date'])
    
    print(f"Dataset shape: {merged_data.shape}")
    print(f"Date range: {merged_data['Date'].min()} to {merged_data['Date'].max()}")
    print(f"Number of observations: {len(merged_data)}")
    
    # Check available columns
    print(f"\nAvailable columns: {list(merged_data.columns)}")
    
    # Identify GPR columns
    gpr_columns = [col for col in merged_data.columns if 'GPR' in col.upper()]
    print(f"GPR columns found: {gpr_columns}")
    
    # Display key variables
    key_vars = ['Ambiguity_Metric']
    if 'GPR_US' in merged_data.columns:
        key_vars.append('GPR_US')
    if 'GPR_China' in merged_data.columns:
        key_vars.append('GPR_China')
    if 'GPR_Japan' in merged_data.columns:
        key_vars.append('GPR_Japan')
    
    print("\nKey variables summary:")
    print(merged_data[key_vars].describe())
    
    return merged_data

def exploratory_analysis(data):
    """Perform exploratory data analysis"""
    print("\n" + "=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    
    # Correlation analysis
    gpr_vars = ['GPR_US', 'GPR_China', 'GPR_Japan']
    ambiguity_vars = ['Ambiguity_Metric']
    
    # Check if New_Ambiguity_Metric exists
    if 'New_Ambiguity_Metric' in data.columns:
        ambiguity_vars.append('New_Ambiguity_Metric')
    
    # Calculate correlations
    corr_matrix = data[gpr_vars + ambiguity_vars].corr()
    
    print("Correlation Matrix: GPR vs Ambiguity")
    print("-" * 40)
    for amb_var in ambiguity_vars:
        print(f"\n{amb_var}:")
        for gpr_var in gpr_vars:
            corr = corr_matrix.loc[amb_var, gpr_var]
            print(f"  {gpr_var}: {corr:.4f}")
    
    # Time series plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Ambiguity over time
    axes[0, 0].plot(data['Date'], data['Ambiguity_Metric'], label='Ambiguity Metric', alpha=0.7)
    axes[0, 0].set_title('Ambiguity Metric Over Time')
    axes[0, 0].set_ylabel('Ambiguity')
    axes[0, 0].legend()
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Plot 2: GPR measures over time
    axes[0, 1].plot(data['Date'], data['GPR_US'], label='US GPR', alpha=0.7)
    axes[0, 1].plot(data['Date'], data['GPR_China'], label='China GPR', alpha=0.7)
    axes[0, 1].plot(data['Date'], data['GPR_Japan'], label='Japan GPR', alpha=0.7)
    axes[0, 1].set_title('GPR Measures Over Time')
    axes[0, 1].set_ylabel('GPR')
    axes[0, 1].legend()
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Plot 3: Scatter plot - US GPR vs Ambiguity
    axes[1, 0].scatter(data['GPR_US'], data['Ambiguity_Metric'], alpha=0.6)
    axes[1, 0].set_xlabel('US GPR')
    axes[1, 0].set_ylabel('Ambiguity')
    axes[1, 0].set_title('US GPR vs Ambiguity')
    
    # Plot 4: Scatter plot - China GPR vs Ambiguity
    axes[1, 1].scatter(data['GPR_China'], data['Ambiguity_Metric'], alpha=0.6)
    axes[1, 1].set_xlabel('China GPR')
    axes[1, 1].set_ylabel('Ambiguity')
    axes[1, 1].set_title('China GPR vs Ambiguity')
    
    plt.tight_layout()
    plt.savefig('/Users/tlxy/Research/Ambiguity/code/Adaptive/GeoUncertainty/gpr_ambiguity_exploratory.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    return corr_matrix

def stationarity_tests(data):
    """Test for stationarity of key variables"""
    print("\n" + "=" * 60)
    print("STATIONARITY TESTS")
    print("=" * 60)
    
    test_vars = ['Ambiguity_Metric', 'GPR_US', 'GPR_China', 'GPR_Japan']
    
    results = {}
    for var in test_vars:
        # Remove NaN values
        series = data[var].dropna()
        
        # Augmented Dickey-Fuller test
        adf_result = adfuller(series, autolag='AIC')
        
        results[var] = {
            'ADF_statistic': adf_result[0],
            'p_value': adf_result[1],
            'critical_values': adf_result[4],
            'is_stationary': adf_result[1] < 0.05
        }
        
        print(f"\n{var}:")
        print(f"  ADF Statistic: {adf_result[0]:.4f}")
        print(f"  p-value: {adf_result[1]:.4f}")
        print(f"  Critical Values: {adf_result[4]}")
        print(f"  Stationary: {'Yes' if adf_result[1] < 0.05 else 'No'}")
    
    return results

def benchmark_regression(data):
    """Run benchmark regression: Ambiguity = f(GPR_US, GPR_China, GPR_Japan)"""
    print("\n" + "=" * 60)
    print("BENCHMARK REGRESSION ANALYSIS")
    print("=" * 60)
    
    # Prepare data
    reg_data = data[['Ambiguity_Metric', 'GPR_US', 'GPR_China', 'GPR_Japan']].dropna()
    
    # Model 1: Basic linear regression
    print("Model 1: Basic Linear Regression")
    print("-" * 40)
    
    y = reg_data['Ambiguity_Metric']
    X = reg_data[['GPR_US', 'GPR_China', 'GPR_Japan']]
    X = sm.add_constant(X)
    
    model1 = sm.OLS(y, X).fit()
    print(model1.summary())
    
    # Model diagnostics
    print("\nModel Diagnostics:")
    print(f"R-squared: {model1.rsquared:.4f}")
    print(f"Adjusted R-squared: {model1.rsquared_adj:.4f}")
    print(f"F-statistic: {model1.fvalue:.4f} (p-value: {model1.f_pvalue:.4f})")
    print(f"Durbin-Watson: {durbin_watson(model1.resid):.4f}")
    
    # Heteroscedasticity test
    bp_test = het_breuschpagan(model1.resid, model1.model.exog)
    print(f"Breusch-Pagan test p-value: {bp_test[1]:.4f}")
    
    # Model 2: With interaction terms
    print("\n" + "=" * 40)
    print("Model 2: With Interaction Terms")
    print("-" * 40)
    
    # Create interaction terms
    reg_data['US_China_Interaction'] = reg_data['GPR_US'] * reg_data['GPR_China']
    reg_data['US_Japan_Interaction'] = reg_data['GPR_US'] * reg_data['GPR_Japan']
    reg_data['China_Japan_Interaction'] = reg_data['GPR_China'] * reg_data['GPR_Japan']
    
    X2 = reg_data[['GPR_US', 'GPR_China', 'GPR_Japan',
                   'US_China_Interaction', 'US_Japan_Interaction', 'China_Japan_Interaction']]
    X2 = sm.add_constant(X2)
    
    model2 = sm.OLS(y, X2).fit()
    print(model2.summary())
    
    # Model 3: Non-linear specifications
    print("\n" + "=" * 40)
    print("Model 3: Non-linear Specifications")
    print("-" * 40)
    
    # Add squared terms
    reg_data['GPR_US_Squared'] = reg_data['GPR_US'] ** 2
    reg_data['GPR_China_Squared'] = reg_data['GPR_China'] ** 2
    reg_data['GPR_Japan_Squared'] = reg_data['GPR_Japan'] ** 2
    
    X3 = reg_data[['GPR_US', 'GPR_China', 'GPR_Japan',
                   'GPR_US_Squared', 'GPR_China_Squared', 'GPR_Japan_Squared']]
    X3 = sm.add_constant(X3)
    
    model3 = sm.OLS(y, X3).fit()
    print(model3.summary())
    
    return model1, model2, model3, reg_data

def lag_analysis(data):
    """Analyze lag structures and Granger causality"""
    print("\n" + "=" * 60)
    print("LAG STRUCTURE AND GRANGER CAUSALITY ANALYSIS")
    print("=" * 60)
    
    # Create lagged variables
    lag_data = data[['Date', 'Ambiguity_Metric', 'GPR_US', 'GPR_China', 'GPR_Japan']].copy()
    
    # Add 1-month and 2-month lags
    for lag in [1, 2]:
        lag_data[f'GPR_US_Lag{lag}'] = lag_data['GPR_US'].shift(lag)
        lag_data[f'GPR_China_Lag{lag}'] = lag_data['GPR_China'].shift(lag)
        lag_data[f'GPR_Japan_Lag{lag}'] = lag_data['GPR_Japan'].shift(lag)
    
    # Model with lags
    print("Model with Lagged GPR Variables")
    print("-" * 40)
    
    lag_vars = ['GPR_US', 'GPR_China', 'GPR_Japan',
                'GPR_US_Lag1', 'GPR_China_Lag1', 'GPR_Japan_Lag1',
                'GPR_US_Lag2', 'GPR_China_Lag2', 'GPR_Japan_Lag2']
    
    reg_lag_data = lag_data[['Ambiguity_Metric'] + lag_vars].dropna()
    
    y_lag = reg_lag_data['Ambiguity_Metric']
    X_lag = reg_lag_data[lag_vars]
    X_lag = sm.add_constant(X_lag)
    
    model_lag = sm.OLS(y_lag, X_lag).fit()
    print(model_lag.summary())
    
    # Granger causality tests
    print("\n" + "=" * 40)
    print("GRANGER CAUSALITY TESTS")
    print("-" * 40)
    
    gpr_vars = ['GPR_US', 'GPR_China', 'GPR_Japan']
    
    for gpr_var in gpr_vars:
        print(f"\nTesting if {gpr_var} Granger-causes Ambiguity:")
        try:
            # Prepare data for Granger test
            test_data = data[['Ambiguity_Metric', gpr_var]].dropna()
            
            # Test with different lag orders
            for max_lag in [1, 2, 3]:
                try:
                    gc_result = grangercausalitytests(test_data, maxlag=max_lag, verbose=False)
                    p_value = gc_result[max_lag][0]['ssr_ftest'][1]
                    print(f"  Lag {max_lag}: F-test p-value = {p_value:.4f}")
                except:
                    print(f"  Lag {max_lag}: Test failed")
        except Exception as e:
            print(f"  Error: {e}")
    
    return model_lag, lag_data

def robustness_checks(data):
    """Perform robustness checks with alternative specifications"""
    print("\n" + "=" * 60)
    print("ROBUSTNESS CHECKS")
    print("=" * 60)
    
    results = {}
    
    # 1. Alternative ambiguity measure
    if 'New_Ambiguity_Metric' in data.columns:
        print("1. Using Alternative Ambiguity Measure")
        print("-" * 40)
        
        reg_data_alt = data[['New_Ambiguity_Metric', 'GPR_US_Recent', 'GPR_China_Recent', 'GPR_Japan_Recent']].dropna()
        
        y_alt = reg_data_alt['New_Ambiguity_Metric']
        X_alt = reg_data_alt[['GPR_US_Recent', 'GPR_China_Recent', 'GPR_Japan_Recent']]
        X_alt = sm.add_constant(X_alt)
        
        model_alt = sm.OLS(y_alt, X_alt).fit()
        print(model_alt.summary())
        results['alternative_ambiguity'] = model_alt
    
    # 2. Historical GPR measures
    print("\n2. Using Historical GPR Measures")
    print("-" * 40)
    
    hist_vars = ['GPR_US_Historical', 'GPR_China_Historical', 'GPR_Japan_Historical']
    reg_data_hist = data[['Ambiguity_Metric'] + hist_vars].dropna()
    
    if len(reg_data_hist) > 10:  # Ensure sufficient data
        y_hist = reg_data_hist['Ambiguity_Metric']
        X_hist = reg_data_hist[hist_vars]
        X_hist = sm.add_constant(X_hist)
        
        model_hist = sm.OLS(y_hist, X_hist).fit()
        print(model_hist.summary())
        results['historical_gpr'] = model_hist
    else:
        print("Insufficient data for historical GPR analysis")
    
    # 3. Standardized variables
    print("\n3. Using Standardized Variables")
    print("-" * 40)
    
    std_data = data[['Ambiguity_Metric', 'GPR_US', 'GPR_China', 'GPR_Japan']].dropna()
    
    # Standardize variables
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    
    std_vars = ['GPR_US', 'GPR_China', 'GPR_Japan']
    std_data[std_vars] = scaler.fit_transform(std_data[std_vars])
    std_data['Ambiguity_Metric'] = (std_data['Ambiguity_Metric'] - std_data['Ambiguity_Metric'].mean()) / std_data['Ambiguity_Metric'].std()
    
    y_std = std_data['Ambiguity_Metric']
    X_std = std_data[std_vars]
    X_std = sm.add_constant(X_std)
    
    model_std = sm.OLS(y_std, X_std).fit()
    print(model_std.summary())
    results['standardized'] = model_std
    
    return results

def create_summary_report(models, data):
    """Create comprehensive summary report"""
    print("\n" + "=" * 60)
    print("COMPREHENSIVE SUMMARY REPORT")
    print("=" * 60)
    
    print("KEY FINDINGS:")
    print("-" * 20)
    
    # Extract key results from benchmark model
    model1 = models[0]  # Basic linear model
    
    print(f"1. Overall Model Performance:")
    print(f"   - R-squared: {model1.rsquared:.4f} ({model1.rsquared*100:.2f}% of ambiguity variation explained)")
    print(f"   - Adjusted R-squared: {model1.rsquared_adj:.4f}")
    print(f"   - F-statistic: {model1.fvalue:.4f} (p-value: {model1.f_pvalue:.4f})")
    
    print(f"\n2. Individual GPR Effects:")
    for i, var in enumerate(['GPR_US_Recent', 'GPR_China_Recent', 'GPR_Japan_Recent']):
        coef = model1.params[var]
        pval = model1.pvalues[var]
        significance = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
        print(f"   - {var}: {coef:.6f} (p-value: {pval:.4f}) {significance}")
    
    print(f"\n3. Economic Interpretation:")
    for i, var in enumerate(['GPR_US_Recent', 'GPR_China_Recent', 'GPR_Japan_Recent']):
        coef = model1.params[var]
        country = var.split('_')[1]
        if coef > 0:
            print(f"   - A 1-unit increase in {country} GPR increases ambiguity by {coef:.6f}")
        else:
            print(f"   - A 1-unit increase in {country} GPR decreases ambiguity by {abs(coef):.6f}")
    
    # Correlation summary
    gpr_vars = ['GPR_US_Recent', 'GPR_China_Recent', 'GPR_Japan_Recent']
    correlations = data[['Ambiguity_Metric'] + gpr_vars].corr()['Ambiguity_Metric'][gpr_vars]
    
    print(f"\n4. Correlation Analysis:")
    for var in gpr_vars:
        corr = correlations[var]
        country = var.split('_')[1]
        print(f"   - {country} GPR correlation with ambiguity: {corr:.4f}")
    
    print(f"\n5. Statistical Significance Summary:")
    significant_vars = []
    for var in ['GPR_US_Recent', 'GPR_China_Recent', 'GPR_Japan_Recent']:
        if model1.pvalues[var] < 0.05:
            significant_vars.append(var.split('_')[1])
    
    if significant_vars:
        print(f"   - Significant GPR effects (p < 0.05): {', '.join(significant_vars)}")
    else:
        print(f"   - No GPR variables are statistically significant at 5% level")
    
    print(f"\n6. Research Implications:")
    if model1.f_pvalue < 0.05:
        print(f"   - Overall model is statistically significant")
        print(f"   - GPR measures collectively explain {model1.rsquared*100:.2f}% of ambiguity variation")
        print(f"   - Evidence supports GPR → Ambiguity causal pathway")
    else:
        print(f"   - Overall model is not statistically significant")
        print(f"   - Limited evidence for GPR → Ambiguity relationship")
    
    if significant_vars:
        print(f"   - {', '.join(significant_vars)} GPR can be used as instrumental variables")
        print(f"   - Proceed with mediation analysis: GPR → Ambiguity → Returns")
    else:
        print(f"   - Consider alternative GPR measures or model specifications")
        print(f"   - May need to explore non-linear relationships")

def main():
    """Main analysis function"""
    print("GPR-AMBIGUITY RELATIONSHIP ANALYSIS")
    print("=" * 60)
    print("Investigating: GPR (US, China, Japan) → Ambiguity")
    print("Foundation for: GPR → Ambiguity → Market Returns")
    print("=" * 60)
    
    # Load and prepare data
    data = load_and_prepare_data()
    
    # Exploratory analysis
    corr_matrix = exploratory_analysis(data)
    
    # Stationarity tests
    stationarity_results = stationarity_tests(data)
    
    # Benchmark regression
    model1, model2, model3, reg_data = benchmark_regression(data)
    
    # Lag analysis
    model_lag, lag_data = lag_analysis(data)
    
    # Robustness checks
    robustness_results = robustness_checks(data)
    
    # Summary report
    create_summary_report([model1, model2, model3], data)
    
    print(f"\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print("Results saved to: gpr_ambiguity_exploratory.png")
    print("Next steps: Use significant GPR variables for mediation analysis")

if __name__ == "__main__":
    main()