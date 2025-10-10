#!/usr/bin/env python3
"""
Two Specific Regressions Analysis
1. Ambiguity & Risk vs Monthly Returns (using optimal 20 bins)
2. Climate Risk & US GPR vs Ambiguity (using optimal 20 bins)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load the enhanced monthly data with all bin configurations"""
    data_path = Path("/Users/tlxy/Research/Ambiguity/code/Adaptive/GeoUncertainty/outputs/results/enhanced_monthly_data_all_bins.csv")
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    print(f"✅ Data loaded successfully: {len(df)} observations")
    print(f"📅 Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    return df

def run_regression_1(df):
    """
    Regression 1: Ambiguity & Risk vs Monthly Returns (using 20 bins)
    Dependent Variable: Monthly_Return_Pct
    Independent Variables: Ambiguity_Bins_20, Risk_Bins_20
    """
    print("\n" + "="*80)
    print("REGRESSION 1: AMBIGUITY & RISK vs MONTHLY RETURNS (20 BINS)")
    print("="*80)
    
    # Prepare data
    y = df['Monthly_Return_Pct'].dropna()
    X = df[['Ambiguity_Bins_20', 'Risk_Bins_20']].dropna()
    
    # Align indices
    common_idx = y.index.intersection(X.index)
    y = y.loc[common_idx]
    X = X.loc[common_idx]
    
    # Add constant
    X = sm.add_constant(X)
    
    # Run regression
    model = sm.OLS(y, X).fit()
    
    # Display results
    print(f"\n📊 MODEL SUMMARY:")
    print(f"   Dependent Variable: Monthly_Return_Pct")
    print(f"   Independent Variables: Ambiguity_Bins_20, Risk_Bins_20")
    print(f"   Observations: {model.nobs}")
    print(f"   R-squared: {model.rsquared:.4f}")
    print(f"   Adjusted R-squared: {model.rsquared_adj:.4f}")
    print(f"   F-statistic p-value: {model.f_pvalue:.4f}")
    print(f"   Durbin-Watson: {sm.stats.durbin_watson(model.resid):.4f}")
    
    print(f"\n📈 COEFFICIENTS:")
    for i, (var, coef) in enumerate(zip(model.params.index, model.params.values)):
        se = model.bse.iloc[i]
        pval = model.pvalues.iloc[i]
        
        # Significance stars
        stars = ""
        if pval < 0.01:
            stars = "***"
        elif pval < 0.05:
            stars = "**"
        elif pval < 0.10:
            stars = "*"
        
        print(f"   {var}: {coef:.4f} (SE: {se:.4f}, p: {pval:.4f}) {stars}")
    
    print(f"\n🔍 ECONOMIC INTERPRETATION:")
    ambiguity_coef = model.params['Ambiguity_Bins_20']
    risk_coef = model.params['Risk_Bins_20']
    ambiguity_pval = model.pvalues['Ambiguity_Bins_20']
    risk_pval = model.pvalues['Risk_Bins_20']
    
    print(f"   • Ambiguity Effect: {ambiguity_coef:.4f} (p={ambiguity_pval:.4f})")
    if ambiguity_pval < 0.10:
        direction = "increases" if ambiguity_coef > 0 else "decreases"
        print(f"     Higher ambiguity significantly {direction} monthly returns")
    else:
        print(f"     Ambiguity effect is not statistically significant")
    
    print(f"   • Risk Effect: {risk_coef:.4f} (p={risk_pval:.4f})")
    if risk_pval < 0.10:
        direction = "increases" if risk_coef > 0 else "decreases"
        print(f"     Higher risk significantly {direction} monthly returns")
    else:
        print(f"     Risk effect is not statistically significant")
    
    return model

def run_regression_2(df):
    """
    Regression 2: Climate Risk & US GPR vs Ambiguity (using 20 bins)
    Dependent Variable: Ambiguity_Bins_20
    Independent Variables: Climate_Risk_Component, GPR_US_y
    """
    print("\n" + "="*80)
    print("REGRESSION 2: CLIMATE RISK & US GPR vs AMBIGUITY (20 BINS)")
    print("="*80)
    
    # Check available climate risk variables
    climate_vars = [col for col in df.columns if 'climate' in col.lower() or 'Climate' in col]
    print(f"📋 Available climate variables: {climate_vars}")
    
    # Use Climate_Risk_Component if available, otherwise skip
    if 'Climate_Risk_Component' not in df.columns:
        print("❌ Climate_Risk_Component not found in dataset")
        return None
    
    # Prepare data
    y = df['Ambiguity_Bins_20'].dropna()
    X = df[['Climate_Risk_Component', 'GPR_US_y']].dropna()
    
    # Align indices
    common_idx = y.index.intersection(X.index)
    y = y.loc[common_idx]
    X = X.loc[common_idx]
    
    # Add constant
    X = sm.add_constant(X)
    
    # Run regression
    model = sm.OLS(y, X).fit()
    
    # Display results
    print(f"\n📊 MODEL SUMMARY:")
    print(f"   Dependent Variable: Ambiguity_Bins_20")
    print(f"   Independent Variables: Climate_Risk_Component, GPR_US_y")
    print(f"   Observations: {model.nobs}")
    print(f"   R-squared: {model.rsquared:.4f}")
    print(f"   Adjusted R-squared: {model.rsquared_adj:.4f}")
    print(f"   F-statistic p-value: {model.f_pvalue:.4f}")
    print(f"   Durbin-Watson: {sm.stats.durbin_watson(model.resid):.4f}")
    
    print(f"\n📈 COEFFICIENTS:")
    for i, (var, coef) in enumerate(zip(model.params.index, model.params.values)):
        se = model.bse.iloc[i]
        pval = model.pvalues.iloc[i]
        
        # Significance stars
        stars = ""
        if pval < 0.01:
            stars = "***"
        elif pval < 0.05:
            stars = "**"
        elif pval < 0.10:
            stars = "*"
        
        print(f"   {var}: {coef:.4f} (SE: {se:.4f}, p: {pval:.4f}) {stars}")
    
    print(f"\n🔍 ECONOMIC INTERPRETATION:")
    climate_coef = model.params['Climate_Risk_Component']
    gpr_coef = model.params['GPR_US_y']
    climate_pval = model.pvalues['Climate_Risk_Component']
    gpr_pval = model.pvalues['GPR_US_y']
    
    print(f"   • Climate Risk Effect: {climate_coef:.4f} (p={climate_pval:.4f})")
    if climate_pval < 0.10:
        direction = "increases" if climate_coef > 0 else "decreases"
        print(f"     Higher climate risk significantly {direction} market ambiguity")
    else:
        print(f"     Climate risk effect on ambiguity is not statistically significant")
    
    print(f"   • US GPR Effect: {gpr_coef:.4f} (p={gpr_pval:.4f})")
    if gpr_pval < 0.10:
        direction = "increases" if gpr_coef > 0 else "decreases"
        print(f"     Higher US geopolitical risk significantly {direction} market ambiguity")
    else:
        print(f"     US GPR effect on ambiguity is not statistically significant")
    
    return model

def generate_summary_comparison(model1, model2):
    """Generate a summary comparison of both models"""
    print("\n" + "="*80)
    print("COMPARATIVE SUMMARY")
    print("="*80)
    
    print(f"\n📊 MODEL COMPARISON:")
    print(f"{'Metric':<25} {'Model 1 (Amb&Risk→Returns)':<30} {'Model 2 (Climate&GPR→Amb)':<30}")
    print(f"{'-'*25} {'-'*30} {'-'*30}")
    
    if model1 and model2:
        print(f"{'R-squared':<25} {model1.rsquared:<30.4f} {model2.rsquared:<30.4f}")
        print(f"{'Adjusted R-squared':<25} {model1.rsquared_adj:<30.4f} {model2.rsquared_adj:<30.4f}")
        print(f"{'F-test p-value':<25} {model1.f_pvalue:<30.4f} {model2.f_pvalue:<30.4f}")
        print(f"{'Observations':<25} {int(model1.nobs):<30} {int(model2.nobs):<30}")
        
        # Determine which model is better
        print(f"\n🏆 MODEL PERFORMANCE:")
        if model1.rsquared_adj > model2.rsquared_adj:
            print(f"   • Model 1 has better explanatory power (Adj R² = {model1.rsquared_adj:.4f})")
        else:
            print(f"   • Model 2 has better explanatory power (Adj R² = {model2.rsquared_adj:.4f})")
        
        if model1.f_pvalue < model2.f_pvalue:
            print(f"   • Model 1 is more statistically significant (F p-value = {model1.f_pvalue:.4f})")
        else:
            print(f"   • Model 2 is more statistically significant (F p-value = {model2.f_pvalue:.4f})")
    
    elif model1:
        print(f"{'R-squared':<25} {model1.rsquared:<30.4f} {'N/A':<30}")
        print(f"{'Adjusted R-squared':<25} {model1.rsquared_adj:<30.4f} {'N/A':<30}")
        print(f"{'F-test p-value':<25} {model1.f_pvalue:<30.4f} {'N/A':<30}")
        print(f"{'Observations':<25} {int(model1.nobs):<30} {'N/A':<30}")

def main():
    """Main analysis function"""
    try:
        # Load data
        df = load_data()
        
        # Run both regressions
        model1 = run_regression_1(df)
        model2 = run_regression_2(df)
        
        # Generate comparison
        generate_summary_comparison(model1, model2)
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        print("\n✅ Both regression analyses completed successfully!")
        
    except Exception as e:
        print(f"❌ Error in analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()