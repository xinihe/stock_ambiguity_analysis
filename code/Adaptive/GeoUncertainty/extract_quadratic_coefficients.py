#!/usr/bin/env python3
"""
Extract Quadratic Risk Model Coefficients
=========================================

This script extracts the exact coefficients for the quadratic risk model:
Returns = β₀ + β₁(Ambiguity) + β₂(Risk) + β₃(Risk²) + ε

Focus on β₀ (intercept) and β₁ (ambiguity coefficient).
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Load and prepare the data for analysis."""
    print("Loading data...")
    
    # Load the data
    df = pd.read_csv('outputs/results/enhanced_monthly_data_all_bins.csv')
    
    # Clean data - remove rows with missing values in key variables
    key_vars = ['Monthly_Return_Pct', 'Ambiguity_Bins_20', 'Risk_Metric']
    df_clean = df[key_vars].dropna()
    
    print(f"Data loaded: {len(df_clean)} observations")
    print(f"Variables: {key_vars}")
    
    return df_clean

def run_quadratic_model(df):
    """Run the quadratic risk model and extract coefficients."""
    print("\n" + "="*60)
    print("QUADRATIC RISK MODEL ANALYSIS")
    print("="*60)
    
    # Prepare variables
    y = df['Monthly_Return_Pct']
    ambiguity = df['Ambiguity_Bins_20']
    risk = df['Risk_Metric']
    risk_squared = risk ** 2
    
    # Create design matrix
    X = pd.DataFrame({
        'const': 1,
        'Ambiguity_Bins_20': ambiguity,
        'Risk_Metric': risk,
        'Risk_Metric_Squared': risk_squared
    })
    
    # Run regression
    model = sm.OLS(y, X).fit()
    
    print(f"\nModel Specification:")
    print(f"Returns = β₀ + β₁(Ambiguity) + β₂(Risk) + β₃(Risk²) + ε")
    print(f"\nObservations: {model.nobs}")
    print(f"R-squared: {model.rsquared:.4f}")
    print(f"Adjusted R-squared: {model.rsquared_adj:.4f}")
    print(f"F-statistic p-value: {model.f_pvalue:.4f}")
    
    print(f"\n" + "="*60)
    print("COEFFICIENT ESTIMATES")
    print("="*60)
    
    # Extract coefficients
    coeffs = model.params
    std_errors = model.bse
    t_stats = model.tvalues
    p_values = model.pvalues
    conf_int = model.conf_int()
    
    # Function to determine significance stars
    def get_stars(p_val):
        if p_val < 0.001:
            return "***"
        elif p_val < 0.01:
            return "**"
        elif p_val < 0.05:
            return "*"
        elif p_val < 0.1:
            return "."
        else:
            return ""
    
    print(f"\n{'Variable':<20} {'Coeff':<12} {'Std Err':<10} {'t-stat':<8} {'p-value':<10} {'Sig':<5} {'95% CI':<25}")
    print("-" * 90)
    
    for var in X.columns:
        coeff = coeffs[var]
        se = std_errors[var]
        t_stat = t_stats[var]
        p_val = p_values[var]
        stars = get_stars(p_val)
        ci_lower = conf_int.loc[var, 0]
        ci_upper = conf_int.loc[var, 1]
        
        print(f"{var:<20} {coeff:>11.4f} {se:>9.4f} {t_stat:>7.3f} {p_val:>9.4f} {stars:<5} [{ci_lower:>7.3f}, {ci_upper:>7.3f}]")
    
    # Highlight key coefficients
    print(f"\n" + "="*60)
    print("KEY COEFFICIENTS (REQUESTED)")
    print("="*60)
    
    beta_0 = coeffs['const']
    beta_1 = coeffs['Ambiguity_Bins_20']
    
    print(f"\nβ₀ (Intercept):     {beta_0:>10.4f}")
    print(f"   Standard Error:  {std_errors['const']:>10.4f}")
    print(f"   t-statistic:     {t_stats['const']:>10.3f}")
    print(f"   p-value:         {p_values['const']:>10.4f} {get_stars(p_values['const'])}")
    
    print(f"\nβ₁ (Ambiguity):    {beta_1:>10.4f}")
    print(f"   Standard Error:  {std_errors['Ambiguity_Bins_20']:>10.4f}")
    print(f"   t-statistic:     {t_stats['Ambiguity_Bins_20']:>10.3f}")
    print(f"   p-value:         {p_values['Ambiguity_Bins_20']:>10.4f} {get_stars(p_values['Ambiguity_Bins_20'])}")
    
    # Additional coefficients for completeness
    print(f"\nβ₂ (Risk):         {coeffs['Risk_Metric']:>10.4f}")
    print(f"   p-value:         {p_values['Risk_Metric']:>10.4f} {get_stars(p_values['Risk_Metric'])}")
    
    print(f"\nβ₃ (Risk²):        {coeffs['Risk_Metric_Squared']:>10.4f}")
    print(f"   p-value:         {p_values['Risk_Metric_Squared']:>10.4f} {get_stars(p_values['Risk_Metric_Squared'])}")
    
    print(f"\n" + "="*60)
    print("MODEL INTERPRETATION")
    print("="*60)
    
    print(f"\n1. INTERCEPT (β₀ = {beta_0:.4f}):")
    if p_values['const'] < 0.05:
        print(f"   • Statistically significant baseline return")
        print(f"   • Expected return when Ambiguity = 0 and Risk = 0")
    else:
        print(f"   • Not statistically significant")
        print(f"   • Baseline return not significantly different from zero")
    
    print(f"\n2. AMBIGUITY EFFECT (β₁ = {beta_1:.4f}):")
    if p_values['Ambiguity_Bins_20'] < 0.05:
        if beta_1 > 0:
            print(f"   • Positive and significant ambiguity premium")
            print(f"   • Higher ambiguity associated with higher returns")
        else:
            print(f"   • Negative and significant ambiguity effect")
            print(f"   • Higher ambiguity associated with lower returns")
    else:
        print(f"   • Not statistically significant")
        print(f"   • Ambiguity does not significantly predict returns")
    
    print(f"\n3. RISK EFFECTS:")
    if p_values['Risk_Metric'] < 0.05 or p_values['Risk_Metric_Squared'] < 0.05:
        print(f"   • Significant non-linear risk-return relationship")
        print(f"   • Risk effect varies with risk level")
    else:
        print(f"   • Risk effects not statistically significant")
    
    return model

def main():
    """Main analysis function."""
    print("QUADRATIC RISK MODEL COEFFICIENT EXTRACTION")
    print("=" * 60)
    
    # Load data
    df = load_and_prepare_data()
    
    # Run quadratic model
    model = run_quadratic_model(df)
    
    print(f"\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nThe quadratic risk model specification is:")
    print(f"Returns = β₀ + β₁(Ambiguity) + β₂(Risk) + β₃(Risk²) + ε")
    print(f"\nKey findings:")
    print(f"• β₀ (intercept): {model.params['const']:.4f}")
    print(f"• β₁ (ambiguity): {model.params['Ambiguity_Bins_20']:.4f}")
    print(f"• Model explains {model.rsquared_adj:.1%} of return variation")

if __name__ == "__main__":
    main()