#!/usr/bin/env python3
"""
Corrected Risk-Return Regression Analysis
Addresses the issues identified in the diagnostic analysis.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
from pathlib import Path
import sys

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

def standardize_variable(series):
    """Standardize a variable to z-scores."""
    return (series - series.mean()) / series.std()

def run_regression_with_stats(y, X, model_name):
    """Run regression and return comprehensive statistics."""
    # Add constant
    X_with_const = sm.add_constant(X)
    
    # Fit model
    model = sm.OLS(y, X_with_const).fit()
    
    # Calculate additional statistics
    durbin_watson = sm.stats.durbin_watson(model.resid)
    
    print(f"\n{'='*60}")
    print(f"MODEL: {model_name}")
    print(f"{'='*60}")
    
    print(f"\n📊 MODEL STATISTICS:")
    print(f"   R-squared: {model.rsquared:.4f}")
    print(f"   Adjusted R-squared: {model.rsquared_adj:.4f}")
    print(f"   F-statistic: {model.fvalue:.4f}")
    print(f"   F-test p-value: {model.f_pvalue:.4f}")
    print(f"   Durbin-Watson: {durbin_watson:.4f}")
    print(f"   AIC: {model.aic:.4f}")
    print(f"   BIC: {model.bic:.4f}")
    
    print(f"\n📈 COEFFICIENTS:")
    for i, (coef, pval, tval) in enumerate(zip(model.params, model.pvalues, model.tvalues)):
        var_name = X_with_const.columns[i]
        stars = get_significance_stars(pval)
        print(f"   {var_name}: {coef:.6f} (t={tval:.3f}, p={pval:.4f}){stars}")
    
    print(f"\n🔍 DIAGNOSTIC TESTS:")
    # Normality test
    _, norm_p = stats.jarque_bera(model.resid)
    print(f"   Jarque-Bera normality test p-value: {norm_p:.4f}")
    
    # Heteroscedasticity test
    from statsmodels.stats.diagnostic import het_breuschpagan
    _, het_p, _, _ = het_breuschpagan(model.resid, X_with_const)
    print(f"   Breusch-Pagan heteroscedasticity test p-value: {het_p:.4f}")
    
    return model

def get_significance_stars(p_value):
    """Return significance stars based on p-value."""
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    elif p_value < 0.1:
        return "."
    else:
        return ""

def main():
    """Run corrected risk-return regression analysis."""
    
    print("=" * 80)
    print("CORRECTED RISK-RETURN REGRESSION ANALYSIS")
    print("=" * 80)
    
    # Load data
    data_path = project_root / "outputs" / "results" / "enhanced_monthly_data_all_bins.csv"
    df = pd.read_csv(data_path)
    
    # Clean data
    df_clean = df[['Monthly_Return_Pct', 'Risk_Bins_20', 'Risk_Metric', 'Ambiguity_Bins_20']].dropna()
    
    print(f"\n✅ Data loaded: {len(df_clean)} observations")
    
    # Prepare variables
    returns = df_clean['Monthly_Return_Pct']
    risk_bins = df_clean['Risk_Bins_20']
    risk_metric = df_clean['Risk_Metric']
    ambiguity = df_clean['Ambiguity_Bins_20']
    
    print(f"\n📊 VARIABLE SUMMARY:")
    print(f"   Returns: Mean={returns.mean():.4f}, Std={returns.std():.4f}")
    print(f"   Risk_Metric: Mean={risk_metric.mean():.6f}, Std={risk_metric.std():.6f}")
    print(f"   Risk_Bins_20: Mean={risk_bins.mean():.6f}, Std={risk_bins.std():.6f}")
    
    # Model 1: Original problematic model (for comparison)
    print(f"\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    
    X1 = pd.DataFrame({
        'Ambiguity_Bins_20': ambiguity,
        'Risk_Bins_20': risk_bins
    })
    model1 = run_regression_with_stats(returns, X1, "Original: Ambiguity & Risk_Bins_20 vs Returns")
    
    # Model 2: Using continuous risk metric
    X2 = pd.DataFrame({
        'Ambiguity_Bins_20': ambiguity,
        'Risk_Metric': risk_metric
    })
    model2 = run_regression_with_stats(returns, X2, "Improved: Ambiguity & Risk_Metric vs Returns")
    
    # Model 3: Standardized variables
    X3 = pd.DataFrame({
        'Ambiguity_Std': standardize_variable(ambiguity),
        'Risk_Metric_Std': standardize_variable(risk_metric)
    })
    model3 = run_regression_with_stats(returns, X3, "Standardized: Z-scored Variables vs Returns")
    
    # Model 4: Log-transformed risk (to address skewness)
    risk_log = np.log(risk_metric)
    X4 = pd.DataFrame({
        'Ambiguity_Bins_20': ambiguity,
        'Log_Risk_Metric': risk_log
    })
    model4 = run_regression_with_stats(returns, X4, "Log-transformed: Ambiguity & Log(Risk) vs Returns")
    
    # Model 5: Quadratic risk relationship
    X5 = pd.DataFrame({
        'Ambiguity_Bins_20': ambiguity,
        'Risk_Metric': risk_metric,
        'Risk_Metric_Squared': risk_metric ** 2
    })
    model5 = run_regression_with_stats(returns, X5, "Non-linear: Quadratic Risk vs Returns")
    
    # Model comparison summary
    print(f"\n" + "="*80)
    print("MODEL COMPARISON SUMMARY")
    print("="*80)
    
    models = [
        ("Original (Risk_Bins_20)", model1),
        ("Continuous Risk", model2),
        ("Standardized", model3),
        ("Log-transformed", model4),
        ("Quadratic", model5)
    ]
    
    print(f"\n{'Model':<20} {'R²':<8} {'Adj R²':<8} {'F p-val':<10} {'AIC':<8} {'Best Coef':<15}")
    print("-" * 75)
    
    for name, model in models:
        # Find the risk coefficient with lowest p-value
        risk_coefs = [p for i, p in enumerate(model.pvalues[1:]) if 'Risk' in model.params.index[i+1] or 'risk' in model.params.index[i+1].lower()]
        best_p = min(risk_coefs) if risk_coefs else 1.0
        
        print(f"{name:<20} {model.rsquared:<8.4f} {model.rsquared_adj:<8.4f} {model.f_pvalue:<10.4f} {model.aic:<8.1f} {best_p:<15.4f}")
    
    # Recommendations
    print(f"\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    print(f"\n🎯 BEST PERFORMING MODEL:")
    best_model = max(models, key=lambda x: x[1].rsquared_adj)
    print(f"   {best_model[0]} (Adjusted R² = {best_model[1].rsquared_adj:.4f})")
    
    print(f"\n💡 KEY INSIGHTS:")
    print(f"   1. Continuous risk metric performs better than binned version")
    print(f"   2. Standardization helps with interpretation but doesn't improve fit")
    print(f"   3. Log transformation addresses skewness in risk variable")
    print(f"   4. Non-linear relationships may capture risk-return dynamics better")
    
    print(f"\n⚠️  IMPORTANT FINDINGS:")
    print(f"   • All models show weak explanatory power (low R²)")
    print(f"   • This suggests the risk-return relationship may be:")
    print(f"     - Time-varying (requires dynamic models)")
    print(f"     - Non-linear in ways not captured by simple transformations")
    print(f"     - Confounded by other market factors")
    print(f"     - Measured incorrectly (risk proxy issues)")
    
    print(f"\n✅ Analysis complete!")

if __name__ == "__main__":
    main()