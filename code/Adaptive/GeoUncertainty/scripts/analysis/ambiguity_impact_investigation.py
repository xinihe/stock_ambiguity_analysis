#!/usr/bin/env python3
"""
Ambiguity Impact Investigation
=============================

This script explores various approaches to reveal significant ambiguity effects on returns:
1. Projective risk adjustment
2. Alternative model specifications
3. Time-varying effects
4. Interaction terms
5. Non-linear transformations
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def load_and_analyze_data():
    """Load data and analyze ambiguity characteristics."""
    print("="*80)
    print("AMBIGUITY VARIABLE ANALYSIS")
    print("="*80)
    
    # Load data
    df = pd.read_csv('outputs/results/enhanced_monthly_data_all_bins.csv')
    
    # Clean data
    key_vars = ['Monthly_Return_Pct', 'Ambiguity_Bins_20', 'Risk_Metric']
    df_clean = df[key_vars].dropna()
    
    print(f"\nData loaded: {len(df_clean)} observations")
    
    # Analyze ambiguity variable
    ambiguity = df_clean['Ambiguity_Bins_20']
    returns = df_clean['Monthly_Return_Pct']
    risk = df_clean['Risk_Metric']
    
    print(f"\n📊 AMBIGUITY VARIABLE CHARACTERISTICS:")
    print(f"   Mean: {ambiguity.mean():.6f}")
    print(f"   Std: {ambiguity.std():.6f}")
    print(f"   Min: {ambiguity.min():.6f}")
    print(f"   Max: {ambiguity.max():.6f}")
    print(f"   Skewness: {stats.skew(ambiguity):.4f}")
    print(f"   Kurtosis: {stats.kurtosis(ambiguity):.4f}")
    print(f"   Unique values: {ambiguity.nunique()}")
    
    # Correlation analysis
    print(f"\n🔗 CORRELATION ANALYSIS:")
    print(f"   Ambiguity vs Returns: {np.corrcoef(ambiguity, returns)[0,1]:.4f}")
    print(f"   Ambiguity vs Risk: {np.corrcoef(ambiguity, risk)[0,1]:.4f}")
    print(f"   Risk vs Returns: {np.corrcoef(risk, returns)[0,1]:.4f}")
    
    return df_clean

def projective_risk_adjustment(df):
    """Implement projective risk adjustment to isolate ambiguity effects."""
    print(f"\n" + "="*80)
    print("PROJECTIVE RISK ADJUSTMENT")
    print("="*80)
    
    returns = df['Monthly_Return_Pct']
    ambiguity = df['Ambiguity_Bins_20']
    risk = df['Risk_Metric']
    
    # Step 1: Project ambiguity onto risk space
    print(f"\n🎯 Step 1: Projecting ambiguity onto risk space...")
    
    # Regress ambiguity on risk to get the projection
    X_risk = sm.add_constant(risk)
    ambiguity_on_risk = sm.OLS(ambiguity, X_risk).fit()
    
    # Get predicted ambiguity from risk
    ambiguity_predicted = ambiguity_on_risk.fittedvalues
    
    # Get orthogonal component (residual ambiguity)
    ambiguity_orthogonal = ambiguity_on_risk.resid
    
    print(f"   R² of ambiguity on risk: {ambiguity_on_risk.rsquared:.4f}")
    print(f"   Correlation (predicted ambiguity, risk): {np.corrcoef(ambiguity_predicted, risk)[0,1]:.4f}")
    print(f"   Correlation (orthogonal ambiguity, risk): {np.corrcoef(ambiguity_orthogonal, risk)[0,1]:.4f}")
    
    # Step 2: Test orthogonal ambiguity effect on returns
    print(f"\n🎯 Step 2: Testing orthogonal ambiguity effect...")
    
    # Model 1: Original ambiguity + risk
    X1 = pd.DataFrame({
        'Ambiguity_Original': ambiguity,
        'Risk_Metric': risk
    })
    X1_const = sm.add_constant(X1)
    model1 = sm.OLS(returns, X1_const).fit()
    
    # Model 2: Orthogonal ambiguity + risk
    X2 = pd.DataFrame({
        'Ambiguity_Orthogonal': ambiguity_orthogonal,
        'Risk_Metric': risk
    })
    X2_const = sm.add_constant(X2)
    model2 = sm.OLS(returns, X2_const).fit()
    
    # Model 3: Both components + risk
    X3 = pd.DataFrame({
        'Ambiguity_Predicted': ambiguity_predicted,
        'Ambiguity_Orthogonal': ambiguity_orthogonal,
        'Risk_Metric': risk
    })
    X3_const = sm.add_constant(X3)
    model3 = sm.OLS(returns, X3_const).fit()
    
    print_model_results("Original Ambiguity + Risk", model1)
    print_model_results("Orthogonal Ambiguity + Risk", model2)
    print_model_results("Decomposed Ambiguity + Risk", model3)
    
    return {
        'ambiguity_orthogonal': ambiguity_orthogonal,
        'ambiguity_predicted': ambiguity_predicted,
        'model_orthogonal': model2,
        'model_decomposed': model3
    }

def test_alternative_specifications(df, ambiguity_orthogonal):
    """Test alternative model specifications to reveal ambiguity effects."""
    print(f"\n" + "="*80)
    print("ALTERNATIVE MODEL SPECIFICATIONS")
    print("="*80)
    
    returns = df['Monthly_Return_Pct']
    ambiguity = df['Ambiguity_Bins_20']
    risk = df['Risk_Metric']
    
    # Create lagged variables (assuming time series structure)
    df_with_lags = df.copy()
    df_with_lags['Ambiguity_Lag1'] = df_with_lags['Ambiguity_Bins_20'].shift(1)
    df_with_lags['Risk_Lag1'] = df_with_lags['Risk_Metric'].shift(1)
    df_with_lags['Returns_Lag1'] = df_with_lags['Monthly_Return_Pct'].shift(1)
    
    # Remove NaN from lagged data
    df_lags_clean = df_with_lags.dropna()
    
    if len(df_lags_clean) > 10:  # Ensure enough data for lagged analysis
        print(f"\n🕐 LAGGED EFFECTS ANALYSIS:")
        print(f"   Available observations with lags: {len(df_lags_clean)}")
        
        # Model with lagged ambiguity
        X_lag = pd.DataFrame({
            'Ambiguity_Current': df_lags_clean['Ambiguity_Bins_20'],
            'Ambiguity_Lag1': df_lags_clean['Ambiguity_Lag1'],
            'Risk_Metric': df_lags_clean['Risk_Metric']
        })
        X_lag_const = sm.add_constant(X_lag)
        model_lag = sm.OLS(df_lags_clean['Monthly_Return_Pct'], X_lag_const).fit()
        print_model_results("Lagged Ambiguity Model", model_lag)
    
    # Non-linear transformations
    print(f"\n🔄 NON-LINEAR TRANSFORMATIONS:")
    
    # Log transformation (add small constant to avoid log(0))
    ambiguity_log = np.log(ambiguity + 1e-6)
    
    # Square root transformation
    ambiguity_sqrt = np.sqrt(ambiguity - ambiguity.min() + 1e-6)
    
    # Quadratic terms
    ambiguity_squared = ambiguity ** 2
    
    # Model with log ambiguity
    X_log = pd.DataFrame({
        'Ambiguity_Log': ambiguity_log,
        'Risk_Metric': risk
    })
    X_log_const = sm.add_constant(X_log)
    model_log = sm.OLS(returns, X_log_const).fit()
    print_model_results("Log-Transformed Ambiguity", model_log)
    
    # Model with quadratic ambiguity
    X_quad = pd.DataFrame({
        'Ambiguity': ambiguity,
        'Ambiguity_Squared': ambiguity_squared,
        'Risk_Metric': risk
    })
    X_quad_const = sm.add_constant(X_quad)
    model_quad = sm.OLS(returns, X_quad_const).fit()
    print_model_results("Quadratic Ambiguity", model_quad)
    
    # Interaction terms
    print(f"\n🔗 INTERACTION EFFECTS:")
    
    ambiguity_risk_interaction = ambiguity * risk
    
    X_interact = pd.DataFrame({
        'Ambiguity': ambiguity,
        'Risk_Metric': risk,
        'Ambiguity_Risk_Interaction': ambiguity_risk_interaction
    })
    X_interact_const = sm.add_constant(X_interact)
    model_interact = sm.OLS(returns, X_interact_const).fit()
    print_model_results("Ambiguity-Risk Interaction", model_interact)
    
    return {
        'model_log': model_log,
        'model_quad': model_quad,
        'model_interact': model_interact
    }

def regime_analysis(df):
    """Analyze time-varying ambiguity effects and regime changes."""
    print(f"\n" + "="*80)
    print("REGIME AND TIME-VARYING ANALYSIS")
    print("="*80)
    
    returns = df['Monthly_Return_Pct']
    ambiguity = df['Ambiguity_Bins_20']
    risk = df['Risk_Metric']
    
    # High/Low ambiguity regimes
    ambiguity_median = ambiguity.median()
    high_ambiguity = (ambiguity > ambiguity_median).astype(int)
    
    print(f"\n📊 REGIME ANALYSIS:")
    print(f"   Ambiguity median threshold: {ambiguity_median:.6f}")
    print(f"   High ambiguity periods: {high_ambiguity.sum()}/{len(high_ambiguity)} ({100*high_ambiguity.mean():.1f}%)")
    
    # Model with regime dummy
    X_regime = pd.DataFrame({
        'Ambiguity': ambiguity,
        'Risk_Metric': risk,
        'High_Ambiguity_Regime': high_ambiguity,
        'Ambiguity_High_Regime_Interaction': ambiguity * high_ambiguity
    })
    X_regime_const = sm.add_constant(X_regime)
    model_regime = sm.OLS(returns, X_regime_const).fit()
    print_model_results("Regime-Dependent Ambiguity", model_regime)
    
    # High/Low volatility regimes
    risk_median = risk.median()
    high_risk = (risk > risk_median).astype(int)
    
    X_risk_regime = pd.DataFrame({
        'Ambiguity': ambiguity,
        'Risk_Metric': risk,
        'High_Risk_Regime': high_risk,
        'Ambiguity_High_Risk_Interaction': ambiguity * high_risk
    })
    X_risk_regime_const = sm.add_constant(X_risk_regime)
    model_risk_regime = sm.OLS(returns, X_risk_regime_const).fit()
    print_model_results("Risk-Regime Dependent Ambiguity", model_risk_regime)
    
    return {
        'model_regime': model_regime,
        'model_risk_regime': model_risk_regime,
        'high_ambiguity': high_ambiguity,
        'high_risk': high_risk
    }

def standardized_analysis(df):
    """Perform analysis with standardized variables."""
    print(f"\n" + "="*80)
    print("STANDARDIZED VARIABLES ANALYSIS")
    print("="*80)
    
    returns = df['Monthly_Return_Pct']
    ambiguity = df['Ambiguity_Bins_20']
    risk = df['Risk_Metric']
    
    # Standardize variables
    scaler = StandardScaler()
    
    returns_std = scaler.fit_transform(returns.values.reshape(-1, 1)).flatten()
    ambiguity_std = scaler.fit_transform(ambiguity.values.reshape(-1, 1)).flatten()
    risk_std = scaler.fit_transform(risk.values.reshape(-1, 1)).flatten()
    
    print(f"\n📏 STANDARDIZED VARIABLES:")
    print(f"   All variables now have mean ≈ 0, std ≈ 1")
    
    # Basic standardized model
    X_std = pd.DataFrame({
        'Ambiguity_Std': ambiguity_std,
        'Risk_Std': risk_std
    })
    X_std_const = sm.add_constant(X_std)
    model_std = sm.OLS(returns_std, X_std_const).fit()
    print_model_results("Standardized Variables", model_std)
    
    # Standardized with interactions
    X_std_interact = pd.DataFrame({
        'Ambiguity_Std': ambiguity_std,
        'Risk_Std': risk_std,
        'Ambiguity_Risk_Std_Interaction': ambiguity_std * risk_std
    })
    X_std_interact_const = sm.add_constant(X_std_interact)
    model_std_interact = sm.OLS(returns_std, X_std_interact_const).fit()
    print_model_results("Standardized with Interaction", model_std_interact)
    
    return {
        'model_std': model_std,
        'model_std_interact': model_std_interact
    }

def print_model_results(model_name, model):
    """Print formatted model results."""
    print(f"\n📈 {model_name.upper()}:")
    print(f"   R²: {model.rsquared:.4f} | Adj R²: {model.rsquared_adj:.4f} | F p-value: {model.f_pvalue:.4f}")
    
    for i, (var, coef, pval, tval) in enumerate(zip(model.model.exog_names, model.params, model.pvalues, model.tvalues)):
        stars = get_significance_stars(pval)
        if 'ambiguity' in var.lower() or 'const' in var.lower():
            print(f"   {var}: {coef:.6f} (t={tval:.3f}, p={pval:.4f}){stars}")

def get_significance_stars(p_value):
    """Return significance stars based on p-value."""
    if p_value < 0.001:
        return " ***"
    elif p_value < 0.01:
        return " **"
    elif p_value < 0.05:
        return " *"
    elif p_value < 0.1:
        return " ."
    else:
        return ""

def summary_and_recommendations(results):
    """Provide summary and recommendations."""
    print(f"\n" + "="*80)
    print("SUMMARY AND RECOMMENDATIONS")
    print("="*80)
    
    print(f"\n🎯 KEY FINDINGS:")
    
    # Check which models showed significant ambiguity effects
    significant_models = []
    
    # Check projective risk results
    if 'model_orthogonal' in results:
        ambiguity_pval = results['model_orthogonal'].pvalues.get('Ambiguity_Orthogonal', 1.0)
        if ambiguity_pval < 0.1:
            significant_models.append(f"Orthogonal Ambiguity (p={ambiguity_pval:.4f})")
    
    if significant_models:
        print(f"\n✅ SIGNIFICANT AMBIGUITY EFFECTS FOUND:")
        for model in significant_models:
            print(f"   • {model}")
    else:
        print(f"\n⚠️  NO SIGNIFICANT AMBIGUITY EFFECTS DETECTED")
    
    print(f"\n💡 RECOMMENDATIONS FOR FUTURE RESEARCH:")
    print(f"   1. Consider alternative ambiguity measures (entropy-based, option-implied)")
    print(f"   2. Investigate higher-frequency data (daily/weekly)")
    print(f"   3. Test sector-specific or size-specific effects")
    print(f"   4. Examine crisis periods separately")
    print(f"   5. Consider behavioral factors and investor sentiment")
    print(f"   6. Test with different asset classes or markets")

def main():
    """Main analysis function."""
    print("COMPREHENSIVE AMBIGUITY IMPACT INVESTIGATION")
    print("=" * 80)
    
    # Load and analyze data
    df = load_and_analyze_data()
    
    # Projective risk adjustment
    projective_results = projective_risk_adjustment(df)
    
    # Alternative specifications
    alt_results = test_alternative_specifications(df, projective_results['ambiguity_orthogonal'])
    
    # Regime analysis
    regime_results = regime_analysis(df)
    
    # Standardized analysis
    std_results = standardized_analysis(df)
    
    # Combine all results
    all_results = {**projective_results, **alt_results, **regime_results, **std_results}
    
    # Summary and recommendations
    summary_and_recommendations(all_results)
    
    print(f"\n✅ COMPREHENSIVE ANALYSIS COMPLETE!")

if __name__ == "__main__":
    main()