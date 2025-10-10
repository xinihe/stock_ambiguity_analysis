#!/usr/bin/env python3
"""
Risk-Return Relationship Diagnostic Analysis
Investigating why risk shows no relationship with returns in the regression
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load the enhanced monthly data"""
    data_path = Path("/Users/tlxy/Research/Ambiguity/code/Adaptive/GeoUncertainty/outputs/results/enhanced_monthly_data_all_bins.csv")
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    print(f"✅ Data loaded successfully: {len(df)} observations")
    print(f"📅 Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    return df

def examine_variable_properties(df):
    """Examine the properties of risk and return variables"""
    print("\n" + "="*80)
    print("VARIABLE PROPERTIES ANALYSIS")
    print("="*80)
    
    # Key variables to examine
    variables = ['Monthly_Return_Pct', 'Risk_Bins_20', 'Ambiguity_Bins_20', 'Risk_Metric']
    
    for var in variables:
        if var in df.columns:
            print(f"\n📊 {var}:")
            print(f"   Count: {df[var].count()}")
            print(f"   Mean: {df[var].mean():.6f}")
            print(f"   Std: {df[var].std():.6f}")
            print(f"   Min: {df[var].min():.6f}")
            print(f"   Max: {df[var].max():.6f}")
            print(f"   Skewness: {stats.skew(df[var].dropna()):.4f}")
            print(f"   Kurtosis: {stats.kurtosis(df[var].dropna()):.4f}")
            print(f"   Unique values: {df[var].nunique()}")
            
            # Check for extreme values
            q99 = df[var].quantile(0.99)
            q01 = df[var].quantile(0.01)
            outliers = ((df[var] > q99) | (df[var] < q01)).sum()
            print(f"   Potential outliers (>99th or <1st percentile): {outliers}")
        else:
            print(f"\n❌ {var}: Variable not found in dataset")

def analyze_correlations(df):
    """Analyze correlations between variables"""
    print("\n" + "="*80)
    print("CORRELATION ANALYSIS")
    print("="*80)
    
    # Select relevant variables
    corr_vars = ['Monthly_Return_Pct', 'Risk_Bins_20', 'Ambiguity_Bins_20', 'Risk_Metric']
    available_vars = [var for var in corr_vars if var in df.columns]
    
    if len(available_vars) < 2:
        print("❌ Insufficient variables for correlation analysis")
        return
    
    # Calculate correlation matrix
    corr_matrix = df[available_vars].corr()
    
    print(f"\n📈 CORRELATION MATRIX:")
    print(corr_matrix.round(4))
    
    # Focus on return correlations
    if 'Monthly_Return_Pct' in available_vars:
        print(f"\n🎯 CORRELATIONS WITH MONTHLY RETURNS:")
        return_corrs = corr_matrix['Monthly_Return_Pct'].drop('Monthly_Return_Pct')
        for var, corr in return_corrs.items():
            print(f"   {var}: {corr:.4f}")
            
            # Statistical significance test
            valid_data = df[[var, 'Monthly_Return_Pct']].dropna()
            if len(valid_data) > 2:
                corr_stat, p_value = stats.pearsonr(valid_data[var], valid_data['Monthly_Return_Pct'])
                significance = "***" if p_value < 0.01 else "**" if p_value < 0.05 else "*" if p_value < 0.10 else ""
                print(f"     (p-value: {p_value:.4f}) {significance}")

def investigate_binning_process(df):
    """Investigate how the binning process affects the risk variable"""
    print("\n" + "="*80)
    print("BINNING PROCESS INVESTIGATION")
    print("="*80)
    
    if 'Risk_Metric' not in df.columns or 'Risk_Bins_20' not in df.columns:
        print("❌ Required variables for binning analysis not found")
        return
    
    # Examine the relationship between original risk metric and binned version
    print(f"\n🔍 ORIGINAL RISK METRIC vs BINNED VERSION:")
    
    # Basic statistics
    print(f"   Original Risk_Metric:")
    print(f"     Range: {df['Risk_Metric'].min():.6f} to {df['Risk_Metric'].max():.6f}")
    print(f"     Unique values: {df['Risk_Metric'].nunique()}")
    
    print(f"   Binned Risk_Bins_20:")
    print(f"     Range: {df['Risk_Bins_20'].min():.6f} to {df['Risk_Bins_20'].max():.6f}")
    print(f"     Unique values: {df['Risk_Bins_20'].nunique()}")
    
    # Check if binning preserved the relationship
    corr_original = df[['Risk_Metric', 'Monthly_Return_Pct']].corr().iloc[0,1]
    corr_binned = df[['Risk_Bins_20', 'Monthly_Return_Pct']].corr().iloc[0,1]
    
    print(f"\n📊 CORRELATION COMPARISON:")
    print(f"   Risk_Metric vs Returns: {corr_original:.6f}")
    print(f"   Risk_Bins_20 vs Returns: {corr_binned:.6f}")
    print(f"   Information loss from binning: {abs(corr_original - corr_binned):.6f}")
    
    # Examine bin distribution
    print(f"\n📈 BIN DISTRIBUTION:")
    bin_counts = df['Risk_Bins_20'].value_counts().sort_index()
    print(f"   Number of bins with data: {len(bin_counts)}")
    print(f"   Min observations per bin: {bin_counts.min()}")
    print(f"   Max observations per bin: {bin_counts.max()}")
    print(f"   Average observations per bin: {bin_counts.mean():.2f}")

def test_alternative_specifications(df):
    """Test alternative model specifications"""
    print("\n" + "="*80)
    print("ALTERNATIVE MODEL SPECIFICATIONS")
    print("="*80)
    
    # Prepare data
    y = df['Monthly_Return_Pct'].dropna()
    
    models_to_test = []
    
    # Model 1: Original specification
    if 'Risk_Bins_20' in df.columns and 'Ambiguity_Bins_20' in df.columns:
        X1 = df[['Risk_Bins_20', 'Ambiguity_Bins_20']].dropna()
        common_idx = y.index.intersection(X1.index)
        if len(common_idx) > 10:
            models_to_test.append(("Original (Binned)", X1.loc[common_idx], y.loc[common_idx]))
    
    # Model 2: Original risk metric (unbinned)
    if 'Risk_Metric' in df.columns:
        X2 = df[['Risk_Metric']].dropna()
        common_idx = y.index.intersection(X2.index)
        if len(common_idx) > 10:
            models_to_test.append(("Original Risk Metric", X2.loc[common_idx], y.loc[common_idx]))
    
    # Model 3: Log-transformed variables
    if 'Risk_Metric' in df.columns:
        # Create log-transformed risk (add small constant to avoid log(0))
        risk_log = np.log(df['Risk_Metric'] + 1e-8)
        X3 = pd.DataFrame({'Risk_Log': risk_log}).dropna()
        common_idx = y.index.intersection(X3.index)
        if len(common_idx) > 10:
            models_to_test.append(("Log Risk Metric", X3.loc[common_idx], y.loc[common_idx]))
    
    # Model 4: Standardized variables
    if 'Risk_Metric' in df.columns:
        risk_std = (df['Risk_Metric'] - df['Risk_Metric'].mean()) / df['Risk_Metric'].std()
        X4 = pd.DataFrame({'Risk_Standardized': risk_std}).dropna()
        common_idx = y.index.intersection(X4.index)
        if len(common_idx) > 10:
            models_to_test.append(("Standardized Risk", X4.loc[common_idx], y.loc[common_idx]))
    
    # Run models
    results = []
    for model_name, X, y_subset in models_to_test:
        try:
            X_with_const = sm.add_constant(X)
            model = sm.OLS(y_subset, X_with_const).fit()
            
            results.append({
                'Model': model_name,
                'R_squared': model.rsquared,
                'Adj_R_squared': model.rsquared_adj,
                'F_pvalue': model.f_pvalue,
                'N_obs': int(model.nobs),
                'Risk_coef': model.params.iloc[1] if len(model.params) > 1 else np.nan,
                'Risk_pvalue': model.pvalues.iloc[1] if len(model.pvalues) > 1 else np.nan
            })
            
        except Exception as e:
            print(f"❌ Error in {model_name}: {str(e)}")
    
    # Display results
    if results:
        results_df = pd.DataFrame(results)
        print(f"\n📊 MODEL COMPARISON:")
        print(results_df.round(4))
        
        # Find best performing model
        best_model = results_df.loc[results_df['Adj_R_squared'].idxmax()]
        print(f"\n🏆 BEST PERFORMING MODEL: {best_model['Model']}")
        print(f"   Adjusted R²: {best_model['Adj_R_squared']:.4f}")
        print(f"   Risk coefficient: {best_model['Risk_coef']:.4f} (p={best_model['Risk_pvalue']:.4f})")

def diagnose_data_issues(df):
    """Diagnose potential data quality issues"""
    print("\n" + "="*80)
    print("DATA QUALITY DIAGNOSIS")
    print("="*80)
    
    # Check for missing values
    print(f"\n🔍 MISSING VALUES:")
    key_vars = ['Monthly_Return_Pct', 'Risk_Bins_20', 'Ambiguity_Bins_20', 'Risk_Metric']
    for var in key_vars:
        if var in df.columns:
            missing = df[var].isna().sum()
            missing_pct = (missing / len(df)) * 100
            print(f"   {var}: {missing} ({missing_pct:.1f}%)")
    
    # Check for constant or near-constant variables
    print(f"\n📊 VARIABLE VARIATION:")
    for var in key_vars:
        if var in df.columns:
            unique_vals = df[var].nunique()
            total_vals = df[var].count()
            variation_ratio = unique_vals / total_vals if total_vals > 0 else 0
            print(f"   {var}: {unique_vals} unique values out of {total_vals} ({variation_ratio:.3f} variation ratio)")
            
            if variation_ratio < 0.1:
                print(f"     ⚠️  WARNING: Low variation detected!")
    
    # Check for extreme outliers
    print(f"\n🎯 EXTREME VALUES ANALYSIS:")
    for var in ['Monthly_Return_Pct', 'Risk_Metric']:
        if var in df.columns:
            Q1 = df[var].quantile(0.25)
            Q3 = df[var].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            extreme_outliers = ((df[var] < lower_bound) | (df[var] > upper_bound)).sum()
            print(f"   {var}: {extreme_outliers} extreme outliers (beyond 3*IQR)")
            
            if extreme_outliers > 0:
                outlier_values = df[var][(df[var] < lower_bound) | (df[var] > upper_bound)]
                print(f"     Range: {outlier_values.min():.6f} to {outlier_values.max():.6f}")

def generate_diagnostic_summary():
    """Generate summary of findings and recommendations"""
    print("\n" + "="*80)
    print("DIAGNOSTIC SUMMARY & RECOMMENDATIONS")
    print("="*80)
    
    print(f"\n🔍 POTENTIAL ISSUES IDENTIFIED:")
    print(f"   1. Binning Process: Converting continuous risk to discrete bins may lose information")
    print(f"   2. Scale Mismatch: Risk and return variables may have very different scales")
    print(f"   3. Data Quality: Outliers or missing values may affect relationships")
    print(f"   4. Model Specification: Linear relationship assumption may not hold")
    
    print(f"\n💡 RECOMMENDATIONS:")
    print(f"   1. Use original continuous risk metric instead of binned version")
    print(f"   2. Standardize variables to comparable scales")
    print(f"   3. Consider log transformations for skewed variables")
    print(f"   4. Test non-linear specifications (polynomial, interaction terms)")
    print(f"   5. Examine time-varying relationships (rolling correlations)")
    print(f"   6. Consider alternative risk measures (volatility, VaR, etc.)")

def main():
    """Main diagnostic function"""
    try:
        # Load data
        df = load_data()
        
        # Run diagnostic analyses
        examine_variable_properties(df)
        analyze_correlations(df)
        investigate_binning_process(df)
        test_alternative_specifications(df)
        diagnose_data_issues(df)
        generate_diagnostic_summary()
        
        print("\n" + "="*80)
        print("DIAGNOSTIC ANALYSIS COMPLETE")
        print("="*80)
        print("\n✅ Risk-return relationship diagnostic completed!")
        
    except Exception as e:
        print(f"❌ Error in diagnostic analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()