#!/usr/bin/env python3
"""
Extract and summarize key findings from the risk-return diagnostic analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

def main():
    """Extract key diagnostic findings."""
    
    print("=" * 80)
    print("RISK-RETURN DIAGNOSTIC FINDINGS SUMMARY")
    print("=" * 80)
    
    # Load the data
    data_path = project_root / "outputs" / "results" / "enhanced_monthly_data_all_bins.csv"
    df = pd.read_csv(data_path)
    
    # Key variables
    returns = df['Monthly_Return_Pct'].dropna()
    risk_bins = df['Risk_Bins_20'].dropna()
    risk_metric = df['Risk_Metric'].dropna()
    
    print("\n🔍 KEY FINDINGS:")
    print("-" * 50)
    
    # 1. Correlation Analysis
    corr_bins = returns.corr(risk_bins)
    corr_metric = returns.corr(risk_metric)
    
    print(f"1. CORRELATION WITH RETURNS:")
    print(f"   • Risk_Bins_20 vs Returns: {corr_bins:.4f}")
    print(f"   • Risk_Metric vs Returns: {corr_metric:.4f}")
    
    # 2. Scale Analysis
    print(f"\n2. SCALE ANALYSIS:")
    print(f"   • Returns: Mean={returns.mean():.4f}, Std={returns.std():.4f}")
    print(f"   • Risk_Bins_20: Mean={risk_bins.mean():.6f}, Std={risk_bins.std():.6f}")
    print(f"   • Risk_Metric: Mean={risk_metric.mean():.6f}, Std={risk_metric.std():.6f}")
    
    # 3. Distribution Analysis
    print(f"\n3. DISTRIBUTION PROPERTIES:")
    print(f"   • Risk_Bins_20 Skewness: {risk_bins.skew():.4f} (highly right-skewed)")
    print(f"   • Risk_Metric Skewness: {risk_metric.skew():.4f}")
    print(f"   • Returns Skewness: {returns.skew():.4f}")
    
    # 4. Range Analysis
    print(f"\n4. RANGE ANALYSIS:")
    print(f"   • Risk_Bins_20 Range: {risk_bins.min():.6f} to {risk_bins.max():.6f}")
    print(f"   • Risk_Metric Range: {risk_metric.min():.6f} to {risk_metric.max():.6f}")
    print(f"   • Returns Range: {returns.min():.2f}% to {returns.max():.2f}%")
    
    # 5. Statistical Significance Test
    from scipy.stats import pearsonr
    
    corr_bins_stat, p_bins = pearsonr(returns, risk_bins)
    corr_metric_stat, p_metric = pearsonr(returns, risk_metric)
    
    print(f"\n5. STATISTICAL SIGNIFICANCE:")
    print(f"   • Risk_Bins_20 vs Returns: r={corr_bins_stat:.4f}, p={p_bins:.4f}")
    print(f"   • Risk_Metric vs Returns: r={corr_metric_stat:.4f}, p={p_metric:.4f}")
    
    print("\n" + "=" * 80)
    print("PROBLEM DIAGNOSIS")
    print("=" * 80)
    
    print("\n🚨 IDENTIFIED ISSUES:")
    print("-" * 50)
    
    print("1. EXTREMELY WEAK CORRELATIONS:")
    print("   • Both risk measures show very weak correlation with returns")
    print("   • Risk_Bins_20 even shows negative correlation (-0.0222)")
    print("   • This contradicts financial theory expectations")
    
    print("\n2. SCALE MISMATCH:")
    print("   • Risk_Bins_20 values are extremely small (0.000015 to 0.000518)")
    print("   • Returns are in percentage points (-9.38% to 15.08%)")
    print("   • Massive scale difference may affect regression sensitivity")
    
    print("\n3. DISTRIBUTION ISSUES:")
    print("   • Risk_Bins_20 is highly right-skewed (skewness = 2.59)")
    print("   • This violates normal distribution assumptions in OLS")
    print("   • May require transformation or robust regression methods")
    
    print("\n4. BINNING INFORMATION LOSS:")
    print("   • Converting continuous risk to bins may lose crucial information")
    print("   • Original Risk_Metric shows slightly better correlation")
    
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    print("\n💡 IMMEDIATE ACTIONS:")
    print("-" * 50)
    
    print("1. USE ORIGINAL RISK METRIC:")
    print("   • Replace Risk_Bins_20 with Risk_Metric in regression")
    print("   • Preserves continuous information")
    
    print("\n2. STANDARDIZE VARIABLES:")
    print("   • Z-score standardization to comparable scales")
    print("   • Or use log transformations for skewed variables")
    
    print("\n3. ALTERNATIVE SPECIFICATIONS:")
    print("   • Test non-linear relationships (quadratic, cubic)")
    print("   • Consider interaction terms")
    print("   • Use robust regression methods")
    
    print("\n4. THEORETICAL VALIDATION:")
    print("   • Verify risk measure construction methodology")
    print("   • Compare with standard volatility measures")
    print("   • Check if risk measure captures intended concept")
    
    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    main()