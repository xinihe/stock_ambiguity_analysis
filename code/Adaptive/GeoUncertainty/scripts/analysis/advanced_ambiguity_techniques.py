#!/usr/bin/env python3
"""
Advanced Ambiguity Techniques
=============================

This script implements advanced econometric and statistical techniques to reveal
potential ambiguity effects on returns that may be hidden by traditional methods.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV, RidgeCV
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load and prepare data."""
    df = pd.read_csv('outputs/results/enhanced_monthly_data_all_bins.csv')
    key_vars = ['Monthly_Return_Pct', 'Ambiguity_Bins_20', 'Risk_Metric']
    df_clean = df[key_vars].dropna()
    return df_clean

def threshold_model_analysis(df):
    """Implement threshold models to detect regime-dependent ambiguity effects."""
    print("="*80)
    print("THRESHOLD MODEL ANALYSIS")
    print("="*80)
    
    returns = df['Monthly_Return_Pct'].values
    ambiguity = df['Ambiguity_Bins_20'].values
    risk = df['Risk_Metric'].values
    
    # Test different threshold variables
    threshold_vars = {
        'Risk_Threshold': risk,
        'Ambiguity_Threshold': ambiguity,
        'Returns_Threshold': returns
    }
    
    best_model = None
    best_r2 = -np.inf
    best_threshold = None
    
    for threshold_name, threshold_var in threshold_vars.items():
        print(f"\n🎯 Testing {threshold_name}:")
        
        # Try different percentiles as thresholds
        percentiles = [25, 33, 50, 67, 75]
        
        for pct in percentiles:
            threshold_value = np.percentile(threshold_var, pct)
            regime = (threshold_var > threshold_value).astype(int)
            
            # Create regime-dependent model
            X = pd.DataFrame({
                'Ambiguity': ambiguity,
                'Risk': risk,
                'Regime': regime,
                'Ambiguity_Regime': ambiguity * regime,
                'Risk_Regime': risk * regime
            })
            X_const = sm.add_constant(X)
            
            try:
                model = sm.OLS(returns, X_const).fit()
                
                # Check if ambiguity or ambiguity_regime is significant
                ambiguity_pval = model.pvalues.get('Ambiguity', 1.0)
                ambiguity_regime_pval = model.pvalues.get('Ambiguity_Regime', 1.0)
                
                min_ambiguity_pval = min(ambiguity_pval, ambiguity_regime_pval)
                
                if model.rsquared_adj > best_r2:
                    best_r2 = model.rsquared_adj
                    best_model = model
                    best_threshold = f"{threshold_name}_{pct}th_percentile"
                
                if min_ambiguity_pval < 0.1:
                    print(f"   ✅ {pct}th percentile: Ambiguity p={ambiguity_pval:.4f}, Regime*Ambiguity p={ambiguity_regime_pval:.4f}")
                    print(f"      R²={model.rsquared:.4f}, Adj R²={model.rsquared_adj:.4f}")
                
            except Exception as e:
                continue
    
    if best_model is not None:
        print(f"\n🏆 BEST THRESHOLD MODEL: {best_threshold}")
        print(f"   R²: {best_model.rsquared:.4f} | Adj R²: {best_model.rsquared_adj:.4f}")
        
        for var in ['Ambiguity', 'Ambiguity_Regime']:
            if var in best_model.params.index:
                coef = best_model.params[var]
                pval = best_model.pvalues[var]
                stars = get_significance_stars(pval)
                print(f"   {var}: {coef:.6f} (p={pval:.4f}){stars}")
    
    return best_model

def polynomial_and_spline_analysis(df):
    """Test polynomial and spline-like transformations."""
    print(f"\n" + "="*80)
    print("POLYNOMIAL AND SPLINE ANALYSIS")
    print("="*80)
    
    returns = df['Monthly_Return_Pct'].values
    ambiguity = df['Ambiguity_Bins_20'].values
    risk = df['Risk_Metric'].values
    
    # Polynomial features
    print(f"\n🔢 POLYNOMIAL FEATURES:")
    
    # Create polynomial features for ambiguity
    poly = PolynomialFeatures(degree=3, include_bias=False)
    ambiguity_poly = poly.fit_transform(ambiguity.reshape(-1, 1))
    
    # Test different polynomial degrees
    for degree in [2, 3]:
        poly_deg = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly = poly_deg.fit_transform(np.column_stack([ambiguity, risk]))
        
        # Get feature names
        feature_names = poly_deg.get_feature_names_out(['Ambiguity', 'Risk'])
        
        X_poly_df = pd.DataFrame(X_poly, columns=feature_names)
        X_poly_const = sm.add_constant(X_poly_df)
        
        try:
            model_poly = sm.OLS(returns, X_poly_const).fit()
            
            print(f"\n   Degree {degree} Polynomial:")
            print(f"   R²: {model_poly.rsquared:.4f} | Adj R²: {model_poly.rsquared_adj:.4f}")
            
            # Check significance of ambiguity terms
            for var in feature_names:
                if 'ambiguity' in var.lower():
                    coef = model_poly.params[var]
                    pval = model_poly.pvalues[var]
                    stars = get_significance_stars(pval)
                    print(f"   {var}: {coef:.6f} (p={pval:.4f}){stars}")
                    
        except Exception as e:
            print(f"   Degree {degree}: Failed - {str(e)}")
    
    # Piecewise linear (spline-like) analysis
    print(f"\n📊 PIECEWISE LINEAR ANALYSIS:")
    
    # Create knots at different percentiles
    knot_percentiles = [25, 50, 75]
    
    for knot_pct in knot_percentiles:
        knot = np.percentile(ambiguity, knot_pct)
        
        # Create piecewise terms
        ambiguity_below = np.minimum(ambiguity, knot)
        ambiguity_above = np.maximum(ambiguity - knot, 0)
        
        X_spline = pd.DataFrame({
            'Ambiguity_Below_Knot': ambiguity_below,
            'Ambiguity_Above_Knot': ambiguity_above,
            'Risk': risk
        })
        X_spline_const = sm.add_constant(X_spline)
        
        try:
            model_spline = sm.OLS(returns, X_spline_const).fit()
            
            print(f"\n   Knot at {knot_pct}th percentile (value={knot:.4f}):")
            print(f"   R²: {model_spline.rsquared:.4f} | Adj R²: {model_spline.rsquared_adj:.4f}")
            
            for var in ['Ambiguity_Below_Knot', 'Ambiguity_Above_Knot']:
                coef = model_spline.params[var]
                pval = model_spline.pvalues[var]
                stars = get_significance_stars(pval)
                print(f"   {var}: {coef:.6f} (p={pval:.4f}){stars}")
                
        except Exception as e:
            print(f"   Knot {knot_pct}%: Failed - {str(e)}")

def machine_learning_feature_importance(df):
    """Use machine learning to assess ambiguity importance."""
    print(f"\n" + "="*80)
    print("MACHINE LEARNING FEATURE IMPORTANCE")
    print("="*80)
    
    returns = df['Monthly_Return_Pct'].values
    ambiguity = df['Ambiguity_Bins_20'].values
    risk = df['Risk_Metric'].values
    
    # Create extended feature set
    features = pd.DataFrame({
        'Ambiguity': ambiguity,
        'Risk': risk,
        'Ambiguity_Squared': ambiguity**2,
        'Risk_Squared': risk**2,
        'Ambiguity_Risk_Interaction': ambiguity * risk,
        'Ambiguity_Log': np.log(ambiguity + 1e-6),
        'Risk_Log': np.log(risk + 1e-6),
        'Ambiguity_Sqrt': np.sqrt(ambiguity - ambiguity.min() + 1e-6),
        'Risk_Sqrt': np.sqrt(risk - risk.min() + 1e-6)
    })
    
    # Random Forest
    print(f"\n🌲 RANDOM FOREST ANALYSIS:")
    rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5)
    rf.fit(features, returns)
    
    # Feature importance
    importance_df = pd.DataFrame({
        'Feature': features.columns,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print(f"   Feature Importance Rankings:")
    for i, (_, row) in enumerate(importance_df.iterrows()):
        print(f"   {i+1:2d}. {row['Feature']:25s}: {row['Importance']:.4f}")
    
    # Check if any ambiguity features are in top 3
    top_3_features = importance_df.head(3)['Feature'].tolist()
    ambiguity_in_top3 = any('ambiguity' in feat.lower() for feat in top_3_features)
    
    if ambiguity_in_top3:
        print(f"   ✅ Ambiguity feature(s) in top 3!")
    else:
        print(f"   ⚠️  No ambiguity features in top 3")
    
    # LASSO regularization
    print(f"\n🎯 LASSO REGULARIZATION:")
    lasso = LassoCV(cv=5, random_state=42, max_iter=1000)
    lasso.fit(features, returns)
    
    # Check which features survive LASSO
    surviving_features = features.columns[lasso.coef_ != 0]
    ambiguity_surviving = [feat for feat in surviving_features if 'ambiguity' in feat.lower()]
    
    print(f"   Features surviving LASSO regularization:")
    for feat in surviving_features:
        coef = lasso.coef_[features.columns.get_loc(feat)]
        print(f"   {feat:25s}: {coef:.6f}")
    
    if ambiguity_surviving:
        print(f"   ✅ Ambiguity features surviving: {ambiguity_surviving}")
    else:
        print(f"   ⚠️  No ambiguity features survived LASSO")
    
    return {
        'rf_importance': importance_df,
        'lasso_coef': lasso.coef_,
        'surviving_features': surviving_features
    }

def instrumental_variable_approach(df):
    """Attempt instrumental variable approach for ambiguity."""
    print(f"\n" + "="*80)
    print("INSTRUMENTAL VARIABLE APPROACH")
    print("="*80)
    
    returns = df['Monthly_Return_Pct'].values
    ambiguity = df['Ambiguity_Bins_20'].values
    risk = df['Risk_Metric'].values
    
    # Create potential instruments (lagged values)
    n = len(df)
    
    # Use lagged risk as instrument for current ambiguity
    if n > 5:  # Need enough observations
        print(f"\n🔧 USING LAGGED RISK AS INSTRUMENT:")
        
        # Create lagged variables
        lag_periods = [1, 2, 3]
        
        for lag in lag_periods:
            if n > lag + 10:  # Ensure enough observations
                # Create lagged instrument
                risk_lag = np.roll(risk, lag)
                
                # Remove first 'lag' observations
                returns_iv = returns[lag:]
                ambiguity_iv = ambiguity[lag:]
                risk_iv = risk[lag:]
                risk_lag_iv = risk_lag[lag:]
                
                # First stage: regress ambiguity on instrument
                X_first = sm.add_constant(risk_lag_iv)
                first_stage = sm.OLS(ambiguity_iv, X_first).fit()
                
                # Get predicted ambiguity
                ambiguity_predicted = first_stage.fittedvalues
                
                # Second stage: regress returns on predicted ambiguity
                X_second = pd.DataFrame({
                    'Ambiguity_Predicted': ambiguity_predicted,
                    'Risk': risk_iv
                })
                X_second_const = sm.add_constant(X_second)
                second_stage = sm.OLS(returns_iv, X_second_const).fit()
                
                print(f"\n   Lag {lag} periods:")
                print(f"   First stage R²: {first_stage.rsquared:.4f}")
                print(f"   Second stage R²: {second_stage.rsquared:.4f}")
                
                # Check significance
                ambiguity_coef = second_stage.params['Ambiguity_Predicted']
                ambiguity_pval = second_stage.pvalues['Ambiguity_Predicted']
                stars = get_significance_stars(ambiguity_pval)
                
                print(f"   Ambiguity_Predicted: {ambiguity_coef:.6f} (p={ambiguity_pval:.4f}){stars}")
                
                # Weak instrument test
                f_stat = first_stage.fvalue
                print(f"   F-statistic (weak instrument test): {f_stat:.2f}")
                if f_stat > 10:
                    print(f"   ✅ Strong instrument (F > 10)")
                else:
                    print(f"   ⚠️  Weak instrument (F < 10)")

def volatility_clustering_analysis(df):
    """Analyze ambiguity effects conditional on volatility clustering."""
    print(f"\n" + "="*80)
    print("VOLATILITY CLUSTERING ANALYSIS")
    print("="*80)
    
    returns = df['Monthly_Return_Pct'].values
    ambiguity = df['Ambiguity_Bins_20'].values
    risk = df['Risk_Metric'].values
    
    # Calculate rolling volatility
    window_sizes = [3, 6, 12]
    
    for window in window_sizes:
        if len(returns) > window + 5:
            print(f"\n📊 {window}-MONTH ROLLING VOLATILITY:")
            
            # Calculate rolling standard deviation
            returns_series = pd.Series(returns)
            rolling_vol = returns_series.rolling(window=window).std()
            
            # Remove NaN values
            valid_idx = ~rolling_vol.isna()
            returns_vol = returns[valid_idx]
            ambiguity_vol = ambiguity[valid_idx]
            risk_vol = risk[valid_idx]
            rolling_vol_clean = rolling_vol[valid_idx].values
            
            # Create high/low volatility regimes
            vol_median = np.median(rolling_vol_clean)
            high_vol_regime = (rolling_vol_clean > vol_median).astype(int)
            
            # Model with volatility regime interaction
            X_vol = pd.DataFrame({
                'Ambiguity': ambiguity_vol,
                'Risk': risk_vol,
                'High_Vol_Regime': high_vol_regime,
                'Ambiguity_High_Vol': ambiguity_vol * high_vol_regime,
                'Rolling_Vol': rolling_vol_clean
            })
            X_vol_const = sm.add_constant(X_vol)
            
            try:
                model_vol = sm.OLS(returns_vol, X_vol_const).fit()
                
                print(f"   R²: {model_vol.rsquared:.4f} | Adj R²: {model_vol.rsquared_adj:.4f}")
                
                # Check ambiguity effects
                for var in ['Ambiguity', 'Ambiguity_High_Vol']:
                    if var in model_vol.params.index:
                        coef = model_vol.params[var]
                        pval = model_vol.pvalues[var]
                        stars = get_significance_stars(pval)
                        print(f"   {var}: {coef:.6f} (p={pval:.4f}){stars}")
                        
            except Exception as e:
                print(f"   Failed: {str(e)}")

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

def main():
    """Main analysis function."""
    print("ADVANCED AMBIGUITY TECHNIQUES")
    print("=" * 80)
    
    # Load data
    df = load_data()
    print(f"Data loaded: {len(df)} observations")
    
    # Run advanced analyses
    threshold_model_analysis(df)
    polynomial_and_spline_analysis(df)
    ml_results = machine_learning_feature_importance(df)
    instrumental_variable_approach(df)
    volatility_clustering_analysis(df)
    
    print(f"\n" + "="*80)
    print("ADVANCED ANALYSIS COMPLETE")
    print("="*80)
    print(f"\n💡 If no significant effects found, consider:")
    print(f"   • Different data frequency (daily/weekly)")
    print(f"   • Alternative ambiguity measures")
    print(f"   • Sector or size-specific analysis")
    print(f"   • Crisis period analysis")
    print(f"   • Cross-sectional rather than time-series analysis")

if __name__ == "__main__":
    main()