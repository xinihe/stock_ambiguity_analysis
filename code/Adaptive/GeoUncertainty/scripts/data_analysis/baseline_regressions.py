#!/usr/bin/env python3
"""
Baseline Regression Analysis for Geopolitical Risk and Ambiguity Paper
Implements the methodology described in geopoliticalAmb02.tex
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

def load_sample_data():
    """Load the sample data generated earlier"""
    data_dir = "/Users/tlxy/Research/Ambiguity/code/Adaptive/GeoUncertainty/outputs/data"

    market_df = pd.read_csv(os.path.join(data_dir, "market_data_sample.csv"), parse_dates=['datetime'], index_col='datetime')
    returns_df = pd.read_csv(os.path.join(data_dir, "firm_returns_sample.csv"), parse_dates=['date'])

    return market_df, returns_df

def prepare_regression_data(market_df):
    """Prepare data for time-series regressions"""

    # Drop missing values
    df = market_df.dropna()

    # Create lagged variables
    df['returns_lag1'] = df['returns'].shift(1)
    df['gpr_lag1'] = df['gpr_index'].shift(1)
    df['ambiguity_lag1'] = df['ambiguity_ce'].shift(1)

    # Create volatility change
    df['vol_change'] = df['realized_vol'].diff()

    # Create log volume
    df['log_volume'] = np.log(df['SSE.000300.volume'])

    # Standardize variables for easier interpretation
    variables_to_standardize = ['returns', 'ambiguity_ce', 'gpr_index', 'realized_vol', 'log_volume']
    for var in variables_to_standardize:
        df[f'{var}_std'] = (df[var] - df[var].mean()) / df[var].std()

    return df.dropna()

def run_time_series_regression(df, dependent_var, independent_vars, regression_name=""):
    """Run time-series regression and return results"""

    y = df[dependent_var]
    X = df[independent_vars]
    X = sm.add_constant(X)

    model = sm.OLS(y, X, missing='drop')
    results = model.fit(cov_type='HAC', cov_kwds={'maxlags': 5})  # Newey-West standard errors

    # Calculate additional statistics
    dw_stat = durbin_watson(results.resid)

    print(f"\n=== {regression_name} ===")
    print(f"Dependent Variable: {dependent_var}")
    print(f"Independent Variables: {', '.join(independent_vars)}")
    print(f"\nR-squared: {results.rsquared:.4f}")
    print(f"Adjusted R-squared: {results.rsquared_adj:.4f}")
    print(f"F-statistic: {results.fvalue:.4f} (p-value: {results.f_pvalue:.4f})")
    print(f"Durbin-Watson: {dw_stat:.4f}")
    print(f"Observations: {int(results.nobs)}")

    print("\nCoefficients:")
    print(results.summary().tables[1])

    return results

def test_h1_gpr_to_ambiguity(df):
    """Test H1: GPR increases ambiguity"""

    print("\n" + "="*60)
    print("TESTING HYPOTHESIS H1: GPR increases ambiguity")
    print("="*60)

    # Model 1: GPR and ambiguity
    independent_vars = ['gpr_index_std', 'returns_std', 'realized_vol_std']
    results = run_time_series_regression(
        df, 'ambiguity_ce_std', independent_vars,
        "GPR Impact on Ambiguity (Model 1)"
    )

    # Model 2: Include lagged GPR
    independent_vars_lag = ['gpr_index_std', 'gpr_lag1', 'returns_std', 'realized_vol_std']
    results_lag = run_time_series_regression(
        df, 'ambiguity_ce_std', independent_vars_lag,
        "GPR Impact on Ambiguity with Lag (Model 2)"
    )

    return results, results_lag

def test_h2_ambiguity_pricing(df):
    """Test H2: Ambiguity is negatively priced"""

    print("\n" + "="*60)
    print("TESTING HYPOTHESIS H2: Ambiguity is negatively priced")
    print("="*60)

    # Model 1: Market factors only
    independent_vars_mkt = ['returns_lag1', 'log_volume_std']
    results_mkt = run_time_series_regression(
        df, 'returns_std', independent_vars_mkt,
        "Baseline Market Model (Model 1)"
    )

    # Model 2: Add GPR
    independent_vars_gpr = ['gpr_index_std', 'returns_lag1', 'log_volume_std']
    results_gpr = run_time_series_regression(
        df, 'returns_std', independent_vars_gpr,
        "Market Model + GPR (Model 2)"
    )

    # Model 3: Add ambiguity
    independent_vars_full = ['gpr_index_std', 'ambiguity_ce_std', 'realized_vol_std',
                           'returns_lag1', 'log_volume_std']
    results_full = run_time_series_regression(
        df, 'returns_std', independent_vars_full,
        "Full Model with Ambiguity (Model 3)"
    )

    return results_mkt, results_gpr, results_full

def calculate_fama_macbeth_betas(returns_df, market_df):
    """Calculate betas for Fama-MacBeth analysis"""

    # Merge firm returns with market factors
    returns_df['date'] = pd.to_datetime(returns_df['date'])
    merged_df = returns_df.merge(market_df[['returns', 'ambiguity_ce', 'realized_vol']],
                                left_on='date', right_index=True, how='inner')

    # Calculate factor changes (innovations)
    market_df['ambiguity_change'] = market_df['ambiguity_ce'].diff()
    market_df['vol_change'] = market_df['realized_vol'].diff()

    # Merge with factor changes
    merged_df = merged_df.merge(market_df[['ambiguity_change', 'vol_change']],
                                left_on='date', right_index=True, how='inner')

    # Calculate betas for each firm
    beta_results = []

    for firm_id in merged_df['firm_id'].unique():
        firm_data = merged_df[merged_df['firm_id'] == firm_id].copy()

        if len(firm_data) > 30:  # Minimum observations for reliable beta estimation
            # Prepare data
            firm_data = firm_data.dropna()

            if len(firm_data) > 10:
                y = firm_data['return']
                X = firm_data[['returns', 'ambiguity_change', 'vol_change']]
                X = sm.add_constant(X)

                try:
                    model = sm.OLS(y, X)
                    results = model.fit()

                    beta_results.append({
                        'firm_id': firm_id,
                        'const_beta': results.params['const'],
                        'market_beta': results.params['returns'],
                        'ambiguity_beta': results.params['ambiguity_change'],
                        'volatility_beta': results.params['vol_change'],
                        'r_squared': results.rsquared,
                        'n_obs': len(firm_data)
                    })
                except:
                    continue

    beta_df = pd.DataFrame(beta_results)
    print(f"Calculated betas for {len(beta_df)} firms")

    return beta_df

def run_fama_macbeth_regression(beta_df, returns_df):
    """Run Fama-MacBeth cross-sectional regression"""

    print("\n" + "="*60)
    print("FAMA-MACBETH CROSS-SECTIONAL ANALYSIS")
    print("="*60)

    # Prepare cross-sectional data
    cs_data = beta_df.merge(returns_df.groupby('firm_id')[['return', 'is_soe', 'market_cap',
                                                           'book_to_market', 'leverage']].mean(),
                           on='firm_id', how='inner')

    # Drop rows with missing betas
    cs_data = cs_data.dropna(subset=['ambiguity_beta', 'volatility_beta'])

    print(f"Cross-sectional sample size: {len(cs_data)} firms")

    # Model 1: Market beta only
    y = cs_data['return']
    X1 = cs_data[['market_beta']]
    X1 = sm.add_constant(X1)
    model1 = sm.OLS(y, X1)
    results1 = model1.fit()

    # Model 2: Add ambiguity beta
    X2 = cs_data[['market_beta', 'ambiguity_beta']]
    X2 = sm.add_constant(X2)
    model2 = sm.OLS(y, X2)
    results2 = model2.fit()

    # Model 3: Add volatility beta
    X3 = cs_data[['market_beta', 'ambiguity_beta', 'volatility_beta']]
    X3 = sm.add_constant(X3)
    model3 = sm.OLS(y, X3)
    results3 = model3.fit()

    # Model 4: Add firm characteristics
    X4 = cs_data[['market_beta', 'ambiguity_beta', 'volatility_beta',
                  'is_soe', 'market_cap', 'book_to_market', 'leverage']]
    X4 = sm.add_constant(X4)
    model4 = sm.OLS(y, X4)
    results4 = model4.fit()

    # Print results
    models = [results1, results2, results3, results4]
    model_names = ["Market Beta Only", "Add Ambiguity Beta", "Add Volatility Beta", "Full Model"]

    for i, (model, name) in enumerate(zip(models, model_names), 1):
        print(f"\n--- Cross-Sectional Model {i}: {name} ---")
        print(f"R-squared: {model.rsquared:.4f}")
        print("Coefficients:")
        print(model.summary().tables[1])

    return results2, results4  # Return key models for interpretation

def test_moderation_effect(beta_df, returns_df):
    """Test H4: SOE status moderates ambiguity pricing"""

    print("\n" + "="*60)
    print("TESTING HYPOTHESIS H4: SOE Moderation Effect")
    print("="*60)

    # Prepare data with interaction term
    cs_data = beta_df.merge(returns_df.groupby('firm_id')[['return', 'is_soe', 'market_cap',
                                                           'book_to_market']].mean(),
                           on='firm_id', how='inner')
    cs_data = cs_data.dropna(subset=['ambiguity_beta'])

    # Create interaction term
    cs_data['ambiguity_beta_x_soe'] = cs_data['ambiguity_beta'] * cs_data['is_soe']

    print(f"Sample size for moderation analysis: {len(cs_data)} firms")
    print(f"SOE firms: {cs_data['is_soe'].sum()} ({cs_data['is_soe'].mean():.1%})")
    print(f"Non-SOE firms: {(1-cs_data['is_soe']).sum()} ({(1-cs_data['is_soe']).mean():.1%})")

    # Model with interaction
    y = cs_data['return']
    X = cs_data[['ambiguity_beta', 'ambiguity_beta_x_soe', 'market_beta', 'is_soe',
                 'market_cap', 'book_to_market']]
    X = sm.add_constant(X)
    model = sm.OLS(y, X)
    results = model.fit()

    print("\n--- Moderation Model Results ---")
    print(f"R-squared: {results.rsquared:.4f}")
    print("Coefficients:")
    print(results.summary().tables[1])

    # Interpret interaction effect
    amb_coef = results.params['ambiguity_beta']
    int_coef = results.params['ambiguity_beta_x_soe']

    print(f"\nInterpretation:")
    print(f"Ambiguity beta for Non-SOEs: {amb_coef:.6f}")
    print(f"Ambiguity beta for SOEs: {amb_coef + int_coef:.6f}")
    print(f"Difference (SOE - Non-SOE): {int_coef:.6f}")

    # Subsample analysis
    soe_data = cs_data[cs_data['is_soe'] == 1]
    non_soe_data = cs_data[cs_data['is_soe'] == 0]

    if len(soe_data) > 20 and len(non_soe_data) > 20:
        # SOE subsample
        y_soe = soe_data['return']
        X_soe = soe_data[['ambiguity_beta', 'market_beta', 'market_cap', 'book_to_market']]
        X_soe = sm.add_constant(X_soe)
        model_soe = sm.OLS(y_soe, X_soe)
        results_soe = model_soe.fit()

        # Non-SOE subsample
        y_non = non_soe_data['return']
        X_non = non_soe_data[['ambiguity_beta', 'market_beta', 'market_cap', 'book_to_market']]
        X_non = sm.add_constant(X_non)
        model_non = sm.OLS(y_non, X_non)
        results_non = model_non.fit()

        print(f"\n--- Subsample Analysis ---")
        print(f"SOE sample: Ambiguity beta = {results_soe.params['ambiguity_beta']:.6f}")
        print(f"Non-SOE sample: Ambiguity beta = {results_non.params['ambiguity_beta']:.6f}")

    return results

def create_regression_results_summary(results_dict):
    """Create a summary table of all regression results"""

    summary_data = []

    for name, result in results_dict.items():
        for var, coef in result.params.items():
            summary_data.append({
                'Model': name,
                'Variable': var,
                'Coefficient': coef,
                'Std Error': result.bse[var],
                't-stat': result.tvalues[var],
                'p-value': result.pvalues[var],
                'Significance': '***' if result.pvalues[var] < 0.001 else
                               '**' if result.pvalues[var] < 0.01 else
                               '*' if result.pvalues[var] < 0.05 else ''
            })

    summary_df = pd.DataFrame(summary_data)
    return summary_df

def save_regression_results(results_dict, summary_df, output_dir):
    """Save regression results to files"""

    # Save individual regression results
    for name, result in results_dict.items():
        filename = f"regression_{name.lower().replace(' ', '_')}.csv"
        # Convert results summary to DataFrame and save
        result_df = pd.read_html(result.summary().tables[1].as_html(), header=0, index_col=0)[0]
        result_df.to_csv(os.path.join(output_dir, filename))

    # Save summary table
    summary_df.to_csv(os.path.join(output_dir, "regression_summary.csv"), index=False)

    print(f"\nRegression results saved to {output_dir}")

def create_visualizations(market_df, beta_df, output_dir):
    """Create visualization of key results"""

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Plot 1: Time series of returns and ambiguity
    ax1 = axes[0, 0]
    sample_data = market_df.tail(100)  # Last 100 days
    ax1_twin = ax1.twinx()
    ax1.plot(sample_data.index, sample_data['returns'] * 100, 'b-', alpha=0.7, label='Returns (%)')
    ax1_twin.plot(sample_data.index, sample_data['ambiguity_ce'], 'r-', alpha=0.7, label='Ambiguity')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Returns (%)', color='b')
    ax1_twin.set_ylabel('Ambiguity Index', color='r')
    ax1.set_title('Returns and Ambiguity (Sample Period)')
    ax1.grid(True, alpha=0.3)

    # Plot 2: GPR vs Ambiguity scatter
    ax2 = axes[0, 1]
    ax2.scatter(market_df['gpr_index'], market_df['ambiguity_ce'], alpha=0.6, s=20)
    z = np.polyfit(market_df['gpr_index'], market_df['ambiguity_ce'], 1)
    p = np.poly1d(z)
    ax2.plot(market_df['gpr_index'], p(market_df['gpr_index']), "r--", alpha=0.8)
    ax2.set_xlabel('GPR Index')
    ax2.set_ylabel('Ambiguity Index')
    ax2.set_title('GPR-Ambiguity Relationship')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Distribution of ambiguity betas
    ax3 = axes[1, 0]
    ax3.hist(beta_df['ambiguity_beta'], bins=30, alpha=0.7, edgecolor='black')
    ax3.axvline(beta_df['ambiguity_beta'].mean(), color='red', linestyle='--',
                label=f'Mean: {beta_df["ambiguity_beta"].mean():.4f}')
    ax3.set_xlabel('Ambiguity Beta')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Distribution of Ambiguity Betas Across Firms')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Ambiguity beta by SOE status
    ax4 = axes[1, 1]
    soe_mask = returns_df.groupby('firm_id')['is_soe'].first() == 1
    beta_with_soe = beta_df.copy()
    beta_with_soe['is_soe'] = soe_mask.reindex(beta_df['firm_id']).values

    soe_betas = beta_with_soe[beta_with_soe['is_soe'] == 1]['ambiguity_beta']
    non_soe_betas = beta_with_soe[beta_with_soe['is_soe'] == 0]['ambiguity_beta']

    ax4.hist(non_soe_betas, bins=20, alpha=0.7, label='Non-SOEs', color='blue')
    ax4.hist(soe_betas, bins=20, alpha=0.7, label='SOEs', color='red')
    ax4.set_xlabel('Ambiguity Beta')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Ambiguity Beta by Ownership Type')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "regression_results_visualization.png"),
                dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function to run all baseline regressions"""

    print("Running Baseline Regression Analysis...")
    print("="*60)

    # Load data
    print("Loading sample data...")
    market_df, returns_df = load_sample_data()

    # Prepare data
    print("Preparing data for regression analysis...")
    df = prepare_regression_data(market_df)

    # Store all results
    results_dict = {}

    # Test H1: GPR increases ambiguity
    results_h1_1, results_h1_2 = test_h1_gpr_to_ambiguity(df)
    results_dict['H1_GPR_to_Ambiguity'] = results_h1_2

    # Test H2: Ambiguity pricing
    results_h2_1, results_h2_2, results_h2_3 = test_h2_ambiguity_pricing(df)
    results_dict['H2_Ambiguity_Pricing'] = results_h2_3

    # Fama-MacBeth analysis
    print("\n" + "="*60)
    print("CALCULATING FIRM-SPECIFIC BETAS FOR FAMA-MACBETH ANALYSIS...")
    beta_df = calculate_fama_macbeth_betas(returns_df, market_df)

    results_fmb_1, results_fmb_2 = run_fama_macbeth_regression(beta_df, returns_df)
    results_dict['Fama_MacBeth_Full'] = results_fmb_2

    # Test H4: SOE moderation
    results_h4 = test_moderation_effect(beta_df, returns_df)
    results_dict['H4_SOE_Moderation'] = results_h4

    # Create summary
    summary_df = create_regression_results_summary(results_dict)

    # Save results
    output_dir = "/Users/tlxy/Research/Ambiguity/code/Adaptive/GeoUncertainty/outputs/results"
    os.makedirs(output_dir, exist_ok=True)
    save_regression_results(results_dict, summary_df, output_dir)

    # Create visualizations
    create_visualizations(market_df, beta_df, output_dir)

    # Print summary
    print("\n" + "="*60)
    print("REGRESSION ANALYSIS SUMMARY")
    print("="*60)

    print(f"\nH1 (GPR increases ambiguity):")
    print(f"  GPR coefficient on ambiguity: {results_h1_2.params['gpr_index_std']:.4f}")
    print(f"  t-statistic: {results_h1_2.tvalues['gpr_index_std']:.2f}")
    print(f"  p-value: {results_h1_2.pvalues['gpr_index_std']:.4f}")

    print(f"\nH2 (Ambiguity negatively priced):")
    print(f"  Ambiguity coefficient on returns: {results_h2_3.params['ambiguity_ce_std']:.4f}")
    print(f"  t-statistic: {results_h2_3.tvalues['ambiguity_ce_std']:.2f}")
    print(f"  p-value: {results_h2_3.pvalues['ambiguity_ce_std']:.4f}")

    print(f"\nFama-MacBeth cross-sectional results:")
    print(f"  Ambiguity beta risk premium: {results_fmb_2.params['ambiguity_beta']:.6f}")
    print(f"  t-statistic: {results_fmb_2.tvalues['ambiguity_beta']:.2f}")

    print(f"\nH4 (SOE moderation):")
    print(f"  Ambiguity beta (Non-SOEs): {results_h4.params['ambiguity_beta']:.6f}")
    print(f"  Interaction term: {results_h4.params['ambiguity_beta_x_soe']:.6f}")
    print(f"  t-statistic (interaction): {results_h4.tvalues['ambiguity_beta_x_soe']:.2f}")

    print(f"\nTotal regressions run: {len(results_dict)}")
    print(f"Results saved to: {output_dir}")

    print("\nRegression analysis complete!")

if __name__ == "__main__":
    main()