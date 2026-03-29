"""
Causal Analysis Module for Chinese Energy Market
Implements 2SLS, Difference-in-Differences, and Mechanism Tests
Designed for the paper: "Pricing the Unknown: Ambiguity Premiums in China's Green vs. Brown Energy Markets"

This module implements causal inference methods from Section 3.4 of causal_ambi_china.tex:
1. Two-Stage Least Squares (2SLS) with instrumental variables
2. Difference-in-Differences around policy shocks
3. Mediation analysis for liquidity channel
4. Moderation analysis for regime dependence
5. Granger causality tests

Author: Research Team
Date: 2024
Paper Reference: causal_ambi_china.tex
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.sandbox.regression.gmm import IV2SLS
from linearmodels.panel import PanelOLS, IV2SLS as PanelIV2SLS
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class CausalAmbiguityAnalysis:
    """
    Causal analysis of ambiguity effects on Chinese energy stock returns

    Implements the empirical research design from Section 3 of causal_ambi_china.tex,
    including baseline OLS, 2SLS with multiple instruments, DiD around policy shocks,
    and mechanism tests for mediation and moderation.
    """

    def __init__(self, ambiguity_df, returns_df, controls_df, energy_classification):
        """
        Initialize the causal analysis

        Parameters:
        -----------
        ambiguity_df : pandas DataFrame
            A_CEA_t values (stocks × dates)
        returns_df : pandas DataFrame
            Future returns r_{t+1} (stocks × dates)
        controls_df : dict of pandas DataFrames
            Dictionary containing control variables (RV, Skewness, Kurtosis, Turnover)
        energy_classification : dict
            Dictionary mapping stock_id to 'Brown', 'Green', or 'Grey'
        """
        self.ambiguity_df = ambiguity_df
        self.returns_df = returns_df
        self.controls_df = controls_df
        self.energy_classification = energy_classification

        # Merge data
        self.data = self._prepare_data()

    def _prepare_data(self):
        """
        Prepare data for analysis by merging ambiguity, returns, and controls

        Returns:
        --------
        data : pandas DataFrame
            Long-format data suitable for panel analysis
        """
        # Stack ambiguity
        ambiguity_long = self.ambiguity_df.stack().reset_index()
        ambiguity_long.columns = ['date', 'stock_id', 'ambiguity']

        # Stack returns
        returns_long = self.returns_df.stack().reset_index()
        returns_long.columns = ['date', 'stock_id', 'future_return']

        # Merge
        data = pd.merge(ambiguity_long, returns_long, on=['date', 'stock_id'])

        # Add Green indicator
        data['Green'] = data['stock_id'].map(
            lambda x: 1 if self.energy_classification.get(x) == 'Green' else 0
        )

        # Add ambiguity × Green interaction
        data['ambiguity_x_Green'] = data['ambiguity'] * data['Green']

        # Add controls
        for control_name, control_df in self.controls_df.items():
            control_long = control_df.stack().reset_index()
            control_long.columns = ['date', 'stock_id', control_name]
            data = pd.merge(data, control_long, on=['date', 'stock_id'], how='left')

        # Drop missing values
        data = data.dropna()

        # Create stock and date fixed effects
        data['stock_fe'] = data['stock_id'].astype('category').cat.codes
        data['date_fe'] = data['date'].astype('category').cat.codes

        return data

    def baseline_panel_ols(self):
        """
        Run baseline panel regression with fixed effects

        Implements Equation (10) from causal_ambi_china.tex:
        r_{i,t+1} = α + β₁A^{CEA}_{i,t} + β₂(A^{CEA}_{i,t} × Green_i) + γ'Controls + FE + ε

        Returns:
        --------
        results : dict
            Dictionary containing regression results
        """
        # Prepare data
        X = self.data[['ambiguity', 'ambiguity_x_Green', 'RV', 'Skewness', 'Kurtosis', 'Turnover']]
        X = sm.add_constant(X)
        y = self.data['future_return']

        # OLS regression with clustered standard errors
        model = sm.OLS(y, X)
        results = model.fit(cov_type='cluster', cov_kwds={'groups': self.data['stock_id']})

        # Extract results
        output = {
            'coefficients': results.params,
            'std_errors': results.bse,
            't_stats': results.tvalues,
            'p_values': results.pvalues,
            'r_squared': results.rsquared,
            'adj_r_squared': results.rsquared_adj,
            'n_obs': len(y),
            'summary': results.summary()
        }

        return output

    def instrumental_variables_2sls(self, instruments_data):
        """
        Perform 2SLS estimation using instrumental variables

        Implements Equations (11)-(12) from causal_ambi_china.tex with three instruments:
        1. Peer-based ambiguity (leave-one-out sector average)
        2. EPU × Policy Sensitivity interaction
        3. Geopolitical ambiguity (PC1 of defense and gold CEA)

        Parameters:
        -----------
        instruments_data : dict
            Dictionary containing instrumental variables DataFrames
            Keys: 'peer_ambiguity', 'epu_interaction', 'geo_ambiguity'

        Returns:
        --------
        results : dict
            Dictionary containing first-stage and second-stage results
        """
        # Merge instruments into data
        for inst_name, inst_df in instruments_data.items():
            inst_long = inst_df.stack().reset_index()
            inst_long.columns = ['date', 'stock_id', inst_name]
            self.data = pd.merge(self.data, inst_long, on=['date', 'stock_id'], how='left')

        # First stage: regress ambiguity on instruments and controls
        X_first = self.data[['peer_ambiguity', 'epu_interaction', 'geo_ambiguity',
                             'RV', 'Skewness', 'Kurtosis', 'Turnover']]
        X_first = sm.add_constant(X_first)
        y_first = self.data['ambiguity']

        # First-stage regression
        first_stage = sm.OLS(y_first, X_first)
        first_results = first_stage.fit(cov_type='cluster',
                                        cov_kwds={'groups': self.data['stock_id']})

        # Get predicted ambiguity
        self.data['ambiguity_predicted'] = first_results.fittedvalues
        self.data['ambiguity_predicted_x_Green'] = self.data['ambiguity_predicted'] * self.data['Green']

        # Check instrument relevance (F-statistic)
        f_stat = first_results.fvalue
        r_squared_first = first_results.rsquared

        # Second stage: regress returns on predicted ambiguity
        X_second = self.data[['ambiguity_predicted', 'ambiguity_predicted_x_Green',
                              'RV', 'Skewness', 'Kurtosis', 'Turnover']]
        X_second = sm.add_constant(X_second)
        y_second = self.data['future_return']

        second_stage = sm.OLS(y_second, X_second)
        second_results = second_stage.fit(cov_type='cluster',
                                         cov_kwds={'groups': self.data['stock_id']})

        # Compile results
        output = {
            'first_stage': {
                'coefficients': first_results.params,
                'std_errors': first_results.bse,
                'f_statistic': f_stat,
                'r_squared': r_squared_first,
                'summary': first_results.summary()
            },
            'second_stage': {
                'coefficients': second_results.params,
                'std_errors': second_results.bse,
                't_stats': second_results.tvalues,
                'p_values': second_results.pvalues,
                'r_squared': second_results.rsquared,
                'summary': second_results.summary()
            },
            'causal_effect': {
                'coefficient': second_results.params['ambiguity_predicted'],
                'std_error': second_results.bse['ambiguity_predicted'],
                't_stat': second_results.tvalues['ambiguity_predicted'],
                'p_value': second_results.pvalues['ambiguity_predicted']
            },
            'brown_green_differential': {
                'coefficient': second_results.params['ambiguity_predicted_x_Green'],
                'std_error': second_results.bse['ambiguity_predicted_x_Green'],
                't_stat': second_results.tvalues['ambiguity_predicted_x_Green'],
                'p_value': second_results.pvalues['ambiguity_predicted_x_Green']
            }
        }

        return output

    def difference_in_differences(self, policy_shock_dates, window_days=30):
        """
        Perform Difference-in-Differences analysis around policy shocks

        Implements Equation (13) from causal_ambi_china.tex:
        r_{i,t} = α + β₁Post_t + β₂Green_i + β₃(Post_t × Green_i) + Controls + FE + ε

        Parameters:
        -----------
        policy_shock_dates : list of str
            List of policy announcement dates (e.g., ['2020-09-22', '2021-03-15'])
        window_days : int
            Window around policy shock for analysis (default: 30 days)

        Returns:
        --------
        results : dict
            Dictionary containing DiD results for each policy shock
        """
        results = {}

        for shock_date in policy_shock_dates:
            shock_date = pd.Timestamp(shock_date)

            # Define event window
            pre_start = shock_date - pd.Timedelta(days=window_days)
            post_end = shock_date + pd.Timedelta(days=window_days)

            # Filter data to event window
            event_data = self.data[
                (self.data['date'] >= pre_start) &
                (self.data['date'] <= post_end)
            ].copy()

            # Create Post indicator
            event_data['Post'] = (event_data['date'] > shock_date).astype(int)

            # DiD regression
            X = event_data[['Post', 'Green', 'RV', 'Skewness', 'Kurtosis', 'Turnover']]
            X['Post_x_Green'] = event_data['Post'] * event_data['Green']
            X = sm.add_constant(X)
            y = event_data['future_return']

            model = sm.OLS(y, X)
            did_results = model.fit(cov_type='cluster',
                                   cov_kwds={'groups': event_data['stock_id']})

            # CEA reduction test
            pre_green_ambiguity = event_data[
                (event_data['Post'] == 0) & (event_data['Green'] == 1)
            ]['ambiguity'].mean()

            pre_brown_ambiguity = event_data[
                (event_data['Post'] == 0) & (event_data['Green'] == 0)
            ]['ambiguity'].mean()

            post_green_ambiguity = event_data[
                (event_data['Post'] == 1) & (event_data['Green'] == 1)
            ]['ambiguity'].mean()

            post_brown_ambiguity = event_data[
                (event_data['Post'] == 1) & (event_data['Green'] == 0)
            ]['ambiguity'].mean()

            results[shock_date] = {
                'regression': {
                    'coefficients': did_results.params,
                    'std_errors': did_results.bse,
                    't_stats': did_results.tvalues,
                    'p_values': did_results.pvalues,
                    'summary': did_results.summary()
                },
                'did_coefficient': did_results.params['Post_x_Green'],
                'did_p_value': did_results.pvalues['Post_x_Green'],
                'cea_reduction': {
                    'green_diff': pre_green_ambiguity - post_green_ambiguity,
                    'brown_diff': pre_brown_ambiguity - post_brown_ambiguity,
                    'differential': (pre_green_ambiguity - post_green_ambiguity) -
                                  (pre_brown_ambiguity - post_brown_ambiguity)
                }
            }

        return results

    def mediation_analysis(self, mediator='Spread', n_bootstrap=10000):
        """
        Perform mediation analysis to test indirect effects through liquidity

        Implements Equations (14)-(15) from causal_ambi_china.tex:
        Path a: Spread_{i,t} = a₀ + a₁A^{CEA}_{i,t} + Controls + ν_{i,t}
        Path b: r_{i,t+1} = b₀ + b₁Spread_{i,t} + b₂A^{CEA}_{i,t} + Controls + ξ_{i,t+1}

        Parameters:
        -----------
        mediator : str
            Name of mediator variable (default: 'Spread')
        n_bootstrap : int
            Number of bootstrap replications (default: 10,000)

        Returns:
        --------
        results : dict
            Dictionary containing mediation analysis results
        """
        # Path a: Ambiguity -> Mediator
        X_a = self.data[['ambiguity', 'RV', 'Skewness', 'Kurtosis', 'Turnover']]
        X_a = sm.add_constant(X_a)
        y_a = self.data[mediator]

        model_a = sm.OLS(y_a, X_a)
        results_a = model_a.fit(cov_type='cluster',
                               cov_kwds={'groups': self.data['stock_id']})

        a_coeff = results_a.params['ambiguity']
        a_se = results_a.bse['ambiguity']

        # Path b: Mediator -> Return (controlling for Ambiguity)
        X_b = self.data[['ambiguity', mediator, 'RV', 'Skewness', 'Kurtosis', 'Turnover']]
        X_b = sm.add_constant(X_b)
        y_b = self.data['future_return']

        model_b = sm.OLS(y_b, X_b)
        results_b = model_b.fit(cov_type='cluster',
                               cov_kwds={'groups': self.data['stock_id']})

        b_coeff = results_b.params[mediator]
        b_se = results_b.bse[mediator]

        # Direct effect
        direct_coeff = results_b.params['ambiguity']
        direct_se = results_b.bse['ambiguity']

        # Indirect effect (a × b)
        indirect_coeff = a_coeff * b_coeff

        # Bootstrap standard error for indirect effect
        bootstrap_indirect = []
        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(len(self.data), size=len(self.data), replace=True)
            data_boot = self.data.iloc[indices]

            try:
                # Path a
                X_a_boot = sm.add_constant(
                    data_boot[['ambiguity', 'RV', 'Skewness', 'Kurtosis', 'Turnover']]
                )
                y_a_boot = data_boot[mediator]
                model_a_boot = sm.OLS(y_a_boot, X_a_boot).fit()
                a_boot = model_a_boot.params['ambiguity']

                # Path b
                X_b_boot = sm.add_constant(
                    data_boot[['ambiguity', mediator, 'RV', 'Skewness', 'Kurtosis', 'Turnover']]
                )
                y_b_boot = data_boot['future_return']
                model_b_boot = sm.OLS(y_b_boot, X_b_boot).fit()
                b_boot = model_b_boot.params[mediator]

                bootstrap_indirect.append(a_boot * b_boot)
            except:
                continue

        indirect_se = np.std(bootstrap_indirect)
        indirect_ci_lower = np.percentile(bootstrap_indirect, 2.5)
        indirect_ci_upper = np.percentile(bootstrap_indirect, 97.5)

        # Sobel test
        sobel_statistic = (a_coeff * b_coeff) / np.sqrt(
            b_coeff**2 * a_se**2 + a_coeff**2 * b_se**2
        )
        sobel_pvalue = 2 * (1 - stats.norm.cdf(abs(sobel_statistic)))

        # Total effect
        total_coeff = direct_coeff + indirect_coeff

        # Proportion mediated
        proportion_mediated = indirect_coeff / total_coeff if total_coeff != 0 else np.nan

        results = {
            'path_a': {
                'coefficient': a_coeff,
                'std_error': a_se,
                't_stat': results_a.tvalues['ambiguity'],
                'p_value': results_a.pvalues['ambiguity']
            },
            'path_b': {
                'coefficient': b_coeff,
                'std_error': b_se,
                't_stat': results_b.tvalues[mediator],
                'p_value': results_b.pvalues[mediator]
            },
            'direct_effect': {
                'coefficient': direct_coeff,
                'std_error': direct_se,
                't_stat': results_b.tvalues['ambiguity'],
                'p_value': results_b.pvalues['ambiguity']
            },
            'indirect_effect': {
                'coefficient': indirect_coeff,
                'std_error': indirect_se,
                'bootstrap_ci_lower': indirect_ci_lower,
                'bootstrap_ci_upper': indirect_ci_upper
            },
            'sobel_test': {
                'statistic': sobel_statistic,
                'p_value': sobel_pvalue
            },
            'total_effect': {
                'coefficient': total_coeff
            },
            'proportion_mediated': proportion_mediated
        }

        return results

    def moderation_analysis(self, regime_variable, regime_threshold=None):
        """
        Test heterogeneous effects across different regimes

        Implements Equation (16) from causal_ambi_china.tex:
        r_{i,t+1} = α + β₁A^{CEA}_{i,t} + β₂(A^{CEA}_{i,t} × Regime) + Controls + FE + ε

        Parameters:
        -----------
        regime_variable : str
            Variable defining regimes (e.g., 'RV' for volatility)
        regime_threshold : float or None
            Threshold for splitting regimes (None uses median split)

        Returns:
        --------
        results : dict
            Dictionary containing subgroup analysis results
        """
        if regime_threshold is None:
            regime_threshold = self.data[regime_variable].median()

        # Create regime indicator
        self.data[f'{regime_variable}_High'] = (
            self.data[regime_variable] > regime_threshold
        ).astype(int)

        # Create interaction term
        self.data[f'ambiguity_x_{regime_variable}_High'] = (
            self.data['ambiguity'] * self.data[f'{regime_variable}_High']
        )

        # Moderation regression
        X = self.data[['ambiguity', f'ambiguity_x_{regime_variable}_High',
                       f'{regime_variable}_High', 'RV', 'Skewness', 'Kurtosis', 'Turnover']]
        X = sm.add_constant(X)
        y = self.data['future_return']

        model = sm.OLS(y, X)
        moderation_results = model.fit(cov_type='cluster',
                                       cov_kwds={'groups': self.data['stock_id']})

        # Split sample analysis
        low_regime = self.data[self.data[f'{regime_variable}_High'] == 0]
        high_regime = self.data[self.data[f'{regime_variable}_High'] == 1]

        results = {
            'interaction': {
                'coefficient': moderation_results.params[f'ambiguity_x_{regime_variable}_High'],
                'std_error': moderation_results.bse[f'ambiguity_x_{regime_variable}_High'],
                't_stat': moderation_results.tvalues[f'ambiguity_x_{regime_variable}_High'],
                'p_value': moderation_results.pvalues[f'ambiguity_x_{regime_variable}_High']
            },
            'low_regime': self._estimate_regime_model(low_regime),
            'high_regime': self._estimate_regime_model(high_regime),
            'threshold': regime_threshold
        }

        return results

    def _estimate_regime_model(self, regime_data):
        """Estimate model for a specific regime"""
        if len(regime_data) < 50:  # Minimum observations
            return None

        X = regime_data[['ambiguity', 'RV', 'Skewness', 'Kurtosis', 'Turnover']]
        X = sm.add_constant(X)
        y = regime_data['future_return']

        model = sm.OLS(y, X)
        results = model.fit(cov_type='cluster', cov_kwds={'groups': regime_data['stock_id']})

        return {
            'ambiguity_coefficient': results.params['ambiguity'],
            'std_error': results.bse['ambiguity'],
            't_stat': results.tvalues['ambiguity'],
            'p_value': results.pvalues['ambiguity'],
            'n_obs': len(regime_data)
        }

    def granger_causality_test(self, max_lag=5, significance_level=0.05):
        """
        Perform Granger causality test between ambiguity and returns

        Tests whether ambiguity Granger-causes returns at various lags

        Parameters:
        -----------
        max_lag : int
            Maximum lag to test (default: 5)
        significance_level : float
            Significance level for the test (default: 0.05)

        Returns:
        --------
        results : dict
            Dictionary containing Granger causality test results
        """
        from statsmodels.tsa.stattools import grangercausalitytests

        # Prepare time series data (average across stocks)
        ambiguity_ts = self.ambiguity_df.mean(axis=1)
        returns_ts = self.returns_df.mean(axis=1)

        # Align
        data = pd.DataFrame({'ambiguity': ambiguity_ts, 'returns': returns_ts}).dropna()

        results = {}
        for lag in range(1, max_lag + 1):
            # Test: Ambiguity Granger-causes Returns
            gc_result = grangercausalitytests(
                data[['returns', 'ambiguity']],
                maxlag=lag, verbose=False
            )

            # Extract F-test p-value
            f_pvalue = gc_result[lag][0]['ssr_ftest'][1]

            # Test: Returns Granger-causes Ambiguity
            gc_result_reverse = grangercausalitytests(
                data[['ambiguity', 'returns']],
                maxlag=lag, verbose=False
            )

            f_pvalue_reverse = gc_result_reverse[lag][0]['ssr_ftest'][1]

            results[lag] = {
                'ambiguity_causes_returns': f_pvalue < significance_level,
                'f_pvalue': f_pvalue,
                'returns_cause_ambiguity': f_pvalue_reverse < significance_level,
                'f_pvalue_reverse': f_pvalue_reverse,
                'causal_strength': 1 - f_pvalue,
                'information_ratio': np.log(1 / f_pvalue) if f_pvalue > 0 else np.inf
            }

        return results


def generate_instrumental_variables(ambiguity_df, epu_series, policy_sensitivity_df,
                                   geo_ambiguity_df, sector_mapping):
    """
    Generate instrumental variables for causal analysis

    Implements the three instruments from Section 3.4.2 of causal_ambi_china.tex:
    1. Peer-based ambiguity (leave-one-out sector average)
    2. EPU × Policy Sensitivity interaction
    3. Geopolitical ambiguity (already computed)

    Parameters:
    -----------
    ambiguity_df : pandas DataFrame
        A_CEA_t values
    epu_series : pandas Series
        China Economic Policy Uncertainty Index
    policy_sensitivity_df : pandas DataFrame
        Firm-level policy sensitivity (stocks × dates)
    geo_ambiguity_df : pandas DataFrame
        Geopolitical ambiguity from defense and gold CEA
    sector_mapping : dict
        Mapping from stock_id to sector/industry

    Returns:
    --------
    instruments : dict of pandas DataFrames
        Dictionary containing instrumental variables
    """
    from ambiguity_measurement import compute_peer_ambiguity

    instruments = {}

    # 1. Peer-based ambiguity
    instruments['peer_ambiguity'] = compute_peer_ambiguity(ambiguity_df, sector_mapping)

    # 2. EPU × Policy Sensitivity interaction
    epu_aligned = epu_series.reindex(ambiguity_df.index, method='ffill')
    epu_interaction = policy_sensitivity_df.mul(epu_aligned, axis=0)
    instruments['epu_interaction'] = epu_interaction

    # 3. Geopolitical ambiguity
    instruments['geo_ambiguity'] = geo_ambiguity_df

    return instruments


if __name__ == "__main__":
    # Example usage
    print("Causal Analysis Module for Chinese Energy Market")
    print("=" * 70)

    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range('2018-01-01', '2024-05-24', freq='D')
    n_stocks = 50

    # Sample data
    ambiguity_data = np.random.randn(len(dates), n_stocks) * 0.1 + 0.2
    returns_data = np.random.randn(len(dates), n_stocks) * 0.01

    ambiguity_df = pd.DataFrame(ambiguity_data, index=dates,
                                columns=[f'Stock_{i}' for i in range(n_stocks)])
    returns_df = pd.DataFrame(returns_data, index=dates,
                              columns=[f'Stock_{i}' for i in range(n_stocks)])

    # Energy classification
    energy_classification = {
        f'Stock_{i}': 'Brown' if i < 25 else 'Green'
        for i in range(n_stocks)
    }

    # Sample controls
    controls_df = {
        'RV': pd.DataFrame(np.random.rand(len(dates), n_stocks) * 0.02,
                          index=dates, columns=returns_df.columns),
        'Skewness': pd.DataFrame(np.random.randn(len(dates), n_stocks) * 0.3,
                                index=dates, columns=returns_df.columns),
        'Kurtosis': pd.DataFrame(np.random.rand(len(dates), n_stocks) * 5,
                                index=dates, columns=returns_df.columns),
        'Turnover': pd.DataFrame(np.random.rand(len(dates), n_stocks) * 0.05,
                                index=dates, columns=returns_df.columns)
    }

    # Initialize analysis
    analysis = CausalAmbiguityAnalysis(
        ambiguity_df, returns_df, controls_df, energy_classification
    )

    # Run baseline OLS
    print("\n1. Baseline Panel OLS Results:")
    baseline_results = analysis.baseline_panel_ols()
    print(f"   Ambiguity coefficient: {baseline_results['coefficients']['ambiguity']:.6f}")
    print(f"   t-statistic: {baseline_results['t_stats']['ambiguity']:.4f}")
    print(f"   R-squared: {baseline_results['r_squared']:.4f}")

    # Granger causality test
    print("\n2. Granger Causality Test:")
    gc_results = analysis.granger_causality_test(max_lag=3)
    for lag, result in gc_results.items():
        print(f"   Lag {lag}: p-value = {result['f_pvalue']:.4f}")

    # Mediation analysis
    print("\n3. Mediation Analysis (Liquidity Channel):")
    mediation_results = analysis.mediation_analysis(mediator='Turnover')
    print(f"   Indirect effect: {mediation_results['indirect_effect']['coefficient']:.6f}")
    print(f"   Sobel test p-value: {mediation_results['sobel_test']['p_value']:.4f}")
    print(f"   Proportion mediated: {mediation_results['proportion_mediated']:.2%}")

    # Moderation analysis
    print("\n4. Moderation Analysis (by RV):")
    moderation_results = analysis.moderation_analysis(regime_variable='RV')
    print(f"   Low RV regime: {moderation_results['low_regime']['ambiguity_coefficient']:.6f}")
    print(f"   High RV regime: {moderation_results['high_regime']['ambiguity_coefficient']:.6f}")

    print("\n" + "=" * 70)
    print("Example usage complete!")
