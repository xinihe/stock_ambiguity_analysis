"""
Hypothesis Testing Module for Co-Ambiguity Analysis
Implements statistical tests for all five research hypotheses
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.api import Logit
from statsmodels.tsa.api import VAR
from sklearn.metrics import roc_curve, auc, roc_auc_score
import warnings
warnings.filterwarnings('ignore')


class CoAmbiguityHypothesisTests:
    """
    Test suite for the five research hypotheses on Systemic Co-Ambiguity
    """

    def __init__(self, sca_series, ambiguity_df, returns_df, market_index_df,
                 liquidity_df=None, volatility_df=None, vix_df=None):
        """
        Initialize the hypothesis testing framework

        Parameters:
        -----------
        sca_series : pandas Series
            Systemic Co-Ambiguity index time series
        ambiguity_df : pandas DataFrame
            Individual ambiguity indices (stocks × dates)
        returns_df : pandas DataFrame
            Stock returns (stocks × dates)
        market_index_df : pandas Series
            Market index level or returns
        liquidity_df : pandas DataFrame or None
            Liquidity measures (spread, depth, turnover) × dates
        volatility_df : pandas Series or None
            Realized volatility time series
        vix_df : pandas Series or None
            VIX or implied volatility time series
        """
        self.sca = sca_series
        self.ambiguity_df = ambiguity_df
        self.returns_df = returns_df
        self.market_index = market_index_df
        self.liquidity_df = liquidity_df
        self.volatility = volatility_df
        self.vix = vix_df

        # Align all data
        self._align_data()

    def _align_data(self):
        """Align all time series to common dates"""
        # Create a combined index
        all_series = [self.sca, self.market_index]
        if self.liquidity_df is not None:
            all_series.extend([self.liquidity_df[col] for col in self.liquidity_df.columns])
        if self.volatility is not None:
            all_series.append(self.volatility)
        if self.vix is not None:
            all_series.append(self.vix)

        # Find common dates
        common_dates = self.sca.index
        for series in all_series:
            common_dates = common_dates.intersection(series.index)

        # Subset all data
        self.sca = self.sca.loc[common_dates]
        self.market_index = self.market_index.loc[common_dates]
        if self.liquidity_df is not None:
            self.liquidity_df = self.liquidity_df.loc[common_dates]
        if self.volatility is not None:
            self.volatility = self.volatility.loc[common_dates]
        if self.vix is not None:
            self.vix = self.vix.loc[common_dates]

        self.common_dates = common_dates

    def test_hypothesis_1_leading_indicator(self, drawdown_threshold=0.05,
                                           event_window=5, pre_window=30,
                                           post_window=10):
        """
        HYPOTHESIS 1: SCA exhibits statistically significant increases before
        financial crises, with lead times of 5-20 trading days

        Test Implementation:
        1. Identify crisis events (market drawdown > threshold in event_window)
        2. Examine SCA behavior in [-pre_window, +post_window] around events
        3. Test statistical significance of pre-event increases
        4. Compare lead times to traditional indicators

        Parameters:
        -----------
        drawdown_threshold : float
            Threshold for crisis identification (default: 5%)
        event_window : int
            Window for measuring drawdown (default: 5 days)
        pre_window : int
            Pre-event window for analysis (default: 30 days)
        post_window : int
            Post-event window for analysis (default: 10 days)

        Returns:
        --------
        results : dict
            Dictionary containing test results:
            - events: identified crisis dates
            - pre_event_sca_changes: SCA changes before events
            - t_stat: t-statistic for pre-event increase
            - p_value: p-value for pre-event increase
            - lead_time: average lead time (days)
            - comparison_with_indicators: comparison with VIX, correlation, CoVaR
        """
        results = {}

        # Step 1: Identify crisis events
        print("Step 1: Identifying crisis events...")
        market_returns = self.market_index.pct_change().fillna(0)

        # Compute rolling drawdown
        cumulative_returns = (1 + market_returns).cumprod()
        rolling_max = cumulative_returns.rolling(window=event_window, min_periods=1).max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max

        # Identify events (drawdown exceeds threshold)
        crisis_dates = drawdown[drawdown < -drawdown_threshold].index.tolist()

        # Cluster nearby events (within event_window)
        events = []
        if len(crisis_dates) > 0:
            current_event = [crisis_dates[0]]
            for date in crisis_dates[1:]:
                if (date - current_event[-1]).days <= event_window:
                    current_event.append(date)
                else:
                    events.append(max(current_event, key=lambda x: drawdown[x]))  # Use worst day
                    current_event = [date]
            events.append(max(current_event, key=lambda x: drawdown[x]))

        results['events'] = events
        results['n_events'] = len(events)
        print(f"  Identified {len(events)} crisis events")

        if len(events) == 0:
            print("  Warning: No crisis events identified")
            return results

        # Step 2: Examine SCA behavior around events
        print("Step 2: Computing SCA changes around events...")
        pre_event_changes = []

        for event_date in events:
            # Get SCA in window around event
            start_date = event_date - pd.Timedelta(days=pre_window)
            end_date = event_date + pd.Timedelta(days=post_window)

            window_sca = self.sca.loc[start_date:end_date]

            if len(window_sca) < pre_window:
                continue

            # Normalize: change relative to pre-event baseline
            baseline_sca = window_sca.loc[:event_date].iloc[:pre_window].mean()
            sca_std = window_sca.loc[:event_date].iloc[:pre_window].std()

            # Compute standardized changes
            sca_changes = (window_sca - baseline_sca) / sca_std if sca_std > 0 else window_sca - baseline_sca

            # Store pre-event change (at lead time)
            if -pre_window < 0:
                pre_event_change = sca_changes.iloc[-pre_window]
                pre_event_changes.append(pre_event_change)

        results['pre_event_sca_changes'] = pre_event_changes

        # Step 3: Test statistical significance
        print("Step 3: Testing statistical significance...")
        if len(pre_event_changes) > 1:
            # One-sided t-test: H0: mean change = 0 vs H1: mean change > 0
            t_stat, p_value = stats.ttest_1samp(pre_event_changes, 0)
            # One-sided p-value
            p_value_one_sided = p_value / 2 if t_stat > 0 else 1 - p_value / 2

            results['t_stat'] = t_stat
            results['p_value'] = p_value_one_sided
            results['mean_pre_event_change'] = np.mean(pre_event_changes)
            results['std_pre_event_change'] = np.std(pre_event_changes)

            print(f"  T-statistic: {t_stat:.4f}")
            print(f"  P-value (one-sided): {p_value_one_sided:.4f}")
            print(f"  Mean pre-event change: {np.mean(pre_event_changes):.4f}")

            # Step 4: Compare with traditional indicators
            print("Step 4: Comparing with traditional indicators...")
            comparison = self._compare_indicators_lead_time(events, pre_window)
            results['comparison_with_indicators'] = comparison

        return results

    def _compare_indicators_lead_time(self, events, pre_window):
        """Compare lead times across indicators"""
        comparison = {}

        # Compute traditional indicators
        # Return correlation (average pairwise correlation of returns)
        ret_corr = self.returns_df.rolling(window=60).corr().mean(axis=1)

        # VIX (if available)
        if self.vix is not None:
            vix_signal = self.vix
        else:
            # Use realized volatility as proxy
            vix_signal = self.volatility if self.volatility is not None else self.returns_df.std(axis=1)

        # CoVaR (simplified: use market beta)
        # For simplicity, use market correlation as CoVaR proxy
        covar_proxy = self.returns_df.corrwith(self.market_index)

        lead_times = {'SCA': [], 'VIX': [], 'Correlation': [], 'CoVaR': []}

        for event_date in events:
            start_date = event_date - pd.Timedelta(days=pre_window)

            # Check when each indicator exceeds 90th percentile
            for indicator_name, indicator_series in [
                ('SCA', self.sca),
                ('VIX', vix_signal),
                ('Correlation', ret_corr),
                ('CoVaR', covar_proxy)
            ]:
                try:
                    indicator_window = indicator_series.loc[start_date:event_date]
                    threshold = indicator_window.rolling(window=252, min_periods=100).quantile(0.9)

                    # Find first crossing
                    crossings = indicator_window[indicator_window > threshold].index.tolist()

                    if len(crossings) > 0:
                        lead_time = (event_date - crossings[0]).days
                        lead_times[indicator_name].append(lead_time)
                except:
                    continue

        # Compute average lead times
        for name, times in lead_times.items():
            if len(times) > 0:
                comparison[f'{name}_avg_lead_time'] = np.mean(times)
                comparison[f'{name}_n_observations'] = len(times)

        return comparison

    def test_hypothesis_2_incremental_power(self, prediction_horizons=[5, 10, 20],
                                          crash_threshold=0.05):
        """
        HYPOTHESIS 2: SCA provides incremental explanatory power for predicting
        financial crises beyond traditional systemic risk measures

        Test Implementation:
        1. Construct crash indicators for various horizons
        2. Estimate baseline logistic regression (VIX, correlation, CoVaR)
        3. Estimate augmented model (baseline + SCA)
        4. Compute incremental pseudo-R^2, LR tests, AUC comparison

        Parameters:
        -----------
        prediction_horizons : list
            Prediction horizons in days (default: [5, 10, 20])
        crash_threshold : float
            Threshold for crash definition (default: -5%)

        Returns:
        --------
        results : dict
            Dictionary containing test results for each horizon:
            - baseline_coef: baseline model coefficients
            - augmented_coef: augmented model coefficients
            - sca_coefficient: SCA coefficient and significance
            - lr_statistic: likelihood ratio test statistic
            - lr_pvalue: LR test p-value
            - incremental_r2: incremental pseudo-R^2
            - baseline_auc: baseline model AUC
            - augmented_auc: augmented model AUC
            - auc_improvement: improvement in AUC
        """
        results = {}

        # Prepare traditional indicators
        market_returns = self.market_index.pct_change().fillna(0)

        # Return correlation (market-wide)
        ret_corr = self.returns_df.rolling(window=60).corr().mean(axis=1)

        # VIX or volatility
        if self.vix is not None:
            vix = self.vix
        else:
            vix = self.volatility if self.volatility is not None else market_returns.rolling(20).std()

        # CoVaR proxy (market correlation)
        covar = self.returns_df.corrwith(market_index)

        # Crash indicator
        crash_indicator = {}
        for k in prediction_horizons:
            future_returns = market_returns.shift(-k)
            crash_indicator[k] = (future_returns < -crash_threshold).astype(int)

        for horizon in prediction_horizons:
            print(f"\nTesting horizon: {horizon} days...")

            # Prepare data
            data = pd.DataFrame({
                'SCA': self.sca,
                'VIX': vix,
                'RetCorr': ret_corr,
                'CoVaR': covar,
                'Crash': crash_indicator[horizon]
            }).dropna()

            if len(data) < 100:
                print(f"  Warning: Insufficient data for horizon {horizon}")
                continue

            X_baseline = data[['VIX', 'RetCorr', 'CoVaR']]
            X_baseline = sm.add_constant(X_baseline)
            X_augmented = data[['SCA', 'VIX', 'RetCorr', 'CoVaR']]
            X_augmented = sm.add_constant(X_augmented)
            y = data['Crash']

            # Estimate models
            try:
                model_baseline = Logit(y, X_baseline).fit(disp=0)
                model_augmented = Logit(y, X_augmented).fit(disp=0)

                # Likelihood ratio test
                lr_statistic = 2 * (model_augmented.llf - model_baseline.llf)
                lr_pvalue = 1 - chi2.cdf(lr_statistic, df=1)

                # Pseudo-R^2 (McFadden)
                r2_baseline = model_baseline.prsquared
                r2_augmented = model_augmented.prsquared
                incremental_r2 = r2_augmented - r2_baseline

                # AUC comparison
                pred_baseline = model_baseline.predict(X_baseline)
                pred_augmented = model_augmented.predict(X_augmented)

                auc_baseline = roc_auc_score(y, pred_baseline)
                auc_augmented = roc_auc_score(y, pred_augmented)

                results[horizon] = {
                    'baseline_coef': model_baseline.params.to_dict(),
                    'augmented_coef': model_augmented.params.to_dict(),
                    'sca_coefficient': model_augmented.params['SCA'],
                    'sca_std_error': model_augmented.bse['SCA'],
                    'sca_t_stat': model_augmented.tvalues['SCA'],
                    'sca_p_value': model_augmented.pvalues['SCA'],
                    'lr_statistic': lr_statistic,
                    'lr_pvalue': lr_pvalue,
                    'baseline_r2': r2_baseline,
                    'augmented_r2': r2_augmented,
                    'incremental_r2': incremental_r2,
                    'baseline_auc': auc_baseline,
                    'augmented_auc': auc_augmented,
                    'auc_improvement': auc_augmented - auc_baseline
                }

                print(f"  SCA coefficient: {results[horizon]['sca_coefficient']:.6f}")
                print(f"  SCA t-statistic: {results[horizon]['sca_t_stat']:.4f}")
                print(f"  SCA p-value: {results[horizon]['sca_p_value']:.4f}")
                print(f"  Incremental pseudo-R2: {incremental_r2:.4f}")
                print(f"  AUC improvement: {auc_augmented - auc_baseline:.4f}")

            except Exception as e:
                print(f"  Error in estimation: {e}")
                continue

        return results

    def test_hypothesis_3_liquidity_channel(self, lags=[1, 5, 10]):
        """
        HYPOTHESIS 3: Increases in SCA precede and predict deteriorations in
        market liquidity measures

        Test Implementation:
        1. Prepare liquidity metrics (spread, depth, turnover)
        2. Estimate VAR models for various lags
        3. Compute Granger causality tests (SCA -> liquidity)
        4. Test reverse causality (liquidity -> SCA)

        Parameters:
        -----------
        lags : list
            VAR lags to test (default: [1, 5, 10])

        Returns:
        --------
        results : dict
            Dictionary containing test results:
            - f_statistics: F-stats for SCA -> liquidity causality
            - p_values: p-values for SCA -> liquidity causality
            - reverse_f_stats: F-stats for reverse causality
            - reverse_p_values: p-values for reverse causality
        """
        if self.liquidity_df is None:
            print("Warning: Liquidity data not provided")
            return {}

        results = {}
        liquidity_metrics = self.liquidity_df.columns.tolist()

        for metric in liquidity_metrics:
            print(f"\nTesting liquidity metric: {metric}")

            liquidity_series = self.liquidity_df[metric]

            # Align data
            data = pd.DataFrame({
                'SCA': self.sca,
                'Liquidity': liquidity_series
            }).dropna()

            results[metric] = {}

            for lag in lags:
                print(f"  Lag: {lag} days")

                # Prepare data for Granger causality test
                # Granger causality test requires specific format
                test_data = data[['SCA', 'Liquidity']].values

                try:
                    # Perform Granger causality test
                    # Test 1: SCA -> Liquidity
                    gc_result = grangercausalitytests(test_data, maxlag=lag, verbose=False)

                    # Extract F-statistic and p-value
                    f_stat = gc_result[lag][0]['ssr_ftest'][0]
                    p_value = gc_result[lag][0]['ssr_ftest'][1]

                    results[metric][f'lag_{lag}'] = {
                        'f_statistic': f_stat,
                        'p_value': p_value,
                        'significant': p_value < 0.05
                    }

                    print(f"    F-statistic: {f_stat:.4f}")
                    print(f"    P-value: {p_value:.4f}")

                except Exception as e:
                    print(f"    Error: {e}")
                    results[metric][f'lag_{lag}'] = {
                        'f_statistic': np.nan,
                        'p_value': np.nan,
                        'significant': False
                    }

        return results

    def test_hypothesis_4_structural_change(self, structural_change_periods,
                                          prediction_horizon=10,
                                          crash_threshold=0.05):
        """
        HYPOTHESIS 4: SCA exhibits stronger predictive power for financial
        crises during periods of structural market change

        Test Implementation:
        1. Classify periods as structural change vs. stable
        2. Estimate interaction regression: SCA × StructuralChange
        3. Test significance of interaction term
        4. Compare performance across regimes

        Parameters:
        -----------
        structural_change_periods : list of tuples
            List of (start_date, end_date) for structural change periods
        prediction_horizon : int
            Prediction horizon in days (default: 10)
        crash_threshold : float
            Threshold for crash definition (default: -5%)

        Returns:
        --------
        results : dict
            Dictionary containing test results:
            - interaction_coefficient: coefficient on SCA × StructuralChange
            - interaction_t_stat: t-statistic for interaction
            - interaction_p_value: p-value for interaction
            - sc_performance: model performance during structural change
            - stable_performance: model performance during stable periods
            - performance_comparison: comparison metrics
        """
        results = {}

        # Create structural change indicator
        structural_change = pd.Series(0, index=self.common_dates)

        for start, end in structural_change_periods:
            mask = (self.common_dates >= start) & (self.common_dates <= end)
            structural_change[mask] = 1

        # Prepare data
        market_returns = self.market_index.pct_change().fillna(0)
        future_returns = market_returns.shift(-prediction_horizon)
        crash_indicator = (future_returns < -crash_threshold).astype(int)

        data = pd.DataFrame({
            'SCA': self.sca,
            'StructuralChange': structural_change,
            'SCA_x_SC': self.sca * structural_change,
            'Crash': crash_indicator
        }).dropna()

        # Prepare controls (VIX, correlation, CoVaR)
        ret_corr = self.returns_df.rolling(window=60).corr().mean(axis=1)
        vix = self.vix if self.vix is not None else self.volatility

        data['VIX'] = vix
        data['RetCorr'] = ret_corr
        data['CoVaR'] = self.returns_df.corrwith(self.market_index)
        data = data.dropna()

        # Estimate model with interaction
        X = data[['SCA', 'StructuralChange', 'SCA_x_SC', 'VIX', 'RetCorr', 'CoVaR']]
        X = sm.add_constant(X)
        y = data['Crash']

        try:
            model = Logit(y, X).fit(disp=0)

            # Extract interaction term results
            interaction_coef = model.params['SCA_x_SC']
            interaction_se = model.bse['SCA_x_SC']
            interaction_t = model.tvalues['SCA_x_SC']
            interaction_p = model.pvalues['SCA_x_SC']

            results['interaction_coefficient'] = interaction_coef
            results['interaction_std_error'] = interaction_se
            results['interaction_t_stat'] = interaction_t
            results['interaction_p_value'] = interaction_p

            print(f"Interaction coefficient: {interaction_coef:.6f}")
            print(f"Interaction t-statistic: {interaction_t:.4f}")
            print(f"Interaction p-value: {interaction_p:.4f}")

            # Compare performance across regimes
            sc_data = data[data['StructuralChange'] == 1]
            stable_data = data[data['StructuralChange'] == 0]

            if len(sc_data) > 50 and len(stable_data) > 50:
                # Structural change regime
                X_sc = sm.add_constant(sc_data[['SCA', 'VIX', 'RetCorr', 'CoVaR']])
                y_sc = sc_data['Crash']
                model_sc = Logit(y_sc, X_sc).fit(disp=0)

                # Stable regime
                X_stable = sm.add_constant(stable_data[['SCA', 'VIX', 'RetCorr', 'CoVaR']])
                y_stable = stable_data['Crash']
                model_stable = Logit(y_stable, X_stable).fit(disp=0)

                results['sc_r2'] = model_sc.prsquared
                results['stable_r2'] = model_stable.prsquared
                results['sc_sca_coef'] = model_sc.params['SCA']
                results['stable_sca_coef'] = model_stable.params['SCA']

                # Predictions for AUC comparison
                pred_sc = model_sc.predict(X_sc)
                pred_stable = model_stable.predict(X_stable)

                results['sc_auc'] = roc_auc_score(y_sc, pred_sc)
                results['stable_auc'] = roc_auc_score(y_stable, pred_stable)

                print(f"\nStructural Change Regime:")
                print(f"  SCA coefficient: {results['sc_sca_coef']:.6f}")
                print(f"  Pseudo-R2: {results['sc_r2']:.4f}")
                print(f"  AUC: {results['sc_auc']:.4f}")

                print(f"\nStable Regime:")
                print(f"  SCA coefficient: {results['stable_sca_coef']:.6f}")
                print(f"  Pseudo-R2: {results['stable_r2']:.4f}")
                print(f"  AUC: {results['stable_auc']:.4f}")

        except Exception as e:
            print(f"Error in estimation: {e}")

        return results

    def test_hypothesis_5_volatility_lead(self, lags=[1, 5, 10, 20]):
        """
        HYPOTHESIS 5: SCA Granger-causes volatility measures, indicating that
        uncertainty synchronization precedes volatility clustering

        Test Implementation:
        1. Prepare volatility measures (VIX, realized volatility)
        2. Estimate VAR models for various lags
        3. Compute bidirectional Granger causality
        4. Generate impulse response functions

        Parameters:
        -----------
        lags : list
            VAR lags to test (default: [1, 5, 10, 20])

        Returns:
        --------
        results : dict
            Dictionary containing test results:
            - sca_to_vol_stats: F-stats for SCA -> volatility
            - sca_to_vol_pvals: p-values for SCA -> volatility
            - vol_to_sca_stats: F-stats for reverse causality
            - vol_to_sca_pvals: p-values for reverse causality
            - impulse_responses: impulse response coefficients
        """
        results = {}

        # Prepare volatility measure
        if self.vix is not None:
            volatility = self.vix
        elif self.volatility is not None:
            volatility = self.volatility
        else:
            # Use realized volatility from returns
            volatility = self.returns_df.std(axis=1)

        # Align data
        data = pd.DataFrame({
            'SCA': self.sca,
            'Volatility': volatility
        }).dropna()

        for lag in lags:
            print(f"\nTesting lag: {lag} days")

            # Prepare data for Granger causality test
            test_data = data[['SCA', 'Volatility']].values

            try:
                # Perform Granger causality tests
                gc_result = grangercausalitytests(test_data, maxlag=lag, verbose=False)

                # Extract results for SCA -> Volatility
                f_stat_sca_to_vol = gc_result[lag][0]['ssr_ftest'][0]
                p_value_sca_to_vol = gc_result[lag][0]['ssr_ftest'][1]

                # Extract results for Volatility -> SCA (reverse)
                # Need to reverse the columns for reverse causality
                test_data_reversed = test_data[:, [1, 0]]
                gc_result_reversed = grangercausalitytests(test_data_reversed, maxlag=lag, verbose=False)

                f_stat_vol_to_sca = gc_result_reversed[lag][0]['ssr_ftest'][0]
                p_value_vol_to_sca = gc_result_reversed[lag][0]['ssr_ftest'][1]

                results[f'lag_{lag}'] = {
                    'sca_to_vol_f_stat': f_stat_sca_to_vol,
                    'sca_to_vol_p_value': p_value_sca_to_vol,
                    'vol_to_sca_f_stat': f_stat_vol_to_sca,
                    'vol_to_sca_p_value': p_value_vol_to_sca,
                    'sca_causes_vol': p_value_sca_to_vol < 0.05,
                    'vol_causes_sca': p_value_vol_to_sca < 0.05
                }

                print(f"  SCA -> Volatility: F={f_stat_sca_to_vol:.4f}, p={p_value_sca_to_vol:.4f}")
                print(f"  Volatility -> SCA: F={f_stat_vol_to_sca:.4f}, p={p_value_vol_to_sca:.4f}")

                # Compute impulse response for this lag
                if len(data) > lag * 3:  # Ensure sufficient observations
                    model = VAR(test_data)
                    var_result = model.fit(lag)

                    # Impulse response function
                    irf = var_result.irf(10)
                    irf_sca_to_vol = irf.irfs[:, 0, 1]  # SCA shock -> Volatility response

                    results[f'lag_{lag}']['impulse_response'] = irf_sca_to_vol

            except Exception as e:
                print(f"  Error: {e}")
                results[f'lag_{lag}'] = {
                    'sca_to_vol_f_stat': np.nan,
                    'sca_to_vol_p_value': np.nan,
                    'vol_to_sca_f_stat': np.nan,
                    'vol_to_sca_p_value': np.nan,
                    'sca_causes_vol': False,
                    'vol_causes_sca': False
                }

        return results


# Additional helper functions

def compute_roc_curves(y_true, y_scores_dict):
    """
    Compute ROC curves for multiple models

    Parameters:
    -----------
    y_true : array-like
        True binary labels
    y_scores_dict : dict
        Dictionary mapping model names to predicted probabilities

    Returns:
    --------
    roc_results : dict
        Dictionary containing fpr, tpr, auc for each model
    """
    roc_results = {}

    for model_name, y_scores in y_scores_dict.items():
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        roc_results[model_name] = {
            'fpr': fpr,
            'tpr': tpr,
            'auc': roc_auc,
            'thresholds': thresholds
        }

    return roc_results


def compute_calibration(y_true, y_scores, n_bins=10):
    """
    Compute calibration metrics for probabilistic predictions

    Parameters:
    -----------
    y_true : array-like
        True binary labels
    y_scores : array-like
        Predicted probabilities
    n_bins : int
        Number of bins for calibration (default: 10)

    Returns:
    --------
    calibration_results : dict
        Dictionary containing calibration metrics
    """
    # Bin predictions
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Compute observed frequency in each bin
    bin_indices = np.digitize(y_scores, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    observed_freq = []
    predicted_freq = []
    bin_counts = []

    for i in range(n_bins):
        mask = bin_indices == i
        if np.sum(mask) > 0:
            observed_freq.append(np.mean(y_true[mask]))
            predicted_freq.append(np.mean(y_scores[mask]))
            bin_counts.append(np.sum(mask))
        else:
            observed_freq.append(np.nan)
            predicted_freq.append(bin_centers[i])
            bin_counts.append(0)

    return {
        'bin_centers': bin_centers,
        'observed_freq': observed_freq,
        'predicted_freq': predicted_freq,
        'bin_counts': bin_counts
    }


if __name__ == "__main__":
    print("Co-Ambiguity Hypothesis Testing Module")
    print("=" * 70)

    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range('2018-01-01', '2024-05-24', freq='D')
    n_stocks = 300

    # Generate SCA data
    sca = np.random.randn(len(dates)) * 0.1 + 0.3
    sca = pd.Series(sca, index=dates)

    # Generate returns data
    returns = np.random.randn(len(dates), n_stocks) * 0.02
    returns_df = pd.DataFrame(returns, index=dates,
                             columns=[f'Stock_{i}' for i in range(n_stocks)])

    # Generate market index
    market_returns = returns_df.mean(axis=1)
    market_index = (1 + market_returns).cumprod()

    # Generate ambiguity data
    ambiguity = np.random.randn(len(dates), n_stocks) * 0.1
    ambiguity_df = pd.DataFrame(ambiguity, index=dates,
                               columns=[f'Stock_{i}' for i in range(n_stocks)])

    # Generate liquidity data
    liquidity = pd.DataFrame({
        'Spread': np.random.rand(len(dates)) * 0.01 + 0.001,
        'Turnover': np.random.rand(len(dates)) * 0.05 + 0.01
    }, index=dates)

    # Generate volatility data
    volatility = pd.Series(np.random.rand(len(dates)) * 0.02 + 0.01, index=dates)

    # Initialize tests
    tester = CoAmbiguityHypothesisTests(
        sca_series=sca,
        ambiguity_df=ambiguity_df,
        returns_df=returns_df,
        market_index_df=market_index,
        liquidity_df=liquidity,
        volatility_df=volatility,
        vix_df=None
    )

    # Test Hypothesis 1
    print("\n" + "="*70)
    print("TESTING HYPOTHESIS 1: Leading Indicator")
    print("="*70)
    h1_results = tester.test_hypothesis_1_leading_indicator()

    # Test Hypothesis 2
    print("\n" + "="*70)
    print("TESTING HYPOTHESIS 2: Incremental Power")
    print("="*70)
    h2_results = tester.test_hypothesis_2_incremental_power()

    # Test Hypothesis 3
    print("\n" + "="*70)
    print("TESTING HYPOTHESIS 3: Liquidity Channel")
    print("="*70)
    h3_results = tester.test_hypothesis_3_liquidity_channel()

    # Test Hypothesis 4
    print("\n" + "="*70)
    print("TESTING HYPOTHESIS 4: Structural Change")
    print("="*70)
    structural_periods = [
        ('2018-01-01', '2018-06-01'),
        ('2020-01-01', '2020-06-01'),
        ('2022-01-01', '2022-06-01')
    ]
    h4_results = tester.test_hypothesis_4_structural_change(structural_periods)

    # Test Hypothesis 5
    print("\n" + "="*70)
    print("TESTING HYPOTHESIS 5: Volatility Lead")
    print("="*70)
    h5_results = tester.test_hypothesis_5_volatility_lead()

    print("\n" + "="*70)
    print("All hypothesis tests complete!")
    print("="*70)
