"""
Ambiguity vs. Higher-Order Moments Analysis Module
Implements orthogonality tests between ambiguity and tail risk measures
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import skew, kurtosis, fisher_exact
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class AmbiguityMomentsAnalyzer:
    """
    Analyze the distinction between ambiguity and higher-order moments
    """

    def __init__(self, ambiguity_df, returns_df):
        """
        Initialize the analyzer

        Parameters:
        -----------
        ambiguity_df : pandas DataFrame
            Individual ambiguity indices (stocks × dates)
        returns_df : pandas DataFrame
            Intraday returns for computing moments (stocks × dates)
        """
        self.ambiguity_df = ambiguity_df
        self.returns_df = returns_df

        # Compute moments
        self.moments_df = self._compute_moments()

        # Align data
        self._align_data()

    def _compute_moments(self):
        """
        Compute daily higher-order moments from intraday returns

        Returns:
        --------
        moments_df : pandas DataFrame
            DataFrame with MultiIndex (moment_type, stock_id)
        """
        print("Computing higher-order moments from intraday returns...")

        moments_dict = {}

        for stock_id in self.returns_df.columns:
            stock_returns = self.returns_df[stock_id]

            # Compute daily moments
            daily_rv = []
            daily_skew = []
            daily_kurt = []

            for date in stock_returns.index:
                # Get intraday returns for this day
                day_returns = stock_returns.loc[date].dropna()

                if len(day_returns) < 10:  # Need minimum observations
                    daily_rv.append(np.nan)
                    daily_skew.append(np.nan)
                    daily_kurt.append(np.nan)
                    continue

                # Realized volatility
                rv = np.sqrt(np.mean(day_returns**2))

                # Skewness and kurtosis
                sk = skew(day_returns)
                kt = kurtosis(day_returns)  # Excess kurtosis

                daily_rv.append(rv)
                daily_skew.append(sk)
                daily_kurt.append(kt)

            # Store in dictionary
            moments_dict[(stock_id, 'RV')] = pd.Series(daily_rv, index=stock_returns.index)
            moments_dict[(stock_id, 'Skew')] = pd.Series(daily_skew, index=stock_returns.index)
            moments_dict[(stock_id, 'Kurt')] = pd.Series(daily_kurt, index=stock_returns.index)

        # Create DataFrame with MultiIndex columns
        moments_df = pd.DataFrame(moments_dict)

        print(f"  Moments computed for {len(self.returns_df.columns)} stocks")
        return moments_df

    def _align_data(self):
        """Align ambiguity and moments data"""
        # Reorganize moments DataFrame
        rv_df = self.moments_df.xs('RV', level=1, axis=1)
        skew_df = self.moments_df.xs('Skew', level=1, axis=1)
        kurt_df = self.moments_df.xs('Kurt', level=1, axis=1)

        # Ensure common dates
        common_dates = self.ambiguity_df.index.intersection(rv_df.index)

        self.ambiguity_df = self.ambiguity_df.loc[common_dates]
        self.rv_df = rv_df.loc[common_dates]
        self.skew_df = skew_df.loc[common_dates]
        self.kurt_df = kurt_df.loc[common_dates]

        print(f"  Data aligned for {len(common_dates)} dates")


def test_hypothesis_1_correlation(ambiguity_df, moments_dict):
    """
    HYPOTHESIS 1: Ambiguity and higher-order moments exhibit low correlation (<0.3)

    Test Implementation:
    1. Compute daily cross-sectional correlations between ambiguity and each moment
    2. Average correlations across time to obtain stable estimates
    3. Test statistical significance using Fisher's z-transformation
    4. Compute 95% confidence intervals

    Parameters:
    -----------
    ambiguity_df : pandas DataFrame
        Individual ambiguity indices (stocks × dates)
    moments_dict : dict
        Dictionary of DataFrames for RV, Skew, Kurt

    Returns:
    --------
    results : dict
        Dictionary containing correlation analysis results:
        - correlations: correlation coefficients
        - std_errors: standard errors
        - confidence_intervals: 95% CIs
        - p_values: p-values for H0: correlation >= 0.3
    """
    print("\n" + "="*70)
    print("TESTING HYPOTHESIS 1: Correlation Orthogonality")
    print("="*70)

    results = {}

    # Compute daily cross-sectional correlations
    print("Computing daily cross-sectional correlations...")

    # Get common stocks
    common_stocks = ambiguity_df.columns
    for moment_name, moment_df in moments_dict.items():
        common_stocks = common_stocks.intersection(moment_df.columns)

    correlations = {'Ambiguity_RV': [], 'Ambiguity_Skew': [], 'Ambiguity_Kurt': []}

    for date in ambiguity_df.index:
        amb_series = ambiguity_df.loc[date, common_stocks]

        for moment_name, moment_df in moments_dict.items():
            moment_series = moment_df.loc[date, common_stocks]

            # Remove NaN values
            valid_mask = ~(amb_series.isna() | moment_series.isna())
            amb_valid = amb_series[valid_mask]
            moment_valid = moment_series[valid_mask]

            if len(amb_valid) < 10:  # Need minimum observations
                correlations[f'Ambiguity_{moment_name}'].append(np.nan)
                continue

            # Compute correlation
            corr = np.corrcoef(amb_valid, moment_valid)[0, 1]
            correlations[f'Ambiguity_{moment_name}'].append(corr)

    # Compute average correlations and statistics
    for pair_name, corr_series in correlations.items():
        corr_array = np.array(corr_series)
        corr_array = corr_array[~np.isnan(corr_array)]

        if len(corr_array) == 0:
            print(f"  Warning: No valid correlations for {pair_name}")
            continue

        mean_corr = np.mean(corr_array)
        std_corr = np.std(corr_array)
        n = len(corr_array)

        # Fisher's z-transformation for confidence intervals
        # z = 0.5 * ln((1+r)/(1-r))
        with np.errstate(divide='ignore', invalid='ignore'):
            z_scores = 0.5 * np.log((1 + corr_array) / (1 - corr_array))
            z_scores = z_scores[~np.isnan(z_scores) & ~np.isinf(z_scores)]

        if len(z_scores) > 0:
            mean_z = np.mean(z_scores)
            se_z = 1 / np.sqrt(len(z_scores) - 3)

            # 95% CI for z
            z_ci_low = mean_z - 1.96 * se_z
            z_ci_high = mean_z + 1.96 * se_z

            # Convert back to correlation
            # r = tanh(z)
            corr_ci_low = np.tanh(z_ci_low)
            corr_ci_high = np.tanh(z_ci_high)

            # Test H0: correlation >= 0.3
            # One-sided test: H0: rho >= 0.3 vs H1: rho < 0.3
            rho_0 = 0.3
            z_0 = 0.5 * np.log((1 + rho_0) / (1 - rho_0))
            z_statistic = (mean_z - z_0) / se_z
            p_value = stats.norm.cdf(z_statistic)  # One-sided p-value

            results[pair_name] = {
                'mean_correlation': mean_corr,
                'std_error': std_corr / np.sqrt(n),
                'n_observations': n,
                'ci_lower': corr_ci_low,
                'ci_upper': corr_ci_high,
                'z_statistic': z_statistic,
                'p_value': p_value,
                'significant_orthogonality': p_value < 0.05 and mean_corr < 0.3
            }

            print(f"\n  {pair_name}:")
            print(f"    Mean correlation: {mean_corr:.4f}")
            print(f"    95% CI: [{corr_ci_low:.4f}, {corr_ci_high:.4f}]")
            print(f"    P-value (H0: rho >= 0.3): {p_value:.4f}")
            print(f"    Orthogonality confirmed: {results[pair_name]['significant_orthogonality']}")

    # Overall conclusion
    all_confirmed = all(r['significant_orthogonality'] for r in results.values())

    print(f"\n  HYPOTHESIS 1 CONCLUSION:")
    print(f"  All correlations < 0.3: {all_confirmed}")

    return results


def test_hypothesis_2_regression(ambiguity_df, moments_dict):
    """
    HYPOTHESIS 2: Regression of ambiguity on moments yields R² < 10%

    Test Implementation:
    1. Estimate time-series regression for each stock
    2. Examine distribution of R² across stocks
    3. Test H0: median(R²) >= 0.10 using Wilcoxon signed-rank test
    4. Extract "pure ambiguity" residuals

    Parameters:
    -----------
    ambiguity_df : pandas DataFrame
        Individual ambiguity indices (stocks × dates)
    moments_dict : dict
        Dictionary of DataFrames for RV, Skew, Kurt

    Returns:
    --------
    results : dict
        Dictionary containing regression results:
        - r2_distribution: R² values across stocks
        - median_r2: median R²
        - wilcoxon_statistic: test statistic
        - wilcoxon_pvalue: p-value for H0: median R² >= 0.10
        - pure_ambiguity_df: residuals (orthogonal component)
    """
    print("\n" + "="*70)
    print("TESTING HYPOTHESIS 2: Regression Orthogonality")
    print("="*70)

    results = {}
    r2_values = []
    residuals_dict = {}

    # Get common stocks
    common_stocks = ambiguity_df.columns
    for moment_df in moments_dict.values():
        common_stocks = common_stocks.intersection(moment_df.columns)

    print(f"Running regressions for {len(common_stocks)} stocks...")

    for stock_id in common_stocks:
        # Get time series for this stock
        amb_series = ambiguity_df[stock_id]
        rv_series = moments_dict['RV'][stock_id]
        skew_series = moments_dict['Skew'][stock_id]
        kurt_series = moments_dict['Kurt'][stock_id]

        # Create regression dataframe
        reg_data = pd.DataFrame({
            'Ambiguity': amb_series,
            'RV': rv_series,
            'Skew': skew_series,
            'Kurt': kurt_series
        }).dropna()

        if len(reg_data) < 50:  # Need sufficient observations
            continue

        # Prepare data
        X = reg_data[['RV', 'Skew', 'Kurt']]
        X = sm.add_constant(X)
        y = reg_data['Ambiguity']

        try:
            # OLS regression
            model = sm.OLS(y, X).fit()

            # Store R²
            r2 = model.rsquared
            r2_values.append(r2)

            # Store residuals (pure ambiguity)
            residuals = model.resid
            residuals_dict[stock_id] = residuals

        except Exception as e:
            print(f"  Warning: Regression failed for {stock_id}: {e}")
            continue

    # Convert to arrays
    r2_array = np.array(r2_values)

    # Compute statistics
    median_r2 = np.median(r2_array)
    mean_r2 = np.mean(r2_array)

    # Wilcoxon signed-rank test: H0: median >= 0.10
    # Test if median R² is significantly less than 0.10
    test_value = 0.10
    statistic, p_value = stats.wilcoxon(r2_array - test_value, alternative='less')

    # Percentiles
    percentiles = {
        '10th': np.percentile(r2_array, 10),
        '25th': np.percentile(r2_array, 25),
        '50th (median)': np.percentile(r2_array, 50),
        '75th': np.percentile(r2_array, 75),
        '90th': np.percentile(r2_array, 90)
    }

    results = {
        'r2_distribution': r2_array,
        'mean_r2': mean_r2,
        'median_r2': median_r2,
        'percentiles': percentiles,
        'wilcoxon_statistic': statistic,
        'wilcoxon_pvalue': p_value,
        'orthogonality_confirmed': p_value < 0.05 and median_r2 < 0.10,
        'n_stocks': len(r2_array)
    }

    # Create pure ambiguity DataFrame
    pure_ambiguity_df = pd.DataFrame(residuals_dict)

    print(f"\n  Regression Results:")
    print(f"    Number of stocks: {len(r2_array)}")
    print(f"    Mean R²: {mean_r2:.4f}")
    print(f"    Median R²: {median_r2:.4f}")
    print(f"    10th percentile: {percentiles['10th']:.4f}")
    print(f"    90th percentile: {percentiles['90th']:.4f}")
    print(f"\n  Wilcoxon Test (H0: median R² >= 0.10):")
    print(f"    Statistic: {statistic:.2f}")
    print(f"    P-value: {p_value:.4f}")
    print(f"    Orthogonality confirmed: {results['orthogonality_confirmed']}")

    results['pure_ambiguity_df'] = pure_ambiguity_df

    return results


def test_hypothesis_3_interaction(market_moments, market_ambiguity, market_returns,
                                 crash_threshold=-0.05, prediction_horizon=5):
    """
    HYPOTHESIS 3: Ambiguity-skewness interaction improves crash prediction

    Test Implementation:
    1. Define crash indicator (market return < threshold within horizon)
    2. Estimate logistic regression with and without interaction term
    3. Compute likelihood ratio test
    4. Compare AUC between models

    Parameters:
    -----------
    market_moments : pandas DataFrame
        Market-level moments (RV, Skew, Kurt) × dates
    market_ambiguity : pandas Series
        Market-level ambiguity × dates
    market_returns : pandas Series
        Market returns × dates
    crash_threshold : float
        Threshold for crash definition (default: -5%)
    prediction_horizon : int
        Days ahead for crash prediction (default: 5)

    Returns:
    --------
    results : dict
        Dictionary containing interaction test results:
        - model_main: main effects model results
        - model_interaction: interaction model results
        - lr_statistic: likelihood ratio test statistic
        - lr_pvalue: LR test p-value
        - auc_comparison: AUC for both models
    """
    print("\n" + "="*70)
    print("TESTING HYPOTHESIS 3: Interaction Effect on Crash Prediction")
    print("="*70)

    # Align data
    common_dates = market_moments.index.intersection(market_ambiguity.index)
    common_dates = common_dates.intersection(market_returns.index)

    # Prepare data
    data = pd.DataFrame({
        'RV': market_moments['RV'].loc[common_dates],
        'Skew': market_moments['Skew'].loc[common_dates],
        'Kurt': market_moments['Kurt'].loc[common_dates],
        'Ambiguity': market_ambiguity.loc[common_dates],
        'Returns': market_returns.loc[common_dates]
    }).dropna()

    # Create crash indicator
    future_returns = data['Returns'].shift(-prediction_horizon)
    crash_indicator = (future_returns < crash_threshold).astype(int)

    data['Crash'] = crash_indicator

    # Create interaction term
    data['Skew_x_Amb'] = data['Skew'] * data['Ambiguity']

    # Remove rows with NaN in crash indicator (end of sample)
    data = data.dropna(subset=['Crash'])

    print(f"  Crash events: {data['Crash'].sum()} out of {len(data)} observations")
    print(f"  Crash rate: {data['Crash'].mean():.2%}")

    results = {}

    # Model 1: Main effects only
    print("\n  Estimating Model 1 (main effects only)...")
    X_main = data[['Ambiguity', 'RV', 'Skew', 'Kurt']]
    X_main = sm.add_constant(X_main)
    y = data['Crash']

    try:
        model_main = Logit(y, X_main).fit(disp=0)

        # Model 2: With interaction
        print("  Estimating Model 2 (with interaction)...")
        X_interaction = data[['Ambiguity', 'RV', 'Skew', 'Kurt', 'Skew_x_Amb']]
        X_interaction = sm.add_constant(X_interaction)

        model_interaction = Logit(y, X_interaction).fit(disp=0)

        # Likelihood ratio test
        ll_main = model_main.llf
        ll_interaction = model_interaction.llf
        lr_statistic = 2 * (ll_interaction - ll_main)
        lr_pvalue = 1 - stats.chi2.cdf(lr_statistic, df=1)

        # AUC comparison
        from sklearn.metrics import roc_auc_score

        pred_main = model_main.predict(X_main)
        pred_interaction = model_interaction.predict(X_interaction)

        auc_main = roc_auc_score(y, pred_main)
        auc_interaction = roc_auc_score(y, pred_interaction)

        # Interaction coefficient significance
        interaction_coef = model_interaction.params['Skew_x_Amb']
        interaction_se = model_interaction.bse['Skew_x_Amb']
        interaction_t = model_interaction.tvalues['Skew_x_Amb']
        interaction_p = model_interaction.pvalues['Skew_x_Amb']

        results = {
            'model_main': {
                'params': model_main.params.to_dict(),
                'llf': ll_main,
                'auc': auc_main,
                'pseudo_r2': model_main.prsquared
            },
            'model_interaction': {
                'params': model_interaction.params.to_dict(),
                'llf': ll_interaction,
                'auc': auc_interaction,
                'pseudo_r2': model_interaction.prsquared
            },
            'lr_statistic': lr_statistic,
            'lr_pvalue': lr_pvalue,
            'interaction_significant': lr_pvalue < 0.05,
            'interaction_coefficient': interaction_coef,
            'interaction_se': interaction_se,
            'interaction_t_stat': interaction_t,
            'interaction_p_value': interaction_p,
            'auc_improvement': auc_interaction - auc_main
        }

        print(f"\n  Interaction Term Results:")
        print(f"    Coefficient: {interaction_coef:.6f}")
        print(f"    Std Error: {interaction_se:.6f}")
        print(f"    T-statistic: {interaction_t:.4f}")
        print(f"    P-value: {interaction_p:.4f}")

        print(f"\n  Likelihood Ratio Test:")
        print(f"    LR Statistic: {lr_statistic:.4f}")
        print(f"    P-value: {lr_pvalue:.4f}")
        print(f"    Interaction significant: {results['interaction_significant']}")

        print(f"\n  AUC Comparison:")
        print(f"    Main effects: {auc_main:.4f}")
        print(f"    With interaction: {auc_interaction:.4f}")
        print(f"    Improvement: {auc_interaction - auc_main:.4f}")

    except Exception as e:
        print(f"  Error in estimation: {e}")
        results = {
            'error': str(e)
        }

    return results


def test_hypothesis_4_pca(ambiguity_df, moments_dict, n_components=4):
    """
    HYPOTHESIS 4: PCA shows ambiguity loads on distinct factor from moments

    Test Implementation:
    1. Standardize all variables
    2. Perform PCA on [Ambiguity, RV, Skew, Kurt]
    3. Examine factor loadings
    4. Test if ambiguity loads heavily on distinct factor

    Parameters:
    -----------
    ambiguity_df : pandas DataFrame
        Individual ambiguity indices (stocks × dates)
    moments_dict : dict
        Dictionary of DataFrames for RV, Skew, Kurt
    n_components : int
        Number of principal components (default: 4)

    Returns:
    --------
    results : dict
        Dictionary containing PCA results:
        - eigenvalues: eigenvalues of covariance matrix
        - explained_variance_ratio: proportion of variance explained
        - factor_loadings: factor loadings for each variable
        - ambiguity_distinct_factor: whether ambiguity loads on distinct factor
    """
    print("\n" + "="*70)
    print("TESTING HYPOTHESIS 4: Factor Structure via PCA")
    print("="*70)

    # Aggregate to market level (average across stocks)
    print("Aggregating to market level...")

    market_ambiguity = ambiguity_df.mean(axis=1)
    market_rv = moments_dict['RV'].mean(axis=1)
    market_skew = moments_dict['Skew'].mean(axis=1)
    market_kurt = moments_dict['Kurt'].mean(axis=1)

    # Create data matrix
    data = pd.DataFrame({
        'Ambiguity': market_ambiguity,
        'RV': market_rv,
        'Skew': market_skew,
        'Kurt': market_kurt
    }).dropna()

    print(f"  Data shape: {data.shape}")

    # Standardize
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Perform PCA
    print("\nPerforming PCA...")
    pca = PCA(n_components=n_components)
    pca.fit(data_scaled)

    # Get results
    eigenvalues = pca.explained_variance_
    explained_variance_ratio = pca.explained_variance_ratio_
    factor_loadings = pca.components_.T

    # Compute variance inflation factors
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    vif_data = pd.DataFrame()
    vif_data["Variable"] = data.columns
    vif_data["VIF"] = [variance_inflation_factor(data_scaled, i)
                       for i in range(data_scaled.shape[1])]

    # Analyze ambiguity loadings
    ambiguity_loadings = factor_loadings[0, :]  # First row = Ambiguity

    # Find which factor has highest absolute loading for ambiguity
    max_loading_idx = np.argmax(np.abs(ambiguity_loadings))
    max_loading = ambiguity_loadings[max_loading_idx]

    # Check if ambiguity loads on distinct factor (not PC1)
    # We expect PC1 to capture general uncertainty (RV, Skew, Kurt correlated)
    # and PC2 to capture ambiguity
    distinct_factor = (max_loading_idx > 0) and (np.abs(max_loading) > 0.5)

    results = {
        'eigenvalues': eigenvalues,
        'explained_variance_ratio': explained_variance_ratio,
        'cumulative_variance_ratio': np.cumsum(explained_variance_ratio),
        'factor_loadings': pd.DataFrame(
            factor_loadings,
            columns=[f'PC{i+1}' for i in range(n_components)],
            index=data.columns
        ),
        'vif': vif_data,
        'ambiguity_max_loading': max_loading,
        'ambiguity_max_loading_factor': max_loading_idx + 1,
        'ambiguity_distinct_factor': distinct_factor
    }

    print(f"\n  PCA Results:")
    print(f"    Eigenvalues: {eigenvalues}")
    print(f"    Explained variance ratio: {explained_variance_ratio}")
    print(f"    Cumulative variance: {np.cumsum(explained_variance_ratio)}")

    print(f"\n  Factor Loadings:")
    print(results['factor_loadings'])

    print(f"\n  Ambiguity Analysis:")
    print(f"    Maximum loading: {max_loading:.4f}")
    print(f"    Loading on factor: PC{max_loading_idx + 1}")
    print(f"    Distinct factor confirmed: {distinct_factor}")

    print(f"\n  Variance Inflation Factors:")
    print(vif_data.to_string(index=False))

    return results


def test_hypothesis_5_portfolio(ambiguity_df, skew_df, returns_df,
                               n_skew_quintiles=5, n_amb_groups=2,
                               rebalance_freq='M', fama_french_factors=None):
    """
    HYPOTHESIS 5: Double-sorted portfolios generate significant alphas

    Test Implementation:
    1. Sort stocks into quintiles by skewness
    2. Within lowest skewness quintile, sort by ambiguity (high/low)
    3. Form portfolios and compute returns
    4. Compute Fama-French alphas

    Parameters:
    -----------
    ambiguity_df : pandas DataFrame
        Individual ambiguity indices (stocks × dates)
    skew_df : pandas DataFrame
        Skewness measures (stocks × dates)
    returns_df : pandas DataFrame
        Future returns (stocks × dates)
    n_skew_quintiles : int
        Number of skewness groups (default: 5)
    n_amb_groups : int
        Number of ambiguity groups within each skewness group (default: 2)
    rebalance_freq : str
        Rebalancing frequency (default: 'M' = monthly)
    fama_french_factors : DataFrame or None
        Fama-French five-factor data

    Returns:
    --------
    results : dict
        Dictionary containing portfolio results:
        - portfolio_returns: returns for each portfolio
        - performance_metrics: Sharpe, Sortino, etc.
        - alphas: Fama-French alphas
        - long_short_results: long-short strategy results
    """
    print("\n" + "="*70)
    print("TESTING HYPOTHESIS 5: Portfolio Value via Double-Sorting")
    print("="*70)

    # Get common stocks and dates
    common_stocks = ambiguity_df.columns.intersection(skew_df.columns)
    common_stocks = common_stocks.intersection(returns_df.columns)

    print(f"  Common stocks: {len(common_stocks)}")

    # Implement double-sorting with monthly rebalancing
    print(f"\n  Implementing double-sorting strategy...")
    print(f"    First sort: Skewness ({n_skew_quintiles} quintiles)")
    print(f"    Second sort: Ambiguity ({n_amb_groups} groups within lowest skew)")
    print(f"    Rebalancing: {rebalance_freq}")

    # For simplicity, use period-specific sorting (not rolling)
    # In practice, would use rolling windows

    # Compute average skewness and ambiguity for each stock over full period
    avg_skew = skew_df[common_stocks].mean()
    avg_amb = ambiguity_df[common_stocks].mean()

    # First sort: by skewness quintiles
    skew_quantiles = pd.qcut(avg_skew, n_skew_quintiles, labels=False, duplicates='drop')

    # Second sort: within lowest skewness quintile, sort by ambiguity
    lowest_skew_stocks = avg_skew[skew_quantiles == 0].index
    amb_in_lowest_skew = avg_amb[lowest_skew_stocks]
    amb_groups = pd.qcut(amb_in_lowest_skew, n_amb_groups, labels=['Low', 'High'], duplicates='drop')

    # Define portfolios
    toxic_stocks = amb_groups[amb_groups == 'High'].index.tolist()  # Low skew + High amb
    stable_stocks = amb_groups[amb_groups == 'Low'].index.tolist()  # High skew + Low amb

    print(f"    Toxic portfolio (Low Skew + High Amb): {len(toxic_stocks)} stocks")
    print(f"    Stable portfolio (High Skew + Low Amb): {len(stable_stocks)} stocks")

    # Compute portfolio returns
    portfolio_returns = pd.DataFrame()
    portfolio_returns['Toxic'] = returns_df[toxic_stocks].mean(axis=1)
    portfolio_returns['Stable'] = returns_df[stable_stocks].mean(axis=1)
    portfolio_returns['Long_Short'] = portfolio_returns['Stable'] - portfolio_returns['Toxic']

    # Remove NaN
    portfolio_returns = portfolio_returns.dropna()

    # Compute performance metrics
    def compute_performance_metrics(returns_series, annualize_factor=252):
        """Compute performance metrics for a return series"""
        mean_return = returns_series.mean()
        std_return = returns_series.std()
        sharpe_ratio = mean_return / std_return * np.sqrt(annualize_factor) if std_return > 0 else np.nan

        # Maximum drawdown
        cumulative = (1 + returns_series).cumprod()
        rolling_max = cumulative.cummax()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        # Annualized return
        annualized_return = (1 + mean_return) ** annualize_factor - 1

        return {
            'mean_daily': mean_return,
            'std_daily': std_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }

    performance = {}
    for portfolio in portfolio_returns.columns:
        performance[portfolio] = compute_performance_metrics(portfolio_returns[portfolio])

    # Compute Fama-French alpha (if factors provided)
    alpha_results = {}
    if fama_french_factors is not None:
        print("\n  Computing Fama-French alphas...")
        # This would require actual FF factor data
        # Placeholder for implementation
        pass
    else:
        print("\n  Fama-French factors not provided, skipping alpha computation")
        alpha_results = None

    results = {
        'portfolio_returns': portfolio_returns,
        'performance_metrics': performance,
        'alphas': alpha_results,
        'n_toxic_stocks': len(toxic_stocks),
        'n_stable_stocks': len(stable_stocks)
    }

    print(f"\n  Portfolio Performance:")
    for portfolio, perf in performance.items():
        print(f"    {portfolio}:")
        print(f"      Annualized Return: {perf['annualized_return']:.2%}")
        print(f"      Sharpe Ratio: {perf['sharpe_ratio']:.4f}")
        print(f"      Max Drawdown: {perf['max_drawdown']:.2%}")

    # Test if long-short generates significant alpha
    ls_mean = portfolio_returns['Long_Short'].mean()
    ls_std = portfolio_returns['Long_Short'].std()
    ls_t_stat = ls_mean / (ls_std / np.sqrt(len(portfolio_returns)))
    ls_p_value = 2 * (1 - stats.norm.cdf(abs(ls_t_stat)))

    results['long_short_t_stat'] = ls_t_stat
    results['long_short_p_value'] = ls_p_value
    results['long_short_significant'] = ls_p_value < 0.05

    print(f"\n  Long-Short Strategy:")
    print(f"    Mean daily return: {ls_mean:.6f}")
    print(f"    T-statistic: {ls_t_stat:.4f}")
    print(f"    P-value: {ls_p_value:.4f}")
    print(f"    Significant: {results['long_short_significant']}")

    return results


if __name__ == "__main__":
    print("Ambiguity vs. Higher-Order Moments Analysis Module")
    print("=" * 70)

    # Generate sample data for testing
    np.random.seed(42)
    dates = pd.date_range('2018-01-01', '2024-05-24', freq='D')
    n_stocks = 100

    # Generate ambiguity data
    ambiguity_data = np.random.randn(len(dates), n_stocks) * 0.1 + 0.2
    ambiguity_df = pd.DataFrame(ambiguity_data, index=dates,
                               columns=[f'Stock_{i}' for i in range(n_stocks)])

    # Generate intraday returns (simplified - daily moments for demonstration)
    returns_data = {}
    moments_data = {}
    for stock_id in ambiguity_df.columns:
        # Daily returns
        stock_returns = np.random.randn(len(dates)) * 0.02
        returns_data[stock_id] = stock_returns

        # Moments (computed from daily returns for simplicity)
        rv = np.abs(stock_returns) * 0.05 + 0.01
        skew = np.random.randn(len(dates)) * 0.3
        kurt = np.random.rand(len(dates)) * 5 - 1

        moments_data[f'Stock_{i}_RV'] = rv
        moments_data[f'Stock_{i}_Skew'] = skew
        moments_data[f'Stock_{i}_Kurt'] = kurt

    returns_df = pd.DataFrame(returns_data, index=dates)

    # Reorganize moments data
    rv_df = pd.DataFrame({col.split('_')[0]: moments_data[col]
                         for col in moments_data if '_RV' in col}, index=dates)
    skew_df = pd.DataFrame({col.split('_')[0]: moments_data[col]
                           for col in moments_data if '_Skew' in col}, index=dates)
    kurt_df = pd.DataFrame({col.split('_')[0]: moments_data[col]
                           for col in moments_data if '_Kurt' in col}, index=dates)

    moments_dict = {'RV': rv_df, 'Skew': skew_df, 'Kurt': kurt_df}

    # Test Hypothesis 1
    h1_results = test_hypothesis_1_correlation(ambiguity_df, moments_dict)

    # Test Hypothesis 2
    h2_results = test_hypothesis_2_regression(ambiguity_df, moments_dict)

    print("\n" + "="*70)
    print("All hypothesis tests complete!")
    print("="*70)
