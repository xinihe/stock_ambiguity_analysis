"""
Backtest and Trading Strategy Module for SCA
Implements out-of-sample validation and trading strategy backtesting
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class SCABacktester:
    """
    Backtest trading strategies based on Systemic Co-Ambiguity signals
    """

    def __init__(self, sca_series, market_returns, risk_free_rate=0.03):
        """
        Initialize the backtester

        Parameters:
        -----------
        sca_series : pandas Series
            Systemic Co-Ambiguity index time series
        market_returns : pandas Series
            Market returns (daily)
        risk_free_rate : float
            Annual risk-free rate (default: 3%)
        """
        self.sca = sca_series
        self.market_returns = market_returns
        self.risk_free_rate = risk_free_rate / 252  # Daily rate

        # Align data
        self._align_data()

    def _align_data(self):
        """Align SCA and returns data"""
        common_dates = self.sca.index.intersection(self.market_returns.index)
        self.sca = self.sca.loc[common_dates]
        self.market_returns = self.market_returns.loc[common_dates]
        self.common_dates = common_dates

    def generate_dynamic_threshold_signals(self, lookback_window=252,
                                          percentile_threshold=90):
        """
        Generate trading signals based on dynamic thresholds

        Strategy: Switch to cash when SCA exceeds (100 - percentile)th percentile
        of rolling lookback window

        Parameters:
        -----------
        lookback_window : int
            Rolling window for threshold computation (default: 252 days = 1 year)
        percentile_threshold : int
            Percentile for signal threshold (default: 90 = top 10% triggers exit)

        Returns:
        --------
        signals : pandas Series
            Trading signals (1 = in market, 0 = in cash)
        """
        # Compute rolling threshold
        rolling_threshold = self.sca.rolling(
            window=lookback_window, min_periods=lookback_window//2
        ).quantile(percentile_threshold / 100)

        # Generate signals
        signals = (self.sca < rolling_threshold).astype(int)

        return signals

    def generate_static_threshold_signals(self, threshold):
        """
        Generate trading signals based on static threshold

        Parameters:
        -----------
        threshold : float
            Static threshold for SCA signal

        Returns:
        --------
        signals : pandas Series
            Trading signals (1 = in market, 0 = in cash)
        """
        signals = (self.sca < threshold).astype(int)
        return signals

    def backtest_strategy(self, signals, transaction_cost=0.001):
        """
        Backtest a trading strategy

        Parameters:
        -----------
        signals : pandas Series
            Trading signals (1 = in market, 0 = in cash)
        transaction_cost : float
            Transaction cost (default: 0.1%)

        Returns:
        --------
        results : dict
            Dictionary containing backtest results
        """
        # Align signals with returns
        aligned_signals = signals.reindex(self.market_returns.index).fillna(1)

        # Compute strategy returns
        strategy_returns = aligned_signals.shift(1) * self.market_returns

        # Subtract transaction costs
        trades = aligned_signals.diff().abs()
        transaction_costs = trades * transaction_cost
        strategy_returns_net = strategy_returns - transaction_costs

        # Compute performance metrics
        total_return = (1 + strategy_returns_net).prod() - 1
        annualized_return = (1 + strategy_returns_net.mean()) ** 252 - 1

        # Volatility
        volatility = strategy_returns_net.std() * np.sqrt(252)

        # Sharpe ratio
        excess_returns = strategy_returns_net - self.risk_free_rate
        sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252)

        # Maximum drawdown
        cumulative_returns = (1 + strategy_returns_net).cumprod()
        rolling_max = cumulative_returns.rolling(window=len(cumulative_returns), min_periods=1).max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else np.nan

        # Sortino ratio
        downside_returns = strategy_returns_net[strategy_returns_net < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252)
        sortino_ratio = (annualized_return - self.risk_free_rate * 252) / downside_deviation if downside_deviation > 0 else np.nan

        # Hit rate (percentage of profitable days when invested)
        invested_days = aligned_signals.shift(1) == 1
        hit_rate = (strategy_returns[invested_days] > 0).mean() if invested_days.sum() > 0 else np.nan

        results = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'sortino_ratio': sortino_ratio,
            'hit_rate': hit_rate,
            'strategy_returns': strategy_returns_net,
            'cumulative_returns': cumulative_returns,
            'signals': aligned_signals
        }

        return results

    def compare_strategies(self, threshold_params=None):
        """
        Compare multiple SCA-based strategies

        Parameters:
        -----------
        threshold_params : list or None
            List of (lookback_window, percentile_threshold) tuples

        Returns:
        --------
        comparison : dict
            Dictionary containing comparison results
        """
        if threshold_params is None:
            threshold_params = [
                (252, 90),   # 1-year lookback, 90th percentile
                (126, 90),   # 6-month lookback, 90th percentile
                (252, 95),   # 1-year lookback, 95th percentile
                (252, 85),   # 1-year lookback, 85th percentile
            ]

        comparison = {}

        # SCA strategies
        for i, (lookback, percentile) in enumerate(threshold_params):
            signals = self.generate_dynamic_threshold_signals(lookback, percentile)
            results = self.backtest_strategy(signals)
            comparison[f'SCA_L{lookback}_P{percentile}'] = results

        # Buy and hold benchmark
        buy_hold_returns = self.market_returns
        buy_hold_cumulative = (1 + buy_hold_returns).cumprod()
        buy_hold_annualized = (1 + buy_hold_returns.mean()) ** 252 - 1
        buy_hold_volatility = buy_hold_returns.std() * np.sqrt(252)
        buy_hold_excess = buy_hold_returns - self.risk_free_rate
        buy_hold_sharpe = buy_hold_excess.mean() / buy_hold_excess.std() * np.sqrt(252)
        buy_hold_max_dd = (buy_hold_cumulative / buy_hold_cumulative.cummax() - 1).min()
        buy_hold_calmar = buy_hold_annualized / abs(buy_hold_max_dd)

        comparison['Buy_Hold'] = {
            'total_return': buy_hold_cumulative.iloc[-1] - 1,
            'annualized_return': buy_hold_annualized,
            'volatility': buy_hold_volatility,
            'sharpe_ratio': buy_hold_sharpe,
            'max_drawdown': buy_hold_max_dd,
            'calmar_ratio': buy_hold_calmar,
            'sortino_ratio': np.nan,  # Not computed for simplicity
            'hit_rate': (buy_hold_returns > 0).mean(),
            'cumulative_returns': buy_hold_cumulative
        }

        return comparison

    def compute_signal_efficiency(self, signals, crash_threshold=-0.05,
                                 prediction_horizon=5):
        """
        Compute signal efficiency metrics (ROC, AUC, confusion matrix)

        Parameters:
        -----------
        signals : pandas Series
            Trading signals (1 = safe, 0 = warning)
        crash_threshold : float
            Threshold for crash definition (default: -5%)
        prediction_horizon : int
            Days ahead for crash prediction (default: 5)

        Returns:
        --------
        efficiency : dict
            Dictionary containing efficiency metrics
        """
        # Create crash indicator
        future_returns = self.market_returns.shift(-prediction_horizon)
        crash_indicator = (future_returns < crash_threshold).astype(int)

        # Align data
        aligned_signals = signals.reindex(crash_indicator.index).fillna(1)

        # Create prediction (signal = 0 predicts crash)
        predictions = (aligned_signals == 0).astype(int)

        # Remove NaN
        valid_mask = ~(crash_indicator.isna() | predictions.isna())
        crash_indicator = crash_indicator[valid_mask]
        predictions = predictions[valid_mask]

        # Compute confusion matrix
        tp = np.sum((predictions == 1) & (crash_indicator == 1))
        tn = np.sum((predictions == 0) & (crash_indicator == 0))
        fp = np.sum((predictions == 1) & (crash_indicator == 0))
        fn = np.sum((predictions == 0) & (crash_indicator == 1))

        # Metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else np.nan
        precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else np.nan

        # False positive rate (Type I error)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan

        # False negative rate (Type II error)
        fnr = fn / (fn + tp) if (fn + tp) > 0 else np.nan

        efficiency = {
            'true_positives': tp,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'false_positive_rate': fpr,
            'false_negative_rate': fnr,
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn
        }

        return efficiency

    def walk_forward_validation(self, train_window=252, test_window=63,
                               percentile_threshold=90):
        """
        Perform walk-forward validation

        Parameters:
        -----------
        train_window : int
            Training window for threshold computation (default: 252 days)
        test_window : int
            Test window (default: 63 days = 3 months)
        percentile_threshold : int
            Percentile for signal threshold (default: 90)

        Returns:
        --------
        wf_results : dict
            Dictionary containing walk-forward results
        """
        dates = self.common_dates
        n_periods = len(dates)

        # Initialize storage
        all_returns = []
        all_signals = []
        period_results = []

        # Slide window through data
        for start_idx in range(0, n_periods - train_window - test_window, test_window):
            # Define train and test periods
            train_dates = dates[start_idx:start_idx + train_window]
            test_dates = dates[start_idx + train_window:start_idx + train_window + test_window]

            # Compute threshold from training period
            sca_train = self.sca.loc[train_dates]
            threshold = sca_train.quantile(percentile_threshold / 100)

            # Generate signals for test period
            sca_test = self.sca.loc[test_dates]
            signals_test = (sca_test < threshold).astype(int)

            # Compute test period returns
            returns_test = self.market_returns.loc[test_dates]
            strategy_returns = signals_test.shift(1) * returns_test

            all_returns.extend(strategy_returns.dropna())
            all_signals.extend(signals_test)

            # Compute period metrics
            period_total_return = (1 + strategy_returns).prod() - 1
            period_volatility = strategy_returns.std() * np.sqrt(252)

            period_results.append({
                'start_date': test_dates[0],
                'end_date': test_dates[-1],
                'total_return': period_total_return,
                'volatility': period_volatility,
                'threshold': threshold
            })

        # Convert to arrays
        all_returns = pd.Series(all_returns)
        all_signals = pd.Series(all_signals)

        # Compute overall metrics
        wf_annualized_return = (1 + all_returns.mean()) ** 252 - 1
        wf_volatility = all_returns.std() * np.sqrt(252)
        wf_cumulative = (1 + all_returns).cumprod()
        wf_max_dd = (wf_cumulative / wf_cumulative.cummax() - 1).min()
        wf_calmar = wf_annualized_return / abs(wf_max_dd) if wf_max_dd != 0 else np.nan

        wf_results = {
            'period_results': period_results,
            'annualized_return': wf_annualized_return,
            'volatility': wf_volatility,
            'max_drawdown': wf_max_dd,
            'calmar_ratio': wf_calmar,
            'all_returns': all_returns,
            'all_signals': all_signals
        }

        return wf_results


def plot_backtest_comparison(comparison_results, save_path=None):
    """
    Plot comparison of backtest results

    Parameters:
    -----------
    comparison_results : dict
        Results from compare_strategies()
    save_path : str or None
        Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Plot 1: Cumulative returns
    ax1 = axes[0, 0]
    for strategy_name, results in comparison_results.items():
        if 'cumulative_returns' in results:
            cumulative = results['cumulative_returns']
            ax1.plot(cumulative.index, cumulative.values, label=strategy_name, alpha=0.7)
    ax1.set_title('Cumulative Returns Comparison')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Cumulative Return')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Performance metrics comparison
    ax2 = axes[0, 1]
    metrics = ['annualized_return', 'sharpe_ratio', 'calmar_ratio', 'max_drawdown']
    metric_labels = ['Annual Return', 'Sharpe Ratio', 'Calmar Ratio', 'Max DD']

    strategy_names = list(comparison_results.keys())
    x = np.arange(len(strategy_names))
    width = 0.2

    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        values = [comparison_results[name].get(metric, 0) for name in strategy_names]
        # Normalize max drawdown for visualization
        if metric == 'max_drawdown':
            values = [-v * 100 for v in values]  # Convert to positive percentage
        else:
            values = [v * 100 for v in values]  # Convert to percentage

        ax2.bar(x + i * width, values, width, label=label, alpha=0.7)

    ax2.set_title('Performance Metrics Comparison')
    ax2.set_xlabel('Strategy')
    ax2.set_ylabel('Value (%)')
    ax2.set_xticks(x + width * 1.5)
    ax2.set_xticklabels(strategy_names, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # Plot 3: Drawdown comparison
    ax3 = axes[1, 0]
    for strategy_name, results in comparison_results.items():
        if 'cumulative_returns' in results:
            cumulative = results['cumulative_returns']
            drawdown = cumulative / cumulative.cummax() - 1
            ax3.plot(drawdown.index, drawdown.values * 100, label=strategy_name, alpha=0.7)
    ax3.set_title('Drawdown Comparison')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Drawdown (%)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Signal efficiency (if available)
    ax4 = axes[1, 1]
    # This would show ROC curve or similar if implemented
    ax4.text(0.5, 0.5, 'Signal Efficiency Analysis\n(ROC/AUC curves)',
             ha='center', va='center', fontsize=12, transform=ax4.transAxes)
    ax4.set_title('Signal Efficiency')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()


def analyze_regime_performance(sca_series, market_returns, n_regimes=3):
    """
    Analyze strategy performance across SCA regimes

    Parameters:
    -----------
    sca_series : pandas Series
        Systemic Co-Ambiguity index
    market_returns : pandas Series
        Market returns
    n_regimes : int
        Number of regimes to identify (default: 3: low, medium, high)

    Returns:
    --------
    regime_analysis : dict
        Dictionary containing regime analysis results
    """
    # Classify SCA into regimes
    sca_percentiles = sca_series.rank(pct=True)

    regime_boundaries = np.linspace(0, 1, n_regimes + 1)
    regime_labels = []

    for pct in sca_percentiles:
        for i in range(n_regimes):
            if regime_boundaries[i] <= pct < regime_boundaries[i + 1]:
                regime_labels.append(i)
                break
        else:
            regime_labels.append(n_regimes - 1)

    regime_labels = pd.Series(regime_labels, index=sca_series.index)

    # Analyze returns by regime
    regime_analysis = {}

    for regime in range(n_regimes):
        regime_mask = regime_labels == regime
        regime_returns = market_returns[regime_mask]

        regime_stats = {
            'n_observations': regime_mask.sum(),
            'mean_return': regime_returns.mean(),
            'std_return': regime_returns.std(),
            'sharpe_ratio': regime_returns.mean() / regime_returns.std() if regime_returns.std() > 0 else np.nan,
            'hit_rate': (regime_returns > 0).mean(),
            'volatility': regime_returns.std() * np.sqrt(252)
        }

        regime_analysis[f'Regime_{regime}'] = regime_stats

    return regime_analysis


if __name__ == "__main__":
    print("SCA Backtest Analysis Module")
    print("=" * 70)

    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range('2018-01-01', '2024-05-24', freq='D')

    # Generate SCA with time-varying properties
    sca_base = np.random.randn(len(dates)) * 0.1 + 0.3
    # Add crisis spikes
    crisis_periods = [
        ('2018-02-01', '2018-04-01'),
        ('2020-02-01', '2020-04-01'),
        ('2022-02-01', '2022-04-01')
    ]
    for start, end in crisis_periods:
        mask = (dates >= start) & (dates <= end)
        sca_base[mask] += 0.3

    sca = pd.Series(sca_base, index=dates)

    # Generate market returns
    market_returns = pd.Series(np.random.randn(len(dates)) * 0.015, index=dates)
    # Add crisis drawdowns
    for start, end in crisis_periods:
        mask = (dates >= start) & (dates <= end)
        crisis_returns = np.random.randn(mask.sum()) * 0.025 - 0.02
        market_returns[mask] = crisis_returns

    # Initialize backtester
    backtester = SCABacktester(sca, market_returns)

    # Generate signals
    signals = backtester.generate_dynamic_threshold_signals(lookback_window=252,
                                                           percentile_threshold=90)

    # Backtest strategy
    print("\nBacktesting SCA strategy...")
    results = backtester.backtest_strategy(signals)

    print(f"Annualized Return: {results['annualized_return']:.2%}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.4f}")
    print(f"Calmar Ratio: {results['calmar_ratio']:.4f}")
    print(f"Maximum Drawdown: {results['max_drawdown']:.2%}")

    # Compare strategies
    print("\nComparing strategies...")
    comparison = backtester.compare_strategies()

    for strategy, perf in comparison.items():
        print(f"\n{strategy}:")
        print(f"  Annual Return: {perf['annualized_return']:.2%}")
        print(f"  Sharpe Ratio: {perf['sharpe_ratio']:.4f}")
        print(f"  Calmar Ratio: {perf['calmar_ratio']:.4f}")

    # Signal efficiency
    print("\nComputing signal efficiency...")
    efficiency = backtester.compute_signal_efficiency(signals)
    print(f"Accuracy: {efficiency['accuracy']:.2%}")
    print(f"Precision: {efficiency['precision']:.2%}")
    print(f"Recall: {efficiency['recall']:.2%}")
    print(f"F1 Score: {efficiency['f1_score']:.4f}")

    print("\n" + "="*70)
    print("Backtest analysis complete!")
