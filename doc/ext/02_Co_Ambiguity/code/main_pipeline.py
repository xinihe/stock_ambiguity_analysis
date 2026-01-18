"""
Main Analysis Pipeline for Co-Ambiguity Research
Orchestrates the complete analysis from SCA computation to hypothesis testing
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sca_measurement import SystemicCoAmbiguity, compute_cross_market_sca
from hypothesis_testing import CoAmbiguityHypothesisTests
from backtest_analysis import SCABacktester, plot_backtest_comparison
import warnings
warnings.filterwarnings('ignore')


class CoAmbiguityResearchPipeline:
    """
    Complete pipeline for Systemic Co-Ambiguity research
    """

    def __init__(self, data_path=None, corr_window=60, weighted_sca=False):
        """
        Initialize the pipeline

        Parameters:
        -----------
        data_path : str or None
            Path to data directory (None generates sample data)
        corr_window : int
            Correlation window for SCA computation (default: 60 days)
        weighted_sca : bool
            Whether to use market-cap-weighted SCA (default: False)
        """
        self.data_path = data_path
        self.corr_window = corr_window
        self.weighted_sca = weighted_sca

        # Initialize components
        self.sca_calculator = SystemicCoAmbiguity(
            corr_window=corr_window,
            weighted=weighted_sca
        )

        # Storage for results
        self.sca = None
        self.ambiguity_df = None
        self.returns_df = None
        self.market_index = None
        self.hypothesis_results = {}
        self.backtest_results = {}

    def load_data(self):
        """
        Load or generate data for analysis

        Returns:
        --------
        data : dict
            Dictionary containing all data
        """
        if self.data_path:
            print(f"Loading data from {self.data_path}...")
            data = self._load_from_files()
        else:
            print("Generating sample data...")
            data = self._generate_sample_data()

        print(f"Data loaded:")
        print(f"  Dates: {data['dates'][0]} to {data['dates'][-1]}")
        print(f"  Stocks: {len(data['stock_ids'])}")
        print(f"  Total observations: {len(data['dates']) * len(data['stock_ids']):,.0f}")

        return data

    def _load_from_files(self):
        """Load data from files (placeholder for actual implementation)"""
        # Implement actual data loading logic
        pass

    def _generate_sample_data(self, n_stocks=300, start_date='2018-01-01',
                             end_date='2024-05-24'):
        """
        Generate comprehensive sample data for testing

        Parameters:
        -----------
        n_stocks : int
            Number of stocks to simulate
        start_date : str
            Start date
        end_date : str
            End date

        Returns:
        --------
        data : dict
            Dictionary containing all generated data
        """
        np.random.seed(42)
        dates = pd.date_range(start_date, end_date, freq='D')
        n_dates = len(dates)

        print(f"Generating data for {n_stocks} stocks over {n_dates} days...")

        # Generate individual ambiguity indices
        # Base ambiguity with time-varying correlation
        ambiguity_base = np.random.randn(n_dates, n_stocks) * 0.1 + 0.2

        # Add crisis periods with high correlation
        crisis_periods = [
            ('2018-02-01', '2018-04-01', 0.4),  # Trade war tensions
            ('2020-02-01', '2020-04-01', 0.5),  # COVID crash
            ('2022-02-01', '2022-04-01', 0.3),  # Regulatory changes
            ('2024-01-01', '2024-03-01', 0.35),  # Market volatility
        ]

        for start, end, shock_magnitude in crisis_periods:
            mask = (dates >= start) & (dates <= end)
            common_shock = np.random.randn(np.sum(mask)) * shock_magnitude
            ambiguity_base[mask] += common_shock.reshape(-1, 1)

        # Create DataFrame
        stock_ids = [f'Stock_{i:04d}' for i in range(n_stocks)]
        ambiguity_df = pd.DataFrame(ambiguity_base, index=dates, columns=stock_ids)

        # Generate returns (correlated with ambiguity during crises)
        returns_base = np.random.randn(n_dates, n_stocks) * 0.02

        for start, end, _ in crisis_periods:
            mask = (dates >= start) & (dates <= end)
            # Negative returns during high ambiguity periods
            crisis_returns = np.random.randn(np.sum(mask), n_stocks) * 0.025 - 0.01
            returns_base[mask] = crisis_returns

        returns_df = pd.DataFrame(returns_base, index=dates, columns=stock_ids)

        # Market index (equal-weighted)
        market_returns = returns_df.mean(axis=1)
        market_index = (1 + market_returns).cumprod()

        # Market caps (for weighted SCA)
        market_caps = pd.DataFrame(
            np.random.rand(n_dates, n_stocks) * 1e10 + 1e9,
            index=dates,
            columns=stock_ids
        )

        # Liquidity measures
        liquidity_df = pd.DataFrame({
            'Spread': np.random.rand(n_dates) * 0.005 + 0.001,
            'Turnover': np.random.rand(n_dates) * 0.05 + 0.01,
            'Depth': np.random.rand(n_dates) * 1e6 + 1e5
        }, index=dates)

        # Deteriorate liquidity during crises
        for start, end, _ in crisis_periods:
            mask = (dates >= start) & (dates <= end)
            liquidity_df.loc[mask, 'Spread'] *= 2.0
            liquidity_df.loc[mask, 'Turnover'] *= 0.6
            liquidity_df.loc[mask, 'Depth'] *= 0.7

        # Volatility measures
        volatility = pd.Series(
            np.random.rand(n_dates) * 0.015 + 0.01,
            index=dates
        )

        # Increase volatility during crises
        for start, end, _ in crisis_periods:
            mask = (dates >= start) & (dates <= end)
            volatility.loc[mask] *= 2.5

        # VIX-equivalent (implied volatility)
        vix = volatility * 1.2  # VIX typically higher than realized

        # Structural change periods
        structural_change_periods = [
            ('2018-01-01', '2018-06-01'),  # Trade war
            ('2020-01-01', '2020-06-01'),  # COVID pandemic
            ('2022-01-01', '2022-06-01'),  # Regulatory overhaul
        ]

        data = {
            'dates': dates,
            'stock_ids': stock_ids,
            'ambiguity_df': ambiguity_df,
            'returns_df': returns_df,
            'market_index': market_index,
            'market_caps': market_caps,
            'liquidity_df': liquidity_df,
            'volatility': volatility,
            'vix': vix,
            'structural_change_periods': structural_change_periods,
            'crisis_periods': crisis_periods
        }

        return data

    def compute_sca(self, ambiguity_df, market_caps=None):
        """
        Compute Systemic Co-Ambiguity Index

        Parameters:
        -----------
        ambiguity_df : pandas DataFrame
            Individual ambiguity indices
        market_caps : pandas DataFrame or None
            Market capitalizations (required if weighted=True)

        Returns:
        --------
        sca : pandas Series
            SCA time series
        """
        print("\nComputing Systemic Co-Ambiguity Index...")
        print(f"  Method: {'Weighted' if self.weighted_sca else 'Unweighted'}")
        print(f"  Correlation window: {self.corr_window} days")

        if self.weighted_sca and market_caps is not None:
            sca = self.sca_calculator.compute_sca_efficient(ambiguity_df, market_caps)
        else:
            sca = self.sca_calculator.compute_sca_efficient(ambiguity_df)

        # Remove NaN values
        valid_sca = sca.dropna()

        print(f"  SCA computed for {len(valid_sca)} days")
        print(f"  Mean SCA: {valid_sca.mean():.6f}")
        print(f"  Std SCA: {valid_sca.std():.6f}")
        print(f"  Min SCA: {valid_sca.min():.6f}")
        print(f"  Max SCA: {valid_sca.max():.6f}")

        self.sca = sca
        self.ambiguity_df = ambiguity_df

        return sca

    def run_hypothesis_tests(self, returns_df, market_index, liquidity_df,
                            volatility, vix, structural_change_periods):
        """
        Run all five hypothesis tests

        Parameters:
        -----------
        returns_df : pandas DataFrame
            Stock returns
        market_index : pandas Series
            Market index
        liquidity_df : pandas DataFrame
            Liquidity measures
        volatility : pandas Series
            Realized volatility
        vix : pandas Series
            VIX or implied volatility
        structural_change_periods : list
            List of structural change periods

        Returns:
        --------
        results : dict
            All hypothesis test results
        """
        print("\n" + "="*70)
        print("RUNNING HYPOTHESIS TESTS")
        print("="*70)

        # Initialize tester
        tester = CoAmbiguityHypothesisTests(
            sca_series=self.sca,
            ambiguity_df=self.ambiguity_df,
            returns_df=returns_df,
            market_index_df=market_index,
            liquidity_df=liquidity_df,
            volatility_df=volatility,
            vix_df=vix
        )

        results = {}

        # Hypothesis 1: Leading Indicator
        print("\n[1/5] Testing Hypothesis 1: Leading Indicator")
        print("-"*70)
        h1_results = tester.test_hypothesis_1_leading_indicator()
        results['hypothesis_1'] = h1_results

        if 't_stat' in h1_results:
            print(f"\nH1 Summary:")
            print(f"  T-statistic: {h1_results['t_stat']:.4f}")
            print(f"  P-value: {h1_results['p_value']:.4f}")
            print(f"  Significant at 5% level: {h1_results['p_value'] < 0.05}")

        # Hypothesis 2: Incremental Power
        print("\n[2/5] Testing Hypothesis 2: Incremental Power")
        print("-"*70)
        h2_results = tester.test_hypothesis_2_incremental_power()
        results['hypothesis_2'] = h2_results

        if h2_results:
            print(f"\nH2 Summary:")
            for horizon, h2_res in h2_results.items():
                print(f"  Horizon {horizon} days:")
                print(f"    SCA coefficient: {h2_res['sca_coefficient']:.6f}")
                print(f"    SCA p-value: {h2_res['sca_p_value']:.4f}")
                print(f"    Incremental pseudo-R2: {h2_res['incremental_r2']:.4f}")
                print(f"    AUC improvement: {h2_res['auc_improvement']:.4f}")

        # Hypothesis 3: Liquidity Channel
        print("\n[3/5] Testing Hypothesis 3: Liquidity Channel")
        print("-"*70)
        h3_results = tester.test_hypothesis_3_liquidity_channel()
        results['hypothesis_3'] = h3_results

        if h3_results:
            print(f"\nH3 Summary:")
            for metric, h3_res in h3_results.items():
                print(f"  Metric: {metric}")
                for lag, lag_res in h3_res.items():
                    if 'f_statistic' in lag_res:
                        print(f"    {lag}: F={lag_res['f_statistic']:.4f}, p={lag_res['p_value']:.4f}")

        # Hypothesis 4: Structural Change
        print("\n[4/5] Testing Hypothesis 4: Structural Change")
        print("-"*70)
        h4_results = tester.test_hypothesis_4_structural_change(
            structural_change_periods
        )
        results['hypothesis_4'] = h4_results

        if h4_results:
            print(f"\nH4 Summary:")
            if 'interaction_coefficient' in h4_results:
                print(f"  Interaction coefficient: {h4_results['interaction_coefficient']:.6f}")
                print(f"  Interaction p-value: {h4_results['interaction_p_value']:.4f}")
            if 'sc_sca_coef' in h4_results:
                print(f"  SCA effect in SC periods: {h4_results['sc_sca_coef']:.6f}")
                print(f"  SCA effect in stable periods: {h4_results['stable_sca_coef']:.6f}")

        # Hypothesis 5: Volatility Lead
        print("\n[5/5] Testing Hypothesis 5: Volatility Lead")
        print("-"*70)
        h5_results = tester.test_hypothesis_5_volatility_lead()
        results['hypothesis_5'] = h5_results

        if h5_results:
            print(f"\nH5 Summary:")
            for lag, h5_res in h5_results.items():
                print(f"  {lag}:")
                print(f"    SCA -> Vol: F={h5_res['sca_to_vol_f_stat']:.4f}, p={h5_res['sca_to_vol_p_value']:.4f}")
                print(f"    Vol -> SCA: F={h5_res['vol_to_sca_f_stat']:.4f}, p={h5_res['vol_to_sca_p_value']:.4f}")

        print("\n" + "="*70)
        print("HYPOTHESIS TESTS COMPLETE")
        print("="*70)

        self.hypothesis_results = results
        return results

    def run_backtests(self, market_returns):
        """
        Run trading strategy backtests

        Parameters:
        -----------
        market_returns : pandas Series
            Market returns

        Returns:
        --------
        backtest_results : dict
            Backtest results
        """
        print("\n" + "="*70)
        print("RUNNING BACKTESTS")
        print("="*70)

        # Initialize backtester
        backtester = SCABacktester(self.sca, market_returns)

        # Compare strategies
        print("\nComparing strategies...")
        comparison = backtester.compare_strategies()

        for strategy, perf in comparison.items():
            print(f"\n{strategy}:")
            print(f"  Annual Return: {perf['annualized_return']:.2%}")
            print(f"  Sharpe Ratio: {perf['sharpe_ratio']:.4f}")
            print(f"  Calmar Ratio: {perf['calmar_ratio']:.4f}")
            print(f"  Max Drawdown: {perf['max_drawdown']:.2%}")

        # Signal efficiency
        print("\nComputing signal efficiency...")
        signals = backtester.generate_dynamic_threshold_signals()
        efficiency = backtester.compute_signal_efficiency(signals)

        print(f"Accuracy: {efficiency['accuracy']:.2%}")
        print(f"Precision: {efficiency['precision']:.2%}")
        print(f"Recall: {efficiency['recall']:.2%}")

        backtest_results = {
            'strategy_comparison': comparison,
            'signal_efficiency': efficiency,
            'backtester': backtester
        }

        self.backtest_results = backtest_results
        return backtest_results

    def visualize_results(self, save_path=None):
        """
        Create comprehensive visualizations

        Parameters:
        -----------
        save_path : str or None
            Path to save figure
        """
        print("\nGenerating visualizations...")

        fig, axes = plt.subplots(3, 2, figsize=(15, 12))

        # Plot 1: SCA time series
        ax1 = axes[0, 0]
        ax1.plot(self.sca.index, self.sca.values, alpha=0.7)
        ax1.set_title('Systemic Co-Ambiguity Index')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('SCA')
        ax1.grid(True, alpha=0.3)

        # Plot 2: SCA vs Market Index
        ax2 = axes[0, 1]
        ax2_twin = ax2.twinx()
        ax2.plot(self.sca.index, self.sca.values, 'b-', alpha=0.7, label='SCA')
        ax2_twin.plot(self.backtest_results['backtester'].market_returns.index,
                      (1 + self.backtest_results['backtester'].market_returns).cumprod().values,
                      'r-', alpha=0.7, label='Market Index')
        ax2.set_title('SCA vs Market Index')
        ax2.set_ylabel('SCA', color='b')
        ax2_twin.set_ylabel('Market Index', color='r')
        ax2.grid(True, alpha=0.3)

        # Plot 3: Strategy comparison
        ax3 = axes[1, 0]
        comparison = self.backtest_results['strategy_comparison']
        strategies = list(comparison.keys())
        metrics = ['annualized_return', 'sharpe_ratio', 'calmar_ratio']
        x = np.arange(len(strategies))
        width = 0.25

        for i, metric in enumerate(metrics):
            values = [comparison[s].get(metric, 0) for s in strategies]
            if metric == 'annualized_return':
                values = [v * 100 for v in values]
            ax3.bar(x + i * width, values, width, label=metric, alpha=0.7)

        ax3.set_title('Strategy Performance Metrics')
        ax3.set_xlabel('Strategy')
        ax3.set_ylabel('Value (%)')
        ax3.set_xticks(x + width)
        ax3.set_xticklabels(strategies, rotation=45, ha='right')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')

        # Plot 4: Drawdown comparison
        ax4 = axes[1, 1]
        for strategy, results in comparison.items():
            if 'cumulative_returns' in results:
                cumulative = results['cumulative_returns']
                drawdown = cumulative / cumulative.cummax() - 1
                ax4.plot(drawdown.index, drawdown.values * 100, label=strategy, alpha=0.7)
        ax4.set_title('Drawdown Comparison')
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Drawdown (%)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # Plot 5: SCA distribution
        ax5 = axes[2, 0]
        ax5.hist(self.sca.dropna().values, bins=50, alpha=0.7, edgecolor='black')
        ax5.axvline(self.sca.mean(), color='red', linestyle='--', label='Mean')
        ax5.axvline(self.sca.quantile(0.9), color='orange', linestyle='--', label='90th percentile')
        ax5.set_title('SCA Distribution')
        ax5.set_xlabel('SCA Value')
        ax5.set_ylabel('Frequency')
        ax5.legend()
        ax5.grid(True, alpha=0.3, axis='y')

        # Plot 6: Signal efficiency summary
        ax6 = axes[2, 1]
        eff = self.backtest_results['signal_efficiency']
        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        metrics_values = [
            eff.get('accuracy', 0) * 100,
            eff.get('precision', 0) * 100,
            eff.get('recall', 0) * 100,
            eff.get('f1_score', 0) * 100
        ]
        ax6.bar(metrics_names, metrics_values, alpha=0.7, edgecolor='black')
        ax6.set_title('Signal Efficiency Metrics')
        ax6.set_ylabel('Value (%)')
        ax6.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")

        plt.show()

    def generate_report(self, save_path=None):
        """
        Generate comprehensive text report

        Parameters:
        -----------
        save_path : str or None
            Path to save report

        Returns:
        --------
        report : str
            Text report
        """
        report_lines = []
        report_lines.append("="*70)
        report_lines.append("SYSTEMIC CO-AMBIGUITY RESEARCH - FINAL REPORT")
        report_lines.append("="*70)
        report_lines.append("")

        # Executive Summary
        report_lines.append("EXECUTIVE SUMMARY")
        report_lines.append("-"*70)
        report_lines.append(f"Analysis Period: {self.sca.index[0]} to {self.sca.index[-1]}")
        report_lines.append(f"Correlation Window: {self.corr_window} days")
        report_lines.append(f"SCA Method: {'Weighted' if self.weighted_sca else 'Unweighted'}")
        report_lines.append("")

        # SCA Statistics
        report_lines.append("SYSTEMIC CO-AMBIGUITY STATISTICS")
        report_lines.append("-"*70)
        sca_valid = self.sca.dropna()
        report_lines.append(f"Observations: {len(sca_valid):,.0f}")
        report_lines.append(f"Mean: {sca_valid.mean():.6f}")
        report_lines.append(f"Std Dev: {sca_valid.std():.6f}")
        report_lines.append(f"Min: {sca_valid.min():.6f}")
        report_lines.append(f"Max: {sca_valid.max():.6f}")
        report_lines.append(f"Skewness: {sca_valid.skew():.4f}")
        report_lines.append(f"Kurtosis: {sca_valid.kurtosis():.4f}")
        report_lines.append("")

        # Hypothesis Test Results
        report_lines.append("HYPOTHESIS TEST RESULTS")
        report_lines.append("-"*70)

        for h_num, h_results in self.hypothesis_results.items():
            report_lines.append(f"\n{h_num.upper().replace('_', ' ')}")
            report_lines.append("-"*40)

            if h_num == 'hypothesis_1':
                if 't_stat' in h_results:
                    report_lines.append(f"Events Identified: {h_results.get('n_events', 0)}")
                    report_lines.append(f"Pre-event T-statistic: {h_results['t_stat']:.4f}")
                    report_lines.append(f"P-value: {h_results['p_value']:.4f}")
                    report_lines.append(f"Significant (5%): {h_results['p_value'] < 0.05}")

            elif h_num == 'hypothesis_2':
                for horizon, h2_res in h_results.items():
                    report_lines.append(f"\n  Horizon {horizon} days:")
                    report_lines.append(f"    SCA Coefficient: {h2_res['sca_coefficient']:.6f}")
                    report_lines.append(f"    SCA P-value: {h2_res['sca_p_value']:.4f}")
                    report_lines.append(f"    Incremental R²: {h2_res['incremental_r2']:.4f}")
                    report_lines.append(f"    AUC Improvement: {h2_res['auc_improvement']:.4f}")

            elif h_num == 'hypothesis_3':
                for metric, h3_res in h_results.items():
                    report_lines.append(f"\n  Metric: {metric}")
                    for lag, lag_res in h3_res.items():
                        if 'f_statistic' in lag_res:
                            report_lines.append(f"    {lag}: F={lag_res['f_statistic']:.4f}, p={lag_res['p_value']:.4f}")

            elif h_num == 'hypothesis_4':
                if 'interaction_coefficient' in h_results:
                    report_lines.append(f"Interaction Coefficient: {h_results['interaction_coefficient']:.6f}")
                    report_lines.append(f"Interaction P-value: {h_results['interaction_p_value']:.4f}")
                if 'sc_sca_coef' in h_results:
                    report_lines.append(f"SC Period SCA Effect: {h_results['sc_sca_coef']:.6f}")
                    report_lines.append(f"Stable Period SCA Effect: {h_results['stable_sca_coef']:.6f}")

            elif h_num == 'hypothesis_5':
                for lag, h5_res in h_results.items():
                    report_lines.append(f"\n  {lag}:")
                    report_lines.append(f"    SCA→Vol: F={h5_res['sca_to_vol_f_stat']:.4f}, p={h5_res['sca_to_vol_p_value']:.4f}")
                    report_lines.append(f"    Vol→SCA: F={h5_res['vol_to_sca_f_stat']:.4f}, p={h5_res['vol_to_sca_p_value']:.4f}")

        report_lines.append("")

        # Backtest Results
        report_lines.append("BACKTEST RESULTS")
        report_lines.append("-"*70)
        comparison = self.backtest_results['strategy_comparison']

        for strategy, perf in comparison.items():
            report_lines.append(f"\n{strategy}:")
            report_lines.append(f"  Annual Return: {perf['annualized_return']:.2%}")
            report_lines.append(f"  Volatility: {perf['volatility']:.2%}")
            report_lines.append(f"  Sharpe Ratio: {perf['sharpe_ratio']:.4f}")
            report_lines.append(f"  Calmar Ratio: {perf['calmar_ratio']:.4f}")
            report_lines.append(f"  Max Drawdown: {perf['max_drawdown']:.2%}")

        # Signal Efficiency
        eff = self.backtest_results['signal_efficiency']
        report_lines.append(f"\nSignal Efficiency:")
        report_lines.append(f"  Accuracy: {eff['accuracy']:.2%}")
        report_lines.append(f"  Precision: {eff['precision']:.2%}")
        report_lines.append(f"  Recall: {eff['recall']:.2%}")
        report_lines.append(f"  F1 Score: {eff['f1_score']:.4f}")

        report_lines.append("")
        report_lines.append("="*70)
        report_lines.append("END OF REPORT")
        report_lines.append("="*70)

        report = "\n".join(report_lines)

        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            print(f"\nReport saved to {save_path}")

        return report

    def run_complete_pipeline(self):
        """
        Run the complete analysis pipeline

        Returns:
        --------
        results : dict
            All results
        """
        print("\n" + "="*70)
        print("SYSTEMIC CO-AMBIGUITY RESEARCH PIPELINE")
        print("="*70)

        # Step 1: Load data
        data = self.load_data()

        # Step 2: Compute SCA
        self.compute_sca(
            data['ambiguity_df'],
            market_caps=data.get('market_caps')
        )

        # Step 3: Run hypothesis tests
        self.run_hypothesis_tests(
            returns_df=data['returns_df'],
            market_index=data['market_index'],
            liquidity_df=data['liquidity_df'],
            volatility=data['volatility'],
            vix=data['vix'],
            structural_change_periods=data['structural_change_periods']
        )

        # Step 4: Run backtests
        market_returns = data['market_index'].pct_change().fillna(0)
        self.run_backtests(market_returns)

        # Step 5: Visualize results
        self.visualize_results()

        # Step 6: Generate report
        report = self.generate_report()
        print("\n" + report)

        # Compile all results
        results = {
            'sca': self.sca,
            'hypothesis_tests': self.hypothesis_results,
            'backtests': self.backtest_results,
            'data': data
        }

        return results


if __name__ == "__main__":
    print("Systemic Co-Ambiguity Research Pipeline")
    print("=" * 70)

    # Initialize pipeline
    pipeline = CoAmbiguityResearchPipeline(
        data_path=None,  # Set to actual data path if available
        corr_window=60,
        weighted_sca=False
    )

    # Run complete pipeline
    results = pipeline.run_complete_pipeline()

    print("\n" + "="*70)
    print("PIPELINE EXECUTION COMPLETE")
    print("="*70)
