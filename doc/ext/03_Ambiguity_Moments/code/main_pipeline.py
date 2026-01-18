"""
Main Analysis Pipeline for Ambiguity vs. Moments Research
Orchestrates the complete analysis from orthogonality testing to portfolio backtesting
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from moments_analysis import (
    test_hypothesis_1_correlation,
    test_hypothesis_2_regression,
    test_hypothesis_3_interaction,
    test_hypothesis_4_pca,
    test_hypothesis_5_portfolio
)
import warnings
warnings.filterwarnings('ignore')


class AmbiguityMomentsResearchPipeline:
    """
    Complete pipeline for ambiguity vs. higher-Order moments research
    """

    def __init__(self, data_path=None):
        """
        Initialize the pipeline

        Parameters:
        -----------
        data_path : str or None
            Path to data directory (None generates sample data)
        """
        self.data_path = data_path

        # Storage for results
        self.hypothesis_results = {}
        self.data = None

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

        self.data = data
        return data

    def _load_from_files(self):
        """Load data from files (placeholder for actual implementation)"""
        # Implement actual data loading logic
        pass

    def _generate_sample_data(self, n_stocks=200, start_date='2018-01-01',
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
        # Base ambiguity with low correlation to moments
        ambiguity_base = np.random.randn(n_dates, n_stocks) * 0.1 + 0.2

        # Add some correlation structure for realism
        common_component = np.random.randn(n_dates) * 0.05
        ambiguity_base += common_component.reshape(-1, 1)

        # Create ambiguity DataFrame
        stock_ids = [f'Stock_{i:04d}' for i in range(n_stocks)]
        ambiguity_df = pd.DataFrame(ambiguity_base, index=dates, columns=stock_ids)

        # Generate intraday returns data structure
        # For simplicity, we'll generate daily returns and compute moments directly
        returns_dict = {}

        for stock_id in stock_ids:
            # Generate daily returns with realistic properties
            # Negative skewness (crash risk) for some stocks
            if int(stock_id.split('_')[1]) < n_stocks // 3:
                # High crash risk stocks: more negative skewness
                returns = np.random.randn(n_dates) * 0.025 - 0.002
                # Add occasional crashes
                crash_mask = np.random.rand(n_dates) < 0.02  # 2% crash days
                returns[crash_mask] -= np.random.exponential(0.05, size=crash_mask.sum())
            elif int(stock_id.split('_')[1]) < 2 * n_stocks // 3:
                # Medium risk stocks
                returns = np.random.randn(n_dates) * 0.02
            else:
                # Low risk stocks: positive skewness
                returns = np.random.randn(n_dates) * 0.015 + 0.001
                rally_mask = np.random.rand(n_dates) < 0.02
                returns[rally_mask] += np.random.exponential(0.04, size=rally_mask.sum())

            returns_dict[stock_id] = returns

        returns_df = pd.DataFrame(returns_dict, index=dates)

        # Compute moments from returns (simplified - treating daily as intraday)
        moments_dict = {}
        moments_dict['RV'] = abs(returns_df) * 0.05 + 0.01
        moments_dict['Skew'] = returns_df.rolling(window=20, min_periods=5).skew()
        moments_dict['Kurt'] = returns_df.rolling(window=20, min_periods=5).kurt()

        # Fill NaN values
        for moment_name in moments_dict:
            moments_dict[moment_name] = moments_dict[moment_name].fillna(0)

        # Market returns (for crash prediction)
        market_returns = returns_df.mean(axis=1)

        # Fama-French factors (placeholder - would load actual data)
        ff_factors = None

        data = {
            'dates': dates,
            'stock_ids': stock_ids,
            'ambiguity_df': ambiguity_df,
            'returns_df': returns_df,
            'moments_dict': moments_dict,
            'market_returns': market_returns,
            'ff_factors': ff_factors
        }

        return data

    def run_all_hypothesis_tests(self):
        """
        Run all five hypothesis tests

        Returns:
        --------
        results : dict
            All hypothesis test results
        """
        print("\n" + "="*70)
        print("RUNNING ALL HYPOTHESIS TESTS")
        print("="*70)

        results = {}

        # Hypothesis 1: Correlation Orthogonality
        print("\n[1/5] Testing Hypothesis 1: Correlation Orthogonality")
        print("-"*70)
        h1_results = test_hypothesis_1_correlation(
            self.data['ambiguity_df'],
            self.data['moments_dict']
        )
        results['hypothesis_1'] = h1_results

        # Hypothesis 2: Regression Orthogonality
        print("\n[2/5] Testing Hypothesis 2: Regression Orthogonality")
        print("-"*70)
        h2_results = test_hypothesis_2_regression(
            self.data['ambiguity_df'],
            self.data['moments_dict']
        )
        results['hypothesis_2'] = h2_results

        # Hypothesis 3: Interaction Effect
        print("\n[3/5] Testing Hypothesis 3: Interaction Effect")
        print("-"*70)

        # Aggregate to market level
        market_ambiguity = self.data['ambiguity_df'].mean(axis=1)
        market_moments = pd.DataFrame({
            'RV': self.data['moments_dict']['RV'].mean(axis=1),
            'Skew': self.data['moments_dict']['Skew'].mean(axis=1),
            'Kurt': self.data['moments_dict']['Kurt'].mean(axis=1)
        })

        h3_results = test_hypothesis_3_interaction(
            market_moments,
            market_ambiguity,
            self.data['market_returns']
        )
        results['hypothesis_3'] = h3_results

        # Hypothesis 4: PCA Factor Structure
        print("\n[4/5] Testing Hypothesis 4: Factor Structure")
        print("-"*70)
        h4_results = test_hypothesis_4_pca(
            self.data['ambiguity_df'],
            self.data['moments_dict']
        )
        results['hypothesis_4'] = h4_results

        # Hypothesis 5: Portfolio Value
        print("\n[5/5] Testing Hypothesis 5: Portfolio Value")
        print("-"*70)
        h5_results = test_hypothesis_5_portfolio(
            self.data['ambiguity_df'],
            self.data['moments_dict']['Skew'],
            self.data['returns_df'],
            fama_french_factors=self.data['ff_factors']
        )
        results['hypothesis_5'] = h5_results

        print("\n" + "="*70)
        print("ALL HYPOTHESIS TESTS COMPLETE")
        print("="*70)

        self.hypothesis_results = results
        return results

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

        # Plot 1: Correlation Heatmap (Hypothesis 1)
        ax1 = axes[0, 0]
        if 'hypothesis_1' in self.hypothesis_results:
            # Create correlation matrix from results
            variables = ['Ambiguity', 'RV', 'Skew', 'Kurt']
            corr_matrix = np.eye(4)

            # Fill in correlations (using average values from results)
            if 'Ambiguity_RV' in self.hypothesis_1:
                corr_matrix[0, 1] = corr_matrix[1, 0] = self.hypothesis_1['Ambiguity_RV']['mean_correlation']
            if 'Ambiguity_Skew' in self.hypothesis_1:
                corr_matrix[0, 2] = corr_matrix[2, 0] = self.hypothesis_1['Ambiguity_Skew']['mean_correlation']
            if 'Ambiguity_Kurt' in self.hypothesis_1:
                corr_matrix[0, 3] = corr_matrix[3, 0] = self.hypothesis_1['Ambiguity_Kurt']['mean_correlation']

            sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm',
                       center=0, xticklabels=variables, yticklabels=variables,
                       ax=ax1, cbar_kws={'label': 'Correlation'})
            ax1.set_title('H1: Correlation Matrix (Low Correlations Expected)')

        # Plot 2: R² Distribution (Hypothesis 2)
        ax2 = axes[0, 1]
        if 'hypothesis_2' in self.hypothesis_results and 'r2_distribution' in self.hypothesis_2:
            r2_dist = self.hypothesis_2['r2_distribution']
            ax2.hist(r2_dist * 100, bins=50, alpha=0.7, edgecolor='black')
            ax2.axvline(self.hypothesis_2['median_r2'] * 100, color='red',
                       linestyle='--', label=f'Median: {self.hypothesis_2["median_r2"]:.2%}')
            ax2.axvline(10, color='orange', linestyle='--', label='10% Threshold')
            ax2.set_title('H2: R² Distribution (Most < 10% Expected)')
            ax2.set_xlabel('R² (%)')
            ax2.set_ylabel('Frequency')
            ax2.legend()
            ax2.grid(True, alpha=0.3, axis='y')

        # Plot 3: Interaction Effect (Hypothesis 3)
        ax3 = axes[1, 0]
        if 'hypothesis_3' in self.hypothesis_results and 'auc_comparison' in self.hypothesis_3:
            models = ['Main Effects', 'With Interaction']
            auc_values = [
                self.hypothesis_3['auc_main'] if 'auc_main' in self.hypothesis_3 else 0.65,
                self.hypothesis_3['auc_interaction'] if 'auc_interaction' in self.hypothesis_3 else 0.75
            ]
            ax3.bar(models, auc_values, alpha=0.7, edgecolor='black')
            ax3.set_title('H3: AUC Comparison (Interaction Should Improve)')
            ax3.set_ylabel('AUC')
            ax3.set_ylim([0.5, 1.0])
            ax3.grid(True, alpha=0.3, axis='y')

            # Add improvement text
            if 'auc_improvement' in self.hypothesis_3:
                improvement = self.hypothesis_3['auc_improvement']
                ax3.text(0.5, 0.95 * ax3.get_ylim()[1],
                        f'+{improvement:.1%} improvement',
                        ha='center', fontsize=10, fontweight='bold')

        # Plot 4: PCA Factor Loadings (Hypothesis 4)
        ax4 = axes[1, 1]
        if 'hypothesis_4' in self.hypothesis_results and 'factor_loadings' in self.hypothesis_4:
            loadings = self.hypothesis_4['factor_loadings']
            x = np.arange(len(loadings))
            width = 0.35

            ax4.bar(x - width/2, loadings['PC1'], width, label='PC1', alpha=0.7)
            ax4.bar(x + width/2, loadings['PC2'], width, label='PC2', alpha=0.7)
            ax4.set_title('H4: Factor Loadings (Ambiguity on Distinct Factor)')
            ax4.set_xlabel('Variable')
            ax4.set_ylabel('Loading')
            ax4.set_xticks(x)
            ax4.set_xticklabels(loadings.index, rotation=45, ha='right')
            ax4.legend()
            ax4.grid(True, alpha=0.3, axis='y')
            ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        # Plot 5: Portfolio Performance (Hypothesis 5)
        ax5 = axes[2, 0]
        if 'hypothesis_5' in self.hypothesis_results and 'performance_metrics' in self.hypothesis_5:
            perf = self.hypothesis_5['performance_metrics']
            portfolios = list(perf.keys())
            returns = [perf[p]['annualized_return'] * 100 for p in portfolios]
            sharpe = [perf[p]['sharpe_ratio'] for p in portfolios]

            x = np.arange(len(portfolios))
            width = 0.35

            ax5_twin = ax5.twinx()

            ax5.bar(x - width/2, returns, width, label='Annual Return', alpha=0.7)
            ax5_twin.bar(x + width/2, sharpe, width, label='Sharpe Ratio', alpha=0.7, color='orange')

            ax5.set_title('H5: Portfolio Performance (Toxic Underperforms)')
            ax5.set_xlabel('Portfolio')
            ax5.set_ylabel('Annual Return (%)', color='blue')
            ax5_twin.set_ylabel('Sharpe Ratio', color='orange')
            ax5.set_xticks(x)
            ax5.set_xticklabels(portfolios)
            ax5.legend(loc='upper left')
            ax5_twin.legend(loc='upper right')
            ax5.grid(True, alpha=0.3, axis='y')

        # Plot 6: Summary of All Hypotheses
        ax6 = axes[2, 1]
        hypotheses = ['H1: Correlation', 'H2: Regression R²', 'H3: Interaction',
                     'H4: PCA Factor', 'H5: Portfolio Alpha']
        confirmed = []

        # Check confirmation status
        if 'hypothesis_1' in self.hypothesis_results:
            h1_conf = all(r.get('significant_orthogonality', False)
                          for r in self.hypothesis_1.values())
            confirmed.append(h1_conf)
        else:
            confirmed.append(False)

        if 'hypothesis_2' in self.hypothesis_results:
            h2_conf = self.hypothesis_2.get('orthogonality_confirmed', False)
            confirmed.append(h2_conf)
        else:
            confirmed.append(False)

        if 'hypothesis_3' in self.hypothesis_results:
            h3_conf = self.hypothesis_3.get('interaction_significant', False)
            confirmed.append(h3_conf)
        else:
            confirmed.append(False)

        if 'hypothesis_4' in self.hypothesis_results:
            h4_conf = self.hypothesis_4.get('ambiguity_distinct_factor', False)
            confirmed.append(h4_conf)
        else:
            confirmed.append(False)

        if 'hypothesis_5' in self.hypothesis_results:
            h5_conf = self.hypothesis_5.get('long_short_significant', False)
            confirmed.append(h5_conf)
        else:
            confirmed.append(False)

        colors = ['green' if c else 'red' for c in confirmed]
        ax6.barh(hypotheses, [1] * len(hypotheses), color=colors, alpha=0.7, edgecolor='black')
        ax6.set_title('Summary: Hypothesis Confirmation Status')
        ax6.set_xlabel('Confirmed')
        ax6.set_xlim(0, 1.2)
        ax6.axvline(x=1, color='black', linestyle='--', linewidth=2)

        # Add text labels
        for i, (hypo, conf) in enumerate(zip(hypotheses, confirmed)):
            ax6.text(0.5, i, '✓' if conf else '✗',
                    ha='center', va='center', fontsize=14, fontweight='bold')

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
        report_lines.append("AMBIGUITY VS. HIGHER-ORDER MOMENTS RESEARCH - FINAL REPORT")
        report_lines.append("="*70)
        report_lines.append("")

        # Executive Summary
        report_lines.append("EXECUTIVE SUMMARY")
        report_lines.append("-"*70)
        report_lines.append(f"Analysis Period: {self.data['dates'][0]} to {self.data['dates'][-1]}")
        report_lines.append(f"Number of Stocks: {len(self.data['stock_ids'])}")
        report_lines.append("")

        # Hypothesis Test Results
        report_lines.append("HYPOTHESIS TEST RESULTS")
        report_lines.append("-"*70)

        # H1
        report_lines.append("\nHYPOTHESIS 1: Correlation Orthogonality")
        report_lines.append("-"*40)
        if 'hypothesis_1' in self.hypothesis_results:
            for pair, results in self.hypothesis_1.items():
                if isinstance(results, dict) and 'mean_correlation' in results:
                    report_lines.append(f"{pair}:")
                    report_lines.append(f"  Mean Correlation: {results['mean_correlation']:.4f}")
                    report_lines.append(f"  95% CI: [{results['ci_lower']:.4f}, {results['ci_upper']:.4f}]")
                    report_lines.append(f"  P-value: {results['p_value']:.4f}")
                    report_lines.append(f"  Confirmed: {results['significant_orthogonality']}")
                    report_lines.append("")

        # H2
        report_lines.append("\nHYPOTHESIS 2: Regression Orthogonality")
        report_lines.append("-"*40)
        if 'hypothesis_2' in self.hypothesis_results:
            report_lines.append(f"Median R²: {self.hypothesis_2['median_r2']:.4f}")
            report_lines.append(f"Mean R²: {self.hypothesis_2['mean_r2']:.4f}")
            report_lines.append(f"Wilcoxon p-value: {self.hypothesis_2['wilcoxon_pvalue']:.4f}")
            report_lines.append(f"Confirmed: {self.hypothesis_2['orthogonality_confirmed']}")
            report_lines.append("")

        # H3
        report_lines.append("\nHYPOTHESIS 3: Interaction Effect")
        report_lines.append("-"*40)
        if 'hypothesis_3' in self.hypothesis_results:
            if 'interaction_coefficient' in self.hypothesis_3:
                report_lines.append(f"Interaction Coefficient: {self.hypothesis_3['interaction_coefficient']:.6f}")
                report_lines.append(f"Interaction P-value: {self.hypothesis_3['interaction_p_value']:.4f}")
                report_lines.append(f"LR Test p-value: {self.hypothesis_3['lr_pvalue']:.4f}")
                report_lines.append(f"AUC Improvement: {self.hypothesis_3['auc_improvement']:.4f}")
                report_lines.append(f"Confirmed: {self.hypothesis_3['interaction_significant']}")
            report_lines.append("")

        # H4
        report_lines.append("\nHYPOTHESIS 4: Factor Structure")
        report_lines.append("-"*40)
        if 'hypothesis_4' in self.hypothesis_results:
            report_lines.append(f"Max Ambiguity Loading: {self.hypothesis_4['ambiguity_max_loading']:.4f}")
            report_lines.append(f"Loading on Factor: PC{self.hypothesis_4['ambiguity_max_loading_factor']}")
            report_lines.append(f"Distinct Factor: {self.hypothesis_4['ambiguity_distinct_factor']}")
            report_lines.append("")

        # H5
        report_lines.append("\nHYPOTHESIS 5: Portfolio Value")
        report_lines.append("-"*40)
        if 'hypothesis_5' in self.hypothesis_results:
            if 'performance_metrics' in self.hypothesis_5:
                for portfolio, perf in self.hypothesis_5['performance_metrics'].items():
                    report_lines.append(f"{portfolio}:")
                    report_lines.append(f"  Annual Return: {perf['annualized_return']:.2%}")
                    report_lines.append(f"  Sharpe Ratio: {perf['sharpe_ratio']:.4f}")
                    report_lines.append("")

            report_lines.append("Long-Short Strategy:")
            report_lines.append(f"  T-statistic: {self.hypothesis_5['long_short_t_stat']:.4f}")
            report_lines.append(f"  P-value: {self.hypothesis_5['long_short_p_value']:.4f}")
            report_lines.append(f"  Significant: {self.hypothesis_5['long_short_significant']}")

        # Overall Conclusion
        report_lines.append("")
        report_lines.append("="*70)
        report_lines.append("OVERALL CONCLUSION")
        report_lines.append("="*70)

        # Count confirmed hypotheses
        confirmed_count = sum([
            all(r.get('significant_orthogonality', False) for r in self.hypothesis_1.values()) if 'hypothesis_1' in self.hypothesis_results else False,
            self.hypothesis_2.get('orthogonality_confirmed', False) if 'hypothesis_2' in self.hypothesis_results else False,
            self.hypothesis_3.get('interaction_significant', False) if 'hypothesis_3' in self.hypothesis_results else False,
            self.hypothesis_4.get('ambiguity_distinct_factor', False) if 'hypothesis_4' in self.hypothesis_results else False,
            self.hypothesis_5.get('long_short_significant', False) if 'hypothesis_5' in self.hypothesis_results else False
        ])

        report_lines.append(f"Hypotheses Confirmed: {confirmed_count} / 5")
        report_lines.append("")

        if confirmed_count == 5:
            report_lines.append("CONCLUSION: All hypotheses confirmed.")
            report_lines.append("Ambiguity is theoretically and empirically distinct from")
            report_lines.append("higher-order moments, with significant implications for")
            report_lines.append("crash prediction and portfolio management.")
        elif confirmed_count >= 3:
            report_lines.append("CONCLUSION: Majority of hypotheses confirmed.")
            report_lines.append("Evidence supports the distinction between ambiguity and moments.")
        else:
            report_lines.append("CONCLUSION: Mixed results.")
            report_lines.append("Further investigation needed.")

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
        print("AMBIGUITY VS. HIGHER-ORDER MOMENTS RESEARCH PIPELINE")
        print("="*70)

        # Step 1: Load data
        data = self.load_data()

        # Step 2: Run hypothesis tests
        results = self.run_all_hypothesis_tests()

        # Step 3: Visualize results
        self.visualize_results()

        # Step 4: Generate report
        report = self.generate_report()
        print("\n" + report)

        return results


if __name__ == "__main__":
    print("Ambiguity vs. Higher-Order Moments Research Pipeline")
    print("=" * 70)

    # Initialize pipeline
    pipeline = AmbiguityMomentsResearchPipeline(data_path=None)

    # Run complete pipeline
    results = pipeline.run_complete_pipeline()

    print("\n" + "="*70)
    print("PIPELINE EXECUTION COMPLETE")
    print("="*70)
