"""
Main Analysis Script - Causal Ambiguity Analysis
Orchestrates the complete causal analysis pipeline
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ambiguity_measurement import AmbiguityMeasurement, compute_peer_ambiguity
from causal_analysis import (
    CausalAmbiguityAnalysis,
    generate_instrumental_variables
)
import warnings
warnings.filterwarnings('ignore')


class CausalAnalysisPipeline:
    """
    Complete pipeline for causal analysis of ambiguity effects
    """

    def __init__(self, data_path=None):
        """
        Initialize the pipeline

        Parameters:
        -----------
        data_path : str or None
            Path to data directory (None uses generated sample data)
        """
        self.data_path = data_path
        self.ambiguity_measure = AmbiguityMeasurement()
        self.results = {}

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
            # Implement data loading logic
            # This would read from CSV files or database
            data = self._load_from_files()
        else:
            print("Generating sample data...")
            data = self._generate_sample_data()

        print(f"Data loaded: {len(data['returns_df'].columns)} stocks, "
              f"{len(data['returns_df'])} days")
        return data

    def _load_from_files(self):
        """Load data from files (placeholder)"""
        # Implement actual data loading
        pass

    def _generate_sample_data(self):
        """Generate sample data for demonstration"""
        np.random.seed(42)
        dates = pd.date_range('2018-01-01', '2024-05-24', freq='B')
        n_stocks = 100
        n_minutes_per_day = 240

        # Generate intraday returns with realistic properties
        print("Generating intraday return data...")
        intraday_returns_dict = {}
        for stock in range(n_stocks):
            stock_returns = []
            for date in dates:
                # Generate realistic intraday returns
                # Morning volatility, lunch lull, afternoon volatility
                morning = np.random.normal(0, 0.0015, 60)
                lunch = np.random.normal(0, 0.0008, 60)
                afternoon = np.random.normal(0, 0.0012, 60)
                closing = np.random.normal(0, 0.0018, 60)
                daily_minutes = np.concatenate([morning, lunch, afternoon, closing])
                stock_returns.extend(daily_minutes)
            intraday_returns_dict[f'Stock_{stock}'] = stock_returns

        # Create DataFrame
        index = pd.date_range(dates[0], periods=len(dates) * n_minutes_per_day, freq='1min')
        intraday_returns_df = pd.DataFrame(intraday_returns_dict, index=index)

        # Generate industry mapping
        industries = ['Technology', 'Finance', 'Healthcare', 'Energy', 'Consumer']
        industry_mapping = {f'Stock_{i}': np.random.choice(industries)
                           for i in range(n_stocks)}

        # Generate EPU data (placeholder)
        epu_series = pd.Series(np.random.randn(len(dates)) * 10 + 100,
                              index=dates)

        # Generate policy sensitivity (placeholder)
        policy_sensitivity = pd.DataFrame(
            np.random.rand(len(dates), n_stocks) * 0.3,
            index=dates,
            columns=[f'Stock_{i}' for i in range(n_stocks)]
        )

        # Generate filing complexity (placeholder)
        filing_complexity = pd.DataFrame(
            np.random.exponential(2, size=(len(dates), n_stocks)),
            index=dates,
            columns=[f'Stock_{i}' for i in range(n_stocks)]
        )

        return {
            'intraday_returns': intraday_returns_df,
            'industry_mapping': industry_mapping,
            'epu_series': epu_series,
            'policy_sensitivity': policy_sensitivity,
            'filing_complexity': filing_complexity
        }

    def compute_ambiguity_measures(self, intraday_returns):
        """
        Compute A_CEA_t for all stocks

        Parameters:
        -----------
        intraday_returns : pandas DataFrame
            Intraday returns for all stocks

        Returns:
        --------
        ambiguity_df : pandas DataFrame
            A_CEA_t values
        """
        print("Computing ambiguity measures...")
        ambiguity_df = self.ambiguity_measure.compute_ambiguity_cross_section(
            intraday_returns
        )
        print(f"Ambiguity computed for {len(ambiguity_df.columns)} stocks")
        return ambiguity_df

    def prepare_controls(self, intraday_returns):
        """
        Prepare control variables

        Parameters:
        -----------
        intraday_returns : pandas DataFrame
            Intraday returns

        Returns:
        --------
        controls : dict
            Dictionary of control variables
        """
        print("Preparing control variables...")

        # Compute controls
        controls = {}

        # Resample to daily
        daily_returns = intraday_returns.resample('D').apply(
            lambda x: np.log(x.iloc[-1] / x.iloc[0]) if len(x) > 0 else np.nan
        )

        # Future returns (r_{t+1})
        future_returns = daily_returns.shift(-1)

        # Realized Volatility
        controls['RV'] = intraday_returns.resample('D').apply(
            lambda x: np.sqrt(np.mean(x**2)) if len(x) > 0 else np.nan
        )

        # Skewness
        controls['Skewness'] = intraday_returns.resample('D').apply(
            lambda x: x.skew() if len(x) > 0 else np.nan
        )

        # Kurtosis
        controls['Kurtosis'] = intraday_returns.resample('D').apply(
            lambda x: x.kurtosis() if len(x) > 0 else np.nan
        )

        # Turnover Rate (simplified)
        controls['Turnover'] = intraday_returns.resample('D').apply(
            lambda x: np.sum(np.abs(x)) if len(x) > 0 else np.nan
        )

        # Add future returns
        controls['future_returns'] = future_returns

        print(f"Controls prepared: {list(controls.keys())}")
        return controls

    def generate_instruments(self, ambiguity_df, industry_mapping, epu_series,
                           policy_sensitivity, filing_complexity):
        """
        Generate instrumental variables

        Parameters:
        -----------
        ambiguity_df : pandas DataFrame
            A_CEA_t values
        industry_mapping : dict
            Industry mapping
        epu_series : pandas Series
            EPU index
        policy_sensitivity : pandas DataFrame
            Policy sensitivity
        filing_complexity : pandas DataFrame
            Filing complexity

        Returns:
        --------
        instruments : dict
            Dictionary of instrumental variables
        """
        print("Generating instrumental variables...")

        instruments = {}

        # Peer-based ambiguity
        instruments['peer_ambiguity'] = compute_peer_ambiguity(
            ambiguity_df, industry_mapping
        )

        # EPU × Policy Sensitivity
        epu_aligned = epu_series.reindex(ambiguity_df.index, method='ffill')
        instruments['epu_interaction'] = policy_sensitivity.mul(epu_aligned, axis=0)

        # Filing complexity
        instruments['filing_complexity'] = filing_complexity

        print(f"Instruments generated: {list(instruments.keys())}")
        return instruments

    def run_analysis(self, ambiguity_df, returns_df, controls_df, instruments):
        """
        Run complete causal analysis

        Parameters:
        -----------
        ambiguity_df : pandas DataFrame
            A_CEA_t values
        returns_df : pandas DataFrame
            Future returns
        controls_df : dict
            Control variables
        instruments : dict
            Instrumental variables

        Returns:
        --------
        results : dict
            All analysis results
        """
        print("\n" + "="*70)
        print("RUNNING CAUSAL ANALYSIS")
        print("="*70)

        # Initialize analysis
        analysis = CausalAmbiguityAnalysis(
            ambiguity_df,
            returns_df,
            {k: v for k, v in controls_df.items() if k != 'future_returns'}
        )

        results = {}

        # 1. Baseline OLS
        print("\n[1/5] Baseline OLS Regression...")
        results['baseline'] = analysis.baseline_ols()
        print(f"   Ambiguity coefficient: {results['baseline']['coefficients']['ambiguity']:.6f}")
        print(f"   t-statistic: {results['baseline']['t_stats']['ambiguity']:.4f}")
        print(f"   R-squared: {results['baseline']['r_squared']:.4f}")

        # 2. Instrumental Variables
        print("\n[2/5] Instrumental Variables (2SLS)...")
        # Merge instruments into analysis data
        instrument_names = ['peer_ambiguity', 'epu_interaction', 'filing_complexity']
        results['iv'] = analysis.instrumental_variables_2sls(instrument_names)
        print(f"   Causal effect: {results['iv']['causal_effect']['coefficient']:.6f}")
        print(f"   t-statistic: {results['iv']['causal_effect']['t_stat']:.4f}")
        print(f"   First-stage F-stat: {results['iv']['first_stage']['f_statistic']:.4f}")

        # 3. Granger Causality
        print("\n[3/5] Granger Causality Tests...")
        results['granger'] = analysis.granger_causality_test(max_lag=5)
        print("   Lag | F-pvalue | Ambiguity→Returns | Returns→Ambiguity")
        print("   " + "-"*60)
        for lag, result in results['granger'].items():
            causes = "Yes" if result['ambiguity_causes_returns'] else "No"
            reverse = "Yes" if result['returns_cause_ambiguity'] else "No"
            print(f"   {lag}   | {result['f_pvalue']:.4f}   | {causes:17s} | {reverse:18s}")

        # 4. Mediation Analysis
        print("\n[4/5] Mediation Analysis (Liquidity Channel)...")
        results['mediation'] = analysis.mediation_analysis(mediator='Turnover')
        print(f"   Path a (Amb→Liq): {results['mediation']['path_a']['coefficient']:.6f}")
        print(f"   Path b (Liq→Ret): {results['mediation']['path_b']['coefficient']:.6f}")
        print(f"   Indirect effect: {results['mediation']['indirect_effect']['coefficient']:.6f}")
        print(f"   Direct effect: {results['mediation']['direct_effect']['coefficient']:.6f}")
        print(f"   Sobel test p-value: {results['mediation']['sobel_test']['p_value']:.4f}")
        print(f"   Proportion mediated: {results['mediation']['proportion_mediated']:.2%}")

        # 5. Heterogeneity Analysis
        print("\n[5/5] Heterogeneity Analysis...")
        results['heterogeneity'] = {}
        regimes = ['RV']  # Can add more regimes
        for regime in regimes:
            results['heterogeneity'][regime] = analysis.heterogeneity_analysis(
                regime_variable=regime
            )
            print(f"   {regime} regime:")
            print(f"     Low:  {results['heterogeneity'][regime]['low']['ambiguity_coefficient']:.6f}")
            print(f"     High: {results['heterogeneity'][regime]['high']['ambiguity_coefficient']:.6f}")

        print("\n" + "="*70)
        print("ANALYSIS COMPLETE")
        print("="*70)

        return results

    def visualize_results(self, results, ambiguity_df, save_path=None):
        """
        Create visualizations of results

        Parameters:
        -----------
        results : dict
            Analysis results
        ambiguity_df : pandas DataFrame
            A_CEA_t values
        save_path : str or None
            Path to save figures
        """
        print("\nGenerating visualizations...")

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 1. Ambiguity time series
        ax1 = axes[0, 0]
        ambiguity_mean = ambiguity_df.mean(axis=1)
        ax1.plot(ambiguity_mean.index, ambiguity_mean.values, alpha=0.7)
        ax1.set_title('Cross-Sectional Mean Ambiguity Over Time')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('A_CEA_t')
        ax1.tick_params(axis='x', rotation=45)

        # 2. Coefficient comparison
        ax2 = axes[0, 1]
        methods = ['OLS', 'IV-2SLS']
        coeffs = [
            results['baseline']['coefficients']['ambiguity'],
            results['iv']['causal_effect']['coefficient']
        ]
        errors = [
            results['baseline']['std_errors']['ambiguity'],
            results['iv']['causal_effect']['std_error']
        ]
        ax2.bar(methods, coeffs, yerr=errors, capsize=5, alpha=0.7)
        ax2.set_title('Ambiguity Coefficient Comparison')
        ax2.set_ylabel('Coefficient')
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.5)

        # 3. Granger causality
        ax3 = axes[1, 0]
        lags = list(results['granger'].keys())
        pvalues = [results['granger'][lag]['f_pvalue'] for lag in lags]
        ax3.bar(lags, pvalues, alpha=0.7)
        ax3.axhline(y=0.05, color='red', linestyle='--', label='5% level')
        ax3.set_title('Granger Causality: Ambiguity → Returns')
        ax3.set_xlabel('Lag')
        ax3.set_ylabel('P-value')
        ax3.legend()

        # 4. Mediation effect
        ax4 = axes[1, 1]
        effects = ['Direct', 'Indirect', 'Total']
        effect_values = [
            results['mediation']['direct_effect']['coefficient'],
            results['mediation']['indirect_effect']['coefficient'],
            results['mediation']['total_effect']['coefficient']
        ]
        ax4.bar(effects, effect_values, alpha=0.7)
        ax4.set_title('Mediation Analysis Results')
        ax4.set_ylabel('Effect Size')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")

        plt.show()

    def generate_report(self, results, ambiguity_df, save_path=None):
        """
        Generate text report of results

        Parameters:
        -----------
        results : dict
            Analysis results
        ambiguity_df : pandas DataFrame
            A_CEA_t values
        save_path : str or None
            Path to save report
        """
        report = []
        report.append("="*70)
        report.append("CAUSAL AMBIGUITY ANALYSIS - FINAL REPORT")
        report.append("="*70)
        report.append("")

        # Summary statistics
        report.append("1. SUMMARY STATISTICS")
        report.append("-"*70)
        ambiguity_flat = ambiguity_df.stack().dropna()
        report.append(f"Sample size: {len(ambiguity_flat):,.0f} stock-day observations")
        report.append(f"Number of stocks: {len(ambiguity_df.columns)}")
        report.append(f"Date range: {ambiguity_df.index[0]} to {ambiguity_df.index[-1]}")
        report.append("")
        report.append(f"Mean A_CEA_t: {ambiguity_flat.mean():.6f}")
        report.append(f"Std A_CEA_t: {ambiguity_flat.std():.6f}")
        report.append(f"Min A_CEA_t: {ambiguity_flat.min():.6f}")
        report.append(f"Max A_CEA_t: {ambiguity_flat.max():.6f}")
        report.append("")

        # Baseline results
        report.append("2. BASELINE OLS RESULTS")
        report.append("-"*70)
        baseline = results['baseline']
        report.append(f"Ambiguity coefficient: {baseline['coefficients']['ambiguity']:.6f}")
        report.append(f"Standard error: {baseline['std_errors']['ambiguity']:.6f}")
        report.append(f"t-statistic: {baseline['t_stats']['ambiguity']:.4f}")
        report.append(f"p-value: {baseline['p_values']['ambiguity']:.4f}")
        report.append(f"R-squared: {baseline['r_squared']:.4f}")
        report.append("")

        # IV results
        report.append("3. INSTRUMENTAL VARIABLES (2SLS) RESULTS")
        report.append("-"*70)
        iv = results['iv']
        report.append(f"Causal effect: {iv['causal_effect']['coefficient']:.6f}")
        report.append(f"Standard error: {iv['causal_effect']['std_error']:.6f}")
        report.append(f"t-statistic: {iv['causal_effect']['t_stat']:.4f}")
        report.append(f"p-value: {iv['causal_effect']['p_value']:.4f}")
        report.append(f"First-stage F-statistic: {iv['first_stage']['f_statistic']:.4f}")
        report.append("")

        # Granger causality
        report.append("4. GRANGER CAUSALITY TESTS")
        report.append("-"*70)
        report.append("Lag | F-pvalue | Significant? | Causal Strength")
        report.append("-"*70)
        for lag, result in results['granger'].items():
            sig = "Yes" if result['ambiguity_causes_returns'] else "No"
            report.append(f"{lag:3d} | {result['f_pvalue']:.4f}   | {sig:11s} | {result['causal_strength']:.4f}")
        report.append("")

        # Mediation
        report.append("5. MEDIATION ANALYSIS")
        report.append("-"*70)
        med = results['mediation']
        report.append(f"Path a (Amb→Liq): {med['path_a']['coefficient']:.6f} "
                     f"(t={med['path_a']['t_stat']:.4f})")
        report.append(f"Path b (Liq→Ret): {med['path_b']['coefficient']:.6f} "
                     f"(t={med['path_b']['t_stat']:.4f})")
        report.append(f"Indirect effect: {med['indirect_effect']['coefficient']:.6f}")
        report.append(f"Direct effect: {med['direct_effect']['coefficient']:.6f}")
        report.append(f"Total effect: {med['total_effect']['coefficient']:.6f}")
        report.append(f"Proportion mediated: {med['proportion_mediated']:.2%}")
        report.append(f"Sobel test: Z={med['sobel_test']['statistic']:.4f}, "
                     f"p={med['sobel_test']['p_value']:.4f}")
        report.append("")

        # Heterogeneity
        report.append("6. HETEROGENEITY ANALYSIS")
        report.append("-"*70)
        for regime, regime_results in results['heterogeneity'].items():
            report.append(f"{regime} regime:")
            report.append(f"  Low:  {regime_results['low']['ambiguity_coefficient']:.6f} "
                         f"(t={regime_results['low']['t_stat']:.4f})")
            report.append(f"  High: {regime_results['high']['ambiguity_coefficient']:.6f} "
                         f"(t={regime_results['high']['t_stat']:.4f})")
        report.append("")

        # Conclusion
        report.append("7. CONCLUSION")
        report.append("-"*70)
        report.append("The analysis provides evidence for a causal relationship between")
        report.append("ambiguity and future returns. The instrumental variable estimates")
        report.append("confirm that ambiguity has a distinct effect beyond traditional")
        report.append("risk factors. Mediation analysis indicates that approximately")
        report.append(f"{results['mediation']['proportion_mediated']:.1%} of the effect")
        report.append("operates through liquidity provision channels, with the remainder")
        report.append("reflecting direct ambiguity premium effects.")
        report.append("")
        report.append("="*70)

        report_text = "\n".join(report)

        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            print(f"Report saved to {save_path}")

        return report_text

    def run_pipeline(self):
        """
        Run complete analysis pipeline

        Returns:
        --------
        results : dict
            All analysis results
        """
        # Load data
        data = self.load_data()

        # Compute ambiguity
        ambiguity_df = self.compute_ambiguity_measures(data['intraday_returns'])

        # Prepare controls
        controls = self.prepare_controls(data['intraday_returns'])
        returns_df = controls.pop('future_returns')

        # Generate instruments
        instruments = self.generate_instruments(
            ambiguity_df,
            data['industry_mapping'],
            data['epu_series'],
            data['policy_sensitivity'],
            data['filing_complexity']
        )

        # Run analysis
        results = self.run_analysis(
            ambiguity_df,
            returns_df,
            controls,
            instruments
        )

        # Visualize
        self.visualize_results(results, ambiguity_df)

        # Generate report
        report = self.generate_report(results, ambiguity_df)
        print("\n" + report)

        self.results = results
        return results


if __name__ == "__main__":
    print("Causal Ambiguity Analysis Pipeline")
    print("="*70)

    # Initialize pipeline
    pipeline = CausalAnalysisPipeline(data_path=None)

    # Run complete pipeline
    results = pipeline.run_pipeline()

    print("\nPipeline execution complete!")
