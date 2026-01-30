"""
Main Analysis Pipeline for Chinese Energy Market Ambiguity Study
Paper: "Pricing the Unknown: Ambiguity Premiums in China's Green vs. Brown Energy Markets"

This script orchestrates the complete analysis pipeline:
1. Data loading and preprocessing
2. Ambiguity measurement (firm, sector, composite, policy, geopolitical)
3. Causal analysis (OLS, 2SLS, DiD, mediation, moderation)
4. Visualization and reporting

Author: Research Team
Date: 2024
Paper Reference: causal_ambi_china.tex
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import project modules
from data_loader import ChinaEnergyDataLoader, prepare_analysis_data
from ambiguity_measurement import AmbiguityMeasurement, EnergySectorAmbiguity
from causal_analysis import CausalAmbiguityAnalysis


class ChinaEnergyPipeline:
    """
    Main pipeline for ambiguity analysis in Chinese energy markets

    This class orchestrates the complete workflow from data loading
    to final causal analysis and visualization.
    """

    def __init__(self, data_path='data/', output_path='output/'):
        """
        Initialize the analysis pipeline

        Parameters:
        -----------
        data_path : str
            Path to data directory
        output_path : str
            Path to save outputs (figures, tables, results)
        """
        self.data_path = data_path
        self.output_path = Path(output_path)
        self.output_path.mkdir(exist_ok=True)

        # Initialize components
        self.data_loader = ChinaEnergyDataLoader(data_path)
        self.ambiguity_meas = AmbiguityMeasurement()
        self.sector_ambiguity = EnergySectorAmbiguity()
        self.causal_analysis = None

        # Data containers
        self.data_dict = None
        self.ambiguity_dict = None
        self.results_dict = None

    def load_data(self, stock_list: Optional[List[str]] = None) -> Dict:
        """
        Load all required data for the analysis

        Parameters:
        -----------
        stock_list : list of str or None
            List of stock IDs to analyze (None uses all available)

        Returns:
        --------
        data_dict : dict
            Dictionary containing all loaded data
        """
        print("\n" + "="*70)
        print("STEP 1: Loading Data")
        print("="*70)

        self.data_dict = prepare_analysis_data(self.data_loader, stock_list)

        print(f"\n✓ Intraday returns loaded: {self.data_dict['intraday_returns'].shape}")
        print(f"✓ Stocks classified: {len(self.data_dict['energy_classification'])}")
        print(f"  - Brown energy: {sum(1 for v in self.data_dict['energy_classification'].values() if v == 'Brown')}")
        print(f"  - Green energy: {sum(1 for v in self.data_dict['energy_classification'].values() if v == 'Green')}")
        print(f"  - Grey energy: {sum(1 for v in self.data_dict['energy_classification'].values() if v == 'Grey')}")
        print(f"✓ Policy shock dates: {len(self.data_dict['policy_shock_dates'])}")

        return self.data_dict

    def compute_ambiguity_measures(self) -> Dict:
        """
        Compute all ambiguity measures for the analysis

        Returns:
        --------
        ambiguity_dict : dict
            Dictionary containing all computed ambiguity measures
        """
        print("\n" + "="*70)
        print("STEP 2: Computing Ambiguity Measures")
        print("="*70)

        self.ambiguity_dict = {}

        # 1. Firm-level ambiguity
        print("\n[1/5] Computing firm-level CEA...")
        intraday_returns = self.data_dict['intraday_returns']
        limit_days = self.data_dict['limit_days']

        firm_ambiguity = self.ambiguity_meas.compute_ambiguity_cross_section(
            returns_data=intraday_returns,
            limit_days_dict=limit_days
        )
        self.ambiguity_dict['firm_level'] = firm_ambiguity
        print(f"    ✓ Firm-level CEA computed: {firm_ambiguity.shape}")

        # 2. Sector ambiguity
        print("\n[2/5] Computing sector-level CEA...")
        sector_ambiguity = self.sector_ambiguity.compute_sector_ambiguity(
            ambiguity_df=firm_ambiguity
        )
        self.ambiguity_dict['sector_level'] = sector_ambiguity
        print(f"    ✓ Sector-level CEA computed")

        # 3. Composite Energy Ambiguity (PCA-based)
        print("\n[3/5] Computing Composite Energy Ambiguity (PCA)...")
        composite_ambiguity = self.sector_ambiguity.compute_composite_ambiguity(
            ambiguity_df=firm_ambiguity,
            n_components=1
        )
        self.ambiguity_dict['composite'] = composite_ambiguity
        print(f"    ✓ Composite Energy Ambiguity computed: {len(composite_ambiguity)} time periods")

        # 4. Policy Ambiguity
        print("\n[4/5] Computing Policy Ambiguity...")
        # Use CSI 300 Energy Index or crude oil futures as proxy
        policy_ambiguity = self.sector_ambiguity.compute_policy_ambiguity(
            index_returns=self.data_dict['control_data']['oil_returns'],
            ambiguity_measure=self.ambiguity_meas
        )
        self.ambiguity_dict['policy'] = policy_ambiguity
        print(f"    ✓ Policy Ambiguity computed: {len(policy_ambiguity)} time periods")

        # 5. Geopolitical Ambiguity
        print("\n[5/5] Computing Geopolitical Ambiguity...")
        geo_data = self.data_dict['geopolitical_data']
        geopolitical_ambiguity = self.sector_ambiguity.compute_geopolitical_ambiguity(
            defense_returns=geo_data['defense'],
            gold_returns=geo_data['gold'],
            ambiguity_measure=self.ambiguity_meas
        )
        self.ambiguity_dict['geopolitical'] = geopolitical_ambiguity
        print(f"    ✓ Geopolitical Ambiguity computed: {len(geopolitical_ambiguity)} time periods")

        return self.ambiguity_dict

    def prepare_analysis_dataset(self) -> pd.DataFrame:
        """
        Prepare the final dataset for causal analysis

        Returns:
        --------
        analysis_df : pandas DataFrame
            DataFrame with all variables for causal analysis
        """
        print("\n" + "="*70)
        print("STEP 3: Preparing Analysis Dataset")
        print("="*70)

        # Merge all data into a single DataFrame
        firm_ambiguity = self.ambiguity_dict['firm_level']

        # Stack to long format
        analysis_df = firm_ambiguity.stack().reset_index()
        analysis_df.columns = ['date', 'stock_id', 'CEA']

        # Add energy classification
        energy_class = self.data_dict['energy_classification']
        analysis_df['energy_type'] = analysis_df['stock_id'].map(energy_class)

        # Create Green dummy
        analysis_df['Green_Dummy'] = (analysis_df['energy_type'] == 'Green').astype(int)

        # Add composite ambiguity (time-series)
        composite_amb = self.ambiguity_dict['composite']
        analysis_df = analysis_df.merge(
            composite_amb.to_frame('Composite_CEA'),
            left_on='date',
            right_index=True,
            how='left'
        )

        # Add geopolitical ambiguity
        geo_amb = self.ambiguity_dict['geopolitical']
        analysis_df = analysis_df.merge(
            geo_amb.to_frame('Geo_CEA'),
            left_on='date',
            right_index=True,
            how='left'
        )

        # Add controls
        controls = self.data_dict['control_data']
        analysis_df = analysis_df.merge(
            controls['oil_returns'].to_frame('oil_return'),
            left_on='date',
            right_index=True,
            how='left'
        )

        analysis_df = analysis_df.merge(
            controls['ets_returns'].to_frame('ets_return'),
            left_on='date',
            right_index=True,
            how='left'
        )

        # Add policy sensitivity
        policy_sens = self.data_dict['policy_sensitivity']
        analysis_df = analysis_df.merge(
            policy_sens.stack().reset_index(),
            on=['date', 'stock_id'],
            how='left'
        )
        analysis_df.rename(columns={0: 'policy_sensitivity'}, inplace=True)

        # Calculate forward returns (next period returns)
        intraday_returns = self.data_dict['intraday_returns']
        daily_returns = intraday_returns.resample('D').sum()
        forward_returns = daily_returns.shift(-1)

        forward_returns_long = forward_returns.stack().reset_index()
        forward_returns_long.columns = ['date', 'stock_id', 'forward_return']

        analysis_df = analysis_df.merge(
            forward_returns_long,
            on=['date', 'stock_id'],
            how='left'
        )

        print(f"\n✓ Analysis dataset prepared: {analysis_df.shape}")
        print(f"  - Date range: {analysis_df['date'].min()} to {analysis_df['date'].max()}")
        print(f"  - Total observations: {len(analysis_df)}")
        print(f"  - Number of stocks: {analysis_df['stock_id'].nunique()}")

        return analysis_df

    def run_causal_analysis(self, analysis_df: pd.DataFrame) -> Dict:
        """
        Run the complete causal analysis pipeline

        Parameters:
        -----------
        analysis_df : pandas DataFrame
            Analysis dataset prepared by prepare_analysis_dataset()

        Returns:
        --------
        results_dict : dict
            Dictionary containing all analysis results
        """
        print("\n" + "="*70)
        print("STEP 4: Running Causal Analysis")
        print("="*70)

        # Initialize causal analysis
        self.causal_analysis = CausalAmbiguityAnalysis(analysis_df)
        self.results_dict = {}

        # 1. Baseline Panel OLS
        print("\n[1/6] Baseline Panel OLS (Equation 10)...")
        ols_results = self.causal_analysis.baseline_panel_ols(
            dependent_var='forward_return',
            ambiguity_var='CEA',
            green_var='Green_Dummy'
        )
        self.results_dict['baseline_ols'] = ols_results
        print("    ✓ OLS results: β_CEA = {:.4f} (t-stat: {:.2f})".format(
            ols_results['params']['CEA'],
            ols_results['tstats']['CEA']
        ))

        # 2. Instrumental Variables (2SLS)
        print("\n[2/6] Instrumental Variables 2SLS (Equations 11-12)...")
        instruments_data = {
            'peer_ambiguity': self._compute_peer_ambiguity(analysis_df),
            'epu': self.data_dict['epu_series'],
            'policy_sensitivity': 'policy_sensitivity'
        }

        iv_results = self.causal_analysis.instrumental_variables_2sls(
            dependent_var='forward_return',
            endogenous_var='CEA',
            instruments=instruments_data,
            green_var='Green_Dummy'
        )
        self.results_dict['iv_2sls'] = iv_results
        print("    ✓ 2SLS results: β_CEA_IV = {:.4f} (t-stat: {:.2f})".format(
            iv_results['second_stage']['params']['CEA'],
            iv_results['second_stage']['tstats']['CEA']
        ))

        # 3. Difference-in-Differences
        print("\n[3/6] Difference-in-Differences (Equation 13)...")
        policy_dates = [pd.Timestamp(d) for d in self.data_dict['policy_shock_dates']]
        did_results = self.causal_analysis.difference_in_differences(
            policy_shock_dates=policy_dates,
            treatment_group='Green',
            window_days=30
        )
        self.results_dict['did'] = did_results
        print("    ✓ DiD estimator: {:.4f} (p-value: {:.4f})".format(
            did_results['did_estimator'],
            did_results['p_value']
        ))

        # 4. Mediation Analysis
        print("\n[4/6] Mediation Analysis (Equations 14-15)...")
        mediation_results = self.causal_analysis.mediation_analysis(
            independent_var='CEA',
            mediator_var='Composite_CEA',
            dependent_var='forward_return',
            n_bootstrap=1000
        )
        self.results_dict['mediation'] = mediation_results
        print("    ✓ Mediation effect: {:.4f} (95% CI: [{:.4f}, {:.4f}])".format(
            mediation_results['indirect_effect'],
            mediation_results['ci_lower'],
            mediation_results['ci_upper']
        ))

        # 5. Moderation Analysis
        print("\n[5/6] Moderation Analysis (Equation 16)...")
        moderation_results = self.causal_analysis.moderation_analysis(
            moderator_var='Green_Dummy'
        )
        self.results_dict['moderation'] = moderation_results
        print("    ✓ Moderation effect (β_Interaction): {:.4f} (p-value: {:.4f})".format(
            moderation_results['interaction_effect'],
            moderation_results['p_value']
        ))

        # 6. Granger Causality
        print("\n[6/6] Granger Causality Tests...")
        granger_results = self.causal_analysis.granger_causality_test(
            ambiguity_var='CEA',
            returns_var='forward_return',
            max_lag=3
        )
        self.results_dict['granger'] = granger_results
        significant = sum(1 for r in granger_results if r['p_value'] < 0.05)
        print(f"    ✓ Granger tests: {significant}/{len(granger_results)} stocks show significant causality")

        return self.results_dict

    def _compute_peer_ambiguity(self, analysis_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute peer ambiguity as an instrument (average CEA of same energy type)

        Parameters:
        -----------
        analysis_df : pandas DataFrame
            Analysis dataset

        Returns:
        --------
        peer_ambiguity : pandas DataFrame
            Peer ambiguity measure for each stock-date
        """
        # Compute average CEA by energy type and date, excluding the stock itself
        peer_amb = []

        for energy_type in ['Brown', 'Green', 'Grey']:
            type_data = analysis_df[analysis_df['energy_type'] == energy_type]

            # Group by date and compute mean CEA across peers
            type_mean = type_data.groupby('date')['CEA'].mean().reset_index()
            type_mean.columns = ['date', 'peer_ambiguity']

            # Map back to original data
            type_data = type_data.merge(type_mean, on='date', how='left')
            peer_amb.append(type_data[['date', 'stock_id', 'peer_ambiguity']])

        peer_amb_df = pd.concat(peer_amb, ignore_index=True)
        return peer_amb_df

    def visualize_results(self):
        """
        Generate all visualizations for the analysis
        """
        print("\n" + "="*70)
        print("STEP 5: Generating Visualizations")
        print("="*70)

        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['font.size'] = 10

        # 1. CEA time series by energy type
        print("\n[1/5] Plotting CEA time series...")
        self._plot_cea_timeseries()

        # 2. CEA distribution
        print("\n[2/5] Plotting CEA distribution...")
        self._plot_cea_distribution()

        # 3. Composite ambiguity components
        print("\n[3/5] Plotting PCA components...")
        self._plot_pca_components()

        # 4. Policy shock events
        print("\n[4/5] Plotting policy shock events...")
        self._plot_policy_shocks()

        # 5. Causal analysis results summary
        print("\n[5/5] Plotting results summary...")
        self._plot_results_summary()

        print(f"\n✓ All visualizations saved to {self.output_path}")

    def _plot_cea_timeseries(self):
        """Plot CEA time series by energy type"""
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))

        firm_ambiguity = self.ambiguity_dict['firm_level']
        energy_class = self.data_dict['energy_classification']

        for idx, energy_type in enumerate(['Brown', 'Green', 'Grey']):
            ax = axes[idx]

            # Get stocks of this type
            type_stocks = [s for s, t in energy_class.items() if t == energy_type]

            if type_stocks:
                type_ambiguity = firm_ambiguity[type_stocks].mean(axis=1)
                type_ambiguity.plot(ax=ax, linewidth=1, alpha=0.7)

                # Mark policy shocks
                for date in self.data_dict['policy_shock_dates']:
                    shock_date = pd.Timestamp(date)
                    if shock_date in type_ambiguity.index:
                        ax.axvline(shock_date, color='red', linestyle='--',
                                  alpha=0.5, linewidth=1)

            ax.set_title(f'{energy_type} Energy - Average CEA')
            ax.set_ylabel('CEA')
            ax.legend(['Average CEA', 'Policy Shocks'], loc='upper left')

        axes[-1].set_xlabel('Date')
        plt.tight_layout()
        plt.savefig(self.output_path / 'cea_timeseries.png', dpi=300)
        plt.close()

    def _plot_cea_distribution(self):
        """Plot CEA distribution by energy type"""
        fig, ax = plt.subplots(figsize=(10, 6))

        analysis_df = self.prepare_analysis_dataset()

        for energy_type in ['Brown', 'Green', 'Grey']:
            type_data = analysis_df[analysis_df['energy_type'] == energy_type]['CEA']
            sns.kdeplot(type_data, label=energy_type, ax=ax, fill=True, alpha=0.3)

        ax.set_xlabel('Cross-Entropy Ambiguity (CEA)')
        ax.set_ylabel('Density')
        ax.set_title('Distribution of Firm-Level CEA by Energy Type')
        ax.legend()
        plt.tight_layout()
        plt.savefig(self.output_path / 'cea_distribution.png', dpi=300)
        plt.close()

    def _plot_pca_components(self):
        """Plot PCA components for composite ambiguity"""
        # Recompute PCA with all components for visualization
        from sklearn.decomposition import PCA

        firm_ambiguity = self.ambiguity_dict['firm_level']
        pca = PCA(n_components=5)
        pca.fit(firm_ambiguity.dropna())

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Scree plot
        ax = axes[0]
        ax.bar(range(1, 6), pca.explained_variance_ratio_, alpha=0.7)
        ax.set_xlabel('Principal Component')
        ax.set_ylabel('Explained Variance Ratio')
        ax.set_title('PCA Scree Plot - Composite Energy Ambiguity')
        ax.set_xticks(range(1, 6))

        # Cumulative variance
        ax = axes[1]
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        ax.plot(range(1, 6), cumvar, marker='o', linewidth=2)
        ax.axhline(y=0.8, color='r', linestyle='--', alpha=0.5)
        ax.set_xlabel('Principal Component')
        ax.set_ylabel('Cumulative Explained Variance')
        ax.set_title('Cumulative Variance Explained')
        ax.set_xticks(range(1, 6))
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_path / 'pca_components.png', dpi=300)
        plt.close()

    def _plot_policy_shocks(self):
        """Plot CEA around policy shock dates"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        firm_ambiguity = self.ambiguity_dict['firm_level']
        energy_class = self.data_dict['energy_classification']

        for idx, shock_date in enumerate(self.data_dict['policy_shock_dates'][:4]):
            ax = axes[idx]

            shock_date = pd.Timestamp(shock_date)
            window = pd.Timedelta(days=30)

            # Get window data
            window_data = firm_ambiguity.loc[shock_date - window:shock_date + window]

            # Plot Brown vs Green
            brown_stocks = [s for s, t in energy_class.items() if t == 'Brown']
            green_stocks = [s for s, t in energy_class.items() if t == 'Green']

            if brown_stocks:
                brown_amb = window_data[brown_stocks].mean(axis=1)
                ax.plot(brown_amb.index, brown_amb.values, label='Brown', color='brown')

            if green_stocks:
                green_amb = window_data[green_stocks].mean(axis=1)
                ax.plot(green_amb.index, green_amb.values, label='Green', color='green')

            ax.axvline(shock_date, color='red', linestyle='--', linewidth=2)
            ax.set_title(f'Policy Shock: {shock_date.strftime("%Y-%m-%d")}')
            ax.set_xlabel('Date')
            ax.set_ylabel('Average CEA')
            ax.legend()
            ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(self.output_path / 'policy_shocks.png', dpi=300)
        plt.close()

    def _plot_results_summary(self):
        """Plot summary of all causal analysis results"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Coefficient comparison
        ax = axes[0, 0]
        methods = ['OLS', '2SLS']
        coefficients = [
            self.results_dict['baseline_ols']['params']['CEA'],
            self.results_dict['iv_2sls']['second_stage']['params']['CEA']
        ]
        errors = [
            self.results_dict['baseline_ols']['std_errors']['CEA'] * 1.96,
            self.results_dict['iv_2sls']['second_stage']['std_errors']['CEA'] * 1.96
        ]

        x_pos = np.arange(len(methods))
        ax.bar(x_pos, coefficients, yerr=errors, alpha=0.7, capsize=5)
        ax.set_xlabel('Estimation Method')
        ax.set_ylabel('CEA Coefficient')
        ax.set_title('Ambiguity Premium: OLS vs 2SLS')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(methods)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        # 2. DiD results
        ax = axes[0, 1]
        did = self.results_dict['did']
        ax.bar(['DiD Estimator'], [did['did_estimator']],
               yerr=[did['did_estimator'] - did['ci_lower']],
               alpha=0.7, capsize=5)
        ax.set_ylabel('Treatment Effect')
        ax.set_title('Difference-in-Differences: Policy Shocks')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        # 3. Mediation effects
        ax = axes[1, 0]
        med = self.results_dict['mediation']
        effects = {
            'Direct': med['direct_effect'],
            'Indirect': med['indirect_effect'],
            'Total': med['total_effect']
        }
        ax.bar(effects.keys(), effects.values(), alpha=0.7)
        ax.set_ylabel('Effect Size')
        ax.set_title('Mediation Analysis: Liquidity Channel')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        # 4. Moderation effect
        ax = axes[1, 1]
        mod = self.results_dict['moderation']
        ax.scatter(['Brown Energy'], [mod['main_effect']], s=100, label='Brown')
        ax.scatter(['Green Energy'], [mod['main_effect'] + mod['interaction_effect']],
                  s=100, label='Green')
        ax.set_ylabel('Ambiguity Premium')
        ax.set_title('Moderation Effect: Green vs Brown')
        ax.legend()
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        plt.tight_layout()
        plt.savefig(self.output_path / 'results_summary.png', dpi=300)
        plt.close()

    def generate_report(self):
        """
        Generate a comprehensive analysis report
        """
        print("\n" + "="*70)
        print("STEP 6: Generating Analysis Report")
        print("="*70)

        report_path = self.output_path / 'analysis_report.txt'

        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("CHINA ENERGY MARKET AMBIGUITY ANALYSIS REPORT\n")
            f.write("Paper: Pricing the Unknown - Ambiguity Premiums in China's Green vs Brown Energy Markets\n")
            f.write("="*70 + "\n\n")

            # Data summary
            f.write("1. DATA SUMMARY\n")
            f.write("-" * 70 + "\n")
            analysis_df = self.prepare_analysis_dataset()
            f.write(f"Date range: {analysis_df['date'].min()} to {analysis_df['date'].max()}\n")
            f.write(f"Total observations: {len(analysis_df)}\n")
            f.write(f"Number of stocks: {analysis_df['stock_id'].nunique()}\n\n")

            for energy_type in ['Brown', 'Green', 'Grey']:
                count = sum(1 for v in self.data_dict['energy_classification'].values() if v == energy_type)
                f.write(f"{energy_type} energy stocks: {count}\n")

            # Ambiguity summary
            f.write("\n2. AMBIGUITY MEASURES SUMMARY\n")
            f.write("-" * 70 + "\n")
            cea = self.ambiguity_dict['firm_level']
            f.write(f"Firm-level CEA mean: {cea.mean().mean():.6f}\n")
            f.write(f"Firm-level CEA std: {cea.std().mean():.6f}\n")

            comp = self.ambiguity_dict['composite']
            f.write(f"Composite CEA mean: {comp.mean():.6f}\n")
            f.write(f"Composite CEA std: {comp.std():.6f}\n\n")

            # Causal results
            f.write("3. CAUSAL ANALYSIS RESULTS\n")
            f.write("-" * 70 + "\n\n")

            f.write("3.1 Baseline OLS (Equation 10)\n")
            ols = self.results_dict['baseline_ols']
            f.write(f"    CEA coefficient: {ols['params']['CEA']:.6f}\n")
            f.write(f"    t-statistic: {ols['tstats']['CEA']:.4f}\n")
            f.write(f"    p-value: {ols['pvalues']['CEA']:.4f}\n\n")

            f.write("3.2 2SLS (Equations 11-12)\n")
            iv = self.results_dict['iv_2sls']
            f.write(f"    First stage F-stat: {iv['first_stage']['f_statistic']:.4f}\n")
            f.write(f"    Second stage CEA coefficient: {iv['second_stage']['params']['CEA']:.6f}\n")
            f.write(f"    t-statistic: {iv['second_stage']['tstats']['CEA']:.4f}\n\n")

            f.write("3.3 Difference-in-Differences (Equation 13)\n")
            did = self.results_dict['did']
            f.write(f"    DiD estimator: {did['did_estimator']:.6f}\n")
            f.write(f"    p-value: {did['p_value']:.4f}\n")
            f.write(f"    95% CI: [{did['ci_lower']:.6f}, {did['ci_upper']:.6f}]\n\n")

            f.write("3.4 Mediation Analysis (Equations 14-15)\n")
            med = self.results_dict['mediation']
            f.write(f"    Direct effect: {med['direct_effect']:.6f}\n")
            f.write(f"    Indirect effect: {med['indirect_effect']:.6f}\n")
            f.write(f"    Total effect: {med['total_effect']:.6f}\n\n")

            f.write("3.5 Moderation Analysis (Equation 16)\n")
            mod = self.results_dict['moderation']
            f.write(f"    Main effect: {mod['main_effect']:.6f}\n")
            f.write(f"    Interaction effect: {mod['interaction_effect']:.6f}\n")
            f.write(f"    p-value: {mod['p_value']:.4f}\n\n")

            # Hypothesis testing summary
            f.write("4. HYPOTHESIS TESTING SUMMARY\n")
            f.write("-" * 70 + "\n")
            f.write("H1 (Ambiguity Premium): ")
            f.write("SUPPORTED\n" if ols['pvalues']['CEA'] < 0.05 else "NOT SUPPORTED\n")
            f.write("H2 (Green Discount): ")
            f.write("SUPPORTED\n" if mod['interaction_effect'] < 0 and mod['p_value'] < 0.05 else "NOT SUPPORTED\n")
            f.write("H3 (Policy Shocks): ")
            f.write("SUPPORTED\n" if did['p_value'] < 0.05 else "NOT SUPPORTED\n")
            f.write("H4 (Liquidity Channel): ")
            f.write("SUPPORTED\n" if med['ci_lower'] > 0 else "NOT SUPPORTED\n")
            f.write("H5 (Regime Dependence): ")
            f.write("SUPPORTED\n" if mod['p_value'] < 0.05 else "NOT SUPPORTED\n")

        print(f"\n✓ Analysis report saved to {report_path}")

    def run_full_pipeline(self):
        """
        Execute the complete analysis pipeline

        This is the main method that runs all steps:
        1. Load data
        2. Compute ambiguity measures
        3. Prepare analysis dataset
        4. Run causal analysis
        5. Generate visualizations
        6. Create report
        """
        print("\n" + "="*70)
        print("CHINA ENERGY MARKET AMBIGUITY ANALYSIS")
        print("Full Pipeline Execution")
        print("="*70)

        try:
            # Step 1: Load data
            self.load_data()

            # Step 2: Compute ambiguity measures
            self.compute_ambiguity_measures()

            # Step 3: Prepare analysis dataset
            analysis_df = self.prepare_analysis_dataset()

            # Step 4: Run causal analysis
            self.run_causal_analysis(analysis_df)

            # Step 5: Generate visualizations
            self.visualize_results()

            # Step 6: Generate report
            self.generate_report()

            print("\n" + "="*70)
            print("PIPELINE COMPLETED SUCCESSFULLY")
            print("="*70)
            print(f"\nAll outputs saved to: {self.output_path.absolute()}")
            print("\nGenerated files:")
            print("  - cea_timeseries.png")
            print("  - cea_distribution.png")
            print("  - pca_components.png")
            print("  - policy_shocks.png")
            print("  - results_summary.png")
            print("  - analysis_report.txt")

        except Exception as e:
            print(f"\n❌ Pipeline failed with error: {str(e)}")
            import traceback
            traceback.print_exc()


def main():
    """
    Main execution function
    """
    # Initialize pipeline
    pipeline = ChinaEnergyPipeline(
        data_path='data/',
        output_path='output/'
    )

    # Run full pipeline
    pipeline.run_full_pipeline()


if __name__ == "__main__":
    main()
