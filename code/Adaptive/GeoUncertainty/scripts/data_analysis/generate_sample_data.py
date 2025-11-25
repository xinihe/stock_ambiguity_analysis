#!/usr/bin/env python3
"""
Generate Sample Data for Geopolitical Risk and Ambiguity Analysis
Created for the geopoliticalAmb02.tex paper Data and Empirical Research Design section
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import matplotlib.pyplot as plt
import seaborn as sns

def set_seed(seed=42):
    """Set random seed for reproducibility"""
    np.random.seed(seed)

def load_csi300_data():
    """Load CSI 300 index data from the existing CSV file"""
    data_path = "/Users/tlxy/Research/Ambiguity/data/SSE.000300.csv"
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        return df
    else:
        print(f"Warning: {data_path} not found. Creating synthetic data.")
        return create_synthetic_csi300_data()

def create_synthetic_csi300_data():
    """Create synthetic CSI 300 data if original file is not available"""
    start_date = datetime(2018, 1, 2)
    end_date = datetime(2023, 12, 29)

    # Create business days
    dates = pd.bdate_range(start=start_date, end=end_date)

    # Simulate price data with realistic characteristics
    initial_price = 4000
    returns = np.random.normal(0.00032, 0.01245, len(dates))  # Daily returns
    prices = [initial_price]

    for ret in returns[1:]:
        new_price = prices[-1] * (1 + ret)
        prices.append(max(new_price, 1000))  # Minimum price floor

    # Create DataFrame
    df = pd.DataFrame({
        'SSE.000300.close': prices,
        'SSE.000300.volume': np.random.lognormal(14, 0.5, len(dates))
    }, index=dates)

    # Calculate OHLC from close
    df['SSE.000300.open'] = df['SSE.000300.close'].shift(1).fillna(df['SSE.000300.close'].iloc[0])
    df['SSE.000300.high'] = df[['SSE.000300.open', 'SSE.000300.close']].max(axis=1) * np.random.uniform(1.0, 1.01, len(dates))
    df['SSE.000300.low'] = df[['SSE.000300.open', 'SSE.000300.close']].min(axis=1) * np.random.uniform(0.99, 1.0, len(dates))

    return df

def calculate_returns(df):
    """Calculate daily and other return measures"""
    df['returns'] = df['SSE.000300.close'].pct_change()
    df['log_returns'] = np.log(df['SSE.000300.close'] / df['SSE.000300.close'].shift(1))
    df['cumulative_returns'] = (1 + df['returns']).cumprod() - 1
    return df

def calculate_realized_volatility(df, window=22):
    """Calculate realized volatility using daily returns"""
    df['realized_vol'] = df['returns'].rolling(window=window).std() * np.sqrt(252)
    return df

def generate_gpr_data(df):
    """Generate synthetic GPR data that correlates with market stress"""
    dates = df.index

    # Base GPR level with trends and shocks
    base_gpr = 100 + 20 * np.sin(np.linspace(0, 6*np.pi, len(dates)))  # Cyclical component

    # Add stress events that correlate with market volatility
    stress_periods = (df['returns'].abs() > df['returns'].abs().quantile(0.9))
    gpr_shocks = np.where(stress_periods, np.random.uniform(20, 80, len(dates)), np.random.uniform(0, 20, len(dates)))

    # Add random walk component
    random_walk = np.cumsum(np.random.normal(0, 5, len(dates)))
    random_walk = random_walk - random_walk.min() + 50  # Normalize to positive values

    gpr_index = base_gpr + gpr_shocks + 0.3 * random_walk
    gpr_index = np.maximum(gpr_index, 10)  # Minimum GPR level

    df['gpr_index'] = gpr_index
    return df

def generate_ambiguity_measures(df):
    """Generate ambiguity measures based on adaptive model uncertainty framework"""

    # Cross-entropy ambiguity (baseline)
    # Higher during market stress and GPR events
    stress_indicator = (df['returns'].abs() > df['returns'].abs().quantile(0.8))
    gpr_stress = (df['gpr_index'] > df['gpr_index'].quantile(0.8))

    # Base ambiguity level
    base_ambiguity = 0.1

    # Stress amplification
    stress_multiplier = np.where(stress_indicator | gpr_stress,
                                np.random.uniform(1.5, 3.0, len(df)),
                                np.random.uniform(0.8, 1.2, len(df)))

    # Add some persistence
    ambiguity_series = []
    current_ambiguity = base_ambiguity

    for i in range(len(df)):
        current_ambiguity = 0.7 * current_ambiguity + 0.3 * base_ambiguity * stress_multiplier[i]
        current_ambiguity += np.random.normal(0, 0.02)
        current_ambiguity = np.clip(current_ambiguity, 0.02, 0.4)
        ambiguity_series.append(current_ambiguity)

    df['ambiguity_ce'] = np.array(ambiguity_series)

    # Model disagreement ambiguity
    df['ambiguity_md'] = df['ambiguity_ce'] * np.random.uniform(0.8, 1.2, len(df))

    # Weight dispersion ambiguity
    df['ambiguity_wd'] = df['ambiguity_ce'] * np.random.uniform(0.6, 1.0, len(df))

    return df

def create_sample_firms_data(n_firms=100):
    """Create sample firm-level data for cross-sectional analysis"""

    # Firm characteristics
    np.random.seed(42)

    industries = ['Manufacturing', 'Financial Services', 'Real Estate', 'Information Technology',
                  'Healthcare', 'Energy & Materials', 'Consumer Discretionary', 'Consumer Staples',
                  'Utilities', 'Others']

    industry_probs = [0.30, 0.08, 0.07, 0.12, 0.09, 0.10, 0.15, 0.08, 0.05, 0.16]
    # Normalize to ensure sum = 1
    industry_probs = np.array(industry_probs) / sum(industry_probs)

    firms = []
    for i in range(n_firms):
        firm_data = {
            'firm_id': f'STOCK_{i+1:04d}',
            'industry': np.random.choice(industries, p=industry_probs),
            'is_soe': np.random.choice([0, 1], p=[0.65, 0.35]),  # 35% SOEs
            'market_cap': np.random.lognormal(3.5, 1.5),  # In billions
            'book_to_market': np.random.beta(2, 3),
            'leverage': np.random.beta(2, 5),
            'profitability': np.random.normal(0.08, 0.05),
            'investment': np.random.normal(0.12, 0.08),
            'momentum': np.random.normal(0.06, 0.15),
        }

        # Ensure realistic constraints
        firm_data['book_to_market'] = np.clip(firm_data['book_to_market'], 0.1, 3.0)
        firm_data['leverage'] = np.clip(firm_data['leverage'], 0.1, 0.8)
        firm_data['profitability'] = np.clip(firm_data['profitability'], -0.1, 0.3)
        firm_data['investment'] = np.clip(firm_data['investment'], -0.2, 0.5)

        # SOEs tend to be larger and have higher leverage
        if firm_data['is_soe'] == 1:
            firm_data['market_cap'] *= 1.5
            firm_data['leverage'] *= 1.2

        firms.append(firm_data)

    return pd.DataFrame(firms)

def generate_firm_returns_data(firms_df, market_df, n_days=100):
    """Generate firm-level returns data"""

    sample_dates = market_df.index[-n_days:]
    returns_data = []

    for _, firm in firms_df.iterrows():
        # Generate firm returns based on market exposure and firm characteristics
        market_beta = np.random.normal(1.0, 0.3)
        ambiguity_beta = np.random.normal(-0.5, 0.2)
        volatility_beta = np.random.normal(-0.3, 0.15)

        # SOEs have lower ambiguity sensitivity
        if firm['is_soe'] == 1:
            ambiguity_beta *= 0.6

        for date in sample_dates:
            if date in market_df.index:
                market_data = market_df.loc[date]

                # Firm-specific return
                firm_return = (
                    0.0001 +  # Base return
                    market_beta * market_data['returns'] +
                    ambiguity_beta * (market_data['ambiguity_ce'] - market_data['ambiguity_ce'].mean()) +
                    volatility_beta * (market_data['realized_vol'] - market_data['realized_vol'].mean()) +
                    np.random.normal(0, 0.02)  # Idiosyncratic component
                )

                returns_data.append({
                    'date': date,
                    'firm_id': firm['firm_id'],
                    'return': firm_return,
                    'industry': firm['industry'],
                    'is_soe': firm['is_soe'],
                    'market_cap': firm['market_cap'],
                    'book_to_market': firm['book_to_market'],
                    'leverage': firm['leverage'],
                    'profitability': firm['profitability'],
                    'investment': firm['investment'],
                    'momentum': firm['momentum'],
                    'market_beta': market_beta,
                    'ambiguity_beta': ambiguity_beta,
                    'volatility_beta': volatility_beta,
                })

    return pd.DataFrame(returns_data)

def calculate_summary_statistics(df, firms_df, returns_df):
    """Calculate summary statistics for tables in the paper"""

    # Market-level summary
    market_vars = ['CSI 300 Returns (%)', 'Ambiguity Index', 'Volatility (RV)', 'GPR Index']
    market_means = [
        df['returns'].mean() * 100,
        df['ambiguity_ce'].mean(),
        df['realized_vol'].mean(),
        df['gpr_index'].mean()
    ]
    market_stds = [
        df['returns'].std() * 100,
        df['ambiguity_ce'].std(),
        df['realized_vol'].std(),
        df['gpr_index'].std()
    ]
    market_mins = [
        df['returns'].min() * 100,
        df['ambiguity_ce'].min(),
        df['realized_vol'].min(),
        df['gpr_index'].min()
    ]
    market_maxs = [
        df['returns'].max() * 100,
        df['ambiguity_ce'].max(),
        df['realized_vol'].max(),
        df['gpr_index'].max()
    ]

    # Firm-level summary
    firm_vars = ['Market Cap (billions)', 'Book-to-Market']
    firm_means = [
        firms_df['market_cap'].mean(),
        firms_df['book_to_market'].mean()
    ]
    firm_stds = [
        firms_df['market_cap'].std(),
        firms_df['book_to_market'].std()
    ]
    firm_mins = [
        firms_df['market_cap'].min(),
        firms_df['book_to_market'].min()
    ]
    firm_maxs = [
        firms_df['market_cap'].max(),
        firms_df['book_to_market'].max()
    ]

    # Combine all statistics
    all_vars = market_vars + firm_vars
    all_means = market_means + firm_means
    all_stds = market_stds + firm_stds
    all_mins = market_mins + firm_mins
    all_maxs = market_maxs + firm_maxs

    market_stats = pd.DataFrame({
        'Variable': all_vars,
        'Mean': all_means,
        'Std. Dev.': all_stds,
        'Min': all_mins,
        'Max': all_maxs
    })

    return market_stats

def create_industry_composition_table(firms_df):
    """Create industry composition table"""

    composition = firms_df.groupby(['industry', 'is_soe']).size().unstack(fill_value=0)
    composition.columns = ['Non-SOEs', 'SOEs']
    composition['Total'] = composition.sum(axis=1)
    composition = composition.reindex(['Manufacturing', 'Financial Services', 'Real Estate',
                                      'Information Technology', 'Healthcare', 'Energy & Materials',
                                      'Consumer Discretionary', 'Consumer Staples', 'Utilities', 'Others'])

    composition.loc['Total'] = composition.sum()

    return composition

def save_data_files(df, firms_df, returns_df, summary_stats, composition):
    """Save all generated data to CSV files"""

    output_dir = "/Users/tlxy/Research/Ambiguity/code/Adaptive/GeoUncertainty/outputs/data"
    os.makedirs(output_dir, exist_ok=True)

    # Save main datasets
    df.to_csv(os.path.join(output_dir, "market_data_sample.csv"))
    firms_df.to_csv(os.path.join(output_dir, "firm_characteristics_sample.csv"))
    returns_df.to_csv(os.path.join(output_dir, "firm_returns_sample.csv"))

    # Save summary tables
    summary_stats.to_csv(os.path.join(output_dir, "summary_statistics.csv"), index=False)
    composition.to_csv(os.path.join(output_dir, "industry_composition.csv"))

    print(f"Data files saved to {output_dir}")

    return output_dir

def create_visualizations(df, firms_df, output_dir):
    """Create visualizations for the data analysis"""

    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Plot 1: CSI 300 Returns and Ambiguity
    ax1 = axes[0, 0]
    ax1_twin = ax1.twinx()
    ax1.plot(df.index, df['returns'] * 100, 'b-', alpha=0.7, label='Returns (%)')
    ax1_twin.plot(df.index, df['ambiguity_ce'], 'r-', alpha=0.7, label='Ambiguity')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Returns (%)', color='b')
    ax1_twin.set_ylabel('Ambiguity Index', color='r')
    ax1.set_title('CSI 300 Returns and Ambiguity Over Time')
    ax1.grid(True, alpha=0.3)

    # Plot 2: GPR and Ambiguity relationship
    ax2 = axes[0, 1]
    scatter = ax2.scatter(df['gpr_index'], df['ambiguity_ce'], alpha=0.6, s=20)
    z = np.polyfit(df['gpr_index'], df['ambiguity_ce'], 1)
    p = np.poly1d(z)
    ax2.plot(df['gpr_index'], p(df['gpr_index']), "r--", alpha=0.8)
    ax2.set_xlabel('GPR Index')
    ax2.set_ylabel('Ambiguity Index')
    ax2.set_title('GPR and Ambiguity Relationship')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Distribution of firm characteristics
    ax3 = axes[1, 0]
    soe_mask = firms_df['is_soe'] == 1
    ax3.hist(firms_df.loc[~soe_mask, 'market_cap'], bins=20, alpha=0.7, label='Non-SOEs')
    ax3.hist(firms_df.loc[soe_mask, 'market_cap'], bins=20, alpha=0.7, label='SOEs')
    ax3.set_xlabel('Market Cap (billions)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Distribution of Market Cap by Ownership Type')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Industry composition
    ax4 = axes[1, 1]
    composition = firms_df.groupby(['industry', 'is_soe']).size().unstack(fill_value=0)
    composition.plot(kind='bar', stacked=True, ax=ax4)
    ax4.set_xlabel('Industry')
    ax4.set_ylabel('Number of Firms')
    ax4.set_title('Sample Composition by Industry and Ownership Type')
    ax4.legend(['Non-SOEs', 'SOEs'])
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "data_visualizations.png"), dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Visualizations saved to {output_dir}")

def main():
    """Main function to generate all sample data"""

    print("Generating sample data for Geopolitical Risk and Ambiguity Analysis...")

    # Set seed for reproducibility
    set_seed(42)

    # Load/create CSI 300 data
    print("1. Loading CSI 300 data...")
    df = load_csi300_data()

    # Calculate returns and volatility
    print("2. Calculating returns and volatility...")
    df = calculate_returns(df)
    df = calculate_realized_volatility(df)

    # Generate GPR data
    print("3. Generating GPR data...")
    df = generate_gpr_data(df)

    # Generate ambiguity measures
    print("4. Generating ambiguity measures...")
    df = generate_ambiguity_measures(df)

    # Create firm data
    print("5. Creating firm-level data...")
    firms_df = create_sample_firms_data(n_firms=2700)  # Match the paper's sample size

    # Generate firm returns
    print("6. Generating firm returns...")
    returns_df = generate_firm_returns_data(firms_df, df, n_days=200)  # Sample for recent period

    # Calculate summary statistics
    print("7. Calculating summary statistics...")
    summary_stats = calculate_summary_statistics(df, firms_df, returns_df)
    composition = create_industry_composition_table(firms_df)

    # Save data files
    print("8. Saving data files...")
    output_dir = save_data_files(df, firms_df, returns_df, summary_stats, composition)

    # Create visualizations
    print("9. Creating visualizations...")
    create_visualizations(df, firms_df, output_dir)

    # Print summary
    print("\n=== DATA GENERATION SUMMARY ===")
    print(f"Sample Period: {df.index.min().date()} to {df.index.max().date()}")
    print(f"Total Trading Days: {len(df)}")
    print(f"Total Firms: {len(firms_df)}")
    print(f"SOE Firms: {firms_df['is_soe'].sum()} ({firms_df['is_soe'].mean():.1%})")
    print(f"Firm-Day Observations: {len(returns_df)}")
    print(f"Average Daily Return: {df['returns'].mean()*100:.3f}%")
    print(f"Average Ambiguity: {df['ambiguity_ce'].mean():.4f}")
    print(f"Average GPR: {df['gpr_index'].mean():.1f}")

    print(f"\nAll files saved to: {output_dir}")
    print("Data generation complete!")

if __name__ == "__main__":
    main()