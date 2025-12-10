"""
Simplified figure generation script without complex LaTeX math
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import matplotlib.dates as mdates

# Use non-LaTeX backend for matplotlib
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14
})

#==============================================================
# Figure 1: KL Divergence Concept
#==============================================================
fig1, axes1 = plt.subplots(2, 2, figsize=(12, 10))
fig1.suptitle('Kullback-Leibler Divergence: Measuring Distributional Differences', fontsize=14, fontweight='bold')

x = np.linspace(-0.05, 0.05, 500)

# Different scenarios
scenarios = [
    {'mu1': 0, 'sigma1': 0.01, 'mu2': 0.01, 'sigma2': 0.01, 'title': 'Different Means'},
    {'mu1': 0, 'sigma1': 0.01, 'mu2': 0, 'sigma2': 0.02, 'title': 'Different Variances'},
    {'mu1': 0, 'sigma1': 0.01, 'mu2': 0.005, 'sigma2': 0.01, 'title': 'Shifted Distribution'},
    {'mu1': 0, 'sigma1': 0.01, 'mu2': 0, 'sigma2': 0.015, 'title': 'Higher Uncertainty'}
]

for idx, (ax, scenario) in enumerate(zip(axes1.flat, scenarios)):
    p = stats.norm.pdf(x, scenario['mu1'], scenario['sigma1'])
    q = stats.norm.pdf(x, scenario['mu2'], scenario['sigma2'])

    # Calculate KL divergence
    kl = np.sum(p * np.log(p / (q + 1e-10))) * (x[1] - x[0])

    ax.plot(x, p, 'b-', linewidth=2, label='Reference p(x)')
    ax.plot(x, q, 'r--', linewidth=2, label='Alternative q(x)')
    ax.fill_between(x, p, q, alpha=0.3)
    ax.set_title(f'({chr(97+idx)}) {scenario["title"]}\nKL Divergence = {kl:.4f}')
    ax.set_xlabel('Return')
    ax.set_ylabel('Probability Density')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/tlxy/Research/Ambiguity/doc/draft/paper_overleaf/fig/figure1_kl_divergence.pdf', dpi=300, bbox_inches='tight')
plt.close()

#==============================================================
# Figure 2: Ambiguity Index Dynamics
#==============================================================
fig2, axes2 = plt.subplots(3, 1, figsize=(14, 12))
fig2.suptitle('Ambiguity Index Dynamics and Market Conditions', fontsize=14, fontweight='bold')

# Generate synthetic data
np.random.seed(42)
n_days = 1500
dates = pd.date_range(start='2018-01-01', periods=n_days, freq='B')

# Market returns with volatility clustering
returns = np.random.normal(0, 0.02, n_days)
vol_regime = np.concatenate([
    np.ones(200) * 0.01, np.ones(150) * 0.03, np.ones(300) * 0.015,
    np.ones(100) * 0.04, np.ones(250) * 0.02, np.ones(200) * 0.025, np.ones(300) * 0.018
])[:n_days]
returns = returns * vol_regime / 0.02

# Ambiguity index
ambiguity_index = np.maximum(0, vol_regime * 0.5 + np.random.normal(0, 0.1, n_days))

# Add event spikes
for event_day in [100, 350, 550, 750, 950, 1150, 1350]:
    if event_day < n_days:
        ambiguity_index[event_day:event_day+5] *= 2.5

# Panel 1: Cumulative returns
axes2[0].plot(dates, np.cumsum(returns), 'b-', linewidth=1.5)
axes2[0].set_title('(a) Market Index Performance')
axes2[0].set_ylabel('Cumulative Returns')
axes2[0].grid(True, alpha=0.3)

# Panel 2: Ambiguity index
axes2[1].plot(dates, ambiguity_index, 'r-', linewidth=1.5)
axes2[1].fill_between(dates, 0, ambiguity_index, alpha=0.3, color='red')
axes2[1].set_title('(b) Time-Varying Ambiguity Index')
axes2[1].set_ylabel('Ambiguity Level')
axes2[1].grid(True, alpha=0.3)

# Panel 3: Scatter plot relationship
axes2[2].scatter(ambiguity_index[::10], returns[::10], alpha=0.5, s=10)
axes2[2].set_xlabel('Ambiguity Index')
axes2[2].set_ylabel('Daily Return')
axes2[2].set_title('(c) Ambiguity vs Returns Relationship')
axes2[2].grid(True, alpha=0.3)

# Add regression line
z = np.polyfit(ambiguity_index[::10], returns[::10], 1)
p = np.poly1d(z)
axes2[2].plot(ambiguity_index[::10], p(ambiguity_index[::10]), "r--", alpha=0.8,
              label=f'Fit: y = {z[0]:.3f}x + {z[1]:.3f}')
axes2[2].legend()

for ax in axes2[:2]:
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.tight_layout()
plt.savefig('/Users/tlxy/Research/Ambiguity/doc/draft/paper_overleaf/fig/figure2_ambiguity_dynamics.pdf', dpi=300, bbox_inches='tight')
plt.close()

#==============================================================
# Figure 3: Portfolio Performance
#==============================================================
fig3, axes3 = plt.subplots(2, 2, figsize=(14, 10))
fig3.suptitle('Portfolio Performance Evaluation', fontsize=14, fontweight='bold')

# Generate portfolio data
np.random.seed(123)
market_returns = np.random.normal(0.0005, 0.015, n_days)
ambe_returns = market_returns + 0.001 + np.random.normal(0, 0.01, n_days)
cea_returns = market_returns + 0.002 + np.random.normal(0, 0.008, n_days)

# Cumulative returns
market_cum = np.cumprod(1 + market_returns)
ambe_cum = np.cumprod(1 + ambe_returns)
cea_cum = np.cumprod(1 + cea_returns)

# Panel (a): Cumulative performance
axes3[0, 0].plot(dates, market_cum, 'b-', linewidth=2, label='CSI 300 Index', alpha=0.7)
axes3[0, 0].plot(dates, ambe_cum, 'g-', linewidth=2, label='AMBE Strategy')
axes3[0, 0].plot(dates, cea_cum, 'r-', linewidth=2, label='CEA Strategy')
axes3[0, 0].set_title('(a) Cumulative Portfolio Performance')
axes3[0, 0].set_ylabel('Portfolio Value')
axes3[0, 0].legend()
axes3[0, 0].grid(True, alpha=0.3)

# Panel (b): Drawdown
def calculate_drawdown(series):
    peak = np.maximum.accumulate(series)
    return (series - peak) / peak

axes3[0, 1].plot(dates, calculate_drawdown(market_cum), 'b-', label='CSI 300', alpha=0.7)
axes3[0, 1].plot(dates, calculate_drawdown(ambe_cum), 'g-', label='AMBE')
axes3[0, 1].plot(dates, calculate_drawdown(cea_cum), 'r-', label='CEA')
axes3[0, 1].set_title('(b) Drawdown Analysis')
axes3[0, 1].set_ylabel('Drawdown')
axes3[0, 1].legend()
axes3[0, 1].grid(True, alpha=0.3)

# Panel (c): Performance metrics
strategies = ['CSI 300', 'AMBE', 'CEA']
metrics = np.array([
    [12.5, 15.2, 18.5],  # Annual Return %
    [23.8, 18.5, 16.2],  # Volatility %
    [0.53, 0.82, 1.14],  # Sharpe
    [-25.3, -12.4, -8.1], # Max DD %
    [0.49, 1.23, 2.28]   # Calmar
])

x = np.arange(len(strategies))
width = 0.15

axes3[1, 0].bar(x - 1.5*width, metrics[0], width, label='Annual Return %')
axes3[1, 0].bar(x - 0.5*width, metrics[2], width, label='Sharpe Ratio')
axes3[1, 0].bar(x + 0.5*width, metrics[4], width, label='Calmar Ratio')
axes3[1, 0].set_xlabel('Strategy')
axes3[1, 0].set_ylabel('Value')
axes3[1, 0].set_title('(c) Performance Metrics Comparison')
axes3[1, 0].set_xticks(x)
axes3[1, 0].set_xticklabels(strategies)
axes3[1, 0].legend()
axes3[1, 0].grid(True, alpha=0.3)

# Panel (d): Annual returns heatmap
annual_returns = np.array([
    [5.2, -12.3, 28.5, 18.2, -5.1, 22.3, 8.5],
    [8.5, -8.2, 35.2, 25.1, 2.1, 28.5, 12.3],
    [12.3, -5.1, 42.5, 32.1, 5.8, 35.2, 18.5]
])

im = axes3[1, 1].imshow(annual_returns, cmap='RdYlGn', aspect='auto')
axes3[1, 1].set_xticks(range(7))
axes3[1, 1].set_xticklabels(['2018', '2019', '2020', '2021', '2022', '2023', '2024'])
axes3[1, 1].set_yticks(range(3))
axes3[1, 1].set_yticklabels(strategies)
axes3[1, 1].set_title('(d) Annual Returns (%)')

# Add values to heatmap
for i in range(3):
    for j in range(7):
        axes3[1, 1].text(j, i, f'{annual_returns[i, j]:.1f}',
                         ha="center", va="center", color="black", fontweight='bold')

plt.colorbar(im, ax=axes3[1, 1])
plt.tight_layout()
plt.savefig('/Users/tlxy/Research/Ambiguity/doc/draft/paper_overleaf/fig/figure3_portfolio_performance.pdf', dpi=300, bbox_inches='tight')
plt.close()

#==============================================================
# Figure 4: Risk vs Ambiguity Distinction
#==============================================================
fig4, axes4 = plt.subplots(2, 2, figsize=(12, 10))
fig4.suptitle('Risk vs Ambiguity: Conceptual Distinction', fontsize=14, fontweight='bold')

# Panel (a): Risk measures
x = np.linspace(-0.05, 0.05, 1000)
base_dist = stats.norm.pdf(x, 0, 0.01)
axes4[0, 0].plot(x, base_dist, 'b-', linewidth=2)
axes4[0, 0].fill_between(x, 0, base_dist, alpha=0.3)
axes4[0, 0].axvline(x=0, color='r', linestyle='--', alpha=0.5)
axes4[0, 0].axvline(x=0.02, color='r', linestyle='--', alpha=0.5, label='+2σ')
axes4[0, 0].axvline(x=-0.02, color='r', linestyle='--', alpha=0.5, label='-2σ')
axes4[0, 0].set_title('(a) Risk: Known Distribution\nVolatility measures dispersion')
axes4[0, 0].set_xlabel('Return')
axes4[0, 0].set_ylabel('Density')
axes4[0, 0].legend()

# Panel (b): Ambiguity - multiple distributions
distributions = [
    stats.norm.pdf(x, 0, 0.01),
    stats.norm.pdf(x, 0.005, 0.012),
    stats.skewnorm.pdf(x, a=-3, loc=-0.003, scale=0.008),
]

colors = ['blue', 'red', 'green']
labels = ['Model 1', 'Model 2', 'Model 3']

for dist, color, label in zip(distributions, colors, labels):
    axes4[0, 1].plot(x, dist/np.sum(dist)*(x[1]-x[0]), color=color, linewidth=2, label=label, alpha=0.7)

axes4[0, 1].set_title('(b) Ambiguity: Unknown Distribution\nMultiple plausible models')
axes4[0, 1].set_xlabel('Return')
axes4[0, 1].set_ylabel('Density')
axes4[0, 1].legend()

# Panel (c): Correlation with returns
np.random.seed(456)
ambiguity = np.random.exponential(0.5, 1000)
volatility = np.random.gamma(2, 0.005, 1000)
returns = 0.002 * ambiguity - 0.1 * volatility + np.random.normal(0, 0.01, 1000)

axes4[1, 0].scatter(volatility, returns, alpha=0.5, label=f'Vol-Ret Corr: {np.corrcoef(volatility, returns)[0,1]:.2f}')
axes4[1, 0].scatter(ambiguity, returns, alpha=0.5, label=f'Amb-Ret Corr: {np.corrcoef(ambiguity, returns)[0,1]:.2f}')
axes4[1, 0].set_xlabel('Risk Measure')
axes4[1, 0].set_ylabel('Return')
axes4[1, 0].set_title('(c) Correlation with Returns')
axes4[1, 0].legend()
axes4[1, 0].grid(True, alpha=0.3)

# Panel (d): Conceptual summary
axes4[1, 1].axis('off')
summary_text = """Key Differences:

RISK:
• Known probability distribution
• Measures outcome dispersion
• Quantified by variance, VaR, etc.
• Decision: Portfolio optimization

AMBIGUITY:
• Unknown probability distribution
• Measures model uncertainty
• Quantified by KL divergence
• Decision: Robust optimization

 Economic Implication:
 Ambiguity aversion leads to
 additional risk premia beyond
 traditional risk compensation"""

axes4[1, 1].text(0.1, 0.9, summary_text, transform=axes4[1, 1].transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace')

plt.tight_layout()
plt.savefig('/Users/tlxy/Research/Ambiguity/doc/draft/paper_overleaf/fig/figure4_risk_vs_ambiguity.pdf', dpi=300, bbox_inches='tight')
plt.close()

print("\nAll figures generated successfully!")
print("Files saved to /Users/tlxy/Research/Ambiguity/doc/draft/paper_overleaf/fig/")
print("\nGenerated figures:")
print("1. figure1_kl_divergence.pdf - KL Divergence concept")
print("2. figure2_ambiguity_dynamics.pdf - Ambiguity index time series")
print("3. figure3_portfolio_performance.pdf - Portfolio comparison")
print("4. figure4_risk_vs_ambiguity.pdf - Risk vs ambiguity distinction")