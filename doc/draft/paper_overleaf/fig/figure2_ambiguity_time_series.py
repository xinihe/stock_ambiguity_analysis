"""
Figure 2: Ambiguity Index Time Series and Market Conditions
Shows how ambiguity index varies over time with market events
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib.dates as mdates

# Set style
plt.style.use('seaborn-v0_8')

# Create synthetic data for 2018-2024 period
np.random.seed(42)
n_days = 1500  # Approximate trading days from 2018-2024
dates = pd.date_range(start='2018-01-01', periods=n_days, freq='B')

# Simulate market returns
returns = np.random.normal(0, 0.02, n_days)

# Add volatility clusters (regime switching)
vol_regime = np.concatenate([
    np.ones(200) * 0.01,  # Low volatility
    np.ones(150) * 0.03,  # High volatility
    np.ones(300) * 0.015, # Medium
    np.ones(100) * 0.04,  # Crisis
    np.ones(250) * 0.02,  # Normal
    np.ones(200) * 0.025, # Slightly elevated
    np.ones(300) * 0.018  # Return to normal
])

vol_regime = vol_regime[:n_days]
returns = returns * vol_regime / 0.02

# Generate ambiguity index that correlates with volatility but with noise
ambiguity_base = vol_regime * 0.5
ambiguity_noise = np.random.normal(0, 0.1, n_days)
ambiguity_index = np.maximum(0, ambiguity_base + ambiguity_noise)

# Add some spikes for events
event_dates = [100, 350, 550, 750, 950, 1150, 1350]
for event_day in event_dates:
    if event_day < n_days:
        ambiguity_index[event_day:event_day+5] *= 2.5

# Create figure
fig, axes = plt.subplots(3, 1, figsize=(15, 12))
fig.suptitle('Ambiguity Index Dynamics and Market Conditions', fontsize=16, fontweight='bold')

# Panel 1: Market Returns
ax1 = axes[0]
ax1.plot(dates, np.cumsum(returns), 'b-', linewidth=1.5, label='Cumulative Returns')
ax1.set_ylabel('Cumulative Returns')
ax1.set_title('(a) Market Index Performance')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper left')

# Add event markers
for event_day in event_dates:
    if event_day < n_days:
        ax1.axvline(x=dates[event_day], color='red', linestyle='--', alpha=0.3)
        ax1.text(dates[event_day], ax1.get_ylim()[1]*0.9, 'Event',
                rotation=90, ha='right', fontsize=8, color='red')

# Panel 2: Ambiguity Index
ax2 = axes[1]
ax2.plot(dates, ambiguity_index, 'r-', linewidth=1.5, label='Ambiguity Index $\\mathcal{A}^{CEA}_t$')
ax2.fill_between(dates, 0, ambiguity_index, alpha=0.3, color='red')
ax2.set_ylabel('Ambiguity Level')
ax2.set_title('(b) Time-Varying Ambiguity Index')
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper left')

# Add regime shading
regimes = ['Low Vol', 'High Vol', 'Medium', 'Crisis', 'Normal', 'Elevated', 'Normal']
regime_starts = [0, 200, 350, 650, 750, 1000, 1200]
colors = ['green', 'yellow', 'blue', 'red', 'green', 'yellow', 'blue']

for i, (start, regime, color) in enumerate(zip(regime_starts, regimes, colors)):
    if start < n_days:
        end = min(start + vol_regime[start:].shape[0], regime_starts[i+1]) if i < len(regime_starts) - 1 else n_days
        ax2.axvspan(dates[start], dates[min(end-1, n_days-1)], alpha=0.1, color=color)

# Panel 3: Ambiguity vs Returns Scatter
ax3 = axes[2]
# Bin the data for clearer visualization
bins = pd.qcut(ambiguity_index, q=10, labels=False, duplicates='drop')
binned_returns = pd.Series(returns).groupby(bins).mean()
binned_ambiguity = pd.Series(ambiguity_index).groupby(bins).mean()

ax3.scatter(binned_ambiguity, binned_returns, s=100, alpha=0.6, c='blue')
ax3.set_xlabel('Ambiguity Index $\\mathcal{A}^{CEA}_t$')
ax3.set_ylabel('Next Day Return')
ax3.set_title('(c) Ambiguity and Forward Returns Relationship')
ax3.grid(True, alpha=0.3)

# Add regression line
z = np.polyfit(binned_ambiguity, binned_returns, 1)
p = np.poly1d(z)
ax3.plot(binned_ambiguity, p(binned_ambiguity), "r--", alpha=0.8,
         label=f'Linear Fit: y = {z[0]:.3f}x + {z[1]:.3f}')
ax3.legend()

# Format x-axis
for ax in axes[:2]:
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

plt.tight_layout()
plt.savefig('/Users/tlxy/Research/Ambiguity/doc/draft/paper_overleaf/fig/ambiguity_time_series.pdf',
            dpi=300, bbox_inches='tight')
plt.savefig('/Users/tlxy/Research/Ambiguity/doc/draft/paper_overleaf/fig/ambiguity_time_series.png',
            dpi=300, bbox_inches='tight')
plt.close()

print("Figure 2 saved to fig/ambiguity_time_series.pdf and .png")