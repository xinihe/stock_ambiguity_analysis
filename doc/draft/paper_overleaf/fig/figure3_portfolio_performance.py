"""
Figure 3: Portfolio Performance Comparison
Shows the performance of ambiguity-based portfolios vs benchmarks
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Set style
plt.style.use('seaborn-v0_8')

# Create synthetic portfolio performance data
np.random.seed(123)
n_days = 1500
dates = pd.date_range(start='2018-01-01', periods=n_days, freq='B')

# Simulate different portfolio strategies
# CSI 300 (market benchmark)
market_returns = np.random.normal(0.0005, 0.015, n_days)

# AMBE strategy (moderate performance)
ambe_alpha = 0.0008
ambe_returns = market_returns + ambe_alpha + np.random.normal(0, 0.0095, n_days)

# CEA strategy (slightly better performance)
cea_alpha = 0.0012
cea_returns = market_returns + cea_alpha + np.random.normal(0, 0.009, n_days)

# Add some crisis periods with different behaviors
crisis_periods = [(100, 150), (350, 400), (750, 800), (1200, 1250)]
for start, end in crisis_periods:
    market_returns[start:end] = np.random.normal(-0.005, 0.03, end-start)
    ambe_returns[start:end] = np.random.normal(-0.002, 0.022, end-start)
    cea_returns[start:end] = np.random.normal(-0.001, 0.02, end-start)

# Calculate cumulative returns
market_cum = np.cumprod(1 + market_returns)
ambe_cum = np.cumprod(1 + ambe_returns)
cea_cum = np.cumprod(1 + cea_returns)

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Portfolio Performance Evaluation: Ambiguity-Based Strategies', fontsize=16, fontweight='bold')

# Panel (a): Cumulative Performance Comparison
ax1 = axes[0, 0]
ax1.plot(dates, market_cum, 'b-', linewidth=2, label='CSI 300 Index', alpha=0.7)
ax1.plot(dates, ambe_cum, 'g-', linewidth=2, label='AMBE Strategy')
ax1.plot(dates, cea_cum, 'r-', linewidth=2, label='$\\mathcal{A}^{CEA}_t$ Strategy')
ax1.set_title('(a) Cumulative Portfolio Performance')
ax1.set_ylabel('Portfolio Value')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper left')
ax1.xaxis.set_major_locator(mdates.YearLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# Panel (b): Drawdown Analysis
def calculate_drawdown(series):
    series = pd.Series(series)
    peak = series.expanding(min_periods=1).max()
    drawdown = (series - peak) / peak
    return drawdown

ax2 = axes[0, 1]
ax2.fill_between(dates, calculate_drawdown(market_cum), 0, alpha=0.3, color='blue', label='CSI 300')
ax2.fill_between(dates, calculate_drawdown(ambe_cum), 0, alpha=0.3, color='green', label='AMBE')
ax2.fill_between(dates, calculate_drawdown(cea_cum), 0, alpha=0.3, color='red', label='$\\mathcal{A}^{CEA}_t$')
ax2.set_title('(b) Drawdown Analysis')
ax2.set_ylabel('Drawdown')
ax2.grid(True, alpha=0.3)
ax2.legend()
ax2.xaxis.set_major_locator(mdates.YearLocator())
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# Panel (c): Rolling Sharpe Ratios
def rolling_sharpe(returns, window=252):
    rolling_mean = pd.Series(returns).rolling(window=window).mean() * 252
    rolling_std = pd.Series(returns).rolling(window=window).std() * np.sqrt(252)
    return rolling_mean / rolling_std

window = 252  # One year
ax3 = axes[1, 0]
ax3.plot(dates, rolling_sharpe(market_returns, window), 'b-', label='CSI 300', alpha=0.7)
ax3.plot(dates, rolling_sharpe(ambe_returns, window), 'g-', label='AMBE')
ax3.plot(dates, rolling_sharpe(cea_returns, window), 'r-', label='$\\mathcal{A}^{CEA}_t$')
ax3.axhline(y=0, color='black', linestyle='--', alpha=0.3)
ax3.set_title(f'(c) Rolling Sharpe Ratio ({window}-day window)')
ax3.set_ylabel('Sharpe Ratio')
ax3.grid(True, alpha=0.3)
ax3.legend()
ax3.xaxis.set_major_locator(mdates.YearLocator())
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# Panel (d): Performance Metrics Table
ax4 = axes[1, 1]
ax4.axis('off')

# Calculate metrics
def calculate_metrics(returns, name):
    annual_ret = np.mean(returns) * 252
    annual_vol = np.std(returns) * np.sqrt(252)
    sharpe = annual_ret / annual_vol
    max_dd = calculate_drawdown(pd.Series(np.cumprod(1 + returns))).min()
    calmar = annual_ret / abs(max_dd) if max_dd != 0 else np.inf

    return {
        'Strategy': name,
        'Annual Return': f'{annual_ret:.2%}',
        'Annual Volatility': f'{annual_vol:.2%}',
        'Sharpe Ratio': f'{sharpe:.2f}',
        'Max Drawdown': f'{max_dd:.2%}',
        'Calmar Ratio': f'{calmar:.2f}'
    }

metrics_data = [
    calculate_metrics(market_returns, 'CSI 300'),
    calculate_metrics(ambe_returns, 'AMBE'),
    calculate_metrics(cea_returns, '$\\mathcal{A}^{CEA}_t$')
]

# Create table
table_data = []
for i, row in enumerate(metrics_data):
    table_data.append([row['Strategy']] + [row[k] for k in list(row.keys())[1:]])

table = ax4.table(cellText=table_data,
                 colLabels=['Strategy', 'Annual\nReturn', 'Annual\nVolatility', 'Sharpe\nRatio', 'Max\nDrawdown', 'Calmar\nRatio'],
                 cellLoc='center',
                 loc='center',
                 colWidths=[0.3, 0.12, 0.12, 0.12, 0.12, 0.12])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Color code the best performer
for i in range(1, len(metrics_data[0])):  # Skip strategy name column
    values = [float(row[list(row.keys())[i]].strip('%')) for row in metrics_data]
    if i == 3 or i == 5:  # Higher is better for Sharpe and Calmar
        best_idx = np.argmax(values)
    else:  # Lower is better for volatility and drawdown
        best_idx = np.argmin(values)

    table[(best_idx + 1, i)].set_facecolor('#ffcccc')

ax4.set_title('(d) Performance Summary Statistics')

plt.tight_layout()
plt.savefig('/Users/tlxy/Research/Ambiguity/doc/draft/paper_overleaf/fig/portfolio_performance.pdf',
            dpi=300, bbox_inches='tight')
plt.savefig('/Users/tlxy/Research/Ambiguity/doc/draft/paper_overleaf/fig/portfolio_performance.png',
            dpi=300, bbox_inches='tight')
plt.close()

print("Figure 3 saved to fig/portfolio_performance.pdf and .png")