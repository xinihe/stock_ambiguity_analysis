"""
Figure 5: Risk vs Ambiguity Distinction
Illustrates the difference between traditional risk measures and ambiguity
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Create figure with multiple subplots
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
fig.suptitle('Risk vs Ambiguity: Conceptual Distinction and Measurement', fontsize=16, fontweight='bold')

# Define x-axis for returns
x = np.linspace(-0.05, 0.05, 1000)

# Panel 1: Traditional Risk Measures (Distribution Moments)
ax1 = fig.add_subplot(gs[0, 0])
# Base distribution
base_dist = stats.norm.pdf(x, 0, 0.01)
ax1.plot(x, base_dist, 'b-', linewidth=2, label='Base Distribution')
ax1.fill_between(x, 0, base_dist, alpha=0.3)
ax1.set_title('(a) Volatility (2nd Moment)\nMeasures dispersion')
ax1.set_xlabel('Return')
ax1.set_ylabel('Density')

# Show volatility
ax1.axvline(x=0, color='r', linestyle='--', alpha=0.5)
ax1.axvline(x=0.02, color='r', linestyle='--', alpha=0.5, label='+2σ')
ax1.axvline(x=-0.02, color='r', linestyle='--', alpha=0.5, label='-2σ')
ax1.legend(fontsize=8)

# Panel 2: Skewness
ax2 = fig.add_subplot(gs[0, 1])
skewed_dist = stats.skewnorm.pdf(x, a=5, loc=0.005, scale=0.01)
skewed_dist = skewed_dist / (np.sum(skewed_dist) * (x[1] - x[0]))
ax2.plot(x, skewed_dist, 'g-', linewidth=2, label='Skewed Distribution')
ax2.plot(x, base_dist, 'b--', linewidth=1, alpha=0.5, label='Symmetric')
ax2.fill_between(x, 0, skewed_dist, alpha=0.3, color='green')
ax2.set_title('(b) Skewness (3rd Moment)\nMeasures asymmetry')
ax2.set_xlabel('Return')
ax2.set_ylabel('Density')
ax2.legend(fontsize=8)

# Panel 3: Kurtosis
ax3 = fig.add_subplot(gs[0, 2])
heavy_tail_dist = stats.t.pdf(x, df=3, loc=0, scale=0.01)
heavy_tail_dist = heavy_tail_dist / (np.sum(heavy_tail_dist) * (x[1] - x[0]))
ax3.plot(x, heavy_tail_dist, 'r-', linewidth=2, label='Heavy-tailed')
ax3.plot(x, base_dist, 'b--', linewidth=1, alpha=0.5, label='Normal')
ax3.fill_between(x, 0, heavy_tail_dist, alpha=0.3, color='red')
ax3.set_title('(c) Kurtosis (4th Moment)\nMeasures tail thickness')
ax3.set_xlabel('Return')
ax3.set_ylabel('Density')
ax3.legend(fontsize=8)

# Panel 4: Ambiguity - Distribution Comparison
ax4 = fig.add_subplot(gs[1, :])
# Multiple distributions for ambiguity illustration
distributions = [
    stats.norm.pdf(x, 0, 0.01),
    stats.norm.pdf(x, 0.005, 0.012),
    stats.skewnorm.pdf(x, a=-3, loc=-0.003, scale=0.008),
    stats.t.pdf(x, df=4, loc=0.002, scale=0.01)
]

# Normalize
for i in range(len(distributions)):
    distributions[i] = distributions[i] / (np.sum(distributions[i]) * (x[1] - x[0]))

colors = ['blue', 'red', 'green', 'orange']
labels = ['$p_1$: Normal', '$p_2$: Shifted', '$p_3$: Skewed', '$p_4$: Heavy-tail']

for dist, color, label in zip(distributions, colors, labels):
    ax4.plot(x, dist, color=color, linewidth=2, label=label, alpha=0.7)

ax4.set_title('(d) Ambiguity: Multiple Plausible Distributions\nUncertainty about the true distribution')
ax4.set_xlabel('Return')
ax4.set_ylabel('Density')
ax4.legend(loc='upper right')

# Panel 5: Correlation Heatmap
ax5 = fig.add_subplot(gs[2, 0])
# Create synthetic correlation matrix
measures = ['Volatility', 'Skewness', 'Kurtosis', 'Ambiguity ($\\mathcal{A}^{CEA}_t$)']
corr_matrix = np.array([
    [1.00, 0.15, 0.30, 0.02],
    [0.15, 1.00, -0.10, 0.00],
    [0.30, -0.10, 1.00, 0.01],
    [0.02, 0.00, 0.01, 1.00]
])

im = ax5.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
ax5.set_xticks(range(len(measures)))
ax5.set_yticks(range(len(measures)))
ax5.set_xticklabels(measures, rotation=45, ha='right', fontsize=10)
ax5.set_yticklabels(measures, fontsize=10)

# Add correlation values
for i in range(len(measures)):
    for j in range(len(measures)):
        text = ax5.text(j, i, f'{corr_matrix[i, j]:.2f}',
                       ha="center", va="center", color="black", fontsize=10)

ax5.set_title('(e) Correlation Matrix\nAmbiguity is orthogonal to risk')

# Panel 6: Information Tree
ax6 = fig.add_subplot(gs[2, 1])
ax6.axis('off')

# Create decision tree visualization
tree_text = """
Information Hierarchy:

Known:
• Mean Return
• Volatility

Unknown (Risk):
• Next period's return
• Magnitude of deviation

Unknown (Ambiguity):
• True distribution
• Model specification
• Parameter stability
• Structural breaks

→ Risk: Uncertainty about outcomes
→ Ambiguity: Uncertainty about models
"""

ax6.text(0.1, 0.9, tree_text, transform=ax6.transAxes,
         fontsize=10, verticalalignment='top', fontfamily='monospace')

# Panel 7: Economic Interpretation
ax7 = fig.add_subplot(gs[2, 2])
ax7.axis('off')

economic_text = """
Economic Implications:

Risk Aversion:
• Preferences over outcomes
• Mean-variance optimization
• Utility: $U = E[r] - \\frac{\\gamma}{2}\\sigma^2$

Ambiguity Aversion:
• Preferences over models
• Model uncertainty penalties
• Utility: $U = \\min_{p \\in \\mathcal{P}} E_p[U]$

Key Insight:
Ambiguity captures uncertainty
beyond traditional risk measures,
reflecting model misspecification
and informational instability.
"""

ax7.text(0.1, 0.9, economic_text, transform=ax7.transAxes,
         fontsize=10, verticalalignment='top', fontfamily='monospace')

plt.savefig('/Users/tlxy/Research/Ambiguity/doc/draft/paper_overleaf/fig/risk_ambiguity_distinction.pdf',
            dpi=300, bbox_inches='tight')
plt.savefig('/Users/tlxy/Research/Ambiguity/doc/draft/paper_overleaf/fig/risk_ambiguity_distinction.png',
            dpi=300, bbox_inches='tight')
plt.close()

print("Figure 5 saved to fig/risk_ambiguity_distinction.pdf and .png")