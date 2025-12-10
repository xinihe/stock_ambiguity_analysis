"""
Figure 1: Visualization of KL Divergence concept
Shows how KL divergence measures the difference between probability distributions
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Create figure with subplots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Kullback-Leibler Divergence: Measuring Distributional Differences', fontsize=16, fontweight='bold')

# Define x-axis
x = np.linspace(-0.05, 0.05, 500)

# Scenario 1: Normal vs Normal (different means)
mu1, sigma1 = 0, 0.01
mu2, sigma2 = 0.01, 0.01
p1 = stats.norm.pdf(x, mu1, sigma1)
q1 = stats.norm.pdf(x, mu2, sigma2)

# Calculate KL divergence
kl1 = np.sum(p1 * np.log(p1 / (q1 + 1e-10))) * (x[1] - x[0])

axes[0, 0].plot(x, p1, 'b-', linewidth=2, label='Reference $p(x)$')
axes[0, 0].plot(x, q1, 'r--', linewidth=2, label='Alternative $q(x)$')
axes[0, 0].fill_between(x, p1, q1, alpha=0.3)
axes[0, 0].set_title(f'(a) Different Means\nKL Divergence = {kl1:.4f}')
axes[0, 0].legend()
axes[0, 0].set_xlabel('Return')
axes[0, 0].set_ylabel('Probability Density')

# Scenario 2: Normal vs Normal (different variances)
mu1, sigma1 = 0, 0.01
mu2, sigma2 = 0, 0.02
p2 = stats.norm.pdf(x, mu1, sigma1)
q2 = stats.norm.pdf(x, mu2, sigma2)

kl2 = np.sum(p2 * np.log(p2 / (q2 + 1e-10))) * (x[1] - x[0])

axes[0, 1].plot(x, p2, 'b-', linewidth=2, label='Reference $p(x)$')
axes[0, 1].plot(x, q2, 'r--', linewidth=2, label='Alternative $q(x)$')
axes[0, 1].fill_between(x, p2, q2, alpha=0.3)
axes[0, 1].set_title(f'(b) Different Variances\n$D_{{KL}}(p \\| q) = {kl2:.4f}$')
axes[0, 1].legend()
axes[0, 1].set_xlabel('Return')
axes[0, 1].set_ylabel('Probability Density')

# Scenario 3: Normal vs Skewed
mu1, sigma1 = 0, 0.01
p3 = stats.norm.pdf(x, mu1, sigma1)
q3 = stats.skewnorm.pdf(x, a=5, loc=0.005, scale=0.01)

# Normalize
q3 = q3 / (np.sum(q3) * (x[1] - x[0]))

kl3 = np.sum(p3 * np.log(p3 / (q3 + 1e-10))) * (x[1] - x[0])

axes[0, 2].plot(x, p3, 'b-', linewidth=2, label='Reference $p(x)$')
axes[0, 2].plot(x, q3, 'r--', linewidth=2, label='Alternative $q(x)$')
axes[0, 2].fill_between(x, p3, q3, alpha=0.3)
axes[0, 2].set_title(f'(c) Skewed Distribution\n$D_{{KL}}(p \\| q) = {kl3:.4f}$')
axes[0, 2].legend()
axes[0, 2].set_xlabel('Return')
axes[0, 2].set_ylabel('Probability Density')

# Scenario 4: Heavy-tailed distributions
df1 = 5  # Degrees of freedom for t-distribution
p4 = stats.t.pdf(x, df=df1, loc=0, scale=0.01)
df2 = 10
q4 = stats.t.pdf(x, df=df2, loc=0, scale=0.01)

# Normalize
p4 = p4 / (np.sum(p4) * (x[1] - x[0]))
q4 = q4 / (np.sum(q4) * (x[1] - x[0]))

kl4 = np.sum(p4 * np.log(p4 / (q4 + 1e-10))) * (x[1] - x[0])

axes[1, 0].plot(x, p4, 'b-', linewidth=2, label='Reference $p(x)$ (df=5)')
axes[1, 0].plot(x, q4, 'r--', linewidth=2, label='Alternative $q(x)$ (df=10)')
axes[1, 0].fill_between(x, p4, q4, alpha=0.3)
axes[1, 0].set_title(f'(d) Heavy Tails\n$D_{{KL}}(p \\| q) = {kl4:.4f}$')
axes[1, 0].legend()
axes[1, 0].set_xlabel('Return')
axes[1, 0].set_ylabel('Probability Density')

# Scenario 5: Bimodal distribution
p5 = stats.norm.pdf(x, 0, 0.01)
q5 = 0.5 * stats.norm.pdf(x, -0.01, 0.005) + 0.5 * stats.norm.pdf(x, 0.01, 0.005)

# Normalize
q5 = q5 / (np.sum(q5) * (x[1] - x[0]))

kl5 = np.sum(p5 * np.log(p5 / (q5 + 1e-10))) * (x[1] - x[0])

axes[1, 1].plot(x, p5, 'b-', linewidth=2, label='Reference $p(x)$')
axes[1, 1].plot(x, q5, 'r--', linewidth=2, label='Alternative $q(x)$')
axes[1, 1].fill_between(x, p5, q5, alpha=0.3)
axes[1, 1].set_title(f'(e) Bimodal Distribution\n$D_{{KL}}(p \\| q) = {kl5:.4f}$')
axes[1, 1].legend()
axes[1, 1].set_xlabel('Return')
axes[1, 1].set_ylabel('Probability Density')

# KL divergence visualization (heat map)
axes[1, 2].axis('off')
text_content = """
Key Properties of KL Divergence:

• Non-negative: D_{KL}(p || q) >= 0
• Zero iff identical: D_{KL}(p || q) = 0 iff p = q
• Asymmetric: D_{KL}(p || q) != D_{KL}(q || p)
• Sensitive to tail events
• Information-theoretic interpretation:
  Measures expected "surprise" when using
  q(x) to approximate p(x)

Financial Interpretation:
Higher KL divergence indicates greater
ambiguity in return distributions,
signaling model uncertainty.
"""
axes[1, 2].text(0.1, 0.9, text_content, transform=axes[1, 2].transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace')

plt.tight_layout()
plt.savefig('/Users/tlxy/Research/Ambiguity/doc/draft/paper_overleaf/fig/kl_divergence_visualization.pdf',
            dpi=300, bbox_inches='tight')
plt.savefig('/Users/tlxy/Research/Ambiguity/doc/draft/paper_overleaf/fig/kl_divergence_visualization.png',
            dpi=300, bbox_inches='tight')
plt.close()

print("Figure 1 saved to fig/kl_divergence_visualization.pdf and .png")