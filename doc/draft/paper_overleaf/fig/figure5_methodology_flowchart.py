"""
Figure 5: Methodology Flowchart
Shows the step-by-step process of calculating the ambiguity index
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch, Circle
import numpy as np

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(14, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Title
ax.text(5, 9.5, 'Cross-Entropy Ambiguity Index: Methodology Framework',
        ha='center', va='center', fontsize=16, fontweight='bold')

# Define process boxes and their positions
processes = [
    # Box format: (x_center, y_center, width, height, text, color)
    (2, 8.5, 2.5, 0.8, 'Step 1:\nIntraday Data Collection\n(1-minute frequency)', 'lightblue'),
    (2, 7, 2.5, 0.8, 'Step 2:\nDaily Return Distribution\nBin returns into 202 bins', 'lightgreen'),
    (5, 7, 2.5, 0.8, 'Step 3:\nSliding Window Analysis\n20-day historical window', 'lightyellow'),
    (8, 8.5, 2.5, 0.8, 'Step 4:\nK-means Clustering\nIdentify 4 market regimes', 'lightgray'),
    (8, 7, 2.5, 0.8, 'Step 5:\nBenchmark Distributions\nPi = {p1, p2, p3, p4}', 'lightcoral'),
    (5, 5, 2.5, 0.8, 'Step 6:\nCalculate KL Divergence\nD KL(q_t || p_i) for all i', 'orange'),
    (5, 3.5, 2.5, 0.8, 'Step 7:\nSelect Minimum Divergence\np* = argmin D KL', 'plum'),
    (5, 2, 2.5, 0.8, 'Step 8:\nAmbiguity Index\nA_CEA_t = D KL(q_t || p*)', 'pink'),
    (2, 3.5, 2.5, 0.8, 'Output:\nAmbiguity Time Series\nDaily measure', 'lightcyan'),
    (8, 3.5, 2.5, 0.8, 'Application:\nPortfolio Optimization\nRisk management', 'mistyrose'),
]

# Draw process boxes
for proc in processes:
    x, y, w, h, text, color = proc

    # Calculate box position (center to corner)
    x_pos = x - w/2
    y_pos = y - h/2

    # Create fancy box
    box = FancyBboxPatch((x_pos, y_pos), w, h,
                         boxstyle="round,pad=0.05",
                         facecolor=color,
                         edgecolor='black',
                         linewidth=1.5)
    ax.add_patch(box)

    # Add text
    ax.text(x, y, text, ha='center', va='center', fontsize=9, fontweight='bold')

# Add flow arrows
arrows = [
    # From Step 1 to Step 2
    ((2, 8.1), (2, 7.4)),
    # From Step 2 to Step 3
    ((3.25, 6.6), (5, 7.4)),
    # From Step 3 to Step 4
    ((6.25, 7.4), (8, 8.1)),
    # From Step 4 to Step 5
    ((8, 8.1), (8, 7.4)),
    # From Step 5 to Step 6
    ((8, 6.6), (6.25, 5.4)),
    # From Step 6 to Step 7
    ((5, 4.6), (5, 3.9)),
    # From Step 7 to Step 8
    ((5, 3.1), (5, 2.4)),
    # From Step 8 to Output
    ((3.75, 2), (3.25, 3.5)),
    # From Step 8 to Application
    ((6.25, 2), (6.75, 3.5)),
]

for start, end in arrows:
    arrow = ConnectionPatch((start[0], start[1]), (end[0], end[1]),
                           "data", "data",
                           arrowstyle="->,head_width=0.3,head_length=0.3",
                           shrinkA=5, shrinkB=5,
                           mutation_scale=20,
                           fc="black", lw=1.5)
    ax.add_patch(arrow)

# Add mathematical formulation box
math_box = FancyBboxPatch((0.5, 0.5), 4, 1.5,
                          boxstyle="round,pad=0.1",
                          facecolor='white',
                          edgecolor='navy',
                          linewidth=2)
ax.add_patch(math_box)

math_text = """Mathematical Formulation:

  Cross-Entropy: H(p,q) = -∑ p(x)log q(x)
  KL Divergence: D_KL(p||q) = ∑ p(x)log[p(x)/q(x)]

  Ambiguity Index: A_CEA_t = min_i D_KL(q_t||p_i)

  where:
    - q_t: Daily return distribution
    - p_i: i-th benchmark distribution
    - i ∈ {1,2,3,4}: Market regimes"""

ax.text(2.5, 1.25, math_text, ha='center', va='center', fontsize=8, fontfamily='monospace')

# Add key insights box
insight_box = FancyBboxPatch((5.5, 0.5), 4, 1.5,
                            boxstyle="round,pad=0.1",
                            facecolor='lightyellow',
                            edgecolor='darkgreen',
                            linewidth=2)
ax.add_patch(insight_box)

insight_text = """Key Features:

  • Captures full distribution shape
  • Sensitive to tail behavior
  • Adaptive to market regimes
  • Information-theoretic foundation
  • Real-time computation
  • Model uncertainty quantification

Advantages over Traditional Risk:
  - Beyond moments (variance, skewness)
  - Distributional comparison
  - Robust to extreme events
  - Dynamic adaptation"""

ax.text(7.5, 1.25, insight_text, ha='center', va='center', fontsize=8, fontfamily='monospace')

# Add cycle indicator
cycle = Circle((1, 1), 0.5, facecolor='lightcyan', edgecolor='blue', linewidth=2)
ax.add_patch(cycle)
ax.text(1, 1, 'Daily\nUpdate', ha='center', va='center', fontsize=10, fontweight='bold')

# Add cycle arrow
cycle_arrow = ConnectionPatch((1, 1.5), (1, 8.5),
                            "data", "data",
                            arrowstyle="->,head_width=0.4,head_length=0.4",
                            shrinkA=5, shrinkB=5,
                            mutation_scale=20,
                            fc="blue", lw=2,
                            linestyle='--', alpha=0.5)
ax.add_patch(cycle_arrow)
ax.text(0.5, 5, 'Iterative\nProcess', ha='center', va='center', fontsize=9,
        rotation=90, color='blue', fontweight='bold')

plt.savefig('/Users/tlxy/Research/Ambiguity/doc/draft/paper_overleaf/fig/figure5_methodology_flowchart.pdf',
            dpi=300, bbox_inches='tight')
plt.close()

print("Figure 5 (Methodology Flowchart) generated successfully!")