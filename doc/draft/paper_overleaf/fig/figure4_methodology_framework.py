"""
Figure 4: Methodology Framework Diagram
Illustrates the cross-entropy ambiguity calculation framework
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import matplotlib.patheffects as path_effects

# Set style
plt.style.use('default')

fig, ax = plt.subplots(1, 1, figsize=(14, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Title
title = ax.text(5, 9.5, 'Cross-Entropy Ambiguity ($\\mathcal{A}^{CEA}_t$) Calculation Framework',
               ha='center', va='center', fontsize=16, fontweight='bold')

# Define components
components = [
    # Intraday Data
    {'box': (0.5, 7.5, 2, 1), 'text': 'Intraday Data\n(1-minute returns)', 'color': 'lightblue'},

    # Daily Distribution
    {'box': (0.5, 5.5, 2, 1), 'text': 'Daily Return\nDistribution $q_t$', 'color': 'lightgreen'},

    # Historical Window
    {'box': (4, 7.5, 2, 1), 'text': 'Historical Window\n(20 days)', 'color': 'lightyellow'},

    # Clustering
    {'box': (4, 5.5, 2, 1), 'text': 'K-means Clustering\n(k=4 clusters)', 'color': 'lightgray'},

    # Benchmark Distributions
    {'box': (7.5, 7.5, 2, 1), 'text': 'Benchmark\nDistributions $\\{p_i\\}$', 'color': 'lightcoral'},

    # KL Divergence
    {'box': (4, 3, 2, 1), 'text': 'KL Divergence\n$D_{KL}(q_t || p_i)$', 'color': 'orange'},

    # Ambiguity Index
    {'box': (4, 0.5, 2, 1), 'text': 'Ambiguity Index\n$\\mathcal{A}^{CEA}_t = \\min_i D_{KL}$', 'color': 'plum'},

    # Decision Making
    {'box': (0.5, 3, 2, 1), 'text': 'Portfolio\nDecision', 'color': 'lightpink'},
]

# Draw boxes
for comp in components:
    x, y, w, h = comp['box']
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.05",
                         facecolor=comp['color'],
                         edgecolor='black',
                         linewidth=1.5)
    ax.add_patch(box)

    # Add text
    text = ax.text(x + w/2, y + h/2, comp['text'],
                   ha='center', va='center', fontsize=10, fontweight='bold')
    text.set_path_effects([path_effects.withStroke(linewidth=3, foreground='white')])

# Add arrows/connections
connections = [
    # From intraday data to daily distribution
    {'start': (1.5, 7.5), 'end': (1.5, 6.5)},

    # From historical window to clustering
    {'start': (5, 7.5), 'end': (5, 6.5)},

    # From clustering to benchmarks
    {'start': (6, 6), 'end': (7.5, 8)},

    # From daily distribution to KL divergence
    {'start': (2.5, 6), 'end': (4, 3.5)},

    # From benchmarks to KL divergence
    {'start': (7.5, 7.5), 'end': (6, 3.5)},

    # From KL divergence to ambiguity index
    {'start': (5, 3), 'end': (5, 1.5)},

    # From ambiguity index to decision
    {'start': (4, 1), 'end': (2.5, 3.5)},
]

for conn in connections:
    arrow = ConnectionPatch(conn['start'], conn['end'],
                           "data", "data",
                           arrowstyle="->,head_width=0.3,head_length=0.3",
                           shrinkA=5, shrinkB=5,
                           mutation_scale=20,
                           fc="black", lw=1.5)
    ax.add_patch(arrow)

# Add mathematical formula box
formula_box = FancyBboxPatch((7.5, 0.5), 2, 2,
                             boxstyle="round,pad=0.1",
                             facecolor='white',
                             edgecolor='black',
                             linewidth=2)
ax.add_patch(formula_box)

formula_text = """KL Divergence:

$D_{KL}(p \\| q) = \\sum_x p(x) \\log\\frac{p(x)}{q(x)}$

Key Features:
• Measures distributional
  distance
• Asymmetric property
• Sensitive to tails
• Information-theoretic
  interpretation"""

ax.text(8.5, 1.5, formula_text, ha='center', va='center', fontsize=9)

# Add annotations
annotations = [
    {'pos': (0.2, 6), 'text': 'High-frequency\ndata collection'},
    {'pos': (3.2, 6), 'text': 'Daily\ncalculation'},
    {'pos': (6.5, 8.8), 'text': '4 representative\ndistributions'},
    {'pos': (8.5, 3), 'text': 'Minimum\nselection'},
    {'pos': (3.5, 2.5), 'text': 'Cross-entropy\nmeasurement'},
]

for ann in annotations:
    ax.text(ann['pos'][0], ann['pos'][1], ann['text'],
            ha='center', va='center', fontsize=8, style='italic', color='darkblue')

# Add timeline indicator
timeline_y = 4.5
ax.plot([0.5, 9.5], [timeline_y, timeline_y], 'k--', alpha=0.3, linewidth=1)
ax.text(10, timeline_y, 'Time $t$', ha='left', va='center', fontsize=10, fontweight='bold')

# Add stage indicators
stages = [
    {'x': 1.5, 'text': 'Data'},
    {'x': 5, 'text': 'Analysis'},
    {'x': 8.5, 'text': 'Decision'},
]

for stage in stages:
    ax.plot([stage['x'], stage['x']], [timeline_y-0.1, timeline_y+0.1], 'k-', linewidth=2)
    ax.text(stage['x'], timeline_y-0.3, stage['text'], ha='center', va='top', fontsize=9)

plt.savefig('/Users/tlxy/Research/Ambiguity/doc/draft/paper_overleaf/fig/methodology_framework.pdf',
            dpi=300, bbox_inches='tight')
plt.savefig('/Users/tlxy/Research/Ambiguity/doc/draft/paper_overleaf/fig/methodology_framework.png',
            dpi=300, bbox_inches='tight')
plt.close()

print("Figure 4 saved to fig/methodology_framework.pdf and .png")