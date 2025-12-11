"""
Figure 4: Methodology Framework Diagram
Illustrates the cross-entropy ambiguity calculation framework with organized layout and clear visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch, Circle, Rectangle
import matplotlib.patheffects as path_effects
from scipy import stats

# Set style
plt.style.use('default')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), gridspec_kw={'width_ratios': [1.1, 1]})
fig.suptitle('Cross-Entropy Ambiguity ($\\mathcal{A}^{CEA}_t$) Calculation Framework', fontsize=16, fontweight='bold')

# Panel (a): Organized Framework Layout
ax1.set_xlim(0, 12)
ax1.set_ylim(0, 12)
ax1.axis('off')

ax1.text(6, 11.5, '(a) Methodology Framework', ha='center', va='center', fontsize=14, fontweight='bold')

# Define components with better spatial organization - no overlaps
components = [
    # Top: Data input
    {'id': 1, 'box': (2, 10, 2.5, 0.8), 'text': '1-Minute\nIntraday Data', 'color': '#A5D8FF'},

    # Distribution creation
    {'id': 2, 'box': (2, 8.5, 2.5, 0.8), 'text': 'Daily Return\nDistribution $q_t$', 'color': '#C5F6FA'},

    # Historical analysis (right side)
    {'id': 3, 'box': (6.5, 9.5, 2.5, 0.8), 'text': 'Historical\nWindow (20d)', 'color': '#FFE0B2'},

    # Clustering
    {'id': 4, 'box': (6.5, 8.5, 2.5, 0.8), 'text': 'K-means\nClustering', 'color': '#E1BEE7'},

    # Regime identification
    {'id': 5, 'box': (6.5, 7, 2.5, 0.8), 'text': '4 Market Regimes\nIdentified', 'color': '#FFCDD2'},

    # Benchmark distributions
    {'id': 6, 'box': (6.5, 5.5, 2.5, 0.8), 'text': 'Benchmark\nDistributions $\\{p_i\\}$', 'color': '#F8BBD9'},

    # KL Divergence (center)
    {'id': 7, 'box': (2, 6.5, 2.5, 0.8), 'text': 'KL Divergence\n$D_{KL}(q_t || p_i)$', 'color': '#FFF9C4'},

    # Decision
    {'id': 8, 'box': (2, 5, 2.5, 0.8), 'text': 'Minimum Selection\n$\\min_i D_{KL}$', 'color': '#FFE0E0'},

    # Output
    {'id': 9, 'box': (2, 3.5, 2.5, 0.8), 'text': 'Ambiguity Index\n$\\mathcal{A}^{CEA}_t$', 'color': '#E8F5E9'},

    # Portfolio decision (bottom left)
    {'id': 10, 'box': (0.5, 2, 2, 0.8), 'text': 'Portfolio\nAllocation', 'color': '#E1F5FE'},
]

# Function to draw shadow effect
def add_shadow(x, y, w, h, ax):
    shadow = FancyBboxPatch((x+0.05, y-0.05), w, h,
                           boxstyle="round,pad=0.03",
                           facecolor='gray',
                           alpha=0.15,
                           edgecolor='none',
                           zorder=1)
    ax.add_patch(shadow)

# Draw components with shadows and clear spacing
for comp in components:
    x, y, w, h = comp['box']

    # Add shadow
    add_shadow(x, y, w, h, ax1)

    # Draw main box
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.05",
                         facecolor=comp['color'],
                         edgecolor='black',
                         linewidth=1.8,
                         alpha=0.9,
                         zorder=2)
    ax1.add_patch(box)

    # Add text
    ax1.text(x + w/2, y + h/2, comp['text'],
            ha='center', va='center', fontsize=10.5, fontweight='bold', zorder=3)

# Define clear arrow paths without any overlaps
connections = [
    # 1: Data to distribution
    {'start': (3.25, 10), 'end': (3.25, 9.3), 'label': '1'},

    # 2: Distribution to KL (direct vertical)
    {'start': (3.25, 8.5), 'end': (3.25, 7.3), 'label': '2'},

    # 3: Historical window to clustering
    {'start': (7.75, 9.5), 'end': (7.75, 9.3), 'label': '3'},

    # 4: Clustering to regimes
    {'start': (7.75, 8.5), 'end': (7.75, 7.8), 'label': '4'},

    # 5: Regimes to benchmarks
    {'start': (7.75, 7), 'end': (7.75, 6.3), 'label': '5'},

    # 6: Benchmarks to KL (clear path)
    {'start': (6.5, 5.9), 'end': (4.5, 6.9), 'label': '6'},

    # 7: KL to decision
    {'start': (3.25, 6.5), 'end': (3.25, 5.8), 'label': '7'},

    # 8: Decision to output
    {'start': (3.25, 5), 'end': (3.25, 4.3), 'label': '8'},

    # 9: Output to portfolio
    {'start': (2, 3.5), 'end': (2.5, 2.8), 'label': '9'},
]

# Draw arrows with clear paths
for conn in connections:
    arrow = ConnectionPatch(conn['start'], conn['end'],
                           "data", "data",
                           arrowstyle="->,head_width=0.2,head_length=0.2",
                           shrinkA=5, shrinkB=5,
                           mutation_scale=15,
                           fc="black", ec="black", lw=1.5,
                           zorder=2)
    ax1.add_patch(arrow)

    # Add label circle - offset to avoid overlap
    if 'label' in conn:
        midx = (conn['start'][0] + conn['end'][0]) / 2
        midy = (conn['start'][1] + conn['end'][1]) / 2

        # Offset based on arrow direction
        if conn['label'] == '1' or conn['label'] == '7' or conn['label'] == '8':
            midx += 0.4
        elif conn['label'] == '6':
            midx -= 0.3
            midy -= 0.2
        elif conn['label'] == '9':
            midx += 0.3
            midy -= 0.2

        circle = Circle((midx, midy), 0.15, facecolor='white',
                       edgecolor='black', linewidth=1.2, alpha=0.95, zorder=3)
        ax1.add_patch(circle)
        ax1.text(midx, midy, conn['label'],
                ha='center', va='center', fontsize=8, fontweight='bold', zorder=4)

# Add Key Processes box
key_processes = [
    ('① Data Processing', 'Transform intraday returns'),
    ('② Distribution Creation', 'Build $q_t$ from returns'),
    ('③ Historical Analysis', '20-day window clustering'),
    ('④ Regime Identification', 'K-means finds 4 patterns'),
    ('⑤ Benchmark Creation', 'Generate $\\{p_i\\}$ distributions'),
    ('⑥ KL Comparison', 'Measure distributional distance'),
    ('⑦ Minimum Selection', 'Find $\\min_i D_{KL}$'),
    ('⑧ Index Output', 'Generate $\\mathcal{A}^{CEA}_t$'),
    ('⑨ Portfolio Decision', 'Guide allocation strategy'),
]

# Key Processes box
key_x, key_y = 9.5, 1
key_box = FancyBboxPatch((key_x, key_y), 2.3, 4,
                        boxstyle="round,pad=0.1",
                        facecolor='white',
                        edgecolor='black',
                        linewidth=1.5,
                        alpha=0.95)
ax1.add_patch(key_box)

ax1.text(key_x + 1.15, key_y + 3.7, 'Key Processes',
        ha='center', va='center', fontsize=11, fontweight='bold')

for i, (title, desc) in enumerate(key_processes):
    y_pos = key_y + 3.3 - i * 0.35
    ax1.text(key_x + 0.1, y_pos, title, fontweight='bold', fontsize=8.5)
    ax1.text(key_x + 0.1, y_pos - 0.15, desc, fontsize=7.5, style='italic')

# Panel (b): Multiple Period Distributions - Optimized Layout
ax2.set_xlim(-3.5, 3.5)
ax2.set_ylim(-0.2, 4.2)
ax2.set_xlabel('Return', fontsize=12, fontweight='bold')
ax2.set_ylabel('Probability Density', fontsize=12, fontweight='bold')
ax2.set_title('(b) Multiple Period Distributions', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Generate distributions with adjusted scales for better separation
x = np.linspace(-3.5, 3.5, 1000)

# Adjust distributions to reduce overlap
dist1 = stats.norm.pdf(x, -1.5, 0.7)  # Shifted left, narrower
dist2 = stats.norm.pdf(x, -0.5, 0.9)  # Slightly left
dist3 = stats.norm.pdf(x, 1.0, 0.6)   # Right, narrow
dist4 = stats.norm.pdf(x, 2.0, 0.8)   # Far right

# Current distribution in center
q_t = stats.norm.pdf(x, 0.3, 0.75)

# Scale distributions to prevent overlap
max_height = 3.5
scale_factors = [
    max_height / np.max(dist1) * 0.9,
    max_height / np.max(dist2) * 0.8,
    max_height / np.max(dist3) * 0.7,
    max_height / np.max(dist4) * 0.6,
    max_height / np.max(q_t) * 1.0
]

dist1 *= scale_factors[0]
dist2 *= scale_factors[1]
dist3 *= scale_factors[2]
dist4 *= scale_factors[3]
q_t *= scale_factors[4]

# Plot distributions with distinct colors
ax2.plot(x, dist1, 'b-', linewidth=2.5, label='Normal Market', alpha=0.8, zorder=2)
ax2.plot(x, dist2, 'g-', linewidth=2.5, label='High Volatility', alpha=0.8, zorder=3)
ax2.plot(x, dist3, 'r-', linewidth=2.5, label='Bull Market', alpha=0.8, zorder=4)
ax2.plot(x, dist4, 'm-', linewidth=2.5, label='Bear Market', alpha=0.8, zorder=5)
ax2.plot(x, q_t, 'k--', linewidth=3, label='Current: $q_t$', alpha=0.9, zorder=6)

# Add vertical lines for means with labels
means = [-1.5, -0.5, 1.0, 2.0, 0.3]
colors = ['b', 'g', 'r', 'm', 'k']
labels = ['μ₁', 'μ₂', 'μ₃', 'μ₄', 'q_t']

for mean, color, label in zip(means, colors, labels):
    ax2.axvline(x=mean, color=color, linestyle=':', alpha=0.5, linewidth=1.5)
    # Add mean label at the top
    if label != 'q_t':
        ax2.text(mean, max_height + 0.1, label, ha='center', fontsize=10,
                color=color, fontweight='bold')

# KL divergence text boxes positioned to avoid overlap
kl_values = [
    {'text': '$D_{KL}(q_t || p_1) = 0.45$', 'pos': (-3.3, 3.0), 'color': '#B3D9FF'},
    {'text': '$D_{KL}(q_t || p_2) = 0.32$', 'pos': (-1.3, 2.5), 'color': '#B3FFB3'},
    {'text': '$D_{KL}(q_t || p_3) = 0.58$', 'pos': (1.3, 2.0), 'color': '#FFB3B3'},
    {'text': '$D_{KL}(q_t || p_4) = 0.91$', 'pos': (2.5, 1.5), 'color': '#FFB3FF'},
]

for kv in kl_values:
    # Background box for text
    bbox_props = dict(boxstyle="round,pad=0.3", facecolor=kv['color'],
                     alpha=0.4, edgecolor='gray', linewidth=1)
    ax2.text(kv['pos'][0], kv['pos'][1], kv['text'], fontsize=9,
            bbox=bbox_props, zorder=7)

# Highlight minimum value
ax2.annotate('Minimum = 0.32\n(Bear Market Regime)',
            xy=(-0.5, 1.5), xytext=(-2.5, 0.5),
            fontsize=10, fontweight='bold', color='darkgreen',
            arrowprops=dict(arrowstyle='->', color='darkgreen', lw=2),
            bbox=dict(boxstyle="round,pad=0.4", facecolor='yellow',
                     alpha=0.6, edgecolor='darkgreen', linewidth=2),
            zorder=8)

# Adjust legend position to avoid overlap
ax2.legend(loc='upper right', fontsize=10, framealpha=0.95,
          bbox_to_anchor=(1.0, 0.98))

plt.tight_layout()
plt.savefig('/Users/tlxy/Research/Ambiguity/doc/draft/paper_overleaf/fig/methodology_framework.pdf',
            dpi=300, bbox_inches='tight')
plt.savefig('/Users/tlxy/Research/Ambiguity/doc/draft/paper_overleaf/fig/methodology_framework.png',
            dpi=300, bbox_inches='tight')
plt.close()

print("Figure 4 saved to fig/methodology_framework.pdf and .png")