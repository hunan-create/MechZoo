"""Fig: per-source L2 coarse-F1, MissDx vs Value-probe (data: main_icde.tex Table IX).
Style: bar_paired_delta (paired bars + dashed reference + delta labels)."""
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'text.usetex': False,
    'font.family': 'serif',
    'font.serif': ['STIXGeneral', 'DejaVu Serif', 'Times New Roman'],
    'mathtext.fontset': 'stix',
    'axes.unicode_minus': False,
})

COLOR_BASELINE = '#A8C8E8'
COLOR_METHOD = '#1B3D6E'
COLOR_DELTA = '#CC2200'
COLOR_DELTA_NEG = '#777777'

groups = ['breast', 'diabetes', 'page-bl.', 'Magic', 'bal-scale',
          'qsar', 'wine', 'syn-ctrl', 'ILPD']
baseline = [51.3, 63.2, 33.3, 77.7, 37.5, 47.2, 9.5, 29.5, 65.6]   # Value-probe
method = [63.1, 41.3, 37.0, 72.4, 47.8, 45.5, 48.3, 44.9, 55.4]    # MissDx
deltas = [11.8, -21.9, 3.7, -5.3, 10.3, -1.7, 38.8, 15.4, -10.2]  # pp

fig, ax = plt.subplots(figsize=(14, 4.8))
fig.subplots_adjust(left=0.05, right=0.99, bottom=0.15, top=0.84)

BAR_W = 0.30
GAP = 0.02
n = len(groups)
x = np.arange(n)

bars_b = ax.bar(x - (BAR_W + GAP) / 2, baseline, width=BAR_W,
                color=COLOR_BASELINE, zorder=3, label='Value-probe')
bars_m = ax.bar(x + (BAR_W + GAP) / 2, method, width=BAR_W,
                color=COLOR_METHOD, zorder=3, label='MissDx')

for i, (bl, me, d) in enumerate(zip(baseline, method, deltas)):
    ax.plot([x[i] - BAR_W, x[i] + BAR_W + GAP / 2], [bl, bl],
            color='black', lw=0.9, ls='--', zorder=4)
    if d > 0:
        ax.annotate('', xy=(x[i] + (BAR_W + GAP) / 2, me - 0.4),
                    xytext=(x[i] + (BAR_W + GAP) / 2, bl + 0.4),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.2),
                    zorder=5)
        label, color = f'+{d:.1f}', COLOR_DELTA
    else:
        label, color = f'{d:.1f}', COLOR_DELTA_NEG
    ax.text(x[i] + (BAR_W + GAP) / 2, max(bl, me) + 1.4, label,
            color=color, ha='center', va='bottom',
            fontsize=9.0, fontweight='bold' if d > 0 else 'normal')

ax.set_xticks(x)
ax.set_xticklabels(groups, fontsize=10.0, fontweight='bold')
ax.set_ylabel(r'L2 coarse-F1 (\%)', fontsize=11.0, fontweight='bold')
ax.set_ylim(0, 90)
ax.yaxis.set_major_locator(plt.MultipleLocator(10))
for spine in ax.spines.values():
    spine.set_linewidth(1.4)
    spine.set_color('black')
ax.tick_params(length=0)
ax.set_axisbelow(True)

ax.text(0.005, 0.97, 'MissDx vs Value-probe (real-injected, per source)',
        transform=ax.transAxes, fontsize=12, fontweight='bold',
        va='top', ha='left', color='#003F6C', fontfamily='serif')
ax.legend(fontsize=10.0, loc='upper right', framealpha=1.0,
          facecolor='white', edgecolor='#C8C8C8')

plt.savefig('../figures/fig_per_source.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print('saved: fig_per_source.png')
