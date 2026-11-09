"""Fig: calibration degradation, coarse ECE per cell (data: main_icde.tex Table VII).
Style: single-series bar with value labels; severity gradient from light to dark red."""
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'text.usetex': False,
    'font.family': 'serif',
    'font.serif': ['STIXGeneral', 'DejaVu Serif', 'Times New Roman'],
    'mathtext.fontset': 'stix',
    'axes.unicode_minus': False,
})

cells = ['in-dist', 'base-fresh', 'held-out', 'real-inj', 'joint-shift']
ece = [4.1, 3.3, 6.7, 22.3, 30.5]   # coarse ECE, %
colors = ['#F2D8D8', '#EAC9C9', '#E8A090', '#D9534F', '#C62828']

fig, ax = plt.subplots(figsize=(7.2, 3.9))
fig.subplots_adjust(left=0.09, right=0.97, bottom=0.15, top=0.86)

x = np.arange(len(cells))
bars = ax.bar(x, ece, width=0.55, color=colors, edgecolor='white',
              linewidth=0.8, zorder=2)
for i, (bar, v) in enumerate(zip(bars, ece)):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.6, f'{v:.1f}',
            ha='center', va='bottom', fontsize=9.2,
            fontweight='bold' if cells[i] == 'joint-shift' else 'normal',
            color='#8B0000' if cells[i] == 'joint-shift' else 'black',
            zorder=3)

ax.set_xticks(x)
ax.set_xticklabels(cells, fontsize=10.5)
ax.set_xlabel('Cell', fontsize=11.2)
ax.set_ylabel(r'Coarse ECE (\%)', fontsize=11.2)
ax.set_ylim(0, 38)
ax.set_xlim(-0.5, 4.5)
ax.set_title('Calibration degradation across cells', fontsize=13.0, pad=5)
ax.yaxis.grid(True, color='#EBEBEB', linewidth=0.7, linestyle='--', zorder=0)
ax.set_axisbelow(True)
for side, spine in ax.spines.items():
    if side in ('top', 'right'):
        spine.set_visible(False)
    else:
        spine.set_linewidth(0.9)
        spine.set_color('#333333')
        spine.set_zorder(4)
ax.tick_params(length=0, labelsize=10.0)

plt.savefig('../figures/fig_calib_ece.png', dpi=300, facecolor='white')
plt.close()
print('saved: fig_calib_ece.png')
