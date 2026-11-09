"""Fig: per-family recall and precision (data: main_icde.tex Table IV).
Style: bar_grouped_hatch (grouped bars, value labels, y-grid)."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams.update({
    'text.usetex': False,
    'font.family': 'serif',
    'font.serif': ['STIXGeneral', 'DejaVu Serif', 'Times New Roman'],
    'mathtext.fontset': 'stix',
    'axes.unicode_minus': False,
})

COLORS = ['#A8C8E8', '#1B3D6E']
HATCHES = ['', '']
LABELS = ['Recall', 'Precision']

FAMILIES = ['MCAR', 'MAR-obs', 'Self-mask', 'Latent-MNAR', 'Skip', 'Mixture']
DATA = {
    'Recall':    [70.7, 56.0, 72.7, 84.0, 98.7, 76.7],
    'Precision': [75.7, 71.2, 71.2, 68.1, 99.3, 74.2],
}

fig, ax = plt.subplots(figsize=(8.6, 4.0))
fig.subplots_adjust(left=0.08, right=0.98, bottom=0.16, top=0.86)

n_groups = len(FAMILIES)
n_methods = len(LABELS)
x = np.arange(n_groups, dtype=float)
bar_w = 0.62 / n_methods
for i, (label, color, hatch) in enumerate(zip(LABELS, COLORS, HATCHES)):
    vals = DATA[label]
    offset = (i - n_methods / 2 + 0.5) * bar_w
    bars = ax.bar(x + offset, vals, width=bar_w, color=color, hatch=hatch,
                  edgecolor='white', linewidth=0.8, zorder=2, label=label)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.8, f'{v:.1f}',
                ha='center', va='bottom', fontsize=8.0,
                fontweight='normal', color='black', zorder=3)

ax.set_xticks(x)
ax.set_xticklabels(FAMILIES, fontsize=10.0)
ax.set_xlabel('Family', fontsize=11.2)
ax.set_ylabel(r'F1 (\%)', fontsize=11.2)
ax.set_ylim(0, 112)
ax.set_xlim(-0.5, 5.5)
ax.set_title('Per-family recall and precision (in-distribution)', fontsize=13.0, pad=5)
ax.yaxis.grid(True, color='#EBEBEB', linewidth=0.7, linestyle='--', zorder=0)
ax.set_axisbelow(True)
for side, spine in ax.spines.items():
    if side in ('top', 'right'):
        spine.set_visible(False)
    else:
        spine.set_linewidth(0.9)
        spine.set_color('#333333')
        spine.set_zorder(4)
ax.tick_params(length=0, labelsize=9.5)
handles = [mpatches.Patch(facecolor=c, hatch=h, edgecolor='white',
                          linewidth=0.8, label=l)
           for l, c, h in zip(LABELS, COLORS, HATCHES)]
ax.legend(handles=handles, fontsize=9.2, loc='upper left',
          bbox_to_anchor=(0.012, 0.985), framealpha=1.0, facecolor='white',
          edgecolor='#C8C8C8', fancybox=False, borderpad=0.28,
          labelspacing=0.26, handlelength=1.7, handletextpad=0.45,
          borderaxespad=0.28)

plt.savefig('../figures/fig_per_family.png', dpi=300, facecolor='white')
plt.close()
print('saved: fig_per_family.png')
