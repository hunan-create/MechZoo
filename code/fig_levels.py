"""Fig: L1/L2 level-structured results (data: main_icde.tex Table II).
Style: bar_grouped_hatch (grouped bars, main method hatched, value labels, y-grid)."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams.update({
    'text.usetex': False,
    'font.family': 'serif',
    'font.serif': ['STIXGeneral', 'DejaVu Serif', 'Times New Roman'],
    'mathtext.fontset': 'stix',
    'axes.unicode_minus': False,
    'hatch.color': 'white',
    'hatch.linewidth': 1.4,
})

COLORS = ['#D3D3D3', '#A9A9A9', '#8C8C8C', '#D00000']
HATCHES = ['', '', '', '//']
LABELS = ['Value-probe', 'Mask-probe', 'Trivial', 'MissDx']
BEST_METHOD = 'MissDx'

# Panel (a): L1 (MCAR detection F1, %); L1 defined for 3 cells
CELLS_A = ['in-dist', 'base-fresh', 'real-inj']
DATA_A = {
    'Value-probe': [92.6, 92.8, 87.0],
    'Mask-probe':  [90.9, 90.9, 87.0],
    'Trivial':     [90.6, 90.7, 80.0],
    'MissDx':      [94.9, 94.7, 73.0],
}
# Panel (b): L2 (MAR-family vs MNAR-family F1, %); 5 cells
CELLS_B = ['in-dist', 'base-fresh', 'held-out', 'real-inj', 'joint-shift']
DATA_B = {
    'Value-probe': [86.2, 84.2, 81.0, 72.1, 79.3],
    'Mask-probe':  [71.0, 69.7, 66.5, 63.0, 70.0],
    'Trivial':     [75.3, 75.2, 80.0, 85.7, 80.0],
    'MissDx':      [85.8, 87.0, 77.6, 69.0, 63.6],
}


def draw_panel(ax, cells, data, title, xlim, ylim, num_fmt='{:.1f}'):
    n_groups = len(cells)
    n_methods = len(LABELS)
    x = np.arange(n_groups, dtype=float)
    bar_w = 0.78 / n_methods
    for i, (label, color, hatch) in enumerate(zip(LABELS, COLORS, HATCHES)):
        vals = data[label]
        offset = (i - n_methods / 2 + 0.5) * bar_w
        bars = ax.bar(x + offset, vals, width=bar_w, color=color, hatch=hatch,
                      edgecolor='white', linewidth=0.8, zorder=2, label=label)
        is_best = (label == BEST_METHOD)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.8,
                    num_fmt.format(v), ha='center', va='bottom',
                    fontsize=8.2,
                    fontweight='bold' if is_best else 'normal',
                    color='#8B0000' if is_best else 'black', zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(cells, fontsize=10.8)
    ax.set_xlabel('Cell', fontsize=11.2)
    ax.set_ylabel(r'F1 (\%)', fontsize=11.2)
    ax.set_ylim(*ylim)
    ax.set_xlim(*xlim)
    ax.set_title(title, fontsize=13.2, pad=5)
    ax.yaxis.grid(True, color='#EBEBEB', linewidth=0.7, linestyle='--', zorder=0)
    ax.set_axisbelow(True)
    for side, spine in ax.spines.items():
        if side in ('top', 'right'):
            spine.set_visible(False)
        else:
            spine.set_linewidth(0.9)
            spine.set_color('#333333')
            spine.set_zorder(4)
    ax.tick_params(length=0, labelsize=10.2)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 4.35))
fig.subplots_adjust(left=0.05, right=0.985, bottom=0.15, top=0.78, wspace=0.18)

draw_panel(ax1, CELLS_A, DATA_A, '(a) L1: MCAR detection',
           xlim=(-0.56, 2.56), ylim=(0, 100))
draw_panel(ax2, CELLS_B, DATA_B, '(b) L2: MAR vs MNAR',
           xlim=(-0.56, 4.56), ylim=(0, 100))

# 共享图例置于图上方，避免覆盖任何数据
handles = [mpatches.Patch(facecolor=c, hatch=h, edgecolor='white',
                          linewidth=0.8, label=l)
           for l, c, h in zip(LABELS, COLORS, HATCHES)]
leg = fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.995),
                 ncol=4, fontsize=9.4, framealpha=1.0, facecolor='white',
                 edgecolor='#C8C8C8', fancybox=False, borderpad=0.30,
                 labelspacing=0.26, handlelength=1.7, handletextpad=0.45,
                 columnspacing=1.0)
for text in leg.get_texts():
    if text.get_text() == BEST_METHOD:
        text.set_fontweight('bold')

plt.savefig('../figures/fig_levels.png', dpi=300, facecolor='white')
plt.close()
print('saved: fig_levels.png')
