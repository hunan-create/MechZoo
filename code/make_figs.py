"""复现论文两张图（R8/R9: 原稿无绘图脚本）
输入: theory_align.json (B3_esn / B4_lambda)
输出: figs/fig_esn.pdf、figs/fig_lambda.pdf
运行: python make_figs.py
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os
_src = "theory_align.json" if os.path.exists("theory_align.json") else os.path.join("..", "results", "theory_align.json")
D = json.load(open(_src, encoding="utf-8"))
OUT = os.path.join("..", "figures")
os.makedirs(OUT, exist_ok=True)


def sn_skew(delta):
    d2 = 2 * delta ** 2 / np.pi
    return ((4 - np.pi) / 2) * (delta * np.sqrt(2 / np.pi)) ** 3 / (1 - d2) ** 1.5


def theory_curve(tau2, b, sigma=None):
    sig = np.sqrt(1 - tau2) if sigma is None else sigma
    alpha1 = b / np.sqrt(1 + b ** 2 * tau2)
    alpha = alpha1 * sig
    return sn_skew(alpha / np.sqrt(1 + alpha ** 2))


def fig_esn():
    g = D["B3_esn"]["grid"]
    taus = sorted({r["tau2"] for r in g})
    bs = np.linspace(0.2, 3.4, 200)
    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(taus)))
    for c, t in zip(colors, taus):
        ax.plot(bs, theory_curve(t, bs), color=c, lw=1.6, label=rf"$\tau_f^2={t:g}$")
        emp = [-r["emp_oracle"] for r in g if r["tau2"] == t]
        ax.scatter([r["b"] for r in g if r["tau2"] == t], emp, color=c, s=22, zorder=3)
    ax.set_xlabel(r"selection slope $b$")
    ax.set_ylabel("pooled residual skewness")
    ax.set_ylim(0, 0.78)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_esn.pdf")
    plt.close(fig)


def fig_lambda():
    rows = D["B4_lambda"]
    lam = [r["lam"] for r in rows]
    mv = [r["mv_signal"] for r in rows]
    ent = [r["entropy"] for r in rows]
    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    ax.plot(lam, mv, "o-", color="#1f4e79", lw=1.6, ms=4, label="mv-channel signal")
    ax.set_xlabel(r"latent loading $\lambda$")
    ax.set_ylabel("mask--value signal", color="#1f4e79")
    ax.tick_params(axis="y", labelcolor="#1f4e79")
    ax2 = ax.twinx()
    ax2.plot(lam, ent, "s--", color="#a33", lw=1.4, ms=4, label="posterior entropy")
    ax2.set_ylabel("posterior entropy (nats)", color="#a33")
    ax2.tick_params(axis="y", labelcolor="#a33")
    ax.set_title("latent-MNAR sweep", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_lambda.pdf")
    plt.close(fig)


if __name__ == "__main__":
    fig_esn()
    fig_lambda()
    print("已存 figs/fig_esn.pdf, figs/fig_lambda.pdf")
