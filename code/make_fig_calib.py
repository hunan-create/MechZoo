"""⑤ Fig.4(calib) — in-distribution 可靠性图 + 逐族后验熵
================================================================
左: 10 分箱可靠性图(in-dist 900 表, fine ECE 标于标题);
右: 按真值族分组的后验熵分布(边界类 latent-MNAR 的中位熵最高)。
输出: ../figures/fig_calib.pdf
运行: python make_fig_calib.py [--ckpt e1_model_iter2.pt]
注: figures/fig_calib.pdf 为论文成稿图; 本脚本按同一数据口径重画。
"""
import argparse, os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from e0_mechanism_probe import MECHS
from e1_amortized import DualAxisSetEncoder, build_pool, ece


def _ck(name):
    """resolve checkpoint path: cwd -> script dir -> ../checkpoints (repo layout)"""
    for p in (name, os.path.join(os.path.dirname(os.path.abspath(__file__)), name),
              os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "checkpoints", name)):
        if os.path.exists(p):
            return p
    return name


FAM_LABEL = {"mcar": "MCAR", "mar_obs": "MAR-obs", "mnar_self": "Self-mask",
             "mnar_lat": "Latent-MNAR", "skip": "Skip", "mix": "Mixture"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="e1_model_iter2.pt")
    args = ap.parse_args()

    dev = torch.device("cpu")
    ck = torch.load(_ck(args.ckpt), map_location=dev, weights_only=False)
    model = DualAxisSetEncoder(h=ck["h"]).to(dev)
    model.load_state_dict(ck["state_dict"]); model.eval()

    dicts, recs = build_pool(150, seed=999)
    with torch.no_grad():
        P = np.concatenate([
            torch.softmax(model(torch.tensor(np.stack([r["tok"] for r in recs[b:b + 64]]),
                                              dtype=torch.float32))[0], -1).numpy()
            for b in range(0, len(recs), 64)])
    y = np.array([r["y"] for r in recs])
    fam = [r["family"] for r in recs]
    conf, pred = P.max(1), P.argmax(1)
    e = ece(P, y)

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4))

    # 左: 可靠性图
    ax = axes[0]
    bins = np.linspace(0, 1, 11)
    ctr, acc = [], []
    for b in range(10):
        sel = (conf > bins[b]) & (conf <= bins[b + 1])
        if sel.sum() < 5:
            continue
        ctr.append(float(conf[sel].mean())); acc.append(float((pred[sel] == y[sel]).mean()))
    ax.plot([0, 1], [0, 1], "--", color="#999", lw=1)
    ax.plot(ctr, acc, "o-", color="#1f4e79", lw=1.6, ms=5)
    ax.set_xlabel("posterior confidence"); ax.set_ylabel("empirical accuracy")
    ax.set_title(f"reliability (in-dist), ECE = {e:.3f}", fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.grid(alpha=.3)

    # 右: 逐族后验熵
    ax = axes[1]
    H = -(P * np.log(P + 1e-12)).sum(1)
    names = [m for m in MECHS]
    data = [H[np.array(fam) == m] for m in names]
    bp = ax.boxplot(data, tick_labels=[FAM_LABEL[m] for m in names], patch_artist=True,
                    medianprops=dict(color="#a33", lw=1.6))
    for i, box in enumerate(bp["boxes"]):
        box.set(facecolor="#F6C9C9" if names[i] == "mnar_lat" else "#DCE8F4",
                edgecolor="#666", lw=.8)
    ax.set_ylabel("posterior entropy (nats)")
    ax.set_title("posterior entropy by true family (red = boundary class)", fontsize=9)
    ax.grid(alpha=.3, axis="y")
    plt.setp(ax.get_xticklabels(), rotation=28, ha="right", fontsize=7.5)

    fig.tight_layout()
    out = os.path.join("..", "figures", "fig_calib.pdf")
    fig.savefig(out)
    print(f"已存 {out}  ECE={e:.4f}")


if __name__ == "__main__":
    main()
