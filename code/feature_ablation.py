"""⑤ §5.5 特征消融 — 值感知探针逐特征置零的 in-distribution macro-F1 下降
================================================================================
协议(与正文 §5.5 一致):
  - 探针: B2 值感知特征(28 维, e0.table_features[1])上的多项逻辑斯蒂,
    训练池 make_tables(MECHS, 250, seed=21) —— 与 levels_report/audit_cells 同协议;
  - 评测池: build_pool(150, seed=999)(in-distribution, 900 表);
  - 消融: 将单个特征列置 0 后重算 macro-F1, drop = full - ablated;
  - 池化残差形状通道(skew/kurt)不在 28 维表级特征中(属逐列 token, 见
    e1_amortized.column_tokens), 另以"扩展臂"复算: 在 28 维后追加逐列
    残差 skew/kurt 的池化统计 2 维, 置零该两维得到形状通道代价。
输出: feature_ablation.json(full / top_columns / channel_ablation)
运行: python feature_ablation.py            # CPU ~1min
对照: results/feature_ablation.json 为论文运行落盘(full=0.763,
      top-5 列 25/14/2/6/27, 代价 1.7/1.6/1.5/1.5/1.2pp; 形状通道 ~0.3pp)。
"""
import json
import numpy as np
import torch

from e0_mechanism_probe import MECHS, table_features, logreg_train, logreg_pred, macro_f1
from e1_amortized import DualAxisSetEncoder, build_pool, column_tokens
from mechzoo import make_tables
import os


def _ck(name):
    """resolve checkpoint path: cwd -> script dir -> ../checkpoints (repo layout)"""
    for p in (name, os.path.join(os.path.dirname(os.path.abspath(__file__)), name),
              os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "checkpoints", name)):
        if os.path.exists(p):
            return p
    return name


CKPT = _ck("e1_model_iter2.pt")


def pooled_shape(cols, types, mask, targets):
    """扩展臂: 逐目标列再拟合残差的 skew/kurt 池化(均值)→ 2 维。"""
    from scipy.stats import skew, kurtosis
    cont = [k for k, v in types.items() if v == "cont"]
    sk, ku = [], []
    for j in targets:
        obs = ~mask[j]
        others = [k for k in cont if k != j and (~mask[k] if k in mask else np.ones(len(obs), bool)).sum() > 30]
        if len(others) < 2:
            continue
        X = np.stack([ (cols[k] - np.nanmean(cols[k][obs])) / (np.nanstd(cols[k][obs]) + 1e-9) for k in others], 1)
        Xo = np.c_[X[obs], np.ones(int(obs.sum()))]
        y = (cols[j][obs] - np.nanmean(cols[j][obs])) / (np.nanstd(cols[j][obs]) + 1e-9)
        w, *_ = np.linalg.lstsq(Xo, y, rcond=None)
        r = y - Xo @ w
        if len(r) > 30:
            sk.append(skew(r)); ku.append(kurtosis(r))
    return [float(np.mean(sk)) if sk else 0.0, float(np.mean(ku)) if ku else 0.0]


def feats(d, extended=False):
    f = list(table_features(*d)[1])
    if extended:
        f += pooled_shape(*d)
    return np.array(f, float)


def main():
    # 与 levels_report/audit_cells 相同的探针协议
    tr_d, tr_r = make_tables(MECHS, 250, seed=21)
    y_tr = np.array([r["y"] for r in tr_r])

    te_d, te_r = build_pool(150, seed=999)
    y_te = np.array([r["y"] for r in te_r])

    out = {}
    for tag, ext in [("base28", False), ("with_shape", True)]:
        Xtr = np.array([feats(d, ext) for d in tr_d])
        Xte = np.array([feats(d, ext) for d in te_d])
        probe = logreg_train(Xtr, y_tr, len(MECHS))
        full = macro_f1(y_te, logreg_pred(probe, Xte).argmax(1), len(MECHS))
        mu = Xtr.mean(0)                    # 消融 = 把特征置回训练均值(标准化后为 0, 即无信息)
        drops = []
        for j in range(Xte.shape[1]):
            Xz = Xte.copy(); Xz[:, j] = mu[j]
            f1j = macro_f1(y_te, logreg_pred(probe, Xz).argmax(1), len(MECHS))
            drops.append((j, round(full - f1j, 4)))
        top = sorted(drops, key=lambda t: -t[1])[:10]
        shape_drop = None
        if ext:
            Xz = Xte.copy(); Xz[:, -2:] = mu[-2:]
            shape_drop = round(full - macro_f1(y_te, logreg_pred(probe, Xz).argmax(1), len(MECHS)), 4)
        out[tag] = {"full": round(full, 3), "top_columns": [[j, d] for j, d in top],
                    "shape_channel_drop": shape_drop}
        print(f"[{tag}] full macro-F1={full:.3f} top5={top[:5]} shape_drop={shape_drop}")

    res = {"channel_ablation": {"full": out["base28"]["full"]},
           "top_columns": out["base28"]["top_columns"],
           "shape_channel": {"full_with_shape": out["with_shape"]["full"],
                             "drop_when_zeroed": out["with_shape"]["shape_channel_drop"]}}
    json.dump(res, open("feature_ablation.json", "w", encoding="utf-8"), indent=2)
    print("已存 feature_ablation.json")


if __name__ == "__main__":
    main()
