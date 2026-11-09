"""⑤ 9 源真实表逐源 L2 — 正文 Table 4 real-inj 行与 per-source 表的数据源
==============================================================================
协议(与 expanded_real_results.json 的产生口径一致):
  - 9 张公开真实表(realdata9, 离线缓存) × 6 基础族 × 8 表, seed 500+k;
  - 标签 = 实际机制(effective_mech): 无 cat 列时 skip 退化为 MCAR;
  - L1 = MCAR 检出 F1(全 432 表); L2 = 非 MCAR 内 MAR 族 vs MNAR 族二元 F1;
  - MissDx = e1_model_iter2.pt(正文 Table 4 行的 checkpoint);
    探针 = B1/B2 逻辑斯蒂(seed 21, 250/类, 与 levels_report 同协议);
  - 逐源 48 表分别报 L2 F1, 并报 9 源合并(pooled)。
输出: expanded_real_results.json(per_source + pooled + L1)
运行: python real_per_source.py          # CPU ~2min
对照: pooled L2 重算 = 论文 Table 4 real-inj 行的 (.690, .721), 逐位一致;
      results/expanded_real_results.json 另保存论文 Table 9 的逐源分解
      (其逐源口径属当次未存档的脚本, 重算逐源值与之有出入, 详见 README)。
"""
import json
import os
import numpy as np
import torch

from e0_mechanism_probe import MECHS, table_features, logreg_train, logreg_pred
from e1_amortized import DualAxisSetEncoder
from mechzoo import make_tables, COARSE
import realdata9

MNAR_FAMS = {"mnar_self", "mnar_lat", "mix", "mnar_logit", "mnar_thresh"}
CMAP = np.array([COARSE[m] for m in MECHS])
IDX_MCAR = MECHS.index("mcar")


def _ck(name):
    """resolve checkpoint path: cwd -> script dir -> ../checkpoints (repo layout)"""
    for p in (name, os.path.join(os.path.dirname(os.path.abspath(__file__)), name),
              os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "checkpoints", name)):
        if os.path.exists(p):
            return p
    return name


def softmax(x):
    e = np.exp(x - x.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def binary_f1(y, p):
    tp = ((p == 1) & (y == 1)).sum(); fp = ((p == 1) & (y == 0)).sum(); fn = ((p == 0) & (y == 1)).sum()
    return float(2 * tp / max(2 * tp + fp + fn, 1))


def l1_l2(P, mech, model_rule=False):
    """P: (n,6) 后验或探针概率; mech: 实际机制标签列表 → (L1_f1, L2_f1)。
    模型 L2 用粗三分 argmax(三类概率和再取 argmax, 复现论文 pooled .690);
    探针 L2 用 6 类 argmax→粗三分(levels_report 口径, 复现论文 pooled .721)。"""
    mech = np.array(mech)
    y1 = (mech != "mcar").astype(int)
    pred1 = (P[:, IDX_MCAR] < 0.5).astype(int)
    f1_1 = binary_f1(y1, pred1)
    sel = y1 == 1
    pc = np.stack([P[:, CMAP == c].sum(1) for c in range(3)], 1)
    C = pc.argmax(1) if model_rule else CMAP[P.argmax(1)]
    pred2 = np.array([1 if c == 2 else 0 for c in C[sel]])
    y2 = np.array([1 if m in MNAR_FAMS else 0 for m in mech[sel]])
    return round(f1_1, 3), round(binary_f1(y2, pred2), 3)


def main():
    dev = torch.device("cpu")
    ck = torch.load(_ck("e1_model_iter2.pt"), map_location=dev, weights_only=False)
    model = DualAxisSetEncoder(h=ck["h"]).to(dev)
    model.load_state_dict(ck["state_dict"]); model.eval()

    tr_d, tr_r = make_tables(MECHS, 250, seed=21)
    probes = {n: logreg_train(np.array([table_features(*d)[i] for d in tr_d]),
                              np.array([r["y"] for r in tr_r]), len(MECHS))
              for n, i in [("B1", 0), ("B2", 1)]}

    per_source, pooled = {}, {"model": [], "B1": [], "B2": [], "mech": []}
    for k, (name, cols, types) in enumerate(realdata9.load_real_tables9()):
        fn = (lambda rr, c=cols, t=types: ({a: v.copy() for a, v in c.items()}, dict(t), len(c), len(c)))
        dicts, recs = make_tables(MECHS, 8, seed=500 + k, table_fn=fn)
        mech = [r["mech"] for r in recs]
        with torch.no_grad():
            P6 = np.concatenate([
                torch.softmax(model(torch.tensor(np.stack([r["tok"] for r in recs[b:b + 64]]),
                                                  dtype=torch.float32))[0], -1).numpy()
                for b in range(0, len(recs), 64)])
        _, l2m = l1_l2(P6, mech, model_rule=True)
        _, l2p = l1_l2(softmax(logreg_pred(probes["B2"],
                     np.array([table_features(*d)[1] for d in dicts]))), mech)
        per_source[name] = {"MissDx": l2m, "probe": l2p, "n": len(recs)}
        pooled["model"].append(P6)
        for n, fi in [("B1", 0), ("B2", 1)]:
            F = np.array([table_features(*d)[fi] for d in dicts])
            pooled[n].append(softmax(logreg_pred(probes[n], F)))
        pooled["mech"] += mech
        print(f"[{name}] MissDx={l2m} probe={l2p} n={len(recs)}")

    Pm = np.concatenate(pooled["model"]); P1 = np.concatenate(pooled["B1"]); P2 = np.concatenate(pooled["B2"])
    l1m, l2m = l1_l2(Pm, pooled["mech"], model_rule=True)
    l11, l21 = l1_l2(P1, pooled["mech"])
    l12, l22 = l1_l2(P2, pooled["mech"])
    mech = np.array(pooled["mech"]); y2 = np.array([1 if m in MNAR_FAMS else 0 for m in mech[mech != "mcar"]])
    res = {"per_source": per_source,
           "pooled": {"MissDx": l2m, "probe": l22,
                      "prevalence": round(float(y2.mean()), 3),
                      "n": int(len(mech)), "n_sources": 9,
                      "L1": {"MissDx": l1m, "mask_probe": l11, "value_probe": l12},
                      "L2": {"MissDx": l2m, "mask_probe": l21, "value_probe": l22}}}
    json.dump(res, open("expanded_real_results.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("pooled:", json.dumps(res["pooled"], indent=1))
    print("已存 expanded_real_results.json")


if __name__ == "__main__":
    main()
