"""A1 实验汇报重构 — 按理论层级(终版命题1)重排主表
=====================================================
Level-1 = MCAR 检出(可检验层, 命题1(i)反向); Level-2 = 非MCAR 内 MAR族 vs MNAR族
(类内辨识层——无约束下不可检验(引理1.1), 受限生成元类内可辨识)。
三系统: v0.5(e1_model_iter2.pt) / 探针B1(仅掩码) / 探针B2(值感知); 与 e2 同种子同池。
运行: python levels_report.py
"""
import json
import numpy as np
import torch
from e0_mechanism_probe import MECHS, table_features, logreg_train
from e1_amortized import DualAxisSetEncoder, build_pool
from mechzoo import make_tables, COARSE, NOVEL, DEGRADED
import realdata

MAR_FAMS = {"mar_obs", "mar_multi", "skip"}
MNAR_FAMS = {"mnar_self", "mnar_lat", "mix", "mnar_logit", "mnar_thresh"}

def sys_probs(recs, dicts, model, dev, probes):
    """三系统的 6 类后验: model softmax / B1 / B2 探针."""
    outs = {"model": [], "B1": [], "B2": []}
    with torch.no_grad():
        for b0 in range(0, len(recs), 64):
            toks = torch.tensor(np.stack([r["tok"] for r in recs[b0:b0 + 64]]),
                                dtype=torch.float32, device=dev)
            outs["model"].append(torch.softmax(model(toks)[0], -1).cpu().numpy())
    for name, fidx in [("B1", 0), ("B2", 1)]:
        F = np.array([table_features(*d)[fidx] for d in dicts])
        logits = logreg_pred_softmax(probes[name], F)
        outs[name] = [logits]   # 列表! concatenate 对裸数组会沿行拼接
    return {k: np.concatenate(v) for k, v in outs.items()}

def logreg_pred_softmax(probe, X):
    from e0_mechanism_probe import logreg_pred
    x = np.exp(logreg_pred(probe, X))
    return x / x.sum(1, keepdims=True)

def binary_metrics(y, p):
    tp = ((p == 1) & (y == 1)).sum(); fp = ((p == 1) & (y == 0)).sum(); fn = ((p == 0) & (y == 1)).sum()
    return float(2 * tp / max(2 * tp + fp + fn, 1)), float((p == y).mean())

import os

def _ck(name):
    """resolve checkpoint path: cwd -> script dir -> ../checkpoints (repo layout)"""
    for p in (name, os.path.join(os.path.dirname(os.path.abspath(__file__)), name),
              os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "checkpoints", name)):
        if os.path.exists(p):
            return p
    return name


def main(shift=0, tag=""):
    import os
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = _ck(os.environ.get("C2_CKPT", "e1_model_iter2.pt"))
    ck = torch.load(ckpt, map_location=dev, weights_only=False)
    model = DualAxisSetEncoder(h=ck["h"]).to(dev); model.load_state_dict(ck["state_dict"]); model.eval()
    tr_dicts, tr_recs = make_tables(MECHS, 250, seed=21)
    probes = {n: logreg_train(np.array([table_features(*d)[i] for d in tr_dicts]),
                              np.array([r["y"] for r in tr_recs]), len(MECHS))
              for n, i in [("B1", 0), ("B2", 1)]}

    pools = {}
    d1, r1 = build_pool(150, seed=999); pools["in_dist"] = (d1, r1)
    pools["base_fresh"] = make_tables(MECHS, 100, seed=777+shift)
    pools["novel"] = make_tables(NOVEL, 100, seed=778+shift)
    real, realn = [], []
    import os
    import realdata9
    real_tabs = realdata9.load_real_tables9() if os.environ.get("C2_REAL9") else realdata.load_real_tables()
    # real-inj: 全部源 × 6 族 × 8；joint-shift: 前 4 源 × 3 留出族 × 20（与论文 432 / 240 口径一致）
    for k, (name, cols, types) in enumerate(real_tabs):
        fn = (lambda r, c=cols, t=types: ({a: v.copy() for a, v in c.items()}, dict(t), len(c), len(c)))
        real.append(make_tables(MECHS, 8, seed=500 + k + shift, table_fn=fn))
        if k < 4:
            realn.append(make_tables(NOVEL, 20, seed=900 + k + shift, table_fn=fn))
    pools["real"] = ([d for p in real for d in p[0]], [r for p in real for r in p[1]])
    pools["real_novel"] = ([d for p in realn for d in p[0]], [r for p in realn for r in p[1]])

    Cmap = np.array([COARSE[MECHS[i]] for i in range(len(MECHS))])
    report = {}
    meta = {"ckpt": ckpt, "shift": int(shift), "tag": tag,
            "probe_train_seed": 21, "probe_per_class": 250,
            "degraded": {f"{a}->{b}": v for (a, b), v in DEGRADED.items()}}
    for pname, (dicts, recs) in pools.items():
        mech = [r["mech"] if "mech" in r else MECHS[r["y"]] for r in recs]
        prob_sys = sys_probs(recs, dicts, model, dev, probes)
        y1_all = np.array([0 if m == "mcar" else 1 for m in mech])
        y2_all = np.array([1 if m in MNAR_FAMS else 0 for m in mech])
        row = {}
        for sysname, P in prob_sys.items():
            mcar_p = P[:, MECHS.index("mcar")]
            lvl1_pred = (mcar_p < 0.5).astype(int)
            f1_1, acc_1 = binary_metrics(y1_all, lvl1_pred)
            sel = y1_all == 1
            C = Cmap[P.argmax(1)]
            lvl2_pred = np.array([1 if c == 2 else 0 for c in C[sel]])
            f1_2, _ = binary_metrics(y2_all[sel], lvl2_pred)
            row[sysname] = {"L1_mcarF1": round(f1_1, 3), "L1_acc": round(acc_1, 3),
                            "L2_marMnarF1": round(f1_2, 3), "L2_n": int(sel.sum())}
        # 平凡基线: L1 = 全预测非 MCAR; L2 = 全预测多数类(按多数类为正类算 F1)
        p1 = float(y1_all.mean())
        p2 = float(y2_all[sel].mean())
        maj2 = max(p2, 1 - p2)
        triv = {"n_tables": int(len(recs)), "p_nonmcar": round(p1, 3),
                "p_mnar_within_nonmcar": round(p2, 3),
                "trivial_L1_F1": round(2 * p1 / (1 + p1), 3),
                "trivial_L2_F1": round(2 * maj2 / (1 + maj2), 3),
                "trivial_L2_acc": round(maj2, 3)}
        report[pname] = {"systems": row, "trivial": triv, "meta": meta}
        print(f"[{pname}] n={len(recs)} trivial={triv}")
        print(json.dumps(row, indent=1))
    json.dump(report, open(f"levels_report{tag}.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("已存 levels_report%s.json  degraded=%s" % (tag, meta["degraded"]))

if __name__ == "__main__":
    import sys
    main(int(sys.argv[1]) if len(sys.argv)>1 else 0, sys.argv[2] if len(sys.argv)>2 else "")
