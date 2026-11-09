"""R9-Fatal-4 修复：TabPFN 零样本基线落盘（同款 B2 特征 / 同池 / 多上下文种子）
====================================================================
任务与正文 §5.1 的句子一致：L2（非 MCAR 内的 MAR 族 vs MNAR 族），
上下文 = MechZoo 训练池(seed 21) 中平衡抽样的 1000 张非 MCAR 表；
预测 = 各 cell 的非 MCAR 表。报 F1 / accuracy，3 个上下文种子。
运行: python tabpfn_baseline.py [--out tabpfn_baseline.json]
"""
import argparse, json, time
import numpy as np

from e0_mechanism_probe import MECHS, table_features
from e1_amortized import build_pool
from mechzoo import NOVEL, make_tables
import realdata

MAR_FAMS = {"mar_obs", "mar_multi", "skip"}
MNAR_FAMS = {"mnar_self", "mnar_lat", "mix", "mnar_logit", "mnar_thresh"}


def feats(dicts):
    return np.array([table_features(*d)[1] for d in dicts], dtype=np.float32)


def labels(recs):
    return np.array([1 if r["mech"] in MNAR_FAMS else 0 for r in recs])


def f1(y, p):
    tp = ((p == 1) & (y == 1)).sum(); fp = ((p == 1) & (y == 0)).sum(); fn = ((p == 0) & (y == 1)).sum()
    return float(2 * tp / max(2 * tp + fp + fn, 1)), float((p == y).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tabpfn_baseline.json")
    ap.add_argument("--ctx", type=int, default=1000)
    ap.add_argument("--seeds", default="0,1,2")
    args = ap.parse_args()
    from tabpfn import TabPFNClassifier
    t0 = time.time()

    tr_d, tr_r = make_tables(MECHS, 250, seed=21)
    keep = [i for i, r in enumerate(tr_r) if r["mech"] != "mcar"]
    Xtr, ytr = feats([tr_d[i] for i in keep]), labels([tr_r[i] for i in keep])

    pools = {}
    d, r = build_pool(150, seed=999); pools["in_dist"] = (d, r)
    pools["base_fresh"] = make_tables(MECHS, 100, seed=777)
    pools["novel"] = make_tables(NOVEL, 100, seed=778)
    real, realn = [], []
    for k, (name, cols, types) in enumerate(realdata.load_real_tables()):
        fn = (lambda rr, c=cols, t=types: ({a: v.copy() for a, v in c.items()}, dict(t), len(c), len(c)))
        real.append(make_tables(MECHS, 8, seed=500 + k, table_fn=fn))
        realn.append(make_tables(NOVEL, 20, seed=900 + k, table_fn=fn))
    pools["real_inj"] = ([x for p in real for x in p[0]], [x for p in real for x in p[1]])
    pools["joint_shift"] = ([x for p in realn for x in p[0]], [x for p in realn for x in p[1]])

    res = {"meta": {"context": args.ctx, "seeds": args.seeds, "train_pool": "seed21, 250/class",
                    "features": "B2 value-aware", "task": "L2 MAR vs MNAR (non-MCAR only)"},
           "cells": {}}
    for seed in [int(s) for s in args.seeds.split(",")]:
        rng = np.random.default_rng(seed)
        pos = np.where(ytr == 1)[0]; neg = np.where(ytr == 0)[0]
        n_pos = min(len(pos), args.ctx // 2); n_neg = min(len(neg), args.ctx - n_pos)
        idx = np.concatenate([rng.choice(pos, n_pos, replace=False),
                              rng.choice(neg, n_neg, replace=False)])
        clf = TabPFNClassifier(device="cuda", n_estimators=8, random_state=seed)
        clf.fit(Xtr[idx], ytr[idx])
        for pname, (dicts, recs) in pools.items():
            sel = np.array([i for i, r0 in enumerate(recs) if r0["mech"] != "mcar"])
            if len(sel) == 0:
                continue
            X = feats([dicts[i] for i in sel]); y = labels([recs[i] for i in sel])
            pr = clf.predict_proba(X)[:, 1]
            f1v, acc = f1(y, (pr >= 0.5).astype(int))
            res["cells"].setdefault(pname, {})[f"seed{seed}"] = {"L2_f1": round(f1v, 3),
                                                                 "acc": round(acc, 3), "n": int(len(sel))}
            print(f"[seed{seed}] {pname:12s} L2 F1={f1v:.3f} acc={acc:.3f} n={len(sel)}")
    for pname, d_ in res["cells"].items():
        vals = [v["L2_f1"] for v in d_.values()]
        res["cells"][pname]["mean_f1"] = round(float(np.mean(vals)), 3)
        res["cells"][pname]["sd_f1"] = round(float(np.std(vals)), 3)
    json.dump(res, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("已存", args.out, f"耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
