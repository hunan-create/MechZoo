"""R6/R8/R9 返修实验：单脚本产出修订版 Table 1 + 校准/覆盖风险 + Little 对照的区间估计
=========================================================================
改动要点（相对 levels_report.py / e2_transfer.py）：
  1) 真值标签来自实际机制（mechzoo.effective_mech）：real 表无 cat 列时 skip 退化为 MCAR 会被正确标为 mcar，
     不再按族名计入 MAR。
  2) 每格同时报平凡基线（L1=全预测非MCAR；L2=多数类），使 "above/below trivial" 可核对。
  3) 每格报 ECE（6 类后验）与 conf>=0.9 的 coverage/risk（选择性风险）。
  4) Little 对照在 base_fresh+novel+real-inj 三池合并上算 AUROC，并给配对 bootstrap 95% CI。
运行: python audit_cells.py [--ckpt e1_model_r6fix.pt] [--out audit_cells.json] [--boot 2000]
"""
import argparse, json, os, time
import numpy as np
import torch

from e0_mechanism_probe import MECHS, table_features, logreg_train, logreg_pred
from e1_amortized import DualAxisSetEncoder, build_pool, ece
from mechzoo import NOVEL, COARSE, make_tables, little_test
import realdata

MAR_FAMS = {"mar_obs", "mar_multi", "skip"}
MNAR_FAMS = {"mnar_self", "mnar_lat", "mix", "mnar_logit", "mnar_thresh"}
CMAP = np.array([COARSE[m] for m in MECHS])          # 6 基类 -> 0/1/2
IDX_MCAR = MECHS.index("mcar")
K6 = len(MECHS)                                       # 模型输出的 6 类标签空间

def _ck(name):
    """resolve checkpoint: cwd -> script dir -> ../checkpoints (repo layout)"""
    for p in (name, os.path.join(os.path.dirname(os.path.abspath(__file__)), name),
              os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "checkpoints", name)):
        if os.path.exists(p):
            return p
    return name



def softmax(x):
    e = np.exp(x - x.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def auroc(score, label):
    from scipy.stats import rankdata
    r = rankdata(score); pos = label.sum(); neg = (1 - label).sum()
    if pos == 0 or neg == 0:
        return float("nan")
    return float((r[label == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def binary_f1(y, p):
    tp = ((p == 1) & (y == 1)).sum(); fp = ((p == 1) & (y == 0)).sum(); fn = ((p == 0) & (y == 1)).sum()
    return float(2 * tp / max(2 * tp + fp + fn, 1)), float((p == y).mean())


def coarse_probs(P6):
    return np.stack([P6[:, CMAP == c].sum(1) for c in range(3)], 1)


def cell_report(P6, mech):
    mech = np.array(mech)
    y1 = (mech != "mcar").astype(int)
    mcar_p = P6[:, IDX_MCAR]
    pred1 = (mcar_p < 0.5).astype(int)
    f1_1, acc_1 = binary_f1(y1, pred1)
    sel = y1 == 1
    pc = coarse_probs(P6)
    C = pc.argmax(1)
    pred2 = np.array([1 if c == 2 else 0 for c in C[sel]])
    y2 = np.array([1 if m in MNAR_FAMS else 0 for m in mech[sel]])
    f1_2, acc_2 = binary_f1(y2, pred2)
    # 平凡基线
    p1 = float(y1.mean()); p2 = float(y2.mean()); maj = max(p2, 1 - p2)
    triv = {"L1_F1_allpos": round(2 * p1 / (1 + p1), 3),
            "L2_F1_majority": round(2 * maj / (1 + maj), 3),
            "L2_acc_majority": round(maj, 3),
            "p_nonmcar": round(p1, 3), "p_mnar": round(p2, 3)}
    return {"L1_f1": round(f1_1, 3), "L1_acc": round(acc_1, 3), "L2_f1": round(f1_2, 3),
            "L2_acc": round(acc_2, 3), "L2_n": int(sel.sum()), "trivial": triv}, (y1, y2, sel)


def coverage_risk(pc, y2, sel, conf_thr=0.9):
    """选择性风险: 只在 conf>=thr 的 L2 表上接受预测。"""
    pm = pc[sel][:, 2]
    pred2 = (pm >= 0.5).astype(int)
    conf = np.maximum(pm, 1 - pm)
    cov = conf >= conf_thr
    risk = float((pred2[cov] != y2[cov]).mean()) if cov.sum() else float("nan")
    return {"coverage": round(float(cov.mean()), 3), "risk": round(risk, 4), "n_cov": int(cov.sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="e1_model_iter2.pt")
    ap.add_argument("--out", default="audit_cells.json")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--shift", type=int, default=0, help="测试池种子偏移(三种子复抽)")
    ap.add_argument("--npz", default="audit_auroc_scores.npz", help="逐表分数落盘(三种子分文件名)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    t0 = time.time()
    dev = torch.device(args.device)
    args.ckpt = _ck(args.ckpt)
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    model = DualAxisSetEncoder(h=ck["h"]).to(dev)
    model.load_state_dict(ck["state_dict"]); model.eval()
    print(f"ckpt={args.ckpt} h={ck['h']}")

    # 探针: 与 levels_report 同协议 (seed 21, 250/class)
    tr_d, tr_r = make_tables(MECHS, 250, seed=21)
    probes = {n: logreg_train(np.array([table_features(*d)[i] for d in tr_d]),
                              np.array([r["y"] for r in tr_r]), len(MECHS))
              for n, i in [("B1", 0), ("B2", 1)]}

    pools = {}
    sfx = args.shift
    d, r = build_pool(150, seed=999 + sfx); pools["in_dist"] = (d, r)
    pools["base_fresh"] = make_tables(MECHS, 100, seed=777 + sfx)
    pools["novel"] = make_tables(NOVEL, 100, seed=778 + sfx)
    real, realn = [], []
    import os
    import realdata9
    real_tabs = realdata9.load_real_tables9() if os.environ.get("C2_REAL9") else realdata.load_real_tables()
    # real-inj: 全部源 × 6 族 × 8；joint-shift: 前 4 源 × 3 留出族 × 20（与论文 432 / 240 口径一致）
    for k, (name, cols, types) in enumerate(real_tabs):
        fn = (lambda rr, c=cols, t=types: ({a: v.copy() for a, v in c.items()}, dict(t), len(c), len(c)))
        real.append(make_tables(MECHS, 8, seed=500 + k + sfx, table_fn=fn))
        if k < 4:
            realn.append(make_tables(NOVEL, 20, seed=900 + k + sfx, table_fn=fn))
    pools["real_inj"] = ([x for p in real for x in p[0]], [x for p in real for x in p[1]])
    pools["joint_shift"] = ([x for p in realn for x in p[0]], [x for p in realn for x in p[1]])

    out = {"meta": {"ckpt": args.ckpt, "h": int(ck["h"]), "probe_seed": 21,
                    "probe_per_class": 250, "boot": args.boot, "shift": int(args.shift)}, "cells": {}}
    pooled = {"score_model": [], "score_little": [], "label": [], "nrow": [], "pool": []}
    for pname, (dicts, recs) in pools.items():
        mech = [r0["mech"] for r0 in recs]
        with torch.no_grad():
            P6 = np.concatenate([
                torch.softmax(model(torch.tensor(np.stack([r0["tok"] for r0 in recs[b0:b0 + 64]]),
                                                  dtype=torch.float32, device=dev))[0], -1).cpu().numpy()
                for b0 in range(0, len(recs), 64)])
        rep, (y1, y2, sel) = cell_report(P6, mech)
        pc = coarse_probs(P6)
        # 校准口径: ece6 只在真值属于模型 6 类标签空间时定义 (in_dist/base_fresh/real_inj);
        # 留出族与 joint-shift 的真值是 9 族索引 (y in 6..8), 6 类后验无法对应 —— 此时报 ece3
        # (MCAR/MAR/MNAR 粗三分, 与 L1/L2 同一层级), 并令 ece6 为 null 而不是被越界标签污染。
        y6 = np.array([r0["y"] for r0 in recs])
        yc = np.array([COARSE[r0["mech"]] for r0 in recs])
        rep["ece6"] = round(float(ece(P6, y6)), 4) if y6.max() < K6 else None
        rep["ece3"] = round(float(ece(pc, yc)), 4)
        rep["covrisk_090"] = coverage_risk(pc, y2, sel)
        for nm, fi in [("B1", 0), ("B2", 1)]:
            F = np.array([table_features(*t)[fi] for t in dicts])
            Pf = softmax(logreg_pred(probes[nm], F))
            f1b1, _ = binary_f1(y1, (Pf[:, IDX_MCAR] < 0.5).astype(int))
            f1b2, _ = binary_f1(y2, np.array([1 if c == 2 else 0
                                              for c in coarse_probs(Pf).argmax(1)[sel]]))
            rep[nm] = {"L1_f1": round(f1b1, 3), "L2_f1": round(f1b2, 3),
                       "ece6": round(float(ece(Pf, y6)), 4) if y6.max() < K6 else None,
                       "ece3": round(float(ece(coarse_probs(Pf), yc)), 4)}        # 合并池的材料 (Little 对照: 只并 base_fresh + novel + real_inj, 与论文口径一致)
        if pname in ("base_fresh", "novel", "real_inj"):
            for r0, p6, d6 in zip(recs, P6, dicts):
                pooled["score_model"].append(1 - p6[IDX_MCAR])
                pooled["score_little"].append(1 - little_test(*d6))
                pooled["label"].append(0 if r0["mech"] == "mcar" else 1)
                pooled["nrow"].append(len(r0["X"]))
                pooled["pool"].append(pname)
        out["cells"][pname] = {"n_tables": len(recs), **rep}
        print(f"[{pname}] n={len(recs)} model L1={rep['L1_f1']} L2={rep['L2_f1']} "
              f"ECE={rep['ece6']} covrisk={rep['covrisk_090']} triv={rep['trivial']}")

    lab = np.array(pooled["label"]); nrow = np.array(pooled["nrow"])
    sm = np.array(pooled["score_model"]); sl = np.array(pooled["score_little"])
    a_m, a_l = auroc(sm, lab), auroc(sl, lab)
    rng = np.random.default_rng(0)
    idx = np.arange(len(lab)); diffs = []
    for _ in range(args.boot):
        b = rng.choice(idx, len(idx), replace=True)
        d_ = auroc(sm[b], lab[b]) - auroc(sl[b], lab[b])
        if d_ == d_:
            diffs.append(d_)
    diffs = np.array(diffs)
    small = nrow < 600
    out["auroc"] = {"n": int(len(lab)), "n_nonmcar": int((lab == 1).sum()),
                    "n_mcar": int((lab == 0).sum()),
                    "model": round(a_m, 4), "little": round(a_l, 4),
                    "diff_pp": round(100 * (a_m - a_l), 2),
                    "diff_ci95": [round(100 * np.percentile(diffs, 2.5), 2),
                                  round(100 * np.percentile(diffs, 97.5), 2)],
                    "model_smalln": round(auroc(sm[small], lab[small]), 4),
                    "little_smalln": round(auroc(sl[small], lab[small]), 4),
                    "smalln_rows_lt": 600, "smalln_n": int(small.sum())}
    np.savez(args.npz, model=sm, little=sl, label=lab, nrow=nrow,
             pool=np.array(pooled["pool"], dtype=object))
    json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(json.dumps(out["auroc"], indent=1))
    print(f"已存 {args.out} 与 {args.npz}  耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
