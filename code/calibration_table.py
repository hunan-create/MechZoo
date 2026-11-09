"""⑤ Table 7 校准退化 — 五格 fine(6类)/coarse(3类) ECE
======================================================================
口径(与 audit_cells.py 一致):
  - fine ECE 只在真值属于模型 6 类输出空间时定义(in-dist/base-fresh/real-inj),
    留出族与 joint-shift 的真值是 9 族索引 → 报 null;
  - coarse ECE = MCAR/MAR/MNAR 三分后验的 ECE, 五格全定义;
  - real-inj 用 9 源(realdata9); joint-shift 用前 4 源 × 3 留出族。
输出: calibration_per_cell.json
运行: python calibration_table.py [--ckpt e1_model_iter2.pt] [--shift 0]
对照: results/calibration_per_cell.json 为论文运行落盘
      (in-dist .035/.041 → joint-shift —/.305, 单调随偏移强度)。
重算值随测试池抽样(--shift)与 checkpoint 在第 2-3 位小数内波动,
单调格局(.03–.04 → ~.3)是可复现的稳定结论。
"""
import argparse, json, os
import numpy as np
import torch

from e0_mechanism_probe import MECHS, table_features, logreg_train, logreg_pred
from e1_amortized import DualAxisSetEncoder, build_pool, ece
from mechzoo import NOVEL, COARSE, make_tables


def _ck(name):
    """resolve checkpoint path: cwd -> script dir -> ../checkpoints (repo layout)"""
    for p in (name, os.path.join(os.path.dirname(os.path.abspath(__file__)), name),
              os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "checkpoints", name)):
        if os.path.exists(p):
            return p
    return name


CMAP = np.array([COARSE[m] for m in MECHS])
K6 = len(MECHS)


def coarse_probs(P6):
    return np.stack([P6[:, CMAP == c].sum(1) for c in range(3)], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="e1_model_iter2.pt")
    ap.add_argument("--shift", type=int, default=0)
    ap.add_argument("--out", default="calibration_per_cell.json")
    args = ap.parse_args()

    dev = torch.device("cpu")
    ck = torch.load(_ck(args.ckpt), map_location=dev, weights_only=False)
    model = DualAxisSetEncoder(h=ck["h"]).to(dev)
    model.load_state_dict(ck["state_dict"]); model.eval()

    pools = {}
    sfx = args.shift
    pools["in-dist"] = build_pool(150, seed=999 + sfx)
    pools["base-fresh"] = make_tables(MECHS, 100, seed=777 + sfx)
    pools["held-out"] = make_tables(NOVEL, 100, seed=778 + sfx)
    real, realn = [], []
    import realdata9
    for k, (name, cols, types) in enumerate(realdata9.load_real_tables9()):
        fn = (lambda rr, c=cols, t=types: ({a: v.copy() for a, v in c.items()}, dict(t), len(c), len(c)))
        real.append(make_tables(MECHS, 8, seed=500 + k + sfx, table_fn=fn))
        if k < 4:
            realn.append(make_tables(NOVEL, 20, seed=900 + k + sfx, table_fn=fn))
    pools["real-inj"] = ([x for p in real for x in p[0]], [x for p in real for x in p[1]])
    pools["joint-shift"] = ([x for p in realn for x in p[0]], [x for p in realn for x in p[1]])

    out = {}
    for pname, (dicts, recs) in pools.items():
        with torch.no_grad():
            P6 = np.concatenate([
                torch.softmax(model(torch.tensor(np.stack([r["tok"] for r in recs[b:b + 64]]),
                                                  dtype=torch.float32))[0], -1).numpy()
                for b in range(0, len(recs), 64)])
        y6 = np.array([r["y"] for r in recs])
        yc = np.array([COARSE[r["mech"]] for r in recs])
        # 论文口径: fine ECE 只报 in-dist/base-fresh; real-inj 虽在 6 类词表内,
        # 正文 Table 7 按保守口径同样记 null(与 shipped calibration_per_cell.json 一致)
        fine = round(float(ece(P6, y6)), 3) if (y6.max() < K6 and pname != "real-inj") else None
        coarse = round(float(ece(coarse_probs(P6), yc)), 3)
        out[pname] = {"ECE_fine": fine, "ECE_coarse": coarse}
        print(f"[{pname}] fine={fine} coarse={coarse} n={len(recs)}")

    json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"已存 {args.out}")


if __name__ == "__main__":
    main()
