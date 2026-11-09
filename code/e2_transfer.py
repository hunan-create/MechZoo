"""C2 E2 迁移门控 — 留出机制族 + 真实表注入 + Little 检验对照
================================================================
门控预注册(研究日志 §4 E2, 阈值操作化见日志 §10 待确认):
  (a) 机制族迁移: novel 池(mar_multi/mnar_logit/mnar_thresh) 粗三分 macro-F1
      (模型) − (探针B1 仅掩码) ≥ 0.05   —— K1 先验支撑风险的直接检验
  (b) 参数: 基础族新鲜表 MAE(有效b̂−b) ≤ 0.60(b∈[0.9,3.0] 量纲)
      且 novel mnar_logit 强度 Spearman ≥ 0.40
  (c) Little 对照: MCAR-vs-not AUROC(模型) − AUROC(Little) ≥ 0.05(全池, 附小n分层)
依赖: e1_model.pt(先跑 python e1_amortized.py; --smoke 模式可现场训 tiny)
运行:
  python e2_transfer.py --smoke --device cpu    # 冒烟(~2min)
  python e2_transfer.py                         # 完整(GPU 推理, 池生成为主)
  python e2_transfer.py --skip-real             # 跳过真实表轴
"""
import argparse, json, sys, time
import os
from pathlib import Path
import numpy as np
import torch
from scipy.stats import rankdata

from e0_mechanism_probe import MECHS, table_features, logreg_train, logreg_pred
from e1_amortized import DualAxisSetEncoder
from mechzoo import NOVEL, ALL_FAMILIES, COARSE, COARSE_NAMES, make_tables, little_test

COARSE_GROUP = {c: [m for m in MECHS if COARSE[m] == c] for c in range(3)}

# ---------------------------------------------------------------- 工具
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

def coarse_macro_f1(y, p):
    f1s = []
    for c in range(3):
        tp = ((p == c) & (y == c)).sum(); fp = ((p == c) & (y != c)).sum(); fn = ((p != c) & (y == c)).sum()
        f1s.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(np.mean(f1s))

def auroc(score, label):
    r = rankdata(score)
    pos, neg = label.sum(), (1 - label).sum()
    if pos == 0 or neg == 0: return float("nan")
    return float((r[label == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))

def model_probs(model, recs, dev, bs=64):
    out = []
    model.eval()
    with torch.no_grad():
        for b0 in range(0, len(recs), bs):
            toks = torch.tensor(np.stack([r["tok"] for r in recs[b0:b0 + bs]]),
                                dtype=torch.float32, device=dev)
            out.append(torch.softmax(model(toks)[0], -1).cpu().numpy())
    return np.concatenate(out)

def probe_coarse_pred(probe, dicts, fidx):
    f = np.array([table_features(*d)[fidx] for d in dicts])   # 0=B1 仅掩码, 1=B2 值感知
    pf = softmax(logreg_pred(probe, f))
    pc = np.stack([pf[:, [MECHS.index(m) for m in COARSE_GROUP[c]]].sum(1) for c in range(3)], 1)
    return pc.argmax(1)

def train_tiny(dev):
    """--smoke 且无 ckpt 时的现场小模型(仅验证流程, 数字无意义)。"""
    _, recs = make_tables(MECHS, 40, seed=11)
    model = DualAxisSetEncoder(h=32).to(dev)
    model.set_norm([r["tok"] for r in recs])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    rng = np.random.default_rng(3)
    for step in range(30):
        idx = rng.choice(len(recs), 16, replace=False)
        toks = torch.tensor(np.stack([recs[i]["tok"] for i in idx]), dtype=torch.float32, device=dev)
        yb = torch.tensor([recs[i]["y"] for i in idx], device=dev)
        loss = torch.nn.functional.cross_entropy(model(toks)[0], yb)
        opt.zero_grad(); loss.backward(); opt.step()
    return model

# ---------------------------------------------------------------- 主流程
def run(args):
    dev = torch.device(args.device)
    t0 = time.time()

    print("[1/6] 加载摊销模型 ...")
    args.ckpt = _ck(args.ckpt)
    if Path(args.ckpt).exists():
        ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
        model = DualAxisSetEncoder(h=ck["h"]).to(dev)
        model.load_state_dict(ck["state_dict"])
        print(f"    已加载 {args.ckpt} (h={ck['h']})")
    elif args.smoke:
        print("    [smoke] 无 ckpt, 现场训 tiny 模型(数字无意义)")
        model = train_tiny(dev)
    else:
        sys.exit(f"缺少 {args.ckpt}: 先运行 python e1_amortized.py")

    print(f"[2/6] 探针训练(基础 6 族 × {args.probe_per_class}) ...")
    tr_dicts, tr_recs = make_tables(MECHS, args.probe_per_class, seed=21)
    F1 = np.array([table_features(*d)[0] for d in tr_dicts])   # B1 仅掩码
    F2 = np.array([table_features(*d)[1] for d in tr_dicts])   # B2 值感知
    yf = np.array([r["y"] for r in tr_recs])
    probe_b1, probe_b2 = logreg_train(F1, yf, len(MECHS)), logreg_train(F2, yf, len(MECHS))

    print(f"[3/6] 测试池: 基础族(新种子) + novel {NOVEL} ..." + ("" if args.skip_real else " + 真实表注入"))
    base_d, base_r = make_tables(MECHS, args.test_per_class, seed=777)
    nov_d, nov_r = make_tables(NOVEL, args.test_per_class, seed=778)
    real_d, real_r, real_names = [], [], []
    if not args.skip_real:
        import realdata
        if args.smoke:
            realdata.SKIP_OPENML = True      # 冒烟只走内置离线表
        for k, (name, cols, types) in enumerate(realdata.load_real_tables()):
            # 深拷贝: mnar_lat 会原地改写目标列, 防跨重复泄漏
            fn = (lambda r, cols=cols, types=types: ({c: v.copy() for c, v in cols.items()},
                                                    dict(types), len(cols), len(cols)))
            d, r = make_tables(MECHS, args.real_per_table, seed=500 + k, table_fn=fn)
            real_d += d; real_r += r; real_names += [name] * len(r)
        print(f"    真实表轴: {len(set(real_names))} 张源表 × 6 族 × {args.real_per_table} = {len(real_r)} 表")
    pools = {"base_fresh": (base_d, base_r), "novel": (nov_d, nov_r)}
    if real_r:
        pools["real_injected"] = (real_d, real_r)

    print("[4/6] 逐池评测(粗三分) ...")
    results, P_all, littles, is_mcar, ns = {}, [], [], [], []
    for pname, (dts, rcs) in pools.items():
        Pm = model_probs(model, rcs, dev)
        yc = np.array([COARSE[r["mech"]] for r in rcs])
        # 粗后验: 对 6 基类按 COARSE 求和(与探针一致)
        Cmap = np.array([COARSE[m] for m in MECHS])
        Pm_c = np.stack([Pm[:, Cmap == c].sum(1) for c in range(3)], 1)
        pred_m = Pm_c.argmax(1)
        p1 = probe_coarse_pred(probe_b1, dts, 0)
        p2 = probe_coarse_pred(probe_b2, dts, 1)
        row = {"model": coarse_macro_f1(yc, pred_m), "B1": coarse_macro_f1(yc, p1),
               "B2": coarse_macro_f1(yc, p2),
               "model_acc": float((pred_m == yc).mean())}
        ent = -(Pm * np.log(Pm + 1e-12)).sum(1)
        row["entropy_mean"] = float(ent.mean())
        results[pname] = row
        print(f"    {pname:13s} 粗macroF1: model={row['model']:.3f} B1={row['B1']:.3f} "
              f"B2={row['B2']:.3f} | 熵={row['entropy_mean']:.2f}")
        # 收集 Little 对照材料(基础+novel+real 全部)
        for dd, rr, pp in zip(dts, rcs, Pm):
            littles.append(little_test(*dd))
            is_mcar.append(rr["mech"] == "mcar")
            P_all.append(pp)
            ns.append(len(rr["X"]))

    print("[5/6] 参数估计 + Little 对照 ...")
    # 参数: 模型回归头 ×3 vs 真值(基础池 s 有定义的表)
    def param_err(rcs):
        preds, trues = [], []
        with torch.no_grad():
            for r in rcs:
                if np.isnan(r["s"]): continue
                toks = torch.tensor(r["tok"][None], dtype=torch.float32, device=dev)
                preds.append(model(toks)[1].item() * 3)
                trues.append(r["s"] * 3)
        return np.array(preds), np.array(trues)
    pb, tb = param_err(base_r)
    mae_base = float(np.abs(pb - tb).mean())
    pn, tn = param_err([r for r in nov_r if r["mech"] == "mnar_logit"])  # 符号携带族,与基础自掩蔽同构
    rho_novel = float(np.corrcoef(rankdata(pn), rankdata(tn))[0, 1]) if len(pn) > 10 else float("nan")
    # Little: MCAR-vs-not
    P_all, littles, is_mcar, ns = map(np.array, (P_all, littles, is_mcar, ns))
    Cmap = np.array([COARSE[m] for m in MECHS])
    score_model = 1 - P_all[:, MECHS.index("mcar")]
    score_little = 1 - littles
    lab = is_mcar.astype(int)  # 1 = MCAR(检验"检出非MCAR"能力 → label=非MCAR 为正)
    lab_not = 1 - lab
    au_model = auroc(score_model, lab_not)
    au_little = auroc(score_little, lab_not)
    small = ns < 600
    au_model_s = auroc(score_model[small], lab_not[small]) if small.sum() > 20 else float("nan")
    au_little_s = auroc(score_little[small], lab_not[small]) if small.sum() > 20 else float("nan")

    print("[6/6] 门控判读 ...")
    g_a = results["novel"]["model"] - results["novel"]["B1"] >= 0.05
    g_b = (mae_base <= 0.60) and (rho_novel == rho_novel and rho_novel >= 0.40)
    g_c = au_model - au_little >= 0.05
    res = {
        "pools": {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in results.items()},
        "param_mae_base": round(mae_base, 3),
        "param_spearman_novel": round(rho_novel, 3) if rho_novel == rho_novel else None,
        "auroc_model": round(au_model, 3), "auroc_little": round(au_little, 3),
        "auroc_model_smallN": round(au_model_s, 3) if au_model_s == au_model_s else None,
        "auroc_little_smallN": round(au_little_s, 3) if au_little_s == au_little_s else None,
        "gate_a": bool(g_a), "gate_b": bool(g_b), "gate_c": bool(g_c),
        "args": vars(args),
    }
    print(json.dumps(res, indent=2, ensure_ascii=False))
    print("\n=== E2 门控判读 ===")
    print(f"(a) novel 粗F1: model−B1 = {results['novel']['model']-results['novel']['B1']:+.3f} (需≥+0.05) -> {'PASS' if g_a else 'FAIL'}")
    print(f"(b) 参数: 基础MAE={mae_base:.3f}(≤0.60) novel ρ={rho_novel:.3f}(≥0.40) -> {'PASS' if g_b else 'FAIL'}")
    print(f"(c) AUROC: model−Little = {au_model-au_little:+.3f} (需≥+0.05) [小n层 {au_model_s if au_model_s==au_model_s else float('nan'):.3f} vs {au_little_s if au_little_s==au_little_s else float('nan'):.3f}] -> {'PASS' if g_c else 'FAIL'}")
    print(f"综合: {'E2 PASS' if (g_a and g_b and g_c) else 'E2 FAIL'}")
    json.dump(res, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"结果已存 {args.out}, 总耗时 {time.time()-t0:.0f}s")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--ckpt", default="e1_model.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--probe_per_class", type=int, default=250)
    ap.add_argument("--test_per_class", type=int, default=100)
    ap.add_argument("--real_per_table", type=int, default=8)
    ap.add_argument("--skip-real", dest="skip_real", action="store_true")
    ap.add_argument("--out", default="e2_results.json")
    args = ap.parse_args()
    if args.smoke:
        args.probe_per_class, args.test_per_class = 15, 8
        args.real_per_table = 2
        args.out = "e2_smoke_results.json"
    run(args)
