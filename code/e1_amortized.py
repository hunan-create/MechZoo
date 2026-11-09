"""C2 E1 摊销机制推断 — v0.5 诊断统计层 + 列集合注意力(MissDx v0.5)
================================================================
门控 v2 预注册(研究日志 2026-10-29 §8 修订版):
  (a) macro-F1(模型) - macro-F1(探针B2, 同测试集) >= 0.05 且 ECE <= 0.10
  (b) 可判层(mcar/skip/mar_obs/mnar_self/mix) macro-F1 差 >= 0.05
      且边界层校准: 后验熵(mnar_lat) - 后验熵(可判层判对表) >= 0.20 nats
  (c) MAR-vs-MNAR(全MNAR合并) 二元 F1 差 >= 0.05
v0.5 变更(依据当日优化诊断, 详见日志 §11):
  - 原始 cell→注意力端到端通路在流式 CE 下不收敛(5 轮修复无效);
    诚实对照: cell-MLP 均值特征线性探针 0.258 vs 手工统计 0.738 → pivot。
  - 列 token = E0 B2 特征逐列化(含未损坏列, 定位信息), 全观察统计;
    集合注意力(无PE, 列置换不变)聚合 → 线性后验/参数读出;
    标准化常数(训练池统计)随 checkpoint 分发。
运行:
  python e1_amortized.py --smoke --device cpu   # 冒烟
  python e1_amortized.py                        # 完整(GPU)
"""
import argparse, json, time
import numpy as np
import torch
import torch.nn as nn

from e0_mechanism_probe import (gen_table, apply_mechanism, MECHS, table_features,
                                logreg_train, logreg_pred, effective_mech)

K = len(MECHS)
DECODABLE = [m for m in MECHS if m != "mnar_lat"]
D_MAX, F_COL = 12, 14

# ---------------------------------------------------------------- 数据管线
def effective_strength(truth):
    vals = [p for mech, p in truth.values() if mech in ("mar_obs", "mnar_self", "mnar_lat", "mix")]
    return float(np.mean(vals)) if vals else None

def table_to_matrix(cols, types, mask):
    """矩阵化(z-score 仅观察统计)——供 e3 插补器使用。"""
    names = list(cols.keys())
    n, d = len(cols[names[0]]), len(names)
    X = np.zeros((n, d), np.float32); M = np.zeros((n, d), np.float32); T = np.zeros(d, np.int64)
    for j, name in enumerate(names):
        t, x, m = types[name], cols[name], mask[name].astype(np.float32)
        obs = m < 0.5
        if t == "cat":
            v = x / max(x.max(), 1)
        else:
            mu, sd = x[obs].mean(), x[obs].std() + 1e-9
            v = np.clip((x - mu) / sd, -5, 5)
        v = v.astype(np.float32); v[~obs] = 0.0
        X[:, j], M[:, j], T[j] = v, m, {"cont": 0, "bin": 1, "cat": 2}[t]
    return X, M, T

def column_tokens(cols, types, mask):
    """逐列诊断统计(E0 B2 逐列化, 含未损坏列) → (D_MAX, F_COL); 全观察数据。"""
    names = list(cols.keys())
    n = len(cols[names[0]])
    cont = [k for k, v in types.items() if v == "cont"]
    Z = {}
    for k in cont:
        obs = ~mask[k]
        mu, sd = cols[k][obs].mean(), cols[k][obs].std() + 1e-9
        Z[k] = np.clip((cols[k] - mu) / sd, -5, 5)
    mvar = [k for k in names if mask[k].std() > 1e-9]
    toks = np.zeros((D_MAX, F_COL), np.float32)
    for j, name in enumerate(names):
        m = mask[name].astype(float); obs = m < 0.5
        f = np.zeros(F_COL, np.float32)
        f[0] = m.mean()
        m_ok = m.std() > 1e-9
        phis = [abs(np.corrcoef(m, mask[k].astype(float))[0, 1]) for k in mvar
                if k != name and m_ok and mask[k].astype(float).std() > 1e-9]
        f[1], f[2] = (max(phis), float(np.mean(phis))) if phis else (0., 0.)
        cors = []
        for k in cont:
            if k == name or not m_ok: continue
            ok = obs & (~mask[k])
            if ok.sum() < 30: continue
            a, b = m[ok], Z[k][ok]
            if a.std() > 1e-9 and b.std() > 1e-9:
                cors.append(abs(np.corrcoef(a, b)[0, 1]))
        f[3], f[4] = (max(cors), float(np.mean(cors))) if cors else (0., 0.)
        if types[name] == "cont":
            others = [k for k in cont if k != name and (~mask[k]).sum() > 30]
            if len(others) >= 2:
                Xd = np.stack([Z[k] for k in others], 1)
                Xo = np.c_[Xd[obs], np.ones(int(obs.sum()))]
                w, *_ = np.linalg.lstsq(Xo, Z[name][obs], rcond=None)
                resid = Z[name][obs] - Xo @ w
                f[5], f[6], f[7] = resid.mean(), abs(resid.mean()), resid.std()
        vs = []
        for k, t in types.items():
            if t != "cat" or k == name: continue
            ok = ~mask[k]
            if ok.sum() < 30: continue
            tab = np.zeros((2, int(cols[k].max()) + 1))
            for mi in (0, 1):
                sel = ok & (m.astype(bool) == bool(mi))
                uv, uc = np.unique(cols[k][sel], return_counts=True)
                for v, c in zip(uv, uc): tab[mi, int(v)] = c
            row, col = tab.sum(1), tab.sum(0)
            if (row < 5).any() or (col < 5).any() or tab.sum() < 20: continue
            exp = np.outer(row, col) / tab.sum()
            chi2 = ((tab - exp) ** 2 / (exp + 1e-12)).sum()
            vs.append(chi2 / (tab.sum() * (min(tab.shape) - 1)))
        f[8] = max(vs) if vs else 0.
        f[9] = np.log10(n) / 4.
        f[10], f[11], f[12] = (float(types[name] == "cont"), float(types[name] == "bin"),
                               float(types[name] == "cat"))
        f[13] = 1.0
        toks[j] = f
    return np.nan_to_num(toks, nan=0.0, posinf=0.0, neginf=0.0)

def build_pool(n_per_class, seed):
    rng = np.random.default_rng(seed)
    dicts, recs = [], []
    for ci, mech in enumerate(MECHS):
        made = 0
        while made < n_per_class:
            cols, types, n, d = gen_table(rng)
            mask, truth = apply_mechanism(mech, cols, types, rng)
            rates = [mask[j].mean() for j in truth]
            if min(rates) < 0.03 or max(rates) > 0.6:
                continue
            eff = effective_mech(mech, truth)
            X, M, T = table_to_matrix(cols, types, mask)
            s = effective_strength(truth)
            dicts.append((cols, types, mask, list(truth.keys())))
            recs.append({"X": X, "M": M, "T": T, "y": MECHS.index(eff), "mech": eff,
                         "family": mech, "tok": column_tokens(cols, types, mask),
                         "s": (s / 3.0) if s is not None else np.nan})
            made += 1
    return dicts, recs

def build_pool_paired(n_tables_per_class, seed):
    """PCC 配对池: 每表 4 次机制干预 [m,θ1],[m,θ2](同族异参)+两个异族。
    逐腐化深拷贝(mnar_lat 原地改写); 返回 (dicts, recs, groups), group=4连续下标。"""
    rng = np.random.default_rng(seed)
    dicts, recs, groups = [], [], []
    for ci, mech in enumerate(MECHS):
        made = 0
        while made < n_tables_per_class:
            cols, types, n, d = gen_table(rng)
            fams = [mech, mech] + list(rng.choice([x for x in MECHS if x != mech], 2, replace=False))
            group, ok = [], True
            for fam in fams:
                cols_c = {k: v.copy() for k, v in cols.items()}
                mask, truth = apply_mechanism(fam, cols_c, types, rng)
                rates = [mask[j].mean() for j in truth]
                if min(rates) < 0.03 or max(rates) > 0.6:
                    ok = False; break
                s = effective_strength(truth)
                eff = effective_mech(fam, truth)
                group.append(len(recs))
                dicts.append((cols_c, types, mask, list(truth.keys())))
                recs.append({"y": MECHS.index(eff), "mech": eff, "family": fam,
                             "tok": column_tokens(cols_c, types, mask),
                             "s": (s / 3.0) if s is not None else np.nan})
            if ok:
                groups.append(group)
                made += 1
    return dicts, recs, groups

def js_div(p, q, eps=1e-8):
    m = 0.5 * (p + q)
    kl = lambda a, b: (a * (a.add(eps).log() - b.add(eps).log())).sum(-1)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)

# ---------------------------------------------------------------- 模型
class DualAxisSetEncoder(nn.Module):
    """v0.5: 逐列诊断 token → 集合注意力(无PE, 列置换不变) → 线性读出。
    标准化常数 mu/sd 为训练池统计(buffer, 随 ckpt 分发)。"""
    def __init__(self, h=96, layers=2, heads=4, ff=192, p_drop=0.1):
        super().__init__()
        self.cell = nn.Sequential(nn.Linear(F_COL, h), nn.SiLU(), nn.Linear(h, h))
        self.attn = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(h, heads, ff, p_drop, batch_first=True, norm_first=True), layers)
        self.drop = nn.Dropout(p_drop)
        self.cls, self.par = nn.Linear(h, K), nn.Linear(h, 1)
        self.register_buffer("mu", torch.zeros(F_COL))
        self.register_buffer("sd", torch.ones(F_COL))

    def set_norm(self, toks):
        t = torch.tensor(np.stack(toks))
        valid = t[..., 13] > 0.5
        x = t[valid]
        self.mu.copy_(x.mean(0)); self.sd.copy_(x.std(0) + 1e-6)

    def forward(self, toks):                        # (B, D_MAX, F_COL)
        pad = toks[..., 13] < 0.5                   # (B,D)
        x = (toks - self.mu) / self.sd
        e = self.cell(x)
        h_ = self.attn(e, src_key_padding_mask=pad)
        w = (~pad).float().unsqueeze(-1)
        z = self.drop((h_ * w).sum(1) / w.sum(1))
        return self.cls(z), self.par(z).squeeze(-1)

# ---------------------------------------------------------------- 指标
def macro_f1(y, p, classes=None):
    cs = range(K) if classes is None else classes
    f1s = []
    for c in cs:
        tp = ((p == c) & (y == c)).sum(); fp = ((p == c) & (y != c)).sum(); fn = ((p != c) & (y == c)).sum()
        f1s.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(np.mean(f1s))

def ece(probs, y, bins=15):
    conf, pred = probs.max(1), probs.argmax(1)
    e = 0.0
    for b in range(bins):
        sel = (conf > b / bins) & (conf <= (b + 1) / bins)
        if sel.sum() == 0: continue
        e += sel.mean() * abs((pred[sel] == y[sel]).mean() - conf[sel].mean())
    return float(e)

def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])

# ---------------------------------------------------------------- 主流程
def run(args):
    dev = torch.device(args.device)
    t0 = time.time()
    print(f"[1/5] 生成训练池 (seed 11) ..." + ("[PCC 配对]" if args.paired else ""))
    if args.paired:
        dicts, recs, groups = build_pool_paired(args.tables_per_class, seed=11)
        print(f"      {len(groups)} 表×4 腐化 = {len(recs)} 项")
    else:
        dicts, recs = build_pool(args.pool_per_class, seed=11)
        print(f"      {len(recs)} 表")
    print(f"[2/5] 新鲜留出测试集 {args.test_per_class}×{K} (seed 999) ...")
    test_dicts, test_recs = build_pool(args.test_per_class, seed=999)
    y_test = np.array([r["y"] for r in test_recs])
    print(f"      {len(test_recs)} 表")

    print(f"[3/5] 探针对照(B2 特征, 类平衡抽样 每类≤{args.probe_n // K}) ...")
    y_all = np.array([r["y"] for r in recs])
    per = max(args.probe_n // K, 60)
    probe_idx = np.concatenate([np.where(y_all == c)[0][:per] for c in range(K)])
    np.random.default_rng(7).shuffle(probe_idx)
    FX = [table_features(*dicts[i])[1] for i in probe_idx]
    Fy = y_all[probe_idx]
    probe = logreg_train(np.array(FX), Fy, K)
    TX = np.array([table_features(*d)[1] for d in test_dicts])
    probe_pred = logreg_pred(probe, TX).argmax(1)
    print(f"      探针留出 macro-F1 = {macro_f1(y_test, probe_pred):.3f}")

    print("[4/5] 训练摊销模型(v0.5 统计层) ...")
    model = DualAxisSetEncoder(h=args.h).to(dev)
    model.set_norm([r["tok"] for r in recs])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    step = 0
    if args.paired:
        G = max(args.bs // 4, 1)
        order_g = np.arange(len(groups))
        steps = args.epochs * (len(groups) // G)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(steps, 1), eta_min=1e-4)
        for ep in range(args.epochs):
            np.random.default_rng(100 + ep).shuffle(order_g)
            for g0 in range(0, len(order_g) - G + 1, G):
                idx = np.concatenate([groups[g] for g in order_g[g0:g0 + G]])
                toks = torch.tensor(np.stack([recs[i]["tok"] for i in idx]), dtype=torch.float32, device=dev)
                yb = torch.tensor([recs[i]["y"] for i in idx], device=dev)
                sb = torch.tensor([0.0 if np.isnan(recs[i]["s"]) else recs[i]["s"] for i in idx], device=dev)
                pm = torch.tensor([0.0 if np.isnan(recs[i]["s"]) else 1.0 for i in idx], device=dev)
                logits, par = model(toks)
                P = torch.softmax(logits, -1)
                loss = nn.functional.cross_entropy(logits, yb)
                if pm.sum() > 0:
                    loss = loss + 0.3 * ((par - sb) ** 2 * pm).sum() / pm.sum()
                coupling, n_c = 0.0, 0
                for b_ in range(G):
                    for a_, b2 in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]:
                        i_, j_ = b_ * 4 + a_, b_ * 4 + b2
                        js = js_div(P[i_], P[j_])
                        if yb[i_] == yb[j_]:                      # 同族异参: attract+参数差分
                            coupling = coupling + 0.1 * js
                            if pm[i_] > 0 and pm[j_] > 0:
                                coupling = coupling + 0.5 * ((par[i_] - par[j_]) - (sb[i_] - sb[j_])) ** 2
                        else:                                    # 异族: repel
                            coupling = coupling + torch.relu(0.25 - js)
                        n_c += 1
                loss = loss + 0.5 * coupling / max(n_c, 1)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); step += 1
                if step % 200 == 0:
                    acc_b = (logits.argmax(1) == yb).float().mean().item()
                    print(f"      ep{ep} step{step} loss {loss.item():.3f} batch_acc {acc_b:.2f} ({time.time()-t0:.0f}s)")
    else:
        steps = args.epochs * (len(recs) // args.bs)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(steps, 1), eta_min=1e-4)
        order = np.arange(len(recs))
        for ep in range(args.epochs):
            np.random.default_rng(100 + ep).shuffle(order)
            for b0 in range(0, len(order) - args.bs + 1, args.bs):
                idx = order[b0:b0 + args.bs]
                toks = torch.tensor(np.stack([recs[i]["tok"] for i in idx]), dtype=torch.float32, device=dev)
                yb = torch.tensor([recs[i]["y"] for i in idx], device=dev)
                sb = torch.tensor([0.0 if np.isnan(recs[i]["s"]) else recs[i]["s"] for i in idx], device=dev)
                pm = torch.tensor([0.0 if np.isnan(recs[i]["s"]) else 1.0 for i in idx], device=dev)
                logits, par = model(toks)
                loss = nn.functional.cross_entropy(logits, yb)
                if pm.sum() > 0:
                    loss = loss + 0.3 * ((par - sb) ** 2 * pm).sum() / pm.sum()
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); step += 1
                if step % 200 == 0:
                    acc_b = (logits.argmax(1) == yb).float().mean().item()
                    print(f"      ep{ep} step{step} loss {loss.item():.3f} batch_acc {acc_b:.2f} ({time.time()-t0:.0f}s)")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"      {n_params/1e6:.2f}M 参数, {step} 步, {time.time()-t0:.0f}s")

    print("[5/5] 评测 + 门控 v2 判读 ...")
    model.eval()
    probs, pars = [], []
    with torch.no_grad():
        for b0 in range(0, len(test_recs), 64):
            chunk = test_recs[b0:b0 + 64]
            toks = torch.tensor(np.stack([r["tok"] for r in chunk]), dtype=torch.float32, device=dev)
            logits, par = model(toks)
            probs.append(torch.softmax(logits, -1).cpu().numpy())
            pars.append(par.cpu().numpy())
    P = np.concatenate(probs); pred = P.argmax(1); s_pred = np.concatenate(pars)
    ent = -(P * np.log(P + 1e-12)).sum(1)

    dec_cls = [MECHS.index(m) for m in DECODABLE]
    sel_dec = np.isin(y_test, dec_cls)
    f1_model, f1_probe = macro_f1(y_test, pred), macro_f1(y_test, probe_pred)
    f1_dec_model = macro_f1(y_test[sel_dec], pred[sel_dec], classes=dec_cls)
    f1_dec_probe = macro_f1(y_test[sel_dec], probe_pred[sel_dec], classes=dec_cls)
    mar_i = MECHS.index("mar_obs"); mn_i = [MECHS.index(m) for m in ("mnar_self", "mnar_lat", "mix")]
    sel_pm = np.isin(y_test, [mar_i] + mn_i)
    def bf1(pp):
        yb = np.isin(y_test[sel_pm], mn_i); pb = np.isin(pp[sel_pm], mn_i)
        tp = (pb & yb).sum(); fp = (pb & ~yb).sum(); fn = (~pb & yb).sum()
        return float(2 * tp / max(2 * tp + fp + fn, 1))
    e_val = ece(P, y_test)
    lat_i = MECHS.index("mnar_lat")
    ent_lat = float(ent[y_test == lat_i].mean())
    dec_correct = (pred == y_test) & sel_dec
    ent_dec = float(ent[dec_correct].mean()) if dec_correct.any() else 0.0
    s_true = np.array([r["s"] for r in test_recs]); ok = ~np.isnan(s_true)
    rho = spearman(s_true[ok], s_pred[ok]) if ok.sum() > 10 else float("nan")
    C = np.zeros((K, K), int)
    for t, p in zip(y_test, pred): C[t, p] += 1

    g_a = (f1_model - f1_probe >= 0.05) and (e_val <= 0.10)
    g_b = (f1_dec_model - f1_dec_probe >= 0.05) and (ent_lat - ent_dec >= 0.20)
    g_c = bf1(pred) - bf1(probe_pred) >= 0.05
    res = {
        "f1_model": round(f1_model, 3), "f1_probe": round(f1_probe, 3),
        "acc_model": round(float((pred == y_test).mean()), 3), "ece": round(e_val, 3),
        "f1_dec_model": round(f1_dec_model, 3), "f1_dec_probe": round(f1_dec_probe, 3),
        "mar_mnar_f1_model": round(bf1(pred), 3), "mar_mnar_f1_probe": round(bf1(probe_pred), 3),
        "entropy_lat": round(ent_lat, 3), "entropy_dec_correct": round(ent_dec, 3),
        "param_spearman": round(rho, 3) if rho == rho else None,
        "n_params_M": round(n_params / 1e6, 2), "steps": step,
        "gate_a": bool(g_a), "gate_b": bool(g_b), "gate_c": bool(g_c),
        "confusion": C.tolist(), "args": vars(args),
    }
    print("\n混淆矩阵(行=真值):")
    print("            " + "".join(f"{m:>10s}" for m in MECHS))
    for i, m in enumerate(MECHS):
        print(f"{m:>10s}  " + "".join(f"{C[i,j]:10d}" for j in range(K)))
    print(json.dumps(res, indent=2, ensure_ascii=False))
    print("\n=== E1 门控 v2 判读 ===")
    print(f"(a) F1差={f1_model-f1_probe:+.3f}(需≥+0.05) ECE={e_val:.3f}(需≤0.10) -> {'PASS' if g_a else 'FAIL'}")
    print(f"(b) 可判层F1差={f1_dec_model-f1_dec_probe:+.3f}(需≥+0.05) 熵分离={ent_lat-ent_dec:+.3f} nats(需≥+0.20) -> {'PASS' if g_b else 'FAIL'}")
    print(f"(c) MAR/MNAR F1差={bf1(pred)-bf1(probe_pred):+.3f}(需≥+0.05) -> {'PASS' if g_c else 'FAIL'}")
    print(f"综合: {'E1 PASS' if (g_a and g_b and g_c) else 'E1 FAIL'}")
    torch.save({"state_dict": model.state_dict(), "h": args.h,
                "families": MECHS, "res": res}, args.ckpt)
    print(f"检查点已存 {args.ckpt}(供 e2_transfer / e3_routing 加载)")
    json.dump(res, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"结果已存 {args.out}, 总耗时 {time.time()-t0:.0f}s")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="CPU 快速验证流程")
    ap.add_argument("--paired", action="store_true", help="PCC 配对腐化对比训练(§12 预注册)")
    ap.add_argument("--tables_per_class", type=int, default=1000, help="paired 模式每类表数(×4腐化)")
    ap.add_argument("--pool_per_class", type=int, default=2500)
    ap.add_argument("--test_per_class", type=int, default=150)
    ap.add_argument("--probe_n", type=int, default=3000)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--h", type=int, default=96)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="e1_results.json")
    ap.add_argument("--ckpt", default="e1_model.pt")
    args = ap.parse_args()
    if args.smoke:
        args.pool_per_class, args.test_per_class = 50, 20
        args.tables_per_class = 12
        args.probe_n, args.epochs, args.h = 200, 2, 32
        args.out = "e1_smoke_results.json"
        args.ckpt = "e1_smoke_model.pt"
    run(args)
