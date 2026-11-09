"""MW1 闭环 — Heckman/Diggle-Kenward 选择检验基线（审稿 R2 要求）
================================================================
Heckit 两步: ①选择方程 probit(_miss ~ z(X₋)) IRLS; ②观察行 OLS x_j ~ X₋ + Mills,
Mills 系数 t 检验 ≈ 自选择(MNAR)得分检验。表级分数 = 损坏列 p 最小值。
预测(理论先行): Gaussian 形边际族内有效; t/logn/mix 边际与 logit/threshold 链上误设失效。
协议: 与 levels_report.py 同池同种子; L2(MAR族 vs MNAR族) 二元 F1。
运行: python dk_baseline.py
"""
import numpy as np
from scipy import stats as sps

def _probit_irls(X, y, iters=30, ridge=1e-6):
    """Newton 求解 probit; X 含截距列; 返回 η=Xβ."""
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        eta = np.clip(X @ beta, -8, 8)
        p = sps.norm.cdf(eta)
        w = sps.norm.pdf(eta) ** 2 / np.clip(p * (1 - p), 1e-9, None)
        v = (y - p) / np.clip(p * (1 - p), 1e-9, None) * w  # 简化 score 步
        z = eta + v / np.clip(w, 1e-9, None)
        A = X.T @ (X * w[:, None]) + ridge * np.eye(X.shape[1])
        beta_new = np.linalg.solve(A, X.T @ (w * z))
        if np.max(np.abs(beta_new - beta)) < 1e-8:
            beta = beta_new; break
        beta = beta_new
    return X @ beta

def heckman_p(cols, types, mask, j_name):
    """列 j 的自选择检验 p 值; 返回 None 若不可算。"""
    names = list(cols.keys()); n = len(cols[names[0]])
    cont = [k for k, v in types.items() if v == "cont" and k != j_name]
    if len(cont) < 2:
        return None
    y_sel = mask[j_name].astype(float)
    cov_obs = ~np.stack([mask[k] if k in mask else np.zeros(n, bool) for k in cont], 1)
    sel_rows = cov_obs.all(1)                     # 协变量全观察的行(选择方程样本)
    if sel_rows.sum() < 60 or y_sel[sel_rows].std() < 1e-9:
        return None
    Z = np.stack([_zc(cols[k], sel_rows)[sel_rows] for k in cont], 1)
    X_sel = np.c_[Z, np.ones(sel_rows.sum())]
    eta_sel = _probit_irls(X_sel, y_sel[sel_rows])
    obs = (~mask[j_name]) & sel_rows              # 结果方程样本: j 且协变量观察
    if obs.sum() < 60:
        return None
    eta_full = np.full(n, np.nan); eta_full[sel_rows] = eta_sel
    eta_o = eta_full[obs]
    mills = sps.norm.pdf(eta_o) / np.clip(sps.norm.cdf(eta_o), 1e-9, None)
    Zo = np.stack([_zc(cols[k], obs)[obs] for k in cont], 1)
    Xo = np.c_[Zo, mills, np.ones(obs.sum())]
    y = _zc(cols[j_name], obs)[obs]
    beta, *_ = np.linalg.lstsq(Xo, y, rcond=None)
    resid = y - Xo @ beta
    dof = obs.sum() - Xo.shape[1]
    if dof < 10:
        return None
    s2 = resid @ resid / dof
    cov = s2 * np.linalg.inv(Xo.T @ Xo + 1e-9 * np.eye(Xo.shape[1]))
    t = beta[-2] / np.sqrt(max(cov[-2, -2], 1e-18))
    return float(2 * sps.norm.sf(abs(t)))

def _zc(x, rows):
    mu, sd = x[rows].mean(), x[rows].std() + 1e-9
    z = np.clip((x - mu) / sd, -5, 5)
    z[~rows] = 0.0
    return z

def table_score(cols, types, mask):
    """表级 MNAR 证据分数 = min p over 损坏连续列。"""
    ps = []
    for k, t in types.items():
        if t != "cont":
            continue
        m = mask[k]
        if m.std() < 1e-9:
            continue
        p = heckman_p(cols, types, mask, k)
        if p is not None:
            ps.append(p)
    return min(ps) if ps else None

MNAR_FAMS = {"mnar_self", "mnar_lat", "mix", "mnar_logit", "mnar_thresh"}

def binary_f1(y, p):
    tp = ((p == 1) & (y == 1)).sum(); fp = ((p == 1) & (y == 0)).sum(); fn = ((p == 0) & (y == 1)).sum()
    return float(2 * tp / max(2 * tp + fp + fn, 1))

def main(shift=0, tag=""):
    from e0_mechanism_probe import MECHS
    from mechzoo import make_tables, NOVEL
    import realdata, json
    # 训练池定阈值(与探针同协议, seed 21)
    tr_d, tr_r = make_tables(MECHS, 150, seed=21)
    tr_scores = [(table_score(d[0], d[1], d[2]), r["mech"]) for d, r in zip(tr_d, tr_r)]
    ts = [(s, m) for s, m in tr_scores if s is not None and m in ("mar_obs", "mnar_self", "mix")]
    sc = np.array([s for s, _ in ts]); yb = np.array([1 if m in MNAR_FAMS else 0 for _, m in ts])
    best_f1, best_thr = -1, 0.05
    for thr in np.arange(0.01, 0.5, 0.01):
        f1 = binary_f1(yb, (sc < thr).astype(int))
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    print(f"[train] threshold={best_thr:.2f} F1={best_f1:.3f} (n={len(ts)})")

    cells = {"base_fresh": make_tables(MECHS, 100, seed=777+shift),
             "novel": make_tables(NOVEL, 100, seed=778+shift)}
    real, realn = [], []
    for k, (name, cols, types) in enumerate(realdata.load_real_tables()):
        fn = (lambda r, c=cols, t=types: ({a: v.copy() for a, v in c.items()}, dict(t), len(c), len(c)))
        real.append(make_tables(MECHS, 8, seed=500 + k + shift, table_fn=fn))
        realn.append(make_tables(NOVEL, 20, seed=900 + k + shift, table_fn=fn))
    cells["real_inj"] = ([d for p in real for d in p[0]], [r for p in real for r in p[1]])
    cells["joint_shift"] = ([d for p in realn for d in p[0]], [r for p in realn for r in p[1]])

    res = {}
    for cname, (dicts, recs) in cells.items():
        scores, ys, nmiss = [], [], 0
        for d, r in zip(dicts, recs):
            if r["mech"] == "mcar":
                continue                       # L2 只评非 MCAR
            s = table_score(d[0], d[1], d[2])
            if s is None:
                nmiss += 1; continue
            scores.append(s); ys.append(1 if r["mech"] in MNAR_FAMS else 0)
        y = np.array(ys); pred = (np.array(scores) < best_thr).astype(int)
        res[cname] = {"L2_f1": round(binary_f1(y, pred), 3),
                      "acc": round(float((pred == y).mean()), 3),
                      "n": len(y), "n_unevaluable": nmiss,
                      "mnar_detected_rate": round(float(pred.mean()), 3)}
        print(cname, res[cname])
    json.dump(res, open(f"dk_baseline_results{tag}.json", "w", encoding="utf-8"), indent=2)
    print("已存 dk_baseline_results.json")

if __name__ == "__main__":
    import sys
    main(int(sys.argv[1]) if len(sys.argv)>1 else 0, sys.argv[2] if len(sys.argv)>2 else "")
