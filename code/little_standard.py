"""⑤ ICDE 审稿修复 — Little (1988) 标准检验实现（替换简化矩检验）
====================================================================
Little's MCAR test: d² = Σ_j n_j(x̄_j-μ̂)'Σ̂_obs(j)⁻¹(x̄_j-μ̂), df=Σ_j n_j p_j - p.
运行: python little_standard.py
"""
import numpy as np, json
from scipy import stats as sps

def little_standard_test(cols, types, mask, targets):
    """标准 Little (1988) MCAR 检验。仅用连续列；模式按 targets 联合缺失模式分组。
    返回 p 值。"""
    names = list(cols.keys()); n = len(cols[names[0]])
    cont = [k for k, v in types.items() if v == "cont"]
    p = len(cont)
    if p < 3 or not targets:
        return None
    # 观察矩阵 (连续列)
    X = np.stack([cols[k] for k in cont], 1)                # n×p
    O = np.stack([~mask.get(k, np.zeros(n, bool)) for k in cont], 1)  # n×p
    # 整体均值/协方差估计: 用完整个案 + 可用个案的 EM 一步简化
    mu_hat = np.array([X[O[:, j], j].mean() for j in range(p)])
    # 合并协方差: 可用个案的 pooled within
    Xc = X - mu_hat
    S = np.zeros((p, p))
    for j in range(p):
        for k in range(p):
            ok = O[:, j] & O[:, k]
            if ok.sum() > 30:
                S[j, k] = (Xc[ok, j] * Xc[ok, k]).mean()
    S += 1e-9 * np.eye(p)
    # 按缺失模式分组（连续列的联合模式）
    patterns = {}
    for i in range(n):
        pat = tuple(O[i].astype(int))
        patterns.setdefault(pat, []).append(i)
    # 检验统计量
    d2, dof = 0.0, 0
    for pat, idx in patterns.items():
        nj = len(idx)
        obs_j = np.array([bool(x) for x in pat])
        pj = int(obs_j.sum())
        if pj == 0 or nj < 10:
            continue
        xbar_j = X[idx][:, obs_j].mean(0)
        mu_j = mu_hat[obs_j]
        S_j = S[np.ix_(obs_j, obs_j)]
        try:
            diff = xbar_j - mu_j
            d2 += nj * diff @ np.linalg.solve(S_j, diff)
            dof += nj * pj
        except np.linalg.LinAlgError:
            continue
    dof = max(dof - p, 1)
    if d2 == 0:
        return 1.0
    return float(sps.chi2.sf(d2, dof))

def binary_f1(y, p):
    tp = ((p == 1) & (y == 1)).sum(); fp = ((p == 1) & (y == 0)).sum(); fn = ((p == 0) & (y == 1)).sum()
    return float(2 * tp / max(2 * tp + fp + fn, 1))

def main():
    from e0_mechanism_probe import MECHS
    from mechzoo import make_tables, NOVEL, little_test
    import realdata
    # 与 e2 同池
    b_d, b_r = make_tables(MECHS, 100, seed=777)
    n_d, n_r = make_tables(NOVEL, 100, seed=778)
    real, realn = [], []
    for k, (name, cols, types) in enumerate(realdata.load_real_tables()):
        fn = (lambda r, c=cols, t=types: ({a: v.copy() for a, v in c.items()}, dict(t), len(c), len(c)))
        real.append(make_tables(MECHS, 8, seed=500 + k, table_fn=fn))
        realn.append(make_tables(NOVEL, 20, seed=900 + k, table_fn=fn))

    res = {}
    cells = [("base_fresh", b_d, b_r), ("novel", n_d, n_r),
             ("real_inj", [d for p in real for d in p[0]], [r for p in real for r in p[1]]),
             ("joint_shift", [d for p in realn for d in p[0]], [r for p in realn for r in p[1]])]
    for pname, dts, rcs in cells:
        # standard vs simplified AUROC (MCAR vs non-MCAR)
        from scipy.stats import rankdata
        std_p, sim_p, labels = [], [], []
        for d, r in zip(dts, rcs):
            std = little_standard_test(d[0], d[1], d[2], d[3])
            sim = little_test(*d)
            if std is None:
                std = 1.0
            std_p.append(std); sim_p.append(sim); labels.append(1 if r["mech"] != "mcar" else 0)
        std_p, sim_p, labels = np.array(std_p), np.array(sim_p), np.array(labels)
        def auroc(s, l):
            r = rankdata(s); pos = l.sum()
            return (r[l == 1].sum() - pos * (pos + 1) / 2) / (pos * (1 - l).sum()) if pos > 0 and (1 - l).sum() > 0 else float("nan")
        au_std = auroc(1 - std_p, labels)
        au_sim = auroc(1 - sim_p, labels)
        res[pname] = {"auroc_standard": round(float(au_std), 3) if au_std == au_std else None,
                      "auroc_simplified": round(float(au_sim), 3) if au_sim == au_sim else None,
                      "n": len(labels)}
        print(f"{pname:12s} standard={au_std:.3f}  simplified={au_sim:.3f}  n={len(labels)}")
    json.dump(res, open("little_standard_results.json", "w", encoding="utf-8"), indent=2)
    print("已存 little_standard_results.json")

if __name__ == "__main__":
    main()
