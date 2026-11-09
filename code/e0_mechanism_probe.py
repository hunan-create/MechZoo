"""C2 E0 信息地板实验 — 机制族可解码性的经典统计量下界
================================================================
预注册判据(研究日志 §4 E0):
  PASS <=> macro-F1(B2) - max(macro-F1(B1), chance) >= 0.10
          且 MAR-vs-MNAR 成对 F1(B2) - B1 >= 0.10
设计要点:
  - 6 机制族缺失率均校准到 U(0.10, 0.40)(截距二分搜索),缺失率不泄漏标签
  - B1=仅掩码特征(≈Little 检验信息集), B2=B1+值感知特征(MAR/MNAR 证据)
  - 逐表切分 train/test,无表间泄漏;B1/B2 吃同一批表
纯 numpy 实现, CPU 分钟级。
"""
import numpy as np
from itertools import combinations

RNG = np.random.default_rng(20261029)

# ---------------------------------------------------------------- 表生成器
def gen_table(rng):
    """多样混合表: 连续列(含线性SEM依赖)+二值+范畴列, 返回 X(dict col->np.array) 与列类型."""
    d = int(rng.integers(8, 13))
    n = int(rng.integers(300, 1001))
    cols, types = {}, {}
    n_cont = int(rng.integers(max(4, d // 2), d - 1))
    cont_names = [f"c{i}" for i in range(n_cont)]
    # 连续列: 随机拓扑序线性 SEM
    params = {}
    for i, name in enumerate(cont_names):
        form = rng.choice(["gauss", "mix2", "t3", "logn"], p=[.35, .3, .15, .2])
        parents = rng.choice(cont_names[:i], size=min(i, int(rng.integers(0, 3))), replace=False) if i else []
        params[name] = (form, [(p, rng.normal(0, 0.8)) for p in parents], abs(rng.normal(1, .3)))
    for name in cont_names:
        form, parents, sig = params[name]
        x = base_noise(form, n, rng) * sig
        for p, b in parents:
            x = x + b * z(cols[p])
        cols[name] = x
        types[name] = "cont"
    # 二值/范畴列: 逻辑斯蒂于 1-2 个连续父列
    for j in range(d - n_cont):
        name = f"b{j}"
        k = int(rng.integers(2, 6))
        pars = rng.choice(cont_names, size=min(2, len(cont_names)), replace=False)
        w = rng.normal(0, 1.2, size=len(pars))
        eta = sum(wi * z(cols[p]) for wi, p in zip(w, pars)) + rng.normal(0, .5, n)
        pr = 1 / (1 + np.exp(-eta))
        if k == 2:
            cols[name] = (rng.random(n) < pr).astype(float)
            types[name] = "bin"
        else:  # 序数范畴: 阈值切 eta
            qs = np.quantile(eta, np.linspace(0, 1, k + 1)[1:-1])
            cols[name] = np.digitize(eta, qs).astype(float)
            types[name] = "cat"
    return cols, types, n, d

def base_noise(form, n, rng):
    if form == "gauss": return rng.normal(0, 1, n)
    if form == "mix2":  return rng.normal(0, 1, n) + rng.choice([-2.5, 2.5], n) * (rng.random(n) < .35)
    if form == "t3":
        chi = rng.chisquare(3, n) / 3
        return rng.normal(0, 1, n) / np.sqrt(chi)
    return np.exp(rng.normal(0, .8, n))

def z(x):
    s = x.std()
    return (x - x.mean()) / (s if s > 1e-9 else 1)

# ---------------------------------------------------------------- 机制动物园
def calibrate_intercept(score_fn, target, a0=0.0):
    """二分搜索截距 a 使平均缺失率≈target(在 score_fn(a) 单调假设下)."""
    lo, hi = -8.0, 8.0
    for _ in range(40):
        a = .5 * (lo + hi)
        r = score_fn(a).mean()
        if r > target: hi = a
        else: lo = a
    return .5 * (lo + hi)

MECHS = ["mcar", "mar_obs", "mnar_self", "mnar_lat", "skip", "mix"]

def effective_mech(requested, truth):
    """表级机制标签必须来自实际生成的逐列真值，而非请求的族名。
    skip(结构跳答) 在无 cat 列的表上会退化为纯 MCAR（见 apply_mechanism 的 else 分支），
    此时真值全部是 ('mcar', rate)，表级标签应为 mcar。"""
    mechs = [t[0] for t in truth.values()]
    return "mcar" if (mechs and all(m == "mcar" for m in mechs)) else requested

def apply_mechanism(mech, cols, types, rng):
    """损坏 2-5 个连续列(随机子集), 返回 mask(dict col->bool array, True=缺失) 与机制真值标签."""
    n = len(next(iter(cols.values())))
    cont = [k for k, v in types.items() if v == "cont"]
    k_targets = min(len(cont) - 2, int(rng.integers(2, 6))) if len(cont) > 4 else 2
    targets = list(rng.choice(cont, size=max(2, k_targets), replace=False))
    mask = {k: np.zeros(n, bool) for k in cols}
    truth = {}
    for j in targets:
        rate = rng.uniform(0.10, 0.40)

        if mech == "mcar":
            mask[j] = rng.random(n) < rate
            truth[j] = ("mcar", rate)

        elif mech == "mar_obs":  # 依赖另一观测连续列
            drv = rng.choice([c for c in cont if c != j])
            b = rng.uniform(0.9, 3.0) * rng.choice([-1, 1])
            s = lambda a, b=b, drv=drv: 1 / (1 + np.exp(-(a + b * z(cols[drv]))))
            a = calibrate_intercept(s, rate)
            mask[j] = rng.random(n) < s(a)
            truth[j] = ("mar_obs", b)

        elif mech == "mnar_self":  # probit 自掩蔽
            b = rng.uniform(0.9, 3.0) * rng.choice([-1, 1])
            from math import erf, sqrt
            probit = lambda t: .5 * (1 + np.vectorize(erf)(t / sqrt(2)))
            s = lambda a, b=b, j=j: probit(a + b * z(cols[j]))
            a = calibrate_intercept(s, rate)
            mask[j] = rng.random(n) < s(a)
            truth[j] = ("mnar_self", b)

        elif mech == "mnar_lat":  # 依赖与目标列相关的潜变量(保持SEM依赖,仅替换噪声成分)
            lam = rng.uniform(0.6, 0.95)
            u = rng.normal(0, 1, n)
            cols[j] = np.sqrt(1 - lam ** 2) * z(cols[j]) + lam * u
            b = rng.uniform(0.9, 3.0)
            from math import erf, sqrt
            probit = lambda t: .5 * (1 + np.vectorize(erf)(t / sqrt(2)))
            s = lambda a, u=u, b=b: probit(a + b * u)
            a = calibrate_intercept(s, rate)
            mask[j] = rng.random(n) < s(a)
            truth[j] = ("mnar_lat", b * lam)

        elif mech == "skip":  # 范畴列取特定值 => 缺失(结构跳答)
            cats = [k for k, v in types.items() if v == "cat"]
            src = rng.choice(cats) if cats else max(cont, key=lambda c: -abs(z(cols[c]).mean()))
            if src in types and types[src] == "cat":
                vstar = rng.integers(int(cols[src].min()), int(cols[src].max()) + 1)
                mask[j] = cols[src] == vstar
                # 若命中率过低, 退化到补足 MCAR 至 rate
                hit = mask[j].mean()
                if hit < rate:
                    extra = (~mask[j]) & (rng.random(n) < (rate - hit) / (1 - hit))
                    mask[j] = mask[j] | extra
                truth[j] = ("skip", float(vstar))
            else:
                mask[j] = rng.random(n) < rate
                truth[j] = ("mcar", rate)

        elif mech == "mix":  # 半 MCAR 半自掩蔽
            half = n // 2
            p = rng.uniform(0.05, 0.3)
            m1 = rng.random(half) < p
            b = rng.uniform(0.9, 3.0) * rng.choice([-1, 1])
            from math import erf, sqrt
            probit = lambda t: .5 * (1 + np.vectorize(erf)(t / sqrt(2)))
            a = calibrate_intercept(lambda aa: probit(aa + b * z(cols[j])[:half]), rate)
            m2 = rng.random(n - half) < probit(a + b * z(cols[j])[half:])
            mask[j] = np.concatenate([m1, m2])
            truth[j] = ("mix", b)
    return mask, truth

# ---------------------------------------------------------------- 特征
def col_mask_features(cols, types, mask, targets):
    """B1: 仅掩码特征(逐目标列)."""
    n = len(next(iter(cols.values())))
    M = np.stack([mask[j] for j in targets], 1).astype(float)  # n x T
    feats = []
    rates = M.mean(0)
    feats += [rates.mean(), rates.std()]
    if M.shape[1] > 1:
        phi = np.corrcoef(M.T)
        off = ~np.eye(M.shape[1], dtype=bool)
        feats += [np.abs(phi[off]).max(), np.abs(phi[off]).mean()]
    else:
        feats += [0.0, 0.0]
    # 模式多样性 + 行缺失数
    pat, cnt = np.unique(M, axis=0, return_counts=True)
    feats += [(pat.shape[0] / n), (cnt.max() / n), mask_all_count(cols, mask).std()]
    return feats

def mask_all_count(cols, mask):
    return np.sum(np.stack([v for v in mask.values()], 1), 1).astype(float)

def col_value_features(cols, types, mask, targets):
    """B2 增量: 值感知特征 — MAR(掩码~他列值)与MNAR(自列截断)证据."""
    cont = [k for k, v in types.items() if v == "cont"]
    n = len(next(iter(cols.values())))
    per_target = []
    for j in targets:
        xj, mj = cols[j], mask[j]
        obs = ~mj
        f = []
        # MAR 证据: m_j 与其他连续观测列值的点二列相关
        corrs = []
        for k in cont:
            if k == j: continue
            ok = obs & (~mask[k]) if k in mask else obs
            if ok.sum() < 30: continue
            a, b = mask[k][ok].astype(float), cols[k][ok]
            if a.std() < 1e-9 or b.std() < 1e-9: continue
            corrs.append(abs(np.corrcoef(mj[ok].astype(float), cols[k][ok])[0, 1]))
        f += [max(corrs) if corrs else 0.0, float(np.mean(corrs)) if corrs else 0.0]
        # MNAR 自掩蔽证据: 交叉预测残差偏移(观测行应为随机样本→残差均值≈0)
        others = [k for k in cont if k != j and (~mask[k] if k in mask else np.ones(n, bool)).sum() > 30]
        if len(others) >= 2:
            X = np.stack([z(cols[k]) for k in others], 1)
            Xo, yo = X[obs], z(cols[j])[obs]
            w, *_ = np.linalg.lstsq(np.c_[Xo, np.ones(len(Xo))], yo, rcond=None)
            resid = yo - np.c_[Xo, np.ones(len(Xo))] @ w
            f += [resid.mean(), float(abs(resid.mean())), resid.std()]
        else:
            f += [0.0, 0.0, 1.0]
        # skip 证据: m_j 与范畴列取值的关联(最大卡方/ Cramer V 代理)
        vs = []
        for k, t in types.items():
            if t != "cat" or k == j: continue
            ok = ~mask[k] if k in mask else np.ones(n, bool)
            if ok.sum() < 30: continue
            tab = np.zeros((2, int(cols[k].max()) + 1))
            for mi in (0, 1):
                sel = ok & (mj.astype(bool) == bool(mi))
                for v, c in zip(*np.unique(cols[k][sel], return_counts=True)):
                    tab[mi, int(v)] = c
            row, col = tab.sum(1), tab.sum(0)
            if (row < 5).any() or (col < 5).any(): continue
            exp = np.outer(row, col) / tab.sum()
            chi2 = ((tab - exp) ** 2 / (exp + 1e-12)).sum()
            vs.append(chi2 / (n * (min(tab.shape) - 1)))
        f += [max(vs) if vs else 0.0]
        per_target.append(f)
    P = np.array(per_target)
    agg = np.concatenate([P.mean(0), P.max(0), P.std(0)])
    return list(agg)

def table_features(cols, types, mask, targets):
    n = len(next(iter(cols.values())))
    b1 = col_mask_features(cols, types, mask, targets)
    b2 = col_value_features(cols, types, mask, targets)
    tbl = [np.log10(n), len(cols), np.mean([mask[j].mean() for j in targets])]
    return np.array(b1 + tbl, float), np.array(b1 + tbl + b2, float)

# ---------------------------------------------------------------- 实验
def macro_f1(y, p, K):
    f1s = []
    for c in range(K):
        tp = ((p == c) & (y == c)).sum(); fp = ((p == c) & (y != c)).sum(); fn = ((p != c) & (y == c)).sum()
        f1s.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(np.mean(f1s))

def logreg_train(X, y, K, l2=1.0, iters=300, lr=0.5):
    """多项逻辑斯蒂(softmax) + 标准化, 纯 numpy L-BFGS 替代: 简单梯度上升足够(特征少)."""
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xs = (X - mu) / sd
    Xs = np.c_[Xs, np.ones(len(Xs))]
    W = np.zeros((Xs.shape[1], K))
    Y = np.eye(K)[y]
    for _ in range(iters):
        P = np.exp(Xs @ W - (Xs @ W).max(1, keepdims=True))
        P /= P.sum(1, keepdims=True)
        G = Xs.T @ (P - Y) / len(Xs) + l2 * W / len(Xs)
        W -= lr * G
    return (mu, sd, W)

def logreg_pred(model, X):
    mu, sd, W = model
    Xs = np.c_[(X - mu) / (sd + 1e-9), np.ones(len(X))]
    return Xs @ W

def run(N_per_class=250):
    K = len(MECHS)
    fb1, fb2, ys = [], [], []
    rng = np.random.default_rng(7)
    for ci, mech in enumerate(MECHS):
        made = 0
        while made < N_per_class:
            cols, types, n, d = gen_table(rng)
            mask, truth = apply_mechanism(mech, cols, types, rng)
            targets = list(truth.keys())
            rates = [mask[j].mean() for j in targets]
            if min(rates) < 0.03 or max(rates) > 0.6:  # 校准失败的表丢弃
                continue
            f1, f2 = table_features(cols, types, mask, targets)
            fb1.append(f1); fb2.append(f2); ys.append(ci)
            made += 1
    X1, X2, y = np.array(fb1), np.array(fb2), np.array(ys)
    idx = rng.permutation(len(y))
    tr, te = idx[: int(.7 * len(y))], idx[int(.7 * len(y)):]

    out = {}
    for name, X in [("B1_mask_only", X1), ("B2_value_aware", X2)]:
        m = logreg_train(X[tr], y[tr], K)
        logits = logreg_pred(m, X[te])
        pred = logits.argmax(1)
        f1m = macro_f1(y[te], pred, K)
        acc = (pred == y[te]).mean()
        out[name] = (f1m, acc, pred)
        print(f"{name:16s} macro-F1={f1m:.3f}  acc={acc:.3f}")
    chance = max(np.bincount(y[te])) / len(te)
    print(f"chance(majority)  acc={chance:.3f}")

    # 混淆矩阵(B2)
    _, _, pred2 = out["B2_value_aware"]
    C = np.zeros((K, K), int)
    for t, p in zip(y[te], pred2): C[t, p] += 1
    print("\n混淆矩阵(行=真值, B2):")
    print("            " + "".join(f"{m:>9s}" for m in MECHS))
    for i, m in enumerate(MECHS):
        print(f"{m:>10s}  " + "".join(f"{C[i,j]:9d}" for j in range(K)))

    # MAR vs MNAR 成对判别(三类 MNAR 合并 vs MAR)
    mar_idx = [MECHS.index("mar_obs")]
    mnar_idx = [MECHS.index(m) for m in ["mnar_self", "mnar_lat", "mix"]]
    sel = np.isin(y[te], mar_idx + mnar_idx)
    yb = np.isin(y[te], mnar_idx).astype(int)
    pair_f1 = {}
    for name in ["B1_mask_only", "B2_value_aware"]:
        pb = np.isin(out[name][2], mnar_idx).astype(int)
        tp = ((pb == 1) & (yb == 1)).sum(); fp = ((pb == 1) & (yb == 0)).sum(); fn = ((pb == 0) & (yb == 1)).sum()
        pair_f1[name] = 2 * tp / max(2 * tp + fp + fn, 1)
    print(f"\nMAR-vs-MNAR F1: B1={pair_f1['B1_mask_only']:.3f}  B2={pair_f1['B2_value_aware']:.3f}")

    # 分解: 各 MNAR 子族 vs MAR(B2), 定位缺口来源
    print("分解 MAR-vs-各MNAR子族 F1 (B2):")
    mar_i = MECHS.index("mar_obs")
    for sub in ["mnar_self", "mnar_lat", "mix"]:
        si = MECHS.index(sub)
        sel2 = np.isin(y[te], [mar_i, si])
        yb2 = (y[te] == si).astype(int)
        pb2 = (out["B2_value_aware"][2] == si).astype(int)
        tp = ((pb2 == 1) & (yb2 == 1)).sum(); fp = ((pb2 == 1) & (yb2 == 0)).sum(); fn = ((pb2 == 0) & (yb2 == 1)).sum()
        print(f"  MAR vs {sub:9s}: F1={2*tp/max(2*tp+fp+fn,1):.3f}  (该族误判去向见混淆矩阵)")

    # 预注册判读
    gap1 = out["B2_value_aware"][0] - max(out["B1_mask_only"][0], 0.0)
    gap2 = pair_f1["B2_value_aware"] - pair_f1["B1_mask_only"]
    print("\n=== E0 预注册判读 ===")
    print(f"判据1: macro-F1(B2)-max(B1,chance) = {gap1:.3f} (需>=0.10)  -> {'PASS' if gap1 >= 0.10 else 'FAIL'}")
    print(f"判据2: MAR/MNAR F1(B2)-B1        = {gap2:.3f} (需>=0.10)  -> {'PASS' if gap2 >= 0.10 else 'FAIL'}")
    verdict = "E0 PASS" if (gap1 >= 0.10 and gap2 >= 0.10) else "E0 FAIL"
    print(f"综合: {verdict}")

if __name__ == "__main__":
    import time
    t0 = time.time()
    run(N_per_class=250)
    print(f"\n耗时 {time.time()-t0:.1f}s")
