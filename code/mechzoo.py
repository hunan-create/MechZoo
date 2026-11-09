"""MechZoo v1 — E2/E3 共享库
================================
内容:
  - NOVEL 三种训练未见机制族(函数形式/驱动结构/删失形态迁移):
      mar_multi  双观测列驱动 logistic MAR
      mnar_logit logistic 自掩蔽(基族为 probit——函数形式迁移)
      mnar_thresh 分位数删失(高值确定性缺失+基底小概率——非光滑形态)
  - COARSE 粗三分法映射(Rubin): 所有族 → MCAR/MAR/MNAR(新族评测的目标粒度)
  - little_test: Little(1988) 风格 MCAR 检验(简化: 模式分组×逐列 Wald,
    合并尺度)——E2 的经典基线
  - make_tables: 通用表池构建(基础族 e0.apply_mechanism / 新族 apply_novel,
    格式与 e1_amortized.build_pool 完全一致, 供探针与模型共用)
依赖链: mechzoo → e1_amortized(模型/矩阵化) → e0_mechanism_probe(生成器)
"""
import numpy as np
from math import erf, sqrt
from scipy import stats as sps

from e0_mechanism_probe import gen_table, apply_mechanism, MECHS, calibrate_intercept, z
from e1_amortized import table_to_matrix, column_tokens  # column_tokens 重导出供 e2/e3

def effective_strength_ext(truth):
    """跨族机制强度(回归目标): 基础族有效 b + mnar_logit 的 b;
    mar_multi 取 |b| 均值(表内一致); mnar_thresh 无斜率参数不计。"""
    vals = [p for mech, p in truth.values()
            if mech in ("mar_obs", "mnar_self", "mnar_lat", "mix", "mnar_logit", "mar_multi")]
    return float(np.mean(vals)) if vals else None

NOVEL = ["mar_multi", "mnar_logit", "mnar_thresh"]
ALL_FAMILIES = MECHS + NOVEL

COARSE = {  # Rubin 粗三分: 机制族 → 0=MCAR, 1=MAR, 2=MNAR
    "mcar": 0, "mar_obs": 1, "skip": 1, "mar_multi": 1,
    "mnar_self": 2, "mnar_lat": 2, "mix": 2, "mnar_logit": 2, "mnar_thresh": 2,
}
COARSE_NAMES = ["MCAR", "MAR", "MNAR"]

_probit = np.vectorize(lambda t: 0.5 * (1 + erf(t / sqrt(2))))

def apply_novel(mech, cols, types, rng):
    """三种训练未见机制族; 返回 (mask, truth) 与 apply_mechanism 同构。
    truth 参数: mar_multi=|b|均值, mnar_logit=b, mnar_thresh=分位 q(无 logit 斜率)。"""
    n = len(next(iter(cols.values())))
    cont = [k for k, v in types.items() if v == "cont"]
    k_t = min(len(cont) - 2, int(rng.integers(2, 6))) if len(cont) > 4 else 2
    targets = list(rng.choice(cont, size=max(2, k_t), replace=False))
    mask = {k: np.zeros(n, bool) for k in cols}
    truth = {}
    for j in targets:
        rate = rng.uniform(0.10, 0.40)
        if mech == "mar_multi":
            d1, d2 = rng.choice([c for c in cont if c != j], size=2, replace=False)
            b1, b2 = rng.uniform(0.6, 2.0) * rng.choice([-1, 1]), rng.uniform(0.6, 2.0) * rng.choice([-1, 1])
            s = lambda a, j=j, b1=b1, b2=b2, d1=d1, d2=d2: 1 / (1 + np.exp(-(a + b1 * z(cols[d1]) + b2 * z(cols[d2]))))
            a = calibrate_intercept(s, rate)
            mask[j] = rng.random(n) < s(a)
            truth[j] = ("mar_multi", float((abs(b1) + abs(b2)) / 2))
        elif mech == "mnar_logit":
            b = rng.uniform(0.9, 3.0) * rng.choice([-1, 1])
            s = lambda a, j=j, b=b: 1 / (1 + np.exp(-(a + b * z(cols[j]))))
            a = calibrate_intercept(s, rate)
            mask[j] = rng.random(n) < s(a)
            truth[j] = ("mnar_logit", b)
        elif mech == "mnar_thresh":
            q = rng.uniform(0.5, 0.8)
            p0 = rng.uniform(0.02, 0.08)
            thr = np.quantile(cols[j], q)
            hi = cols[j] > thr
            p_hi = float(np.clip((rate - p0 * q) / (1 - q), 0.05, 1.0))
            mask[j] = np.where(hi, rng.random(n) < p_hi, rng.random(n) < p0)
            truth[j] = ("mnar_thresh", q)
    return mask, truth

def apply_family(mech, cols, types, rng):
    return apply_novel(mech, cols, types, rng) if mech in NOVEL \
        else apply_mechanism(mech, cols, types, rng)

# ---------------------------------------------------------------- Little 检验
def little_test(cols, types, mask, targets, min_group=20):
    """Little(1988) 风格 MCAR 检验·简化实现: 按目标列缺失模式分组, 对全部
    连续列做组均值-全局均值 Wald 统计(合并观察方差尺度), χ² 尾概率。
    分组不足 2 组(如纯 MCAR 单列情形退化)返回 p=1。"""
    n = len(next(iter(cols.values())))
    M = np.stack([mask[j] for j in targets], 1)
    use = [k for k, v in types.items() if v == "cont"]
    X = np.stack([z(cols[k]) for k in use], 1)
    obs = ~np.stack([mask[k] if k in mask else np.zeros(n, bool) for k in use], 1)
    groups = {}
    for i in range(n):
        groups.setdefault(tuple(M[i]), []).append(i)
    valid = [np.array(g) for g in groups.values() if len(g) >= min_group]
    if len(valid) < 2:
        return 1.0
    C = len(use)
    means = np.full((len(valid), C), np.nan)
    for gi, g in enumerate(valid):
        for c in range(C):
            m = obs[g, c]
            if m.sum() > 5:
                means[gi, c] = X[g, c][m].mean()
    overall = np.nanmean(means, 0)
    sd = np.array([X[:, c][obs[:, c]].std() for c in range(C)]) + 1e-9
    stat, dof = 0.0, 0
    for gi, g in enumerate(valid):
        for c in range(C):
            if np.isnan(means[gi, c]) or np.isnan(overall[c]):
                continue
            stat += len(g) * ((means[gi, c] - overall[c]) / sd[c]) ** 2
            dof += 1
    dof = max(dof - C, 1)
    return float(sps.chi2.sf(stat, dof))

# ---------------------------------------------------------------- 通用池构建
# 退化计数: 请求的族 vs 实际生成的机制 (R6/R8 修复: 真值必须来自实际机制, 不来自族名)
DEGRADED = {}

def effective_mech(requested, truth):
    """表级机制标签取实际逐列真值; 统计退化次数供报告使用。"""
    from e0_mechanism_probe import effective_mech as _eff
    eff = _eff(requested, truth)
    if eff != requested:
        DEGRADED[(requested, eff)] = DEGRADED.get((requested, eff), 0) + 1
    return eff

def make_tables(families, n_per_class, seed, table_fn=None):
    """按 families 列表逐族构建表池; 返回 (dicts, recs), 与 e1.build_pool 同构。
    table_fn(rng) → (cols, types, n, d) 可注入真实表采样器(E2 real 轴)。"""
    from e0_mechanism_probe import gen_table as _gt
    tf = table_fn or (lambda r: _gt(r))
    rng = np.random.default_rng(seed)
    dicts, recs = [], []
    for mech in families:
        made = 0
        while made < n_per_class:
            cols, types, *_ = tf(rng)
            mask, truth = apply_family(mech, cols, types, rng)
            rates = [mask[j].mean() for j in truth]
            if min(rates) < 0.03 or max(rates) > 0.6:
                continue
            eff = effective_mech(mech, truth)
            X, M, T = table_to_matrix(cols, types, mask)
            s = effective_strength_ext(truth)
            dicts.append((cols, types, mask, list(truth.keys())))
            recs.append({"X": X, "M": M, "T": T, "y": ALL_FAMILIES.index(eff),
                         "mech": eff, "family": mech, "truth": truth,
                         "tok": column_tokens(cols, types, mask),
                         "s": (s / 3.0) if s is not None else np.nan})
            made += 1
    return dicts, recs
