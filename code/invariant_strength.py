"""⑤ §5.8 链相对强度 — 可观测投影分位对比估计器 Ĉ 的族内/跨链秩相关
================================================================================
命题 3 的经验对应: 原始斜率 b 是链相对的(跨链 ρ≤.11, 预注册失败),
而分位对比 C = s(q90)-s(q10) 是链不变泛函; 在"可观测投影"(目标列的
交叉验证预测成分 f̂, 观察行 OLS)上用 probit 拟合 P(miss|f̂) 可部分恢复 C。
协议:
  - 三族: probit 自掩蔽 / logit 自掩蔽 / 分位删失(threshold), 各 120 表;
    目标列 c0 = β·c1 + √(1-β²)·ε(可预测成分, β 均匀抽样), d=8, n=800;
  - 真值 C: s(x)=P(miss|c0=x) 在真列 q90/q10 上的差;
  - 估计 Ĉ: 观察行 OLS 得 f̂ → probit IRLS 拟合 m~f̂ →
    Ĉ = Φ(â+b̂ f̂q90) - Φ(â+b̂ f̂q10);
  - 族内 Spearman ρ(Ĉ, C) 与三族合并 ρ。
输出: invariant_results.json + invariant_pooled.json
运行: python invariant_strength.py        # CPU ~1min
对照: results/invariant_results.json(论文运行: probit .87 / logit .918 /
      threshold NaN, n=120)与 invariant_pooled.json(pooled .549)。
重算值随随机种子在 ±.05 内波动; 稳定结论是"族内高(≈.9)、跨链部分转移(≈.5)"。
"""
import json
import numpy as np
from scipy import stats as sps

N_TABLES = 120


def gen_table(rng, link):
    """一张自掩蔽表: 返回 (obs_fhat 估计材料, 真列, mask, 真值 C)。"""
    n, d = 800, 8
    beta = rng.uniform(0.3, 0.9)
    cols = {f"c{i}": rng.normal(0, 1, n) for i in range(d)}
    eps = rng.normal(0, np.sqrt(1 - beta ** 2), n)
    x0 = beta * cols["c1"] + eps
    b = rng.uniform(0.7, 3.0) * rng.choice([-1, 1])
    z0 = (x0 - x0.mean()) / (x0.std() + 1e-9)

    if link == "threshold":
        q = rng.uniform(0.5, 0.8); p0 = rng.uniform(0.02, 0.08)
        rate = rng.uniform(0.10, 0.40)
        thr = np.quantile(x0, q)
        p_hi = float(np.clip((rate - p0 * q) / (1 - q), 0.05, 1.0))
        s = lambda x: np.where(x > thr, p_hi, p0)
        mask = np.where(x0 > thr, rng.random(n) < p_hi, rng.random(n) < p0)
    else:
        rate = rng.uniform(0.10, 0.40)
        if link == "probit":
            s_raw = lambda a, x: sps.norm.cdf(a + b * x)
        else:
            s_raw = lambda a, x: 1 / (1 + np.exp(-(a + b * x)))
        lo, hi = -6.0, 6.0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if s_raw(mid, z0).mean() > rate: hi = mid
            else: lo = mid
        a = 0.5 * (lo + hi)
        s = lambda x: s_raw(a, x)
        mask = rng.random(n) < s(z0)

    q90t, q10t = np.quantile(x0, .9), np.quantile(x0, .1)
    C_true = float(np.mean(s(np.array([q90t]))) - np.mean(s(np.array([q10t]))))

    # 可观测投影: 观察行 OLS 用其余连续列预测 c0 → f̂(全行)
    obs = ~mask
    Xo = np.c_[np.stack([cols[f"c{i}"] for i in range(1, d)], 1)[obs], np.ones(int(obs.sum()))]
    w, *_ = np.linalg.lstsq(Xo, x0[obs], rcond=None)
    Xa = np.c_[np.stack([cols[f"c{i}"] for i in range(1, d)], 1), np.ones(n)]
    fhat = Xa @ w
    return fhat, mask, C_true, link


def probit_irls(x, m, iters=25):
    """一维 probit: P(m=1)=Φ(a+b x); 返回 (a, b)。"""
    X = np.c_[np.ones(len(x)), x]
    w = np.zeros(2)
    for _ in range(iters):
        eta = X @ w
        p = np.clip(sps.norm.cdf(eta), 1e-6, 1 - 1e-6)
        lam = sps.norm.pdf(eta) / p
        # 标准 IRLS: W = pdf²/(p(1-p)), z = eta + (m-p)/lam
        Wt = (sps.norm.pdf(eta) ** 2 / (p * (1 - p)))
        z = eta + (m - p) / np.clip(lam, 1e-8, None)
        wn = np.linalg.solve(X.T @ (X * Wt[:, None]) + 1e-6 * np.eye(2), X.T @ (Wt * z))
        if np.max(np.abs(wn - w)) < 1e-8:
            w = wn; break
        w = wn
    return w[0], w[1]


def main():
    rng = np.random.default_rng(7)
    out, allC, allChat = {}, [], []
    for link in ["probit", "logit", "threshold"]:
        Cs, Chats = [], []
        for _ in range(N_TABLES):
            fhat, mask, C_true, _ = gen_table(rng, link)
            a, b = probit_irls(fhat, mask.astype(float))
            q90, q10 = np.quantile(fhat, .9), np.quantile(fhat, .1)
            Chat = float(sps.norm.cdf(a + b * q90) - sps.norm.cdf(a + b * q10))
            Cs.append(C_true); Chats.append(Chat)
        Cs, Chats = np.array(Cs), np.array(Chats)
        if np.std(Chats) < 1e-9 or np.std(Cs) < 1e-9:
            rho = float("nan")
        else:
            rho = float(sps.spearmanr(Chats, Cs).statistic)
        out[link] = {"rho_Cest_vs_C": None if rho != rho else round(rho, 3), "n": N_TABLES}
        print(f"[{link}] rho(C_hat, C) = {out[link]['rho_Cest_vs_C']}")
        allC.append(Cs); allChat.append(Chats)
    pooled_C = np.concatenate(allC); pooled_Chat = np.concatenate(allChat)
    ok = (np.abs(pooled_Chat) < 5)  # 数值发散的极端拟合剔除
    rho_p = float(sps.spearmanr(pooled_Chat[ok], pooled_C[ok]).statistic)
    json.dump(out, open("invariant_results.json", "w", encoding="utf-8"), indent=2)
    json.dump({"pooled_rho": round(rho_p, 3), "n": int(ok.sum())},
              open("invariant_pooled.json", "w", encoding="utf-8"), indent=2)
    print(f"pooled rho = {rho_p:.3f} (n={int(ok.sum())}); 已存 invariant_results.json / invariant_pooled.json")


if __name__ == "__main__":
    main()
