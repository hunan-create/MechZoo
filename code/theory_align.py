"""B3 ESN衰减律验证 + B4 λ扫描 + A2 链不变泛函表（§20 实验-理论对齐批）
运行: python theory_align.py
"""
import json
import os
import numpy as np
from scipy import stats as sps

out = {}

# ---------------- B3: ESN 衰减律 (引理2.1(c) 预测验证)
# 受控生成: x1~N(0,1), x_j = β·x1 + ε, ε~N(0, 1-β²) → τ_f²=β², σ²=1-β²
# 缺失 P(M=1)=Φ(b_s·z(x_j)), a=0(对称50%) → 观察侧 w(r)=Φ(-b_s(βx1+r))
# 理论: 观察残差~ESN(0,σ,α,0), α=α1σ, α1=b_s/√(1+b_s²β²) → SN 偏度闭式
def sn_skew(delta):
    d2 = 2 * delta ** 2 / np.pi
    return ((4 - np.pi) / 2) * (delta * np.sqrt(2 / np.pi)) ** 3 / (1 - d2) ** 1.5

rows = []
rng = np.random.default_rng(1)
for beta in [0.0, 0.45, 0.71, 0.9]:           # τ_f² = β²
    sigma = np.sqrt(1 - beta ** 2)
    for b_s in [0.5, 1.0, 2.0, 3.0]:
        sk_or, sk_fit = [], []
        for rep in range(50):
            n = 2000
            x1 = rng.normal(0, 1, n)
            eps = rng.normal(0, sigma, n)
            xj = beta * x1 + eps
            p_miss = sps.norm.cdf(b_s * xj)
            miss = rng.random(n) < p_miss
            obs = ~miss
            r_or = eps[obs]                                   # 预言机残差(ε 真值)
            # 再拟合残差: OLS on 观察行
            A = np.c_[x1[obs], np.ones(obs.sum())]
            w_hat, *_ = np.linalg.lstsq(A, xj[obs], rcond=None)
            r_fit = xj[obs] - A @ w_hat
            sk_or.append(sps.skew(r_or)); sk_fit.append(sps.skew(r_fit))
        alpha1 = b_s / np.sqrt(1 + b_s ** 2 * beta ** 2)
        alpha = alpha1 * sigma
        delta = alpha / np.sqrt(1 + alpha ** 2)
        theory = sn_skew(delta)
        rows.append({"tau2": round(beta ** 2, 2), "b": b_s,
                     "theory_skew": round(float(theory), 4),
                     "emp_oracle": round(float(np.mean(sk_or)), 4),
                     "emp_refit": round(float(np.mean(sk_fit)), 4)})
corr_or = np.corrcoef([r["theory_skew"] for r in rows], [r["emp_oracle"] for r in rows])[0, 1]
corr_fit = np.corrcoef([r["theory_skew"] for r in rows], [r["emp_refit"] for r in rows])[0, 1]
out["B3_esn"] = {"grid": rows, "corr_theory_oracle": round(float(corr_or), 4),
                 "corr_theory_refit": round(float(corr_fit), 4)}
print("B3 ESN验证: 理论vs预言机残差 ρ=%.4f | 理论vs再拟合残差 ρ=%.4f" % (corr_or, corr_fit))
for r in rows[::4]:
    print("   τ²=%.2f b=%.1f 理论=%.3f 预言机=%.3f 再拟合=%.3f" %
          (r["tau2"], r["b"], r["theory_skew"], r["emp_oracle"], r["emp_refit"]))

# ---------------- B4: mnar_lat 的 λ 扫描 (熵与mv信号单调衰减)
import torch
from e1_amortized import DualAxisSetEncoder, column_tokens
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def _ck(name):
    """resolve checkpoint path: cwd -> script dir -> ../checkpoints (repo layout)"""
    for p in (name, os.path.join(os.path.dirname(os.path.abspath(__file__)), name),
              os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "checkpoints", name)):
        if os.path.exists(p):
            return p
    return name


ck = torch.load(_ck("e1_model_iter2.pt"), map_location=dev, weights_only=False)
model = DualAxisSetEncoder(h=ck["h"]).to(dev); model.load_state_dict(ck["state_dict"]); model.eval()

def lat_table(lam, rng, b=1.5, n=800):
    d = 9
    cols = {f"c{i}": rng.normal(0, 1, n) for i in range(d)}
    cols["c0"] = cols["c1"] + 0.6 * cols["c2"] + rng.normal(0, 0.5, n)   # 有预测结构
    types = {k: "cont" for k in cols}
    u = rng.normal(0, 1, n)
    cols["c0"] = np.sqrt(1 - lam ** 2) * cols["c0"] + lam * u
    z0 = (cols["c0"] - cols["c0"].mean()) / (cols["c0"].std() + 1e-9)
    p_miss = sps.norm.cdf(0.5 * b * z0)   # rate≈0.31
    mask = {k: np.zeros(n, bool) for k in cols}
    mask["c0"] = rng.random(n) < p_miss
    return cols, types, mask

lam_rows = []
for lam in [0.2, 0.4, 0.6, 0.8, 0.95]:
    ents, mvs = [], []
    rng = np.random.default_rng(77)
    for _ in range(40):
        cols, types, mask = lat_table(lam, rng)
        tok = column_tokens(cols, types, mask)
        with torch.no_grad():
            P = torch.softmax(model(torch.tensor(tok[None], dtype=torch.float32, device=dev))[0], -1)[0]
            ents.append(float(-(P * torch.log(P + 1e-12)).sum()))
        m = mask["c0"].astype(float)
        mt = (m - m.mean()) / (m.std() + 1e-12)
        cors = [abs(np.corrcoef(mt, cols[k])[0, 1]) for k in ["c1", "c2"]]
        mvs.append(max(cors))
    lam_rows.append({"lam": lam, "entropy": round(float(np.mean(ents)), 3),
                     "mv_signal": round(float(np.mean(mvs)), 3)})
out["B4_lambda"] = lam_rows
print("B4 λ扫描:", lam_rows)

# ---------------- A2: 链不变泛函 (命题3 基准侧特权演示)
# 同一潜在选择强度, 用分位选择对比 C = s(z90)-s(z10) 表达时跨链一致所需参数不同
z90, z10 = sps.norm.ppf(0.9), sps.norm.ppf(0.1)
def C_probit(b, a): return sps.norm.cdf(a + b * z90) - sps.norm.cdf(a + b * z10)
def C_logit(b, a):
    s = lambda t: 1 / (1 + np.exp(-(a + b * t)))
    return s(z90) - s(z10)
rows2 = []
from scipy.optimize import brentq
for name, link, Cf in [("probit", lambda t: sps.norm.cdf(t), C_probit),
                       ("logit", lambda t: 1/(1+np.exp(-t)), C_logit)]:
    for b in [1.0, 2.0, 3.0]:
        a = brentq(lambda aa: link(aa) - 0.25, -5, 5)      # 零点处率=0.25
        rows2.append({"link": name, "b": b, "a": round(a, 3),
                      "C_q90_q10": round(float(Cf(b, a)), 3)})
out["A2_invariant"] = rows2
print("A2 链不变泛函(率均校准25%):")
for r in rows2:
    print("   ", r)

json.dump(out, open("theory_align.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("已存 theory_align.json")
