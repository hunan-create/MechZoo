"""⑤ §5.3 补充 — α₀≠0 扩展 ESN 分支的六格验证(缺失率 25%)
==========================================================================
主网格(α₀=0, 16 格)由 theory_align.py 的 B3_esn 覆盖; 本脚本补正文提到的
"rate 0.25 (α₀≠0) 六格扩展网格": 对 (τ_f², b) ∈ {0,.5}×{1,2,3},
生成器(β 可预测成分 + probit 自选择, 截距按平均缺失率 0.25 校准)的
观察侧预言机残差偏度 vs 精确 ESN 密度采样的偏度(观察侧符号)。
输出: esn_extended.json
运行: python esn_extended.py          # CPU ~20s
对照: results/icde_fix_batch.json 的 esn_alpha0neq0 字段为论文运行落盘
      (|dev| ≤ 0.033, 与 α₀=0 主网格同量级)。
"""
import json
import numpy as np
from scipy import stats as sps


def sn_theory_skew(alpha0, alpha1, sigma, n_s=400_000, rng=None):
    """精确 ESN(0,σ,α₁σ,α₀) 采样后的偏度(拒绝采样, φσ 提案)。"""
    rng = rng or np.random.default_rng(0)
    keep = []
    while sum(map(len, keep)) < n_s:
        x = rng.normal(0, sigma, 4 * n_s)
        w = sps.norm.cdf(alpha0 + alpha1 * x)
        u = rng.random(len(x))
        keep.append(x[u < w])
    r = np.concatenate(keep)[:n_s]
    return float(sps.skew(r))


def cell(beta, b_s, rng, rate=0.25, n=2000, reps=50):
    sigma = np.sqrt(1 - beta ** 2)
    # 截距按平均缺失率校准(观察侧等价于缺失侧平均率)
    z = lambda v: (v - v.mean()) / (v.std() + 1e-9)
    lo, hi = -6.0, 6.0
    for _ in range(40):
        a = 0.5 * (lo + hi)
        # 平均率用数值期望近似: x_j~N(0,1) 时 E[Φ(a+b·x)]≈Φ(a/√(1+b²))
        if sps.norm.cdf(a / np.sqrt(1 + b_s ** 2)) > rate:
            hi = a
        else:
            lo = a
    a = 0.5 * (lo + hi)

    sk_gen = []
    for _ in range(reps):
        x1 = rng.normal(0, 1, n)
        eps = rng.normal(0, sigma, n)
        xj = beta * x1 + eps
        miss = rng.random(n) < sps.norm.cdf(a + b_s * xj)
        obs = ~miss
        sk_gen.append(sps.skew(eps[obs]))          # 预言机残差, 观察侧
    gen = float(np.mean(sk_gen))

    # 观察侧 ESN 参数(取负), α₁=b/√(1+b²τ²), α₀=(a+bμ_f)/√(...), μ_f=0
    alpha1 = b_s / np.sqrt(1 + b_s ** 2 * beta ** 2)
    alpha0 = a / np.sqrt(1 + b_s ** 2 * beta ** 2)
    return gen, alpha0, alpha1, sigma


def main():
    rng = np.random.default_rng(3)
    rows = []
    for beta, tau2 in [(0.0, 0.0), (0.71, 0.5)]:
        for b_s in [1.0, 2.0, 3.0]:
            gen, alpha0, alpha1, sigma = cell(beta, b_s, rng)
            # ESN 密度: f(r) ∝ φ_{σ}(r)·Φ(α₀+α₁·r)  (r 的系数就是 α₁ 本身)
            esn = sn_theory_skew(-alpha0, -alpha1, sigma, rng=rng)
            dev = abs(gen - esn)
            rows.append({"t2": tau2, "b": b_s, "gen": round(gen, 4),
                         "esn": round(esn, 4), "esn_abs": round(abs(esn), 4),
                         "dev": round(dev, 4)})
            print(f"τ²={tau2} b={b_s}  gen={gen:+.4f}  esn={esn:+.4f}  dev={dev:.4f}")
    json.dump({"esn_alpha0neq0": rows, "rate": 0.25},
              open("esn_extended.json", "w", encoding="utf-8"), indent=2)
    print("已存 esn_extended.json  max dev =", max(r["dev"] for r in rows))


if __name__ == "__main__":
    main()
