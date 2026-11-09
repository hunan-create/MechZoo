"""R4-W4 — Heckit 纯高斯正控：正确设定下的 size/power
======================================================
生成: x_k ~ N(0,1) iid, d=8, n=600。正确设定的 Heckit 世界:
  MAR:     P(miss_j)=Φ(a+b·z(x_k)), k≠j (probit 选择于观测列)
  MNAR-self: P(miss_j)=Φ(a+b·z(x_j))  (probit 自选择)
预测: MAR 表检出率(假阳)≈α 水平; MNAR 检出率(功效)随 b 升高。
若假阳失控 → 实现/渐近问题, 论文归因须撤回。
运行: python heckit_control.py
"""
import numpy as np
from scipy import stats as sps
from dk_baseline import heckman_p

def gauss_table(rng, mech, b, rate=0.25, n=600, d=8):
    X = rng.normal(0, 1, (n, d))
    cols = {f"c{i}": X[:, i].copy() for i in range(d)}
    types = {k: "cont" for k in cols}
    j, k = "c0", "c1"
    zj = (cols[j] - cols[j].mean()) / cols[j].std()
    zk = (cols[k] - cols[k].mean()) / cols[k].std()
    drv = zj if mech == "mnar" else zk
    from scipy.optimize import brentq
    a = brentq(lambda aa: sps.norm.cdf(aa + b * 0) - rate, -6, 6)  # z=0 处率=rate
    p = sps.norm.cdf(a + b * drv)
    mask = {k2: np.zeros(n, bool) for k2 in cols}
    mask[j] = rng.random(n) < p
    return cols, types, mask

def main():
    rng = np.random.default_rng(5)
    print("正确设定(纯高斯+probit)下的 Heckit: MAR检出率(假阳/size) vs MNAR检出率(功效)")
    for b in [0.5, 1.0, 2.0, 3.0]:
        fp, pw = [], []
        for rep in range(60):
            for mech, sink in [("mar", fp), ("mnar", pw)]:
                cols, types, mask = gauss_table(rng, mech, b)
                p = heckman_p(cols, types, mask, "c0")
                sink.append(1 if (p is not None and p < 0.05) else 0)
        print(f"  b={b:.1f}:  MAR 假阳率={np.mean(fp):.2f} (期望≈0.05)   "
              f"MNAR 功效={np.mean(pw):.2f}")
    # 论文归因的对照: 把边际换成 t3/logn(误设), size 应失控
    print("\n误设对照(t3 边际, 其余同):")
    for b in [2.0]:
        fp, pw = [], []
        for rep in range(60):
            for mech, sink in [("mar", fp), ("mnar", pw)]:
                cols, types, mask = gauss_table(rng, mech, b)
                # 注入重尾: c0 与驱动列加 t 噪声
                cols["c0"] = cols["c0"] + rng.standard_t(3, 600) * 0.8
                cols["c1"] = cols["c1"] + rng.standard_t(3, 600) * 0.8
                p = heckman_p(cols, types, mask, "c0")
                sink.append(1 if (p is not None and p < 0.05) else 0)
        print(f"  b={b:.1f}:  MAR 假阳率={np.mean(fp):.2f}   MNAR 检出率={np.mean(pw):.2f}")

if __name__ == "__main__":
    main()
