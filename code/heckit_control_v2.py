"""R9-Fatal-3 修复：Heckit 正控 v2（带 τ_f² 旋钮 + 落盘 JSON + Wilson 区间）
======================================================================
相对 heckit_control.py 的改动：
  1) 生成器加入可预测成分: c0 = β·z(c1) + √(1−β²)·ε  ⇒ τ_f² = β²  (总方差 1)
  2) 截距按**平均缺失率**=rate 校准（原版只在 z=0 处令 Φ(a)=rate，等价于平均率≈38%）
  3) t3 误设臂在**生成数据时**注入重尾（原版在掩码之后加噪声，混淆了边际误设与测量误差）
  4) 全部结果写 heckit_control_results.json；每格给 Wilson 95% CI
运行: python heckit_control_v2.py [--reps 150]
"""
import argparse, json
import numpy as np
from scipy import stats as sps
from scipy.optimize import brentq
from dk_baseline import heckman_p


def z(x):
    sd = x.std()
    return (x - x.mean()) / (sd if sd > 1e-12 else 1.0)


def gauss_table(rng, mech, b, beta=0.0, rate=0.25, n=600, d=8, marginal="gauss"):
    base = (lambda k: rng.standard_t(3, k) / np.sqrt(3.0)) if marginal == "t3" else (lambda k: rng.normal(size=k))
    X = base((n, d))
    X[:, 0] = beta * z(X[:, 1]) + np.sqrt(max(1 - beta ** 2, 0.0)) * base(n)
    cols = {f"c{i}": X[:, i].copy() for i in range(d)}
    types = {k: "cont" for k in cols}
    drv = z(cols["c0"]) if mech == "mnar" else z(cols["c1"])
    a = brentq(lambda aa: sps.norm.cdf(aa + b * drv).mean() - rate, -10, 10)
    p = sps.norm.cdf(a + b * drv)
    mask = {k2: np.zeros(n, bool) for k2 in cols}
    mask["c0"] = rng.random(n) < p
    return cols, types, mask


def wilson(k, n, z=1.96):
    if n == 0:
        return [float("nan")] * 2
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(max(c - h, 0.0), 3), round(min(c + h, 1.0), 3)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=150)
    ap.add_argument("--out", default="heckit_control_results.json")
    args = ap.parse_args()
    rng = np.random.default_rng(5)
    out = {"design": {"n": 600, "d": 8, "rate": 0.25, "reps": args.reps,
                      "tau2_grid": [0.0, 0.0625, 0.24, 0.49, 0.66],
                      "note": "c0 = beta*z(c1)+sqrt(1-beta^2)*eps, tau_f^2=beta^2; "
                              "intercept calibrated on the mean rate"},
           "grid": [], "t3": []}

    for beta, tau2 in zip([0.0, 0.25, 0.49, 0.7, 0.81], [0.0, 0.0625, 0.24, 0.49, 0.66]):
        for b in [0.5, 1.0, 2.0, 3.0]:
            for mech, key in [("mar", "fpr"), ("mnar", "power")]:
                k = 0
                for _ in range(args.reps):
                    cols, types, mask = gauss_table(rng, mech, b, beta)
                    p = heckman_p(cols, types, mask, "c0")
                    k += int(p is not None and p < 0.05)
                row = {"tau2": round(beta ** 2, 4), "beta": beta, "b": b, "mech": mech,
                       "rate": round(k / args.reps, 3), "ci95": wilson(k, args.reps)}
                if mech == "mar":
                    row["fpr"] = row.pop("rate")
                else:
                    row["power"] = row.pop("rate")
                out["grid"].append(row)
            print(f"tau2={beta**2:.3f} b={b}: FPR={out['grid'][-2]['fpr']} "
                  f"power={out['grid'][-1]['power']}")

    for b in [2.0]:
        for mech, key in [("mar", "fpr"), ("mnar", "power")]:
            k = 0
            for _ in range(args.reps):
                cols, types, mask = gauss_table(rng, mech, b, 0.7, marginal="t3")
                p = heckman_p(cols, types, mask, "c0")
                k += int(p is not None and p < 0.05)
            out["t3"].append({"tau2": 0.49, "b": b, "mech": mech, key: round(k / args.reps, 3),
                              "ci95": wilson(k, args.reps)})
            print(f"t3 tau2=0.49 b={b} {mech}: {out['t3'][-1][key]}")

    json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("已存", args.out)


if __name__ == "__main__":
    main()
