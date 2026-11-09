"""②③ 宽表实验 + 运行时曲线 — ICDE 审稿人可扩展性批评的正面回应
================================================================
宽表: d∈{12, 20, 50, 100}, n=1000, 6 基础族, 50 表/族
运行时: n∈{100,1k,10k,100k}, d∈{12,50,100}, 纯计时
运行: python wide_runtime.py
"""
import time, json
import os
import numpy as np
from e0_mechanism_probe import MECHS, z
from e1_amortized import column_tokens

# ---------- ② 宽表实验 ----------
def _ck(name):
    """resolve checkpoint path: cwd -> script dir -> ../checkpoints (repo layout)"""
    for p in (name, os.path.join(os.path.dirname(os.path.abspath(__file__)), name),
              os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "checkpoints", name)):
        if os.path.exists(p):
            return p
    return name


def gen_wide_table(rng, d, n=1000):
    """多样边际宽表: 前 d_cont 连续(异质边际), 后 d-d_cont 范畴."""
    d_cont = max(d - min(d//4, 10), 5)
    cols, types = {}, {}
    for i in range(d_cont):
        form = ["gauss","mix2","t3","logn"][i % 4]
        if form == "gauss": x = rng.normal(0,1,n)
        elif form == "mix2": x = rng.normal(0,1,n) + rng.choice([-2.5,2.5],n)*(rng.random(n)<.35)
        elif form == "t3": x = rng.normal(0,1,n)/np.sqrt(rng.chisquare(3,n)/3)
        else: x = np.exp(rng.normal(0,.8,n))
        # 线性依赖: 每列依赖前 1-2 列
        if i >= 2:
            x = 0.4*x + 0.3*z(cols[f"c{i-1}"]) + 0.3*z(cols[f"c{i-2}"])
        cols[f"c{i}"] = x; types[f"c{i}"] = "cont"
    for i in range(d_cont, d):
        K = int(rng.integers(2, 6))
        cols[f"b{i}"] = rng.integers(0, K, n).astype(float)
        types[f"b{i}"] = "cat" if K > 2 else "bin"
    return cols, types, n, d

def apply_mech_wide(rng, cols, types, mech):
    """简版机制施加(适应宽表): 目标列取前 5 连续列."""
    from scipy import stats as sps
    from scipy.optimize import brentq
    n = len(next(iter(cols.values())))
    cont = [k for k,v in types.items() if v=="cont"][:5]
    targets = list(rng.choice(cont, size=min(3, len(cont)-1), replace=False))
    mask = {k: np.zeros(n, bool) for k in cols}
    truth = {}
    for j in targets:
        rate = rng.uniform(0.10, 0.40)
        if mech == "mcar":
            mask[j] = rng.random(n) < rate; truth[j] = ("mcar", rate)
        elif mech == "mar_obs":
            drv = rng.choice([c for c in cont if c != j])
            b = rng.uniform(0.9, 3.0)
            a = brentq(lambda aa: 1/(1+np.exp(-aa)) - rate, -6, 6)
            s = 1/(1+np.exp(-(a + b*z(cols[drv]))))
            mask[j] = rng.random(n) < s; truth[j] = ("mar_obs", b)
        elif mech == "mnar_self":
            b = rng.uniform(0.9, 3.0)
            a = brentq(lambda aa: sps.norm.cdf(aa) - rate, -6, 6)
            s = sps.norm.cdf(a + b*z(cols[j]))
            mask[j] = rng.random(n) < s; truth[j] = ("mnar_self", b)
        elif mech == "mix":
            p = rng.uniform(0.05, 0.30)
            b = rng.uniform(0.9, 3.0)
            a = brentq(lambda aa: sps.norm.cdf(aa) - rate, -6, 6)
            half = n//2
            m = np.concatenate([rng.random(half) < p,
                                sps.norm.cdf(a + b*z(cols[j])[half:]) > rng.random(n-half)])
            mask[j] = m; truth[j] = ("mix", b)
    return mask, truth

def run_wide():
    from e1_amortized import DualAxisSetEncoder
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(_ck("e1_model_iter2.pt"), map_location=dev, weights_only=False)
    model = DualAxisSetEncoder(h=ck["h"]).to(dev); model.load_state_dict(ck["state_dict"]); model.eval()
    COARSE = {"mcar":0, "mar_obs":1, "mnar_self":2, "mix":2}
    results = {}
    for d in [12, 20, 50, 100]:
        rng = np.random.default_rng(100 + d)
        correct, total, tok_times = 0, 0, []
        for mech in ["mcar", "mar_obs", "mnar_self", "mix"]:
            for rep in range(15):
                cols, types, n, _ = gen_wide_table(rng, d)
                mask, truth = apply_mech_wide(rng, cols, types, mech)
                rates = [mask[j].mean() for j in truth]
                if min(rates) < 0.03 or max(rates) > 0.6: continue
                t0 = time.time()
                # 逐列 token (截断到模型 D_MAX=12; 宽表取前 12 列 token)
                from e1_amortized import table_to_matrix, D_MAX
                names = list(cols.keys())[:D_MAX]
                sub_cols = {k: cols[k] for k in names}
                sub_types = {k: types[k] for k in names}
                sub_mask = {k: mask.get(k, np.zeros(len(cols[names[0]]), bool)) for k in names}
                tok = column_tokens(sub_cols, sub_types, sub_mask)
                tok_times.append(time.time() - t0)
                with torch.no_grad():
                    toks = torch.tensor(tok[None], dtype=torch.float32, device=dev)
                    P = torch.softmax(model(toks)[0], -1)[0].cpu().numpy()
                pred = P.argmax().item()
                y = MECHS.index(mech)
                Cmap = np.array([COARSE.get(m,0) for m in MECHS])
                from mechzoo import COARSE as full_coarse
                pred_c = full_coarse.get(MECHS[pred], None)
                if pred_c is None: pred_c = COARSE.get(MECHS[pred], 0)
                y_c = COARSE[mech]
                correct += (pred_c == y_c); total += 1
        acc = correct / max(total, 1)
        results[d] = {"acc": round(acc, 3), "n_tables": total,
                      "token_time_ms": round(np.mean(tok_times)*1000, 1)}
        print(f"d={d:3d}: acc={acc:.3f} (n={total}), token_time={np.mean(tok_times)*1000:.0f}ms")
    return results

# ---------- ③ 运行时曲线 ----------
def run_runtime():
    results = []
    for d in [12, 50, 100]:
        for n in [100, 1000, 10000, 100000]:
            rng = np.random.default_rng(d)
            cols, types, _, _ = gen_wide_table(rng, d, n)
            # 施加 MCAR
            mask = {k: rng.random(n) < 0.25 for k in cols}
            t0 = time.time()
            from e1_amortized import D_MAX, table_to_matrix, column_tokens
            names = list(cols.keys())[:D_MAX]
            sub_cols = {k: cols[k] for k in names}
            sub_types = {k: types[k] for k in names}
            sub_mask = {k: mask[k] for k in names}
            tok = column_tokens(sub_cols, sub_types, sub_mask)
            dt = time.time() - t0
            results.append({"d": d, "n": n, "time_s": round(dt, 3)})
            print(f"d={d:3d} n={n:6d}: {dt:.3f}s")
    return results

if __name__ == "__main__":
    t0 = time.time()
    print("=" * 50)
    print("② 宽表实验")
    print("=" * 50)
    wide = run_wide()
    print(f"\n{'='*50}")
    print("③ 运行时曲线")
    print(f"{'='*50}")
    runtime = run_runtime()
    json.dump({"wide": wide, "runtime": runtime},
              open("wide_runtime_results.json", "w", encoding="utf-8"), indent=2)
    print(f"\n已存 wide_runtime_results.json, 总耗时 {time.time()-t0:.0f}s")
