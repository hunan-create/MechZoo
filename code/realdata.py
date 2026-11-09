"""真实表获取(E2 real 轴) — 三级回退, 永不阻塞
====================================================
  1) 本地缓存 data_cache/real_tables.npz(一旦建成永久离线)
  2) sklearn/statsmodels 内置真实数据集(离线保底: breast_cancer/diabetes)
  3) OpenML fetch_openml 尽力扩充(www.openml.org API 已验证可达;
     数据主机失败则跳过; Miss-Bench 克隆在当前网络下不稳, 留作手动路径)
产出: list[(name, cols{col->1d float array}, types{col->cont/bin/cat})]
约束: 6 ≤ 列数 ≤ 12(模型 D_MAX=12), 行数 300~800(子采样), 与 MechZoo 格式一致。
"""
import socket
from pathlib import Path
import numpy as np

CACHE = Path(__file__).parent / "data_cache" / "real_tables.npz"
MAX_TABLES, MIN_ROWS, MAX_ROWS = 8, 300, 800
RNG = np.random.default_rng(42)
SKIP_OPENML = False   # 冒烟/离线模式可置 True(只走内置保底表)

def _select_columns(df, rng):
    """混合类型列选择: 连续优先(≤8) + 低基数范畴(≤3), 共 6~12 列。"""
    num, cat = [], []
    for c in df.columns:
        try:
            x = df[c].astype(float).to_numpy()
        except (ValueError, TypeError):
            continue
        nu = len(np.unique(x[~np.isnan(x)]))
        if nu <= 1:
            continue
        if nu == 2:
            cat.append((c, x, "bin"))
        elif nu <= 8 and nu < 0.05 * len(x):
            cat.append((c, x, "cat"))
        elif nu > 0.2 * len(x):
            num.append((c, x, "cont"))
    rng.shuffle(num); rng.shuffle(cat)
    keep = num[:8] + cat[:3]
    rng.shuffle(keep)
    return keep[:12] if len(keep) >= 6 else (num + cat)[:12]

def _finalize(name, df, rng):
    """df → (name, cols, types); 行子采样 + 范畴编码 + 列选择。"""
    df = df.dropna(axis=1, thresh=int(0.8 * len(df)))
    if len(df) < MIN_ROWS:
        return None
    if len(df) > MAX_ROWS:
        df = df.iloc[rng.choice(len(df), MAX_ROWS, replace=False)]
    keep = _select_columns(df, rng)
    if len(keep) < 6 or sum(1 for _, _, t in keep if t == "cont") < 3:
        return None  # 机制动物园需要 ≥3 连续列做损坏目标
    cols, types = {}, {}
    for c, x, t in keep:
        if t == "cat":
            vals, codes = np.unique(x[~np.isnan(x)], return_inverse=True)
            xc = np.full(len(x), np.nan)
            xc[~np.isnan(x)] = codes
        else:
            xc = x
        ok = ~np.isnan(xc)
        mu, sd = xc[ok].mean(), xc[ok].std() + 1e-9
        cols[c] = np.nan_to_num((xc - mu) / sd, nan=0.0).astype(np.float64)
        types[c] = t
    return (name, cols, types)

def _bundled(rng):
    out = []
    try:
        from sklearn.datasets import load_breast_cancer, load_diabetes
        import pandas as pd
        bc = load_breast_cancer()
        out.append(_finalize("sklearn_breast_cancer", pd.DataFrame(bc.data, columns=bc.feature_names), rng))
        db = load_diabetes()
        out.append(_finalize("sklearn_diabetes", pd.DataFrame(db.data, columns=db.feature_names), rng))
    except Exception as e:
        print(f"  [bundled] 失败: {e}")
    return [t for t in out if t]

_OPENML_IDS = [61, 2, 30, 37, 31, 23, 1120, 50, 1461]  # 经典 UCI 镜像, 逐个 try

def _openml(rng, need):
    out = []
    if SKIP_OPENML:
        print("  [openml] 已跳过(SKIP_OPENML=True)")
        return out
    try:
        import pandas as pd
        from sklearn.datasets import fetch_openml
        socket.setdefaulttimeout(20)
        for did in _OPENML_IDS[: need + 3]:
            if len(out) >= need:
                break
            try:
                print(f"  [openml] 拉取 data_id={did} ...")
                ds = fetch_openml(data_id=did, as_frame=True, parser="auto")
                t = _finalize(f"openml_{did}", ds.frame.drop(columns=[ds.target.name]
                            if hasattr(ds.target, "name") else [], errors="ignore"), rng)
                if t:
                    out.append(t)
                    print(f"    -> {t[0]}: {len(t[1])} 列 × {len(next(iter(t[1].values())))} 行")
            except Exception as e:
                print(f"    -> 跳过: {type(e).__name__}: {str(e)[:80]}")
    except Exception as e:
        print(f"  [openml] 不可用: {e}")
    return out

def load_real_tables(refresh=False):
    """主入口: 缓存 → 内置 → OpenML, 结果写缓存。"""
    if CACHE.exists() and not refresh:
        z = np.load(CACHE, allow_pickle=True)
        tables = [(n, dict(zip(z["names"][i], [z["cols"][i][j] for j in range(len(z["names"][i]))])),
                   dict(zip(z["names"][i], z["types"][i]))) for i, n in enumerate(z["table_names"])]
        print(f"  [cache] {len(tables)} 张真实表")
        return tables
    rng = np.random.default_rng(42)
    tables = _bundled(rng)
    print(f"  [bundled] {len(tables)} 张")
    if len(tables) < MAX_TABLES:
        tables += _openml(rng, MAX_TABLES - len(tables))
    tables = tables[:MAX_TABLES]
    # 写缓存
    data_cache = CACHE.parent
    data_cache.mkdir(exist_ok=True)
    names = [list(t[1].keys()) for t in tables]
    cols = [[t[1][c] for c in ns] for ns, t in zip(names, tables)]
    types = [[t[2][c] for c in ns] for ns, t in zip(names, tables)]
    np.savez(CACHE, table_names=[t[0] for t in tables],
             names=np.array(names, dtype=object), cols=np.array(cols, dtype=object),
             types=np.array(types, dtype=object))
    print(f"  已缓存 {len(tables)} 张 → {CACHE}")
    return tables

if __name__ == "__main__":  # 手动预取/刷新
    for name, cols, types in load_real_tables(refresh=True):
        print(f"{name:28s} {len(cols)}列 × {len(next(iter(cols.values())))}行  "
              f"({sum(1 for t in types.values() if t=='cont')}cont)")
