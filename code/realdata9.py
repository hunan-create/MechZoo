"""9 源真实表加载器（ICDE 版 real-inj 432 表口径）
=====================================================
前 4 源沿用 data_cache/real_tables.npz（与历史结果逐字节一致），
后 5 源按名字从 OpenML 抓取，复用 realdata._finalize 的列选择/行子采样/标准化口径，
抓取后写入 data_cache/real_tables9.npz 永久离线。

设计约束（与 realdata 相同）：6~12 列、300~800 行、≥3 连续列、与 MechZoo 同格式。
OpenML 缓存目录强制落在工作区内（沙箱不允许写 ~/scikit_learn_data）。

运行: python realdata9.py        # 预取/刷新缓存并打印 9 源清单
"""
import socket
from pathlib import Path
import numpy as np

import realdata
from realdata import _finalize

CACHE4 = realdata.CACHE
CACHE9 = Path(__file__).parent / "data_cache" / "real_tables9.npz"
SK_HOME = str(Path(__file__).parent / "data_cache" / "sk")

# 新增 5 源：OpenML 数据集名（大小写不对时依次回退）。
# balancing-scale（4 列全低基数范畴）与 synthetic-control（OpenML 无同名活动集）在本管线不可用，
# 按同一口径替换为两张公开表（见 研究日志 §46），其余 3 源与原清单一致。
NEW_SOURCES = [
    ("ecoli", ["ecoli"]),                      # 替补 balancing-scale
    ("qsar-biodeg", ["qsar-biodeg"]),
    ("wine-quality-white", ["wine-quality-white"]),
    ("climate-model-simulation-crashes", ["climate-model-simulation-crashes"]),  # 替补 synthetic-control
    ("ILPD", ["ILPD", "ilpd"]),
]


def _load_cached4():
    z = np.load(CACHE4, allow_pickle=True)
    return [(n, dict(zip(z["names"][i], [z["cols"][i][j] for j in range(len(z["names"][i]))])),
             dict(zip(z["names"][i], z["types"][i]))) for i, n in enumerate(z["table_names"])]


def _select_columns_relaxed(df, rng):
    """比 realdata._select_columns 宽一档：数值列只要 nu>8 即视为连续（原规则要求 nu>20% 行数，
    会把 wine-quality / ILPD 这类中等基数数值列整列丢掉）。仅用于新增源。"""
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
        else:
            num.append((c, x, "cont"))
    if len(num) + len(cat) < 4 or len(num) < 3:
        return []
    rng.shuffle(num); rng.shuffle(cat)
    keep = num[:8] + cat[:3]
    rng.shuffle(keep)
    return keep[:12]


def _finalize_relaxed(name, df, rng, min_cols=4):
    """与 realdata._finalize 同口径，但列数下限放宽到 min_cols（balance-scale 只有 4 列），
    仍需 ≥3 连续列且 ≤12 列。仅用于新增源，不改动已缓存的 4 源。"""
    df = df.dropna(axis=1, thresh=int(0.8 * len(df)))
    if len(df) < realdata.MIN_ROWS:
        return None
    if len(df) > realdata.MAX_ROWS:
        df = df.iloc[rng.choice(len(df), realdata.MAX_ROWS, replace=False)]
    keep = _select_columns_relaxed(df, rng)
    if len(keep) < min_cols or sum(1 for _, _, t in keep if t == "cont") < 3:
        return None
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


def _fetch(name_candidates, rng):
    from sklearn.datasets import fetch_openml
    socket.setdefaulttimeout(30)
    last = None
    for nm in name_candidates:
        try:
            ds = fetch_openml(name=nm, as_frame=True, parser="auto", data_home=SK_HOME)
            df = ds.frame.drop(columns=[ds.target.name] if hasattr(ds.target, "name") else [],
                               errors="ignore")
            t = _finalize(nm, df, rng) or _finalize_relaxed(nm, df, rng)
            if t:
                print(f"  [openml] {nm}: {len(t[1])} 列 × {len(next(iter(t[1].values())))} 行")
                return t
            last = "finalize 拒绝(列数/连续列/行数不达标)"
        except Exception as e:
            last = f"{type(e).__name__}: {str(e)[:120]}"
        print(f"  [openml] {nm} 失败: {last}")
    raise RuntimeError(f"抓取失败: {name_candidates} -> {last}")


def load_real_tables9(refresh=False):
    """返回 9 源 list[(name, cols, types)]；缓存缺失时抓取并落盘。"""
    if CACHE9.exists() and not refresh:
        z = np.load(CACHE9, allow_pickle=True)
        tables = [(n, dict(zip(z["names"][i], [z["cols"][i][j] for j in range(len(z["names"][i]))])),
                   dict(zip(z["names"][i], z["types"][i]))) for i, n in enumerate(z["table_names"])]
        print(f"  [cache9] {len(tables)} 张真实表")
        return tables

    tables = _load_cached4()
    print(f"  [cache4] {len(tables)} 张")
    rng = np.random.default_rng(42)
    for label, cands in NEW_SOURCES:
        tables.append(_fetch(cands, rng))

    names = [list(t[1].keys()) for t in tables]
    cols = [[t[1][c] for c in ns] for ns, t in zip(names, tables)]
    types = [[t[2][c] for c in ns] for ns, t in zip(names, tables)]
    np.savez(CACHE9, table_names=[t[0] for t in tables],
             names=np.array(names, dtype=object), cols=np.array(cols, dtype=object),
             types=np.array(types, dtype=object))
    print(f"  已缓存 {len(tables)} 张 → {CACHE9}")
    return tables


if __name__ == "__main__":
    for name, cols, types in load_real_tables9(refresh=True):
        print(f"{name:28s} {len(cols):2d}列 × {len(next(iter(cols.values()))):4d}行 "
              f"({sum(1 for t in types.values() if t == 'cont')}cont)")
