"""§29(a) E3b — 小样本机制条件处置重设计（预注册判据 a/b/c）
============================================================
任务表 n≤150 行；下游=RandomForest；β任务；路由 blind/oracle/model/B2/Little。
判据: (a) oracle增益≥3pp 于某层; (b) model regret≤0.5×oracle增益; (c) MNAR β-MAE 改善≥20%。
运行: python e3b.py
"""
import json, time
import os
import numpy as np
import torch
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from e0_mechanism_probe import gen_table, MECHS, z, table_features, logreg_train, logreg_pred
from e1_amortized import DualAxisSetEncoder, table_to_matrix
from mechzoo import apply_family, little_test, COARSE

POL_PRED = ["mean", "knn", "mice", "maskfeat"]
POL_BIAS = ["mean", "knn", "mice", "cc"]

def _ck(name):
    """resolve checkpoint path: cwd -> script dir -> ../checkpoints (repo layout)"""
    for p in (name, os.path.join(os.path.dirname(os.path.abspath(__file__)), name),
              os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "checkpoints", name)):
        if os.path.exists(p):
            return p
    return name


def make_small_tables(families, n_per, seed, n_hi=150, n_lo=80):
    rng = np.random.default_rng(seed)
    recs = []
    for mech in families:
        made = 0
        while made < n_per:
            cols, types, n, d = gen_table(rng)
            cont = [k for k, v in types.items() if v == "cont"]
            if len(cont) < 5:
                continue
            if len(cols) > 11:
                for k in [k for k, v in types.items() if v == "cat"] + list(cols):
                    if len(cols) <= 11: break
                    cols.pop(k); types.pop(k)
                cont = [k for k, v in types.items() if v == "cont"]
                if len(cont) < 5: continue
            n_t = int(rng.integers(n_lo, n_hi + 1))
            sel = rng.choice(n, n_t, replace=False)
            cols = {k: v[sel] for k, v in cols.items()}
            f1, f2 = rng.choice(cont, 2, replace=False)
            pr = 1 / (1 + np.exp(-(z(cols[f1]) + 0.8 * z(cols[f2]) + 0.3 * rng.normal(0, 1, n_t))))
            cols["label_y"] = (rng.random(n_t) < pr).astype(float)
            types["label_y"] = "bin"
            mask, truth = apply_family(mech, cols, types, rng)
            rates = [mask[j].mean() for j in truth]
            if min(rates) < 0.03 or max(rates) > 0.6:
                continue
            targets = list(truth.keys())
            best = None
            for a in list(rng.permutation(targets)):
                if types[a] != "cont": continue
                cands = sorted((abs(np.corrcoef(cols[a], cols[b])[0, 1]), b)
                               for b in cont if b not in targets and b != a)
                if cands and cands[-1][0] >= 0.25:
                    best = (cands[-1][0], a, cands[-1][1]); break
            if best is None: continue
            _, a_, b_ = best
            names = list(cols.keys())
            beta_full = np.polyfit(cols[a_], cols[b_], 1)[0]
            X, M, T = table_to_matrix(cols, types, mask)
            from e1_amortized import column_tokens
            recs.append({"X": X, "M": M, "T": T, "mech": mech, "y": cols["label_y"],
                         "ia": names.index(a_), "ib": names.index(b_), "iy": names.index("label_y"),
                         "beta_full": beta_full, "tok": column_tokens(cols, types, mask),
                         "feats": table_features(cols, types, mask, targets)[1],
                         "cols": cols, "types": types, "mask": mask, "targets": targets})
            made += 1
    return recs

def impute(X, M, policy):
    Xn = X.copy(); Xn[M > 0.5] = np.nan
    if policy == "mean":
        mu = np.nanmean(Xn, 0)
        return np.where(np.isnan(Xn), mu, Xn)
    if policy == "knn":
        return KNNImputer(n_neighbors=5).fit_transform(Xn)
    if policy == "mice":
        return IterativeImputer(max_iter=5, random_state=0).fit_transform(Xn)
    raise ValueError(policy)

def eval_pred(r, policy):
    keep = [j for j in range(r["X"].shape[1]) if j != r["iy"]]
    Xf, Mf = r["X"][:, keep], r["M"][:, keep]
    Xi = impute(Xf, Mf, policy if policy != "maskfeat" else "mean")
    if policy == "maskfeat":
        Xi = np.c_[Xi, Mf]
    n = len(r["y"]); rng = np.random.default_rng(0)
    idx = rng.permutation(n); tr, te = idx[: int(0.7 * n)], idx[int(0.7 * n):]
    clf = RandomForestClassifier(n_estimators=200, random_state=0).fit(Xi[tr], r["y"][tr])
    return float(clf.score(Xi[te], r["y"][te]))

def eval_bias(r, policy):
    if policy == "cc":
        ok = (r["M"][:, r["ia"]] < 0.5) & (r["M"][:, r["ib"]] < 0.5)
        if ok.sum() < 20: return abs(r["beta_full"]) * 2
        xa, xb = r["X"][ok, r["ia"]], r["X"][ok, r["ib"]]
    else:
        Xi = impute(r["X"], r["M"], policy)
        xa, xb = Xi[:, r["ia"]], Xi[:, r["ib"]]
    v = xa.var()
    return float(abs(np.cov(xa, xb)[0, 1] / (v + 1e-12) - r["beta_full"])) if v > 1e-6 else abs(r["beta_full"]) * 2

def main():
    t0 = time.time()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(_ck("e1_model_iter2.pt"), map_location=dev, weights_only=False)
    model = DualAxisSetEncoder(h=ck["h"]).to(dev); model.load_state_dict(ck["state_dict"]); model.eval()
    by_c = {c: [m for m in MECHS if COARSE[m] == c] for c in range(3)}
    tr_r = make_small_tables(MECHS, 20, seed=21)
    probe = logreg_train(np.array([r["feats"] for r in tr_r]), np.array([MECHS.index(r["mech"]) for r in tr_r]), len(MECHS))

    def posteriors(recs):
        out = []
        with torch.no_grad():
            for b0 in range(0, len(recs), 64):
                toks = torch.tensor(np.stack([r["tok"] for r in recs[b0:b0 + 64]]),
                                    dtype=torch.float32, device=dev)
                pf = torch.softmax(model(toks)[0], -1).cpu().numpy()
                out.append(pf)
        return np.concatenate(out)
    def probe_pred(recs):
        return logreg_pred(probe, np.array([r["feats"] for r in recs])).argmax(1)
    Cmap = np.array([COARSE[m] for m in MECHS])
    def coarse(p): return np.stack([p[:, Cmap == c].sum(1) for c in range(3)], 1)
    def little_cls(recs):
        return np.array([0 if little_test(r["cols"], r["types"], r["mask"], r["targets"]) > 0.05 else 1 for r in recs])

    print(f"[1/4] 校准池(30/粗类) {time.time()-t0:.0f}s")
    cal = {c: make_small_tables(by_c[c], 30, seed=100 + c) for c in range(3)}
    Vp = np.zeros((3, len(POL_PRED))); Vb = np.zeros((3, len(POL_BIAS)))
    for c in range(3):
        for i, p in enumerate(POL_PRED): Vp[c, i] = np.mean([eval_pred(r, p) for r in cal[c]])
        for i, p in enumerate(POL_BIAS): Vb[c, i] = np.mean([eval_bias(r, p) for r in cal[c]])
        print(f"   类{c}: pred={np.round(Vp[c],3)} bias={np.round(Vb[c],3)}")

    print(f"[2/4] 测试池 {time.time()-t0:.0f}s")
    test = {c: make_small_tables(by_c[c], 30, seed=200 + c) for c in range(3)}
    allr = sum(test.values(), [])
    yc = np.array([COARSE[r["mech"]] for r in allr])
    Pc = coarse(posteriors(allr)); ppc = coarse(np.eye(len(MECHS))[probe_pred(allr)])
    lcls = little_cls(allr)
    accs = {p: np.array([eval_pred(r, p) for r in allr]) for p in POL_PRED}
    biases = {p: np.array([eval_bias(r, p) for p in [r]][0]) for p in []} if False else \
        {p: np.array([eval_bias(r, p) for r in allr]) for p in POL_BIAS}

    print(f"[3/4] 路由 {time.time()-t0:.0f}s")
    blind_p = POL_PRED[int(np.mean(Vp, 0).argmax())]; blind_b = POL_BIAS[int(np.mean(Vb, 0).argmin())]
    def route_acc(kind, fn):
        vals = [accs[fn(i)][i] if kind == "pred" else biases[fn(i)][i] for i in range(len(allr))]
        sel = yc == 2 if kind == "bias" else np.ones(len(allr), bool)
        return float(np.mean([v for v, s in zip(vals, sel) if s]))
    r_blind = lambda i: blind_p; r_oracle = lambda i: POL_PRED[Vp[yc[i]].argmax()]
    r_model = lambda i: POL_PRED[(Pc[i] @ Vp).argmax()]
    r_probe = lambda i: POL_PRED[(ppc[i] @ Vp).argmax()]
    r_little = lambda i: "mean" if lcls[i] == 0 else "mice"
    pred_sc = {n: route_acc("pred", f) for n, f in [("blind", r_blind), ("oracle", r_oracle),
                                                    ("model", r_model), ("probe", r_probe), ("little", r_little)]}
    rb_blind = lambda i: blind_b; rb_oracle = lambda i: POL_BIAS[Vb[yc[i]].argmin()]
    rb_model = lambda i: POL_BIAS[(Pc[i] @ (-Vb)).argmin()]
    bias_sc = {n: route_acc("bias", f) for n, f in [("blind", rb_blind), ("oracle", rb_oracle), ("model", rb_model)]}

    print(f"[4/4] 判读 {time.time()-t0:.0f}s")
    og = pred_sc["oracle"] - pred_sc["blind"]
    regret = pred_sc["oracle"] - pred_sc["model"]
    ga = og >= 0.03
    gb = (regret <= 0.5 * og) if og > 0 else (regret <= 0.01)
    gc = bias_sc["model"] <= 0.8 * bias_sc["blind"]
    res = {"Vp": Vp.round(3).tolist(), "Vb": Vb.round(3).tolist(),
           "pred_acc": {k: round(v, 3) for k, v in pred_sc.items()},
           "bias_mnar": {k: round(v, 3) for k, v in bias_sc.items()},
           "oracle_gain": round(og, 4), "regret": round(regret, 4),
           "gate_a": bool(ga), "gate_b": bool(gb), "gate_c": bool(gc),
           "n_tables": len(allr)}
    print(json.dumps(res, indent=1))
    print(f"判据(a) oracle增益={og:+.4f}≥0.03? {'PASS' if ga else 'FAIL'} | "
          f"(b) regret={regret:+.4f}≤0.5×增益? {'PASS' if gb else 'FAIL'} | "
          f"(c) β改善={bias_sc['model']:.3f}≤0.8×blind({bias_sc['blind']:.3f})? {'PASS' if gc else 'FAIL'}")
    print("综合:", "E3b PASS" if (ga and gb and gc) else "E3b FAIL" + ("（判据a不满足→按预注册记死）" if not ga else ""))
    json.dump(res, open("e3b_results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
