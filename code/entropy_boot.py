"""熵分离的逐表 bootstrap 95% CI（换 checkpoint 后重算口径不变的统计量）
统计量: mean H(posterior | 真值=mnar_lat) - mean H(posterior | 可判层且判对)
运行: python entropy_boot.py --ckpt e1_model_retrain.pt --out entropy_boot_retrain.json
"""
import argparse, json, os
import numpy as np
import torch

from e0_mechanism_probe import MECHS
from e1_amortized import DualAxisSetEncoder, build_pool, DECODABLE


def _ck(name):
    """resolve checkpoint path: cwd -> script dir -> ../checkpoints (repo layout)"""
    for p in (name, os.path.join(os.path.dirname(os.path.abspath(__file__)), name),
              os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "checkpoints", name)):
        if os.path.exists(p):
            return p
    return name



ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="e1_model_retrain.pt")
ap.add_argument("--out", default="entropy_boot_retrain.json")
ap.add_argument("--boot", type=int, default=2000)
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
a = ap.parse_args()

dev = torch.device(a.device)
a.ckpt = _ck(a.ckpt)
ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
model = DualAxisSetEncoder(h=ck["h"]).to(dev)
model.load_state_dict(ck["state_dict"]); model.eval()

_, recs = build_pool(150, seed=999)
with torch.no_grad():
    P = np.concatenate([
        torch.softmax(model(torch.tensor(np.stack([r["tok"] for r in recs[i:i + 64]]),
                                          dtype=torch.float32, device=dev))[0], -1).cpu().numpy()
        for i in range(0, len(recs), 64)])
y = np.array([r["y"] for r in recs])
pred = P.argmax(1)
H = -(P * np.log(P + 1e-12)).sum(1)

lat = y == MECHS.index("mnar_lat")
dec_ok = np.isin(y, [MECHS.index(m) for m in DECODABLE]) & (pred == y)
sep = float(H[lat].mean() - H[dec_ok].mean())
rng = np.random.default_rng(0)
bs = []
idx_all = np.arange(len(y))
for _ in range(a.boot):
    b = rng.choice(idx_all, len(idx_all), replace=True)
    lb, db = lat[b], dec_ok[b]
    if lb.sum() == 0 or db.sum() == 0:
        continue
    bs.append(H[b][lb].mean() - H[b][db].mean())
bs = np.array(bs)
res = {"ckpt": a.ckpt, "n": int(len(y)), "n_lat": int(lat.sum()), "n_dec_ok": int(dec_ok.sum()),
       "entropy_lat": round(float(H[lat].mean()), 3),
       "entropy_dec_correct": round(float(H[dec_ok].mean()), 3),
       "sep": round(sep, 3),
       "ci95": [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)],
       "boot": a.boot}
print(json.dumps(res, indent=1, ensure_ascii=False))
json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
