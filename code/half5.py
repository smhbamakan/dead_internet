import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(__file__)); __file__x = __file__
src = open(os.path.join(os.path.dirname(__file__), "theory5.py")).read()
exec(src.split("# ---------- (d)")[0]); exec("# ---------- (d)" + src.split("# ---------- (d)")[1].split("CACHE =")[0])
S = np.linspace(0.005, 0.5, 100); out = {}; curves = {}
for name, kw in [("base", {}), ("obs_m1", dict(model="obs", m=1, alpha=1.0, beta_=0.0)), ("obs_m3", dict(model="obs", m=3, alpha=0.9, beta_=0.1))]:
    for pi in [0.0, 0.3, 0.6, 0.9]:
        y = 1.0; ys = []
        for s in S:
            for _ in range(300): y = Phi(y, s, pi, 0.6, 0.05, **kw)
            ys.append(y)
        ys = np.array(ys); curves[f"{name}|{pi}"] = ys.tolist()
        out[f"{name}|{pi}"] = float(S[np.argmax(ys < 0.5)]) if (ys < 0.5).any() else None
json.dump(dict(half=out, S=S.tolist(), curves=curves), open(os.path.join(OUT, "half5.json"), "w"))
print(out)
