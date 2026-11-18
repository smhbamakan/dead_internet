"""Paired statistics, direct critical fraction, half-trust thresholds."""
import os, sys, json
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
sys.path.insert(0, os.path.dirname(__file__))
import abm5 as M
OUT = M.OUT; R = {}
d = pd.read_csv(os.path.join(OUT, "main.csv"))
def paired(sc, s, r1, r2, col):
    a = d[(d.scenario == sc) & (d.sigma == s) & (d.regime == r1)].sort_values("run")[col].values
    b = d[(d.scenario == sc) & (d.sigma == s) & (d.regime == r2)].sort_values("run")[col].values
    diff = a - b; rng = np.random.default_rng(0)
    bs = [rng.choice(diff, len(diff)).mean() for _ in range(4000)]
    p = 1.0 if np.allclose(diff, 0) else wilcoxon(a, b).pvalue
    return dict(c=f"{sc} s={s} {r1}-{r2} {col}", diff=float(diff.mean()), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)), p=float(p), n_ties=int((diff == 0).sum()))
C = [("S2", 0.2, "D2C", "D2", "fake"), ("S2", 0.2, "D3", "D2C", "fake"), ("S2", 0.2, "D3B", "D3", "fake"), ("S2", 0.2, "D3W", "D3", "fake"),
     ("S2", 0.2, "D4", "D3", "fake"), ("S1", 0.2, "D3", "D2C", "auth"), ("S1", 0.2, "D3W", "D2C", "auth"), ("S1", 0.2, "D2", "D0", "auth"),
     ("S1", 0.05, "D3", "D3W", "fake"), ("S1", 0.05, "D3", "D3W", "auth"), ("S2", 0.05, "D3B", "D3", "fake"), ("S1", 0.05, "D4", "D3", "fake"),
     ("S2", 0.2, "D2C", "D2", "peak_hs"), ("S2", 0.2, "D3", "D2", "over_kvah")]
res = [paired(*c) for c in C]
ps = np.array([r["p"] for r in res]); order = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
for k, i in enumerate(order):
    run = max(run, min(1, (m - k) * ps[i])); adj[i] = run
for r, a in zip(res, adj): r["p_holm"] = float(a)
R["paired"] = res
# direct critical fraction from power-flow table, Monte Carlo over EV assignment
rng = np.random.default_rng(1); xs = np.linspace(0, 0.2, 201); P = M.PAR
peak = []
for x in xs:
    vals = []
    for _ in range(400):
        act = rng.random(P["hh_feeder"]) < x; ev = rng.random(P["hh_feeder"]) < P["ev_share"]
        vals.append(M.loading_of((act * np.where(ev, P["ev_kw"], P["other_kw"])).sum()))
    peak.append([np.mean(vals), np.percentile(vals, 2.5), np.percentile(vals, 97.5)])
peak = np.array(peak)
def cross(col): i = np.argmax(peak[:, col] >= 100); return float(xs[i])
R["xstar_direct"] = dict(mean=cross(0), lo=cross(2), hi=cross(1))
np.save(os.path.join(OUT, "xstar_curve.npy"), np.c_[xs, peak])
# regression on simulated runs (for comparison)
x = d[d.fake > 0]; c = np.polyfit(x.fake, x.peak_load, 1); R["regression"] = dict(intercept=float(c[1]), slope=float(c[0]), xstar=float((100 - c[1]) / c[0]))
T = M.pf_table(); R["pf_slope_kw"] = [(float(T[i, 0]), float(T[i, 1])) for i in [0, 5, 10, 20, 30, 40]]
json.dump(R, open(os.path.join(OUT, "stats5.json"), "w"), indent=1)
for r in res: print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
print(R["xstar_direct"], R["regression"], R["pf_slope_kw"])
