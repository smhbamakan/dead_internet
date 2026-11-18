"""Revision-5 coupled model. Paired design (common random numbers), eight regimes, three scenarios,
noisy / spoofed consistency channel, time-resolved transformer loading and IEC 60076-7 thermal response.
Usage:
  python3 -I abm5.py main        -> ../out5/main.csv
  python3 -I abm5.py pareto      -> ../out5/pareto.csv
  python3 -I abm5.py robust      -> ../out5/robust.csv
  python3 -I abm5.py attack      -> ../out5/attack.csv
"""
import os, sys, json, itertools, warnings, copy
import numpy as np, pandas as pd, networkx as nx
sys.path.insert(0, os.path.dirname(__file__))
import sl
warnings.filterwarnings("ignore")
OUT = os.path.join(os.path.dirname(__file__), "..", "out5"); os.makedirs(OUT, exist_ok=True)

PAR = dict(N=3000, m=4, T=16, c=0.25, c_conc=None, share_p=0.5, bot_post_p=0.6, bot_followers=40,
           s=0.5, kappa=0.6, rho=0.75, tpr=0.30, fpr=0.05, f_util=0.20, pi_ecosys=0.6, phc_weight=0.1,
           sig_hat_mult=1.0, eps_con=0.0, psi=0.0, graph="BA", tau=0.4, w_c=0.6, w_k=0.4,
           con_rel=0.9, ev_share=0.35, ev_kw=7.4, other_kw=1.5, hh_feeder=240, mapping="random")
W_UTIL = sl.op(0.95, 0.0, 0.05); W_AGG = sl.op(0.80, 0.0, 0.20)
TRUST = {"util": W_UTIL, "agg": W_AGG}
REGIMES = ["D0", "D1", "D2", "D2C", "D3", "D3B", "D3W", "D4"]


def base_rate_unsigned(sigma, pi, q=0.0):
    num = (1 - sigma) * (1 - pi); den = num + sigma * (1 - q)
    return num / den if den > 0 else 1.0


def observe_con(authentic, rng, p):
    """Cross-plane consistency observation: None (unavailable), +1 consistent, -1 contradicts."""
    if rng.random() >= p["kappa"]:
        return None
    if authentic:
        return -1 if rng.random() < p["eps_con"] else +1
    return +1 if rng.random() < p["psi"] else -1   # psi: spoofed tariff channel (A3)


def accept(regime, cred, authentic, sigma, rng, p):
    if regime == "D0":
        return True
    if regime == "D1":
        return not (rng.random() < (p["fpr"] if authentic else p["tpr"]))
    if regime == "D2":
        return cred is not None
    con = observe_con(authentic, rng, p)
    if regime == "D2C":
        return cred is not None and con != -1
    a = base_rate_unsigned(min(sigma * p["sig_hat_mult"], 0.99), p["pi_ecosys"])
    if regime in ("D3", "D4"):
        w_c = sl.discount(TRUST[cred], sl.op(1, 0, 0, a)) if cred else sl.vacuous(a)
        w_c[3] = a
        rel = p["con_rel"]
        w_k = sl.vacuous(a) if con is None else (sl.op(rel, 0, 1 - rel, a) if con > 0 else sl.op(0, rel, 1 - rel, a))
        return sl.P(sl.fuse(w_c, w_k)) >= p["rho"]
    if regime == "D3B":   # Bayesian fusion of the same two channels
        p0 = sl.P(sl.discount(TRUST[cred], sl.op(1, 0, 0, a))) if cred else a
        if con is not None:
            lr = p["con_rel"] / (1 - p["con_rel"]); lr = lr if con > 0 else 1 / lr
            p0 = p0 * lr / (p0 * lr + 1 - p0)
        return p0 >= p["rho"]
    if regime == "D3W":   # weighted evidence score, no base rate
        sc = (p["w_c"] * TRUST[cred][0] if cred else 0.0) + (0.0 if con is None else p["w_k"] * con)
        return sc >= p["tau"]
    raise ValueError(regime)


def build_network(sigma, rng, p):
    N = p["N"]; seed = int(rng.integers(1e9))
    if p["graph"] == "BA":
        G = nx.barabasi_albert_graph(N, p["m"], seed=seed)
    elif p["graph"] == "WS":
        G = nx.watts_strogatz_graph(N, 2 * p["m"], 0.1, seed=seed)
    elif p["graph"] == "SBM":   # 10 homophilous communities, mean degree ~8
        k = 10; sizes = [N // k] * k; pin, pout = 7.0 / (N // k), 1.0 / (N - N // k)
        G = nx.stochastic_block_model(sizes, [[pin if i == j else pout for j in range(k)] for i in range(k)], seed=seed)
    nb = int(round(sigma / (1 - sigma) * N))
    bt = [rng.choice(N, p["bot_followers"], replace=False) for _ in range(nb)]
    nbrs = [list(G.neighbors(i)) for i in range(N)]
    cvec = np.full(N, p["c"]) if not p["c_conc"] else rng.beta(p["c"] * p["c_conc"], (1 - p["c"]) * p["c_conc"], N)
    return dict(G=G, nbrs=nbrs, nb=nb, bt=bt, cvec=cvec)


def spread(net, regime, scenario, authentic, sigma, seed, p):
    """Returns believe array and exposure counters. Common random numbers: same seed across regimes."""
    rng = np.random.default_rng(seed)
    N = p["N"]; believe = np.zeros(N, bool); evaluated = np.zeros(N, bool); queue = []
    exp_bot = exp_hum = 0
    def emit(cred, targets, from_bot):
        nonlocal exp_bot, exp_hum
        for t in targets:
            c = cred
            if c is not None and rng.random() < p["s"]:
                c = None
            w = p["phc_weight"] if (from_bot and regime == "D4") else 1.0
            if rng.random() < w:
                queue.append((t, c))
                if from_bot: exp_bot += 1
                else: exp_hum += 1
    if authentic:
        for t in rng.choice(N, int(p["f_util"] * N), replace=False):
            queue.append((t, "util"))
    fake_cred = None if scenario == "S1" else "agg"
    for step in range(p["T"]):
        if not authentic:
            for b in range(net["nb"]):
                if rng.random() < p["bot_post_p"]:
                    if regime == "D1" and rng.random() < p["tpr"]:
                        continue
                    emit(fake_cred, net["bt"][b], True)
        new = []
        for t, cred in queue:
            if believe[t] or evaluated[t]:
                continue
            evaluated[t] = True
            if accept(regime, cred, authentic, sigma, rng, p) and rng.random() < net["cvec"][t]:
                believe[t] = True; new.append((t, cred))
            else:
                evaluated[t] = rng.random() < 0.5
        queue = []
        for t, cred in new:
            if rng.random() < p["share_p"]:
                emit(cred, net["nbrs"][t], False)
    return believe, exp_bot, exp_hum


# ---------------- physical plane ----------------
_PF = None
def pf_table():
    """Transformer loading and min voltage as a function of added residential kW (proportional allocation)."""
    global _PF
    if _PF is None:
        f = os.path.join(OUT, "pf_table.npy")
        if os.path.exists(f):
            _PF = np.load(f)
        else:
            import pandapower as pp, pandapower.networks as pn
            net = pn.create_cigre_network_lv(); res = net.load[net.load.name.str.startswith("Load R")].index
            base = net.load.loc[res, "p_mw"].values.copy(); q0 = net.load.loc[res, "q_mvar"].values.copy(); sh = base / base.sum()
            rows = []
            for kw in np.arange(0, 1001, 10):
                n = copy.deepcopy(net); add = kw / 1000 * sh
                n.load.loc[res, "p_mw"] = base + add; n.load.loc[res, "q_mvar"] = q0 * (base + add) / base
                pp.runpp(n)
                rows.append([kw, n.res_trafo.loc[0, "loading_percent"], n.res_bus.vm_pu.min(), n.res_line.loading_percent.max()])
            _PF = np.array(rows); np.save(f, _PF)
    return _PF


def loading_of(kw):
    T = pf_table(); return np.interp(kw, T[:, 0], T[:, 1])


def vmin_of(kw):
    T = pf_table(); return np.interp(kw, T[:, 0], T[:, 2])


def feeder_households(net, rng, p):
    if p["mapping"] == "random":
        return rng.choice(p["N"], p["hh_feeder"], replace=False)
    # clustered: BFS neighbourhood (spatially correlated response)
    start = int(rng.integers(p["N"])); seen = [start]; S = {start}; i = 0
    while len(seen) < p["hh_feeder"] and i < len(seen):
        for v in net["nbrs"][seen[i]]:
            if v not in S:
                S.add(v); seen.append(v)
                if len(seen) >= p["hh_feeder"]: break
        i += 1
    return np.array(seen[:p["hh_feeder"]])


def thermal(load_pu, dt=1.0, amb=20.0, K0=None):
    """IEC 60076-7 difference-equation model, ONAN distribution transformer defaults.
    load_pu: array of per-unit load per minute. Returns hot-spot temperature series (degC)."""
    R, x, y, d_or, Hgr, k11, k21, k22, to, tw = 5.0, 0.8, 1.6, 60.0, 23.0, 1.0, 1.0, 2.0, 180.0, 4.0
    K0 = load_pu[0] if K0 is None else K0
    Do = d_or * ((1 + R * K0 ** 2) / (1 + R)) ** x
    Dh1 = k21 * Hgr * K0 ** y; Dh2 = (k21 - 1) * Hgr * K0 ** y
    out = []
    for K in load_pu:
        Do += dt / (k11 * to) * (d_or * ((1 + R * K ** 2) / (1 + R)) ** x - Do)
        Dh1 += dt / (k22 * tw) * (k21 * Hgr * K ** y - Dh1)
        Dh2 += dt / (to / k22) * ((k21 - 1) * Hgr * K ** y - Dh2)
        out.append(amb + Do + Dh1 - Dh2)
    return np.array(out)


def physical(believe, net, rng, p, horizon=240):
    """Time-resolved response: believers start flexible load at window start (t=0) for 60 min (other)
    or 60-150 min (EV). Base load constant at evening peak. Returns resilience metrics."""
    hh = feeder_households(net, rng, p)
    act = believe[hh]; ev = rng.random(len(hh)) < p["ev_share"]
    dur = np.where(ev, rng.integers(60, 151, len(hh)), 60)
    kw = np.where(ev, p["ev_kw"], p["other_kw"])
    t = np.arange(horizon)
    add = ((t[:, None] < dur[None, :]) * (act * kw)[None, :]).sum(1)
    L = loading_of(add); hs = thermal(L / 100.0, K0=float(loading_of(0.0)) / 100.0)
    over = L > 100
    rec = int(np.argmax(~over)) if over[0] else 0
    V = 2.0 ** ((hs - 98.0) / 6.0)
    return dict(x_feeder=act.mean(), peak_load=L.max(), min_over=int(over.sum()),
                over_kvah=float(np.clip(L - 100, 0, None).sum() / 100 * 500 / 60), recovery_min=rec,
                peak_hs=float(hs.max()), min_hs120=int((hs > 120).sum()), lol_min=float(V.sum() - horizon),
                vmin=float(vmin_of(add.max())))


# ---------------- experiments ----------------
def one_main(args):
    sigma, run, scen_list, regimes, p = args
    rng = np.random.default_rng(10_000 * run + int(sigma * 1000) + 7)
    net = build_network(sigma, rng, p)
    rows = []
    for scen in scen_list:
        seed = int(rng.integers(1e9)); pseed = int(rng.integers(1e9)); aseed = int(rng.integers(1e9))
        for reg in regimes:
            b, eb, eh = spread(net, reg, scen, False, sigma, seed, p) if net["nb"] > 0 else (np.zeros(p["N"], bool), 0, 0)
            ph = physical(b, net, np.random.default_rng(pseed), p)
            auth = spread(net, reg, scen, True, sigma, aseed, p)[0].mean() if scen == scen_list[0] else np.nan
            rows.append(dict(sigma=sigma, run=run, scenario=scen, regime=reg, fake=b.mean(), auth=auth,
                             exp_bot=eb, exp_hum=eh, **ph))
    return rows


def pool_map(f, jobs):
    from multiprocessing import Pool
    with Pool(2) as P:
        res = P.map(f, jobs, chunksize=1)
    return [r for rr in res for r in rr]


def main(R=50):
    sig = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40]
    jobs = [(s, r, ["S1", "S2"], REGIMES, PAR) for s in sig for r in range(R)]
    pd.DataFrame(pool_map(one_main, jobs)).to_csv(os.path.join(OUT, "main.csv"), index=False)


def pareto(R=30):
    jobs = []
    for s in [0.05, 0.20]:
        for reg, key, vals in [("D3", "rho", [0.55, 0.65, 0.75, 0.85, 0.9, 0.95, 0.98]),
                               ("D3B", "rho", [0.55, 0.65, 0.75, 0.85, 0.9, 0.95, 0.98]),
                               ("D3W", "tau", [0.0, 0.2, 0.35, 0.45, 0.5, 0.55, 0.65])]:
            for v in vals:
                p = dict(PAR, **{key: v})
                jobs += [(s, r, ["S1", "S2"], [reg], dict(p, _tag=f"{reg}:{key}={v}")) for r in range(R)]
        jobs += [(s, r, ["S1", "S2"], ["D2", "D2C", "D0"], dict(PAR, _tag="fixed")) for r in range(R)]
    rows = pool_map(one_main_tag, jobs)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "pareto.csv"), index=False)


def one_main_tag(args):
    rows = one_main(args)
    for r in rows:
        r["tag"] = args[4].get("_tag", ""); r["rho"] = args[4]["rho"]; r["tau"] = args[4]["tau"]
    return rows


def robust(R=30):
    jobs = []
    for name, upd in [("BA", {}), ("WS", {"graph": "WS"}), ("SBM", {"graph": "SBM"}), ("het_c", {"c_conc": 4.0}),
                      ("clustered", {"mapping": "clustered"}), ("EV60", {"ev_share": 0.6})]:
        jobs += [(0.2, r, ["S1", "S2"], ["D2", "D2C", "D3", "D3W"], dict(PAR, _tag=name, **upd)) for r in range(R)]
    pd.DataFrame(pool_map(one_main_tag, jobs)).to_csv(os.path.join(OUT, "robust.csv"), index=False)


def attack(R=30):
    jobs = []
    for psi in [0.0, 0.1, 0.25, 0.5, 0.75]:
        jobs += [(0.2, r, ["S2"], ["D2C", "D3", "D3B", "D3W"], dict(PAR, psi=psi, _tag=f"psi={psi}")) for r in range(R)]
    for eps in [0.05, 0.15, 0.3]:
        jobs += [(0.2, r, ["S2"], ["D2C", "D3", "D3B", "D3W"], dict(PAR, eps_con=eps, _tag=f"eps={eps}")) for r in range(R)]
    pd.DataFrame(pool_map(one_main_tag, jobs)).to_csv(os.path.join(OUT, "attack.csv"), index=False)


if __name__ == "__main__":
    pf_table()
    {"main": main, "pareto": pareto, "robust": robust, "attack": attack}[sys.argv[1]]()
