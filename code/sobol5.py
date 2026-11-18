"""Sobol sensitivity for D3 at sigma=0.2 (revision 5): N=512 base samples (4,608 model evaluations),
convergence from nested prefixes N'=64,128,256,512. Writes ../out5/sobol5.json"""
import os, sys, json
import numpy as np
from SALib.sample import sobol as ss
from SALib.analyze import sobol as sa
sys.path.insert(0, os.path.dirname(__file__))
import abm5 as M

PROB = {"num_vars": 7, "names": ["c", "s", "kappa", "rho", "pi_ecosys", "bot_followers", "sig_hat_mult"],
        "bounds": [[0.15, 0.35], [0.2, 0.8], [0.3, 0.9], [0.6, 0.9], [0.3, 0.9], [20, 60], [0.5, 2.0]]}


def ev(args):
    i, x = args
    p = dict(M.PAR); p.update({k: v for k, v in zip(PROB["names"], x)}); p["bot_followers"] = int(round(p["bot_followers"]))
    rng = np.random.default_rng(1000 + i)
    net = M.build_network(0.2, rng, p)
    f = M.spread(net, "D3", "S2", False, 0.2, int(rng.integers(1e9)), p)[0].mean()
    a = M.spread(net, "D3", "S1", True, 0.2, int(rng.integers(1e9)), p)[0].mean()
    return [f, a]


if __name__ == "__main__":
    from multiprocessing import Pool
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 512
    X = ss.sample(PROB, N, calc_second_order=True, seed=11)
    with Pool(2) as P:
        Y = np.array(P.map(ev, list(enumerate(X)), chunksize=8))
    np.save(os.path.join(M.OUT, "sobol5_XY.npy"), np.hstack([X, Y]))
    D = PROB["num_vars"]; blk = 2 * D + 2; out = {}
    for n in [64, 128, 256, N]:
        rows = n * blk
        for j, name in enumerate(["fake_S2", "authentic"]):
            r = sa.analyze(PROB, Y[:rows, j], calc_second_order=True, seed=5)
            out.setdefault(name, {})[n] = {k: dict(S1=float(r["S1"][i]), S1c=float(r["S1_conf"][i]), ST=float(r["ST"][i]), STc=float(r["ST_conf"][i]))
                                           for i, k in enumerate(PROB["names"])}
            if n == N:
                S2 = r["S2"]; pairs = []
                for a in range(D):
                    for b in range(a + 1, D):
                        pairs.append((PROB["names"][a], PROB["names"][b], float(S2[a, b])))
                out[name]["S2_top"] = sorted(pairs, key=lambda t: -abs(t[2]))[:5]
                out[name]["mean"] = float(Y[:, j].mean()); out[name]["p5_50_95"] = np.percentile(Y[:, j], [5, 50, 95]).tolist()
                if j == 0:
                    out[name]["share_below_xstar"] = float((Y[:, 0] < 0.078).mean())
    json.dump(out, open(os.path.join(M.OUT, "sobol5.json"), "w"), indent=1)
    print("done")
