"""Analyses added in revision 5. Writes ../out5/theory5.json and arrays for figures."""
import os, sys, json, time
import numpy as np
from scipy.stats import beta as B, binom
sys.path.insert(0, os.path.dirname(__file__))
import sl, cpag
OUT = os.path.join(os.path.dirname(__file__), "..", "out5"); os.makedirs(OUT, exist_ok=True)
R = {}

# ---------- (a) generalised posteriors with synthetic-credential probability q ----------
def a_uns(s, pi, q): n = (1 - s) * (1 - pi); return n / (n + s * (1 - q))
def a_sig(s, pi, q): n = (1 - s) * pi; return n / (n + s * q) if n + s * q > 0 else 1.0
def sstar_u(pi, rho, q): n = (1 - pi) * (1 - rho); return n / (n + rho * (1 - q))
def sstar_s(pi, rho, q): n = pi * (1 - rho); return n / (n + rho * q) if q > 0 else 1.0
R["sstar_q"] = {f"q={q}": {f"pi={pi}": [round(sstar_u(pi, 0.75, q), 4), round(sstar_s(pi, 0.75, q), 4)]
                for pi in [0.3, 0.6, 0.9]} for q in [0.0, 0.05, 0.2, 0.5]}

# ---------- (b) aggregate welfare with a full utility matrix ----------
def welfare(s, pi, q, rho, G=1.0, M=0.5, L=4.0):
    """Expected utility per message. Accept iff posterior >= rho. Utilities: act&auth=G, not&auth=-M,
    act&syn=-L, not&syn=0. Also returns false-rejection of authentic unsigned and false acceptance."""
    pa, ps = a_uns(s, pi, q), a_sig(s, pi, q)
    acc_u, acc_s = pa >= rho, ps >= rho
    A_s, A_u = (1 - s) * pi, (1 - s) * (1 - pi)           # authentic signed / unsigned mass
    S_s, S_u = s * q, s * (1 - q)                          # synthetic signed / unsigned mass
    U = (A_s * (G if acc_s else -M) + A_u * (G if acc_u else -M) - L * (S_s * acc_s + S_u * acc_u))
    fr_auth = (A_u * (not acc_u) + A_s * (not acc_s)) / (1 - s)
    fa = (S_s * acc_s + S_u * acc_u) / s if s > 0 else 0
    return U, fr_auth, fa
rho_star = 4.0 / (1.0 + 0.5 + 4.0)
PI = np.linspace(0, 0.99, 100)
W = {}
for s in [0.05, 0.15, 0.3]:
    for q in [0.0, 0.1]:
        W[f"s={s},q={q}"] = np.array([welfare(s, p, q, rho_star) for p in PI]).tolist()
np.save(os.path.join(OUT, "welfare.npy"), np.array([[welfare(s, p, q, rho_star) for p in PI] for s in [0.05, 0.15, 0.3] for q in [0.0, 0.1]]))
R["rho_full_utility"] = round(rho_star, 4)

# ---------- (c) projected probability along delegation chains ----------
def P_chain(k, bbar, bx, dx, a):
    Bk = bbar ** k
    return a + Bk * (bx - a * (bx + dx))
def kmax(bbar, bx, dx, a, rho):
    if a >= rho:
        return np.inf
    num = (rho - a) / (bx - a * (bx + dx))
    return np.floor(np.log(num) / np.log(bbar)) if 0 < num < 1 else (0 if num >= 1 else np.inf)
R["kmax"] = {f"a={a}": {f"bbar={bb}": (None if np.isinf(kmax(bb, .95, 0, a, .75)) else int(kmax(bb, .95, 0, a, .75)))
             for bb in [0.8, 0.9, 0.95]} for a in [0.3, 0.5, 0.615, 0.7, 0.8]}

# ---------- (d) fixed points, folds and stability of the collective map ----------
FA, FB = 22.5, 7.5
_g = np.linspace(0, 1, 20001); _F = B.cdf(_g, FA, FB); _f = B.pdf(_g, FA, FB)
F = lambda x: np.interp(x, _g, _F); f = lambda x: np.interp(x, _g, _f)

def Phi(y, s, pi, kappa, eta, model="base", m=3, alpha=0.9, beta_=0.1, gamma=0.0):
    a = a_uns(s, pi, 0.0)
    if model == "base":
        pc = a * y / (a * y + (1 - a) * s) if a * y + (1 - a) * s > 0 else 1.0
        return eta + (1 - eta) * (kappa * F(pc) + (1 - kappa) * F(a))
    # observation process: m queried peers; honest trusting peers confirm authentic w.p. alpha and
    # synthetic w.p. beta_; distrusting honest peers never confirm; bots confirm synthetic always and authentic w.p. gamma
    ca = (1 - s) * y * alpha + s * gamma; cs = (1 - s) * y * beta_ + s
    tot = 0.0
    for k in range(m + 1):
        la, ls = binom.pmf(k, m, ca), binom.pmf(k, m, cs)
        post = a * la / (a * la + (1 - a) * ls) if a * la + (1 - a) * ls > 0 else a
        tot += la * F(post)
    return eta + (1 - eta) * (kappa * tot + (1 - kappa) * F(a))

def branches(s, pi, kappa=0.6, eta=0.05, n=2001, **kw):
    ys = np.linspace(0, 1, n); g = np.array([Phi(y, s, pi, kappa, eta, **kw) for y in ys]) - ys
    roots = []
    for i in range(n - 1):
        if g[i] == 0 or g[i] * g[i + 1] < 0:
            lo, hi = ys[i], ys[i + 1]
            for _ in range(50):
                mid = (lo + hi) / 2
                if (Phi(lo, s, pi, kappa, eta, **kw) - lo) * (Phi(mid, s, pi, kappa, eta, **kw) - mid) <= 0: hi = mid
                else: lo = mid
            y0 = (lo + hi) / 2; h = 1e-5
            d = (Phi(y0 + h, s, pi, kappa, eta, **kw) - Phi(y0 - h, s, pi, kappa, eta, **kw)) / (2 * h)
            roots.append((y0, d < 1))
    return roots

def folds(pi, model="base", kappa=0.6, eta=0.05, nS=300, ny=801, **kw):
    S = np.linspace(0.005, 0.5, nS); nroot = []; diag = []
    for s in S:
        r = branches(s, pi, kappa, eta, n=ny, model=model, **kw); nroot.append(len(r)); diag.append(r)
    nroot = np.array(nroot); idx = np.where(np.diff(nroot) != 0)[0]
    fp = [float((S[i] + S[i + 1]) / 2) for i in idx]
    return S, diag, fp

CACHE = os.path.join(OUT, "folds_cache.json")
_cache = json.load(open(CACHE)) if os.path.exists(CACHE) else None
BIF = {}
for pi in ([] if _cache else [0.0, 0.3, 0.6, 0.9]):
    S, diag, fp = folds(pi)
    BIF[pi] = fp
    np.save(os.path.join(OUT, f"bif_pi{pi}.npy"), np.array([(s, y, st) for s, d in zip(S, diag) for y, st in d]))
R["folds_base"] = _cache["base"] if _cache else {str(k): [round(x, 4) for x in v] for k, v in BIF.items()}
ALT = _cache["alt"] if _cache else {}
for name, kw in ([] if _cache else [("obs_m1_beta0", dict(model="obs", m=1, alpha=1.0, beta_=0.0)),
                 ("obs_m3_beta0.1", dict(model="obs", m=3, alpha=0.9, beta_=0.1)),
                 ("obs_m3_beta0.1_gamma0.3", dict(model="obs", m=3, alpha=0.9, beta_=0.1, gamma=0.3))]):
    ALT[name] = {str(pi): [round(x, 4) for x in folds(pi, nS=120, ny=301, **kw)[2]] for pi in [0.0, 0.6, 0.9]}
R["folds_alt"] = ALT
json.dump({"base": R["folds_base"], "alt": ALT}, open(CACHE, "w"))
# Proposition 4 check (kappa=1, eta=0)
R["prop4_check"] = {str(s): [(round(y, 4), bool(st)) for y, st in branches(s, 0.6, 1.0, 0.0)] for s in [0.05, 0.15, 0.3]}

# ---------- (e) dependence-aware fusion vs naive (Sybil replication) ----------
def sybil(k, honest_contra=True):
    nodes = {"agg": dict(prop="authorised", prior=sl.op(0.8, 0, 0.2, 0.5)),
             "meter": dict(prop="data_integrity", prior=sl.op(0.95, 0, 0.05, 0.5)),
             "reading": dict(prop="content_truth", claim=sl.op(0, 0.9, 0.1, 0.615) if honest_contra else sl.op(0.9, 0, 0.1, 0.615), a=0.615),
             "notice": dict(prop="content_truth", a=0.615)}
    edges = [("meter", "reading", "DEL"), ("reading", "notice", "CON")]
    for i in range(k):
        nodes[f"relay{i}"] = dict(prop="content_truth", a=0.615)
        edges += [("agg", f"relay{i}", "DEL"), (f"relay{i}", "notice", "CC")]
    op, fl = cpag.evaluate(nodes, edges, theta_c=0.5)
    nv = cpag.naive_evaluate(nodes, edges)
    return sl.P(op["notice"]), sl.P(nv["notice"]), fl["notice"]
R["sybil"] = {k: [round(x, 3) if not isinstance(x, bool) else x for x in sybil(k)] for k in [1, 2, 3, 5, 10, 20]}

# ---------- worked non-trivial graph ----------
def worked(fake=True):
    a = 0.615
    nodes = {"utility_DID": dict(prop="authorised", prior=sl.op(0.95, 0, 0.05)),
             "aggregator_VC": dict(prop="authorised", prior=sl.op(0.80, 0, 0.20)),
             "TPM": dict(prop="device_state", prior=sl.op(0.9, 0, 0.1)),
             "meter": dict(prop="data_integrity"),
             "tariff_reading": dict(prop="content_truth", a=a, claim=sl.op(0, 0.9, 0.1, a) if fake else sl.op(0.9, 0, 0.1, a)),
             "agentA": dict(prop="content_truth", a=a), "agentB": dict(prop="content_truth", a=a),
             "notice": dict(prop="content_truth", a=a)}
    edges = [("TPM", "meter", "ATT"), ("meter", "tariff_reading", "DEL"), ("aggregator_VC", "agentA", "DEL"),
             ("aggregator_VC", "agentB", "DEL"), ("agentA", "agentB", "DEL"), ("agentB", "agentA", "DEL"),
             ("agentA", "notice", "CC"), ("agentB", "notice", "CC"), ("tariff_reading", "notice", "CON")]
    if not fake:
        edges = [e for e in edges if e[0] != "aggregator_VC"] + [("utility_DID", "agentB", "DEL")]
    op, fl = cpag.evaluate(nodes, edges, theta_c=0.5)
    return {k: [round(float(x), 3) for x in v[:3]] + [round(float(sl.P(v)), 3), bool(fl[k])] for k, v in op.items()}
R["worked_fake"] = worked(True); R["worked_auth"] = worked(False)

# ---------- (g) Table 4 trace with evidence ----------
def ev(w):
    r, s_ = sl.to_evidence(w) if w[2] > 0 else (np.inf, np.inf); return [round(float(r), 2), round(float(s_), 2)]
a = 0.615; trace = {}
creds = {"utility": sl.discount(sl.op(.95, 0, .05), sl.op(1, 0, 0, a)), "aggregator": sl.discount(sl.op(.8, 0, .2), sl.op(1, 0, 0, a)), "none": sl.vacuous(a)}
cons = {"consistent": sl.op(.9, 0, .1, a), "contradicts": sl.op(0, .9, .1, a), "unavailable": sl.vacuous(a)}
for cn, wc in creds.items():
    wc[3] = a
    for kn, wk in cons.items():
        fz = sl.fuse(wc, wk); fz[3] = a
        trace[f"{cn}|{kn}"] = dict(cred=[round(x, 3) for x in wc[:3]], cred_ev=ev(wc), con=[round(x, 3) for x in wk[:3]], con_ev=ev(wk),
                                   fused=[round(x, 3) for x in fz[:3]], fused_ev=ev(fz), P=round(float(sl.P(fz)), 3),
                                   DC=round(float(sl.conflict(wc, wk)), 3))
R["table4_trace"] = trace
# conflict threshold: review rates
R["theta_c"] = {str(th): dict(auth_review_if_contra={k: trace[f"{k}|contradicts"]["DC"] > th for k in ["utility", "none"]},
                              fake_review_if_contra=trace["aggregator|contradicts"]["DC"] > th) for th in [0.3, 0.5, 0.7, 0.8]}

# ---------- (i) overhead benchmark ----------
try:
    from cryptography.hazmat.primitives.asymmetric import ed25519, ec
    from cryptography.hazmat.primitives import hashes
    msg = os.urandom(512)
    sk = ed25519.Ed25519PrivateKey.generate(); pk = sk.public_key(); sig = sk.sign(msg)
    t = time.perf_counter(); n = 3000
    for _ in range(n): pk.verify(sig, msg)
    ed = (time.perf_counter() - t) / n * 1e6
    sk2 = ec.generate_private_key(ec.SECP256R1()); pk2 = sk2.public_key(); sig2 = sk2.sign(msg, ec.ECDSA(hashes.SHA256()))
    t = time.perf_counter()
    for _ in range(n): pk2.verify(sig2, msg, ec.ECDSA(hashes.SHA256()))
    p256 = (time.perf_counter() - t) / n * 1e6
    t = time.perf_counter()
    for _ in range(20000): sl.fuse(sl.op(.9, 0, .1, .6), sl.op(.8, .1, .1, .6))
    fu = (time.perf_counter() - t) / 20000 * 1e6
    R["overhead_us"] = dict(ed25519_verify=round(ed, 1), ecdsa_p256_verify=round(p256, 1), sl_fuse_python=round(fu, 2))
except Exception as e:
    R["overhead_us"] = str(e)

json.dump(R, open(os.path.join(OUT, "theory5.json"), "w"), indent=1, default=str)
print(json.dumps({k: R[k] for k in ["sstar_q", "kmax", "folds_base", "folds_alt", "prop4_check", "sybil", "worked_fake", "worked_auth", "overhead_us", "theta_c"]}, indent=1, default=str))
