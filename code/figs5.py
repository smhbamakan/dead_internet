"""Figures for revision 5 (standard matplotlib output, serif font; concept figures monochrome)."""
import os, sys, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import figs_v3 as V   # rcParams, node/arr helpers
import abm5 as M
OUT = M.OUT; FIG = os.path.join(os.path.dirname(__file__), "..", "fig5"); os.makedirs(FIG, exist_ok=True)


def save(f, n):
    f.savefig(os.path.join(FIG, n + ".png"), bbox_inches="tight", dpi=600); f.savefig(os.path.join(FIG, n + ".pdf"), bbox_inches="tight"); plt.close(f)


def fig_chain():
    f, ax = plt.subplots(figsize=(7.0, 2.8)); ax.set_xlim(0, 15); ax.set_ylim(0, 5.3); ax.axis("off")
    st = [("Bot swarm", "A1 + A2, fake notice"), ("Diffusion", "threshold contagion"), ("Belief", "household decision"),
          ("Load surge", "EV, heat pump"), ("Thermal stress", "rating, hot-spot")]
    for i, (t, s) in enumerate(st):
        V.node(ax, 0.2 + 3.0 * i, 2.7, 2.5, 1.3, t, s)
        if i < 4: V.arr(ax, (2.7 + 3.0 * i, 3.35), (3.2 + 3.0 * i, 3.35))
    for i, t in [(1, "D1: removal"), (2, "D2–D4: credential"), (3, "consistency edge")]:
        x = 0.2 + 3.0 * i + 1.25
        ax.text(x, 0.75, t, ha="center", fontsize=8, bbox=dict(fc="white", ec="black", lw=0.7, ls=(0, (3, 2)), pad=3))
        V.arr(ax, (x, 1.1), (x, 2.68), ls=(0, (3, 2)))
    ax.text(7.45, 4.85, "A4 (S2, S3): signature from a compromised aggregator", fontsize=8, style="italic", ha="center")
    V.arr(ax, (7.45, 4.6), (7.45, 4.02))
    ax.text(12.0, 0.2, "A3 (S3): spoofed tariff signal", fontsize=8, style="italic", ha="center")
    save(f, "fig1_killchain")


def fig_threshold():
    sstar_u = lambda pi, rho, q: (1 - pi) * (1 - rho) / ((1 - pi) * (1 - rho) + rho * (1 - q))
    P = np.linspace(0, 0.99, 200)
    f, ax = plt.subplots(1, 3, figsize=(7.0, 2.4))
    for i, q in enumerate([0.0, 0.05, 0.2, 0.5]):
        ax[0].plot(P, [sstar_u(p, 0.75, q) for p in P], label=f"$q={q}$")
    ax[0].set_xlabel(r"provenance coverage $\pi$"); ax[0].set_ylabel(r"$\sigma^{*}_{u}$"); ax[0].legend(); ax[0].set_title("(a)", fontsize=9)
    Wt = np.load(os.path.join(OUT, "welfare.npy"))   # order: s in (.05,.15,.3) x q in (0,.1)
    lab = [(0.05, 0), (0.05, 0.1), (0.15, 0), (0.15, 0.1), (0.3, 0), (0.3, 0.1)]
    for k, (s, q) in enumerate(lab):
        ls = "-" if q == 0 else "--"; col = f"C{[0.05, 0.15, 0.3].index(s)}"
        ax[1].plot(np.linspace(0, 0.99, 100), Wt[k][:, 0], ls=ls, color=col, label=rf"$\sigma={s}$" if q == 0 else None)
        ax[2].plot(np.linspace(0, 0.99, 100), Wt[k][:, 1], ls=ls, color=col)
    ax[1].set_xlabel(r"$\pi$"); ax[1].set_ylabel("expected utility per message"); ax[1].legend(); ax[1].set_title("(b)", fontsize=9)
    ax[2].set_xlabel(r"$\pi$"); ax[2].set_ylabel("authentic messages rejected"); ax[2].set_title("(c)", fontsize=9)
    f.tight_layout(); save(f, "fig3_threshold_welfare")


def fig_collective():
    f, ax = plt.subplots(1, 2, figsize=(7.0, 2.8))
    for i, pi in enumerate([0.0, 0.6, 0.9]):
        A = np.load(os.path.join(OUT, f"bif_pi{pi}.npy"))
        st, un = A[A[:, 2] == 1], A[A[:, 2] == 0]
        ax[0].plot(st[:, 0], st[:, 1], ".", ms=1.5, color=f"C{i}", label=rf"$\pi={pi}$")
        ax[0].plot(un[:, 0], un[:, 1], ".", ms=1.0, color=f"C{i}", alpha=0.35)
    ax[0].set_xlabel(r"synthetic share $\sigma$"); ax[0].set_ylabel(r"equilibrium $y^{*}$"); ax[0].legend(markerscale=6)
    ax[0].set_title("(a) conditional-corroboration model", fontsize=9)
    H = json.load(open(os.path.join(OUT, "half5.json"))); S = np.array(H["S"])
    for name, ls in [("base", "-"), ("obs_m1", "--"), ("obs_m3", ":")]:
        for i, pi in enumerate([0.0, 0.6]):
            ax[1].plot(S, H["curves"][f"{name}|{pi}"], ls=ls, color=f"C{i}", label=f"{name}, $\\pi={pi}$")
    ax[1].set_xlabel(r"synthetic share $\sigma$"); ax[1].set_ylabel(r"$y^{*}$ (continued from $y=1$)"); ax[1].legend(fontsize=6)
    ax[1].set_title("(b) three corroboration models", fontsize=9)
    f.tight_layout(); save(f, "fig4_collective")


def series(reg, scen, sigma=0.2, R=12):
    Ls, Hs = [], []
    for r in range(R):
        rng = np.random.default_rng(500 + r); net = M.build_network(sigma, rng, M.PAR)
        b = M.spread(net, reg, scen, False, sigma, int(rng.integers(1e9)), M.PAR)[0]
        prng = np.random.default_rng(900 + r); hh = M.feeder_households(net, prng, M.PAR)
        act = b[hh]; ev = prng.random(len(hh)) < M.PAR["ev_share"]; dur = np.where(ev, prng.integers(60, 151, len(hh)), 60)
        kw = np.where(ev, M.PAR["ev_kw"], M.PAR["other_kw"]); t = np.arange(240)
        add = ((t[:, None] < dur[None, :]) * (act * kw)[None, :]).sum(1); L = M.loading_of(add)
        Ls.append(L); Hs.append(M.thermal(L / 100, K0=float(M.loading_of(0.0)) / 100))
    return np.mean(Ls, 0), np.mean(Hs, 0)


def fig_time():
    f, ax = plt.subplots(1, 2, figsize=(7.0, 2.7))
    for reg in ["D0", "D1", "D2", "D2C"]:
        L, H = series(reg, "S2")
        lab = reg if reg != "D2C" else "D2C = D3"
        ax[0].plot(np.arange(240), L, label=lab); ax[1].plot(np.arange(240), H, label=lab)
    ax[0].axhline(100, color="k", ls="--", lw=0.8); ax[1].axhline(120, color="k", ls="--", lw=0.8); ax[1].axhline(140, color="k", ls=":", lw=0.8)
    ax[0].set_xlabel("minutes after window start"); ax[0].set_ylabel("transformer loading (%)"); ax[0].legend(); ax[0].set_title(r"(a) S2, $\sigma=0.2$", fontsize=9)
    ax[1].set_xlabel("minutes after window start"); ax[1].set_ylabel("hot-spot temperature (°C)"); ax[1].set_title("(b) IEC 60076-7 thermal model", fontsize=9)
    f.tight_layout(); save(f, "fig5_time")


def fig_pareto():
    g = pd.read_csv(os.path.join(OUT, "pareto_summary.csv"))
    f, ax = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)
    for k, s in enumerate([0.05, 0.2]):
        x = g[g.sigma == s]
        for fam, mk in [("D3:", "D"), ("D3B:", "o"), ("D3W:", "s")]:
            y = x[x.key.str.startswith(fam)]
            ax[k].plot(y.fake, y.FR, marker=mk, ms=4, lw=1, label=fam.rstrip(":") + (r" ($\rho$ sweep)" if fam != "D3W:" else r" ($\tau$ sweep)"))
        for key, mk in [("D2", "^"), ("D2C", "v"), ("D0", "x")]:
            y = x[x.key == key]; ax[k].plot(y.fake, y.FR, mk, ms=7, color="k", mfc="none" if key != "D0" else None, label=key)
        ax[k].axvline(0.078, color="0.5", ls=":", lw=0.8)
        ax[k].set_xlabel("fake adoption (mean of S1, S2)"); ax[k].set_title(rf"({'ab'[k]}) $\sigma={s}$", fontsize=9)
    ax[0].set_ylabel("false rejection of authentic notice"); ax[1].legend(fontsize=6.5)
    f.tight_layout(); save(f, "fig6_pareto")


def fig_attack():
    a = pd.read_csv(os.path.join(OUT, "attack.csv"))
    f, ax = plt.subplots(1, 2, figsize=(7.0, 2.6))
    ps = a[a.tag.str.startswith("psi")].copy(); ps["x"] = ps.tag.str.split("=").str[1].astype(float)
    for reg, mk in [("D2C", "v"), ("D3", "D"), ("D3B", "o"), ("D3W", "s")]:
        g = ps[ps.regime == reg].groupby("x").fake.mean(); ax[0].plot(g.index, g.values, marker=mk, ms=4, label=reg)
    ax[0].axhline(0.078, color="k", ls="--", lw=0.8)
    ax[0].set_xlabel(r"spoofing probability $\psi$"); ax[0].set_ylabel("fake adoption (S3)"); ax[0].legend(); ax[0].set_title("(a) spoofed consistency channel", fontsize=9)
    ep = a[a.tag.str.startswith("eps") | (a.tag == "psi=0.0")].copy(); ep["x"] = np.where(ep.tag == "psi=0.0", 0.0, ep.tag.str.split("=").str[1].astype(float))
    for reg, mk in [("D2C", "v"), ("D3", "D"), ("D3B", "o"), ("D3W", "s")]:
        g = ep[ep.regime == reg].groupby("x").auth.mean(); ax[1].plot(g.index, g.values, marker=mk, ms=4, label=reg)
    ax[1].set_xlabel(r"false-contradiction rate $\varepsilon$"); ax[1].set_ylabel("authentic-notice uptake"); ax[1].set_title("(b) noisy consistency channel", fontsize=9)
    f.tight_layout(); save(f, "fig7_attack")


def fig_sobol():
    S = json.load(open(os.path.join(OUT, "sobol5.json")))
    names = ["c", "s", "kappa", "rho", "pi_ecosys", "bot_followers", "sig_hat_mult"]
    tex = [r"$c$", r"$s$", r"$\kappa$", r"$\rho$", r"$\pi$", r"$k_b$", r"$\hat\sigma/\sigma$"]
    f, ax = plt.subplots(1, 3, figsize=(7.0, 2.6))
    Ns = sorted(int(k) for k in S["fake_S2"] if k.isdigit())
    for i, n in enumerate(names):
        ax[0].errorbar(Ns, [S["fake_S2"][str(N)][n]["ST"] for N in Ns], yerr=[S["fake_S2"][str(N)][n]["STc"] for N in Ns], marker="o", ms=3, capsize=2, label=tex[i])
    ax[0].set_xscale("log", base=2); ax[0].set_xlabel("base samples $N$"); ax[0].set_ylabel("$S_T$, fake (S2)"); ax[0].legend(fontsize=6, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.28)); ax[0].set_title("(a) convergence", fontsize=9)
    N = str(Ns[-1]); y = np.arange(len(names))
    for j, out in enumerate(["fake_S2", "authentic"]):
        s1 = [S[out][N][n]["S1"] for n in names]; st = [S[out][N][n]["ST"] for n in names]
        e1 = [S[out][N][n]["S1c"] for n in names]; et = [S[out][N][n]["STc"] for n in names]
        ax[j + 1].barh(y + 0.2, s1, 0.4, xerr=e1, capsize=2, label="$S_1$"); ax[j + 1].barh(y - 0.2, st, 0.4, xerr=et, capsize=2, label="$S_T$")
        ax[j + 1].set_yticks(y); ax[j + 1].set_yticklabels(tex); ax[j + 1].axvline(0, color="k", lw=0.5)
        ax[j + 1].set_title(["(b) fake adoption (S2)", "(c) authentic uptake"][j], fontsize=9); ax[j + 1].set_xlabel("index")
    ax[1].legend(fontsize=6.5); f.tight_layout(); save(f, "fig8_sobol")


if __name__ == "__main__":
    which = sys.argv[1:] or ["chain", "threshold", "collective", "time", "pareto", "attack", "sobol"]
    for w in which:
        globals()["fig_" + w](); print(w)


def fig_s1():
    import matplotlib.patches as mp
    f, ax = plt.subplots(figsize=(7.0, 3.2)); ax.set_xlim(0, 14); ax.set_ylim(0, 6.4); ax.axis("off")
    P = {"TPM": (0.3, 4.6), "meter": (0.3, 2.6), "reading": (3.3, 2.6), "agg": (3.3, 4.8), "A": (6.6, 5.2), "B": (6.6, 3.4),
         "notice": (10.0, 3.6), "util": (3.3, 0.6)}
    L = {"TPM": ("TPM", "device state"), "meter": ("Meter", "data integrity"), "reading": ("Tariff reading", "content truth"),
         "agg": ("Aggregator VC", "authorised"), "A": ("Agent A", "content truth"), "B": ("Agent B", "content truth"),
         "notice": ("Notice", "content truth"), "util": ("Utility DID", "authorised (auth. case)")}
    W, H = 2.6, 1.1
    for k, (x, y) in P.items():
        V.node(ax, x, y, W, H, *L[k])
    c = lambda k, dx, dy: (P[k][0] + W * dx, P[k][1] + H * dy)
    V.arr(ax, c("TPM", .5, 0), c("meter", .5, 1)); V.lab(ax, 1.6, 4.15, "ATT")
    V.arr(ax, c("meter", 1, .5), c("reading", 0, .5)); V.lab(ax, 3.1, 3.4, "DEL")
    V.arr(ax, c("agg", 1, .7), c("A", 0, .5)); V.arr(ax, c("agg", 1, .3), c("B", 0, .5)); V.lab(ax, 6.2, 4.75, "DEL")
    V.arr(ax, c("A", .4, 0), c("B", .4, 1)); V.arr(ax, c("B", .6, 1), c("A", .6, 0)); V.lab(ax, 8.05, 4.85, "cycle (ignored)", ha="left")
    V.arr(ax, c("A", 1, .5), c("notice", 0, .8)); V.arr(ax, c("B", 1, .5), c("notice", 0, .5)); V.lab(ax, 9.6, 5.15, "CC")
    V.arr(ax, c("reading", 1, .3), c("notice", 0, .2), ls=(0, (5, 2))); V.lab(ax, 7.6, 2.55, "CON")
    V.arr(ax, c("util", 1, .5), c("B", .5, 0), ls=(0, (1, 2))); V.lab(ax, 7.4, 0.9, "DEL (authentic case)", ha="left")
    ax.text(10.0, 1.2, "fake: $P=0.498$, review flag\nauthentic: $P=0.975$", fontsize=8)
    save(f, "figS1_worked_graph")
