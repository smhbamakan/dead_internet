"""Figures in Elsevier artwork format, generated from the original outputs in out5/.

Sizes follow Elsevier guidance: full width 190 mm, 1.5 column 140 mm. Text is Times New Roman
(Liberation Serif, which has the same metrics, is used where Times New Roman is not installed), 8-9 pt at
final size. Vector PDF (fonts embedded) and 600 dpi TIFF/PNG are written to fig_elsevier/.
Multi-panel result figures are stacked vertically so that every panel is printed large.

Usage: python3 -I code/figs_elsevier.py [names...]
"""
import os, sys, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import figs_v3 as V
import figs5 as F
import abm5 as M

MM = 1 / 25.4
FULL, ONEHALF = 190 * MM, 140 * MM
OUT = M.OUT
FIG = os.path.join(os.path.dirname(__file__), "..", "fig_elsevier"); os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "TeX Gyre Termes", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "legend.frameon": True,
    "axes.linewidth": 0.6, "lines.linewidth": 1.2, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 600})


def save(f, n):
    for ext in ("pdf", "png", "tiff"):
        kw = {"pil_kwargs": {"compression": "tiff_lzw"}} if ext == "tiff" else {}
        f.savefig(os.path.join(FIG, f"{n}.{ext}"), bbox_inches="tight", pad_inches=0.02, **kw)
    plt.close(f)


def title(ax, t):
    ax.set_title(t, loc="left", fontsize=9)


# ---------------- concept figures (redrawn at full width) ----------------
def _concept(fn, name, w, h):
    orig_subplots, orig_save = plt.subplots, F.save
    plt.subplots = lambda *a, figsize=None, **k: orig_subplots(*a, figsize=(w, h), **k)
    F.save = V.save = lambda f, n: save(f, name)
    try:
        fn()
    finally:
        plt.subplots, F.save, V.save = orig_subplots, orig_save, orig_save


def fig_killchain(): _concept(F.fig_chain, "fig1_killchain", FULL, 2.9)
def fig_cpag(): _concept(V.fig_cpag, "fig2_cpag", FULL, 4.1)
def fig_worked():
    """Worked example of Section 4.2 (cycle between two agents); layout redrawn for print."""
    f, ax = plt.subplots(figsize=(FULL, 3.4)); ax.set_xlim(0, 15.2); ax.set_ylim(0, 6.6); ax.axis("off")
    W, H = 2.6, 1.1
    P = {"TPM": (0.2, 5.0), "meter": (0.2, 2.8), "reading": (3.7, 2.8), "agg": (3.7, 5.0),
         "A": (7.6, 5.0), "B": (7.6, 2.8), "notice": (12.2, 3.9), "util": (3.7, 0.4)}
    L = {"TPM": ("TPM", "device state"), "meter": ("Meter", "data integrity"), "reading": ("Tariff reading", "content truth"),
         "agg": ("Aggregator VC", "authorised"), "A": ("Agent A", "content truth"), "B": ("Agent B", "content truth"),
         "notice": ("Notice", "content truth"), "util": ("Utility DID", "authorised")}
    for k, (x, y) in P.items(): V.node(ax, x, y, W, H, *L[k])
    c = lambda k, dx, dy: (P[k][0] + W * dx, P[k][1] + H * dy)
    V.arr(ax, c("TPM", .5, 0), c("meter", .5, 1)); V.lab(ax, 1.5, 4.45, "ATT")
    V.arr(ax, c("meter", 1, .5), c("reading", 0, .5)); V.lab(ax, 3.25, 3.65, "DEL")
    V.arr(ax, c("agg", 1, .5), c("A", 0, .5)); V.arr(ax, c("agg", 1, .2), c("B", 0, .8)); V.lab(ax, 6.9, 5.85, "DEL")
    V.arr(ax, c("A", .35, 0), c("B", .35, 1)); V.arr(ax, c("B", .65, 1), c("A", .65, 0))
    V.lab(ax, 9.4, 4.45, "cycle\n(ignored)", ha="left")
    V.arr(ax, c("A", 1, .5), c("notice", 0, .8)); V.arr(ax, c("B", 1, .5), c("notice", 0, .2)); V.lab(ax, 11.3, 5.25, "CC")
    V.arr(ax, c("reading", 1, .2), (P["notice"][0] + 0.9, P["notice"][1]), ls=(0, (5, 2)), rad=0.25); V.lab(ax, 10.6, 2.35, "CON")
    V.arr(ax, c("util", 1, .5), c("B", .5, 0), ls=(0, (1, 2))); V.lab(ax, 7.75, 1.45, "DEL (authentic case only)", ha="left")
    ax.text(12.2, 2.9, "fake: $P=0.498$, review flag\nauthentic: $P=0.975$", fontsize=8, va="top")
    save(f, "fig3_worked_graph")


# ---------------- analytical figures ----------------
def fig_threshold():
    su = lambda pi, rho, q: (1 - pi) * (1 - rho) / ((1 - pi) * (1 - rho) + rho * (1 - q))
    P = np.linspace(0, 0.99, 200); Pw = np.linspace(0, 0.99, 100)
    f, ax = plt.subplots(3, 1, figsize=(ONEHALF, 6.6), sharex=True)
    for q in [0.0, 0.05, 0.2, 0.5]:
        ax[0].plot(P, su(P, 0.75, q), label=f"$q={q}$")
    ax[0].set_ylabel(r"$\sigma^{*}_{u}$"); ax[0].legend(); title(ax[0], r"(a) rejection threshold for unsigned notices ($\rho=0.75$)")
    Wt = np.load(os.path.join(OUT, "welfare.npy"))
    lab = [(0.05, 0), (0.05, 0.1), (0.15, 0), (0.15, 0.1), (0.3, 0), (0.3, 0.1)]
    for k, (s, q) in enumerate(lab):
        ls = "-" if q == 0 else "--"; col = f"C{[0.05, 0.15, 0.3].index(s)}"
        ax[1].plot(Pw, Wt[k][:, 0], ls=ls, color=col, label=rf"$\sigma={s}$" if q == 0 else None)
        ax[2].plot(Pw, Wt[k][:, 1], ls=ls, color=col, label=rf"$\sigma={s}$" if q == 0 else None)
    ax[1].plot([], [], "k-", label="$q=0$"); ax[1].plot([], [], "k--", label="$q=0.1$")
    ax[1].set_ylabel("expected utility per message"); ax[1].legend(ncol=2, loc="lower right"); title(ax[1], "(b) expected utility")
    ax[2].set_ylabel("authentic messages rejected"); ax[2].set_xlabel(r"provenance coverage $\pi$"); title(ax[2], "(c) rejected authentic messages")
    f.tight_layout(); save(f, "fig4_threshold_welfare")


def fig_collective():
    f, ax = plt.subplots(2, 1, figsize=(ONEHALF, 4.6), sharex=True)
    for i, pi in enumerate([0.0, 0.6, 0.9]):
        A = np.load(os.path.join(OUT, f"bif_pi{pi}.npy"))
        st, un = A[A[:, 2] == 1], A[A[:, 2] == 0]
        ax[0].plot(st[:, 0], st[:, 1], ".", ms=2.0, color=f"C{i}", label=rf"$\pi={pi}$")
        ax[0].plot(un[:, 0], un[:, 1], ".", ms=1.2, color=f"C{i}", alpha=0.35)
    ax[0].set_ylabel(r"equilibrium $y^{*}$"); ax[0].legend(markerscale=5); title(ax[0], "(a) conditional-corroboration model")
    H = json.load(open(os.path.join(OUT, "half5.json"))); S = np.array(H["S"])
    names = {"base": "conditional", "obs_m1": "observation, $m=1$", "obs_m3": r"observation, $m=3$"}
    for name, ls in [("base", "-"), ("obs_m1", "--"), ("obs_m3", ":")]:
        for i, pi in enumerate([0.0, 0.6]):
            ax[1].plot(S, H["curves"][f"{name}|{pi}"], ls=ls, color=f"C{i}", label=f"{names[name]}, $\\pi={pi}$")
    ax[1].set_xlabel(r"synthetic share $\sigma$"); ax[1].set_ylabel(r"$y^{*}$ (continued from $y=1$)")
    ax[1].legend(fontsize=7, ncol=2); title(ax[1], "(b) three corroboration models")
    f.tight_layout(); save(f, "fig5_collective")


# ---------------- simulation figures ----------------
def fig_time():
    f, ax = plt.subplots(2, 1, figsize=(ONEHALF, 4.8), sharex=True)
    for reg in ["D0", "D1", "D2", "D2C"]:
        L, Hs = F.series(reg, "S2")
        lab = reg if reg != "D2C" else "D2C = D3 = D3B = D3W"
        ax[0].plot(np.arange(240), L, label=lab); ax[1].plot(np.arange(240), Hs, label=lab)
    ax[0].axhline(100, color="k", ls="--", lw=0.7); ax[1].axhline(120, color="k", ls="--", lw=0.7); ax[1].axhline(140, color="k", ls=":", lw=0.7)
    ax[0].set_ylabel("transformer loading (%)"); ax[0].legend(); title(ax[0], r"(a) transformer loading, S2, $\sigma=0.2$")
    ax[1].set_xlabel("minutes after window start"); ax[1].set_ylabel("hot-spot temperature (°C)"); title(ax[1], "(b) hot-spot temperature (IEC 60076-7)")
    f.tight_layout(); save(f, "fig6_time")


def fig_pareto():
    g = pd.read_csv(os.path.join(OUT, "pareto_summary.csv"))
    f, ax = plt.subplots(2, 1, figsize=(ONEHALF, 5.0), sharex=True)
    for k, s in enumerate([0.05, 0.2]):
        x = g[g.sigma == s]
        for fam, mk in [("D3:", "D"), ("D3B:", "o"), ("D3W:", "s")]:
            y = x[x.key.str.startswith(fam)]
            ax[k].plot(y.fake, y.FR, marker=mk, ms=4, lw=1, label=fam.rstrip(":") + (r" ($\rho$ sweep)" if fam != "D3W:" else r" ($\tau$ sweep)"))
        for key, mk in [("D2", "^"), ("D2C", "v"), ("D0", "x")]:
            y = x[x.key == key]; ax[k].plot(y.fake, y.FR, mk, ms=7, color="k", mfc="none" if key != "D0" else None, label=key)
        ax[k].axvline(0.078, color="0.5", ls=":", lw=0.8)
        ax[k].set_ylabel("false rejection of authentic notice"); title(ax[k], rf"({'ab'[k]}) $\sigma={s}$")
    ax[1].set_xlabel("fake adoption (mean of S1, S2)"); ax[0].legend(ncol=2)
    f.tight_layout(); save(f, "fig7_pareto")


def fig_attack():
    a = pd.read_csv(os.path.join(OUT, "attack.csv"))
    f, ax = plt.subplots(2, 1, figsize=(ONEHALF, 4.8))
    ps = a[a.tag.str.startswith("psi")].copy(); ps["x"] = ps.tag.str.split("=").str[1].astype(float)
    ep = a[a.tag.str.startswith("eps") | (a.tag == "psi=0.0")].copy()
    ep["x"] = np.where(ep.tag == "psi=0.0", 0.0, ep.tag.str.split("=").str[1].astype(float))
    for reg, mk in [("D2C", "v"), ("D3", "D"), ("D3B", "o"), ("D3W", "s")]:
        g = ps[ps.regime == reg].groupby("x").fake.mean(); ax[0].plot(g.index, g.values, marker=mk, ms=4, label=reg)
        g = ep[ep.regime == reg].groupby("x").auth.mean(); ax[1].plot(g.index, g.values, marker=mk, ms=4, label=reg)
    ax[0].axhline(0.078, color="k", ls="--", lw=0.7)
    ax[0].set_xlabel(r"spoofing probability $\psi$"); ax[0].set_ylabel("fake adoption (S3)"); ax[0].legend(); title(ax[0], "(a) spoofed consistency channel")
    ax[1].set_xlabel(r"false-contradiction rate $\varepsilon$"); ax[1].set_ylabel("authentic-notice uptake"); title(ax[1], "(b) noisy consistency channel")
    f.tight_layout(); save(f, "fig8_attack")


def fig_sobol():
    S = json.load(open(os.path.join(OUT, "sobol5.json")))
    names = ["c", "s", "kappa", "rho", "pi_ecosys", "bot_followers", "sig_hat_mult"]
    tex = [r"$c$", r"$s$", r"$\kappa$", r"$\rho$", r"$\pi$", r"$k_b$", r"$\hat\sigma/\sigma$"]
    f, ax = plt.subplots(3, 1, figsize=(ONEHALF, 7.2), gridspec_kw={"height_ratios": [1.1, 1, 1]})
    Ns = sorted(int(k) for k in S["fake_S2"] if k.isdigit())
    for i, n in enumerate(names):
        ax[0].errorbar(Ns, [S["fake_S2"][str(N)][n]["ST"] for N in Ns], yerr=[S["fake_S2"][str(N)][n]["STc"] for N in Ns],
                       marker="o", ms=3, capsize=2, label=tex[i])
    ax[0].set_xscale("log", base=2); ax[0].set_xlabel("base samples $N$"); ax[0].set_ylabel("$S_T$, fake adoption (S2)")
    ax[0].legend(ncol=7, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.28), columnspacing=1.0, handletextpad=0.3); title(ax[0], "(a) convergence of total-order indices")
    N = str(Ns[-1]); y = np.arange(len(names))
    for j, out in enumerate(["fake_S2", "authentic"]):
        a = ax[j + 1]
        s1 = [S[out][N][n]["S1"] for n in names]; st = [S[out][N][n]["ST"] for n in names]
        e1 = [S[out][N][n]["S1c"] for n in names]; et = [S[out][N][n]["STc"] for n in names]
        a.barh(y + 0.2, s1, 0.4, xerr=e1, capsize=2, label="$S_1$"); a.barh(y - 0.2, st, 0.4, xerr=et, capsize=2, label="$S_T$")
        a.set_yticks(y); a.set_yticklabels(tex); a.axvline(0, color="k", lw=0.5); a.invert_yaxis()
        a.set_xlabel("Sobol index"); title(a, ["(b) fake adoption (S2)", "(c) authentic uptake"][j])
    ax[1].legend(loc="lower right"); f.tight_layout(); save(f, "fig9_sobol")


ALL = ["killchain", "cpag", "worked", "threshold", "collective", "time", "pareto", "attack", "sobol"]
if __name__ == "__main__":
    for w in sys.argv[1:] or ALL:
        globals()["fig_" + w](); print("done", w)
