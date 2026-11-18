"""Figures for manuscript v3. Concept figures: monochrome, serif. Result figures: standard matplotlib output, serif."""
import os, json, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

D = os.path.join(os.path.dirname(__file__), "..")
OUT, FIG = os.path.join(D, "out"), os.path.join(D, "fig3")
os.makedirs(FIG, exist_ok=True)
plt.style.use("default")
plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "TeX Gyre Termes", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 7.5,
                     "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
                     "savefig.dpi": 600, "figure.dpi": 100})
REG = ["D0", "D1", "D2", "D2C", "D3", "D4"]
LAB = {"D0": "D0 no defence", "D1": "D1 detection", "D2": "D2 provenance-only", "D2C": "D2C provenance + consistency veto",
       "D3": "D3 CPAG fusion", "D4": "D4 CPAG + PHC weighting"}
MK = dict(zip(REG, ["o", "s", "^", "v", "D", "P"]))


def save(f, n):
    f.savefig(os.path.join(FIG, n + ".png"), bbox_inches="tight"); f.savefig(os.path.join(FIG, n + ".pdf"), bbox_inches="tight"); plt.close(f)


# ---------------- concept figures (monochrome) ----------------
def node(ax, x, y, w, h, title, sub, lw=0.9):
    ax.add_patch(Rectangle((x, y), w, h, fc="white", ec="black", lw=lw))
    ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center", fontsize=9, weight="bold")
    ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center", fontsize=7.5, style="italic")


def arr(ax, p, q, ls="-", rad=0.0, lw=0.9):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=9, color="black", lw=lw, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


def lab(ax, x, y, t, ha="center"):
    ax.text(x, y, t, ha=ha, va="center", fontsize=8, bbox=dict(fc="white", ec="none", pad=0.6))


def fig_cpag():
    f, ax = plt.subplots(figsize=(7.0, 3.9)); ax.set_xlim(0, 14); ax.set_ylim(0, 7.8); ax.axis("off")
    for i, t in enumerate(["PHYSICAL PLANE", "CYBER PLANE", "SOCIAL PLANE"]):
        ax.text(2.3 + 4.7 * i, 7.45, t, ha="center", fontsize=9, weight="bold")
    for xv in [4.65, 9.35]:
        ax.plot([xv, xv], [0.4, 7.7], color="black", lw=0.6, ls=(0, (4, 3)))
    W, H = 3.4, 1.2
    P = {"dev": (0.6, 5.6), "data": (0.6, 3.4), "act": (0.6, 0.9), "proc": (5.3, 5.6), "cont": (5.3, 3.4),
         "actor": (10.0, 5.6), "dec": (10.0, 2.2)}
    node(ax, *P["dev"], W, H, "Device", "TPM / TEE, sensor")
    node(ax, *P["data"], W, H, "Data", "telemetry, AMI")
    node(ax, *P["act"], W, H, "Actuation", "load, route, dose")
    node(ax, *P["proc"], W, H, "Process", "model, pipeline")
    node(ax, *P["cont"], W, H, "Content", "media, notices")
    node(ax, *P["actor"], W, H, "Actor", "person, organisation, agent")
    node(ax, *P["dec"], W, H, "Decision", r"$P(\omega)\geq\rho$ ?")
    c = lambda k, dx=0.5, dy=0.5: (P[k][0] + W * dx, P[k][1] + H * dy)
    arr(ax, c("dev", .5, 0), c("data", .5, 1)); lab(ax, 2.3, 5.0, "ATT")
    arr(ax, c("proc", .5, 0), c("cont", .5, 1)); lab(ax, 7.0, 5.0, "TRN")
    arr(ax, c("actor", 0, .5), c("cont", 1, .75)); lab(ax, 8.85, 5.55, "CC / DEL")
    arr(ax, c("actor", .5, 0), c("dec", .5, 1)); lab(ax, 11.7, 4.6, "DEL")
    arr(ax, c("cont", 1, .25), c("dec", 0, .75)); lab(ax, 9.35, 3.35, r"$\oplus$")
    arr(ax, c("data", 1, .2), c("dec", 0, .3), ls=(0, (5, 2)), rad=0.18); lab(ax, 7.0, 2.25, "CON (cross-plane consistency)")
    arr(ax, c("dec", .5, 0), (P["dec"][0] + W * .5, 1.5)); ax.plot([P["dec"][0] + W * .5, c("act", 1, .5)[0] + 0.02], [1.5, 1.5], color="black", lw=0.9)
    arr(ax, (8.0, 1.5), c("act", 1, .5))
    ax.text(7.0, 0.15, r"Each node carries an opinion $\omega=(b,d,u,a)$; edges propagate it by operators R1–R5.", ha="center", fontsize=8, style="italic")
    save(f, "fig1_cpag")


def fig_chain():
    f, ax = plt.subplots(figsize=(7.0, 2.6)); ax.set_xlim(0, 15); ax.set_ylim(0, 5); ax.axis("off")
    st = [("Bot swarm", "A1 + A2, fake notice"), ("Diffusion", "threshold contagion"), ("Belief", "household decision"),
          ("Load surge", "EV, heat pump"), ("Overload", r"$x > x^{*}\approx 0.08$")]
    for i, (t, s) in enumerate(st):
        node(ax, 0.2 + 3.0 * i, 2.7, 2.5, 1.3, t, s)
        if i < 4:
            arr(ax, (2.7 + 3.0 * i, 3.35), (3.2 + 3.0 * i, 3.35))
    dd = [(1, "D1: removal"), (2, "D2–D4: credential"), (3, "D3: CON edge")]
    for i, t in dd:
        x = 0.2 + 3.0 * i + 1.25
        ax.text(x, 0.75, t, ha="center", fontsize=8, bbox=dict(fc="white", ec="black", lw=0.7, ls=(0, (3, 2)), pad=3))
        arr(ax, (x, 1.1), (x, 2.68), ls=(0, (3, 2)))
    ax.text(7.45, 4.6, "A4 (S2): signature from a compromised aggregator", fontsize=8, style="italic", ha="center")
    arr(ax, (7.45, 4.35), (7.45, 4.02))
    save(f, "fig2_killchain")


# ---------------- result figures (standard matplotlib) ----------------
def fig_threshold():
    from phase import sigma_star
    P = np.linspace(0, 0.99, 200); R = np.linspace(0.3, 0.97, 200)
    g = np.array([[sigma_star(p, r) for p in P] for r in R])
    f, ax = plt.subplots(1, 2, figsize=(7.0, 2.8))
    cs = ax[0].contour(P, R, g, levels=[0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5], colors="k", linewidths=0.8)
    ax[0].clabel(cs, fmt="%.2f", fontsize=7)
    ax[0].plot(0.6, 0.75, "o", color="C3"); ax[0].annotate(r"$\sigma^{*}=0.118$", (0.6, 0.75), (0.63, 0.86), fontsize=8)
    ax[0].set_xlabel(r"provenance coverage $\pi$"); ax[0].set_ylabel(r"action threshold $\rho$"); ax[0].set_title(r"(a) contours of $\sigma^{*}(\pi,\rho)$", fontsize=9)
    s = np.linspace(0, 0.5, 300)
    for i, pi in enumerate([0.0, 0.3, 0.6, 0.9]):
        n = (1 - s) * (1 - pi); ax[1].plot(s, n / (n + s), marker=MK[REG[i]], markevery=50, ms=4, label=rf"$\pi={pi}$")
    ax[1].axhline(0.75, color="k", ls="--", lw=0.8)
    ax[1].set_xlabel(r"synthetic share $\sigma$"); ax[1].set_ylabel(r"$a(\sigma,\pi)$"); ax[1].set_title("(b) base rate of unsigned signals", fontsize=9)
    ax[1].legend(); ax[1].set_ylim(0, 1.02)
    f.tight_layout(); save(f, "fig3_threshold")


def fig_phase():
    L = np.load(os.path.join(OUT, "loops.npy")); s = np.load(os.path.join(OUT, "loops_sig.npy"))
    phase = np.load(os.path.join(OUT, "phase.npy")); S = np.linspace(0.005, 0.5, 120); PI = np.linspace(0, 0.95, 96)
    f, ax = plt.subplots(1, 2, figsize=(7.0, 2.8))
    for i, pi in enumerate([0.0, 0.3, 0.6, 0.9]):
        ax[0].plot(s, L[i, 0], color=f"C{i}", label=rf"$\pi={pi}$"); ax[0].plot(s, L[i, 1], color=f"C{i}", ls="--")
    ax[0].set_xlabel(r"synthetic share $\sigma$"); ax[0].set_ylabel(r"trusting fraction $y^{*}$"); ax[0].legend(loc="upper right")
    ax[0].set_title(r"(a) increasing (solid) and decreasing (dashed) $\sigma$", fontsize=9)
    im = ax[1].contourf(S, PI, phase, levels=[-0.5, 0.5, 1.5, 2.5], colors=["#ffffff", "#7f7f7f", "#d0d0d0"])
    ax[1].contour(S, PI, phase, levels=[0.5, 1.5], colors="k", linewidths=0.6)
    ax[1].text(0.05, 0.25, "trust", fontsize=9); ax[1].text(0.36, 0.55, "distrust", fontsize=9)
    ax[1].annotate("bistable", (0.205, 0.52), (0.26, 0.85), fontsize=9, arrowprops=dict(arrowstyle="->", lw=0.7))
    ax[1].set_xlabel(r"synthetic share $\sigma$"); ax[1].set_ylabel(r"provenance coverage $\pi$")
    ax[1].set_title(r"(b) regimes, $\kappa=0.6$, $\eta=0.05$", fontsize=9)
    f.tight_layout(); save(f, "fig4_phase")


def fig_grid(d):
    f, ax = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
    for k, sc in enumerate(["S1", "S2"]):
        x = d[d.scenario == sc]
        for i, r in enumerate(REG):
            g = x[x.regime == r].groupby("sigma").trafo
            m, ci = g.mean(), 1.96 * g.std() / np.sqrt(g.count())
            ax[k].errorbar(m.index + (i - 2.5) * 0.004, m.values, yerr=ci.values, marker=MK[r], ms=4, capsize=2, lw=1.2, label=LAB[r])
        ax[k].axhline(100, color="k", ls="--", lw=0.8)
        ax[k].set_xlabel(r"bot share $\sigma$"); ax[k].set_title(["(a) S1: unsigned fake notice", "(b) S2: compromised-issuer signature"][k], fontsize=9)
    ax[0].set_ylabel("transformer loading (%)")
    h, l = ax[0].get_legend_handles_labels(); f.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.1))
    f.tight_layout(rect=(0, 0.08, 1, 1)); save(f, "fig5_grid")


def fig_tradeoffs(d, sw):
    f, ax = plt.subplots(1, 3, figsize=(7.0, 2.6))
    x = d[d.scenario == "S1"]
    for r in REG:
        g = x[x.regime == r].groupby("sigma").auth_adopt.mean(); ax[0].plot(g.index, g.values, marker=MK[r], ms=4, label=r)
    ax[0].axvline(0.1176, color="k", ls=":", lw=0.8)
    ax[0].set_xlabel(r"bot share $\sigma$"); ax[0].set_ylabel("authentic-notice uptake"); ax[0].set_title("(a)", fontsize=9); ax[0].set_ylim(0, 0.22)
    ax[0].legend(ncol=2, fontsize=6.5, loc="lower left")
    s = sw[sw.sweep == "strip"]
    for r in ["D0", "D2", "D3"]:
        g = s[s.regime == r].groupby("x").auth.mean(); ax[1].plot(g.index, g.values, marker=MK[r], ms=4, color=f"C{REG.index(r)}", label=r)
    ax[1].set_xlabel("stripping rate $s$"); ax[1].set_title(r"(b) $\sigma=0.2$", fontsize=9); ax[1].set_ylim(0, 0.22); ax[1].legend()
    k = sw[sw.sweep == "kappa"].groupby("x")[["fake_S2", "auth"]].mean()
    ax[2].plot(k.index, k.fake_S2, marker="s", ms=4, label="fake (S2)"); ax[2].plot(k.index, k.auth, marker="^", ms=4, label="authentic")
    ax[2].axhline(0.08, color="k", ls="--", lw=0.8)
    ax[2].set_xlabel(r"verification capacity $\kappa$"); ax[2].set_title(r"(c) D3, $\sigma=0.2$", fontsize=9); ax[2].legend()
    f.tight_layout(); save(f, "fig6_tradeoffs")


def fig_xcrit(d):
    x = d[d.fake_adopt > 0]; c = np.polyfit(x.fake_adopt, x.trafo, 1)
    pred = np.polyval(c, x.fake_adopt); r2 = 1 - ((x.trafo - pred) ** 2).sum() / ((x.trafo - x.trafo.mean()) ** 2).sum()
    f, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.scatter(x.fake_adopt, x.trafo, s=4, alpha=0.4)
    xx = np.linspace(0, 0.45, 10); ax.plot(xx, np.polyval(c, xx), "k-", lw=1)
    xc = (100 - c[1]) / c[0]; ax.axhline(100, color="k", ls="--", lw=0.8); ax.axvline(xc, color="C3", ls=":", lw=1)
    ax.set_xlabel("fraction of households acting $x$"); ax.set_ylabel("transformer loading (%)")
    save(f, "fig7_xcrit"); return dict(slope=c[0], intercept=c[1], xcrit=xc, r2=r2, n=len(x))


def fig_sobol():
    S = json.load(open(os.path.join(OUT, "sobol.json")))
    names = ["c", "s", "kappa", "rho", "pi_ecosys", "bot_followers", "sig_hat_mult"]
    tex = [r"$c$", r"$s$", r"$\kappa$", r"$\rho$", r"$\pi$", r"$k_b$", r"$\hat\sigma/\sigma$"]
    f, ax = plt.subplots(1, 2, figsize=(7.0, 2.5), sharey=True)
    for j, out in enumerate(["fake_S2", "authentic"]):
        st = [S[out][n]["ST"] for n in names]; se = [S[out][n]["ST_conf"] for n in names]
        s1 = [S[out][n]["S1"] for n in names]; s1e = [S[out][n]["S1_conf"] for n in names]
        y = np.arange(len(names))
        ax[j].barh(y + 0.2, s1, 0.4, xerr=s1e, label="$S_1$", capsize=2); ax[j].barh(y - 0.2, st, 0.4, xerr=se, label="$S_T$", capsize=2)
        ax[j].set_yticks(y); ax[j].set_yticklabels(tex); ax[j].set_title(["(a) fake (S2) adoption", "(b) authentic uptake"][j], fontsize=9)
        ax[j].set_xlabel("Sobol index"); ax[j].axvline(0, color="k", lw=0.5)
    ax[0].legend(); f.tight_layout(); save(f, "fig8_sobol")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    fig_cpag(); fig_chain(); fig_threshold(); fig_phase()
    if os.path.exists(os.path.join(OUT, "abm_results.csv")) and "D2C" in open(os.path.join(OUT, "abm_results.csv")).read():
        d = pd.read_csv(os.path.join(OUT, "abm_results.csv")); sw = pd.read_csv(os.path.join(OUT, "sweeps.csv"))
        fig_grid(d); fig_tradeoffs(d, sw); print(fig_xcrit(d))
    if os.path.exists(os.path.join(OUT, "sobol.json")):
        fig_sobol()
    print("done")
