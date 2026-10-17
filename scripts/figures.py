import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import final_mc as F
import empirical as E

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 9,
    "axes.linewidth": 0.7, "axes.edgecolor": "#28251D", "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.frameon": False,
    "legend.fontsize": 8, "axes.grid": True, "grid.color": "#D4D1CA",
    "grid.linewidth": 0.4, "grid.alpha": 0.8, "figure.dpi": 200,
})
TEAL, RUST, DARK, GOLD = "#20808D", "#A84B2F", "#1B474D", "#B8860B"

# ---------------------------------------------------------------- figure 1 --
# relative absolute-risk bias as a function of the linear index, lambda = 224
X, w = F.cells()
b2 = np.array([0.30, 0.60, 0.85])
lam_t, N = 224, 275_638
beta = np.concatenate([[F.solve_intercept(X, w, b2, lam_t / N)], b2])
p = 1 / (1 + np.exp(-(X @ beta)))
q = p @ w; lam = N * q; m = p / q
Om = (X * (w * m)[:, None]).T @ X
Omi = np.linalg.inv(Om)
quad = np.einsum("ij,jk,ik->i", X, Omi, X)
om = ((w * m * quad)[:, None] * X).sum(0)
lin = X @ (Omi @ om)
eta = X @ beta
b_mle = 100 * (quad - lin) / (2 * lam)
b_fir = 100 * quad / (2 * lam)

fig, ax = plt.subplots(1, 2, figsize=(6.9, 2.85))
o = np.argsort(eta)
for cm, ls in ((1.0, "-"), (0.0, "--")):
    for kk, col in ((0.0, TEAL), (1.0, RUST)):
        s = (X[:, 1] == cm) & (X[:, 3] == kk)
        oo = np.argsort(eta[s])
        lbl = ("commercial" if cm else "Medicaid") + \
              (", comorbid" if kk else ", not comorbid")
        ax[0].plot(eta[s][oo], b_mle[s][oo], ls, color=col, lw=1.2, label=lbl)
ax[0].axhline(0, color="#28251D", lw=0.7)
ax[0].set_xlabel(r"linear index $x'\beta_n$")
ax[0].set_ylabel(r"relative bias of $\hat p(x)$, per cent")
ax[0].set_title("(a) Maximum likelihood: redistribution", fontsize=9, loc="left")
ax[0].set_ylim(-0.75, 4.4)
ax[0].legend(loc="upper center", ncol=2, fontsize=7, frameon=True, facecolor="white", edgecolor="none", framealpha=1)
neg = w @ (b_mle < 0)
ax[0].annotate(f"{100*neg:.0f}% of covariate mass\nbiased downward",
               xy=(0.66, 0.52), xycoords="axes fraction", fontsize=7,
               color="#28251D")

edges = np.linspace(-0.6, 3.4, 70)
wm = w * m
ax[1].hist(b_mle, bins=edges, weights=wm, color=TEAL, alpha=0.65,
           label="maximum likelihood")
ax[1].hist(b_fir, bins=edges, weights=wm, color=DARK, alpha=0.65, label="penalized (Firth)")
ax[1].axvline(0, color=TEAL, lw=1.1, ls="--")
ax[1].axvline(100 * X.shape[1] / (2 * lam), color=RUST, lw=1.1, ls="--")
ax[1].annotate(r"mean $=0$", xy=(0.19, 0.90), xycoords="axes fraction",
               fontsize=7.5, color=TEAL)
ax[1].annotate(r"mean $=K/2\lambda_n$", xy=(0.42, 0.78),
               xycoords="axes fraction", fontsize=7.5, color=RUST)
ax[1].set_xlabel(r"relative bias of $\hat p(x)$, per cent")
ax[1].set_ylabel("event-weighted covariate mass")
ax[1].set_title("(b) Event-weighted distribution of bias", fontsize=9, loc="left")
ax[1].legend(loc="center right")
ax[0].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}"))
fig.tight_layout()
fig.savefig("fig1_bias.pdf", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- figure 2 --
res = json.load(open("empirical_results.json"))
tab = res["table"]
ev = np.array([r["events"] for r in tab], float)
fb9 = np.array([r["firth_bias_pct_K9"] for r in tab])
fb5 = np.array([r["firth_bias_pct_K5"] for r in tab])
fb13 = np.array([r["firth_bias_pct_K13"] for r in tab])
rr = np.array([r["rel_regret_pct"] for r in tab])
names = [r["outcome"] for r in tab]

fig, ax = plt.subplots(1, 2, figsize=(6.9, 2.95))
ax[0].fill_between(ev, fb5, fb13, color=TEAL, alpha=0.18, label=r"$K\in\{5,13\}$")
ax[0].plot(ev, fb9, "o-", color=TEAL, ms=3.5, lw=1.2, label=r"$K=9$")
ax[0].set_xscale("log"); ax[0].set_yscale("log")
ax[0].set_xlabel("observed events in cohort")
ax[0].set_ylabel("penalized inflation of absolute risk, per cent")
ax[0].set_title("(a) Absolute-risk inflation, $K/2\\lambda$", fontsize=9, loc="left")
ax[0].legend(loc="upper right")
SHORT = {0: ("transfusion", 7, 7, "left"), 4: ("serious adverse event", 7, 7, "left"),
         8: ("any primary AE", -7, 8, "right"),
         10: ("broad composite", -7, 9, "right")}
BOX = dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.88)
for i, (lb, dx, dy, ha) in SHORT.items():
    ax[0].annotate(lb, xy=(ev[i], fb9[i]), xytext=(dx, dy),
                   textcoords="offset points", fontsize=6.6, color="#28251D",
                   ha=ha, bbox=BOX, zorder=6)

ax[1].plot(ev, rr, "s-", color=RUST, ms=3.5, lw=1.2)
ax[1].set_xscale("log"); ax[1].set_yscale("log")
ax[1].set_xlabel("observed events in cohort")
ax[1].set_ylabel("first-order regret, per cent of first best")
ax[1].set_title("(b) Targeting regret", fontsize=9, loc="left")
for i, (dx, dy, ha) in {0: (7, -2, "left"), 4: (7, 3, "left")}.items():
    ax[1].annotate(f"{rr[i]:.2f}%", xy=(ev[i], rr[i]), xytext=(dx, dy),
                   textcoords="offset points", fontsize=6.8, color="#28251D",
                   ha=ha, bbox=BOX, zorder=6)
fig.tight_layout()
fig.savefig("fig3_empirical.pdf", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- figure 3 --
mc = json.load(open("mc_results.json"))
lams = np.array([r["lam"] for r in mc])
fig, ax = plt.subplots(1, 2, figsize=(6.9, 2.95))
for key, lbl, col, mk in (("lamreg_m", "maximum likelihood", TEAL, "o"),
                          ("lamreg_f", "penalized (Firth)", RUST, "s"),
                          ("lamreg_c", "probability-scale corrected", GOLD, "^")):
    v = np.array([r[key][0] for r in mc]); s = np.array([r[key][1] for r in mc])
    ax[0].errorbar(lams, v, yerr=1.96 * s, fmt=mk + "-", color=col, ms=3.5,
                   lw=1.0, capsize=2, label=lbl)
ax[0].plot(lams, [r["half_trMOmi"] for r in mc], "k--", lw=1.0,
           label=r"theory $\frac{1}{2}\mathrm{tr}(M\Omega^{-1})$")
ax[0].set_xscale("log")
ax[0].set_xlabel(r"expected events $\lambda_n$")
ax[0].set_ylabel(r"$\lambda_n\,\mathcal{R}/\tau$")
ax[0].set_title("(a) Regret: invariant", fontsize=9, loc="left")
ax[0].legend(loc="lower right")
ax[0].set_ylim(0.44, 0.66)

for key, lbl, col, mk in (("relval_m", "maximum likelihood", TEAL, "o"),
                          ("relval_f", "penalized (Firth)", RUST, "s"),
                          ("relval_w", "welfare-calibrated", DARK, "D")):
    v = np.array([r[key][0] for r in mc]); s = np.array([r[key][1] for r in mc])
    ax[1].errorbar(lams, v, yerr=1.96 * s, fmt=mk + "-", color=col, ms=3.5,
                   lw=1.0, capsize=2, label=lbl)
ax[1].axhline(0, color="#28251D", lw=0.7)
ax[1].set_xscale("log")
ax[1].set_xlabel(r"expected events $\lambda_n$")
ax[1].set_ylabel("bias in program value, per cent")
ax[1].set_title("(b) Valuation: not invariant", fontsize=9, loc="left")
ax[1].legend(loc="upper right")
fig.tight_layout(w_pad=2.2)
fig.savefig("fig2_mc.pdf", bbox_inches="tight")
plt.close(fig)
print("figures written")
