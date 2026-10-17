"""
Final Monte Carlo evidence for "Welfare-Calibrated Estimation for Rare Events".
Produces mc_results.json used to typeset Tables 1 to 3.

Verified objects (paper numbering):
  Prop. 1 : index bias      E[bhat] - beta = -(2 lam)^{-1} Om^{-1} om + O(lam^{-2})
  Thm  2  : second-order relative absolute-risk bias
              b2(x) = (2 lam)^{-1} x'Om^{-1}(x - om)      [MLE]
              b2F(x)= (2 lam)^{-1} x'Om^{-1}x             [Firth, uniformly > 0]
            checked against the MC mean of the truncated stochastic expansion
              x'd + (x'd)^2/2,  d = bhat - beta
            and against the exact MC mean of phat/p - 1 for Firth.
  Thm  3  : lam * Regret / tau -> (1/2) tr(M Om^{-1}), identically for MLE,
            Firth and the probability-scale corrected rule.
  Thm  4  : relative program-value bias, first order in 1/lam.
  Thm  5  : WCPML is relative-unbiased to O(lam^{-2}) and value-unbiased.
"""
import json
import numpy as np
from scipy.special import ndtri

K = 4
NAGE = 500
N_COHORT = 275_638


# ---------------------------------------------------------------- design ----
def cells(p_comm=0.7938, p_com=0.0655, nage=NAGE):
    e = np.linspace(0, 1, nage + 1)[1:-1]
    cuts = ndtri(e)
    lo = np.concatenate([[-np.inf], cuts]); hi = np.concatenate([cuts, [np.inf]])
    phi = lambda z: np.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi)
    rep = (np.where(np.isfinite(lo), phi(lo), 0.0)
           - np.where(np.isfinite(hi), phi(hi), 0.0)) * nage
    rows, wts = [], []
    for cm, wc in ((1.0, p_comm), (0.0, 1 - p_comm)):
        for kk, wk in ((1.0, p_com), (0.0, 1 - p_com)):
            for a in rep:
                rows.append([1.0, cm, a, kk]); wts.append(wc * wk / nage)
    X = np.asarray(rows); w = np.asarray(wts)
    return X, w / w.sum()


def solve_intercept(X, w, b2, target_q):
    lo, hi = -30.0, 6.0
    for _ in range(400):
        mid = 0.5 * (lo + hi)
        p = 1.0 / (1.0 + np.exp(-(mid + X[:, 1:] @ b2)))
        if p @ w > target_q:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def fit(X, nc, yc, firth=False, tol=1e-12, maxit=500):
    N = nc.astype(float); Y = yc.astype(float)
    b = np.zeros(X.shape[1])
    b[0] = np.log(max(Y.sum(), .5) / max(N.sum() - Y.sum(), .5))
    for _ in range(maxit):
        p = 1.0 / (1.0 + np.exp(-(X @ b)))
        wv = np.clip(p * (1 - p), 1e-15, None)
        H = (X * (N * wv)[:, None]).T @ X
        try:
            Hi = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            return None
        if firth:
            h = wv * np.einsum("ij,jk,ik->i", X, Hi, X)
            g = X.T @ (Y - N * p + N * h * (0.5 - p))
        else:
            g = X.T @ (Y - N * p)
        s = Hi @ g
        if not np.all(np.isfinite(s)):
            return None
        nrm = np.max(np.abs(s))
        if nrm > 2.0:
            s = s * (2.0 / nrm)
        b = b + s
        if nrm < tol:
            break
    return b if np.all(np.isfinite(b)) else None


# ------------------------------------------------------------ experiment ----
def experiment(lam_target, reps, n=N_COHORT, b2=(0.30, 0.60, 0.85),
               theta_q=0.85, seed=2026):
    rng = np.random.default_rng(seed + int(lam_target))
    X, w = cells()
    b2 = np.asarray(b2, float)
    beta = np.concatenate([[solve_intercept(X, w, b2, lam_target / n)], b2])

    p = 1.0 / (1.0 + np.exp(-(X @ beta)))
    q = p @ w
    lam = n * q
    m = p / q
    Om = (X * (w * m)[:, None]).T @ X
    Omi = np.linalg.inv(Om)
    quad = np.einsum("ij,jk,ik->i", X, Omi, X)
    om = ((w * m * quad)[:, None] * X).sum(0)
    lin = X @ (Omi @ om)

    b2_mle_th = (quad - lin) / (2 * lam)
    b2_fir_th = quad / (2 * lam)
    idxb_th = -(Omi @ om) / (2 * lam)

    eta = X @ beta
    o = np.argsort(eta); cw = np.cumsum(w[o])
    theta = eta[o][np.searchsorted(cw, theta_q)]
    t = 1.0 / (1.0 + np.exp(-theta)); tau = n * t
    mu = n * p
    dstar = (eta >= theta).astype(float)
    W_star = w @ (dstar * (mu - tau))
    bw = 1.5 * np.std(eta) / np.sqrt(lam)
    ker = np.exp(-0.5 * ((eta - theta) / bw) ** 2) / (bw * np.sqrt(2 * np.pi))
    M = (X * (w * ker)[:, None]).T @ X
    trM = np.trace(M @ Omi)

    valb_mle_th = w @ (dstar * mu * (quad - lin)) / (2 * lam)
    valb_fir_th = w @ (dstar * mu * quad) / (2 * lam)

    nc = np.round(n * w).astype(np.int64); nc[0] += n - nc.sum()
    th_hat = np.log(t / (1 - t))

    acc = {k: [] for k in ("idxb", "exp_m", "exp_f", "rb_f", "rb_w",
                           "reg_m", "reg_f", "reg_c", "val_m", "val_f", "val_w",
                           "neg_m", "neg_f", "sep")}
    for _ in range(reps):
        yc = rng.binomial(nc, p)
        bh = fit(X, nc, yc); bf = fit(X, nc, yc, firth=True)
        if bf is None:
            continue
        if bh is None or np.max(np.abs(bh - beta)) > 25:
            acc["sep"].append(1.0)
            bh = bf.copy()          # MLE unusable; record and fall back
        else:
            acc["sep"].append(0.0)
        acc["idxb"].append(bh - beta)

        dm = X @ (bh - beta); df = X @ (bf - beta)
        acc["exp_m"].append(dm + 0.5 * dm ** 2)     # truncated expansion, MLE
        acc["exp_f"].append(df + 0.5 * df ** 2)     # truncated expansion, Firth

        pf = 1.0 / (1.0 + np.exp(-(X @ bf)))
        pm = 1.0 / (1.0 + np.exp(-(X @ bh)))
        acc["rb_f"].append(pf / p - 1.0)

        qs = pf @ w; lam_s = n * qs
        Om_s = (X * (w * (pf / qs))[:, None]).T @ X
        qd = np.einsum("ij,jk,ik->i", X, np.linalg.inv(Om_s), X)
        pw = pf * np.exp(-qd / (2 * lam_s))
        acc["rb_w"].append(pw / p - 1.0)

        for key, bb in (("reg_m", bh), ("reg_f", bf)):
            d = (X @ bb >= th_hat).astype(float)
            acc[key].append(W_star - w @ (d * (mu - tau)))
        pk = pf + 0.5 * pf * qd / lam_s
        acc["reg_c"].append(W_star - w @ ((pk >= t).astype(float) * (mu - tau)))

        acc["val_m"].append(w @ (dstar * (n * pm - tau)))
        acc["val_f"].append(w @ (dstar * (n * pf - tau)))
        acc["val_w"].append(w @ (dstar * (n * pw - tau)))

    A = {k: np.asarray(v) for k, v in acc.items()}
    R = len(A["idxb"])

    def ms(a):                          # mean and se of a scalar array
        return float(a.mean()), float(a.std(ddof=1) / np.sqrt(len(a)))

    exp_m = A["exp_m"].mean(0); exp_f = A["exp_f"].mean(0)
    rb_f = A["rb_f"].mean(0)

    out = dict(
        lam=float(lam), q=float(q), n=n, tau=float(tau), t=float(t),
        theta=float(theta), reps=R, sep_rate=float(A["sep"].mean()),
        trMOmi=float(trM), half_trMOmi=float(0.5 * trM), W_star=float(W_star),
        beta=beta.tolist(),
        idxb_mc=A["idxb"].mean(0).tolist(), idxb_th=idxb_th.tolist(),
        # Theorem 2: truncated-expansion check, weighted mean and OLS slope
        exp_m_mean=float(w @ exp_m), b2_mle_th_mean=float(w @ b2_mle_th),
        exp_m_slope=float(np.polyfit(b2_mle_th, exp_m, 1)[0]),
        exp_f_mean=float(w @ exp_f), b2_fir_th_mean=float(w @ b2_fir_th),
        exp_f_slope=float(np.polyfit(b2_fir_th, exp_f, 1)[0]),
        # sign reversal
        pneg_mc=float(w @ (exp_m < 0)), pneg_th=float(w @ (b2_mle_th < 0)),
        pneg_fir_mc=float(w @ (rb_f < 0)),
        # Firth exact relative bias
        rb_f_mean=float(w @ rb_f), rb_w_mean=float(w @ A["rb_w"].mean(0)),
        # Theorem 3
        lamreg_m=ms(lam * A["reg_m"] / tau), lamreg_f=ms(lam * A["reg_f"] / tau),
        lamreg_c=ms(lam * A["reg_c"] / tau),
        # Theorem 4 (relative value bias, percent of W*)
        relval_m=ms(100 * (A["val_m"] - W_star) / W_star),
        relval_f=ms(100 * (A["val_f"] - W_star) / W_star),
        relval_w=ms(100 * (A["val_w"] - W_star) / W_star),
        relval_m_th=float(100 * valb_mle_th / W_star),
        relval_f_th=float(100 * valb_fir_th / W_star),
    )
    return out


if __name__ == "__main__":
    import sys
    res = []
    for lam, reps in [(224, 20000), (1122, 20000), (5152, 12000), (18139, 8000)]:
        r = experiment(lam, reps)
        res.append(r)
        print(f"--- lam={r['lam']:.0f}  q={r['q']:.5f}  reps={r['reps']}  "
              f"sep/unstable MLE rate={r['sep_rate']:.4f}")
        print(f"  idx bias  MC {np.array2string(np.array(r['idxb_mc']),precision=5)}")
        print(f"            TH {np.array2string(np.array(r['idxb_th']),precision=5)}")
        print(f"  T2 MLE  E[trunc exp]  {r['exp_m_mean']:+.6f}  th {r['b2_mle_th_mean']:+.6f}"
              f"  slope {r['exp_m_slope']:.3f}   P(neg) MC {r['pneg_mc']:.3f} TH {r['pneg_th']:.3f}")
        print(f"  T2 Firth E[trunc exp] {r['exp_f_mean']:+.6f}  th {r['b2_fir_th_mean']:+.6f}"
              f"  slope {r['exp_f_slope']:.3f}   exact rel-bias {r['rb_f_mean']:+.6f}"
              f"  P(neg) {r['pneg_fir_mc']:.3f}")
        print(f"  T5 WCPML exact rel-bias {r['rb_w_mean']:+.6f}")
        print(f"  T3 lam*Reg/tau  MLE {r['lamreg_m'][0]:.4f}({r['lamreg_m'][1]:.4f})  "
              f"Firth {r['lamreg_f'][0]:.4f}({r['lamreg_f'][1]:.4f})  "
              f"corr {r['lamreg_c'][0]:.4f}({r['lamreg_c'][1]:.4f})  | th {r['half_trMOmi']:.4f}")
        print(f"  T4 rel value bias %  MLE {r['relval_m'][0]:+.3f}({r['relval_m'][1]:.3f}) "
              f"th {r['relval_m_th']:+.3f} | Firth {r['relval_f'][0]:+.3f}"
              f"({r['relval_f'][1]:.3f}) th {r['relval_f_th']:+.3f} | "
              f"WCPML {r['relval_w'][0]:+.3f}({r['relval_w'][1]:.3f})")
        sys.stdout.flush()
    json.dump(res, open("mc_results.json", "w"), indent=1)
    print("\nwrote mc_results.json")
