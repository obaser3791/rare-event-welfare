"""
Monte Carlo verification of the asymptotic expansions in
"Welfare-Calibrated Estimation for Rare Events".

Everything is computed on a finite cell partition of the covariate space, so
population expectations are EXACT sums (no integration error) and sampling is
cell-wise binomial.

Notation (paper Section 2):
  x_c = (1, commercial_c, a_c, k_c)'   in R^K,  cell weight w_c,  sum w_c = 1
  p_c = Lam(x_c' beta_n),  q_n = sum_c w_c p_c,  m_c = p_c / q_n
  Om  = sum_c w_c m_c x_c x_c'                       (normalized information)
  om  = sum_c w_c m_c (x_c' Om^{-1} x_c) x_c         (bias direction)
  lam = n q_n                                        (expected events)
  I_n = lam * Om + O(lam q_n)

Claims verified:
  (P1) MLE index bias      = -(1/(2 lam)) Om^{-1} om + O(lam^{-2})
  (T2) MLE relative bias of fitted probability
                           = (1/(2 lam)) x' Om^{-1} (x - om) + O(lam^{-2})
       -> sign reverses across the ellipsoid x'Om^{-1}(x - om) = 0
  (T2c) Firth relative bias = + (1/(2 lam)) x' Om^{-1} x  (uniformly positive)
  (T3) lam * Regret / tau  -> (1/2) tr(M Om^{-1})  for MLE and for Firth alike
  (T4) program-value bias is first order in 1/lam, sign-ambiguous for MLE,
       strictly positive for Firth
  (T5) the welfare-calibrated estimator
         p_check(x) = Lam(x' beta_firth) * exp( - x' Om^{-1} x / (2 lam) )
       is relative-unbiased to O(lam^{-2}), gives an unbiased program value,
       and (being a monotone transform of the index along the boundary only up
       to the quadratic term) is used for VALUATION while the FIRTH INDEX rule
       is used for ASSIGNMENT -- the paper's separation principle.
"""
import numpy as np

# ---------------------------------------------------------------------------
# cell design
# ---------------------------------------------------------------------------
NAGE = 60


def build_cells(p_commercial=0.7938, p_comorb=0.0655, nage=NAGE, seed=0):
    """payer (2) x comorbidity (2) x standardized age (nage bins)."""
    # age bins: equal-probability bins of a standard normal
    edges = np.linspace(0, 1, nage + 1)[1:-1]
    from math import sqrt
    from scipy.special import ndtri
    cuts = ndtri(edges)
    # bin representative = conditional mean of a standard normal in the bin
    lo = np.concatenate([[-np.inf], cuts])
    hi = np.concatenate([cuts, [np.inf]])
    phi = lambda z: np.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi)
    plo = np.where(np.isfinite(lo), phi(lo), 0.0)
    phi_hi = np.where(np.isfinite(hi), phi(hi), 0.0)
    reps = (plo - phi_hi) * nage            # E[Z | bin] for equal-prob bins
    wage = np.full(nage, 1.0 / nage)

    rows, wts = [], []
    for cm, wc in ((1.0, p_commercial), (0.0, 1 - p_commercial)):
        for kk, wk in ((1.0, p_comorb), (0.0, 1 - p_comorb)):
            for a, wa in zip(reps, wage):
                rows.append([1.0, cm, a, kk])
                wts.append(wc * wk * wa)
    X = np.array(rows)
    w = np.array(wts)
    w = w / w.sum()
    return X, w


# ---------------------------------------------------------------------------
# population objects
# ---------------------------------------------------------------------------
def solve_intercept(X, w, beta2, target_q):
    lo, hi = -30.0, 5.0
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        p = 1.0 / (1.0 + np.exp(-(mid + X[:, 1:] @ beta2)))
        if p @ w > target_q:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def pop_objects(X, w, beta, n):
    p = 1.0 / (1.0 + np.exp(-(X @ beta)))
    q = p @ w
    m = p / q
    Om = (X * (w * m)[:, None]).T @ X
    Omi = np.linalg.inv(Om)
    quad = np.einsum("ij,jk,ik->i", X, Omi, X)          # x'Om^{-1}x
    om = ((w * m * quad)[:, None] * X).sum(axis=0)
    lin = X @ (Omi @ om)                               # x'Om^{-1}om
    return dict(p=p, q=q, m=m, Om=Om, Omi=Omi, quad=quad, om=om, lin=lin,
                lam=n * q)


def boundary_form(X, w, beta, theta, bw=0.25):
    """M = int_H x x' Gamma(dx) with Gamma the boundary density of P_X.
    Estimated by a symmetric kernel shell in index units."""
    s = X @ beta - theta
    ker = np.exp(-0.5 * (s / bw) ** 2) / (bw * np.sqrt(2 * np.pi))
    ww = w * ker
    return (X * ww[:, None]).T @ X


# ---------------------------------------------------------------------------
# estimators on grouped binomial data
# ---------------------------------------------------------------------------
def fit(X, nc, yc, firth=False, tol=1e-11, maxit=400):
    keep = nc > 0
    Xf, N, Y = X[keep], nc[keep].astype(float), yc[keep].astype(float)
    b = np.zeros(X.shape[1])
    b[0] = np.log(max(Y.sum(), 0.5) / max(N.sum() - Y.sum(), 0.5))
    for _ in range(maxit):
        eta = Xf @ b
        p = 1.0 / (1.0 + np.exp(-eta))
        wv = np.clip(p * (1 - p), 1e-14, None)
        H = (Xf * (N * wv)[:, None]).T @ Xf
        try:
            Hi = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            return None
        if firth:
            h = wv * np.einsum("ij,jk,ik->i", Xf, Hi, Xf)
            g = Xf.T @ (Y - N * p + N * h * (0.5 - p))
        else:
            g = Xf.T @ (Y - N * p)
        step = Hi @ g
        if not np.all(np.isfinite(step)):
            return None
        nrm = np.max(np.abs(step))
        if nrm > 2.0:
            step *= 2.0 / nrm
        b = b + step
        if nrm < tol:
            break
    return b if np.all(np.isfinite(b)) else None


# ---------------------------------------------------------------------------
# experiment
# ---------------------------------------------------------------------------
def run(n, lam_target, reps, beta2=(0.30, 0.60, 0.85), theta_q=0.85,
        seed=11, verbose=True):
    rng = np.random.default_rng(seed)
    X, w = build_cells()
    beta2 = np.asarray(beta2, float)
    b1 = solve_intercept(X, w, beta2, lam_target / n)
    beta = np.concatenate([[b1], beta2])
    P = pop_objects(X, w, beta, n)
    lam, Omi, quad, lin = P["lam"], P["Omi"], P["quad"], P["lin"]

    # ---- decision problem -------------------------------------------------
    eta = X @ beta
    order = np.argsort(eta)
    cw = np.cumsum(w[order])
    theta = eta[order][np.searchsorted(cw, theta_q)]
    t = 1.0 / (1.0 + np.exp(-theta))
    tau = n * t
    mu = n * P["p"]
    dstar = (eta >= theta).astype(float)
    W_star = w @ (dstar * (mu - tau))
    M = boundary_form(X, w, beta, theta)
    trMOmi = np.trace(M @ Omi)

    # ---- theory -----------------------------------------------------------
    idx_bias_th = -(Omi @ P["om"]) / (2 * lam)
    rb_mle_th = (quad - lin) / (2 * lam)
    rb_fir_th = quad / (2 * lam)
    regret_th = tau * trMOmi / (2 * lam)
    valbias_mle_th = w @ (dstar * mu * (quad - lin)) / (2 * lam)
    valbias_fir_th = w @ (dstar * mu * quad) / (2 * lam)

    # ---- Monte Carlo ------------------------------------------------------
    ncell = np.round(n * w).astype(np.int64)
    ncell[0] += n - ncell.sum()
    thetahat = np.log(t / (1 - t))

    S = dict(bh=[], bf=[], rb_m=[], rb_f=[], rb_w=[],
             reg_m=[], reg_f=[], reg_kz=[], val_m=[], val_f=[], val_w=[])
    nfail = 0
    for _ in range(reps):
        yc = rng.binomial(ncell, P["p"])
        if yc.sum() < 3:
            nfail += 1
            continue
        bh = fit(X, ncell, yc, firth=False)
        bf = fit(X, ncell, yc, firth=True)
        if bh is None or bf is None:
            nfail += 1
            continue
        S["bh"].append(bh); S["bf"].append(bf)

        pm = 1.0 / (1.0 + np.exp(-(X @ bh)))
        pf = 1.0 / (1.0 + np.exp(-(X @ bf)))
        # sample analogues for the calibration factor
        qs = pf @ w
        lam_s = n * qs
        Om_s = (X * (w * (pf / qs))[:, None]).T @ X
        Omi_s = np.linalg.inv(Om_s)
        quad_s = np.einsum("ij,jk,ik->i", X, Omi_s, X)
        pw = pf * np.exp(-quad_s / (2 * lam_s))

        S["rb_m"].append(pm / P["p"] - 1)
        S["rb_f"].append(pf / P["p"] - 1)
        S["rb_w"].append(pw / P["p"] - 1)

        for key, bb in (("reg_m", bh), ("reg_f", bf)):
            d = (X @ bb >= thetahat).astype(float)
            S[key].append(W_star - w @ (d * (mu - tau)))
        p_kz = pf + 0.5 * pf * quad_s / lam_s          # probability-scale corr.
        d_kz = (p_kz >= t).astype(float)
        S["reg_kz"].append(W_star - w @ (d_kz * (mu - tau)))

        S["val_m"].append(w @ (dstar * (n * pm - tau)))
        S["val_f"].append(w @ (dstar * (n * pf - tau)))
        S["val_w"].append(w @ (dstar * (n * pw - tau)))

    A = {k: np.array(v) for k, v in S.items()}
    rb_m, rb_f, rb_w = A["rb_m"].mean(0), A["rb_f"].mean(0), A["rb_w"].mean(0)
    idx_bias_mc = A["bh"].mean(0) - beta

    R = dict(n=n, lam=lam, tau=tau, t=t, theta=theta, trMOmi=trMOmi,
             W_star=W_star, nfail=nfail, reps=len(A["bh"]), beta=beta,
             idx_bias_mc=idx_bias_mc, idx_bias_th=idx_bias_th,
             regret_th=regret_th, reg_m=A["reg_m"].mean(),
             reg_f=A["reg_f"].mean(), reg_kz=A["reg_kz"].mean(),
             val_m=A["val_m"].mean(), val_f=A["val_f"].mean(),
             val_w=A["val_w"].mean(),
             valbias_mle_th=valbias_mle_th, valbias_fir_th=valbias_fir_th,
             wbar_mle=w @ rb_m, wbar_mle_th=w @ rb_mle_th,
             wbar_fir=w @ rb_f, wbar_fir_th=w @ rb_fir_th,
             wbar_wc=w @ rb_w,
             slope_mle=np.polyfit(rb_mle_th, rb_m, 1)[0],
             slope_fir=np.polyfit(rb_fir_th, rb_f, 1)[0],
             fracneg_mle_mc=w @ (rb_m < 0), fracneg_mle_th=w @ (rb_mle_th < 0),
             fracneg_fir_mc=w @ (rb_f < 0),
             rb_m=rb_m, rb_f=rb_f, rb_w=rb_w,
             rb_mle_th=rb_mle_th, rb_fir_th=rb_fir_th, X=X, w=w)

    if verbose:
        print("=" * 80)
        print(f"n={n:,}  lam={lam:,.1f}  tau={tau:.3f}  tr(M Om^-1)={trMOmi:.4f}  "
              f"reps={R['reps']}  fails={nfail}")
        print(f" P1 index bias   MC {np.array2string(idx_bias_mc, precision=5)}")
        print(f"                 TH {np.array2string(idx_bias_th, precision=5)}")
        print(f" T2 MLE  rel-bias  mean MC {R['wbar_mle']:+.5f}  TH {R['wbar_mle_th']:+.5f}"
              f"   regr-slope {R['slope_mle']:.3f}"
              f"   P(neg) MC {R['fracneg_mle_mc']:.3f} TH {R['fracneg_mle_th']:.3f}")
        print(f" T2c Firth rel-bias mean MC {R['wbar_fir']:+.5f}  TH {R['wbar_fir_th']:+.5f}"
              f"   regr-slope {R['slope_fir']:.3f}"
              f"   P(neg) MC {R['fracneg_fir_mc']:.3f} TH 0.000")
        print(f" T5 WCPML rel-bias  mean MC {R['wbar_wc']:+.5f}   (target 0)")
        print(f" T3 regret  TH {regret_th:.6f}   MLE {R['reg_m']:.6f}   "
              f"Firth {R['reg_f']:.6f}   prob-scale-corrected {R['reg_kz']:.6f}")
        print(f" T4 W*={W_star:.4f}   value bias: MLE {R['val_m']-W_star:+.4f} "
              f"(TH {valbias_mle_th:+.4f})  Firth {R['val_f']-W_star:+.4f} "
              f"(TH {valbias_fir_th:+.4f})  WCPML {R['val_w']-W_star:+.4f}")
    return R


if __name__ == "__main__":
    import sys
    for n, lam, reps in [(275_638, 224, 4000), (275_638, 1122, 4000),
                         (275_638, 5152, 3000), (275_638, 18139, 2000)]:
        run(n, lam, reps)
        sys.stdout.flush()
