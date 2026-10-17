"""Diagnostics: (a) does lam*Regret/tau converge to (1/2)tr(M Om^-1)?
(b) is the second-order relative-risk bias expansion right when evaluated as a
stochastic expansion (trimmed mean) rather than an exact moment?"""
import numpy as np
import theory_check as tc

NAGE = 500


def cells(p_commercial=0.7938, p_comorb=0.0655, nage=NAGE):
    from scipy.special import ndtri
    e = np.linspace(0, 1, nage + 1)[1:-1]
    cuts = ndtri(e)
    lo = np.concatenate([[-np.inf], cuts]); hi = np.concatenate([cuts, [np.inf]])
    phi = lambda z: np.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi)
    reps = (np.where(np.isfinite(lo), phi(lo), 0) - np.where(np.isfinite(hi), phi(hi), 0)) * nage
    rows, wts = [], []
    for cm, wc in ((1.0, p_commercial), (0.0, 1 - p_commercial)):
        for kk, wk in ((1.0, p_comorb), (0.0, 1 - p_comorb)):
            for a in reps:
                rows.append([1.0, cm, a, kk]); wts.append(wc * wk / nage)
    X = np.array(rows); w = np.array(wts); return X, w / w.sum()


def one(n, lam_t, reps, beta2=(0.30, 0.60, 0.85), theta_q=0.85, seed=5, bw=None):
    rng = np.random.default_rng(seed)
    X, w = cells()
    beta2 = np.asarray(beta2, float)
    b1 = tc.solve_intercept(X, w, beta2, lam_t / n)
    beta = np.concatenate([[b1], beta2])
    P = tc.pop_objects(X, w, beta, n)
    lam, Omi, quad, lin = P["lam"], P["Omi"], P["quad"], P["lin"]
    eta = X @ beta
    o = np.argsort(eta); cwt = np.cumsum(w[o])
    theta = eta[o][np.searchsorted(cwt, theta_q)]
    t = 1 / (1 + np.exp(-theta)); tau = n * t
    mu = n * P["p"]; dstar = (eta >= theta).astype(float)
    Ws = w @ (dstar * (mu - tau))
    if bw is None:
        bw = 1.5 * np.std(eta) / np.sqrt(lam)      # shrink with lam
    M = tc.boundary_form(X, w, beta, theta, bw=bw)
    trM = np.trace(M @ Omi)

    ncell = np.round(n * w).astype(np.int64); ncell[0] += n - ncell.sum()
    th_hat = np.log(t / (1 - t))
    regs_m, regs_f, regs_kz, rbm, rbf, rbw, bhs = [], [], [], [], [], [], []
    for _ in range(reps):
        yc = rng.binomial(ncell, P["p"])
        bh = tc.fit(X, ncell, yc); bf = tc.fit(X, ncell, yc, firth=True)
        if bh is None or bf is None:
            continue
        bhs.append(bh)
        for L, bb in ((regs_m, bh), (regs_f, bf)):
            d = (X @ bb >= th_hat).astype(float)
            L.append(Ws - w @ (d * (mu - tau)))
        pf = 1 / (1 + np.exp(-(X @ bf))); pm = 1 / (1 + np.exp(-(X @ bh)))
        qs = pf @ w; lam_s = n * qs
        Om_s = (X * (w * (pf / qs))[:, None]).T @ X
        qd = np.einsum("ij,jk,ik->i", X, np.linalg.inv(Om_s), X)
        pk = pf + 0.5 * pf * qd / lam_s
        regs_kz.append(Ws - w @ ((pk >= t).astype(float) * (mu - tau)))
        rbm.append(pm / P["p"] - 1); rbf.append(pf / P["p"] - 1)
        rbw.append(pf * np.exp(-qd / (2 * lam_s)) / P["p"] - 1)

    def trim(A, f=0.05):
        A = np.array(A); k = int(len(A) * f)
        S = np.sort(A, axis=0)
        return S[k:len(A) - k].mean(0)

    rbm_t, rbf_t, rbw_t = trim(rbm), trim(rbf), trim(rbw)
    th_m = (quad - lin) / (2 * lam); th_f = quad / (2 * lam)
    return dict(lam=lam, tau=tau, trM=trM, bw=bw, reps=len(bhs),
                lamreg_m=lam * np.mean(regs_m) / tau,
                lamreg_f=lam * np.mean(regs_f) / tau,
                lamreg_kz=lam * np.mean(regs_kz) / tau,
                half_trM=0.5 * trM,
                idxb_mc=np.array(bhs).mean(0) - beta,
                idxb_th=-(Omi @ P["om"]) / (2 * lam),
                rb_slope_m=np.polyfit(th_m, rbm_t, 1)[0],
                rb_slope_f=np.polyfit(th_f, rbf_t, 1)[0],
                rb_mean_m=(w @ rbm_t, w @ th_m),
                rb_mean_f=(w @ rbf_t, w @ th_f),
                rb_mean_w=w @ rbw_t,
                pneg_m=(w @ (rbm_t < 0), w @ (th_m < 0)),
                pneg_f=w @ (rbf_t < 0))


if __name__ == "__main__":
    import sys
    print("lam       lam*Reg/tau  MLE     Firth   probcorr | (1/2)tr(MOm^-1)  bw")
    for lam_t, reps in [(200, 3000), (800, 3000), (3200, 2000), (12800, 1200), (51200, 800)]:
        r = one(275_638, lam_t, reps)
        print(f"{r['lam']:9.0f}   {r['lamreg_m']:7.4f} {r['lamreg_f']:7.4f} "
              f"{r['lamreg_kz']:7.4f} | {r['half_trM']:7.4f}  bw={r['bw']:.4f}")
        print(f"    idx bias MC {np.array2string(r['idxb_mc'],precision=5)} "
              f"TH {np.array2string(r['idxb_th'],precision=5)}")
        print(f"    relbias MLE  trimmed-mean {r['rb_mean_m'][0]:+.5f} th {r['rb_mean_m'][1]:+.5f}"
              f"  slope {r['rb_slope_m']:.3f}  P(neg) {r['pneg_m'][0]:.3f}/{r['pneg_m'][1]:.3f}")
        print(f"    relbias Firth trimmed-mean {r['rb_mean_f'][0]:+.5f} th {r['rb_mean_f'][1]:+.5f}"
              f"  slope {r['rb_slope_f']:.3f}  P(neg) {r['pneg_f']:.3f}")
        print(f"    relbias WCPML trimmed-mean {r['rb_mean_w']:+.5f}")
        sys.stdout.flush()
