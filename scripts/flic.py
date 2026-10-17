"""
Intercept recalibration (FLIC, Puhr et al. 2017) versus welfare calibration.

Theory (Proposition 12 of the JoE version): FLIC refits the intercept by ML
holding the Firth slopes fixed, so sum_i p_FLIC(x_i) = sum_i y_i exactly.
Its second-order relative bias is (q(x) - K)/(2 lam): zero on average over
the events, but pointwise nonzero, positive where q(x) > K and negative where
q(x) < K.  In the binary design x = (1, z)' this is (1/pi - 2)/(2 lam) for the
attribute group and (1/(1-pi) - 2)/(2 lam) for the complement.

Part A: two-cell design calibrated to the transfusion row (lam = 224, K = 2).
Part B: the K = 4 Monte Carlo design of Section 8 at lam = 224.
"""
import json, sys
import numpy as np
import final_mc as F


def flic_intercept(X, nc, yc, slopes, a0):
    off = X[:, 1:] @ slopes
    a = a0
    for _ in range(200):
        p = 1 / (1 + np.exp(-(a + off)))
        g = yc.sum() - nc @ p
        h = nc @ (p * (1 - p))
        s = g / h
        a += s
        if abs(s) < 1e-13:
            break
    return a


def wcpml(X, w, n, pf):
    qs = pf @ w; ls = n * qs
    Om_s = (X * (w * (pf / qs))[:, None]).T @ X
    qd = np.einsum("ij,jk,ik->i", X, np.linalg.inv(Om_s), X)
    return pf * np.exp(-qd / (2 * ls))


def part_a(slope=0.90, prev=0.0655, n=275_638, lam_t=224, reps=40_000, seed=11):
    X = np.array([[1.0, 0.0], [1.0, 1.0]]); w = np.array([1 - prev, prev])
    b0 = F.solve_intercept(X, w, np.array([slope]), lam_t / n)
    beta = np.array([b0, slope]); p = 1 / (1 + np.exp(-(X @ beta)))
    q = p @ w; lam = n * q; m = p / q; pi = float(w[1] * m[1]); K = 2
    th = dict(firth=[1 / (2 * lam * (1 - pi)), 1 / (2 * lam * pi)],
              flic=[(1 / (1 - pi) - K) / (2 * lam), (1 / pi - K) / (2 * lam)])
    nc = np.round(n * w).astype(np.int64); nc[0] += n - nc.sum()
    rng = np.random.default_rng(seed)
    acc = {k: [] for k in ("ml", "firth", "flic", "wc")}
    cnt = {k: [] for k in acc}
    for _ in range(reps):
        yc = rng.binomial(nc, p)
        bh = F.fit(X, nc, yc); bf = F.fit(X, nc, yc, firth=True)
        if bh is None or bf is None or np.max(np.abs(bh - beta)) > 25:
            continue
        pm = 1 / (1 + np.exp(-(X @ bh))); pf = 1 / (1 + np.exp(-(X @ bf)))
        a = flic_intercept(X, nc, yc, bf[1:], bf[0])
        pl = 1 / (1 + np.exp(-(a + X[:, 1:] @ bf[1:])))
        pw = wcpml(X, w, n, pf)
        for k, v in (("ml", pm), ("firth", pf), ("flic", pl), ("wc", pw)):
            acc[k].append(v / p - 1); cnt[k].append(nc @ v - nc @ p)
    out = dict(pi=pi, lam=float(lam), reps=len(acc["ml"]), theory=th)
    for k in acc:
        A = np.asarray(acc[k]); C = np.asarray(cnt[k])
        out[k] = dict(mean=(100 * A.mean(0)).tolist(),
                      se=(100 * A.std(0, ddof=1) / np.sqrt(len(A))).tolist(),
                      count=float(C.mean()), count_se=float(C.std(ddof=1) / np.sqrt(len(C))))
    return out


def part_b(lam_target=224, reps=20000, n=F.N_COHORT, b2=(0.30, 0.60, 0.85),
           theta_q=0.85, seed=99):
    rng = np.random.default_rng(seed)
    X, w = F.cells(); b2 = np.asarray(b2, float)
    beta = np.concatenate([[F.solve_intercept(X, w, b2, lam_target / n)], b2])
    p = 1 / (1 + np.exp(-(X @ beta))); q = p @ w; lam = n * q; m = p / q
    Om = (X * (w * m)[:, None]).T @ X; Omi = np.linalg.inv(Om)
    quad = np.einsum("ij,jk,ik->i", X, Omi, X); K = X.shape[1]
    eta = X @ beta; o = np.argsort(eta); cw = np.cumsum(w[o])
    theta = eta[o][np.searchsorted(cw, theta_q)]
    t = 1 / (1 + np.exp(-theta)); tau = n * t; mu = n * p
    dstar = (eta >= theta).astype(float); W_star = w @ (dstar * (mu - tau))
    th_flic = (quad - K) / (2 * lam)
    th_val_flic = 100 * (w @ (dstar * mu * (quad - K)) / (2 * lam)) / W_star
    th_val_firth = 100 * (w @ (dstar * mu * quad) / (2 * lam)) / W_star
    nc = np.round(n * w).astype(np.int64); nc[0] += n - nc.sum()
    comorb = X[:, 3] == 1
    rbl, rbw, rbf, vl, vw, vf = [], [], [], [], [], []
    for _ in range(reps):
        yc = rng.binomial(nc, p)
        bf = F.fit(X, nc, yc, firth=True)
        if bf is None:
            continue
        pf = 1 / (1 + np.exp(-(X @ bf)))
        a = flic_intercept(X, nc, yc, bf[1:], bf[0])
        pl = 1 / (1 + np.exp(-(a + X[:, 1:] @ bf[1:])))
        pw = wcpml(X, w, n, pf)
        rbf.append(pf / p - 1); rbl.append(pl / p - 1); rbw.append(pw / p - 1)
        vf.append(w @ (dstar * (n * pf - tau)))
        vl.append(w @ (dstar * (n * pl - tau)))
        vw.append(w @ (dstar * (n * pw - tau)))
    R = len(rbl)
    rbl = np.asarray(rbl); rbw = np.asarray(rbw); rbf = np.asarray(rbf)
    em = w * m
    def grp(A, mask):  # event-weighted mean relative bias within a group, per cent
        v = A.mean(0); ww = em * mask
        return float(100 * (ww @ v) / ww.sum())
    def grp_se(A, mask):
        ww = em * mask; s = (A @ ww) / ww.sum()
        return float(100 * s.std(ddof=1) / np.sqrt(len(s)))
    def th_grp(v, mask):
        ww = em * mask; return float(100 * (ww @ v) / ww.sum())
    ones = np.ones_like(w, dtype=bool)
    def ms(a):
        a = 100 * (np.asarray(a) - W_star) / W_star
        return float(a.mean()), float(a.std(ddof=1) / np.sqrt(len(a)))
    out = dict(lam=float(lam), reps=R, K=K,
               flic_all=grp(rbl, ones), flic_all_se=grp_se(rbl, ones),
               flic_com=grp(rbl, comorb), flic_com_se=grp_se(rbl, comorb),
               flic_non=grp(rbl, ~comorb), flic_non_se=grp_se(rbl, ~comorb),
               th_flic_com=th_grp(th_flic, comorb), th_flic_non=th_grp(th_flic, ~comorb),
               firth_com=grp(rbf, comorb), firth_non=grp(rbf, ~comorb),
               th_firth_com=th_grp(quad / (2 * lam), comorb),
               th_firth_non=th_grp(quad / (2 * lam), ~comorb),
               wc_com=grp(rbw, comorb), wc_com_se=grp_se(rbw, comorb),
               wc_non=grp(rbw, ~comorb), wc_non_se=grp_se(rbw, ~comorb),
               val_flic=ms(vl), val_wc=ms(vw), val_firth=ms(vf),
               th_val_flic=float(th_val_flic), th_val_firth=float(th_val_firth),
               slope_flic=float(np.polyfit(th_flic, rbl.mean(0), 1, w=np.sqrt(em))[0]),
               # paired differences against the welfare-calibrated fit, which is
               # unbiased to O(lam^-2); differencing removes common sampling noise
               d_com=grp(rbl - rbw, comorb), d_com_se=grp_se(rbl - rbw, comorb),
               d_non=grp(rbl - rbw, ~comorb), d_non_se=grp_se(rbl - rbw, ~comorb),
               dv_flic=ms(np.asarray(vl) - np.asarray(vw) + W_star),
               dv_firth=ms(np.asarray(vf) - np.asarray(vw) + W_star))
    return out


if __name__ == "__main__":
    A = part_a(); print(json.dumps(A, indent=1)); sys.stdout.flush()
    B = part_b(); print(json.dumps(B, indent=1))
    json.dump(dict(two_cell=A, k4=B), open("flic_results.json", "w"), indent=1)
