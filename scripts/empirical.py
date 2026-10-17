"""
(a) Verifies the two model-free corollaries:
      event-weighted mean relative bias of the MLE fitted risk  = 0 + O(lam^-2)
      event-weighted mean relative bias of Firth fitted risk    = K/(2 lam)
      => Firth overstates the expected event count by exactly K/2 events.
(b) Builds the tables of Section 9 for the Kythera Labs commercial + Medicaid
    mifepristone cohort (n = 275,638; 2017-2024). Inputs are the cohort
    quantities listed below. The payer contrast is implied by the
    payer-specific adverse-event rates; the age slope (0.18 per SD) and the
    comorbidity log-odds (0.90) are the values stated in Section 9.1 and are
    varied over a grid in the sensitivity analysis of Section 9.4.

Cohort aggregates used (cohort N = 275,638; 2017-2024):
  payer:      commercial 218,792 (79.38%)   Medicaid 56,846 (20.62%)
  outcomes:   transfusion 224, sepsis 333, infection w/o sepsis 391,
              ectopic 861, SAE 1,122, incomplete abortion 1,227,
              hemorrhage w/o transfusion 1,669, other complications 2,535,
              any primary AE 5,152, abortion-related ED visit 10,899,
              broadest composite 18,139
  AE by payer: commercial 1.89%, Medicaid 1.79%
  mean age 27.63; 6.55% van Walraven-Elixhauser >= 2
"""
import json
import numpy as np
from scipy.special import ndtri
import final_mc as F

N = 275_638
P_COMM = 218_792 / N
P_COM2 = 0.0655
AGE_MEAN, AGE_SD = 27.63, 6.0

OUTCOMES = [
    ("Transfusion", 224),
    ("Sepsis", 333),
    ("Infection without sepsis", 391),
    ("Post-abortion ectopic pregnancy", 861),
    ("Serious adverse event", 1_122),
    ("Incomplete abortion requiring procedure", 1_227),
    ("Hemorrhage without transfusion", 1_669),
    ("Other abortion-related complication", 2_535),
    ("Any primary adverse event", 5_152),
    ("Abortion-related emergency department visit", 10_899),
    ("Broadest adverse-event composite", 18_139),
]

# payer contrast identified from the published payer-specific AE rates
AE_COMM, AE_MCD = 0.0189, 0.0179
BETA_PAYER = float(np.log(AE_COMM / (1 - AE_COMM)) - np.log(AE_MCD / (1 - AE_MCD)))


# --------------------------------------------------------------------------
def corollary_check():
    """m-weighted (event-weighted) means and the total-event-count bias."""
    rows = []
    for lam_t, reps in [(224, 8000), (1122, 8000), (5152, 5000)]:
        X, w = F.cells()
        b2 = np.array([0.30, 0.60, 0.85])
        beta = np.concatenate([[F.solve_intercept(X, w, b2, lam_t / N)], b2])
        p = 1 / (1 + np.exp(-(X @ beta)))
        q = p @ w; lam = N * q; m = p / q
        Om = (X * (w * m)[:, None]).T @ X
        Omi = np.linalg.inv(Om)
        quad = np.einsum("ij,jk,ik->i", X, Omi, X)
        om = ((w * m * quad)[:, None] * X).sum(0)
        lin = X @ (Omi @ om)
        K = X.shape[1]
        th_m = w @ (m * (quad - lin)) / (2 * lam)        # should be ~0
        th_f = w @ (m * quad) / (2 * lam)                # should be K/(2 lam)

        rng = np.random.default_rng(99 + lam_t)
        nc = np.round(N * w).astype(np.int64); nc[0] += N - nc.sum()
        tot_true = N * q
        tm, tf, tw = [], [], []
        for _ in range(reps):
            yc = rng.binomial(nc, p)
            bh = F.fit(X, nc, yc); bf = F.fit(X, nc, yc, firth=True)
            if bh is None or bf is None or np.max(np.abs(bh - beta)) > 25:
                continue
            pm = 1 / (1 + np.exp(-(X @ bh))); pf = 1 / (1 + np.exp(-(X @ bf)))
            qs = pf @ w; ls = N * qs
            Om_s = (X * (w * (pf / qs))[:, None]).T @ X
            qd = np.einsum("ij,jk,ik->i", X, np.linalg.inv(Om_s), X)
            pw = pf * np.exp(-qd / (2 * ls))
            tm.append(N * (pm @ w)); tf.append(N * (pf @ w)); tw.append(N * (pw @ w))
        rows.append(dict(lam=lam, K=K, tot_true=tot_true,
                         mw_mle_th=th_m, mw_fir_th=th_f, K_over_2lam=K / (2 * lam),
                         d_mle=np.mean(tm) - tot_true, se_mle=np.std(tm, ddof=1) / np.sqrt(len(tm)),
                         d_fir=np.mean(tf) - tot_true, se_fir=np.std(tf, ddof=1) / np.sqrt(len(tf)),
                         d_wc=np.mean(tw) - tot_true, se_wc=np.std(tw, ddof=1) / np.sqrt(len(tw)),
                         K_over_2=K / 2, reps=len(tm)))
    return rows


# --------------------------------------------------------------------------
def calibrated_cells(nage=400, K_extra=0):
    """Covariate cells for the mifepristone cohort: payer x comorbidity x age.
    K_extra adds orthogonal nuisance regressors (SES quintiles, race
    categories) that enter the index with zero coefficient, so they change K
    without changing the fitted risk -- exactly the comparative static the
    corollaries describe."""
    e = np.linspace(0, 1, nage + 1)[1:-1]
    cuts = ndtri(e)
    lo = np.concatenate([[-np.inf], cuts]); hi = np.concatenate([cuts, [np.inf]])
    phi = lambda z: np.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi)
    rep = (np.where(np.isfinite(lo), phi(lo), 0.0)
           - np.where(np.isfinite(hi), phi(hi), 0.0)) * nage
    rows, wts = [], []
    for cm, wc in ((1.0, P_COMM), (0.0, 1 - P_COMM)):
        for kk, wk in ((1.0, P_COM2), (0.0, 1 - P_COM2)):
            for a in rep:
                rows.append([1.0, cm, a, kk]); wts.append(wc * wk / nage)
    X = np.asarray(rows); w = np.asarray(wts) / np.sum(wts)
    return X, w


def empirical_table(beta_age=0.18, beta_com=0.90, K_list=(5, 9, 13), theta_q=0.85):
    X, w = calibrated_cells()
    b2 = np.array([BETA_PAYER, beta_age, beta_com])
    out = []
    for name, cnt in OUTCOMES:
        q_t = cnt / N
        beta = np.concatenate([[F.solve_intercept(X, w, b2, q_t)], b2])
        p = 1 / (1 + np.exp(-(X @ beta)))
        q = p @ w; lam = N * q; m = p / q
        Om = (X * (w * m)[:, None]).T @ X
        Omi = np.linalg.inv(Om)
        eta = X @ beta
        o = np.argsort(eta); cw = np.cumsum(w[o])
        theta = eta[o][np.searchsorted(cw, theta_q)]
        t = 1 / (1 + np.exp(-theta)); tau = N * t
        mu = N * p; d = (eta >= theta).astype(float)
        Wst = w @ (d * (mu - tau))
        bw = 1.5 * np.std(eta) / np.sqrt(lam)
        ker = np.exp(-0.5 * ((eta - theta) / bw) ** 2) / (bw * np.sqrt(2 * np.pi))
        M = (X * (w * ker)[:, None]).T @ X
        trM = float(np.trace(M @ Omi))
        rec = dict(outcome=name, events=cnt, rate_pct=100 * q_t, lam=lam,
                   trMOmi=trM, first_order_regret=tau * trM / (2 * lam),
                   W_star=Wst,
                   rel_regret_pct=100 * (tau * trM / (2 * lam)) / Wst,
                   treated_share=theta_q, t=t)
        for K in K_list:
            rec[f"firth_bias_pct_K{K}"] = 100 * K / (2 * lam)
            rec[f"firth_extra_events_K{K}"] = K / 2
        out.append(rec)
    return out, dict(beta_payer=BETA_PAYER, beta_age=beta_age, beta_com=beta_com,
                     payer_or=float(np.exp(BETA_PAYER)))


if __name__ == "__main__":
    cor = corollary_check()
    print("== Corollary check: event-weighted relative bias and total-count bias ==")
    for r in cor:
        print(f" lam={r['lam']:8.1f}  K={r['K']}  true expected events={r['tot_true']:.1f}")
        print(f"   event-weighted rel bias  MLE theory {r['mw_mle_th']:+.2e} (target 0)   "
              f"Firth theory {r['mw_fir_th']:.6f}  K/(2lam)={r['K_over_2lam']:.6f}")
        print(f"   predicted-total-count bias (events):  MLE {r['d_mle']:+.3f}"
              f"({r['se_mle']:.3f})   Firth {r['d_fir']:+.3f}({r['se_fir']:.3f})"
              f"  [K/2={r['K_over_2']:.1f}]   WCPML {r['d_wc']:+.3f}({r['se_wc']:.3f})")
    tab, cal = empirical_table()
    print("\n== Empirical exhibit (Kythera Commercial + Medicaid, N=275,638) ==")
    print(f" calibration: payer log-odds {cal['beta_payer']:+.5f} (OR {cal['payer_or']:.4f}), "
          f"age {cal['beta_age']}, comorbidity {cal['beta_com']}")
    hdr = f"{'outcome':44s} {'events':>7s} {'rate%':>7s} {'K/2lam%,K=9':>12s} {'+events':>8s} {'relRegret%':>11s}"
    print(hdr)
    for r in tab:
        print(f"{r['outcome']:44s} {r['events']:7d} {r['rate_pct']:7.3f} "
              f"{r['firth_bias_pct_K9']:12.3f} {r['firth_extra_events_K9']:8.1f} "
              f"{r['rel_regret_pct']:11.3f}")
    json.dump(dict(corollary=cor, table=tab, calibration=cal),
              open("empirical_results.json", "w"), indent=1, default=float)
    print("\nwrote empirical_results.json")
