"""Two-cell transfusion design (Table 6): paired differences against ML, which is
exactly unbiased in a saturated design (E[y_c/n_c] = p_c), isolate estimator bias."""
import json, numpy as np, final_mc as F
from flic import flic_intercept, wcpml
N, PREV, LAM, SLOPE, REPS = 275_638, 0.0655, 224, 0.90, 40_000
X = np.array([[1.0, 0.0], [1.0, 1.0]]); w = np.array([1 - PREV, PREV])
b0 = F.solve_intercept(X, w, np.array([SLOPE]), LAM / N); beta = np.array([b0, SLOPE])
p = 1 / (1 + np.exp(-(X @ beta))); q = p @ w; lam = N * q; m = p / q; pi = float(w[1] * m[1])
nc = np.round(N * w).astype(np.int64); nc[0] += N - nc.sum()
rng = np.random.default_rng(11); R = {k: [] for k in ("ml", "firth", "flic", "wc")}; C = {k: [] for k in R}
for _ in range(REPS):
    yc = rng.binomial(nc, p)
    bh = F.fit(X, nc, yc); bf = F.fit(X, nc, yc, firth=True)
    pm = 1 / (1 + np.exp(-(X @ bh))); pf = 1 / (1 + np.exp(-(X @ bf)))
    a = flic_intercept(X, nc, yc, bf[1:], bf[0]); pl = 1 / (1 + np.exp(-(a + X[:, 1:] @ bf[1:])))
    pw = wcpml(X, w, N, pf)
    for k, v in (("ml", pm), ("firth", pf), ("flic", pl), ("wc", pw)):
        R[k].append(v / p - 1); C[k].append(nc @ v - nc @ p)
R = {k: np.asarray(v) for k, v in R.items()}; C = {k: np.asarray(v) for k, v in C.items()}
out = dict(pi=pi, lam=float(lam), reps=REPS,
           theory=dict(firth=[100 / (2 * lam * (1 - pi)), 100 / (2 * lam * pi)],
                       flic=[100 * (1 / (1 - pi) - 2) / (2 * lam), 100 * (1 / pi - 2) / (2 * lam)],
                       wc=[0.0, 0.0]))
for k in R:
    d = R[k] - R["ml"]
    out[k] = dict(raw=(100 * R[k].mean(0)).tolist(), raw_se=(100 * R[k].std(0, ddof=1) / np.sqrt(REPS)).tolist(),
                  net=(100 * d.mean(0)).tolist(), net_se=(100 * d.std(0, ddof=1) / np.sqrt(REPS)).tolist(),
                  count=float(C[k].mean()), count_se=float(C[k].std(ddof=1) / np.sqrt(REPS)))
print(json.dumps(out, indent=1)); json.dump(out, open("flic_twocell.json", "w"), indent=1)
