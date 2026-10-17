"""
Verifies the distributional-incidence corollary.

Claim: if a binary attribute has event-weighted share pi, then in the two-
parameter design x = (1, z)' the quadratic form takes exactly two values,
  quad(1,1) = 1/pi        (minority / attribute present)
  quad(1,0) = 1/(1-pi)    (majority / attribute absent)
so the penalized relative inflation of fitted risk is 1/(2 lam pi) for the
minority and 1/(2 lam (1-pi)) for the majority, a ratio of (1-pi)/pi, while the
event-weighted average is exactly K/(2 lam) = 1/lam.

Calibrated to cohort aggregates: comorbidity (van Walraven-Elixhauser >= 2)
has population prevalence 6.55 per cent; n = 275,638; lam = 224 (transfusion).
"""
import numpy as np
import final_mc as F

N, PREV, LAM_T, REPS = 275_638, 0.0655, 224, 40_000
SLOPE = 0.90   # comorbidity log-odds, same calibration value as Table 4

X = np.array([[1.0, 0.0], [1.0, 1.0]])
w = np.array([1 - PREV, PREV])
b0 = F.solve_intercept(X, w, np.array([SLOPE]), LAM_T / N)
beta = np.array([b0, SLOPE])
p = 1 / (1 + np.exp(-(X @ beta)))
q = p @ w
lam = N * q
m = p / q
pi = float(w[1] * m[1])                      # event-weighted minority share
Om = (X * (w * m)[:, None]).T @ X
Omi = np.linalg.inv(Om)
quad = np.einsum("ij,jk,ik->i", X, Omi, X)
K = 2

print(f"population prevalence      {PREV:.4f}")
print(f"event-weighted share pi    {pi:.6f}")
print(f"lambda                     {lam:.1f}   (target {LAM_T})")
print(f"quad  majority  {quad[0]:.4f}   claim 1/(1-pi) = {1/(1-pi):.4f}")
print(f"quad  minority  {quad[1]:.4f}   claim 1/pi     = {1/pi:.4f}")
print(f"event-weighted mean quad {w @ (m * quad):.6f}  claim K = {K}")
print()
print("theoretical penalized relative inflation, per cent")
print(f"  majority {100*quad[0]/(2*lam):.4f}   closed form {100/(2*lam*(1-pi)):.4f}")
print(f"  minority {100*quad[1]/(2*lam):.4f}   closed form {100/(2*lam*pi):.4f}")
print(f"  ratio    {quad[1]/quad[0]:.3f}       closed form {(1-pi)/pi:.3f}")
print(f"  event-weighted mean {100*K/(2*lam):.4f}")
print()

rng = np.random.default_rng(7)
nc = np.round(N * w).astype(np.int64); nc[0] += N - nc.sum()
acc_f = np.zeros(2); acc_w = np.zeros(2); acc_m = np.zeros(2); ok = 0
sq_f = np.zeros(2)
for _ in range(REPS):
    yc = rng.binomial(nc, p)
    bh = F.fit(X, nc, yc); bf = F.fit(X, nc, yc, firth=True)
    if bh is None or bf is None or np.max(np.abs(bh - beta)) > 25:
        continue
    pm = 1 / (1 + np.exp(-(X @ bh)))
    pf = 1 / (1 + np.exp(-(X @ bf)))
    qs = pf @ w; ls = N * qs
    Om_s = (X * (w * (pf / qs))[:, None]).T @ X
    qd = np.einsum("ij,jk,ik->i", X, np.linalg.inv(Om_s), X)
    pw = pf * np.exp(-qd / (2 * ls))
    acc_m += pm / p; acc_f += pf / p; acc_w += pw / p
    sq_f += (pf / p) ** 2
    ok += 1
acc_m /= ok; acc_f /= ok; acc_w /= ok
se_f = np.sqrt(np.maximum(sq_f / ok - acc_f ** 2, 0) / ok)
print(f"Monte Carlo, {ok} usable replications, relative bias in per cent")
print(f"  penalized  majority {100*(acc_f[0]-1):+.4f} ({100*se_f[0]:.4f})"
      f"   theory {100*quad[0]/(2*lam):+.4f}")
print(f"  penalized  minority {100*(acc_f[1]-1):+.4f} ({100*se_f[1]:.4f})"
      f"   theory {100*quad[1]/(2*lam):+.4f}")
print(f"  max lik    majority {100*(acc_m[0]-1):+.4f}")
print(f"  max lik    minority {100*(acc_m[1]-1):+.4f}")
print(f"  calibrated majority {100*(acc_w[0]-1):+.4f}")
print(f"  calibrated minority {100*(acc_w[1]-1):+.4f}")

print("\n-- incidence across the eleven published outcome counts, K=2 --")
for name, cnt in [("transfusion", 224), ("serious adverse event", 1122),
                  ("any primary adverse event", 5152),
                  ("broadest composite", 18139)]:
    b0c = F.solve_intercept(X, w, np.array([SLOPE]), cnt / N)
    pc = 1 / (1 + np.exp(-(X @ np.array([b0c, SLOPE]))))
    qc = pc @ w; lc = N * qc; mc = pc / qc
    pic = float(w[1] * mc[1])
    print(f"  {name:26s} lam={lc:8.1f}  minority {100/(2*lc*pic):6.3f}%"
          f"   majority {100/(2*lc*(1-pic)):6.3f}%   ratio {(1-pic)/pic:5.2f}")
