# rare-event-welfare

Reference implementation and replication package for

> Onur Baser (2026), "Welfare-calibrated estimation for rare events: Targeting, adoption,
> and the limits of bias correction." Submitted to the Journal of Econometrics.

The paper introduces the welfare-calibrated penalized maximum likelihood
(WCPML) estimator for rare-event logistic models and derives the
second-order bias of fitted risk, the regret expansion, and the
program-valuation decomposition. This
repository provides:

1. A small Python library, `rewelfare`, that implements the four
   estimators discussed in the paper (maximum likelihood, Firth, intercept-recalibrated Firth and welfare-calibrated) on a common Fisher-scoring backbone.
2. The Monte Carlo, empirical, diagnostic, and figure scripts referenced
   in the Data and code availability statement.
3. Numerical verification of the trace identity (Lemmas 1 and 4) and the
   second-order expansions used in Section 3.

## Contents

```
rare-event-welfare/
├── src/rewelfare/
│   ├── __init__.py         public API
│   ├── estimators.py       fit_ml, fit_firth, fit_wcpml, predict_prob
│   ├── policy.py           policy_rule, welfare_value, regret_at
│   └── diagnostics.py      check_trace_identity
├── scripts/
│   ├── final_mc.py         Monte Carlo driver (Tables 1 to 3)
│   ├── theory_check.py     second-order bias verification
│   ├── empirical.py        application to the claims cohort (Section 9)
│   ├── figures.py          regenerates Figures 1 to 3
│   ├── diag.py             separation and diagnostic checks
│   └── minority.py         subgroup analysis
├── tests/
│   └── test_estimators.py  pytest suite
├── figures/                compiled figures shipped with the paper
├── pyproject.toml
├── LICENSE                 MIT
└── README.md
```

## Installation

```bash
git clone https://github.com/obaser3791/rare-event-welfare
cd rare-event-welfare
pip install -e .[scripts,test]
```

Python 3.9 or later is required. The core library depends only on
NumPy; the scripts additionally use pandas, matplotlib, and scipy.

## Quick start

```python
import numpy as np
from rewelfare import fit_ml, fit_firth, fit_wcpml, logistic

rng = np.random.default_rng(0)
n, k = 5000, 4
X = np.column_stack([np.ones(n), rng.normal(size=(n, k - 1))])
beta_true = np.array([-4.0, 0.6, -0.4, 0.3])   # roughly 2% event rate
y = (rng.uniform(size=n) < logistic(X @ beta_true)).astype(int)

fit_a = fit_ml(X, y)
fit_b = fit_firth(X, y)
fit_c = fit_wcpml(X, y)

print("ML     :", fit_a.coef)
print("Firth  :", fit_b.coef)
print("WCPML  :", fit_c.coef, "calibration =", fit_c.calibration)

# WCPML restores aggregate unbiasedness on the training sample:
assert abs(fit_c.fitted_prob.sum() - y.sum()) < 1e-8
```

## Policy targeting and program valuation

Given a threshold `tau_over_b` (the treatment cost divided by the
per-event benefit), the plug-in policy, its welfare, and the
finite-sample regret against the oracle are:

```python
from rewelfare import policy_rule, welfare_value, regret_at

tau_over_b = 0.05
d_hat = policy_rule(fit_c, X, tau_over_b)          # 0/1 policy
W_hat = welfare_value(fit_c, X, tau_over_b, y)     # sample welfare
p_true = logistic(X @ beta_true)
reg   = regret_at(fit_c, X, tau_over_b, p_true)    # E[(d*-d_hat)(bp-tau)]
```

## Reproducing the paper

Every table and figure can be regenerated from the driver scripts.
The application in Section 9 of the paper takes the cohort inputs listed at
the top of `scripts/empirical.py` (event counts for the eleven outcomes,
payer mix, age and comorbidity distribution, and the three index slopes).
The underlying Kythera Labs claims are licensed and are not distributed here.

```bash
# Monte Carlo evidence (Tables 1 to 3):
python scripts/final_mc.py
python scripts/theory_check.py

# Figures (writes into figures/):
python scripts/figures.py

# Intercept recalibration (Table 4):
python scripts/flic.py
python scripts/flic_twocell_paired.py

# Claims application and sensitivity (Tables 5 and 6, Section 9.4):
python scripts/empirical.py
python scripts/sensitivity.py
python scripts/diag.py
python scripts/minority.py
```

## Verifying the trace identity

The trace identity behind Lemmas 1 and 4 of the paper is

$$ \sum_{i=1}^n w_i \, x_i' I^{-1} x_i \;=\; K, $$

where $I = X'\mathrm{diag}(w)X$ and $w_i = p_i(1-p_i)$. The identity
holds at every $\beta$ and to machine precision:

```python
from rewelfare import check_trace_identity
out = check_trace_identity(X, fit_c.coef)
print(out)   # {"lhs": ..., "rhs": K, "diff": ~1e-12}
```

## Tests

```bash
pip install -e .[test]
pytest -q
```

## Citation

```bibtex
@article{Baser2026WCPML,
  author  = {Baser, Onur},
  title   = {Welfare-Calibrated Estimation for Rare Events},
  year    = {2026},
  note    = {Working paper.}
}
```

Also cited in the paper:

- Athey, S. and Wager, S. (2021), "Policy Learning With Observational Data,"
  Econometrica 89(1), 133-161.
- Firth, D. (1993), "Bias Reduction of Maximum Likelihood Estimates,"
  Biometrika 80(1), 27-38.
- King, G. and Zeng, L. (2001), "Logistic Regression in Rare Events Data,"
  Political Analysis 9(2), 137-163.
- Kitagawa, T. and Tetenov, A. (2018), "Who Should Be Treated? Empirical
  Welfare Maximization Methods for Treatment Choice," Econometrica 86(2),
  591-616.

## License

MIT. See LICENSE.
