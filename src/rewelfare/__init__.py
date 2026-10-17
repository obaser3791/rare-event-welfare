"""
rewelfare: welfare-calibrated estimation for rare events.

Reference implementation of the three estimators from

    O. Baser (2026), "Welfare-Calibrated Estimation for Rare Events."

Public API:
    logistic(eta)                    logistic link
    fit_ml(X, y)                     maximum likelihood
    fit_firth(X, y)                  Firth penalized maximum likelihood
    fit_flic(X, y)                   Firth with intercept correction (FLIC)
    fit_wcpml(X, y)                  welfare-calibrated PML
    predict_prob(fit, X)             fitted probabilities on new X
    policy_rule(fit, X, tau_over_b)  threshold policy 1{p_hat(x) >= tau/b}
    welfare_value(fit, X, tau_over_b, y)   sample analog of W(d)
    regret_at(fit, X, tau_over_b, p_true)  finite-sample regret vs oracle

The three estimators return a common `Fit` dataclass so the same
downstream routines (predict, policy, valuation, regret) work with
any of them.
"""
from .estimators import (
    Fit,
    logistic,
    fit_ml,
    fit_firth,
    fit_flic,
    fit_wcpml,
    predict_prob,
)
from .policy import (
    policy_rule,
    welfare_value,
    regret_at,
)
from .diagnostics import (
    check_trace_identity,
)

__all__ = [
    "Fit",
    "logistic",
    "fit_ml",
    "fit_firth",
    "fit_flic",
    "fit_wcpml",
    "predict_prob",
    "policy_rule",
    "welfare_value",
    "regret_at",
    "check_trace_identity",
]

__version__ = "0.2.0"
