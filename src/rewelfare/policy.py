"""
Policy targeting and program valuation.

These utilities implement the objects in Sections 4 and 5 of Baser
(2026):

    policy_rule(fit, X, tau_over_b)
        1{ p_hat(x) >= tau/b }, the plug-in threshold rule.

    welfare_value(fit, X, tau_over_b, y)
        Sample analog of the welfare functional
        W(d) = E[ d(X) ( b*Y - tau ) ]
        with b normalized to 1 so tau_over_b is the threshold.

    regret_at(fit, X, tau_over_b, p_true)
        Finite-sample regret W(d^*) - W(d_hat) when the true
        probabilities p_true are known (used in Monte Carlo).
"""
from __future__ import annotations

import numpy as np

from .estimators import Fit, predict_prob


def policy_rule(fit: Fit, X: np.ndarray, tau_over_b: float) -> np.ndarray:
    """Threshold policy: 1 if predicted probability is at least tau/b."""
    p = predict_prob(fit, X)
    return (p >= tau_over_b).astype(int)


def welfare_value(
    fit: Fit,
    X: np.ndarray,
    tau_over_b: float,
    y: np.ndarray,
    b: float = 1.0,
) -> float:
    """
    Sample analog W_hat(d) = (1/n) sum_i d(x_i) ( b y_i - tau ),
    with tau = tau_over_b * b.
    """
    d = policy_rule(fit, X, tau_over_b)
    tau = tau_over_b * b
    return float(np.mean(d * (b * np.asarray(y) - tau)))


def regret_at(
    fit: Fit,
    X: np.ndarray,
    tau_over_b: float,
    p_true: np.ndarray,
    b: float = 1.0,
) -> float:
    """
    Population regret at the true probabilities p_true:
        Regret = E[ (d^*(x) - d_hat(x)) ( b p_true(x) - tau ) ]
    approximated by the empirical mean on the supplied X.

    d^*(x) = 1{ p_true(x) >= tau/b } is the oracle rule.
    """
    p_true = np.asarray(p_true, float)
    d_star = (p_true >= tau_over_b).astype(int)
    d_hat = policy_rule(fit, X, tau_over_b)
    tau = tau_over_b * b
    return float(np.mean((d_star - d_hat) * (b * p_true - tau)))
