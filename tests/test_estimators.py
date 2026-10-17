"""
Tests for the three estimators in rewelfare.
"""
import numpy as np
import pytest

from rewelfare import (
    fit_ml, fit_firth, fit_flic, fit_wcpml,
    predict_prob, policy_rule, welfare_value, regret_at,
    check_trace_identity, logistic,
)


rng = np.random.default_rng(20260101)


def _simulate(n=5000, k=4, event_rate=0.02, seed=0):
    """Simulate a rare-event logistic design."""
    r = np.random.default_rng(seed)
    X = np.column_stack([np.ones(n), r.normal(size=(n, k - 1))])
    # Pick intercept so the marginal event rate matches event_rate.
    beta = np.concatenate([[0.0], r.normal(scale=0.5, size=k - 1)])
    # Adjust intercept by bisection.
    def rate(a):
        b = beta.copy(); b[0] = a
        return logistic(X @ b).mean()
    lo, hi = -20.0, 5.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if rate(mid) > event_rate:
            hi = mid
        else:
            lo = mid
    beta[0] = 0.5 * (lo + hi)
    p_true = logistic(X @ beta)
    y = (r.uniform(size=n) < p_true).astype(int)
    return X, y, beta, p_true


def test_ml_recovers_intercept_dense_case():
    n = 20000
    r = np.random.default_rng(1)
    X = np.column_stack([np.ones(n), r.normal(size=(n, 3))])
    beta_true = np.array([0.5, -0.3, 0.2, 0.7])
    p = logistic(X @ beta_true)
    y = (r.uniform(size=n) < p).astype(int)
    fit = fit_ml(X, y)
    assert fit.converged
    # Coefficients should be within a few standard errors.
    assert np.allclose(fit.coef, beta_true, atol=0.05)


def test_all_three_fit_a_rare_event_design():
    X, y, _, _ = _simulate(n=8000, k=3, event_rate=0.02, seed=42)
    for fitter in (fit_ml, fit_firth, fit_flic, fit_wcpml):
        fit = fitter(X, y)
        assert fit.converged
        assert fit.fitted_prob.shape == (X.shape[0],)
        assert np.all(fit.fitted_prob >= 0) and np.all(fit.fitted_prob <= 1)


def test_flic_is_aggregate_calibrated_on_sample():
    """FLIC (Puhr et al. 2017): sum(p) == sum(y) exactly on training data."""
    X, y, _, _ = _simulate(n=6000, k=3, event_rate=0.03, seed=7)
    fit = fit_flic(X, y)
    assert np.isclose(fit.fitted_prob.sum(), y.sum(), atol=1e-8)
    fb = fit_firth(X, y)
    assert np.allclose(fit.coef[1:], fb.coef[1:])


def test_firth_overstates_event_count_by_about_K_over_2():
    """Corollary 9: sum(p_firth) - sum(y) = sum_i h_i (1/2 - p_i), close to K/2."""
    X, y, _, _ = _simulate(n=20000, k=4, event_rate=0.01, seed=5)
    fb = fit_firth(X, y)
    excess = fb.fitted_prob.sum() - y.sum()
    assert 0.4 * X.shape[1] < excess < 0.5 * X.shape[1] + 1e-6


def test_wcpml_factor_is_pointwise_and_below_one():
    """Equation (6.2): p_check = p_tilde * exp(-q(x)/(2 lambda)), factor varies with x."""
    X, y, _, _ = _simulate(n=6000, k=3, event_rate=0.03, seed=7)
    fw = fit_wcpml(X, y); fb = fit_firth(X, y)
    ratio = fw.fitted_prob / fb.fitted_prob
    assert np.all(ratio < 1.0) and np.all(ratio > 0.0)
    assert ratio.std() > 0.0
    assert np.allclose(fw.coef, fb.coef)
    # sample-average removal of the K/2 excess, up to O(1) sampling terms
    assert abs(fw.fitted_prob.sum() - y.sum()) < abs(fb.fitted_prob.sum() - y.sum())


def test_firth_shrinks_toward_zero_versus_ml():
    """Firth penalty shrinks slope magnitude relative to ML in rare regime."""
    X, y, _, _ = _simulate(n=2000, k=3, event_rate=0.01, seed=11)
    fit_a = fit_ml(X, y)
    fit_b = fit_firth(X, y)
    # Slope coefficients: exclude intercept.
    assert np.linalg.norm(fit_b.coef[1:]) <= np.linalg.norm(fit_a.coef[1:]) + 1e-6


def test_trace_identity_holds_to_machine_precision():
    """Lemma 2 of the paper: sum w_i x_i' I^{-1} x_i = K, exactly."""
    r = np.random.default_rng(99)
    n, k = 4000, 5
    X = np.column_stack([np.ones(n), r.normal(size=(n, k - 1))])
    beta = r.normal(size=k, scale=0.3)
    out = check_trace_identity(X, beta)
    assert abs(out["diff"]) < 1e-9


def test_predict_prob_on_new_design_matches_calibration():
    X, y, _, _ = _simulate(n=3000, k=3, event_rate=0.02, seed=3)
    fit = fit_wcpml(X, y)
    # Predict on same X: should equal fitted_prob.
    p_hat = predict_prob(fit, X)
    assert np.allclose(p_hat, fit.fitted_prob)


def test_policy_and_welfare_signatures():
    X, y, _, p_true = _simulate(n=4000, k=3, event_rate=0.02, seed=17)
    fit = fit_firth(X, y)
    tau_over_b = 0.05
    d = policy_rule(fit, X, tau_over_b)
    assert set(np.unique(d)).issubset({0, 1})
    w = welfare_value(fit, X, tau_over_b, y)
    reg = regret_at(fit, X, tau_over_b, p_true)
    # Regret must be non-negative (up to Monte Carlo noise, small n).
    assert reg >= -1e-6
    assert np.isfinite(w)
