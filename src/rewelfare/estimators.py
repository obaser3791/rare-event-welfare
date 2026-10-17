"""
Estimators for logistic rare-event models.

All fits share a common Fisher-scoring backbone:

    fit_ml     : plain maximum likelihood.
    fit_firth  : Firth (1993) penalized maximum likelihood; fitted risk has
                 second-order relative bias q(x)/(2 lambda_n) > 0.
    fit_flic   : Firth slopes with the intercept re-estimated by maximum
                 likelihood (FLIC, Puhr et al. 2017). Aggregate-calibrated
                 on the sample, but its relative bias (q(x) - K)/(2 lambda_n)
                 is nonzero pointwise (Proposition 12 of the paper).
    fit_wcpml  : welfare-calibrated penalized maximum likelihood,
                 p_check(x) = Lambda(x' beta_tilde) * exp(-q_tilde(x) / (2 lambda_tilde)),
                 Definition 1 / equation (6.2) of the paper. The correction
                 varies with x and removes the pointwise bias to O(lambda^-2).

Each fit returns a Fit dataclass whose fields are used by policy.py
and diagnostics.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# --------------------------------------------------------------------
# Link
# --------------------------------------------------------------------
def logistic(eta: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function 1/(1+exp(-eta))."""
    eta = np.asarray(eta, dtype=float)
    out = np.empty_like(eta)
    pos = eta >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-eta[pos]))
    e = np.exp(eta[~pos])
    out[~pos] = e / (1.0 + e)
    return out


# --------------------------------------------------------------------
# Result container
# --------------------------------------------------------------------
@dataclass
class Fit:
    """Container returned by fit_ml, fit_firth, fit_wcpml."""

    coef: np.ndarray             # (k,) coefficient vector
    fitted_prob: np.ndarray      # (n,) fitted P(Y=1|X) on training set
    information: np.ndarray      # (k,k) Fisher information X' W X
    method: str                  # "ml" | "firth" | "wcpml"
    iterations: int
    converged: bool
    calibration: float = 1.0     # sample-average multiplicative factor (wcpml, flic)
    omega_inv: Optional[np.ndarray] = None   # inverse event-weighted moment (wcpml)
    lam_tilde: Optional[float] = None        # fitted expected event count (wcpml)
    X: Optional[np.ndarray] = None
    y: Optional[np.ndarray] = None


# --------------------------------------------------------------------
# Newton-Raphson / Fisher scoring core
# --------------------------------------------------------------------
def _check_inputs(X: np.ndarray, y: np.ndarray) -> None:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be 2-dimensional.")
    if y.ndim != 1 or len(y) != X.shape[0]:
        raise ValueError("y must be length nrow(X).")
    if not np.all(np.isin(y, [0.0, 1.0])):
        raise ValueError("y must be binary (0/1).")
    if not np.all(np.isfinite(X)):
        raise ValueError("X must be finite.")


def _fisher_score(
    X: np.ndarray,
    y: np.ndarray,
    beta: np.ndarray,
    firth: bool,
    ridge: float = 1e-10,
):
    """
    Compute Fisher information I, score s, and the hat-matrix
    diagonal h (only when firth=True) at the current beta.
    """
    eta = X @ beta
    p = logistic(eta)
    w = p * (1.0 - p)
    # Information X' W X with a tiny ridge for numerical stability.
    I = (X * w[:, None]).T @ X + ridge * np.eye(X.shape[1])
    if firth:
        # h_i = w_i * x_i' I^{-1} x_i, the hat-matrix diagonal.
        # Compute I^{-1} X' once via a solve.
        try:
            IinvXT = np.linalg.solve(I, X.T)   # (k,n)
        except np.linalg.LinAlgError:
            IinvXT = np.linalg.pinv(I) @ X.T
        h = w * np.einsum("ij,ji->i", X, IinvXT)
        # Firth score: X' ( y - p + h*(0.5 - p) )
        score = X.T @ (y - p + h * (0.5 - p))
    else:
        score = X.T @ (y - p)
    return I, score, p


def _newton(
    X: np.ndarray,
    y: np.ndarray,
    firth: bool,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> Fit:
    k = X.shape[1]
    beta = np.zeros(k)
    converged = False
    for it in range(1, max_iter + 1):
        I, s, _ = _fisher_score(X, y, beta, firth=firth)
        try:
            step = np.linalg.solve(I, s)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(I) @ s
        beta_new = beta + step
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            converged = True
            break
        beta = beta_new
    eta = X @ beta
    p = logistic(eta)
    w = p * (1.0 - p)
    I = (X * w[:, None]).T @ X
    method = "firth" if firth else "ml"
    return Fit(
        coef=beta,
        fitted_prob=p,
        information=I,
        method=method,
        iterations=it,
        converged=converged,
        X=X,
        y=y,
    )


# --------------------------------------------------------------------
# Public estimators
# --------------------------------------------------------------------
def fit_ml(X: np.ndarray, y: np.ndarray,
           max_iter: int = 100, tol: float = 1e-8) -> Fit:
    """Plain maximum-likelihood logistic fit."""
    _check_inputs(X, y)
    return _newton(np.asarray(X, float), np.asarray(y, float),
                   firth=False, max_iter=max_iter, tol=tol)


def fit_firth(X: np.ndarray, y: np.ndarray,
              max_iter: int = 100, tol: float = 1e-8) -> Fit:
    """Firth (1993) penalized maximum-likelihood fit."""
    _check_inputs(X, y)
    return _newton(np.asarray(X, float), np.asarray(y, float),
                   firth=True, max_iter=max_iter, tol=tol)


def _wc_factor(X: np.ndarray, omega_inv: np.ndarray, lam_tilde: float) -> np.ndarray:
    quad = np.einsum("ij,jk,ik->i", X, omega_inv, X)
    return np.exp(-quad / (2.0 * lam_tilde))


def fit_wcpml(X: np.ndarray, y: np.ndarray,
              max_iter: int = 100, tol: float = 1e-8) -> Fit:
    """
    Welfare-calibrated penalized maximum-likelihood estimator
    (Baser 2026, equations (6.1) and (6.2)):

        p_tilde(x)   = Lambda(x' beta_tilde)                (Firth fit)
        q_tilde      = n^{-1} sum_i p_tilde(x_i),  lambda_tilde = n q_tilde
        Omega_tilde  = n^{-1} sum_i {p_tilde(x_i) / q_tilde} x_i x_i'
        p_check(x)   = p_tilde(x) * exp{ -x' Omega_tilde^{-1} x / (2 lambda_tilde) }

    The coefficient vector is the Firth vector, so the index rule
    1{x' beta_tilde >= theta} is unchanged ("rank on the penalized index,
    value on the calibrated risk").
    """
    X = np.asarray(X, float)
    base = fit_firth(X, y, max_iter=max_iter, tol=tol)
    pt = base.fitted_prob
    n = X.shape[0]
    q_t = float(pt.mean())
    if q_t <= 0:
        raise RuntimeError("Firth fit produced non-positive event mass.")
    lam_t = n * q_t
    Om = (X * (pt / q_t)[:, None]).T @ X / n
    Om_inv = np.linalg.inv(Om)
    fac = _wc_factor(X, Om_inv, lam_t)
    base.fitted_prob = pt * fac
    base.calibration = float(base.fitted_prob.sum() / pt.sum())
    base.omega_inv = Om_inv
    base.lam_tilde = lam_t
    base.method = "wcpml"
    return base


def fit_flic(X: np.ndarray, y: np.ndarray,
             max_iter: int = 100, tol: float = 1e-10) -> Fit:
    """
    Firth logistic regression with intercept correction (Puhr et al. 2017):
    keep the Firth slopes, re-estimate the intercept by maximum likelihood
    with the slope index as an offset, so that sum_i p(x_i) = sum_i y_i.
    The first column of X must be the intercept.
    """
    X = np.asarray(X, float); y = np.asarray(y, float)
    if not np.allclose(X[:, 0], 1.0):
        raise ValueError("fit_flic requires an intercept in the first column of X.")
    base = fit_firth(X, y, max_iter=max_iter, tol=tol)
    off = X[:, 1:] @ base.coef[1:]
    a = float(base.coef[0])
    for _ in range(200):
        p = logistic(a + off)
        step = (y.sum() - p.sum()) / max(float((p * (1 - p)).sum()), 1e-300)
        a += step
        if abs(step) < tol:
            break
    coef = base.coef.copy(); coef[0] = a
    p = logistic(X @ coef)
    base.calibration = float(p.sum() / base.fitted_prob.sum())
    base.coef = coef
    base.fitted_prob = p
    base.information = (X * (p * (1 - p))[:, None]).T @ X
    base.method = "flic"
    return base


def predict_prob(fit: Fit, newX: Optional[np.ndarray] = None) -> np.ndarray:
    """Predicted probabilities on new design newX (or fit.X if None)."""
    if newX is None:
        return fit.fitted_prob
    newX = np.asarray(newX, float)
    p = logistic(newX @ fit.coef)
    if fit.method == "wcpml":
        p = p * _wc_factor(newX, fit.omega_inv, fit.lam_tilde)
    return p
