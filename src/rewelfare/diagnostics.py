"""
Numerical verification of Lemma 2 of Baser (2026) (the "trace
identity"): for a design matrix X and event probabilities p,

    sum_i p_i (1 - p_i) * x_i' I^{-1} x_i  =  K,

where I = X' diag(p(1-p)) X and K = ncol(X). Equivalently,
tr( W X (X'WX)^{-1} X' ) = K.

This is a global finite-sample identity, not an asymptotic one, and
holds at every beta. The routine below returns the two sides
separately so downstream tests can assert their equality to machine
precision.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from .estimators import logistic


def check_trace_identity(X: np.ndarray, beta: np.ndarray) -> Dict[str, float]:
    """
    Return a dictionary with:
        lhs  = sum_i w_i * x_i' I^{-1} x_i
        rhs  = ncol(X)
        diff = lhs - rhs
    """
    X = np.asarray(X, float)
    beta = np.asarray(beta, float)
    if X.shape[1] != beta.shape[0]:
        raise ValueError("X and beta dimensions must match.")
    p = logistic(X @ beta)
    w = p * (1.0 - p)
    I = (X * w[:, None]).T @ X
    IinvXT = np.linalg.solve(I, X.T)
    lhs = float(np.sum(w * np.einsum("ij,ji->i", X, IinvXT)))
    rhs = float(X.shape[1])
    return {"lhs": lhs, "rhs": rhs, "diff": lhs - rhs}
