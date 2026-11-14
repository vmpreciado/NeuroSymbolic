"""Controlled Fitzsimmons--Liu PDI-loss baseline for the swing application.

This is an independent implementation of the spectral positive-part objective
in Fitzsimmons and Liu (2024), Eq. (5). It is a common-class/common-verifier
adaptation, not a reproduction of their complete LyZNet/dReal experiment.
See comparison_sources/fitzsimmons_liu_2024.json for the precise mapping.

The common candidate is M = alpha I + L.T L, alpha = 0.1. Taking S = alpha I
in the positivity penalty of Eq. (5) makes that term identically zero. To
match the target-rate condition used by both arms, use

    R = Mdot + A.T M + M A + 2 rate M + raw_margin I.

The PDI objective averages squared positive parts of BOTH eigenvalues of R.
The loss is a floating-point training heuristic. Only the separate outward-
rounded verifier can certify a candidate.
"""
import autograd.numpy as np


DEFAULT_RAW_MARGIN = 0.01


def eigenvalues_2x2(c11, c12, c22):
    """Return both nominal eigenvalues, with a negligible gradient guard.

    The tiny radicand guard follows the existing learner's convention at
    repeated eigenvalues. It is not part of any certification calculation.
    """
    center = 0.5 * (c11 + c22)
    radius = np.sqrt((0.5 * (c11 - c22)) ** 2 + c12 ** 2 + 1e-300)
    return center - radius, center + radius


def pdi_loss_from_entries(c11, c12, c22, raw_margin=DEFAULT_RAW_MARGIN):
    """Equation (5)'s spectral PDI term, averaged over samples and dimension."""
    lo, hi = eigenvalues_2x2(c11 + raw_margin, c12, c22 + raw_margin)
    return 0.5 * np.mean(np.maximum(lo, 0.0) ** 2 + np.maximum(hi, 0.0) ** 2)


def baseline_loss(theta, X, rate, raw_margin=DEFAULT_RAW_MARGIN):
    """Raw squared PDI objective under the shared neural representation."""
    from application_verifier import contraction_entries
    c11, c12, c22, _trace = contraction_entries(theta, X, rate)
    return pdi_loss_from_entries(c11, c12, c22, raw_margin)


def protocol_description():
    """Machine-readable scientific identity; experiment budgets live in driver."""
    return {
        "name": "Fitzsimmons--Liu PDI objective (controlled adaptation)",
        "source_equation": "Fitzsimmons and Liu (2024), Eq. (5), with Eqs. (6)--(7)",
        "objective": "mean over samples and both eigenvalues of max(lambda_i(C_rate+qI),0)^2",
        "positivity_penalty": "zero identically: S=alpha I and M=alpha I+L^T L",
        "normalization": "none",
        "parameterization": "common six-unit tanh triangular metric factor; 39 parameters; alpha=0.1",
        "data_policy": "fixed common initial collocation set; no verifier feedback",
        "certification": "common outward-rounded verifier; positive training loss is not a certificate",
        "scope": "learning-objective and diagnostic-feedback comparison; not original full-method reproduction",
    }
