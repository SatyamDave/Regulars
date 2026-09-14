"""Power for a correlation under clustering (§6 Problem 3), via the Fisher z transform.

``atanh(r)`` is approximately normal with SD ``1 / sqrt(n - 3)``. Clustering within person and
group inflates that variance by the design effect ``1 + (m - 1) · ICC`` (Kish), so the
effective sample size is ``n / design_effect``.
"""

from __future__ import annotations

import math

from scipy.stats import norm


def design_effect(cluster_size: float, icc: float) -> float:
    """Kish design effect for clusters of ``cluster_size`` with intraclass correlation ``icc``."""
    if not 0.0 <= icc <= 1.0:
        raise ValueError("icc must be in [0, 1]")
    if cluster_size < 1:
        raise ValueError("cluster_size must be >= 1")
    return 1.0 + (cluster_size - 1.0) * icc


def effective_n(n: int, cluster_size: float, icc: float) -> float:
    """Effective sample size of ``n`` clustered observations."""
    return n / design_effect(cluster_size, icc)


def power_for_rho(rho: float, n: int, alpha: float = 0.05, design_effect: float = 1.0) -> float:
    """Two-sided power to reject ρ = 0 when the true correlation is ``rho``."""
    _check(rho, alpha, design_effect)
    n_eff = n / design_effect
    if n_eff <= 3:
        return 0.0
    z = math.atanh(rho)
    se = 1.0 / math.sqrt(n_eff - 3.0)
    crit = float(norm.ppf(1.0 - alpha / 2.0))
    shift = z / se
    return float(norm.cdf(shift - crit) + norm.cdf(-shift - crit))


def n_needed(
    rho: float = 0.35, power: float = 0.8, alpha: float = 0.05, design_effect: float = 1.0
) -> int:
    """Smallest raw ``n`` (before clustering loss) reaching ``power`` for ``rho``."""
    _check(rho, alpha, design_effect)
    if not 0.0 < power < 1.0:
        raise ValueError("power must be in (0, 1)")
    if rho == 0.0:
        raise ValueError("rho must be non-zero")
    z_alpha = float(norm.ppf(1.0 - alpha / 2.0))
    z_beta = float(norm.ppf(power))
    n_eff = ((z_alpha + z_beta) / math.atanh(abs(rho))) ** 2 + 3.0
    return math.ceil(n_eff * design_effect)


def _check(rho: float, alpha: float, design_effect: float) -> None:
    if not -1.0 < rho < 1.0:
        raise ValueError("rho must be in (-1, 1)")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if design_effect < 1.0:
        raise ValueError("design_effect must be >= 1")
