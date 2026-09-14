"""Mixed model with random intercepts for rater, ratee and group (§6 Problem 3).

``outcome ~ y_hat`` with ``groups=group_id`` (random intercept per group) and variance
components for rater and ratee via ``vc_formula``. statsmodels nests variance components inside
the top-level group, so a rater who sits in two groups gets two independent intercepts. That is
exact for the six-person round-robin (§5: a user is in one group at a time) and a conservative
approximation once continuity carries people across groups.

The outcome is ``LabeledDyad.outcome`` (``felt_connected_1_7`` where present, else 0/1). A
linear mixed model on a 1–7 scale is the standard SRM working model; on 0/1 it is a linear
probability model, fine for the slope's sign and rough size, not for its tails.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from regulars.eval.records import LabeledDyad

MIN_DYADS = 10


@dataclass(frozen=True, slots=True)
class MixedResult:
    beta_yhat: float
    se: float
    p: float
    var_rater: float
    var_ratee: float
    var_group: float
    var_resid: float
    converged: bool
    n: int

    @property
    def total_var(self) -> float:
        return self.var_rater + self.var_ratee + self.var_group + self.var_resid

    def share(self, component: str) -> float:
        """Share of total variance in ``rater`` | ``ratee`` | ``group`` | ``resid``."""
        value = float(getattr(self, f"var_{component}"))
        return value / self.total_var if self.total_var > 0 else math.nan


def _frame(dyads: Sequence[LabeledDyad], use_calibrated: bool) -> pd.DataFrame:
    def score(d: LabeledDyad) -> float:
        if use_calibrated:
            if d.y_hat_calibrated is None:
                raise ValueError("use_calibrated requires y_hat_calibrated on every dyad")
            return d.y_hat_calibrated
        return d.y_hat

    return pd.DataFrame(
        {
            "y": [d.outcome for d in dyads],
            "y_hat": [score(d) for d in dyads],
            "rater": [d.rater for d in dyads],
            "ratee": [d.ratee for d in dyads],
            "group_id": [d.group_id for d in dyads],
        }
    )


def fit_mixed(
    dyads: Sequence[LabeledDyad], use_calibrated: bool = False, maxiter: int = 500
) -> MixedResult:
    if len(dyads) < MIN_DYADS:
        raise ValueError(f"need at least {MIN_DYADS} dyads, got {len(dyads)}")
    df = _frame(dyads, use_calibrated)
    if df["y_hat"].nunique() < 2 or df["y"].nunique() < 2:
        raise ValueError("y_hat or outcome is constant; mixed model undefined")

    model = MixedLM.from_formula(
        "y ~ y_hat",
        data=df,
        groups="group_id",
        re_formula="1",
        vc_formula={"rater": "0 + C(rater)", "ratee": "0 + C(ratee)"},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        fit: Any = model.fit(reml=True, maxiter=maxiter)

    converged = bool(getattr(fit, "converged", False))
    var_group = float(np.asarray(fit.cov_re)[0, 0]) if fit.cov_re.size else 0.0
    vcomp = {name: float(v) for name, v in zip(model.exog_vc.names, fit.vcomp, strict=True)}
    return MixedResult(
        beta_yhat=float(fit.params["y_hat"]),
        se=float(fit.bse["y_hat"]),
        p=float(fit.pvalues["y_hat"]),
        var_rater=vcomp.get("rater", 0.0),
        var_ratee=vcomp.get("ratee", 0.0),
        var_group=var_group,
        var_resid=float(fit.scale),
        converged=converged,
        n=len(dyads),
    )
