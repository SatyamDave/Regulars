"""Isotonic calibration of judge scores (technical-architecture.md §5 "Calibration").

Raw ``y_hat`` is not a probability. This maps ``y_hat`` monotonically onto observed
``P(would_see_again)`` so that reported values mean what they say.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.isotonic import IsotonicRegression

MIN_CALIBRATION_N = 30
"""Floor on labeled dyads for a calibrator.

Isotonic regression is the lowest-capacity monotone fit there is, but it is still
nonparametric: PAVA produces a step function whose number of steps grows roughly with
n^(1/3), and every step's height is the mean of the labels pooled into it. Below ~30 labels
there are only two or three steps, each averaging a handful of binary outcomes, so the
"calibrated" probabilities carry standard errors of +/-0.2 or worse and a ten-bin reliability
table has three observations per bin. That is not calibration, it is a lookup table of noise.
§5 says to report calibrated values only; below this floor there are none worth reporting.
"""


@dataclass(frozen=True, slots=True)
class ReliabilityRow:
    lower: float
    upper: float
    n: int
    mean_predicted: float
    mean_observed: float


class IsotonicCalibrator:
    """``fit(y_hat, y)`` then ``predict_proba(y_hat)``; monotone non-decreasing by construction."""

    def __init__(self) -> None:
        self._model: IsotonicRegression | None = None
        self._y_hat: NDArray[np.float64] | None = None
        self._y: NDArray[np.float64] | None = None

    def fit(self, y_hat: Sequence[float], y: Sequence[float]) -> IsotonicCalibrator:
        yh = np.asarray(y_hat, dtype=np.float64)
        yy = np.asarray(y, dtype=np.float64)
        if yh.ndim != 1 or yh.shape != yy.shape:
            raise ValueError("y_hat and y must be 1-d and the same length")
        if yh.size < MIN_CALIBRATION_N:
            raise ValueError(
                f"isotonic calibration needs at least {MIN_CALIBRATION_N} labeled dyads, "
                f"got {yh.size}; see technical-architecture.md §5 'Calibration' and "
                "regulars.judge.calibration.MIN_CALIBRATION_N for why."
            )
        if not np.all((yy >= 0.0) & (yy <= 1.0)):
            raise ValueError("y must be in [0, 1] (binary outcome or observed rate)")
        model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        model.fit(yh, yy)
        self._model = model
        self._y_hat = yh
        self._y = yy
        return self

    def _require_fit(self) -> IsotonicRegression:
        if self._model is None:
            raise RuntimeError("call fit() first")
        return self._model

    def predict_proba(self, y_hat: Sequence[float] | NDArray[np.float64]) -> NDArray[np.float64]:
        model = self._require_fit()
        out = model.predict(np.asarray(y_hat, dtype=np.float64))
        return np.asarray(out, dtype=np.float64)

    def _data(
        self, y_hat: Sequence[float] | None, y: Sequence[float] | None
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        self._require_fit()
        if (y_hat is None) != (y is None):
            raise ValueError("pass both y_hat and y, or neither (uses the fit data)")
        if y_hat is None or y is None:
            assert self._y_hat is not None and self._y is not None
            return self._y_hat, self._y
        return np.asarray(y_hat, dtype=np.float64), np.asarray(y, dtype=np.float64)

    def brier(
        self, y_hat: Sequence[float] | None = None, y: Sequence[float] | None = None
    ) -> float:
        """Mean squared error of calibrated probabilities; 0 is perfect, 0.25 is a coin flip."""
        yh, yy = self._data(y_hat, y)
        p = self.predict_proba(yh)
        return float(np.mean((p - yy) ** 2))

    def reliability_table(
        self,
        y_hat: Sequence[float] | None = None,
        y: Sequence[float] | None = None,
        bins: int = 10,
    ) -> list[ReliabilityRow]:
        """Equal-width bins over calibrated probability; empty bins are omitted."""
        if bins < 1:
            raise ValueError("bins must be >= 1")
        yh, yy = self._data(y_hat, y)
        p = self.predict_proba(yh)
        edges = np.linspace(0.0, 1.0, bins + 1)
        idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, bins - 1)
        rows: list[ReliabilityRow] = []
        for b in range(bins):
            mask = idx == b
            if not mask.any():
                continue
            rows.append(
                ReliabilityRow(
                    lower=float(edges[b]),
                    upper=float(edges[b + 1]),
                    n=int(mask.sum()),
                    mean_predicted=float(p[mask].mean()),
                    mean_observed=float(yy[mask].mean()),
                )
            )
        return rows
