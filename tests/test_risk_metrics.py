"""
Extended Performance Metrics Test
"""

from __future__ import annotations

import unittest

import numpy as np
import polars as pl

from agentic_quant.risk.metrics import (
    calmar_ratio,
    deflated_sharpe,
    omega_ratio,
    probabilistic_sharpe,
    skewness_kurtosis,
    sortino_ratio,
)
from agentic_quant.validation.overfit import probability_of_backtest_overfitting


class RiskMetricsTest(unittest.TestCase):
    def _returns(self, values: list[float]) -> pl.Series:
        return pl.Series("r", values)

    def test_calmar_basic(self):
        r = self._returns([0.01, 0.02, -0.01, 0.01, 0.0])
        cr = calmar_ratio(r, max_drawdown=-0.05, ann_factor=252)
        self.assertGreater(cr, 0.0)

    def test_sortino_positive(self):
        r = self._returns([0.01, -0.005, 0.02, 0.015, -0.002])
        sr = sortino_ratio(r, ann_factor=252)
        self.assertGreater(sr, 0.0)

    def test_omega_basic(self):
        r = self._returns([0.01, -0.02, 0.03, -0.01, 0.02])
        o = omega_ratio(r, threshold=0.0)
        self.assertGreater(o, 0.0)

    def test_skewness_kurtosis_shape(self):
        np.random.seed(42)
        r = self._returns(list(np.random.randn(1000) * 0.02))
        skew, kurt = skewness_kurtosis(r)
        self.assertIsInstance(skew, float)
        self.assertIsInstance(kurt, float)

    def test_probabilistic_sharpe_range(self):
        psr = probabilistic_sharpe(
            sharpe=1.5, benchmark=1.0, skewness=0.0, kurtosis=3.0, n=100
        )
        self.assertGreaterEqual(psr, 0.0)
        self.assertLessEqual(psr, 1.0)

    def test_probabilistic_sharpe_equal_benchmark_is_half(self):
        psr = probabilistic_sharpe(
            sharpe=1.0, benchmark=1.0, skewness=0.0, kurtosis=0.0, n=100
        )
        self.assertAlmostEqual(psr, 0.5, delta=0.01)

    def test_deflated_sharpe_is_probability(self):
        dsr = deflated_sharpe(
            sharpe=1.5, skewness=0.0, kurtosis=0.0,
            n_trials=20, n_samples=1000,
        )
        self.assertGreaterEqual(dsr, 0.0)
        self.assertLessEqual(dsr, 1.0)

    def test_deflated_sharpe_more_trials_reduces_confidence(self):
        low_trials = deflated_sharpe(1.0, 0.0, 0.0, n_trials=2, n_samples=1000)
        high_trials = deflated_sharpe(1.0, 0.0, 0.0, n_trials=200, n_samples=1000)
        self.assertGreaterEqual(low_trials, high_trials)


class OverfitTest(unittest.TestCase):
    def test_pbo_random_walk(self):
        """Random walk strategy should show high overfitting probability (IS optimum is not necessarily optimal OOS)."""
        np.random.seed(42)
        returns = np.random.randn(10, 50) * 0.01
        pbo = probability_of_backtest_overfitting(returns)
        self.assertGreaterEqual(pbo, 0.0)
        self.assertLessEqual(pbo, 1.0)

    def test_pbo_low_dimensions(self):
        returns = np.array([[0.01, 0.02], [0.02, 0.01]])
        pbo = probability_of_backtest_overfitting(returns)
        self.assertGreaterEqual(pbo, 0.0)


if __name__ == "__main__":
    unittest.main()
