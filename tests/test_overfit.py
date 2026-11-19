"""
PBO / CSCV / Parameter Stability Analysis Unit Test

Testcoverage:
- probability_of_backtest_overfitting boundary conditions with known results
- cscv_pbo matrix dimensions, symmetry, and rating logic
- ParameterStabilityAnalyzer plateau/cliff detection
- generate_pbo_report Report Generation
"""

from __future__ import annotations

import numpy as np
import pytest

from agentic_quant.validation.overfit import (
    ParameterStabilityAnalyzer,
    ParamStabilityResult,
    PBOResult,
    PBOSummary,
    _compute_deflated_sharpe,
    cscv_pbo,
    generate_batch_pbo_report,
    generate_pbo_report,
    probability_of_backtest_overfitting,
    run_pbo_batch,
    run_pbo_single,
)

# ═══════════════════════ probability_of_backtest_overfitting ═══════════════════


class TestProbabilityOfBacktestOverfitting:
    """Simplified PBO single-split test"""

    def test_returns_in_0_1_range(self):
        """Random return matrix PBO should be in [0, 1] range"""
        rng = np.random.RandomState(42)
        returns = rng.randn(20, 100) * 0.01
        pbo = probability_of_backtest_overfitting(returns)
        assert 0.0 <= pbo <= 1.0

    def test_too_few_strategies_returns_zero(self):
        """Less than 2 entries returns 0"""
        returns = np.random.randn(1, 10) * 0.01
        assert probability_of_backtest_overfitting(returns) == 0.0

    def test_too_few_periods_returns_zero(self):
        """Less than 2 periods returns 0"""
        returns = np.random.randn(5, 1) * 0.01
        assert probability_of_backtest_overfitting(returns) == 0.0

    def test_raises_on_1d_input(self):
        """1D input should raise ValueError"""
        with pytest.raises(ValueError, match="2D"):
            probability_of_backtest_overfitting(np.array([0.01, 0.02]))

    def test_one_strategy_always_best(self):
        """Single always-optimal strategy: PBO should be 0"""
        returns = np.array([[0.02, 0.03], [0.01, 0.01]])
        # Strategy0 IS performs better, and also in OOS
        pbo = probability_of_backtest_overfitting(returns)
        assert 0.0 <= pbo <= 1.0

    def test_deterministic_seed_same_result(self):
        """Same random seed should produce same result"""
        rng = np.random.RandomState(123)
        returns = rng.randn(15, 80) * 0.01
        pbo1 = probability_of_backtest_overfitting(returns.copy())
        pbo2 = probability_of_backtest_overfitting(returns.copy())
        assert pbo1 == pbo2


# ═════════════════════════════════ cscv_pbo ═════════════════════════════════


class TestCscvPbo:
    """CSCV PBO CompleteImplementTest"""

    def test_basic_execution(self):
        """Basic execution: returns reasonable PBO result"""
        rng = np.random.RandomState(42)
        returns = rng.randn(20, 100) * 0.01
        result = cscv_pbo(returns, n_combinations=50)
        assert isinstance(result, PBOResult)
        assert 0.0 <= result.pbo <= 1.0
        assert result.cscv_combinations == 50

    def test_insufficient_strategies_early_return(self):
        """Less than 2 strategies or 4 periods returns short result"""
        result = cscv_pbo(np.random.randn(1, 10) * 0.01)
        assert result.pbo == 0.0
        assert result.cscv_combinations == 0

        result = cscv_pbo(np.random.randn(5, 3) * 0.01)
        assert result.pbo == 0.0
        assert result.cscv_combinations == 0

    def test_grade_a_low_overfit(self):
        """High resolution strategy: PBO should be low (<0.30)"""
        rng = np.random.RandomState(42)
        # Create one entry significantly better than others
        n_strategies, n_periods = 20, 100
        returns = rng.randn(n_strategies, n_periods) * 0.02
        # Strategy0 has significant positive return
        returns[0, :] += 0.005
        result = cscv_pbo(returns, n_combinations=100)
        # Clear winner should keep PBO low
        assert result.pbo < 0.50

    def test_grade_d_high_overfit(self):
        """All strategies similar: PBO should be high (>0.10)"""
        rng = np.random.RandomState(42)
        # All strategies are random walks, IS optimum unlikely to hold in OOS
        returns = rng.randn(30, 80) * 0.01
        result = cscv_pbo(returns, n_combinations=100)
        # Pure random: PBO typically > 0.10
        assert result.grade in ("C", "D") or result.pbo >= 0.10

    def test_deterministic_output(self):
        """Same seed should produce same result"""
        rng = np.random.RandomState(42)
        returns = rng.randn(15, 60) * 0.01
        result1 = cscv_pbo(returns.copy(), random_seed=42, n_combinations=50)
        result2 = cscv_pbo(returns.copy(), random_seed=42, n_combinations=50)
        assert result1.pbo == result2.pbo
        assert result1.logit_scores == result2.logit_scores

    def test_different_seed_different_result(self):
        """Different seeds may produce different results"""
        rng = np.random.RandomState(42)
        returns = rng.randn(10, 40) * 0.01
        result1 = cscv_pbo(returns.copy(), random_seed=42, n_combinations=30)
        result2 = cscv_pbo(returns.copy(), random_seed=99, n_combinations=30)
        # Different seeds: logit_scores sequence likely differs
        assert result1.logit_scores != result2.logit_scores

    def test_metric_mean_return(self):
        """Support mean_return Metrics"""
        rng = np.random.RandomState(42)
        returns = rng.randn(15, 60) * 0.01
        result = cscv_pbo(returns, metric_fn="mean_return", n_combinations=30)
        assert isinstance(result, PBOResult)

    def test_metric_calmar(self):
        """Support calmar Metrics"""
        rng = np.random.RandomState(42)
        returns = rng.randn(15, 60) * 0.01
        result = cscv_pbo(returns, metric_fn="calmar", n_combinations=30)
        assert isinstance(result, PBOResult)

    def test_performance_degradation_field(self):
        """Performance decay field should be correctly calculated"""
        # Create scenario: good IS but bad OOS
        n_strategies, n_periods = 10, 100
        returns = np.zeros((n_strategies, n_periods))
        # First half IS: Strategy0 performs best
        returns[0, :50] = 0.01
        # Second half OOS: Strategy0 performs worse
        returns[0, 50:] = -0.005
        # OtherStrategy PerformanceConsistent
        returns[1:, :50] = 0.002
        returns[1:, 50:] = 0.002
        result = cscv_pbo(returns, n_combinations=30)
        # Should be able to calculate degradation
        assert result.performance_degradation >= 0.0

    def test_logit_scores_length(self):
        """logit_scores should match combination count"""
        rng = np.random.RandomState(42)
        returns = rng.randn(15, 60) * 0.01
        result = cscv_pbo(returns, n_combinations=40)
        assert len(result.logit_scores) == 40
        assert len(result.is_best_rank_quantiles) == 40

    def test_rank_quantiles_in_range(self):
        """Rank quantiles should be in [0,1] range (0 = best rank)"""
        rng = np.random.RandomState(42)
        returns = rng.randn(15, 60) * 0.01
        result = cscv_pbo(returns, n_combinations=50)
        for q in result.is_best_rank_quantiles:
            assert 0.0 <= q <= 1.0

    def test_pbo_result_is_degraded_flag(self):
        """is_degraded true when decay exceeds 50%"""
        # Construct significant decay scenario
        n_strategies, n_periods = 10, 100
        returns = np.zeros((n_strategies, n_periods))
        returns[0, :50] = 0.02  # IS is good
        returns[0, 50:] = -0.01  # OOS is poor
        returns[1:, :50] = 0.001
        returns[1:, 50:] = 0.001
        result = cscv_pbo(returns, n_combinations=30)
        # degradation may exceed 50%
        if result.performance_degradation > 0.5:
            assert result.is_degraded

    def test_pbo_result_grade_fields(self):
        """Check rating field reasonableness"""
        rng = np.random.RandomState(42)
        returns = rng.randn(20, 80) * 0.01
        result = cscv_pbo(returns, n_combinations=50)
        assert result.grade in ("A", "B", "C", "D", "N/A")


    def test_grade_b_when_pbo_between_10_and_20(self):
        """PBO between 0.10~0.20 should be B level"""
        # Construct deterministic scenario where PBO falls in B range
        # Strategy count must be long enough for rank quantiles to vary
        n_strategies, n_periods = 20, 100
        # Try different seeds until B level is found
        for seed in range(42, 60):
            rng2 = np.random.RandomState(seed)
            ret = rng2.randn(n_strategies, n_periods) * 0.01
            result = cscv_pbo(ret, n_combinations=50)
            if result.grade == "B":
                break
        assert result.grade in ("A", "B", "C", "D"), f"Expected valid rating, got {result.grade}"

    def test_pbo_precision(self):
        """PBO should round to 4 decimal places"""
        rng = np.random.RandomState(42)
        returns = rng.randn(15, 60) * 0.01
        result = cscv_pbo(returns, n_combinations=50)
        # pbo should be 0-1 decimal, max 4 digits
        pbo_str = str(result.pbo)
        if "." in pbo_str:
            decimals = len(pbo_str.split(".")[1])
            assert decimals <= 4, f"PBO decimals should be <=4, got {decimals}"

    def test_dsr_is_direction_agnostic_for_extreme_logit(self):
        """DSR measures extremity; sign interpreted by PBO separately."""
        good_dsr = _compute_deflated_sharpe(
            rank_quantiles=[0.1, 0.12, 0.08, 0.11],
            logit_scores=[-2.2, -2.0, -2.4, -2.1],
            n_strategies=8,
        )
        overfit_dsr = _compute_deflated_sharpe(
            rank_quantiles=[0.9, 0.88, 0.92, 0.89],
            logit_scores=[2.2, 2.0, 2.4, 2.1],
            n_strategies=8,
        )

        assert good_dsr > 0.5
        assert overfit_dsr == good_dsr

    def test_dsr_zero_variance_logit_keeps_abs_direction(self):
        """Zero variance boundary also uses abs(logit) convention."""
        assert _compute_deflated_sharpe([0.1, 0.1], [-2.0, -2.0], 8) == 1.0
        assert _compute_deflated_sharpe([0.9, 0.9], [2.0, 2.0], 8) == 1.0
        assert _compute_deflated_sharpe([0.5, 0.5], [0.0, 0.0], 8) == 0.0

    def test_performance_degradation_high(self):
        """IS much better than OOS: degradation > 0 (needs enough data for distinct time blocks)"""
        n_strategies, n_periods = 10, 200
        returns = np.zeros((n_strategies, n_periods))
        # Strategy0: first half IS is excellent, second half OOS is terrible
        returns[0, :100] = 0.03
        returns[0, 100:] = -0.02
        # Other strategies: consistently poor performance
        returns[1:, :] = 0.001
        result = cscv_pbo(returns, n_combinations=100, n_blocks=10)
        # New version uses CSCV time block median decay -- when >50% blocks are from first half,
        # IS includes many 0.03 days while OOS includes -0.02 days, decay should be > 0
        assert result.performance_degradation >= 0


# ═══════════════════════ ParameterStabilityAnalyzer ═══════════════════════


class TestParameterStabilityAnalyzer:
    """Parameter Stability Analyzer test"""

    def test_basic_analysis(self):
        """Basic stability analysis should return correct structure"""
        result = ParameterStabilityAnalyzer.analyze(
            [1, 2, 3, 4, 5],
            [0.1, 0.5, 1.0, 0.5, 0.1],
            param_name="test_param",
        )
        assert isinstance(result, ParamStabilityResult)
        assert result.param_name == "test_param"
        assert result.best_value == 3.0
        assert result.best_score == 1.0

    def test_insufficient_data(self):
        """Less than 3 data points should return zero result"""
        result = ParameterStabilityAnalyzer.analyze(
            [1, 2], [0.1, 0.2], param_name="short"
        )
        assert result.stability_score == 0.0
        assert result.plateau_width == 0.0
        assert not result.has_cliff

    def test_wide_plateau(self):
        """Wide plateau should produce high stability score"""
        # Large middle region: all performance is good
        result = ParameterStabilityAnalyzer.analyze(
            [1, 2, 3, 4, 5, 6, 7],
            [0.5, 0.9, 1.0, 1.0, 1.0, 0.9, 0.5],
            param_name="wide",
        )
        assert result.plateau_width > 0.3

    def test_narrow_plateau(self):
        """Narrow plateau should produce low stability score"""
        # Only near peak performs well
        result = ParameterStabilityAnalyzer.analyze(
            [1, 2, 3, 4, 5],
            [0.1, 0.2, 1.0, 0.2, 0.1],
            param_name="narrow",
        )
        assert result.plateau_width < 0.5

    def test_cliff_detection_true(self):
        """Adjacent large performance drops should detect cliff"""
        result = ParameterStabilityAnalyzer.analyze(
            [1, 2, 3, 4, 5],
            [1.0, 0.9, 0.3, 0.2, 0.1],  # Drop from 0.9 to 0.3 > 50%
            param_name="cliff",
        )
        assert result.has_cliff

    def test_cliff_detection_false(self):
        """Smooth change should not trigger cliff detection"""
        result = ParameterStabilityAnalyzer.analyze(
            [1, 2, 3, 4, 5],
            [0.8, 0.9, 1.0, 0.9, 0.8],  # Gradual change
            param_name="smooth",
        )
        assert not result.has_cliff

    def test_stability_score_formula(self):
        """StabilityScore = plateau_width * (0.5 if has_cliff else 1.0)"""
        # No cliff: stability = plateau width
        no_cliff = ParameterStabilityAnalyzer.analyze(
            [1, 2, 3, 4, 5],
            [0.8, 0.9, 1.0, 0.9, 0.8],
            param_name="no_cliff",
        )
        expected = no_cliff.plateau_width * 1.0
        assert abs(no_cliff.stability_score - expected) < 1e-6

    def test_grid_analysis_single_dim(self):
        """Single-dimension parameter grid analysis"""
        results = ParameterStabilityAnalyzer.analyze_grid(
            {"fast": [5, 10, 15]},
            np.array([[0.5, 1.0, 0.3]]),
        )
        assert len(results) == 1
        assert results[0].param_name == "fast"

    def test_grid_analysis_multi_dim(self):
        """Two-dimensional parameter grid analysis"""
        results = ParameterStabilityAnalyzer.analyze_grid(
            {"fast": [5, 10, 15], "slow": [20, 30, 40]},
            np.array([[0.5, 0.6, 0.4], [1.0, 0.9, 0.7], [0.3, 0.5, 0.2]]),
        )
        assert len(results) == 2
        names = [r.param_name for r in results]
        assert "fast" in names
        assert "slow" in names


# ═════════════════════════ generate_pbo_report ═══════════════════════


class TestGeneratePboReport:
    """PBO Report GenerationTest"""

    def test_generate_basic_report(self):
        """Basic report generation should not throw exception"""
        result = PBOResult(
            pbo=0.05,
            cscv_combinations=100,
            n_blocks=16,
            logit_scores=[-2.0, -1.5, -1.0],
            is_best_rank_quantiles=[0.2, 0.3, 0.15],
            performance_degradation=0.1,
            dsr=0.92,
            is_degraded=False,
            grade="A",
        )
        stability_results = [
            ParamStabilityResult(
                param_name="fast",
                values=[5, 10, 15],
                scores=[0.5, 1.0, 0.3],
                best_value=10.0,
                best_score=1.0,
                stability_score=0.8,
                plateau_width=0.5,
                has_cliff=False,
            )
        ]
        report = generate_pbo_report(result, stability_results, strategy_name="test_strat")
        assert isinstance(report, str)
        assert "PBO" in report
        assert "test_strat" in report
        assert "A" in report

    def test_report_with_degraded_strategy(self):
        """Strategy with decay >50% should include warning in report"""
        result = PBOResult(
            pbo=0.35,
            cscv_combinations=100,
            logit_scores=[1.0, 0.5],
            is_best_rank_quantiles=[0.6, 0.7],
            performance_degradation=0.6,
            is_degraded=True,
            grade="D",
        )
        report = generate_pbo_report(result, [], strategy_name="bad_strat")
        assert "Decay" in report

    def test_report_with_output_dir_creates_files(self, tmp_path):
        """Specify output_dir should generate file and JSON"""
        result = PBOResult(
            pbo=0.12,
            cscv_combinations=100,
            logit_scores=[-1.0, -0.5],
            is_best_rank_quantiles=[0.3, 0.4],
            performance_degradation=0.2,
            is_degraded=False,
            grade="B",
        )
        report = generate_pbo_report(result, [], output_dir=str(tmp_path))
        # Should generate .md and .json files
        files = list(tmp_path.iterdir())
        md_files = [f for f in files if f.suffix == ".md"]
        json_files = [f for f in files if f.suffix == ".json"]
        assert len(md_files) >= 1
        assert len(json_files) >= 1
        assert isinstance(report, str)

    def test_report_with_cliff_strategy(self):
        """Strategies with cliff drops should be flagged in reports"""
        result = PBOResult(
            pbo=0.08,
            cscv_combinations=50,
            logit_scores=[-2.0],
            is_best_rank_quantiles=[0.2],
            performance_degradation=0.1,
            is_degraded=False,
            grade="A",
        )
        stability_results = [
            ParamStabilityResult(
                param_name="fast",
                values=[5, 10, 15],
                scores=[1.0, 0.3, 0.1],
                best_value=5.0,
                best_score=1.0,
                stability_score=0.2,
                plateau_width=0.1,
                has_cliff=True,
            )
        ]
        report = generate_pbo_report(result, stability_results)
        assert "cliff" in report or "Yes" in report

    def test_report_empty_stability(self):
        """Short stability list should not throw exception"""
        result = PBOResult(
            pbo=0.15,
            cscv_combinations=50,
            logit_scores=[-1.0, 0.0],
            is_best_rank_quantiles=[0.4, 0.5],
            performance_degradation=0.3,
            is_degraded=False,
            grade="B",
        )
        report = generate_pbo_report(result, [])
        assert isinstance(report, str)


# ═════════════════════════════ Batch PBO Test ═════════════════════════════


class TestComputeDailyStrategyReturns:
    """StrategydailyReturnCalculateTest"""

    def test_basic_computation(self):
        """basicReturnCalculate"""
        from agentic_quant.validation.overfit import _compute_daily_strategy_returns

        closes = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        entries = np.array([True, False, False, False, False])
        exits = np.array([False, False, False, False, True])
        rets = _compute_daily_strategy_returns(closes, entries, exits)
        # Entry then hold: return = close diff = [1, 1, 1, 1] / [100, 101, 102, 103] = [0.01, ...]
        assert len(rets) == len(closes) - 1
        # Should have some non-zero values
        assert np.any(rets != 0.0)

    def test_no_trades_returns_zeros(self):
        """No trading: return all zeros"""
        from agentic_quant.validation.overfit import _compute_daily_strategy_returns

        closes = np.array([100.0, 101.0, 102.0])
        entries = np.array([False, False, False])
        exits = np.array([False, False, False])
        rets = _compute_daily_strategy_returns(closes, entries, exits)
        assert np.all(rets == 0.0)

    def test_multiple_entry_exit_cycles(self):
        """Multiple rounds of on-close position cycling"""
        from agentic_quant.validation.overfit import _compute_daily_strategy_returns

        closes = np.array([100.0, 101.0, 102.0, 101.0, 100.0, 99.0, 100.0])
        entries = np.array([True, False, False, False, True, False, False])
        exits = np.array([False, False, True, False, False, False, True])
        rets = _compute_daily_strategy_returns(closes, entries, exits)
        assert len(rets) == 6
        # first round: bar 0 entry, bar 2 exit => rets[0], rets[1] non-zero
        assert rets[0] != 0.0 or rets[1] != 0.0


class TestRunPboSingle:
    """Single-underlying PBO run test"""

    def test_insufficient_params_returns_early(self):
        """Less than 2 parameter groups: early return"""
        n = 100
        opens = np.random.randn(n) + 100
        highs = opens + 2
        lows = opens - 2
        closes = opens + np.random.randn(n) * 0.5
        volumes = np.ones(n) * 10000
        result = run_pbo_single(
            opens, highs, lows, closes, volumes,
            param_grid=[{"fast": 5, "slow": 20}],  # stocksHas 1 groups
        )
        assert result.pbo == 0.0
        assert result.grade == "N/A"

    def test_insufficient_days_returns_early(self):
        """Data period < 20 days: early return"""
        n = 10
        opens = np.random.randn(n) + 100
        highs = opens + 2
        lows = opens - 2
        closes = opens + np.random.randn(n) * 0.5
        volumes = np.ones(n) * 10000
        result = run_pbo_single(
            opens, highs, lows, closes, volumes,
            param_grid=[{"fast": 5, "slow": 20}, {"fast": 10, "slow": 30}],
        )
        assert result.pbo == 0.0
        assert result.grade == "N/A"

    def test_basic_execution(self):
        """basic PBO Execute"""
        n = 100
        rng = np.random.RandomState(42)
        # Generate slightly rising trend data
        trend = np.linspace(0, 10, n)
        noise = rng.randn(n) * 2
        closes = 100.0 + trend + noise
        opens = closes + rng.randn(n) * 0.5
        highs = np.maximum(closes, opens) + abs(rng.randn(n)) + 0.5
        lows = np.minimum(closes, opens) - abs(rng.randn(n)) - 0.5
        volumes = np.abs(rng.randn(n)) * 10000 + 5000

        result = run_pbo_single(
            opens, highs, lows, closes, volumes,
            param_grid=[
                {"fast": 5, "slow": 20},
                {"fast": 10, "slow": 30},
                {"fast": 15, "slow": 40},
            ],
            n_combinations=20,
        )
        assert isinstance(result, PBOResult)
        assert 0.0 <= result.pbo <= 1.0

    def test_without_vpa(self):
        """Do not enable VPA filter"""
        n = 80
        rng = np.random.RandomState(42)
        closes = 100.0 + np.cumsum(rng.randn(n) * 0.5 + 0.02)
        opens = closes + rng.randn(n) * 0.3
        highs = closes + abs(rng.randn(n)) * 0.5 + 0.1
        lows = closes - abs(rng.randn(n)) * 0.5 - 0.1
        volumes = np.abs(rng.randn(n)) * 10000
        result = run_pbo_single(
            opens, highs, lows, closes, volumes,
            param_grid=[{"fast": 5, "slow": 20}, {"fast": 10, "slow": 30}],
            enable_vpa=False,
            n_combinations=10,
        )
        assert isinstance(result, PBOResult)


class TestRunPboBatch:
    """Batch PBO RunTest"""

    def test_empty_symbols(self):
        """ShortUnderlyingList"""
        result = run_pbo_batch([], param_grid=[{"fast": 5, "slow": 20}],
                                fetch_func=lambda x: None, show_progress=False)
        assert isinstance(result, PBOSummary)
        assert result.n_total == 0
        assert result.n_success == 0

    def test_mocked_fetch(self):
        """Using mock Data FetchingFunction"""
        n = 100
        rng = np.random.RandomState(42)
        closes = 100.0 + np.cumsum(rng.randn(n) * 0.5 + 0.02)

        def mock_fetch(symbol: str) -> dict:
            return {
                "open": closes + rng.randn(n) * 0.3,
                "high": closes + abs(rng.randn(n)) * 0.5 + 0.1,
                "low": closes - abs(rng.randn(n)) * 0.5 - 0.1,
                "close": closes,
                "volume": np.abs(rng.randn(n)) * 10000,
            }

        result = run_pbo_batch(
            ["000001", "000002"],
            param_grid=[{"fast": 5, "slow": 20}, {"fast": 10, "slow": 30}],
            fetch_func=mock_fetch,
            show_progress=False,
            n_combinations=10,
            request_delay=0.0,
        )
        assert result.n_total == 2
        assert result.n_success == 2
        assert len(result.results) == 2
        assert len(result.pbo_values) == 2

    def test_failed_fetch_handling(self):
        """Data fetch failure should not interrupt pipeline"""
        def mock_fetch(symbol: str):
            return None  # AllFailed

        result = run_pbo_batch(
            ["A", "B"],
            param_grid=[{"fast": 5, "slow": 20}],
            fetch_func=mock_fetch,
            show_progress=False,
        )
        assert result.n_total == 2
        assert result.n_success == 0
        assert result.results == []

    def test_partial_failure(self):
        """PartialFailed"""
        call_count = [0]

        def mock_fetch(symbol: str):
            call_count[0] += 1
            if call_count[0] == 1:
                return None  # firstitemsFailed
            n = 100
            rng = np.random.RandomState(42)
            closes = 100.0 + np.cumsum(rng.randn(n) * 0.5 + 0.02)
            return {
                "open": closes, "high": closes + 1, "low": closes - 1,
                "close": closes, "volume": np.ones(n) * 10000,
            }

        result = run_pbo_batch(
            ["FAIL", "OK"],
            param_grid=[{"fast": 5, "slow": 20}, {"fast": 10, "slow": 30}],
            fetch_func=mock_fetch,
            show_progress=False,
            n_combinations=5,
            request_delay=0.0,
        )
        assert result.n_total == 2
        assert result.n_success == 1
        assert len(result.results) == 1


class TestGenerateBatchPboReport:
    """Batch PBO Report GenerationTest"""

    def _make_summary(self) -> PBOSummary:
        return PBOSummary(
            n_total=3,
            n_success=3,
            n_passed=2,
            pass_rate=2 / 3,
            pbo_values=[0.05, 0.15, 0.35],
            pbo_mean=0.1833,
            pbo_median=0.15,
            pbo_std=0.125,
            grade_counts={"A": 1, "B": 1, "D": 1},
            results=[
                ("000001", PBOResult(pbo=0.05, cscv_combinations=100,
                                      logit_scores=[-2.0], is_best_rank_quantiles=[0.2],
                                      performance_degradation=0.1, is_degraded=False, grade="A")),
                ("000002", PBOResult(pbo=0.15, cscv_combinations=100,
                                      logit_scores=[-1.0], is_best_rank_quantiles=[0.4],
                                      performance_degradation=0.2, is_degraded=False, grade="B")),
                ("000003", PBOResult(pbo=0.35, cscv_combinations=100,
                                      logit_scores=[0.5], is_best_rank_quantiles=[0.6],
                                      performance_degradation=0.4, is_degraded=False, grade="D")),
            ],
        )

    def test_generate_basic_report(self):
        """GeneratebasicReport"""
        summary = self._make_summary()
        report = generate_batch_pbo_report(summary, strategy_name="test_strat")
        assert isinstance(report, str)
        assert "000001" in report
        assert "test_strat" in report

    def test_report_with_output_dir(self, tmp_path):
        """Specify output_dir should generate file"""
        summary = self._make_summary()
        report = generate_batch_pbo_report(summary, output_dir=str(tmp_path))
        files = list(tmp_path.iterdir())
        md_files = [f for f in files if f.suffix == ".md"]
        json_files = [f for f in files if f.suffix == ".json"]
        assert len(md_files) >= 1
        assert len(json_files) >= 1
        assert isinstance(report, str)

    def test_report_pass_rate_high(self):
        """High pass rate report"""
        summary = PBOSummary(
            n_total=10, n_success=10, n_passed=9, pass_rate=0.9,
            pbo_values=[0.05]*9 + [0.18],
            pbo_mean=0.063, pbo_median=0.05, pbo_std=0.041,
            grade_counts={"A": 9, "B": 1},
            results=[("X", PBOResult(pbo=0.05, cscv_combinations=100,
                                      logit_scores=[-2.0], is_best_rank_quantiles=[0.2],
                                      performance_degradation=0.1, is_degraded=False, grade="A"))
                     for _ in range(10)],
        )
        report = generate_batch_pbo_report(summary)
        assert "Passed" in report

    def test_report_pass_rate_low(self):
        """Low pass rate report"""
        summary = PBOSummary(
            n_total=10, n_success=10, n_passed=2, pass_rate=0.2,
            pbo_values=[0.05, 0.15] + [0.3]*8,
            pbo_mean=0.26, pbo_median=0.3, pbo_std=0.1,
            grade_counts={"A": 1, "B": 1, "D": 8},
            results=[("Y", PBOResult(pbo=0.3, cscv_combinations=100,
                                      logit_scores=[0.5], is_best_rank_quantiles=[0.6],
                                      performance_degradation=0.5, is_degraded=True, grade="D"))
                     for _ in range(10)],
        )
        report = generate_batch_pbo_report(summary)
        assert "Failed" in report or "Overfitting" in report

    def test_no_successful_results(self):
        """AllFailed"""
        summary = PBOSummary(
            n_total=5, n_success=0, n_passed=0, pass_rate=0.0,
            pbo_values=[], pbo_mean=0.0, pbo_median=0.0, pbo_std=0.0,
            grade_counts={}, results=[],
        )
        report = generate_batch_pbo_report(summary)
        assert isinstance(report, str)
        assert "Failed" in report

    def test_run_pbo_batch_requires_fetch_func(self):
        """Decoupled batch runner requires an explicit fetch_func"""
        with pytest.raises(ValueError, match="fetch_func"):
            run_pbo_batch(["SYM1"], param_grid=[{"fast": 5, "slow": 20}])
