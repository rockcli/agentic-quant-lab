# agentic-quant-lab

[![CI](https://github.com/rockcli/agentic-quant-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/rockcli/agentic-quant-lab/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/rockcli/agentic-quant-lab?display_name=tag&color=10B981)](https://github.com/rockcli/agentic-quant-lab/releases/tag/v0.1.0)
[![Python >=3.11](https://img.shields.io/badge/python-%3E%3D3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

[English](README.md) | [简体中文](README.zh-CN.md)

**Knowledge-grounded agentic quantitative strategy research & anti-overfitting validation framework.**

`agentic-quant-lab` (`agentic_quant`) combines a local RAG-grounded LLM research loop (Generate -> Backtest -> Diagnose -> Iterate -> Track) with a statistical anti-overfitting validation suite (CSCV/PBO, Deflated Sharpe Ratio, Purged Walk-Forward Analysis, 4-Quadrant Market Regime decomposition, and Out-of-Sample Embargo tracking).

Market data ingestion and live broker execution are intentionally **decoupled**: every pipeline, backtest engine, and validation gate operates purely on caller-supplied `Bar` lists, NumPy OHLCV arrays, or an explicit `fetch_func` callback.

---

## Architecture

```
+-------------------------------------------------------------------------+
|                        1. Knowledge-Grounded RAG                        |
|  docs/knowledge_base/*.md                                               |
|    -> MarkdownChunker -> TfidfEmbedder / SentenceTransformersEmbedder   |
|    -> InMemoryVectorStore -> HybridRetriever / BookRetriever            |
+------------------------------------+------------------------------------+
                                     | retrieved theory snippets
                                     v
+-------------------------------------------------------------------------+
|                       2. Agentic Research Loop                          |
|  ResearchOrchestrator                                                   |
|    -> StrategyGenerationPipeline (natural language -> AST-checked code) |
|    -> BacktestEngine (event-driven, close / next_open execution)        |
|    -> DiagnosisPipeline (AFML-inspired flaw & left-tail risk diagnosis) |
|    -> AutoIterationPipeline <-> ExperimentTracker (JSONL lineage log)   |
+------------------------------------+------------------------------------+
                                     | candidate strategies & param grids
                                     v
+-------------------------------------------------------------------------+
|                    3. Anti-Overfitting Validation Gate                  |
|  +-----------------------+  +--------------------+  +-----------------+ |
|  | Overfit Detection     |  | Purged Walk-Forward|  | Embargo &       | |
|  | - CSCV / PBO          |  | - Purge gap        |  | Regime Analysis | |
|  | - Deflated Sharpe     |  | - Majority rule    |  | - 4-quadrant    | |
|  | - Parameter stability |  | - Catastrophic veto|  |   trend x vol   | |
|  |   & cliff detection   |  | - Regime breakdown |  | - OOS tracking  | |
|  +-----------------------+  +--------------------+  +-----------------+ |
+-------------------------------------------------------------------------+
```

---

## Key Capabilities

### 1. Agentic Strategy Research Loop (`agentic_quant.ai`)
- **Zero-bloat LLM Client (`LLMClient`)**: Standard-library `urllib` HTTP client supporting OpenAI-compatible endpoints (`DEFAULT_LLM_PROVIDER`, `<PROVIDER>_API_KEY`, `<PROVIDER>_BASE_URL`), with automatic retries, token accounting, and SHA-256 disk response caching (`24h` TTL).
- **Local Hybrid RAG (`agentic_quant.ai.rag`)**:
  - `MarkdownChunker` + `TfidfEmbedder` (zero external ML dependencies) + `InMemoryVectorStore` + `HybridRetriever` (semantic cosine + keyword matching, optional `SentenceTransformersEmbedder`).
  - `BookRetriever`: lightweight BM25-style retriever over local markdown knowledge bases (`docs/knowledge_base/`).
- **End-to-End Orchestration (`ResearchOrchestrator`)**:
  - `StrategyGenerationPipeline`: retrieves domain context, constructs prompts, extracts Python code, and validates syntax via `ast.parse` plus `Strategy` subclass checks.
  - `DiagnosisPipeline`: summarizes backtest metrics, worst-performing trades, and drawdown windows into structured JSON diagnoses (`DiagnosisReport`).
  - `AutoIterationPipeline` & `ExperimentTracker`: runs iterative parameter refinement rounds and persists experiment lineage (`records.jsonl`) for parent-child comparison.

### 2. Anti-Overfitting Validation Suite (`agentic_quant.validation` & `agentic_quant.optimization`)
- **CSCV & Probability of Backtest Overfitting (`overfit.py`)**:
  - `cscv_pbo`: Combinatorially Symmetric Cross-Validation over chronological time blocks (preserving autocorrelation within each block), outputting `PBOResult` with PBO, logit scores, IS-to-OOS performance degradation, Deflated Sharpe Ratio (`_compute_deflated_sharpe`), and letter grades (`A`/`B`/`C`/`D`).
  - `ParameterStabilityAnalyzer`: 1D and grid marginal parameter plateau width and cliff detection (>50% adjacent drop).
  - `run_pbo_single` & `run_pbo_batch`: parameter-grid PBO evaluation over OHLCV arrays or caller-supplied `fetch_func`.
- **Purged Walk-Forward Analysis (`walk_forward.py`)**:
  - `WalkForwardOptimizer` & `WalkForwardAnalyzer`: rolling IS-OOS splits with explicit **purge gap** to prevent label leakage, **majority rule** (>=70% positive OOS windows), **catastrophic veto** on severe window drawdowns, parameter stability scoring, and Markdown/JSON report generation.
- **4-Quadrant Market Regime Classifier (`regime.py`)**:
  - `MarketRegime`: classifies bars into `bull_high_vol`, `bull_low_vol`, `bear_high_vol`, and `bear_low_vol` using MA(200) + rolling slope and rolling 20-day volatility vs. historical median, enabling regime-segmented WFA evaluation.
- **Out-of-Sample Embargo Tracking (`embargo.py`)**:
  - `EmbargoManager`: logs post-backtest OOS signals, verifies realized next-day OHLCV outcomes, tracks cumulative PnL, triggers stop-loss / daily / cumulative circuit breakers, and produces daily reports.
- **Pure-NumPy Signal Engine & Risk Metrics (`signal_engine.py`, `labeling.py`, `vpa.py`, `weekly.py`, `risk/metrics.py`)**:
  - Vectorized signals (`sma`, `kama`, `donchian`, `donchian_regime`, `bollinger`), Volume-Price Analysis confirmation, Triple-Barrier labeling (`triple_barrier_labels`), and non-normal Sharpe diagnostics (`probabilistic_sharpe`, `deflated_sharpe`, `calmar_ratio`, `sortino_ratio`, `omega_ratio`).

### 3. Event-Driven Backtest Engine & Sample Strategies (`agentic_quant.backtest`, `agentic_quant.strategy`)
- `BacktestEngine`: bar-by-bar simulation supporting `execution_timing="close"` or `"next_open"` (to avoid look-ahead bias), commissions, slippage, and `BacktestResult` metrics via Polars.
- Generic textbook sample strategies in `agentic_quant.strategy.examples`: `SmaCrossStrategy`, `DonchianBreakout`, `BollingerMeanReversion`, `PairsTrading`, `VolumeProfileStrategy`, `MetaLabelingVpaStrategy`.

---

## Installation

Requires Python `>=3.11`. Core runtime dependencies are only `numpy` and `polars`.

```bash
# With uv (recommended)
uv sync --extra dev

# Or with pip
pip install -e ".[dev]"
```

Optional extras:
- `.[rag]` — adds `sentence-transformers` for dense embeddings in `SentenceTransformersEmbedder`
- `.[optimize]` — adds `optuna` for `BayesianOptimizer`

---

## Quickstart

### 1. CSCV / PBO Overfitting Check

```python
import numpy as np
from agentic_quant.validation.overfit import cscv_pbo, ParameterStabilityAnalyzer

# Shape: (n_strategy_variants, n_periods)
rng = np.random.default_rng(42)
returns_matrix = rng.normal(0.0002, 0.01, size=(12, 240))

pbo_res = cscv_pbo(returns_matrix, n_combinations=100, n_blocks=16)
print(f"PBO={pbo_res.pbo:.2%}, DSR={pbo_res.dsr:.4f}, Grade={pbo_res.grade}")

stability = ParameterStabilityAnalyzer.analyze(
    param_values=[5, 10, 15, 20, 25],
    performance_scores=[0.6, 1.1, 1.15, 1.05, 0.7],
    param_name="fast_window",
)
print(f"Plateau={stability.plateau_width:.1%}, HasCliff={stability.has_cliff}")
```

### 2. Backtest + Purged Walk-Forward Analysis

```python
from datetime import datetime, timedelta
import numpy as np
from agentic_quant.backtest.data.structures import Bar
from agentic_quant.optimization.walk_forward import WalkForwardAnalyzer
from agentic_quant.strategy.examples import SmaCrossStrategy

rng = np.random.default_rng(7)
prices = 100.0 + np.cumsum(rng.normal(0.05, 1.0, size=300))
start = datetime(2024, 1, 1)
bars = [
    Bar(
        timestamp=start + timedelta(days=i),
        open=float(p),
        high=float(p + 0.5),
        low=float(p - 0.5),
        close=float(p),
        volume=10_000.0,
    )
    for i, p in enumerate(prices)
]

wfa = WalkForwardAnalyzer(
    total_start=bars[0].timestamp,
    total_end=bars[-1].timestamp,
    train_rolls=5,
    train_pct=0.6,
    purge_days=5,
)
report = wfa.analyze(
    strategy_cls=SmaCrossStrategy,
    bars=bars,
    param_grid=[{"fast": f, "slow": s} for f in (5, 10) for s in (20, 30)],
)
print(f"OOS Positive Rate={report.oos_positive_rate:.1%}, Grade={report.grade}")
```

### 3. Agentic Research Loop (Offline Mock or Live LLM)

Place domain markdown notes under `docs/knowledge_base/`, configure your OpenAI-compatible LLM environment variables (e.g., `DEFAULT_LLM_PROVIDER=deepseek`, `DEEPSEEK_API_KEY`, `DEEPSEEK_BASE_URL`), and run:

```python
from agentic_quant.ai.pipeline.orchestrator import ResearchOrchestrator

orch = ResearchOrchestrator(docs_dir="docs/knowledge_base")
session = orch.run(
    description="Trend-following moving average crossover with volatility-scaled stop loss",
    bars=bars,
    max_iterations=2,
)
print(session.diagnosis)
```

---

## Project Layout

```text
src/agentic_quant/
├── ai/
│   ├── experiment/tracker.py       # JSONL experiment lineage tracker
│   ├── pipeline/
│   │   ├── generate.py             # RAG + LLM code generation & AST check
│   │   ├── diagnose.py             # Structured failure & tail-risk diagnosis
│   │   ├── iterate.py              # Automated parameter mutation & re-backtest
│   │   └── orchestrator.py         # End-to-end research session coordinator
│   ├── rag/                        # Markdown chunker, TF-IDF/BM25/hybrid retrievers, prompts
│   ├── llm_client.py               # Stdlib urllib LLM client with disk cache
│   └── mvp_validation.py           # Snapshot & risk-review prompt builder
├── backtest/                       # Event-driven engine, portfolio, result, Bar/Tick, features
├── indicators/                     # Wilder ATR
├── optimization/                   # GridSearch, Bayesian (optional optuna), Purged WFA
├── risk/                           # Calmar, Sortino, Omega, PSR, DSR
├── strategy/                       # Strategy registry + generic sample strategies
└── validation/                     # CSCV/PBO, Embargo, Regime, Signal Engine, VPA, Labeling, Weekly
tests/                              # Offline unit & contract test suite
```

---

## Running Tests

```bash
uv run --with pytest --with numpy --with polars pytest
```

All tests are offline and self-contained using synthetic market series and mocked LLM responses.

---

## Non-Goals

- **Not a live trading, brokerage, or market-data platform** — Market data feeds and order routing are decoupled (`fetch_func=None` raises `ValueError`).
- **Not an autonomous sandboxed strategy runner** — `ResearchOrchestrator.run()` intentionally does **not** `exec()` LLM-generated strategies (it runs a placeholder `SmaCrossStrategy` until a human reviews and integrates generated `.py` files).
- **Not an intraday tick/order-book or margin simulator** — `BacktestEngine` simulates single-asset bars with flat commission rates and additive slippage, without borrow/funding fees or partial fills.
- **Not a persistent vector database** — `InMemoryVectorStore` holds NumPy arrays in RAM and rebuilds its TF-IDF index per process.
- **Not financial advice** — Statistical validation gates reduce overfitting risk on historical bars but do not guarantee future performance.

## Known Pitfalls

- **Default `close` timing vs strict T+1 execution** — `BacktestEngine` defaults to `execution_timing="close"` for legacy compatibility across WFA/grid/iteration loops; strict next-open T+1 execution (which excludes overnight entry gaps) is enforced in `validation/signal_engine.py` (`run_strategy()`).
- **CSCV sampling & direction-agnostic DSR** — `cscv_pbo()` uses Monte Carlo block sampling rather than exhaustive `C(S, S/2)` combinations, and `_compute_deflated_sharpe()` uses `abs(mean_logit)` so a consistently overfitting strategy (`logit > 0`) also scores high `dsr`; always evaluate `dsr` together with `pbo`.
- **Helper look-ahead & short-window boundaries** — `apply_weekly_filter()` broadcasts each week's final close across Mon–Fri unless explicitly lagged by 1 week; `MarketRegime.classify()` computes full-sample median volatility for post-hoc WFA attribution (use `classify_regime_percentile()` for causal features); default `purge_days=5` drops short intraday windows whose test span is shorter than the purge gap.

## Security & Trust Boundaries

- **Zero dynamic execution in core package** — `src/agentic_quant/` contains zero `subprocess`, `eval`, `exec`, or `pickle` calls; generated strategies are only AST-parsed for structure (`ast.parse`) and must be audited manually before import/execution.
- **Scoped network & disk cache** — Only `LLMClient.chat()` touches the network (`urllib.request.urlopen` to `<PROVIDER>_BASE_URL/chat/completions` with a Bearer token); treat `base_url` as trusted configuration (no URL scheme/host allowlist is enforced). Plaintext LLM prompts and responses are cached for 24 hours under `data/cache/llm_cache/` keyed by a 16-hex-char SHA-256 prefix.
- **Trusted local inputs** — Markdown files in `docs/knowledge_base/` are injected verbatim into RAG prompts without prompt-injection filtering, and `StrategyGenerationPipeline.save()` builds output filenames from lightly sanitized `description` prefixes.

---

## Methodology Notes

- This library provides **research and statistical validation tooling**; it makes no claims of future trading returns or live execution readiness without your own out-of-sample, transaction-cost, and slippage verification.
- Passing `fetch_func=None` to `run_pbo_batch` intentionally raises `ValueError` because live market data providers are decoupled from the core package.

## License

MIT © [rockcli](LICENSE)
