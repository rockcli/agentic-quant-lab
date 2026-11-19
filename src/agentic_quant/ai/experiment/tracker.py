"""
Experiment tracker: log multi-round iteration strategy, parameters, backtest results, diagnosis suggestions

Ref: Experimentation for Engineers (Kohavi et al.), Ch.3
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from agentic_quant.ai.pipeline.diagnose import DiagnosisReport
from agentic_quant.backtest.core.result import BacktestResult


@dataclass
class ExperimentRecord:
    """Single experiment log"""

    iteration: int
    timestamp: str
    strategy_name: str
    strategy_code: str
    params: dict[str, Any]
    result_metrics: dict[str, Any]
    diagnosis: dict[str, Any] = field(default_factory=dict)
    parent_iteration: int | None = None


class ExperimentTracker:
    """
    Experiment tracker

    - Organized by experiment name directory
    - Save JSON Lines per iteration
    - Supports comparing multi-round results, extracting optimal experiment
    """

    def __init__(self, experiment_name: str, base_dir: str | Path = "data/experiments"):
        self.experiment_name = experiment_name
        self.base_dir = Path(base_dir)
        self.exp_dir = self.base_dir / experiment_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.records: list[ExperimentRecord] = []
        self._load_existing()

    def _load_existing(self) -> None:
        """Load existing logs"""
        path = self.exp_dir / "records.jsonl"
        if not path.exists():
            return
        for line in path.read_text(encoding="utf-8").strip().split("\n"):
            if not line.strip():
                continue
            data = json.loads(line)
            self.records.append(ExperimentRecord(**data))

    def log(
        self,
        iteration: int,
        strategy_name: str,
        strategy_code: str,
        params: dict[str, Any],
        result: BacktestResult,
        diagnosis: DiagnosisReport | None = None,
        parent_iteration: int | None = None,
    ) -> None:
        """Log one round of experiment"""
        record = ExperimentRecord(
            iteration=iteration,
            timestamp=datetime.now().isoformat(),
            strategy_name=strategy_name,
            strategy_code=strategy_code,
            params=params,
            result_metrics=result.to_dict(),
            diagnosis={
                "summary": diagnosis.summary if diagnosis else "",
                "suggestions": diagnosis.suggestions if diagnosis else [],
                "priority": diagnosis.priority if diagnosis else [],
            },
            parent_iteration=parent_iteration,
        )
        self.records.append(record)
        self._append(record)

    def _append(self, record: ExperimentRecord) -> None:
        """Append to JSON Lines"""
        path = self.exp_dir / "records.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record.__dict__, ensure_ascii=False, default=str) + "\n")

    def best_by(self, metric: str = "sharpe_ratio", maximize: bool = True) -> ExperimentRecord | None:
        """Find optimal experiment by metric"""
        if not self.records:
            return None
        valid = [r for r in self.records if metric in r.result_metrics]
        if not valid:
            return None
        return max(valid, key=lambda r: r.result_metrics[metric]) if maximize else min(
            valid, key=lambda r: r.result_metrics[metric]
        )

    def compare(self, metric: str = "sharpe_ratio") -> list[dict[str, Any]]:
        """Compare specified metrics across experiment rounds"""
        return [
            {
                "iteration": r.iteration,
                "strategy": r.strategy_name,
                metric: r.result_metrics.get(metric),
                "parent": r.parent_iteration,
            }
            for r in self.records
        ]

    def lineage(self, iteration: int) -> list[ExperimentRecord]:
        """Trace the parent chain of a given experiment round"""
        record_map = {r.iteration: r for r in self.records}
        chain = []
        current = record_map.get(iteration)
        while current:
            chain.append(current)
            current = record_map.get(current.parent_iteration)  # type: ignore[arg-type]
        return list(reversed(chain))

    def summary(self) -> dict[str, Any]:
        """Experiment summary"""
        return {
            "experiment": self.experiment_name,
            "total_iterations": len(self.records),
            "best_sharpe": self.best_by("sharpe_ratio", maximize=True),
            "best_return": self.best_by("total_return", maximize=True),
            "best_drawdown": self.best_by("max_drawdown", maximize=False),
            "records": self.compare(),
        }
