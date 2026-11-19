"""
Strategy generation pipeline

Natural language -> retrieval theory -> LLM generate code -> validate -> save
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path

from agentic_quant.ai.llm_client import LLMClient
from agentic_quant.ai.rag.retriever import HybridRetriever


@dataclass
class GeneratedStrategy:
    description: str
    code: str
    code_valid: bool
    references: list[dict]
    error: str = ""


class StrategyGenerationPipeline:
    """
    Strategy generation pipeline
    """

    SYSTEM_PROMPT = """You are a quantitative strategy engineer. Based on strategy objectives and knowledge base theory snippets, generate runnable Python strategy code.

Requirements:
1. Code must inherit Strategy base class, implement on_bar method
2. Use Signal(timestamp, action, size, price) to generate signals
3. Include Chinese comments explaining WHY
4. Do not output pseudo-code, code must be directly runnable
5. Append 3-5 entries of "consistency explanation with knowledge base" after the code
"""

    def __init__(self, retriever: HybridRetriever, llm: LLMClient | None = None):
        self.retriever = retriever
        self.llm = llm or LLMClient()

    def run(self, description: str, books: list[str] | None = None) -> GeneratedStrategy:
        """
        Execute strategy generation pipeline

        Args:
            description: Natural language strategy description
            books: Books to limit retrieval to (optional)
        """
        # 1. Retrieve relevant theory
        results = self.retriever.search(description, top_k=5, filter_books=books)
        context = "\n\n".join(
            f"[{i+1}] {r['chunk'].title}\n{r['chunk'].text[:300]}..."
            for i, r in enumerate(results)
        )

        # 2. Assemble prompt
        user_prompt = self._build_prompt(description, context)

        # 3. call LLM
        try:
            response = self.llm.chat(self.SYSTEM_PROMPT, user_prompt)
            code = self._extract_code(response.content)
        except Exception as exc:
            return GeneratedStrategy(
                description=description,
                code="",
                code_valid=False,
                references=[{"chunk_id": r["chunk"].chunk_id} for r in results],
                error=f"LLM callFailed: {exc}",
            )

        # 4. CodeValidate
        valid, error = self._validate_code(code)

        return GeneratedStrategy(
            description=description,
            code=code,
            code_valid=valid,
            references=[{"chunk_id": r["chunk"].chunk_id} for r in results],
            error=error,
        )

    @staticmethod
    def _build_prompt(description: str, context: str) -> str:
        return f"""[Strategy Objective]
{description}

[Knowledge Base Retrieval Snippet]
{context}

[Output Requirements]
Please output complete Python code, including:
1. from agentic_quant.backtest.strategy.base import Strategy, Signal
2. from agentic_quant.backtest.data.structures import Bar
3. Class definition, inheriting Strategy
4. __init__ method to set parameters
5. on_bar method to implement trading logic
6. Chinese comments explaining WHY

Code format:
```python
# Your code
```
"""

    @staticmethod
    def _extract_code(content: str) -> str:
        """Extract Python code block from LLM output."""
        if "```python" in content:
            start = content.index("```python") + len("```python")
            end = content.index("```", start)
            return content[start:end].strip()
        if "```" in content:
            start = content.index("```") + len("```")
            end = content.index("```", start)
            return content[start:end].strip()
        return content.strip()

    @staticmethod
    def _validate_code(code: str) -> tuple[bool, str]:
        """
        CodeValidatelayer：
        - AST Parse
        - CheckInheritance Strategy
        - CheckImplement on_bar
        """
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            return False, f"Syntax error: {exc}"

        has_strategy_class = False
        has_on_bar = False

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                bases = [isinstance(b, ast.Name) and b.id == "Strategy" for b in node.bases]
                if any(bases):
                    has_strategy_class = True
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef) and item.name == "on_bar":
                            has_on_bar = True

        if not has_strategy_class:
            return False, "No class inheriting Strategy found"
        if not has_on_bar:
            return False, "No on_bar method found"

        return True, ""

    def save(self, result: GeneratedStrategy, output_dir: Path) -> Path:
        """Save generated strategy to file."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        safe_name = result.description[:20].replace(" ", "_").replace("/", "_")
        path = output_dir / f"generated_{safe_name}.py"

        header = f'"""\nGenerated by AI: {result.description}\nValidation Status: {"Passed" if result.code_valid else "Failed"}\n"""\n\n'
        path.write_text(header + result.code, encoding="utf-8")

        meta_path = path.with_suffix(".meta.json")
        meta_path.write_text(
            json.dumps(
                {
                    "description": result.description,
                    "code_valid": result.code_valid,
                    "error": result.error,
                    "references": result.references,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        return path
