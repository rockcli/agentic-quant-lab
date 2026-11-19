"""
LLM Interface abstraction layer

Support multi-provider switching, automatic retry, caching, cost tracking.
Uses standard library urllib for HTTP requests, avoids extra dependencies.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from urllib import request

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    content: str
    usage: dict
    model: str


class LLMClient:
    """
    Universal LLM HTTP client (OpenAI/DeepSeek/Kimi compatible format)

    Passed via environment variable configuration:
    - DEFAULT_LLM_PROVIDER=deepseek
    - DEEPSEEK_API_KEY=sk-xxx
    - DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
    """

    def __init__(
        self,
        provider: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        cache_dir: str | Path = "data/cache/llm_cache",
    ):
        self.provider = provider or os.getenv("DEFAULT_LLM_PROVIDER", "deepseek") or "deepseek"
        self.api_key = str(api_key or os.getenv(f"{self.provider.upper()}_API_KEY", "") or "")
        raw_base_url = base_url or os.getenv(f"{self.provider.upper()}_BASE_URL", "") or ""
        self.base_url = raw_base_url.rstrip("/")
        self.model = model or self._default_model()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0

    def _default_model(self) -> str:
        defaults = {
            "openai": "gpt-4o-mini",
            "deepseek": "deepseek-chat",
            "kimi": "moonshot-v1-8k",
        }
        return defaults.get(self.provider, "deepseek-chat")

    def _cache_path(self, prompt_hash: str) -> Path:
        return self.cache_dir / f"{prompt_hash}.json"

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_retries: int = 3,
        use_cache: bool = True,
    ) -> LLMResponse:
        """
        Send chat request, supports caching and retry.
        """
        cache_key = {
            "provider": self.provider,
            "model": self.model,
            "temperature": temperature,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
        }
        cache_payload = json.dumps(cache_key, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        prompt_hash = hashlib.sha256(cache_payload.encode()).hexdigest()[:16]
        cache_path = self._cache_path(prompt_hash)

        if use_cache and cache_path.exists():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                # 24h cache validity period
                if time.time() - cached.get("cached_at", 0) < 86400:
                    logger.debug("LLM cache hit: %s", prompt_hash)
                    return LLMResponse(
                        content=cached["content"],
                        usage=cached["usage"],
                        model=cached["model"],
                    )
            except Exception:
                pass

        if not self.api_key or not self.base_url:
            raise RuntimeError(
                f"LLM not configured: provider={self.provider}, "
                f"please set {self.provider.upper()}_API_KEY and BASE_URL"
            )

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = request.Request(
            f"{self.base_url}/chat/completions",
            data=data,
            headers=headers,
            method="POST",
        )

        last_exc: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                with request.urlopen(req, timeout=60) as resp:
                    result = json.loads(resp.read().decode("utf-8"))
                break
            except Exception as exc:
                last_exc = exc
                logger.warning("LLM RequestFailed(%s/%s): %s", attempt, max_retries, exc)
                if attempt < max_retries:
                    time.sleep(2**attempt)
        else:
            raise RuntimeError(f"LLM request ultimately failed: {last_exc}")

        choice = result["choices"][0]
        content = choice["message"]["content"]
        usage = result.get("usage", {})
        model = result.get("model", self.model)

        self._total_prompt_tokens += usage.get("prompt_tokens", 0)
        self._total_completion_tokens += usage.get("completion_tokens", 0)

        if use_cache:
            cache_path.write_text(
                json.dumps(
                    {
                        "content": content,
                        "usage": usage,
                        "model": model,
                        "cached_at": time.time(),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

        return LLMResponse(content=content, usage=usage, model=model)

    def cost_summary(self) -> dict:
        """Track accumulated token consumption."""
        return {
            "provider": self.provider,
            "prompt_tokens": self._total_prompt_tokens,
            "completion_tokens": self._total_completion_tokens,
        }
