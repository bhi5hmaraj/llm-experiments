"""
LiteLLM client wrapper exposing the same interface as OpenAIClient.

Usage expectations:
- Provide a model string via ctor or env var `LITELLM_MODEL`.
- Provide an API key via provider-specific env var (preferred):
  - Gemini direct: `GEMINI_API_KEY`
  - OpenRouter (routes to Gemini): `OPENROUTER_API_KEY`
  - Or generic: `LITELLM_API_KEY`

This class mirrors `OpenAIClient.completion()` signature used by RLM_REPL.
"""

from __future__ import annotations

import os
from typing import Optional, Union, List, Dict

try:
    from dotenv import load_dotenv  # optional
    load_dotenv()
except Exception:
    pass


class LiteLLMClient:
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.5-flash-lite"):
        # Model can be overridden by env var
        self.model = os.getenv("LITELLM_MODEL", model)

        # Key handling: prefer provider-specific envs; fallback to LITELLM_API_KEY; finally param
        # Do not print or log keys.
        # Prefer proxy key when provided; fall back to provider keys for direct mode
        self.api_key = (
            api_key
            or os.getenv("LITELLM_API_KEY")
            or os.getenv("OPENROUTER_API_KEY")
            or os.getenv("GEMINI_API_KEY")
        )

        if not self.api_key:
            raise ValueError(
                "Missing API key. Set GEMINI_API_KEY or OPENROUTER_API_KEY or LITELLM_API_KEY."
            )

        # Surface to litellm via env; provider will read the right var.
        # We set a generic var as last resort to support some provider setups.
        os.environ.setdefault("LITELLM_API_KEY", self.api_key)

        # Lazy import to avoid making litellm a hard requirement for offline tests
        try:
            import litellm  # noqa: F401
        except Exception as e:
            raise RuntimeError(
                "litellm is not installed. Run `pip install litellm`"
            ) from e

    def completion(
        self,
        messages: Union[List[Dict[str, str]], Dict[str, str], str],
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        from litellm import completion as llm_completion

        # Normalize messages into list[dict]
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        elif isinstance(messages, dict):
            messages = [messages]

        # Allow proxy base URL via env
        api_base = (
            os.getenv("LITELLM_API_BASE")
            or os.getenv("LITELLM_BASE_URL")
            or os.getenv("OPENAI_API_BASE")  # some proxies reuse this var
        )

        params = {
            "model": self.model,
            "messages": messages,
        }
        if max_tokens is not None:
            params["max_tokens"] = max_tokens

        if api_base:
            # Ensure OpenAI-compatible routing for LiteLLM Proxy
            # Append /v1 if the base does not already include it
            base = api_base.rstrip("/")
            if not base.endswith("/v1"):
                base = base + "/v1"
            params["api_base"] = base
            params["custom_llm_provider"] = "openai"
            # Pass key explicitly for proxies; also set provider-specific key
            params["api_key"] = self.api_key
            params["openai_api_key"] = self.api_key
            # Ensure env var fallback also has the correct key (some code paths read env)
            os.environ["OPENAI_API_KEY"] = self.api_key
        params.update(kwargs)

        resp = llm_completion(**params)

        # Try attribute access first; fallback to dict style
        try:
            content = getattr(resp.choices[0].message, "content", None)  # type: ignore[attr-defined]
            return content or ""
        except Exception:
            try:
                content = resp["choices"][0]["message"].get("content")  # type: ignore[index]
                return content or ""
            except Exception as e:
                raise RuntimeError(f"Unexpected LiteLLM response shape: {type(resp)}") from e
