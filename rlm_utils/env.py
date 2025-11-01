from __future__ import annotations

import os
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


def normalize_api_base(api_base: Optional[str]) -> Optional[str]:
    if not api_base:
        return None
    base = api_base.rstrip("/")
    if not base.endswith("/v1"):
        base = base + "/v1"
    return base


def apply_proxy_env(api_base_arg: Optional[str] = None) -> None:
    """Apply LiteLLM proxy env vars consistently.

    - Normalizes LITELLM_API_BASE to end with /v1
    - Mirrors LITELLM_API_KEY -> OPENAI_API_KEY for OpenAI‑compatible proxies
    """
    api_base = normalize_api_base(api_base_arg or os.getenv("LITELLM_API_BASE"))
    if api_base:
        os.environ["LITELLM_API_BASE"] = api_base

    key = os.getenv("LITELLM_API_KEY")
    if key:
        os.environ["OPENAI_API_KEY"] = key
    else:
        os.environ.setdefault("OPENAI_API_KEY", "placeholder")


def effective_model(default: str = "gemini-2.5-flash-lite") -> str:
    return os.getenv("LITELLM_MODEL", default)

