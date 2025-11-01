from __future__ import annotations

import os
from typing import Any

from .pathing import bootstrap_paths


def monkey_patch_litellm() -> None:
    """Patch the vendored RLM to use our LiteLLM client for both root + sub calls."""
    bootstrap_paths()
    # Import after path bootstrap
    from rlm.utils.litellm_client import LiteLLMClient  # type: ignore
    import rlm.utils.llm as llm_mod  # type: ignore
    import rlm.rlm_repl as rlm_repl_mod  # type: ignore

    llm_mod.OpenAIClient = LiteLLMClient  # type: ignore[attr-defined]
    rlm_repl_mod.OpenAIClient = LiteLLMClient  # type: ignore[attr-defined]


def build_rlm(model: str, max_iterations: int = 6, enable_logging: bool = True) -> Any:
    """Return an RLM_REPL instance with our chosen model and settings."""
    bootstrap_paths()
    from rlm.rlm_repl import RLM_REPL  # type: ignore

    return RLM_REPL(
        model=model,
        recursive_model=model,
        enable_logging=enable_logging,
        max_iterations=max_iterations,
    )

