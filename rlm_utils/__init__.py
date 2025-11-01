"""Shared helpers for running and tracing RLM scripts.

Modules:
- pathing: ensure vendor path on sys.path
- env: load .env, normalize proxy base, align keys
- sampling: file/directory small sampling helpers
- rlm_adapter: monkey‑patch LiteLLM client + build RLM_REPL
- tracing: lightweight function call tracer + Mermaid export
"""

