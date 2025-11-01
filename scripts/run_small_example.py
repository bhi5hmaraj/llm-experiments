#!/usr/bin/env python3
"""
Run RLM_REPL on a tiny sampled context using LiteLLM.

This script:
- Samples a few small files from `data/` (first N bytes each)
- Monkey‑patches the upstream client to use LiteLLM
- Runs the controller with a short iteration budget

Setup (choose one provider and set its env var):
- Gemini direct:    export GEMINI_API_KEY=...; export LITELLM_MODEL="gemini/gemini-2.5-flash-lite"
- OpenRouter route: export OPENROUTER_API_KEY=...; export LITELLM_MODEL="google/gemini-2.5-flash-lite"
- Or generic:       export LITELLM_API_KEY=...

Dependencies: pip install litellm rich dotenv
"""

from __future__ import annotations

import argparse
import os

import sys
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from rlm_utils.env import apply_proxy_env, effective_model  # type: ignore
from rlm_utils.pathing import bootstrap_paths  # type: ignore
from rlm_utils.rlm_adapter import monkey_patch_litellm, build_rlm  # type: ignore
from rlm_utils.sampling import small_sample_from_dir, small_sample_from_file  # type: ignore


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "..", "data"), help="data folder path")
    ap.add_argument("--file", default=None, help="optional: run on a single file path instead of sampling a directory")
    ap.add_argument("--k", type=int, default=3, help="number of files to sample")
    ap.add_argument("--bytes", type=int, default=24_000, help="bytes to read per file")
    ap.add_argument(
        "--query",
        default=(
            "From these snippets, list 5 salient topics and cite the source file."
        ),
        help="user query",
    )
    ap.add_argument("--max-iters", type=int, default=6)
    ap.add_argument("--api-base", default=None, help="LiteLLM proxy base URL, e.g. https://your-proxy.example.com")
    ap.add_argument("--all", action="store_true", help="include all file types (not only texty)")
    args = ap.parse_args()

    # Prepare tiny context
    if args.file:
        context = small_sample_from_file(args.file, args.bytes)
    else:
        context = small_sample_from_dir(args.data, args.k, args.bytes, include_all=args.all)

    # Import upstream modules
    bootstrap_paths()
    monkey_patch_litellm()
    apply_proxy_env(args.api_base)

    # Model selection (neutral alias works with LiteLLM proxy)
    model = effective_model("gemini-2.5-flash-lite")

    # Build RLM with short budget
    rlm = build_rlm(model, max_iterations=args.max_iters, enable_logging=True)

    print("Running RLM_REPL on a tiny sampled context...\n")
    result = rlm.completion(context=context, query=args.query)
    print("\n=== FINAL ANSWER ===\n" + str(result))


if __name__ == "__main__":
    main()
