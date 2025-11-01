#!/usr/bin/env python3
"""
Trace a small RLM run and output a CLI call tree + Mermaid call graph.

Usage:
  python scripts/trace_run.py --file data/fed_papers.txt --bytes 5000 --mermaid artifacts/callgraph.mmd
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
from rlm_utils.tracing import run_with_trace, render_cli_tree, export_mermaid  # type: ignore


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "..", "data"))
    ap.add_argument("--file", default=None)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--bytes", type=int, default=24_000)
    ap.add_argument("--query", default="List 5 salient topics and cite the source file.")
    ap.add_argument("--max-iters", type=int, default=4)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--api-base", default=None)
    ap.add_argument("--mermaid", default="artifacts/callgraph.mmd")
    args = ap.parse_args()

    # Build context
    if args.file:
        context = small_sample_from_file(args.file, args.bytes)
    else:
        context = small_sample_from_dir(args.data, args.k, args.bytes, include_all=args.all)

    # Env + model
    bootstrap_paths()
    monkey_patch_litellm()
    apply_proxy_env(args.api_base)
    model = effective_model("gemini-2.5-flash-lite")
    rlm = build_rlm(model, max_iterations=args.max_iters, enable_logging=False)

    # Trace the completion call
    def _run():
        return rlm.completion(context=context, query=args.query)

    result, roots, edges = run_with_trace(_run)

    print("\n=== FINAL ANSWER ===\n", result)
    print("\n=== CALL TREE (filtered to vendor/rlm + scripts) ===\n")
    render_cli_tree(roots)

    if args.mermaid:
        export_mermaid(edges, args.mermaid)
        print(f"\nMermaid call graph written to: {args.mermaid}")


if __name__ == "__main__":
    main()
