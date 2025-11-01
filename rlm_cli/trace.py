from __future__ import annotations

import argparse
import os

from rlm_utils.env import apply_proxy_env, effective_model
from rlm_utils.pathing import bootstrap_paths
from rlm_utils.rlm_adapter import monkey_patch_litellm, build_rlm
from rlm_utils.sampling import small_sample_from_dir, small_sample_from_file
from rlm_utils.tracing import run_with_trace, render_cli_tree, export_mermaid


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(os.getcwd(), "data"))
    ap.add_argument("--file", default=None)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--bytes", type=int, default=24_000)
    ap.add_argument("--query", default="List 5 salient topics and cite the source file.")
    ap.add_argument("--max-iters", type=int, default=4)
    ap.add_argument("--max-depth", type=int, default=1, help="recursive depth for sub-LLM calls")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--api-base", default=None)
    ap.add_argument("--mermaid", default="artifacts/callgraph.mmd")
    ap.add_argument("--min-ms", type=float, default=1.0, help="hide edges/nodes under this duration (ms)")
    ap.add_argument("--max-depth", type=int, default=3, help="maximum depth to render in tree")
    ap.add_argument(
        "--exclude",
        default=r"rlm\\.logger|repl_logger|_capture_output|_temp_working_directory|add_execution_result_to_messages|format_execution_result",
        help="regex of functions/files to exclude",
    )
    ap.add_argument("--top", type=int, default=8, help="print top-N heaviest edges")
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
    rlm = build_rlm(model, max_iterations=args.max_iters, enable_logging=False, max_depth=args.max_depth)

    # Trace the completion call
    def _run():
        return rlm.completion(context=context, query=args.query)

    result, roots, edges = run_with_trace(_run)

    print("\n=== FINAL ANSWER ===\n", result)
    from rlm_utils.tracing import filter_trace, render_top_edges
    filtered_roots = filter_trace(roots, min_ms=args.min_ms, max_depth=args.max_depth, exclude_patterns=args.exclude)
    print("\n=== CALL TREE (filtered) ===\n")
    render_cli_tree(filtered_roots)
    print("\n=== TOP EDGES ===\n")
    render_top_edges(edges, top_n=args.top, min_ms=args.min_ms, exclude_patterns=args.exclude)

    if args.mermaid:
        export_mermaid(edges, args.mermaid, min_ms=args.min_ms, exclude_patterns=args.exclude)
        print(f"\nMermaid call graph written to: {args.mermaid}")


if __name__ == "__main__":
    main()
