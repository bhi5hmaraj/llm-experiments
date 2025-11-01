from __future__ import annotations

import argparse
import os

from rlm_utils.env import apply_proxy_env, effective_model
from rlm_utils.pathing import bootstrap_paths
from rlm_utils.rlm_adapter import monkey_patch_litellm, build_rlm
from rlm_utils.sampling import small_sample_from_dir, small_sample_from_file
from rlm_utils.sequence import export_sequence_mermaid
from rlm_utils.event_log import get_logger, reset_logger
from rlm_utils.summary import print_summary


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
    ap.add_argument("--mermaid", default="docs/graphs/sequence.mmd")
    ap.add_argument("--log", action="store_true", help="print a concise per-iteration summary")
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

    reset_logger()
    result = rlm.completion(context=context, query=args.query)
    events = get_logger().dump()
    export_sequence_mermaid(events, args.mermaid)

    print("\n=== FINAL ANSWER (truncated) ===\n", str(result)[:500])
    if args.log:
        print("\n=== RUN SUMMARY ===")
        print_summary(events)
    print(f"\nSequence diagram written to: {args.mermaid}")


if __name__ == "__main__":
    main()
