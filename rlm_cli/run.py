from __future__ import annotations

import argparse
import os

from rlm_utils.env import apply_proxy_env, effective_model
from rlm_utils.pathing import bootstrap_paths
from rlm_utils.rlm_adapter import monkey_patch_litellm, build_rlm
from rlm_utils.sampling import small_sample_from_dir, small_sample_from_file


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(os.getcwd(), "data"), help="data folder path")
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
    ap.add_argument("--all", action="store_true", help="include all file types (not only texty)")
    ap.add_argument("--api-base", default=None, help="LiteLLM proxy base URL")
    args = ap.parse_args()

    # Prepare tiny context
    if args.file:
        context = small_sample_from_file(args.file, args.bytes)
    else:
        context = small_sample_from_dir(args.data, args.k, args.bytes, include_all=args.all)

    # Env + model + RLM
    bootstrap_paths()
    monkey_patch_litellm()
    apply_proxy_env(args.api_base)
    model = effective_model("gemini-2.5-flash-lite")
    rlm = build_rlm(model, max_iterations=args.max_iters, enable_logging=True)

    print("Running RLM_REPL on a tiny sampled context...\n")
    result = rlm.completion(context=context, query=args.query)
    print("\n=== FINAL ANSWER ===\n" + str(result))


if __name__ == "__main__":
    main()

