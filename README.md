Recursive Language Models (RLM) — Author’s Minimal Repo + Offline Tests

What’s included
- Vendored upstream: `vendor/rlm` (alexzhang13/rlm — minimal RLM + REPL).
- Offline unittest suite: `tests/` exercising the REPL environment and utils.
- No network or API calls are needed to run the tests.

What is RLM?
- RLM (Recursive Language Model) wraps a base LLM in a control loop that can iteratively write and run code inside a persistent REPL and optionally call sub‑LLMs on demand. The persistence (shared globals/locals) gives the model a working memory so it can chunk, summarize, and aggregate information before producing a final answer.
- In this minimal repo, recursion depth > 1 is disabled by default: the REPL exposes `llm_query()` backed by a simple sub‑LM client. Replacing that sub‑LM with `RLM_REPL` enables deeper recursion, though you may want stronger isolation between nested REPLs.

How it works
- Controller (`rlm/rlm_repl.py`):
  - Seeds messages with a system prompt instructing the model to use the REPL and to end with `FINAL(...)` or `FINAL_VAR(...)`.
  - On each iteration, asks for the next action, parses ```repl``` code blocks, executes them, appends outputs back to the conversation, and checks for a final answer.
  - `FINAL(text)` is read from the assistant message text (not a REPL function). `FINAL_VAR(name)` is a real REPL helper that returns a variable from REPL locals.
- REPL (`rlm/repl.py`):
  - Sandboxed Python with persistent state, a `context` variable, `llm_query(prompt)`, and `FINAL_VAR(varname)`.
  - Captures `stdout`/`stderr`; prints the last bare expression result.
  - Runs inside a temp working directory.

Pseudo‑flow
```
messages = build_system_prompt()
repl = REPLEnv(context_str|json)
for step in range(max_iterations):
  response = LLM(messages + [next_action_prompt(...)])
  code_blocks = find_code_blocks(response)
  if code_blocks:
    for code in code_blocks:
      out = repl.code_execution(code)
      messages = add_execution_result_to_messages(messages, code, out)
  else:
    messages.append({"role":"assistant","content": response})
  final = check_for_final_answer(response, repl)
  if final:
    return final
return llm.completion(messages)
```

Why it helps with long context
- The REPL gives the model a scratch‑space to persist intermediate structures (lists, dicts, partial summaries). The sub‑LLM functions like a large “attention buffer” the model can query with curated chunks, enabling scalable search and synthesis beyond a single prompt/response.

Limits and safety
- This minimal REPL uses `exec` with a curated builtins allowlist but still runs arbitrary Python. Treat it as untrusted: sandbox it, avoid secrets, and prefer containers when possible.
- Depth > 1 recursion is possible but requires careful REPL isolation to avoid state leakage across nested calls.

Run tests
- `python -m unittest discover -s tests -p 'test_*.py' -v`

Notes on the vendored repo
- The upstream RLM implementation uses OpenAI via `openai`, plus `dotenv` and `rich` for logging. These are listed in `vendor/rlm/requirements.txt`.
- The REPL (`rlm/repl.py`) embeds a persistent Python environment and exposes `llm_query()` and `FINAL_VAR()` into that environment. Our tests verify:
  - State persists across executions.
  - Expression results print automatically.
  - `FINAL_VAR()` returns a variable from the REPL locals.
  - Disallowed builtins (e.g., `eval`, `input`) are sandboxed.
  - Execution runs inside a temp working directory.

Run the example (requires network + API key)
1) `python -m venv .venv && source .venv/bin/activate`
2) `pip install -r vendor/rlm/requirements.txt`
3) Export your key: `export OPENAI_API_KEY=...`
4) `python vendor/rlm/main.py`

Tip: the example generates ~1M-line context and will incur model usage; tweak `num_lines` and model names as needed.

Extending tests to the full RLM loop

Run a tiny example with LiteLLM (recommended)
- Install deps: `pip install openai litellm rich dotenv`
- Configure env vars:
  - `cp .env.example .env` and edit `.env` to add your key.
  - Or export in your shell: `export LITELLM_API_BASE=...; export LITELLM_API_KEY=...; export LITELLM_MODEL=...`
- Using a LiteLLM proxy:
  - `export LITELLM_API_BASE=https://<your-proxy>`
  - `export LITELLM_API_KEY=...`
  - `export LITELLM_MODEL="gemini-2.5-flash-lite"` (or your proxy alias)
  - The client forces `custom_llm_provider=litellm` when `LITELLM_API_BASE` is set, avoiding provider auto‑detection (e.g., Google).
- Run the tiny sampler over your `data/` folder:
  - `python scripts/run_small_example.py --data data --k 3 --bytes 24000 --max-iters 6 \
     --query "From these snippets, list 5 salient topics and cite the source file."`

What the tiny example does
- Randomly samples `k` files from `data/` and reads only the first `--bytes` bytes of each (keeps things small).
- Monkey‑patches the upstream client to use LiteLLM so you can point to Gemini (or any provider LiteLLM supports).
- Runs the RLM REPL controller for `--max-iters` steps and prints the final answer.
- The upstream `RLM_REPL` imports `openai` and `rich` at module import time via its logger and client. To test the full loop offline, you can stub modules before import or inject a mock `OpenAIClient`. The simpler path is to run with the real deps installed and an API key, then assert end-to-end behavior.

Tracing and call graphs
- CLI call tree + Mermaid output:
  - `python scripts/trace_run.py --file data/fed_papers.txt --bytes 5000 --max-iters 4 --mermaid artifacts/callgraph.mmd`
- Render Mermaid to SVG (optional):
  - `npm i -g @mermaid-js/mermaid-cli`
  - `mmdc -i artifacts/callgraph.mmd -o artifacts/callgraph.svg`

Code reuse across scripts
- Shared helpers live in `rlm_utils/`:
  - `env.py` — loads .env, normalizes proxy base, aligns keys
  - `pathing.py` — ensures vendored `rlm` is on `sys.path`
  - `sampling.py` — single-file/dir sampling
  - `rlm_adapter.py` — monkey‑patch LiteLLM + build `RLM_REPL`
  - `tracing.py` — run with tracer, render tree, export Mermaid
