from __future__ import annotations

from typing import Dict, Any, List


def export_sequence_mermaid(events: List[Dict[str, Any]], outfile: str, *, preview: int = 80) -> None:
    """Export a concise Mermaid sequence diagram of root iterations, sub-LLM calls,
    and code executions. Designed to be low-noise and highlight recursion.
    """
    import os
    os.makedirs(os.path.dirname(outfile), exist_ok=True)

    # Group by iteration
    by_iter: Dict[int, List[Dict[str, Any]]] = {}
    for e in events:
        it = int(e.get("iteration", 0) or 0)
        by_iter.setdefault(it, []).append(e)

    def trunc(s: str) -> str:
        s = s.replace("\n", " ")
        return s[:preview] + ("…" if len(s) > preview else "")

    with open(outfile, "w") as f:
        f.write("sequenceDiagram\n")
        f.write("  participant RootLM\n  participant REPL\n  participant SubLM\n")
        for it in sorted(by_iter.keys()):
            f.write(f"  rect rgba(200,200,200,0.15)\n  note right of RootLM: Iteration {it}\n")
            for e in by_iter[it]:
                k = e.get("kind") or e.get("kind", "")
                if k == "root_llm_call":
                    p = trunc(e.get("prompt_preview", ""))
                    f.write(f"  RootLM->>REPL: decide next action ({p})\n")
                elif k == "code_exec":
                    lines = e.get("lines", 0)
                    prev = trunc(e.get("preview", ""))
                    f.write(f"  RootLM->>REPL: exec code ({lines} lines)\n")
                    if prev:
                        f.write(f"  note over REPL: {prev}\n")
                elif k == "sub_llm_call":
                    mode = e.get("mode")
                    if mode == "prompt":
                        prev = trunc(e.get("prompt_preview", ""))
                        f.write(f"  REPL->>SubLM: llm_query {prev}\n")
                    else:
                        instr = trunc(e.get("instruction_preview", ""))
                        text_len = int(e.get("text_len", 0))
                        f.write(f"  REPL->>SubLM: llm_query_text (len={text_len}) {instr}\n")
            f.write("  end\n")

