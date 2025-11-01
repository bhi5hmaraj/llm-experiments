from __future__ import annotations

from typing import Any, Dict, List


def _group_by_iteration(events: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    by_it: Dict[int, List[Dict[str, Any]]] = {}
    for e in events:
        it = int(e.get("iteration", 0) or 0)
        by_it.setdefault(it, []).append(e)
    return by_it


def build_summary(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_it = _group_by_iteration(events)
    rows: List[Dict[str, Any]] = []
    for it, evs in sorted(by_it.items()):
        root_calls = sum(1 for e in evs if e.get("kind") == "root_llm_call")
        root_resp = [e for e in evs if e.get("kind") == "root_llm_response"]
        has_code = sum(1 for e in root_resp if e.get("has_code"))
        code_exec = [e for e in evs if e.get("kind") == "code_exec"]
        sub_calls = [e for e in evs if e.get("kind") == "sub_llm_call"]
        total_lines = sum(int(e.get("lines", 0) or 0) for e in code_exec)
        total_sub_text = sum(int(e.get("text_len", 0) or 0) for e in sub_calls)
        instr_samples = []
        for e in sub_calls[:3]:
            mode = e.get("mode")
            if mode == "text":
                instr_samples.append((e.get("instruction_preview") or "").strip())
            else:
                instr_samples.append((e.get("prompt_preview") or "").strip())
        rows.append(
            dict(
                iteration=it,
                root_calls=root_calls,
                root_has_code=has_code,
                code_exec=len(code_exec),
                code_lines=total_lines,
                sub_calls=len(sub_calls),
                sub_text_kb=round(total_sub_text / 1024.0, 1),
                samples=instr_samples,
            )
        )
    return rows


def print_summary(events: List[Dict[str, Any]], *, show_samples: bool = True) -> None:
    try:
        from rich.table import Table
        from rich.console import Console
    except Exception:
        for row in build_summary(events):
            print(row)
        return
    rows = build_summary(events)
    table = Table(title="RLM run summary")
    table.add_column("iter", justify="right")
    table.add_column("root")
    table.add_column("has_code")
    table.add_column("exec")
    table.add_column("lines", justify="right")
    table.add_column("sub_calls", justify="right")
    table.add_column("sub_text_kb", justify="right")
    if show_samples:
        table.add_column("samples")
    for r in rows:
        table.add_row(
            str(r["iteration"]),
            str(r["root_calls"]),
            str(r["root_has_code"]),
            str(r["code_exec"]),
            str(r["code_lines"]),
            str(r["sub_calls"]),
            str(r["sub_text_kb"]),
            "; ".join([s[:60] for s in r["samples"]]) if show_samples else "",
        )
    Console().print(table)

