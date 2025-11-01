from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class CallRecord:
    func: str
    file: str
    line: int
    started_at: float
    children: List["CallRecord"] = field(default_factory=list)
    duration: float = 0.0
    count: int = 1


def _func_name(frame) -> str:
    code = frame.f_code
    mod = frame.f_globals.get("__name__", "?")
    qual = code.co_name
    return f"{mod}.{qual}"


def _file_in_scope(filename: str) -> bool:
    if not filename:
        return False
    filename = os.path.abspath(filename)
    # Limit to our repo: vendor/rlm/rlm and scripts
    return (
        "/vendor/rlm/rlm/" in filename.replace("\\", "/")
        or "/scripts/" in filename.replace("\\", "/")
        or "/rlm_cli/" in filename.replace("\\", "/")
    )


def run_with_trace(fn, *args, **kwargs):
    """Run a function with a filtered call tracer; returns (result, root_records, edges).

    edges: Dict[(caller, callee), (count, total_time)]
    """
    stack: List[CallRecord] = []
    roots: List[CallRecord] = []
    edges: Dict[Tuple[str, str], Tuple[int, float]] = {}

    def tracer(frame, event, arg):
        if event not in ("call", "return"):
            return tracer
        filename = frame.f_code.co_filename
        if not _file_in_scope(filename):
            return tracer

        now = time.perf_counter()
        if event == "call":
            rec = CallRecord(
                func=_func_name(frame), file=filename, line=frame.f_code.co_firstlineno, started_at=now
            )
            if stack:
                stack[-1].children.append(rec)
            else:
                roots.append(rec)
            stack.append(rec)
            return tracer
        # return event
        if stack:
            rec = stack.pop()
            rec.duration += now - rec.started_at
            if stack:
                caller = stack[-1].func
                key = (caller, rec.func)
                count, total = edges.get(key, (0, 0.0))
                edges[key] = (count + 1, total + rec.duration)
        return tracer

    sys.setprofile(tracer)
    try:
        result = fn(*args, **kwargs)
    finally:
        sys.setprofile(None)
    return result, roots, edges


def _should_exclude(func: str, file: str, exclude_patterns: Optional[str]) -> bool:
    if not exclude_patterns:
        return False
    import re
    return re.search(exclude_patterns, func) or re.search(exclude_patterns, file)


def _prune_and_collapse(
    rec: CallRecord,
    *,
    min_ms: float,
    max_depth: int,
    depth: int = 0,
    exclude_patterns: Optional[str] = None,
) -> Optional[CallRecord]:
    # Filter children first
    new_children: List[CallRecord] = []
    for ch in rec.children:
        kept = _prune_and_collapse(
            ch,
            min_ms=min_ms,
            max_depth=max_depth,
            depth=depth + 1,
            exclude_patterns=exclude_patterns,
        )
        if kept is not None:
            new_children.append(kept)

    # Collapse consecutive children with same func
    collapsed: List[CallRecord] = []
    for ch in new_children:
        if collapsed and collapsed[-1].func == ch.func:
            prev = collapsed[-1]
            prev.count += ch.count
            prev.duration += ch.duration
            prev.children.extend(ch.children)
        else:
            collapsed.append(ch)
    rec.children = collapsed

    # Exclude this node?
    if _should_exclude(rec.func, rec.file, exclude_patterns):
        # Promote children upwards by returning a synthetic node when we're at root depth > 0,
        # otherwise if top-level, keep children for caller to attach
        if rec.children:
            # If excluded but has children, return a virtual node by merging children into parent level
            # Caller will append these children; indicate by returning a dummy None and handling above.
            # Instead, return a minimal container with 0 duration but same children so parent can attach.
            container = CallRecord(func="", file=rec.file, line=rec.line, started_at=rec.started_at)
            container.children = rec.children
            container.duration = 0.0
            container.count = 0
            return container
        # No children and excluded: drop
        return None

    # Depth / duration pruning (keep if has children even if short)
    too_deep = depth >= max_depth
    too_short = (rec.duration * 1000.0) < min_ms
    if (too_short and not rec.children) or too_deep:
        return None

    return rec


def filter_trace(
    roots: List[CallRecord], *, min_ms: float = 1.0, max_depth: int = 3, exclude_patterns: Optional[str] = None
) -> List[CallRecord]:
    filtered: List[CallRecord] = []
    for r in roots:
        kept = _prune_and_collapse(r, min_ms=min_ms, max_depth=max_depth, exclude_patterns=exclude_patterns)
        if kept is None:
            continue
        # If we got a container (func==""), lift its children
        if kept.func == "" and kept.children:
            filtered.extend(kept.children)
        else:
            filtered.append(kept)
    return filtered


def render_cli_tree(roots: List[CallRecord], max_children: int = 8) -> None:
    try:
        from rich.tree import Tree
        from rich.console import Console
    except Exception:
        print("Install 'rich' to render CLI tree")
        return

    def add_node(tree, rec: CallRecord):
        name = rec.func.split(".")[-1]
        label = f"{os.path.basename(rec.file)}:{rec.line} • {name} ({rec.duration*1000:.1f}ms" + (f" ×{rec.count}" if rec.count > 1 else ")")
        node = tree.add(label)
        for child in rec.children[:max_children]:
            add_node(node, child)

    console = Console()
    root_tree = Tree("Call Tree")
    for r in roots:
        add_node(root_tree, r)
    console.print(root_tree)


def export_mermaid(
    edges: Dict[Tuple[str, str], Tuple[int, float]],
    outfile: str,
    *,
    min_ms: float = 1.0,
    exclude_patterns: Optional[str] = None,
    label_max: int = 40,
) -> None:
    """Export a Mermaid call graph with safe node IDs + readable labels.

    - Uses stable node IDs (n0, n1, ...) and bracket labels [module.func]
    - Filters short/irrelevant edges via min_ms and exclude regex
    """
    import re

    os.makedirs(os.path.dirname(outfile), exist_ok=True)

    def short(name: str) -> str:
        # module.func → keep the last two segments when available
        parts = name.split(".")
        if len(parts) >= 2:
            lbl = parts[-2] + "." + parts[-1]
        else:
            lbl = name
        # trim
        if len(lbl) > label_max:
            lbl = lbl[:label_max - 1] + "…"
        # escape bracket terminators/quotes minimally
        return lbl.replace("\n", " ").replace("]", ")").replace("\"", "'")

    # Build node id map for stable identifiers
    node_id: Dict[str, str] = {}
    def nid(name: str) -> str:
        if name not in node_id:
            node_id[name] = f"n{len(node_id)}"
        return node_id[name]

    with open(outfile, "w") as f:
        f.write("graph TD\n")
        for (caller, callee), (count, total) in edges.items():
            if exclude_patterns and (re.search(exclude_patterns, caller) or re.search(exclude_patterns, callee)):
                continue
            if (total * 1000.0) < min_ms:
                continue
            edge_label = f"{count}x / {total*1000:.0f}ms"
            f.write(
                f"  {nid(caller)}[{short(caller)}] -->|{edge_label}| {nid(callee)}[{short(callee)}]\n"
            )


def render_top_edges(edges: Dict[Tuple[str, str], Tuple[int, float]], *, top_n: int = 10, min_ms: float = 1.0, exclude_patterns: Optional[str] = None) -> None:
    try:
        from rich.table import Table
        from rich.console import Console
    except Exception:
        return
    rows = []
    for (caller, callee), (count, total) in edges.items():
        if (total * 1000.0) < min_ms:
            continue
        if exclude_patterns:
            import re
            if re.search(exclude_patterns, caller) or re.search(exclude_patterns, callee):
                continue
        rows.append((total, count, caller.split(".")[-1], callee.split(".")[-1]))
    rows.sort(key=lambda x: x[0], reverse=True)
    rows = rows[:top_n]
    table = Table(title="Top Edges")
    table.add_column("Total ms", justify="right")
    table.add_column("Count", justify="right")
    table.add_column("Caller")
    table.add_column("Callee")
    for total, count, caller, callee in rows:
        table.add_row(f"{total*1000:.1f}", str(count), caller, callee)
    Console().print(table)
