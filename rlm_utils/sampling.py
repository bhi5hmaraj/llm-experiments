from __future__ import annotations

import glob
import os
from typing import List


TEXT_EXTS = {".txt", ".md", ".markdown", ".json", ".csv", ".tsv"}


def small_sample_from_file(file_path: str, bytes_per_file: int) -> str:
    fp = os.path.abspath(file_path)
    if not os.path.isfile(fp):
        raise FileNotFoundError(fp)
    with open(fp, "rb") as f:
        buf = f.read(bytes_per_file)
    try:
        text = buf.decode("utf-8", errors="replace")
    except Exception:
        text = buf.decode("latin-1", errors="replace")
    return f"### FILE: {os.path.basename(fp)}\n{text}\n"


def small_sample_from_dir(data_dir: str, k: int, bytes_per_file: int, include_all: bool = False) -> str:
    def _is_texty(path: str) -> bool:
        if include_all:
            return True
        return os.path.splitext(path)[1].lower() in TEXT_EXTS

    paths = [
        p
        for p in glob.glob(os.path.join(data_dir, "**", "*"), recursive=True)
        if os.path.isfile(p) and _is_texty(p)
    ]
    if not paths:
        raise RuntimeError(f"No files found in {data_dir}")
    import random
    sample = random.sample(paths, min(k, len(paths)))

    chunks: List[str] = []
    for p in sample:
        try:
            with open(p, "rb") as f:
                data = f.read(bytes_per_file)
            try:
                text = data.decode("utf-8", errors="replace")
            except Exception:
                text = data.decode("latin-1", errors="replace")
            chunks.append(f"### FILE: {os.path.relpath(p, data_dir)}\n{text}\n")
        except Exception as e:
            chunks.append(f"### FILE: {os.path.relpath(p, data_dir)}\n[read error: {e}]\n")
    return "\n".join(chunks)

