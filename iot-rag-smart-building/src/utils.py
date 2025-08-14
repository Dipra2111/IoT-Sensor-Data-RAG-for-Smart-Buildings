from __future__ import annotations
import os
from typing import List, Dict, Any
def get_env_bool(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None: return default
    return val.lower() in {"1","true","yes","y","on"}
def format_retrieval(chunks: List[Dict[str, Any]]) -> str:
    lines = []
    for ch in chunks:
        src = ch.get("source", "unknown")
        content = ch.get("page_content", "")
        lines.append(f"[{src}]\n{content}\n")
    return "\n---\n".join(lines)
