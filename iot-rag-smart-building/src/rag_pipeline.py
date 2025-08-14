from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any
from langchain_community.vectorstores import Chroma
from .ingest import build_or_load_vectorstore, CorpusPaths
from .retriever import retrieve
from .utils import format_retrieval
@dataclass
class RAGEngine:
    vs: Chroma
    @classmethod
    def initialize(cls, manuals_dir: str, specs_dir: str, persist_dir: str="chroma_db")->"RAGEngine":
        vs = build_or_load_vectorstore(CorpusPaths(manuals_dir, specs_dir, persist_dir)); return cls(vs)
    def ask(self, query: str, k: int=4) -> Dict[str, Any]:
        chunks = retrieve(self.vs, query, k=k); context = format_retrieval(chunks)
        answer=f"""Based on retrieved manuals & specs:

{context}

Provisional guidance:
- Match steps above to your equipment and symptoms.
- Follow safety SOPs (lockout/tagout) before intervention.
- If vibration > 7.1 mm/s or temperature > 28°C, prioritize bearing/alignment checks and cooling airflow.

(To enable full LLM generation, connect an API/local model here.)"""
        return {"answer": answer.strip(), "chunks": chunks}
