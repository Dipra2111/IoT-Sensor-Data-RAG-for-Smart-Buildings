from __future__ import annotations
from typing import List, Dict, Any
from langchain_community.vectorstores import Chroma
def retrieve(vs: Chroma, query: str, k: int = 4) -> List[Dict[str, Any]]:
    docs = vs.similarity_search(query, k=k)
    return [{"page_content": d.page_content, **d.metadata} for d in docs]
