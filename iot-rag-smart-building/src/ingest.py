from __future__ import annotations
import os, glob
from dataclasses import dataclass
from typing import List
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.docstore.document import Document
@dataclass
class CorpusPaths:
    manuals_dir: str
    specs_dir: str
    persist_dir: str = "chroma_db"
def load_documents(paths: CorpusPaths) -> List[Document]:
    docs: List[Document] = []
    for base in [paths.manuals_dir, paths.specs_dir]:
        for fp in glob.glob(os.path.join(base, "**", "*"), recursive=True):
            if os.path.isdir(fp): continue
            if fp.lower().endswith(".pdf"):
                try:
                    loader = PyPDFLoader(fp); docs.extend(loader.load())
                except Exception as e: print("PDF load error:", fp, e)
            elif fp.lower().endswith((".txt",".md")):
                docs.extend(TextLoader(fp, encoding="utf-8").load())
    return docs
def split_documents(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=120)
    return splitter.split_documents(docs)
def build_embeddings(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    return HuggingFaceEmbeddings(model_name=model_name)
def build_or_load_vectorstore(paths: CorpusPaths, embeddings=None) -> Chroma:
    os.makedirs(paths.persist_dir, exist_ok=True)
    embeddings = embeddings or build_embeddings()
    try:
        vs = Chroma(persist_directory=paths.persist_dir, embedding_function=embeddings)
        if vs._collection.count() > 0: return vs
    except Exception: pass
    docs = split_documents(load_documents(paths))
    for d in docs:
        d.metadata["source"] = d.metadata.get("source", d.metadata.get("file_path", "doc"))
    vs = Chroma.from_documents(docs, embeddings, persist_directory=paths.persist_dir); vs.persist(); return vs
