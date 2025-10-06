import os
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from .vectorstore import ensure_collection, recreate_collection
from .config import AppConfig


def load_pdf_as_documents(file_path: str) -> List[Document]:
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    for d in docs:
        d.metadata = dict(d.metadata or {})
        d.metadata["source_path"] = os.path.abspath(file_path)
        d.metadata["source"] = os.path.basename(file_path)
    return docs


def chunk_documents(documents: List[Document], chunk_size: int = 800, chunk_overlap: int = 120) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)


def ingest_pdfs_into_qdrant(config: AppConfig, vectorstore, embeddings, pdf_paths: List[str], replace: bool = True) -> int:
    if not pdf_paths:
        return 0
    # Ensure collection exists using embedding dimension
    sample_vec = embeddings.embed_query("dimension_probing_string")
    # Some vectorstores (Chroma) don't have a client/property; guard Qdrant-only ops
    client = getattr(vectorstore, "client", None)
    if client is not None:
        if replace:
            recreate_collection(client, config.qdrant_collection, len(sample_vec))
        else:
            ensure_collection(client, config.qdrant_collection, len(sample_vec))

    total_chunks = 0
    batch_size = 64  # keep batches modest for Qdrant Cloud
    for path in pdf_paths:
        docs = load_pdf_as_documents(path)
        chunks = chunk_documents(docs)
        # upload in smaller batches to avoid timeouts
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            vectorstore.add_documents(batch)
        total_chunks += len(chunks)
    return total_chunks

