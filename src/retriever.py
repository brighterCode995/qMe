"""
Retriever Module
================
Students must implement and customize retrieval strategies.
"""

from typing import List, Optional, Dict
from langchain_community.vectorstores import Chroma, FAISS
from langchain_core.documents import Document
from langchain_classic.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_community.retrievers import BM25Retriever


def create_vectorstore(
    documents: List[Document],
    embedder,
    db_type: str = "chroma",
    persist_dir: Optional[str] = None
):
    """
    Create a vector store from documents.

    IMPLEMENTED:
    - ChromaDB for persistent storage (survives restarts)
    - FAISS for fast in-memory search
    - Metadata fields (source, page, chunk_id) preserved from loader
    """
    if db_type == "chroma":
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embedder,
            persist_directory=persist_dir
        )
    elif db_type == "faiss":
        vectorstore = FAISS.from_documents(
            documents=documents,
            embedding=embedder
        )
        if persist_dir:
            vectorstore.save_local(persist_dir)
    else:
        raise ValueError(f"Unknown DB type: {db_type}")

    return vectorstore


def get_retriever(
    vectorstore,
    search_type: str = "mmr",
    k: int = 6,
    score_threshold: Optional[float] = 0.3,
    filter_criteria: Optional[Dict] = None
):
    """
    Create a retriever with customizable search parameters.

    IMPLEMENTED:
    - search_type = 'mmr' for diverse, non-redundant results
    - k = 6 for better coverage of the Ubuntu manual
    - fetch_k = k * 3 so MMR has more candidates to pick from
    - metadata filters supported via filter_criteria
    """
    search_kwargs = {"k": k}

    if search_type == "mmr":
        search_kwargs["fetch_k"] = k * 3

    if score_threshold is not None and search_type == "similarity_score_threshold":
        search_kwargs["score_threshold"] = score_threshold

    if filter_criteria is not None:
        search_kwargs["filter"] = filter_criteria

    retriever = vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs
    )

    return retriever


def retrieve_with_hybrid_search(
    vectorstore,
    documents: List[Document],
    query: str,
    k: int = 6,
    alpha: float = 0.5
) -> List[Document]:
    """
    IMPLEMENTED: Hybrid search combining BM25 and vector similarity.

    - BM25: keyword-based (good for exact Linux commands)
    - Vector: semantic-based (good for meaning-based queries)
    - alpha=0.5 means equal weighting between both
    """
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = k

    vector_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[1 - alpha, alpha]
    )

    results = ensemble_retriever.invoke(query)
    return results


def retrieve_with_reranking(
    retriever,
    llm,
    query: str,
    k: int = 6
) -> List[Document]:
    """
    IMPLEMENTED: Reranking using LangChain contextual compression.

    - LLMChainExtractor uses the LLM to extract only relevant parts
    - Wraps base retriever with compression for higher quality results
    """
    compressor = LLMChainExtractor.from_llm(llm)

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=retriever
    )

    results = compression_retriever.invoke(query)
    return results


if __name__ == "__main__":
    from src.embedder import get_embedder
    from src.loader import load_documents, chunk_documents

    docs = load_documents()
    chunks = chunk_documents(docs)
    embedder = get_embedder()

    print("=" * 60)
    print("TEST 1: MMR Retriever")
    print("=" * 60)
    vectorstore = create_vectorstore(chunks, embedder)
    retriever = get_retriever(vectorstore, k=6)
    results = retriever.invoke("how to configure SSH on Ubuntu")
    print(f"Retrieved {len(results)} documents")
    for i, doc in enumerate(results):
        print(f"\n[{i+1}] Page {doc.metadata.get('page', 'N/A')}")
        print(f"     {doc.page_content[:100]}...")

    print()
    print("=" * 60)
    print("TEST 2: Hybrid Search")
    print("=" * 60)
    results = retrieve_with_hybrid_search(
        vectorstore, chunks, "how to configure SSH on Ubuntu", k=6
    )
    print(f"Retrieved {len(results)} documents")
    for i, doc in enumerate(results):
        print(f"\n[{i+1}] Page {doc.metadata.get('page', 'N/A')}")
        print(f"     {doc.page_content[:100]}...")