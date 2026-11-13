"""
Embedding Model
===============
Students must choose and configure the embedding model.
"""
import os
os.environ["USER_AGENT"] = "rag-starter/1.0"
from typing import List, Optional
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings


def get_embedder(
    provider: str = "huggingface",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",  # CHANGED: back to fastest model
    **kwargs
):
    """
    Get an embedding model.

    Students: Modify this:
    - Try different embedding models (e.g., BAAI/bge-small-en)
    - Test Ollama embeddings for local inference

    Recommended models by speed/quality tradeoff:
    - sentence-transformers/all-MiniLM-L6-v2  → fastest, good quality (RECOMMENDED)
    - BAAI/bge-small-en-v1.5                  → slightly better quality, slower
    - BAAI/bge-base-en-v1.5                   → best quality, much slower

    Args:
        provider: "huggingface", "openai", or "ollama"
        model_name: Name of the embedding model
    """
    if provider == "huggingface":
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"}
        )
    elif provider == "openai":
        return OpenAIEmbeddings(model=model_name)
    elif provider == "ollama":
        return OllamaEmbeddings(model=model_name)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def embed_documents(embedder, documents: List) -> List[List[float]]:
    """
    Embed a list of documents.
    Processes in batches of 100 to avoid memory issues on large PDFs.
    """
    texts = [doc.page_content for doc in documents]

    # IMPLEMENTED: batch processing for large document sets
    batch_size = 100
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_embeddings = embedder.embed_documents(batch)
        all_embeddings.extend(batch_embeddings)

    return all_embeddings


def embed_query(embedder, query: str) -> List[float]:
    """
    Embed a query string.
    """
    return embedder.embed_query(query)


if __name__ == "__main__":
    embedder = get_embedder()
    test_text = "What is Retrieval-Augmented Generation?"
    embedding = embed_query(embedder, test_text)
    print(f"Embedding dimension: {len(embedding)}")
    print(f"Model: sentence-transformers/all-MiniLM-L6-v2")
    print(f"Device: CPU")