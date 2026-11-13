"""
Document Loader and Chunker
===========================
"""

from pathlib import Path
from typing import List, Optional
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    WebBaseLoader,
    DirectoryLoader
)
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    TokenTextSplitter
)


def load_documents(data_dir: str = "data/") -> List:
    """
    Load documents from the data directory and all subfolders.
    Supports .txt and .pdf files recursively.
    """
    path = Path(data_dir)
    documents = []

    # Load all text files recursively through subfolders
    for file_path in path.rglob("*.txt"):
        try:
            loader = TextLoader(str(file_path), encoding="utf-8")
            docs = loader.load()
            documents.extend(docs)
            print(f"  Loaded: {file_path} ({len(docs)} pages)")
        except Exception as e:
            print(f"  Skipped {file_path}: {e}")

    # Load all PDF files recursively through subfolders
    for pdf_path in path.rglob("*.pdf"):
        try:
            loader = PyPDFLoader(str(pdf_path))
            docs = loader.load()
            documents.extend(docs)
            print(f"  Loaded: {pdf_path} ({len(docs)} pages)")
        except Exception as e:
            print(f"  Skipped {pdf_path}: {e}")

    return documents


def chunk_documents(
    documents: List,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    chunking_strategy: str = "recursive"
) -> List:
    """
    Split documents into chunks.

    Chunking strategy chosen: recursive
    - chunk_size=500: small enough to stay focused on one topic per chunk
    - chunk_overlap=50: ensures context is not lost between chunks
    - separators prioritize paragraph breaks then line breaks then words
    """
    if chunking_strategy == "recursive":
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
    elif chunking_strategy == "markdown":
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
    elif chunking_strategy == "token":
        splitter = TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    else:
        raise ValueError(f"Unknown chunking strategy: {chunking_strategy}")

    chunks = splitter.split_documents(documents)

    # Add metadata to every chunk for source citation
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["chunk_size"] = len(chunk.page_content)
        chunk.metadata["source"] = chunk.metadata.get("source", "unknown")
        chunk.metadata["page"] = chunk.metadata.get("page", "N/A")

    return chunks


if __name__ == "__main__":
    docs = load_documents()
    chunks = chunk_documents(docs)
    print(f"\nTotal: {len(docs)} pages, {len(chunks)} chunks")