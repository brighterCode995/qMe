"""
RAG Pipeline Orchestration
===========================
Main pipeline that ties together all RAG components.
"""
import os
import json
import hashlib
import logging
from pathlib import Path
from typing import Optional, List, Dict
from langchain_community.vectorstores import Chroma
from src.loader import load_documents, chunk_documents
from src.embedder import get_embedder
from src.retriever import create_vectorstore, get_retriever
from src.generator import get_llm, create_rag_prompt, create_qa_chain, generate_response


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OUT_OF_SCOPE_PHRASES = [
    "i don't know", "i am not sure", "i cannot find",
    "no information", "not found in the manual", "not in the manual",
    "could not find this in the manual", "consult your lab instructor",
    "outside the scope", "not covered", "misunderstanding",
    "general knowledge",
]

OUT_OF_SCOPE_RESPONSE = (
    "I could not find this in the manual. Please consult your lab instructor."
)

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "up", "about", "into", "through",
    "how", "what", "when", "where", "who", "which", "that", "this",
    "these", "those", "i", "you", "he", "she", "it", "we", "they",
    "and", "but", "or", "not", "just", "also", "then", "than",
}


def _is_relevant(query: str, retrieved_docs: list, min_keyword_overlap: int = 1) -> bool:
    query_keywords = {
        w for w in query.lower().split()
        if w.isalpha() and w not in STOPWORDS and len(w) > 2
    }
    if not query_keywords:
        return True
    for doc in retrieved_docs:
        doc_words = set(doc.page_content.lower().split())
        if len(query_keywords & doc_words) >= min_keyword_overlap:
            return True
    return False


def compute_file_hash(filepath: str) -> str:
    with open(filepath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def load_hash_cache(cache_path: str) -> Dict:
    if Path(cache_path).exists():
        with open(cache_path, "r") as f:
            return json.load(f)
    return {}


def save_hash_cache(cache_path: str, cache: Dict):
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2)


class RAGPipeline:

    def __init__(
        self,
        data_dir: str = "data/",
        embedder_provider: str = "huggingface",
        embedder_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_provider: str = "ollama",
        llm_model: str = "phi3",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        retrieval_k: int = 4,
        vectorstore_type: str = "chroma",
        persist_dir: Optional[str] = "vectorstore"
    ):
        self.data_dir = data_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.retrieval_k = retrieval_k
        self.persist_dir = persist_dir
        self.hash_cache_path = os.path.join(persist_dir, "file_hashes.json") if persist_dir else None

        logger.info("Initializing embedder...")
        self.embedder = get_embedder(
            provider=embedder_provider,
            model_name=embedder_model
        )

        logger.info("Initializing LLM...")
        self.llm = get_llm(
            provider=llm_provider,
            model_name=llm_model
        )

        self.vectorstore = None
        self.retriever = None
        self.qa_chain = None

    def _get_changed_files(self) -> List[str]:
        cache = load_hash_cache(self.hash_cache_path)
        changed = []
        data_path = Path(self.data_dir)
        for file_path in data_path.glob("*.txt"):
            filepath_str = str(file_path)
            current_hash = compute_file_hash(filepath_str)
            if cache.get(filepath_str) != current_hash:
                changed.append(filepath_str)
        return changed

    def _update_hash_cache(self):
        cache = {}
        data_path = Path(self.data_dir)
        for file_path in data_path.glob("*.txt"):
            filepath_str = str(file_path)
            cache[filepath_str] = compute_file_hash(filepath_str)
        save_hash_cache(self.hash_cache_path, cache)

    def _preprocess_query(self, question: str) -> str:
        question = question.strip()
        question = " ".join(question.split())
        abbreviations = {
            "vs code": "Visual Studio Code",
            "venv": "virtual environment",
            "cmd": "command prompt",
            "os": "operating system",
            "lan": "local area network",
            "ip": "IP address",
            "pc": "computer",
        }
        question_lower = question.lower()
        for short, full in abbreviations.items():
            if short in question_lower:
                question = question_lower.replace(short, full)
        return question

    def _postprocess_response(self, answer: str) -> str:
        answer = answer.strip()
        answer = "\n".join(line for line in answer.splitlines() if line.strip())
        return answer

    def load_and_index(self, force_rebuild: bool = False):
        persist_path = Path(self.persist_dir) if self.persist_dir else None

        if persist_path and persist_path.exists() and not force_rebuild:
            logger.info("Loading existing vectorstore...")
            self.vectorstore = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embedder
            )
            changed_files = self._get_changed_files()
            if changed_files:
                logger.info(f"Detected {len(changed_files)} new/changed file(s) — re-indexing...")
                for filepath in changed_files:
                    from langchain_community.document_loaders import TextLoader
                    loader = TextLoader(filepath, encoding="utf-8")
                    new_docs = loader.load()
                    new_chunks = chunk_documents(
                        new_docs,
                        chunk_size=self.chunk_size,
                        chunk_overlap=self.chunk_overlap
                    )
                    self.vectorstore.add_documents(new_chunks)
                    logger.info(f"Re-indexed: {filepath}")
                self._update_hash_cache()
            else:
                logger.info("No file changes detected — using existing index.")
        else:
            logger.info("Loading and chunking all documents...")
            documents = load_documents(self.data_dir)
            chunks = chunk_documents(
                documents,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
            logger.info("Creating vector index...")
            self.vectorstore = create_vectorstore(
                chunks,
                self.embedder,
                db_type="chroma",
                persist_dir=self.persist_dir
            )
            self._update_hash_cache()

        logger.info("Setting up retriever...")
        self.retriever = get_retriever(
            self.vectorstore,
            k=self.retrieval_k
        )
        logger.info("Creating QA chain...")
        prompt = create_rag_prompt()
        self.qa_chain = create_qa_chain(self.llm, self.retriever, prompt)

    def query(self, question: str, return_sources: bool = True) -> Dict:
        if self.qa_chain is None:
            raise RuntimeError("Pipeline not initialized. Call load_and_index() first.")

        # STEP 1: Preprocess
        question = self._preprocess_query(question)
        logger.info(f"Querying: {question}")

        # STEP 2: Retrieve
        retrieved_docs = self.retriever.invoke(question)
        if not retrieved_docs:
            return {"answer": OUT_OF_SCOPE_RESPONSE, "sources": [], "confidence": 0.0}

        # RELEVANCE GATE
        if not _is_relevant(question, retrieved_docs):
            logger.info("Query failed relevance gate — returning out-of-scope response.")
            return {"answer": OUT_OF_SCOPE_RESPONSE, "sources": [], "confidence": 0.0}

        # STEP 3: Generate
        response = generate_response(self.qa_chain, self.retriever, question, return_sources)

        # STEP 4: Postprocess — FIXED: removed logger.info raw answer (was causing double print)
        answer = self._postprocess_response(response.get("answer", ""))

        # STEP 5: Catch LLM slippage
        answer_lower = answer.lower()
        is_out_of_scope = any(phrase in answer_lower for phrase in OUT_OF_SCOPE_PHRASES)

        if is_out_of_scope and len(answer) < 300:
            confidence = 0.3
            answer = OUT_OF_SCOPE_RESPONSE
            response["sources"] = []
        else:
            confidence = 1.0

        response["answer"] = answer
        response["confidence"] = confidence
        return response

    def evaluate(self, test_queries: List[Dict]) -> Dict:
        results = []
        total_precision = 0.0
        hallucination_count = 0

        for item in test_queries:
            query = item["question"]
            expected = item.get("expected_answer", "")

            response = self.query(query, return_sources=True)
            answer = response["answer"]

            retrieved_docs = self.retriever.invoke(self._preprocess_query(query))
            keywo