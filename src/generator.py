"""
LLM Generator Module
====================
Lab/Manual Troubleshooting Assistant
Students: customize the prompt and LLM configuration.
"""

from typing import List, Optional, Dict
from pathlib import Path
from langchain_ollama import OllamaLLM
from langchain_community.llms import HuggingFaceHub
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage


# ─────────────────────────────────────────────
# TASK 2: Customize the LLM Configuration
# ─────────────────────────────────────────────
def get_llm(
    provider: str = "ollama",
    model_name: str = "phi3",
    temperature: float = 0.1,
    **kwargs
):
    if provider == "ollama":
        return OllamaLLM(
            model=model_name,
            temperature=temperature,
            num_predict=512         # prevents answer from being cut off mid-sentence
        )
    elif provider == "huggingface":
        return HuggingFaceHub(
            repo_id=model_name,
            model_kwargs={"temperature": temperature}
        )
    elif provider == "openai":
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")


# ─────────────────────────────────────────────
# TASK 1: Customize the RAG Prompt
# ─────────────────────────────────────────────
def create_rag_prompt(
    system_message: Optional[str] = None,
    template: Optional[str] = None
) -> ChatPromptTemplate:

    if system_message is None:
        system_message = """You are a strict Ubuntu Linux lab assistant.
You ONLY answer using the MANUAL CONTEXT provided.

STRICT RULES:
- ONLY use information from the MANUAL CONTEXT. Nothing else.
- NEVER add URLs, links, or external references.
- NEVER use your own knowledge or general IT knowledge.
- NEVER pad your answer with unrelated steps.
- Give a SHORT, direct, step-by-step answer using ONLY what the manual says.
- Reference the chapter or page number if visible in the context.
- If the answer is not in the context say ONLY:
  "I could not find this in the manual. Please consult your lab instructor."
- If the question is not about Linux or Ubuntu say ONLY:
  "This is outside the scope of this lab manual."
- Do NOT greet the user or add closing remarks.
- Stop writing as soon as you have answered the question. Do not continue."""

    if template is None:
        # FIXED: removed "Question:" label to stop LLM reading ahead into the template
        template = """MANUAL CONTEXT:
{context}

{question}

Answer (be concise, use ONLY the manual context, stop after answering):"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", template)
    ])

    return prompt


def create_qa_chain(llm, retriever, prompt: Optional[ChatPromptTemplate] = None):
    if prompt is None:
        prompt = create_rag_prompt()

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    qa_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return qa_chain


def generate_response(
    qa_chain,
    retriever,
    query: str,
    return_sources: bool = True
) -> Dict:
    answer = qa_chain.invoke(query)
    response = {"answer": answer}

    if return_sources:
        source_docs = retriever.invoke(query)

        # Deduplicate by page number
        seen = set()
        unique_sources = []
        for doc in source_docs:
            page = doc.metadata.get("page", "N/A")
            source = doc.metadata.get("source", "")
            key = (source, page)
            if key not in seen:
                seen.add(key)
                unique_sources.append({
                    "filename": Path(source).name if source else "Unknown",
                    "page": page
                })

        response["sources"] = unique_sources

    return response


if __name__ == "__main__":
    prompt = create_rag_prompt()
    print("=" * 60)
    print("Lab Troubleshooting Assistant — Prompt Messages")
    print("=" * 60)
    for message in prompt.messages:
        print(f"[{message.__class__.__name__}]")
        print(message.content)
        print()
    print("Default LLM Configuration:")
    print("  Provider   : ollama")
    print("  Model      : phi3")
    print("  Temperature: 0.1")