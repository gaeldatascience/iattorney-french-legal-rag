"""
Utility functions used in RAG chains.
"""

from typing import List
from langchain.schema import Document
from models.embeddings import retriever

def clean_subquestions(raw_text: str) -> List[str]:
    """
    Cleans the raw output of sub-questions to obtain a clean list.
    
    Args:
        raw_text (str): The raw text containing the generated sub-questions.
    
    Returns:
        List[str]: A list of cleaned sub-questions.
    """
    lines = raw_text.strip().split("\n")
    return [
        line.lstrip("1234567890.- ").strip()
        for line in lines
        if line.strip()
    ]


def enrich_with_neighbors(docs: List[Document], k: int = 5) -> List[Document]:
    """
    Adds relevant neighboring documents from the same file/source
    as the first returned document to broaden the context.
    
    Args:
        docs (List[Document]): Initial list of documents.
        k (int): Number of neighboring documents to retrieve (default is 5).
    
    Returns:
        List[Document]: Enriched list of documents.
    """
    if not docs:
        return []

    top_doc = docs[0]
    source = top_doc.metadata.get("source")
    vectorstore = retriever.vectorstore

    # Search for similar neighbors in the same file
    similar_docs = vectorstore.similarity_search(query=top_doc.page_content, k=k+1)
    filtered = [d for d in similar_docs if d.metadata.get("source") == source]

    # De-duplicate by content
    seen, final_docs = set(), []
    for doc in docs + filtered:
        content = doc.page_content.strip()
        if content not in seen:
            seen.add(content)
            final_docs.append(doc)

    return final_docs