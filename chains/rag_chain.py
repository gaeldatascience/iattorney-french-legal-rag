from langchain.schema.runnable import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from models.embeddings import retriever
from .utils import clean_subquestions, enrich_with_neighbors
from prompts import prompt_decomposition, prompt_hyde, final_prompt

# LLM and output parser
llm = ChatOpenAI(temperature=0)
parser = StrOutputParser()

# =========================
# RAG Chain Steps
# =========================

# Decompose the main question into sub-questions
decompose_chain = (
    prompt_decomposition
    | llm
    | parser
    | RunnableLambda(clean_subquestions)
)

# Generate fictitious articles for each sub-question (HyDE)
hyde_single = prompt_hyde | llm | parser
hyde_map_chain = (
    RunnableLambda(lambda subqs: [{"sub_question": q} for q in subqs])
    | hyde_single.map()
)

# Retrieve documents via FAISS
retrieve_chain = RunnableLambda(
    lambda pseudo_articles: [
        doc for article in pseudo_articles
        for doc in retriever.invoke(article)
    ]
)

# Build the final context
context_builder = RunnableLambda(
    lambda docs: {"context": "\n\n".join([doc.page_content for doc in docs])}
)

# Final answer
answer_chain = final_prompt | llm | parser

# =========================
# Full RAG Chain
# =========================

rag_chain = (
    RunnableLambda(lambda x: {"question": x["question"], "history": x.get("history", "")})
    | {
        "question": lambda x: x["question"],
        "sub_questions": lambda x: decompose_chain.invoke({"question": x["question"]}),
        "history": lambda x: x["history"]
    }
    | {
        "question": lambda x: x["question"],
        "pseudo_articles": lambda x: hyde_map_chain.invoke(x["sub_questions"]),
        "history": lambda x: x["history"]
    }
    | {
        "question": lambda x: x["question"],
        "retrieved_docs": lambda x: retrieve_chain.invoke(x["pseudo_articles"]),
        "history": lambda x: x["history"]
    }
    | {
        "question": lambda x: x["question"],
        "retrieved_docs": lambda x: x["retrieved_docs"],
        "expanded_docs": lambda x: enrich_with_neighbors(x["retrieved_docs"]),
        "history": lambda x: x["history"]
    }
    | {
        "question": lambda x: x["question"],
        "context": lambda x: context_builder.invoke(x["expanded_docs"])["context"],
        "history": lambda x: x["history"]
    }
    | answer_chain
)
