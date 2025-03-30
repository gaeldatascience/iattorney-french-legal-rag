from langchain.schema.runnable import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from rapidfuzz import process
from data.list_codes import LEGAL_CODES_LIST
from prompts import prompt_articles_to_json, article_answer_prompt
from models.embeddings import vector_store
import json
import re
"""
FAISS retriever does not support filters, so it is unused.
"""


llm = ChatOpenAI(temperature=0)
parser = StrOutputParser()

# ======= Article normalization =======
def normalize_article(article: str) -> str:
    article = article.strip().replace(" ", "").upper()
    match = re.match(r'^([RL]?)[\.-]?(\d{1,4})(?:[\.-]?(\d{1,4}))?(?:[\.-]?(\d{1,4}))?$', article)
    if match:
        parts = [match.group(1)] + [g for g in match.groups()[1:] if g]
        return parts[0] + "-".join(parts[1:])
    return article

# ======= Match code name using RapidFuzz =======
def match_code_name(user_code: str) -> str:
    match, score, _ = process.extractOne(user_code, LEGAL_CODES_LIST)
    return match

# ======= Convert LLM output to normalized and matched structure =======
def process_extracted_articles(response: str):
    entries = json.loads(response)
    for entry in entries:
        entry["article_normalized"] = normalize_article(entry["article"])
        entry["code_matched"] = match_code_name(entry["code"])
    return entries

# ======= Retrieve content from knowledge base =======
def retrieve_articles_from_store(entries):
    results = []
    for entry in entries:
        code = entry['code_matched']
        article = entry['article_normalized']

        # On construit un retriever avec filtres
        filtered_retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 1,
                "include_score": True,
                "filters": {
                    "source": code,
                    "id": article
                }
            }
        )
        docs = filtered_retriever.invoke("")
        print([doc.metadata for doc in docs])
        print(f"source : {code}, id : {article}")
        top_doc = docs[0] if docs else None
        entry["retrieved_content"] = [top_doc.page_content] if top_doc else []
        results.append(entry)
    return results


# ======= Article Extraction Chain =======
article_extraction_chain = (
    prompt_articles_to_json
    | llm
    | parser
    | RunnableLambda(process_extracted_articles)
    | RunnableLambda(retrieve_articles_from_store)
)

article_context_builder = RunnableLambda(lambda entries: {
    "context": "\n\n".join(
        f"Article {entry['article_normalized']} du {entry['code_matched']} :\n" + "\n\n".join(entry["retrieved_content"])
        for entry in entries if entry["retrieved_content"]
    )
})

# ======= Legal Answer Generation Chain (Articles) =======
article_rag_chain = (
    RunnableLambda(lambda x: {"question": x["question"], "history": x.get("history", "")})
    | {
        "question": lambda x: x["question"],
        "extracted_articles": lambda x: article_extraction_chain.invoke({"question": x["question"]}),
        "history": lambda x: x["history"]
    }
    | {
        "question": lambda x: x["question"],
        "context": lambda x: article_context_builder.invoke(x["extracted_articles"])['context'],
        "history": lambda x: x["history"]
    }
    | article_answer_prompt
    | llm
    | parser
)
