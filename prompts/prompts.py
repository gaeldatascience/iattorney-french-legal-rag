from langchain.prompts import ChatPromptTemplate

prompt_decomposition = ChatPromptTemplate.from_template(
    """You are a helpful assistant that generates multiple sub-questions related to an input question. 
Generate multiple search queries related to: {question} 
Output (3 queries):"""
)

prompt_hyde = ChatPromptTemplate.from_template(
    """For the legal sub-question below, generate a plausible legal paragraph or pseudo-article of law 
that could answer it.

Sub-question: {sub_question}
Output:"""
)

final_prompt = ChatPromptTemplate.from_template(
    """You are a legal expert assistant. 
Here is the conversation history so far:
{history}

Now, based on the following legal context and conversation history, answer the question precisely and clearly.

Refer explicitly to the relevant articles by quoting or paraphrasing them, and explain how each supports your conclusion.

Lors de la référence à un article de loi, écris toujours le nom au format : « l’article [numéro] du [Nom du Code] ».
Par exemple, si le contexte contient « Code Pénal - L121-1 », tu dois écrire : « l’article L121-1 du Code Pénal ».


Context:
{context}

Question:
{question}

Answer:"""
)

prompt_articles_to_json = ChatPromptTemplate.from_template("""
You are a legal assistant.

Your task is to extract all the legal articles and their corresponding legal codes
mentioned in the user’s sentence.

Your response must be a **strictly formatted JSON** like this:

[
  {{
    "article": "<article number exactly as mentioned>",
    "code": "<associated legal code name>"
  }},
  ...
]

If multiple articles are mentioned, return one JSON entry per article.

Do not include any explanation or comments — only return the JSON.

User's sentence: {question}
"""
)

article_answer_prompt = ChatPromptTemplate.from_template("""
You are a legal expert assistant.

Here is the conversation history so far:
{history}

You have access to the content of the legal articles explicitly mentioned by the user.
Your job is to help the user in one of the following ways:

- If the user asked you to cite or summarize the articles, do so clearly and precisely.
- If the user previously asked a question and now shares legal articles, use those articles to complete or refine your previous answer.
- If the user's question is more general, and these articles help answer it, refer to them precisely.

Always reference articles using the format: "l’article [numéro] du [Nom du Code]".

Context:
{context}

User question:
{question}

Your answer:
""")

routing_prompt_template = ChatPromptTemplate.from_template("""
You are a legal assistant. Your task is to analyze the following user question and decide:
- If it is unrelated to French law, respond exactly with: "NON-LEGAL"
- Otherwise, respond with: "LEGAL"

User question: {question}
""")