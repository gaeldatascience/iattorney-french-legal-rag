from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from prompts import routing_prompt_template
from chains import rag_chain

# LLM and output parser
llm = ChatOpenAI(temperature=0)
parser = StrOutputParser()

router_chain = routing_prompt_template | llm | parser

# ======= Main Router Logic =======
def route_question(question: str, history: str = "") -> str:
    route = router_chain.invoke({"question": question}).strip()

    if route == "NON-LEGAL":
        return "Je suis spécialisé dans le droit français. Si vous avez une question relative à ce domaine, je serai ravi d'y répondre."

    else:
        return rag_chain.invoke({"question": question, "history": history})