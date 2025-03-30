import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SECRET_KEY = os.getenv("SECRET_KEY", "dev_secret")

FAISS_PATH = "data/faiss_juridique_code_with_metadatas"
EMBEDDING_MODEL = "camembert_model"

LANGSMITH_TRACING_V2=os.getenv("LANGSMITH_TRACING_V2")
LANGSMITH_ENDPOINT=os.getenv("LANGSMITH_ENDPOINT")
LANGSMITH_API_KEY=os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT=os.getenv("LANGSMITH_PROJECT")