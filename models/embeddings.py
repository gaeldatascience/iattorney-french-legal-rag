from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from config import FAISS_PATH, EMBEDDING_MODEL

embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

vector_store = FAISS.load_local(
        FAISS_PATH,
        embedding_function,
        allow_dangerous_deserialization=True
    )

retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3, "include_score": True}
    )