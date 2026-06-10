import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pydantic import SecretStr
from src.vectorstore import FaissVectorStore

load_dotenv()


class RAGSearch:

    def __init__(
        self,
        persist_dir: str = "faiss_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "llama-3.1-8b-instant"
    ):

        self.vectorstore = FaissVectorStore(
            persist_dir,
            embedding_model
        )

        faiss_path = os.path.join(
            persist_dir,
            "faiss.index"
        )

        meta_path = os.path.join(
            persist_dir,
            "metadata.pkl"
        )

        if (
            os.path.exists(faiss_path)
            and
            os.path.exists(meta_path)
        ):
            self.vectorstore.load()
        else:
            from src.data_loader import (
                load_all_documents
            )

            docs = load_all_documents("data")

            self.vectorstore.build_from_documents(
                docs
            )

        groq_api_key = os.getenv(
            "GROQ_API_KEY"
        )

        if not groq_api_key:
            raise ValueError(
                "GROQ_API_KEY not found"
            )

        self.llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=SecretStr(groq_api_key)
      )

        print(
            f"[INFO] Groq model: {llm_model}"
        )

    def search_and_summarize(
        self,
        query: str,
        top_k: int = 5
    ) -> str:

        results = self.vectorstore.query(
            query,
            top_k
        )

        texts = [
            r["metadata"]["text"]
            for r in results
            if r["metadata"]
        ]

        if not texts:
            return (
                "No relevant documents found."
            )

        context = "\n\n".join(texts)

        prompt = f"""
Answer the question using only the context.

Question:
{query}

Context:
{context}

Answer:
"""

        response = self.llm.invoke(prompt)

        return str(response.content)


if __name__ == "__main__":

    rag = RAGSearch()

    answer = rag.search_and_summarize(
        "How to create ABC ID?",
        top_k=3
    )

    print(answer)