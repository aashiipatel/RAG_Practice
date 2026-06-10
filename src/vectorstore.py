import os
import pickle
from typing import List, Any, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.embeddings import EmbeddingPipeline


class FaissVectorStore:

    def __init__(
        self,
        persist_dir: str = "faiss_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)

        self.index: Optional[faiss.Index] = None
        self.metadata: List[Any] = []

        self.embedding_model = embedding_model
        self.model = SentenceTransformer(embedding_model)

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        print(f"[INFO] Loaded embedding model: {embedding_model}")

    def build_from_documents(self, documents: List[Any]):

        print(
            f"[INFO] Building vector store from "
            f"{len(documents)} documents..."
        )

        emb_pipe = EmbeddingPipeline(
            model_name=self.embedding_model,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

        chunks = emb_pipe.chunk_documents(documents)

        embeddings = emb_pipe.embed_chunks(chunks)

        metadatas = [
            {
                "text": chunk.page_content,
                "metadata": chunk.metadata
            }
            for chunk in chunks
        ]

        self.add_embeddings(
            embeddings.astype(np.float32),
            metadatas
        )

        self.save()

    def add_embeddings(
        self,
        embeddings: np.ndarray,
        metadatas: Optional[List[Any]] = None
    ):

        dim = embeddings.shape[1]

        if self.index is None:
            self.index = faiss.IndexFlatL2(dim)

        self.index.add(embeddings)

        if metadatas:
            self.metadata.extend(metadatas)

        print(
            f"[INFO] Added "
            f"{embeddings.shape[0]} vectors."
        )

    def save(self):

        if self.index is None:
            raise ValueError("No FAISS index to save")

        faiss.write_index(
            self.index,
            os.path.join(
                self.persist_dir,
                "faiss.index"
            )
        )

        with open(
            os.path.join(
                self.persist_dir,
                "metadata.pkl"
            ),
            "wb"
        ) as f:
            pickle.dump(self.metadata, f)

    def load(self):

        self.index = faiss.read_index(
            os.path.join(
                self.persist_dir,
                "faiss.index"
            )
        )

        with open(
            os.path.join(
                self.persist_dir,
                "metadata.pkl"
            ),
            "rb"
        ) as f:
            self.metadata = pickle.load(f)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5
    ):

        if self.index is None:
            raise ValueError(
                "FAISS index not loaded"
            )

        distances, indices = self.index.search(
            query_embedding,
            top_k
        )

        results = []

        for idx, dist in zip(
            indices[0],
            distances[0]
        ):
            results.append(
                {
                    "index": int(idx),
                    "distance": float(dist),
                    "metadata":
                        self.metadata[idx]
                        if idx < len(self.metadata)
                        else None
                }
            )

        return results

    def query(
        self,
        query_text: str,
        top_k: int = 5
    ):

        query_emb = self.model.encode(
            [query_text],
            convert_to_numpy=True
        ).astype(np.float32)

        return self.search(
            query_emb,
            top_k
        )