import os
import pickle

import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util


class RetrievalService:
    def __init__(self):
        self.documents = []
        self.embedding_model = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

        self.document_embeddings = None

        self.cache_dir = "cache"

        self.documents_cache_path = os.path.join(
            self.cache_dir,
            "turkish_retrieval_documents.pkl"
        )

        self.embeddings_cache_path = os.path.join(
            self.cache_dir,
            "turkish_retrieval_embeddings.pt"
        )

        self._prepare_cache_dir()
        self._load_data()

    def _prepare_cache_dir(self):
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def _load_data(self):
        if self._cache_exists():
            self._load_from_cache()
        else:
            self._build_and_save_cache()

    def _cache_exists(self):
        return (
            os.path.exists(self.documents_cache_path)
            and os.path.exists(self.embeddings_cache_path)
        )

    def _load_from_cache(self):
        with open(self.documents_cache_path, "rb") as file:
            self.documents = pickle.load(file)

        self.document_embeddings = torch.load(
            self.embeddings_cache_path
        )

    def _build_and_save_cache(self):
        data = pd.read_excel(
            "data/retrieval_tr/turkish_news.xls"
        )

        data = data[["headline", "content"]].dropna()

        data["combined"] = (
            data["headline"].astype(str)
            + " "
            + data["content"].astype(str)
        )

        # Test için ilk etapta sınırlıyoruz
        data = data.sample(n=3000, random_state=42)

        self.documents = data["combined"].tolist()

        self.document_embeddings = self.embedding_model.encode(
            self.documents,
            convert_to_tensor=True,
            show_progress_bar=True
        )

        with open(self.documents_cache_path, "wb") as file:
            pickle.dump(self.documents, file)

        torch.save(
            self.document_embeddings,
            self.embeddings_cache_path
        )

    def retrieve(self, query: str, top_k: int = 3):
        if not query or not query.strip():
            return []

        query_embedding = self.embedding_model.encode(
            query,
            convert_to_tensor=True
        )

        similarities = util.cos_sim(
            query_embedding,
            self.document_embeddings
        )[0]

        top_k = min(top_k, len(self.documents))

        top_results = np.argpartition(
            -similarities.cpu().numpy(),
            range(top_k)
        )[:top_k]

        ranked_results = sorted(
            [
                (idx, similarities[idx].item())
                for idx in top_results
            ],
            key=lambda x: x[1],
            reverse=True
        )

        results = []

        for idx, score in ranked_results:
            results.append(
                self.documents[idx][:700]
            )

        return results