"""
Hybrid Retriever for CST Cloud Computing Regulatory Framework (RS10)
Supports:
1. Arabic Text Normalization & Tokenization
2. Lexical Keyword & Field-Boosted Matching (BM25-style)
3. Vector / Semantic Cosine Similarity (Gemini Embedding or TF-IDF Vectorizer)
4. Regulatory Chunk Ranking & Highlighting
"""

import os
import json
import re
import math
from typing import List, Dict, Any, Optional
import numpy as np

def normalize_arabic(text: str) -> str:
    """Normalize Arabic text for high-accuracy search matching."""
    if not text:
        return ""
    text = text.lower()
    # Remove tashkeel (diacritics)
    text = re.sub(r'[\u064B-\u0652\u0670]', '', text)
    # Remove tatweel
    text = re.sub(r'\u0640', '', text)
    # Normalize Alefs
    text = re.sub(r'[إأآٱ]', 'ا', text)
    # Normalize Teh Marbuta to Heh
    text = re.sub(r'ة', 'ه', text)
    # Normalize Yeh/Alef Maksura
    text = re.sub(r'ى', 'ي', text)
    # Remove excessive whitespace & punctuation
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

class HybridRetriever:
    """
    Retrieves regulatory chunks using a hybrid approach:
    Semantic similarity + Lexical keyword scoring + Metadata boosting.
    """

    def __init__(self, chunks_path: str, gemini_api_key: Optional[str] = None):
        self.chunks_path = chunks_path
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        self.chunks: List[Dict[str, Any]] = []
        self.vocabulary: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.doc_vectors: List[np.ndarray] = []
        self._load_chunks()
        self._build_index()

    def _load_chunks(self):
        """Loads structured chunks from JSON."""
        if not os.path.exists(self.chunks_path):
            raise FileNotFoundError(f"Chunks file not found at: {self.chunks_path}")
        with open(self.chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

    def _build_index(self):
        """Builds in-memory TF-IDF index for fast, local lexical-semantic scoring."""
        doc_count = len(self.chunks)
        df: Dict[str, int] = {}

        # Tokenize each chunk (including title, content, keywords, legal articles)
        chunk_tokens_list = []
        for chunk in self.chunks:
            full_text = " ".join([
                chunk.get("title", ""),
                chunk.get("classification", ""),
                chunk.get("entity_type", ""),
                " ".join(chunk.get("keywords", [])),
                chunk.get("content", ""),
                chunk.get("storage_location", ""),
                chunk.get("provider_class", "")
            ])
            norm = normalize_arabic(full_text)
            tokens = set(norm.split())
            chunk_tokens_list.append(norm.split())
            for t in tokens:
                df[t] = df.get(t, 0) + 1

        # Calculate IDF
        self.vocabulary = {t: idx for idx, t in enumerate(sorted(df.keys()))}
        for t, freq in df.items():
            self.idf[t] = math.log((1 + doc_count) / (1 + freq)) + 1.0

        # Build normalized TF-IDF vector for each chunk
        dim = len(self.vocabulary)
        self.doc_vectors = []
        for tokens in chunk_tokens_list:
            vec = np.zeros(dim, dtype=np.float32)
            for t in tokens:
                if t in self.vocabulary:
                    vec[self.vocabulary[t]] += 1.0
            # Apply IDF
            for t in set(tokens):
                if t in self.vocabulary:
                    vec[self.vocabulary[t]] *= self.idf[t]
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            self.doc_vectors.append(vec)

    def _get_query_vector(self, query: str) -> np.ndarray:
        """Transforms normalized query into TF-IDF vector."""
        norm_q = normalize_arabic(query)
        dim = len(self.vocabulary)
        vec = np.zeros(dim, dtype=np.float32)
        tokens = norm_q.split()
        for t in tokens:
            if t in self.vocabulary:
                vec[self.vocabulary[t]] += 1.0
        for t in set(tokens):
            if t in self.vocabulary:
                vec[self.vocabulary[t]] *= self.idf.get(t, 1.0)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieves top_k chunks matching the query using hybrid scoring.
        """
        norm_query = normalize_arabic(query)
        q_tokens = set(norm_query.split())
        q_vec = self._get_query_vector(query)

        scored_results = []
        for idx, chunk in enumerate(self.chunks):
            # 1. Cosine similarity score
            cos_sim = float(np.dot(q_vec, self.doc_vectors[idx])) if np.linalg.norm(q_vec) > 0 else 0.0

            # 2. Keyword boost
            keyword_score = 0.0
            keywords = [normalize_arabic(k) for k in chunk.get("keywords", [])]
            for k in keywords:
                if k in norm_query:
                    keyword_score += 0.35
                for token in q_tokens:
                    if token in k.split():
                        keyword_score += 0.15

            # 3. Direct Classification Title Boost
            title_norm = normalize_arabic(chunk.get("title", ""))
            class_norm = normalize_arabic(chunk.get("classification", ""))
            if class_norm and class_norm in norm_query:
                keyword_score += 0.5
            if any(term in norm_query for term in title_norm.split()):
                keyword_score += 0.2

            # Combined score
            final_score = (0.5 * cos_sim) + (0.5 * min(keyword_score, 1.0))

            scored_results.append({
                "chunk": chunk,
                "score": round(float(final_score), 4),
                "cos_sim": round(cos_sim, 4),
                "keyword_score": round(keyword_score, 4)
            })

        # Sort descending by final score
        scored_results.sort(key=lambda x: x["score"], reverse=True)
        return scored_results[:top_k]
