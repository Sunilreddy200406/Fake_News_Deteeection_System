from __future__ import annotations

from typing import List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # Optional dependency.
    SentenceTransformer = None


def tfidf_similarity_score(text: str, candidates: List[str]) -> Tuple[float, int]:
    if not text.strip() or not candidates:
        return 0.0, -1

    corpus = [text] + candidates
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(corpus)
    scores = cosine_similarity(matrix[0:1], matrix[1:]).flatten()
    idx = int(np.argmax(scores))
    return float(scores[idx]), idx


def embedding_similarity_score(text: str, candidates: List[str]) -> Tuple[float, int]:
    if SentenceTransformer is None or not text.strip() or not candidates:
        return 0.0, -1

    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode([text] + candidates, normalize_embeddings=True)
        source = embeddings[0]
        targets = embeddings[1:]
        scores = np.dot(targets, source)
        idx = int(np.argmax(scores))
        return float(scores[idx]), idx
    except Exception:
        return 0.0, -1

