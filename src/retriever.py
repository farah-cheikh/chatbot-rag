import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

model = SentenceTransformer("BAAI/bge-m3")

index = faiss.read_index("data/vector_store/index.faiss")

with open("data/vector_store/metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

# Préparer BM25
documents = []
for doc in metadata:
    documents.append(doc["text"].lower().split())

bm25 = BM25Okapi(documents)


def retrieve(query, top_k=5, k_dense=10, k_bm25=10, rrf_k=60):

    # Recherche FAISS (on récupère plus large que top_k pour bien fusionner après)
    query_embedding = model.encode(query, normalize_embeddings=True)
    query_embedding = np.array([query_embedding], dtype="float32")

    scores_faiss, indices_faiss = index.search(query_embedding, k_dense)
    indices_faiss = indices_faiss[0]

    # Recherche BM25
    query_tokens = query.lower().split()
    scores_bm25 = bm25.get_scores(query_tokens)
    indices_bm25 = np.argsort(scores_bm25)[::-1][:k_bm25]

    # Fusion par rang 
    rrf_scores = {}

    for rank, idx in enumerate(indices_faiss):
        idx = int(idx)
        if idx == -1:
            continue
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (rrf_k + rank + 1)

    for rank, idx in enumerate(indices_bm25):
        idx = int(idx)
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (rrf_k + rank + 1)

    # Trier par score RRF décroissant
    sorted_indices = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    top_indices = [idx for idx, score in sorted_indices[:top_k]]

    results = []
    for i in top_indices:
        results.append(metadata[i])

    return results