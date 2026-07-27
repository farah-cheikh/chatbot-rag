from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("BAAI/bge-m3")

def create_embeddings(chunks):
    embeddings = []

    for chunk in chunks:
        vector = model.encode(
            chunk["text"],
            normalize_embeddings=True
        )
        embeddings.append({
            "filename": chunk["filename"],
            "text": chunk["text"],
            "embedding": vector
        })


    # Vérification de dimension
    if len(embeddings) > 0:
        print("Nombre d'embeddings :", len(embeddings))

    dimension_ok = True

    for emb in embeddings:
        if len(emb["embedding"]) != len(embeddings[0]["embedding"]):
            dimension_ok = False
            break

    if dimension_ok:
        print("Tous les vecteurs sont de même dimension")
    else:
        print("Dimensions différentes")

    # Sanity check
    if len(embeddings) >= 2:
        sim = cosine_similarity(
            [embeddings[0]["embedding"]],
            [embeddings[1]["embedding"]]
        )[0][0]

        print("-- Test similarité --")
        print(f"similarité entre le chunk 1 et le chunk 2 : {sim:.4f}")

    return embeddings