from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-m3")

def create_embeddings(chunks):
    embeddings = []
    for chunk in chunks:
        vector = model.encode(
            chunk["text"],
            normalize_embeddings=True )
        embeddings.append({
            "filename": chunk["filename"],
            "text": chunk["text"],
            "embedding": vector})
    return embeddings