from src.extract import load_documents
from src.quality import check_quality
from src.preprocessing import preprocess_documents
from src.chunking import create_chunks
from src.embedding import create_embeddings
# 1. Charger les documents
documents = load_documents("data/raw")

# 2. Vérifier la qualité
quality_df = check_quality(documents)

# 3. Nettoyer les documents
processed_documents = preprocess_documents(documents)

# 4. Créer les chunks
chunks = create_chunks(processed_documents)

print(f"\nNombre total de chunks : {len(chunks)}")

print("\nPremier chunk :\n")
print(chunks[0]["text"][:1000])

# Créer les embeddings
embedding_documents = create_embeddings(chunks)

print("Nombre d'embeddings :", len(embedding_documents))
print("Dimension du premier embedding :", len(embedding_documents[0]["embedding"]))