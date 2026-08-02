from src.extract import load_documents
from src.quality import check_quality
from src.preprocessing import preprocess_documents
from src.chunking import create_chunks
from src.embedding import create_embeddings
from src.vector_store import save_vector_store
from src.retriever import retrieve
from src.llm import generate_answer

# 1. Charger les documents
documents = load_documents("data/raw")

# 2. Vérifier la qualité
quality_df = check_quality(documents)

# 3. Nettoyer les documents
processed_documents = preprocess_documents(documents)

# 4. Créer les chunks
chunks = create_chunks(processed_documents)

print("\nPremier chunk :\n")
print(chunks[0]["text"][:1000])

# 5. Créer les embeddings
embedding_documents = create_embeddings(chunks)

print("\nNombre d'embeddings :", len(embedding_documents))

# 6. Sauvegarder dans FAISS
save_vector_store(embedding_documents)


# 7. Tester le retrieval

question = "Quelle est la règle de transformation du champ montant à partir de bkhis.mon et bkhis.sen"
results = retrieve(question, top_k=5)
print("Question :", question)

for i, result in enumerate(results, start=1):
    print(f"\n Résultat {i}")
    print("Document :", result["filename"])
    print("Texte :")
    print(result["text"])
# 8.reponse 
context = "\n".join(
    result["text"]
    for result in results
)
answer = generate_answer(
    question=question,
    context=context,
    model="qwen2.5:3b"
)
print("\nRÉPONSE DU CHATBOT ")
print(answer)