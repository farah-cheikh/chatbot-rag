import pickle

with open("data/vector_store/metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

print(" Tous les chunks du document COMMISION_AMP \n")

for i, chunk in enumerate(metadata):
    if chunk["filename"] == "Fiche_COMMISION_AMP.docx":
        print(f" Chunk numéro {i} (taille: {len(chunk['text'])} caractères)")
        print(chunk["text"])
        print("\n" + "="*80 + "\n")