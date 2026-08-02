import faiss
import numpy as np
import os 
import pickle

def save_vector_store(embedding_documents):

    os.makedirs("data/vector_store", exist_ok=True)
    vectors=[]
    for doc in embedding_documents:
        vectors.append(doc["embedding"])
    vectors=np.array(vectors,dtype="float32")
    
    if len(vectors)==0:
        raise ValueError("Aucun embedding trouvé")
    #creation dindex faiss
    index= faiss.IndexFlatIP(vectors.shape[1])

    index.add(vectors)
# sauvegarder l'index
    faiss.write_index(index,"data/vector_store/index.faiss")

    #sauvegarder les metadonnées
    metadata=[]
    for doc in embedding_documents:
        metadata.append({
            "filename":doc["filename"],
            "text": doc["text"]
        })

    with open("data/vector_store/metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)
    print("vector store sauvegardé")
 