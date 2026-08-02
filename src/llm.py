import ollama

def generate_answer(question, context, model="qwen2.5:3b"):

    prompt = f"""
Tu es un assistant spécialisé dans la documentation bancaire.Réponds à la question uniquement à partir du contexte fourni.
Si la réponse n'est pas présente dans le contexte,dis clairement que l'information n'est pas trouvée dans les documents.
Ne crée pas d'information.
Contexte :
{context}
Question :
{question}
Réponse :
"""
    response=ollama.chat(model=model,messages=[{"role":"user","content":prompt}])
    return response["message"]["content"]
