import os
import json
import time
from src.llm import generate_answer
from src.retriever import retrieve

MODELS = [
    "qwen2.5:3b",
    "llama3.2:3b",
    "phi3.5:3.8b",
    "llama3.1:latest"
]

QUESTIONS = [
    "que signifie le guichet vr?",
    "Quel est le plafond de financement pour le crédit etudes ?",
    "C'est quoi la différence entre TAHSSIN et TAHSSIN PRO",
    "Est-ce que j'ai droit à un crédit si j'ai eu un chèque impayé"
]

def test_model(model, questions, retriever):
    results = []
    for question in questions:
        print("modele :", model)
        print("question:", question)

        # retrieval
        retrieved_documents = retriever(question)

        # contexte
        context = "\n".join(document["text"] for document in retrieved_documents)

        # temps de génération
        start_time = time.perf_counter()
        answer = generate_answer(question=question, context=context, model=model)
        end_time = time.perf_counter()
        response_time = end_time - start_time

        result = {
            "model": model,
            "question": question,
            "context": context,
            "answer": answer,
            "response_time": round(response_time, 3),
            "answer_length": len(answer)
        }

        results.append(result)
        print("\nresponse:")
        print(answer)
        print(f"\nTemps de réponse : {response_time:.3f} secondes")
    return results

def save_results(results, model):
    os.makedirs("modeles/test_modeles", exist_ok=True)
    filename = model.replace(":", "_").replace(".", "_")
    filepath = f"modeles/test_modeles/{filename}.json"

    with open(filepath, "w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=4)

    print(f"\nrésultats sauvegardés dans : {filepath}")

if __name__ == "__main__":
    for model in MODELS:
        results = test_model(model, QUESTIONS, retrieve)
        save_results(results, model)