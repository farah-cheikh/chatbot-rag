import re

def clean_text(text):
    text = re.sub(r"[ \t]+", " ", text)         
    text = re.sub(r"\n{3,}", "\n\n", text)      
    text = "\n".join(line.strip() for line in text.split("\n"))  
    text = text.strip()
    return text


def preprocess_documents(documents):
    processed_documents = []
    for doc in documents:
        cleaned_text = clean_text(doc["text"])
        processed_documents.append({
            "filename": doc["filename"],
            "text": cleaned_text
        })
    return processed_documents