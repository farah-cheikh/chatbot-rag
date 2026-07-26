import re
import spacy

nlp = spacy.load("fr_core_news_md")

def clean_text(text):

    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    doc = nlp(text)
    tokens = []
    for token in doc:
        if token.is_space:
            continue

        if token.is_punct:
            continue

        if token.like_num:
            tokens.append(token.text)
            continue

        if token.text.isupper():
            tokens.append(token.text)
            continue

        tokens.append(token.text)

    return " ".join(tokens)

def preprocess_documents(documents):
    processed_documents = []
    for doc in documents:
        cleaned_text = clean_text(doc["text"])
        processed_documents.append({
            "filename": doc["filename"],
            "text": cleaned_text
        })
    return processed_documents