from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document


def create_chunks(processed_documents):

    splitter = SentenceSplitter(
        chunk_size=512,
        chunk_overlap=50
    )

    all_chunks = []

    for doc in processed_documents:

        # transformer le texte en document LlamaIndex
        llama_doc = Document(
            text=doc["text"],
            metadata={"filename": doc["filename"]}
        )

        nodes = splitter.get_nodes_from_documents([llama_doc])

        for node in nodes:
            all_chunks.append({
                "filename": doc["filename"],
                "text": node.text
            })

    return all_chunks