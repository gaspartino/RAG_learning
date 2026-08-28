import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


def extract_text(pdf_path):
    reader = PdfReader(pdf_path)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


def create_chunks(text, chunk_size=1000, overlap=200):
    chunks = []

    start = 0

    while start < len(text):
        end = start + chunk_size

        chunk = text[start:end]

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def create_embeddings(pdf_path, chunk_size=1000, overlap=200):
    model = SentenceTransformer("all-MiniLM-L6-v2")

    text = extract_text(pdf_path)

    chunks = create_chunks(
        text,
        chunk_size,
        overlap
    )

    embeddings = model.encode(chunks)

    return chunks, embeddings


def create_faiss_index(embeddings):
    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    return index


def search_faiss(query, model, index, chunks, k=5):
    query_embedding = model.encode([query])

    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = index.search(query_embedding, k)

    results = []

    for distance, index_position in zip(distances[0], indices[0]):
        results.append({
            "chunk": chunks[index_position],
            "distance": distance
        })

    return results
