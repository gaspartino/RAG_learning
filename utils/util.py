import faiss
import numpy as np
from pypdf import PdfReader

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


def create_embeddings(model, pdf_path, chunk_size=1000, overlap=200):
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

def create_context(results):
    context = ""

    for result in results:
        context += result["chunk"] + "\n\n"

    return context

def create_prompt(query, context):
    prompt = f"""
        Você é um agente de inteligência artificial especializado em tirar dúvidas
        sobre os conteúdos apresentados no material de aprendizado fornecido.
        
        Seu conhecimento, para fins desta tarefa, está restrito ao contexto recuperado
        do material, que aborda os seguintes temas:
        - Retrieval-Augmented Generation (RAG);
        - FAISS (Facebook AI Similarity Search);
        - MCP (Model Context Protocol).
        
        Sua função é auxiliar o usuário no aprendizado desses conceitos, respondendo
        às perguntas de maneira clara, precisa, didática e culta.
        
        Regras para elaborar a resposta:
        
        1. Utilize prioritariamente e exclusivamente as informações presentes no
           contexto fornecido abaixo.
        
        2. Não invente informações, exemplos, definições ou detalhes que não estejam
           presentes no contexto.
        
        3. Caso a pergunta não possa ser respondida com base nas informações
           disponíveis no contexto, informe claramente que não foi possível encontrar
           a resposta no material fornecido.
        
        4. Quando apropriado, explique os conceitos de maneira didática, utilizando
           exemplos presentes no próprio contexto.
        
        5. Mantenha uma linguagem culta, clara e objetiva, evitando respostas
           excessivamente informais.
        
        6. Não mencione que você é um modelo de linguagem, a menos que isso seja
           diretamente solicitado pelo usuário.
        
        7. Não responda apenas reproduzindo o contexto. Organize e sintetize as
           informações para fornecer uma resposta adequada à pergunta.
        
        Contexto de aprendizado:
        {context}
        
        Pergunta do usuário:
        {query}
        
        Resposta:
        """

    return prompt
