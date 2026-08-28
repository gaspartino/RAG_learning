from model import (
    get_model,
    generate_response
)

from utils import (
    create_embeddings,
    create_faiss_index,
    search_faiss,
    create_context,
    create_prompt
)


def main():
    pdf_path = "data/rag_guides.pdf"

    embedding_model = get_model("embedding")
    llm = get_model("LLM")

    chunks, embeddings = create_embeddings(
        pdf_path,
        embedding_model
    )

    index = create_faiss_index(embeddings)

    query = input("Digite sua pergunta: ")

    results = search_faiss(
        query,
        embedding_model,
        index,
        chunks,
        k=5
    )

    context = create_context(results)

    prompt = create_prompt(
        query,
        context
    )

    response = generate_response(
        llm,
        prompt
    )

    print("\nResposta:")
    print(response)


if __name__ == "__main__":
    main()
