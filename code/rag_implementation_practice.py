import os

import chromadb
import torch
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from paths import ROOT_DIR
os.environ["TOKENIZERS_PARALLELISM"] = "false"
MY_DATA_DIR = os.path.join(ROOT_DIR, "data_practice")

def initialize_chroma_db() -> chromadb.Collection:
    client = chromadb.CloudClient(
        api_key=os.getenv("CHROMA_API_KEY"),
        tenant=os.getenv("CHROMA_TENANT"),
        database=os.getenv("CHROMA_DATABASE")
    )
    return client.get_or_create_collection(
        name="indian_states",
        metadata={
            "hnsw:space": "cosine"
        },
    )

def load_publications() -> list[str]:
    """Load all publication text files from the DATA_DIR and return their contents as a list of strings."""
    publication_list: list[str] = []
    langchain_documents = []
    for file_name in os.listdir(MY_DATA_DIR):
        if file_name.endswith(".md"):
            file_path = os.path.join(MY_DATA_DIR, file_name)
            langchain_documents.extend(TextLoader(file_path, encoding="utf-8").load())
    for document in langchain_documents:
        publication_list.append(document.page_content)
    return publication_list

def initialize_embedding_model():
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device for embeddings: {device}")
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": device},
    )

def create_chunks_from_publications(publications):
    text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=500,
      chunk_overlap=100,
      separators= ["\n\n", "\n", ".",  " ", ""],

    )
    chunk_data = [
        {
            "publication_index": pub_index,
            "chunk_index": chunk_index,
            "chunk_text": chunk,
        }
        for pub_index, pub in enumerate(publications)
        for chunk_index, chunk in enumerate(text_splitter.split_text(pub))
    ]
    return chunk_data

def insert_chunks_into_db(collection, chunks, embedding_model):
    next_id = collection.count()
    ids = list(range(next_id, next_id + len(chunks)))
    ids = [f"document_{id}" for id in ids]
    chunk_texts = [chunk["chunk_text"] for chunk in chunks]
    embeddings = embedding_model.embed_documents(chunk_texts)
    collection.add(
        embeddings=embeddings,
        ids=ids,
        documents=chunk_texts,
    )

def find_similar_chunks(embedding_model, query, collection, top_k=3):
    query_vector = embedding_model.embed_query(query)

    query_results =collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=[ "documents", "metadatas", "distances"]
    )
    similar_chunks = []
    for idx, document in enumerate(query_results["documents"][0]):
        similar_chunks.append({
            "document": document,
            "metadata": query_results["metadatas"][0][idx],
            "similarity_score": 1 - query_results["distances"][0][idx],
        })
    return similar_chunks

def create_context(similar_chunks):
    combined_context = "\n".join(
        [chunk["document"] for chunk in similar_chunks]
    )
    return combined_context

def build_prompt(question: str, context: str) -> str:
    prompt_template = PromptTemplate(
        input_variables=["question", "context"],
        template=(
            """ 
                Based on the following context
                {context}
                answer the question:
                {question}
                
            """
        ),
    )
    return prompt_template.format(question=question, context=context)

def print_similar_chunks(similar_chunks):
    print("\n=== Similar Chunks ===")
    for idx, chunk in enumerate(similar_chunks, 1):
        print(f"\nChunk {idx} (Similarity: {chunk['similarity_score']:.4f}):")
        print(f"{chunk['document'][:200]}...")  # Print first 200 chars
    print("=" * 50 + "\n")

def main():
    collection = initialize_chroma_db()
    print("ChromaDB collection initialized.")
    publications = load_publications()
    print("Publications loaded.")
    embedding_model = initialize_embedding_model()
    print("Embedding model initialized.")
    chunks = create_chunks_from_publications(publications)
    print("Chunks created from publications.")
    insert_chunks_into_db(collection, chunks, embedding_model)
    print("Chunks inserted into database.")
    query = "Where in India can we find backwaters?"
    print("Query:", query)
    similar_chunks = find_similar_chunks(embedding_model, query, collection)
    print_similar_chunks(similar_chunks)
    combined_context = create_context(similar_chunks)
    prompt = build_prompt(query, combined_context)
    print(f"Constructed Prompt:\n{prompt}")
    llm  = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.7,
        api_key=os.getenv("GROQ_API_KEY")
    )
    response = llm.invoke(prompt)
    print(response.content)





if __name__ == "__main__":
    load_dotenv()
    main()