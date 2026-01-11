import os
from typing import Any

import chromadb
import torch
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def insert_publications_to_db(collection: chromadb.Collection, publications: list[str]):
    """Iterate over the publications, chunk them, generate embeddings, and insert them into the ChromaDB collection."""
    next_id = collection.count()
    for publication in publications:
        publication_chunks = chunk_publication(publication, chunk_size=1000, chunk_overlap=200)
        # chunk_texts = [chunk["content"] for chunk in publication_chunks]
        embeddings = embed_document_chunks(publication_chunks)
        ids = list(range(next_id, next_id + len(publication_chunks)))
        ids = [f"document_{idn}" for idn in ids] # using idn to avoid shadowing built-in id()
        # metadata = [{"chunk_id": chunk["chunk_id"]} for chunk in publication_chunks]
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=publication_chunks,
        )
        next_id += len(publication_chunks)
        print(f"Inserted {len(publication_chunks)} chunks into the database. Total count: {collection.count()}")



def embed_document_chunks(chunks: list[str]) -> list[list[float]]:
    """Generate embeddings for each document chunk using HuggingFaceEmbeddings."""
    device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": device},
    )
    embeddings = model.embed_documents(chunks)
    return embeddings

def chunk_publication(publication: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> list[Any]:
    """Chunk the publication into smaller documents using a RecursiveCharacterTextSplitter."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunk_data = [
        {
            "chunk_id": idx, 
            "title": publication.title, 
            "content": chunk
        }
        for chunk in [text_splitter.split_text(publication)]
        for idx, chunk in enumerate(chunk)
    ]
    return chunk_data


def load_publications(document_path) -> list[str]:
    """Add all text files from the specified path to a list as LangChain Documents to a list.
    Add the page_content of each Document to a publications list and return it.
    Args:
        document_path: Path to the directory containing text documents.
    Returns:
        List of publication contents as strings.
    """
    documents: list[Document] = [] # no need to mention the type here, added just for clarity
    publications: list[str] = []
    for file_name in os.listdir(document_path):
        if file_name.endswith(".txt"):
            file_path = os.path.join(document_path, file_name)
            try:
                """Convert each file into a LangChain Document. Each Document contains a page_content (text content)
                 and metadata"""
                documents.extend(TextLoader(file_path, encoding="utf-8").load())
                print(f"Loaded document {file_name}")
            except Exception as e:
                print(f"Error loading {file_name}: {e}")
    print(f"Loaded {len(documents)} documents")
    for doc in documents:
        publications.append(doc.page_content)
    return publications


def initialize_chroma_db() -> chromadb.Collection:
    """Initialize ChromaDB persistent client and collection."""
    client = chromadb.PersistentClient(database="/.research_db")
    return client.get_or_create_collection(
        name="research_collection",
        metadata={
            "hnsw:space": "cosine",
            # "hnsw:batch_size": 10000,
        },
    )


def __main__():
    """Steps to implement a Retrieval-Augmented Generation (RAG) system.
        1. Set up a vector database to store and index documents. 📚
        2. Load the documents. 📖
        3. Chunk the documents into smaller pieces. 📄
        4. Generate embeddings for the document chunks. 🔍
        5. Insert the embeddings and chunks into the vector database. 💾
        6. Implement a retrieval mechanism to intelligently fetch relevant document chunks based on user queries. 🎯
        7. Integrate the retrieval mechanism with a language model to generate context-aware responses. 🤖
    """
    collection = initialize_chroma_db()
    publications = load_publications("") # TODO add path to documents
    # document_chunks = chunk_publications(publications = publications, chunk_size=1000, chunk_overlap=200)
    insert_publications_to_db(collection = collection, publications = publications)


