import os
from typing import Any

import chromadb
import torch
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from paths import DATA_DIR


def insert_publications_to_db():
    """Iterate over the publications, chunk them, generate embeddings, and insert them into the ChromaDB collection."""
    next_id = collection.count()
    for publication in publications:
        publication_chunks = chunk_publication(publication, chunk_size=1000, chunk_overlap=200)
        # chunk_texts = [chunk["content"] for chunk in publication_chunks]
        embeddings = embedding_model.embed_documents(publication_chunks)
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




def chunk_publication(publication: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> list[Any]:
    """Chunk the publication into smaller documents using a RecursiveCharacterTextSplitter."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunk_data = [
        # {
        #     "chunk_id": idx,
        #     "title": publication.title,
        #     "content": chunk
        # }
        chunk
        for chunk in [text_splitter.split_text(publication)]
        for idx, chunk in enumerate(chunk)
    ]
    return chunk_data


def load_publications() -> list[str]:
    """Add all text files from the specified path to a list as LangChain Documents to a list.
    Add the page_content of each Document to a publications list and return it.
    Args:
        document_path: Path to the directory containing text langchain_documents.
    Returns:
        List of publication contents as strings.
    """
    langchain_documents: list[Document] = [] # no need to mention the type here, added just for clarity
    publication_list: list[str] = []
    for file_name in os.listdir(DATA_DIR):
        # if file_name.endswith(".txt"):
        if file_name.endswith(".txt") or file_name.endswith(".md"):
            file_path = os.path.join(DATA_DIR, file_name)
            try:
                """Convert each file into a LangChain Document. Each LangChain Document has page_content and metadata 
                    attributes."""
                langchain_documents.extend(TextLoader(file_path, encoding="utf-8").load())
                print(f"Loaded document {file_name}")
            except Exception as e:
                print(f"Error loading {file_name}: {e}")
    print(f"Loaded {len(langchain_documents)} langchain_documents")
    for doc in langchain_documents:
        """"???
            Why do we need to create LangChain Documents first and then extract page_content if we only need the text?
        """
        publication_list.append(doc.page_content)
    return publication_list


def initialize_chroma_db() -> chromadb.Collection:
    # client = chromadb.PersistentClient(database="/.research_db")

    client = chromadb.CloudClient(
        api_key=os.getenv("CHROMA_DB_API_KEY") ,
        tenant=os.getenv("CHROMA_TENANT"),
        database=os.getenv("CHROMA_DATABASE")
    )



    return client.get_or_create_collection(
        name="research_collection",
        metadata={
            "hnsw:space": "cosine",
            # "hnsw:batch_size": 10000,
        },
    )

def initialize_embedding_model() -> HuggingFaceEmbeddings:
    device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Initializing embedding model on device: {device}")
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": device},
    )

if __name__ == "__main__":
    """Steps to implement a Retrieval-Augmented Generation (RAG) system.
        1. Set up a vector database to store and index documents. 📚
        2. Load the documents. 📖
        3. Chunk the documents into smaller pieces. 📄
        4. Generate embeddings for the document chunks. 🔍
        5. Insert the embeddings and chunks into the vector database. 💾
        6. Implement a retrieval mechanism to intelligently fetch relevant document chunks based on user queries. 🎯
        7. Integrate the retrieval mechanism with a language model to generate context-aware responses. 🤖
    """
    load_dotenv()
    collection = initialize_chroma_db()
    print(f"ChromaDB collection {collection} initialized.")
    publications = load_publications()
    print(f"Loaded {len(publications)} publications from documents")
    # document_chunks = chunk_publications(publications = publications, chunk_size=1000, chunk_overlap=200)
    embedding_model = initialize_embedding_model()
    print("Embedding model initialized.")
    insert_publications_to_db()
