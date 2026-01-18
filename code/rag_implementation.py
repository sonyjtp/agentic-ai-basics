import os
from typing import Any

import chromadb
import torch
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from paths import DATA_DIR

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def search_similar_documents(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Search for similar documents in the ChromaDB collection based on the query.
    Args:
        query: The input query string.
        top_k: The number of top similar documents to retrieve.
    Returns:
        A list of dictionaries containing the similar documents and their metadata.
    """
    query_vector = embedding_model.embed_query(query)
    query_results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    similar_documents = []
    for idx, doc in enumerate(query_results['documents'][0]):
        similar_documents.append({
            "document": doc,
            "chunk_index": query_results['metadatas'][0][idx].get("chunk_index", "-1"),
            "similarity_score": 1 - query_results['distances'][0][idx]
        })
    return similar_documents

def create_context(query: str, top_k: int) -> str:
    """Create a context for the question by retrieving similar documents from the ChromaDB collection.
    Args:
        query: The input question string.
        top_k: The number of top similar documents to retrieve for context.
    Returns:
        A combined context string from the similar documents.
    """
    similar_documents = search_similar_documents(query, top_k=top_k)
    combined_context = "\n".join(
        [doc["document"] for doc in similar_documents]
    )
    # Here you would typically call your LLM with the combined_context to generate an answer.
    # For simplicity, we'll just return the combined context as the "answer".
    return combined_context

def insert_publications_to_db():
    """Iterate over the publications, chunk them, generate embeddings, and insert them into the ChromaDB collection."""
    next_id = collection.count()
    for publication in publications:
        publication_chunks = chunk_publication(publication, chunk_size=1000, chunk_overlap=200)
        # chunk_texts = [chunk["content"] for chunk in publication_chunks]
        embeddings = embedding_model.embed_documents(publication_chunks)
        ids = list(range(next_id, next_id + len(publication_chunks)))
        keys = [f"document_{idn}" for idn in ids] # using idn to avoid shadowing built-in id()
        metadata = [
            {
                "chunk_index": idx
            }
            for idx in range(len(publication_chunks))
        ]
        collection.add(
            ids=keys,
            embeddings=embeddings,
            documents=publication_chunks,
            metadatas=metadata,
        )
        next_id += len(publication_chunks)
        print(f"Inserted {len(publication_chunks)} chunks into the database. Total count: {collection.count()}")




def chunk_publication(publication: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> list[str]:
    """Chunk the publication into smaller documents using a RecursiveCharacterTextSplitter."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    # chunk_data = [
        # {
        #     "chunk_id": idx,
        #     "title": publication.title,
        #     "content": chunk
        # }
    #     chunk
    #     for chunks in [text_splitter.split_text(publication)]
    #     # for idx, chunk in enumerate(chunks)
    # ]
    # return chunk_data
    return text_splitter.split_text(publication)


def load_publications() -> list[str]:
    """Add all text files from the specified path to a list as LangChain Documents to a list.
    Add the page_content of each Document to a publications list and return it.
    Returns:
        List of publication contents as strings.
    """
    langchain_documents = []
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
        api_key=os.getenv("CHROMA_API_KEY") ,
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


def build_question_prompt():
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=""""
                Based on the following research context, answer the question:
                Research Context:
                {context}
                Research Question:
                {question}
                Answer: Provide a detailed answer based on the research context above.
            """
    )
    return prompt_template.format(context=question_context, question=f"{question}")


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
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.7,
        api_key=os.getenv("GROQ_API_KEY")
    )
    collection = initialize_chroma_db()
    print(f"ChromaDB collection {collection} initialized.")
    publications = load_publications()
    print(f"Loaded {len(publications)} publications from documents")
    # document_chunks = chunk_publications(publications = publications, chunk_size=1000, chunk_overlap=200)
    embedding_model = initialize_embedding_model()
    print("Embedding model initialized.")
    insert_publications_to_db()
    question = "Applications of Variational Autoencoders"
    question_context = create_context(query=question, top_k=3)
    prompt = build_question_prompt()
    response = llm.invoke(prompt)
    print("AI Response:", response.content)
    print("RAG process complete.")

