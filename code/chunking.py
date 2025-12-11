from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
def process_file(file_path):
    # 1. Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    # 2. Create chunks by intelligent splitting
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_text(text)
    # 3. Create documents from chunks
    documents = [
        Document(
            page_content = chunk,
            metadata = {"source": file_path, "chunk_id": i}
        ) for i, chunk in enumerate(chunks)
    ]
    # 4. Create searchable vector store
    embeddings = HuggingFaceEmbeddings(model_name = 'all-MiniLM-L6-v2')
    vector_store = Chroma.from_documents(documents, embeddings)
    return vector_store
