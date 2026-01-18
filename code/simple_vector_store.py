from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

load_dotenv()

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

texts = [
    """Rasgulla is a popular Indian dessert made from ball-shaped dumplings of chhena and semolina dough, cooked in 
        light syrup made of sugar.""",
    """Tamil Nadu is a state in southern India known for its rich culture and history.""",
    """Sambar Deer is a large deer native to the Indian subcontinent, known for its distinctive antlers and 
        spotted coat.""",
    """The Aghori are a small group of ascetic Shaiva sadhus based in India, known for their extreme and unconventional 
        practices aimed at achieving spiritual enlightenment.""",
    """Chandragupta Maurya was the founder of the Maurya Empire in ancient India, known for unifying most of the Indian
        subcontinent under one rule.""",
    """The Indian Space Research Organisation (ISRO) is the space agency of the Government of India, responsible for
        space research and exploration.""",
    """ The Banyan tree (Ficus benghalensis) is the national tree of India, known for its extensive aerial root system
        and large canopy.""",
    """ Theyyam is a ritual dance form of Kerala, India, characterized by elaborate costumes, vibrant makeup, and
    energetic performances that depict mythological stories and local legends.""",

]

metadata = [
    {"topic": "Rasgulla", "type": "food"},
    {"topic": "Tamil Nadu", "type": "geography"},
    {"topic": "Sambar Deer", "type": "wildlife"},
    {"topic": "Aghori", "type": "culture"},
    {"topic": "Chandragupta Maurya", "type": "history"},
    {"topic": "ISRO", "type": "science"},
    {"topic": "Banyan Tree", "type": "nature"},
    {"topic": "Theyyam", "type": "arts"}
]

documents = [
    Document(
        page_content=text, metadata=metadata[i]
    ) for i, text in enumerate(texts)
]

vector_store = Chroma.from_documents(
    documents=documents,
    embedding=embedding_model
)

result =  vector_store.similarity_search_with_score("Martial arts of India", k=3)
for doc, score in result:
    print(f"Score: {score:.4f}, Document: {doc.page_content}, Metadata: {doc.metadata}")