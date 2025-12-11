from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# 1. Choose a model that turns text into embeddings
embeddings = HuggingFaceEmbeddings(model_name = 'all-MiniLM-L6-v2')

# 2. Prepare the documents with metadata
texts = [' Kerala is located in the southwestern region of India. ',
         ' Kerala is known for its beautiful backwaters and beaches. ',
         ' Kerala has a rich cultural heritage with traditional dance forms like Kathakali and Mohini]yattam. ',
         ' Kerala is famous for its spices, including black pepper, cardamom, and cinnamon. ',
         ' Kerala has a high literacy rate and is known for its quality education system. ',
         ' Kerala is a popular tourist destination, attracting visitors from around the world. ',
         ' Kerala has a diverse cuisine, with dishes like appam, puttu, and fish curry being popular. ',
         ' Kerala is home to several wildlife sanctuaries and national parks, including Periyar and] Wayanad. ',
         ' Kerala celebrates several festivals, including Onam, Vishu, and Thrissur Pooram. ',
         ' Kerala has a tropical climate, with heavy monsoon rains from June to September. '
        ]
metadata = [
    {"topic": "geography", "type": "fact"},
    {"topic": "tourism", "type": "fact"},
    {"topic": "culture", "type": "fact"},
    {"topic": "spices", "type": "fact"},
    {"topic": "education", "type": "fact"},
    {"topic": "tourism", "type": "fact"},
    {"topic": "cuisine", "type": "fact"},
    {"topic": "wildlife", "type": "fact"},
    {"topic": "festivals", "type": "fact"},
    {"topic": "climate", "type": "fact"}
]

# Python list comprehension to create Document objects
documents = [
    Document(
        page_content=text,
        metadata=metadata[i]
    )
    for i, text in enumerate(texts)
]
# 3. Create the Vector Store using Chroma
vector_store = Chroma.from_documents(documents)
# 4. Perform a similarity search
query = "How hot is it in Kerala?"
results = vector_store.similarity_search_with_score(query, k=3)
for doc, score in results:
    print(f"Score: {score:.4f}, Document: {doc.page_content}, Metadata: {doc.metadata}")