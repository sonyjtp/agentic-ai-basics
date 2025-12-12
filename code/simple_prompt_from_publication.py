import os

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq

load_dotenv()
llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    temperature=0.0,
    api_key=os.getenv("GROQ_API_KEY")
)

publication = """
    Title: One Model, Five Superpowers: The Versatility of Variational Autoencoders
    TL;DR: Variational Autoencoders (VAEs) are powerful generative models capable of learning complex data distributions. 
    This paper explores five key applications of VAEs: image generation, anomaly detection, data compression, 
    semi-supervised learning, and representation learning. Through extensive experiments, we demonstrate the 
    effectiveness of VAEs in these domains and discuss their potential for future research.
"""

messages = [
    SystemMessage('You are an expert at creating insightful questions based on academic publications.'),
    HumanMessage(f"""Based on the following publication, generate five insightful questions that 
        could be explored further:\n{publication}"""
    )
]

response = llm.invoke(messages)
print("content:", response.content)