from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import os

load_dotenv()
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.7,
    api_key=os.getenv("GROQ_API_KEY"),
)

publication = """
    Title: One Model, Five Superpowers: The Versatility of Variational Autoencoders
    TL;DR: Variational Autoencoders (VAEs) are powerful generative models capable of learning complex data distributions.
    This paper explores five key applications of VAEs: image generation, anomaly detection, data compression,
    semi-supervised learning, and representation learning. Through extensive experiments, we demonstrate the
    effectiveness of VAEs in these domains and discuss their potential for future research.
"""

conversation: list[HumanMessage|SystemMessage|AIMessage] = [SystemMessage
    (
    f"""
            You are a friendly and knowledgeable virtual assistant. Answer the user's questions based on the provided 
            publication: {publication} 
        """
), HumanMessage(
    """
        Can you summarize the key applications of Variational Autoencoders (VAEs) mentioned in the publication?
    """
)]

response1 = llm.invoke(conversation)
print("AI Response 1:")
print(response1.content)
conversation.append(AIMessage(content=response1.content))
conversation.append(HumanMessage(
        content="""
            That's interesting! Could you explain how VAEs are used in anomaly detection?
        """
    )
)
response2 = llm.invoke(conversation)
print("\nAI Response 2:")
print(response2.content)


