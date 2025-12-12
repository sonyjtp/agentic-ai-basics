import os

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_groq import ChatGroq

load_dotenv()
llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    temperature=0.0,
    api_key=os.getenv("GROQ_API_KEY")
)

publication_content = """
Title: One Model, Five Superpowers: The Versatility of Variational Autoencoders
TL;DR: Variational Autoencoders (VAEs) are powerful generative models capable of learning complex data distributions. 
This paper explores five key applications of VAEs: image generation, anomaly detection, data compression, 
semi-supervised learning, and representation learning. Through extensive experiments, we demonstrate the 
effectiveness of VAEs in these domains and discuss their potential for future research. 

Introduction
Variational Autoencoders (VAEs) have emerged as a prominent class of generative models in the field of machine learning.
They use a probabilistic approach to model the underlying distribution of data, enabling them to generate new samples 
that resemble the training data. VAEs consist of an encoder that maps input data to a latent space and a decoder that 
reconstructs data from the latent representation. This paper delves into five significant applications of VAEs,
highlighting their versatility and effectiveness. In case of anomaly detection, 
VAEs can be trained on normal data to learn its distribution. During inference, data points that deviate significantly 
from the learned distribution can be flagged as anomalies. The reconstruction error or the likelihood of the data point
under the learned distribution can be used as an anomaly score. This approach is particularly useful in scenarios such as 
fraud detection, network security, and fault detection in industrial systems. By leveraging the generative capabilities 
of VAEs, we can effectively identify anomalous patterns in data, enhancing the reliability and security of various 
applications.
"""

conversation = [
    SystemMessage(
        content=f"""
                You are an expert at creating insightful questions based on academic publications.
                Base your questions on the following publication:\n{publication_content}
            """
    )
]

conversation.append(
    HumanMessage(
        content="What are  variational autoencoders used for?"
    )
)

response1 = llm.invoke(conversation)
print("🤖 AI Response to Question 1:")
print(response1.content)
print("\n" + "="*50 + "\n")
conversation.append(
    AIMessage(
        content=response1.content
    )
)
conversation.append(
   HumanMessage(
       content = """
        How does it work in case of anomaly detection?
       """
   )
)
response2 = llm.invoke(conversation)
print("🤖 AI Response to Question 2:")
print(response2.content)