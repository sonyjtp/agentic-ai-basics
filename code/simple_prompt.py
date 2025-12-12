import os

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq

load_dotenv()
llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    temperature=0.7,
    api_key=os.getenv("GROQ_API_KEY"),
)

messages = [
    SystemMessage(
        content='You are a helpful assistant that provides concise and accurate information.'
    ),
    HumanMessage(
        content='What are the key benefits of using LangChain for building applications with large language models?'
    )
]
response = llm.invoke(messages)
print("content:", response.content)
print("response_metadata:", response.response_metadata)
print("id:", response.id)
