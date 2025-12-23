import os

from dotenv import load_dotenv
from langchain_classic.chains.conversation.base import ConversationChain
from langchain_classic.memory import ConversationBufferMemory, ConversationBufferWindowMemory, ConversationSummaryMemory
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.7,
    api_key=os.getenv("GROQ_API_KEY"),
)

# 1. Stuff Everything In Memory Strategy
buffer_memory = ConversationBufferMemory(
    return_messages=True
)

# 2. Trim Older Messages Memory Strategy
buffer_window_memory = ConversationBufferWindowMemory(
    k=3,
    return_messages=True
)

# 3. Summarize or Refine History Memory Strategy
summarized_memory = ConversationSummaryMemory(
    llm=llm,
    return_messages=True
)

summarized_window_memory = ConversationBufferWindowMemory(
    k=3,
    return_messages=True
)

def build_conversation(memory):
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=True
    )
    return conversation.invoke(
        {"input": "Hello! Can you tell me a joke about computers?"}
    )




print("""=== Using ConversationBufferMemory (Stuff Everything In Memory) ===""")
print("Response:", build_conversation(buffer_memory))
print("""=== Using ConversationBufferWindowMemory (Trim Older Messages) ===""")
print("Response:", build_conversation(buffer_window_memory))
print("""=== Using ConversationSummaryMemory ===""")
print("Response:", build_conversation(summarized_memory))