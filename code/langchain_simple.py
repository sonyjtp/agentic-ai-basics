from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from llms import get_model

load_dotenv()
prompt = PromptTemplate(
    input_variables = ['topic'],
    template = "Generate five insightful questions about {topic}."
)
llm = get_model("llama-3.1-8b-instant")

question_chain = prompt | llm
response = question_chain.invoke({'topic': 'artificial intelligence in healthcare'})
print(response.content)