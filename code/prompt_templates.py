from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langchain_core.prompts import PromptTemplate

from llms import get_model

load_dotenv()

llm = get_model("llama-3.1-8b-instant")
countries = input("Enter country names separated by commas: ").split(",")

message = [SystemMessage(content="You are a helpful assistant that provides information about countries.")]

template = PromptTemplate(
    input_variables = ['country'],
    template = "What is the capital, 3 major languages, and 3 main cities of {country}?"
)

for country in countries:
    formatted_prompt = template.format(country=country)
    print(f"Country: {country.strip()}")
    response = llm.invoke(formatted_prompt)
    print(response.content)
    print("\n" + "="*50 + "\n")


