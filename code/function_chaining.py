from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

load_dotenv()
# 1. Define the model
model = ChatGroq(
            model_name="llama-3.1-8b-instant",
            temperature=0.0
)
# 2. Define the prompts
question_prompt = PromptTemplate(
    input_variables=['topic'],
    template='Generate five insightful questions about {topic}.'
)
answer_prompt = PromptTemplate(
    input_variables=['questions'],
    template='Generate answers for the following questions:\n{questions}'
)
# 3. Create the question and answer chains using the pipe operator
output_parser = StrOutputParser()
question_chain = question_prompt | model |  output_parser
answer_chain = answer_prompt | model | output_parser

def build_answer_input_from_questions(questions: str) -> dict:
    return {'questions': questions}

# 4. Combine the chains into a full chain
full_chain = question_chain | build_answer_input_from_questions |  answer_chain
# 5. Invoke the full chain with a specific topic, and print the response
response = full_chain.invoke({'topic': 'Kerala is the best place to visit in India'})
print('ANSWERS:\n')
print(response)
