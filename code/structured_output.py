from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from file_utils import load_publication
from llms import get_model


class Entity(BaseModel):
    type: str = Field(description="The type of the entity. Either 'model' or 'task'")
    name: str = Field(description="The name of the entity")

class Entities(BaseModel):
    entities: list[Entity] = Field(
        description="The entities mentioned in the publication"
    )

def no_structured_output() -> str:
    prompt_text = f"""
    Provide a list of entities mentioned in the publication. An entity is either a model or a task.
    <publication>
    {publication_content}
    </publication>
    """
    return prompt_text

def with_prompting_to_structure_output() -> str:
    prompt_text = f"""
    Provide a list of entities mentioned in the publication. An entity is either a model or a task.
    Format the output as JSON with the following structure:
    {{
      "entities": [
        {{
          "type": "model" or "task",
          "name": "name of the entity"
        }},
        ...
      ]
    }}
    <publication>
    {publication_content}
    </publication>
    """
    return prompt_text

def with_output_parser():
    output_parser = PydanticOutputParser(pydantic_object=Entities)
    format_instructions = output_parser.get_format_instructions()
    prompt_text = f"""
        Provide a list of entities mentioned in the publication. An entity is either a model or a task.
        Format the output as JSON matching the Entities Pydantic model.
        <publication>
        {publication_content}
        </publication>
        {format_instructions}
    """

    return prompt_text


if __name__ == "__main__":
    llm = get_model("openai/gpt-oss-120b")
    publication_content = load_publication()
    unstructured_output = llm.invoke(no_structured_output())
    print(f"===== Unstructured Output =====\n{unstructured_output.content}\n\n")
    # structured_output_using_prompts = llm.invoke(with_prompting_to_structure_output())
    # print(f"===== Structured Output Using Prompts =====\n{structured_output_using_prompts.content}\n\n")
    # structured_output_using_parsers= llm.invoke(with_output_parser())
    # print(f"===== Structured Output Using Parsers =====\n{structured_output_using_parsers.content}\n\n")
    llm_for_model_native_structured_output = get_model("gpt-4o-mini")
    structured_output_using_model_native_parsers = llm_for_model_native_structured_output.invoke(with_output_parser())
    print(f"===== Structured Output Using Model Native Parsers =====\n{structured_output_using_model_native_parsers.content}")



