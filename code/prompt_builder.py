from str_utils import add_prefix
from constants import PUBLICATION_CONTENT_FOOTER
from file_utils import load_yaml, load_publication

from paths import PROMPT_CONFIG_FPATH
from constants import PUBLICATION_CONTENT_HEADER


def lowercase_first_char(text: str) -> str:
    """Lowercases the first character of a string.

    Args:
        text: Input string.

    Returns:
        The input string with the first character lowercased.
    """
    return text[0].lower() + text[1:] if text else text

def format_prompt_section(lead_in: str, value: str | list[str]) -> str:
    """Formats a prompt section by joining a lead-in with content.

    Args:
        lead_in: Introduction sentence for the section.
        value: Section content, as a string or list of strings.

    Returns:
        A formatted string with the lead-in followed by the content.
    """
    if isinstance(value, list):
        formatted_value = "\n".join(f"- {item}" for item in value)
    else:
        formatted_value = value
    return f"{lead_in}\n{formatted_value}"


def build_system_prompt_from_config(publication_external_id = "yzN0OCQT7hUS") -> str:
    """ Build system prompt from configuration.
    System prompt includes scope (publication), role/personality, constraints, style, format, and goal.
    """
    print("Loading publication content...")
    publication_content = load_publication(publication_external_id)
    prompt_configs = load_yaml(PROMPT_CONFIG_FPATH)
    system_prompt_config_name = "ai_assistant_system_prompt_advanced"
    system_prompt_config = prompt_configs.get(system_prompt_config_name)
    if not system_prompt_config:
        raise ValueError(f"System prompt config '{system_prompt_config_name}' not found")
    prompt_parts = []

    # Role is required for system prompts
    if role := system_prompt_config.get("role", "You are a helpful AI assistant."):
        prompt_parts.append(f"You are {lowercase_first_char(role.strip())}.")
    else:
        raise ValueError("Missing required field: 'role'")

    # Add behavioral constraints
    if constraints := system_prompt_config.get("output_constraints"):
        prompt_parts.append(
            format_prompt_section(
                lead_in="Follow these important guidelines:",
                value=constraints
            )
        )

    # Add style and tone guidelines
    if tone := system_prompt_config.get("style_or_tone"):
        prompt_parts.append(
            format_prompt_section(
                "Communication style:", tone
            )
        )

    # Add output format requirements
    if format_ := system_prompt_config.get("output_format"):
        prompt_parts.append(
            format_prompt_section("Response formatting:", format_)
        )

    # Add goal if specified
    if goal := system_prompt_config.get("goal"):
        prompt_parts.append(
            format_prompt_section("Overall goal:", goal)
        )

    # Include publication content, if provided
    if publication_content:
        prompt_parts.append(
            "Base your responses on this publication content:\n\n"
            f"{PUBLICATION_CONTENT_HEADER}\n"
            f"{publication_content.strip()}\n"
            f"{PUBLICATION_CONTENT_FOOTER}"
        )

    return "\n\n".join(prompt_parts)


def load_system_prompts(
        key: str,
        publication_external_id: str
) -> str:
    """Loads system prompt configuration from YAML file and builds the system prompt string.
    The system prompt includes role, style/tone, output constraints, output format, and goal.
    Args:
        key (str): The key to identify the specific system prompt configuration.
    Returns:
        str: The constructed system prompt.
    Raises:
        ValueError: If the specified key is not found in the prompt configuration.
        :param key:
        :param publication_external_id:
    """
    print("loading system prompts...")
    prompt_config = load_yaml(PROMPT_CONFIG_FPATH)
    system_prompt_config = prompt_config.get(key)
    if not system_prompt_config:
        raise ValueError(f"System prompt config for '{key}' not found")
    system_prompts = []
    # Add role to the system prompt
    if role := system_prompt_config.get("role", "helpful AI assistant."):
        system_prompts.append(
            add_prefix(
                lead_in="You are a ",
                append_value=f"{role.strip().lower()}.\n"
            )
        )
        print("✓ Added role to system prompt.")
    if tone := system_prompt_config.get("style_or_tone"):
        system_prompts.append(
            add_prefix(
                lead_in="Adopt the following style or tone:",
                append_value=tone
            )
        )
        print("✓ Added style/tone to system prompt.")

    if constraints := system_prompt_config.get("output_constraints"):
        system_prompts.append(
            add_prefix(
                lead_in="Follow these output constraints:",
                append_value=constraints
            )
        )
        print("✓ Added output constraints to system prompt.")

    if format_ := system_prompt_config.get("output_format"):
        system_prompts.append(
            add_prefix(
                lead_in="Use the following output format:",
                append_value=format_
            )
        )
        print("✓ Added output format to system prompt.")
    if goal := system_prompt_config.get("goal"):
        system_prompts.append(
            add_prefix(
                lead_in="Keep in mind the overall goal:",
                append_value=goal
            )
        )
        print("✓ Added goal to system prompt.")
    if publication_external_id is not None:
        if publication := load_publication(publication_external_id):
            system_prompts.append(
               "Base your responses on this publication content:\n\n" 
                f"{PUBLICATION_CONTENT_HEADER}\n"
                f"{publication.strip()}\n"
                f"{PUBLICATION_CONTENT_FOOTER}"
            )
            print("✓ Added publication content to system prompt.")
        else:
            raise ValueError(f"Publication for id {publication_external_id} not found")
    print("✓ System prompt construction complete.")
    return "\n".join(system_prompts)