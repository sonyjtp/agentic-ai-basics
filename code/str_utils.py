from typing import Union


def add_prefix(lead_in: str, append_value: str | list[str]) -> str:
    """Add a prefix to a string or each item in a list.

    Args:
        lead_in (str): The prefix to add.
        append_value (str | list): The string or list of strings to which the prefix will be added.

    Returns:
        str: The resulting string with prefixes added.
    """
    if isinstance(append_value, list):
        formatted_value = "\n".join(f"- {item}" for item in append_value)
    else:
        formatted_value = append_value
    return f"{lead_in}\n{formatted_value}"

def capitalize_first_char(text: str) -> str:
    """Capitalizes the first character of a string.

    Args:
        text (str): Input string.

    Returns:
        str: The input string with the first character capitalized.
    """
    return text[0].upper() + text[1:] if text else text