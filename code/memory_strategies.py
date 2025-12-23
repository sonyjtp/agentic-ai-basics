import os

import tiktoken
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_groq import ChatGroq

from constants import PUBLICATION_CONTENT_HEADER, PUBLICATION_CONTENT_FOOTER
from prompt_builder import load_system_prompts
from paths import APP_CONFIG_FPATH, DATA_DIR, OUTPUTS_DIR
from file_utils import load_yaml, save_text_to_file
from str_utils import capitalize_first_char


def save_strategy_results(strategy: str, qa_pairs: list[dict], final_prompt: str, final_response: str, token_progression: list, questions: list) -> None:
    """Saves the results of a memory strategy run to output files."""
    content = [f"# {strategy.upper()} STRATEGY RESULTS", "=" * 60, ""]

    # Strategy description
    descriptions = {
        "stuffing": "Keeps ALL previous messages in conversation history.",
        "trimming": "Keeps only the most recent N messages in conversation history.",
        "summarization": "Summarizes older messages and keeps recent messages for context."
    }

    content.append("## Strategy Description")
    content.append(descriptions.get(strategy, "Unknown strategy"))
    content.append("")

    # Token progression
    content.append("## Token Usage Progression")
    content.append("| Question | Prompt Tokens | Response Tokens | Total |")
    content.append("|----------|---------------|-----------------|-------|")
    for token_data in token_progression:
        content.append(f"| {token_data['question_num']} | {token_data['prompt_tokens']:,} | {token_data['response_tokens']:,} | {token_data['total_tokens']:,} |")
    content.append("")

    # Final prompt
    if questions:
        content.append("## Complete Final Prompt for Last Question")
        content.append(f"**Last Question:** '{questions[-1]}'")
        content.append("")
        content.append("```")
        content.append(final_prompt)
        content.append("```")
        content.append("")
        content.append("**Final Response:**")
        content.append("```")
        content.append(final_response)
        content.append("```")
        content.append("")

    # All Q&A pairs
    content.append("## All Q&A Pairs")
    content.append("")
    for i, qa in enumerate(qa_pairs, 1):
        content.append(f"### Question {i}")
        content.append(f"**User:** {qa['question']}")
        content.append("")
        content.append(f"**Assistant:** {qa['response']}")
        content.append("")
        content.append("-" * 40)
        content.append("")

    # Save file
    filename = f"lesson3a_strategy_{strategy}_results.md"
    save_text_to_file(
        "\n".join(content),
        os.path.join(OUTPUTS_DIR, filename),
        header=f"{strategy.title()} Strategy Results"
    )
    print(f"    ✓ Results saved to {filename}")

def count_tokens(text: str) -> int:
    """Estimates the number of tokens in a given text. If the model encoding is not found, falls back to a
    word-based estimate. """
    try:
        encoding = tiktoken.encoding_for_model(llm.model_name)
        return len(encoding.encode(text))
    except (KeyError, ValueError):
        return int(len(text.split()) * 1.3) # Fallback estimate

def remove_publication(system_content)-> str:
    """Removes publication content from the system message. If markers are not found, returns original content.
    Else, replaces publication content with a placeholder."""
    start_marker = PUBLICATION_CONTENT_HEADER
    end_marker = PUBLICATION_CONTENT_FOOTER
    start_idx = system_content.find(start_marker)
    end_idx = system_content.find(start_marker)
    if start_idx != -1 and end_idx != -1:
        before_pub = system_content[:start_idx]
        after_pub = system_content[end_idx + len(end_marker):]
        return before_pub + "[PUBLICATION CONTENT OMITTED FOR READABILITY]" + after_pub
    print("⚠️ Publication content markers not found in system message.")
    return system_content

def messages_to_string(messages: list, include_publication: bool = False) -> str:
    """Converts a list of messages to a single string for token counting."""
    content = ""
    question_count = 0
    for idx, message in enumerate(messages):
        if isinstance(message, SystemMessage):
            system_content = message.content
            if not include_publication and PUBLICATION_CONTENT_HEADER in system_content:
                system_content = remove_publication(system_content)
            content += f"SYSTEM: {system_content}\n\n"
        elif isinstance(message, HumanMessage):
            question_count += 1
            if idx > 0:
                content += "=" * 80 + "\n"
            content += f"Q{question_count}: {message.content}\n"
        elif isinstance(message, AIMessage):
            content += f"AI: {message.content}\n\n"
    return content

def apply_stuffing_strategy(conversation: list) -> list:
    """Strategy 1: Keep all messages."""
    print("Applying stuffing strategy: keeping all messages.")
    return  system_msg + conversation


def apply_trimming_strategy(conversation: list, window_size: int) -> list:
    """Strategy 2: Keep only recent N messages."""
    if len(conversation) <= window_size:
        return system_msg + conversation
    else:
        return system_msg + conversation[-window_size:]


def apply_summarization_strategy(conversation: list, max_tokens: int) -> list:
    """Strategy 3: Summarize old messages, keep recent ones."""

    # If conversation is short, no need to summarize
    current_tokens = count_tokens(messages_to_string(system_msg + conversation))
    if current_tokens <= max_tokens:
        return system_msg + conversation

    # Keep last 6 messages and summarize the rest
    recent_messages = conversation[-6:] if len(conversation) > 6 else conversation
    older_messages = conversation[:-6] if len(conversation) > 6 else []

    if not older_messages:
        return system_msg + conversation

    # Create summary
    try:
        older_text = ""
        for msg in older_messages:
            if isinstance(msg, HumanMessage):
                older_text += f"Q: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                older_text += f"AI: {msg.content}\n"

        summary_prompt = f"""Provide a concise summary of this conversation history:{older_text}

Focus on main topics and key information. Keep under 200 words."""

        summary_response = llm.invoke([HumanMessage(content=summary_prompt)])
        summary_message = SystemMessage(content=f"Summary of earlier conversation: {summary_response.content}")

        return system_msg + [summary_message] + recent_messages

    except Exception as e:
        print(f"  ⚠️ Summarization failed, using trimming: {e}")
        return apply_trimming_strategy(conversation, system_prompts, 8)

def apply_strategy(strategy, conversation_history) -> list:
    """Applies the specified memory strategy to the conversation history."""
    curr = []
    match strategy:
        case "stuffing":
            curr = apply_stuffing_strategy(conversation=conversation_history[:-1])
        case "trimming":
            curr = apply_trimming_strategy(
                conversation=conversation_history[:-1],
                window_size=memory_cfg.get("trimming_window_size", 8)
            )
        case "summarization":
            curr = apply_summarization_strategy(
                conversation=conversation_history[:-1],
                max_tokens=memory_cfg.get("summarization_max_tokens", 1000)
            )
        case _:
            raise ValueError(f"Unknown strategy: {strategy}")
    return curr

def run_conversation_using_memory_strategy(strategy: str, user_questions: list[str]) -> dict:
    """Runs a conversation using the specified memory strategy and user questions."""
    max_tokens_before_summarization = memory_cfg.get("summarization_max_tokens", 1000)
    print(f"\n🔧 Running {strategy.upper()} strategy on {len(user_questions)} questions")

    # Track conversation history (without system prompt)
    conversation_history = []
    qa_pairs = []
    token_progression = []
    for idx, question in enumerate(user_questions, start=1):
        print(f"\n❓ Question {idx}/{len(user_questions)}: {capitalize_first_char(question)}?")
        # Add question to conversation history and then apply strategy
        conversation_history.append(HumanMessage(content=question))
        currrent_messages = apply_strategy(strategy, conversation_history)
        # Add current question to current messages
        currrent_messages.append(HumanMessage(content=question))

        # Count the tokens before invoking LLM
        prompt_tokens = count_tokens(messages_to_string(currrent_messages))
        try:
            response = llm.invoke(currrent_messages)
            response_tokens = count_tokens(response.content)
            total_tokens = response_tokens + prompt_tokens
            print(f" Answer: {response.content}")
            conversation_history.append(AIMessage(content=response.content))
            qa_pairs.append({
                "question": question,
                "answer": response.content
            })
            token_progression.append({
                'question_num': idx,
                'prompt_tokens': prompt_tokens,
                'response_tokens': response_tokens,
                'total_tokens': total_tokens
            })
            print(f"  🪙 Token count for this interaction: {total_tokens}")
        except Exception as e:
            print(f"  ❌ Error at question {idx}: {e}")
            break
        # Generate final prompt for the last question
        final_messages = []
    if questions:
        final_messages = apply_strategy(strategy, conversation_history)
        final_messages.append(HumanMessage(content=questions[-1]))
        final_prompt = messages_to_string(final_messages, include_publication=False)
        final_response = conversation_history[-1].content if conversation_history else "No response"
    else:
        final_prompt = ""
        final_response = ""
    save_strategy_results(strategy, qa_pairs, final_prompt, final_response, token_progression, user_questions)

def run_single_strategy():
    """Prompts the user to select a memory strategy and sets the strategy variable."""
    choice = input("Select a memory strategy by number (default = 1): ").strip()
    strategy = strategy_map.get(str(choice), "trimming")
    print(f"Selected strategy:{strategy.upper()}")

    # Ask how many questions
    num_questions = input(f"How many questions to process? (max = {len(questions)}, default = 10): ").strip()
    try:
        """ num_questions should be an integer between 1 and len(questions) """
        num_questions = min(int(num_questions) if num_questions else 10, len(questions))
    except ValueError:
        num_questions = 10
    print(f"Number of questions:{num_questions}")
    selected_questions = questions[:num_questions]
    # run strategy
    stats = run_conversation_using_memory_strategy(
        strategy=strategy,
        user_questions=selected_questions
    )

def load_questions() -> list[str]:
    """Loads user questions from a YAML configuration file."""
    questions_cfg = load_yaml(os.path.join(DATA_DIR, "yzN0OCQT7hUS_questions.yaml"))
    user_questions = questions_cfg.get("questions", [])
    print("✓ User questions loaded.")
    return user_questions


def bootstrap() -> tuple[dict, ChatGroq, str, list[str], list[str]]:
    """Bootstraps the LLM and system prompts for the AI assistant application.
    Returns:
        tuple: A tuple containing the initialized ChatGroq LLM instance and the system prompt string.
    """
    load_dotenv()
    app_cfg = load_yaml(APP_CONFIG_FPATH)
    print("✓ Application configuration loaded.")
    llm_client = ChatGroq(
        model=app_cfg.get("llm", "llama-3.1-8b-instant"),
        temperature=0.7,
        api_key=os.getenv("GROQ_API_KEY"),
    )
    print("✓ LLM client initialized.")
    sys_prompts = load_system_prompts(
        key="ai_assistant_system_prompt_advanced",
        publication_external_id="yzN0OCQT7hUS"
    )
    print("✓ System prompts loaded.")
    memory_strategies = ["stuffing", "trimming", "summarization"]
    print("✓ Memory strategies defined.")
    return app_cfg, llm_client, sys_prompts, memory_strategies, load_questions()


if __name__ == "__main__":
    print("Bootstrapping App Config, LLM and system prompts...")
    app_config, llm, system_prompts, strategies, questions = bootstrap()
    memory_cfg = app_config.get("memory_strategies", {})
    system_msg = [SystemMessage(content=system_prompts)]
    print("Added system prompts to system message.")
    strategy_map: dict[str, str] = {}
    print("✓ Bootstrap complete.\n")
    print("Available memory strategies:")
    for i, stgy in enumerate(strategies, start=1):
        strategy_map[str(i)] = stgy
        print(f"{str(i)}: {stgy}")
    run_single_strategy()





