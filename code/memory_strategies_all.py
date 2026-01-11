import os

import tiktoken
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_groq import ChatGroq

from constants import PUBLICATION_CONTENT_FOOTER, PUBLICATION_CONTENT_HEADER
from paths import APP_CONFIG_FPATH, DATA_DIR, OUTPUTS_DIR
from prompt_builder import build_system_prompt_from_config
from code.file_utils import load_yaml, save_text_to_file



def save_comparison_summary(all_stats: list):
    """ Save comparison summary of all strategies to a markdown file. """
    content = []
    content.append("# Memory Strategy Comparison Summary\n")
    content.append("=" * 80 + "\n")
    content.append("| Strategy      | Questions Processed | Prompt Tokens | Response Tokens | Total Tokens |\n")
    content.append("|---------------|---------------------|---------------|-----------------|--------------|\n")
    for stat in all_stats:
        content.append(f"| {stat['strategy']:13} | "
                      f"{stat['questions_processed']:19} | "
                      f"{stat['total_prompt_tokens']:13} | "
                      f"{stat['total_response_tokens']:15} | "
                      f"{stat['total_tokens']:12} |\n")

    # Efficiency comparison
    if (len(all_stats) > 1):
        baseline = next((s for s in all_stats if s['strategy'] == 'stuffing'), all_stats[0])
        content.append("\n## Efficiency Comparison (vs Stuffing Strategy)\n")
        for stats in all_stats:
            if stats['strategy'] !=baseline['strategy']:
                prompt_savings = 100 * ((baseline['total_tokens'] - stats['total_tokens']) / baseline['total_tokens'])
                content.append(f"- **{stats['strategy'].title()}** vs {baseline['strategy'].title()}: "
                               f"{prompt_savings:.1f}% token savings\n")

    # Analysis
    content.append("\n## Analysis\n")
    content.append(""" When to use each strategy:""")
    content.append("""
        - **Stuffing**: Best when full context is critical and token limits are not a concern.
        - **Trimming**: Useful for conversations where recent context is most relevant, and token efficiency is needed.
        - **Summarization**: Ideal for long conversations where maintaining overall context is important, balancing 
        detail and token usage.
    """)
    # Save to file
    save_text_to_file("\n".join(content), os.path.join(OUTPUTS_DIR, "comparison_summary.md"),
                        header="Memory Strategy Comparison Summary"
    )
    print("✓ Comparison statistics saved to comparison_summary.md")


def save_strategy_results(
    strategy: str,
    qa_pairs: list,
    final_prompt: str,
    final_response: str,
    token_progression: list,
    questions: list
):
    content = []
    content.append(f"# {strategy.upper()} Strategy Results\n")
    content.append("=" * 80 + "\n")

    # Strategy description
    strategy_descriptions = {
        "stuffing": "Keeps all previous messages in the conversation history.",
        "trimming": "Keeps only the most recent N messages in the conversation history.",
        "summarization": "Summarizes older messages to retain context while keeping recent messages intact."
    }
    content.append("## Strategy Description\n")
    content.append(strategy_descriptions.get(strategy, "Unknown strategy") + "\n")

    # Token progression
    content.append("## Token Usage Progression\n")
    content.append("| Question Index | Prompt Tokens | Response Tokens | Total Tokens |\n")
    content.append("|----------------|---------------|-----------------|--------------|\n")
    for token_info in token_progression:
        content.append(
            f"| {token_info['question_index']} | "
           f"{token_info['prompt_tokens']} | "
           f"{token_info['response_tokens']} | "
           f"{token_info['total_tokens']} |"
           f"\n"
       )

    # Final prompt
    if questions:
        content.append("\n## Final Prompt for last Question\n")
        content.append(f"**Last Question:** '{questions[-1]}'\n")
        content.append("```\n")
        content.append(final_prompt)
        content.append("\n```\n")
        # Final response
        content.append("## Final Response\n")
        content.append("```\n")
        content.append(final_response)
        content.append("\n```\n")

    # All QA pairs
    content.append("## All Question-Answer Pairs\n")
    for i, qa in enumerate(qa_pairs, start=1):
        content.append(f"### Q{i}: {qa['question']}\n")
        content.append(f"**A{i}:** {qa['answer']}\n\n")

    # Save file
    filename = f"strategy_{strategy}_results.md"
    save_text_to_file("\n".join(content), os.path.join(OUTPUTS_DIR, filename))
    print(f"    ✓ Results saved to {filename}")


def count_tokens(text: str, model: str = "gpt-3.5-turbo"):
    """Count tokens using tiktoken with fallback."""
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except:
        # Fallback estimation
        return int(len(text.split()) * 1.3)

def messages_to_string(messages: list, include_publication: bool = False) -> str:
    """Convert list of messages to a single readable string"""
    content = ""
    user_question_count = 0
    for i, msg in enumerate(messages):
        if isinstance(msg, SystemMessage):
            system_content = msg.content
            # Remove publication content if not needed
            if not include_publication and PUBLICATION_CONTENT_HEADER in system_content:
                system_content = remove_publication(system_content)
            content += f"SYSTEM: {system_content}\n"
        elif isinstance(msg, HumanMessage):
            user_question_count += 1
            # Add separator before user messages (except if it's the first message)
            if i > 0:
                content += "=" * 80 + "\n"
            content += f"USER: Q{user_question_count}\n: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            content += f"ASSISTANT: {msg.content}\n"
    return content


def remove_publication(system_content: str | list[str | dict]) -> str | list[str | dict]:
    start_idx = system_content.find(PUBLICATION_CONTENT_HEADER)
    end_idx = system_content.find(PUBLICATION_CONTENT_FOOTER) + len(PUBLICATION_CONTENT_FOOTER)
    if start_idx != -1 and end_idx != -1:
        # Remove the publication content section
        system_content = system_content[:start_idx] + system_content[end_idx:]
    return system_content


def apply_stuffing_strategy(conversation: list) -> list:
    """Strategy 1: Keep all messages."""
    return [SystemMessage(content=system_prompt)] + conversation


def apply_trimming_strategy(conversation: list, system_prompt: str, window_size: int = 8) -> list:
    """Strategy 2: Keep only recent N messages."""
    system_msg = [SystemMessage(content=system_prompt)]
    if len(conversation) <= window_size:
        return system_msg + conversation
    else:
        return system_msg + conversation[-window_size:]


def apply_summarization_strategy(conversation: list, system_prompt: str, llm, max_tokens: int = 1000) -> list:
    """Strategy 3: Summarize old messages, keep recent ones."""
    system_msg = [SystemMessage(content=system_prompt)]

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
                older_text += f"User: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                older_text += f"Assistant: {msg.content}\n"

        summary_prompt = f"""Provide a concise summary of this conversation history:{older_text}

Focus on main topics and key information. Keep under 200 words."""

        summary_response = llm.invoke([HumanMessage(content=summary_prompt)])
        summary_message = SystemMessage(content=f"Summary of earlier conversation: {summary_response.content}")

        return system_msg + [summary_message] + recent_messages

    except Exception as e:
        print(f"  ⚠️ Summarization failed, using trimming: {e}")
        return apply_trimming_strategy(conversation, system_prompt, 8)


def run_memory_strategy_conversation(
    strategy: str,
    questions: list
):
    """ Run a memory strategy conversation. """

    # Get memory config
    memory_config = app_config.get("memory_strategies", {}).get(strategy, {})
    window_size = memory_config.get("window_size", 8)
    max_tokens = memory_config.get("max_tokens", 1000)
    print(f"""Running strategy: {strategy.upper()} | window size {window_size} | 
        max tokens {max_tokens} | questions: {len(questions)}""")
    # Track conversation history (without system prompt)
    conversation_history = []
    qa_pairs = []
    token_progression = []

    # process each question
    for i, question in enumerate(questions, start=1):
        print(f"Processing question {i}/{len(questions)}: {question[: 50]}...'")

        # Add user message to history
        conversation_history.append(HumanMessage(content=question))

        # Apply memory strategy to build prompt
        if strategy == "stuffing":
            current_messages = apply_stuffing_strategy(conversation_history[:-1], system_prompt)
        elif strategy == "trimming":
            current_messages = apply_trimming_strategy(conversation_history[:-1], system_prompt, window_size)
        elif strategy == "summarization":
            current_messages = apply_summarization_strategy(
                conversation_history[:-1],
                system_prompt,
                llm,
                max_tokens
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Add the latest user question
        current_messages.append(HumanMessage(content=question))

        # Count tokens before invoking LLM
        prompt_tokens = count_tokens(messages_to_string(current_messages))

        # Invoke LLM
        try:
            response = llm.invoke(current_messages)
            response_tokens = count_tokens(messages_to_string((current_messages)))

            # Add AI response to history
            conversation_history.append(AIMessage(content=response.content))

            # Record QA pair and token usage
            qa_pairs.append({
                "question": question,
                "answer": response.content,
            })
            # Track token usage
            token_progression.append({
                "question_index": i,
                "prompt_tokens": prompt_tokens,
                "response_tokens": response_tokens,
                "total_tokens": prompt_tokens + response_tokens
            })

            if i % 5 == 0 or i == len(questions):
                print(f"  ✓ Completed {i}/{len(questions)} questions, current prompt: {prompt_tokens:,} tokens")
        except Exception as e:
            print(f"  ❌ Error processing question {i}: {e}")
            break

        # Generate final prompt for last question
        final_messages = []
        if questions:
            if strategy == "stuffing":
                final_messages = apply_stuffing_strategy(conversation_history[:-1], system_prompt)
            elif strategy == "trimming":
                final_messages = apply_trimming_strategy(conversation_history[:-1], system_prompt, window_size)
            elif strategy == "summarization":
                final_messages = apply_summarization_strategy(
                    conversation_history[:-1],
                    system_prompt,
                    llm,
                    max_tokens
                )

            final_messages.append(HumanMessage(content=questions[-1]))
            final_prompt_str = messages_to_string(final_messages, include_publication=False) # Exclude publication for readability
            final_response = conversation_history[-1].content if conversation_history else "No response"
        else:
            final_prompt_str = ""
            final_response = "No questions provided"

        # Save strategy-specific results
        save_strategy_results(
            strategy=strategy,
            qa_pairs=qa_pairs,
            final_prompt=final_prompt_str,
            final_response=final_response,
            token_progression=token_progression,
            questions=questions
        )

        # calculate totals
        total_prompt_tokens = sum(tp["prompt_tokens"] for tp in token_progression)
        total_response_tokens = sum(tp["response_tokens"] for tp in token_progression)
        total_tokens = total_prompt_tokens + total_response_tokens

        return {
            "strategy": strategy,
            "total_prompt_tokens": total_prompt_tokens,
            "total_response_tokens": total_response_tokens,
            "total_tokens": total_tokens,
            "questions_processed": len(token_progression),
            "token_progression": token_progression
        }




def run_single_strategy():
    """ Run a single memory strategy. """
    #Pick a strategy
    print("Available strategies:")
    for i, strategy in enumerate(strategies, start=1):
        print(f"{i}. {strategy}")
    strategy_choice = input(f"Select a strategy to run (1-{len(strategies)}), default=1: ").strip()
    strategy_map = {str(i): strategy for i, strategy in enumerate(strategies, start=1)}
    strategy = strategy_map.get(strategy_choice, "stuffing")

    # Ask how many questions to run
    print(f"Loaded {len(user_questions)} questions.")
    num_questions = input(f"How many questions to run? (max {len(user_questions)}), default=10): ").strip()
    try:
        num_questions = int(num_questions) if num_questions else 10
        num_questions = min(num_questions, len(user_questions))
    except ValueError:
        num_questions = 10
    selected_questions = user_questions[:num_questions]
    # Run the selected strategy on the selected questions
    stats = run_memory_strategy_conversation(
        strategy=strategy,
        questions=selected_questions,
        system_prompt=system_prompt,
        app_config=app_config
    )

def compare_all_strategies():
    """ Compare all memory strategies. """
    num_questions = input(f"How many questions to run? (max {len(user_questions)}), default=10): ").strip()
    try:
        num_questions = int(num_questions) if num_questions else 10
        num_questions = min(num_questions, len(user_questions))
    except ValueError:
        num_questions = 10
    selected_questions = user_questions[:num_questions]

    all_stats = []
    print(f"\n🏁 Running comparison with {len(selected_questions)} questions...")
    for strategy in strategies:
        stats = run_memory_strategy_conversation(
            strategy=strategy,
            questions=selected_questions,
            system_prompt=system_prompt,
            app_config=app_config
        )
        all_stats.append(stats)

    # Save comparison summary
    save_comparison_summary(all_stats)

    # Print comparison summary
    print("\nMemory Strategy Comparison Summary:")
    print("|---------------|---------------------|---------------|-----------------|--------------|")
    for stat in all_stats:
        print(f"{stat['strategy'].title():15} | {stat['total_tokens']:,} total tokens")




def main():
    """ Main entry point for memory strategies module. """
    print("Choose mode:")
    print("1. Run a single strategy")
    print("2. Compare all strategies")
    choice = input("Enter 1 or 2 (default=2): ").strip()
    if choice == "1":
        run_single_strategy()
    else:
        compare_all_strategies()


if __name__ == "__main__":
    print("Loading environment variables...")
    load_dotenv()
    print("Loading application configuration...") # for reasoning strategies and model selection
    app_config = load_yaml(APP_CONFIG_FPATH)
    model_name = app_config["llm"]
    llm = ChatGroq(
        model=model_name,
        temperature=0.0,
        api_key=os.getenv("GROQ_API_KEY")
    )
    system_prompt = build_system_prompt_from_config("yzN0OCQT7hUS")
    strategies = ["stuffing", "trimming", "summarization"]
    questions_config = load_yaml(os.path.join(DATA_DIR, "yzN0OCQT7hUS_questions.yaml"))
    user_questions = questions_config.get("questions", [])
    main()