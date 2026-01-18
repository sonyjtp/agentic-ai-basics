# Gen AI Examples

This project contains a collection of examples demonstrating generative AI concepts and implementations, organized by increasing complexity.

## How to Learn

Work through the examples in the order listed below. Each example builds on concepts introduced in previous ones. Start with the fundamentals and progress toward more advanced agentic patterns.

## Learning Examples

### Beginner Level

1. **simple_prompt.py** - Learn how to craft effective prompts with proper context and input structure.
2. **simple_prompt_from_publication.py** - Apply basic prompting to extract information from published content.
3. **multiturn_conversation.py** - Build multi-turn conversation systems with context management.
4. **prompt_templates.py** - Create reusable prompt templates with variable substitution for different use cases.

### Intermediate Level

1. **structured_output.py** - Generate and validate structured output using Pydantic models.
2. **function_chaining.py** - Chain multiple function calls to solve complex problems step by step.
3. **simple_vector_store.py** - Build and manage vector databases for semantic search and retrieval using Chroma.

### Advanced Level

1. **memory_strategies.py** - Apply memory strategies in a practical multi-turn conversation context.
2. **rag_implementation.py** - Implement Retrieval-Augmented Generation (RAG) for knowledge-enhanced responses.

## Helper Modules & Configuration Files

These utility modules and configuration files support the learning examples:

- **llms.py** - LLM integration and model initialization utilities.
- **prompt_builder.py** - Helper functions for constructing and validating prompts.
- **memory_strategies.py** - Core implementations of memory management strategies (stuffing, trimming, summarization).
- **chunking.py** - Text chunking utilities for preparing documents for embeddings.
- **config/config.yaml** - Main configuration file for project settings.
- **config/prompt_config.yaml** - Prompt templates and configurations.

## Project Structure

```
├── main.py                           # Entry point for the project
├── code/
│   ├── *.py                         # Individual example modules
│   ├── config/
│   │   ├── config.yaml              # Main configuration file
│   │   └── prompt_config.yaml       # Prompt-specific configurations
│   └── output/                      # Generated outputs from examples
├── data/                            # Sample data and publications
└── outputs/                         # Results and artifacts
```

## Getting Started

1. Open the examples in order of complexity listed above.
2. Read the comments and docstrings in each file.
3. Modify and experiment with the code to deepen understanding.
4. Test your changes using PyCharm's built-in run and debug tools.
5. Check the `outputs/` folder to see results from each example.

## Key Concepts Covered

- **Prompt Engineering**: Crafting effective prompts with system messages and context
- **Memory Strategies**: Managing conversation history and context windows
- **Structured Output**: Using Pydantic for validated, structured responses
- **RAG (Retrieval-Augmented Generation)**: Combining document retrieval with generation
- **Vector Databases**: Storing and retrieving semantic embeddings
- **Function Chaining**: Orchestrating multiple AI operations sequentially
- **Agentic Patterns**: Building autonomous systems with AI agents
