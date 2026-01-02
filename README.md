# Glenn Bot - Local AI Collaboration Assistant

A fully local AI assistant that helps you solve problems, build products, and produce content based on your personal values, principles, and preferred frameworks.

## Features

- **100% Local**: Runs entirely on your machine using Ollama
- **Personal Knowledge Base**: Store your values, principles, preferences, and problem-solving frameworks
- **Multi-Agent System**: Specialized agents for frameworks, values alignment, and preferences
- **Rich Terminal Interface**: Beautiful CLI with syntax highlighting and formatted responses
- **Conversation History**: Save and load previous conversations
- **Vector Search**: Semantic search through your knowledge base using ChromaDB

## Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai/) installed and running
- Required Ollama models:
  ```bash
  ollama pull llama3:8b
  ollama pull nomic-embed-text
  ```

## Quick Start

1. Clone the repository and navigate to the project directory

2. Run the setup script:
   ```bash
   ./setup.sh
   ```

3. Activate the virtual environment:
   ```bash
   source venv/bin/activate
   ```

4. Run Glenn Bot:
   ```bash
   python run.py
   ```

## Customizing Your Knowledge Base

Add your personal content to the `knowledge/` directory:

### Values (`knowledge/values.json`)
Define your core values and principles that should guide the AI's responses.

### Frameworks (`knowledge/frameworks/*.json`)
Add problem-solving frameworks, methodologies, and structured approaches.

### Preferences (`knowledge/preferences.json`)
Specify your preferences for communication style, work approaches, and more.

## Usage

Once running, you can:
- Ask questions or describe problems you want to solve
- Request specific frameworks to be applied
- Get recommendations based on your values and preferences
- Use commands:
  - `/help` - Show available commands
  - `/knowledge` - Display knowledge base statistics
  - `/frameworks` - List available frameworks
  - `/history` - Show conversation history
  - `/new` - Start a new conversation
  - `/exit` - Exit the application

## Testing

The project includes comprehensive unit tests for all core modules.

### Running Tests Locally

1. Install test dependencies (included in requirements.txt):
   ```bash
   pip install -r requirements.txt
   ```

2. Run all tests:
   ```bash
   pytest
   ```

3. Run tests with coverage:
   ```bash
   pytest --cov=src --cov-report=term-missing
   ```

4. Run a specific test file:
   ```bash
   pytest tests/test_memory_system.py -v
   ```

### Test Structure

```
tests/
├── conftest.py           # Shared fixtures
├── test_memory_system.py # Memory and context tests
├── test_conversation.py  # Conversation manager tests
├── test_knowledge_base.py# Knowledge base tests
└── test_agents.py        # Agent and orchestrator tests
```

### CI/CD

Tests run automatically on:
- Every push to `main` branch
- Every pull request targeting `main`

The GitHub Actions workflow tests against Python 3.10, 3.11, and 3.12.

## Architecture

- **Ollama**: Local LLM runtime
- **ChromaDB**: Vector database for knowledge storage
- **Rich**: Terminal UI framework
- **LangGraph**: Agent orchestration
- **LlamaIndex**: Document processing and retrieval

## License

MIT