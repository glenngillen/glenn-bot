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

## Architecture

- **Ollama**: Local LLM runtime
- **ChromaDB**: Vector database for knowledge storage
- **Rich**: Terminal UI framework
- **LangGraph**: Agent orchestration
- **BeautifulSoup4**: Web content extraction

## License

MIT