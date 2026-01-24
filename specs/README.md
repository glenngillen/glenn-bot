# Glenn-Bot Specifications

A 100% local AI collaboration assistant that helps solve problems, build products, and create content based on personal values, principles, and problem-solving frameworks.

## Purpose

Glenn-bot is a personal AI assistant designed for:
- **Personal AI assistant**: A private, local AI that knows preferences and context
- **Knowledge management**: Organizing and retrieving personal knowledge, frameworks, and values
- **Decision support**: Making decisions aligned with values and principles
- **Project context switching**: Managing multiple projects/life areas with relevant context

## Architecture Overview

```
User Input → Terminal UI → GlennBot Orchestrator
                                    ↓
              ┌─────────────────────┼─────────────────────┐
              ↓                     ↓                     ↓
      Context Detection      Memory System        Knowledge Base
              ↓                     ↓                     ↓
              └─────────────────────┼─────────────────────┘
                                    ↓
                          Agent Orchestrator
            ┌──────────┬─────────┬──────────┬──────────┬──────────┐
            ↓          ↓         ↓          ↓          ↓          ↓
        Framework   Values   Preference    Quotes   Planning   Reasoning
          Agent      Agent      Agent       Agent     Agent     Agent
            └──────────┴─────────┴──────────┴──────────┴──────────┘
                                ↓
                         Reviewer Agent
                                ↓
                       Response + Learning
```

## Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| LLM Runtime | Ollama | Local language model inference |
| Vector Database | ChromaDB | Semantic search for knowledge/memories |
| Agent Orchestration | Custom (AgentOrchestrator + TaskExecutor) | Routing, planning, synthesis |
| Document Processing | requests + aiohttp + BeautifulSoup + markdownify | Web scraping and ingestion |
| Terminal UI | Rich + prompt-toolkit | Interactive command-line interface |

## Spec Index

| Spec | Description | Status |
|------|-------------|--------|
| [AGENTS.md](./AGENTS.md) | Multi-agent system with routing and scoring | Stable |
| [MEMORY.md](./MEMORY.md) | Context and memory persistence system | Stable |
| [KNOWLEDGE-BASE.md](./KNOWLEDGE-BASE.md) | ChromaDB-backed semantic search | Stable |
| [CONVERSATION.md](./CONVERSATION.md) | Conversation history management | Stable |
| [DOCUMENT-INGESTION.md](./DOCUMENT-INGESTION.md) | URL and text content ingestion | Stable |
| [CONTEXT-DETECTION.md](./CONTEXT-DETECTION.md) | Automatic context switching | Experimental |
| [FEEDBACK.md](./FEEDBACK.md) | User feedback and learning system | Experimental |
| [QUOTES.md](./QUOTES.md) | Inspirational quotes management | Stable |
| [TERMINAL-UI.md](./TERMINAL-UI.md) | Rich terminal interface | Stable |
| [COMMANDS.md](./COMMANDS.md) | Available slash commands reference | Stable |

## Key Files

| File | Purpose |
|------|---------|
| `src/main.py` | Entry point and GlennBot orchestrator |
| `src/agents.py` | Agent definitions and orchestration |
| `src/reasoning_agents.py` | Planning, Reasoning, Reviewer agents |
| `src/memory_system.py` | Memory and context management |
| `src/knowledge_base.py` | ChromaDB knowledge storage |
| `src/conversation.py` | Conversation history |
| `src/document_ingestion.py` | URL/text ingestion |
| `src/context_detector.py` | Auto-context detection |
| `src/feedback_system.py` | Feedback collection |
| `src/quotes_system.py` | Quotes management |
| `src/terminal_ui.py` | Terminal UI components |
| `src/ollama_client.py` | Ollama LLM interface |
| `src/config.py` | Settings management |

## Data Directories

| Directory | Contents |
|-----------|----------|
| `data/chroma/` | ChromaDB vector store |
| `data/conversations/` | Conversation history (JSON) |
| `data/memory/` | Memory system storage |
| `data/memory/contexts/` | Context definitions |
| `data/quotes/` | Quotes database |
| `data/feedback/` | User feedback records |
| `knowledge/` | User-defined values, preferences, frameworks |

## Running the Project

```bash
# Setup (first time)
./setup.sh

# Run
python run.py

# Tests
pytest
```

## Required Ollama Models

- `llama3:8b` - Main language model
- `nomic-embed-text` - Embeddings for vector search
