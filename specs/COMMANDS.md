# Commands Reference

## Overview

Glenn-bot provides slash commands for managing conversations, contexts, knowledge, quotes, and feedback. Commands are processed in `src/main.py` in the `handle_command()` method.

## Command Categories

### Conversation

| Command | Description |
|---------|-------------|
| `/help` | Show help panel with common commands |
| `/history` | List recent conversations |
| `/load <id>` | Load a previous conversation |
| `/new` | Start a new conversation |
| `/clear` | Clear the terminal screen |
| `/exit`, `/quit` | Exit the application |

### Context & Memory

| Command | Description |
|---------|-------------|
| `/contexts` | Show all available contexts |
| `/context <id>` | Switch to a specific context |
| `/new-context <name> <type> [description]` | Create a new context |
| `/delete-context <id>` | Delete a context and its memories |
| `/focus` | Show current focus |
| `/focus <new_focus>` | Update current context focus |
| `/memory` | Show memory statistics |
| `/remember <content>` | Manually add a memory |
| `/recall <query>` | Search and display relevant memories |

### Auto-Context Detection

| Command | Description |
|---------|-------------|
| `/auto-context` | Show auto-context status and settings |
| `/auto-context on` | Enable auto-context switching |
| `/auto-context off` | Disable auto-context switching |
| `/auto-context threshold <0.0-1.0>` | Set auto-switch threshold |

### Knowledge Base

| Command | Description |
|---------|-------------|
| `/knowledge` | Display knowledge base statistics |
| `/frameworks` | List available problem-solving frameworks |
| `/add-url <url> [context]` | Add content from webpage |
| `/add-urls [context]` | Batch add multiple URLs with progress |
| `/add-text <name> [context]` | Add manually entered text (prompts for multiline input) |
| `/clean-knowledge` | Remove duplicate entries |
| `/export-knowledge [filename]` | Export to JSON backup |
| `/import-knowledge <filename>` | Import from JSON backup |

### Debugging (Knowledge)

| Command | Description |
|---------|-------------|
| `/debug-search <query>` | Debug semantic search results |
| `/list-knowledge` | List all knowledge items |
| `/show-doc <name>` | Show a specific document |
| `/debug-agents <query>` | Debug agent selection for a query |

### Quotes & Inspiration

| Command | Description |
|---------|-------------|
| `/add-quote "<quote>" "<author>" [context]` | Add inspirational quote |
| `/reflect` | Get random quote for reflection |
| `/search-quotes <query>` | Search quotes semantically |
| `/quotes-stats` | Show quotes collection statistics |

### Response Feedback

| Command | Description |
|---------|-------------|
| `/rate` | Interactive rating for last response |
| `/rate +`, `/rate up` | Quick thumbs up |
| `/rate -`, `/rate down` | Quick thumbs down |
| `/rate <1-5>` | Star rating (1-5 scale) |
| `/feedback-stats` | Show feedback statistics |
| `/best-responses` | View highest-rated responses |
| `/worst-responses` | View lowest-rated for improvement |
| `/feedback-insights` | Get insights from feedback |

## Behavior

### Command Parsing

Commands start with `/` and are case-sensitive. Arguments are space-separated.
Most commands use simple space splitting:
- `/add-text <name> [context]` and `/new-context <name> <type> [description]` treat the first token as the name, so names cannot include spaces.
- `/add-url <url> [context]` and `/add-urls [context]` treat the remainder as context, so context strings may include spaces.
`/add-quote` uses simple splitting: it strips surrounding quotes if present but does not support spaces inside the quote or author tokens. The optional context is the remainder of the line (so it can include spaces). If the context is omitted, the command prompts with `Why does this quote resonate with you?`.

### Unknown Commands

Returns error: `Unknown command: {command}`

### Natural Language Queries

Any input not starting with `/` is processed as a query through the agent system.

### Pending Context Switch

After context detection prompts `Switch context? (y/n)`:
- `y` or `yes`: Apply the switch
- `n` or `no`: Cancel the switch
- Other input: Cancel and process as normal query

### Interactive Commands

- `/add-urls [context]`: Prompts for one URL per line until EOF (Ctrl+D).
- `/add-text <name> [context]`: Prompts for multiline content until EOF (Ctrl+D).
- `/add-quote "<quote>" "<author>" [context]`: Prompts for context if omitted, then uses AI to categorize the quote.
- `/rate -` and low star ratings (1-2) prompt for optional text feedback.
- `/rate` accepts quick tokens: `+`, `up`, `👍`, `good` → thumbs up; `-`, `down`, `👎`, `bad` → thumbs down.
- `/auto-context threshold <value>`: Updates only the auto-switch threshold (prompt threshold remains unchanged).

### Help Output

The `/help` panel is curated and currently omits debug commands and auto-context commands.

## Implementation Location

All command handling is in `src/main.py`:

| Method | Purpose |
|--------|---------|
| `handle_command()` | Main command router |
| `_show_contexts()` | Display contexts table |
| `_switch_context()` | Change current context |
| `_create_context()` | Add new context |
| `_delete_context()` | Remove context |
| `_show_memory_stats()` | Memory statistics |
| `_add_manual_memory()` | Create memory from input |
| `_recall_memories()` | Search memories |
| `_handle_url_ingestion()` | Single URL processing |
| `_handle_batch_url_ingestion()` | Multiple URL processing |
| `_handle_text_ingestion()` | Manual text entry |
| `_add_quote()` | Add new quote |
| `_reflect_on_quote()` | Random reflection |
| `_search_quotes()` | Quote search |
| `_show_quotes_stats()` | Quote statistics |
| `_rate_last_response()` | Feedback rating |
| `_show_feedback_stats()` | Feedback statistics |
| `_show_best_responses()` | Top-rated responses |
| `_show_worst_responses()` | Low-rated responses |
| `_show_feedback_insights()` | Feedback analysis |
| `_export_knowledge()` | Knowledge export |
| `_import_knowledge()` | Knowledge import |
| `_show_auto_context_status()` | Auto-context settings |
| `_set_auto_context()` | Enable/disable auto-context |
| `_set_auto_context_threshold()` | Adjust threshold |
