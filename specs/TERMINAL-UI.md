# Terminal UI

## Overview

Provides a rich terminal interface using the Rich library and prompt-toolkit for interactive command-line usage with tables, markdown rendering, and input history.

## Key Components

- `src/terminal_ui.py`: TerminalUI class

## Technologies

| Library | Purpose |
|---------|---------|
| `rich` | Panels, tables, markdown rendering |
| `prompt-toolkit` | Input with history, auto-suggest |

## Features

### User Input

- Command history saved to `~/.glenn_bot_history`
- Auto-suggest from history
- Ctrl+C or Ctrl+D returns `/exit`

### Display Methods

| Method | Purpose |
|--------|---------|
| `display_welcome()` | Welcome panel with command overview |
| `display_response(text)` | Assistant response in green panel |
| `display_error(text)` | Red error message |
| `display_help()` | Help panel with common commands (excludes debug/auto-context) |
| `display_knowledge_stats(stats)` | Knowledge base table |
| `display_frameworks(list)` | Frameworks table |
| `display_conversation_history(list)` | Conversation history table |
| `clear_screen()` | Clear terminal |
| `show_thinking_indicator()` | Spinner during processing |

## Panel Styles

| Content | Title Color | Border Color |
|---------|-------------|--------------|
| Welcome | Blue | Blue |
| Response | Green | Green |
| Help | Yellow | Yellow |
| Error | Red (inline) | - |

## Table Displays

### Knowledge Stats
```
┌──────────────────────────────────┐
│ Knowledge Base Statistics        │
├─────────────────┬────────────────┤
│ Type            │ Count          │
├─────────────────┼────────────────┤
│ Total Documents │ 42             │
│ Value           │ 10             │
│ Framework       │ 5              │
└─────────────────┴────────────────┘
```

### Conversation History
```
┌──────────────────────────────────────────────────────┐
│ Conversation History                                 │
├──────────┬───────────────┬──────────┬────────────────┤
│ ID       │ Started       │ Messages │ Preview        │
├──────────┼───────────────┼──────────┼────────────────┤
│ 20240115 │ 2024-01-15... │ 5        │ Hello, I...    │
└──────────┴───────────────┴──────────┴────────────────┘
```

## Behavior

### Welcome Message

Displays on startup with:
- Application title
- Brief description
- Core commands overview

### Help Panel

The help panel is a curated list of common commands and currently omits debug commands and auto-context commands.

### Response Rendering

- Markdown formatted using `rich.markdown.Markdown`
- Wrapped in panel with padding
- Can show "Thinking..." indicator before response

### Thinking Indicator

```python
with ui.show_thinking_indicator():
    # Long-running operation
    response = process_query(...)
```

Shows animated dots spinner: `Thinking...`

## Edge Cases

- **KeyboardInterrupt during input**: Returns `/exit`
- **EOFError during input**: Returns `/exit`
- **Empty input**: Returns empty string (handled by caller)

## Dependencies

- `rich`: Console, Panel, Table, Markdown, Syntax, Text
- `prompt_toolkit`: prompt, FileHistory, AutoSuggestFromHistory
- `pathlib.Path`: For history file path
