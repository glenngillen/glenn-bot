from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.table import Table
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.text import Text
from prompt_toolkit import prompt
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from typing import Optional
import asyncio
from pathlib import Path

class TerminalUI:
    def __init__(self):
        self.console = Console()
        self.history_file = Path.home() / ".glenn_bot_history"
        
    def display_welcome(self):
        """Display welcome message."""
        welcome_text = """
# Glenn Bot - AI Collaboration Assistant

Welcome! I'm here to help you solve problems, build products, and produce content.
I have access to your values, principles, preferences, and various problem-solving frameworks.

## Commands:
- Type your question or request
- `/help` - Show available commands
- `/knowledge` - Show knowledge base stats
- `/frameworks` - List available frameworks
- `/history` - Show conversation history
- `/new` - Start a new conversation
- `/exit` or `/quit` - Exit the application

Let's collaborate!
        """
        
        panel = Panel(
            Markdown(welcome_text),
            title="[bold blue]Glenn Bot[/bold blue]",
            border_style="blue"
        )
        self.console.print(panel)
        
    def get_user_input(self) -> str:
        """Get input from user with history support."""
        try:
            user_input = prompt(
                "You> ",
                history=FileHistory(str(self.history_file)),
                auto_suggest=AutoSuggestFromHistory(),
                multiline=False
            )
            return user_input.strip()
        except (KeyboardInterrupt, EOFError):
            return "/exit"
            
    def display_response(self, response: str, thinking: bool = False):
        """Display AI response."""
        if thinking:
            self.console.print("[dim]Thinking...[/dim]")
            
        panel = Panel(
            Markdown(response),
            title="[bold green]Assistant[/bold green]",
            border_style="green",
            padding=(1, 2)
        )
        self.console.print(panel)
        
    def display_error(self, error: str):
        """Display error message."""
        self.console.print(f"[bold red]Error:[/bold red] {error}")
        
    def display_knowledge_stats(self, stats: dict):
        """Display knowledge base statistics."""
        table = Table(title="Knowledge Base Statistics")
        table.add_column("Type", style="cyan")
        table.add_column("Count", style="magenta")
        
        table.add_row("Total Documents", str(stats['total_documents']))
        
        for doc_type, count in stats.get('types', {}).items():
            table.add_row(doc_type.capitalize(), str(count))
            
        self.console.print(table)
        
    def display_frameworks(self, frameworks: list):
        """Display available frameworks."""
        table = Table(title="Available Frameworks")
        table.add_column("Name", style="cyan")
        table.add_column("Category", style="magenta")
        table.add_column("Description", style="white")
        
        for fw in frameworks:
            table.add_row(
                fw['name'],
                fw.get('category', 'General'),
                fw.get('description', '')[:50] + "..."
            )
            
        self.console.print(table)
        
    def display_conversation_history(self, conversations: list):
        """Display conversation history."""
        table = Table(title="Conversation History")
        table.add_column("ID", style="cyan")
        table.add_column("Started", style="magenta")
        table.add_column("Messages", style="green")
        table.add_column("Preview", style="white")
        
        for conv in conversations[:10]:  # Show last 10
            table.add_row(
                conv['id'],
                conv.get('started_at', 'Unknown'),
                str(conv['message_count']),
                conv.get('first_message', '')
            )
            
        self.console.print(table)
        
    def display_help(self):
        """Display help information."""
        help_text = """
## Available Commands:

### Conversation
- **Natural Language Input**: Just type your question or request
- `/help` - Show this help message
- `/history` - Show recent conversations
- `/load <id>` - Load a previous conversation
- `/new` - Start a new conversation
- `/clear` - Clear the screen
- `/exit` or `/quit` - Exit the application

### Context & Memory
- `/contexts` - Show available operating contexts
- `/context <id>` - Switch to a specific context
- `/new-context <name> <type> [description]` - Create new context
- `/delete-context <id>` - Delete a context and its memories
- `/focus` - Show current focus
- `/focus <new_focus>` - Update current context focus
- `/memory` - Show memory statistics
- `/remember <content>` - Manually add a memory
- `/recall <query>` - Search and display relevant memories

### Knowledge Base
- `/knowledge` - Display knowledge base statistics
- `/frameworks` - List available problem-solving frameworks
- `/add-url <url> [context]` - Add content from a webpage to knowledge base
- `/add-text <name> [context]` - Add manually entered text to knowledge base
- `/clean-knowledge` - Remove duplicate knowledge base entries

### Quotes & Inspiration
- `/add-quote "<quote>" "<author>" [context]` - Add an inspirational quote
- `/reflect` - Get a random quote for reflection
- `/search-quotes <query>` - Search quotes by content, author, or tags
- `/quotes-stats` - Show quotes collection statistics

## Tips:
- **Context Switching**: Use contexts to maintain separate project spaces
- **Memory**: The system automatically remembers important information from conversations
- **Frameworks**: Be specific about which frameworks or principles you want to apply
- **Continuous Learning**: Use knowledge ingestion to expand capabilities
        """
        
        panel = Panel(
            Markdown(help_text),
            title="[bold yellow]Help[/bold yellow]",
            border_style="yellow"
        )
        self.console.print(panel)
        
    def clear_screen(self):
        """Clear the console screen."""
        self.console.clear()
        
    def show_thinking_indicator(self):
        """Show a thinking indicator."""
        return self.console.status("[bold green]Thinking...[/bold green]", spinner="dots")