import logging
from pathlib import Path
from typing import Optional
from datetime import datetime
from src.config import settings
from src.terminal_ui import TerminalUI
from src.knowledge_base import KnowledgeBase
from src.conversation import ConversationManager
from src.ollama_client import OllamaClient
from src.agents import AgentOrchestrator
from src.document_ingestion import DocumentIngestionTool, BatchProgress, ProcessingStatus
from src.memory_system import MemorySystem, MemoryType
from src.quotes_system import QuotesSystem
from src.feedback_system import FeedbackManager, FeedbackType
from src.context_detector import ContextDetector

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Silence noisy loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)
logging.getLogger("src.knowledge_base").setLevel(logging.WARNING)
logging.getLogger("src.memory_system").setLevel(logging.WARNING)
logging.getLogger("src.agents").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

class GlennBot:
    def __init__(self):
        self.ui = TerminalUI()
        self.knowledge_base = KnowledgeBase()
        self.conversation = ConversationManager()
        self.ollama_client = OllamaClient()
        self.memory_system = MemorySystem(self.knowledge_base, self.ollama_client)
        self.quotes_system = QuotesSystem(self.memory_system, self.knowledge_base, self.ollama_client)
        self.orchestrator = AgentOrchestrator(self.knowledge_base, self.ollama_client, self.quotes_system)
        self.ingestion_tool = DocumentIngestionTool(self.knowledge_base, self.ollama_client)
        self.feedback_manager = FeedbackManager()

        # Context detection with thresholds from memory system settings
        self.context_detector = ContextDetector(
            self.ollama_client,
            auto_switch_threshold=self.memory_system.auto_switch_threshold,
            prompt_threshold=self.memory_system.prompt_threshold
        )

        # Track pending context switch (awaiting user confirmation)
        self.pending_context_switch = None
        
    def initialize(self):
        """Initialize the bot and load knowledge base."""
        self.ui.display_welcome()
        
        # Load knowledge files if they exist
        if settings.knowledge_dir.exists():
            with self.ui.show_thinking_indicator():
                self.knowledge_base.load_knowledge_files(settings.knowledge_dir)
            
        stats = self.knowledge_base.get_stats()
        if stats['total_documents'] > 0:
            self.ui.console.print(f"[green]Loaded {stats['total_documents']} knowledge items[/green]")
        else:
            self.ui.console.print("[yellow]No knowledge files found. Add them to the 'knowledge' directory.[/yellow]")
            
    def handle_command(self, command: str) -> bool:
        """Handle special commands. Returns True if should continue, False to exit."""
        
        if command in ["/exit", "/quit"]:
            return False
            
        elif command == "/help":
            self.ui.display_help()
            
        elif command == "/knowledge":
            stats = self.knowledge_base.get_stats()
            self.ui.display_knowledge_stats(stats)
            
        elif command == "/frameworks":
            frameworks = self.knowledge_base.search(
                "framework",
                n_results=20,
                filter_metadata={"type": {"$eq": "framework"}}
            )
            self.ui.display_frameworks([
                {
                    "name": fw['metadata']['name'],
                    "category": fw['metadata'].get('category', 'General'),
                    "description": fw['content'].split('\n')[1] if '\n' in fw['content'] else fw['content']
                }
                for fw in frameworks
            ])
            
        elif command == "/history":
            conversations = self.conversation.list_conversations()
            self.ui.display_conversation_history(conversations)
            
        elif command.startswith("/load "):
            conv_id = command.split(" ", 1)[1]
            try:
                self.conversation.load_conversation(conv_id)
                self.ui.console.print(f"[green]Loaded conversation {conv_id}[/green]")
            except Exception as e:
                self.ui.display_error(f"Failed to load conversation: {e}")
                
        elif command == "/new":
            self.conversation.start_new_conversation()
            self.ui.console.print("[green]Started new conversation[/green]")
            
        elif command == "/clear":
            self.ui.clear_screen()
            
        elif command.startswith("/add-url "):
            parts = command.split(" ", 2)
            if len(parts) < 2:
                self.ui.display_error("Usage: /add-url <url> [context]")
            else:
                url = parts[1]
                context = parts[2] if len(parts) > 2 else ""
                self._handle_url_ingestion(url, context)
                
        elif command.startswith("/add-text "):
            parts = command.split(" ", 2)
            if len(parts) < 2:
                self.ui.display_error("Usage: /add-text <name> [context]")
            else:
                name = parts[1]
                context = parts[2] if len(parts) > 2 else ""
                self._handle_text_ingestion(name, context)

        elif command == "/add-urls":
            self._handle_batch_url_ingestion()

        elif command.startswith("/add-urls "):
            context = command[10:].strip()  # Remove "/add-urls "
            self._handle_batch_url_ingestion(context)

        elif command == "/contexts":
            self._show_contexts()
            
        elif command.startswith("/context "):
            context_id = command.split(" ", 1)[1]
            self._switch_context(context_id)
            
        elif command.startswith("/new-context "):
            parts = command.split(" ", 3)
            if len(parts) < 3:
                self.ui.display_error("Usage: /new-context <name> <type> [description]")
            else:
                name = parts[1]
                context_type = parts[2]
                description = parts[3] if len(parts) > 3 else ""
                self._create_context(name, context_type, description)
                
        elif command == "/memory":
            self._show_memory_stats()
            
        elif command.startswith("/remember "):
            content = command[10:].strip()  # Remove "/remember "
            self._add_manual_memory(content)
            
        elif command.startswith("/recall "):
            query = command[8:].strip()  # Remove "/recall "
            self._recall_memories(query)
            
        elif command == "/focus":
            if self.memory_system.current_context:
                self.ui.console.print(f"[yellow]Current focus: {self.memory_system.current_context.current_focus}[/yellow]")
            else:
                self.ui.display_error("No context selected")
                
        elif command.startswith("/focus "):
            new_focus = command[7:].strip()  # Remove "/focus "
            self._update_focus(new_focus)
            
        elif command.startswith("/delete-context "):
            context_id = command.split(" ", 1)[1]
            self._delete_context(context_id)
            
        elif command == "/clean-knowledge":
            self._clean_knowledge_duplicates()
            
        elif command.startswith("/debug-search "):
            query = command[14:].strip()  # Remove "/debug-search "
            self._debug_search(query)
            
        elif command == "/list-knowledge":
            self._list_all_knowledge()
            
        elif command.startswith("/show-doc "):
            doc_name = command[10:].strip()  # Remove "/show-doc "
            self._show_document(doc_name)
            
        elif command.startswith("/debug-agents "):
            query = command[14:].strip()  # Remove "/debug-agents "
            self._debug_agent_selection(query)
            
        elif command.startswith("/add-quote "):
            parts = command.split(" ", 3)
            if len(parts) < 3:
                self.ui.display_error("Usage: /add-quote \"<quote>\" \"<author>\" [context]")
            else:
                quote_text = parts[1].strip('"')
                author = parts[2].strip('"')
                context = parts[3].strip('"') if len(parts) > 3 else ""
                self._add_quote(quote_text, author, context)
                
        elif command == "/reflect":
            self._reflect_on_quote()
            
        elif command.startswith("/search-quotes "):
            query = command[15:].strip()  # Remove "/search-quotes "
            self._search_quotes(query)
            
        elif command == "/quotes-stats":
            self._show_quotes_stats()

        # Feedback commands
        elif command == "/rate":
            self._rate_last_response()

        elif command.startswith("/rate "):
            rating_str = command[6:].strip()
            self._rate_last_response(rating_str)

        elif command == "/feedback-stats":
            self._show_feedback_stats()

        elif command == "/best-responses":
            self._show_best_responses()

        elif command == "/worst-responses":
            self._show_worst_responses()

        elif command == "/feedback-insights":
            self._show_feedback_insights()

        # Knowledge export/import commands
        elif command == "/export-knowledge" or command.startswith("/export-knowledge "):
            parts = command.split(" ", 1)
            filename = parts[1] if len(parts) > 1 else None
            self._export_knowledge(filename)

        elif command.startswith("/import-knowledge "):
            filename = command.split(" ", 1)[1]
            self._import_knowledge(filename)

        # Auto-context commands
        elif command == "/auto-context":
            self._show_auto_context_status()

        elif command == "/auto-context on":
            self._set_auto_context(True)

        elif command == "/auto-context off":
            self._set_auto_context(False)

        elif command.startswith("/auto-context threshold "):
            value_str = command[24:].strip()
            self._set_auto_context_threshold(value_str)

        else:
            self.ui.display_error(f"Unknown command: {command}")
            
        return True
        
    def process_query(self, query: str):
        """Process a user query through the agent system."""
        # Handle pending context switch confirmation
        if self.pending_context_switch:
            if query.lower() in ['y', 'yes']:
                self._apply_context_switch(self.pending_context_switch)
                self.pending_context_switch = None
                return
            elif query.lower() in ['n', 'no']:
                self.ui.console.print("[yellow]Context switch cancelled[/yellow]")
                self.pending_context_switch = None
                return
            else:
                # Clear pending switch and process as normal query
                self.pending_context_switch = None

        # Auto-detect context if enabled
        if self.memory_system.auto_switch_enabled:
            self._detect_and_handle_context_switch(query)

        # If there's a pending switch that needs confirmation, don't process query yet
        if self.pending_context_switch:
            return

        # Add to conversation
        self.conversation.add_message("user", query)

        # Get memory context
        memory_context = self.memory_system.get_context_for_query(query)

        # Prepare full context
        context = {
            "conversation_context": self.conversation.get_conversation_context(),
            "memory_context": memory_context,
            "current_context": memory_context["current_context"],
            "relevant_memories": memory_context["relevant_memories"]
        }

        try:
            with self.ui.show_thinking_indicator():
                response = self.orchestrator.process_query(query, context)

            self.conversation.add_message("assistant", response)
            self.ui.display_response(response)

            # Extract and save important memories from this interaction
            recent_conversation = self.conversation.get_conversation_context(max_messages=4)
            if len(recent_conversation.split('\n')) > 6:  # Only if substantial conversation
                extracted_memories = self.memory_system.extract_memories_from_conversation(recent_conversation)
                if extracted_memories:
                    self.ui.console.print(f"[dim]💾 Saved {len(extracted_memories)} memories[/dim]")
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            self.ui.display_error(f"Failed to process query: {e}")

    def _detect_and_handle_context_switch(self, query: str):
        """Detect context from query and handle switching."""
        try:
            current_context_id = (
                self.memory_system.current_context.id
                if self.memory_system.current_context
                else None
            )

            # Get recent conversation context for better classification
            recent_context = self.conversation.get_conversation_context(max_messages=3)

            # Detect context
            result = self.context_detector.detect_and_recommend(
                query,
                current_context_id=current_context_id,
                recent_context=recent_context
            )

            if result.should_switch and result.recommended_context:
                # Check if the recommended context exists in memory system
                if result.recommended_context not in self.memory_system.contexts:
                    # Context doesn't exist yet, skip switch
                    return

                if result.needs_confirmation:
                    # Prompt user for confirmation
                    context = self.memory_system.contexts[result.recommended_context]
                    self.ui.console.print(
                        f"[yellow]Detected topic shift to '{context.name}' "
                        f"(confidence: {result.confidence:.0%})[/yellow]"
                    )
                    self.ui.console.print(
                        f"[yellow]Switch context? (y/n)[/yellow]"
                    )
                    self.pending_context_switch = {
                        "to_context": result.recommended_context,
                        "from_context": current_context_id,
                        "confidence": result.confidence
                    }
                else:
                    # Auto-switch silently
                    self._apply_context_switch({
                        "to_context": result.recommended_context,
                        "from_context": current_context_id,
                        "confidence": result.confidence
                    }, silent=True)

        except Exception as e:
            logger.warning(f"Context detection failed: {e}")
            # Continue without context switching on failure

    def _apply_context_switch(self, switch_info: dict, silent: bool = False):
        """Apply a context switch."""
        to_context = switch_info["to_context"]
        from_context = switch_info.get("from_context")
        confidence = switch_info.get("confidence", 1.0)

        if self.memory_system.switch_context(to_context):
            context = self.memory_system.current_context
            if not silent:
                self.ui.console.print(f"[green]Switched to: {context.name}[/green]")
            else:
                self.ui.console.print(
                    f"[dim]Auto-switched to '{context.name}' context[/dim]"
                )

            # Record the switch for learning
            self.memory_system.record_context_switch(
                from_context=from_context or "none",
                to_context=to_context,
                was_auto=silent,
                confidence=confidence
            )
        else:
            self.ui.display_error(f"Failed to switch to context '{to_context}'")

    def _handle_url_ingestion(self, url: str, context: str):
        """Handle adding content from a URL."""
        try:
            with self.ui.show_thinking_indicator():
                classification = self.ingestion_tool.add_web_content(url, context)
                
            self.ui.console.print(f"[green]✓ Added {classification['type']}: '{classification['name']}'[/green]")
            self.ui.console.print(f"[dim]Category: {classification.get('category', 'N/A')}[/dim]")
            self.ui.console.print(f"[dim]Description: {classification['description']}[/dim]")
            
        except Exception as e:
            logger.error(f"Error ingesting URL {url}: {e}")
            self.ui.display_error(f"Failed to ingest URL: {e}")
            
    def _handle_text_ingestion(self, name: str, context: str):
        """Handle adding manually entered text content."""
        try:
            self.ui.console.print(f"[yellow]Enter your content for '{name}' (press Ctrl+D when done):[/yellow]")
            
            lines = []
            try:
                while True:
                    line = input()
                    lines.append(line)
            except EOFError:
                pass
                
            content = "\n".join(lines).strip()
            
            if not content:
                self.ui.display_error("No content provided")
                return
                
            with self.ui.show_thinking_indicator():
                classification = self.ingestion_tool.add_text_content(content, context, name)
                
            self.ui.console.print(f"[green]✓ Added {classification['type']}: '{classification['name']}'[/green]")
            self.ui.console.print(f"[dim]Category: {classification.get('category', 'N/A')}[/dim]")
            self.ui.console.print(f"[dim]Description: {classification['description']}[/dim]")
            
        except Exception as e:
            logger.error(f"Error ingesting text content: {e}")
            self.ui.display_error(f"Failed to ingest content: {e}")

    def _handle_batch_url_ingestion(self, context: str = ""):
        """Handle adding content from multiple URLs concurrently."""
        try:
            self.ui.console.print("[yellow]Enter URLs to process (one per line, press Ctrl+D when done):[/yellow]")

            urls = []
            try:
                while True:
                    line = input().strip()
                    if line:
                        urls.append(line)
            except EOFError:
                pass

            if not urls:
                self.ui.display_error("No URLs provided")
                return

            self.ui.console.print(f"\n[cyan]Processing {len(urls)} URLs concurrently...[/cyan]")

            # Create a progress display callback
            from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=self.ui.console,
                transient=True
            ) as progress_bar:
                task_id = progress_bar.add_task("Processing URLs", total=len(urls))

                def progress_callback(batch_progress: BatchProgress):
                    progress_bar.update(
                        task_id,
                        completed=batch_progress.completed + batch_progress.failed,
                        description=f"Processing: {batch_progress.current_url or 'waiting...'}"[:50]
                    )

                # Run batch processing
                result = self.ingestion_tool.batch_process_urls_sync(
                    urls=urls,
                    user_context=context,
                    max_concurrent=5,
                    progress_callback=progress_callback
                )

            # Display results
            self.ui.console.print(f"\n[green]✓ Completed: {result.completed}/{result.total}[/green]")
            if result.failed > 0:
                self.ui.console.print(f"[red]✗ Failed: {result.failed}[/red]")

            # Show details for each result
            from rich.table import Table
            table = Table(title="Batch Processing Results")
            table.add_column("Status", style="cyan")
            table.add_column("URL", style="white", max_width=50)
            table.add_column("Result", style="green")

            for res in result.results:
                if res.status == ProcessingStatus.COMPLETED:
                    status_icon = "✓"
                    result_text = f"{res.classification['type']}: {res.classification['name']}"
                else:
                    status_icon = "✗"
                    result_text = res.error or "Unknown error"

                # Truncate URL for display
                display_url = res.url
                if len(display_url) > 50:
                    display_url = display_url[:47] + "..."

                table.add_row(status_icon, display_url, result_text[:40])

            self.ui.console.print(table)

        except KeyboardInterrupt:
            self.ui.console.print("\n[yellow]Batch processing cancelled[/yellow]")
        except Exception as e:
            logger.error(f"Error in batch URL ingestion: {e}")
            self.ui.display_error(f"Failed to process URLs: {e}")

    def _show_contexts(self):
        """Show available contexts."""
        contexts = list(self.memory_system.contexts.values())
        if not contexts:
            self.ui.console.print("[yellow]No contexts available[/yellow]")
            return
            
        from rich.table import Table
        table = Table(title="Available Contexts")
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="green")
        table.add_column("Type", style="magenta")
        table.add_column("Status", style="yellow")
        table.add_column("Focus", style="white")
        table.add_column("Current", style="red")
        
        for context in sorted(contexts, key=lambda x: x.last_used, reverse=True):
            is_current = "🎯" if self.memory_system.current_context and context.id == self.memory_system.current_context.id else ""
            table.add_row(
                context.id,
                context.name,
                context.context_type,
                context.status,
                context.current_focus[:30] + "..." if len(context.current_focus) > 30 else context.current_focus,
                is_current
            )
            
        self.ui.console.print(table)

    def _show_auto_context_status(self):
        """Show current auto-context settings."""
        enabled = self.memory_system.auto_switch_enabled
        auto_threshold = self.memory_system.auto_switch_threshold
        prompt_threshold = self.memory_system.prompt_threshold

        from rich.panel import Panel
        from rich.text import Text

        status_text = Text()
        status_text.append("Auto-context: ", style="bold")
        status_text.append(
            "Enabled" if enabled else "Disabled",
            style="green" if enabled else "red"
        )
        status_text.append("\n\n")
        status_text.append("Auto-switch threshold: ", style="bold")
        status_text.append(f"{auto_threshold:.0%}", style="cyan")
        status_text.append(" (switches silently above this)\n")
        status_text.append("Prompt threshold: ", style="bold")
        status_text.append(f"{prompt_threshold:.0%}", style="cyan")
        status_text.append(" (asks for confirmation above this)\n\n")
        status_text.append("Commands:\n", style="bold")
        status_text.append("  /auto-context on       - Enable auto-switching\n", style="dim")
        status_text.append("  /auto-context off      - Disable auto-switching\n", style="dim")
        status_text.append("  /auto-context threshold <value>  - Set auto-switch threshold (0.0-1.0)\n", style="dim")

        panel = Panel(status_text, title="Auto-Context Settings", border_style="blue")
        self.ui.console.print(panel)

    def _set_auto_context(self, enabled: bool):
        """Enable or disable auto-context switching."""
        self.memory_system.auto_switch_enabled = enabled
        status = "enabled" if enabled else "disabled"
        self.ui.console.print(f"[green]Auto-context switching {status}[/green]")

    def _set_auto_context_threshold(self, value_str: str):
        """Set the auto-context switching threshold."""
        try:
            value = float(value_str)
            if value < 0.0 or value > 1.0:
                self.ui.display_error("Threshold must be between 0.0 and 1.0")
                return

            self.memory_system.auto_switch_threshold = value
            # Update the context detector's threshold too
            self.context_detector.auto_switch_threshold = value
            self.ui.console.print(f"[green]Auto-switch threshold set to {value:.0%}[/green]")

        except ValueError:
            self.ui.display_error(f"Invalid threshold value: {value_str}")

    def _switch_context(self, context_id: str):
        """Switch to a different context."""
        if self.memory_system.switch_context(context_id):
            context = self.memory_system.current_context
            self.ui.console.print(f"[green]✓ Switched to: {context.name}[/green]")
            self.ui.console.print(f"[dim]{context.description}[/dim]")
            self.ui.console.print(f"[dim]Current focus: {context.current_focus}[/dim]")
        else:
            self.ui.display_error(f"Context '{context_id}' not found")
            
    def _create_context(self, name: str, context_type: str, description: str):
        """Create a new context."""
        try:
            context = self.memory_system.create_context(name, description, context_type)
            self.ui.console.print(f"[green]✓ Created context: {context.name} ({context.id})[/green]")
            self.ui.console.print(f"[dim]{context.description}[/dim]")
        except Exception as e:
            logger.error(f"Error creating context: {e}")
            self.ui.display_error(f"Failed to create context: {e}")
            
    def _show_memory_stats(self):
        """Show memory system statistics."""
        total_memories = len(self.memory_system.memories)
        
        if total_memories == 0:
            self.ui.console.print("[yellow]No memories stored yet[/yellow]")
            return
            
        # Count by type
        type_counts = {}
        importance_sum = 0
        
        for memory in self.memory_system.memories.values():
            memory_type = memory.memory_type.value
            type_counts[memory_type] = type_counts.get(memory_type, 0) + 1
            importance_sum += memory.importance
            
        avg_importance = importance_sum / total_memories if total_memories > 0 else 0
        
        from rich.table import Table
        table = Table(title="Memory Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total Memories", str(total_memories))
        table.add_row("Average Importance", f"{avg_importance:.1f}/10")
        
        if self.memory_system.current_context:
            context_memories = [m for m in self.memory_system.memories.values() 
                             if self.memory_system.current_context.id in m.projects]
            table.add_row("Current Context Memories", str(len(context_memories)))
            
        for mem_type, count in sorted(type_counts.items()):
            table.add_row(f"{mem_type.title()} Memories", str(count))
            
        self.ui.console.print(table)
        
    def _add_manual_memory(self, content: str):
        """Add a manual memory."""
        if not content:
            self.ui.display_error("Please provide memory content")
            return
            
        # Ask for memory type
        self.ui.console.print("[yellow]Memory types:[/yellow]")
        for i, mem_type in enumerate(MemoryType, 1):
            self.ui.console.print(f"{i}. {mem_type.value.title()}")
            
        try:
            choice = input("Select type (1-8): ")
            type_index = int(choice) - 1
            memory_types = list(MemoryType)
            
            if 0 <= type_index < len(memory_types):
                memory_type = memory_types[type_index]
                
                # Ask for importance
                importance = input("Importance (1-10, default 5): ").strip()
                importance = int(importance) if importance.isdigit() else 5
                importance = max(1, min(10, importance))
                
                memory = self.memory_system.add_memory(content, memory_type, importance)
                self.ui.console.print(f"[green]✓ Added {memory_type.value} memory[/green]")
            else:
                self.ui.display_error("Invalid selection")
                
        except (ValueError, KeyboardInterrupt):
            self.ui.display_error("Cancelled")
            
    def _recall_memories(self, query: str):
        """Recall and display memories."""
        memories = self.memory_system.recall_memories(query, limit=8)
        
        if not memories:
            self.ui.console.print("[yellow]No relevant memories found[/yellow]")
            return
            
        from rich.table import Table
        table = Table(title=f"Recalled Memories for: '{query}'")
        table.add_column("Type", style="cyan")
        table.add_column("Content", style="white", max_width=50)
        table.add_column("Context", style="magenta")
        table.add_column("Importance", style="yellow")
        table.add_column("Age", style="dim")
        
        for memory in memories:
            age_days = (datetime.now() - memory.created_at).days
            age_str = f"{age_days}d" if age_days > 0 else "today"
            
            table.add_row(
                memory.memory_type.value,
                memory.content[:100] + "..." if len(memory.content) > 100 else memory.content,
                memory.context,
                f"{memory.importance}/10",
                age_str
            )
            
        self.ui.console.print(table)
        
    def _update_focus(self, new_focus: str):
        """Update the current context's focus."""
        if not self.memory_system.current_context:
            self.ui.display_error("No context selected")
            return
            
        old_focus = self.memory_system.current_context.current_focus
        self.memory_system.update_context_focus(new_focus)
        
        self.ui.console.print(f"[green]✓ Updated focus[/green]")
        self.ui.console.print(f"[dim]From: {old_focus}[/dim]")
        self.ui.console.print(f"[dim]To: {new_focus}[/dim]")
        
        # Add this as a memory
        self.memory_system.add_memory(
            f"Changed focus from '{old_focus}' to '{new_focus}'",
            MemoryType.PROJECT,
            importance=6
        )
        
    def _delete_context(self, context_id: str):
        """Delete a context."""
        if context_id not in self.memory_system.contexts:
            self.ui.display_error(f"Context '{context_id}' not found")
            return
            
        context = self.memory_system.contexts[context_id]
        
        # Confirm deletion
        self.ui.console.print(f"[yellow]Are you sure you want to delete context '{context.name}' ({context_id})?[/yellow]")
        self.ui.console.print(f"[dim]This will also delete associated memories.[/dim]")
        
        try:
            confirm = input("Type 'yes' to confirm: ").strip().lower()
            if confirm == 'yes':
                # Remove from current context if it's the active one
                if self.memory_system.current_context and self.memory_system.current_context.id == context_id:
                    self.memory_system.current_context = None
                    
                # Delete context file
                context_file = self.memory_system.contexts_dir / f"{context_id}.json"
                if context_file.exists():
                    context_file.unlink()
                    
                # Remove from contexts dict
                del self.memory_system.contexts[context_id]
                
                # Remove associated memories
                memories_to_remove = []
                for mem_id, memory in self.memory_system.memories.items():
                    if context_id in memory.projects or memory.context == context_id:
                        memories_to_remove.append(mem_id)
                        
                for mem_id in memories_to_remove:
                    del self.memory_system.memories[mem_id]
                    
                # Save updated memories
                self.memory_system._save_memories()
                
                self.ui.console.print(f"[green]✓ Deleted context '{context.name}' and {len(memories_to_remove)} associated memories[/green]")
                
                if self.memory_system.current_context is None and self.memory_system.contexts:
                    # Switch to first available context
                    first_context = list(self.memory_system.contexts.keys())[0]
                    self.memory_system.switch_context(first_context)
                    self.ui.console.print(f"[dim]Switched to: {self.memory_system.current_context.name}[/dim]")
            else:
                self.ui.console.print("[dim]Cancelled[/dim]")
                
        except KeyboardInterrupt:
            self.ui.console.print("\n[dim]Cancelled[/dim]")
            
    def _clean_knowledge_duplicates(self):
        """Clean up duplicate knowledge base entries."""
        try:
            self.ui.console.print("[yellow]Cleaning up knowledge base duplicates...[/yellow]")
            
            # Get all documents
            all_docs = self.knowledge_base.collection.get()
            
            # Track unique items and duplicates
            seen_items = {}
            duplicates_to_remove = []
            
            for i, (doc_id, metadata) in enumerate(zip(all_docs['ids'], all_docs['metadatas'])):
                if not metadata:
                    continue
                    
                # Create a key for deduplication
                item_type = metadata.get('type', 'unknown')
                item_name = metadata.get('name', 'unknown')
                
                # Special handling for preferences
                if item_type == 'preference':
                    category = metadata.get('category', 'unknown')
                    key = f"{item_type}:{category}:{item_name}"
                else:
                    key = f"{item_type}:{item_name}"
                
                if key in seen_items:
                    # This is a duplicate
                    duplicates_to_remove.append(doc_id)
                else:
                    # First time seeing this item
                    seen_items[key] = doc_id
            
            if duplicates_to_remove:
                # Remove duplicates
                self.knowledge_base.collection.delete(ids=duplicates_to_remove)
                
                self.ui.console.print(f"[green]✓ Removed {len(duplicates_to_remove)} duplicate knowledge items[/green]")
                
                # Show new stats
                stats = self.knowledge_base.get_stats()
                self.ui.console.print(f"[dim]Knowledge base now has {stats['total_documents']} items[/dim]")
            else:
                self.ui.console.print("[green]No duplicates found[/green]")
                
        except Exception as e:
            logger.error(f"Error cleaning knowledge duplicates: {e}")
            self.ui.display_error(f"Failed to clean duplicates: {e}")
            
    def _debug_search(self, query: str):
        """Debug search functionality."""
        try:
            self.ui.console.print(f"[yellow]Debugging search for: '{query}'[/yellow]")
            
            # Search all knowledge without filters
            all_results = self.knowledge_base.search(query, n_results=10)
            
            self.ui.console.print(f"\n[cyan]All search results ({len(all_results)}):[/cyan]")
            for i, result in enumerate(all_results):
                self.ui.console.print(f"{i+1}. [bold]{result['metadata'].get('type', 'unknown')}[/bold]: {result['metadata'].get('name', 'unknown')}")
                self.ui.console.print(f"   Distance: {result.get('distance', 'N/A'):.3f}")
                self.ui.console.print(f"   Content: {result['content'][:100]}...")
                self.ui.console.print()
            
            # Search values specifically
            values_results = self.knowledge_base.search(query, n_results=5, filter_metadata={"type": {"$eq": "value"}})
            
            self.ui.console.print(f"[cyan]Values search results ({len(values_results)}):[/cyan]")
            for i, result in enumerate(values_results):
                self.ui.console.print(f"{i+1}. [bold]{result['metadata'].get('name', 'unknown')}[/bold]")
                self.ui.console.print(f"   Distance: {result.get('distance', 'N/A'):.3f}")
                self.ui.console.print(f"   Content: {result['content'][:200]}...")
                self.ui.console.print()
            
            # Test memory context
            memory_context = self.memory_system.get_context_for_query(query)
            self.ui.console.print(f"[cyan]Memory context:[/cyan]")
            self.ui.console.print(f"Relevant memories: {len(memory_context['relevant_memories'])}")
            
        except Exception as e:
            logger.error(f"Error in debug search: {e}")
            self.ui.display_error(f"Debug search failed: {e}")
            
    def _list_all_knowledge(self):
        """List all knowledge base items."""
        try:
            all_docs = self.knowledge_base.collection.get()
            
            self.ui.console.print(f"[yellow]All Knowledge Base Items ({len(all_docs['ids'])}):[/yellow]")
            
            from rich.table import Table
            table = Table()
            table.add_column("ID", style="dim")
            table.add_column("Type", style="cyan")
            table.add_column("Name", style="green")
            table.add_column("Source", style="magenta")
            table.add_column("Content Preview", style="white", max_width=50)
            
            for doc_id, content, metadata in zip(all_docs['ids'], all_docs['documents'], all_docs['metadatas']):
                if metadata:
                    table.add_row(
                        doc_id[:8] + "...",
                        metadata.get('type', 'unknown'),
                        metadata.get('name', 'unknown'),
                        metadata.get('source', 'unknown'),
                        content[:50] + "..." if len(content) > 50 else content
                    )
                    
            self.ui.console.print(table)
            
        except Exception as e:
            logger.error(f"Error listing knowledge: {e}")
            self.ui.display_error(f"Failed to list knowledge: {e}")
            
    def _show_document(self, doc_name: str):
        """Show full content of a specific document."""
        try:
            all_docs = self.knowledge_base.collection.get()
            
            found = False
            for doc_id, content, metadata in zip(all_docs['ids'], all_docs['documents'], all_docs['metadatas']):
                if metadata and metadata.get('name', '').lower() == doc_name.lower():
                    found = True
                    self.ui.console.print(f"[yellow]Document: {metadata.get('name', 'Unknown')}[/yellow]")
                    self.ui.console.print(f"[dim]ID: {doc_id}[/dim]")
                    self.ui.console.print(f"[dim]Type: {metadata.get('type', 'unknown')}[/dim]")
                    self.ui.console.print(f"[dim]Source: {metadata.get('source', 'unknown')}[/dim]")
                    self.ui.console.print()
                    self.ui.console.print("[cyan]Full Content:[/cyan]")
                    self.ui.console.print(content)
                    break
                    
            if not found:
                self.ui.display_error(f"Document '{doc_name}' not found")
                
        except Exception as e:
            logger.error(f"Error showing document: {e}")
            self.ui.display_error(f"Failed to show document: {e}")
            
    def _debug_agent_selection(self, query: str):
        """Debug which agents are selected for a query."""
        try:
            self.ui.console.print(f"[yellow]Agent selection for: '{query}'[/yellow]")
            
            from rich.table import Table
            table = Table()
            table.add_column("Agent", style="cyan")
            table.add_column("Confidence", style="green")
            table.add_column("Selected", style="magenta")
            
            # Get confidence scores from all agents
            agent_scores = []
            for agent in self.orchestrator.agents:
                if agent != self.orchestrator.planning_agent:  # Skip planning for this test
                    score = agent.can_handle(query)
                    agent_scores.append((agent, score))
                    
            # Sort by confidence
            agent_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Show selection logic
            active_agents = [(agent, score) for agent, score in agent_scores if score > 0.3]
            
            for agent, score in agent_scores:
                is_active = "✓" if score > 0.3 else ""
                is_primary = "PRIMARY" if active_agents and agent == active_agents[0][0] and score > 0.7 else ""
                
                table.add_row(
                    agent.name,
                    f"{score:.3f}",
                    f"{is_active} {is_primary}".strip()
                )
                
            self.ui.console.print(table)
            
            # Show what would happen
            if active_agents and active_agents[0][1] > 0.7:
                self.ui.console.print(f"[green]Would use: {active_agents[0][0].name} (primary)[/green]")
            elif active_agents:
                self.ui.console.print(f"[yellow]Would use: Reasoning Agent (synthesis)[/yellow]")
            else:
                self.ui.console.print(f"[red]Would use: Reasoning Agent (fallback)[/red]")
                
        except Exception as e:
            logger.error(f"Error debugging agent selection: {e}")
            self.ui.display_error(f"Failed to debug agents: {e}")
            
    def _add_quote(self, quote_text: str, author: str, context: str):
        """Add a new inspirational quote."""
        try:
            if not quote_text or not author:
                self.ui.display_error("Quote text and author are required")
                return
                
            # Use AI to help categorize the quote if no context provided
            if not context:
                context = input("Why does this quote resonate with you? ")
                
            with self.ui.show_thinking_indicator():
                # Get AI suggestions for categorization
                categorization = self.quotes_system.categorize_quote(quote_text, author, context)
                
                # Add the quote with AI suggestions
                quote = self.quotes_system.add_quote(
                    text=quote_text,
                    author=author,
                    context=context,
                    category=categorization.get("category", "inspiration"),
                    importance=categorization.get("importance", 5),
                    tags=set(categorization.get("tags", []))
                )
                
            self.ui.console.print(f"[green]✓ Added quote by {author}[/green]")
            self.ui.console.print(f"[dim]Category: {quote.category}[/dim]")
            self.ui.console.print(f"[dim]Importance: {quote.importance}/10[/dim]")
            self.ui.console.print(f"[dim]Tags: {', '.join(quote.tags)}[/dim]")
            
            if categorization.get("explanation"):
                self.ui.console.print(f"[dim]AI insight: {categorization['explanation']}[/dim]")
                
        except Exception as e:
            logger.error(f"Error adding quote: {e}")
            self.ui.display_error(f"Failed to add quote: {e}")
            
    def _reflect_on_quote(self):
        """Present a random quote for reflection."""
        try:
            quote = self.quotes_system.get_random_quote()
            
            if not quote:
                self.ui.console.print("[yellow]No quotes available. Add some quotes first with /add-quote[/yellow]")
                return
                
            reflection_prompt = self.quotes_system.get_reflection_prompt(quote)
            self.ui.console.print("[cyan]" + "="*60 + "[/cyan]")
            self.ui.console.print(reflection_prompt)
            self.ui.console.print("[cyan]" + "="*60 + "[/cyan]")
            
        except Exception as e:
            logger.error(f"Error during reflection: {e}")
            self.ui.display_error(f"Failed to get reflection quote: {e}")
            
    def _search_quotes(self, query: str):
        """Search quotes by content, author, or tags."""
        try:
            quotes = self.quotes_system.search_quotes(query, limit=8)
            
            if not quotes:
                self.ui.console.print(f"[yellow]No quotes found matching '{query}'[/yellow]")
                return
                
            from rich.table import Table
            table = Table(title=f"Quotes matching: '{query}'")
            table.add_column("Quote", style="white", max_width=40)
            table.add_column("Author", style="cyan")
            table.add_column("Category", style="magenta")
            table.add_column("Importance", style="yellow")
            table.add_column("Reflections", style="dim")
            
            for quote in quotes:
                table.add_row(
                    f'"{quote.text[:80]}{"..." if len(quote.text) > 80 else ""}"',
                    quote.author,
                    quote.category,
                    f"{quote.importance}/10",
                    str(quote.reflection_count)
                )
                
            self.ui.console.print(table)
            
        except Exception as e:
            logger.error(f"Error searching quotes: {e}")
            self.ui.display_error(f"Failed to search quotes: {e}")
            
    def _show_quotes_stats(self):
        """Show quotes system statistics."""
        try:
            stats = self.quotes_system.get_stats()
            
            if stats["total_quotes"] == 0:
                self.ui.console.print("[yellow]No quotes stored yet[/yellow]")
                return
                
            from rich.table import Table
            table = Table(title="Quotes Statistics")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            
            table.add_row("Total Quotes", str(stats["total_quotes"]))
            table.add_row("Average Importance", f"{stats['average_importance']:.1f}/10")
            table.add_row("Total Reflections", str(stats["total_reflections"]))
            
            if stats.get("recent_quotes", 0) > 0:
                table.add_row("Added This Week", str(stats["recent_quotes"]))
                
            self.ui.console.print(table)
            
            # Show category breakdown
            if stats.get("categories"):
                self.ui.console.print("\n[cyan]Categories:[/cyan]")
                for category, count in stats["categories"].items():
                    self.ui.console.print(f"  {category}: {count}")
                    
            # Show top authors
            if stats.get("top_authors"):
                self.ui.console.print("\n[cyan]Top Authors:[/cyan]")
                for author, count in list(stats["top_authors"].items())[:5]:
                    self.ui.console.print(f"  {author}: {count}")
                    
        except Exception as e:
            logger.error(f"Error showing quotes stats: {e}")
            self.ui.display_error(f"Failed to show quotes stats: {e}")

    def _rate_last_response(self, rating_str: Optional[str] = None):
        """Rate the last assistant response."""
        try:
            # Get the last assistant message
            msg_pair = self.conversation.get_message_pair_for_feedback()

            if not msg_pair:
                self.ui.console.print("[yellow]No response to rate yet[/yellow]")
                return

            # Show what we're rating
            response_preview = msg_pair["assistant_content"][:200]
            if len(msg_pair["assistant_content"]) > 200:
                response_preview += "..."

            self.ui.console.print(f"[cyan]Rating response:[/cyan]")
            self.ui.console.print(f"[dim]{response_preview}[/dim]")
            self.ui.console.print()

            # Parse rating if provided
            if rating_str:
                if rating_str in ["+", "up", "👍", "good"]:
                    feedback = self.feedback_manager.add_thumbs_up(
                        conversation_id=self.conversation.conversation_id,
                        message_index=msg_pair["assistant_index"],
                        user_query=msg_pair["user_content"],
                        assistant_response=msg_pair["assistant_content"]
                    )
                    self.ui.console.print("[green]✓ Marked as 👍 (thumbs up)[/green]")
                    return

                elif rating_str in ["-", "down", "👎", "bad"]:
                    # Ask for optional feedback
                    text_feedback = input("What could be improved? (press Enter to skip): ").strip()
                    feedback = self.feedback_manager.add_thumbs_down(
                        conversation_id=self.conversation.conversation_id,
                        message_index=msg_pair["assistant_index"],
                        text_feedback=text_feedback if text_feedback else None,
                        user_query=msg_pair["user_content"],
                        assistant_response=msg_pair["assistant_content"]
                    )
                    self.ui.console.print("[yellow]✓ Marked as 👎 (thumbs down)[/yellow]")
                    return

                elif rating_str.isdigit() and 1 <= int(rating_str) <= 5:
                    rating = int(rating_str)
                    text_feedback = None
                    if rating <= 2:
                        text_feedback = input("What could be improved? (press Enter to skip): ").strip()

                    feedback = self.feedback_manager.add_rating(
                        conversation_id=self.conversation.conversation_id,
                        message_index=msg_pair["assistant_index"],
                        rating=rating,
                        text_feedback=text_feedback if text_feedback else None,
                        user_query=msg_pair["user_content"],
                        assistant_response=msg_pair["assistant_content"]
                    )
                    stars = "⭐" * rating + "☆" * (5 - rating)
                    self.ui.console.print(f"[green]✓ Rated {stars} ({rating}/5)[/green]")
                    return

            # Interactive rating mode
            self.ui.console.print("[yellow]How would you rate this response?[/yellow]")
            self.ui.console.print("  1. 👍 Thumbs up (good response)")
            self.ui.console.print("  2. 👎 Thumbs down (needs improvement)")
            self.ui.console.print("  3. Rate 1-5 stars")
            self.ui.console.print("  4. Cancel")

            try:
                choice = input("Select option (1-4): ").strip()

                if choice == "1":
                    feedback = self.feedback_manager.add_thumbs_up(
                        conversation_id=self.conversation.conversation_id,
                        message_index=msg_pair["assistant_index"],
                        user_query=msg_pair["user_content"],
                        assistant_response=msg_pair["assistant_content"]
                    )
                    self.ui.console.print("[green]✓ Marked as 👍 (thumbs up)[/green]")

                elif choice == "2":
                    text_feedback = input("What could be improved? (press Enter to skip): ").strip()
                    feedback = self.feedback_manager.add_thumbs_down(
                        conversation_id=self.conversation.conversation_id,
                        message_index=msg_pair["assistant_index"],
                        text_feedback=text_feedback if text_feedback else None,
                        user_query=msg_pair["user_content"],
                        assistant_response=msg_pair["assistant_content"]
                    )
                    self.ui.console.print("[yellow]✓ Marked as 👎 (thumbs down)[/yellow]")

                elif choice == "3":
                    rating_input = input("Enter rating (1-5): ").strip()
                    if rating_input.isdigit() and 1 <= int(rating_input) <= 5:
                        rating = int(rating_input)
                        text_feedback = None
                        if rating <= 2:
                            text_feedback = input("What could be improved? (press Enter to skip): ").strip()

                        feedback = self.feedback_manager.add_rating(
                            conversation_id=self.conversation.conversation_id,
                            message_index=msg_pair["assistant_index"],
                            rating=rating,
                            text_feedback=text_feedback if text_feedback else None,
                            user_query=msg_pair["user_content"],
                            assistant_response=msg_pair["assistant_content"]
                        )
                        stars = "⭐" * rating + "☆" * (5 - rating)
                        self.ui.console.print(f"[green]✓ Rated {stars} ({rating}/5)[/green]")
                    else:
                        self.ui.display_error("Invalid rating. Please enter 1-5")

                else:
                    self.ui.console.print("[dim]Cancelled[/dim]")

            except (KeyboardInterrupt, EOFError):
                self.ui.console.print("\n[dim]Cancelled[/dim]")

        except Exception as e:
            logger.error(f"Error rating response: {e}")
            self.ui.display_error(f"Failed to rate response: {e}")

    def _show_feedback_stats(self):
        """Show feedback statistics."""
        try:
            stats = self.feedback_manager.get_statistics()

            if stats["total_feedback"] == 0:
                self.ui.console.print("[yellow]No feedback recorded yet[/yellow]")
                self.ui.console.print("[dim]Use /rate to provide feedback on responses[/dim]")
                return

            from rich.table import Table
            table = Table(title="Response Feedback Statistics")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")

            table.add_row("Total Feedback", str(stats["total_feedback"]))
            table.add_row("Positive Responses", f"{stats['positive_count']} ({stats['positive_rate']*100:.1f}%)")
            table.add_row("Negative Responses", str(stats["negative_count"]))
            table.add_row("Average Score", f"{stats['average_score']*100:.1f}%")
            table.add_row("With Text Feedback", str(stats["feedback_with_text"]))

            self.ui.console.print(table)

            # Show breakdown by type
            if stats["by_type"]:
                self.ui.console.print("\n[cyan]By Feedback Type:[/cyan]")
                for type_name, data in stats["by_type"].items():
                    self.ui.console.print(f"  {type_name}: {data['count']} (avg: {data['avg_rating']:.1f})")

        except Exception as e:
            logger.error(f"Error showing feedback stats: {e}")
            self.ui.display_error(f"Failed to show feedback stats: {e}")

    def _show_best_responses(self):
        """Show the best-rated responses."""
        try:
            best = self.feedback_manager.get_best_responses(limit=10)

            if not best:
                self.ui.console.print("[yellow]No positive feedback recorded yet[/yellow]")
                return

            from rich.table import Table
            table = Table(title="Best Rated Responses")
            table.add_column("Query", style="cyan", max_width=30)
            table.add_column("Response", style="white", max_width=40)
            table.add_column("Rating", style="green")
            table.add_column("Date", style="dim")

            for fb in best:
                query = (fb.user_query or "N/A")[:60]
                if len(fb.user_query or "") > 60:
                    query += "..."

                response = (fb.assistant_response or "N/A")[:80]
                if len(fb.assistant_response or "") > 80:
                    response += "..."

                if fb.feedback_type == FeedbackType.RATING_1_5:
                    rating_str = "⭐" * fb.rating
                else:
                    rating_str = "👍" if fb.is_positive else "👎"

                table.add_row(
                    query,
                    response,
                    rating_str,
                    fb.timestamp.strftime("%Y-%m-%d")
                )

            self.ui.console.print(table)

        except Exception as e:
            logger.error(f"Error showing best responses: {e}")
            self.ui.display_error(f"Failed to show best responses: {e}")

    def _show_worst_responses(self):
        """Show the worst-rated responses."""
        try:
            worst = self.feedback_manager.get_worst_responses(limit=10)

            if not worst:
                self.ui.console.print("[yellow]No negative feedback recorded yet[/yellow]")
                return

            from rich.table import Table
            table = Table(title="Lowest Rated Responses (Improvement Opportunities)")
            table.add_column("Query", style="cyan", max_width=30)
            table.add_column("Response", style="white", max_width=35)
            table.add_column("Rating", style="red")
            table.add_column("Feedback", style="yellow", max_width=25)

            for fb in worst:
                query = (fb.user_query or "N/A")[:60]
                if len(fb.user_query or "") > 60:
                    query += "..."

                response = (fb.assistant_response or "N/A")[:70]
                if len(fb.assistant_response or "") > 70:
                    response += "..."

                if fb.feedback_type == FeedbackType.RATING_1_5:
                    rating_str = "⭐" * fb.rating + "☆" * (5 - fb.rating)
                else:
                    rating_str = "👎"

                feedback_text = (fb.text_feedback or "")[:50]
                if len(fb.text_feedback or "") > 50:
                    feedback_text += "..."

                table.add_row(
                    query,
                    response,
                    rating_str,
                    feedback_text
                )

            self.ui.console.print(table)

        except Exception as e:
            logger.error(f"Error showing worst responses: {e}")
            self.ui.display_error(f"Failed to show worst responses: {e}")

    def _show_feedback_insights(self):
        """Show insights from collected feedback."""
        try:
            insights = self.feedback_manager.get_improvement_insights()
            stats = self.feedback_manager.get_statistics()

            if stats["total_feedback"] == 0:
                self.ui.console.print("[yellow]No feedback recorded yet[/yellow]")
                return

            self.ui.console.print("[cyan]" + "="*60 + "[/cyan]")
            self.ui.console.print("[bold cyan]Feedback Insights[/bold cyan]")
            self.ui.console.print("[cyan]" + "="*60 + "[/cyan]")

            # Summary
            self.ui.console.print(f"\n[green]Overall Performance:[/green]")
            self.ui.console.print(f"  • {stats['positive_count']}/{stats['total_feedback']} responses rated positively ({stats['positive_rate']*100:.1f}%)")
            self.ui.console.print(f"  • Average satisfaction score: {stats['average_score']*100:.1f}%")

            # Improvement suggestions
            if insights["common_issues"]:
                self.ui.console.print(f"\n[yellow]Areas for Improvement ({len(insights['common_issues'])} feedback items):[/yellow]")
                for i, issue in enumerate(insights["common_issues"][:5], 1):
                    self.ui.console.print(f"  {i}. {issue}")

            # Successful patterns
            if insights["successful_patterns"]:
                self.ui.console.print(f"\n[green]What's Working Well:[/green]")
                for i, pattern in enumerate(insights["successful_patterns"][:5], 1):
                    self.ui.console.print(f"  {i}. {pattern}")

            # Few-shot examples available
            examples = self.feedback_manager.get_few_shot_examples(limit=3)
            if examples:
                self.ui.console.print(f"\n[cyan]Top-Rated Response Examples Available:[/cyan] {len(examples)}")
                self.ui.console.print("[dim]These can be used as few-shot examples for future responses[/dim]")

            self.ui.console.print("[cyan]" + "="*60 + "[/cyan]")

        except Exception as e:
            logger.error(f"Error showing feedback insights: {e}")
            self.ui.display_error(f"Failed to show feedback insights: {e}")

    def _export_knowledge(self, filename: Optional[str] = None):
        """Export the knowledge base to a JSON file."""
        try:
            # Generate default filename if not provided
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"knowledge_backup_{timestamp}.json"

            # Ask for optional filters
            self.ui.console.print("[yellow]Export Options:[/yellow]")
            self.ui.console.print("1. Export all documents")
            self.ui.console.print("2. Export by type (value, framework, preference, memory, etc.)")
            self.ui.console.print("3. Export by source (knowledge_files, web, etc.)")

            try:
                choice = input("Select option (1-3, default 1): ").strip() or "1"

                filter_type = None
                filter_source = None

                if choice == "2":
                    # Show available types
                    stats = self.knowledge_base.get_stats()
                    self.ui.console.print(f"[dim]Available types: {', '.join(stats['types'].keys())}[/dim]")
                    filter_type = input("Enter type to export: ").strip()
                elif choice == "3":
                    filter_source = input("Enter source to export: ").strip()

                with self.ui.show_thinking_indicator():
                    export_data = self.knowledge_base.export_knowledge(
                        filter_type=filter_type,
                        filter_source=filter_source
                    )

                # Write to file
                output_path = Path(filename)
                with open(output_path, 'w') as f:
                    json.dump(export_data, f, indent=2)

                self.ui.console.print(f"[green]✓ Exported {export_data['total_documents']} documents to {filename}[/green]")
                self.ui.console.print(f"[dim]Export timestamp: {export_data['export_timestamp']}[/dim]")

                if filter_type or filter_source:
                    self.ui.console.print(f"[dim]Filters applied - Type: {filter_type}, Source: {filter_source}[/dim]")

            except KeyboardInterrupt:
                self.ui.console.print("\n[dim]Export cancelled[/dim]")

        except Exception as e:
            logger.error(f"Error exporting knowledge: {e}")
            self.ui.display_error(f"Failed to export knowledge: {e}")

    def _import_knowledge(self, filename: str):
        """Import knowledge from a JSON backup file."""
        try:
            input_path = Path(filename)

            if not input_path.exists():
                self.ui.display_error(f"File not found: {filename}")
                return

            # Load the file
            with open(input_path, 'r') as f:
                import_data = json.load(f)

            # Show import info
            self.ui.console.print(f"[yellow]Import File: {filename}[/yellow]")
            self.ui.console.print(f"[dim]Export version: {import_data.get('export_version', 'unknown')}[/dim]")
            self.ui.console.print(f"[dim]Export timestamp: {import_data.get('export_timestamp', 'unknown')}[/dim]")
            self.ui.console.print(f"[dim]Documents in file: {import_data.get('total_documents', len(import_data.get('documents', [])))}[/dim]")

            # Ask for duplicate handling
            self.ui.console.print("\n[yellow]Duplicate Handling:[/yellow]")
            self.ui.console.print("1. Skip - Don't import documents with existing IDs (default)")
            self.ui.console.print("2. Update - Replace existing documents with import data")
            self.ui.console.print("3. Fail - Stop import if any duplicates found")

            try:
                choice = input("Select option (1-3, default 1): ").strip() or "1"

                duplicate_map = {"1": "skip", "2": "update", "3": "fail"}
                duplicate_handling = duplicate_map.get(choice, "skip")

                # Confirm import
                confirm = input(f"Proceed with import ({duplicate_handling} duplicates)? [y/N]: ").strip().lower()
                if confirm != 'y':
                    self.ui.console.print("[dim]Import cancelled[/dim]")
                    return

                with self.ui.show_thinking_indicator():
                    stats = self.knowledge_base.import_knowledge(
                        import_data,
                        duplicate_handling=duplicate_handling
                    )

                # Show results
                self.ui.console.print(f"[green]✓ Import complete![/green]")
                self.ui.console.print(f"  Documents in file: {stats['total_in_file']}")
                self.ui.console.print(f"  Added: {stats['added']}")
                self.ui.console.print(f"  Updated: {stats['updated']}")
                self.ui.console.print(f"  Skipped: {stats['skipped']}")

                if stats['errors'] > 0:
                    self.ui.console.print(f"[yellow]  Errors: {stats['errors']}[/yellow]")
                    for error_msg in stats['error_messages'][:5]:  # Show first 5 errors
                        self.ui.console.print(f"[dim]    - {error_msg}[/dim]")

                # Show new stats
                new_stats = self.knowledge_base.get_stats()
                self.ui.console.print(f"\n[dim]Knowledge base now has {new_stats['total_documents']} total documents[/dim]")

            except KeyboardInterrupt:
                self.ui.console.print("\n[dim]Import cancelled[/dim]")

        except json.JSONDecodeError as e:
            self.ui.display_error(f"Invalid JSON file: {e}")
        except Exception as e:
            logger.error(f"Error importing knowledge: {e}")
            self.ui.display_error(f"Failed to import knowledge: {e}")

    def run(self):
        """Main application loop."""
        self.initialize()
        
        while True:
            try:
                user_input = self.ui.get_user_input()
                
                if not user_input:
                    continue
                    
                # Check if it's a command
                if user_input.startswith("/"):
                    if not self.handle_command(user_input):
                        break
                else:
                    # Process as a query
                    self.process_query(user_input)
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                self.ui.display_error(f"Unexpected error: {e}")
                
        self.ui.console.print("\n[blue]Goodbye! Thanks for collaborating.[/blue]")

def main():
    """Entry point."""
    bot = GlennBot()
    bot.run()

if __name__ == "__main__":
    main()