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
from src.document_ingestion import DocumentIngestionTool
from src.memory_system import MemorySystem, MemoryType
from src.quotes_system import QuotesSystem

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
        
        else:
            self.ui.display_error(f"Unknown command: {command}")
            
        return True
        
    def process_query(self, query: str):
        """Process a user query through the agent system."""
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