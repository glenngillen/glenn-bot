"""Tests for library CLI commands in GlennBot."""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestBuildLibraryCommand:
    """Tests for the /build-library command handler."""

    def test_build_library_command_calls_builder(self, temp_dir):
        """Test that /build-library command invokes LibraryBuilder.build()."""
        # Import inside test with patches already active
        with patch.dict("sys.modules", {"chromadb": MagicMock()}):
            with patch("src.config.settings") as mock_settings:
                # Set up mock settings
                mock_settings.log_level = "WARNING"
                mock_settings.knowledge_dir = temp_dir / "knowledge"
                mock_settings.conversation_history_dir = temp_dir / "conversations"
                mock_settings.chroma_persist_directory = temp_dir / "chroma"
                mock_settings.chroma_collection_name = "test_collection"
                mock_settings.library_data_dir = temp_dir / "library"
                mock_settings.library_site_dir = temp_dir / "library-site"
                mock_settings.library_server_port = 8080
                mock_settings.ollama_host = "http://localhost:11434"
                mock_settings.ollama_model = "llama3:8b"
                mock_settings.ollama_embedding_model = "nomic-embed-text"
                mock_settings.context_persistence_dir = temp_dir / "contexts"
                mock_settings.memory_persistence_file = temp_dir / "memories.json"
                mock_settings.quotes_persistence_file = temp_dir / "quotes.json"
                mock_settings.feedback_persistence_dir = temp_dir / "feedback"

                # Create necessary directories
                mock_settings.knowledge_dir.mkdir(parents=True, exist_ok=True)
                mock_settings.library_data_dir.mkdir(parents=True, exist_ok=True)
                mock_settings.library_site_dir.mkdir(parents=True, exist_ok=True)

                with patch("src.knowledge_base.KnowledgeBase") as mock_kb_class, \
                     patch("src.ollama_client.OllamaClient") as mock_ollama_class, \
                     patch("src.terminal_ui.TerminalUI") as mock_ui_class, \
                     patch("src.conversation.ConversationManager"), \
                     patch("src.memory_system.MemorySystem") as mock_mem_class, \
                     patch("src.quotes_system.QuotesSystem"), \
                     patch("src.agents.AgentOrchestrator"), \
                     patch("src.document_ingestion.DocumentIngestionTool"), \
                     patch("src.feedback_system.FeedbackManager"), \
                     patch("src.context_detector.ContextDetector"):

                    # Set up mock returns
                    mock_kb = MagicMock()
                    mock_kb.get_stats.return_value = {"total_documents": 10, "types": {}}
                    mock_kb_class.return_value = mock_kb

                    mock_ollama = MagicMock()
                    mock_ollama_class.return_value = mock_ollama

                    mock_mem = MagicMock()
                    mock_mem.auto_switch_threshold = 0.8
                    mock_mem.prompt_threshold = 0.5
                    mock_mem_class.return_value = mock_mem

                    mock_ui = MagicMock()
                    mock_ui.show_thinking_indicator.return_value.__enter__ = MagicMock()
                    mock_ui.show_thinking_indicator.return_value.__exit__ = MagicMock()
                    mock_ui_class.return_value = mock_ui

                    from src.main import GlennBot
                    bot = GlennBot()
                    bot.ui = mock_ui
                    bot.knowledge_base = mock_kb
                    bot.ollama_client = mock_ollama

                    with patch("src.main.LibraryBuilder") as mock_builder_class:
                        mock_builder = MagicMock()
                        mock_builder.build.return_value = {
                            "items_count": 5,
                            "themes_count": 3,
                            "pages_generated": 12,
                            "last_build": "2024-01-15T10:00:00",
                        }
                        mock_builder_class.return_value = mock_builder

                        # Call the command
                        result = bot.handle_command("/build-library")

                        # Verify builder was created with correct arguments
                        mock_builder_class.assert_called_once_with(
                            bot.knowledge_base,
                            bot.ollama_client,
                            mock_settings.library_data_dir,
                            mock_settings.library_site_dir,
                        )

                        # Verify build was called
                        mock_builder.build.assert_called_once_with(force=False)

                        # Verify command returns True (continue)
                        assert result is True

    def test_build_library_displays_success_message(self, temp_dir):
        """Test that /build-library displays build statistics on success."""
        with patch.dict("sys.modules", {"chromadb": MagicMock()}):
            with patch("src.config.settings") as mock_settings:
                mock_settings.log_level = "WARNING"
                mock_settings.knowledge_dir = temp_dir / "knowledge"
                mock_settings.conversation_history_dir = temp_dir / "conversations"
                mock_settings.chroma_persist_directory = temp_dir / "chroma"
                mock_settings.chroma_collection_name = "test_collection"
                mock_settings.library_data_dir = temp_dir / "library"
                mock_settings.library_site_dir = temp_dir / "library-site"
                mock_settings.library_server_port = 8080
                mock_settings.ollama_host = "http://localhost:11434"
                mock_settings.ollama_model = "llama3:8b"
                mock_settings.ollama_embedding_model = "nomic-embed-text"
                mock_settings.context_persistence_dir = temp_dir / "contexts"
                mock_settings.memory_persistence_file = temp_dir / "memories.json"
                mock_settings.quotes_persistence_file = temp_dir / "quotes.json"
                mock_settings.feedback_persistence_dir = temp_dir / "feedback"

                mock_settings.knowledge_dir.mkdir(parents=True, exist_ok=True)
                mock_settings.library_data_dir.mkdir(parents=True, exist_ok=True)
                mock_settings.library_site_dir.mkdir(parents=True, exist_ok=True)

                with patch("src.knowledge_base.KnowledgeBase") as mock_kb_class, \
                     patch("src.ollama_client.OllamaClient") as mock_ollama_class, \
                     patch("src.terminal_ui.TerminalUI") as mock_ui_class, \
                     patch("src.conversation.ConversationManager"), \
                     patch("src.memory_system.MemorySystem") as mock_mem_class, \
                     patch("src.quotes_system.QuotesSystem"), \
                     patch("src.agents.AgentOrchestrator"), \
                     patch("src.document_ingestion.DocumentIngestionTool"), \
                     patch("src.feedback_system.FeedbackManager"), \
                     patch("src.context_detector.ContextDetector"):

                    mock_kb = MagicMock()
                    mock_kb.get_stats.return_value = {"total_documents": 10, "types": {}}
                    mock_kb_class.return_value = mock_kb

                    mock_ollama = MagicMock()
                    mock_ollama_class.return_value = mock_ollama

                    mock_mem = MagicMock()
                    mock_mem.auto_switch_threshold = 0.8
                    mock_mem.prompt_threshold = 0.5
                    mock_mem_class.return_value = mock_mem

                    mock_ui = MagicMock()
                    mock_ui.show_thinking_indicator.return_value.__enter__ = MagicMock()
                    mock_ui.show_thinking_indicator.return_value.__exit__ = MagicMock()
                    mock_ui_class.return_value = mock_ui

                    from src.main import GlennBot
                    bot = GlennBot()
                    bot.ui = mock_ui
                    bot.knowledge_base = mock_kb
                    bot.ollama_client = mock_ollama

                    with patch("src.main.LibraryBuilder") as mock_builder_class:
                        mock_builder = MagicMock()
                        mock_builder.build.return_value = {
                            "items_count": 5,
                            "themes_count": 3,
                            "pages_generated": 12,
                            "last_build": "2024-01-15T10:00:00",
                        }
                        mock_builder_class.return_value = mock_builder

                        bot.handle_command("/build-library")

                        # Verify success message was displayed
                        bot.ui.console.print.assert_called()
                        print_calls = [str(c) for c in bot.ui.console.print.call_args_list]
                        assert any("5" in c and "item" in c.lower() for c in print_calls) or \
                               any("green" in c.lower() for c in print_calls)

    def test_build_library_shows_thinking_indicator(self, temp_dir):
        """Test that /build-library shows the thinking indicator during build."""
        with patch.dict("sys.modules", {"chromadb": MagicMock()}):
            with patch("src.config.settings") as mock_settings:
                mock_settings.log_level = "WARNING"
                mock_settings.knowledge_dir = temp_dir / "knowledge"
                mock_settings.conversation_history_dir = temp_dir / "conversations"
                mock_settings.chroma_persist_directory = temp_dir / "chroma"
                mock_settings.chroma_collection_name = "test_collection"
                mock_settings.library_data_dir = temp_dir / "library"
                mock_settings.library_site_dir = temp_dir / "library-site"
                mock_settings.library_server_port = 8080
                mock_settings.ollama_host = "http://localhost:11434"
                mock_settings.ollama_model = "llama3:8b"
                mock_settings.ollama_embedding_model = "nomic-embed-text"
                mock_settings.context_persistence_dir = temp_dir / "contexts"
                mock_settings.memory_persistence_file = temp_dir / "memories.json"
                mock_settings.quotes_persistence_file = temp_dir / "quotes.json"
                mock_settings.feedback_persistence_dir = temp_dir / "feedback"

                mock_settings.knowledge_dir.mkdir(parents=True, exist_ok=True)
                mock_settings.library_data_dir.mkdir(parents=True, exist_ok=True)
                mock_settings.library_site_dir.mkdir(parents=True, exist_ok=True)

                with patch("src.knowledge_base.KnowledgeBase") as mock_kb_class, \
                     patch("src.ollama_client.OllamaClient") as mock_ollama_class, \
                     patch("src.terminal_ui.TerminalUI") as mock_ui_class, \
                     patch("src.conversation.ConversationManager"), \
                     patch("src.memory_system.MemorySystem") as mock_mem_class, \
                     patch("src.quotes_system.QuotesSystem"), \
                     patch("src.agents.AgentOrchestrator"), \
                     patch("src.document_ingestion.DocumentIngestionTool"), \
                     patch("src.feedback_system.FeedbackManager"), \
                     patch("src.context_detector.ContextDetector"):

                    mock_kb = MagicMock()
                    mock_kb.get_stats.return_value = {"total_documents": 10, "types": {}}
                    mock_kb_class.return_value = mock_kb

                    mock_ollama = MagicMock()
                    mock_ollama_class.return_value = mock_ollama

                    mock_mem = MagicMock()
                    mock_mem.auto_switch_threshold = 0.8
                    mock_mem.prompt_threshold = 0.5
                    mock_mem_class.return_value = mock_mem

                    mock_ui = MagicMock()
                    mock_ui.show_thinking_indicator.return_value.__enter__ = MagicMock()
                    mock_ui.show_thinking_indicator.return_value.__exit__ = MagicMock()
                    mock_ui_class.return_value = mock_ui

                    from src.main import GlennBot
                    bot = GlennBot()
                    bot.ui = mock_ui
                    bot.knowledge_base = mock_kb
                    bot.ollama_client = mock_ollama

                    with patch("src.main.LibraryBuilder") as mock_builder_class:
                        mock_builder = MagicMock()
                        mock_builder.build.return_value = {
                            "items_count": 5,
                            "themes_count": 3,
                            "pages_generated": 12,
                            "last_build": "2024-01-15T10:00:00",
                        }
                        mock_builder_class.return_value = mock_builder

                        bot.handle_command("/build-library")

                        # Verify thinking indicator was used
                        bot.ui.show_thinking_indicator.assert_called()

    def test_build_library_handles_build_error(self, temp_dir):
        """Test that /build-library handles errors gracefully."""
        with patch.dict("sys.modules", {"chromadb": MagicMock()}):
            with patch("src.config.settings") as mock_settings:
                mock_settings.log_level = "WARNING"
                mock_settings.knowledge_dir = temp_dir / "knowledge"
                mock_settings.conversation_history_dir = temp_dir / "conversations"
                mock_settings.chroma_persist_directory = temp_dir / "chroma"
                mock_settings.chroma_collection_name = "test_collection"
                mock_settings.library_data_dir = temp_dir / "library"
                mock_settings.library_site_dir = temp_dir / "library-site"
                mock_settings.library_server_port = 8080
                mock_settings.ollama_host = "http://localhost:11434"
                mock_settings.ollama_model = "llama3:8b"
                mock_settings.ollama_embedding_model = "nomic-embed-text"
                mock_settings.context_persistence_dir = temp_dir / "contexts"
                mock_settings.memory_persistence_file = temp_dir / "memories.json"
                mock_settings.quotes_persistence_file = temp_dir / "quotes.json"
                mock_settings.feedback_persistence_dir = temp_dir / "feedback"

                mock_settings.knowledge_dir.mkdir(parents=True, exist_ok=True)
                mock_settings.library_data_dir.mkdir(parents=True, exist_ok=True)
                mock_settings.library_site_dir.mkdir(parents=True, exist_ok=True)

                with patch("src.knowledge_base.KnowledgeBase") as mock_kb_class, \
                     patch("src.ollama_client.OllamaClient") as mock_ollama_class, \
                     patch("src.terminal_ui.TerminalUI") as mock_ui_class, \
                     patch("src.conversation.ConversationManager"), \
                     patch("src.memory_system.MemorySystem") as mock_mem_class, \
                     patch("src.quotes_system.QuotesSystem"), \
                     patch("src.agents.AgentOrchestrator"), \
                     patch("src.document_ingestion.DocumentIngestionTool"), \
                     patch("src.feedback_system.FeedbackManager"), \
                     patch("src.context_detector.ContextDetector"):

                    mock_kb = MagicMock()
                    mock_kb.get_stats.return_value = {"total_documents": 10, "types": {}}
                    mock_kb_class.return_value = mock_kb

                    mock_ollama = MagicMock()
                    mock_ollama_class.return_value = mock_ollama

                    mock_mem = MagicMock()
                    mock_mem.auto_switch_threshold = 0.8
                    mock_mem.prompt_threshold = 0.5
                    mock_mem_class.return_value = mock_mem

                    mock_ui = MagicMock()
                    mock_ui.show_thinking_indicator.return_value.__enter__ = MagicMock()
                    mock_ui.show_thinking_indicator.return_value.__exit__ = MagicMock()
                    mock_ui_class.return_value = mock_ui

                    from src.main import GlennBot
                    bot = GlennBot()
                    bot.ui = mock_ui
                    bot.knowledge_base = mock_kb
                    bot.ollama_client = mock_ollama

                    with patch("src.main.LibraryBuilder") as mock_builder_class:
                        mock_builder = MagicMock()
                        mock_builder.build.side_effect = Exception("Build failed")
                        mock_builder_class.return_value = mock_builder

                        # Should not raise, should display error
                        result = bot.handle_command("/build-library")

                        # Verify error was displayed
                        bot.ui.display_error.assert_called()
                        error_call = str(bot.ui.display_error.call_args)
                        assert "Build failed" in error_call or "Failed" in error_call

                        # Command should still return True (continue)
                        assert result is True


class TestBuildLibraryForceFlag:
    """Tests for the /build-library --force flag."""

    def test_build_library_force_flag_passes_force_true(self, temp_dir):
        """Test that /build-library --force passes force=True to builder."""
        with patch.dict("sys.modules", {"chromadb": MagicMock()}):
            with patch("src.config.settings") as mock_settings:
                mock_settings.log_level = "WARNING"
                mock_settings.knowledge_dir = temp_dir / "knowledge"
                mock_settings.conversation_history_dir = temp_dir / "conversations"
                mock_settings.chroma_persist_directory = temp_dir / "chroma"
                mock_settings.chroma_collection_name = "test_collection"
                mock_settings.library_data_dir = temp_dir / "library"
                mock_settings.library_site_dir = temp_dir / "library-site"
                mock_settings.library_server_port = 8080
                mock_settings.ollama_host = "http://localhost:11434"
                mock_settings.ollama_model = "llama3:8b"
                mock_settings.ollama_embedding_model = "nomic-embed-text"
                mock_settings.context_persistence_dir = temp_dir / "contexts"
                mock_settings.memory_persistence_file = temp_dir / "memories.json"
                mock_settings.quotes_persistence_file = temp_dir / "quotes.json"
                mock_settings.feedback_persistence_dir = temp_dir / "feedback"

                mock_settings.knowledge_dir.mkdir(parents=True, exist_ok=True)
                mock_settings.library_data_dir.mkdir(parents=True, exist_ok=True)
                mock_settings.library_site_dir.mkdir(parents=True, exist_ok=True)

                with patch("src.knowledge_base.KnowledgeBase") as mock_kb_class, \
                     patch("src.ollama_client.OllamaClient") as mock_ollama_class, \
                     patch("src.terminal_ui.TerminalUI") as mock_ui_class, \
                     patch("src.conversation.ConversationManager"), \
                     patch("src.memory_system.MemorySystem") as mock_mem_class, \
                     patch("src.quotes_system.QuotesSystem"), \
                     patch("src.agents.AgentOrchestrator"), \
                     patch("src.document_ingestion.DocumentIngestionTool"), \
                     patch("src.feedback_system.FeedbackManager"), \
                     patch("src.context_detector.ContextDetector"):

                    mock_kb = MagicMock()
                    mock_kb.get_stats.return_value = {"total_documents": 10, "types": {}}
                    mock_kb_class.return_value = mock_kb

                    mock_ollama = MagicMock()
                    mock_ollama_class.return_value = mock_ollama

                    mock_mem = MagicMock()
                    mock_mem.auto_switch_threshold = 0.8
                    mock_mem.prompt_threshold = 0.5
                    mock_mem_class.return_value = mock_mem

                    mock_ui = MagicMock()
                    mock_ui.show_thinking_indicator.return_value.__enter__ = MagicMock()
                    mock_ui.show_thinking_indicator.return_value.__exit__ = MagicMock()
                    mock_ui_class.return_value = mock_ui

                    from src.main import GlennBot
                    bot = GlennBot()
                    bot.ui = mock_ui
                    bot.knowledge_base = mock_kb
                    bot.ollama_client = mock_ollama

                    with patch("src.main.LibraryBuilder") as mock_builder_class:
                        mock_builder = MagicMock()
                        mock_builder.build.return_value = {
                            "items_count": 5,
                            "themes_count": 3,
                            "pages_generated": 12,
                            "last_build": "2024-01-15T10:00:00",
                        }
                        mock_builder_class.return_value = mock_builder

                        bot.handle_command("/build-library --force")

                        # Verify build was called with force=True
                        mock_builder.build.assert_called_once_with(force=True)


class TestBuildLibraryServeFlag:
    """Tests for the /build-library --serve flag."""

    def test_build_library_serve_flag_starts_server_after_build(self, temp_dir):
        """Test that /build-library --serve starts the server after building."""
        with patch.dict("sys.modules", {"chromadb": MagicMock()}):
            with patch("src.config.settings") as mock_settings:
                mock_settings.log_level = "WARNING"
                mock_settings.knowledge_dir = temp_dir / "knowledge"
                mock_settings.conversation_history_dir = temp_dir / "conversations"
                mock_settings.chroma_persist_directory = temp_dir / "chroma"
                mock_settings.chroma_collection_name = "test_collection"
                mock_settings.library_data_dir = temp_dir / "library"
                mock_settings.library_site_dir = temp_dir / "library-site"
                mock_settings.library_server_port = 8080
                mock_settings.ollama_host = "http://localhost:11434"
                mock_settings.ollama_model = "llama3:8b"
                mock_settings.ollama_embedding_model = "nomic-embed-text"
                mock_settings.context_persistence_dir = temp_dir / "contexts"
                mock_settings.memory_persistence_file = temp_dir / "memories.json"
                mock_settings.quotes_persistence_file = temp_dir / "quotes.json"
                mock_settings.feedback_persistence_dir = temp_dir / "feedback"

                mock_settings.knowledge_dir.mkdir(parents=True, exist_ok=True)
                mock_settings.library_data_dir.mkdir(parents=True, exist_ok=True)
                mock_settings.library_site_dir.mkdir(parents=True, exist_ok=True)

                with patch("src.knowledge_base.KnowledgeBase") as mock_kb_class, \
                     patch("src.ollama_client.OllamaClient") as mock_ollama_class, \
                     patch("src.terminal_ui.TerminalUI") as mock_ui_class, \
                     patch("src.conversation.ConversationManager"), \
                     patch("src.memory_system.MemorySystem") as mock_mem_class, \
                     patch("src.quotes_system.QuotesSystem"), \
                     patch("src.agents.AgentOrchestrator"), \
                     patch("src.document_ingestion.DocumentIngestionTool"), \
                     patch("src.feedback_system.FeedbackManager"), \
                     patch("src.context_detector.ContextDetector"):

                    mock_kb = MagicMock()
                    mock_kb.get_stats.return_value = {"total_documents": 10, "types": {}}
                    mock_kb_class.return_value = mock_kb

                    mock_ollama = MagicMock()
                    mock_ollama_class.return_value = mock_ollama

                    mock_mem = MagicMock()
                    mock_mem.auto_switch_threshold = 0.8
                    mock_mem.prompt_threshold = 0.5
                    mock_mem_class.return_value = mock_mem

                    mock_ui = MagicMock()
                    mock_ui.show_thinking_indicator.return_value.__enter__ = MagicMock()
                    mock_ui.show_thinking_indicator.return_value.__exit__ = MagicMock()
                    mock_ui_class.return_value = mock_ui

                    from src.main import GlennBot
                    bot = GlennBot()
                    bot.ui = mock_ui
                    bot.knowledge_base = mock_kb
                    bot.ollama_client = mock_ollama

                    with patch("src.main.LibraryBuilder") as mock_builder_class, \
                         patch("src.main.LibraryServer") as mock_server_class:
                        mock_builder = MagicMock()
                        mock_builder.build.return_value = {
                            "items_count": 5,
                            "themes_count": 3,
                            "pages_generated": 12,
                            "last_build": "2024-01-15T10:00:00",
                        }
                        mock_builder_class.return_value = mock_builder

                        mock_server = MagicMock()
                        mock_server.serve.return_value = "http://localhost:8080"
                        mock_server_class.return_value = mock_server

                        bot.handle_command("/build-library --serve")

                        # Verify build was called first
                        mock_builder.build.assert_called_once()

                        # Verify server was created and started
                        mock_server_class.assert_called_once_with(
                            mock_settings.library_site_dir,
                            mock_settings.library_server_port,
                        )
                        mock_server.serve.assert_called_once()

    def test_build_library_force_and_serve_together(self, temp_dir):
        """Test that --force and --serve can be used together."""
        with patch.dict("sys.modules", {"chromadb": MagicMock()}):
            with patch("src.config.settings") as mock_settings:
                mock_settings.log_level = "WARNING"
                mock_settings.knowledge_dir = temp_dir / "knowledge"
                mock_settings.conversation_history_dir = temp_dir / "conversations"
                mock_settings.chroma_persist_directory = temp_dir / "chroma"
                mock_settings.chroma_collection_name = "test_collection"
                mock_settings.library_data_dir = temp_dir / "library"
                mock_settings.library_site_dir = temp_dir / "library-site"
                mock_settings.library_server_port = 8080
                mock_settings.ollama_host = "http://localhost:11434"
                mock_settings.ollama_model = "llama3:8b"
                mock_settings.ollama_embedding_model = "nomic-embed-text"
                mock_settings.context_persistence_dir = temp_dir / "contexts"
                mock_settings.memory_persistence_file = temp_dir / "memories.json"
                mock_settings.quotes_persistence_file = temp_dir / "quotes.json"
                mock_settings.feedback_persistence_dir = temp_dir / "feedback"

                mock_settings.knowledge_dir.mkdir(parents=True, exist_ok=True)
                mock_settings.library_data_dir.mkdir(parents=True, exist_ok=True)
                mock_settings.library_site_dir.mkdir(parents=True, exist_ok=True)

                with patch("src.knowledge_base.KnowledgeBase") as mock_kb_class, \
                     patch("src.ollama_client.OllamaClient") as mock_ollama_class, \
                     patch("src.terminal_ui.TerminalUI") as mock_ui_class, \
                     patch("src.conversation.ConversationManager"), \
                     patch("src.memory_system.MemorySystem") as mock_mem_class, \
                     patch("src.quotes_system.QuotesSystem"), \
                     patch("src.agents.AgentOrchestrator"), \
                     patch("src.document_ingestion.DocumentIngestionTool"), \
                     patch("src.feedback_system.FeedbackManager"), \
                     patch("src.context_detector.ContextDetector"):

                    mock_kb = MagicMock()
                    mock_kb.get_stats.return_value = {"total_documents": 10, "types": {}}
                    mock_kb_class.return_value = mock_kb

                    mock_ollama = MagicMock()
                    mock_ollama_class.return_value = mock_ollama

                    mock_mem = MagicMock()
                    mock_mem.auto_switch_threshold = 0.8
                    mock_mem.prompt_threshold = 0.5
                    mock_mem_class.return_value = mock_mem

                    mock_ui = MagicMock()
                    mock_ui.show_thinking_indicator.return_value.__enter__ = MagicMock()
                    mock_ui.show_thinking_indicator.return_value.__exit__ = MagicMock()
                    mock_ui_class.return_value = mock_ui

                    from src.main import GlennBot
                    bot = GlennBot()
                    bot.ui = mock_ui
                    bot.knowledge_base = mock_kb
                    bot.ollama_client = mock_ollama

                    with patch("src.main.LibraryBuilder") as mock_builder_class, \
                         patch("src.main.LibraryServer") as mock_server_class:
                        mock_builder = MagicMock()
                        mock_builder.build.return_value = {
                            "items_count": 5,
                            "themes_count": 3,
                            "pages_generated": 12,
                            "last_build": "2024-01-15T10:00:00",
                        }
                        mock_builder_class.return_value = mock_builder

                        mock_server = MagicMock()
                        mock_server.serve.return_value = "http://localhost:8080"
                        mock_server_class.return_value = mock_server

                        bot.handle_command("/build-library --force --serve")

                        # Verify build was called with force=True
                        mock_builder.build.assert_called_once_with(force=True)

                        # Verify server was started
                        mock_server.serve.assert_called_once()


class TestServeLibraryCommand:
    """Tests for the /serve-library command."""

    def test_serve_library_starts_server(self, temp_dir):
        """Test that /serve-library starts the HTTP server."""
        with patch.dict("sys.modules", {"chromadb": MagicMock()}):
            with patch("src.config.settings") as mock_settings:
                mock_settings.log_level = "WARNING"
                mock_settings.knowledge_dir = temp_dir / "knowledge"
                mock_settings.conversation_history_dir = temp_dir / "conversations"
                mock_settings.chroma_persist_directory = temp_dir / "chroma"
                mock_settings.chroma_collection_name = "test_collection"
                mock_settings.library_data_dir = temp_dir / "library"
                mock_settings.library_site_dir = temp_dir / "library-site"
                mock_settings.library_server_port = 8080
                mock_settings.ollama_host = "http://localhost:11434"
                mock_settings.ollama_model = "llama3:8b"
                mock_settings.ollama_embedding_model = "nomic-embed-text"
                mock_settings.context_persistence_dir = temp_dir / "contexts"
                mock_settings.memory_persistence_file = temp_dir / "memories.json"
                mock_settings.quotes_persistence_file = temp_dir / "quotes.json"
                mock_settings.feedback_persistence_dir = temp_dir / "feedback"

                mock_settings.knowledge_dir.mkdir(parents=True, exist_ok=True)
                mock_settings.library_data_dir.mkdir(parents=True, exist_ok=True)
                mock_settings.library_site_dir.mkdir(parents=True, exist_ok=True)

                with patch("src.knowledge_base.KnowledgeBase") as mock_kb_class, \
                     patch("src.ollama_client.OllamaClient") as mock_ollama_class, \
                     patch("src.terminal_ui.TerminalUI") as mock_ui_class, \
                     patch("src.conversation.ConversationManager"), \
                     patch("src.memory_system.MemorySystem") as mock_mem_class, \
                     patch("src.quotes_system.QuotesSystem"), \
                     patch("src.agents.AgentOrchestrator"), \
                     patch("src.document_ingestion.DocumentIngestionTool"), \
                     patch("src.feedback_system.FeedbackManager"), \
                     patch("src.context_detector.ContextDetector"):

                    mock_kb = MagicMock()
                    mock_kb.get_stats.return_value = {"total_documents": 10, "types": {}}
                    mock_kb_class.return_value = mock_kb

                    mock_ollama = MagicMock()
                    mock_ollama_class.return_value = mock_ollama

                    mock_mem = MagicMock()
                    mock_mem.auto_switch_threshold = 0.8
                    mock_mem.prompt_threshold = 0.5
                    mock_mem_class.return_value = mock_mem

                    mock_ui = MagicMock()
                    mock_ui.show_thinking_indicator.return_value.__enter__ = MagicMock()
                    mock_ui.show_thinking_indicator.return_value.__exit__ = MagicMock()
                    mock_ui_class.return_value = mock_ui

                    from src.main import GlennBot
                    bot = GlennBot()
                    bot.ui = mock_ui
                    bot.knowledge_base = mock_kb
                    bot.ollama_client = mock_ollama

                    with patch("src.main.LibraryServer") as mock_server_class:
                        mock_server = MagicMock()
                        mock_server.serve.return_value = "http://localhost:8080"
                        mock_server_class.return_value = mock_server

                        result = bot.handle_command("/serve-library")

                        # Verify server was created
                        mock_server_class.assert_called_once_with(
                            mock_settings.library_site_dir,
                            mock_settings.library_server_port,
                        )

                        # Verify serve was called
                        mock_server.serve.assert_called_once()

                        assert result is True

    def test_serve_library_handles_missing_site_directory(self, temp_dir):
        """Test that /serve-library handles missing site directory."""
        with patch.dict("sys.modules", {"chromadb": MagicMock()}):
            with patch("src.config.settings") as mock_settings:
                mock_settings.log_level = "WARNING"
                mock_settings.knowledge_dir = temp_dir / "knowledge"
                mock_settings.conversation_history_dir = temp_dir / "conversations"
                mock_settings.chroma_persist_directory = temp_dir / "chroma"
                mock_settings.chroma_collection_name = "test_collection"
                mock_settings.library_data_dir = temp_dir / "library"
                mock_settings.library_site_dir = temp_dir / "library-site"
                mock_settings.library_server_port = 8080
                mock_settings.ollama_host = "http://localhost:11434"
                mock_settings.ollama_model = "llama3:8b"
                mock_settings.ollama_embedding_model = "nomic-embed-text"
                mock_settings.context_persistence_dir = temp_dir / "contexts"
                mock_settings.memory_persistence_file = temp_dir / "memories.json"
                mock_settings.quotes_persistence_file = temp_dir / "quotes.json"
                mock_settings.feedback_persistence_dir = temp_dir / "feedback"

                mock_settings.knowledge_dir.mkdir(parents=True, exist_ok=True)
                mock_settings.library_data_dir.mkdir(parents=True, exist_ok=True)
                mock_settings.library_site_dir.mkdir(parents=True, exist_ok=True)

                with patch("src.knowledge_base.KnowledgeBase") as mock_kb_class, \
                     patch("src.ollama_client.OllamaClient") as mock_ollama_class, \
                     patch("src.terminal_ui.TerminalUI") as mock_ui_class, \
                     patch("src.conversation.ConversationManager"), \
                     patch("src.memory_system.MemorySystem") as mock_mem_class, \
                     patch("src.quotes_system.QuotesSystem"), \
                     patch("src.agents.AgentOrchestrator"), \
                     patch("src.document_ingestion.DocumentIngestionTool"), \
                     patch("src.feedback_system.FeedbackManager"), \
                     patch("src.context_detector.ContextDetector"):

                    mock_kb = MagicMock()
                    mock_kb.get_stats.return_value = {"total_documents": 10, "types": {}}
                    mock_kb_class.return_value = mock_kb

                    mock_ollama = MagicMock()
                    mock_ollama_class.return_value = mock_ollama

                    mock_mem = MagicMock()
                    mock_mem.auto_switch_threshold = 0.8
                    mock_mem.prompt_threshold = 0.5
                    mock_mem_class.return_value = mock_mem

                    mock_ui = MagicMock()
                    mock_ui.show_thinking_indicator.return_value.__enter__ = MagicMock()
                    mock_ui.show_thinking_indicator.return_value.__exit__ = MagicMock()
                    mock_ui_class.return_value = mock_ui

                    from src.main import GlennBot
                    bot = GlennBot()
                    bot.ui = mock_ui
                    bot.knowledge_base = mock_kb
                    bot.ollama_client = mock_ollama

                    with patch("src.main.LibraryServer") as mock_server_class:
                        # Simulate ValueError when directory doesn't exist
                        mock_server_class.side_effect = ValueError(
                            "Site directory does not exist"
                        )

                        result = bot.handle_command("/serve-library")

                        # Should display error
                        bot.ui.display_error.assert_called()

                        assert result is True

    def test_serve_library_handles_port_in_use(self, temp_dir):
        """Test that /serve-library handles port-in-use errors."""
        with patch.dict("sys.modules", {"chromadb": MagicMock()}):
            with patch("src.config.settings") as mock_settings:
                mock_settings.log_level = "WARNING"
                mock_settings.knowledge_dir = temp_dir / "knowledge"
                mock_settings.conversation_history_dir = temp_dir / "conversations"
                mock_settings.chroma_persist_directory = temp_dir / "chroma"
                mock_settings.chroma_collection_name = "test_collection"
                mock_settings.library_data_dir = temp_dir / "library"
                mock_settings.library_site_dir = temp_dir / "library-site"
                mock_settings.library_server_port = 8080
                mock_settings.ollama_host = "http://localhost:11434"
                mock_settings.ollama_model = "llama3:8b"
                mock_settings.ollama_embedding_model = "nomic-embed-text"
                mock_settings.context_persistence_dir = temp_dir / "contexts"
                mock_settings.memory_persistence_file = temp_dir / "memories.json"
                mock_settings.quotes_persistence_file = temp_dir / "quotes.json"
                mock_settings.feedback_persistence_dir = temp_dir / "feedback"

                mock_settings.knowledge_dir.mkdir(parents=True, exist_ok=True)
                mock_settings.library_data_dir.mkdir(parents=True, exist_ok=True)
                mock_settings.library_site_dir.mkdir(parents=True, exist_ok=True)

                with patch("src.knowledge_base.KnowledgeBase") as mock_kb_class, \
                     patch("src.ollama_client.OllamaClient") as mock_ollama_class, \
                     patch("src.terminal_ui.TerminalUI") as mock_ui_class, \
                     patch("src.conversation.ConversationManager"), \
                     patch("src.memory_system.MemorySystem") as mock_mem_class, \
                     patch("src.quotes_system.QuotesSystem"), \
                     patch("src.agents.AgentOrchestrator"), \
                     patch("src.document_ingestion.DocumentIngestionTool"), \
                     patch("src.feedback_system.FeedbackManager"), \
                     patch("src.context_detector.ContextDetector"):

                    mock_kb = MagicMock()
                    mock_kb.get_stats.return_value = {"total_documents": 10, "types": {}}
                    mock_kb_class.return_value = mock_kb

                    mock_ollama = MagicMock()
                    mock_ollama_class.return_value = mock_ollama

                    mock_mem = MagicMock()
                    mock_mem.auto_switch_threshold = 0.8
                    mock_mem.prompt_threshold = 0.5
                    mock_mem_class.return_value = mock_mem

                    mock_ui = MagicMock()
                    mock_ui.show_thinking_indicator.return_value.__enter__ = MagicMock()
                    mock_ui.show_thinking_indicator.return_value.__exit__ = MagicMock()
                    mock_ui_class.return_value = mock_ui

                    from src.main import GlennBot
                    bot = GlennBot()
                    bot.ui = mock_ui
                    bot.knowledge_base = mock_kb
                    bot.ollama_client = mock_ollama

                    with patch("src.main.LibraryServer") as mock_server_class:
                        mock_server = MagicMock()
                        mock_server.serve.side_effect = OSError("Port already in use")
                        mock_server_class.return_value = mock_server

                        result = bot.handle_command("/serve-library")

                        # Should display error
                        bot.ui.display_error.assert_called()

                        assert result is True


class TestLibraryStatusCommand:
    """Tests for the /library-status command."""

    def test_library_status_shows_build_state(self, temp_dir):
        """Test that /library-status displays build state information."""
        with patch.dict("sys.modules", {"chromadb": MagicMock()}):
            with patch("src.config.settings") as mock_settings:
                mock_settings.log_level = "WARNING"
                mock_settings.knowledge_dir = temp_dir / "knowledge"
                mock_settings.conversation_history_dir = temp_dir / "conversations"
                mock_settings.chroma_persist_directory = temp_dir / "chroma"
                mock_settings.chroma_collection_name = "test_collection"
                mock_settings.library_data_dir = temp_dir / "library"
                mock_settings.library_site_dir = temp_dir / "library-site"
                mock_settings.library_server_port = 8080
                mock_settings.ollama_host = "http://localhost:11434"
                mock_settings.ollama_model = "llama3:8b"
                mock_settings.ollama_embedding_model = "nomic-embed-text"
                mock_settings.context_persistence_dir = temp_dir / "contexts"
                mock_settings.memory_persistence_file = temp_dir / "memories.json"
                mock_settings.quotes_persistence_file = temp_dir / "quotes.json"
                mock_settings.feedback_persistence_dir = temp_dir / "feedback"

                mock_settings.knowledge_dir.mkdir(parents=True, exist_ok=True)
                mock_settings.library_data_dir.mkdir(parents=True, exist_ok=True)
                mock_settings.library_site_dir.mkdir(parents=True, exist_ok=True)

                # Create a build state file
                build_state = {
                    "last_build": "2024-01-15T10:00:00",
                    "item_hashes": {"item1": "abc", "item2": "def"},
                    "theme_version": "1.0",
                }
                build_state_file = mock_settings.library_site_dir / "_build_state.json"
                with open(build_state_file, "w") as f:
                    json.dump(build_state, f)

                with patch("src.knowledge_base.KnowledgeBase") as mock_kb_class, \
                     patch("src.ollama_client.OllamaClient") as mock_ollama_class, \
                     patch("src.terminal_ui.TerminalUI") as mock_ui_class, \
                     patch("src.conversation.ConversationManager"), \
                     patch("src.memory_system.MemorySystem") as mock_mem_class, \
                     patch("src.quotes_system.QuotesSystem"), \
                     patch("src.agents.AgentOrchestrator"), \
                     patch("src.document_ingestion.DocumentIngestionTool"), \
                     patch("src.feedback_system.FeedbackManager"), \
                     patch("src.context_detector.ContextDetector"):

                    mock_kb = MagicMock()
                    mock_kb.get_stats.return_value = {"total_documents": 10, "types": {}}
                    mock_kb_class.return_value = mock_kb

                    mock_ollama = MagicMock()
                    mock_ollama_class.return_value = mock_ollama

                    mock_mem = MagicMock()
                    mock_mem.auto_switch_threshold = 0.8
                    mock_mem.prompt_threshold = 0.5
                    mock_mem_class.return_value = mock_mem

                    mock_ui = MagicMock()
                    mock_ui.show_thinking_indicator.return_value.__enter__ = MagicMock()
                    mock_ui.show_thinking_indicator.return_value.__exit__ = MagicMock()
                    mock_ui_class.return_value = mock_ui

                    from src.main import GlennBot
                    bot = GlennBot()
                    bot.ui = mock_ui
                    bot.knowledge_base = mock_kb
                    bot.ollama_client = mock_ollama

                    result = bot.handle_command("/library-status")

                    # Check output includes relevant info
                    bot.ui.console.print.assert_called()

                    assert result is True

    def test_library_status_shows_no_build_when_empty(self, temp_dir):
        """Test that /library-status shows appropriate message when not built."""
        with patch.dict("sys.modules", {"chromadb": MagicMock()}):
            with patch("src.config.settings") as mock_settings:
                mock_settings.log_level = "WARNING"
                mock_settings.knowledge_dir = temp_dir / "knowledge"
                mock_settings.conversation_history_dir = temp_dir / "conversations"
                mock_settings.chroma_persist_directory = temp_dir / "chroma"
                mock_settings.chroma_collection_name = "test_collection"
                mock_settings.library_data_dir = temp_dir / "library"
                mock_settings.library_site_dir = temp_dir / "library-site"
                mock_settings.library_server_port = 8080
                mock_settings.ollama_host = "http://localhost:11434"
                mock_settings.ollama_model = "llama3:8b"
                mock_settings.ollama_embedding_model = "nomic-embed-text"
                mock_settings.context_persistence_dir = temp_dir / "contexts"
                mock_settings.memory_persistence_file = temp_dir / "memories.json"
                mock_settings.quotes_persistence_file = temp_dir / "quotes.json"
                mock_settings.feedback_persistence_dir = temp_dir / "feedback"

                mock_settings.knowledge_dir.mkdir(parents=True, exist_ok=True)
                mock_settings.library_data_dir.mkdir(parents=True, exist_ok=True)
                mock_settings.library_site_dir.mkdir(parents=True, exist_ok=True)

                with patch("src.knowledge_base.KnowledgeBase") as mock_kb_class, \
                     patch("src.ollama_client.OllamaClient") as mock_ollama_class, \
                     patch("src.terminal_ui.TerminalUI") as mock_ui_class, \
                     patch("src.conversation.ConversationManager"), \
                     patch("src.memory_system.MemorySystem") as mock_mem_class, \
                     patch("src.quotes_system.QuotesSystem"), \
                     patch("src.agents.AgentOrchestrator"), \
                     patch("src.document_ingestion.DocumentIngestionTool"), \
                     patch("src.feedback_system.FeedbackManager"), \
                     patch("src.context_detector.ContextDetector"):

                    mock_kb = MagicMock()
                    mock_kb.get_stats.return_value = {"total_documents": 10, "types": {}}
                    mock_kb_class.return_value = mock_kb

                    mock_ollama = MagicMock()
                    mock_ollama_class.return_value = mock_ollama

                    mock_mem = MagicMock()
                    mock_mem.auto_switch_threshold = 0.8
                    mock_mem.prompt_threshold = 0.5
                    mock_mem_class.return_value = mock_mem

                    mock_ui = MagicMock()
                    mock_ui.show_thinking_indicator.return_value.__enter__ = MagicMock()
                    mock_ui.show_thinking_indicator.return_value.__exit__ = MagicMock()
                    mock_ui_class.return_value = mock_ui

                    from src.main import GlennBot
                    bot = GlennBot()
                    bot.ui = mock_ui
                    bot.knowledge_base = mock_kb
                    bot.ollama_client = mock_ollama

                    result = bot.handle_command("/library-status")

                    # Should indicate no build exists
                    bot.ui.console.print.assert_called()
                    print_calls = [str(c) for c in bot.ui.console.print.call_args_list]
                    # Should show "not built" or "never" or similar
                    assert any(
                        "not" in c.lower() or "never" in c.lower() or "no build" in c.lower()
                        for c in print_calls
                    )

                    assert result is True
