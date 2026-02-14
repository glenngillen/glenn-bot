"""Tests for library CLI commands in GlennBot."""

import json
import sys
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


# Mock out heavy dependencies before importing src.main
@pytest.fixture(autouse=True)
def mock_chromadb():
    """Mock chromadb module before any imports."""
    if "chromadb" not in sys.modules:
        sys.modules["chromadb"] = MagicMock()
        sys.modules["chromadb.config"] = MagicMock()


class TestBuildLibraryCommand:
    """Tests for the /build-library command handler."""

    def test_build_library_command_calls_builder(self, temp_dir, mock_chromadb):
        """Test that /build-library command invokes LibraryBuilder.build()."""
        with patch.dict("sys.modules", {"chromadb": MagicMock(), "chromadb.config": MagicMock()}):
            # Import after mocking
            import src.main as main_module

            with patch.object(main_module, "LibraryBuilder") as mock_builder_class, \
                 patch.object(main_module, "LibraryServer"), \
                 patch.object(main_module, "settings") as mock_settings:
                mock_settings.library_data_dir = temp_dir / "library"
                mock_settings.library_site_dir = temp_dir / "library-site"
                mock_settings.library_server_port = 8080
                mock_settings.library_data_dir.mkdir(parents=True, exist_ok=True)
                mock_settings.library_site_dir.mkdir(parents=True, exist_ok=True)

                mock_builder = MagicMock()
                mock_builder.build.return_value = {
                    "items_count": 5,
                    "themes_count": 3,
                    "pages_generated": 12,
                    "last_build": "2024-01-15T10:00:00",
                }
                mock_builder_class.return_value = mock_builder

                # Create a mock bot
                mock_bot = MagicMock()
                mock_bot.ui = MagicMock()
                mock_bot.ui.show_thinking_indicator.return_value.__enter__ = MagicMock()
                mock_bot.ui.show_thinking_indicator.return_value.__exit__ = MagicMock()
                mock_bot.knowledge_base = MagicMock()
                mock_bot.ollama_client = MagicMock()

                # Call the method
                main_module.GlennBot._build_library(mock_bot, "/build-library")

                # Verify builder was created
                mock_builder_class.assert_called_once_with(
                    mock_bot.knowledge_base,
                    mock_bot.ollama_client,
                    mock_settings.library_data_dir,
                    mock_settings.library_site_dir,
                )

                # Verify build was called
                mock_builder.build.assert_called_once_with(force=False)

    def test_build_library_displays_success_message(self, temp_dir, mock_chromadb):
        """Test that /build-library displays build statistics on success."""
        with patch.dict("sys.modules", {"chromadb": MagicMock(), "chromadb.config": MagicMock()}):
            import src.main as main_module

            with patch.object(main_module, "LibraryBuilder") as mock_builder_class, \
                 patch.object(main_module, "LibraryServer"), \
                 patch.object(main_module, "settings") as mock_settings:
                mock_settings.library_data_dir = temp_dir / "library"
                mock_settings.library_site_dir = temp_dir / "library-site"
                mock_settings.library_server_port = 8080
                mock_settings.library_data_dir.mkdir(parents=True, exist_ok=True)
                mock_settings.library_site_dir.mkdir(parents=True, exist_ok=True)

                mock_builder = MagicMock()
                mock_builder.build.return_value = {
                    "items_count": 5,
                    "themes_count": 3,
                    "pages_generated": 12,
                    "last_build": "2024-01-15T10:00:00",
                }
                mock_builder_class.return_value = mock_builder

                mock_bot = MagicMock()
                mock_bot.ui = MagicMock()
                mock_bot.ui.show_thinking_indicator.return_value.__enter__ = MagicMock()
                mock_bot.ui.show_thinking_indicator.return_value.__exit__ = MagicMock()
                mock_bot.knowledge_base = MagicMock()
                mock_bot.ollama_client = MagicMock()

                main_module.GlennBot._build_library(mock_bot, "/build-library")

                # Verify success message was displayed
                mock_bot.ui.console.print.assert_called()
                print_calls = [str(c) for c in mock_bot.ui.console.print.call_args_list]
                assert any("5" in c for c in print_calls) or \
                       any("green" in c.lower() for c in print_calls)

    def test_build_library_shows_thinking_indicator(self, temp_dir, mock_chromadb):
        """Test that /build-library shows the thinking indicator during build."""
        with patch.dict("sys.modules", {"chromadb": MagicMock(), "chromadb.config": MagicMock()}):
            import src.main as main_module

            with patch.object(main_module, "LibraryBuilder") as mock_builder_class, \
                 patch.object(main_module, "LibraryServer"), \
                 patch.object(main_module, "settings") as mock_settings:
                mock_settings.library_data_dir = temp_dir / "library"
                mock_settings.library_site_dir = temp_dir / "library-site"
                mock_settings.library_server_port = 8080
                mock_settings.library_data_dir.mkdir(parents=True, exist_ok=True)
                mock_settings.library_site_dir.mkdir(parents=True, exist_ok=True)

                mock_builder = MagicMock()
                mock_builder.build.return_value = {
                    "items_count": 5,
                    "themes_count": 3,
                    "pages_generated": 12,
                    "last_build": "2024-01-15T10:00:00",
                }
                mock_builder_class.return_value = mock_builder

                mock_bot = MagicMock()
                mock_bot.ui = MagicMock()
                mock_bot.ui.show_thinking_indicator.return_value.__enter__ = MagicMock()
                mock_bot.ui.show_thinking_indicator.return_value.__exit__ = MagicMock()
                mock_bot.knowledge_base = MagicMock()
                mock_bot.ollama_client = MagicMock()

                main_module.GlennBot._build_library(mock_bot, "/build-library")

                # Verify thinking indicator was used
                mock_bot.ui.show_thinking_indicator.assert_called()

    def test_build_library_handles_build_error(self, temp_dir, mock_chromadb):
        """Test that /build-library handles errors gracefully."""
        with patch.dict("sys.modules", {"chromadb": MagicMock(), "chromadb.config": MagicMock()}):
            import src.main as main_module

            with patch.object(main_module, "LibraryBuilder") as mock_builder_class, \
                 patch.object(main_module, "LibraryServer"), \
                 patch.object(main_module, "settings") as mock_settings:
                mock_settings.library_data_dir = temp_dir / "library"
                mock_settings.library_site_dir = temp_dir / "library-site"
                mock_settings.library_server_port = 8080
                mock_settings.library_data_dir.mkdir(parents=True, exist_ok=True)
                mock_settings.library_site_dir.mkdir(parents=True, exist_ok=True)

                mock_builder = MagicMock()
                mock_builder.build.side_effect = Exception("Build failed")
                mock_builder_class.return_value = mock_builder

                mock_bot = MagicMock()
                mock_bot.ui = MagicMock()
                mock_bot.ui.show_thinking_indicator.return_value.__enter__ = MagicMock()
                mock_bot.ui.show_thinking_indicator.return_value.__exit__ = MagicMock()
                mock_bot.knowledge_base = MagicMock()
                mock_bot.ollama_client = MagicMock()

                # Should not raise
                main_module.GlennBot._build_library(mock_bot, "/build-library")

                # Verify error was displayed
                mock_bot.ui.display_error.assert_called()
                error_call = str(mock_bot.ui.display_error.call_args)
                assert "Build failed" in error_call or "Failed" in error_call


class TestBuildLibraryForceFlag:
    """Tests for the /build-library --force flag."""

    def test_build_library_force_flag_passes_force_true(self, temp_dir, mock_chromadb):
        """Test that /build-library --force passes force=True to builder."""
        with patch.dict("sys.modules", {"chromadb": MagicMock(), "chromadb.config": MagicMock()}):
            import src.main as main_module

            with patch.object(main_module, "LibraryBuilder") as mock_builder_class, \
                 patch.object(main_module, "LibraryServer"), \
                 patch.object(main_module, "settings") as mock_settings:
                mock_settings.library_data_dir = temp_dir / "library"
                mock_settings.library_site_dir = temp_dir / "library-site"
                mock_settings.library_server_port = 8080
                mock_settings.library_data_dir.mkdir(parents=True, exist_ok=True)
                mock_settings.library_site_dir.mkdir(parents=True, exist_ok=True)

                mock_builder = MagicMock()
                mock_builder.build.return_value = {
                    "items_count": 5,
                    "themes_count": 3,
                    "pages_generated": 12,
                    "last_build": "2024-01-15T10:00:00",
                }
                mock_builder_class.return_value = mock_builder

                mock_bot = MagicMock()
                mock_bot.ui = MagicMock()
                mock_bot.ui.show_thinking_indicator.return_value.__enter__ = MagicMock()
                mock_bot.ui.show_thinking_indicator.return_value.__exit__ = MagicMock()
                mock_bot.knowledge_base = MagicMock()
                mock_bot.ollama_client = MagicMock()

                main_module.GlennBot._build_library(mock_bot, "/build-library --force")

                # Verify build was called with force=True
                mock_builder.build.assert_called_once_with(force=True)


class TestBuildLibraryServeFlag:
    """Tests for the /build-library --serve flag."""

    def test_build_library_serve_flag_starts_server_after_build(self, temp_dir, mock_chromadb):
        """Test that /build-library --serve starts the server after building."""
        with patch.dict("sys.modules", {"chromadb": MagicMock(), "chromadb.config": MagicMock()}):
            import src.main as main_module

            with patch.object(main_module, "LibraryBuilder") as mock_builder_class, \
                 patch.object(main_module, "LibraryServer") as mock_server_class, \
                 patch.object(main_module, "settings") as mock_settings:
                mock_settings.library_data_dir = temp_dir / "library"
                mock_settings.library_site_dir = temp_dir / "library-site"
                mock_settings.library_server_port = 8080
                mock_settings.library_data_dir.mkdir(parents=True, exist_ok=True)
                mock_settings.library_site_dir.mkdir(parents=True, exist_ok=True)

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

                mock_bot = MagicMock()
                mock_bot.ui = MagicMock()
                mock_bot.ui.show_thinking_indicator.return_value.__enter__ = MagicMock()
                mock_bot.ui.show_thinking_indicator.return_value.__exit__ = MagicMock()
                mock_bot.knowledge_base = MagicMock()
                mock_bot.ollama_client = MagicMock()

                # Make _serve_library actually call the real implementation
                mock_bot._serve_library = lambda: main_module.GlennBot._serve_library(mock_bot)

                main_module.GlennBot._build_library(mock_bot, "/build-library --serve")

                # Verify build was called first
                mock_builder.build.assert_called_once()

                # Verify server was created and started
                mock_server_class.assert_called_once_with(
                    mock_settings.library_site_dir,
                    mock_settings.library_server_port,
                )
                mock_server.serve.assert_called_once()

    def test_build_library_force_and_serve_together(self, temp_dir, mock_chromadb):
        """Test that --force and --serve can be used together."""
        with patch.dict("sys.modules", {"chromadb": MagicMock(), "chromadb.config": MagicMock()}):
            import src.main as main_module

            with patch.object(main_module, "LibraryBuilder") as mock_builder_class, \
                 patch.object(main_module, "LibraryServer") as mock_server_class, \
                 patch.object(main_module, "settings") as mock_settings:
                mock_settings.library_data_dir = temp_dir / "library"
                mock_settings.library_site_dir = temp_dir / "library-site"
                mock_settings.library_server_port = 8080
                mock_settings.library_data_dir.mkdir(parents=True, exist_ok=True)
                mock_settings.library_site_dir.mkdir(parents=True, exist_ok=True)

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

                mock_bot = MagicMock()
                mock_bot.ui = MagicMock()
                mock_bot.ui.show_thinking_indicator.return_value.__enter__ = MagicMock()
                mock_bot.ui.show_thinking_indicator.return_value.__exit__ = MagicMock()
                mock_bot.knowledge_base = MagicMock()
                mock_bot.ollama_client = MagicMock()

                # Make _serve_library actually call the real implementation
                mock_bot._serve_library = lambda: main_module.GlennBot._serve_library(mock_bot)

                main_module.GlennBot._build_library(mock_bot, "/build-library --force --serve")

                # Verify build was called with force=True
                mock_builder.build.assert_called_once_with(force=True)

                # Verify server was started
                mock_server.serve.assert_called_once()


class TestServeLibraryCommand:
    """Tests for the /serve-library command."""

    def test_serve_library_starts_server(self, temp_dir, mock_chromadb):
        """Test that /serve-library starts the HTTP server."""
        with patch.dict("sys.modules", {"chromadb": MagicMock(), "chromadb.config": MagicMock()}):
            import src.main as main_module

            with patch.object(main_module, "LibraryServer") as mock_server_class, \
                 patch.object(main_module, "settings") as mock_settings:
                mock_settings.library_site_dir = temp_dir / "library-site"
                mock_settings.library_server_port = 8080
                mock_settings.library_site_dir.mkdir(parents=True, exist_ok=True)

                mock_server = MagicMock()
                mock_server.serve.return_value = "http://localhost:8080"
                mock_server_class.return_value = mock_server

                mock_bot = MagicMock()
                mock_bot.ui = MagicMock()
                mock_bot.knowledge_base = MagicMock()
                mock_bot.ollama_client = MagicMock()

                main_module.GlennBot._serve_library(mock_bot)

                # Verify server was created
                mock_server_class.assert_called_once_with(
                    mock_settings.library_site_dir,
                    mock_settings.library_server_port,
                )

                # Verify serve was called
                mock_server.serve.assert_called_once()

    def test_serve_library_handles_missing_site_directory(self, temp_dir, mock_chromadb):
        """Test that /serve-library handles missing site directory."""
        with patch.dict("sys.modules", {"chromadb": MagicMock(), "chromadb.config": MagicMock()}):
            import src.main as main_module

            with patch.object(main_module, "LibraryServer") as mock_server_class, \
                 patch.object(main_module, "settings") as mock_settings:
                mock_settings.library_site_dir = temp_dir / "library-site"
                mock_settings.library_server_port = 8080
                # Don't create the directory

                # Simulate ValueError when directory doesn't exist
                mock_server_class.side_effect = ValueError(
                    "Site directory does not exist"
                )

                mock_bot = MagicMock()
                mock_bot.ui = MagicMock()

                main_module.GlennBot._serve_library(mock_bot)

                # Should display error
                mock_bot.ui.display_error.assert_called()

    def test_serve_library_handles_port_in_use(self, temp_dir, mock_chromadb):
        """Test that /serve-library handles port-in-use errors."""
        with patch.dict("sys.modules", {"chromadb": MagicMock(), "chromadb.config": MagicMock()}):
            import src.main as main_module

            with patch.object(main_module, "LibraryServer") as mock_server_class, \
                 patch.object(main_module, "settings") as mock_settings:
                mock_settings.library_site_dir = temp_dir / "library-site"
                mock_settings.library_server_port = 8080
                mock_settings.library_site_dir.mkdir(parents=True, exist_ok=True)

                mock_server = MagicMock()
                mock_server.serve.side_effect = OSError("Port already in use")
                mock_server_class.return_value = mock_server

                mock_bot = MagicMock()
                mock_bot.ui = MagicMock()

                main_module.GlennBot._serve_library(mock_bot)

                # Should display error
                mock_bot.ui.display_error.assert_called()


class TestLibraryStatusCommand:
    """Tests for the /library-status command."""

    def test_library_status_shows_build_state(self, temp_dir, mock_chromadb):
        """Test that /library-status displays build state information."""
        with patch.dict("sys.modules", {"chromadb": MagicMock(), "chromadb.config": MagicMock()}):
            import src.main as main_module

            with patch.object(main_module, "settings") as mock_settings:
                mock_settings.library_site_dir = temp_dir / "library-site"
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

                mock_bot = MagicMock()
                mock_bot.ui = MagicMock()

                main_module.GlennBot._show_library_status(mock_bot)

                # Check output includes relevant info
                mock_bot.ui.console.print.assert_called()

    def test_library_status_shows_no_build_when_empty(self, temp_dir, mock_chromadb):
        """Test that /library-status shows appropriate message when not built."""
        with patch.dict("sys.modules", {"chromadb": MagicMock(), "chromadb.config": MagicMock()}):
            import src.main as main_module

            with patch.object(main_module, "settings") as mock_settings:
                mock_settings.library_site_dir = temp_dir / "library-site"
                mock_settings.library_site_dir.mkdir(parents=True, exist_ok=True)
                # Don't create build state file

                mock_bot = MagicMock()
                mock_bot.ui = MagicMock()

                main_module.GlennBot._show_library_status(mock_bot)

                # Should indicate no build exists
                mock_bot.ui.console.print.assert_called()
                print_calls = [str(c) for c in mock_bot.ui.console.print.call_args_list]
                # Should show "not built" or similar
                assert any(
                    "not" in c.lower() for c in print_calls
                )
