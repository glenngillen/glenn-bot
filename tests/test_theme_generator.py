"""Tests for theme generator module.

This module tests the ThemeGenerator class which uses AI (via OllamaClient)
to automatically generate thematic categories that group related content
across all content types in the knowledge library.

Test classes are organized by the methods they test:
- TestThemeGeneratorInitialization: ThemeGenerator class initialization
- TestSaveThemes: save_themes() method for persisting themes to JSON
- TestLoadThemes: load_themes() method for reading themes from JSON
- TestSaveAssignments: save_assignments() method for persisting assignments
- TestLoadAssignments: load_assignments() method for reading assignments
- TestBuildThemeGenerationPrompt: _build_theme_generation_prompt() method
- TestParseThemesFromResponse: _parse_themes_from_response() method
- TestGenerateThemes: generate_themes() orchestration
- TestBuildAssignmentPrompt: _build_assignment_prompt() method
- TestParseAssignmentsFromResponse: _parse_assignments_from_response() method
- TestAssignItemsToThemes: assign_items_to_themes() method
- TestUpdateAssignments: update_assignments() for incremental updates
- TestMiscellaneousTheme: Catch-all theme for low confidence items
- TestGetItemsForTheme: get_items_for_theme() method
"""

import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch
import json

from src.library.models import Theme, ThemeAssignment, LibraryItem, ContentType


class TestThemeGeneratorInitialization:
    """Tests for ThemeGenerator class initialization."""

    def test_theme_generator_can_be_imported(self):
        """ThemeGenerator class should be importable from the module."""
        from src.library.theme_generator import ThemeGenerator
        assert ThemeGenerator is not None

    def test_theme_generator_init_with_ollama_client(self, mock_ollama_client, temp_dir):
        """ThemeGenerator should accept an OllamaClient instance."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        data_dir.mkdir(parents=True, exist_ok=True)

        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        assert generator.ollama_client is mock_ollama_client

    def test_theme_generator_init_with_data_dir(self, mock_ollama_client, temp_dir):
        """ThemeGenerator should accept a data_dir path."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        data_dir.mkdir(parents=True, exist_ok=True)

        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        assert generator.data_dir == data_dir

    def test_theme_generator_init_creates_data_dir_if_not_exists(self, mock_ollama_client, temp_dir):
        """ThemeGenerator should create the data_dir if it doesn't exist."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library" / "subdir"
        assert not data_dir.exists()

        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        assert data_dir.exists()

    def test_theme_generator_init_data_dir_as_string(self, mock_ollama_client, temp_dir):
        """ThemeGenerator should accept data_dir as a string and convert to Path."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = str(temp_dir / "library")

        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        assert isinstance(generator.data_dir, Path)
        assert generator.data_dir == Path(data_dir)

    def test_theme_generator_has_themes_file_path(self, mock_ollama_client, temp_dir):
        """ThemeGenerator should have a themes_file path attribute."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        data_dir.mkdir(parents=True, exist_ok=True)

        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        assert generator.themes_file == data_dir / "themes.json"

    def test_theme_generator_has_assignments_file_path(self, mock_ollama_client, temp_dir):
        """ThemeGenerator should have an assignments_file path attribute."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        data_dir.mkdir(parents=True, exist_ok=True)

        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        assert generator.assignments_file == data_dir / "assignments.json"

    def test_theme_generator_initializes_empty_themes_list(self, mock_ollama_client, temp_dir):
        """ThemeGenerator should initialize with an empty themes list."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        data_dir.mkdir(parents=True, exist_ok=True)

        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        assert generator.themes == []

    def test_theme_generator_initializes_empty_assignments_list(self, mock_ollama_client, temp_dir):
        """ThemeGenerator should initialize with an empty assignments list."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        data_dir.mkdir(parents=True, exist_ok=True)

        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        assert generator.assignments == []
