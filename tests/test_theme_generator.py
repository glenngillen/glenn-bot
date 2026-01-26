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


class TestSaveThemes:
    """Tests for save_themes() method that persists themes to themes.json."""

    def test_save_themes_creates_themes_file(self, mock_ollama_client, temp_dir):
        """save_themes() should create themes.json file in data_dir."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        data_dir.mkdir(parents=True, exist_ok=True)

        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        # Add a theme to save
        theme = Theme(
            id="personal-growth",
            name="Personal Growth",
            description="Self-improvement and learning",
            keywords=["growth", "learning", "habits"],
            item_count=5,
            created_at=datetime(2026, 1, 15, 10, 0, 0),
            updated_at=datetime(2026, 1, 15, 10, 0, 0),
        )
        generator.themes = [theme]

        generator.save_themes()

        assert (data_dir / "themes.json").exists()

    def test_save_themes_writes_valid_json(self, mock_ollama_client, temp_dir):
        """save_themes() should write valid JSON to themes.json."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        data_dir.mkdir(parents=True, exist_ok=True)

        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        theme = Theme(
            id="personal-growth",
            name="Personal Growth",
            description="Self-improvement and learning",
            keywords=["growth", "learning", "habits"],
            item_count=5,
            created_at=datetime(2026, 1, 15, 10, 0, 0),
            updated_at=datetime(2026, 1, 15, 10, 0, 0),
        )
        generator.themes = [theme]

        generator.save_themes()

        # Should be able to parse the JSON
        with open(data_dir / "themes.json") as f:
            data = json.load(f)

        assert isinstance(data, list)

    def test_save_themes_serializes_theme_data_correctly(self, mock_ollama_client, temp_dir):
        """save_themes() should serialize all theme fields correctly."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        data_dir.mkdir(parents=True, exist_ok=True)

        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        theme = Theme(
            id="systems-thinking",
            name="Systems Thinking",
            description="Understanding complex systems and feedback loops",
            keywords=["systems", "feedback", "complexity"],
            item_count=12,
            created_at=datetime(2026, 1, 20, 14, 30, 0),
            updated_at=datetime(2026, 1, 25, 9, 15, 0),
        )
        generator.themes = [theme]

        generator.save_themes()

        with open(data_dir / "themes.json") as f:
            data = json.load(f)

        assert len(data) == 1
        assert data[0]["id"] == "systems-thinking"
        assert data[0]["name"] == "Systems Thinking"
        assert data[0]["description"] == "Understanding complex systems and feedback loops"
        assert data[0]["keywords"] == ["systems", "feedback", "complexity"]
        assert data[0]["item_count"] == 12
        assert data[0]["created_at"] == "2026-01-20T14:30:00"
        assert data[0]["updated_at"] == "2026-01-25T09:15:00"

    def test_save_themes_saves_multiple_themes(self, mock_ollama_client, temp_dir):
        """save_themes() should save multiple themes to themes.json."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        data_dir.mkdir(parents=True, exist_ok=True)

        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        theme1 = Theme(
            id="personal-growth",
            name="Personal Growth",
            description="Self-improvement and learning",
            keywords=["growth", "learning"],
            item_count=5,
            created_at=datetime(2026, 1, 15, 10, 0, 0),
            updated_at=datetime(2026, 1, 15, 10, 0, 0),
        )
        theme2 = Theme(
            id="technology",
            name="Technology",
            description="Software and tech innovations",
            keywords=["software", "tech"],
            item_count=8,
            created_at=datetime(2026, 1, 16, 11, 0, 0),
            updated_at=datetime(2026, 1, 16, 11, 0, 0),
        )
        generator.themes = [theme1, theme2]

        generator.save_themes()

        with open(data_dir / "themes.json") as f:
            data = json.load(f)

        assert len(data) == 2
        assert data[0]["id"] == "personal-growth"
        assert data[1]["id"] == "technology"

    def test_save_themes_overwrites_existing_file(self, mock_ollama_client, temp_dir):
        """save_themes() should overwrite existing themes.json file."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        data_dir.mkdir(parents=True, exist_ok=True)

        # Create initial themes file with old data
        old_data = [{"id": "old-theme", "name": "Old Theme"}]
        with open(data_dir / "themes.json", "w") as f:
            json.dump(old_data, f)

        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        theme = Theme(
            id="new-theme",
            name="New Theme",
            description="A new theme",
            keywords=["new"],
            item_count=1,
            created_at=datetime(2026, 1, 26, 12, 0, 0),
            updated_at=datetime(2026, 1, 26, 12, 0, 0),
        )
        generator.themes = [theme]

        generator.save_themes()

        with open(data_dir / "themes.json") as f:
            data = json.load(f)

        assert len(data) == 1
        assert data[0]["id"] == "new-theme"
        assert data[0]["name"] == "New Theme"

    def test_save_themes_empty_list(self, mock_ollama_client, temp_dir):
        """save_themes() should save empty list when no themes exist."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        data_dir.mkdir(parents=True, exist_ok=True)

        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)
        generator.themes = []

        generator.save_themes()

        with open(data_dir / "themes.json") as f:
            data = json.load(f)

        assert data == []

    def test_save_themes_uses_indentation_for_readability(self, mock_ollama_client, temp_dir):
        """save_themes() should use indentation for human-readable JSON."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        data_dir.mkdir(parents=True, exist_ok=True)

        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        theme = Theme(
            id="test-theme",
            name="Test Theme",
            description="A test",
            keywords=["test"],
            item_count=1,
            created_at=datetime(2026, 1, 26, 12, 0, 0),
            updated_at=datetime(2026, 1, 26, 12, 0, 0),
        )
        generator.themes = [theme]

        generator.save_themes()

        with open(data_dir / "themes.json") as f:
            content = f.read()

        # Check for newlines (indented JSON has newlines)
        assert "\n" in content
