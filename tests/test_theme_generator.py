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


class TestLoadThemes:
    """Tests for load_themes() method that reads themes from themes.json."""

    def test_load_themes_reads_from_themes_file(self, mock_ollama_client, temp_dir):
        """load_themes() should read from themes.json file in data_dir."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        data_dir.mkdir(parents=True, exist_ok=True)

        # Create a themes.json file
        theme_data = [
            {
                "id": "personal-growth",
                "name": "Personal Growth",
                "description": "Self-improvement and learning",
                "keywords": ["growth", "learning", "habits"],
                "item_count": 5,
                "created_at": "2026-01-15T10:00:00",
                "updated_at": "2026-01-15T10:00:00",
            }
        ]
        with open(data_dir / "themes.json", "w") as f:
            json.dump(theme_data, f)

        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        generator.load_themes()

        assert len(generator.themes) == 1

    def test_load_themes_parses_theme_objects_correctly(self, mock_ollama_client, temp_dir):
        """load_themes() should parse JSON into Theme objects with correct fields."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        data_dir.mkdir(parents=True, exist_ok=True)

        theme_data = [
            {
                "id": "systems-thinking",
                "name": "Systems Thinking",
                "description": "Understanding complex systems and feedback loops",
                "keywords": ["systems", "feedback", "complexity"],
                "item_count": 12,
                "created_at": "2026-01-20T14:30:00",
                "updated_at": "2026-01-25T09:15:00",
            }
        ]
        with open(data_dir / "themes.json", "w") as f:
            json.dump(theme_data, f)

        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        generator.load_themes()

        theme = generator.themes[0]
        assert isinstance(theme, Theme)
        assert theme.id == "systems-thinking"
        assert theme.name == "Systems Thinking"
        assert theme.description == "Understanding complex systems and feedback loops"
        assert theme.keywords == ["systems", "feedback", "complexity"]
        assert theme.item_count == 12
        assert theme.created_at == datetime(2026, 1, 20, 14, 30, 0)
        assert theme.updated_at == datetime(2026, 1, 25, 9, 15, 0)

    def test_load_themes_loads_multiple_themes(self, mock_ollama_client, temp_dir):
        """load_themes() should load multiple themes from themes.json."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        data_dir.mkdir(parents=True, exist_ok=True)

        theme_data = [
            {
                "id": "personal-growth",
                "name": "Personal Growth",
                "description": "Self-improvement",
                "keywords": ["growth"],
                "item_count": 5,
                "created_at": "2026-01-15T10:00:00",
                "updated_at": "2026-01-15T10:00:00",
            },
            {
                "id": "technology",
                "name": "Technology",
                "description": "Tech innovations",
                "keywords": ["tech"],
                "item_count": 8,
                "created_at": "2026-01-16T11:00:00",
                "updated_at": "2026-01-16T11:00:00",
            },
            {
                "id": "business",
                "name": "Business",
                "description": "Business strategy",
                "keywords": ["business"],
                "item_count": 3,
                "created_at": "2026-01-17T12:00:00",
                "updated_at": "2026-01-17T12:00:00",
            },
        ]
        with open(data_dir / "themes.json", "w") as f:
            json.dump(theme_data, f)

        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        generator.load_themes()

        assert len(generator.themes) == 3
        assert generator.themes[0].id == "personal-growth"
        assert generator.themes[1].id == "technology"
        assert generator.themes[2].id == "business"

    def test_load_themes_handles_empty_file(self, mock_ollama_client, temp_dir):
        """load_themes() should handle empty themes list gracefully."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        data_dir.mkdir(parents=True, exist_ok=True)

        # Create an empty JSON array
        with open(data_dir / "themes.json", "w") as f:
            json.dump([], f)

        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        generator.load_themes()

        assert generator.themes == []

    def test_load_themes_handles_missing_file(self, mock_ollama_client, temp_dir):
        """load_themes() should return empty list when themes.json doesn't exist."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        data_dir.mkdir(parents=True, exist_ok=True)

        # Don't create themes.json file
        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        generator.load_themes()

        assert generator.themes == []

    def test_load_themes_replaces_existing_themes(self, mock_ollama_client, temp_dir):
        """load_themes() should replace any existing themes in memory."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        data_dir.mkdir(parents=True, exist_ok=True)

        # Create themes.json with new data
        theme_data = [
            {
                "id": "new-theme",
                "name": "New Theme",
                "description": "A new theme from file",
                "keywords": ["new"],
                "item_count": 1,
                "created_at": "2026-01-26T12:00:00",
                "updated_at": "2026-01-26T12:00:00",
            }
        ]
        with open(data_dir / "themes.json", "w") as f:
            json.dump(theme_data, f)

        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        # Set some existing themes in memory
        existing_theme = Theme(
            id="existing-theme",
            name="Existing Theme",
            description="An existing theme",
            keywords=["existing"],
            item_count=10,
            created_at=datetime(2026, 1, 1, 0, 0, 0),
            updated_at=datetime(2026, 1, 1, 0, 0, 0),
        )
        generator.themes = [existing_theme]

        generator.load_themes()

        # Should have replaced with file contents
        assert len(generator.themes) == 1
        assert generator.themes[0].id == "new-theme"

    def test_load_themes_returns_loaded_themes(self, mock_ollama_client, temp_dir):
        """load_themes() should return the list of loaded themes."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        data_dir.mkdir(parents=True, exist_ok=True)

        theme_data = [
            {
                "id": "test-theme",
                "name": "Test Theme",
                "description": "A test theme",
                "keywords": ["test"],
                "item_count": 1,
                "created_at": "2026-01-26T12:00:00",
                "updated_at": "2026-01-26T12:00:00",
            }
        ]
        with open(data_dir / "themes.json", "w") as f:
            json.dump(theme_data, f)

        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        result = generator.load_themes()

        assert result == generator.themes
        assert len(result) == 1
        assert result[0].id == "test-theme"


class TestSaveAssignments:
    """Tests for save_assignments() method that persists assignments to assignments.json."""

    def test_save_assignments_creates_assignments_file(self, mock_ollama_client, temp_dir):
        """save_assignments() should create assignments.json file in data_dir."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        data_dir.mkdir(parents=True, exist_ok=True)

        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        # Add an assignment to save
        assignment = ThemeAssignment(
            item_id="item-123",
            theme_id="personal-growth",
            confidence=0.85,
            assigned_at=datetime(2026, 1, 15, 10, 0, 0),
        )
        generator.assignments = [assignment]

        generator.save_assignments()

        assert (data_dir / "assignments.json").exists()

    def test_save_assignments_writes_valid_json(self, mock_ollama_client, temp_dir):
        """save_assignments() should write valid JSON to assignments.json."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        data_dir.mkdir(parents=True, exist_ok=True)

        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        assignment = ThemeAssignment(
            item_id="item-123",
            theme_id="personal-growth",
            confidence=0.85,
            assigned_at=datetime(2026, 1, 15, 10, 0, 0),
        )
        generator.assignments = [assignment]

        generator.save_assignments()

        # Should be able to parse the JSON
        with open(data_dir / "assignments.json") as f:
            data = json.load(f)

        assert isinstance(data, list)

    def test_save_assignments_serializes_assignment_data_correctly(self, mock_ollama_client, temp_dir):
        """save_assignments() should serialize all assignment fields correctly."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        data_dir.mkdir(parents=True, exist_ok=True)

        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        assignment = ThemeAssignment(
            item_id="book-456",
            theme_id="systems-thinking",
            confidence=0.92,
            assigned_at=datetime(2026, 1, 20, 14, 30, 0),
        )
        generator.assignments = [assignment]

        generator.save_assignments()

        with open(data_dir / "assignments.json") as f:
            data = json.load(f)

        assert len(data) == 1
        assert data[0]["item_id"] == "book-456"
        assert data[0]["theme_id"] == "systems-thinking"
        assert data[0]["confidence"] == 0.92
        assert data[0]["assigned_at"] == "2026-01-20T14:30:00"

    def test_save_assignments_saves_multiple_assignments(self, mock_ollama_client, temp_dir):
        """save_assignments() should save multiple assignments to assignments.json."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        data_dir.mkdir(parents=True, exist_ok=True)

        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        assignment1 = ThemeAssignment(
            item_id="item-123",
            theme_id="personal-growth",
            confidence=0.85,
            assigned_at=datetime(2026, 1, 15, 10, 0, 0),
        )
        assignment2 = ThemeAssignment(
            item_id="item-123",
            theme_id="technology",
            confidence=0.45,
            assigned_at=datetime(2026, 1, 15, 10, 0, 0),
        )
        assignment3 = ThemeAssignment(
            item_id="item-456",
            theme_id="personal-growth",
            confidence=0.72,
            assigned_at=datetime(2026, 1, 16, 11, 0, 0),
        )
        generator.assignments = [assignment1, assignment2, assignment3]

        generator.save_assignments()

        with open(data_dir / "assignments.json") as f:
            data = json.load(f)

        assert len(data) == 3
        assert data[0]["item_id"] == "item-123"
        assert data[0]["theme_id"] == "personal-growth"
        assert data[1]["item_id"] == "item-123"
        assert data[1]["theme_id"] == "technology"
        assert data[2]["item_id"] == "item-456"

    def test_save_assignments_overwrites_existing_file(self, mock_ollama_client, temp_dir):
        """save_assignments() should overwrite existing assignments.json file."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        data_dir.mkdir(parents=True, exist_ok=True)

        # Create initial assignments file with old data
        old_data = [{"item_id": "old-item", "theme_id": "old-theme", "confidence": 0.5}]
        with open(data_dir / "assignments.json", "w") as f:
            json.dump(old_data, f)

        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        assignment = ThemeAssignment(
            item_id="new-item",
            theme_id="new-theme",
            confidence=0.99,
            assigned_at=datetime(2026, 1, 26, 12, 0, 0),
        )
        generator.assignments = [assignment]

        generator.save_assignments()

        with open(data_dir / "assignments.json") as f:
            data = json.load(f)

        assert len(data) == 1
        assert data[0]["item_id"] == "new-item"
        assert data[0]["theme_id"] == "new-theme"

    def test_save_assignments_empty_list(self, mock_ollama_client, temp_dir):
        """save_assignments() should save empty list when no assignments exist."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        data_dir.mkdir(parents=True, exist_ok=True)

        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)
        generator.assignments = []

        generator.save_assignments()

        with open(data_dir / "assignments.json") as f:
            data = json.load(f)

        assert data == []

    def test_save_assignments_uses_indentation_for_readability(self, mock_ollama_client, temp_dir):
        """save_assignments() should use indentation for human-readable JSON."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        data_dir.mkdir(parents=True, exist_ok=True)

        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        assignment = ThemeAssignment(
            item_id="item-123",
            theme_id="test-theme",
            confidence=0.75,
            assigned_at=datetime(2026, 1, 26, 12, 0, 0),
        )
        generator.assignments = [assignment]

        generator.save_assignments()

        with open(data_dir / "assignments.json") as f:
            content = f.read()

        # Check for newlines (indented JSON has newlines)
        assert "\n" in content

    def test_save_assignments_preserves_float_precision(self, mock_ollama_client, temp_dir):
        """save_assignments() should preserve confidence score precision."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        data_dir.mkdir(parents=True, exist_ok=True)

        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        assignment = ThemeAssignment(
            item_id="item-123",
            theme_id="test-theme",
            confidence=0.123456789,
            assigned_at=datetime(2026, 1, 26, 12, 0, 0),
        )
        generator.assignments = [assignment]

        generator.save_assignments()

        with open(data_dir / "assignments.json") as f:
            data = json.load(f)

        # JSON should preserve float precision
        assert data[0]["confidence"] == 0.123456789


class TestLoadAssignments:
    """Tests for load_assignments() method that reads assignments from assignments.json."""

    def test_load_assignments_reads_from_assignments_file(self, mock_ollama_client, temp_dir):
        """load_assignments() should read from assignments.json file in data_dir."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        data_dir.mkdir(parents=True, exist_ok=True)

        # Create an assignments.json file
        assignment_data = [
            {
                "item_id": "item-123",
                "theme_id": "personal-growth",
                "confidence": 0.85,
                "assigned_at": "2026-01-15T10:00:00",
            }
        ]
        with open(data_dir / "assignments.json", "w") as f:
            json.dump(assignment_data, f)

        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        generator.load_assignments()

        assert len(generator.assignments) == 1

    def test_load_assignments_parses_assignment_objects_correctly(self, mock_ollama_client, temp_dir):
        """load_assignments() should parse JSON into ThemeAssignment objects with correct fields."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        data_dir.mkdir(parents=True, exist_ok=True)

        assignment_data = [
            {
                "item_id": "book-456",
                "theme_id": "systems-thinking",
                "confidence": 0.92,
                "assigned_at": "2026-01-20T14:30:00",
            }
        ]
        with open(data_dir / "assignments.json", "w") as f:
            json.dump(assignment_data, f)

        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        generator.load_assignments()

        assignment = generator.assignments[0]
        assert isinstance(assignment, ThemeAssignment)
        assert assignment.item_id == "book-456"
        assert assignment.theme_id == "systems-thinking"
        assert assignment.confidence == 0.92
        assert assignment.assigned_at == datetime(2026, 1, 20, 14, 30, 0)

    def test_load_assignments_loads_multiple_assignments(self, mock_ollama_client, temp_dir):
        """load_assignments() should load multiple assignments from assignments.json."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        data_dir.mkdir(parents=True, exist_ok=True)

        assignment_data = [
            {
                "item_id": "item-123",
                "theme_id": "personal-growth",
                "confidence": 0.85,
                "assigned_at": "2026-01-15T10:00:00",
            },
            {
                "item_id": "item-123",
                "theme_id": "technology",
                "confidence": 0.45,
                "assigned_at": "2026-01-15T10:00:00",
            },
            {
                "item_id": "item-456",
                "theme_id": "personal-growth",
                "confidence": 0.72,
                "assigned_at": "2026-01-16T11:00:00",
            },
        ]
        with open(data_dir / "assignments.json", "w") as f:
            json.dump(assignment_data, f)

        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        generator.load_assignments()

        assert len(generator.assignments) == 3
        assert generator.assignments[0].item_id == "item-123"
        assert generator.assignments[0].theme_id == "personal-growth"
        assert generator.assignments[1].item_id == "item-123"
        assert generator.assignments[1].theme_id == "technology"
        assert generator.assignments[2].item_id == "item-456"

    def test_load_assignments_handles_empty_file(self, mock_ollama_client, temp_dir):
        """load_assignments() should handle empty assignments list gracefully."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        data_dir.mkdir(parents=True, exist_ok=True)

        # Create an empty JSON array
        with open(data_dir / "assignments.json", "w") as f:
            json.dump([], f)

        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        generator.load_assignments()

        assert generator.assignments == []

    def test_load_assignments_handles_missing_file(self, mock_ollama_client, temp_dir):
        """load_assignments() should return empty list when assignments.json doesn't exist."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        data_dir.mkdir(parents=True, exist_ok=True)

        # Don't create assignments.json file
        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        generator.load_assignments()

        assert generator.assignments == []

    def test_load_assignments_replaces_existing_assignments(self, mock_ollama_client, temp_dir):
        """load_assignments() should replace any existing assignments in memory."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        data_dir.mkdir(parents=True, exist_ok=True)

        # Create assignments.json with new data
        assignment_data = [
            {
                "item_id": "new-item",
                "theme_id": "new-theme",
                "confidence": 0.99,
                "assigned_at": "2026-01-26T12:00:00",
            }
        ]
        with open(data_dir / "assignments.json", "w") as f:
            json.dump(assignment_data, f)

        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        # Set some existing assignments in memory
        existing_assignment = ThemeAssignment(
            item_id="existing-item",
            theme_id="existing-theme",
            confidence=0.5,
            assigned_at=datetime(2026, 1, 1, 0, 0, 0),
        )
        generator.assignments = [existing_assignment]

        generator.load_assignments()

        # Should have replaced with file contents
        assert len(generator.assignments) == 1
        assert generator.assignments[0].item_id == "new-item"

    def test_load_assignments_returns_loaded_assignments(self, mock_ollama_client, temp_dir):
        """load_assignments() should return the list of loaded assignments."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        data_dir.mkdir(parents=True, exist_ok=True)

        assignment_data = [
            {
                "item_id": "test-item",
                "theme_id": "test-theme",
                "confidence": 0.75,
                "assigned_at": "2026-01-26T12:00:00",
            }
        ]
        with open(data_dir / "assignments.json", "w") as f:
            json.dump(assignment_data, f)

        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        result = generator.load_assignments()

        assert result == generator.assignments
        assert len(result) == 1
        assert result[0].item_id == "test-item"

    def test_load_assignments_preserves_float_precision(self, mock_ollama_client, temp_dir):
        """load_assignments() should preserve confidence score precision when loading."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        data_dir.mkdir(parents=True, exist_ok=True)

        assignment_data = [
            {
                "item_id": "item-123",
                "theme_id": "test-theme",
                "confidence": 0.123456789,
                "assigned_at": "2026-01-26T12:00:00",
            }
        ]
        with open(data_dir / "assignments.json", "w") as f:
            json.dump(assignment_data, f)

        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        generator.load_assignments()

        assert generator.assignments[0].confidence == 0.123456789
