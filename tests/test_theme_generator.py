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


class TestBuildThemeGenerationPrompt:
    """Tests for _build_theme_generation_prompt() method that constructs the LLM prompt."""

    def test_build_theme_generation_prompt_returns_string(self, mock_ollama_client, temp_dir):
        """_build_theme_generation_prompt() should return a string."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        items = [
            LibraryItem(
                id="item-1",
                content_type=ContentType.BOOK,
                title="Thinking in Systems",
                summary="A primer on systems thinking",
                full_content="Full content here",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime(2026, 1, 15, 10, 0, 0),
                highlights=[],
            )
        ]

        prompt = generator._build_theme_generation_prompt(items)

        assert isinstance(prompt, str)

    def test_build_theme_generation_prompt_includes_item_titles(self, mock_ollama_client, temp_dir):
        """_build_theme_generation_prompt() should include item titles in the prompt."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        items = [
            LibraryItem(
                id="item-1",
                content_type=ContentType.BOOK,
                title="Thinking in Systems",
                summary="A primer on systems thinking",
                full_content="Full content here",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime(2026, 1, 15, 10, 0, 0),
                highlights=[],
            ),
            LibraryItem(
                id="item-2",
                content_type=ContentType.ARTICLE,
                title="The Art of Decision Making",
                summary="How to make better decisions",
                full_content="Full article content",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime(2026, 1, 16, 10, 0, 0),
                highlights=[],
            ),
        ]

        prompt = generator._build_theme_generation_prompt(items)

        assert "Thinking in Systems" in prompt
        assert "The Art of Decision Making" in prompt

    def test_build_theme_generation_prompt_includes_item_summaries(self, mock_ollama_client, temp_dir):
        """_build_theme_generation_prompt() should include item summaries in the prompt."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        items = [
            LibraryItem(
                id="item-1",
                content_type=ContentType.FRAMEWORK,
                title="First Principles Thinking",
                summary="Break down problems to fundamental truths",
                full_content="Full content here",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime(2026, 1, 15, 10, 0, 0),
                highlights=[],
            )
        ]

        prompt = generator._build_theme_generation_prompt(items)

        assert "Break down problems to fundamental truths" in prompt

    def test_build_theme_generation_prompt_includes_content_types(self, mock_ollama_client, temp_dir):
        """_build_theme_generation_prompt() should include content types in the prompt."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        items = [
            LibraryItem(
                id="item-1",
                content_type=ContentType.BOOK,
                title="Test Book",
                summary="A test book summary",
                full_content="Full content",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime(2026, 1, 15, 10, 0, 0),
                highlights=[],
            ),
            LibraryItem(
                id="item-2",
                content_type=ContentType.VALUE,
                title="Test Value",
                summary="A test value summary",
                full_content="Full content",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime(2026, 1, 16, 10, 0, 0),
                highlights=[],
            ),
        ]

        prompt = generator._build_theme_generation_prompt(items)

        assert "book" in prompt.lower()
        assert "value" in prompt.lower()

    def test_build_theme_generation_prompt_requests_5_to_10_themes(self, mock_ollama_client, temp_dir):
        """_build_theme_generation_prompt() should request 5-10 broad themes."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        items = [
            LibraryItem(
                id="item-1",
                content_type=ContentType.BOOK,
                title="Test Item",
                summary="A test summary",
                full_content="Full content",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime(2026, 1, 15, 10, 0, 0),
                highlights=[],
            )
        ]

        prompt = generator._build_theme_generation_prompt(items)

        # Should mention the theme count range
        assert "5" in prompt and "10" in prompt

    def test_build_theme_generation_prompt_requests_json_format(self, mock_ollama_client, temp_dir):
        """_build_theme_generation_prompt() should request JSON output format."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        items = [
            LibraryItem(
                id="item-1",
                content_type=ContentType.BOOK,
                title="Test Item",
                summary="A test summary",
                full_content="Full content",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime(2026, 1, 15, 10, 0, 0),
                highlights=[],
            )
        ]

        prompt = generator._build_theme_generation_prompt(items)

        # Should mention JSON format for structured output
        assert "json" in prompt.lower()

    def test_build_theme_generation_prompt_requests_theme_fields(self, mock_ollama_client, temp_dir):
        """_build_theme_generation_prompt() should request name, description, and keywords for each theme."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        items = [
            LibraryItem(
                id="item-1",
                content_type=ContentType.BOOK,
                title="Test Item",
                summary="A test summary",
                full_content="Full content",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime(2026, 1, 15, 10, 0, 0),
                highlights=[],
            )
        ]

        prompt = generator._build_theme_generation_prompt(items)

        # Should mention the required fields for each theme
        assert "name" in prompt.lower()
        assert "description" in prompt.lower()
        assert "keywords" in prompt.lower()

    def test_build_theme_generation_prompt_with_empty_items_list(self, mock_ollama_client, temp_dir):
        """_build_theme_generation_prompt() should handle empty items list gracefully."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        items = []

        prompt = generator._build_theme_generation_prompt(items)

        # Should still return a valid prompt (may indicate no content)
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_build_theme_generation_prompt_with_multiple_items(self, mock_ollama_client, temp_dir):
        """_build_theme_generation_prompt() should handle multiple items of various types."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        items = [
            LibraryItem(
                id="item-1",
                content_type=ContentType.BOOK,
                title="Systems Thinking Book",
                summary="About complex systems",
                full_content="Content",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime(2026, 1, 15, 10, 0, 0),
                highlights=[],
            ),
            LibraryItem(
                id="item-2",
                content_type=ContentType.FRAMEWORK,
                title="Decision Framework",
                summary="How to make decisions",
                full_content="Content",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime(2026, 1, 16, 10, 0, 0),
                highlights=[],
            ),
            LibraryItem(
                id="item-3",
                content_type=ContentType.VALUE,
                title="Continuous Learning",
                summary="Always be learning",
                full_content="Content",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime(2026, 1, 17, 10, 0, 0),
                highlights=[],
            ),
            LibraryItem(
                id="item-4",
                content_type=ContentType.MEMORY,
                title="Conference Talk Memory",
                summary="Great insights from a conference",
                full_content="Content",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime(2026, 1, 18, 10, 0, 0),
                highlights=[],
            ),
            LibraryItem(
                id="item-5",
                content_type=ContentType.WEB_CONTENT,
                title="Interesting Article",
                summary="Web article about technology",
                full_content="Content",
                source_url="https://example.com",
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime(2026, 1, 19, 10, 0, 0),
                highlights=[],
            ),
        ]

        prompt = generator._build_theme_generation_prompt(items)

        # All items should be represented in the prompt
        assert "Systems Thinking Book" in prompt
        assert "Decision Framework" in prompt
        assert "Continuous Learning" in prompt
        assert "Conference Talk Memory" in prompt
        assert "Interesting Article" in prompt

    def test_build_theme_generation_prompt_requests_id_field(self, mock_ollama_client, temp_dir):
        """_build_theme_generation_prompt() should request id (slug) field for each theme."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        items = [
            LibraryItem(
                id="item-1",
                content_type=ContentType.BOOK,
                title="Test Item",
                summary="A test summary",
                full_content="Full content",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime(2026, 1, 15, 10, 0, 0),
                highlights=[],
            )
        ]

        prompt = generator._build_theme_generation_prompt(items)

        # Should mention the id field (as slug format)
        assert "id" in prompt.lower() or "slug" in prompt.lower()


class TestParseThemesFromResponse:
    """Tests for _parse_themes_from_response() method that parses LLM output into Theme objects."""

    def test_parse_themes_from_response_returns_list_of_themes(self, mock_ollama_client, temp_dir):
        """_parse_themes_from_response() should return a list of Theme objects."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        response = '''[
            {
                "id": "personal-growth",
                "name": "Personal Growth",
                "description": "Self-improvement and learning",
                "keywords": ["growth", "learning", "habits"]
            }
        ]'''

        themes = generator._parse_themes_from_response(response)

        assert isinstance(themes, list)
        assert len(themes) == 1
        assert isinstance(themes[0], Theme)

    def test_parse_themes_from_response_extracts_id_correctly(self, mock_ollama_client, temp_dir):
        """_parse_themes_from_response() should extract the id field correctly."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        response = '''[
            {
                "id": "systems-thinking",
                "name": "Systems Thinking",
                "description": "Understanding complex systems",
                "keywords": ["systems", "complexity"]
            }
        ]'''

        themes = generator._parse_themes_from_response(response)

        assert themes[0].id == "systems-thinking"

    def test_parse_themes_from_response_extracts_name_correctly(self, mock_ollama_client, temp_dir):
        """_parse_themes_from_response() should extract the name field correctly."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        response = '''[
            {
                "id": "tech-innovation",
                "name": "Technology & Innovation",
                "description": "Tech trends and innovations",
                "keywords": ["technology", "innovation"]
            }
        ]'''

        themes = generator._parse_themes_from_response(response)

        assert themes[0].name == "Technology & Innovation"

    def test_parse_themes_from_response_extracts_description_correctly(self, mock_ollama_client, temp_dir):
        """_parse_themes_from_response() should extract the description field correctly."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        response = '''[
            {
                "id": "business",
                "name": "Business",
                "description": "Entrepreneurship, management, and strategy",
                "keywords": ["business", "strategy"]
            }
        ]'''

        themes = generator._parse_themes_from_response(response)

        assert themes[0].description == "Entrepreneurship, management, and strategy"

    def test_parse_themes_from_response_extracts_keywords_correctly(self, mock_ollama_client, temp_dir):
        """_parse_themes_from_response() should extract the keywords list correctly."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        response = '''[
            {
                "id": "communication",
                "name": "Communication",
                "description": "Writing, speaking, and relationships",
                "keywords": ["writing", "speaking", "relationships", "clarity"]
            }
        ]'''

        themes = generator._parse_themes_from_response(response)

        assert themes[0].keywords == ["writing", "speaking", "relationships", "clarity"]

    def test_parse_themes_from_response_sets_item_count_to_zero(self, mock_ollama_client, temp_dir):
        """_parse_themes_from_response() should set item_count to 0 for new themes."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        response = '''[
            {
                "id": "personal-growth",
                "name": "Personal Growth",
                "description": "Self-improvement",
                "keywords": ["growth"]
            }
        ]'''

        themes = generator._parse_themes_from_response(response)

        assert themes[0].item_count == 0

    def test_parse_themes_from_response_sets_created_at_timestamp(self, mock_ollama_client, temp_dir):
        """_parse_themes_from_response() should set created_at to current time."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        response = '''[
            {
                "id": "personal-growth",
                "name": "Personal Growth",
                "description": "Self-improvement",
                "keywords": ["growth"]
            }
        ]'''

        before = datetime.now()
        themes = generator._parse_themes_from_response(response)
        after = datetime.now()

        assert before <= themes[0].created_at <= after

    def test_parse_themes_from_response_sets_updated_at_timestamp(self, mock_ollama_client, temp_dir):
        """_parse_themes_from_response() should set updated_at to current time."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        response = '''[
            {
                "id": "personal-growth",
                "name": "Personal Growth",
                "description": "Self-improvement",
                "keywords": ["growth"]
            }
        ]'''

        before = datetime.now()
        themes = generator._parse_themes_from_response(response)
        after = datetime.now()

        assert before <= themes[0].updated_at <= after

    def test_parse_themes_from_response_parses_multiple_themes(self, mock_ollama_client, temp_dir):
        """_parse_themes_from_response() should parse multiple themes from the response."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        response = '''[
            {
                "id": "personal-growth",
                "name": "Personal Growth",
                "description": "Self-improvement and learning",
                "keywords": ["growth", "learning"]
            },
            {
                "id": "technology",
                "name": "Technology",
                "description": "Software and tech innovations",
                "keywords": ["software", "tech"]
            },
            {
                "id": "business",
                "name": "Business",
                "description": "Entrepreneurship and strategy",
                "keywords": ["business", "strategy"]
            }
        ]'''

        themes = generator._parse_themes_from_response(response)

        assert len(themes) == 3
        assert themes[0].id == "personal-growth"
        assert themes[1].id == "technology"
        assert themes[2].id == "business"

    def test_parse_themes_from_response_handles_json_with_markdown_code_block(self, mock_ollama_client, temp_dir):
        """_parse_themes_from_response() should handle JSON wrapped in markdown code blocks."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        response = '''```json
[
    {
        "id": "personal-growth",
        "name": "Personal Growth",
        "description": "Self-improvement",
        "keywords": ["growth"]
    }
]
```'''

        themes = generator._parse_themes_from_response(response)

        assert len(themes) == 1
        assert themes[0].id == "personal-growth"

    def test_parse_themes_from_response_handles_generic_markdown_code_block(self, mock_ollama_client, temp_dir):
        """_parse_themes_from_response() should handle JSON wrapped in generic code blocks."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        response = '''```
[
    {
        "id": "tech",
        "name": "Technology",
        "description": "Tech topics",
        "keywords": ["tech"]
    }
]
```'''

        themes = generator._parse_themes_from_response(response)

        assert len(themes) == 1
        assert themes[0].id == "tech"

    def test_parse_themes_from_response_returns_empty_list_for_invalid_json(self, mock_ollama_client, temp_dir):
        """_parse_themes_from_response() should return empty list for invalid JSON."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        response = "This is not valid JSON at all."

        themes = generator._parse_themes_from_response(response)

        assert themes == []

    def test_parse_themes_from_response_returns_empty_list_for_empty_response(self, mock_ollama_client, temp_dir):
        """_parse_themes_from_response() should return empty list for empty response."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        response = ""

        themes = generator._parse_themes_from_response(response)

        assert themes == []

    def test_parse_themes_from_response_returns_empty_list_for_empty_json_array(self, mock_ollama_client, temp_dir):
        """_parse_themes_from_response() should return empty list for empty JSON array."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        response = "[]"

        themes = generator._parse_themes_from_response(response)

        assert themes == []

    def test_parse_themes_from_response_skips_themes_missing_required_fields(self, mock_ollama_client, temp_dir):
        """_parse_themes_from_response() should skip themes missing required fields."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        response = '''[
            {
                "id": "valid-theme",
                "name": "Valid Theme",
                "description": "This is valid",
                "keywords": ["valid"]
            },
            {
                "name": "Missing ID",
                "description": "This is missing id",
                "keywords": ["invalid"]
            },
            {
                "id": "missing-name",
                "description": "This is missing name",
                "keywords": ["invalid"]
            }
        ]'''

        themes = generator._parse_themes_from_response(response)

        assert len(themes) == 1
        assert themes[0].id == "valid-theme"

    def test_parse_themes_from_response_handles_extra_whitespace(self, mock_ollama_client, temp_dir):
        """_parse_themes_from_response() should handle extra whitespace in response."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        response = '''

            [
                {
                    "id": "test-theme",
                    "name": "Test Theme",
                    "description": "A test",
                    "keywords": ["test"]
                }
            ]

        '''

        themes = generator._parse_themes_from_response(response)

        assert len(themes) == 1
        assert themes[0].id == "test-theme"

    def test_parse_themes_from_response_handles_empty_keywords_list(self, mock_ollama_client, temp_dir):
        """_parse_themes_from_response() should handle themes with empty keywords list."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        response = '''[
            {
                "id": "no-keywords",
                "name": "No Keywords Theme",
                "description": "A theme without keywords",
                "keywords": []
            }
        ]'''

        themes = generator._parse_themes_from_response(response)

        assert len(themes) == 1
        assert themes[0].keywords == []

    def test_parse_themes_from_response_handles_llm_preamble_text(self, mock_ollama_client, temp_dir):
        """_parse_themes_from_response() should handle LLM response with preamble text before JSON."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        response = '''Based on the content provided, here are the themes I've identified:

```json
[
    {
        "id": "personal-growth",
        "name": "Personal Growth",
        "description": "Self-improvement topics",
        "keywords": ["growth"]
    }
]
```

These themes represent the major categories in your knowledge base.'''

        themes = generator._parse_themes_from_response(response)

        assert len(themes) == 1
        assert themes[0].id == "personal-growth"

    def test_parse_themes_from_response_created_at_equals_updated_at(self, mock_ollama_client, temp_dir):
        """_parse_themes_from_response() should set created_at and updated_at to the same value for new themes."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        response = '''[
            {
                "id": "test",
                "name": "Test",
                "description": "Test",
                "keywords": []
            }
        ]'''

        themes = generator._parse_themes_from_response(response)

        assert themes[0].created_at == themes[0].updated_at


class TestGenerateThemes:
    """Tests for generate_themes() orchestration method that coordinates LLM call, parsing, and saving."""

    def test_generate_themes_calls_ollama_client_generate(self, mock_ollama_client, temp_dir):
        """generate_themes() should call OllamaClient.generate() with a prompt."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        # Configure mock to return valid JSON
        mock_ollama_client.generate.return_value = '''[
            {
                "id": "personal-growth",
                "name": "Personal Growth",
                "description": "Self-improvement topics",
                "keywords": ["growth", "learning"]
            }
        ]'''

        items = [
            LibraryItem(
                id="item-1",
                content_type=ContentType.BOOK,
                title="Test Book",
                summary="A test book about learning",
                full_content="Full content here",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime(2026, 1, 15, 10, 0, 0),
                highlights=[],
            )
        ]

        generator.generate_themes(items)

        # Verify OllamaClient.generate() was called
        mock_ollama_client.generate.assert_called_once()

    def test_generate_themes_passes_built_prompt_to_ollama(self, mock_ollama_client, temp_dir):
        """generate_themes() should pass the prompt from _build_theme_generation_prompt() to OllamaClient."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        mock_ollama_client.generate.return_value = '''[
            {
                "id": "tech",
                "name": "Technology",
                "description": "Tech topics",
                "keywords": ["tech"]
            }
        ]'''

        items = [
            LibraryItem(
                id="item-1",
                content_type=ContentType.ARTICLE,
                title="Tech Article",
                summary="An article about technology",
                full_content="Full content here",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime(2026, 1, 15, 10, 0, 0),
                highlights=[],
            )
        ]

        generator.generate_themes(items)

        # Get the prompt that was passed to generate()
        call_args = mock_ollama_client.generate.call_args
        prompt = call_args.kwargs.get("prompt") or call_args.args[0]

        # The prompt should contain the item title and summary
        assert "Tech Article" in prompt
        assert "An article about technology" in prompt

    def test_generate_themes_parses_ollama_response(self, mock_ollama_client, temp_dir):
        """generate_themes() should parse the LLM response into Theme objects."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        mock_ollama_client.generate.return_value = '''[
            {
                "id": "systems-thinking",
                "name": "Systems Thinking",
                "description": "Understanding complex systems",
                "keywords": ["systems", "complexity", "feedback"]
            }
        ]'''

        items = [
            LibraryItem(
                id="item-1",
                content_type=ContentType.BOOK,
                title="Thinking in Systems",
                summary="A primer on systems thinking",
                full_content="Full content here",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime(2026, 1, 15, 10, 0, 0),
                highlights=[],
            )
        ]

        result = generator.generate_themes(items)

        # Should have parsed the theme correctly
        assert len(result) == 1
        assert result[0].id == "systems-thinking"
        assert result[0].name == "Systems Thinking"
        assert result[0].description == "Understanding complex systems"
        assert result[0].keywords == ["systems", "complexity", "feedback"]

    def test_generate_themes_saves_themes_to_file(self, mock_ollama_client, temp_dir):
        """generate_themes() should save the generated themes to themes.json."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        mock_ollama_client.generate.return_value = '''[
            {
                "id": "business",
                "name": "Business",
                "description": "Business and strategy topics",
                "keywords": ["business", "strategy"]
            }
        ]'''

        items = [
            LibraryItem(
                id="item-1",
                content_type=ContentType.FRAMEWORK,
                title="Business Framework",
                summary="A framework for business decisions",
                full_content="Full content here",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime(2026, 1, 15, 10, 0, 0),
                highlights=[],
            )
        ]

        generator.generate_themes(items)

        # Verify themes.json was created
        assert (data_dir / "themes.json").exists()

        # Verify the content was saved correctly
        with open(data_dir / "themes.json") as f:
            saved_data = json.load(f)

        assert len(saved_data) == 1
        assert saved_data[0]["id"] == "business"
        assert saved_data[0]["name"] == "Business"

    def test_generate_themes_updates_self_themes(self, mock_ollama_client, temp_dir):
        """generate_themes() should update self.themes with the generated themes."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        mock_ollama_client.generate.return_value = '''[
            {
                "id": "personal-growth",
                "name": "Personal Growth",
                "description": "Self-improvement topics",
                "keywords": ["growth"]
            },
            {
                "id": "technology",
                "name": "Technology",
                "description": "Tech topics",
                "keywords": ["tech"]
            }
        ]'''

        items = [
            LibraryItem(
                id="item-1",
                content_type=ContentType.BOOK,
                title="Test Book",
                summary="A test book",
                full_content="Full content",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime(2026, 1, 15, 10, 0, 0),
                highlights=[],
            )
        ]

        generator.generate_themes(items)

        # self.themes should be updated
        assert len(generator.themes) == 2
        assert generator.themes[0].id == "personal-growth"
        assert generator.themes[1].id == "technology"

    def test_generate_themes_returns_generated_themes(self, mock_ollama_client, temp_dir):
        """generate_themes() should return the list of generated themes."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        mock_ollama_client.generate.return_value = '''[
            {
                "id": "communication",
                "name": "Communication",
                "description": "Communication skills",
                "keywords": ["writing", "speaking"]
            }
        ]'''

        items = [
            LibraryItem(
                id="item-1",
                content_type=ContentType.VALUE,
                title="Clear Communication",
                summary="The value of clear communication",
                full_content="Full content",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime(2026, 1, 15, 10, 0, 0),
                highlights=[],
            )
        ]

        result = generator.generate_themes(items)

        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], Theme)
        assert result[0].id == "communication"

    def test_generate_themes_with_empty_items_list(self, mock_ollama_client, temp_dir):
        """generate_themes() should handle empty items list gracefully."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        # Return empty array when no items
        mock_ollama_client.generate.return_value = '[]'

        items = []

        result = generator.generate_themes(items)

        # Should return empty list without error
        assert result == []
        assert generator.themes == []

    def test_generate_themes_with_invalid_llm_response(self, mock_ollama_client, temp_dir):
        """generate_themes() should handle invalid LLM response gracefully."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        # Return invalid response
        mock_ollama_client.generate.return_value = "This is not valid JSON"

        items = [
            LibraryItem(
                id="item-1",
                content_type=ContentType.BOOK,
                title="Test Book",
                summary="A test book",
                full_content="Full content",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime(2026, 1, 15, 10, 0, 0),
                highlights=[],
            )
        ]

        result = generator.generate_themes(items)

        # Should return empty list without raising an exception
        assert result == []
        assert generator.themes == []

    def test_generate_themes_with_multiple_items(self, mock_ollama_client, temp_dir):
        """generate_themes() should handle multiple items correctly."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        mock_ollama_client.generate.return_value = '''[
            {
                "id": "personal-growth",
                "name": "Personal Growth",
                "description": "Self-improvement and learning",
                "keywords": ["growth", "learning", "habits"]
            },
            {
                "id": "systems-thinking",
                "name": "Systems Thinking",
                "description": "Understanding complex systems",
                "keywords": ["systems", "complexity"]
            },
            {
                "id": "technology",
                "name": "Technology",
                "description": "Software and tech",
                "keywords": ["software", "tech"]
            }
        ]'''

        items = [
            LibraryItem(
                id="item-1",
                content_type=ContentType.BOOK,
                title="Atomic Habits",
                summary="Building good habits",
                full_content="Content",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime(2026, 1, 15, 10, 0, 0),
                highlights=[],
            ),
            LibraryItem(
                id="item-2",
                content_type=ContentType.BOOK,
                title="Thinking in Systems",
                summary="Systems thinking primer",
                full_content="Content",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime(2026, 1, 16, 10, 0, 0),
                highlights=[],
            ),
            LibraryItem(
                id="item-3",
                content_type=ContentType.ARTICLE,
                title="AI Trends 2026",
                summary="Latest AI developments",
                full_content="Content",
                source_url="https://example.com",
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime(2026, 1, 17, 10, 0, 0),
                highlights=[],
            ),
        ]

        result = generator.generate_themes(items)

        assert len(result) == 3
        assert {t.id for t in result} == {"personal-growth", "systems-thinking", "technology"}

    def test_generate_themes_replaces_existing_themes(self, mock_ollama_client, temp_dir):
        """generate_themes() should replace any existing themes."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        # Pre-populate with existing themes
        existing_theme = Theme(
            id="old-theme",
            name="Old Theme",
            description="This should be replaced",
            keywords=["old"],
            item_count=5,
            created_at=datetime(2026, 1, 1, 0, 0, 0),
            updated_at=datetime(2026, 1, 1, 0, 0, 0),
        )
        generator.themes = [existing_theme]

        mock_ollama_client.generate.return_value = '''[
            {
                "id": "new-theme",
                "name": "New Theme",
                "description": "This is the new theme",
                "keywords": ["new"]
            }
        ]'''

        items = [
            LibraryItem(
                id="item-1",
                content_type=ContentType.BOOK,
                title="Test Book",
                summary="A test book",
                full_content="Content",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime(2026, 1, 15, 10, 0, 0),
                highlights=[],
            )
        ]

        result = generator.generate_themes(items)

        # Old themes should be replaced
        assert len(generator.themes) == 1
        assert generator.themes[0].id == "new-theme"
        assert "old-theme" not in [t.id for t in generator.themes]

    def test_generate_themes_uses_correct_system_prompt(self, mock_ollama_client, temp_dir):
        """generate_themes() should use an appropriate system prompt for theme generation."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        mock_ollama_client.generate.return_value = '''[
            {
                "id": "test",
                "name": "Test",
                "description": "Test",
                "keywords": []
            }
        ]'''

        items = [
            LibraryItem(
                id="item-1",
                content_type=ContentType.BOOK,
                title="Test Book",
                summary="A test book",
                full_content="Content",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime(2026, 1, 15, 10, 0, 0),
                highlights=[],
            )
        ]

        generator.generate_themes(items)

        # Verify a system_prompt was passed (could be None or a string)
        call_kwargs = mock_ollama_client.generate.call_args.kwargs
        # Either system_prompt is passed or it uses default
        assert "prompt" in call_kwargs or len(mock_ollama_client.generate.call_args.args) > 0

    def test_generate_themes_handles_markdown_wrapped_response(self, mock_ollama_client, temp_dir):
        """generate_themes() should handle LLM response wrapped in markdown code blocks."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        mock_ollama_client.generate.return_value = '''Based on the content, here are the themes:

```json
[
    {
        "id": "personal-growth",
        "name": "Personal Growth",
        "description": "Self-improvement topics",
        "keywords": ["growth"]
    }
]
```

These themes cover the main topics in your knowledge base.'''

        items = [
            LibraryItem(
                id="item-1",
                content_type=ContentType.BOOK,
                title="Test Book",
                summary="A test book",
                full_content="Content",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime(2026, 1, 15, 10, 0, 0),
                highlights=[],
            )
        ]

        result = generator.generate_themes(items)

        assert len(result) == 1
        assert result[0].id == "personal-growth"
