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


class TestBuildAssignmentPrompt:
    """Tests for _build_assignment_prompt() method that constructs the prompt for item-to-theme assignment."""

    def test_build_assignment_prompt_returns_string(self, mock_ollama_client, temp_dir):
        """_build_assignment_prompt() should return a string."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        themes = [
            Theme(
                id="personal-growth",
                name="Personal Growth",
                description="Self-improvement and learning",
                keywords=["growth", "learning", "habits"],
                item_count=0,
                created_at=datetime(2026, 1, 15, 10, 0, 0),
                updated_at=datetime(2026, 1, 15, 10, 0, 0),
            )
        ]

        items = [
            LibraryItem(
                id="item-1",
                content_type=ContentType.BOOK,
                title="Atomic Habits",
                summary="Building good habits for life",
                full_content="Full content here",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime(2026, 1, 15, 10, 0, 0),
                highlights=[],
            )
        ]

        prompt = generator._build_assignment_prompt(items, themes)

        assert isinstance(prompt, str)

    def test_build_assignment_prompt_includes_theme_names(self, mock_ollama_client, temp_dir):
        """_build_assignment_prompt() should include theme names in the prompt."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        themes = [
            Theme(
                id="personal-growth",
                name="Personal Growth",
                description="Self-improvement topics",
                keywords=["growth"],
                item_count=0,
                created_at=datetime(2026, 1, 15, 10, 0, 0),
                updated_at=datetime(2026, 1, 15, 10, 0, 0),
            ),
            Theme(
                id="technology",
                name="Technology",
                description="Tech topics",
                keywords=["tech"],
                item_count=0,
                created_at=datetime(2026, 1, 15, 10, 0, 0),
                updated_at=datetime(2026, 1, 15, 10, 0, 0),
            ),
        ]

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

        prompt = generator._build_assignment_prompt(items, themes)

        assert "Personal Growth" in prompt
        assert "Technology" in prompt

    def test_build_assignment_prompt_includes_theme_ids(self, mock_ollama_client, temp_dir):
        """_build_assignment_prompt() should include theme IDs for assignment output."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        themes = [
            Theme(
                id="systems-thinking",
                name="Systems Thinking",
                description="Understanding complex systems",
                keywords=["systems"],
                item_count=0,
                created_at=datetime(2026, 1, 15, 10, 0, 0),
                updated_at=datetime(2026, 1, 15, 10, 0, 0),
            )
        ]

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

        prompt = generator._build_assignment_prompt(items, themes)

        assert "systems-thinking" in prompt

    def test_build_assignment_prompt_includes_theme_descriptions(self, mock_ollama_client, temp_dir):
        """_build_assignment_prompt() should include theme descriptions to help LLM understand context."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        themes = [
            Theme(
                id="business",
                name="Business",
                description="Entrepreneurship, management, and strategy",
                keywords=["business", "strategy"],
                item_count=0,
                created_at=datetime(2026, 1, 15, 10, 0, 0),
                updated_at=datetime(2026, 1, 15, 10, 0, 0),
            )
        ]

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

        prompt = generator._build_assignment_prompt(items, themes)

        assert "Entrepreneurship, management, and strategy" in prompt

    def test_build_assignment_prompt_includes_theme_keywords(self, mock_ollama_client, temp_dir):
        """_build_assignment_prompt() should include theme keywords for better matching."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        themes = [
            Theme(
                id="communication",
                name="Communication",
                description="Communication skills",
                keywords=["writing", "speaking", "relationships"],
                item_count=0,
                created_at=datetime(2026, 1, 15, 10, 0, 0),
                updated_at=datetime(2026, 1, 15, 10, 0, 0),
            )
        ]

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

        prompt = generator._build_assignment_prompt(items, themes)

        # Keywords should appear in the prompt
        assert "writing" in prompt.lower() or "speaking" in prompt.lower() or "relationships" in prompt.lower()

    def test_build_assignment_prompt_includes_item_ids(self, mock_ollama_client, temp_dir):
        """_build_assignment_prompt() should include item IDs for assignment output."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        themes = [
            Theme(
                id="personal-growth",
                name="Personal Growth",
                description="Self-improvement",
                keywords=["growth"],
                item_count=0,
                created_at=datetime(2026, 1, 15, 10, 0, 0),
                updated_at=datetime(2026, 1, 15, 10, 0, 0),
            )
        ]

        items = [
            LibraryItem(
                id="book-atomic-habits",
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
            )
        ]

        prompt = generator._build_assignment_prompt(items, themes)

        assert "book-atomic-habits" in prompt

    def test_build_assignment_prompt_includes_item_titles(self, mock_ollama_client, temp_dir):
        """_build_assignment_prompt() should include item titles in the prompt."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        themes = [
            Theme(
                id="personal-growth",
                name="Personal Growth",
                description="Self-improvement",
                keywords=["growth"],
                item_count=0,
                created_at=datetime(2026, 1, 15, 10, 0, 0),
                updated_at=datetime(2026, 1, 15, 10, 0, 0),
            )
        ]

        items = [
            LibraryItem(
                id="item-1",
                content_type=ContentType.BOOK,
                title="The Art of Learning",
                summary="How to master any skill",
                full_content="Content",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime(2026, 1, 15, 10, 0, 0),
                highlights=[],
            )
        ]

        prompt = generator._build_assignment_prompt(items, themes)

        assert "The Art of Learning" in prompt

    def test_build_assignment_prompt_includes_item_summaries(self, mock_ollama_client, temp_dir):
        """_build_assignment_prompt() should include item summaries in the prompt."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        themes = [
            Theme(
                id="technology",
                name="Technology",
                description="Tech topics",
                keywords=["tech"],
                item_count=0,
                created_at=datetime(2026, 1, 15, 10, 0, 0),
                updated_at=datetime(2026, 1, 15, 10, 0, 0),
            )
        ]

        items = [
            LibraryItem(
                id="item-1",
                content_type=ContentType.ARTICLE,
                title="AI Trends",
                summary="Exploring the latest developments in artificial intelligence",
                full_content="Content",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime(2026, 1, 15, 10, 0, 0),
                highlights=[],
            )
        ]

        prompt = generator._build_assignment_prompt(items, themes)

        assert "Exploring the latest developments in artificial intelligence" in prompt

    def test_build_assignment_prompt_includes_item_content_types(self, mock_ollama_client, temp_dir):
        """_build_assignment_prompt() should include item content types in the prompt."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        themes = [
            Theme(
                id="personal-growth",
                name="Personal Growth",
                description="Self-improvement",
                keywords=["growth"],
                item_count=0,
                created_at=datetime(2026, 1, 15, 10, 0, 0),
                updated_at=datetime(2026, 1, 15, 10, 0, 0),
            )
        ]

        items = [
            LibraryItem(
                id="item-1",
                content_type=ContentType.FRAMEWORK,
                title="Decision Framework",
                summary="A framework for making decisions",
                full_content="Content",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime(2026, 1, 15, 10, 0, 0),
                highlights=[],
            )
        ]

        prompt = generator._build_assignment_prompt(items, themes)

        assert "framework" in prompt.lower()

    def test_build_assignment_prompt_requests_json_format(self, mock_ollama_client, temp_dir):
        """_build_assignment_prompt() should request JSON output format."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        themes = [
            Theme(
                id="personal-growth",
                name="Personal Growth",
                description="Self-improvement",
                keywords=["growth"],
                item_count=0,
                created_at=datetime(2026, 1, 15, 10, 0, 0),
                updated_at=datetime(2026, 1, 15, 10, 0, 0),
            )
        ]

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

        prompt = generator._build_assignment_prompt(items, themes)

        assert "json" in prompt.lower()

    def test_build_assignment_prompt_requests_confidence_scores(self, mock_ollama_client, temp_dir):
        """_build_assignment_prompt() should request confidence scores for each assignment."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        themes = [
            Theme(
                id="personal-growth",
                name="Personal Growth",
                description="Self-improvement",
                keywords=["growth"],
                item_count=0,
                created_at=datetime(2026, 1, 15, 10, 0, 0),
                updated_at=datetime(2026, 1, 15, 10, 0, 0),
            )
        ]

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

        prompt = generator._build_assignment_prompt(items, themes)

        assert "confidence" in prompt.lower()

    def test_build_assignment_prompt_requests_item_id_in_output(self, mock_ollama_client, temp_dir):
        """_build_assignment_prompt() should request item_id in the output format."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        themes = [
            Theme(
                id="personal-growth",
                name="Personal Growth",
                description="Self-improvement",
                keywords=["growth"],
                item_count=0,
                created_at=datetime(2026, 1, 15, 10, 0, 0),
                updated_at=datetime(2026, 1, 15, 10, 0, 0),
            )
        ]

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

        prompt = generator._build_assignment_prompt(items, themes)

        assert "item_id" in prompt.lower()

    def test_build_assignment_prompt_requests_theme_id_in_output(self, mock_ollama_client, temp_dir):
        """_build_assignment_prompt() should request theme_id in the output format."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        themes = [
            Theme(
                id="personal-growth",
                name="Personal Growth",
                description="Self-improvement",
                keywords=["growth"],
                item_count=0,
                created_at=datetime(2026, 1, 15, 10, 0, 0),
                updated_at=datetime(2026, 1, 15, 10, 0, 0),
            )
        ]

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

        prompt = generator._build_assignment_prompt(items, themes)

        assert "theme_id" in prompt.lower()

    def test_build_assignment_prompt_with_multiple_items(self, mock_ollama_client, temp_dir):
        """_build_assignment_prompt() should handle multiple items correctly."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        themes = [
            Theme(
                id="personal-growth",
                name="Personal Growth",
                description="Self-improvement",
                keywords=["growth"],
                item_count=0,
                created_at=datetime(2026, 1, 15, 10, 0, 0),
                updated_at=datetime(2026, 1, 15, 10, 0, 0),
            )
        ]

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
                content_type=ContentType.ARTICLE,
                title="Deep Work",
                summary="Focused work strategies",
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
                content_type=ContentType.FRAMEWORK,
                title="Eisenhower Matrix",
                summary="Prioritization framework",
                full_content="Content",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime(2026, 1, 17, 10, 0, 0),
                highlights=[],
            ),
        ]

        prompt = generator._build_assignment_prompt(items, themes)

        # All items should be in the prompt
        assert "Atomic Habits" in prompt
        assert "Deep Work" in prompt
        assert "Eisenhower Matrix" in prompt

    def test_build_assignment_prompt_with_multiple_themes(self, mock_ollama_client, temp_dir):
        """_build_assignment_prompt() should include all available themes."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        themes = [
            Theme(
                id="personal-growth",
                name="Personal Growth",
                description="Self-improvement and learning",
                keywords=["growth", "learning"],
                item_count=0,
                created_at=datetime(2026, 1, 15, 10, 0, 0),
                updated_at=datetime(2026, 1, 15, 10, 0, 0),
            ),
            Theme(
                id="technology",
                name="Technology",
                description="Software and tech innovations",
                keywords=["software", "tech"],
                item_count=0,
                created_at=datetime(2026, 1, 15, 10, 0, 0),
                updated_at=datetime(2026, 1, 15, 10, 0, 0),
            ),
            Theme(
                id="business",
                name="Business",
                description="Entrepreneurship and strategy",
                keywords=["business", "strategy"],
                item_count=0,
                created_at=datetime(2026, 1, 15, 10, 0, 0),
                updated_at=datetime(2026, 1, 15, 10, 0, 0),
            ),
        ]

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

        prompt = generator._build_assignment_prompt(items, themes)

        # All themes should be listed
        assert "Personal Growth" in prompt
        assert "Technology" in prompt
        assert "Business" in prompt

    def test_build_assignment_prompt_with_empty_items_list(self, mock_ollama_client, temp_dir):
        """_build_assignment_prompt() should handle empty items list gracefully."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        themes = [
            Theme(
                id="personal-growth",
                name="Personal Growth",
                description="Self-improvement",
                keywords=["growth"],
                item_count=0,
                created_at=datetime(2026, 1, 15, 10, 0, 0),
                updated_at=datetime(2026, 1, 15, 10, 0, 0),
            )
        ]

        items = []

        prompt = generator._build_assignment_prompt(items, themes)

        # Should still return a valid string
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_build_assignment_prompt_with_empty_themes_list(self, mock_ollama_client, temp_dir):
        """_build_assignment_prompt() should handle empty themes list gracefully."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        themes = []

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

        prompt = generator._build_assignment_prompt(items, themes)

        # Should still return a valid string
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_build_assignment_prompt_allows_multiple_theme_assignments(self, mock_ollama_client, temp_dir):
        """_build_assignment_prompt() should indicate that items can be assigned to multiple themes."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        themes = [
            Theme(
                id="personal-growth",
                name="Personal Growth",
                description="Self-improvement",
                keywords=["growth"],
                item_count=0,
                created_at=datetime(2026, 1, 15, 10, 0, 0),
                updated_at=datetime(2026, 1, 15, 10, 0, 0),
            ),
            Theme(
                id="technology",
                name="Technology",
                description="Tech topics",
                keywords=["tech"],
                item_count=0,
                created_at=datetime(2026, 1, 15, 10, 0, 0),
                updated_at=datetime(2026, 1, 15, 10, 0, 0),
            ),
        ]

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

        prompt = generator._build_assignment_prompt(items, themes)

        # Should mention multiple themes or assignments
        assert "multiple" in prompt.lower() or "more than one" in prompt.lower() or "themes" in prompt.lower()

    def test_build_assignment_prompt_specifies_confidence_range(self, mock_ollama_client, temp_dir):
        """_build_assignment_prompt() should specify that confidence should be between 0.0 and 1.0."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        themes = [
            Theme(
                id="personal-growth",
                name="Personal Growth",
                description="Self-improvement",
                keywords=["growth"],
                item_count=0,
                created_at=datetime(2026, 1, 15, 10, 0, 0),
                updated_at=datetime(2026, 1, 15, 10, 0, 0),
            )
        ]

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

        prompt = generator._build_assignment_prompt(items, themes)

        # Should mention the confidence score range
        assert ("0" in prompt and "1" in prompt) or "0.0" in prompt or "1.0" in prompt


class TestParseAssignmentsFromResponse:
    """Tests for _parse_assignments_from_response() method that parses LLM output into ThemeAssignment objects."""

    def test_parse_assignments_from_response_returns_list_of_assignments(self, mock_ollama_client, temp_dir):
        """_parse_assignments_from_response() should return a list of ThemeAssignment objects."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        response = '''[
            {
                "item_id": "item-123",
                "theme_id": "personal-growth",
                "confidence": 0.85
            }
        ]'''

        assignments = generator._parse_assignments_from_response(response)

        assert isinstance(assignments, list)
        assert len(assignments) == 1
        assert isinstance(assignments[0], ThemeAssignment)

    def test_parse_assignments_from_response_extracts_item_id_correctly(self, mock_ollama_client, temp_dir):
        """_parse_assignments_from_response() should extract the item_id field correctly."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        response = '''[
            {
                "item_id": "book-atomic-habits",
                "theme_id": "personal-growth",
                "confidence": 0.9
            }
        ]'''

        assignments = generator._parse_assignments_from_response(response)

        assert assignments[0].item_id == "book-atomic-habits"

    def test_parse_assignments_from_response_extracts_theme_id_correctly(self, mock_ollama_client, temp_dir):
        """_parse_assignments_from_response() should extract the theme_id field correctly."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        response = '''[
            {
                "item_id": "item-1",
                "theme_id": "systems-thinking",
                "confidence": 0.75
            }
        ]'''

        assignments = generator._parse_assignments_from_response(response)

        assert assignments[0].theme_id == "systems-thinking"

    def test_parse_assignments_from_response_extracts_confidence_correctly(self, mock_ollama_client, temp_dir):
        """_parse_assignments_from_response() should extract the confidence field correctly."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        response = '''[
            {
                "item_id": "item-1",
                "theme_id": "personal-growth",
                "confidence": 0.92
            }
        ]'''

        assignments = generator._parse_assignments_from_response(response)

        assert assignments[0].confidence == 0.92

    def test_parse_assignments_from_response_sets_assigned_at_timestamp(self, mock_ollama_client, temp_dir):
        """_parse_assignments_from_response() should set assigned_at to current time."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        response = '''[
            {
                "item_id": "item-1",
                "theme_id": "personal-growth",
                "confidence": 0.85
            }
        ]'''

        before = datetime.now()
        assignments = generator._parse_assignments_from_response(response)
        after = datetime.now()

        assert before <= assignments[0].assigned_at <= after

    def test_parse_assignments_from_response_parses_multiple_assignments(self, mock_ollama_client, temp_dir):
        """_parse_assignments_from_response() should parse multiple assignments from the response."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        response = '''[
            {
                "item_id": "item-1",
                "theme_id": "personal-growth",
                "confidence": 0.85
            },
            {
                "item_id": "item-1",
                "theme_id": "technology",
                "confidence": 0.65
            },
            {
                "item_id": "item-2",
                "theme_id": "personal-growth",
                "confidence": 0.95
            }
        ]'''

        assignments = generator._parse_assignments_from_response(response)

        assert len(assignments) == 3
        assert assignments[0].item_id == "item-1"
        assert assignments[0].theme_id == "personal-growth"
        assert assignments[1].item_id == "item-1"
        assert assignments[1].theme_id == "technology"
        assert assignments[2].item_id == "item-2"

    def test_parse_assignments_from_response_handles_json_with_markdown_code_block(self, mock_ollama_client, temp_dir):
        """_parse_assignments_from_response() should handle JSON wrapped in markdown code blocks."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        response = '''```json
[
    {
        "item_id": "item-1",
        "theme_id": "personal-growth",
        "confidence": 0.85
    }
]
```'''

        assignments = generator._parse_assignments_from_response(response)

        assert len(assignments) == 1
        assert assignments[0].item_id == "item-1"

    def test_parse_assignments_from_response_handles_generic_markdown_code_block(self, mock_ollama_client, temp_dir):
        """_parse_assignments_from_response() should handle JSON wrapped in generic code blocks."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        response = '''```
[
    {
        "item_id": "item-2",
        "theme_id": "technology",
        "confidence": 0.7
    }
]
```'''

        assignments = generator._parse_assignments_from_response(response)

        assert len(assignments) == 1
        assert assignments[0].theme_id == "technology"

    def test_parse_assignments_from_response_returns_empty_list_for_invalid_json(self, mock_ollama_client, temp_dir):
        """_parse_assignments_from_response() should return empty list for invalid JSON."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        response = "This is not valid JSON at all."

        assignments = generator._parse_assignments_from_response(response)

        assert assignments == []

    def test_parse_assignments_from_response_returns_empty_list_for_empty_response(self, mock_ollama_client, temp_dir):
        """_parse_assignments_from_response() should return empty list for empty response."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        response = ""

        assignments = generator._parse_assignments_from_response(response)

        assert assignments == []

    def test_parse_assignments_from_response_returns_empty_list_for_empty_json_array(self, mock_ollama_client, temp_dir):
        """_parse_assignments_from_response() should return empty list for empty JSON array."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        response = "[]"

        assignments = generator._parse_assignments_from_response(response)

        assert assignments == []

    def test_parse_assignments_from_response_skips_assignments_missing_item_id(self, mock_ollama_client, temp_dir):
        """_parse_assignments_from_response() should skip assignments missing item_id."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        response = '''[
            {
                "item_id": "valid-item",
                "theme_id": "personal-growth",
                "confidence": 0.85
            },
            {
                "theme_id": "technology",
                "confidence": 0.7
            }
        ]'''

        assignments = generator._parse_assignments_from_response(response)

        assert len(assignments) == 1
        assert assignments[0].item_id == "valid-item"

    def test_parse_assignments_from_response_skips_assignments_missing_theme_id(self, mock_ollama_client, temp_dir):
        """_parse_assignments_from_response() should skip assignments missing theme_id."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        response = '''[
            {
                "item_id": "item-1",
                "theme_id": "personal-growth",
                "confidence": 0.85
            },
            {
                "item_id": "item-2",
                "confidence": 0.7
            }
        ]'''

        assignments = generator._parse_assignments_from_response(response)

        assert len(assignments) == 1
        assert assignments[0].theme_id == "personal-growth"

    def test_parse_assignments_from_response_skips_assignments_missing_confidence(self, mock_ollama_client, temp_dir):
        """_parse_assignments_from_response() should skip assignments missing confidence."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        response = '''[
            {
                "item_id": "item-1",
                "theme_id": "personal-growth",
                "confidence": 0.85
            },
            {
                "item_id": "item-2",
                "theme_id": "technology"
            }
        ]'''

        assignments = generator._parse_assignments_from_response(response)

        assert len(assignments) == 1
        assert assignments[0].item_id == "item-1"

    def test_parse_assignments_from_response_handles_extra_whitespace(self, mock_ollama_client, temp_dir):
        """_parse_assignments_from_response() should handle extra whitespace in response."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        response = '''

            [
                {
                    "item_id": "item-1",
                    "theme_id": "personal-growth",
                    "confidence": 0.85
                }
            ]

        '''

        assignments = generator._parse_assignments_from_response(response)

        assert len(assignments) == 1
        assert assignments[0].item_id == "item-1"

    def test_parse_assignments_from_response_handles_llm_preamble_text(self, mock_ollama_client, temp_dir):
        """_parse_assignments_from_response() should handle LLM response with preamble text before JSON."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        response = '''Based on the content provided, here are the theme assignments:

```json
[
    {
        "item_id": "item-1",
        "theme_id": "personal-growth",
        "confidence": 0.9
    }
]
```

These assignments reflect the relevance of each item to the available themes.'''

        assignments = generator._parse_assignments_from_response(response)

        assert len(assignments) == 1
        assert assignments[0].item_id == "item-1"
        assert assignments[0].confidence == 0.9

    def test_parse_assignments_from_response_handles_confidence_as_integer(self, mock_ollama_client, temp_dir):
        """_parse_assignments_from_response() should handle confidence as integer (e.g., 1 instead of 1.0)."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        response = '''[
            {
                "item_id": "item-1",
                "theme_id": "personal-growth",
                "confidence": 1
            }
        ]'''

        assignments = generator._parse_assignments_from_response(response)

        assert len(assignments) == 1
        assert assignments[0].confidence == 1.0

    def test_parse_assignments_from_response_handles_zero_confidence(self, mock_ollama_client, temp_dir):
        """_parse_assignments_from_response() should handle confidence of 0."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        response = '''[
            {
                "item_id": "item-1",
                "theme_id": "personal-growth",
                "confidence": 0
            }
        ]'''

        assignments = generator._parse_assignments_from_response(response)

        assert len(assignments) == 1
        assert assignments[0].confidence == 0.0

    def test_parse_assignments_from_response_skips_non_dict_entries(self, mock_ollama_client, temp_dir):
        """_parse_assignments_from_response() should skip non-dictionary entries in the array."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        response = '''[
            {
                "item_id": "item-1",
                "theme_id": "personal-growth",
                "confidence": 0.85
            },
            "not a dict",
            123,
            null
        ]'''

        assignments = generator._parse_assignments_from_response(response)

        assert len(assignments) == 1
        assert assignments[0].item_id == "item-1"

    def test_parse_assignments_from_response_returns_empty_list_for_non_array_json(self, mock_ollama_client, temp_dir):
        """_parse_assignments_from_response() should return empty list if JSON is not an array."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        response = '''{
            "item_id": "item-1",
            "theme_id": "personal-growth",
            "confidence": 0.85
        }'''

        assignments = generator._parse_assignments_from_response(response)

        assert assignments == []

    def test_parse_assignments_from_response_handles_whitespace_only_response(self, mock_ollama_client, temp_dir):
        """_parse_assignments_from_response() should return empty list for whitespace-only response."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        response = "   \n\t\n   "

        assignments = generator._parse_assignments_from_response(response)

        assert assignments == []

    def test_parse_assignments_from_response_preserves_order(self, mock_ollama_client, temp_dir):
        """_parse_assignments_from_response() should preserve the order of assignments."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        response = '''[
            {
                "item_id": "first-item",
                "theme_id": "theme-a",
                "confidence": 0.9
            },
            {
                "item_id": "second-item",
                "theme_id": "theme-b",
                "confidence": 0.8
            },
            {
                "item_id": "third-item",
                "theme_id": "theme-c",
                "confidence": 0.7
            }
        ]'''

        assignments = generator._parse_assignments_from_response(response)

        assert len(assignments) == 3
        assert assignments[0].item_id == "first-item"
        assert assignments[1].item_id == "second-item"
        assert assignments[2].item_id == "third-item"

    def test_parse_assignments_from_response_all_assignments_have_same_timestamp(self, mock_ollama_client, temp_dir):
        """_parse_assignments_from_response() should set the same assigned_at for all assignments in one call."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        response = '''[
            {
                "item_id": "item-1",
                "theme_id": "personal-growth",
                "confidence": 0.85
            },
            {
                "item_id": "item-2",
                "theme_id": "technology",
                "confidence": 0.75
            }
        ]'''

        assignments = generator._parse_assignments_from_response(response)

        assert len(assignments) == 2
        assert assignments[0].assigned_at == assignments[1].assigned_at


class TestAssignItemsToThemes:
    """Tests for assign_items_to_themes() method that assigns items to themes using AI."""

    def test_assign_items_to_themes_returns_list_of_assignments(self, mock_ollama_client, temp_dir):
        """assign_items_to_themes() should return a list of ThemeAssignment objects."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        # Set up themes
        themes = [
            Theme(
                id="personal-growth",
                name="Personal Growth",
                description="Self-improvement and learning",
                keywords=["growth", "learning"],
                item_count=0,
                created_at=datetime(2026, 1, 15, 10, 0, 0),
                updated_at=datetime(2026, 1, 15, 10, 0, 0),
            ),
            Theme(
                id="technology",
                name="Technology",
                description="Software and tech innovations",
                keywords=["software", "tech"],
                item_count=0,
                created_at=datetime(2026, 1, 15, 10, 0, 0),
                updated_at=datetime(2026, 1, 15, 10, 0, 0),
            ),
        ]
        generator.themes = themes

        # Set up items
        items = [
            LibraryItem(
                id="item-1",
                content_type=ContentType.BOOK,
                title="Learning to Code",
                summary="A book about learning programming",
                full_content="Full content about coding",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime(2026, 1, 15, 10, 0, 0),
                highlights=[],
            )
        ]

        # Mock the LLM response
        mock_ollama_client.generate.return_value = '''[
            {
                "item_id": "item-1",
                "theme_id": "personal-growth",
                "confidence": 0.85
            },
            {
                "item_id": "item-1",
                "theme_id": "technology",
                "confidence": 0.75
            }
        ]'''

        result = generator.assign_items_to_themes(items)

        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(a, ThemeAssignment) for a in result)

    def test_assign_items_to_themes_calls_ollama_client(self, mock_ollama_client, temp_dir):
        """assign_items_to_themes() should call OllamaClient.generate() with correct parameters."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        # Set up themes
        themes = [
            Theme(
                id="personal-growth",
                name="Personal Growth",
                description="Self-improvement",
                keywords=["growth"],
                item_count=0,
                created_at=datetime(2026, 1, 15, 10, 0, 0),
                updated_at=datetime(2026, 1, 15, 10, 0, 0),
            )
        ]
        generator.themes = themes

        # Set up items
        items = [
            LibraryItem(
                id="item-1",
                content_type=ContentType.VALUE,
                title="Test Value",
                summary="A test value",
                full_content="Full content",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime(2026, 1, 15, 10, 0, 0),
                highlights=[],
            )
        ]

        # Mock the LLM response
        mock_ollama_client.generate.return_value = '''[
            {
                "item_id": "item-1",
                "theme_id": "personal-growth",
                "confidence": 0.9
            }
        ]'''

        generator.assign_items_to_themes(items)

        # Verify generate was called
        mock_ollama_client.generate.assert_called_once()
        call_kwargs = mock_ollama_client.generate.call_args[1]
        assert "prompt" in call_kwargs
        assert "system_prompt" in call_kwargs

    def test_assign_items_to_themes_updates_self_assignments(self, mock_ollama_client, temp_dir):
        """assign_items_to_themes() should update self.assignments with new assignments."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        # Set up themes
        themes = [
            Theme(
                id="technology",
                name="Technology",
                description="Tech topics",
                keywords=["tech"],
                item_count=0,
                created_at=datetime(2026, 1, 15, 10, 0, 0),
                updated_at=datetime(2026, 1, 15, 10, 0, 0),
            )
        ]
        generator.themes = themes

        # Set up items
        items = [
            LibraryItem(
                id="item-1",
                content_type=ContentType.ARTICLE,
                title="Tech Article",
                summary="An article about technology",
                full_content="Full content",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime(2026, 1, 15, 10, 0, 0),
                highlights=[],
            )
        ]

        # Mock the LLM response
        mock_ollama_client.generate.return_value = '''[
            {
                "item_id": "item-1",
                "theme_id": "technology",
                "confidence": 0.95
            }
        ]'''

        generator.assign_items_to_themes(items)

        assert len(generator.assignments) == 1
        assert generator.assignments[0].item_id == "item-1"
        assert generator.assignments[0].theme_id == "technology"
        assert generator.assignments[0].confidence == 0.95

    def test_assign_items_to_themes_saves_assignments_to_disk(self, mock_ollama_client, temp_dir):
        """assign_items_to_themes() should save assignments to assignments.json."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        # Set up themes
        themes = [
            Theme(
                id="business",
                name="Business",
                description="Business topics",
                keywords=["business"],
                item_count=0,
                created_at=datetime(2026, 1, 15, 10, 0, 0),
                updated_at=datetime(2026, 1, 15, 10, 0, 0),
            )
        ]
        generator.themes = themes

        # Set up items
        items = [
            LibraryItem(
                id="item-1",
                content_type=ContentType.FRAMEWORK,
                title="Business Framework",
                summary="A business framework",
                full_content="Full content",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime(2026, 1, 15, 10, 0, 0),
                highlights=[],
            )
        ]

        # Mock the LLM response
        mock_ollama_client.generate.return_value = '''[
            {
                "item_id": "item-1",
                "theme_id": "business",
                "confidence": 0.88
            }
        ]'''

        generator.assign_items_to_themes(items)

        # Check that the file was created
        assert (data_dir / "assignments.json").exists()

        # Check the file contents
        with open(data_dir / "assignments.json") as f:
            data = json.load(f)

        assert len(data) == 1
        assert data[0]["item_id"] == "item-1"
        assert data[0]["theme_id"] == "business"
        assert data[0]["confidence"] == 0.88

    def test_assign_items_to_themes_assigns_item_to_multiple_themes(self, mock_ollama_client, temp_dir):
        """assign_items_to_themes() should allow an item to be assigned to multiple themes."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        # Set up themes
        themes = [
            Theme(
                id="personal-growth",
                name="Personal Growth",
                description="Self-improvement",
                keywords=["growth"],
                item_count=0,
                created_at=datetime(2026, 1, 15, 10, 0, 0),
                updated_at=datetime(2026, 1, 15, 10, 0, 0),
            ),
            Theme(
                id="technology",
                name="Technology",
                description="Tech topics",
                keywords=["tech"],
                item_count=0,
                created_at=datetime(2026, 1, 15, 10, 0, 0),
                updated_at=datetime(2026, 1, 15, 10, 0, 0),
            ),
            Theme(
                id="business",
                name="Business",
                description="Business topics",
                keywords=["business"],
                item_count=0,
                created_at=datetime(2026, 1, 15, 10, 0, 0),
                updated_at=datetime(2026, 1, 15, 10, 0, 0),
            ),
        ]
        generator.themes = themes

        # Set up items
        items = [
            LibraryItem(
                id="cross-domain-item",
                content_type=ContentType.BOOK,
                title="Tech Entrepreneurship for Personal Growth",
                summary="A book about tech startups and personal development",
                full_content="Full content",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime(2026, 1, 15, 10, 0, 0),
                highlights=[],
            )
        ]

        # Mock the LLM response - item assigned to all three themes
        mock_ollama_client.generate.return_value = '''[
            {
                "item_id": "cross-domain-item",
                "theme_id": "personal-growth",
                "confidence": 0.85
            },
            {
                "item_id": "cross-domain-item",
                "theme_id": "technology",
                "confidence": 0.90
            },
            {
                "item_id": "cross-domain-item",
                "theme_id": "business",
                "confidence": 0.75
            }
        ]'''

        result = generator.assign_items_to_themes(items)

        # Should have 3 assignments for the one item
        assert len(result) == 3

        # All assignments should be for the same item
        item_ids = [a.item_id for a in result]
        assert all(id == "cross-domain-item" for id in item_ids)

        # Should have assignments to all three themes
        theme_ids = [a.theme_id for a in result]
        assert "personal-growth" in theme_ids
        assert "technology" in theme_ids
        assert "business" in theme_ids

    def test_assign_items_to_themes_handles_multiple_items(self, mock_ollama_client, temp_dir):
        """assign_items_to_themes() should correctly assign multiple items to themes."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        # Set up themes
        themes = [
            Theme(
                id="personal-growth",
                name="Personal Growth",
                description="Self-improvement",
                keywords=["growth"],
                item_count=0,
                created_at=datetime(2026, 1, 15, 10, 0, 0),
                updated_at=datetime(2026, 1, 15, 10, 0, 0),
            ),
            Theme(
                id="technology",
                name="Technology",
                description="Tech topics",
                keywords=["tech"],
                item_count=0,
                created_at=datetime(2026, 1, 15, 10, 0, 0),
                updated_at=datetime(2026, 1, 15, 10, 0, 0),
            ),
        ]
        generator.themes = themes

        # Set up multiple items
        items = [
            LibraryItem(
                id="item-1",
                content_type=ContentType.VALUE,
                title="Growth Mindset",
                summary="A value about growth mindset",
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
                content_type=ContentType.ARTICLE,
                title="Python Best Practices",
                summary="An article about Python",
                full_content="Full content",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime(2026, 1, 15, 10, 0, 0),
                highlights=[],
            ),
            LibraryItem(
                id="item-3",
                content_type=ContentType.BOOK,
                title="Learning Python",
                summary="A book about learning Python programming",
                full_content="Full content",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime(2026, 1, 15, 10, 0, 0),
                highlights=[],
            ),
        ]

        # Mock the LLM response
        mock_ollama_client.generate.return_value = '''[
            {
                "item_id": "item-1",
                "theme_id": "personal-growth",
                "confidence": 0.95
            },
            {
                "item_id": "item-2",
                "theme_id": "technology",
                "confidence": 0.90
            },
            {
                "item_id": "item-3",
                "theme_id": "technology",
                "confidence": 0.85
            },
            {
                "item_id": "item-3",
                "theme_id": "personal-growth",
                "confidence": 0.65
            }
        ]'''

        result = generator.assign_items_to_themes(items)

        # Should have 4 total assignments
        assert len(result) == 4

        # Check item-1 assignments
        item1_assignments = [a for a in result if a.item_id == "item-1"]
        assert len(item1_assignments) == 1
        assert item1_assignments[0].theme_id == "personal-growth"

        # Check item-2 assignments
        item2_assignments = [a for a in result if a.item_id == "item-2"]
        assert len(item2_assignments) == 1
        assert item2_assignments[0].theme_id == "technology"

        # Check item-3 assignments (should have 2)
        item3_assignments = [a for a in result if a.item_id == "item-3"]
        assert len(item3_assignments) == 2
        theme_ids = [a.theme_id for a in item3_assignments]
        assert "technology" in theme_ids
        assert "personal-growth" in theme_ids

    def test_assign_items_to_themes_captures_confidence_scores(self, mock_ollama_client, temp_dir):
        """assign_items_to_themes() should correctly capture confidence scores."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        # Set up themes
        themes = [
            Theme(
                id="theme-1",
                name="Theme 1",
                description="First theme",
                keywords=["theme1"],
                item_count=0,
                created_at=datetime(2026, 1, 15, 10, 0, 0),
                updated_at=datetime(2026, 1, 15, 10, 0, 0),
            ),
            Theme(
                id="theme-2",
                name="Theme 2",
                description="Second theme",
                keywords=["theme2"],
                item_count=0,
                created_at=datetime(2026, 1, 15, 10, 0, 0),
                updated_at=datetime(2026, 1, 15, 10, 0, 0),
            ),
        ]
        generator.themes = themes

        # Set up item
        items = [
            LibraryItem(
                id="item-1",
                content_type=ContentType.INSIGHT,
                title="Test Insight",
                summary="A test insight",
                full_content="Full content",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime(2026, 1, 15, 10, 0, 0),
                highlights=[],
            )
        ]

        # Mock the LLM response with different confidence scores
        mock_ollama_client.generate.return_value = '''[
            {
                "item_id": "item-1",
                "theme_id": "theme-1",
                "confidence": 0.92
            },
            {
                "item_id": "item-1",
                "theme_id": "theme-2",
                "confidence": 0.45
            }
        ]'''

        result = generator.assign_items_to_themes(items)

        # Check confidence scores
        theme1_assignment = next(a for a in result if a.theme_id == "theme-1")
        theme2_assignment = next(a for a in result if a.theme_id == "theme-2")

        assert theme1_assignment.confidence == 0.92
        assert theme2_assignment.confidence == 0.45

    def test_assign_items_to_themes_returns_empty_list_for_empty_items(self, mock_ollama_client, temp_dir):
        """assign_items_to_themes() should return empty list when given empty items list."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        # Set up themes
        themes = [
            Theme(
                id="personal-growth",
                name="Personal Growth",
                description="Self-improvement",
                keywords=["growth"],
                item_count=0,
                created_at=datetime(2026, 1, 15, 10, 0, 0),
                updated_at=datetime(2026, 1, 15, 10, 0, 0),
            )
        ]
        generator.themes = themes

        result = generator.assign_items_to_themes([])

        assert result == []
        # Should not have called the LLM for empty items
        mock_ollama_client.generate.assert_not_called()

    def test_assign_items_to_themes_returns_empty_list_for_no_themes(self, mock_ollama_client, temp_dir):
        """assign_items_to_themes() should return empty list when no themes exist."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        # No themes set
        generator.themes = []

        # Set up items
        items = [
            LibraryItem(
                id="item-1",
                content_type=ContentType.MEMORY,
                title="Test Memory",
                summary="A test memory",
                full_content="Full content",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime(2026, 1, 15, 10, 0, 0),
                highlights=[],
            )
        ]

        result = generator.assign_items_to_themes(items)

        assert result == []
        # Should not have called the LLM when no themes exist
        mock_ollama_client.generate.assert_not_called()

    def test_assign_items_to_themes_uses_build_assignment_prompt(self, mock_ollama_client, temp_dir):
        """assign_items_to_themes() should use _build_assignment_prompt() to construct the prompt."""
        from src.library.theme_generator import ThemeGenerator
        from unittest.mock import patch

        data_dir = temp_dir / "library"
        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        # Set up themes
        themes = [
            Theme(
                id="test-theme",
                name="Test Theme",
                description="A test theme",
                keywords=["test"],
                item_count=0,
                created_at=datetime(2026, 1, 15, 10, 0, 0),
                updated_at=datetime(2026, 1, 15, 10, 0, 0),
            )
        ]
        generator.themes = themes

        # Set up items
        items = [
            LibraryItem(
                id="item-1",
                content_type=ContentType.SKILL,
                title="Test Skill",
                summary="A test skill",
                full_content="Full content",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime(2026, 1, 15, 10, 0, 0),
                highlights=[],
            )
        ]

        # Mock the LLM response
        mock_ollama_client.generate.return_value = '''[
            {
                "item_id": "item-1",
                "theme_id": "test-theme",
                "confidence": 0.8
            }
        ]'''

        with patch.object(generator, '_build_assignment_prompt', wraps=generator._build_assignment_prompt) as mock_build:
            generator.assign_items_to_themes(items)

            # Verify _build_assignment_prompt was called with items and themes
            mock_build.assert_called_once_with(items, themes)

    def test_assign_items_to_themes_uses_parse_assignments_from_response(self, mock_ollama_client, temp_dir):
        """assign_items_to_themes() should use _parse_assignments_from_response() to parse the response."""
        from src.library.theme_generator import ThemeGenerator
        from unittest.mock import patch

        data_dir = temp_dir / "library"
        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        # Set up themes
        themes = [
            Theme(
                id="test-theme",
                name="Test Theme",
                description="A test theme",
                keywords=["test"],
                item_count=0,
                created_at=datetime(2026, 1, 15, 10, 0, 0),
                updated_at=datetime(2026, 1, 15, 10, 0, 0),
            )
        ]
        generator.themes = themes

        # Set up items
        items = [
            LibraryItem(
                id="item-1",
                content_type=ContentType.GOAL,
                title="Test Goal",
                summary="A test goal",
                full_content="Full content",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime(2026, 1, 15, 10, 0, 0),
                highlights=[],
            )
        ]

        # Mock the LLM response
        llm_response = '''[
            {
                "item_id": "item-1",
                "theme_id": "test-theme",
                "confidence": 0.8
            }
        ]'''
        mock_ollama_client.generate.return_value = llm_response

        with patch.object(generator, '_parse_assignments_from_response', wraps=generator._parse_assignments_from_response) as mock_parse:
            generator.assign_items_to_themes(items)

            # Verify _parse_assignments_from_response was called with the LLM response
            mock_parse.assert_called_once_with(llm_response)

    def test_assign_items_to_themes_sets_assigned_at_timestamp(self, mock_ollama_client, temp_dir):
        """assign_items_to_themes() should set assigned_at timestamp on all assignments."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        # Set up themes
        themes = [
            Theme(
                id="test-theme",
                name="Test Theme",
                description="A test theme",
                keywords=["test"],
                item_count=0,
                created_at=datetime(2026, 1, 15, 10, 0, 0),
                updated_at=datetime(2026, 1, 15, 10, 0, 0),
            )
        ]
        generator.themes = themes

        # Set up items
        items = [
            LibraryItem(
                id="item-1",
                content_type=ContentType.PREFERENCE,
                title="Test Preference",
                summary="A test preference",
                full_content="Full content",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime(2026, 1, 15, 10, 0, 0),
                highlights=[],
            )
        ]

        # Mock the LLM response
        mock_ollama_client.generate.return_value = '''[
            {
                "item_id": "item-1",
                "theme_id": "test-theme",
                "confidence": 0.8
            }
        ]'''

        before = datetime.now()
        result = generator.assign_items_to_themes(items)
        after = datetime.now()

        assert len(result) == 1
        assert before <= result[0].assigned_at <= after

    def test_assign_items_to_themes_handles_llm_failure_gracefully(self, mock_ollama_client, temp_dir):
        """assign_items_to_themes() should return empty list when LLM returns invalid response."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        # Set up themes
        themes = [
            Theme(
                id="test-theme",
                name="Test Theme",
                description="A test theme",
                keywords=["test"],
                item_count=0,
                created_at=datetime(2026, 1, 15, 10, 0, 0),
                updated_at=datetime(2026, 1, 15, 10, 0, 0),
            )
        ]
        generator.themes = themes

        # Set up items
        items = [
            LibraryItem(
                id="item-1",
                content_type=ContentType.WEB_CONTENT,
                title="Test Web Content",
                summary="A test web content",
                full_content="Full content",
                source_url="https://example.com",
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime(2026, 1, 15, 10, 0, 0),
                highlights=[],
            )
        ]

        # Mock the LLM response with invalid JSON
        mock_ollama_client.generate.return_value = "This is not valid JSON at all."

        result = generator.assign_items_to_themes(items)

        # Should return empty list for invalid response
        assert result == []
        assert generator.assignments == []


class TestUpdateAssignments:
    """Tests for update_assignments() method that incrementally updates theme assignments for new items."""

    def test_update_assignments_only_assigns_new_items(self, mock_ollama_client, temp_dir):
        """update_assignments() should only assign items that don't have existing assignments."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        # Set up themes
        themes = [
            Theme(
                id="personal-growth",
                name="Personal Growth",
                description="Self-improvement and learning",
                keywords=["growth", "learning"],
                item_count=1,
                created_at=datetime(2026, 1, 15, 10, 0, 0),
                updated_at=datetime(2026, 1, 15, 10, 0, 0),
            ),
            Theme(
                id="technology",
                name="Technology",
                description="Software and tech innovations",
                keywords=["software", "tech"],
                item_count=0,
                created_at=datetime(2026, 1, 15, 10, 0, 0),
                updated_at=datetime(2026, 1, 15, 10, 0, 0),
            ),
        ]
        generator.themes = themes

        # Set up existing assignments for item-1
        existing_assignment = ThemeAssignment(
            item_id="item-1",
            theme_id="personal-growth",
            confidence=0.85,
            assigned_at=datetime(2026, 1, 20, 10, 0, 0),
        )
        generator.assignments = [existing_assignment]

        # Set up items - item-1 already has assignment, item-2 is new
        items = [
            LibraryItem(
                id="item-1",
                content_type=ContentType.BOOK,
                title="Existing Book",
                summary="A book that already has assignments",
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
                content_type=ContentType.ARTICLE,
                title="New Article",
                summary="A new article without assignments",
                full_content="Full content",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime(2026, 1, 25, 10, 0, 0),
                highlights=[],
            ),
        ]

        # Mock the LLM response - should only be for item-2
        mock_ollama_client.generate.return_value = '''[
            {
                "item_id": "item-2",
                "theme_id": "technology",
                "confidence": 0.90
            }
        ]'''

        result = generator.update_assignments(items)

        # Should have the original assignment plus the new one
        assert len(generator.assignments) == 2

        # Original assignment should be preserved
        item1_assignments = [a for a in generator.assignments if a.item_id == "item-1"]
        assert len(item1_assignments) == 1
        assert item1_assignments[0].confidence == 0.85

        # New assignment should be added
        item2_assignments = [a for a in generator.assignments if a.item_id == "item-2"]
        assert len(item2_assignments) == 1
        assert item2_assignments[0].theme_id == "technology"

    def test_update_assignments_returns_only_new_assignments(self, mock_ollama_client, temp_dir):
        """update_assignments() should return only the newly created assignments."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        # Set up themes
        themes = [
            Theme(
                id="technology",
                name="Technology",
                description="Tech topics",
                keywords=["tech"],
                item_count=0,
                created_at=datetime(2026, 1, 15, 10, 0, 0),
                updated_at=datetime(2026, 1, 15, 10, 0, 0),
            ),
        ]
        generator.themes = themes

        # Set up existing assignment
        existing_assignment = ThemeAssignment(
            item_id="item-1",
            theme_id="technology",
            confidence=0.80,
            assigned_at=datetime(2026, 1, 20, 10, 0, 0),
        )
        generator.assignments = [existing_assignment]

        # Set up items - only item-2 is new
        items = [
            LibraryItem(
                id="item-1",
                content_type=ContentType.BOOK,
                title="Existing Book",
                summary="Already assigned",
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
                content_type=ContentType.ARTICLE,
                title="New Article",
                summary="Not yet assigned",
                full_content="Full content",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime(2026, 1, 25, 10, 0, 0),
                highlights=[],
            ),
        ]

        # Mock LLM response
        mock_ollama_client.generate.return_value = '''[
            {
                "item_id": "item-2",
                "theme_id": "technology",
                "confidence": 0.75
            }
        ]'''

        result = generator.update_assignments(items)

        # Return value should only contain new assignments
        assert len(result) == 1
        assert result[0].item_id == "item-2"

    def test_update_assignments_calls_ollama_only_for_new_items(self, mock_ollama_client, temp_dir):
        """update_assignments() should only call OllamaClient with new items."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        # Set up themes
        themes = [
            Theme(
                id="business",
                name="Business",
                description="Business topics",
                keywords=["business"],
                item_count=0,
                created_at=datetime(2026, 1, 15, 10, 0, 0),
                updated_at=datetime(2026, 1, 15, 10, 0, 0),
            ),
        ]
        generator.themes = themes

        # Set up existing assignment for item-1
        existing_assignment = ThemeAssignment(
            item_id="item-1",
            theme_id="business",
            confidence=0.90,
            assigned_at=datetime(2026, 1, 20, 10, 0, 0),
        )
        generator.assignments = [existing_assignment]

        # Set up items
        items = [
            LibraryItem(
                id="item-1",
                content_type=ContentType.VALUE,
                title="Existing Value",
                summary="Already assigned",
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
                content_type=ContentType.FRAMEWORK,
                title="New Framework",
                summary="A new framework",
                full_content="Full content",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime(2026, 1, 25, 10, 0, 0),
                highlights=[],
            ),
        ]

        # Mock LLM response
        mock_ollama_client.generate.return_value = '''[
            {
                "item_id": "item-2",
                "theme_id": "business",
                "confidence": 0.70
            }
        ]'''

        generator.update_assignments(items)

        # Verify generate was called
        mock_ollama_client.generate.assert_called_once()

        # Check that the prompt only includes item-2 (not item-1)
        call_kwargs = mock_ollama_client.generate.call_args[1]
        prompt = call_kwargs["prompt"]
        assert "item-2" in prompt
        assert "New Framework" in prompt
        # item-1 should not be in the prompt since it already has assignments
        assert "item-1" not in prompt
        assert "Existing Value" not in prompt

    def test_update_assignments_does_not_call_ollama_when_no_new_items(self, mock_ollama_client, temp_dir):
        """update_assignments() should not call OllamaClient when all items already have assignments."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        # Set up themes
        themes = [
            Theme(
                id="technology",
                name="Technology",
                description="Tech topics",
                keywords=["tech"],
                item_count=0,
                created_at=datetime(2026, 1, 15, 10, 0, 0),
                updated_at=datetime(2026, 1, 15, 10, 0, 0),
            ),
        ]
        generator.themes = themes

        # Set up existing assignment
        existing_assignment = ThemeAssignment(
            item_id="item-1",
            theme_id="technology",
            confidence=0.85,
            assigned_at=datetime(2026, 1, 20, 10, 0, 0),
        )
        generator.assignments = [existing_assignment]

        # Only item-1 which already has assignment
        items = [
            LibraryItem(
                id="item-1",
                content_type=ContentType.ARTICLE,
                title="Existing Article",
                summary="Already assigned",
                full_content="Full content",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime(2026, 1, 15, 10, 0, 0),
                highlights=[],
            ),
        ]

        result = generator.update_assignments(items)

        # Should not call LLM
        mock_ollama_client.generate.assert_not_called()

        # Should return empty list (no new assignments)
        assert result == []

        # Original assignment should still be there
        assert len(generator.assignments) == 1

    def test_update_assignments_saves_to_disk(self, mock_ollama_client, temp_dir):
        """update_assignments() should save all assignments (old and new) to disk."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        # Set up themes
        themes = [
            Theme(
                id="personal-growth",
                name="Personal Growth",
                description="Self-improvement",
                keywords=["growth"],
                item_count=0,
                created_at=datetime(2026, 1, 15, 10, 0, 0),
                updated_at=datetime(2026, 1, 15, 10, 0, 0),
            ),
        ]
        generator.themes = themes

        # Set up existing assignment
        existing_assignment = ThemeAssignment(
            item_id="item-1",
            theme_id="personal-growth",
            confidence=0.80,
            assigned_at=datetime(2026, 1, 20, 10, 0, 0),
        )
        generator.assignments = [existing_assignment]

        # New item
        items = [
            LibraryItem(
                id="item-1",
                content_type=ContentType.BOOK,
                title="Existing Book",
                summary="Already assigned",
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
                content_type=ContentType.INSIGHT,
                title="New Insight",
                summary="A new insight",
                full_content="Full content",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime(2026, 1, 25, 10, 0, 0),
                highlights=[],
            ),
        ]

        # Mock LLM response
        mock_ollama_client.generate.return_value = '''[
            {
                "item_id": "item-2",
                "theme_id": "personal-growth",
                "confidence": 0.75
            }
        ]'''

        generator.update_assignments(items)

        # Check file was created/updated
        assert (data_dir / "assignments.json").exists()

        # Check file contains both old and new assignments
        with open(data_dir / "assignments.json") as f:
            data = json.load(f)

        assert len(data) == 2
        item_ids = [a["item_id"] for a in data]
        assert "item-1" in item_ids
        assert "item-2" in item_ids

    def test_update_assignments_preserves_existing_assignment_details(self, mock_ollama_client, temp_dir):
        """update_assignments() should preserve all details of existing assignments."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        # Set up themes
        themes = [
            Theme(
                id="technology",
                name="Technology",
                description="Tech topics",
                keywords=["tech"],
                item_count=0,
                created_at=datetime(2026, 1, 15, 10, 0, 0),
                updated_at=datetime(2026, 1, 15, 10, 0, 0),
            ),
        ]
        generator.themes = themes

        # Set up existing assignment with specific timestamp and confidence
        original_timestamp = datetime(2026, 1, 10, 8, 30, 45)
        existing_assignment = ThemeAssignment(
            item_id="item-1",
            theme_id="technology",
            confidence=0.92,
            assigned_at=original_timestamp,
        )
        generator.assignments = [existing_assignment]

        # New item
        items = [
            LibraryItem(
                id="item-1",
                content_type=ContentType.ARTICLE,
                title="Existing Article",
                summary="Already assigned",
                full_content="Full content",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime(2026, 1, 5, 10, 0, 0),
                highlights=[],
            ),
            LibraryItem(
                id="item-2",
                content_type=ContentType.BOOK,
                title="New Book",
                summary="A new book",
                full_content="Full content",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime(2026, 1, 25, 10, 0, 0),
                highlights=[],
            ),
        ]

        # Mock LLM response
        mock_ollama_client.generate.return_value = '''[
            {
                "item_id": "item-2",
                "theme_id": "technology",
                "confidence": 0.65
            }
        ]'''

        generator.update_assignments(items)

        # Find the original assignment
        item1_assignments = [a for a in generator.assignments if a.item_id == "item-1"]
        assert len(item1_assignments) == 1

        # All original details should be preserved
        assert item1_assignments[0].theme_id == "technology"
        assert item1_assignments[0].confidence == 0.92
        assert item1_assignments[0].assigned_at == original_timestamp

    def test_update_assignments_handles_multiple_new_items(self, mock_ollama_client, temp_dir):
        """update_assignments() should correctly handle multiple new items at once."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        # Set up themes
        themes = [
            Theme(
                id="personal-growth",
                name="Personal Growth",
                description="Self-improvement",
                keywords=["growth"],
                item_count=0,
                created_at=datetime(2026, 1, 15, 10, 0, 0),
                updated_at=datetime(2026, 1, 15, 10, 0, 0),
            ),
            Theme(
                id="technology",
                name="Technology",
                description="Tech topics",
                keywords=["tech"],
                item_count=0,
                created_at=datetime(2026, 1, 15, 10, 0, 0),
                updated_at=datetime(2026, 1, 15, 10, 0, 0),
            ),
        ]
        generator.themes = themes

        # Start with no existing assignments
        generator.assignments = []

        # Multiple new items
        items = [
            LibraryItem(
                id="item-1",
                content_type=ContentType.BOOK,
                title="Python Handbook",
                summary="Learn Python programming",
                full_content="Full content",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime(2026, 1, 20, 10, 0, 0),
                highlights=[],
            ),
            LibraryItem(
                id="item-2",
                content_type=ContentType.VALUE,
                title="Growth Mindset",
                summary="Embrace continuous learning",
                full_content="Full content",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime(2026, 1, 21, 10, 0, 0),
                highlights=[],
            ),
            LibraryItem(
                id="item-3",
                content_type=ContentType.ARTICLE,
                title="AI Trends 2026",
                summary="Latest AI developments",
                full_content="Full content",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime(2026, 1, 22, 10, 0, 0),
                highlights=[],
            ),
        ]

        # Mock LLM response
        mock_ollama_client.generate.return_value = '''[
            {
                "item_id": "item-1",
                "theme_id": "technology",
                "confidence": 0.88
            },
            {
                "item_id": "item-1",
                "theme_id": "personal-growth",
                "confidence": 0.55
            },
            {
                "item_id": "item-2",
                "theme_id": "personal-growth",
                "confidence": 0.95
            },
            {
                "item_id": "item-3",
                "theme_id": "technology",
                "confidence": 0.92
            }
        ]'''

        result = generator.update_assignments(items)

        # Should have 4 assignments total
        assert len(result) == 4
        assert len(generator.assignments) == 4

        # Check specific assignments
        item1_assignments = [a for a in result if a.item_id == "item-1"]
        assert len(item1_assignments) == 2

        item2_assignments = [a for a in result if a.item_id == "item-2"]
        assert len(item2_assignments) == 1
        assert item2_assignments[0].theme_id == "personal-growth"

    def test_update_assignments_returns_empty_list_when_no_themes(self, mock_ollama_client, temp_dir):
        """update_assignments() should return empty list when no themes exist."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        # No themes
        generator.themes = []
        generator.assignments = []

        items = [
            LibraryItem(
                id="item-1",
                content_type=ContentType.MEMORY,
                title="Test Memory",
                summary="A test memory",
                full_content="Full content",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime(2026, 1, 25, 10, 0, 0),
                highlights=[],
            ),
        ]

        result = generator.update_assignments(items)

        assert result == []
        mock_ollama_client.generate.assert_not_called()

    def test_update_assignments_returns_empty_list_when_empty_items(self, mock_ollama_client, temp_dir):
        """update_assignments() should return empty list when given empty items list."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        # Set up themes
        themes = [
            Theme(
                id="technology",
                name="Technology",
                description="Tech topics",
                keywords=["tech"],
                item_count=0,
                created_at=datetime(2026, 1, 15, 10, 0, 0),
                updated_at=datetime(2026, 1, 15, 10, 0, 0),
            ),
        ]
        generator.themes = themes
        generator.assignments = []

        result = generator.update_assignments([])

        assert result == []
        mock_ollama_client.generate.assert_not_called()

    def test_update_assignments_identifies_new_items_by_assignment_not_presence(self, mock_ollama_client, temp_dir):
        """update_assignments() should consider an item 'new' if it has no assignments, not based on created_at."""
        from src.library.theme_generator import ThemeGenerator

        data_dir = temp_dir / "library"
        generator = ThemeGenerator(ollama_client=mock_ollama_client, data_dir=data_dir)

        # Set up themes
        themes = [
            Theme(
                id="technology",
                name="Technology",
                description="Tech topics",
                keywords=["tech"],
                item_count=0,
                created_at=datetime(2026, 1, 15, 10, 0, 0),
                updated_at=datetime(2026, 1, 15, 10, 0, 0),
            ),
        ]
        generator.themes = themes

        # Existing assignment only for item-2 (even though item-1 was created first)
        existing_assignment = ThemeAssignment(
            item_id="item-2",
            theme_id="technology",
            confidence=0.85,
            assigned_at=datetime(2026, 1, 20, 10, 0, 0),
        )
        generator.assignments = [existing_assignment]

        # item-1 was created first but has no assignments
        # item-2 was created later but already has an assignment
        items = [
            LibraryItem(
                id="item-1",
                content_type=ContentType.BOOK,
                title="Old Unassigned Book",
                summary="Created first but never assigned",
                full_content="Full content",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime(2026, 1, 1, 10, 0, 0),  # Created earlier
                highlights=[],
            ),
            LibraryItem(
                id="item-2",
                content_type=ContentType.ARTICLE,
                title="Newer Assigned Article",
                summary="Created later but already assigned",
                full_content="Full content",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime(2026, 1, 15, 10, 0, 0),  # Created later
                highlights=[],
            ),
        ]

        # Mock LLM response - should only be asked about item-1
        mock_ollama_client.generate.return_value = '''[
            {
                "item_id": "item-1",
                "theme_id": "technology",
                "confidence": 0.70
            }
        ]'''

        result = generator.update_assignments(items)

        # Should return assignment only for item-1
        assert len(result) == 1
        assert result[0].item_id == "item-1"

        # Prompt should only contain item-1
        call_kwargs = mock_ollama_client.generate.call_args[1]
        prompt = call_kwargs["prompt"]
        assert "item-1" in prompt
        assert "Old Unassigned Book" in prompt
        assert "item-2" not in prompt
