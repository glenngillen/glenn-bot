"""Tests for library builder module.

This module tests the LibraryBuilder class which orchestrates the entire build process
for generating the static library site from ChromaDB knowledge base content.

Test classes are organized by the methods they test:
- TestLibraryBuilderInitialization: LibraryBuilder class initialization
- TestLoadBuildState: _load_build_state() reading _build_state.json
- TestSaveBuildState: _save_build_state() persisting state
- TestComputeItemHash: _compute_item_hash() generating content hash
- TestGetChangedItems: _get_changed_items() detecting modifications
- TestBuild: build() orchestration
- TestBuildWithForce: build() with force=True
- TestExportLibraryJson: _export_library_json() creating debug file
"""

import json
import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch, call

from src.library.models import LibraryItem, ContentType, Theme, ThemeAssignment


class TestLibraryBuilderInitialization:
    """Tests for LibraryBuilder class initialization."""

    def test_library_builder_can_be_imported(self):
        """LibraryBuilder class should be importable from the module."""
        from src.library.builder import LibraryBuilder

        assert LibraryBuilder is not None

    def test_library_builder_init_with_dependencies(self, temp_dir):
        """LibraryBuilder should accept required dependencies."""
        from src.library.builder import LibraryBuilder

        mock_knowledge_base = MagicMock()
        mock_ollama_client = MagicMock()
        data_dir = temp_dir / "library"
        site_dir = temp_dir / "library-site"

        builder = LibraryBuilder(
            knowledge_base=mock_knowledge_base,
            ollama_client=mock_ollama_client,
            data_dir=data_dir,
            site_dir=site_dir,
        )

        assert builder.knowledge_base == mock_knowledge_base
        assert builder.ollama_client == mock_ollama_client
        assert builder.data_dir == data_dir
        assert builder.site_dir == site_dir

    def test_library_builder_init_creates_directories(self, temp_dir):
        """LibraryBuilder should create data and site directories if they don't exist."""
        from src.library.builder import LibraryBuilder

        mock_knowledge_base = MagicMock()
        mock_ollama_client = MagicMock()
        data_dir = temp_dir / "library" / "nested"
        site_dir = temp_dir / "library-site" / "nested"

        assert not data_dir.exists()
        assert not site_dir.exists()

        builder = LibraryBuilder(
            knowledge_base=mock_knowledge_base,
            ollama_client=mock_ollama_client,
            data_dir=data_dir,
            site_dir=site_dir,
        )

        assert data_dir.exists()
        assert site_dir.exists()

    def test_library_builder_init_accepts_string_paths(self, temp_dir):
        """LibraryBuilder should accept string paths and convert to Path objects."""
        from src.library.builder import LibraryBuilder

        mock_knowledge_base = MagicMock()
        mock_ollama_client = MagicMock()
        data_dir = str(temp_dir / "library")
        site_dir = str(temp_dir / "library-site")

        builder = LibraryBuilder(
            knowledge_base=mock_knowledge_base,
            ollama_client=mock_ollama_client,
            data_dir=data_dir,
            site_dir=site_dir,
        )

        assert isinstance(builder.data_dir, Path)
        assert isinstance(builder.site_dir, Path)

    def test_library_builder_has_build_state_file_attribute(self, temp_dir):
        """LibraryBuilder should have a build_state_file attribute pointing to _build_state.json."""
        from src.library.builder import LibraryBuilder

        mock_knowledge_base = MagicMock()
        mock_ollama_client = MagicMock()
        data_dir = temp_dir / "library"
        site_dir = temp_dir / "library-site"

        builder = LibraryBuilder(
            knowledge_base=mock_knowledge_base,
            ollama_client=mock_ollama_client,
            data_dir=data_dir,
            site_dir=site_dir,
        )

        assert hasattr(builder, "build_state_file")
        assert builder.build_state_file == site_dir / "_build_state.json"


class TestLoadBuildState:
    """Tests for _load_build_state() method that reads _build_state.json."""

    def test_load_build_state_returns_empty_dict_when_file_not_exists(self, temp_dir):
        """_load_build_state() should return empty dict when _build_state.json doesn't exist."""
        from src.library.builder import LibraryBuilder

        mock_knowledge_base = MagicMock()
        mock_ollama_client = MagicMock()
        data_dir = temp_dir / "library"
        site_dir = temp_dir / "library-site"

        builder = LibraryBuilder(
            knowledge_base=mock_knowledge_base,
            ollama_client=mock_ollama_client,
            data_dir=data_dir,
            site_dir=site_dir,
        )

        result = builder._load_build_state()

        assert result == {}

    def test_load_build_state_reads_existing_file(self, temp_dir):
        """_load_build_state() should read and parse _build_state.json."""
        from src.library.builder import LibraryBuilder

        mock_knowledge_base = MagicMock()
        mock_ollama_client = MagicMock()
        data_dir = temp_dir / "library"
        site_dir = temp_dir / "library-site"
        site_dir.mkdir(parents=True, exist_ok=True)

        # Create a build state file
        build_state = {
            "last_build": "2024-01-15T10:00:00",
            "item_hashes": {"doc_1": "hash123", "doc_2": "hash456"},
            "theme_version": "1.0",
        }
        (site_dir / "_build_state.json").write_text(json.dumps(build_state))

        builder = LibraryBuilder(
            knowledge_base=mock_knowledge_base,
            ollama_client=mock_ollama_client,
            data_dir=data_dir,
            site_dir=site_dir,
        )

        result = builder._load_build_state()

        assert result == build_state
        assert result["last_build"] == "2024-01-15T10:00:00"
        assert result["item_hashes"]["doc_1"] == "hash123"

    def test_load_build_state_handles_invalid_json(self, temp_dir):
        """_load_build_state() should return empty dict for invalid JSON."""
        from src.library.builder import LibraryBuilder

        mock_knowledge_base = MagicMock()
        mock_ollama_client = MagicMock()
        data_dir = temp_dir / "library"
        site_dir = temp_dir / "library-site"
        site_dir.mkdir(parents=True, exist_ok=True)

        # Create an invalid JSON file
        (site_dir / "_build_state.json").write_text("not valid json {{{")

        builder = LibraryBuilder(
            knowledge_base=mock_knowledge_base,
            ollama_client=mock_ollama_client,
            data_dir=data_dir,
            site_dir=site_dir,
        )

        result = builder._load_build_state()

        assert result == {}


class TestSaveBuildState:
    """Tests for _save_build_state() method that persists state."""

    def test_save_build_state_creates_file(self, temp_dir):
        """_save_build_state() should create _build_state.json file."""
        from src.library.builder import LibraryBuilder

        mock_knowledge_base = MagicMock()
        mock_ollama_client = MagicMock()
        data_dir = temp_dir / "library"
        site_dir = temp_dir / "library-site"

        builder = LibraryBuilder(
            knowledge_base=mock_knowledge_base,
            ollama_client=mock_ollama_client,
            data_dir=data_dir,
            site_dir=site_dir,
        )

        build_state = {"last_build": "2024-01-15T10:00:00"}
        builder._save_build_state(build_state)

        assert (site_dir / "_build_state.json").exists()

    def test_save_build_state_writes_correct_content(self, temp_dir):
        """_save_build_state() should write state as JSON."""
        from src.library.builder import LibraryBuilder

        mock_knowledge_base = MagicMock()
        mock_ollama_client = MagicMock()
        data_dir = temp_dir / "library"
        site_dir = temp_dir / "library-site"

        builder = LibraryBuilder(
            knowledge_base=mock_knowledge_base,
            ollama_client=mock_ollama_client,
            data_dir=data_dir,
            site_dir=site_dir,
        )

        build_state = {
            "last_build": "2024-01-15T10:00:00",
            "item_hashes": {"doc_1": "abc123"},
        }
        builder._save_build_state(build_state)

        content = (site_dir / "_build_state.json").read_text()
        loaded = json.loads(content)

        assert loaded == build_state

    def test_save_build_state_overwrites_existing_file(self, temp_dir):
        """_save_build_state() should overwrite existing _build_state.json."""
        from src.library.builder import LibraryBuilder

        mock_knowledge_base = MagicMock()
        mock_ollama_client = MagicMock()
        data_dir = temp_dir / "library"
        site_dir = temp_dir / "library-site"
        site_dir.mkdir(parents=True, exist_ok=True)

        # Create an existing file
        (site_dir / "_build_state.json").write_text('{"old": "data"}')

        builder = LibraryBuilder(
            knowledge_base=mock_knowledge_base,
            ollama_client=mock_ollama_client,
            data_dir=data_dir,
            site_dir=site_dir,
        )

        new_state = {"new": "state"}
        builder._save_build_state(new_state)

        content = (site_dir / "_build_state.json").read_text()
        loaded = json.loads(content)

        assert loaded == new_state
        assert "old" not in loaded


class TestComputeItemHash:
    """Tests for _compute_item_hash() method that generates content hash."""

    def test_compute_item_hash_returns_string(self, temp_dir):
        """_compute_item_hash() should return a hash string."""
        from src.library.builder import LibraryBuilder

        mock_knowledge_base = MagicMock()
        mock_ollama_client = MagicMock()
        data_dir = temp_dir / "library"
        site_dir = temp_dir / "library-site"

        builder = LibraryBuilder(
            knowledge_base=mock_knowledge_base,
            ollama_client=mock_ollama_client,
            data_dir=data_dir,
            site_dir=site_dir,
        )

        item = LibraryItem(
            id="doc_1",
            content_type=ContentType.BOOK,
            title="Test Book",
            summary="A test book summary",
            full_content="Full content here",
            source_url=None,
            cover_image_url=None,
            metadata={},
            themes=[],
            created_at=datetime(2024, 1, 15),
            highlights=[],
        )

        result = builder._compute_item_hash(item)

        assert isinstance(result, str)
        assert len(result) > 0

    def test_compute_item_hash_same_content_same_hash(self, temp_dir):
        """_compute_item_hash() should return same hash for identical items."""
        from src.library.builder import LibraryBuilder

        mock_knowledge_base = MagicMock()
        mock_ollama_client = MagicMock()
        data_dir = temp_dir / "library"
        site_dir = temp_dir / "library-site"

        builder = LibraryBuilder(
            knowledge_base=mock_knowledge_base,
            ollama_client=mock_ollama_client,
            data_dir=data_dir,
            site_dir=site_dir,
        )

        item1 = LibraryItem(
            id="doc_1",
            content_type=ContentType.BOOK,
            title="Test Book",
            summary="A test book summary",
            full_content="Full content here",
            source_url="https://example.com",
            cover_image_url=None,
            metadata={"key": "value"},
            themes=["theme-1"],
            created_at=datetime(2024, 1, 15),
            highlights=["highlight 1"],
        )

        item2 = LibraryItem(
            id="doc_1",
            content_type=ContentType.BOOK,
            title="Test Book",
            summary="A test book summary",
            full_content="Full content here",
            source_url="https://example.com",
            cover_image_url=None,
            metadata={"key": "value"},
            themes=["theme-1"],
            created_at=datetime(2024, 1, 15),
            highlights=["highlight 1"],
        )

        hash1 = builder._compute_item_hash(item1)
        hash2 = builder._compute_item_hash(item2)

        assert hash1 == hash2

    def test_compute_item_hash_different_content_different_hash(self, temp_dir):
        """_compute_item_hash() should return different hash for different items."""
        from src.library.builder import LibraryBuilder

        mock_knowledge_base = MagicMock()
        mock_ollama_client = MagicMock()
        data_dir = temp_dir / "library"
        site_dir = temp_dir / "library-site"

        builder = LibraryBuilder(
            knowledge_base=mock_knowledge_base,
            ollama_client=mock_ollama_client,
            data_dir=data_dir,
            site_dir=site_dir,
        )

        item1 = LibraryItem(
            id="doc_1",
            content_type=ContentType.BOOK,
            title="Test Book",
            summary="A test book summary",
            full_content="Full content here",
            source_url=None,
            cover_image_url=None,
            metadata={},
            themes=[],
            created_at=datetime(2024, 1, 15),
            highlights=[],
        )

        item2 = LibraryItem(
            id="doc_1",
            content_type=ContentType.BOOK,
            title="Different Title",  # Changed
            summary="A test book summary",
            full_content="Full content here",
            source_url=None,
            cover_image_url=None,
            metadata={},
            themes=[],
            created_at=datetime(2024, 1, 15),
            highlights=[],
        )

        hash1 = builder._compute_item_hash(item1)
        hash2 = builder._compute_item_hash(item2)

        assert hash1 != hash2

    def test_compute_item_hash_changes_on_content_change(self, temp_dir):
        """_compute_item_hash() should detect changes in full_content."""
        from src.library.builder import LibraryBuilder

        mock_knowledge_base = MagicMock()
        mock_ollama_client = MagicMock()
        data_dir = temp_dir / "library"
        site_dir = temp_dir / "library-site"

        builder = LibraryBuilder(
            knowledge_base=mock_knowledge_base,
            ollama_client=mock_ollama_client,
            data_dir=data_dir,
            site_dir=site_dir,
        )

        item1 = LibraryItem(
            id="doc_1",
            content_type=ContentType.BOOK,
            title="Test Book",
            summary="Summary",
            full_content="Original content",
            source_url=None,
            cover_image_url=None,
            metadata={},
            themes=[],
            created_at=datetime(2024, 1, 15),
            highlights=[],
        )

        item2 = LibraryItem(
            id="doc_1",
            content_type=ContentType.BOOK,
            title="Test Book",
            summary="Summary",
            full_content="Modified content",  # Changed
            source_url=None,
            cover_image_url=None,
            metadata={},
            themes=[],
            created_at=datetime(2024, 1, 15),
            highlights=[],
        )

        hash1 = builder._compute_item_hash(item1)
        hash2 = builder._compute_item_hash(item2)

        assert hash1 != hash2


class TestGetChangedItems:
    """Tests for _get_changed_items() method that detects modifications."""

    def test_get_changed_items_all_items_when_no_previous_state(self, temp_dir):
        """_get_changed_items() should return all items when no previous build state exists."""
        from src.library.builder import LibraryBuilder

        mock_knowledge_base = MagicMock()
        mock_ollama_client = MagicMock()
        data_dir = temp_dir / "library"
        site_dir = temp_dir / "library-site"

        builder = LibraryBuilder(
            knowledge_base=mock_knowledge_base,
            ollama_client=mock_ollama_client,
            data_dir=data_dir,
            site_dir=site_dir,
        )

        items = [
            LibraryItem(
                id="doc_1",
                content_type=ContentType.BOOK,
                title="Book 1",
                summary="Summary 1",
                full_content="Content 1",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime(2024, 1, 15),
                highlights=[],
            ),
            LibraryItem(
                id="doc_2",
                content_type=ContentType.ARTICLE,
                title="Article 1",
                summary="Summary 2",
                full_content="Content 2",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime(2024, 1, 16),
                highlights=[],
            ),
        ]

        previous_state = {}
        changed = builder._get_changed_items(items, previous_state)

        assert len(changed) == 2
        assert changed[0].id == "doc_1"
        assert changed[1].id == "doc_2"

    def test_get_changed_items_detects_new_items(self, temp_dir):
        """_get_changed_items() should detect items not in previous state."""
        from src.library.builder import LibraryBuilder

        mock_knowledge_base = MagicMock()
        mock_ollama_client = MagicMock()
        data_dir = temp_dir / "library"
        site_dir = temp_dir / "library-site"

        builder = LibraryBuilder(
            knowledge_base=mock_knowledge_base,
            ollama_client=mock_ollama_client,
            data_dir=data_dir,
            site_dir=site_dir,
        )

        items = [
            LibraryItem(
                id="doc_1",
                content_type=ContentType.BOOK,
                title="Book 1",
                summary="Summary 1",
                full_content="Content 1",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime(2024, 1, 15),
                highlights=[],
            ),
            LibraryItem(
                id="doc_2",  # New item
                content_type=ContentType.ARTICLE,
                title="Article 1",
                summary="Summary 2",
                full_content="Content 2",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime(2024, 1, 16),
                highlights=[],
            ),
        ]

        # Only doc_1 in previous state
        previous_state = {"item_hashes": {"doc_1": "somehash"}}

        # Patch _compute_item_hash to return same hash for doc_1
        with patch.object(builder, "_compute_item_hash") as mock_hash:
            mock_hash.side_effect = lambda item: "somehash" if item.id == "doc_1" else "newhash"
            changed = builder._get_changed_items(items, previous_state)

        # Only doc_2 should be in changed (new item)
        assert len(changed) == 1
        assert changed[0].id == "doc_2"

    def test_get_changed_items_detects_modified_items(self, temp_dir):
        """_get_changed_items() should detect items with different hashes."""
        from src.library.builder import LibraryBuilder

        mock_knowledge_base = MagicMock()
        mock_ollama_client = MagicMock()
        data_dir = temp_dir / "library"
        site_dir = temp_dir / "library-site"

        builder = LibraryBuilder(
            knowledge_base=mock_knowledge_base,
            ollama_client=mock_ollama_client,
            data_dir=data_dir,
            site_dir=site_dir,
        )

        items = [
            LibraryItem(
                id="doc_1",
                content_type=ContentType.BOOK,
                title="Modified Title",  # Changed
                summary="Summary 1",
                full_content="Content 1",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime(2024, 1, 15),
                highlights=[],
            ),
        ]

        # Previous state has different hash
        previous_state = {"item_hashes": {"doc_1": "oldhash"}}

        # Patch _compute_item_hash to return new hash
        with patch.object(builder, "_compute_item_hash") as mock_hash:
            mock_hash.return_value = "newhash"
            changed = builder._get_changed_items(items, previous_state)

        assert len(changed) == 1
        assert changed[0].id == "doc_1"

    def test_get_changed_items_returns_empty_when_no_changes(self, temp_dir):
        """_get_changed_items() should return empty list when nothing changed."""
        from src.library.builder import LibraryBuilder

        mock_knowledge_base = MagicMock()
        mock_ollama_client = MagicMock()
        data_dir = temp_dir / "library"
        site_dir = temp_dir / "library-site"

        builder = LibraryBuilder(
            knowledge_base=mock_knowledge_base,
            ollama_client=mock_ollama_client,
            data_dir=data_dir,
            site_dir=site_dir,
        )

        items = [
            LibraryItem(
                id="doc_1",
                content_type=ContentType.BOOK,
                title="Book 1",
                summary="Summary 1",
                full_content="Content 1",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime(2024, 1, 15),
                highlights=[],
            ),
        ]

        # Previous state has same hash
        previous_state = {"item_hashes": {"doc_1": "samehash"}}

        # Patch _compute_item_hash to return same hash
        with patch.object(builder, "_compute_item_hash") as mock_hash:
            mock_hash.return_value = "samehash"
            changed = builder._get_changed_items(items, previous_state)

        assert len(changed) == 0


class TestBuild:
    """Tests for build() method that orchestrates the full build process."""

    def test_build_exports_content_from_knowledge_base(self, temp_dir):
        """build() should call ContentExporter.export_all()."""
        from src.library.builder import LibraryBuilder

        mock_knowledge_base = MagicMock()
        mock_knowledge_base.export_knowledge.return_value = {"documents": []}
        mock_ollama_client = MagicMock()
        data_dir = temp_dir / "library"
        site_dir = temp_dir / "library-site"

        builder = LibraryBuilder(
            knowledge_base=mock_knowledge_base,
            ollama_client=mock_ollama_client,
            data_dir=data_dir,
            site_dir=site_dir,
        )

        with patch("src.library.builder.ContentExporter") as MockExporter:
            mock_exporter = MagicMock()
            mock_exporter.export_all.return_value = []
            MockExporter.return_value = mock_exporter

            with patch("src.library.builder.ThemeGenerator"):
                with patch("src.library.builder.CoverResolver"):
                    with patch("src.library.builder.SearchIndexer"):
                        with patch("src.library.builder.StaticGenerator"):
                            builder.build()

            mock_exporter.export_all.assert_called_once()

    def test_build_resolves_cover_images(self, temp_dir):
        """build() should call CoverResolver.resolve_all_covers()."""
        from src.library.builder import LibraryBuilder

        mock_knowledge_base = MagicMock()
        mock_ollama_client = MagicMock()
        data_dir = temp_dir / "library"
        site_dir = temp_dir / "library-site"

        builder = LibraryBuilder(
            knowledge_base=mock_knowledge_base,
            ollama_client=mock_ollama_client,
            data_dir=data_dir,
            site_dir=site_dir,
        )

        sample_items = [
            LibraryItem(
                id="doc_1",
                content_type=ContentType.BOOK,
                title="Book 1",
                summary="Summary",
                full_content="Content",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime(2024, 1, 15),
                highlights=[],
            )
        ]

        with patch("src.library.builder.ContentExporter") as MockExporter:
            mock_exporter = MagicMock()
            mock_exporter.export_all.return_value = sample_items
            MockExporter.return_value = mock_exporter

            with patch("src.library.builder.CoverResolver") as MockResolver:
                mock_resolver = MagicMock()
                mock_resolver.resolve_all_covers.return_value = sample_items
                MockResolver.return_value = mock_resolver

                with patch("src.library.builder.ThemeGenerator") as MockThemeGen:
                    mock_theme_gen = MagicMock()
                    mock_theme_gen.themes = []
                    mock_theme_gen.assignments = []
                    mock_theme_gen.load_themes.return_value = []
                    mock_theme_gen.generate_themes.return_value = []
                    MockThemeGen.return_value = mock_theme_gen

                    with patch("src.library.builder.SearchIndexer"):
                        with patch("src.library.builder.StaticGenerator"):
                            builder.build()

                mock_resolver.resolve_all_covers.assert_called_once()

    def test_build_generates_themes(self, temp_dir):
        """build() should call ThemeGenerator.generate_themes() for new builds."""
        from src.library.builder import LibraryBuilder

        mock_knowledge_base = MagicMock()
        mock_ollama_client = MagicMock()
        data_dir = temp_dir / "library"
        site_dir = temp_dir / "library-site"

        builder = LibraryBuilder(
            knowledge_base=mock_knowledge_base,
            ollama_client=mock_ollama_client,
            data_dir=data_dir,
            site_dir=site_dir,
        )

        sample_items = [
            LibraryItem(
                id="doc_1",
                content_type=ContentType.BOOK,
                title="Book 1",
                summary="Summary",
                full_content="Content",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime(2024, 1, 15),
                highlights=[],
            )
        ]

        with patch("src.library.builder.ContentExporter") as MockExporter:
            mock_exporter = MagicMock()
            mock_exporter.export_all.return_value = sample_items
            MockExporter.return_value = mock_exporter

            with patch("src.library.builder.CoverResolver") as MockResolver:
                mock_resolver = MagicMock()
                mock_resolver.resolve_all_covers.return_value = sample_items
                MockResolver.return_value = mock_resolver

                with patch("src.library.builder.ThemeGenerator") as MockThemeGen:
                    mock_theme_gen = MagicMock()
                    mock_theme_gen.themes = []
                    mock_theme_gen.assignments = []
                    mock_theme_gen.load_themes.return_value = []  # No existing themes
                    mock_theme_gen.generate_themes.return_value = []
                    mock_theme_gen.assign_items_to_themes.return_value = []
                    MockThemeGen.return_value = mock_theme_gen

                    with patch("src.library.builder.SearchIndexer"):
                        with patch("src.library.builder.StaticGenerator"):
                            builder.build()

                    # Should generate themes because load_themes returned empty
                    mock_theme_gen.generate_themes.assert_called()

    def test_build_generates_search_index(self, temp_dir):
        """build() should call SearchIndexer.write_index()."""
        from src.library.builder import LibraryBuilder

        mock_knowledge_base = MagicMock()
        mock_ollama_client = MagicMock()
        data_dir = temp_dir / "library"
        site_dir = temp_dir / "library-site"

        builder = LibraryBuilder(
            knowledge_base=mock_knowledge_base,
            ollama_client=mock_ollama_client,
            data_dir=data_dir,
            site_dir=site_dir,
        )

        with patch("src.library.builder.ContentExporter") as MockExporter:
            mock_exporter = MagicMock()
            mock_exporter.export_all.return_value = []
            MockExporter.return_value = mock_exporter

            with patch("src.library.builder.CoverResolver"):
                with patch("src.library.builder.ThemeGenerator") as MockThemeGen:
                    mock_theme_gen = MagicMock()
                    mock_theme_gen.themes = []
                    mock_theme_gen.assignments = []
                    mock_theme_gen.load_themes.return_value = []
                    MockThemeGen.return_value = mock_theme_gen

                    with patch("src.library.builder.SearchIndexer") as MockIndexer:
                        mock_indexer = MagicMock()
                        MockIndexer.return_value = mock_indexer

                        with patch("src.library.builder.StaticGenerator"):
                            builder.build()

                        mock_indexer.write_index.assert_called_once()

    def test_build_generates_static_site(self, temp_dir):
        """build() should call StaticGenerator.generate_all()."""
        from src.library.builder import LibraryBuilder

        mock_knowledge_base = MagicMock()
        mock_ollama_client = MagicMock()
        data_dir = temp_dir / "library"
        site_dir = temp_dir / "library-site"

        builder = LibraryBuilder(
            knowledge_base=mock_knowledge_base,
            ollama_client=mock_ollama_client,
            data_dir=data_dir,
            site_dir=site_dir,
        )

        with patch("src.library.builder.ContentExporter") as MockExporter:
            mock_exporter = MagicMock()
            mock_exporter.export_all.return_value = []
            MockExporter.return_value = mock_exporter

            with patch("src.library.builder.CoverResolver"):
                with patch("src.library.builder.ThemeGenerator") as MockThemeGen:
                    mock_theme_gen = MagicMock()
                    mock_theme_gen.themes = []
                    mock_theme_gen.assignments = []
                    mock_theme_gen.load_themes.return_value = []
                    MockThemeGen.return_value = mock_theme_gen

                    with patch("src.library.builder.SearchIndexer"):
                        with patch("src.library.builder.StaticGenerator") as MockGenerator:
                            mock_generator = MagicMock()
                            mock_generator.generate_all.return_value = {"pages_generated": 3}
                            MockGenerator.return_value = mock_generator

                            builder.build()

                            mock_generator.generate_all.assert_called_once()

    def test_build_saves_build_state(self, temp_dir):
        """build() should save build state with item hashes and timestamp."""
        from src.library.builder import LibraryBuilder

        mock_knowledge_base = MagicMock()
        mock_ollama_client = MagicMock()
        data_dir = temp_dir / "library"
        site_dir = temp_dir / "library-site"

        builder = LibraryBuilder(
            knowledge_base=mock_knowledge_base,
            ollama_client=mock_ollama_client,
            data_dir=data_dir,
            site_dir=site_dir,
        )

        sample_items = [
            LibraryItem(
                id="doc_1",
                content_type=ContentType.BOOK,
                title="Book 1",
                summary="Summary",
                full_content="Content",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime(2024, 1, 15),
                highlights=[],
            )
        ]

        with patch("src.library.builder.ContentExporter") as MockExporter:
            mock_exporter = MagicMock()
            mock_exporter.export_all.return_value = sample_items
            MockExporter.return_value = mock_exporter

            with patch("src.library.builder.CoverResolver") as MockResolver:
                mock_resolver = MagicMock()
                mock_resolver.resolve_all_covers.return_value = sample_items
                MockResolver.return_value = mock_resolver

                with patch("src.library.builder.ThemeGenerator") as MockThemeGen:
                    mock_theme_gen = MagicMock()
                    mock_theme_gen.themes = []
                    mock_theme_gen.assignments = []
                    mock_theme_gen.load_themes.return_value = []
                    MockThemeGen.return_value = mock_theme_gen

                    with patch("src.library.builder.SearchIndexer"):
                        with patch("src.library.builder.StaticGenerator"):
                            builder.build()

        # Check that build state was saved
        assert (site_dir / "_build_state.json").exists()
        state = json.loads((site_dir / "_build_state.json").read_text())
        assert "last_build" in state
        assert "item_hashes" in state

    def test_build_returns_summary(self, temp_dir):
        """build() should return a summary dict with build information."""
        from src.library.builder import LibraryBuilder

        mock_knowledge_base = MagicMock()
        mock_ollama_client = MagicMock()
        data_dir = temp_dir / "library"
        site_dir = temp_dir / "library-site"

        builder = LibraryBuilder(
            knowledge_base=mock_knowledge_base,
            ollama_client=mock_ollama_client,
            data_dir=data_dir,
            site_dir=site_dir,
        )

        with patch("src.library.builder.ContentExporter") as MockExporter:
            mock_exporter = MagicMock()
            mock_exporter.export_all.return_value = []
            MockExporter.return_value = mock_exporter

            with patch("src.library.builder.CoverResolver"):
                with patch("src.library.builder.ThemeGenerator") as MockThemeGen:
                    mock_theme_gen = MagicMock()
                    mock_theme_gen.themes = []
                    mock_theme_gen.assignments = []
                    mock_theme_gen.load_themes.return_value = []
                    MockThemeGen.return_value = mock_theme_gen

                    with patch("src.library.builder.SearchIndexer"):
                        with patch("src.library.builder.StaticGenerator") as MockGenerator:
                            mock_generator = MagicMock()
                            mock_generator.generate_all.return_value = {"pages_generated": 5}
                            MockGenerator.return_value = mock_generator

                            result = builder.build()

        assert result is not None
        assert isinstance(result, dict)
        # Should contain some summary information
        assert "items_count" in result or "pages_generated" in result


class TestBuildWithForce:
    """Tests for build() method with force=True flag."""

    def test_build_force_regenerates_themes(self, temp_dir):
        """build(force=True) should regenerate themes even if they exist."""
        from src.library.builder import LibraryBuilder

        mock_knowledge_base = MagicMock()
        mock_ollama_client = MagicMock()
        data_dir = temp_dir / "library"
        site_dir = temp_dir / "library-site"

        builder = LibraryBuilder(
            knowledge_base=mock_knowledge_base,
            ollama_client=mock_ollama_client,
            data_dir=data_dir,
            site_dir=site_dir,
        )

        existing_themes = [
            Theme(
                id="existing-theme",
                name="Existing Theme",
                description="Already exists",
                keywords=["existing"],
                item_count=1,
                created_at=datetime(2024, 1, 10),
                updated_at=datetime(2024, 1, 10),
            )
        ]

        with patch("src.library.builder.ContentExporter") as MockExporter:
            mock_exporter = MagicMock()
            mock_exporter.export_all.return_value = []
            MockExporter.return_value = mock_exporter

            with patch("src.library.builder.CoverResolver"):
                with patch("src.library.builder.ThemeGenerator") as MockThemeGen:
                    mock_theme_gen = MagicMock()
                    mock_theme_gen.themes = existing_themes
                    mock_theme_gen.assignments = []
                    mock_theme_gen.load_themes.return_value = existing_themes
                    mock_theme_gen.generate_themes.return_value = []
                    MockThemeGen.return_value = mock_theme_gen

                    with patch("src.library.builder.SearchIndexer"):
                        with patch("src.library.builder.StaticGenerator"):
                            builder.build(force=True)

                    # Should call generate_themes even though themes exist
                    mock_theme_gen.generate_themes.assert_called()

    def test_build_force_clears_cover_cache(self, temp_dir):
        """build(force=True) should resolve covers without using cache."""
        from src.library.builder import LibraryBuilder

        mock_knowledge_base = MagicMock()
        mock_ollama_client = MagicMock()
        data_dir = temp_dir / "library"
        site_dir = temp_dir / "library-site"

        # Create a pre-existing cover cache
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "cover_cache.json").write_text('{"doc_1": {"url": "old_url"}}')

        builder = LibraryBuilder(
            knowledge_base=mock_knowledge_base,
            ollama_client=mock_ollama_client,
            data_dir=data_dir,
            site_dir=site_dir,
        )

        sample_items = [
            LibraryItem(
                id="doc_1",
                content_type=ContentType.BOOK,
                title="Book 1",
                summary="Summary",
                full_content="Content",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime(2024, 1, 15),
                highlights=[],
            )
        ]

        with patch("src.library.builder.ContentExporter") as MockExporter:
            mock_exporter = MagicMock()
            mock_exporter.export_all.return_value = sample_items
            MockExporter.return_value = mock_exporter

            with patch("src.library.builder.CoverResolver") as MockResolver:
                mock_resolver = MagicMock()
                mock_resolver.resolve_all_covers.return_value = sample_items
                mock_resolver.cache = {}
                MockResolver.return_value = mock_resolver

                with patch("src.library.builder.ThemeGenerator") as MockThemeGen:
                    mock_theme_gen = MagicMock()
                    mock_theme_gen.themes = []
                    mock_theme_gen.assignments = []
                    mock_theme_gen.load_themes.return_value = []
                    MockThemeGen.return_value = mock_theme_gen

                    with patch("src.library.builder.SearchIndexer"):
                        with patch("src.library.builder.StaticGenerator"):
                            builder.build(force=True)

                # Should have cleared cache or created new resolver
                # The implementation should handle this
                mock_resolver.resolve_all_covers.assert_called()

    def test_build_force_ignores_previous_build_state(self, temp_dir):
        """build(force=True) should not use previous build state for change detection."""
        from src.library.builder import LibraryBuilder

        mock_knowledge_base = MagicMock()
        mock_ollama_client = MagicMock()
        data_dir = temp_dir / "library"
        site_dir = temp_dir / "library-site"

        # Create pre-existing build state
        site_dir.mkdir(parents=True, exist_ok=True)
        (site_dir / "_build_state.json").write_text(json.dumps({
            "last_build": "2024-01-10T10:00:00",
            "item_hashes": {"doc_1": "oldhash"},
        }))

        builder = LibraryBuilder(
            knowledge_base=mock_knowledge_base,
            ollama_client=mock_ollama_client,
            data_dir=data_dir,
            site_dir=site_dir,
        )

        sample_items = [
            LibraryItem(
                id="doc_1",
                content_type=ContentType.BOOK,
                title="Book 1",
                summary="Summary",
                full_content="Content",  # Same content as before
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime(2024, 1, 15),
                highlights=[],
            )
        ]

        with patch("src.library.builder.ContentExporter") as MockExporter:
            mock_exporter = MagicMock()
            mock_exporter.export_all.return_value = sample_items
            MockExporter.return_value = mock_exporter

            with patch("src.library.builder.CoverResolver") as MockResolver:
                mock_resolver = MagicMock()
                mock_resolver.resolve_all_covers.return_value = sample_items
                MockResolver.return_value = mock_resolver

                with patch("src.library.builder.ThemeGenerator") as MockThemeGen:
                    mock_theme_gen = MagicMock()
                    mock_theme_gen.themes = []
                    mock_theme_gen.assignments = []
                    mock_theme_gen.load_themes.return_value = []
                    mock_theme_gen.generate_themes.return_value = []
                    MockThemeGen.return_value = mock_theme_gen

                    with patch("src.library.builder.SearchIndexer"):
                        with patch("src.library.builder.StaticGenerator") as MockGenerator:
                            mock_generator = MagicMock()
                            MockGenerator.return_value = mock_generator

                            builder.build(force=True)

                            # Should still generate the site (not skip due to no changes)
                            mock_generator.generate_all.assert_called()


class TestExportLibraryJson:
    """Tests for _export_library_json() method that creates debug file."""

    def test_export_library_json_creates_file(self, temp_dir):
        """_export_library_json() should create library.json in _data directory."""
        from src.library.builder import LibraryBuilder

        mock_knowledge_base = MagicMock()
        mock_ollama_client = MagicMock()
        data_dir = temp_dir / "library"
        site_dir = temp_dir / "library-site"

        builder = LibraryBuilder(
            knowledge_base=mock_knowledge_base,
            ollama_client=mock_ollama_client,
            data_dir=data_dir,
            site_dir=site_dir,
        )

        items = [
            LibraryItem(
                id="doc_1",
                content_type=ContentType.BOOK,
                title="Book 1",
                summary="Summary",
                full_content="Content",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=["theme-1"],
                created_at=datetime(2024, 1, 15),
                highlights=[],
            )
        ]
        themes = [
            Theme(
                id="theme-1",
                name="Theme 1",
                description="Description",
                keywords=["keyword"],
                item_count=1,
                created_at=datetime(2024, 1, 15),
                updated_at=datetime(2024, 1, 15),
            )
        ]

        builder._export_library_json(items, themes)

        assert (site_dir / "_data" / "library.json").exists()

    def test_export_library_json_contains_items(self, temp_dir):
        """_export_library_json() should include all items in the output."""
        from src.library.builder import LibraryBuilder

        mock_knowledge_base = MagicMock()
        mock_ollama_client = MagicMock()
        data_dir = temp_dir / "library"
        site_dir = temp_dir / "library-site"

        builder = LibraryBuilder(
            knowledge_base=mock_knowledge_base,
            ollama_client=mock_ollama_client,
            data_dir=data_dir,
            site_dir=site_dir,
        )

        items = [
            LibraryItem(
                id="doc_1",
                content_type=ContentType.BOOK,
                title="Book 1",
                summary="Summary 1",
                full_content="Content 1",
                source_url="https://example.com",
                cover_image_url=None,
                metadata={"key": "value"},
                themes=["theme-1"],
                created_at=datetime(2024, 1, 15),
                highlights=["highlight 1"],
            )
        ]
        themes = []

        builder._export_library_json(items, themes)

        content = (site_dir / "_data" / "library.json").read_text()
        data = json.loads(content)

        assert "items" in data
        assert len(data["items"]) == 1
        assert data["items"][0]["id"] == "doc_1"
        assert data["items"][0]["title"] == "Book 1"

    def test_export_library_json_contains_themes(self, temp_dir):
        """_export_library_json() should include all themes in the output."""
        from src.library.builder import LibraryBuilder

        mock_knowledge_base = MagicMock()
        mock_ollama_client = MagicMock()
        data_dir = temp_dir / "library"
        site_dir = temp_dir / "library-site"

        builder = LibraryBuilder(
            knowledge_base=mock_knowledge_base,
            ollama_client=mock_ollama_client,
            data_dir=data_dir,
            site_dir=site_dir,
        )

        items = []
        themes = [
            Theme(
                id="theme-1",
                name="Theme 1",
                description="Description 1",
                keywords=["keyword1", "keyword2"],
                item_count=5,
                created_at=datetime(2024, 1, 15),
                updated_at=datetime(2024, 1, 20),
            )
        ]

        builder._export_library_json(items, themes)

        content = (site_dir / "_data" / "library.json").read_text()
        data = json.loads(content)

        assert "themes" in data
        assert len(data["themes"]) == 1
        assert data["themes"][0]["id"] == "theme-1"
        assert data["themes"][0]["name"] == "Theme 1"

    def test_export_library_json_creates_data_directory(self, temp_dir):
        """_export_library_json() should create _data directory if it doesn't exist."""
        from src.library.builder import LibraryBuilder

        mock_knowledge_base = MagicMock()
        mock_ollama_client = MagicMock()
        data_dir = temp_dir / "library"
        site_dir = temp_dir / "library-site"

        builder = LibraryBuilder(
            knowledge_base=mock_knowledge_base,
            ollama_client=mock_ollama_client,
            data_dir=data_dir,
            site_dir=site_dir,
        )

        # site_dir exists but _data doesn't
        assert not (site_dir / "_data").exists()

        builder._export_library_json([], [])

        assert (site_dir / "_data").exists()
        assert (site_dir / "_data" / "library.json").exists()
