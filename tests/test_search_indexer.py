"""Tests for search indexer module.

This module tests the SearchIndexer class which builds a pre-built search index
for client-side search functionality. The index is generated at build time
and used by Fuse.js for fuzzy search on the static site.

Test classes are organized by the methods they test:
- TestSearchIndexerInitialization: SearchIndexer class initialization
- TestExtractKeywords: _extract_keywords() from themes and content
- TestBuildSearchItem: _build_search_item() creating index entry
- TestBuildIndex: build_index() returning search items
- TestWriteIndex: write_index() outputting search-index.json
"""

import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock
import json

from src.library.models import LibraryItem, ContentType, Theme, ThemeAssignment


class TestSearchIndexerInitialization:
    """Tests for SearchIndexer class initialization."""

    def test_search_indexer_can_be_imported(self):
        """SearchIndexer class should be importable from the module."""
        from src.library.search_indexer import SearchIndexer

        assert SearchIndexer is not None

    def test_search_indexer_init_with_output_dir(self, temp_dir):
        """SearchIndexer should accept an output_dir path."""
        from src.library.search_indexer import SearchIndexer

        output_dir = temp_dir / "library-site"
        output_dir.mkdir(parents=True, exist_ok=True)

        indexer = SearchIndexer(output_dir=output_dir)

        assert indexer.output_dir == output_dir

    def test_search_indexer_init_creates_output_dir_if_not_exists(self, temp_dir):
        """SearchIndexer should create the output_dir if it doesn't exist."""
        from src.library.search_indexer import SearchIndexer

        output_dir = temp_dir / "library-site" / "subdir"
        assert not output_dir.exists()

        indexer = SearchIndexer(output_dir=output_dir)

        assert output_dir.exists()

    def test_search_indexer_init_output_dir_as_string(self, temp_dir):
        """SearchIndexer should accept output_dir as a string and convert to Path."""
        from src.library.search_indexer import SearchIndexer

        output_dir = str(temp_dir / "library-site")

        indexer = SearchIndexer(output_dir=output_dir)

        assert isinstance(indexer.output_dir, Path)
        assert indexer.output_dir == Path(output_dir)

    def test_search_indexer_has_index_file_path(self, temp_dir):
        """SearchIndexer should have an index_file path attribute."""
        from src.library.search_indexer import SearchIndexer

        output_dir = temp_dir / "library-site"
        output_dir.mkdir(parents=True, exist_ok=True)

        indexer = SearchIndexer(output_dir=output_dir)

        # Index file should be in assets/js/search-index.json
        assert indexer.index_file == output_dir / "assets" / "js" / "search-index.json"


class TestExtractKeywords:
    """Tests for _extract_keywords() method that extracts keywords from themes and content."""

    def test_extract_keywords_from_theme_keywords(self, temp_dir):
        """_extract_keywords() should include keywords from assigned themes."""
        from src.library.search_indexer import SearchIndexer

        output_dir = temp_dir / "library-site"
        output_dir.mkdir(parents=True, exist_ok=True)
        indexer = SearchIndexer(output_dir=output_dir)

        # Create a theme with keywords
        themes = {
            "systems-thinking": Theme(
                id="systems-thinking",
                name="Systems Thinking",
                description="Understanding complex systems",
                keywords=["systems", "feedback", "complexity"],
                item_count=5,
                created_at=datetime(2024, 1, 15, 10, 0, 0),
                updated_at=datetime(2024, 1, 20, 14, 30, 0),
            )
        }

        # Item assigned to this theme
        item = LibraryItem(
            id="doc_123",
            content_type=ContentType.BOOK,
            title="Thinking in Systems",
            summary="A primer on systems thinking",
            full_content="Full content here...",
            source_url="https://example.com",
            cover_image_url=None,
            metadata={},
            themes=["systems-thinking"],
            created_at=datetime(2024, 1, 15, 10, 30, 0),
            highlights=[],
        )

        keywords = indexer._extract_keywords(item, themes)

        assert "systems" in keywords
        assert "feedback" in keywords
        assert "complexity" in keywords

    def test_extract_keywords_from_multiple_themes(self, temp_dir):
        """_extract_keywords() should combine keywords from all assigned themes."""
        from src.library.search_indexer import SearchIndexer

        output_dir = temp_dir / "library-site"
        output_dir.mkdir(parents=True, exist_ok=True)
        indexer = SearchIndexer(output_dir=output_dir)

        themes = {
            "systems-thinking": Theme(
                id="systems-thinking",
                name="Systems Thinking",
                description="Understanding complex systems",
                keywords=["systems", "feedback"],
                item_count=5,
                created_at=datetime(2024, 1, 15),
                updated_at=datetime(2024, 1, 20),
            ),
            "mental-models": Theme(
                id="mental-models",
                name="Mental Models",
                description="Thinking frameworks",
                keywords=["models", "frameworks"],
                item_count=3,
                created_at=datetime(2024, 1, 15),
                updated_at=datetime(2024, 1, 20),
            ),
        }

        item = LibraryItem(
            id="doc_123",
            content_type=ContentType.BOOK,
            title="Test Book",
            summary="A test book",
            full_content="Content...",
            source_url=None,
            cover_image_url=None,
            metadata={},
            themes=["systems-thinking", "mental-models"],
            created_at=datetime(2024, 1, 15),
            highlights=[],
        )

        keywords = indexer._extract_keywords(item, themes)

        # Should have keywords from both themes
        assert "systems" in keywords
        assert "feedback" in keywords
        assert "models" in keywords
        assert "frameworks" in keywords

    def test_extract_keywords_from_item_metadata_tags(self, temp_dir):
        """_extract_keywords() should include tags from item metadata if present."""
        from src.library.search_indexer import SearchIndexer

        output_dir = temp_dir / "library-site"
        output_dir.mkdir(parents=True, exist_ok=True)
        indexer = SearchIndexer(output_dir=output_dir)

        item = LibraryItem(
            id="doc_123",
            content_type=ContentType.ARTICLE,
            title="Test Article",
            summary="A test article",
            full_content="Content...",
            source_url=None,
            cover_image_url=None,
            metadata={"tags": ["programming", "python", "testing"]},
            themes=[],
            created_at=datetime(2024, 1, 15),
            highlights=[],
        )

        keywords = indexer._extract_keywords(item, {})

        assert "programming" in keywords
        assert "python" in keywords
        assert "testing" in keywords

    def test_extract_keywords_no_duplicates(self, temp_dir):
        """_extract_keywords() should not include duplicate keywords."""
        from src.library.search_indexer import SearchIndexer

        output_dir = temp_dir / "library-site"
        output_dir.mkdir(parents=True, exist_ok=True)
        indexer = SearchIndexer(output_dir=output_dir)

        themes = {
            "theme-1": Theme(
                id="theme-1",
                name="Theme 1",
                description="First theme",
                keywords=["shared", "unique1"],
                item_count=5,
                created_at=datetime(2024, 1, 15),
                updated_at=datetime(2024, 1, 20),
            ),
            "theme-2": Theme(
                id="theme-2",
                name="Theme 2",
                description="Second theme",
                keywords=["shared", "unique2"],
                item_count=3,
                created_at=datetime(2024, 1, 15),
                updated_at=datetime(2024, 1, 20),
            ),
        }

        item = LibraryItem(
            id="doc_123",
            content_type=ContentType.BOOK,
            title="Test Book",
            summary="A test book",
            full_content="Content...",
            source_url=None,
            cover_image_url=None,
            metadata={"tags": ["shared"]},  # Same keyword as themes
            themes=["theme-1", "theme-2"],
            created_at=datetime(2024, 1, 15),
            highlights=[],
        )

        keywords = indexer._extract_keywords(item, themes)

        # Count occurrences of "shared" - should only appear once
        assert keywords.count("shared") == 1

    def test_extract_keywords_empty_when_no_themes_or_tags(self, temp_dir):
        """_extract_keywords() should return empty list when no themes or tags."""
        from src.library.search_indexer import SearchIndexer

        output_dir = temp_dir / "library-site"
        output_dir.mkdir(parents=True, exist_ok=True)
        indexer = SearchIndexer(output_dir=output_dir)

        item = LibraryItem(
            id="doc_123",
            content_type=ContentType.MEMORY,
            title="Test Memory",
            summary="A test memory",
            full_content="Content...",
            source_url=None,
            cover_image_url=None,
            metadata={},
            themes=[],
            created_at=datetime(2024, 1, 15),
            highlights=[],
        )

        keywords = indexer._extract_keywords(item, {})

        assert keywords == []

    def test_extract_keywords_handles_missing_theme(self, temp_dir):
        """_extract_keywords() should gracefully handle themes not in themes dict."""
        from src.library.search_indexer import SearchIndexer

        output_dir = temp_dir / "library-site"
        output_dir.mkdir(parents=True, exist_ok=True)
        indexer = SearchIndexer(output_dir=output_dir)

        item = LibraryItem(
            id="doc_123",
            content_type=ContentType.BOOK,
            title="Test Book",
            summary="A test book",
            full_content="Content...",
            source_url=None,
            cover_image_url=None,
            metadata={},
            themes=["non-existent-theme"],  # Theme not in dict
            created_at=datetime(2024, 1, 15),
            highlights=[],
        )

        # Should not raise, should return empty list
        keywords = indexer._extract_keywords(item, {})

        assert keywords == []


class TestBuildSearchItem:
    """Tests for _build_search_item() method that creates a search index entry."""

    def test_build_search_item_basic_fields(self, temp_dir):
        """_build_search_item() should include id, title, summary, content_type, url."""
        from src.library.search_indexer import SearchIndexer

        output_dir = temp_dir / "library-site"
        output_dir.mkdir(parents=True, exist_ok=True)
        indexer = SearchIndexer(output_dir=output_dir)

        item = LibraryItem(
            id="doc_123",
            content_type=ContentType.BOOK,
            title="Thinking in Systems",
            summary="A primer on systems thinking that emphasizes feedback loops.",
            full_content="Full content here...",
            source_url="https://example.com",
            cover_image_url=None,
            metadata={},
            themes=[],
            created_at=datetime(2024, 1, 15, 10, 30, 0),
            highlights=[],
        )

        search_item = indexer._build_search_item(item, {})

        assert search_item["id"] == "doc_123"
        assert search_item["title"] == "Thinking in Systems"
        assert search_item["summary"] == "A primer on systems thinking that emphasizes feedback loops."
        assert search_item["content_type"] == "book"
        assert search_item["url"] == "/item/doc_123/"

    def test_build_search_item_includes_keywords(self, temp_dir):
        """_build_search_item() should include extracted keywords."""
        from src.library.search_indexer import SearchIndexer

        output_dir = temp_dir / "library-site"
        output_dir.mkdir(parents=True, exist_ok=True)
        indexer = SearchIndexer(output_dir=output_dir)

        themes = {
            "systems-thinking": Theme(
                id="systems-thinking",
                name="Systems Thinking",
                description="Understanding complex systems",
                keywords=["systems", "feedback", "complexity"],
                item_count=5,
                created_at=datetime(2024, 1, 15),
                updated_at=datetime(2024, 1, 20),
            )
        }

        item = LibraryItem(
            id="doc_123",
            content_type=ContentType.BOOK,
            title="Thinking in Systems",
            summary="A primer on systems thinking",
            full_content="Full content...",
            source_url=None,
            cover_image_url=None,
            metadata={},
            themes=["systems-thinking"],
            created_at=datetime(2024, 1, 15),
            highlights=[],
        )

        search_item = indexer._build_search_item(item, themes)

        assert "keywords" in search_item
        assert "systems" in search_item["keywords"]
        assert "feedback" in search_item["keywords"]
        assert "complexity" in search_item["keywords"]

    def test_build_search_item_includes_theme_names(self, temp_dir):
        """_build_search_item() should include theme names (not IDs)."""
        from src.library.search_indexer import SearchIndexer

        output_dir = temp_dir / "library-site"
        output_dir.mkdir(parents=True, exist_ok=True)
        indexer = SearchIndexer(output_dir=output_dir)

        themes = {
            "systems-thinking": Theme(
                id="systems-thinking",
                name="Systems Thinking",
                description="Understanding complex systems",
                keywords=["systems"],
                item_count=5,
                created_at=datetime(2024, 1, 15),
                updated_at=datetime(2024, 1, 20),
            ),
            "mental-models": Theme(
                id="mental-models",
                name="Mental Models",
                description="Thinking frameworks",
                keywords=["models"],
                item_count=3,
                created_at=datetime(2024, 1, 15),
                updated_at=datetime(2024, 1, 20),
            ),
        }

        item = LibraryItem(
            id="doc_123",
            content_type=ContentType.BOOK,
            title="Test Book",
            summary="A test book",
            full_content="Content...",
            source_url=None,
            cover_image_url=None,
            metadata={},
            themes=["systems-thinking", "mental-models"],
            created_at=datetime(2024, 1, 15),
            highlights=[],
        )

        search_item = indexer._build_search_item(item, themes)

        assert "themes" in search_item
        assert "Systems Thinking" in search_item["themes"]
        assert "Mental Models" in search_item["themes"]

    def test_build_search_item_handles_special_characters_in_id(self, temp_dir):
        """_build_search_item() should properly encode special characters in URL."""
        from src.library.search_indexer import SearchIndexer

        output_dir = temp_dir / "library-site"
        output_dir.mkdir(parents=True, exist_ok=True)
        indexer = SearchIndexer(output_dir=output_dir)

        item = LibraryItem(
            id="doc-with-special_chars",
            content_type=ContentType.ARTICLE,
            title="Test Article",
            summary="A test article",
            full_content="Content...",
            source_url=None,
            cover_image_url=None,
            metadata={},
            themes=[],
            created_at=datetime(2024, 1, 15),
            highlights=[],
        )

        search_item = indexer._build_search_item(item, {})

        # URL should be constructed correctly
        assert search_item["url"] == "/item/doc-with-special_chars/"

    def test_build_search_item_empty_themes_list(self, temp_dir):
        """_build_search_item() should return empty themes list when item has no themes."""
        from src.library.search_indexer import SearchIndexer

        output_dir = temp_dir / "library-site"
        output_dir.mkdir(parents=True, exist_ok=True)
        indexer = SearchIndexer(output_dir=output_dir)

        item = LibraryItem(
            id="doc_123",
            content_type=ContentType.MEMORY,
            title="Test Memory",
            summary="A test memory",
            full_content="Content...",
            source_url=None,
            cover_image_url=None,
            metadata={},
            themes=[],
            created_at=datetime(2024, 1, 15),
            highlights=[],
        )

        search_item = indexer._build_search_item(item, {})

        assert search_item["themes"] == []


class TestBuildIndex:
    """Tests for build_index() method that creates the full search index."""

    def test_build_index_returns_list_of_search_items(self, temp_dir):
        """build_index() should return a list of search item dictionaries."""
        from src.library.search_indexer import SearchIndexer

        output_dir = temp_dir / "library-site"
        output_dir.mkdir(parents=True, exist_ok=True)
        indexer = SearchIndexer(output_dir=output_dir)

        items = [
            LibraryItem(
                id="doc_1",
                content_type=ContentType.BOOK,
                title="Book 1",
                summary="First book",
                full_content="Content...",
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
                summary="First article",
                full_content="Content...",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime(2024, 1, 16),
                highlights=[],
            ),
        ]

        index = indexer.build_index(items, {})

        assert isinstance(index, list)
        assert len(index) == 2
        assert all(isinstance(item, dict) for item in index)

    def test_build_index_with_themes(self, temp_dir):
        """build_index() should incorporate theme data into index items."""
        from src.library.search_indexer import SearchIndexer

        output_dir = temp_dir / "library-site"
        output_dir.mkdir(parents=True, exist_ok=True)
        indexer = SearchIndexer(output_dir=output_dir)

        themes = {
            "systems-thinking": Theme(
                id="systems-thinking",
                name="Systems Thinking",
                description="Understanding complex systems",
                keywords=["systems", "feedback"],
                item_count=1,
                created_at=datetime(2024, 1, 15),
                updated_at=datetime(2024, 1, 20),
            )
        }

        items = [
            LibraryItem(
                id="doc_1",
                content_type=ContentType.BOOK,
                title="Systems Book",
                summary="A systems book",
                full_content="Content...",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=["systems-thinking"],
                created_at=datetime(2024, 1, 15),
                highlights=[],
            )
        ]

        index = indexer.build_index(items, themes)

        assert len(index) == 1
        assert index[0]["themes"] == ["Systems Thinking"]
        assert "systems" in index[0]["keywords"]

    def test_build_index_empty_items(self, temp_dir):
        """build_index() should return empty list for empty items."""
        from src.library.search_indexer import SearchIndexer

        output_dir = temp_dir / "library-site"
        output_dir.mkdir(parents=True, exist_ok=True)
        indexer = SearchIndexer(output_dir=output_dir)

        index = indexer.build_index([], {})

        assert index == []

    def test_build_index_preserves_item_order(self, temp_dir):
        """build_index() should preserve the order of items."""
        from src.library.search_indexer import SearchIndexer

        output_dir = temp_dir / "library-site"
        output_dir.mkdir(parents=True, exist_ok=True)
        indexer = SearchIndexer(output_dir=output_dir)

        items = [
            LibraryItem(
                id=f"doc_{i}",
                content_type=ContentType.BOOK,
                title=f"Book {i}",
                summary=f"Book number {i}",
                full_content="Content...",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime(2024, 1, i + 1),
                highlights=[],
            )
            for i in range(5)
        ]

        index = indexer.build_index(items, {})

        for i, search_item in enumerate(index):
            assert search_item["id"] == f"doc_{i}"


class TestWriteIndex:
    """Tests for write_index() method that outputs search-index.json."""

    def test_write_index_creates_file(self, temp_dir):
        """write_index() should create search-index.json file."""
        from src.library.search_indexer import SearchIndexer

        output_dir = temp_dir / "library-site"
        output_dir.mkdir(parents=True, exist_ok=True)
        indexer = SearchIndexer(output_dir=output_dir)

        items = [
            LibraryItem(
                id="doc_1",
                content_type=ContentType.BOOK,
                title="Test Book",
                summary="A test book",
                full_content="Content...",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime(2024, 1, 15),
                highlights=[],
            )
        ]

        indexer.write_index(items, {})

        assert indexer.index_file.exists()

    def test_write_index_creates_parent_directories(self, temp_dir):
        """write_index() should create parent directories if they don't exist."""
        from src.library.search_indexer import SearchIndexer

        output_dir = temp_dir / "library-site"
        output_dir.mkdir(parents=True, exist_ok=True)
        indexer = SearchIndexer(output_dir=output_dir)

        # Ensure the assets/js directory doesn't exist
        assets_js = output_dir / "assets" / "js"
        assert not assets_js.exists()

        items = [
            LibraryItem(
                id="doc_1",
                content_type=ContentType.BOOK,
                title="Test Book",
                summary="A test book",
                full_content="Content...",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime(2024, 1, 15),
                highlights=[],
            )
        ]

        indexer.write_index(items, {})

        assert assets_js.exists()
        assert indexer.index_file.exists()

    def test_write_index_valid_json_format(self, temp_dir):
        """write_index() should output valid JSON with items array."""
        from src.library.search_indexer import SearchIndexer

        output_dir = temp_dir / "library-site"
        output_dir.mkdir(parents=True, exist_ok=True)
        indexer = SearchIndexer(output_dir=output_dir)

        items = [
            LibraryItem(
                id="doc_1",
                content_type=ContentType.BOOK,
                title="Test Book",
                summary="A test book",
                full_content="Content...",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime(2024, 1, 15),
                highlights=[],
            )
        ]

        indexer.write_index(items, {})

        # Read and parse the file
        with open(indexer.index_file) as f:
            data = json.load(f)

        assert "items" in data
        assert isinstance(data["items"], list)
        assert len(data["items"]) == 1

    def test_write_index_item_structure(self, temp_dir):
        """write_index() should write items with correct structure."""
        from src.library.search_indexer import SearchIndexer

        output_dir = temp_dir / "library-site"
        output_dir.mkdir(parents=True, exist_ok=True)
        indexer = SearchIndexer(output_dir=output_dir)

        themes = {
            "systems-thinking": Theme(
                id="systems-thinking",
                name="Systems Thinking",
                description="Understanding complex systems",
                keywords=["systems", "feedback"],
                item_count=1,
                created_at=datetime(2024, 1, 15),
                updated_at=datetime(2024, 1, 20),
            )
        }

        items = [
            LibraryItem(
                id="doc_123",
                content_type=ContentType.BOOK,
                title="Thinking in Systems",
                summary="A primer on systems thinking",
                full_content="Content...",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=["systems-thinking"],
                created_at=datetime(2024, 1, 15),
                highlights=[],
            )
        ]

        indexer.write_index(items, themes)

        with open(indexer.index_file) as f:
            data = json.load(f)

        item = data["items"][0]
        assert item["id"] == "doc_123"
        assert item["title"] == "Thinking in Systems"
        assert item["summary"] == "A primer on systems thinking"
        assert item["content_type"] == "book"
        assert item["keywords"] == ["systems", "feedback"]
        assert item["themes"] == ["Systems Thinking"]
        assert item["url"] == "/item/doc_123/"

    def test_write_index_overwrites_existing_file(self, temp_dir):
        """write_index() should overwrite existing search-index.json."""
        from src.library.search_indexer import SearchIndexer

        output_dir = temp_dir / "library-site"
        output_dir.mkdir(parents=True, exist_ok=True)
        indexer = SearchIndexer(output_dir=output_dir)

        # Create initial index
        items1 = [
            LibraryItem(
                id="doc_1",
                content_type=ContentType.BOOK,
                title="First Book",
                summary="First book summary",
                full_content="Content...",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime(2024, 1, 15),
                highlights=[],
            )
        ]
        indexer.write_index(items1, {})

        # Create second index
        items2 = [
            LibraryItem(
                id="doc_2",
                content_type=ContentType.ARTICLE,
                title="Second Article",
                summary="Second article summary",
                full_content="Content...",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime(2024, 1, 16),
                highlights=[],
            )
        ]
        indexer.write_index(items2, {})

        # Read the file
        with open(indexer.index_file) as f:
            data = json.load(f)

        # Should only have the second item
        assert len(data["items"]) == 1
        assert data["items"][0]["id"] == "doc_2"

    def test_write_index_empty_items(self, temp_dir):
        """write_index() should create valid JSON even with empty items."""
        from src.library.search_indexer import SearchIndexer

        output_dir = temp_dir / "library-site"
        output_dir.mkdir(parents=True, exist_ok=True)
        indexer = SearchIndexer(output_dir=output_dir)

        indexer.write_index([], {})

        with open(indexer.index_file) as f:
            data = json.load(f)

        assert data == {"items": []}

    def test_write_index_returns_index_file_path(self, temp_dir):
        """write_index() should return the path to the written file."""
        from src.library.search_indexer import SearchIndexer

        output_dir = temp_dir / "library-site"
        output_dir.mkdir(parents=True, exist_ok=True)
        indexer = SearchIndexer(output_dir=output_dir)

        items = [
            LibraryItem(
                id="doc_1",
                content_type=ContentType.BOOK,
                title="Test Book",
                summary="A test book",
                full_content="Content...",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime(2024, 1, 15),
                highlights=[],
            )
        ]

        result = indexer.write_index(items, {})

        assert result == indexer.index_file
