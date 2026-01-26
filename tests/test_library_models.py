"""Tests for library data models."""

import pytest
from datetime import datetime


class TestContentType:
    """Tests for the ContentType enum."""

    def test_content_type_has_book(self):
        """ContentType should have a BOOK value."""
        from src.library.models import ContentType
        assert hasattr(ContentType, 'BOOK')
        assert ContentType.BOOK.value == "book"

    def test_content_type_has_article(self):
        """ContentType should have an ARTICLE value."""
        from src.library.models import ContentType
        assert hasattr(ContentType, 'ARTICLE')
        assert ContentType.ARTICLE.value == "article"

    def test_content_type_has_framework(self):
        """ContentType should have a FRAMEWORK value."""
        from src.library.models import ContentType
        assert hasattr(ContentType, 'FRAMEWORK')
        assert ContentType.FRAMEWORK.value == "framework"

    def test_content_type_has_value(self):
        """ContentType should have a VALUE value."""
        from src.library.models import ContentType
        assert hasattr(ContentType, 'VALUE')
        assert ContentType.VALUE.value == "value"

    def test_content_type_has_preference(self):
        """ContentType should have a PREFERENCE value."""
        from src.library.models import ContentType
        assert hasattr(ContentType, 'PREFERENCE')
        assert ContentType.PREFERENCE.value == "preference"

    def test_content_type_has_memory(self):
        """ContentType should have a MEMORY value."""
        from src.library.models import ContentType
        assert hasattr(ContentType, 'MEMORY')
        assert ContentType.MEMORY.value == "memory"

    def test_content_type_has_insight(self):
        """ContentType should have an INSIGHT value."""
        from src.library.models import ContentType
        assert hasattr(ContentType, 'INSIGHT')
        assert ContentType.INSIGHT.value == "insight"

    def test_content_type_has_goal(self):
        """ContentType should have a GOAL value."""
        from src.library.models import ContentType
        assert hasattr(ContentType, 'GOAL')
        assert ContentType.GOAL.value == "goal"

    def test_content_type_has_skill(self):
        """ContentType should have a SKILL value."""
        from src.library.models import ContentType
        assert hasattr(ContentType, 'SKILL')
        assert ContentType.SKILL.value == "skill"

    def test_content_type_has_web_content(self):
        """ContentType should have a WEB_CONTENT value."""
        from src.library.models import ContentType
        assert hasattr(ContentType, 'WEB_CONTENT')
        assert ContentType.WEB_CONTENT.value == "web_content"

    def test_content_type_has_exactly_ten_types(self):
        """ContentType should have exactly 10 content types."""
        from src.library.models import ContentType
        assert len(ContentType) == 10

    def test_content_type_values_are_strings(self):
        """All ContentType values should be strings."""
        from src.library.models import ContentType
        for content_type in ContentType:
            assert isinstance(content_type.value, str)

    def test_content_type_can_be_created_from_string(self):
        """ContentType should be creatable from string value."""
        from src.library.models import ContentType
        assert ContentType("book") == ContentType.BOOK
        assert ContentType("framework") == ContentType.FRAMEWORK
        assert ContentType("web_content") == ContentType.WEB_CONTENT


class TestLibraryItem:
    """Tests for the LibraryItem dataclass."""

    def test_library_item_has_required_fields(self):
        """LibraryItem should have all required fields."""
        from src.library.models import LibraryItem, ContentType

        item = LibraryItem(
            id="test-123",
            content_type=ContentType.BOOK,
            title="Test Book",
            summary="A test book summary",
            full_content="Full content of the test book goes here.",
            source_url="https://example.com/book",
            cover_image_url="https://example.com/cover.jpg",
            metadata={"author": "Test Author"},
            themes=["theme-1", "theme-2"],
            created_at=datetime(2024, 1, 15, 10, 30, 0),
            highlights=["Key insight 1", "Key insight 2"],
        )

        assert item.id == "test-123"
        assert item.content_type == ContentType.BOOK
        assert item.title == "Test Book"
        assert item.summary == "A test book summary"
        assert item.full_content == "Full content of the test book goes here."
        assert item.source_url == "https://example.com/book"
        assert item.cover_image_url == "https://example.com/cover.jpg"
        assert item.metadata == {"author": "Test Author"}
        assert item.themes == ["theme-1", "theme-2"]
        assert item.created_at == datetime(2024, 1, 15, 10, 30, 0)
        assert item.highlights == ["Key insight 1", "Key insight 2"]

    def test_library_item_source_url_is_optional(self):
        """LibraryItem should allow None for source_url."""
        from src.library.models import LibraryItem, ContentType

        item = LibraryItem(
            id="test-123",
            content_type=ContentType.MEMORY,
            title="Test Memory",
            summary="A memory without a source",
            full_content="Full content here.",
            source_url=None,
            cover_image_url=None,
            metadata={},
            themes=[],
            created_at=datetime(2024, 1, 15, 10, 30, 0),
            highlights=[],
        )

        assert item.source_url is None

    def test_library_item_cover_image_url_is_optional(self):
        """LibraryItem should allow None for cover_image_url."""
        from src.library.models import LibraryItem, ContentType

        item = LibraryItem(
            id="test-123",
            content_type=ContentType.INSIGHT,
            title="Test Insight",
            summary="An insight without a cover",
            full_content="Full content here.",
            source_url=None,
            cover_image_url=None,
            metadata={},
            themes=[],
            created_at=datetime(2024, 1, 15, 10, 30, 0),
            highlights=[],
        )

        assert item.cover_image_url is None

    def test_library_item_themes_can_be_empty(self):
        """LibraryItem should allow an empty themes list."""
        from src.library.models import LibraryItem, ContentType

        item = LibraryItem(
            id="test-123",
            content_type=ContentType.VALUE,
            title="Test Value",
            summary="A value with no themes",
            full_content="Full content here.",
            source_url=None,
            cover_image_url=None,
            metadata={},
            themes=[],
            created_at=datetime(2024, 1, 15, 10, 30, 0),
            highlights=[],
        )

        assert item.themes == []

    def test_library_item_can_have_multiple_themes(self):
        """LibraryItem should support multiple themes."""
        from src.library.models import LibraryItem, ContentType

        item = LibraryItem(
            id="test-123",
            content_type=ContentType.FRAMEWORK,
            title="Test Framework",
            summary="A framework with multiple themes",
            full_content="Full content here.",
            source_url=None,
            cover_image_url=None,
            metadata={},
            themes=["theme-1", "theme-2", "theme-3"],
            created_at=datetime(2024, 1, 15, 10, 30, 0),
            highlights=[],
        )

        assert len(item.themes) == 3
        assert "theme-1" in item.themes
        assert "theme-2" in item.themes
        assert "theme-3" in item.themes

    def test_library_item_highlights_can_be_empty(self):
        """LibraryItem should allow an empty highlights list."""
        from src.library.models import LibraryItem, ContentType

        item = LibraryItem(
            id="test-123",
            content_type=ContentType.PREFERENCE,
            title="Test Preference",
            summary="A preference without highlights",
            full_content="Full content here.",
            source_url=None,
            cover_image_url=None,
            metadata={},
            themes=[],
            created_at=datetime(2024, 1, 15, 10, 30, 0),
            highlights=[],
        )

        assert item.highlights == []

    def test_library_item_metadata_can_contain_various_types(self):
        """LibraryItem metadata should accept various value types."""
        from src.library.models import LibraryItem, ContentType

        item = LibraryItem(
            id="test-123",
            content_type=ContentType.BOOK,
            title="Test Book",
            summary="A book with rich metadata",
            full_content="Full content here.",
            source_url="https://example.com",
            cover_image_url=None,
            metadata={
                "author": "Test Author",
                "year": 2024,
                "tags": ["programming", "python"],
                "is_favorite": True,
            },
            themes=[],
            created_at=datetime(2024, 1, 15, 10, 30, 0),
            highlights=[],
        )

        assert item.metadata["author"] == "Test Author"
        assert item.metadata["year"] == 2024
        assert item.metadata["tags"] == ["programming", "python"]
        assert item.metadata["is_favorite"] is True

    def test_library_item_content_type_accepts_enum_value(self):
        """LibraryItem content_type should accept ContentType enum values."""
        from src.library.models import LibraryItem, ContentType

        for content_type in ContentType:
            item = LibraryItem(
                id="test-123",
                content_type=content_type,
                title="Test Item",
                summary="Test summary",
                full_content="Full content here.",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime(2024, 1, 15, 10, 30, 0),
                highlights=[],
            )
            assert item.content_type == content_type

    def test_library_item_is_dataclass(self):
        """LibraryItem should be a dataclass."""
        from dataclasses import is_dataclass
        from src.library.models import LibraryItem

        assert is_dataclass(LibraryItem)
