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
