"""Tests for content exporter module."""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch

from src.library.models import ContentType


class TestContentExporterInitialization:
    """Tests for ContentExporter initialization with KnowledgeBase."""

    def test_content_exporter_accepts_knowledge_base(self, mock_knowledge_base):
        """ContentExporter should accept a KnowledgeBase instance."""
        from src.library.content_exporter import ContentExporter

        exporter = ContentExporter(mock_knowledge_base)

        assert exporter.knowledge_base == mock_knowledge_base

    def test_content_exporter_stores_knowledge_base_reference(self, mock_knowledge_base):
        """ContentExporter should store a reference to the KnowledgeBase."""
        from src.library.content_exporter import ContentExporter

        exporter = ContentExporter(mock_knowledge_base)

        assert exporter.knowledge_base is mock_knowledge_base

    def test_content_exporter_can_be_instantiated(self, mock_knowledge_base):
        """ContentExporter should be instantiable with a KnowledgeBase."""
        from src.library.content_exporter import ContentExporter

        exporter = ContentExporter(mock_knowledge_base)

        assert exporter is not None


class TestInferContentType:
    """Tests for _infer_content_type() method mapping ChromaDB types to ContentType."""

    def test_infer_content_type_value_maps_to_value(self, mock_knowledge_base):
        """ChromaDB 'value' type should map to ContentType.VALUE."""
        from src.library.content_exporter import ContentExporter

        exporter = ContentExporter(mock_knowledge_base)

        result = exporter._infer_content_type("value")

        assert result == ContentType.VALUE

    def test_infer_content_type_framework_maps_to_framework(self, mock_knowledge_base):
        """ChromaDB 'framework' type should map to ContentType.FRAMEWORK."""
        from src.library.content_exporter import ContentExporter

        exporter = ContentExporter(mock_knowledge_base)

        result = exporter._infer_content_type("framework")

        assert result == ContentType.FRAMEWORK

    def test_infer_content_type_web_maps_to_web_content(self, mock_knowledge_base):
        """ChromaDB 'web' type should map to ContentType.WEB_CONTENT."""
        from src.library.content_exporter import ContentExporter

        exporter = ContentExporter(mock_knowledge_base)

        result = exporter._infer_content_type("web")

        assert result == ContentType.WEB_CONTENT

    def test_infer_content_type_preference_maps_to_preference(self, mock_knowledge_base):
        """ChromaDB 'preference' type should map to ContentType.PREFERENCE."""
        from src.library.content_exporter import ContentExporter

        exporter = ContentExporter(mock_knowledge_base)

        result = exporter._infer_content_type("preference")

        assert result == ContentType.PREFERENCE

    def test_infer_content_type_memory_maps_to_memory(self, mock_knowledge_base):
        """ChromaDB 'memory' type should map to ContentType.MEMORY."""
        from src.library.content_exporter import ContentExporter

        exporter = ContentExporter(mock_knowledge_base)

        result = exporter._infer_content_type("memory")

        assert result == ContentType.MEMORY

    def test_infer_content_type_book_maps_to_book(self, mock_knowledge_base):
        """ChromaDB 'book' type should map to ContentType.BOOK."""
        from src.library.content_exporter import ContentExporter

        exporter = ContentExporter(mock_knowledge_base)

        result = exporter._infer_content_type("book")

        assert result == ContentType.BOOK

    def test_infer_content_type_article_maps_to_article(self, mock_knowledge_base):
        """ChromaDB 'article' type should map to ContentType.ARTICLE."""
        from src.library.content_exporter import ContentExporter

        exporter = ContentExporter(mock_knowledge_base)

        result = exporter._infer_content_type("article")

        assert result == ContentType.ARTICLE

    def test_infer_content_type_insight_maps_to_insight(self, mock_knowledge_base):
        """ChromaDB 'insight' type should map to ContentType.INSIGHT."""
        from src.library.content_exporter import ContentExporter

        exporter = ContentExporter(mock_knowledge_base)

        result = exporter._infer_content_type("insight")

        assert result == ContentType.INSIGHT

    def test_infer_content_type_goal_maps_to_goal(self, mock_knowledge_base):
        """ChromaDB 'goal' type should map to ContentType.GOAL."""
        from src.library.content_exporter import ContentExporter

        exporter = ContentExporter(mock_knowledge_base)

        result = exporter._infer_content_type("goal")

        assert result == ContentType.GOAL

    def test_infer_content_type_skill_maps_to_skill(self, mock_knowledge_base):
        """ChromaDB 'skill' type should map to ContentType.SKILL."""
        from src.library.content_exporter import ContentExporter

        exporter = ContentExporter(mock_knowledge_base)

        result = exporter._infer_content_type("skill")

        assert result == ContentType.SKILL

    def test_infer_content_type_unknown_defaults_to_web_content(self, mock_knowledge_base):
        """Unknown ChromaDB types should default to ContentType.WEB_CONTENT."""
        from src.library.content_exporter import ContentExporter

        exporter = ContentExporter(mock_knowledge_base)

        result = exporter._infer_content_type("unknown_type")

        assert result == ContentType.WEB_CONTENT

    def test_infer_content_type_handles_case_insensitive(self, mock_knowledge_base):
        """Content type inference should handle case variations."""
        from src.library.content_exporter import ContentExporter

        exporter = ContentExporter(mock_knowledge_base)

        # Should work with uppercase
        assert exporter._infer_content_type("VALUE") == ContentType.VALUE
        # Should work with mixed case
        assert exporter._infer_content_type("Framework") == ContentType.FRAMEWORK

    def test_infer_content_type_handles_none_input(self, mock_knowledge_base):
        """Content type inference should handle None input gracefully."""
        from src.library.content_exporter import ContentExporter

        exporter = ContentExporter(mock_knowledge_base)

        result = exporter._infer_content_type(None)

        assert result == ContentType.WEB_CONTENT

    def test_infer_content_type_handles_empty_string(self, mock_knowledge_base):
        """Content type inference should handle empty string input."""
        from src.library.content_exporter import ContentExporter

        exporter = ContentExporter(mock_knowledge_base)

        result = exporter._infer_content_type("")

        assert result == ContentType.WEB_CONTENT
