"""Tests for content exporter module."""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch


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
