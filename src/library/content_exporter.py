"""
Content exporter for the browsable library.

This module provides the ContentExporter class which transforms ChromaDB
documents into LibraryItem instances suitable for display in the static
website library.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.knowledge_base import KnowledgeBase


class ContentExporter:
    """Exports knowledge base content to LibraryItem format.

    ContentExporter reads documents from ChromaDB via the KnowledgeBase
    class and transforms them into LibraryItem instances with all
    required fields for display in the browsable library.
    """

    def __init__(self, knowledge_base: "KnowledgeBase") -> None:
        """Initialize the ContentExporter with a KnowledgeBase.

        Args:
            knowledge_base: The KnowledgeBase instance to export content from.
        """
        self.knowledge_base = knowledge_base
