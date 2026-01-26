"""
Content exporter for the browsable library.

This module provides the ContentExporter class which transforms ChromaDB
documents into LibraryItem instances suitable for display in the static
website library.
"""

from typing import TYPE_CHECKING, Optional

from src.library.models import ContentType

if TYPE_CHECKING:
    from src.knowledge_base import KnowledgeBase


# Mapping from ChromaDB type strings to ContentType enum values
_TYPE_MAPPING: dict[str, ContentType] = {
    "value": ContentType.VALUE,
    "framework": ContentType.FRAMEWORK,
    "web": ContentType.WEB_CONTENT,
    "preference": ContentType.PREFERENCE,
    "memory": ContentType.MEMORY,
    "book": ContentType.BOOK,
    "article": ContentType.ARTICLE,
    "insight": ContentType.INSIGHT,
    "goal": ContentType.GOAL,
    "skill": ContentType.SKILL,
}


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

    def _infer_content_type(self, type_string: Optional[str]) -> ContentType:
        """Infer the ContentType from a ChromaDB type string.

        Args:
            type_string: The type string from ChromaDB metadata, or None.

        Returns:
            The corresponding ContentType enum value. Returns WEB_CONTENT
            for unknown types, None, or empty strings.
        """
        if not type_string:
            return ContentType.WEB_CONTENT

        normalized = type_string.lower()
        return _TYPE_MAPPING.get(normalized, ContentType.WEB_CONTENT)

    def _generate_title(
        self, metadata: Optional[dict], content: Optional[str]
    ) -> str:
        """Generate a display title from metadata or content.

        Uses the 'name' field from metadata if available and non-empty.
        Otherwise, generates a title from the first 50 characters of
        the content, with ellipsis if truncated.

        Args:
            metadata: Document metadata dict, may contain 'name' field.
            content: Document content text.

        Returns:
            A title string. Returns 'Untitled' if no valid title can be
            generated from either metadata or content.
        """
        # Try to use name from metadata first
        if metadata:
            name = metadata.get("name")
            if name and str(name).strip():
                return str(name)

        # Fall back to generating from content
        if not content or not content.strip():
            return "Untitled"

        # Strip leading/trailing whitespace
        stripped = content.strip()

        # Truncate at 50 characters with ellipsis if needed
        if len(stripped) <= 50:
            return stripped

        return stripped[:50] + "..."
