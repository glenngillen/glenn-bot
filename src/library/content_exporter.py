"""
Content exporter for the browsable library.

This module provides the ContentExporter class which transforms ChromaDB
documents into LibraryItem instances suitable for display in the static
website library.
"""

from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

from src.library.models import ContentType, LibraryItem

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

    def _generate_summary(self, content: Optional[str]) -> str:
        """Generate a display summary from content.

        Truncates content to 200 characters with ellipsis if necessary.
        Leading and trailing whitespace is stripped before truncation.

        Args:
            content: Document content text, may be None.

        Returns:
            A summary string, truncated to 200 characters with "..."
            if the content exceeds that length. Returns empty string
            for None, empty, or whitespace-only content.
        """
        if not content:
            return ""

        stripped = content.strip()
        if not stripped:
            return ""

        # Truncate at 200 characters with ellipsis if needed
        if len(stripped) <= 200:
            return stripped

        return stripped[:200] + "..."

    def _extract_highlights(
        self, metadata: Optional[dict[str, Any]], content_type: ContentType
    ) -> list[str]:
        """Extract highlight strings from metadata based on content type.

        For VALUE and INSIGHT types, extracts the 'key_points' field.
        For FRAMEWORK types, extracts the 'steps' field.
        Other content types return an empty list.

        Args:
            metadata: Document metadata dict, may contain 'key_points' or 'steps'.
            content_type: The ContentType of the document.

        Returns:
            A list of highlight strings. Returns empty list if metadata is
            None/empty, the relevant field doesn't exist, or the field is
            not a list.
        """
        if not metadata:
            return []

        # Determine which field to extract based on content type
        if content_type in (ContentType.VALUE, ContentType.INSIGHT):
            field_name = "key_points"
        elif content_type == ContentType.FRAMEWORK:
            field_name = "steps"
        else:
            return []

        # Get the field value
        items = metadata.get(field_name)

        # Must be a list
        if not isinstance(items, list):
            return []

        # Return empty list if the list is empty
        if not items:
            return []

        # Convert all items to strings
        return [str(item) for item in items]

    def _convert_document(self, document: dict[str, Any]) -> LibraryItem:
        """Convert a ChromaDB document to a LibraryItem.

        Transforms a raw document from the knowledge base into a LibraryItem
        instance with all required fields for display in the browsable library.

        Args:
            document: A dictionary with 'id', 'content', and 'metadata' keys
                from ChromaDB.

        Returns:
            A LibraryItem instance with all fields populated.
        """
        doc_id = document.get("id", "")
        content = document.get("content") or ""
        metadata = document.get("metadata") or {}

        # Infer content type from metadata
        type_string = metadata.get("type") if metadata else None
        content_type = self._infer_content_type(type_string)

        # Generate title and summary
        title = self._generate_title(metadata, content)
        summary = self._generate_summary(content)

        # Extract source URL from metadata
        source_url = metadata.get("source") if metadata else None

        # Extract highlights based on content type
        highlights = self._extract_highlights(metadata, content_type)

        return LibraryItem(
            id=doc_id,
            content_type=content_type,
            title=title,
            summary=summary,
            full_content=content,
            source_url=source_url,
            cover_image_url=None,  # Resolved later by CoverResolver
            metadata=metadata,
            themes=[],  # Assigned later by ThemeGenerator
            created_at=datetime.now(),
            highlights=highlights,
        )
