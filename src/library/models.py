"""
Data models for the browsable library.

This module defines:
- ContentType: Enum of all supported content types
- LibraryItem: Unified data structure for all knowledge base items
- Theme: AI-generated thematic category
- ThemeAssignment: Item-to-theme mapping with confidence score
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class ContentType(Enum):
    """Enum of supported content types for library items."""

    BOOK = "book"
    ARTICLE = "article"
    FRAMEWORK = "framework"
    VALUE = "value"
    PREFERENCE = "preference"
    MEMORY = "memory"
    INSIGHT = "insight"
    GOAL = "goal"
    SKILL = "skill"
    WEB_CONTENT = "web_content"


@dataclass
class LibraryItem:
    """Unified data structure for all knowledge base items.

    Represents a single item in the browsable library, supporting
    all content types with sufficient metadata for display, grouping,
    and search.
    """

    id: str
    content_type: ContentType
    title: str
    summary: str
    full_content: str
    source_url: Optional[str]
    cover_image_url: Optional[str]
    metadata: dict[str, Any]
    themes: list[str]
    created_at: datetime
    highlights: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert the LibraryItem to a dictionary for serialization.

        Returns:
            A dictionary with all fields serialized. ContentType is converted
            to its string value, and datetime is converted to ISO format.
        """
        return {
            "id": self.id,
            "content_type": self.content_type.value,
            "title": self.title,
            "summary": self.summary,
            "full_content": self.full_content,
            "source_url": self.source_url,
            "cover_image_url": self.cover_image_url,
            "metadata": self.metadata,
            "themes": self.themes,
            "created_at": self.created_at.isoformat(),
            "highlights": self.highlights,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LibraryItem":
        """Create a LibraryItem from a dictionary.

        Args:
            data: A dictionary with LibraryItem fields. The content_type
                should be a string value, and created_at should be an
                ISO format datetime string.

        Returns:
            A new LibraryItem instance.
        """
        return cls(
            id=data["id"],
            content_type=ContentType(data["content_type"]),
            title=data["title"],
            summary=data["summary"],
            full_content=data["full_content"],
            source_url=data["source_url"],
            cover_image_url=data["cover_image_url"],
            metadata=data["metadata"],
            themes=data["themes"],
            created_at=datetime.fromisoformat(data["created_at"]),
            highlights=data["highlights"],
        )


# Placeholder - implementations will be added following TDD
Theme = None
ThemeAssignment = None
