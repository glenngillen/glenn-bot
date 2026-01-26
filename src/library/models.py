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


# Placeholder - implementations will be added following TDD
Theme = None
ThemeAssignment = None
