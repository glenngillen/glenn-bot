"""
Data models for the browsable library.

This module defines:
- ContentType: Enum of all supported content types
- LibraryItem: Unified data structure for all knowledge base items
- Theme: AI-generated thematic category
- ThemeAssignment: Item-to-theme mapping with confidence score
"""

from enum import Enum


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


# Placeholder - implementations will be added following TDD
LibraryItem = None
Theme = None
ThemeAssignment = None
