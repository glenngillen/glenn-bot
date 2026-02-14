"""
Search index generation for the browsable library.

This module provides the SearchIndexer class which builds a pre-built search index
for client-side search functionality. The index is generated at build time and
used by Fuse.js for fuzzy search on the static site.

Index Fields & Weights (for Fuse.js configuration):
- title: 1.0 (highest priority)
- keywords: 0.8 (theme keywords + item tags)
- summary: 0.6 (card summary text)
- themes: 0.4 (theme names)
- content_type: 0.2 (allow filtering by type)
"""

import json
from pathlib import Path
from typing import Any

from src.library.models import LibraryItem, Theme


class SearchIndexer:
    """Builds a search index for client-side search functionality.

    The SearchIndexer creates a JSON index file containing all library items
    with their searchable fields (title, summary, keywords, themes, content_type).
    This index is used by Fuse.js on the client side for fuzzy search.

    Attributes:
        output_dir: Path to the library site output directory.
        index_file: Path to the search-index.json file.
    """

    def __init__(self, output_dir: str | Path) -> None:
        """Initialize the SearchIndexer with an output directory.

        Args:
            output_dir: Path to the library site output directory. Will be
                created if it doesn't exist. Can be a string or Path.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.output_dir / "assets" / "js" / "search-index.json"

    def _extract_keywords(
        self, item: LibraryItem, themes: dict[str, Theme]
    ) -> list[str]:
        """Extract keywords from themes and item metadata.

        Combines keywords from all themes the item belongs to, plus any
        tags from the item's metadata. Removes duplicates while preserving
        order.

        Args:
            item: The library item to extract keywords for.
            themes: Dictionary mapping theme IDs to Theme objects.

        Returns:
            List of unique keywords extracted from themes and item tags.
        """
        keywords: list[str] = []
        seen: set[str] = set()

        # Extract keywords from assigned themes
        for theme_id in item.themes:
            if theme_id in themes:
                theme = themes[theme_id]
                for keyword in theme.keywords:
                    if keyword not in seen:
                        keywords.append(keyword)
                        seen.add(keyword)

        # Extract tags from item metadata if present
        tags = item.metadata.get("tags", [])
        if isinstance(tags, list):
            for tag in tags:
                if tag not in seen:
                    keywords.append(tag)
                    seen.add(tag)

        return keywords

    def _build_search_item(
        self, item: LibraryItem, themes: dict[str, Theme]
    ) -> dict[str, Any]:
        """Build a search index entry for a library item.

        Creates a dictionary with all the fields needed for client-side search:
        id, title, summary, content_type, url, keywords, and theme names.

        Args:
            item: The library item to create an index entry for.
            themes: Dictionary mapping theme IDs to Theme objects.

        Returns:
            Dictionary with search index fields for the item.
        """
        # Get theme names (not IDs) for display
        theme_names: list[str] = []
        for theme_id in item.themes:
            if theme_id in themes:
                theme_names.append(themes[theme_id].name)

        return {
            "id": item.id,
            "title": item.title,
            "summary": item.summary,
            "content_type": item.content_type.value,
            "url": f"/item/{item.id}/",
            "keywords": self._extract_keywords(item, themes),
            "themes": theme_names,
        }

    def build_index(
        self, items: list[LibraryItem], themes: dict[str, Theme]
    ) -> list[dict[str, Any]]:
        """Build the complete search index from library items.

        Creates a list of search index entries for all library items,
        preserving the order of the input items.

        Args:
            items: List of library items to index.
            themes: Dictionary mapping theme IDs to Theme objects.

        Returns:
            List of search index entry dictionaries.
        """
        return [self._build_search_item(item, themes) for item in items]

    def write_index(
        self, items: list[LibraryItem], themes: dict[str, Theme]
    ) -> Path:
        """Build and write the search index to a JSON file.

        Creates the search-index.json file with all library items indexed
        for client-side search. The output format is:
        {"items": [<search_item>, ...]}

        Args:
            items: List of library items to index.
            themes: Dictionary mapping theme IDs to Theme objects.

        Returns:
            Path to the written search-index.json file.
        """
        # Ensure parent directories exist
        self.index_file.parent.mkdir(parents=True, exist_ok=True)

        # Build the index
        index = self.build_index(items, themes)

        # Write to file
        with open(self.index_file, "w", encoding="utf-8") as f:
            json.dump({"items": index}, f, ensure_ascii=False, indent=2)

        return self.index_file
