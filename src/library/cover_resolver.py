"""Cover image resolution for library items.

This module provides the CoverResolver class which resolves cover images for
library items. Books use API lookups via Open Library and Google Books APIs,
while other content types use placeholder images based on content type.

Resolution Flow for Books:
1. If book has ISBN in metadata, try Open Library by ISBN
2. If no ISBN or API fails, try Open Library by title
3. If still no result, try Google Books API by title
4. If all fail, use book placeholder image

Non-book content types are assigned placeholder images based on their type.

Caching:
- Resolved URLs are cached in cover_cache.json
- Cache never expires (covers don't change)
- Force refresh available via clear_cache() or refresh flag
"""

import json
from pathlib import Path
from typing import Any, Union


class CoverResolver:
    """Resolves cover images for library items.

    Books use API lookups (Open Library, Google Books), while other content
    types use placeholder images based on content type.

    Attributes:
        cache_dir: Path to the directory for storing the cache file.
        cache_file: Path to the cover_cache.json file.
        cache: Dictionary mapping item IDs to cached cover data.
        placeholder_base_path: Path to the placeholder images directory.
    """

    def __init__(self, cache_dir: Union[str, Path]) -> None:
        """Initialize the CoverResolver.

        Args:
            cache_dir: Path to the directory for storing the cache file.
                Will be created if it doesn't exist. Can be a string or Path.
        """
        self.cache_dir = Path(cache_dir) if isinstance(cache_dir, str) else cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.cache_file = self.cache_dir / "cover_cache.json"

        # Path to placeholder images (relative to the library module)
        self.placeholder_base_path = (
            Path(__file__).parent / "assets" / "images" / "placeholders"
        )

        # Load existing cache on initialization
        self.cache = self._load_cache()

    def _load_cache(self) -> dict[str, Any]:
        """Load the cover cache from disk.

        Returns:
            A dictionary mapping item IDs to cached cover data.
            Returns an empty dict if the cache file doesn't exist
            or contains invalid JSON.
        """
        if not self.cache_file.exists():
            return {}

        try:
            with open(self.cache_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}

    def _save_cache(self) -> None:
        """Save the cover cache to disk.

        Writes the current cache dictionary to cover_cache.json.
        Overwrites any existing file.
        """
        with open(self.cache_file, "w") as f:
            json.dump(self.cache, f, indent=2)
