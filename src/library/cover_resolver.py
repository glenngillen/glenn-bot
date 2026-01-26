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
from datetime import datetime
from typing import Any, Optional, Union

import requests

from src.library.models import ContentType, LibraryItem


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

    def _fetch_cover_by_isbn(self, isbn: str) -> Optional[str]:
        """Fetch a book cover URL from Open Library API by ISBN.

        Constructs the Open Library cover URL and verifies the image exists
        by making a HEAD request to check the response status and content type.

        Args:
            isbn: The book's ISBN (10 or 13 digits). Hyphens will be stripped.

        Returns:
            The cover URL if a valid image exists, None otherwise.
            Returns None on 404, non-image content type, or network errors.
        """
        # Strip hyphens from ISBN
        clean_isbn = isbn.replace("-", "")

        # Construct the Open Library cover URL
        url = f"https://covers.openlibrary.org/b/isbn/{clean_isbn}-L.jpg"

        try:
            # Make a HEAD request to verify the image exists
            response = requests.head(url, timeout=10)

            # Check if the request was successful
            if response.status_code != 200:
                return None

            # Verify the content type is an image
            content_type = response.headers.get("content-type", "")
            if not content_type.startswith("image/"):
                return None

            return url

        except (requests.RequestException, requests.Timeout):
            return None

    def _fetch_cover_by_title(self, title: str) -> Optional[str]:
        """Fetch a book cover URL from Open Library API by title.

        Constructs the Open Library cover URL using the book title and verifies
        the image exists by making a HEAD request to check the response status
        and content type.

        This is a fallback when ISBN lookup fails or is not available.

        Args:
            title: The book's title. Will be URL-encoded. Leading/trailing
                whitespace will be stripped.

        Returns:
            The cover URL if a valid image exists, None otherwise.
            Returns None for empty/whitespace-only titles, 404 responses,
            non-image content types, or network errors.
        """
        # Handle empty or whitespace-only titles
        if not title or not title.strip():
            return None

        # Strip leading/trailing whitespace and URL-encode the title
        clean_title = title.strip()

        # URL-encode special characters using urllib
        from urllib.parse import quote

        encoded_title = quote(clean_title, safe="")

        # Construct the Open Library cover URL
        url = f"https://covers.openlibrary.org/b/title/{encoded_title}-L.jpg"

        try:
            # Make a HEAD request to verify the image exists
            response = requests.head(url, timeout=10)

            # Check if the request was successful
            if response.status_code != 200:
                return None

            # Verify the content type is an image
            content_type = response.headers.get("content-type", "")
            if not content_type.startswith("image/"):
                return None

            return url

        except (requests.RequestException, requests.Timeout):
            return None

    def _fetch_cover_from_google_books(self, title: str) -> Optional[str]:
        """Fetch a book cover URL from Google Books API by title.

        Searches the Google Books API for the given title and extracts the
        thumbnail URL from the first result's volumeInfo.imageLinks.

        This is a fallback when Open Library lookups fail.

        Args:
            title: The book's title. Will be URL-encoded. Leading/trailing
                whitespace will be stripped.

        Returns:
            The cover thumbnail URL if found, None otherwise.
            Returns None for empty/whitespace-only titles, API errors,
            no results, missing imageLinks, or network errors.
            Prefers 'thumbnail' over 'smallThumbnail' if both are available.
        """
        # Handle empty or whitespace-only titles
        if not title or not title.strip():
            return None

        # Strip leading/trailing whitespace
        clean_title = title.strip()

        # URL-encode the title for the search query
        from urllib.parse import quote

        encoded_title = quote(clean_title, safe="")

        # Construct the Google Books API URL
        url = f"https://www.googleapis.com/books/v1/volumes?q=intitle:{encoded_title}"

        try:
            # Make a GET request to the API
            response = requests.get(url, timeout=10)

            # Check if the request was successful
            if response.status_code != 200:
                return None

            # Parse the JSON response
            try:
                data = response.json()
            except ValueError:
                return None

            # Check if there are any items in the response
            items = data.get("items")
            if not items:
                return None

            # Get the first result
            first_item = items[0]

            # Extract volumeInfo
            volume_info = first_item.get("volumeInfo")
            if not volume_info:
                return None

            # Extract imageLinks
            image_links = volume_info.get("imageLinks")
            if not image_links:
                return None

            # Prefer 'thumbnail' (larger) over 'smallThumbnail'
            if "thumbnail" in image_links:
                return image_links["thumbnail"]
            elif "smallThumbnail" in image_links:
                return image_links["smallThumbnail"]

            return None

        except (requests.RequestException, requests.Timeout):
            return None

    def get_placeholder_url(self, content_type: ContentType) -> str:
        """Get the placeholder image URL for a given content type.

        Returns a path to an SVG placeholder image specific to the content type.
        Each of the 10 content types has its own distinct placeholder image.

        Args:
            content_type: The ContentType enum value for the item.

        Returns:
            A string path to the placeholder SVG image within the
            assets/images/placeholders directory. The path contains
            the content type value (e.g., "book", "article") and ends with .svg.
        """
        # Use the content type's value (e.g., "book", "web_content") as filename
        filename = f"{content_type.value}.svg"

        # Return the path relative to placeholders directory
        return f"assets/images/placeholders/{filename}"

    def resolve_cover(self, item: LibraryItem, _save: bool = True) -> str:
        """Resolve the cover image URL for a library item.

        Implements the full resolution flow:
        1. Check cache - if item is cached, return the cached URL
        2. For books: try ISBN lookup (if ISBN available) -> title lookup -> Google Books
        3. For non-books: use placeholder directly
        4. Cache the result and save to disk (unless _save=False)

        Args:
            item: The LibraryItem to resolve a cover for.
            _save: If True (default), save cache to disk after resolving.
                Set to False for batch processing to avoid redundant saves.

        Returns:
            The resolved cover image URL. For books, this may be an Open Library
            or Google Books URL. For non-books (or when all API lookups fail),
            this will be a placeholder image path.
        """
        # Step 1: Check cache first
        if item.id in self.cache:
            return self.cache[item.id]["url"]

        # Step 2: Determine the cover URL based on content type
        cover_url: Optional[str] = None
        source: str = "placeholder"

        if item.content_type == ContentType.BOOK:
            # For books, try API lookups in order of priority
            isbn = item.metadata.get("isbn") if item.metadata else None

            # Try ISBN lookup first (if ISBN is available)
            if isbn:
                cover_url = self._fetch_cover_by_isbn(isbn)
                if cover_url:
                    source = "open_library_isbn"

            # Fall back to title lookup
            if not cover_url:
                cover_url = self._fetch_cover_by_title(item.title)
                if cover_url:
                    source = "open_library_title"

            # Fall back to Google Books
            if not cover_url:
                cover_url = self._fetch_cover_from_google_books(item.title)
                if cover_url:
                    source = "google_books"

        # Step 3: Use placeholder if no cover found (or for non-books)
        if not cover_url:
            cover_url = self.get_placeholder_url(item.content_type)
            source = "placeholder"

        # Step 4: Cache the result with metadata
        self.cache[item.id] = {
            "url": cover_url,
            "resolved_at": datetime.now().isoformat(),
            "source": source,
        }

        # Step 5: Save cache to disk (if requested)
        if _save:
            self._save_cache()

        return cover_url

    def resolve_all_covers(self, items: list[LibraryItem]) -> list[LibraryItem]:
        """Resolve cover images for a list of library items.

        Processes each item in the list and resolves its cover image URL.
        Items are not modified in place; new LibraryItem instances are returned
        with the cover_image_url field populated.

        This method is optimized for batch processing:
        - Uses cached values when available
        - Continues processing if an individual item fails
        - Saves the cache once at the end, not after each item

        Args:
            items: A list of LibraryItem instances to resolve covers for.

        Returns:
            A list of new LibraryItem instances with cover_image_url populated.
            The order of items is preserved.
        """
        if not items:
            return []

        result: list[LibraryItem] = []

        for item in items:
            # Resolve the cover URL for this item
            # Pass _save=False to avoid saving cache after each item
            cover_url = self.resolve_cover(item, _save=False)

            # Create a new LibraryItem with the cover_image_url set
            new_item = LibraryItem(
                id=item.id,
                content_type=item.content_type,
                title=item.title,
                summary=item.summary,
                full_content=item.full_content,
                source_url=item.source_url,
                cover_image_url=cover_url,
                metadata=item.metadata,
                themes=item.themes,
                created_at=item.created_at,
                highlights=item.highlights,
            )
            result.append(new_item)

        # Save cache once at the end
        self._save_cache()

        return result
