"""Tests for cover resolver module.

This module tests the CoverResolver class which resolves cover images for
library items. Books use API lookups (Open Library, Google Books), while
other content types use placeholder images based on content type.

Test classes are organized by the methods they test:
- TestCoverResolverInitialization: CoverResolver class initialization
- TestLoadCache: _load_cache() method for reading cover_cache.json
- TestSaveCache: _save_cache() method for persisting cover_cache.json
- TestFetchCoverByIsbn: _fetch_cover_by_isbn() for Open Library API by ISBN
- TestFetchCoverByTitle: _fetch_cover_by_title() for Open Library API by title
- TestFetchCoverFromGoogleBooks: _fetch_cover_from_google_books() API fallback
- TestGetPlaceholderUrl: get_placeholder_url() for content type placeholders
- TestResolveCover: resolve_cover() orchestration (cache -> API -> placeholder)
- TestResolveAllCovers: resolve_all_covers() batch processing
- TestApiErrorHandling: API error handling with exponential backoff
"""

import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch, Mock
import json

from src.library.models import LibraryItem, ContentType


class TestCoverResolverInitialization:
    """Tests for CoverResolver class initialization."""

    def test_cover_resolver_can_be_imported(self):
        """CoverResolver class should be importable from the module."""
        from src.library.cover_resolver import CoverResolver
        assert CoverResolver is not None

    def test_cover_resolver_init_with_cache_dir(self, temp_dir):
        """CoverResolver should accept a cache_dir path."""
        from src.library.cover_resolver import CoverResolver

        cache_dir = temp_dir / "library"
        cache_dir.mkdir(parents=True, exist_ok=True)

        resolver = CoverResolver(cache_dir=cache_dir)

        assert resolver.cache_dir == cache_dir

    def test_cover_resolver_init_creates_cache_dir_if_not_exists(self, temp_dir):
        """CoverResolver should create the cache_dir if it doesn't exist."""
        from src.library.cover_resolver import CoverResolver

        cache_dir = temp_dir / "library" / "subdir"
        assert not cache_dir.exists()

        resolver = CoverResolver(cache_dir=cache_dir)

        assert cache_dir.exists()

    def test_cover_resolver_init_cache_dir_as_string(self, temp_dir):
        """CoverResolver should accept cache_dir as a string and convert to Path."""
        from src.library.cover_resolver import CoverResolver

        cache_dir = str(temp_dir / "library")

        resolver = CoverResolver(cache_dir=cache_dir)

        assert isinstance(resolver.cache_dir, Path)
        assert resolver.cache_dir == Path(cache_dir)

    def test_cover_resolver_has_cache_file_path(self, temp_dir):
        """CoverResolver should have a cache_file path attribute."""
        from src.library.cover_resolver import CoverResolver

        cache_dir = temp_dir / "library"
        cache_dir.mkdir(parents=True, exist_ok=True)

        resolver = CoverResolver(cache_dir=cache_dir)

        assert resolver.cache_file == cache_dir / "cover_cache.json"

    def test_cover_resolver_initializes_empty_cache(self, temp_dir):
        """CoverResolver should initialize with an empty cache dict."""
        from src.library.cover_resolver import CoverResolver

        cache_dir = temp_dir / "library"
        cache_dir.mkdir(parents=True, exist_ok=True)

        resolver = CoverResolver(cache_dir=cache_dir)

        assert resolver.cache == {}

    def test_cover_resolver_loads_existing_cache_on_init(self, temp_dir):
        """CoverResolver should load existing cache file on initialization."""
        from src.library.cover_resolver import CoverResolver

        cache_dir = temp_dir / "library"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Create an existing cache file
        cache_data = {
            "item_123": {
                "url": "https://example.com/cover.jpg",
                "resolved_at": "2024-01-15T10:30:00",
                "source": "open_library_isbn"
            }
        }
        with open(cache_dir / "cover_cache.json", "w") as f:
            json.dump(cache_data, f)

        resolver = CoverResolver(cache_dir=cache_dir)

        assert resolver.cache == cache_data

    def test_cover_resolver_has_placeholder_base_path(self, temp_dir):
        """CoverResolver should have a placeholder_base_path attribute."""
        from src.library.cover_resolver import CoverResolver

        cache_dir = temp_dir / "library"
        cache_dir.mkdir(parents=True, exist_ok=True)

        resolver = CoverResolver(cache_dir=cache_dir)

        # Should point to the assets/images/placeholders directory
        assert hasattr(resolver, "placeholder_base_path")
        assert "placeholders" in str(resolver.placeholder_base_path)


class TestLoadCache:
    """Tests for _load_cache() method that reads cover_cache.json."""

    def test_load_cache_returns_empty_dict_when_file_not_exists(self, temp_dir):
        """_load_cache() should return empty dict when cache file doesn't exist."""
        from src.library.cover_resolver import CoverResolver

        cache_dir = temp_dir / "library"
        cache_dir.mkdir(parents=True, exist_ok=True)

        resolver = CoverResolver(cache_dir=cache_dir)
        # Reset cache and reload
        resolver.cache = None
        result = resolver._load_cache()

        assert result == {}

    def test_load_cache_reads_existing_json(self, temp_dir):
        """_load_cache() should read and parse existing cache file."""
        from src.library.cover_resolver import CoverResolver

        cache_dir = temp_dir / "library"
        cache_dir.mkdir(parents=True, exist_ok=True)

        cache_data = {
            "item_456": {
                "url": "https://covers.openlibrary.org/b/isbn/1234567890-L.jpg",
                "resolved_at": "2024-01-20T14:30:00",
                "source": "open_library_isbn"
            }
        }
        with open(cache_dir / "cover_cache.json", "w") as f:
            json.dump(cache_data, f)

        resolver = CoverResolver(cache_dir=cache_dir)
        # Force reload
        resolver.cache = None
        result = resolver._load_cache()

        assert result == cache_data

    def test_load_cache_handles_invalid_json(self, temp_dir):
        """_load_cache() should return empty dict for invalid JSON."""
        from src.library.cover_resolver import CoverResolver

        cache_dir = temp_dir / "library"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Write invalid JSON
        with open(cache_dir / "cover_cache.json", "w") as f:
            f.write("not valid json {{{")

        resolver = CoverResolver(cache_dir=cache_dir)
        # Force reload
        resolver.cache = None
        result = resolver._load_cache()

        assert result == {}


class TestSaveCache:
    """Tests for _save_cache() method that persists cover_cache.json."""

    def test_save_cache_creates_cache_file(self, temp_dir):
        """_save_cache() should create cover_cache.json file in cache_dir."""
        from src.library.cover_resolver import CoverResolver

        cache_dir = temp_dir / "library"
        cache_dir.mkdir(parents=True, exist_ok=True)

        resolver = CoverResolver(cache_dir=cache_dir)
        resolver.cache = {
            "item_123": {
                "url": "https://example.com/cover.jpg",
                "resolved_at": "2024-01-15T10:30:00",
                "source": "open_library_isbn"
            }
        }

        resolver._save_cache()

        assert (cache_dir / "cover_cache.json").exists()

    def test_save_cache_writes_valid_json(self, temp_dir):
        """_save_cache() should write valid JSON to cover_cache.json."""
        from src.library.cover_resolver import CoverResolver

        cache_dir = temp_dir / "library"
        cache_dir.mkdir(parents=True, exist_ok=True)

        resolver = CoverResolver(cache_dir=cache_dir)
        resolver.cache = {
            "item_123": {
                "url": "https://example.com/cover.jpg",
                "resolved_at": "2024-01-15T10:30:00",
                "source": "open_library_isbn"
            }
        }

        resolver._save_cache()

        # Should be able to parse the JSON
        with open(cache_dir / "cover_cache.json") as f:
            data = json.load(f)

        assert isinstance(data, dict)

    def test_save_cache_serializes_cache_data_correctly(self, temp_dir):
        """_save_cache() should serialize all cache fields correctly."""
        from src.library.cover_resolver import CoverResolver

        cache_dir = temp_dir / "library"
        cache_dir.mkdir(parents=True, exist_ok=True)

        resolver = CoverResolver(cache_dir=cache_dir)
        resolver.cache = {
            "item_789": {
                "url": "https://covers.openlibrary.org/b/title/Test-L.jpg",
                "resolved_at": "2024-01-22T08:15:00",
                "source": "open_library_title"
            }
        }

        resolver._save_cache()

        with open(cache_dir / "cover_cache.json") as f:
            data = json.load(f)

        assert data["item_789"]["url"] == "https://covers.openlibrary.org/b/title/Test-L.jpg"
        assert data["item_789"]["resolved_at"] == "2024-01-22T08:15:00"
        assert data["item_789"]["source"] == "open_library_title"

    def test_save_cache_overwrites_existing_file(self, temp_dir):
        """_save_cache() should overwrite existing cache file."""
        from src.library.cover_resolver import CoverResolver

        cache_dir = temp_dir / "library"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Create initial cache file
        with open(cache_dir / "cover_cache.json", "w") as f:
            json.dump({"old_item": {"url": "old_url"}}, f)

        resolver = CoverResolver(cache_dir=cache_dir)
        resolver.cache = {
            "new_item": {
                "url": "https://example.com/new.jpg",
                "resolved_at": "2024-01-25T12:00:00",
                "source": "google_books"
            }
        }

        resolver._save_cache()

        with open(cache_dir / "cover_cache.json") as f:
            data = json.load(f)

        assert "new_item" in data
        assert "old_item" not in data
