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


class TestFetchCoverByIsbn:
    """Tests for _fetch_cover_by_isbn() method that calls Open Library API by ISBN."""

    def test_fetch_cover_by_isbn_constructs_correct_url(self, temp_dir):
        """_fetch_cover_by_isbn() should construct the correct Open Library URL."""
        from src.library.cover_resolver import CoverResolver

        cache_dir = temp_dir / "library"
        cache_dir.mkdir(parents=True, exist_ok=True)

        resolver = CoverResolver(cache_dir=cache_dir)

        # Mock the requests.head call
        with patch("src.library.cover_resolver.requests.head") as mock_head:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.headers = {"content-type": "image/jpeg"}
            mock_head.return_value = mock_response

            resolver._fetch_cover_by_isbn("9780134685991")

            # Verify the URL construction
            expected_url = "https://covers.openlibrary.org/b/isbn/9780134685991-L.jpg"
            mock_head.assert_called_once()
            actual_url = mock_head.call_args[0][0]
            assert actual_url == expected_url

    def test_fetch_cover_by_isbn_returns_url_on_success(self, temp_dir):
        """_fetch_cover_by_isbn() should return the cover URL when image exists."""
        from src.library.cover_resolver import CoverResolver

        cache_dir = temp_dir / "library"
        cache_dir.mkdir(parents=True, exist_ok=True)

        resolver = CoverResolver(cache_dir=cache_dir)

        with patch("src.library.cover_resolver.requests.head") as mock_head:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.headers = {"content-type": "image/jpeg"}
            mock_head.return_value = mock_response

            result = resolver._fetch_cover_by_isbn("9780134685991")

            expected_url = "https://covers.openlibrary.org/b/isbn/9780134685991-L.jpg"
            assert result == expected_url

    def test_fetch_cover_by_isbn_returns_none_on_404(self, temp_dir):
        """_fetch_cover_by_isbn() should return None when image not found (404)."""
        from src.library.cover_resolver import CoverResolver

        cache_dir = temp_dir / "library"
        cache_dir.mkdir(parents=True, exist_ok=True)

        resolver = CoverResolver(cache_dir=cache_dir)

        with patch("src.library.cover_resolver.requests.head") as mock_head:
            mock_response = Mock()
            mock_response.status_code = 404
            mock_head.return_value = mock_response

            result = resolver._fetch_cover_by_isbn("0000000000")

            assert result is None

    def test_fetch_cover_by_isbn_returns_none_on_non_image_content_type(self, temp_dir):
        """_fetch_cover_by_isbn() should return None when response is not an image."""
        from src.library.cover_resolver import CoverResolver

        cache_dir = temp_dir / "library"
        cache_dir.mkdir(parents=True, exist_ok=True)

        resolver = CoverResolver(cache_dir=cache_dir)

        with patch("src.library.cover_resolver.requests.head") as mock_head:
            # Open Library returns a 1x1 pixel GIF for missing covers
            # We should detect non-image or very small images
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.headers = {"content-type": "text/html"}
            mock_head.return_value = mock_response

            result = resolver._fetch_cover_by_isbn("0000000000")

            assert result is None

    def test_fetch_cover_by_isbn_handles_request_exception(self, temp_dir):
        """_fetch_cover_by_isbn() should return None on network errors."""
        from src.library.cover_resolver import CoverResolver
        import requests

        cache_dir = temp_dir / "library"
        cache_dir.mkdir(parents=True, exist_ok=True)

        resolver = CoverResolver(cache_dir=cache_dir)

        with patch("src.library.cover_resolver.requests.head") as mock_head:
            mock_head.side_effect = requests.RequestException("Connection error")

            result = resolver._fetch_cover_by_isbn("9780134685991")

            assert result is None

    def test_fetch_cover_by_isbn_handles_timeout(self, temp_dir):
        """_fetch_cover_by_isbn() should return None on request timeout."""
        from src.library.cover_resolver import CoverResolver
        import requests

        cache_dir = temp_dir / "library"
        cache_dir.mkdir(parents=True, exist_ok=True)

        resolver = CoverResolver(cache_dir=cache_dir)

        with patch("src.library.cover_resolver.requests.head") as mock_head:
            mock_head.side_effect = requests.Timeout("Request timed out")

            result = resolver._fetch_cover_by_isbn("9780134685991")

            assert result is None

    def test_fetch_cover_by_isbn_strips_hyphens_from_isbn(self, temp_dir):
        """_fetch_cover_by_isbn() should handle ISBN with hyphens."""
        from src.library.cover_resolver import CoverResolver

        cache_dir = temp_dir / "library"
        cache_dir.mkdir(parents=True, exist_ok=True)

        resolver = CoverResolver(cache_dir=cache_dir)

        with patch("src.library.cover_resolver.requests.head") as mock_head:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.headers = {"content-type": "image/jpeg"}
            mock_head.return_value = mock_response

            resolver._fetch_cover_by_isbn("978-0-13-468599-1")

            # URL should have hyphens removed
            actual_url = mock_head.call_args[0][0]
            assert "978-0-13-468599-1" not in actual_url
            assert "9780134685991" in actual_url

    def test_fetch_cover_by_isbn_uses_timeout(self, temp_dir):
        """_fetch_cover_by_isbn() should use a reasonable timeout for the request."""
        from src.library.cover_resolver import CoverResolver

        cache_dir = temp_dir / "library"
        cache_dir.mkdir(parents=True, exist_ok=True)

        resolver = CoverResolver(cache_dir=cache_dir)

        with patch("src.library.cover_resolver.requests.head") as mock_head:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.headers = {"content-type": "image/jpeg"}
            mock_head.return_value = mock_response

            resolver._fetch_cover_by_isbn("9780134685991")

            # Verify a timeout is specified
            assert "timeout" in mock_head.call_args.kwargs
            # Timeout should be reasonable (5-30 seconds)
            timeout = mock_head.call_args.kwargs["timeout"]
            assert 5 <= timeout <= 30

    def test_fetch_cover_by_isbn_accepts_png_content_type(self, temp_dir):
        """_fetch_cover_by_isbn() should accept PNG images as valid covers."""
        from src.library.cover_resolver import CoverResolver

        cache_dir = temp_dir / "library"
        cache_dir.mkdir(parents=True, exist_ok=True)

        resolver = CoverResolver(cache_dir=cache_dir)

        with patch("src.library.cover_resolver.requests.head") as mock_head:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.headers = {"content-type": "image/png"}
            mock_head.return_value = mock_response

            result = resolver._fetch_cover_by_isbn("9780134685991")

            expected_url = "https://covers.openlibrary.org/b/isbn/9780134685991-L.jpg"
            assert result == expected_url

    def test_fetch_cover_by_isbn_accepts_gif_content_type(self, temp_dir):
        """_fetch_cover_by_isbn() should accept GIF images as valid covers."""
        from src.library.cover_resolver import CoverResolver

        cache_dir = temp_dir / "library"
        cache_dir.mkdir(parents=True, exist_ok=True)

        resolver = CoverResolver(cache_dir=cache_dir)

        with patch("src.library.cover_resolver.requests.head") as mock_head:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.headers = {"content-type": "image/gif"}
            mock_head.return_value = mock_response

            result = resolver._fetch_cover_by_isbn("9780134685991")

            expected_url = "https://covers.openlibrary.org/b/isbn/9780134685991-L.jpg"
            assert result == expected_url


class TestFetchCoverByTitle:
    """Tests for _fetch_cover_by_title() method that calls Open Library API by title."""

    def test_fetch_cover_by_title_constructs_correct_url(self, temp_dir):
        """_fetch_cover_by_title() should construct the correct Open Library URL."""
        from src.library.cover_resolver import CoverResolver

        cache_dir = temp_dir / "library"
        cache_dir.mkdir(parents=True, exist_ok=True)

        resolver = CoverResolver(cache_dir=cache_dir)

        with patch("src.library.cover_resolver.requests.head") as mock_head:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.headers = {"content-type": "image/jpeg"}
            mock_head.return_value = mock_response

            resolver._fetch_cover_by_title("Thinking in Systems")

            # Verify the URL construction
            expected_url = "https://covers.openlibrary.org/b/title/Thinking%20in%20Systems-L.jpg"
            mock_head.assert_called_once()
            actual_url = mock_head.call_args[0][0]
            assert actual_url == expected_url

    def test_fetch_cover_by_title_returns_url_on_success(self, temp_dir):
        """_fetch_cover_by_title() should return the cover URL when image exists."""
        from src.library.cover_resolver import CoverResolver

        cache_dir = temp_dir / "library"
        cache_dir.mkdir(parents=True, exist_ok=True)

        resolver = CoverResolver(cache_dir=cache_dir)

        with patch("src.library.cover_resolver.requests.head") as mock_head:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.headers = {"content-type": "image/jpeg"}
            mock_head.return_value = mock_response

            result = resolver._fetch_cover_by_title("Thinking in Systems")

            expected_url = "https://covers.openlibrary.org/b/title/Thinking%20in%20Systems-L.jpg"
            assert result == expected_url

    def test_fetch_cover_by_title_returns_none_on_404(self, temp_dir):
        """_fetch_cover_by_title() should return None when image not found (404)."""
        from src.library.cover_resolver import CoverResolver

        cache_dir = temp_dir / "library"
        cache_dir.mkdir(parents=True, exist_ok=True)

        resolver = CoverResolver(cache_dir=cache_dir)

        with patch("src.library.cover_resolver.requests.head") as mock_head:
            mock_response = Mock()
            mock_response.status_code = 404
            mock_head.return_value = mock_response

            result = resolver._fetch_cover_by_title("NonExistent Book Title")

            assert result is None

    def test_fetch_cover_by_title_returns_none_on_non_image_content_type(self, temp_dir):
        """_fetch_cover_by_title() should return None when response is not an image."""
        from src.library.cover_resolver import CoverResolver

        cache_dir = temp_dir / "library"
        cache_dir.mkdir(parents=True, exist_ok=True)

        resolver = CoverResolver(cache_dir=cache_dir)

        with patch("src.library.cover_resolver.requests.head") as mock_head:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.headers = {"content-type": "text/html"}
            mock_head.return_value = mock_response

            result = resolver._fetch_cover_by_title("Some Book")

            assert result is None

    def test_fetch_cover_by_title_handles_request_exception(self, temp_dir):
        """_fetch_cover_by_title() should return None on network errors."""
        from src.library.cover_resolver import CoverResolver
        import requests

        cache_dir = temp_dir / "library"
        cache_dir.mkdir(parents=True, exist_ok=True)

        resolver = CoverResolver(cache_dir=cache_dir)

        with patch("src.library.cover_resolver.requests.head") as mock_head:
            mock_head.side_effect = requests.RequestException("Connection error")

            result = resolver._fetch_cover_by_title("Thinking in Systems")

            assert result is None

    def test_fetch_cover_by_title_handles_timeout(self, temp_dir):
        """_fetch_cover_by_title() should return None on request timeout."""
        from src.library.cover_resolver import CoverResolver
        import requests

        cache_dir = temp_dir / "library"
        cache_dir.mkdir(parents=True, exist_ok=True)

        resolver = CoverResolver(cache_dir=cache_dir)

        with patch("src.library.cover_resolver.requests.head") as mock_head:
            mock_head.side_effect = requests.Timeout("Request timed out")

            result = resolver._fetch_cover_by_title("Thinking in Systems")

            assert result is None

    def test_fetch_cover_by_title_url_encodes_special_characters(self, temp_dir):
        """_fetch_cover_by_title() should URL-encode special characters in title."""
        from src.library.cover_resolver import CoverResolver

        cache_dir = temp_dir / "library"
        cache_dir.mkdir(parents=True, exist_ok=True)

        resolver = CoverResolver(cache_dir=cache_dir)

        with patch("src.library.cover_resolver.requests.head") as mock_head:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.headers = {"content-type": "image/jpeg"}
            mock_head.return_value = mock_response

            # Title with special characters: ampersand, apostrophe, colon
            resolver._fetch_cover_by_title("The Art & Science of Java: A Beginner's Guide")

            actual_url = mock_head.call_args[0][0]
            # URL should be encoded (spaces, ampersand, apostrophe, colon)
            assert " " not in actual_url
            assert "&" not in actual_url or "%26" in actual_url
            assert "'" not in actual_url or "%27" in actual_url
            assert ":" not in actual_url or "%3A" in actual_url

    def test_fetch_cover_by_title_uses_timeout(self, temp_dir):
        """_fetch_cover_by_title() should use a reasonable timeout for the request."""
        from src.library.cover_resolver import CoverResolver

        cache_dir = temp_dir / "library"
        cache_dir.mkdir(parents=True, exist_ok=True)

        resolver = CoverResolver(cache_dir=cache_dir)

        with patch("src.library.cover_resolver.requests.head") as mock_head:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.headers = {"content-type": "image/jpeg"}
            mock_head.return_value = mock_response

            resolver._fetch_cover_by_title("Test Book")

            # Verify a timeout is specified
            assert "timeout" in mock_head.call_args.kwargs
            # Timeout should be reasonable (5-30 seconds)
            timeout = mock_head.call_args.kwargs["timeout"]
            assert 5 <= timeout <= 30

    def test_fetch_cover_by_title_accepts_png_content_type(self, temp_dir):
        """_fetch_cover_by_title() should accept PNG images as valid covers."""
        from src.library.cover_resolver import CoverResolver

        cache_dir = temp_dir / "library"
        cache_dir.mkdir(parents=True, exist_ok=True)

        resolver = CoverResolver(cache_dir=cache_dir)

        with patch("src.library.cover_resolver.requests.head") as mock_head:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.headers = {"content-type": "image/png"}
            mock_head.return_value = mock_response

            result = resolver._fetch_cover_by_title("Test Book")

            expected_url = "https://covers.openlibrary.org/b/title/Test%20Book-L.jpg"
            assert result == expected_url

    def test_fetch_cover_by_title_handles_empty_title(self, temp_dir):
        """_fetch_cover_by_title() should return None for empty title."""
        from src.library.cover_resolver import CoverResolver

        cache_dir = temp_dir / "library"
        cache_dir.mkdir(parents=True, exist_ok=True)

        resolver = CoverResolver(cache_dir=cache_dir)

        result = resolver._fetch_cover_by_title("")

        assert result is None

    def test_fetch_cover_by_title_handles_whitespace_only_title(self, temp_dir):
        """_fetch_cover_by_title() should return None for whitespace-only title."""
        from src.library.cover_resolver import CoverResolver

        cache_dir = temp_dir / "library"
        cache_dir.mkdir(parents=True, exist_ok=True)

        resolver = CoverResolver(cache_dir=cache_dir)

        result = resolver._fetch_cover_by_title("   ")

        assert result is None

    def test_fetch_cover_by_title_strips_leading_trailing_whitespace(self, temp_dir):
        """_fetch_cover_by_title() should strip leading/trailing whitespace from title."""
        from src.library.cover_resolver import CoverResolver

        cache_dir = temp_dir / "library"
        cache_dir.mkdir(parents=True, exist_ok=True)

        resolver = CoverResolver(cache_dir=cache_dir)

        with patch("src.library.cover_resolver.requests.head") as mock_head:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.headers = {"content-type": "image/jpeg"}
            mock_head.return_value = mock_response

            resolver._fetch_cover_by_title("  Thinking in Systems  ")

            actual_url = mock_head.call_args[0][0]
            # Should not have leading/trailing encoded spaces
            assert actual_url == "https://covers.openlibrary.org/b/title/Thinking%20in%20Systems-L.jpg"


class TestFetchCoverFromGoogleBooks:
    """Tests for _fetch_cover_from_google_books() method that calls Google Books API."""

    def test_fetch_cover_from_google_books_constructs_correct_url(self, temp_dir):
        """_fetch_cover_from_google_books() should construct the correct Google Books API URL."""
        from src.library.cover_resolver import CoverResolver

        cache_dir = temp_dir / "library"
        cache_dir.mkdir(parents=True, exist_ok=True)

        resolver = CoverResolver(cache_dir=cache_dir)

        with patch("src.library.cover_resolver.requests.get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "items": [
                    {
                        "volumeInfo": {
                            "imageLinks": {
                                "thumbnail": "https://books.google.com/books/content?id=abc123"
                            }
                        }
                    }
                ]
            }
            mock_get.return_value = mock_response

            resolver._fetch_cover_from_google_books("Thinking in Systems")

            mock_get.assert_called_once()
            actual_url = mock_get.call_args[0][0]
            assert "googleapis.com/books/v1/volumes" in actual_url
            assert "intitle:Thinking" in actual_url or "intitle%3AThinking" in actual_url

    def test_fetch_cover_from_google_books_returns_thumbnail_url_on_success(self, temp_dir):
        """_fetch_cover_from_google_books() should return the thumbnail URL when found."""
        from src.library.cover_resolver import CoverResolver

        cache_dir = temp_dir / "library"
        cache_dir.mkdir(parents=True, exist_ok=True)

        resolver = CoverResolver(cache_dir=cache_dir)

        with patch("src.library.cover_resolver.requests.get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "items": [
                    {
                        "volumeInfo": {
                            "imageLinks": {
                                "thumbnail": "https://books.google.com/books/content?id=abc123"
                            }
                        }
                    }
                ]
            }
            mock_get.return_value = mock_response

            result = resolver._fetch_cover_from_google_books("Thinking in Systems")

            assert result == "https://books.google.com/books/content?id=abc123"

    def test_fetch_cover_from_google_books_returns_none_on_404(self, temp_dir):
        """_fetch_cover_from_google_books() should return None when API returns 404."""
        from src.library.cover_resolver import CoverResolver

        cache_dir = temp_dir / "library"
        cache_dir.mkdir(parents=True, exist_ok=True)

        resolver = CoverResolver(cache_dir=cache_dir)

        with patch("src.library.cover_resolver.requests.get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 404
            mock_get.return_value = mock_response

            result = resolver._fetch_cover_from_google_books("NonExistent Book")

            assert result is None

    def test_fetch_cover_from_google_books_returns_none_on_empty_items(self, temp_dir):
        """_fetch_cover_from_google_books() should return None when no items in response."""
        from src.library.cover_resolver import CoverResolver

        cache_dir = temp_dir / "library"
        cache_dir.mkdir(parents=True, exist_ok=True)

        resolver = CoverResolver(cache_dir=cache_dir)

        with patch("src.library.cover_resolver.requests.get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"totalItems": 0}
            mock_get.return_value = mock_response

            result = resolver._fetch_cover_from_google_books("NonExistent Book Title")

            assert result is None

    def test_fetch_cover_from_google_books_returns_none_when_no_image_links(self, temp_dir):
        """_fetch_cover_from_google_books() should return None when item has no imageLinks."""
        from src.library.cover_resolver import CoverResolver

        cache_dir = temp_dir / "library"
        cache_dir.mkdir(parents=True, exist_ok=True)

        resolver = CoverResolver(cache_dir=cache_dir)

        with patch("src.library.cover_resolver.requests.get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "items": [
                    {
                        "volumeInfo": {
                            "title": "A Book Without Cover"
                        }
                    }
                ]
            }
            mock_get.return_value = mock_response

            result = resolver._fetch_cover_from_google_books("A Book Without Cover")

            assert result is None

    def test_fetch_cover_from_google_books_handles_request_exception(self, temp_dir):
        """_fetch_cover_from_google_books() should return None on network errors."""
        from src.library.cover_resolver import CoverResolver
        import requests

        cache_dir = temp_dir / "library"
        cache_dir.mkdir(parents=True, exist_ok=True)

        resolver = CoverResolver(cache_dir=cache_dir)

        with patch("src.library.cover_resolver.requests.get") as mock_get:
            mock_get.side_effect = requests.RequestException("Connection error")

            result = resolver._fetch_cover_from_google_books("Thinking in Systems")

            assert result is None

    def test_fetch_cover_from_google_books_handles_timeout(self, temp_dir):
        """_fetch_cover_from_google_books() should return None on request timeout."""
        from src.library.cover_resolver import CoverResolver
        import requests

        cache_dir = temp_dir / "library"
        cache_dir.mkdir(parents=True, exist_ok=True)

        resolver = CoverResolver(cache_dir=cache_dir)

        with patch("src.library.cover_resolver.requests.get") as mock_get:
            mock_get.side_effect = requests.Timeout("Request timed out")

            result = resolver._fetch_cover_from_google_books("Thinking in Systems")

            assert result is None

    def test_fetch_cover_from_google_books_handles_json_decode_error(self, temp_dir):
        """_fetch_cover_from_google_books() should return None on invalid JSON response."""
        from src.library.cover_resolver import CoverResolver

        cache_dir = temp_dir / "library"
        cache_dir.mkdir(parents=True, exist_ok=True)

        resolver = CoverResolver(cache_dir=cache_dir)

        with patch("src.library.cover_resolver.requests.get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.side_effect = ValueError("Invalid JSON")
            mock_get.return_value = mock_response

            result = resolver._fetch_cover_from_google_books("Test Book")

            assert result is None

    def test_fetch_cover_from_google_books_url_encodes_special_characters(self, temp_dir):
        """_fetch_cover_from_google_books() should URL-encode special characters in title."""
        from src.library.cover_resolver import CoverResolver

        cache_dir = temp_dir / "library"
        cache_dir.mkdir(parents=True, exist_ok=True)

        resolver = CoverResolver(cache_dir=cache_dir)

        with patch("src.library.cover_resolver.requests.get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "items": [
                    {
                        "volumeInfo": {
                            "imageLinks": {
                                "thumbnail": "https://books.google.com/books/content?id=xyz"
                            }
                        }
                    }
                ]
            }
            mock_get.return_value = mock_response

            # Title with special characters
            resolver._fetch_cover_from_google_books("The Art & Science: A Beginner's Guide")

            actual_url = mock_get.call_args[0][0]
            # Spaces should be encoded as %20 or +
            assert " " not in actual_url or "+" in actual_url

    def test_fetch_cover_from_google_books_uses_timeout(self, temp_dir):
        """_fetch_cover_from_google_books() should use a reasonable timeout for the request."""
        from src.library.cover_resolver import CoverResolver

        cache_dir = temp_dir / "library"
        cache_dir.mkdir(parents=True, exist_ok=True)

        resolver = CoverResolver(cache_dir=cache_dir)

        with patch("src.library.cover_resolver.requests.get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "items": [
                    {
                        "volumeInfo": {
                            "imageLinks": {
                                "thumbnail": "https://books.google.com/books/content?id=abc"
                            }
                        }
                    }
                ]
            }
            mock_get.return_value = mock_response

            resolver._fetch_cover_from_google_books("Test Book")

            # Verify a timeout is specified
            assert "timeout" in mock_get.call_args.kwargs
            # Timeout should be reasonable (5-30 seconds)
            timeout = mock_get.call_args.kwargs["timeout"]
            assert 5 <= timeout <= 30

    def test_fetch_cover_from_google_books_handles_empty_title(self, temp_dir):
        """_fetch_cover_from_google_books() should return None for empty title."""
        from src.library.cover_resolver import CoverResolver

        cache_dir = temp_dir / "library"
        cache_dir.mkdir(parents=True, exist_ok=True)

        resolver = CoverResolver(cache_dir=cache_dir)

        result = resolver._fetch_cover_from_google_books("")

        assert result is None

    def test_fetch_cover_from_google_books_handles_whitespace_only_title(self, temp_dir):
        """_fetch_cover_from_google_books() should return None for whitespace-only title."""
        from src.library.cover_resolver import CoverResolver

        cache_dir = temp_dir / "library"
        cache_dir.mkdir(parents=True, exist_ok=True)

        resolver = CoverResolver(cache_dir=cache_dir)

        result = resolver._fetch_cover_from_google_books("   ")

        assert result is None

    def test_fetch_cover_from_google_books_strips_whitespace(self, temp_dir):
        """_fetch_cover_from_google_books() should strip leading/trailing whitespace."""
        from src.library.cover_resolver import CoverResolver

        cache_dir = temp_dir / "library"
        cache_dir.mkdir(parents=True, exist_ok=True)

        resolver = CoverResolver(cache_dir=cache_dir)

        with patch("src.library.cover_resolver.requests.get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "items": [
                    {
                        "volumeInfo": {
                            "imageLinks": {
                                "thumbnail": "https://books.google.com/books/content?id=abc"
                            }
                        }
                    }
                ]
            }
            mock_get.return_value = mock_response

            resolver._fetch_cover_from_google_books("  Test Book  ")

            actual_url = mock_get.call_args[0][0]
            # Should not have leading/trailing encoded spaces in the search query
            assert "%20%20" not in actual_url
            assert "++Test" not in actual_url
            assert "Book++" not in actual_url

    def test_fetch_cover_from_google_books_prefers_larger_image_if_available(self, temp_dir):
        """_fetch_cover_from_google_books() should prefer smallThumbnail over thumbnail if both present."""
        from src.library.cover_resolver import CoverResolver

        cache_dir = temp_dir / "library"
        cache_dir.mkdir(parents=True, exist_ok=True)

        resolver = CoverResolver(cache_dir=cache_dir)

        with patch("src.library.cover_resolver.requests.get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            # Google Books provides both thumbnail and smallThumbnail
            mock_response.json.return_value = {
                "items": [
                    {
                        "volumeInfo": {
                            "imageLinks": {
                                "smallThumbnail": "https://books.google.com/small.jpg",
                                "thumbnail": "https://books.google.com/thumbnail.jpg"
                            }
                        }
                    }
                ]
            }
            mock_get.return_value = mock_response

            result = resolver._fetch_cover_from_google_books("Test Book")

            # Should return thumbnail (larger) if available
            assert result == "https://books.google.com/thumbnail.jpg"

    def test_fetch_cover_from_google_books_falls_back_to_small_thumbnail(self, temp_dir):
        """_fetch_cover_from_google_books() should fall back to smallThumbnail if no thumbnail."""
        from src.library.cover_resolver import CoverResolver

        cache_dir = temp_dir / "library"
        cache_dir.mkdir(parents=True, exist_ok=True)

        resolver = CoverResolver(cache_dir=cache_dir)

        with patch("src.library.cover_resolver.requests.get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            # Only smallThumbnail available
            mock_response.json.return_value = {
                "items": [
                    {
                        "volumeInfo": {
                            "imageLinks": {
                                "smallThumbnail": "https://books.google.com/small.jpg"
                            }
                        }
                    }
                ]
            }
            mock_get.return_value = mock_response

            result = resolver._fetch_cover_from_google_books("Test Book")

            assert result == "https://books.google.com/small.jpg"

    def test_fetch_cover_from_google_books_returns_none_on_missing_volume_info(self, temp_dir):
        """_fetch_cover_from_google_books() should return None when volumeInfo is missing."""
        from src.library.cover_resolver import CoverResolver

        cache_dir = temp_dir / "library"
        cache_dir.mkdir(parents=True, exist_ok=True)

        resolver = CoverResolver(cache_dir=cache_dir)

        with patch("src.library.cover_resolver.requests.get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "items": [
                    {
                        "kind": "books#volume"
                    }
                ]
            }
            mock_get.return_value = mock_response

            result = resolver._fetch_cover_from_google_books("Test Book")

            assert result is None


class TestGetPlaceholderUrl:
    """Tests for get_placeholder_url() method that returns placeholder URLs by content type."""

    def test_get_placeholder_url_returns_book_placeholder(self, temp_dir):
        """get_placeholder_url() should return a book placeholder for BOOK content type."""
        from src.library.cover_resolver import CoverResolver

        cache_dir = temp_dir / "library"
        cache_dir.mkdir(parents=True, exist_ok=True)

        resolver = CoverResolver(cache_dir=cache_dir)

        result = resolver.get_placeholder_url(ContentType.BOOK)

        assert result is not None
        assert "book" in result.lower()
        assert result.endswith(".svg")

    def test_get_placeholder_url_returns_article_placeholder(self, temp_dir):
        """get_placeholder_url() should return an article placeholder for ARTICLE content type."""
        from src.library.cover_resolver import CoverResolver

        cache_dir = temp_dir / "library"
        cache_dir.mkdir(parents=True, exist_ok=True)

        resolver = CoverResolver(cache_dir=cache_dir)

        result = resolver.get_placeholder_url(ContentType.ARTICLE)

        assert result is not None
        assert "article" in result.lower()
        assert result.endswith(".svg")

    def test_get_placeholder_url_returns_framework_placeholder(self, temp_dir):
        """get_placeholder_url() should return a framework placeholder for FRAMEWORK content type."""
        from src.library.cover_resolver import CoverResolver

        cache_dir = temp_dir / "library"
        cache_dir.mkdir(parents=True, exist_ok=True)

        resolver = CoverResolver(cache_dir=cache_dir)

        result = resolver.get_placeholder_url(ContentType.FRAMEWORK)

        assert result is not None
        assert "framework" in result.lower()
        assert result.endswith(".svg")

    def test_get_placeholder_url_returns_value_placeholder(self, temp_dir):
        """get_placeholder_url() should return a value placeholder for VALUE content type."""
        from src.library.cover_resolver import CoverResolver

        cache_dir = temp_dir / "library"
        cache_dir.mkdir(parents=True, exist_ok=True)

        resolver = CoverResolver(cache_dir=cache_dir)

        result = resolver.get_placeholder_url(ContentType.VALUE)

        assert result is not None
        assert "value" in result.lower()
        assert result.endswith(".svg")

    def test_get_placeholder_url_returns_preference_placeholder(self, temp_dir):
        """get_placeholder_url() should return a preference placeholder for PREFERENCE content type."""
        from src.library.cover_resolver import CoverResolver

        cache_dir = temp_dir / "library"
        cache_dir.mkdir(parents=True, exist_ok=True)

        resolver = CoverResolver(cache_dir=cache_dir)

        result = resolver.get_placeholder_url(ContentType.PREFERENCE)

        assert result is not None
        assert "preference" in result.lower()
        assert result.endswith(".svg")

    def test_get_placeholder_url_returns_memory_placeholder(self, temp_dir):
        """get_placeholder_url() should return a memory placeholder for MEMORY content type."""
        from src.library.cover_resolver import CoverResolver

        cache_dir = temp_dir / "library"
        cache_dir.mkdir(parents=True, exist_ok=True)

        resolver = CoverResolver(cache_dir=cache_dir)

        result = resolver.get_placeholder_url(ContentType.MEMORY)

        assert result is not None
        assert "memory" in result.lower()
        assert result.endswith(".svg")

    def test_get_placeholder_url_returns_insight_placeholder(self, temp_dir):
        """get_placeholder_url() should return an insight placeholder for INSIGHT content type."""
        from src.library.cover_resolver import CoverResolver

        cache_dir = temp_dir / "library"
        cache_dir.mkdir(parents=True, exist_ok=True)

        resolver = CoverResolver(cache_dir=cache_dir)

        result = resolver.get_placeholder_url(ContentType.INSIGHT)

        assert result is not None
        assert "insight" in result.lower()
        assert result.endswith(".svg")

    def test_get_placeholder_url_returns_goal_placeholder(self, temp_dir):
        """get_placeholder_url() should return a goal placeholder for GOAL content type."""
        from src.library.cover_resolver import CoverResolver

        cache_dir = temp_dir / "library"
        cache_dir.mkdir(parents=True, exist_ok=True)

        resolver = CoverResolver(cache_dir=cache_dir)

        result = resolver.get_placeholder_url(ContentType.GOAL)

        assert result is not None
        assert "goal" in result.lower()
        assert result.endswith(".svg")

    def test_get_placeholder_url_returns_skill_placeholder(self, temp_dir):
        """get_placeholder_url() should return a skill placeholder for SKILL content type."""
        from src.library.cover_resolver import CoverResolver

        cache_dir = temp_dir / "library"
        cache_dir.mkdir(parents=True, exist_ok=True)

        resolver = CoverResolver(cache_dir=cache_dir)

        result = resolver.get_placeholder_url(ContentType.SKILL)

        assert result is not None
        assert "skill" in result.lower()
        assert result.endswith(".svg")

    def test_get_placeholder_url_returns_web_content_placeholder(self, temp_dir):
        """get_placeholder_url() should return a web_content placeholder for WEB_CONTENT content type."""
        from src.library.cover_resolver import CoverResolver

        cache_dir = temp_dir / "library"
        cache_dir.mkdir(parents=True, exist_ok=True)

        resolver = CoverResolver(cache_dir=cache_dir)

        result = resolver.get_placeholder_url(ContentType.WEB_CONTENT)

        assert result is not None
        assert "web_content" in result.lower()
        assert result.endswith(".svg")

    def test_get_placeholder_url_returns_distinct_placeholders_for_all_types(self, temp_dir):
        """get_placeholder_url() should return 10 distinct placeholder URLs for all content types."""
        from src.library.cover_resolver import CoverResolver

        cache_dir = temp_dir / "library"
        cache_dir.mkdir(parents=True, exist_ok=True)

        resolver = CoverResolver(cache_dir=cache_dir)

        # Get placeholders for all content types
        placeholders = set()
        for content_type in ContentType:
            url = resolver.get_placeholder_url(content_type)
            placeholders.add(url)

        # Should have 10 distinct placeholders
        assert len(placeholders) == 10

    def test_get_placeholder_url_returns_path_in_assets_directory(self, temp_dir):
        """get_placeholder_url() should return a path within the assets/images/placeholders directory."""
        from src.library.cover_resolver import CoverResolver

        cache_dir = temp_dir / "library"
        cache_dir.mkdir(parents=True, exist_ok=True)

        resolver = CoverResolver(cache_dir=cache_dir)

        result = resolver.get_placeholder_url(ContentType.BOOK)

        # The path should contain the placeholders directory
        assert "placeholders" in result

    def test_get_placeholder_url_returns_consistent_results(self, temp_dir):
        """get_placeholder_url() should return the same URL for the same content type."""
        from src.library.cover_resolver import CoverResolver

        cache_dir = temp_dir / "library"
        cache_dir.mkdir(parents=True, exist_ok=True)

        resolver = CoverResolver(cache_dir=cache_dir)

        # Call twice with the same content type
        result1 = resolver.get_placeholder_url(ContentType.FRAMEWORK)
        result2 = resolver.get_placeholder_url(ContentType.FRAMEWORK)

        assert result1 == result2

    def test_get_placeholder_url_uses_content_type_value_in_filename(self, temp_dir):
        """get_placeholder_url() should use the content type value as part of the filename."""
        from src.library.cover_resolver import CoverResolver

        cache_dir = temp_dir / "library"
        cache_dir.mkdir(parents=True, exist_ok=True)

        resolver = CoverResolver(cache_dir=cache_dir)

        # Each content type should have its value in the filename
        for content_type in ContentType:
            result = resolver.get_placeholder_url(content_type)
            assert content_type.value in result.lower()


class TestResolveCover:
    """Tests for resolve_cover() method that orchestrates cache -> API -> placeholder fallback."""

    def test_resolve_cover_returns_cached_url_if_exists(self, temp_dir):
        """resolve_cover() should return cached URL without making API calls."""
        from src.library.cover_resolver import CoverResolver

        cache_dir = temp_dir / "library"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Create a LibraryItem for a book
        item = LibraryItem(
            id="item_123",
            content_type=ContentType.BOOK,
            title="Thinking in Systems",
            summary="A primer on systems thinking",
            full_content="Full content here",
            source_url=None,
            cover_image_url=None,
            metadata={"isbn": "9781603580557"},
            themes=[],
            created_at=datetime.now(),
            highlights=[]
        )

        resolver = CoverResolver(cache_dir=cache_dir)
        # Pre-populate cache
        resolver.cache["item_123"] = {
            "url": "https://cached.example.com/cover.jpg",
            "resolved_at": "2024-01-15T10:30:00",
            "source": "open_library_isbn"
        }

        with patch("src.library.cover_resolver.requests.head") as mock_head:
            result = resolver.resolve_cover(item)

            # Should not make any API calls
            mock_head.assert_not_called()
            assert result == "https://cached.example.com/cover.jpg"

    def test_resolve_cover_for_book_with_isbn_tries_isbn_first(self, temp_dir):
        """resolve_cover() should try ISBN lookup first for books with ISBN."""
        from src.library.cover_resolver import CoverResolver

        cache_dir = temp_dir / "library"
        cache_dir.mkdir(parents=True, exist_ok=True)

        item = LibraryItem(
            id="item_456",
            content_type=ContentType.BOOK,
            title="Thinking in Systems",
            summary="A primer on systems thinking",
            full_content="Full content here",
            source_url=None,
            cover_image_url=None,
            metadata={"isbn": "9781603580557"},
            themes=[],
            created_at=datetime.now(),
            highlights=[]
        )

        resolver = CoverResolver(cache_dir=cache_dir)

        with patch("src.library.cover_resolver.requests.head") as mock_head:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.headers = {"content-type": "image/jpeg"}
            mock_head.return_value = mock_response

            result = resolver.resolve_cover(item)

            # Should call Open Library with ISBN
            expected_url = "https://covers.openlibrary.org/b/isbn/9781603580557-L.jpg"
            assert result == expected_url

    def test_resolve_cover_for_book_falls_back_to_title_when_isbn_fails(self, temp_dir):
        """resolve_cover() should try title lookup when ISBN lookup fails."""
        from src.library.cover_resolver import CoverResolver

        cache_dir = temp_dir / "library"
        cache_dir.mkdir(parents=True, exist_ok=True)

        item = LibraryItem(
            id="item_789",
            content_type=ContentType.BOOK,
            title="Thinking in Systems",
            summary="A primer on systems thinking",
            full_content="Full content here",
            source_url=None,
            cover_image_url=None,
            metadata={"isbn": "0000000000"},  # Invalid ISBN
            themes=[],
            created_at=datetime.now(),
            highlights=[]
        )

        resolver = CoverResolver(cache_dir=cache_dir)

        with patch("src.library.cover_resolver.requests.head") as mock_head:
            # First call (ISBN) fails, second call (title) succeeds
            mock_response_fail = Mock()
            mock_response_fail.status_code = 404

            mock_response_success = Mock()
            mock_response_success.status_code = 200
            mock_response_success.headers = {"content-type": "image/jpeg"}

            mock_head.side_effect = [mock_response_fail, mock_response_success]

            result = resolver.resolve_cover(item)

            # Should have made two calls: ISBN, then title
            assert mock_head.call_count == 2
            # Should return the title-based URL
            assert "title" in result
            assert "Thinking" in result

    def test_resolve_cover_for_book_falls_back_to_google_books_when_open_library_fails(self, temp_dir):
        """resolve_cover() should try Google Books when Open Library lookups fail."""
        from src.library.cover_resolver import CoverResolver

        cache_dir = temp_dir / "library"
        cache_dir.mkdir(parents=True, exist_ok=True)

        item = LibraryItem(
            id="item_101",
            content_type=ContentType.BOOK,
            title="Some Obscure Book",
            summary="A book not found on Open Library",
            full_content="Full content here",
            source_url=None,
            cover_image_url=None,
            metadata={"isbn": "1234567890"},
            themes=[],
            created_at=datetime.now(),
            highlights=[]
        )

        resolver = CoverResolver(cache_dir=cache_dir)

        with patch("src.library.cover_resolver.requests.head") as mock_head, \
             patch("src.library.cover_resolver.requests.get") as mock_get:
            # Both Open Library calls fail
            mock_head_response = Mock()
            mock_head_response.status_code = 404
            mock_head.return_value = mock_head_response

            # Google Books API succeeds
            mock_get_response = Mock()
            mock_get_response.status_code = 200
            mock_get_response.json.return_value = {
                "items": [
                    {
                        "volumeInfo": {
                            "imageLinks": {
                                "thumbnail": "https://books.google.com/books/content?id=xyz123"
                            }
                        }
                    }
                ]
            }
            mock_get.return_value = mock_get_response

            result = resolver.resolve_cover(item)

            # Should have tried Open Library (ISBN and title) then Google Books
            assert mock_head.call_count == 2  # ISBN and title
            assert mock_get.call_count == 1  # Google Books
            assert result == "https://books.google.com/books/content?id=xyz123"

    def test_resolve_cover_for_book_uses_placeholder_when_all_apis_fail(self, temp_dir):
        """resolve_cover() should use placeholder when all API lookups fail for books."""
        from src.library.cover_resolver import CoverResolver

        cache_dir = temp_dir / "library"
        cache_dir.mkdir(parents=True, exist_ok=True)

        item = LibraryItem(
            id="item_102",
            content_type=ContentType.BOOK,
            title="Completely Unknown Book",
            summary="A book not found anywhere",
            full_content="Full content here",
            source_url=None,
            cover_image_url=None,
            metadata={"isbn": "0000000000"},
            themes=[],
            created_at=datetime.now(),
            highlights=[]
        )

        resolver = CoverResolver(cache_dir=cache_dir)

        with patch("src.library.cover_resolver.requests.head") as mock_head, \
             patch("src.library.cover_resolver.requests.get") as mock_get:
            # All APIs fail
            mock_head_response = Mock()
            mock_head_response.status_code = 404
            mock_head.return_value = mock_head_response

            mock_get_response = Mock()
            mock_get_response.status_code = 200
            mock_get_response.json.return_value = {"totalItems": 0}
            mock_get.return_value = mock_get_response

            result = resolver.resolve_cover(item)

            # Should fall back to book placeholder
            assert "book" in result.lower()
            assert result.endswith(".svg")

    def test_resolve_cover_for_non_book_uses_placeholder_directly(self, temp_dir):
        """resolve_cover() should use placeholder for non-book content types without API calls."""
        from src.library.cover_resolver import CoverResolver

        cache_dir = temp_dir / "library"
        cache_dir.mkdir(parents=True, exist_ok=True)

        item = LibraryItem(
            id="item_200",
            content_type=ContentType.FRAMEWORK,
            title="Decision Making Framework",
            summary="A framework for making decisions",
            full_content="Full content here",
            source_url=None,
            cover_image_url=None,
            metadata={},
            themes=[],
            created_at=datetime.now(),
            highlights=[]
        )

        resolver = CoverResolver(cache_dir=cache_dir)

        with patch("src.library.cover_resolver.requests.head") as mock_head, \
             patch("src.library.cover_resolver.requests.get") as mock_get:

            result = resolver.resolve_cover(item)

            # Should not make any API calls for non-books
            mock_head.assert_not_called()
            mock_get.assert_not_called()

            # Should return framework placeholder
            assert "framework" in result.lower()
            assert result.endswith(".svg")

    def test_resolve_cover_caches_result_after_api_lookup(self, temp_dir):
        """resolve_cover() should cache the resolved URL after successful API lookup."""
        from src.library.cover_resolver import CoverResolver

        cache_dir = temp_dir / "library"
        cache_dir.mkdir(parents=True, exist_ok=True)

        item = LibraryItem(
            id="item_300",
            content_type=ContentType.BOOK,
            title="Test Book",
            summary="A test book",
            full_content="Full content here",
            source_url=None,
            cover_image_url=None,
            metadata={"isbn": "9781234567890"},
            themes=[],
            created_at=datetime.now(),
            highlights=[]
        )

        resolver = CoverResolver(cache_dir=cache_dir)
        assert "item_300" not in resolver.cache

        with patch("src.library.cover_resolver.requests.head") as mock_head:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.headers = {"content-type": "image/jpeg"}
            mock_head.return_value = mock_response

            result = resolver.resolve_cover(item)

            # Cache should be updated
            assert "item_300" in resolver.cache
            assert resolver.cache["item_300"]["url"] == result
            assert "resolved_at" in resolver.cache["item_300"]
            assert "source" in resolver.cache["item_300"]

    def test_resolve_cover_caches_placeholder_for_non_books(self, temp_dir):
        """resolve_cover() should cache the placeholder URL for non-book items."""
        from src.library.cover_resolver import CoverResolver

        cache_dir = temp_dir / "library"
        cache_dir.mkdir(parents=True, exist_ok=True)

        item = LibraryItem(
            id="item_301",
            content_type=ContentType.VALUE,
            title="Core Value",
            summary="An important value",
            full_content="Full content here",
            source_url=None,
            cover_image_url=None,
            metadata={},
            themes=[],
            created_at=datetime.now(),
            highlights=[]
        )

        resolver = CoverResolver(cache_dir=cache_dir)

        result = resolver.resolve_cover(item)

        # Cache should be updated with placeholder
        assert "item_301" in resolver.cache
        assert resolver.cache["item_301"]["url"] == result
        assert resolver.cache["item_301"]["source"] == "placeholder"

    def test_resolve_cover_records_correct_source_for_isbn_lookup(self, temp_dir):
        """resolve_cover() should record 'open_library_isbn' as source when ISBN lookup succeeds."""
        from src.library.cover_resolver import CoverResolver

        cache_dir = temp_dir / "library"
        cache_dir.mkdir(parents=True, exist_ok=True)

        item = LibraryItem(
            id="item_400",
            content_type=ContentType.BOOK,
            title="Test Book",
            summary="A test book",
            full_content="Full content here",
            source_url=None,
            cover_image_url=None,
            metadata={"isbn": "9781234567890"},
            themes=[],
            created_at=datetime.now(),
            highlights=[]
        )

        resolver = CoverResolver(cache_dir=cache_dir)

        with patch("src.library.cover_resolver.requests.head") as mock_head:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.headers = {"content-type": "image/jpeg"}
            mock_head.return_value = mock_response

            resolver.resolve_cover(item)

            assert resolver.cache["item_400"]["source"] == "open_library_isbn"

    def test_resolve_cover_records_correct_source_for_title_lookup(self, temp_dir):
        """resolve_cover() should record 'open_library_title' as source when title lookup succeeds."""
        from src.library.cover_resolver import CoverResolver

        cache_dir = temp_dir / "library"
        cache_dir.mkdir(parents=True, exist_ok=True)

        item = LibraryItem(
            id="item_401",
            content_type=ContentType.BOOK,
            title="Test Book",
            summary="A test book",
            full_content="Full content here",
            source_url=None,
            cover_image_url=None,
            metadata={"isbn": "0000000000"},  # Invalid ISBN
            themes=[],
            created_at=datetime.now(),
            highlights=[]
        )

        resolver = CoverResolver(cache_dir=cache_dir)

        with patch("src.library.cover_resolver.requests.head") as mock_head:
            # ISBN fails, title succeeds
            mock_response_fail = Mock()
            mock_response_fail.status_code = 404

            mock_response_success = Mock()
            mock_response_success.status_code = 200
            mock_response_success.headers = {"content-type": "image/jpeg"}

            mock_head.side_effect = [mock_response_fail, mock_response_success]

            resolver.resolve_cover(item)

            assert resolver.cache["item_401"]["source"] == "open_library_title"

    def test_resolve_cover_records_correct_source_for_google_books(self, temp_dir):
        """resolve_cover() should record 'google_books' as source when Google Books succeeds."""
        from src.library.cover_resolver import CoverResolver

        cache_dir = temp_dir / "library"
        cache_dir.mkdir(parents=True, exist_ok=True)

        item = LibraryItem(
            id="item_402",
            content_type=ContentType.BOOK,
            title="Test Book",
            summary="A test book",
            full_content="Full content here",
            source_url=None,
            cover_image_url=None,
            metadata={"isbn": "0000000000"},
            themes=[],
            created_at=datetime.now(),
            highlights=[]
        )

        resolver = CoverResolver(cache_dir=cache_dir)

        with patch("src.library.cover_resolver.requests.head") as mock_head, \
             patch("src.library.cover_resolver.requests.get") as mock_get:
            # Open Library fails
            mock_head_response = Mock()
            mock_head_response.status_code = 404
            mock_head.return_value = mock_head_response

            # Google Books succeeds
            mock_get_response = Mock()
            mock_get_response.status_code = 200
            mock_get_response.json.return_value = {
                "items": [
                    {
                        "volumeInfo": {
                            "imageLinks": {
                                "thumbnail": "https://books.google.com/thumbnail.jpg"
                            }
                        }
                    }
                ]
            }
            mock_get.return_value = mock_get_response

            resolver.resolve_cover(item)

            assert resolver.cache["item_402"]["source"] == "google_books"

    def test_resolve_cover_for_book_without_isbn_skips_isbn_lookup(self, temp_dir):
        """resolve_cover() should skip ISBN lookup for books without ISBN in metadata."""
        from src.library.cover_resolver import CoverResolver

        cache_dir = temp_dir / "library"
        cache_dir.mkdir(parents=True, exist_ok=True)

        item = LibraryItem(
            id="item_500",
            content_type=ContentType.BOOK,
            title="Book Without ISBN",
            summary="A book without ISBN",
            full_content="Full content here",
            source_url=None,
            cover_image_url=None,
            metadata={},  # No ISBN
            themes=[],
            created_at=datetime.now(),
            highlights=[]
        )

        resolver = CoverResolver(cache_dir=cache_dir)

        with patch("src.library.cover_resolver.requests.head") as mock_head:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.headers = {"content-type": "image/jpeg"}
            mock_head.return_value = mock_response

            result = resolver.resolve_cover(item)

            # Should only call once (title lookup), not twice (no ISBN lookup)
            assert mock_head.call_count == 1
            # URL should be title-based
            assert "title" in result

    def test_resolve_cover_saves_cache_to_disk(self, temp_dir):
        """resolve_cover() should persist cache to disk after resolving."""
        from src.library.cover_resolver import CoverResolver

        cache_dir = temp_dir / "library"
        cache_dir.mkdir(parents=True, exist_ok=True)

        item = LibraryItem(
            id="item_600",
            content_type=ContentType.ARTICLE,
            title="Test Article",
            summary="A test article",
            full_content="Full content here",
            source_url=None,
            cover_image_url=None,
            metadata={},
            themes=[],
            created_at=datetime.now(),
            highlights=[]
        )

        resolver = CoverResolver(cache_dir=cache_dir)
        resolver.resolve_cover(item)

        # Cache file should exist and contain the entry
        assert (cache_dir / "cover_cache.json").exists()

        with open(cache_dir / "cover_cache.json") as f:
            saved_cache = json.load(f)

        assert "item_600" in saved_cache

    def test_resolve_cover_handles_all_content_types(self, temp_dir):
        """resolve_cover() should return a placeholder for all non-book content types."""
        from src.library.cover_resolver import CoverResolver

        cache_dir = temp_dir / "library"
        cache_dir.mkdir(parents=True, exist_ok=True)

        resolver = CoverResolver(cache_dir=cache_dir)

        non_book_types = [
            ContentType.ARTICLE,
            ContentType.FRAMEWORK,
            ContentType.VALUE,
            ContentType.PREFERENCE,
            ContentType.MEMORY,
            ContentType.INSIGHT,
            ContentType.GOAL,
            ContentType.SKILL,
            ContentType.WEB_CONTENT,
        ]

        for content_type in non_book_types:
            item = LibraryItem(
                id=f"item_{content_type.value}",
                content_type=content_type,
                title=f"Test {content_type.value}",
                summary=f"A test {content_type.value}",
                full_content="Full content here",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime.now(),
                highlights=[]
            )

            result = resolver.resolve_cover(item)

            # Should return a placeholder path containing the content type
            assert content_type.value in result.lower()
            assert result.endswith(".svg")


class TestResolveAllCovers:
    """Tests for resolve_all_covers() method that processes multiple items in batch."""

    def test_resolve_all_covers_accepts_list_of_items(self, temp_dir):
        """resolve_all_covers() should accept a list of LibraryItems."""
        from src.library.cover_resolver import CoverResolver

        cache_dir = temp_dir / "library"
        cache_dir.mkdir(parents=True, exist_ok=True)

        resolver = CoverResolver(cache_dir=cache_dir)

        items = [
            LibraryItem(
                id="item_1",
                content_type=ContentType.FRAMEWORK,
                title="Test Framework",
                summary="A test framework",
                full_content="Full content",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime.now(),
                highlights=[]
            )
        ]

        # Should not raise an error
        result = resolver.resolve_all_covers(items)

        assert result is not None

    def test_resolve_all_covers_returns_list_of_items(self, temp_dir):
        """resolve_all_covers() should return a list of LibraryItems."""
        from src.library.cover_resolver import CoverResolver

        cache_dir = temp_dir / "library"
        cache_dir.mkdir(parents=True, exist_ok=True)

        resolver = CoverResolver(cache_dir=cache_dir)

        items = [
            LibraryItem(
                id="item_1",
                content_type=ContentType.VALUE,
                title="Test Value",
                summary="A test value",
                full_content="Full content",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime.now(),
                highlights=[]
            )
        ]

        result = resolver.resolve_all_covers(items)

        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], LibraryItem)

    def test_resolve_all_covers_populates_cover_image_url(self, temp_dir):
        """resolve_all_covers() should populate cover_image_url for each item."""
        from src.library.cover_resolver import CoverResolver

        cache_dir = temp_dir / "library"
        cache_dir.mkdir(parents=True, exist_ok=True)

        resolver = CoverResolver(cache_dir=cache_dir)

        items = [
            LibraryItem(
                id="item_1",
                content_type=ContentType.ARTICLE,
                title="Test Article",
                summary="A test article",
                full_content="Full content",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime.now(),
                highlights=[]
            )
        ]

        result = resolver.resolve_all_covers(items)

        assert result[0].cover_image_url is not None
        assert "article" in result[0].cover_image_url.lower()

    def test_resolve_all_covers_processes_multiple_items(self, temp_dir):
        """resolve_all_covers() should process all items in the list."""
        from src.library.cover_resolver import CoverResolver

        cache_dir = temp_dir / "library"
        cache_dir.mkdir(parents=True, exist_ok=True)

        resolver = CoverResolver(cache_dir=cache_dir)

        items = [
            LibraryItem(
                id="item_1",
                content_type=ContentType.FRAMEWORK,
                title="Framework 1",
                summary="First framework",
                full_content="Full content",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime.now(),
                highlights=[]
            ),
            LibraryItem(
                id="item_2",
                content_type=ContentType.VALUE,
                title="Value 1",
                summary="First value",
                full_content="Full content",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime.now(),
                highlights=[]
            ),
            LibraryItem(
                id="item_3",
                content_type=ContentType.INSIGHT,
                title="Insight 1",
                summary="First insight",
                full_content="Full content",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime.now(),
                highlights=[]
            )
        ]

        result = resolver.resolve_all_covers(items)

        assert len(result) == 3
        assert all(item.cover_image_url is not None for item in result)

    def test_resolve_all_covers_handles_empty_list(self, temp_dir):
        """resolve_all_covers() should handle an empty list gracefully."""
        from src.library.cover_resolver import CoverResolver

        cache_dir = temp_dir / "library"
        cache_dir.mkdir(parents=True, exist_ok=True)

        resolver = CoverResolver(cache_dir=cache_dir)

        result = resolver.resolve_all_covers([])

        assert result == []

    def test_resolve_all_covers_preserves_item_order(self, temp_dir):
        """resolve_all_covers() should preserve the order of items."""
        from src.library.cover_resolver import CoverResolver

        cache_dir = temp_dir / "library"
        cache_dir.mkdir(parents=True, exist_ok=True)

        resolver = CoverResolver(cache_dir=cache_dir)

        items = [
            LibraryItem(
                id="first",
                content_type=ContentType.GOAL,
                title="First",
                summary="First item",
                full_content="Full content",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime.now(),
                highlights=[]
            ),
            LibraryItem(
                id="second",
                content_type=ContentType.SKILL,
                title="Second",
                summary="Second item",
                full_content="Full content",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime.now(),
                highlights=[]
            ),
            LibraryItem(
                id="third",
                content_type=ContentType.MEMORY,
                title="Third",
                summary="Third item",
                full_content="Full content",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime.now(),
                highlights=[]
            )
        ]

        result = resolver.resolve_all_covers(items)

        assert result[0].id == "first"
        assert result[1].id == "second"
        assert result[2].id == "third"

    def test_resolve_all_covers_uses_resolve_cover_for_each_item(self, temp_dir):
        """resolve_all_covers() should call resolve_cover() for each item."""
        from src.library.cover_resolver import CoverResolver

        cache_dir = temp_dir / "library"
        cache_dir.mkdir(parents=True, exist_ok=True)

        resolver = CoverResolver(cache_dir=cache_dir)

        items = [
            LibraryItem(
                id="item_1",
                content_type=ContentType.PREFERENCE,
                title="Preference 1",
                summary="First preference",
                full_content="Full content",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime.now(),
                highlights=[]
            ),
            LibraryItem(
                id="item_2",
                content_type=ContentType.WEB_CONTENT,
                title="Web Content 1",
                summary="First web content",
                full_content="Full content",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime.now(),
                highlights=[]
            )
        ]

        with patch.object(resolver, 'resolve_cover') as mock_resolve_cover:
            # Return a placeholder URL for each call
            mock_resolve_cover.side_effect = [
                "assets/images/placeholders/preference.svg",
                "assets/images/placeholders/web_content.svg"
            ]

            resolver.resolve_all_covers(items)

            assert mock_resolve_cover.call_count == 2

    def test_resolve_all_covers_handles_mixed_content_types(self, temp_dir):
        """resolve_all_covers() should handle books and non-books in the same batch."""
        from src.library.cover_resolver import CoverResolver

        cache_dir = temp_dir / "library"
        cache_dir.mkdir(parents=True, exist_ok=True)

        resolver = CoverResolver(cache_dir=cache_dir)

        items = [
            LibraryItem(
                id="book_1",
                content_type=ContentType.BOOK,
                title="Test Book",
                summary="A test book",
                full_content="Full content",
                source_url=None,
                cover_image_url=None,
                metadata={"isbn": "9781234567890"},
                themes=[],
                created_at=datetime.now(),
                highlights=[]
            ),
            LibraryItem(
                id="framework_1",
                content_type=ContentType.FRAMEWORK,
                title="Test Framework",
                summary="A test framework",
                full_content="Full content",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime.now(),
                highlights=[]
            )
        ]

        with patch("src.library.cover_resolver.requests.head") as mock_head:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.headers = {"content-type": "image/jpeg"}
            mock_head.return_value = mock_response

            result = resolver.resolve_all_covers(items)

            # Book should have API-resolved cover
            assert "isbn" in result[0].cover_image_url or "openlibrary" in result[0].cover_image_url
            # Framework should have placeholder
            assert "framework" in result[1].cover_image_url.lower()

    def test_resolve_all_covers_caches_all_items(self, temp_dir):
        """resolve_all_covers() should cache all resolved covers."""
        from src.library.cover_resolver import CoverResolver

        cache_dir = temp_dir / "library"
        cache_dir.mkdir(parents=True, exist_ok=True)

        resolver = CoverResolver(cache_dir=cache_dir)

        items = [
            LibraryItem(
                id="cache_test_1",
                content_type=ContentType.ARTICLE,
                title="Article 1",
                summary="First article",
                full_content="Full content",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime.now(),
                highlights=[]
            ),
            LibraryItem(
                id="cache_test_2",
                content_type=ContentType.INSIGHT,
                title="Insight 1",
                summary="First insight",
                full_content="Full content",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime.now(),
                highlights=[]
            )
        ]

        resolver.resolve_all_covers(items)

        # Both items should be in the cache
        assert "cache_test_1" in resolver.cache
        assert "cache_test_2" in resolver.cache

    def test_resolve_all_covers_uses_existing_cache(self, temp_dir):
        """resolve_all_covers() should use cached values for already-resolved items."""
        from src.library.cover_resolver import CoverResolver

        cache_dir = temp_dir / "library"
        cache_dir.mkdir(parents=True, exist_ok=True)

        resolver = CoverResolver(cache_dir=cache_dir)

        # Pre-populate cache for one item
        resolver.cache["cached_item"] = {
            "url": "https://cached.example.com/cover.jpg",
            "resolved_at": "2024-01-15T10:30:00",
            "source": "open_library_isbn"
        }

        items = [
            LibraryItem(
                id="cached_item",
                content_type=ContentType.BOOK,
                title="Cached Book",
                summary="A cached book",
                full_content="Full content",
                source_url=None,
                cover_image_url=None,
                metadata={"isbn": "1234567890"},
                themes=[],
                created_at=datetime.now(),
                highlights=[]
            ),
            LibraryItem(
                id="new_item",
                content_type=ContentType.FRAMEWORK,
                title="New Framework",
                summary="A new framework",
                full_content="Full content",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime.now(),
                highlights=[]
            )
        ]

        with patch("src.library.cover_resolver.requests.head") as mock_head:
            result = resolver.resolve_all_covers(items)

            # Should not make API calls for cached item
            # (only the framework would potentially need API, but it's non-book)
            mock_head.assert_not_called()

            # Cached item should use cached URL
            assert result[0].cover_image_url == "https://cached.example.com/cover.jpg"
            # New item should have placeholder
            assert "framework" in result[1].cover_image_url.lower()

    def test_resolve_all_covers_preserves_other_item_fields(self, temp_dir):
        """resolve_all_covers() should not modify other fields of the LibraryItem."""
        from src.library.cover_resolver import CoverResolver

        cache_dir = temp_dir / "library"
        cache_dir.mkdir(parents=True, exist_ok=True)

        resolver = CoverResolver(cache_dir=cache_dir)

        original_created_at = datetime.now()
        items = [
            LibraryItem(
                id="preserve_test",
                content_type=ContentType.VALUE,
                title="Original Title",
                summary="Original Summary",
                full_content="Original Full Content",
                source_url="https://example.com/source",
                cover_image_url=None,
                metadata={"key": "value"},
                themes=["theme1", "theme2"],
                created_at=original_created_at,
                highlights=["highlight1"]
            )
        ]

        result = resolver.resolve_all_covers(items)

        # All other fields should be preserved
        assert result[0].id == "preserve_test"
        assert result[0].content_type == ContentType.VALUE
        assert result[0].title == "Original Title"
        assert result[0].summary == "Original Summary"
        assert result[0].full_content == "Original Full Content"
        assert result[0].source_url == "https://example.com/source"
        assert result[0].metadata == {"key": "value"}
        assert result[0].themes == ["theme1", "theme2"]
        assert result[0].created_at == original_created_at
        assert result[0].highlights == ["highlight1"]
        # And cover_image_url should be set
        assert result[0].cover_image_url is not None

    def test_resolve_all_covers_does_not_modify_original_items(self, temp_dir):
        """resolve_all_covers() should not modify the original input items."""
        from src.library.cover_resolver import CoverResolver

        cache_dir = temp_dir / "library"
        cache_dir.mkdir(parents=True, exist_ok=True)

        resolver = CoverResolver(cache_dir=cache_dir)

        original_item = LibraryItem(
            id="original_item",
            content_type=ContentType.GOAL,
            title="Original Goal",
            summary="A goal",
            full_content="Full content",
            source_url=None,
            cover_image_url=None,
            metadata={},
            themes=[],
            created_at=datetime.now(),
            highlights=[]
        )

        result = resolver.resolve_all_covers([original_item])

        # Original item should be unchanged
        assert original_item.cover_image_url is None
        # Result item should have cover_image_url set
        assert result[0].cover_image_url is not None

    def test_resolve_all_covers_continues_on_individual_failures(self, temp_dir):
        """resolve_all_covers() should continue processing if one item fails."""
        from src.library.cover_resolver import CoverResolver

        cache_dir = temp_dir / "library"
        cache_dir.mkdir(parents=True, exist_ok=True)

        resolver = CoverResolver(cache_dir=cache_dir)

        items = [
            LibraryItem(
                id="item_1",
                content_type=ContentType.BOOK,
                title="Book 1",
                summary="First book",
                full_content="Full content",
                source_url=None,
                cover_image_url=None,
                metadata={"isbn": "bad_isbn"},
                themes=[],
                created_at=datetime.now(),
                highlights=[]
            ),
            LibraryItem(
                id="item_2",
                content_type=ContentType.FRAMEWORK,
                title="Framework 1",
                summary="First framework",
                full_content="Full content",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime.now(),
                highlights=[]
            )
        ]

        with patch("src.library.cover_resolver.requests.head") as mock_head, \
             patch("src.library.cover_resolver.requests.get") as mock_get:
            # All API calls fail for the book
            mock_head_response = Mock()
            mock_head_response.status_code = 404
            mock_head.return_value = mock_head_response

            mock_get_response = Mock()
            mock_get_response.status_code = 200
            mock_get_response.json.return_value = {"totalItems": 0}
            mock_get.return_value = mock_get_response

            result = resolver.resolve_all_covers(items)

            # Should return both items with covers (book with placeholder, framework with placeholder)
            assert len(result) == 2
            assert result[0].cover_image_url is not None  # Falls back to placeholder
            assert result[1].cover_image_url is not None  # Framework placeholder

    def test_resolve_all_covers_saves_cache_once_at_end(self, temp_dir):
        """resolve_all_covers() should save cache efficiently (not after every item)."""
        from src.library.cover_resolver import CoverResolver

        cache_dir = temp_dir / "library"
        cache_dir.mkdir(parents=True, exist_ok=True)

        resolver = CoverResolver(cache_dir=cache_dir)

        items = [
            LibraryItem(
                id=f"item_{i}",
                content_type=ContentType.ARTICLE,
                title=f"Article {i}",
                summary=f"Article summary {i}",
                full_content="Full content",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime.now(),
                highlights=[]
            )
            for i in range(5)
        ]

        with patch.object(resolver, '_save_cache') as mock_save_cache:
            resolver.resolve_all_covers(items)

            # Should save cache once at the end, not 5 times
            assert mock_save_cache.call_count == 1


class TestApiErrorHandling:
    """Tests for API error handling with exponential backoff."""

    def test_fetch_cover_by_isbn_retries_on_429_rate_limit(self, temp_dir):
        """_fetch_cover_by_isbn() should retry with backoff on 429 rate limit response."""
        from src.library.cover_resolver import CoverResolver

        cache_dir = temp_dir / "library"
        cache_dir.mkdir(parents=True, exist_ok=True)

        resolver = CoverResolver(cache_dir=cache_dir)

        with patch("src.library.cover_resolver.requests.head") as mock_head, \
             patch("time.sleep") as mock_sleep:
            # First call returns 429 (rate limited), second call succeeds
            mock_response_429 = Mock()
            mock_response_429.status_code = 429
            mock_response_429.headers = {}

            mock_response_success = Mock()
            mock_response_success.status_code = 200
            mock_response_success.headers = {"content-type": "image/jpeg"}

            mock_head.side_effect = [mock_response_429, mock_response_success]

            result = resolver._fetch_cover_by_isbn("9781234567890")

            # Should have retried after backoff
            assert mock_head.call_count == 2
            # Should have slept between retries
            assert mock_sleep.call_count >= 1
            # Should eventually return the URL
            assert result is not None
            assert "9781234567890" in result

    def test_fetch_cover_by_isbn_uses_exponential_backoff(self, temp_dir):
        """_fetch_cover_by_isbn() should use exponential backoff between retries."""
        from src.library.cover_resolver import CoverResolver

        cache_dir = temp_dir / "library"
        cache_dir.mkdir(parents=True, exist_ok=True)

        resolver = CoverResolver(cache_dir=cache_dir)

        with patch("src.library.cover_resolver.requests.head") as mock_head, \
             patch("time.sleep") as mock_sleep:
            # Multiple 429s then success
            mock_response_429 = Mock()
            mock_response_429.status_code = 429
            mock_response_429.headers = {}

            mock_response_success = Mock()
            mock_response_success.status_code = 200
            mock_response_success.headers = {"content-type": "image/jpeg"}

            mock_head.side_effect = [mock_response_429, mock_response_429, mock_response_success]

            result = resolver._fetch_cover_by_isbn("9781234567890")

            # Should have called sleep multiple times with increasing delays
            assert mock_sleep.call_count >= 2
            sleep_calls = [call[0][0] for call in mock_sleep.call_args_list]
            # Each delay should be >= the previous (exponential)
            for i in range(1, len(sleep_calls)):
                assert sleep_calls[i] >= sleep_calls[i - 1]

    def test_fetch_cover_by_isbn_gives_up_after_max_retries(self, temp_dir):
        """_fetch_cover_by_isbn() should return None after max retries on persistent 429."""
        from src.library.cover_resolver import CoverResolver

        cache_dir = temp_dir / "library"
        cache_dir.mkdir(parents=True, exist_ok=True)

        resolver = CoverResolver(cache_dir=cache_dir)

        with patch("src.library.cover_resolver.requests.head") as mock_head, \
             patch("time.sleep") as mock_sleep:
            # Always return 429
            mock_response_429 = Mock()
            mock_response_429.status_code = 429
            mock_response_429.headers = {}
            mock_head.return_value = mock_response_429

            result = resolver._fetch_cover_by_isbn("9781234567890")

            # Should eventually give up and return None
            assert result is None
            # Should have made multiple attempts (but not infinite)
            assert mock_head.call_count <= 5  # Max retries should be capped

    def test_fetch_cover_by_isbn_retries_on_503_service_unavailable(self, temp_dir):
        """_fetch_cover_by_isbn() should retry on 503 service unavailable."""
        from src.library.cover_resolver import CoverResolver

        cache_dir = temp_dir / "library"
        cache_dir.mkdir(parents=True, exist_ok=True)

        resolver = CoverResolver(cache_dir=cache_dir)

        with patch("src.library.cover_resolver.requests.head") as mock_head, \
             patch("time.sleep") as mock_sleep:
            # First call returns 503, second succeeds
            mock_response_503 = Mock()
            mock_response_503.status_code = 503
            mock_response_503.headers = {}

            mock_response_success = Mock()
            mock_response_success.status_code = 200
            mock_response_success.headers = {"content-type": "image/jpeg"}

            mock_head.side_effect = [mock_response_503, mock_response_success]

            result = resolver._fetch_cover_by_isbn("9781234567890")

            # Should have retried
            assert mock_head.call_count == 2
            assert result is not None

    def test_fetch_cover_by_title_retries_on_rate_limit(self, temp_dir):
        """_fetch_cover_by_title() should retry with backoff on rate limit."""
        from src.library.cover_resolver import CoverResolver

        cache_dir = temp_dir / "library"
        cache_dir.mkdir(parents=True, exist_ok=True)

        resolver = CoverResolver(cache_dir=cache_dir)

        with patch("src.library.cover_resolver.requests.head") as mock_head, \
             patch("time.sleep") as mock_sleep:
            # First call returns 429, second succeeds
            mock_response_429 = Mock()
            mock_response_429.status_code = 429
            mock_response_429.headers = {}

            mock_response_success = Mock()
            mock_response_success.status_code = 200
            mock_response_success.headers = {"content-type": "image/jpeg"}

            mock_head.side_effect = [mock_response_429, mock_response_success]

            result = resolver._fetch_cover_by_title("Thinking in Systems")

            assert mock_head.call_count == 2
            assert mock_sleep.call_count >= 1
            assert result is not None

    def test_fetch_cover_by_title_gives_up_after_max_retries(self, temp_dir):
        """_fetch_cover_by_title() should return None after max retries."""
        from src.library.cover_resolver import CoverResolver

        cache_dir = temp_dir / "library"
        cache_dir.mkdir(parents=True, exist_ok=True)

        resolver = CoverResolver(cache_dir=cache_dir)

        with patch("src.library.cover_resolver.requests.head") as mock_head, \
             patch("time.sleep") as mock_sleep:
            # Always return 429
            mock_response_429 = Mock()
            mock_response_429.status_code = 429
            mock_response_429.headers = {}
            mock_head.return_value = mock_response_429

            result = resolver._fetch_cover_by_title("Thinking in Systems")

            assert result is None
            assert mock_head.call_count <= 5

    def test_fetch_cover_from_google_books_retries_on_rate_limit(self, temp_dir):
        """_fetch_cover_from_google_books() should retry with backoff on rate limit."""
        from src.library.cover_resolver import CoverResolver

        cache_dir = temp_dir / "library"
        cache_dir.mkdir(parents=True, exist_ok=True)

        resolver = CoverResolver(cache_dir=cache_dir)

        with patch("src.library.cover_resolver.requests.get") as mock_get, \
             patch("time.sleep") as mock_sleep:
            # First call returns 429, second succeeds
            mock_response_429 = Mock()
            mock_response_429.status_code = 429
            mock_response_429.headers = {}

            mock_response_success = Mock()
            mock_response_success.status_code = 200
            mock_response_success.json.return_value = {
                "items": [{
                    "volumeInfo": {
                        "imageLinks": {"thumbnail": "https://books.google.com/thumb.jpg"}
                    }
                }]
            }

            mock_get.side_effect = [mock_response_429, mock_response_success]

            result = resolver._fetch_cover_from_google_books("Test Book")

            assert mock_get.call_count == 2
            assert mock_sleep.call_count >= 1
            assert result == "https://books.google.com/thumb.jpg"

    def test_fetch_cover_from_google_books_gives_up_after_max_retries(self, temp_dir):
        """_fetch_cover_from_google_books() should return None after max retries."""
        from src.library.cover_resolver import CoverResolver

        cache_dir = temp_dir / "library"
        cache_dir.mkdir(parents=True, exist_ok=True)

        resolver = CoverResolver(cache_dir=cache_dir)

        with patch("src.library.cover_resolver.requests.get") as mock_get, \
             patch("time.sleep") as mock_sleep:
            # Always return 429
            mock_response_429 = Mock()
            mock_response_429.status_code = 429
            mock_response_429.headers = {}
            mock_get.return_value = mock_response_429

            result = resolver._fetch_cover_from_google_books("Test Book")

            assert result is None
            assert mock_get.call_count <= 5

    def test_resolve_cover_gracefully_handles_all_apis_rate_limited(self, temp_dir):
        """resolve_cover() should return placeholder when all APIs are rate limited."""
        from src.library.cover_resolver import CoverResolver

        cache_dir = temp_dir / "library"
        cache_dir.mkdir(parents=True, exist_ok=True)

        resolver = CoverResolver(cache_dir=cache_dir)

        item = LibraryItem(
            id="rate_limited_book",
            content_type=ContentType.BOOK,
            title="Rate Limited Book",
            summary="A book when APIs are rate limited",
            full_content="Full content",
            source_url=None,
            cover_image_url=None,
            metadata={"isbn": "9781234567890"},
            themes=[],
            created_at=datetime.now(),
            highlights=[]
        )

        with patch("src.library.cover_resolver.requests.head") as mock_head, \
             patch("src.library.cover_resolver.requests.get") as mock_get, \
             patch("time.sleep") as mock_sleep:
            # All APIs return 429
            mock_response_429 = Mock()
            mock_response_429.status_code = 429
            mock_response_429.headers = {}
            mock_head.return_value = mock_response_429
            mock_get.return_value = mock_response_429

            result = resolver.resolve_cover(item)

            # Should fall back to placeholder gracefully
            assert result is not None
            assert "book" in result.lower()
            assert result.endswith(".svg")

    def test_resolve_all_covers_continues_when_apis_rate_limited(self, temp_dir):
        """resolve_all_covers() should continue processing when APIs are rate limited."""
        from src.library.cover_resolver import CoverResolver

        cache_dir = temp_dir / "library"
        cache_dir.mkdir(parents=True, exist_ok=True)

        resolver = CoverResolver(cache_dir=cache_dir)

        items = [
            LibraryItem(
                id="book_1",
                content_type=ContentType.BOOK,
                title="Book 1",
                summary="First book",
                full_content="Full content",
                source_url=None,
                cover_image_url=None,
                metadata={"isbn": "1111111111"},
                themes=[],
                created_at=datetime.now(),
                highlights=[]
            ),
            LibraryItem(
                id="book_2",
                content_type=ContentType.BOOK,
                title="Book 2",
                summary="Second book",
                full_content="Full content",
                source_url=None,
                cover_image_url=None,
                metadata={"isbn": "2222222222"},
                themes=[],
                created_at=datetime.now(),
                highlights=[]
            ),
            LibraryItem(
                id="framework_1",
                content_type=ContentType.FRAMEWORK,
                title="Framework 1",
                summary="A framework",
                full_content="Full content",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime.now(),
                highlights=[]
            )
        ]

        with patch("src.library.cover_resolver.requests.head") as mock_head, \
             patch("src.library.cover_resolver.requests.get") as mock_get, \
             patch("time.sleep") as mock_sleep:
            # All APIs return 429
            mock_response_429 = Mock()
            mock_response_429.status_code = 429
            mock_response_429.headers = {}
            mock_head.return_value = mock_response_429
            mock_get.return_value = mock_response_429

            result = resolver.resolve_all_covers(items)

            # All items should have covers (placeholders for rate-limited books)
            assert len(result) == 3
            assert all(item.cover_image_url is not None for item in result)
            # Books should fall back to book placeholder
            assert "book" in result[0].cover_image_url.lower()
            assert "book" in result[1].cover_image_url.lower()
            # Framework should have its own placeholder
            assert "framework" in result[2].cover_image_url.lower()

    def test_fetch_cover_by_isbn_does_not_retry_on_404(self, temp_dir):
        """_fetch_cover_by_isbn() should not retry on 404 (not found is permanent)."""
        from src.library.cover_resolver import CoverResolver

        cache_dir = temp_dir / "library"
        cache_dir.mkdir(parents=True, exist_ok=True)

        resolver = CoverResolver(cache_dir=cache_dir)

        with patch("src.library.cover_resolver.requests.head") as mock_head, \
             patch("time.sleep") as mock_sleep:
            mock_response_404 = Mock()
            mock_response_404.status_code = 404
            mock_head.return_value = mock_response_404

            result = resolver._fetch_cover_by_isbn("0000000000")

            # Should not retry on 404 - just return None immediately
            assert mock_head.call_count == 1
            assert mock_sleep.call_count == 0
            assert result is None

    def test_fetch_cover_by_isbn_retries_on_connection_error(self, temp_dir):
        """_fetch_cover_by_isbn() should retry on connection errors."""
        from src.library.cover_resolver import CoverResolver
        import requests

        cache_dir = temp_dir / "library"
        cache_dir.mkdir(parents=True, exist_ok=True)

        resolver = CoverResolver(cache_dir=cache_dir)

        with patch("src.library.cover_resolver.requests.head") as mock_head, \
             patch("time.sleep") as mock_sleep:
            mock_response_success = Mock()
            mock_response_success.status_code = 200
            mock_response_success.headers = {"content-type": "image/jpeg"}

            # First call raises connection error, second succeeds
            mock_head.side_effect = [
                requests.ConnectionError("Connection refused"),
                mock_response_success
            ]

            result = resolver._fetch_cover_by_isbn("9781234567890")

            # Should have retried after the connection error
            assert mock_head.call_count == 2
            assert mock_sleep.call_count >= 1
            assert result is not None

    def test_exponential_backoff_caps_at_maximum_delay(self, temp_dir):
        """Exponential backoff should cap at a reasonable maximum delay."""
        from src.library.cover_resolver import CoverResolver

        cache_dir = temp_dir / "library"
        cache_dir.mkdir(parents=True, exist_ok=True)

        resolver = CoverResolver(cache_dir=cache_dir)

        with patch("src.library.cover_resolver.requests.head") as mock_head, \
             patch("time.sleep") as mock_sleep:
            # Always return 429 to force maximum retries
            mock_response_429 = Mock()
            mock_response_429.status_code = 429
            mock_response_429.headers = {}
            mock_head.return_value = mock_response_429

            resolver._fetch_cover_by_isbn("9781234567890")

            # Check that sleep delays are capped (no delay > 30 seconds)
            for call in mock_sleep.call_args_list:
                delay = call[0][0]
                assert delay <= 30, f"Backoff delay {delay} exceeds maximum of 30 seconds"

    def test_retries_respect_retry_after_header(self, temp_dir):
        """_fetch_cover_by_isbn() should respect Retry-After header if present."""
        from src.library.cover_resolver import CoverResolver

        cache_dir = temp_dir / "library"
        cache_dir.mkdir(parents=True, exist_ok=True)

        resolver = CoverResolver(cache_dir=cache_dir)

        with patch("src.library.cover_resolver.requests.head") as mock_head, \
             patch("time.sleep") as mock_sleep:
            # First call returns 429 with Retry-After header
            mock_response_429 = Mock()
            mock_response_429.status_code = 429
            mock_response_429.headers = {"Retry-After": "5"}

            mock_response_success = Mock()
            mock_response_success.status_code = 200
            mock_response_success.headers = {"content-type": "image/jpeg"}

            mock_head.side_effect = [mock_response_429, mock_response_success]

            result = resolver._fetch_cover_by_isbn("9781234567890")

            # Should have waited at least 5 seconds (or more due to backoff)
            assert mock_sleep.call_count >= 1
            first_sleep = mock_sleep.call_args_list[0][0][0]
            assert first_sleep >= 5, f"Should wait at least 5 seconds per Retry-After, got {first_sleep}"
