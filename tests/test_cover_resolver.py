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
