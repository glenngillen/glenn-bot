"""Tests for library server module.

This module tests the LibraryServer class which provides a simple HTTP server
for serving the generated static library site locally.

Test classes are organized by the methods they test:
- TestLibraryServerInitialization: LibraryServer class initialization
- TestServe: serve() starting HTTP server
- TestPortHandling: port-in-use handling and fallback logic
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, call
import socket
import threading
import time


class TestLibraryServerInitialization:
    """Tests for LibraryServer class initialization."""

    def test_library_server_can_be_imported(self):
        """LibraryServer class should be importable from the module."""
        from src.library.server import LibraryServer

        assert LibraryServer is not None

    def test_library_server_init_with_site_dir(self, temp_dir):
        """LibraryServer should accept site_dir as a required argument."""
        from src.library.server import LibraryServer

        site_dir = temp_dir / "library-site"
        site_dir.mkdir(parents=True, exist_ok=True)

        server = LibraryServer(site_dir=site_dir)

        assert server.site_dir == site_dir

    def test_library_server_init_with_port(self, temp_dir):
        """LibraryServer should accept port as an optional argument."""
        from src.library.server import LibraryServer

        site_dir = temp_dir / "library-site"
        site_dir.mkdir(parents=True, exist_ok=True)

        server = LibraryServer(site_dir=site_dir, port=9000)

        assert server.port == 9000

    def test_library_server_init_default_port(self, temp_dir):
        """LibraryServer should use default port 8080 if not specified."""
        from src.library.server import LibraryServer

        site_dir = temp_dir / "library-site"
        site_dir.mkdir(parents=True, exist_ok=True)

        server = LibraryServer(site_dir=site_dir)

        assert server.port == 8080

    def test_library_server_init_accepts_string_path(self, temp_dir):
        """LibraryServer should accept string path and convert to Path object."""
        from src.library.server import LibraryServer

        site_dir = str(temp_dir / "library-site")
        Path(site_dir).mkdir(parents=True, exist_ok=True)

        server = LibraryServer(site_dir=site_dir)

        assert isinstance(server.site_dir, Path)

    def test_library_server_init_raises_if_site_dir_not_exists(self, temp_dir):
        """LibraryServer should raise ValueError if site_dir doesn't exist."""
        from src.library.server import LibraryServer

        site_dir = temp_dir / "nonexistent-site"

        with pytest.raises(ValueError, match="does not exist"):
            LibraryServer(site_dir=site_dir)

    def test_library_server_has_httpd_attribute_initially_none(self, temp_dir):
        """LibraryServer should have httpd attribute initialized to None."""
        from src.library.server import LibraryServer

        site_dir = temp_dir / "library-site"
        site_dir.mkdir(parents=True, exist_ok=True)

        server = LibraryServer(site_dir=site_dir)

        assert hasattr(server, "httpd")
        assert server.httpd is None
