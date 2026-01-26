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


class TestServe:
    """Tests for serve() starting HTTP server."""

    def test_serve_method_exists(self, temp_dir):
        """LibraryServer should have a serve() method."""
        from src.library.server import LibraryServer

        site_dir = temp_dir / "library-site"
        site_dir.mkdir(parents=True, exist_ok=True)

        server = LibraryServer(site_dir=site_dir)

        assert hasattr(server, "serve")
        assert callable(server.serve)

    def test_serve_starts_http_server(self, temp_dir):
        """serve() should start an HTTP server and set httpd attribute."""
        from src.library.server import LibraryServer

        site_dir = temp_dir / "library-site"
        site_dir.mkdir(parents=True, exist_ok=True)
        # Create an index.html file
        (site_dir / "index.html").write_text("<html><body>Test</body></html>")

        server = LibraryServer(site_dir=site_dir, port=18080)

        # Start server in a thread
        server_thread = threading.Thread(target=server.serve, kwargs={"blocking": False})
        server_thread.daemon = True
        server_thread.start()

        # Give server time to start
        time.sleep(0.5)

        try:
            assert server.httpd is not None
        finally:
            if server.httpd:
                server.httpd.shutdown()

    def test_serve_serves_files_from_site_dir(self, temp_dir):
        """serve() should serve files from the site directory."""
        import urllib.request
        from src.library.server import LibraryServer

        site_dir = temp_dir / "library-site"
        site_dir.mkdir(parents=True, exist_ok=True)
        (site_dir / "index.html").write_text("<html><body>Hello Library</body></html>")

        server = LibraryServer(site_dir=site_dir, port=18081)

        # Start server in a thread
        server_thread = threading.Thread(target=server.serve, kwargs={"blocking": False})
        server_thread.daemon = True
        server_thread.start()

        # Give server time to start
        time.sleep(0.5)

        try:
            # Request the index page
            response = urllib.request.urlopen("http://localhost:18081/index.html")
            content = response.read().decode("utf-8")
            assert "Hello Library" in content
        finally:
            if server.httpd:
                server.httpd.shutdown()

    def test_serve_returns_server_url(self, temp_dir):
        """serve() should return the URL where the server is accessible."""
        from src.library.server import LibraryServer

        site_dir = temp_dir / "library-site"
        site_dir.mkdir(parents=True, exist_ok=True)
        (site_dir / "index.html").write_text("<html><body>Test</body></html>")

        server = LibraryServer(site_dir=site_dir, port=18082)

        # Start server in a thread and capture the URL
        result = {"url": None}

        def serve_and_capture():
            result["url"] = server.serve(blocking=False)

        server_thread = threading.Thread(target=serve_and_capture)
        server_thread.daemon = True
        server_thread.start()

        # Give server time to start
        time.sleep(0.5)

        try:
            assert result["url"] == "http://localhost:18082"
        finally:
            if server.httpd:
                server.httpd.shutdown()

    def test_serve_with_blocking_true_blocks_until_interrupted(self, temp_dir):
        """serve() with blocking=True should block execution."""
        from src.library.server import LibraryServer

        site_dir = temp_dir / "library-site"
        site_dir.mkdir(parents=True, exist_ok=True)
        (site_dir / "index.html").write_text("<html><body>Test</body></html>")

        server = LibraryServer(site_dir=site_dir, port=18083)

        # Track if serve() blocks
        completed = {"value": False}

        def serve_blocking():
            server.serve(blocking=True)
            completed["value"] = True

        server_thread = threading.Thread(target=serve_blocking)
        server_thread.daemon = True
        server_thread.start()

        # Give server time to start
        time.sleep(0.5)

        try:
            # Server should still be running (blocking)
            assert not completed["value"]
            # Shutdown the server to unblock
            if server.httpd:
                server.httpd.shutdown()
            server_thread.join(timeout=2.0)
            assert completed["value"]
        finally:
            if server.httpd:
                server.httpd.shutdown()

    def test_serve_has_stop_method(self, temp_dir):
        """LibraryServer should have a stop() method to stop the server."""
        from src.library.server import LibraryServer

        site_dir = temp_dir / "library-site"
        site_dir.mkdir(parents=True, exist_ok=True)

        server = LibraryServer(site_dir=site_dir)

        assert hasattr(server, "stop")
        assert callable(server.stop)

    def test_stop_shuts_down_server(self, temp_dir):
        """stop() should shut down the running server."""
        from src.library.server import LibraryServer

        site_dir = temp_dir / "library-site"
        site_dir.mkdir(parents=True, exist_ok=True)
        (site_dir / "index.html").write_text("<html><body>Test</body></html>")

        server = LibraryServer(site_dir=site_dir, port=18084)

        # Start server in blocking mode in a thread
        server_thread = threading.Thread(target=server.serve, kwargs={"blocking": True})
        server_thread.daemon = True
        server_thread.start()

        # Give server time to start
        time.sleep(0.5)

        # Server should be running
        assert server.httpd is not None

        # Stop the server
        server.stop()

        # Wait for thread to complete
        server_thread.join(timeout=2.0)

        # Server should be stopped
        assert not server_thread.is_alive()

    def test_stop_when_not_running_does_nothing(self, temp_dir):
        """stop() should not raise error when server is not running."""
        from src.library.server import LibraryServer

        site_dir = temp_dir / "library-site"
        site_dir.mkdir(parents=True, exist_ok=True)

        server = LibraryServer(site_dir=site_dir)

        # Should not raise an error
        server.stop()

    def test_serve_changes_to_site_directory(self, temp_dir):
        """serve() should serve files relative to site_dir."""
        import urllib.request
        from src.library.server import LibraryServer

        site_dir = temp_dir / "library-site"
        site_dir.mkdir(parents=True, exist_ok=True)
        sub_dir = site_dir / "theme" / "test-theme"
        sub_dir.mkdir(parents=True, exist_ok=True)
        (sub_dir / "index.html").write_text("<html><body>Theme Page</body></html>")

        server = LibraryServer(site_dir=site_dir, port=18085)

        server_thread = threading.Thread(target=server.serve, kwargs={"blocking": False})
        server_thread.daemon = True
        server_thread.start()

        time.sleep(0.5)

        try:
            response = urllib.request.urlopen("http://localhost:18085/theme/test-theme/index.html")
            content = response.read().decode("utf-8")
            assert "Theme Page" in content
        finally:
            if server.httpd:
                server.httpd.shutdown()


class TestPortHandling:
    """Tests for port-in-use handling and fallback logic."""

    def test_serve_raises_error_when_port_in_use(self, temp_dir):
        """serve() should raise OSError when the port is already in use."""
        import socketserver
        from src.library.server import LibraryServer

        site_dir = temp_dir / "library-site"
        site_dir.mkdir(parents=True, exist_ok=True)
        (site_dir / "index.html").write_text("<html><body>Test</body></html>")

        # Occupy the port with another server
        class DummyHandler(socketserver.BaseRequestHandler):
            def handle(self):
                pass

        socketserver.TCPServer.allow_reuse_address = True
        blocking_server = socketserver.TCPServer(("", 18090), DummyHandler)

        try:
            server = LibraryServer(site_dir=site_dir, port=18090)

            # Should raise an OSError when port is in use
            with pytest.raises(OSError):
                server.serve(blocking=False)
        finally:
            blocking_server.server_close()

    def test_serve_with_auto_port_finds_available_port(self, temp_dir):
        """serve() with auto_port=True should find an available port if requested port is in use."""
        import socketserver
        from src.library.server import LibraryServer

        site_dir = temp_dir / "library-site"
        site_dir.mkdir(parents=True, exist_ok=True)
        (site_dir / "index.html").write_text("<html><body>Test</body></html>")

        # Occupy the port with another server
        class DummyHandler(socketserver.BaseRequestHandler):
            def handle(self):
                pass

        socketserver.TCPServer.allow_reuse_address = True
        blocking_server = socketserver.TCPServer(("", 18091), DummyHandler)

        try:
            server = LibraryServer(site_dir=site_dir, port=18091)

            # With auto_port=True, should find next available port
            url = server.serve(blocking=False, auto_port=True)

            # Should have started on a different port
            assert server.httpd is not None
            assert server.port != 18091
            assert str(server.port) in url
        finally:
            blocking_server.server_close()
            if server.httpd:
                server.httpd.shutdown()

    def test_serve_with_auto_port_increments_port(self, temp_dir):
        """serve() with auto_port=True should try incrementing ports."""
        import socketserver
        from src.library.server import LibraryServer

        site_dir = temp_dir / "library-site"
        site_dir.mkdir(parents=True, exist_ok=True)
        (site_dir / "index.html").write_text("<html><body>Test</body></html>")

        # Occupy ports 18092 and 18093
        class DummyHandler(socketserver.BaseRequestHandler):
            def handle(self):
                pass

        socketserver.TCPServer.allow_reuse_address = True
        blocking_server1 = socketserver.TCPServer(("", 18092), DummyHandler)
        blocking_server2 = socketserver.TCPServer(("", 18093), DummyHandler)

        try:
            server = LibraryServer(site_dir=site_dir, port=18092)

            # With auto_port=True, should skip 18092 and 18093
            url = server.serve(blocking=False, auto_port=True)

            # Should have started on port 18094 or higher
            assert server.port >= 18094
            assert str(server.port) in url
        finally:
            blocking_server1.server_close()
            blocking_server2.server_close()
            if server.httpd:
                server.httpd.shutdown()

    def test_serve_with_auto_port_gives_up_after_max_attempts(self, temp_dir):
        """serve() with auto_port=True should give up after trying multiple ports."""
        import socketserver
        from src.library.server import LibraryServer

        site_dir = temp_dir / "library-site"
        site_dir.mkdir(parents=True, exist_ok=True)
        (site_dir / "index.html").write_text("<html><body>Test</body></html>")

        # Occupy 10 consecutive ports
        class DummyHandler(socketserver.BaseRequestHandler):
            def handle(self):
                pass

        socketserver.TCPServer.allow_reuse_address = True
        blocking_servers = []
        for i in range(10):
            srv = socketserver.TCPServer(("", 18100 + i), DummyHandler)
            blocking_servers.append(srv)

        try:
            server = LibraryServer(site_dir=site_dir, port=18100)

            # With auto_port=True but all ports occupied, should raise error
            with pytest.raises(OSError, match="Could not find an available port"):
                server.serve(blocking=False, auto_port=True)
        finally:
            for srv in blocking_servers:
                srv.server_close()

    def test_serve_without_auto_port_does_not_retry(self, temp_dir):
        """serve() without auto_port should not try other ports."""
        import socketserver
        from src.library.server import LibraryServer

        site_dir = temp_dir / "library-site"
        site_dir.mkdir(parents=True, exist_ok=True)
        (site_dir / "index.html").write_text("<html><body>Test</body></html>")

        # Occupy the port
        class DummyHandler(socketserver.BaseRequestHandler):
            def handle(self):
                pass

        socketserver.TCPServer.allow_reuse_address = True
        blocking_server = socketserver.TCPServer(("", 18095), DummyHandler)

        try:
            server = LibraryServer(site_dir=site_dir, port=18095)

            # Without auto_port, should raise immediately
            with pytest.raises(OSError):
                server.serve(blocking=False, auto_port=False)
        finally:
            blocking_server.server_close()

    def test_serve_updates_port_attribute_when_using_fallback(self, temp_dir):
        """serve() should update the port attribute when using a fallback port."""
        import socketserver
        from src.library.server import LibraryServer

        site_dir = temp_dir / "library-site"
        site_dir.mkdir(parents=True, exist_ok=True)
        (site_dir / "index.html").write_text("<html><body>Test</body></html>")

        # Occupy the port
        class DummyHandler(socketserver.BaseRequestHandler):
            def handle(self):
                pass

        socketserver.TCPServer.allow_reuse_address = True
        blocking_server = socketserver.TCPServer(("", 18096), DummyHandler)

        try:
            server = LibraryServer(site_dir=site_dir, port=18096)
            original_port = server.port

            server.serve(blocking=False, auto_port=True)

            # Port attribute should have been updated
            assert server.port != original_port
            assert server.port > original_port
        finally:
            blocking_server.server_close()
            if server.httpd:
                server.httpd.shutdown()

    def test_serve_auto_port_default_is_false(self, temp_dir):
        """serve() should have auto_port=False as default."""
        import socketserver
        from src.library.server import LibraryServer

        site_dir = temp_dir / "library-site"
        site_dir.mkdir(parents=True, exist_ok=True)
        (site_dir / "index.html").write_text("<html><body>Test</body></html>")

        # Occupy the port
        class DummyHandler(socketserver.BaseRequestHandler):
            def handle(self):
                pass

        socketserver.TCPServer.allow_reuse_address = True
        blocking_server = socketserver.TCPServer(("", 18097), DummyHandler)

        try:
            server = LibraryServer(site_dir=site_dir, port=18097)

            # By default (no auto_port arg), should raise error on port conflict
            with pytest.raises(OSError):
                server.serve(blocking=False)
        finally:
            blocking_server.server_close()
