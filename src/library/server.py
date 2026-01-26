"""
Library server module for serving the static library site locally.

This module provides a simple HTTP server for local development and viewing
of the generated library site.
"""

from pathlib import Path
from typing import Optional, Union
import http.server
import socketserver


class LibraryServer:
    """Simple HTTP server for serving the library site locally.

    Provides a basic HTTP server using Python's http.server module
    to serve the generated static library site for local viewing.

    Attributes:
        site_dir: Path to the generated library site directory.
        port: Port number to serve on (default 8080).
        httpd: The HTTP server instance (None until serve() is called).
    """

    def __init__(
        self,
        site_dir: Union[str, Path],
        port: int = 8080,
    ) -> None:
        """Initialize the LibraryServer.

        Args:
            site_dir: Path to the generated library site directory.
                Must exist before serving.
            port: Port number to serve on. Defaults to 8080.

        Raises:
            ValueError: If site_dir does not exist.
        """
        self.site_dir = Path(site_dir) if isinstance(site_dir, str) else site_dir
        self.port = port
        self.httpd: Optional[socketserver.TCPServer] = None

        if not self.site_dir.exists():
            raise ValueError(f"Site directory does not exist: {self.site_dir}")

    def serve(self, blocking: bool = True) -> str:
        """Start the HTTP server to serve the library site.

        Args:
            blocking: If True, blocks until server is interrupted.
                If False, returns immediately after server starts.

        Returns:
            The URL where the server is accessible (e.g., "http://localhost:8080").
        """
        # Create a custom handler that serves from our site directory
        class SiteHandler(http.server.SimpleHTTPRequestHandler):
            def __init__(handler_self, *args, **kwargs):
                super().__init__(*args, directory=str(self.site_dir), **kwargs)

            def log_message(handler_self, format, *args):
                # Suppress default logging
                pass

        # Allow address reuse to avoid "Address already in use" errors
        socketserver.TCPServer.allow_reuse_address = True

        self.httpd = socketserver.TCPServer(("", self.port), SiteHandler)

        url = f"http://localhost:{self.port}"

        if blocking:
            self.httpd.serve_forever()
        else:
            # Start server in a separate thread for non-blocking mode
            import threading

            server_thread = threading.Thread(target=self.httpd.serve_forever)
            server_thread.daemon = True
            server_thread.start()

        return url

    def stop(self) -> None:
        """Stop the running HTTP server.

        Safe to call even if server is not running.
        """
        if self.httpd is not None:
            self.httpd.shutdown()
