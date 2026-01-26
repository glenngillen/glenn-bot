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
