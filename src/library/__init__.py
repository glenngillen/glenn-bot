"""
Library module for generating a browsable static website from ChromaDB knowledge base.

This module provides:
- Content export from ChromaDB to LibraryItem format
- AI-powered theme generation and content grouping
- Book cover resolution via Open Library and Google Books APIs
- Client-side search index generation
- Static HTML/CSS/JS site generation using Jinja2 templates
- Local development server

CLI Commands:
- /build-library: Generate the static site
- /build-library --serve: Generate and serve locally
- /build-library --force: Force regeneration of themes and covers
- /serve-library: Serve an existing build
- /library-status: Show build status and statistics
"""

from .models import ContentType, LibraryItem, Theme, ThemeAssignment

__all__ = [
    "ContentType",
    "LibraryItem",
    "Theme",
    "ThemeAssignment",
]
