"""Library builder module for orchestrating the static site build process.

This module provides the LibraryBuilder class which orchestrates the entire build
process for generating the static library site from ChromaDB knowledge base content.

Build Process:
1. Export content from ChromaDB via ContentExporter
2. Resolve cover images via CoverResolver
3. Generate/update themes via ThemeGenerator
4. Build search index via SearchIndexer
5. Generate static HTML pages via StaticGenerator
6. Save build state for incremental updates
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

from src.library.content_exporter import ContentExporter
from src.library.cover_resolver import CoverResolver
from src.library.models import LibraryItem, Theme
from src.library.search_indexer import SearchIndexer
from src.library.static_generator import StaticGenerator
from src.library.theme_generator import ThemeGenerator

if TYPE_CHECKING:
    from src.knowledge_base import KnowledgeBase
    from src.ollama_client import OllamaClient


class LibraryBuilder:
    """Orchestrates the library build process.

    LibraryBuilder coordinates the entire process of building a static website
    from the knowledge base content. It manages the pipeline of:
    - Exporting content from ChromaDB
    - Resolving cover images for books
    - Generating AI-powered themes
    - Building search indexes
    - Generating static HTML pages

    It also tracks build state for incremental updates, allowing subsequent
    builds to only process changed content.

    Attributes:
        knowledge_base: The KnowledgeBase instance to export content from.
        ollama_client: The OllamaClient instance for AI theme generation.
        data_dir: Path to the persistent data directory (themes, cache).
        site_dir: Path to the generated site output directory.
        build_state_file: Path to the _build_state.json file.
    """

    def __init__(
        self,
        knowledge_base: "KnowledgeBase",
        ollama_client: "OllamaClient",
        data_dir: Union[str, Path],
        site_dir: Union[str, Path],
    ) -> None:
        """Initialize the LibraryBuilder.

        Args:
            knowledge_base: The KnowledgeBase instance to export content from.
            ollama_client: The OllamaClient instance for AI theme generation.
            data_dir: Path to the persistent data directory. Will be created
                if it doesn't exist.
            site_dir: Path to the generated site output directory. Will be
                created if it doesn't exist.
        """
        self.knowledge_base = knowledge_base
        self.ollama_client = ollama_client

        # Convert string paths to Path objects
        self.data_dir = Path(data_dir) if isinstance(data_dir, str) else data_dir
        self.site_dir = Path(site_dir) if isinstance(site_dir, str) else site_dir

        # Create directories if they don't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.site_dir.mkdir(parents=True, exist_ok=True)

        # Build state file location
        self.build_state_file = self.site_dir / "_build_state.json"

    def _load_build_state(self) -> dict[str, Any]:
        """Load the build state from disk.

        Reads the _build_state.json file and returns its contents. This state
        is used to detect which items have changed since the last build.

        Returns:
            A dictionary containing the build state. Returns an empty dict
            if the file doesn't exist or contains invalid JSON.
        """
        if not self.build_state_file.exists():
            return {}

        try:
            with open(self.build_state_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}

    def _save_build_state(self, state: dict[str, Any]) -> None:
        """Save the build state to disk.

        Writes the build state to _build_state.json. This persists information
        about item hashes and build timestamps for incremental updates.

        Args:
            state: The build state dictionary to persist.
        """
        with open(self.build_state_file, "w") as f:
            json.dump(state, f, indent=2)

    def _compute_item_hash(self, item: LibraryItem) -> str:
        """Compute a hash for a library item's content.

        Creates a hash based on the item's key fields that affect rendering:
        id, title, summary, full_content, source_url, metadata, themes, and
        highlights. This hash is used to detect changes between builds.

        Args:
            item: The LibraryItem to compute a hash for.

        Returns:
            A hexadecimal string hash of the item's content.
        """
        # Create a string representation of the key fields
        content_parts = [
            item.id,
            item.content_type.value,
            item.title,
            item.summary,
            item.full_content,
            str(item.source_url),
            json.dumps(item.metadata, sort_keys=True),
            json.dumps(item.themes, sort_keys=True),
            json.dumps(item.highlights, sort_keys=True),
        ]

        combined = "|".join(content_parts)
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

    def _get_changed_items(
        self, items: list[LibraryItem], previous_state: dict[str, Any]
    ) -> list[LibraryItem]:
        """Identify items that have changed since the last build.

        Compares current item hashes against the previous build state to
        determine which items are new or modified.

        Args:
            items: List of current LibraryItem objects.
            previous_state: The previous build state dictionary. Should have
                an 'item_hashes' key mapping item IDs to hash strings.

        Returns:
            A list of LibraryItem objects that are new or have changed.
        """
        previous_hashes = previous_state.get("item_hashes", {})
        changed = []

        for item in items:
            current_hash = self._compute_item_hash(item)
            previous_hash = previous_hashes.get(item.id)

            # Item is new or has changed if hash doesn't match
            if previous_hash is None or previous_hash != current_hash:
                changed.append(item)

        return changed

    def _export_library_json(
        self, items: list[LibraryItem], themes: list[Theme]
    ) -> Path:
        """Export library data to a JSON file for debugging.

        Creates a library.json file in the _data directory containing all
        items and themes in JSON format. This is useful for debugging and
        can be used by JavaScript for dynamic functionality.

        Args:
            items: List of LibraryItem objects to export.
            themes: List of Theme objects to export.

        Returns:
            Path to the created library.json file.
        """
        # Ensure _data directory exists
        data_dir = self.site_dir / "_data"
        data_dir.mkdir(parents=True, exist_ok=True)

        # Serialize items and themes
        data = {
            "items": [item.to_dict() for item in items],
            "themes": [theme.to_dict() for theme in themes],
        }

        output_path = data_dir / "library.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return output_path

    def build(self, force: bool = False) -> dict[str, Any]:
        """Build the static library site.

        Orchestrates the complete build process:
        1. Export content from ChromaDB
        2. Resolve cover images
        3. Generate/update themes
        4. Assign items to themes
        5. Build search index
        6. Generate static HTML pages
        7. Export debug library.json
        8. Save build state

        Args:
            force: If True, forces a full rebuild ignoring previous state.
                Regenerates themes and resolves covers without using cache.

        Returns:
            A summary dictionary with build information including:
            - items_count: Number of items processed
            - themes_count: Number of themes
            - pages_generated: Number of HTML pages created
        """
        # Step 1: Export content from ChromaDB
        exporter = ContentExporter(self.knowledge_base)
        items = exporter.export_all()

        # Step 2: Load previous build state (unless forcing rebuild)
        previous_state = {} if force else self._load_build_state()

        # Step 3: Resolve cover images
        if force:
            # Clear cover cache for forced rebuild by creating fresh resolver
            cover_cache_file = self.data_dir / "cover_cache.json"
            if cover_cache_file.exists():
                cover_cache_file.unlink()

        cover_resolver = CoverResolver(self.data_dir)
        items = cover_resolver.resolve_all_covers(items)

        # Step 4: Generate/update themes
        theme_generator = ThemeGenerator(self.ollama_client, self.data_dir)

        # Load existing themes (unless forcing rebuild)
        existing_themes = [] if force else theme_generator.load_themes()

        if not existing_themes or force:
            # Generate new themes
            theme_generator.generate_themes(items)
        else:
            # Use existing themes
            theme_generator.themes = existing_themes

        # Ensure minimum number of themes for small libraries
        theme_generator.ensure_minimum_themes()

        # Load existing assignments (unless forcing rebuild)
        if not force:
            theme_generator.load_assignments()

        # Assign items to themes
        if force or not theme_generator.assignments:
            theme_generator.assign_items_to_themes(items)
        else:
            # Incremental update for new items only
            theme_generator.update_assignments(items)

        # Apply Miscellaneous fallback for items with low confidence
        theme_generator.apply_miscellaneous_fallback(items)

        # Update item themes based on assignments
        items = self._apply_theme_assignments_to_items(items, theme_generator)

        # Step 5: Build search index
        themes_dict = {theme.id: theme for theme in theme_generator.themes}
        search_indexer = SearchIndexer(self.site_dir)
        search_indexer.write_index(items, themes_dict)

        # Step 6: Generate static HTML pages
        static_generator = StaticGenerator(self.site_dir)

        # Build assignments dict for theme pages (theme_id -> [item_ids])
        assignments_dict = self._build_assignments_dict(theme_generator)

        # Update theme item counts
        for theme in theme_generator.themes:
            theme.item_count = len(assignments_dict.get(theme.id, []))

        generation_result = static_generator.generate_all(
            themes=theme_generator.themes,
            items=items,
            assignments=assignments_dict,
        )

        # Step 7: Export debug library.json
        self._export_library_json(items, theme_generator.themes)

        # Step 8: Save build state
        item_hashes = {item.id: self._compute_item_hash(item) for item in items}
        new_state = {
            "last_build": datetime.now().isoformat(),
            "item_hashes": item_hashes,
            "theme_version": "1.0",
        }
        self._save_build_state(new_state)

        # Return summary
        return {
            "items_count": len(items),
            "themes_count": len(theme_generator.themes),
            "pages_generated": generation_result.get("pages_generated", 0),
            "last_build": new_state["last_build"],
        }

    def _apply_theme_assignments_to_items(
        self, items: list[LibraryItem], theme_generator: ThemeGenerator
    ) -> list[LibraryItem]:
        """Update items' themes list based on theme assignments.

        Creates new LibraryItem instances with their themes field populated
        based on the theme generator's assignments.

        Args:
            items: List of LibraryItem objects.
            theme_generator: ThemeGenerator with assignments.

        Returns:
            List of new LibraryItem instances with themes populated.
        """
        # Build a mapping of item_id -> [theme_ids]
        item_themes: dict[str, list[str]] = {}
        for assignment in theme_generator.assignments:
            if assignment.item_id not in item_themes:
                item_themes[assignment.item_id] = []
            item_themes[assignment.item_id].append(assignment.theme_id)

        # Create new items with updated themes
        updated_items = []
        for item in items:
            new_item = LibraryItem(
                id=item.id,
                content_type=item.content_type,
                title=item.title,
                summary=item.summary,
                full_content=item.full_content,
                source_url=item.source_url,
                cover_image_url=item.cover_image_url,
                metadata=item.metadata,
                themes=item_themes.get(item.id, []),
                created_at=item.created_at,
                highlights=item.highlights,
            )
            updated_items.append(new_item)

        return updated_items

    def _build_assignments_dict(
        self, theme_generator: ThemeGenerator
    ) -> dict[str, list[str]]:
        """Build a dictionary mapping theme IDs to item IDs.

        Args:
            theme_generator: ThemeGenerator with assignments.

        Returns:
            Dictionary mapping theme_id to list of item_ids.
        """
        assignments_dict: dict[str, list[str]] = {}

        for assignment in theme_generator.assignments:
            if assignment.theme_id not in assignments_dict:
                assignments_dict[assignment.theme_id] = []
            assignments_dict[assignment.theme_id].append(assignment.item_id)

        return assignments_dict
