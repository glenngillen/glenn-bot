"""Static site generator for the browsable library.

This module generates a static HTML/CSS/JS website from the ChromaDB knowledge base.
It uses Jinja2 templates to render pages and copies static assets to the output directory.

The generated site includes:
- Home page with theme cards and stats
- Theme pages showing items grouped by theme
- Individual item detail pages
- All content page with sorting and pagination
- Search page with client-side search functionality
"""

import shutil
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from .models import ContentType, LibraryItem, Theme


class StaticGenerator:
    """Generates a static HTML/CSS/JS website from library data.

    This class handles the complete static site generation process:
    - Setting up Jinja2 templating environment
    - Creating output directory structure
    - Rendering HTML pages from templates
    - Copying static assets (CSS, JS, images)
    """

    def __init__(self, output_dir: str | Path):
        """Initialize the StaticGenerator.

        Args:
            output_dir: Path to the output directory where the site will be generated.
                Can be a string or Path object. Will be created if it doesn't exist.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set up paths to templates and assets
        # Templates and assets are located relative to this file
        module_dir = Path(__file__).parent
        self.template_dir = module_dir / "templates"
        self.assets_dir = module_dir / "assets"

        # Set up Jinja2 environment
        self.env = self._setup_jinja_env()

    def _setup_jinja_env(self) -> Environment:
        """Set up the Jinja2 templating environment.

        Returns:
            A configured Jinja2 Environment with templates loaded and autoescape enabled.
        """
        env = Environment(
            loader=FileSystemLoader(self.template_dir),
            autoescape=select_autoescape(["html", "xml"]),
        )
        return env

    def _ensure_output_dirs(self) -> None:
        """Create the output directory structure.

        Creates all required directories for the static site:
        - all/ for the all content page
        - theme/ for theme pages
        - item/ for item detail pages
        - search/ for the search page
        - assets/css/, assets/js/, assets/images/ for static assets
        - _data/ for debug/data output
        """
        dirs_to_create = [
            self.output_dir / "all",
            self.output_dir / "theme",
            self.output_dir / "item",
            self.output_dir / "search",
            self.output_dir / "assets" / "css",
            self.output_dir / "assets" / "js",
            self.output_dir / "assets" / "images",
            self.output_dir / "_data",
        ]

        for dir_path in dirs_to_create:
            dir_path.mkdir(parents=True, exist_ok=True)

    def generate_home_page(
        self, themes: list[Theme], items: list[LibraryItem]
    ) -> Path:
        """Generate the home page (index.html).

        Args:
            themes: List of Theme objects to display as cards.
            items: List of all LibraryItem objects (for stats).

        Returns:
            Path to the generated index.html file.
        """
        template = self.env.get_template("index.html")

        # Prepare theme data for template - convert to dict if needed
        theme_dicts = []
        for theme in themes:
            if hasattr(theme, "to_dict"):
                theme_dicts.append(theme.to_dict())
            elif isinstance(theme, dict):
                theme_dicts.append(theme)
            else:
                theme_dicts.append({
                    "id": theme.id,
                    "name": theme.name,
                    "description": theme.description,
                    "item_count": theme.item_count,
                })

        content = template.render(
            themes=theme_dicts,
            total_items=len(items),
            total_themes=len(themes),
            active_page="home",
        )

        output_path = self.output_dir / "index.html"
        output_path.write_text(content)

        return output_path

    def generate_theme_pages(
        self,
        themes: list[Theme],
        items: list[LibraryItem],
        assignments: dict[str, list[str]],
    ) -> list[Path]:
        """Generate pages for each theme.

        Args:
            themes: List of Theme objects to generate pages for.
            items: List of all LibraryItem objects.
            assignments: Dict mapping theme_id to list of item_ids.

        Returns:
            List of paths to the generated theme page files.
        """
        template = self.env.get_template("theme.html")

        # Create a lookup dict for items by ID
        items_by_id = {item.id: item for item in items}

        generated_paths = []

        for theme in themes:
            # Get items assigned to this theme
            assigned_item_ids = assignments.get(theme.id, [])
            theme_items = []
            for item_id in assigned_item_ids:
                if item_id in items_by_id:
                    item = items_by_id[item_id]
                    theme_items.append(self._item_to_template_dict(item))

            # Prepare theme dict for template
            theme_dict = {
                "id": theme.id,
                "name": theme.name,
                "description": theme.description,
                "item_count": theme.item_count,
            }

            content = template.render(
                theme=theme_dict,
                items=theme_items,
            )

            # Create theme directory and write file
            theme_dir = self.output_dir / "theme" / theme.id
            theme_dir.mkdir(parents=True, exist_ok=True)

            output_path = theme_dir / "index.html"
            output_path.write_text(content)
            generated_paths.append(output_path)

        return generated_paths

    def generate_item_pages(
        self,
        items: list[LibraryItem],
        themes: dict[str, Theme],
    ) -> list[Path]:
        """Generate detail pages for each item.

        Args:
            items: List of LibraryItem objects to generate pages for.
            themes: Dict mapping theme_id to Theme object (for displaying theme names).

        Returns:
            List of paths to the generated item page files.
        """
        template = self.env.get_template("item.html")

        # Create theme name lookup
        theme_names = {}
        for theme_id, theme in themes.items():
            if hasattr(theme, "name"):
                theme_names[theme_id] = theme.name
            elif isinstance(theme, dict):
                theme_names[theme_id] = theme.get("name", theme_id)
            else:
                theme_names[theme_id] = theme_id

        generated_paths = []

        for item in items:
            item_dict = self._item_to_template_dict(item)

            content = template.render(
                item=item_dict,
                theme_names=theme_names,
            )

            # Create item directory and write file
            item_dir = self.output_dir / "item" / item.id
            item_dir.mkdir(parents=True, exist_ok=True)

            output_path = item_dir / "index.html"
            output_path.write_text(content)
            generated_paths.append(output_path)

        return generated_paths

    def generate_all_page(self, items: list[LibraryItem]) -> Path:
        """Generate the all content page (all/index.html).

        Args:
            items: List of all LibraryItem objects.

        Returns:
            Path to the generated all/index.html file.
        """
        template = self.env.get_template("all.html")

        # Convert items to template dicts
        item_dicts = [self._item_to_template_dict(item) for item in items]

        content = template.render(
            items=item_dicts,
            active_page="all",
        )

        output_path = self.output_dir / "all" / "index.html"
        output_path.write_text(content)

        return output_path

    def generate_search_page(self) -> Path:
        """Generate the search page (search/index.html).

        Returns:
            Path to the generated search/index.html file.
        """
        template = self.env.get_template("search.html")

        content = template.render()

        output_path = self.output_dir / "search" / "index.html"
        output_path.write_text(content)

        return output_path

    def copy_assets(self) -> list[Path]:
        """Copy static assets (CSS, JS, images) to the output directory.

        Returns:
            List of paths to the copied asset files.
        """
        copied_files = []

        # Copy CSS files
        css_src = self.assets_dir / "css"
        css_dest = self.output_dir / "assets" / "css"
        if css_src.exists():
            for css_file in css_src.glob("*.css"):
                dest_path = css_dest / css_file.name
                shutil.copy2(css_file, dest_path)
                copied_files.append(dest_path)

        # Copy JS files
        js_src = self.assets_dir / "js"
        js_dest = self.output_dir / "assets" / "js"
        if js_src.exists():
            for js_file in js_src.glob("*.js"):
                dest_path = js_dest / js_file.name
                shutil.copy2(js_file, dest_path)
                copied_files.append(dest_path)

        # Copy placeholder images
        placeholders_src = self.assets_dir / "images" / "placeholders"
        placeholders_dest = self.output_dir / "assets" / "images" / "placeholders"
        if placeholders_src.exists():
            placeholders_dest.mkdir(parents=True, exist_ok=True)
            for svg_file in placeholders_src.glob("*.svg"):
                dest_path = placeholders_dest / svg_file.name
                shutil.copy2(svg_file, dest_path)
                copied_files.append(dest_path)

        return copied_files

    def generate_all(
        self,
        themes: list[Theme],
        items: list[LibraryItem],
        assignments: dict[str, list[str]],
    ) -> dict[str, Any]:
        """Generate the complete static site.

        This method orchestrates the full site generation:
        1. Ensures output directories exist
        2. Generates home page
        3. Generates theme pages
        4. Generates item pages
        5. Generates all content page
        6. Generates search page
        7. Copies static assets

        Args:
            themes: List of Theme objects.
            items: List of LibraryItem objects.
            assignments: Dict mapping theme_id to list of item_ids.

        Returns:
            A summary dict with information about generated files.
        """
        # Ensure output directories exist
        self._ensure_output_dirs()

        # Generate home page
        home_path = self.generate_home_page(themes=themes, items=items)

        # Generate theme pages
        theme_paths = self.generate_theme_pages(
            themes=themes, items=items, assignments=assignments
        )

        # Generate item pages (convert themes list to dict for lookup)
        themes_dict = {theme.id: theme for theme in themes}
        item_paths = self.generate_item_pages(items=items, themes=themes_dict)

        # Generate all content page
        all_path = self.generate_all_page(items=items)

        # Generate search page
        search_path = self.generate_search_page()

        # Copy static assets
        asset_paths = self.copy_assets()

        # Return summary
        return {
            "home": home_path,
            "theme_pages": theme_paths,
            "item_pages": item_paths,
            "all_page": all_path,
            "search_page": search_path,
            "assets": asset_paths,
            "pages_generated": 1 + len(theme_paths) + len(item_paths) + 2,
        }

    def _item_to_template_dict(self, item: LibraryItem) -> dict[str, Any]:
        """Convert a LibraryItem to a dictionary suitable for templates.

        Args:
            item: The LibraryItem to convert.

        Returns:
            A dictionary with all item data ready for template rendering.
        """
        return {
            "id": item.id,
            "content_type": item.content_type.value if isinstance(item.content_type, ContentType) else item.content_type,
            "title": item.title,
            "summary": item.summary,
            "full_content": item.full_content,
            "source_url": item.source_url,
            "cover_image_url": item.cover_image_url,
            "metadata": item.metadata,
            "themes": item.themes,
            "created_at": item.created_at,
            "highlights": item.highlights,
        }
