"""Tests for static site generator module.

This module tests the StaticGenerator class which generates a static HTML/CSS/JS
website from the ChromaDB knowledge base. It integrates with Jinja2 templates
and produces the browsable library site.

Test classes are organized by the methods they test:
- TestStaticGeneratorInitialization: StaticGenerator class initialization
- TestSetupJinjaEnv: _setup_jinja_env() loading templates
- TestEnsureOutputDirs: _ensure_output_dirs() creating structure
- TestGenerateHomePage: generate_home_page() rendering index.html
- TestGenerateThemePages: generate_theme_pages() creating theme/{id}/index.html
- TestGenerateItemPages: generate_item_pages() creating item/{id}/index.html
- TestGenerateAllPage: generate_all_page() rendering all/index.html
- TestGenerateSearchPage: generate_search_page() rendering search/index.html
- TestCopyAssets: copy_assets() copying CSS, JS, images
- TestGenerateAll: generate_all() orchestrating full generation
"""

import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch
import json

from src.library.models import LibraryItem, ContentType, Theme, ThemeAssignment


class TestStaticGeneratorInitialization:
    """Tests for StaticGenerator class initialization."""

    def test_static_generator_can_be_imported(self):
        """StaticGenerator class should be importable from the module."""
        from src.library.static_generator import StaticGenerator

        assert StaticGenerator is not None

    def test_static_generator_init_with_output_dir(self, temp_dir):
        """StaticGenerator should accept an output_dir path."""
        from src.library.static_generator import StaticGenerator

        output_dir = temp_dir / "library-site"
        output_dir.mkdir(parents=True, exist_ok=True)

        generator = StaticGenerator(output_dir=output_dir)

        assert generator.output_dir == output_dir

    def test_static_generator_init_creates_output_dir_if_not_exists(self, temp_dir):
        """StaticGenerator should create the output_dir if it doesn't exist."""
        from src.library.static_generator import StaticGenerator

        output_dir = temp_dir / "library-site" / "subdir"
        assert not output_dir.exists()

        generator = StaticGenerator(output_dir=output_dir)

        assert output_dir.exists()

    def test_static_generator_init_output_dir_as_string(self, temp_dir):
        """StaticGenerator should accept output_dir as a string and convert to Path."""
        from src.library.static_generator import StaticGenerator

        output_dir = str(temp_dir / "library-site")

        generator = StaticGenerator(output_dir=output_dir)

        assert isinstance(generator.output_dir, Path)
        assert generator.output_dir == Path(output_dir)

    def test_static_generator_has_template_dir_attribute(self, temp_dir):
        """StaticGenerator should have a template_dir attribute pointing to templates."""
        from src.library.static_generator import StaticGenerator

        output_dir = temp_dir / "library-site"
        output_dir.mkdir(parents=True, exist_ok=True)

        generator = StaticGenerator(output_dir=output_dir)

        assert hasattr(generator, "template_dir")
        assert generator.template_dir.exists()
        assert (generator.template_dir / "base.html").exists()

    def test_static_generator_has_assets_dir_attribute(self, temp_dir):
        """StaticGenerator should have an assets_dir attribute pointing to assets."""
        from src.library.static_generator import StaticGenerator

        output_dir = temp_dir / "library-site"
        output_dir.mkdir(parents=True, exist_ok=True)

        generator = StaticGenerator(output_dir=output_dir)

        assert hasattr(generator, "assets_dir")
        assert generator.assets_dir.exists()

    def test_static_generator_has_jinja_env(self, temp_dir):
        """StaticGenerator should initialize a Jinja2 environment."""
        from src.library.static_generator import StaticGenerator

        output_dir = temp_dir / "library-site"
        output_dir.mkdir(parents=True, exist_ok=True)

        generator = StaticGenerator(output_dir=output_dir)

        assert hasattr(generator, "env")
        assert generator.env is not None


class TestSetupJinjaEnv:
    """Tests for _setup_jinja_env() method that loads templates."""

    def test_setup_jinja_env_loads_base_template(self, temp_dir):
        """_setup_jinja_env() should be able to load the base.html template."""
        from src.library.static_generator import StaticGenerator

        output_dir = temp_dir / "library-site"
        output_dir.mkdir(parents=True, exist_ok=True)

        generator = StaticGenerator(output_dir=output_dir)
        template = generator.env.get_template("base.html")

        assert template is not None

    def test_setup_jinja_env_loads_index_template(self, temp_dir):
        """_setup_jinja_env() should be able to load the index.html template."""
        from src.library.static_generator import StaticGenerator

        output_dir = temp_dir / "library-site"
        output_dir.mkdir(parents=True, exist_ok=True)

        generator = StaticGenerator(output_dir=output_dir)
        template = generator.env.get_template("index.html")

        assert template is not None

    def test_setup_jinja_env_loads_theme_template(self, temp_dir):
        """_setup_jinja_env() should be able to load the theme.html template."""
        from src.library.static_generator import StaticGenerator

        output_dir = temp_dir / "library-site"
        output_dir.mkdir(parents=True, exist_ok=True)

        generator = StaticGenerator(output_dir=output_dir)
        template = generator.env.get_template("theme.html")

        assert template is not None

    def test_setup_jinja_env_loads_item_template(self, temp_dir):
        """_setup_jinja_env() should be able to load the item.html template."""
        from src.library.static_generator import StaticGenerator

        output_dir = temp_dir / "library-site"
        output_dir.mkdir(parents=True, exist_ok=True)

        generator = StaticGenerator(output_dir=output_dir)
        template = generator.env.get_template("item.html")

        assert template is not None

    def test_setup_jinja_env_loads_all_template(self, temp_dir):
        """_setup_jinja_env() should be able to load the all.html template."""
        from src.library.static_generator import StaticGenerator

        output_dir = temp_dir / "library-site"
        output_dir.mkdir(parents=True, exist_ok=True)

        generator = StaticGenerator(output_dir=output_dir)
        template = generator.env.get_template("all.html")

        assert template is not None

    def test_setup_jinja_env_loads_search_template(self, temp_dir):
        """_setup_jinja_env() should be able to load the search.html template."""
        from src.library.static_generator import StaticGenerator

        output_dir = temp_dir / "library-site"
        output_dir.mkdir(parents=True, exist_ok=True)

        generator = StaticGenerator(output_dir=output_dir)
        template = generator.env.get_template("search.html")

        assert template is not None

    def test_setup_jinja_env_loads_macros(self, temp_dir):
        """_setup_jinja_env() should be able to load the macros.html template."""
        from src.library.static_generator import StaticGenerator

        output_dir = temp_dir / "library-site"
        output_dir.mkdir(parents=True, exist_ok=True)

        generator = StaticGenerator(output_dir=output_dir)
        template = generator.env.get_template("macros.html")

        assert template is not None

    def test_setup_jinja_env_has_autoescape_enabled(self, temp_dir):
        """_setup_jinja_env() should enable autoescape for HTML safety."""
        from src.library.static_generator import StaticGenerator

        output_dir = temp_dir / "library-site"
        output_dir.mkdir(parents=True, exist_ok=True)

        generator = StaticGenerator(output_dir=output_dir)

        # Jinja2 autoescape is typically enabled for .html files
        assert generator.env.autoescape is True or callable(generator.env.autoescape)


class TestEnsureOutputDirs:
    """Tests for _ensure_output_dirs() method that creates directory structure."""

    def test_ensure_output_dirs_creates_all_directory(self, temp_dir):
        """_ensure_output_dirs() should create the all/ directory."""
        from src.library.static_generator import StaticGenerator

        output_dir = temp_dir / "library-site"

        generator = StaticGenerator(output_dir=output_dir)
        generator._ensure_output_dirs()

        assert (output_dir / "all").exists()
        assert (output_dir / "all").is_dir()

    def test_ensure_output_dirs_creates_theme_directory(self, temp_dir):
        """_ensure_output_dirs() should create the theme/ directory."""
        from src.library.static_generator import StaticGenerator

        output_dir = temp_dir / "library-site"

        generator = StaticGenerator(output_dir=output_dir)
        generator._ensure_output_dirs()

        assert (output_dir / "theme").exists()
        assert (output_dir / "theme").is_dir()

    def test_ensure_output_dirs_creates_item_directory(self, temp_dir):
        """_ensure_output_dirs() should create the item/ directory."""
        from src.library.static_generator import StaticGenerator

        output_dir = temp_dir / "library-site"

        generator = StaticGenerator(output_dir=output_dir)
        generator._ensure_output_dirs()

        assert (output_dir / "item").exists()
        assert (output_dir / "item").is_dir()

    def test_ensure_output_dirs_creates_search_directory(self, temp_dir):
        """_ensure_output_dirs() should create the search/ directory."""
        from src.library.static_generator import StaticGenerator

        output_dir = temp_dir / "library-site"

        generator = StaticGenerator(output_dir=output_dir)
        generator._ensure_output_dirs()

        assert (output_dir / "search").exists()
        assert (output_dir / "search").is_dir()

    def test_ensure_output_dirs_creates_assets_directories(self, temp_dir):
        """_ensure_output_dirs() should create assets/css, assets/js, assets/images."""
        from src.library.static_generator import StaticGenerator

        output_dir = temp_dir / "library-site"

        generator = StaticGenerator(output_dir=output_dir)
        generator._ensure_output_dirs()

        assert (output_dir / "assets" / "css").exists()
        assert (output_dir / "assets" / "js").exists()
        assert (output_dir / "assets" / "images").exists()

    def test_ensure_output_dirs_creates_data_directory(self, temp_dir):
        """_ensure_output_dirs() should create _data/ directory for debug output."""
        from src.library.static_generator import StaticGenerator

        output_dir = temp_dir / "library-site"

        generator = StaticGenerator(output_dir=output_dir)
        generator._ensure_output_dirs()

        assert (output_dir / "_data").exists()
        assert (output_dir / "_data").is_dir()

    def test_ensure_output_dirs_is_idempotent(self, temp_dir):
        """_ensure_output_dirs() should be safe to call multiple times."""
        from src.library.static_generator import StaticGenerator

        output_dir = temp_dir / "library-site"

        generator = StaticGenerator(output_dir=output_dir)
        generator._ensure_output_dirs()
        generator._ensure_output_dirs()  # Call again

        assert (output_dir / "all").exists()
        assert (output_dir / "theme").exists()


class TestGenerateHomePage:
    """Tests for generate_home_page() method that renders index.html."""

    def test_generate_home_page_creates_index_html(self, temp_dir):
        """generate_home_page() should create index.html in output directory."""
        from src.library.static_generator import StaticGenerator

        output_dir = temp_dir / "library-site"
        generator = StaticGenerator(output_dir=output_dir)
        generator._ensure_output_dirs()

        themes = []
        items = []

        generator.generate_home_page(themes=themes, items=items)

        assert (output_dir / "index.html").exists()

    def test_generate_home_page_includes_themes(self, temp_dir):
        """generate_home_page() should render theme cards in the output."""
        from src.library.static_generator import StaticGenerator

        output_dir = temp_dir / "library-site"
        generator = StaticGenerator(output_dir=output_dir)
        generator._ensure_output_dirs()

        themes = [
            Theme(
                id="systems-thinking",
                name="Systems Thinking",
                description="Understanding complex systems",
                keywords=["systems", "feedback"],
                item_count=5,
                created_at=datetime(2024, 1, 15),
                updated_at=datetime(2024, 1, 20),
            )
        ]
        items = []

        generator.generate_home_page(themes=themes, items=items)

        content = (output_dir / "index.html").read_text()
        assert "Systems Thinking" in content
        assert "Understanding complex systems" in content

    def test_generate_home_page_shows_stats(self, temp_dir):
        """generate_home_page() should show total items and themes count."""
        from src.library.static_generator import StaticGenerator

        output_dir = temp_dir / "library-site"
        generator = StaticGenerator(output_dir=output_dir)
        generator._ensure_output_dirs()

        themes = [
            Theme(
                id="theme-1",
                name="Theme 1",
                description="Description 1",
                keywords=[],
                item_count=3,
                created_at=datetime(2024, 1, 15),
                updated_at=datetime(2024, 1, 20),
            ),
            Theme(
                id="theme-2",
                name="Theme 2",
                description="Description 2",
                keywords=[],
                item_count=2,
                created_at=datetime(2024, 1, 15),
                updated_at=datetime(2024, 1, 20),
            ),
        ]
        items = [
            LibraryItem(
                id=f"doc_{i}",
                content_type=ContentType.BOOK,
                title=f"Book {i}",
                summary=f"Summary {i}",
                full_content="Content",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime(2024, 1, 15),
                highlights=[],
            )
            for i in range(5)
        ]

        generator.generate_home_page(themes=themes, items=items)

        content = (output_dir / "index.html").read_text()
        # Should show 5 total items and 2 themes
        assert "5" in content  # Total items
        assert "2" in content  # Total themes

    def test_generate_home_page_empty_themes_shows_message(self, temp_dir):
        """generate_home_page() should show 'No themes yet' when themes is empty."""
        from src.library.static_generator import StaticGenerator

        output_dir = temp_dir / "library-site"
        generator = StaticGenerator(output_dir=output_dir)
        generator._ensure_output_dirs()

        generator.generate_home_page(themes=[], items=[])

        content = (output_dir / "index.html").read_text()
        assert "No themes yet" in content

    def test_generate_home_page_returns_path(self, temp_dir):
        """generate_home_page() should return the path to the generated file."""
        from src.library.static_generator import StaticGenerator

        output_dir = temp_dir / "library-site"
        generator = StaticGenerator(output_dir=output_dir)
        generator._ensure_output_dirs()

        result = generator.generate_home_page(themes=[], items=[])

        assert result == output_dir / "index.html"


class TestGenerateThemePages:
    """Tests for generate_theme_pages() method that creates theme/{id}/index.html."""

    def test_generate_theme_pages_creates_theme_directories(self, temp_dir):
        """generate_theme_pages() should create a directory for each theme."""
        from src.library.static_generator import StaticGenerator

        output_dir = temp_dir / "library-site"
        generator = StaticGenerator(output_dir=output_dir)
        generator._ensure_output_dirs()

        themes = [
            Theme(
                id="systems-thinking",
                name="Systems Thinking",
                description="Understanding complex systems",
                keywords=["systems"],
                item_count=1,
                created_at=datetime(2024, 1, 15),
                updated_at=datetime(2024, 1, 20),
            ),
            Theme(
                id="mental-models",
                name="Mental Models",
                description="Thinking frameworks",
                keywords=["models"],
                item_count=1,
                created_at=datetime(2024, 1, 15),
                updated_at=datetime(2024, 1, 20),
            ),
        ]
        items = []
        assignments = {}

        generator.generate_theme_pages(
            themes=themes, items=items, assignments=assignments
        )

        assert (output_dir / "theme" / "systems-thinking").exists()
        assert (output_dir / "theme" / "mental-models").exists()

    def test_generate_theme_pages_creates_index_html_for_each_theme(self, temp_dir):
        """generate_theme_pages() should create index.html in each theme directory."""
        from src.library.static_generator import StaticGenerator

        output_dir = temp_dir / "library-site"
        generator = StaticGenerator(output_dir=output_dir)
        generator._ensure_output_dirs()

        themes = [
            Theme(
                id="systems-thinking",
                name="Systems Thinking",
                description="Understanding complex systems",
                keywords=["systems"],
                item_count=1,
                created_at=datetime(2024, 1, 15),
                updated_at=datetime(2024, 1, 20),
            )
        ]
        items = []
        assignments = {}

        generator.generate_theme_pages(
            themes=themes, items=items, assignments=assignments
        )

        assert (output_dir / "theme" / "systems-thinking" / "index.html").exists()

    def test_generate_theme_pages_includes_theme_name_and_description(self, temp_dir):
        """generate_theme_pages() should render theme name and description."""
        from src.library.static_generator import StaticGenerator

        output_dir = temp_dir / "library-site"
        generator = StaticGenerator(output_dir=output_dir)
        generator._ensure_output_dirs()

        themes = [
            Theme(
                id="systems-thinking",
                name="Systems Thinking",
                description="Understanding complex systems and feedback loops",
                keywords=["systems"],
                item_count=0,
                created_at=datetime(2024, 1, 15),
                updated_at=datetime(2024, 1, 20),
            )
        ]
        items = []
        assignments = {}

        generator.generate_theme_pages(
            themes=themes, items=items, assignments=assignments
        )

        content = (
            output_dir / "theme" / "systems-thinking" / "index.html"
        ).read_text()
        assert "Systems Thinking" in content
        assert "Understanding complex systems and feedback loops" in content

    def test_generate_theme_pages_includes_assigned_items(self, temp_dir):
        """generate_theme_pages() should show items assigned to the theme."""
        from src.library.static_generator import StaticGenerator

        output_dir = temp_dir / "library-site"
        generator = StaticGenerator(output_dir=output_dir)
        generator._ensure_output_dirs()

        themes = [
            Theme(
                id="systems-thinking",
                name="Systems Thinking",
                description="Understanding complex systems",
                keywords=["systems"],
                item_count=1,
                created_at=datetime(2024, 1, 15),
                updated_at=datetime(2024, 1, 20),
            )
        ]
        items = [
            LibraryItem(
                id="doc_1",
                content_type=ContentType.BOOK,
                title="Thinking in Systems",
                summary="A primer on systems thinking",
                full_content="Content...",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=["systems-thinking"],
                created_at=datetime(2024, 1, 15),
                highlights=[],
            )
        ]
        # Assignments map theme_id to list of item_ids
        assignments = {"systems-thinking": ["doc_1"]}

        generator.generate_theme_pages(
            themes=themes, items=items, assignments=assignments
        )

        content = (
            output_dir / "theme" / "systems-thinking" / "index.html"
        ).read_text()
        assert "Thinking in Systems" in content

    def test_generate_theme_pages_empty_theme_shows_message(self, temp_dir):
        """generate_theme_pages() should show 'No items' message for empty themes."""
        from src.library.static_generator import StaticGenerator

        output_dir = temp_dir / "library-site"
        generator = StaticGenerator(output_dir=output_dir)
        generator._ensure_output_dirs()

        themes = [
            Theme(
                id="empty-theme",
                name="Empty Theme",
                description="A theme with no items",
                keywords=[],
                item_count=0,
                created_at=datetime(2024, 1, 15),
                updated_at=datetime(2024, 1, 20),
            )
        ]
        items = []
        assignments = {}

        generator.generate_theme_pages(
            themes=themes, items=items, assignments=assignments
        )

        content = (output_dir / "theme" / "empty-theme" / "index.html").read_text()
        assert "No items" in content or "0 item" in content

    def test_generate_theme_pages_returns_list_of_paths(self, temp_dir):
        """generate_theme_pages() should return list of generated file paths."""
        from src.library.static_generator import StaticGenerator

        output_dir = temp_dir / "library-site"
        generator = StaticGenerator(output_dir=output_dir)
        generator._ensure_output_dirs()

        themes = [
            Theme(
                id="theme-1",
                name="Theme 1",
                description="Description 1",
                keywords=[],
                item_count=0,
                created_at=datetime(2024, 1, 15),
                updated_at=datetime(2024, 1, 20),
            ),
            Theme(
                id="theme-2",
                name="Theme 2",
                description="Description 2",
                keywords=[],
                item_count=0,
                created_at=datetime(2024, 1, 15),
                updated_at=datetime(2024, 1, 20),
            ),
        ]

        result = generator.generate_theme_pages(themes=themes, items=[], assignments={})

        assert len(result) == 2
        assert output_dir / "theme" / "theme-1" / "index.html" in result
        assert output_dir / "theme" / "theme-2" / "index.html" in result


class TestGenerateItemPages:
    """Tests for generate_item_pages() method that creates item/{id}/index.html."""

    def test_generate_item_pages_creates_item_directories(self, temp_dir):
        """generate_item_pages() should create a directory for each item."""
        from src.library.static_generator import StaticGenerator

        output_dir = temp_dir / "library-site"
        generator = StaticGenerator(output_dir=output_dir)
        generator._ensure_output_dirs()

        items = [
            LibraryItem(
                id="doc_1",
                content_type=ContentType.BOOK,
                title="Book 1",
                summary="Summary 1",
                full_content="Content...",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime(2024, 1, 15),
                highlights=[],
            ),
            LibraryItem(
                id="doc_2",
                content_type=ContentType.ARTICLE,
                title="Article 1",
                summary="Summary 2",
                full_content="Content...",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime(2024, 1, 16),
                highlights=[],
            ),
        ]
        themes = {}

        generator.generate_item_pages(items=items, themes=themes)

        assert (output_dir / "item" / "doc_1").exists()
        assert (output_dir / "item" / "doc_2").exists()

    def test_generate_item_pages_creates_index_html_for_each_item(self, temp_dir):
        """generate_item_pages() should create index.html in each item directory."""
        from src.library.static_generator import StaticGenerator

        output_dir = temp_dir / "library-site"
        generator = StaticGenerator(output_dir=output_dir)
        generator._ensure_output_dirs()

        items = [
            LibraryItem(
                id="doc_1",
                content_type=ContentType.BOOK,
                title="Book 1",
                summary="Summary 1",
                full_content="Content...",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime(2024, 1, 15),
                highlights=[],
            )
        ]
        themes = {}

        generator.generate_item_pages(items=items, themes=themes)

        assert (output_dir / "item" / "doc_1" / "index.html").exists()

    def test_generate_item_pages_includes_item_details(self, temp_dir):
        """generate_item_pages() should render title, summary, content type."""
        from src.library.static_generator import StaticGenerator

        output_dir = temp_dir / "library-site"
        generator = StaticGenerator(output_dir=output_dir)
        generator._ensure_output_dirs()

        items = [
            LibraryItem(
                id="doc_1",
                content_type=ContentType.BOOK,
                title="Thinking in Systems",
                summary="A primer on systems thinking with detailed analysis",
                full_content="Full content here...",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime(2024, 1, 15),
                highlights=[],
            )
        ]
        themes = {}

        generator.generate_item_pages(items=items, themes=themes)

        content = (output_dir / "item" / "doc_1" / "index.html").read_text()
        assert "Thinking in Systems" in content
        assert "A primer on systems thinking" in content
        assert "Book" in content  # Content type badge

    def test_generate_item_pages_includes_source_link(self, temp_dir):
        """generate_item_pages() should include source link when available."""
        from src.library.static_generator import StaticGenerator

        output_dir = temp_dir / "library-site"
        generator = StaticGenerator(output_dir=output_dir)
        generator._ensure_output_dirs()

        items = [
            LibraryItem(
                id="doc_1",
                content_type=ContentType.WEB_CONTENT,
                title="Web Article",
                summary="An interesting article",
                full_content="Content...",
                source_url="https://example.com/article",
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime(2024, 1, 15),
                highlights=[],
            )
        ]
        themes = {}

        generator.generate_item_pages(items=items, themes=themes)

        content = (output_dir / "item" / "doc_1" / "index.html").read_text()
        assert "https://example.com/article" in content
        assert "View Source" in content

    def test_generate_item_pages_includes_highlights(self, temp_dir):
        """generate_item_pages() should render highlights/excerpts."""
        from src.library.static_generator import StaticGenerator

        output_dir = temp_dir / "library-site"
        generator = StaticGenerator(output_dir=output_dir)
        generator._ensure_output_dirs()

        items = [
            LibraryItem(
                id="doc_1",
                content_type=ContentType.BOOK,
                title="Book Title",
                summary="Summary",
                full_content="Content...",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime(2024, 1, 15),
                highlights=[
                    "This is an important excerpt from the book",
                    "Another key takeaway to remember",
                ],
            )
        ]
        themes = {}

        generator.generate_item_pages(items=items, themes=themes)

        content = (output_dir / "item" / "doc_1" / "index.html").read_text()
        assert "This is an important excerpt from the book" in content
        assert "Another key takeaway to remember" in content

    def test_generate_item_pages_includes_theme_badges(self, temp_dir):
        """generate_item_pages() should show theme badges for assigned themes."""
        from src.library.static_generator import StaticGenerator

        output_dir = temp_dir / "library-site"
        generator = StaticGenerator(output_dir=output_dir)
        generator._ensure_output_dirs()

        themes = {
            "systems-thinking": Theme(
                id="systems-thinking",
                name="Systems Thinking",
                description="Understanding complex systems",
                keywords=["systems"],
                item_count=1,
                created_at=datetime(2024, 1, 15),
                updated_at=datetime(2024, 1, 20),
            )
        }
        items = [
            LibraryItem(
                id="doc_1",
                content_type=ContentType.BOOK,
                title="Book Title",
                summary="Summary",
                full_content="Content...",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=["systems-thinking"],
                created_at=datetime(2024, 1, 15),
                highlights=[],
            )
        ]

        generator.generate_item_pages(items=items, themes=themes)

        content = (output_dir / "item" / "doc_1" / "index.html").read_text()
        assert "Systems Thinking" in content
        assert "/theme/systems-thinking/" in content

    def test_generate_item_pages_hides_source_link_when_none(self, temp_dir):
        """generate_item_pages() should not show source link when source_url is None."""
        from src.library.static_generator import StaticGenerator

        output_dir = temp_dir / "library-site"
        generator = StaticGenerator(output_dir=output_dir)
        generator._ensure_output_dirs()

        items = [
            LibraryItem(
                id="doc_1",
                content_type=ContentType.VALUE,
                title="A Core Value",
                summary="Description of the value",
                full_content="Content...",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime(2024, 1, 15),
                highlights=[],
            )
        ]

        generator.generate_item_pages(items=items, themes={})

        content = (output_dir / "item" / "doc_1" / "index.html").read_text()
        # View Source button should not appear
        assert "View Source" not in content or "display: none" in content

    def test_generate_item_pages_returns_list_of_paths(self, temp_dir):
        """generate_item_pages() should return list of generated file paths."""
        from src.library.static_generator import StaticGenerator

        output_dir = temp_dir / "library-site"
        generator = StaticGenerator(output_dir=output_dir)
        generator._ensure_output_dirs()

        items = [
            LibraryItem(
                id=f"doc_{i}",
                content_type=ContentType.BOOK,
                title=f"Book {i}",
                summary=f"Summary {i}",
                full_content="Content...",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime(2024, 1, i + 1),
                highlights=[],
            )
            for i in range(3)
        ]

        result = generator.generate_item_pages(items=items, themes={})

        assert len(result) == 3
        for i in range(3):
            assert output_dir / "item" / f"doc_{i}" / "index.html" in result


class TestGenerateAllPage:
    """Tests for generate_all_page() method that renders all/index.html."""

    def test_generate_all_page_creates_index_html(self, temp_dir):
        """generate_all_page() should create all/index.html."""
        from src.library.static_generator import StaticGenerator

        output_dir = temp_dir / "library-site"
        generator = StaticGenerator(output_dir=output_dir)
        generator._ensure_output_dirs()

        generator.generate_all_page(items=[])

        assert (output_dir / "all" / "index.html").exists()

    def test_generate_all_page_includes_all_items(self, temp_dir):
        """generate_all_page() should include cards for all items."""
        from src.library.static_generator import StaticGenerator

        output_dir = temp_dir / "library-site"
        generator = StaticGenerator(output_dir=output_dir)
        generator._ensure_output_dirs()

        items = [
            LibraryItem(
                id="doc_1",
                content_type=ContentType.BOOK,
                title="First Book",
                summary="Summary 1",
                full_content="Content...",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime(2024, 1, 15),
                highlights=[],
            ),
            LibraryItem(
                id="doc_2",
                content_type=ContentType.ARTICLE,
                title="Second Article",
                summary="Summary 2",
                full_content="Content...",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime(2024, 1, 16),
                highlights=[],
            ),
        ]

        generator.generate_all_page(items=items)

        content = (output_dir / "all" / "index.html").read_text()
        assert "First Book" in content
        assert "Second Article" in content

    def test_generate_all_page_shows_item_count(self, temp_dir):
        """generate_all_page() should display the total item count."""
        from src.library.static_generator import StaticGenerator

        output_dir = temp_dir / "library-site"
        generator = StaticGenerator(output_dir=output_dir)
        generator._ensure_output_dirs()

        items = [
            LibraryItem(
                id=f"doc_{i}",
                content_type=ContentType.BOOK,
                title=f"Book {i}",
                summary=f"Summary {i}",
                full_content="Content...",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime(2024, 1, i + 1),
                highlights=[],
            )
            for i in range(10)
        ]

        generator.generate_all_page(items=items)

        content = (output_dir / "all" / "index.html").read_text()
        assert "10" in content  # Should show 10 items

    def test_generate_all_page_includes_sort_controls(self, temp_dir):
        """generate_all_page() should include sort dropdown (Recent, A-Z, Type)."""
        from src.library.static_generator import StaticGenerator

        output_dir = temp_dir / "library-site"
        generator = StaticGenerator(output_dir=output_dir)
        generator._ensure_output_dirs()

        generator.generate_all_page(items=[])

        content = (output_dir / "all" / "index.html").read_text()
        assert "sort-select" in content or "Sort" in content
        assert "Recent" in content
        assert "A-Z" in content

    def test_generate_all_page_empty_shows_message(self, temp_dir):
        """generate_all_page() should show 'No content yet' when items is empty."""
        from src.library.static_generator import StaticGenerator

        output_dir = temp_dir / "library-site"
        generator = StaticGenerator(output_dir=output_dir)
        generator._ensure_output_dirs()

        generator.generate_all_page(items=[])

        content = (output_dir / "all" / "index.html").read_text()
        assert "No content" in content or "empty" in content.lower()

    def test_generate_all_page_returns_path(self, temp_dir):
        """generate_all_page() should return the path to the generated file."""
        from src.library.static_generator import StaticGenerator

        output_dir = temp_dir / "library-site"
        generator = StaticGenerator(output_dir=output_dir)
        generator._ensure_output_dirs()

        result = generator.generate_all_page(items=[])

        assert result == output_dir / "all" / "index.html"


class TestGenerateSearchPage:
    """Tests for generate_search_page() method that renders search/index.html."""

    def test_generate_search_page_creates_index_html(self, temp_dir):
        """generate_search_page() should create search/index.html."""
        from src.library.static_generator import StaticGenerator

        output_dir = temp_dir / "library-site"
        generator = StaticGenerator(output_dir=output_dir)
        generator._ensure_output_dirs()

        generator.generate_search_page()

        assert (output_dir / "search" / "index.html").exists()

    def test_generate_search_page_includes_search_input(self, temp_dir):
        """generate_search_page() should include search input field."""
        from src.library.static_generator import StaticGenerator

        output_dir = temp_dir / "library-site"
        generator = StaticGenerator(output_dir=output_dir)
        generator._ensure_output_dirs()

        generator.generate_search_page()

        content = (output_dir / "search" / "index.html").read_text()
        assert 'type="search"' in content or 'search-input' in content

    def test_generate_search_page_includes_results_container(self, temp_dir):
        """generate_search_page() should include container for search results."""
        from src.library.static_generator import StaticGenerator

        output_dir = temp_dir / "library-site"
        generator = StaticGenerator(output_dir=output_dir)
        generator._ensure_output_dirs()

        generator.generate_search_page()

        content = (output_dir / "search" / "index.html").read_text()
        assert "search-results" in content

    def test_generate_search_page_includes_no_results_state(self, temp_dir):
        """generate_search_page() should include 'no results' message element."""
        from src.library.static_generator import StaticGenerator

        output_dir = temp_dir / "library-site"
        generator = StaticGenerator(output_dir=output_dir)
        generator._ensure_output_dirs()

        generator.generate_search_page()

        content = (output_dir / "search" / "index.html").read_text()
        assert "no-results" in content or "No results" in content

    def test_generate_search_page_returns_path(self, temp_dir):
        """generate_search_page() should return the path to the generated file."""
        from src.library.static_generator import StaticGenerator

        output_dir = temp_dir / "library-site"
        generator = StaticGenerator(output_dir=output_dir)
        generator._ensure_output_dirs()

        result = generator.generate_search_page()

        assert result == output_dir / "search" / "index.html"


class TestCopyAssets:
    """Tests for copy_assets() method that copies CSS, JS, images."""

    def test_copy_assets_copies_css_file(self, temp_dir):
        """copy_assets() should copy styles.css to assets/css/."""
        from src.library.static_generator import StaticGenerator

        output_dir = temp_dir / "library-site"
        generator = StaticGenerator(output_dir=output_dir)
        generator._ensure_output_dirs()

        generator.copy_assets()

        assert (output_dir / "assets" / "css" / "styles.css").exists()

    def test_copy_assets_copies_js_file(self, temp_dir):
        """copy_assets() should copy search.js to assets/js/."""
        from src.library.static_generator import StaticGenerator

        output_dir = temp_dir / "library-site"
        generator = StaticGenerator(output_dir=output_dir)
        generator._ensure_output_dirs()

        generator.copy_assets()

        assert (output_dir / "assets" / "js" / "search.js").exists()

    def test_copy_assets_copies_placeholder_images(self, temp_dir):
        """copy_assets() should copy placeholder images to assets/images/placeholders/."""
        from src.library.static_generator import StaticGenerator

        output_dir = temp_dir / "library-site"
        generator = StaticGenerator(output_dir=output_dir)
        generator._ensure_output_dirs()

        generator.copy_assets()

        placeholders_dir = output_dir / "assets" / "images" / "placeholders"
        assert placeholders_dir.exists()
        # Check for at least one placeholder file
        placeholder_files = list(placeholders_dir.glob("*.svg"))
        assert len(placeholder_files) > 0

    def test_copy_assets_preserves_file_content(self, temp_dir):
        """copy_assets() should copy files with correct content."""
        from src.library.static_generator import StaticGenerator

        output_dir = temp_dir / "library-site"
        generator = StaticGenerator(output_dir=output_dir)
        generator._ensure_output_dirs()

        generator.copy_assets()

        # Read the copied CSS file
        css_content = (output_dir / "assets" / "css" / "styles.css").read_text()
        # Should contain some CSS (background color from design system)
        assert "#FAFAFA" in css_content or "fafafa" in css_content.lower()

    def test_copy_assets_is_idempotent(self, temp_dir):
        """copy_assets() should be safe to call multiple times."""
        from src.library.static_generator import StaticGenerator

        output_dir = temp_dir / "library-site"
        generator = StaticGenerator(output_dir=output_dir)
        generator._ensure_output_dirs()

        generator.copy_assets()
        generator.copy_assets()  # Call again

        assert (output_dir / "assets" / "css" / "styles.css").exists()
        assert (output_dir / "assets" / "js" / "search.js").exists()

    def test_copy_assets_returns_list_of_copied_files(self, temp_dir):
        """copy_assets() should return list of copied file paths."""
        from src.library.static_generator import StaticGenerator

        output_dir = temp_dir / "library-site"
        generator = StaticGenerator(output_dir=output_dir)
        generator._ensure_output_dirs()

        result = generator.copy_assets()

        assert isinstance(result, list)
        assert len(result) > 0
        # Should include at least CSS and JS
        css_copied = any("styles.css" in str(p) for p in result)
        js_copied = any("search.js" in str(p) for p in result)
        assert css_copied
        assert js_copied


class TestGenerateAll:
    """Tests for generate_all() method that orchestrates full generation."""

    def test_generate_all_creates_complete_site_structure(self, temp_dir):
        """generate_all() should create complete site with all directories."""
        from src.library.static_generator import StaticGenerator

        output_dir = temp_dir / "library-site"
        generator = StaticGenerator(output_dir=output_dir)

        themes = [
            Theme(
                id="theme-1",
                name="Theme 1",
                description="Description 1",
                keywords=["keyword1"],
                item_count=1,
                created_at=datetime(2024, 1, 15),
                updated_at=datetime(2024, 1, 20),
            )
        ]
        items = [
            LibraryItem(
                id="doc_1",
                content_type=ContentType.BOOK,
                title="Book 1",
                summary="Summary 1",
                full_content="Content...",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=["theme-1"],
                created_at=datetime(2024, 1, 15),
                highlights=[],
            )
        ]
        assignments = {"theme-1": ["doc_1"]}

        generator.generate_all(themes=themes, items=items, assignments=assignments)

        # Check all expected files/directories exist
        assert (output_dir / "index.html").exists()
        assert (output_dir / "all" / "index.html").exists()
        assert (output_dir / "search" / "index.html").exists()
        assert (output_dir / "theme" / "theme-1" / "index.html").exists()
        assert (output_dir / "item" / "doc_1" / "index.html").exists()
        assert (output_dir / "assets" / "css" / "styles.css").exists()
        assert (output_dir / "assets" / "js" / "search.js").exists()

    def test_generate_all_with_empty_library(self, temp_dir):
        """generate_all() should handle empty library gracefully."""
        from src.library.static_generator import StaticGenerator

        output_dir = temp_dir / "library-site"
        generator = StaticGenerator(output_dir=output_dir)

        generator.generate_all(themes=[], items=[], assignments={})

        # Should still create basic structure
        assert (output_dir / "index.html").exists()
        assert (output_dir / "all" / "index.html").exists()
        assert (output_dir / "search" / "index.html").exists()
        assert (output_dir / "assets" / "css" / "styles.css").exists()

    def test_generate_all_returns_summary(self, temp_dir):
        """generate_all() should return a summary of generated files."""
        from src.library.static_generator import StaticGenerator

        output_dir = temp_dir / "library-site"
        generator = StaticGenerator(output_dir=output_dir)

        themes = [
            Theme(
                id="theme-1",
                name="Theme 1",
                description="Description 1",
                keywords=[],
                item_count=1,
                created_at=datetime(2024, 1, 15),
                updated_at=datetime(2024, 1, 20),
            )
        ]
        items = [
            LibraryItem(
                id="doc_1",
                content_type=ContentType.BOOK,
                title="Book 1",
                summary="Summary 1",
                full_content="Content...",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=["theme-1"],
                created_at=datetime(2024, 1, 15),
                highlights=[],
            )
        ]

        result = generator.generate_all(
            themes=themes, items=items, assignments={"theme-1": ["doc_1"]}
        )

        # Should return some kind of summary
        assert result is not None
        assert isinstance(result, dict)
        # Should include counts or paths
        assert "pages_generated" in result or "home" in result

    def test_generate_all_generates_valid_html(self, temp_dir):
        """generate_all() should produce valid HTML with proper structure."""
        from src.library.static_generator import StaticGenerator

        output_dir = temp_dir / "library-site"
        generator = StaticGenerator(output_dir=output_dir)

        generator.generate_all(themes=[], items=[], assignments={})

        # Check home page has valid HTML structure
        content = (output_dir / "index.html").read_text()
        assert "<!DOCTYPE html>" in content
        assert "<html" in content
        assert "</html>" in content
        assert "<head>" in content
        assert "</head>" in content
        assert "<body>" in content
        assert "</body>" in content

    def test_generate_all_links_css_and_js(self, temp_dir):
        """generate_all() should link to CSS and JS files in generated pages."""
        from src.library.static_generator import StaticGenerator

        output_dir = temp_dir / "library-site"
        generator = StaticGenerator(output_dir=output_dir)

        generator.generate_all(themes=[], items=[], assignments={})

        content = (output_dir / "index.html").read_text()
        assert "/assets/css/styles.css" in content
        assert "/assets/js/search.js" in content

    def test_generate_all_escapes_special_characters(self, temp_dir):
        """generate_all() should properly escape special characters in content."""
        from src.library.static_generator import StaticGenerator

        output_dir = temp_dir / "library-site"
        generator = StaticGenerator(output_dir=output_dir)

        items = [
            LibraryItem(
                id="doc_1",
                content_type=ContentType.ARTICLE,
                title="Article with <script>alert('XSS')</script>",
                summary="Testing & escaping of <special> characters",
                full_content="Content with \"quotes\" and 'apostrophes'",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime(2024, 1, 15),
                highlights=[],
            )
        ]

        generator.generate_all(themes=[], items=items, assignments={})

        content = (output_dir / "item" / "doc_1" / "index.html").read_text()
        # Should not contain unescaped script tags
        assert "<script>alert" not in content
        # Should have escaped versions
        assert "&lt;script&gt;" in content or "script" not in content.lower()
