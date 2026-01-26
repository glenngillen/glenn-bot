"""Tests for library edge cases and polish.

This module tests edge cases in the library build process including:
- Empty knowledge base handling (Task 173)
- Long content truncation (Task 175)
- Special character escaping (Task 177)
- Items without source_url (Task 179)
- Items without highlights (Task 181)
- Small knowledge base with minimum themes (Task 183)

These tests ensure the library handles edge conditions gracefully and
displays appropriate messages to users.
"""

import json
import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.library.models import LibraryItem, ContentType, Theme, ThemeAssignment


class TestEmptyKnowledgeBaseHandling:
    """Tests for empty knowledge base handling (Task 173).

    When the knowledge base is empty, the library should:
    - Still generate a valid site structure
    - Display a "No content yet" message in appropriate places
    - Handle the build process without errors
    - Allow the user to add content later and rebuild
    """

    def test_builder_handles_empty_knowledge_base(self, temp_dir):
        """LibraryBuilder should complete successfully with empty knowledge base."""
        from src.library.builder import LibraryBuilder

        mock_knowledge_base = MagicMock()
        # Return empty documents from knowledge base
        mock_knowledge_base.export_knowledge.return_value = {
            "documents": [],
            "total_documents": 0,
        }

        mock_ollama_client = MagicMock()
        mock_ollama_client.generate.return_value = "[]"

        data_dir = temp_dir / "library"
        site_dir = temp_dir / "library-site"

        builder = LibraryBuilder(
            knowledge_base=mock_knowledge_base,
            ollama_client=mock_ollama_client,
            data_dir=data_dir,
            site_dir=site_dir,
        )

        # Build should complete without error
        result = builder.build()

        # Should return valid summary with zero items
        assert result is not None
        assert result["items_count"] == 0

    def test_builder_creates_site_structure_with_empty_knowledge_base(self, temp_dir):
        """LibraryBuilder should create valid site structure even when empty."""
        from src.library.builder import LibraryBuilder

        mock_knowledge_base = MagicMock()
        mock_knowledge_base.export_knowledge.return_value = {"documents": []}

        mock_ollama_client = MagicMock()
        mock_ollama_client.generate.return_value = "[]"

        data_dir = temp_dir / "library"
        site_dir = temp_dir / "library-site"

        builder = LibraryBuilder(
            knowledge_base=mock_knowledge_base,
            ollama_client=mock_ollama_client,
            data_dir=data_dir,
            site_dir=site_dir,
        )

        builder.build()

        # Core site structure should still exist
        assert (site_dir / "index.html").exists()
        assert (site_dir / "all" / "index.html").exists()
        assert (site_dir / "search" / "index.html").exists()
        assert (site_dir / "assets" / "css" / "styles.css").exists()
        assert (site_dir / "assets" / "js" / "search.js").exists()

    def test_content_exporter_returns_empty_list_for_empty_knowledge_base(self, temp_dir):
        """ContentExporter should return empty list when knowledge base has no documents."""
        from src.library.content_exporter import ContentExporter

        mock_knowledge_base = MagicMock()
        mock_knowledge_base.export_knowledge.return_value = {
            "documents": [],
            "total_documents": 0,
        }

        exporter = ContentExporter(mock_knowledge_base)
        items = exporter.export_all()

        assert items == []
        assert isinstance(items, list)

    def test_home_page_shows_empty_message_when_no_themes(self, temp_dir):
        """Home page should display 'No themes yet' when themes list is empty."""
        from src.library.static_generator import StaticGenerator

        output_dir = temp_dir / "library-site"
        generator = StaticGenerator(output_dir=output_dir)

        generator.generate_home_page(themes=[], items=[])

        content = (output_dir / "index.html").read_text()

        # Should contain empty state message
        assert "No themes yet" in content or "no themes" in content.lower()

    def test_all_page_shows_empty_message_when_no_items(self, temp_dir):
        """All page should display 'No content yet' when items list is empty."""
        from src.library.static_generator import StaticGenerator

        output_dir = temp_dir / "library-site"
        generator = StaticGenerator(output_dir=output_dir)

        # Ensure directories exist first
        generator._ensure_output_dirs()

        generator.generate_all_page(items=[])

        content = (output_dir / "all" / "index.html").read_text()

        # Should contain empty state message
        assert "No content" in content or "empty" in content.lower()

    def test_theme_generator_handles_empty_items_list(self, temp_dir):
        """ThemeGenerator should handle empty items list gracefully."""
        from src.library.theme_generator import ThemeGenerator

        mock_ollama_client = MagicMock()
        mock_ollama_client.generate.return_value = "[]"

        data_dir = temp_dir / "library"
        generator = ThemeGenerator(
            ollama_client=mock_ollama_client, data_dir=data_dir
        )

        # Should not raise an exception
        themes = generator.generate_themes([])

        # Should return empty list or minimal themes
        assert isinstance(themes, list)

    def test_cover_resolver_handles_empty_items_list(self, temp_dir):
        """CoverResolver should handle empty items list gracefully."""
        from src.library.cover_resolver import CoverResolver

        cache_dir = temp_dir / "library"
        resolver = CoverResolver(cache_dir=cache_dir)

        # Should not raise an exception
        result = resolver.resolve_all_covers([])

        assert result == []
        assert isinstance(result, list)

    def test_search_indexer_handles_empty_items_list(self, temp_dir):
        """SearchIndexer should handle empty items list gracefully."""
        from src.library.search_indexer import SearchIndexer

        output_dir = temp_dir / "library-site"
        indexer = SearchIndexer(output_dir=output_dir)

        # Should not raise an exception
        indexer.write_index(items=[], themes={})

        # Search index file should still be created
        index_file = output_dir / "assets" / "js" / "search-index.json"
        assert index_file.exists()

        # Should contain empty items array
        content = json.loads(index_file.read_text())
        assert content["items"] == []

    def test_build_state_saved_for_empty_knowledge_base(self, temp_dir):
        """Build state should be saved even when knowledge base is empty."""
        from src.library.builder import LibraryBuilder

        mock_knowledge_base = MagicMock()
        mock_knowledge_base.export_knowledge.return_value = {"documents": []}

        mock_ollama_client = MagicMock()
        mock_ollama_client.generate.return_value = "[]"

        data_dir = temp_dir / "library"
        site_dir = temp_dir / "library-site"

        builder = LibraryBuilder(
            knowledge_base=mock_knowledge_base,
            ollama_client=mock_ollama_client,
            data_dir=data_dir,
            site_dir=site_dir,
        )

        builder.build()

        # Build state should exist
        build_state_file = site_dir / "_build_state.json"
        assert build_state_file.exists()

        # Should contain valid state
        state = json.loads(build_state_file.read_text())
        assert "last_build" in state
        assert "item_hashes" in state
        assert state["item_hashes"] == {}  # Empty since no items

    def test_library_json_created_for_empty_knowledge_base(self, temp_dir):
        """Library.json debug file should be created even when empty."""
        from src.library.builder import LibraryBuilder

        mock_knowledge_base = MagicMock()
        mock_knowledge_base.export_knowledge.return_value = {"documents": []}

        mock_ollama_client = MagicMock()
        mock_ollama_client.generate.return_value = "[]"

        data_dir = temp_dir / "library"
        site_dir = temp_dir / "library-site"

        builder = LibraryBuilder(
            knowledge_base=mock_knowledge_base,
            ollama_client=mock_ollama_client,
            data_dir=data_dir,
            site_dir=site_dir,
        )

        builder.build()

        # library.json should exist
        library_json_file = site_dir / "_data" / "library.json"
        assert library_json_file.exists()

        # Should contain empty items and themes
        data = json.loads(library_json_file.read_text())
        assert data["items"] == []
        assert isinstance(data["themes"], list)

    def test_incremental_build_works_after_empty_initial_build(self, temp_dir):
        """Second build should work correctly after an initial empty build."""
        from src.library.builder import LibraryBuilder
        from src.library.models import ContentType

        mock_knowledge_base = MagicMock()
        mock_ollama_client = MagicMock()

        data_dir = temp_dir / "library"
        site_dir = temp_dir / "library-site"

        builder = LibraryBuilder(
            knowledge_base=mock_knowledge_base,
            ollama_client=mock_ollama_client,
            data_dir=data_dir,
            site_dir=site_dir,
        )

        # First build - empty
        mock_knowledge_base.export_knowledge.return_value = {"documents": []}
        mock_ollama_client.generate.return_value = "[]"

        result1 = builder.build()
        assert result1["items_count"] == 0

        # Second build - with content
        mock_knowledge_base.export_knowledge.return_value = {
            "documents": [
                {
                    "id": "doc_1",
                    "content": "Test content",
                    "metadata": {"type": "book", "name": "Test Book"},
                }
            ]
        }
        mock_ollama_client.generate.return_value = json.dumps([
            {
                "id": "test-theme",
                "name": "Test Theme",
                "description": "A test theme",
                "keywords": ["test"],
            }
        ])

        result2 = builder.build()
        assert result2["items_count"] == 1

        # Verify new content pages were created
        assert (site_dir / "item" / "doc_1" / "index.html").exists()

    def test_empty_state_message_styling(self, temp_dir):
        """Empty state messages should include styling classes for proper display."""
        from src.library.static_generator import StaticGenerator

        output_dir = temp_dir / "library-site"
        generator = StaticGenerator(output_dir=output_dir)

        # Ensure directories exist first
        generator._ensure_output_dirs()

        generator.generate_all_page(items=[])

        content = (output_dir / "all" / "index.html").read_text()

        # Should include empty-state CSS class for styling
        assert "empty-state" in content

    def test_home_page_shows_zero_stats_when_empty(self, temp_dir):
        """Home page should show 0 total items when knowledge base is empty."""
        from src.library.static_generator import StaticGenerator

        output_dir = temp_dir / "library-site"
        generator = StaticGenerator(output_dir=output_dir)

        generator.generate_home_page(themes=[], items=[])

        content = (output_dir / "index.html").read_text()

        # Should show zero items in stats
        assert "0" in content  # At minimum, should display the count

    def test_search_works_with_empty_index(self, temp_dir):
        """Search functionality should work correctly with empty search index."""
        from src.library.search_indexer import SearchIndexer

        output_dir = temp_dir / "library-site"
        indexer = SearchIndexer(output_dir=output_dir)

        indexer.write_index(items=[], themes={})

        # Search index should have valid structure even when empty
        index_file = output_dir / "assets" / "js" / "search-index.json"
        content = json.loads(index_file.read_text())

        assert "items" in content
        assert isinstance(content["items"], list)


class TestLongContentTruncation:
    """Tests for long content truncation (Task 175).

    When content is longer than display limits, the library should:
    - Truncate titles to max 2 lines with ellipsis (via CSS)
    - Truncate summaries to max 3 lines with ellipsis (via CSS)
    - Truncate very long full content on detail pages
    - Show full title on hover (via title attribute)
    """

    def test_css_includes_title_truncation(self):
        """CSS should include line-clamp for title truncation (2 lines)."""
        css_path = Path(__file__).parent.parent / "src" / "library" / "assets" / "css" / "styles.css"
        css_content = css_path.read_text()

        # Should have -webkit-line-clamp: 2 for titles
        assert "-webkit-line-clamp: 2" in css_content
        # Should have webkit-box-orient: vertical for line-clamp to work
        assert "-webkit-box-orient: vertical" in css_content
        # Should have overflow: hidden
        assert "overflow: hidden" in css_content

    def test_css_includes_summary_truncation(self):
        """CSS should include line-clamp for summary truncation (3 lines)."""
        css_path = Path(__file__).parent.parent / "src" / "library" / "assets" / "css" / "styles.css"
        css_content = css_path.read_text()

        # Should have -webkit-line-clamp: 3 for summaries
        assert "-webkit-line-clamp: 3" in css_content

    def test_card_title_class_has_truncation_styles(self):
        """card-title class should have all required truncation CSS properties."""
        css_path = Path(__file__).parent.parent / "src" / "library" / "assets" / "css" / "styles.css"
        css_content = css_path.read_text()

        # Find the card-title block and verify it has truncation
        import re
        card_title_match = re.search(r'\.card-title\s*\{[^}]+\}', css_content, re.DOTALL)
        assert card_title_match is not None, "card-title class not found in CSS"

        card_title_css = card_title_match.group()
        assert "-webkit-line-clamp" in card_title_css, "card-title missing line-clamp"
        assert "-webkit-box-orient: vertical" in card_title_css, "card-title missing box-orient"
        assert "overflow: hidden" in card_title_css, "card-title missing overflow"

    def test_card_summary_class_has_truncation_styles(self):
        """card-summary class should have all required truncation CSS properties."""
        css_path = Path(__file__).parent.parent / "src" / "library" / "assets" / "css" / "styles.css"
        css_content = css_path.read_text()

        # Find the card-summary block and verify it has truncation
        import re
        card_summary_match = re.search(r'\.card-summary\s*\{[^}]+\}', css_content, re.DOTALL)
        assert card_summary_match is not None, "card-summary class not found in CSS"

        card_summary_css = card_summary_match.group()
        assert "-webkit-line-clamp" in card_summary_css, "card-summary missing line-clamp"
        assert "-webkit-box-orient: vertical" in card_summary_css, "card-summary missing box-orient"
        assert "overflow: hidden" in card_summary_css, "card-summary missing overflow"

    def test_long_title_rendered_in_card_template(self, temp_dir):
        """Long titles should be rendered in cards and truncated by CSS."""
        from src.library.static_generator import StaticGenerator
        from src.library.models import LibraryItem, ContentType

        output_dir = temp_dir / "library-site"
        generator = StaticGenerator(output_dir=output_dir)

        # Create item with very long title
        long_title = "This is a very long title that spans multiple lines and should be truncated by CSS rules using the webkit-line-clamp property"
        item = LibraryItem(
            id="long-title-item",
            content_type=ContentType.BOOK,
            title=long_title,
            summary="Short summary",
            full_content="Full content",
            source_url=None,
            cover_image_url=None,
            metadata={},
            themes=[],
            created_at=datetime.now(),
            highlights=[],
        )

        generator._ensure_output_dirs()
        generator.generate_all_page(items=[item])

        content = (output_dir / "all" / "index.html").read_text()

        # Long title should be present (truncation is handled by CSS)
        assert long_title in content
        # Card class should be present for CSS to apply
        assert "card-title" in content

    def test_long_summary_rendered_in_card_template(self, temp_dir):
        """Long summaries should be rendered in cards and truncated by CSS."""
        from src.library.static_generator import StaticGenerator
        from src.library.models import LibraryItem, ContentType

        output_dir = temp_dir / "library-site"
        generator = StaticGenerator(output_dir=output_dir)

        # Create item with very long summary
        long_summary = "A" * 500  # 500 character summary
        item = LibraryItem(
            id="long-summary-item",
            content_type=ContentType.ARTICLE,
            title="Test Article",
            summary=long_summary,
            full_content="Full content",
            source_url=None,
            cover_image_url=None,
            metadata={},
            themes=[],
            created_at=datetime.now(),
            highlights=[],
        )

        generator._ensure_output_dirs()
        generator.generate_all_page(items=[item])

        content = (output_dir / "all" / "index.html").read_text()

        # Long summary should be present (truncation is handled by CSS)
        assert long_summary in content
        # Card summary class should be present for CSS to apply
        assert "card-summary" in content

    def test_very_long_full_content_truncated_on_detail_page(self, temp_dir):
        """Very long full_content should be truncated on detail pages."""
        from src.library.static_generator import StaticGenerator
        from src.library.models import LibraryItem, ContentType

        output_dir = temp_dir / "library-site"
        generator = StaticGenerator(output_dir=output_dir)

        # Create item with very long full_content (over 5000 chars)
        long_content = "A" * 6000  # 6000 characters
        item = LibraryItem(
            id="long-content-item",
            content_type=ContentType.BOOK,
            title="Long Content Book",
            summary="Short summary",
            full_content=long_content,
            source_url=None,
            cover_image_url=None,
            metadata={},
            themes=[],
            created_at=datetime.now(),
            highlights=[],
        )

        generator._ensure_output_dirs()
        generator.generate_item_pages(items=[item], themes={})

        content = (output_dir / "item" / "long-content-item" / "index.html").read_text()

        # Full content should be truncated (not all 6000 chars)
        assert long_content not in content
        # Should contain truncation indicator
        assert "..." in content
        # Should still contain the first part of the content
        assert "A" * 100 in content

    def test_content_exporter_truncates_summary(self, temp_dir):
        """ContentExporter should truncate summary to ~200 characters."""
        from src.library.content_exporter import ContentExporter

        mock_knowledge_base = MagicMock()
        long_content = "A" * 500  # 500 character content

        mock_knowledge_base.export_knowledge.return_value = {
            "documents": [
                {
                    "id": "doc-1",
                    "content": long_content,
                    "metadata": {"type": "book", "name": "Test Book"},
                }
            ]
        }

        exporter = ContentExporter(mock_knowledge_base)
        items = exporter.export_all()

        assert len(items) == 1
        # Summary should be truncated (max 200 chars + ellipsis)
        assert len(items[0].summary) <= 203  # 200 chars + "..."
        # Full content should be preserved
        assert items[0].full_content == long_content

    def test_long_title_has_full_text_in_item_page(self, temp_dir):
        """Item detail page should show full title (not truncated)."""
        from src.library.static_generator import StaticGenerator
        from src.library.models import LibraryItem, ContentType

        output_dir = temp_dir / "library-site"
        generator = StaticGenerator(output_dir=output_dir)

        long_title = "This is a very long title that would be truncated in cards but should appear in full on the detail page"
        item = LibraryItem(
            id="detail-item",
            content_type=ContentType.BOOK,
            title=long_title,
            summary="Summary",
            full_content="Content",
            source_url=None,
            cover_image_url=None,
            metadata={},
            themes=[],
            created_at=datetime.now(),
            highlights=[],
        )

        generator._ensure_output_dirs()
        generator.generate_item_pages(items=[item], themes={})

        content = (output_dir / "item" / "detail-item" / "index.html").read_text()

        # Full title should appear on detail page
        assert long_title in content


class TestSpecialCharacterEscaping:
    """Tests for special character escaping (Task 177).

    When content contains special HTML characters, the library should:
    - Escape < > & " ' characters in HTML output
    - Prevent XSS by escaping script tags and JavaScript
    - Use Jinja2 autoescape for titles, summaries, theme names
    - Handle Unicode characters correctly
    """

    def test_jinja2_autoescape_is_enabled(self, temp_dir):
        """Verify that Jinja2 autoescape is enabled for HTML templates."""
        from src.library.static_generator import StaticGenerator

        output_dir = temp_dir / "library-site"
        generator = StaticGenerator(output_dir=output_dir)

        # Check that autoescape is enabled in the environment
        assert generator.env.autoescape is True or callable(generator.env.autoescape)

    def test_html_tags_escaped_in_title(self, temp_dir):
        """HTML tags in titles should be escaped, not rendered as HTML."""
        from src.library.static_generator import StaticGenerator
        from src.library.models import LibraryItem, ContentType

        output_dir = temp_dir / "library-site"
        generator = StaticGenerator(output_dir=output_dir)

        # Create item with HTML in title
        malicious_title = "<script>alert('XSS')</script>Title"
        item = LibraryItem(
            id="xss-item",
            content_type=ContentType.BOOK,
            title=malicious_title,
            summary="Normal summary",
            full_content="Normal content",
            source_url=None,
            cover_image_url=None,
            metadata={},
            themes=[],
            created_at=datetime.now(),
            highlights=[],
        )

        generator._ensure_output_dirs()
        generator.generate_all_page(items=[item])

        content = (output_dir / "all" / "index.html").read_text()

        # HTML should be escaped (< becomes &lt;, > becomes &gt;)
        assert "<script>" not in content
        assert "&lt;script&gt;" in content

    def test_html_tags_escaped_in_summary(self, temp_dir):
        """HTML tags in summaries should be escaped."""
        from src.library.static_generator import StaticGenerator
        from src.library.models import LibraryItem, ContentType

        output_dir = temp_dir / "library-site"
        generator = StaticGenerator(output_dir=output_dir)

        # Create item with HTML in summary
        malicious_summary = "Summary with <img src=x onerror=alert('XSS')> image"
        item = LibraryItem(
            id="xss-summary-item",
            content_type=ContentType.ARTICLE,
            title="Normal Title",
            summary=malicious_summary,
            full_content="Normal content",
            source_url=None,
            cover_image_url=None,
            metadata={},
            themes=[],
            created_at=datetime.now(),
            highlights=[],
        )

        generator._ensure_output_dirs()
        generator.generate_all_page(items=[item])

        content = (output_dir / "all" / "index.html").read_text()

        # HTML should be escaped
        assert "<img src=x" not in content
        assert "&lt;img" in content

    def test_ampersand_escaped_in_content(self, temp_dir):
        """Ampersands (&) should be escaped as &amp; in HTML."""
        from src.library.static_generator import StaticGenerator
        from src.library.models import LibraryItem, ContentType

        output_dir = temp_dir / "library-site"
        generator = StaticGenerator(output_dir=output_dir)

        # Create item with ampersand in title
        title_with_ampersand = "AT&T Corporation & Others"
        item = LibraryItem(
            id="ampersand-item",
            content_type=ContentType.ARTICLE,
            title=title_with_ampersand,
            summary="Summary",
            full_content="Content",
            source_url=None,
            cover_image_url=None,
            metadata={},
            themes=[],
            created_at=datetime.now(),
            highlights=[],
        )

        generator._ensure_output_dirs()
        generator.generate_all_page(items=[item])

        content = (output_dir / "all" / "index.html").read_text()

        # Ampersands should be escaped
        assert "&amp;" in content

    def test_quotes_escaped_in_attributes(self, temp_dir):
        """Quotes in content should be properly escaped when used in attributes."""
        from src.library.static_generator import StaticGenerator
        from src.library.models import LibraryItem, ContentType

        output_dir = temp_dir / "library-site"
        generator = StaticGenerator(output_dir=output_dir)

        # Create item with quotes in title (used in img alt attribute)
        title_with_quotes = 'Book "Title" with \'quotes\''
        item = LibraryItem(
            id="quotes-item",
            content_type=ContentType.BOOK,
            title=title_with_quotes,
            summary="Summary",
            full_content="Content",
            source_url=None,
            cover_image_url="http://example.com/image.jpg",
            metadata={},
            themes=[],
            created_at=datetime.now(),
            highlights=[],
        )

        generator._ensure_output_dirs()
        generator.generate_item_pages(items=[item], themes={})

        content = (output_dir / "item" / "quotes-item" / "index.html").read_text()

        # The raw quotes should not break HTML attributes
        # Check that the page is valid (contains the escaped title in a heading)
        assert "Book" in content
        assert "Title" in content
        # Should not have broken HTML
        assert 'alt=""Title"' not in content

    def test_less_than_greater_than_escaped(self, temp_dir):
        """Less than (<) and greater than (>) should be escaped."""
        from src.library.static_generator import StaticGenerator
        from src.library.models import LibraryItem, ContentType

        output_dir = temp_dir / "library-site"
        generator = StaticGenerator(output_dir=output_dir)

        # Create item with comparison operators
        title_with_symbols = "Values where x < 10 and y > 5"
        item = LibraryItem(
            id="symbols-item",
            content_type=ContentType.FRAMEWORK,
            title=title_with_symbols,
            summary="Math content: a < b > c",
            full_content="Content",
            source_url=None,
            cover_image_url=None,
            metadata={},
            themes=[],
            created_at=datetime.now(),
            highlights=[],
        )

        generator._ensure_output_dirs()
        generator.generate_all_page(items=[item])

        content = (output_dir / "all" / "index.html").read_text()

        # < and > should be escaped
        assert "&lt;" in content  # < escaped
        assert "&gt;" in content  # > escaped

    def test_unicode_characters_preserved(self, temp_dir):
        """Unicode characters should be preserved correctly."""
        from src.library.static_generator import StaticGenerator
        from src.library.models import LibraryItem, ContentType

        output_dir = temp_dir / "library-site"
        generator = StaticGenerator(output_dir=output_dir)

        # Create item with Unicode characters
        unicode_title = "日本語タイトル with emoji 🚀 and accents éàü"
        item = LibraryItem(
            id="unicode-item",
            content_type=ContentType.ARTICLE,
            title=unicode_title,
            summary="Summary with émojis 🎉 and ñ",
            full_content="Content",
            source_url=None,
            cover_image_url=None,
            metadata={},
            themes=[],
            created_at=datetime.now(),
            highlights=[],
        )

        generator._ensure_output_dirs()
        generator.generate_all_page(items=[item])

        content = (output_dir / "all" / "index.html").read_text()

        # Unicode should be preserved
        assert "日本語タイトル" in content
        assert "🚀" in content
        assert "éàü" in content

    def test_theme_names_escaped(self, temp_dir):
        """Theme names with special characters should be escaped."""
        from src.library.static_generator import StaticGenerator
        from src.library.models import LibraryItem, ContentType, Theme

        output_dir = temp_dir / "library-site"
        generator = StaticGenerator(output_dir=output_dir)

        # Create theme with special characters
        theme = Theme(
            id="special-theme",
            name="Theme with <script>alert('XSS')</script>",
            description="Description with & symbols",
            keywords=["test"],
            item_count=1,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        generator._ensure_output_dirs()
        generator.generate_home_page(themes=[theme], items=[])

        content = (output_dir / "index.html").read_text()

        # HTML in theme name should be escaped
        assert "<script>" not in content
        assert "&lt;script&gt;" in content

    def test_highlight_text_escaped(self, temp_dir):
        """Highlight/excerpt text should be escaped."""
        from src.library.static_generator import StaticGenerator
        from src.library.models import LibraryItem, ContentType

        output_dir = temp_dir / "library-site"
        generator = StaticGenerator(output_dir=output_dir)

        # Create item with special characters in highlights
        item = LibraryItem(
            id="highlight-item",
            content_type=ContentType.BOOK,
            title="Test Book",
            summary="Summary",
            full_content="Content",
            source_url=None,
            cover_image_url=None,
            metadata={},
            themes=[],
            created_at=datetime.now(),
            highlights=[
                "Highlight with <em>tags</em>",
                "Highlight with & ampersand",
            ],
        )

        generator._ensure_output_dirs()
        generator.generate_item_pages(items=[item], themes={})

        content = (output_dir / "item" / "highlight-item" / "index.html").read_text()

        # HTML in highlights should be escaped
        assert "<em>" not in content
        assert "&lt;em&gt;" in content
        assert "&amp;" in content

    def test_source_url_not_escaped_improperly(self, temp_dir):
        """Source URLs should preserve query parameters correctly."""
        from src.library.static_generator import StaticGenerator
        from src.library.models import LibraryItem, ContentType

        output_dir = temp_dir / "library-site"
        generator = StaticGenerator(output_dir=output_dir)

        # Create item with URL containing query parameters
        url_with_params = "https://example.com/page?foo=1&bar=2"
        item = LibraryItem(
            id="url-item",
            content_type=ContentType.WEB_CONTENT,
            title="Web Page",
            summary="Summary",
            full_content="Content",
            source_url=url_with_params,
            cover_image_url=None,
            metadata={},
            themes=[],
            created_at=datetime.now(),
            highlights=[],
        )

        generator._ensure_output_dirs()
        generator.generate_item_pages(items=[item], themes={})

        content = (output_dir / "item" / "url-item" / "index.html").read_text()

        # URL should be present and properly escaped in href
        # The & in URLs is typically escaped as &amp; in HTML attributes
        assert "https://example.com/page" in content
        # Either the raw URL or properly escaped version should work
        assert "foo=1" in content and "bar=2" in content

    def test_newlines_in_content_handled(self, temp_dir):
        """Newlines in content should not break HTML structure."""
        from src.library.static_generator import StaticGenerator
        from src.library.models import LibraryItem, ContentType

        output_dir = temp_dir / "library-site"
        generator = StaticGenerator(output_dir=output_dir)

        # Create item with newlines
        content_with_newlines = "Line 1\nLine 2\nLine 3"
        item = LibraryItem(
            id="newline-item",
            content_type=ContentType.ARTICLE,
            title="Multi-line Title\nWith Newline",
            summary=content_with_newlines,
            full_content=content_with_newlines,
            source_url=None,
            cover_image_url=None,
            metadata={},
            themes=[],
            created_at=datetime.now(),
            highlights=[],
        )

        generator._ensure_output_dirs()
        generator.generate_all_page(items=[item])

        content = (output_dir / "all" / "index.html").read_text()

        # Content should still be present (newlines are allowed in HTML)
        assert "Line 1" in content
        assert "Line 2" in content

    def test_script_injection_prevented_in_card_title(self, temp_dir):
        """Script injection via card titles should be prevented."""
        from src.library.static_generator import StaticGenerator
        from src.library.models import LibraryItem, ContentType

        output_dir = temp_dir / "library-site"
        generator = StaticGenerator(output_dir=output_dir)

        # Various XSS attack vectors
        xss_payloads = [
            "<script>alert(1)</script>",
            '<img src="x" onerror="alert(1)">',
            "javascript:alert(1)",
            '<svg onload="alert(1)">',
            "<<script>script>alert(1)<</script>/script>",
        ]

        items = []
        for i, payload in enumerate(xss_payloads):
            items.append(
                LibraryItem(
                    id=f"xss-{i}",
                    content_type=ContentType.ARTICLE,
                    title=payload,
                    summary="Safe summary",
                    full_content="Safe content",
                    source_url=None,
                    cover_image_url=None,
                    metadata={},
                    themes=[],
                    created_at=datetime.now(),
                    highlights=[],
                )
            )

        generator._ensure_output_dirs()
        generator.generate_all_page(items=items)

        content = (output_dir / "all" / "index.html").read_text()

        # None of the dangerous strings should appear unescaped
        assert "<script>" not in content
        assert 'onerror="' not in content
        assert 'onload="' not in content
        # The escaped versions should be present
        assert "&lt;script&gt;" in content

    def test_metadata_values_escaped(self, temp_dir):
        """Metadata values displayed in templates should be escaped."""
        from src.library.static_generator import StaticGenerator
        from src.library.models import LibraryItem, ContentType

        output_dir = temp_dir / "library-site"
        generator = StaticGenerator(output_dir=output_dir)

        # Create item with special characters in metadata
        item = LibraryItem(
            id="metadata-item",
            content_type=ContentType.BOOK,
            title="Test Book",
            summary="Summary",
            full_content="Content",
            source_url=None,
            cover_image_url=None,
            metadata={
                "author": "Author <script>alert('XSS')</script>",
                "isbn": "123-456-789",
            },
            themes=[],
            created_at=datetime.now(),
            highlights=[],
        )

        generator._ensure_output_dirs()
        generator.generate_item_pages(items=[item], themes={})

        content = (output_dir / "item" / "metadata-item" / "index.html").read_text()

        # Author value should be escaped
        assert "<script>" not in content
        assert "&lt;script&gt;" in content


class TestItemsWithoutSourceUrl:
    """Tests for items without source_url (Task 179).

    When an item has no source_url, the library should:
    - Not display the "View Source" button on item detail pages
    - Still render all other item information correctly
    - Handle None and empty string source_url values
    """

    def test_item_page_without_source_url_no_button(self, temp_dir):
        """Item detail page should not show View Source button when source_url is None."""
        from src.library.static_generator import StaticGenerator
        from src.library.models import LibraryItem, ContentType

        output_dir = temp_dir / "library-site"
        generator = StaticGenerator(output_dir=output_dir)

        # Create item without source_url
        item = LibraryItem(
            id="no-source-item",
            content_type=ContentType.BOOK,
            title="Book Without Source",
            summary="A book with no source URL",
            full_content="Full content of the book",
            source_url=None,
            cover_image_url=None,
            metadata={"author": "Test Author"},
            themes=[],
            created_at=datetime.now(),
            highlights=["A highlight from the book"],
        )

        generator._ensure_output_dirs()
        generator.generate_item_pages(items=[item], themes={})

        content = (output_dir / "item" / "no-source-item" / "index.html").read_text()

        # View Source button should NOT be present
        assert "View Source" not in content
        # The button link structure should not exist
        assert 'class="btn btn-primary"' not in content or "View Source" not in content

    def test_item_page_with_source_url_shows_button(self, temp_dir):
        """Item detail page should show View Source button when source_url exists."""
        from src.library.static_generator import StaticGenerator
        from src.library.models import LibraryItem, ContentType

        output_dir = temp_dir / "library-site"
        generator = StaticGenerator(output_dir=output_dir)

        # Create item with source_url
        item = LibraryItem(
            id="has-source-item",
            content_type=ContentType.WEB_CONTENT,
            title="Web Page With Source",
            summary="A web page with source URL",
            full_content="Full content",
            source_url="https://example.com/article",
            cover_image_url=None,
            metadata={},
            themes=[],
            created_at=datetime.now(),
            highlights=[],
        )

        generator._ensure_output_dirs()
        generator.generate_item_pages(items=[item], themes={})

        content = (output_dir / "item" / "has-source-item" / "index.html").read_text()

        # View Source button SHOULD be present
        assert "View Source" in content
        # Link should contain the URL
        assert "https://example.com/article" in content

    def test_item_page_without_source_url_still_renders_other_content(self, temp_dir):
        """Item without source_url should still render title, summary, highlights, etc."""
        from src.library.static_generator import StaticGenerator
        from src.library.models import LibraryItem, ContentType

        output_dir = temp_dir / "library-site"
        generator = StaticGenerator(output_dir=output_dir)

        item = LibraryItem(
            id="full-content-no-source",
            content_type=ContentType.FRAMEWORK,
            title="Test Framework",
            summary="A framework for testing",
            full_content="Detailed explanation of the framework",
            source_url=None,
            cover_image_url=None,
            metadata={"author": "Framework Author"},
            themes=["test-theme"],
            created_at=datetime.now(),
            highlights=["Key insight from the framework"],
        )

        generator._ensure_output_dirs()
        generator.generate_item_pages(items=[item], themes={"test-theme": "Test Theme"})

        content = (output_dir / "item" / "full-content-no-source" / "index.html").read_text()

        # All other content should be present
        assert "Test Framework" in content
        assert "A framework for testing" in content
        assert "Key insight from the framework" in content
        assert "Framework Author" in content

    def test_item_card_renders_without_source_url(self, temp_dir):
        """Item cards on all page should render correctly without source_url."""
        from src.library.static_generator import StaticGenerator
        from src.library.models import LibraryItem, ContentType

        output_dir = temp_dir / "library-site"
        generator = StaticGenerator(output_dir=output_dir)

        item = LibraryItem(
            id="card-no-source",
            content_type=ContentType.VALUE,
            title="Core Value",
            summary="An important value",
            full_content="Full description",
            source_url=None,
            cover_image_url=None,
            metadata={},
            themes=[],
            created_at=datetime.now(),
            highlights=[],
        )

        generator._ensure_output_dirs()
        generator.generate_all_page(items=[item])

        content = (output_dir / "all" / "index.html").read_text()

        # Card should render correctly
        assert "Core Value" in content
        assert "An important value" in content
        # Link to detail page should exist
        assert "/item/card-no-source/" in content

    def test_empty_string_source_url_treated_as_no_source(self, temp_dir):
        """Empty string source_url should be treated the same as None."""
        from src.library.static_generator import StaticGenerator
        from src.library.models import LibraryItem, ContentType

        output_dir = temp_dir / "library-site"
        generator = StaticGenerator(output_dir=output_dir)

        # Create item with empty string source_url
        item = LibraryItem(
            id="empty-source-item",
            content_type=ContentType.ARTICLE,
            title="Article With Empty Source",
            summary="An article",
            full_content="Content",
            source_url="",  # Empty string
            cover_image_url=None,
            metadata={},
            themes=[],
            created_at=datetime.now(),
            highlights=[],
        )

        generator._ensure_output_dirs()
        generator.generate_item_pages(items=[item], themes={})

        content = (output_dir / "item" / "empty-source-item" / "index.html").read_text()

        # View Source button should NOT be present for empty string
        assert "View Source" not in content

    def test_multiple_items_mixed_source_url_presence(self, temp_dir):
        """Page with multiple items should correctly show/hide source buttons."""
        from src.library.static_generator import StaticGenerator
        from src.library.models import LibraryItem, ContentType

        output_dir = temp_dir / "library-site"
        generator = StaticGenerator(output_dir=output_dir)

        items = [
            LibraryItem(
                id="item-with-source",
                content_type=ContentType.WEB_CONTENT,
                title="With Source",
                summary="Has source",
                full_content="Content",
                source_url="https://example.com",
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime.now(),
                highlights=[],
            ),
            LibraryItem(
                id="item-without-source",
                content_type=ContentType.VALUE,
                title="Without Source",
                summary="No source",
                full_content="Content",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime.now(),
                highlights=[],
            ),
        ]

        generator._ensure_output_dirs()
        generator.generate_item_pages(items=items, themes={})

        # Check item with source
        content_with = (output_dir / "item" / "item-with-source" / "index.html").read_text()
        assert "View Source" in content_with

        # Check item without source
        content_without = (output_dir / "item" / "item-without-source" / "index.html").read_text()
        assert "View Source" not in content_without


class TestItemsWithoutHighlights:
    """Tests for items without highlights (Task 181).

    When an item has no highlights, the library should:
    - Not display the "Key Highlights" section on item detail pages
    - Still render all other item information correctly
    - Handle empty list and None highlights values
    """

    def test_item_page_without_highlights_no_section(self, temp_dir):
        """Item detail page should not show highlights section when highlights is empty."""
        from src.library.static_generator import StaticGenerator
        from src.library.models import LibraryItem, ContentType

        output_dir = temp_dir / "library-site"
        generator = StaticGenerator(output_dir=output_dir)

        # Create item without highlights (empty list)
        item = LibraryItem(
            id="no-highlights-item",
            content_type=ContentType.BOOK,
            title="Book Without Highlights",
            summary="A book with no highlights",
            full_content="Full content of the book",
            source_url="https://example.com/book",
            cover_image_url=None,
            metadata={"author": "Test Author"},
            themes=[],
            created_at=datetime.now(),
            highlights=[],  # Empty list
        )

        generator._ensure_output_dirs()
        generator.generate_item_pages(items=[item], themes={})

        content = (output_dir / "item" / "no-highlights-item" / "index.html").read_text()

        # Highlights section should NOT be present
        assert "Key Highlights" not in content
        assert "highlights-section" not in content

    def test_item_page_with_highlights_shows_section(self, temp_dir):
        """Item detail page should show highlights section when highlights exist."""
        from src.library.static_generator import StaticGenerator
        from src.library.models import LibraryItem, ContentType

        output_dir = temp_dir / "library-site"
        generator = StaticGenerator(output_dir=output_dir)

        # Create item with highlights
        item = LibraryItem(
            id="has-highlights-item",
            content_type=ContentType.BOOK,
            title="Book With Highlights",
            summary="A book with highlights",
            full_content="Full content",
            source_url=None,
            cover_image_url=None,
            metadata={},
            themes=[],
            created_at=datetime.now(),
            highlights=["First important insight", "Second key takeaway"],
        )

        generator._ensure_output_dirs()
        generator.generate_item_pages(items=[item], themes={})

        content = (output_dir / "item" / "has-highlights-item" / "index.html").read_text()

        # Highlights section SHOULD be present
        assert "Key Highlights" in content
        assert "highlights-section" in content
        # Highlight text should appear
        assert "First important insight" in content
        assert "Second key takeaway" in content

    def test_item_page_without_highlights_still_renders_other_content(self, temp_dir):
        """Item without highlights should still render title, summary, source, etc."""
        from src.library.static_generator import StaticGenerator
        from src.library.models import LibraryItem, ContentType

        output_dir = temp_dir / "library-site"
        generator = StaticGenerator(output_dir=output_dir)

        item = LibraryItem(
            id="full-content-no-highlights",
            content_type=ContentType.FRAMEWORK,
            title="Test Framework",
            summary="A framework for testing",
            full_content="Detailed explanation of the framework",
            source_url="https://example.com/framework",
            cover_image_url=None,
            metadata={"author": "Framework Author"},
            themes=["test-theme"],
            created_at=datetime.now(),
            highlights=[],  # No highlights
        )

        generator._ensure_output_dirs()
        generator.generate_item_pages(items=[item], themes={"test-theme": "Test Theme"})

        content = (output_dir / "item" / "full-content-no-highlights" / "index.html").read_text()

        # All other content should be present
        assert "Test Framework" in content
        assert "A framework for testing" in content
        assert "View Source" in content
        assert "Framework Author" in content

    def test_single_highlight_shows_section(self, temp_dir):
        """Item with a single highlight should still show highlights section."""
        from src.library.static_generator import StaticGenerator
        from src.library.models import LibraryItem, ContentType

        output_dir = temp_dir / "library-site"
        generator = StaticGenerator(output_dir=output_dir)

        item = LibraryItem(
            id="single-highlight-item",
            content_type=ContentType.INSIGHT,
            title="Single Insight",
            summary="An insight with one highlight",
            full_content="Content",
            source_url=None,
            cover_image_url=None,
            metadata={},
            themes=[],
            created_at=datetime.now(),
            highlights=["The only important takeaway"],
        )

        generator._ensure_output_dirs()
        generator.generate_item_pages(items=[item], themes={})

        content = (output_dir / "item" / "single-highlight-item" / "index.html").read_text()

        # Highlights section should be present with single highlight
        assert "Key Highlights" in content
        assert "The only important takeaway" in content

    def test_multiple_items_mixed_highlights_presence(self, temp_dir):
        """Page with multiple items should correctly show/hide highlights sections."""
        from src.library.static_generator import StaticGenerator
        from src.library.models import LibraryItem, ContentType

        output_dir = temp_dir / "library-site"
        generator = StaticGenerator(output_dir=output_dir)

        items = [
            LibraryItem(
                id="item-with-highlights",
                content_type=ContentType.BOOK,
                title="With Highlights",
                summary="Has highlights",
                full_content="Content",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime.now(),
                highlights=["Important point 1", "Important point 2"],
            ),
            LibraryItem(
                id="item-without-highlights",
                content_type=ContentType.VALUE,
                title="Without Highlights",
                summary="No highlights",
                full_content="Content",
                source_url=None,
                cover_image_url=None,
                metadata={},
                themes=[],
                created_at=datetime.now(),
                highlights=[],
            ),
        ]

        generator._ensure_output_dirs()
        generator.generate_item_pages(items=items, themes={})

        # Check item with highlights
        content_with = (output_dir / "item" / "item-with-highlights" / "index.html").read_text()
        assert "Key Highlights" in content_with
        assert "Important point 1" in content_with

        # Check item without highlights
        content_without = (output_dir / "item" / "item-without-highlights" / "index.html").read_text()
        assert "Key Highlights" not in content_without
        assert "highlights-section" not in content_without

    def test_highlights_with_special_characters_escaped(self, temp_dir):
        """Highlights with special characters should be properly escaped."""
        from src.library.static_generator import StaticGenerator
        from src.library.models import LibraryItem, ContentType

        output_dir = temp_dir / "library-site"
        generator = StaticGenerator(output_dir=output_dir)

        item = LibraryItem(
            id="special-char-highlights",
            content_type=ContentType.BOOK,
            title="Book with Special Chars",
            summary="Summary",
            full_content="Content",
            source_url=None,
            cover_image_url=None,
            metadata={},
            themes=[],
            created_at=datetime.now(),
            highlights=[
                "Highlight with <script>alert('XSS')</script> attack",
                "Highlight with & ampersand",
                "Highlight with 'quotes' and \"double quotes\"",
            ],
        )

        generator._ensure_output_dirs()
        generator.generate_item_pages(items=[item], themes={})

        content = (output_dir / "item" / "special-char-highlights" / "index.html").read_text()

        # Highlights should be present
        assert "Key Highlights" in content
        # HTML should be escaped
        assert "<script>" not in content
        assert "&lt;script&gt;" in content
        # Ampersand should be escaped
        assert "&amp;" in content

    def test_highlight_styling_applied(self, temp_dir):
        """Highlights should have proper CSS styling classes."""
        from src.library.static_generator import StaticGenerator
        from src.library.models import LibraryItem, ContentType

        output_dir = temp_dir / "library-site"
        generator = StaticGenerator(output_dir=output_dir)

        item = LibraryItem(
            id="styled-highlights-item",
            content_type=ContentType.BOOK,
            title="Book",
            summary="Summary",
            full_content="Content",
            source_url=None,
            cover_image_url=None,
            metadata={},
            themes=[],
            created_at=datetime.now(),
            highlights=["A styled highlight"],
        )

        generator._ensure_output_dirs()
        generator.generate_item_pages(items=[item], themes={})

        content = (output_dir / "item" / "styled-highlights-item" / "index.html").read_text()

        # Should have highlight styling class
        assert "highlight" in content
        # Should be in a blockquote or similar styled element
        assert "blockquote" in content.lower() or "highlight" in content

    def test_many_highlights_all_rendered(self, temp_dir):
        """All highlights should be rendered even if there are many."""
        from src.library.static_generator import StaticGenerator
        from src.library.models import LibraryItem, ContentType

        output_dir = temp_dir / "library-site"
        generator = StaticGenerator(output_dir=output_dir)

        many_highlights = [f"Highlight number {i}" for i in range(10)]
        item = LibraryItem(
            id="many-highlights-item",
            content_type=ContentType.BOOK,
            title="Book with Many Highlights",
            summary="Summary",
            full_content="Content",
            source_url=None,
            cover_image_url=None,
            metadata={},
            themes=[],
            created_at=datetime.now(),
            highlights=many_highlights,
        )

        generator._ensure_output_dirs()
        generator.generate_item_pages(items=[item], themes={})

        content = (output_dir / "item" / "many-highlights-item" / "index.html").read_text()

        # All 10 highlights should be present
        for i in range(10):
            assert f"Highlight number {i}" in content

    def test_highlight_with_newlines_handled(self, temp_dir):
        """Highlights with newlines should not break HTML structure."""
        from src.library.static_generator import StaticGenerator
        from src.library.models import LibraryItem, ContentType

        output_dir = temp_dir / "library-site"
        generator = StaticGenerator(output_dir=output_dir)

        item = LibraryItem(
            id="newline-highlight-item",
            content_type=ContentType.BOOK,
            title="Book",
            summary="Summary",
            full_content="Content",
            source_url=None,
            cover_image_url=None,
            metadata={},
            themes=[],
            created_at=datetime.now(),
            highlights=[
                "First line\nSecond line\nThird line",
            ],
        )

        generator._ensure_output_dirs()
        generator.generate_item_pages(items=[item], themes={})

        content = (output_dir / "item" / "newline-highlight-item" / "index.html").read_text()

        # Content should be present
        assert "First line" in content
        # Page should still have valid structure
        assert "Key Highlights" in content
