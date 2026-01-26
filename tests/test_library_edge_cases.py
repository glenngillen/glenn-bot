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
