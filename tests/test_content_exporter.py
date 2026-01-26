"""Tests for content exporter module."""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch

from src.library.models import ContentType


class TestContentExporterInitialization:
    """Tests for ContentExporter initialization with KnowledgeBase."""

    def test_content_exporter_accepts_knowledge_base(self, mock_knowledge_base):
        """ContentExporter should accept a KnowledgeBase instance."""
        from src.library.content_exporter import ContentExporter

        exporter = ContentExporter(mock_knowledge_base)

        assert exporter.knowledge_base == mock_knowledge_base

    def test_content_exporter_stores_knowledge_base_reference(self, mock_knowledge_base):
        """ContentExporter should store a reference to the KnowledgeBase."""
        from src.library.content_exporter import ContentExporter

        exporter = ContentExporter(mock_knowledge_base)

        assert exporter.knowledge_base is mock_knowledge_base

    def test_content_exporter_can_be_instantiated(self, mock_knowledge_base):
        """ContentExporter should be instantiable with a KnowledgeBase."""
        from src.library.content_exporter import ContentExporter

        exporter = ContentExporter(mock_knowledge_base)

        assert exporter is not None


class TestInferContentType:
    """Tests for _infer_content_type() method mapping ChromaDB types to ContentType."""

    def test_infer_content_type_value_maps_to_value(self, mock_knowledge_base):
        """ChromaDB 'value' type should map to ContentType.VALUE."""
        from src.library.content_exporter import ContentExporter

        exporter = ContentExporter(mock_knowledge_base)

        result = exporter._infer_content_type("value")

        assert result == ContentType.VALUE

    def test_infer_content_type_framework_maps_to_framework(self, mock_knowledge_base):
        """ChromaDB 'framework' type should map to ContentType.FRAMEWORK."""
        from src.library.content_exporter import ContentExporter

        exporter = ContentExporter(mock_knowledge_base)

        result = exporter._infer_content_type("framework")

        assert result == ContentType.FRAMEWORK

    def test_infer_content_type_web_maps_to_web_content(self, mock_knowledge_base):
        """ChromaDB 'web' type should map to ContentType.WEB_CONTENT."""
        from src.library.content_exporter import ContentExporter

        exporter = ContentExporter(mock_knowledge_base)

        result = exporter._infer_content_type("web")

        assert result == ContentType.WEB_CONTENT

    def test_infer_content_type_preference_maps_to_preference(self, mock_knowledge_base):
        """ChromaDB 'preference' type should map to ContentType.PREFERENCE."""
        from src.library.content_exporter import ContentExporter

        exporter = ContentExporter(mock_knowledge_base)

        result = exporter._infer_content_type("preference")

        assert result == ContentType.PREFERENCE

    def test_infer_content_type_memory_maps_to_memory(self, mock_knowledge_base):
        """ChromaDB 'memory' type should map to ContentType.MEMORY."""
        from src.library.content_exporter import ContentExporter

        exporter = ContentExporter(mock_knowledge_base)

        result = exporter._infer_content_type("memory")

        assert result == ContentType.MEMORY

    def test_infer_content_type_book_maps_to_book(self, mock_knowledge_base):
        """ChromaDB 'book' type should map to ContentType.BOOK."""
        from src.library.content_exporter import ContentExporter

        exporter = ContentExporter(mock_knowledge_base)

        result = exporter._infer_content_type("book")

        assert result == ContentType.BOOK

    def test_infer_content_type_article_maps_to_article(self, mock_knowledge_base):
        """ChromaDB 'article' type should map to ContentType.ARTICLE."""
        from src.library.content_exporter import ContentExporter

        exporter = ContentExporter(mock_knowledge_base)

        result = exporter._infer_content_type("article")

        assert result == ContentType.ARTICLE

    def test_infer_content_type_insight_maps_to_insight(self, mock_knowledge_base):
        """ChromaDB 'insight' type should map to ContentType.INSIGHT."""
        from src.library.content_exporter import ContentExporter

        exporter = ContentExporter(mock_knowledge_base)

        result = exporter._infer_content_type("insight")

        assert result == ContentType.INSIGHT

    def test_infer_content_type_goal_maps_to_goal(self, mock_knowledge_base):
        """ChromaDB 'goal' type should map to ContentType.GOAL."""
        from src.library.content_exporter import ContentExporter

        exporter = ContentExporter(mock_knowledge_base)

        result = exporter._infer_content_type("goal")

        assert result == ContentType.GOAL

    def test_infer_content_type_skill_maps_to_skill(self, mock_knowledge_base):
        """ChromaDB 'skill' type should map to ContentType.SKILL."""
        from src.library.content_exporter import ContentExporter

        exporter = ContentExporter(mock_knowledge_base)

        result = exporter._infer_content_type("skill")

        assert result == ContentType.SKILL

    def test_infer_content_type_unknown_defaults_to_web_content(self, mock_knowledge_base):
        """Unknown ChromaDB types should default to ContentType.WEB_CONTENT."""
        from src.library.content_exporter import ContentExporter

        exporter = ContentExporter(mock_knowledge_base)

        result = exporter._infer_content_type("unknown_type")

        assert result == ContentType.WEB_CONTENT

    def test_infer_content_type_handles_case_insensitive(self, mock_knowledge_base):
        """Content type inference should handle case variations."""
        from src.library.content_exporter import ContentExporter

        exporter = ContentExporter(mock_knowledge_base)

        # Should work with uppercase
        assert exporter._infer_content_type("VALUE") == ContentType.VALUE
        # Should work with mixed case
        assert exporter._infer_content_type("Framework") == ContentType.FRAMEWORK

    def test_infer_content_type_handles_none_input(self, mock_knowledge_base):
        """Content type inference should handle None input gracefully."""
        from src.library.content_exporter import ContentExporter

        exporter = ContentExporter(mock_knowledge_base)

        result = exporter._infer_content_type(None)

        assert result == ContentType.WEB_CONTENT

    def test_infer_content_type_handles_empty_string(self, mock_knowledge_base):
        """Content type inference should handle empty string input."""
        from src.library.content_exporter import ContentExporter

        exporter = ContentExporter(mock_knowledge_base)

        result = exporter._infer_content_type("")

        assert result == ContentType.WEB_CONTENT


class TestGenerateTitle:
    """Tests for _generate_title() method."""

    def test_generate_title_uses_name_from_metadata(self, mock_knowledge_base):
        """When metadata has 'name' field, use it as title."""
        from src.library.content_exporter import ContentExporter

        exporter = ContentExporter(mock_knowledge_base)
        metadata = {"name": "My Custom Title", "type": "value"}
        content = "This is the content that should not be used."

        result = exporter._generate_title(metadata, content)

        assert result == "My Custom Title"

    def test_generate_title_uses_content_when_no_name(self, mock_knowledge_base):
        """When metadata lacks 'name', generate title from content."""
        from src.library.content_exporter import ContentExporter

        exporter = ContentExporter(mock_knowledge_base)
        metadata = {"type": "value"}
        content = "This is a short piece of content."

        result = exporter._generate_title(metadata, content)

        assert result == "This is a short piece of content."

    def test_generate_title_truncates_long_content_at_50_chars(self, mock_knowledge_base):
        """When generating from content, truncate at 50 characters."""
        from src.library.content_exporter import ContentExporter

        exporter = ContentExporter(mock_knowledge_base)
        metadata = {}
        content = "This is a much longer piece of content that definitely exceeds fifty characters in length and should be truncated."

        result = exporter._generate_title(metadata, content)

        assert len(result) <= 53  # 50 chars + "..."
        assert result.endswith("...")
        assert result == "This is a much longer piece of content that defini..."

    def test_generate_title_no_ellipsis_for_short_content(self, mock_knowledge_base):
        """Content under 50 chars should not have ellipsis."""
        from src.library.content_exporter import ContentExporter

        exporter = ContentExporter(mock_knowledge_base)
        metadata = {}
        content = "Short content"

        result = exporter._generate_title(metadata, content)

        assert result == "Short content"
        assert not result.endswith("...")

    def test_generate_title_exactly_50_chars_no_ellipsis(self, mock_knowledge_base):
        """Content exactly 50 chars should not have ellipsis."""
        from src.library.content_exporter import ContentExporter

        exporter = ContentExporter(mock_knowledge_base)
        metadata = {}
        # Exactly 50 characters
        content = "12345678901234567890123456789012345678901234567890"

        result = exporter._generate_title(metadata, content)

        assert result == content
        assert len(result) == 50
        assert not result.endswith("...")

    def test_generate_title_handles_empty_name(self, mock_knowledge_base):
        """Empty 'name' field should fall back to content."""
        from src.library.content_exporter import ContentExporter

        exporter = ContentExporter(mock_knowledge_base)
        metadata = {"name": "", "type": "value"}
        content = "Content used as fallback"

        result = exporter._generate_title(metadata, content)

        assert result == "Content used as fallback"

    def test_generate_title_handles_none_name(self, mock_knowledge_base):
        """None 'name' field should fall back to content."""
        from src.library.content_exporter import ContentExporter

        exporter = ContentExporter(mock_knowledge_base)
        metadata = {"name": None, "type": "value"}
        content = "Content used as fallback"

        result = exporter._generate_title(metadata, content)

        assert result == "Content used as fallback"

    def test_generate_title_handles_empty_content(self, mock_knowledge_base):
        """Empty content should return 'Untitled' placeholder."""
        from src.library.content_exporter import ContentExporter

        exporter = ContentExporter(mock_knowledge_base)
        metadata = {}
        content = ""

        result = exporter._generate_title(metadata, content)

        assert result == "Untitled"

    def test_generate_title_handles_none_content(self, mock_knowledge_base):
        """None content should return 'Untitled' placeholder."""
        from src.library.content_exporter import ContentExporter

        exporter = ContentExporter(mock_knowledge_base)
        metadata = {}
        content = None

        result = exporter._generate_title(metadata, content)

        assert result == "Untitled"

    def test_generate_title_handles_whitespace_only_content(self, mock_knowledge_base):
        """Whitespace-only content should return 'Untitled' placeholder."""
        from src.library.content_exporter import ContentExporter

        exporter = ContentExporter(mock_knowledge_base)
        metadata = {}
        content = "   \n\t  "

        result = exporter._generate_title(metadata, content)

        assert result == "Untitled"

    def test_generate_title_strips_leading_whitespace(self, mock_knowledge_base):
        """Content with leading whitespace should be stripped."""
        from src.library.content_exporter import ContentExporter

        exporter = ContentExporter(mock_knowledge_base)
        metadata = {}
        content = "   Leading whitespace content"

        result = exporter._generate_title(metadata, content)

        assert result == "Leading whitespace content"

    def test_generate_title_handles_none_metadata(self, mock_knowledge_base):
        """None metadata should fall back to content."""
        from src.library.content_exporter import ContentExporter

        exporter = ContentExporter(mock_knowledge_base)
        metadata = None
        content = "Content used as title"

        result = exporter._generate_title(metadata, content)

        assert result == "Content used as title"

    def test_generate_title_prefers_name_over_content(self, mock_knowledge_base):
        """Name in metadata should always be preferred over content."""
        from src.library.content_exporter import ContentExporter

        exporter = ContentExporter(mock_knowledge_base)
        metadata = {"name": "Preferred Name"}
        content = "This is the content"

        result = exporter._generate_title(metadata, content)

        assert result == "Preferred Name"

    def test_generate_title_with_newlines_in_content(self, mock_knowledge_base):
        """Newlines in content should be handled (use first line or strip)."""
        from src.library.content_exporter import ContentExporter

        exporter = ContentExporter(mock_knowledge_base)
        metadata = {}
        content = "First line of content\nSecond line\nThird line"

        result = exporter._generate_title(metadata, content)

        # Should take first 50 chars, which includes the first line
        assert "First line of content" in result


class TestGenerateSummary:
    """Tests for _generate_summary() method truncation at 200 characters."""

    def test_generate_summary_truncates_at_200_chars(self, mock_knowledge_base):
        """Summary should be truncated at 200 characters with ellipsis."""
        from src.library.content_exporter import ContentExporter

        exporter = ContentExporter(mock_knowledge_base)
        # Create content longer than 200 characters
        content = "A" * 250

        result = exporter._generate_summary(content)

        assert len(result) <= 203  # 200 chars + "..."
        assert result.endswith("...")

    def test_generate_summary_preserves_short_content(self, mock_knowledge_base):
        """Content under 200 chars should not be truncated."""
        from src.library.content_exporter import ContentExporter

        exporter = ContentExporter(mock_knowledge_base)
        content = "This is a short summary."

        result = exporter._generate_summary(content)

        assert result == content
        assert not result.endswith("...")

    def test_generate_summary_exactly_200_chars_no_ellipsis(self, mock_knowledge_base):
        """Content exactly 200 chars should not have ellipsis."""
        from src.library.content_exporter import ContentExporter

        exporter = ContentExporter(mock_knowledge_base)
        # Exactly 200 characters
        content = "A" * 200

        result = exporter._generate_summary(content)

        assert result == content
        assert len(result) == 200
        assert not result.endswith("...")

    def test_generate_summary_201_chars_gets_ellipsis(self, mock_knowledge_base):
        """Content at 201 chars should be truncated with ellipsis."""
        from src.library.content_exporter import ContentExporter

        exporter = ContentExporter(mock_knowledge_base)
        # 201 characters
        content = "A" * 201

        result = exporter._generate_summary(content)

        assert len(result) == 203  # 200 chars + "..."
        assert result == "A" * 200 + "..."

    def test_generate_summary_handles_empty_content(self, mock_knowledge_base):
        """Empty content should return empty string."""
        from src.library.content_exporter import ContentExporter

        exporter = ContentExporter(mock_knowledge_base)
        content = ""

        result = exporter._generate_summary(content)

        assert result == ""

    def test_generate_summary_handles_none_content(self, mock_knowledge_base):
        """None content should return empty string."""
        from src.library.content_exporter import ContentExporter

        exporter = ContentExporter(mock_knowledge_base)

        result = exporter._generate_summary(None)

        assert result == ""

    def test_generate_summary_strips_whitespace(self, mock_knowledge_base):
        """Leading and trailing whitespace should be stripped before truncation."""
        from src.library.content_exporter import ContentExporter

        exporter = ContentExporter(mock_knowledge_base)
        content = "   Content with whitespace   "

        result = exporter._generate_summary(content)

        assert result == "Content with whitespace"

    def test_generate_summary_whitespace_only_returns_empty(self, mock_knowledge_base):
        """Whitespace-only content should return empty string."""
        from src.library.content_exporter import ContentExporter

        exporter = ContentExporter(mock_knowledge_base)
        content = "   \n\t  "

        result = exporter._generate_summary(content)

        assert result == ""

    def test_generate_summary_preserves_internal_whitespace(self, mock_knowledge_base):
        """Internal whitespace should be preserved."""
        from src.library.content_exporter import ContentExporter

        exporter = ContentExporter(mock_knowledge_base)
        content = "Word one  Word two\tWord three"

        result = exporter._generate_summary(content)

        assert result == content

    def test_generate_summary_handles_newlines(self, mock_knowledge_base):
        """Newlines in content should be preserved (for display handling)."""
        from src.library.content_exporter import ContentExporter

        exporter = ContentExporter(mock_knowledge_base)
        content = "First line\nSecond line\nThird line"

        result = exporter._generate_summary(content)

        assert result == content

    def test_generate_summary_truncates_at_word_boundary_optionally(
        self, mock_knowledge_base
    ):
        """Truncation at 200 chars is character-based, not word-based."""
        from src.library.content_exporter import ContentExporter

        exporter = ContentExporter(mock_knowledge_base)
        # Create content that is 205 chars with a word spanning the 200 boundary
        words = "word " * 50  # "word " is 5 chars, 50 times = 250 chars
        content = words[:205]

        result = exporter._generate_summary(content)

        # Should truncate at exactly 200 chars + ellipsis
        assert len(result) == 203
        assert result.endswith("...")

    def test_generate_summary_long_realistic_content(self, mock_knowledge_base):
        """Test with realistic content that exceeds 200 characters."""
        from src.library.content_exporter import ContentExporter

        exporter = ContentExporter(mock_knowledge_base)
        content = (
            "This is a longer piece of content that simulates a real knowledge base "
            "entry. It contains multiple sentences and spans well over two hundred "
            "characters to test the truncation behavior properly. We want to ensure "
            "it gets truncated at exactly 200 characters."
        )

        result = exporter._generate_summary(content)

        assert len(result) == 203  # 200 + "..."
        assert result.endswith("...")
        assert result[:200] == content[:200]


class TestExtractHighlights:
    """Tests for _extract_highlights() extracting key_points from values and steps from frameworks."""

    def test_extract_highlights_returns_key_points_for_values(self, mock_knowledge_base):
        """Values with key_points metadata should return those as highlights."""
        from src.library.content_exporter import ContentExporter
        from src.library.models import ContentType

        exporter = ContentExporter(mock_knowledge_base)
        metadata = {
            "type": "value",
            "key_points": [
                "Appreciate the care others put into their work",
                "Be curious and ask questions without assumption",
                "Believe in magic and keep dreaming",
            ],
        }

        result = exporter._extract_highlights(metadata, ContentType.VALUE)

        assert result == [
            "Appreciate the care others put into their work",
            "Be curious and ask questions without assumption",
            "Believe in magic and keep dreaming",
        ]

    def test_extract_highlights_returns_steps_for_frameworks(self, mock_knowledge_base):
        """Frameworks with steps metadata should return those as highlights."""
        from src.library.content_exporter import ContentExporter
        from src.library.models import ContentType

        exporter = ContentExporter(mock_knowledge_base)
        metadata = {
            "type": "framework",
            "steps": [
                "Define the problem clearly",
                "Gather relevant information",
                "Generate potential solutions",
            ],
        }

        result = exporter._extract_highlights(metadata, ContentType.FRAMEWORK)

        assert result == [
            "Define the problem clearly",
            "Gather relevant information",
            "Generate potential solutions",
        ]

    def test_extract_highlights_returns_empty_for_value_without_key_points(
        self, mock_knowledge_base
    ):
        """Values without key_points should return empty list."""
        from src.library.content_exporter import ContentExporter
        from src.library.models import ContentType

        exporter = ContentExporter(mock_knowledge_base)
        metadata = {"type": "value", "name": "Some Value"}

        result = exporter._extract_highlights(metadata, ContentType.VALUE)

        assert result == []

    def test_extract_highlights_returns_empty_for_framework_without_steps(
        self, mock_knowledge_base
    ):
        """Frameworks without steps should return empty list."""
        from src.library.content_exporter import ContentExporter
        from src.library.models import ContentType

        exporter = ContentExporter(mock_knowledge_base)
        metadata = {"type": "framework", "name": "Some Framework"}

        result = exporter._extract_highlights(metadata, ContentType.FRAMEWORK)

        assert result == []

    def test_extract_highlights_returns_empty_for_other_content_types(
        self, mock_knowledge_base
    ):
        """Other content types should return empty list (no highlight extraction)."""
        from src.library.content_exporter import ContentExporter
        from src.library.models import ContentType

        exporter = ContentExporter(mock_knowledge_base)
        metadata = {"type": "memory", "content": "Some memory content"}

        result = exporter._extract_highlights(metadata, ContentType.MEMORY)

        assert result == []

    def test_extract_highlights_handles_none_metadata(self, mock_knowledge_base):
        """None metadata should return empty list."""
        from src.library.content_exporter import ContentExporter
        from src.library.models import ContentType

        exporter = ContentExporter(mock_knowledge_base)

        result = exporter._extract_highlights(None, ContentType.VALUE)

        assert result == []

    def test_extract_highlights_handles_empty_metadata(self, mock_knowledge_base):
        """Empty metadata dict should return empty list."""
        from src.library.content_exporter import ContentExporter
        from src.library.models import ContentType

        exporter = ContentExporter(mock_knowledge_base)

        result = exporter._extract_highlights({}, ContentType.FRAMEWORK)

        assert result == []

    def test_extract_highlights_handles_non_list_key_points(self, mock_knowledge_base):
        """Non-list key_points should return empty list."""
        from src.library.content_exporter import ContentExporter
        from src.library.models import ContentType

        exporter = ContentExporter(mock_knowledge_base)
        metadata = {"type": "value", "key_points": "This is a string, not a list"}

        result = exporter._extract_highlights(metadata, ContentType.VALUE)

        assert result == []

    def test_extract_highlights_handles_non_list_steps(self, mock_knowledge_base):
        """Non-list steps should return empty list."""
        from src.library.content_exporter import ContentExporter
        from src.library.models import ContentType

        exporter = ContentExporter(mock_knowledge_base)
        metadata = {"type": "framework", "steps": "Single step as string"}

        result = exporter._extract_highlights(metadata, ContentType.FRAMEWORK)

        assert result == []

    def test_extract_highlights_handles_empty_key_points_list(self, mock_knowledge_base):
        """Empty key_points list should return empty list."""
        from src.library.content_exporter import ContentExporter
        from src.library.models import ContentType

        exporter = ContentExporter(mock_knowledge_base)
        metadata = {"type": "value", "key_points": []}

        result = exporter._extract_highlights(metadata, ContentType.VALUE)

        assert result == []

    def test_extract_highlights_handles_empty_steps_list(self, mock_knowledge_base):
        """Empty steps list should return empty list."""
        from src.library.content_exporter import ContentExporter
        from src.library.models import ContentType

        exporter = ContentExporter(mock_knowledge_base)
        metadata = {"type": "framework", "steps": []}

        result = exporter._extract_highlights(metadata, ContentType.FRAMEWORK)

        assert result == []

    def test_extract_highlights_for_insight_returns_key_points(self, mock_knowledge_base):
        """Insights with key_points should return them as highlights (same as values)."""
        from src.library.content_exporter import ContentExporter
        from src.library.models import ContentType

        exporter = ContentExporter(mock_knowledge_base)
        metadata = {
            "type": "insight",
            "key_points": ["First insight", "Second insight"],
        }

        result = exporter._extract_highlights(metadata, ContentType.INSIGHT)

        assert result == ["First insight", "Second insight"]

    def test_extract_highlights_for_book_returns_empty(self, mock_knowledge_base):
        """Books should return empty list (highlights come from elsewhere)."""
        from src.library.content_exporter import ContentExporter
        from src.library.models import ContentType

        exporter = ContentExporter(mock_knowledge_base)
        metadata = {"type": "book", "title": "Some Book"}

        result = exporter._extract_highlights(metadata, ContentType.BOOK)

        assert result == []

    def test_extract_highlights_for_web_content_returns_empty(self, mock_knowledge_base):
        """Web content should return empty list."""
        from src.library.content_exporter import ContentExporter
        from src.library.models import ContentType

        exporter = ContentExporter(mock_knowledge_base)
        metadata = {"type": "web", "source": "https://example.com"}

        result = exporter._extract_highlights(metadata, ContentType.WEB_CONTENT)

        assert result == []

    def test_extract_highlights_preserves_order_of_key_points(self, mock_knowledge_base):
        """Key points should be returned in the same order as in metadata."""
        from src.library.content_exporter import ContentExporter
        from src.library.models import ContentType

        exporter = ContentExporter(mock_knowledge_base)
        metadata = {
            "key_points": ["First", "Second", "Third", "Fourth"],
        }

        result = exporter._extract_highlights(metadata, ContentType.VALUE)

        assert result == ["First", "Second", "Third", "Fourth"]

    def test_extract_highlights_preserves_order_of_steps(self, mock_knowledge_base):
        """Steps should be returned in the same order as in metadata."""
        from src.library.content_exporter import ContentExporter
        from src.library.models import ContentType

        exporter = ContentExporter(mock_knowledge_base)
        metadata = {
            "steps": ["Step 1", "Step 2", "Step 3", "Step 4"],
        }

        result = exporter._extract_highlights(metadata, ContentType.FRAMEWORK)

        assert result == ["Step 1", "Step 2", "Step 3", "Step 4"]

    def test_extract_highlights_filters_non_string_items_in_key_points(
        self, mock_knowledge_base
    ):
        """Non-string items in key_points should be converted to strings."""
        from src.library.content_exporter import ContentExporter
        from src.library.models import ContentType

        exporter = ContentExporter(mock_knowledge_base)
        metadata = {
            "key_points": ["Valid string", 123, None, "Another string"],
        }

        result = exporter._extract_highlights(metadata, ContentType.VALUE)

        # Should convert non-strings to strings or filter them
        assert len(result) == 4
        assert "Valid string" in result
        assert "Another string" in result

    def test_extract_highlights_filters_non_string_items_in_steps(
        self, mock_knowledge_base
    ):
        """Non-string items in steps should be converted to strings."""
        from src.library.content_exporter import ContentExporter
        from src.library.models import ContentType

        exporter = ContentExporter(mock_knowledge_base)
        metadata = {
            "steps": ["Step one", 42, None, "Step two"],
        }

        result = exporter._extract_highlights(metadata, ContentType.FRAMEWORK)

        # Should convert non-strings to strings or filter them
        assert len(result) == 4
        assert "Step one" in result
        assert "Step two" in result
