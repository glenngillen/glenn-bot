"""Tests for document ingestion module with async support."""

import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from src.document_ingestion import (
    DocumentIngestionTool,
    ProcessingStatus,
    URLProcessingResult,
    BatchProgress
)


class TestBatchProgress:
    """Tests for BatchProgress dataclass."""

    def test_initial_state(self):
        """Test BatchProgress initial state."""
        progress = BatchProgress(total=5)
        assert progress.total == 5
        assert progress.completed == 0
        assert progress.failed == 0
        assert progress.pending == 5
        assert progress.percentage == 0.0
        assert progress.current_url is None
        assert len(progress.results) == 0

    def test_pending_calculation(self):
        """Test pending count calculation."""
        progress = BatchProgress(total=10, completed=3, failed=2)
        assert progress.pending == 5

    def test_percentage_calculation(self):
        """Test percentage calculation."""
        progress = BatchProgress(total=10, completed=5, failed=2)
        assert progress.percentage == 70.0

    def test_percentage_with_zero_total(self):
        """Test percentage with zero total (edge case)."""
        progress = BatchProgress(total=0)
        assert progress.percentage == 100.0


class TestURLProcessingResult:
    """Tests for URLProcessingResult dataclass."""

    def test_successful_result(self):
        """Test successful processing result."""
        result = URLProcessingResult(
            url="https://example.com",
            status=ProcessingStatus.COMPLETED,
            classification={"type": "reference", "name": "Example"}
        )
        assert result.url == "https://example.com"
        assert result.status == ProcessingStatus.COMPLETED
        assert result.classification["type"] == "reference"
        assert result.error is None

    def test_failed_result(self):
        """Test failed processing result."""
        result = URLProcessingResult(
            url="https://example.com",
            status=ProcessingStatus.FAILED,
            error="Connection timeout"
        )
        assert result.status == ProcessingStatus.FAILED
        assert result.error == "Connection timeout"
        assert result.classification is None


class TestDocumentIngestionTool:
    """Tests for DocumentIngestionTool."""

    @pytest.fixture
    def ingestion_tool(self, mock_knowledge_base, mock_ollama_client):
        """Create a DocumentIngestionTool with mocked dependencies."""
        # Configure mock to return valid classification
        mock_ollama_client.generate.return_value = '''
        {
            "type": "reference",
            "name": "Test Content",
            "category": "general",
            "description": "Test description",
            "key_points": ["Point 1", "Point 2"]
        }
        '''
        return DocumentIngestionTool(mock_knowledge_base, mock_ollama_client)

    def test_classify_content(self, ingestion_tool):
        """Test content classification."""
        content = "This is some test content"
        context = "Testing classification"

        classification = ingestion_tool.classify_content(content, context)

        assert classification["type"] == "reference"
        assert classification["name"] == "Test Content"
        assert "key_points" in classification

    def test_classify_content_fallback(self, ingestion_tool, mock_ollama_client):
        """Test content classification fallback on invalid response."""
        mock_ollama_client.generate.return_value = "Invalid JSON response"

        classification = ingestion_tool.classify_content("content", "context")

        # Should fall back to default
        assert classification["type"] == "reference"
        assert classification["name"] == "Web Content"


class TestAsyncDocumentIngestion:
    """Tests for async document ingestion functionality."""

    @pytest.fixture
    def ingestion_tool(self, mock_knowledge_base, mock_ollama_client):
        """Create a DocumentIngestionTool with mocked dependencies."""
        mock_ollama_client.generate.return_value = '''
        {
            "type": "reference",
            "name": "Test Content",
            "category": "general",
            "description": "Test description",
            "key_points": ["Point 1", "Point 2"]
        }
        '''
        return DocumentIngestionTool(mock_knowledge_base, mock_ollama_client)

    @pytest.mark.asyncio
    async def test_fetch_webpage_async(self, ingestion_tool):
        """Test async webpage fetching."""
        mock_response = AsyncMock()
        mock_response.text = AsyncMock(return_value="<html><body>Test content</body></html>")
        mock_response.raise_for_status = MagicMock()

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_response),
            __aexit__=AsyncMock(return_value=None)
        ))

        # Patch aiohttp.ClientSession
        with patch('aiohttp.ClientSession', return_value=mock_session):
            content = await ingestion_tool.fetch_webpage_async("https://example.com")
            assert "Test content" in content

    @pytest.mark.asyncio
    async def test_fetch_webpage_async_google_docs(self, ingestion_tool):
        """Test async fetching of Google Docs URLs."""
        mock_response = AsyncMock()
        mock_response.text = AsyncMock(return_value="Plain text content from Google Docs")
        mock_response.raise_for_status = MagicMock()

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_response),
            __aexit__=AsyncMock(return_value=None)
        ))

        with patch('aiohttp.ClientSession', return_value=mock_session):
            # Google Docs URLs should be converted to export format
            content = await ingestion_tool.fetch_webpage_async(
                "https://docs.google.com/document/d/abc123/edit",
                session=mock_session
            )
            # For Google Docs export URLs, content should be returned as-is
            assert "Plain text content" in content or content == "Plain text content from Google Docs"

    @pytest.mark.asyncio
    async def test_batch_process_urls_success(self, ingestion_tool):
        """Test batch URL processing with successful results."""
        urls = ["https://example1.com", "https://example2.com"]

        # Mock the async methods
        mock_classification = {
            "type": "reference",
            "name": "Test",
            "description": "Test description",
            "key_points": ["Point 1"]
        }

        with patch.object(
            ingestion_tool,
            'add_web_content_async',
            new_callable=AsyncMock,
            return_value=mock_classification
        ):
            result = await ingestion_tool.batch_process_urls(urls, "test context")

            assert result.total == 2
            assert result.completed == 2
            assert result.failed == 0
            assert len(result.results) == 2
            assert all(r.status == ProcessingStatus.COMPLETED for r in result.results)

    @pytest.mark.asyncio
    async def test_batch_process_urls_with_failures(self, ingestion_tool):
        """Test batch URL processing with some failures."""
        urls = ["https://good.com", "https://bad.com"]

        call_count = 0

        async def mock_add_content(url, context, session=None):
            nonlocal call_count
            call_count += 1
            if "bad" in url:
                raise Exception("Connection failed")
            return {
                "type": "reference",
                "name": "Test",
                "description": "Test",
                "key_points": []
            }

        with patch.object(
            ingestion_tool,
            'add_web_content_async',
            side_effect=mock_add_content
        ):
            result = await ingestion_tool.batch_process_urls(urls)

            assert result.total == 2
            assert result.completed == 1
            assert result.failed == 1

    @pytest.mark.asyncio
    async def test_batch_process_urls_progress_callback(self, ingestion_tool):
        """Test that progress callback is called during batch processing."""
        urls = ["https://example.com"]
        progress_updates = []

        def progress_callback(progress):
            progress_updates.append({
                "completed": progress.completed,
                "failed": progress.failed,
                "current_url": progress.current_url
            })

        mock_classification = {
            "type": "reference",
            "name": "Test",
            "description": "Test",
            "key_points": []
        }

        with patch.object(
            ingestion_tool,
            'add_web_content_async',
            new_callable=AsyncMock,
            return_value=mock_classification
        ):
            await ingestion_tool.batch_process_urls(
                urls,
                progress_callback=progress_callback
            )

            # Should have been called multiple times (initial, during, final)
            assert len(progress_updates) >= 2

    @pytest.mark.asyncio
    async def test_batch_process_urls_concurrency_limit(self, ingestion_tool):
        """Test that concurrency limit is respected."""
        urls = [f"https://example{i}.com" for i in range(10)]
        max_concurrent = 3
        concurrent_calls = []
        max_concurrent_calls = 0

        async def mock_add_content(url, context, session=None):
            nonlocal max_concurrent_calls
            concurrent_calls.append(1)
            current_concurrent = sum(concurrent_calls)
            max_concurrent_calls = max(max_concurrent_calls, current_concurrent)

            # Simulate some processing time
            await asyncio.sleep(0.01)

            concurrent_calls.pop()
            return {
                "type": "reference",
                "name": "Test",
                "description": "Test",
                "key_points": []
            }

        with patch.object(
            ingestion_tool,
            'add_web_content_async',
            side_effect=mock_add_content
        ):
            await ingestion_tool.batch_process_urls(
                urls,
                max_concurrent=max_concurrent
            )

            # Concurrency should not exceed the limit
            assert max_concurrent_calls <= max_concurrent

    def test_batch_process_urls_sync(self, ingestion_tool):
        """Test synchronous wrapper for batch processing."""
        urls = ["https://example.com"]

        mock_classification = {
            "type": "reference",
            "name": "Test",
            "description": "Test",
            "key_points": []
        }

        with patch.object(
            ingestion_tool,
            'add_web_content_async',
            new_callable=AsyncMock,
            return_value=mock_classification
        ):
            result = ingestion_tool.batch_process_urls_sync(urls)

            assert result.total == 1
            assert result.completed == 1
            assert result.failed == 0


class TestProcessingStatus:
    """Tests for ProcessingStatus enum."""

    def test_status_values(self):
        """Test ProcessingStatus enum values."""
        assert ProcessingStatus.PENDING.value == "pending"
        assert ProcessingStatus.IN_PROGRESS.value == "in_progress"
        assert ProcessingStatus.COMPLETED.value == "completed"
        assert ProcessingStatus.FAILED.value == "failed"
