"""Shared pytest fixtures for glenn-bot tests."""

import pytest
from pathlib import Path
from datetime import datetime
from unittest.mock import MagicMock, patch
import tempfile
import shutil


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test data."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def mock_ollama_client():
    """Create a mock OllamaClient."""
    mock = MagicMock()
    mock.generate.return_value = "Mock LLM response"
    mock.get_embeddings.return_value = [0.1] * 768  # Mock embedding vector
    mock.list_models.return_value = [{"name": "llama3:8b"}]
    return mock


@pytest.fixture
def mock_knowledge_base():
    """Create a mock KnowledgeBase."""
    mock = MagicMock()
    mock.search.return_value = []
    mock.add_document.return_value = None
    mock.get_stats.return_value = {"total_documents": 0, "types": {}}
    return mock


@pytest.fixture
def mock_memory_system(mock_knowledge_base, mock_ollama_client):
    """Create a mock MemorySystem."""
    mock = MagicMock()
    mock.current_context = None
    mock.memories = {}
    mock.contexts = {}
    mock.knowledge_base = mock_knowledge_base
    mock.ollama_client = mock_ollama_client
    mock.add_memory.return_value = MagicMock(id="test_memory_1")
    mock.recall_memories.return_value = []
    mock.get_context_summary.return_value = "No context selected"
    return mock


@pytest.fixture
def mock_settings(temp_dir):
    """Create mock settings with temp directories."""
    mock = MagicMock()
    mock.conversation_history_dir = temp_dir / "conversations"
    mock.chroma_persist_directory = temp_dir / "chroma"
    mock.chroma_collection_name = "test_collection"
    mock.knowledge_dir = temp_dir / "knowledge"
    mock.ollama_host = "http://localhost:11434"
    mock.ollama_model = "llama3:8b"
    mock.ollama_embedding_model = "nomic-embed-text"

    # Create directories
    mock.conversation_history_dir.mkdir(parents=True, exist_ok=True)
    mock.chroma_persist_directory.mkdir(parents=True, exist_ok=True)
    mock.knowledge_dir.mkdir(parents=True, exist_ok=True)

    return mock


@pytest.fixture
def sample_conversation_data():
    """Sample conversation data for testing."""
    return {
        "id": "20240101_120000",
        "started_at": "2024-01-01T12:00:00",
        "messages": [
            {
                "role": "user",
                "content": "Hello, how are you?",
                "timestamp": "2024-01-01T12:00:00",
                "metadata": {}
            },
            {
                "role": "assistant",
                "content": "I'm doing well, thank you!",
                "timestamp": "2024-01-01T12:00:01",
                "metadata": {}
            }
        ]
    }


@pytest.fixture
def sample_memory_data():
    """Sample memory data for testing."""
    return {
        "id": "mem_1_20240101_120000",
        "memory_type": "personal",
        "content": "User prefers direct communication",
        "context": "general",
        "importance": 7,
        "created_at": "2024-01-01T12:00:00",
        "last_accessed": "2024-01-01T12:00:00",
        "access_count": 1,
        "tags": ["communication", "preferences"],
        "projects": ["personal_brand"],
        "metadata": {}
    }


@pytest.fixture
def sample_context_data():
    """Sample context data for testing."""
    return {
        "id": "test_context",
        "name": "Test Context",
        "description": "A test context for unit testing",
        "context_type": "testing",
        "goals": ["Test goal 1", "Test goal 2"],
        "key_people": ["Tester"],
        "current_focus": "Writing tests",
        "status": "active",
        "created_at": "2024-01-01T12:00:00",
        "last_used": "2024-01-01T12:00:00",
        "metadata": {}
    }


@pytest.fixture
def sample_quote_data():
    """Sample quote data for testing."""
    return {
        "id": "quote_1_20240101_120000",
        "text": "The only way to do great work is to love what you do.",
        "author": "Steve Jobs",
        "source": "Stanford Commencement Speech",
        "context": "Reminder to find passion in work",
        "tags": ["passion", "work", "success"],
        "category": "inspiration",
        "created_at": "2024-01-01T12:00:00",
        "last_reflected": None,
        "reflection_count": 0,
        "importance": 9,
        "projects": ["personal_brand"]
    }


# Library module fixtures

@pytest.fixture
def sample_chromadb_document():
    """Sample ChromaDB document as returned by KnowledgeBase.export_knowledge()."""
    return {
        "id": "doc_123",
        "content": "Systems thinking is a holistic approach to analysis that focuses on the way a system's constituent parts interrelate.",
        "metadata": {
            "type": "framework",
            "name": "Systems Thinking Framework",
            "source": "https://example.com/systems-thinking",
            "category": "mental_models"
        },
        "embedding": [0.1] * 768
    }


@pytest.fixture
def sample_library_item_data():
    """Sample LibraryItem data for testing."""
    return {
        "id": "doc_123",
        "content_type": "framework",
        "title": "Systems Thinking Framework",
        "summary": "Systems thinking is a holistic approach to analysis that focuses on the way a system's constituent parts interrelate.",
        "full_content": "Systems thinking is a holistic approach to analysis that focuses on the way a system's constituent parts interrelate. It emphasizes feedback loops, emergence, and interconnectedness.",
        "source_url": "https://example.com/systems-thinking",
        "cover_image_url": None,
        "metadata": {
            "category": "mental_models"
        },
        "themes": ["systems-thinking", "mental-models"],
        "created_at": "2024-01-15T10:30:00",
        "highlights": [
            "Focus on feedback loops and interconnections",
            "Consider the whole rather than individual parts"
        ]
    }


@pytest.fixture
def sample_theme_data():
    """Sample Theme data for testing."""
    return {
        "id": "systems-thinking",
        "name": "Systems Thinking",
        "description": "Content related to understanding complex systems, feedback loops, and emergent behavior.",
        "keywords": ["systems", "feedback", "complexity", "emergence", "holistic"],
        "item_count": 5,
        "created_at": "2024-01-15T10:00:00",
        "updated_at": "2024-01-20T14:30:00"
    }


@pytest.fixture
def sample_theme_assignment_data():
    """Sample ThemeAssignment data for testing."""
    return {
        "item_id": "doc_123",
        "theme_id": "systems-thinking",
        "confidence": 0.85,
        "assigned_at": "2024-01-15T10:30:00"
    }


@pytest.fixture
def mock_library_settings(temp_dir):
    """Create mock settings with library directories."""
    mock = MagicMock()
    mock.library_data_dir = temp_dir / "library"
    mock.library_site_dir = temp_dir / "library-site"
    mock.library_server_port = 8080

    # Also include standard settings
    mock.conversation_history_dir = temp_dir / "conversations"
    mock.chroma_persist_directory = temp_dir / "chroma"
    mock.chroma_collection_name = "test_collection"
    mock.knowledge_dir = temp_dir / "knowledge"
    mock.ollama_host = "http://localhost:11434"
    mock.ollama_model = "llama3:8b"
    mock.ollama_embedding_model = "nomic-embed-text"

    # Create library directories
    mock.library_data_dir.mkdir(parents=True, exist_ok=True)
    mock.library_site_dir.mkdir(parents=True, exist_ok=True)

    return mock
