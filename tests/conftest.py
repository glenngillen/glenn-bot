"""Shared pytest fixtures for glenn-bot tests."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from datetime import datetime
import tempfile
import shutil


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def mock_settings(temp_dir):
    """Mock settings with temporary directories."""
    with patch('src.config.settings') as mock:
        mock.conversation_history_dir = temp_dir / "conversations"
        mock.chroma_persist_directory = temp_dir / "chroma"
        mock.chroma_collection_name = "test_collection"
        mock.knowledge_dir = temp_dir / "knowledge"
        mock.ollama_host = "http://localhost:11434"
        mock.ollama_model = "llama3:8b"
        mock.ollama_embedding_model = "nomic-embed-text"
        yield mock


@pytest.fixture
def mock_ollama_client():
    """Create a mock OllamaClient."""
    client = MagicMock()
    client.generate.return_value = "Mock response"
    client.get_embeddings.return_value = [0.1] * 384  # Typical embedding dimension
    return client


@pytest.fixture
def mock_knowledge_base(mock_ollama_client):
    """Create a mock KnowledgeBase."""
    kb = MagicMock()
    kb.search.return_value = []
    kb.add_document.return_value = None
    return kb


@pytest.fixture
def mock_chroma_collection():
    """Create a mock ChromaDB collection."""
    collection = MagicMock()
    collection.get.return_value = {'ids': [], 'metadatas': [], 'documents': []}
    collection.query.return_value = {
        'ids': [[]],
        'documents': [[]],
        'metadatas': [[]],
        'distances': [[]]
    }
    collection.add.return_value = None
    return collection


@pytest.fixture
def sample_message_data():
    """Sample message data for testing."""
    return {
        "role": "user",
        "content": "Hello, this is a test message",
        "timestamp": datetime.now().isoformat(),
        "metadata": {"source": "test"}
    }


@pytest.fixture
def sample_memory_data():
    """Sample memory data for testing."""
    return {
        "id": "mem_0_20240101_120000",
        "memory_type": "personal",
        "content": "User prefers morning meetings",
        "context": "general",
        "importance": 7,
        "created_at": datetime.now().isoformat(),
        "last_accessed": datetime.now().isoformat(),
        "access_count": 1,
        "tags": ["preferences", "scheduling"],
        "projects": [],
        "metadata": {}
    }


@pytest.fixture
def sample_context_data():
    """Sample context data for testing."""
    return {
        "id": "test_project",
        "name": "Test Project",
        "description": "A test project for unit testing",
        "context_type": "product_development",
        "goals": ["Test goal 1", "Test goal 2"],
        "key_people": ["Person A"],
        "current_focus": "Testing",
        "status": "active",
        "created_at": datetime.now().isoformat(),
        "last_used": datetime.now().isoformat(),
        "metadata": {}
    }
