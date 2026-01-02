"""Tests for knowledge_base.py module."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open
import json

from src.knowledge_base import KnowledgeBase


class TestKnowledgeBase:
    """Tests for KnowledgeBase class."""

    @pytest.fixture
    def mock_chroma_client(self, mock_chroma_collection):
        """Create a mock ChromaDB client."""
        client = MagicMock()
        client.get_or_create_collection.return_value = mock_chroma_collection
        return client

    @pytest.fixture
    def knowledge_base(self, temp_dir, mock_chroma_client, mock_ollama_client):
        """Create a KnowledgeBase with mocked dependencies."""
        with patch('src.knowledge_base.settings') as mock_settings, \
             patch('src.knowledge_base.chromadb.PersistentClient', return_value=mock_chroma_client), \
             patch('src.knowledge_base.OllamaClient', return_value=mock_ollama_client):

            mock_settings.chroma_persist_directory = temp_dir / "chroma"
            mock_settings.chroma_collection_name = "test_collection"

            kb = KnowledgeBase()
            kb.ollama_client = mock_ollama_client  # Ensure mock is used
            yield kb

    def test_initialization(self, knowledge_base):
        """Test KnowledgeBase initialization."""
        assert knowledge_base.collection is not None
        assert knowledge_base.ollama_client is not None

    def test_add_document(self, knowledge_base, mock_chroma_collection, mock_ollama_client):
        """Test adding a document to the knowledge base."""
        content = "This is a test document about Python programming."
        metadata = {"type": "test", "topic": "python"}

        knowledge_base.add_document(content, metadata)

        # Verify embeddings were generated
        mock_ollama_client.get_embeddings.assert_called_with(content)

        # Verify document was added to collection
        mock_chroma_collection.add.assert_called_once()
        call_args = mock_chroma_collection.add.call_args
        assert call_args.kwargs["documents"] == [content]
        assert call_args.kwargs["metadatas"] == [metadata]

    def test_add_document_with_custom_id(self, knowledge_base, mock_chroma_collection, mock_ollama_client):
        """Test adding a document with a custom ID."""
        content = "Custom ID document"
        metadata = {"type": "custom"}
        doc_id = "custom_doc_123"

        knowledge_base.add_document(content, metadata, document_id=doc_id)

        call_args = mock_chroma_collection.add.call_args
        assert call_args.kwargs["ids"] == [doc_id]

    def test_add_document_error_handling(self, knowledge_base, mock_ollama_client):
        """Test error handling when adding document fails."""
        mock_ollama_client.get_embeddings.side_effect = Exception("Embedding error")

        with pytest.raises(Exception):
            knowledge_base.add_document("Test content", {"type": "test"})

    def test_search(self, knowledge_base, mock_chroma_collection, mock_ollama_client):
        """Test searching the knowledge base."""
        # Setup mock search results
        mock_chroma_collection.query.return_value = {
            'ids': [['doc_1', 'doc_2']],
            'documents': [['First document', 'Second document']],
            'metadatas': [[{'type': 'test'}, {'type': 'test'}]],
            'distances': [[0.1, 0.2]]
        }

        results = knowledge_base.search("test query", n_results=5)

        # Verify embedding was generated for query
        mock_ollama_client.get_embeddings.assert_called_with("test query")

        # Verify results format
        assert len(results) == 2
        assert results[0]["content"] == "First document"
        assert results[0]["metadata"]["type"] == "test"
        assert results[0]["distance"] == 0.1
        assert results[0]["id"] == "doc_1"

    def test_search_with_filter(self, knowledge_base, mock_chroma_collection, mock_ollama_client):
        """Test searching with metadata filter."""
        mock_chroma_collection.query.return_value = {
            'ids': [['doc_1']],
            'documents': [['Filtered document']],
            'metadatas': [[{'type': 'filtered'}]],
            'distances': [[0.05]]
        }

        filter_metadata = {"type": {"$eq": "framework"}}
        results = knowledge_base.search("query", filter_metadata=filter_metadata)

        call_args = mock_chroma_collection.query.call_args
        assert call_args.kwargs["where"] == filter_metadata

    def test_search_empty_results(self, knowledge_base, mock_chroma_collection, mock_ollama_client):
        """Test search with no results."""
        mock_chroma_collection.query.return_value = {
            'ids': [[]],
            'documents': [[]],
            'metadatas': [[]],
            'distances': [[]]
        }

        results = knowledge_base.search("nonexistent query")

        assert results == []

    def test_search_error_handling(self, knowledge_base, mock_ollama_client):
        """Test error handling when search fails."""
        mock_ollama_client.get_embeddings.side_effect = Exception("Search error")

        with pytest.raises(Exception):
            knowledge_base.search("test query")

    def test_get_stats(self, knowledge_base, mock_chroma_collection):
        """Test getting knowledge base statistics."""
        mock_chroma_collection.get.return_value = {
            'ids': ['doc_1', 'doc_2', 'doc_3'],
            'metadatas': [
                {'type': 'value'},
                {'type': 'value'},
                {'type': 'framework'}
            ]
        }

        stats = knowledge_base.get_stats()

        assert stats["total_documents"] == 3
        assert stats["types"]["value"] == 2
        assert stats["types"]["framework"] == 1

    def test_get_stats_empty(self, knowledge_base, mock_chroma_collection):
        """Test getting stats from empty knowledge base."""
        mock_chroma_collection.get.return_value = {
            'ids': [],
            'metadatas': []
        }

        stats = knowledge_base.get_stats()

        assert stats["total_documents"] == 0
        assert stats["types"] == {}


class TestKnowledgeFileLoading:
    """Tests for loading knowledge files."""

    @pytest.fixture
    def mock_chroma_client(self, mock_chroma_collection):
        """Create a mock ChromaDB client."""
        client = MagicMock()
        client.get_or_create_collection.return_value = mock_chroma_collection
        return client

    @pytest.fixture
    def knowledge_base(self, temp_dir, mock_chroma_client, mock_ollama_client, mock_chroma_collection):
        """Create a KnowledgeBase with mocked dependencies."""
        with patch('src.knowledge_base.settings') as mock_settings, \
             patch('src.knowledge_base.chromadb.PersistentClient', return_value=mock_chroma_client), \
             patch('src.knowledge_base.OllamaClient', return_value=mock_ollama_client):

            mock_settings.chroma_persist_directory = temp_dir / "chroma"
            mock_settings.chroma_collection_name = "test_collection"

            kb = KnowledgeBase()
            kb.ollama_client = mock_ollama_client
            kb.collection = mock_chroma_collection
            yield kb

    def test_load_knowledge_files_values(self, knowledge_base, temp_dir, mock_chroma_collection):
        """Test loading values from knowledge files."""
        # Reset the get mock to return empty initially
        mock_chroma_collection.get.return_value = {'ids': [], 'metadatas': []}

        # Create knowledge directory with values file
        knowledge_dir = temp_dir / "knowledge"
        knowledge_dir.mkdir()

        values_data = {
            "values": [
                {
                    "name": "Innovation",
                    "description": "Always seek new and better ways",
                    "key_points": ["Think creatively", "Embrace change"],
                    "source": "Personal experience"
                }
            ]
        }

        with open(knowledge_dir / "values.json", 'w') as f:
            json.dump(values_data, f)

        knowledge_base.load_knowledge_files(knowledge_dir)

        # Verify document was added
        mock_chroma_collection.add.assert_called()

    def test_load_knowledge_files_frameworks(self, knowledge_base, temp_dir, mock_chroma_collection):
        """Test loading frameworks from knowledge files."""
        mock_chroma_collection.get.return_value = {'ids': [], 'metadatas': []}

        # Create knowledge directory with frameworks
        knowledge_dir = temp_dir / "knowledge"
        frameworks_dir = knowledge_dir / "frameworks"
        frameworks_dir.mkdir(parents=True)

        framework_data = {
            "name": "Problem Solving",
            "description": "A structured approach to solving problems",
            "category": "thinking",
            "steps": ["Define problem", "Analyze", "Generate solutions", "Implement"]
        }

        with open(frameworks_dir / "problem_solving.json", 'w') as f:
            json.dump(framework_data, f)

        knowledge_base.load_knowledge_files(knowledge_dir)

        mock_chroma_collection.add.assert_called()

    def test_load_knowledge_files_preferences(self, knowledge_base, temp_dir, mock_chroma_collection):
        """Test loading preferences from knowledge files."""
        mock_chroma_collection.get.return_value = {'ids': [], 'metadatas': []}

        # Create knowledge directory with preferences
        knowledge_dir = temp_dir / "knowledge"
        knowledge_dir.mkdir()

        preferences_data = {
            "communication": {
                "style": "direct",
                "format": "structured"
            },
            "work": {
                "environment": "quiet"
            }
        }

        with open(knowledge_dir / "preferences.json", 'w') as f:
            json.dump(preferences_data, f)

        knowledge_base.load_knowledge_files(knowledge_dir)

        # Should have added multiple preference documents
        assert mock_chroma_collection.add.call_count >= 3

    def test_load_knowledge_files_already_loaded(self, knowledge_base, temp_dir, mock_chroma_collection):
        """Test that already loaded knowledge files are not reloaded."""
        # Mock that knowledge files are already loaded
        mock_chroma_collection.get.return_value = {
            'ids': ['existing_doc'],
            'metadatas': [{'source': 'knowledge_files'}]
        }

        knowledge_dir = temp_dir / "knowledge"
        knowledge_dir.mkdir()

        # Create a values file
        with open(knowledge_dir / "values.json", 'w') as f:
            json.dump({"values": [{"name": "Test", "description": "Test value"}]}, f)

        # Clear any previous calls
        mock_chroma_collection.add.reset_mock()

        knowledge_base.load_knowledge_files(knowledge_dir)

        # Should not add any new documents
        mock_chroma_collection.add.assert_not_called()

    def test_load_knowledge_files_empty_directory(self, knowledge_base, temp_dir, mock_chroma_collection):
        """Test loading from empty knowledge directory."""
        mock_chroma_collection.get.return_value = {'ids': [], 'metadatas': []}

        knowledge_dir = temp_dir / "knowledge"
        knowledge_dir.mkdir()

        # Clear any previous calls
        mock_chroma_collection.add.reset_mock()

        knowledge_base.load_knowledge_files(knowledge_dir)

        # Should not add any documents from empty directory
        mock_chroma_collection.add.assert_not_called()

    def test_load_knowledge_files_nonexistent_directory(self, knowledge_base, temp_dir, mock_chroma_collection):
        """Test loading from non-existent knowledge directory."""
        mock_chroma_collection.get.return_value = {'ids': [], 'metadatas': []}

        nonexistent_dir = temp_dir / "nonexistent"

        # Should not raise error, just not load anything
        knowledge_base.load_knowledge_files(nonexistent_dir)

    def test_load_values_with_all_fields(self, knowledge_base, temp_dir, mock_chroma_collection):
        """Test loading values with all optional fields."""
        mock_chroma_collection.get.return_value = {'ids': [], 'metadatas': []}

        knowledge_dir = temp_dir / "knowledge"
        knowledge_dir.mkdir()

        values_data = {
            "values": [
                {
                    "name": "Comprehensive Value",
                    "description": "A value with all fields",
                    "key_points": ["Point 1", "Point 2"],
                    "source": "Personal development",
                    "user_context": "Applied daily in work"
                }
            ]
        }

        with open(knowledge_dir / "values.json", 'w') as f:
            json.dump(values_data, f)

        knowledge_base.load_knowledge_files(knowledge_dir)

        # Verify the content includes all fields
        call_args = mock_chroma_collection.add.call_args
        content = call_args.kwargs["documents"][0]

        assert "Comprehensive Value" in content
        assert "Point 1" in content
        assert "Personal development" in content
        assert "Applied daily in work" in content
