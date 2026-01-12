"""Tests for the knowledge_base module."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, Mock
import json


class TestKnowledgeBase:
    """Tests for the KnowledgeBase class."""

    @patch('src.knowledge_base.OllamaClient')
    @patch('src.knowledge_base.chromadb')
    @patch('src.knowledge_base.settings')
    def test_init_creates_directory(self, mock_settings, mock_chromadb, mock_ollama, temp_dir):
        """Test that initialization creates the persist directory."""
        mock_settings.chroma_persist_directory = temp_dir / "chroma"
        mock_settings.chroma_collection_name = "test_collection"

        # Mock chromadb client and collection
        mock_collection = MagicMock()
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        from src.knowledge_base import KnowledgeBase

        kb = KnowledgeBase()

        assert mock_settings.chroma_persist_directory.exists()
        mock_chromadb.PersistentClient.assert_called_once()
        mock_client.get_or_create_collection.assert_called_once()

    @patch('src.knowledge_base.OllamaClient')
    @patch('src.knowledge_base.chromadb')
    @patch('src.knowledge_base.settings')
    def test_add_document(self, mock_settings, mock_chromadb, mock_ollama_class, temp_dir):
        """Test adding a document to the knowledge base."""
        mock_settings.chroma_persist_directory = temp_dir / "chroma"
        mock_settings.chroma_collection_name = "test_collection"

        # Mock chromadb
        mock_collection = MagicMock()
        mock_collection.get.return_value = {"ids": []}
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        # Mock ollama client instance
        mock_ollama_instance = MagicMock()
        mock_ollama_instance.get_embeddings.return_value = [0.1] * 768
        mock_ollama_class.return_value = mock_ollama_instance

        from src.knowledge_base import KnowledgeBase

        kb = KnowledgeBase()
        kb.add_document(
            content="Test document content",
            metadata={"type": "test", "source": "unit_test"},
            document_id="test_doc_1"
        )

        mock_ollama_instance.get_embeddings.assert_called_once_with("Test document content")
        mock_collection.add.assert_called_once()
        call_kwargs = mock_collection.add.call_args
        assert call_kwargs[1]["documents"] == ["Test document content"]
        assert call_kwargs[1]["ids"] == ["test_doc_1"]

    @patch('src.knowledge_base.OllamaClient')
    @patch('src.knowledge_base.chromadb')
    @patch('src.knowledge_base.settings')
    def test_add_document_generates_id(self, mock_settings, mock_chromadb, mock_ollama_class, temp_dir):
        """Test that document ID is auto-generated if not provided."""
        mock_settings.chroma_persist_directory = temp_dir / "chroma"
        mock_settings.chroma_collection_name = "test_collection"

        # Mock chromadb
        mock_collection = MagicMock()
        mock_collection.get.return_value = {"ids": ["doc_0", "doc_1"]}
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        # Mock ollama
        mock_ollama_instance = MagicMock()
        mock_ollama_instance.get_embeddings.return_value = [0.1] * 768
        mock_ollama_class.return_value = mock_ollama_instance

        from src.knowledge_base import KnowledgeBase

        kb = KnowledgeBase()
        kb.add_document(
            content="Auto ID document",
            metadata={"type": "test"}
        )

        call_kwargs = mock_collection.add.call_args
        # Should generate ID based on current collection size
        assert call_kwargs[1]["ids"] == ["doc_2"]

    @patch('src.knowledge_base.OllamaClient')
    @patch('src.knowledge_base.chromadb')
    @patch('src.knowledge_base.settings')
    def test_search(self, mock_settings, mock_chromadb, mock_ollama_class, temp_dir):
        """Test searching the knowledge base."""
        mock_settings.chroma_persist_directory = temp_dir / "chroma"
        mock_settings.chroma_collection_name = "test_collection"

        # Mock chromadb
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [["Doc 1 content", "Doc 2 content"]],
            "metadatas": [[{"type": "test"}, {"type": "test"}]],
            "distances": [[0.1, 0.2]],
            "ids": [["doc_1", "doc_2"]]
        }
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        # Mock ollama
        mock_ollama_instance = MagicMock()
        mock_ollama_instance.get_embeddings.return_value = [0.1] * 768
        mock_ollama_class.return_value = mock_ollama_instance

        from src.knowledge_base import KnowledgeBase

        kb = KnowledgeBase()
        results = kb.search("test query", n_results=2)

        mock_ollama_instance.get_embeddings.assert_called_with("test query")
        assert len(results) == 2
        assert results[0]["content"] == "Doc 1 content"
        assert results[0]["distance"] == 0.1
        assert results[1]["id"] == "doc_2"

    @patch('src.knowledge_base.OllamaClient')
    @patch('src.knowledge_base.chromadb')
    @patch('src.knowledge_base.settings')
    def test_search_with_filter(self, mock_settings, mock_chromadb, mock_ollama_class, temp_dir):
        """Test searching with metadata filter."""
        mock_settings.chroma_persist_directory = temp_dir / "chroma"
        mock_settings.chroma_collection_name = "test_collection"

        # Mock chromadb
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [["Filtered doc"]],
            "metadatas": [[{"type": "value"}]],
            "distances": [[0.05]],
            "ids": [["value_doc_1"]]
        }
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        # Mock ollama
        mock_ollama_instance = MagicMock()
        mock_ollama_instance.get_embeddings.return_value = [0.1] * 768
        mock_ollama_class.return_value = mock_ollama_instance

        from src.knowledge_base import KnowledgeBase

        kb = KnowledgeBase()
        results = kb.search(
            "values query",
            n_results=5,
            filter_metadata={"type": {"$eq": "value"}}
        )

        call_kwargs = mock_collection.query.call_args
        assert call_kwargs[1]["where"] == {"type": {"$eq": "value"}}

    @patch('src.knowledge_base.OllamaClient')
    @patch('src.knowledge_base.chromadb')
    @patch('src.knowledge_base.settings')
    def test_get_stats(self, mock_settings, mock_chromadb, mock_ollama_class, temp_dir):
        """Test getting knowledge base statistics."""
        mock_settings.chroma_persist_directory = temp_dir / "chroma"
        mock_settings.chroma_collection_name = "test_collection"

        # Mock chromadb
        mock_collection = MagicMock()
        mock_collection.get.return_value = {
            "ids": ["doc_1", "doc_2", "doc_3"],
            "metadatas": [
                {"type": "value"},
                {"type": "framework"},
                {"type": "value"}
            ]
        }
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        # Mock ollama
        mock_ollama_instance = MagicMock()
        mock_ollama_class.return_value = mock_ollama_instance

        from src.knowledge_base import KnowledgeBase

        kb = KnowledgeBase()
        stats = kb.get_stats()

        assert stats["total_documents"] == 3
        assert stats["types"]["value"] == 2
        assert stats["types"]["framework"] == 1

    @patch('src.knowledge_base.OllamaClient')
    @patch('src.knowledge_base.chromadb')
    @patch('src.knowledge_base.settings')
    def test_load_knowledge_files_values(self, mock_settings, mock_chromadb, mock_ollama_class, temp_dir):
        """Test loading knowledge files from directory."""
        mock_settings.chroma_persist_directory = temp_dir / "chroma"
        mock_settings.chroma_collection_name = "test_collection"

        # Create knowledge directory and files
        knowledge_dir = temp_dir / "knowledge"
        knowledge_dir.mkdir(parents=True, exist_ok=True)

        values_data = {
            "values": [
                {
                    "name": "Test Value",
                    "description": "A test value for unit testing",
                    "key_points": ["Point 1", "Point 2"]
                }
            ]
        }
        with open(knowledge_dir / "values.json", 'w') as f:
            json.dump(values_data, f)

        # Mock chromadb - return empty so files will be loaded
        mock_collection = MagicMock()
        mock_collection.get.return_value = {"ids": [], "metadatas": []}
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        # Mock ollama
        mock_ollama_instance = MagicMock()
        mock_ollama_instance.get_embeddings.return_value = [0.1] * 768
        mock_ollama_class.return_value = mock_ollama_instance

        from src.knowledge_base import KnowledgeBase

        kb = KnowledgeBase()
        kb.load_knowledge_files(knowledge_dir)

        # Should have called add to add the value
        mock_collection.add.assert_called()

    @patch('src.knowledge_base.OllamaClient')
    @patch('src.knowledge_base.chromadb')
    @patch('src.knowledge_base.settings')
    def test_load_knowledge_files_skips_if_loaded(self, mock_settings, mock_chromadb, mock_ollama_class, temp_dir):
        """Test that knowledge files are not re-loaded if already present."""
        mock_settings.chroma_persist_directory = temp_dir / "chroma"
        mock_settings.chroma_collection_name = "test_collection"

        knowledge_dir = temp_dir / "knowledge"
        knowledge_dir.mkdir(parents=True, exist_ok=True)

        # Mock chromadb - return existing knowledge files marker
        mock_collection = MagicMock()
        mock_collection.get.return_value = {
            "ids": ["existing_1"],
            "metadatas": [{"source": "knowledge_files"}]
        }
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        # Mock ollama
        mock_ollama_instance = MagicMock()
        mock_ollama_class.return_value = mock_ollama_instance

        from src.knowledge_base import KnowledgeBase

        kb = KnowledgeBase()
        kb.load_knowledge_files(knowledge_dir)

        # Should not have called add since files already loaded
        mock_collection.add.assert_not_called()

    @patch('src.knowledge_base.OllamaClient')
    @patch('src.knowledge_base.chromadb')
    @patch('src.knowledge_base.settings')
    def test_load_knowledge_files_frameworks(self, mock_settings, mock_chromadb, mock_ollama_class, temp_dir):
        """Test loading framework files."""
        mock_settings.chroma_persist_directory = temp_dir / "chroma"
        mock_settings.chroma_collection_name = "test_collection"

        # Create knowledge directory and framework files
        knowledge_dir = temp_dir / "knowledge"
        frameworks_dir = knowledge_dir / "frameworks"
        frameworks_dir.mkdir(parents=True, exist_ok=True)

        framework_data = {
            "name": "Test Framework",
            "description": "A framework for testing",
            "steps": ["Step 1", "Step 2", "Step 3"],
            "category": "testing"
        }
        with open(frameworks_dir / "test_framework.json", 'w') as f:
            json.dump(framework_data, f)

        # Mock chromadb
        mock_collection = MagicMock()
        mock_collection.get.return_value = {"ids": [], "metadatas": []}
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        # Mock ollama
        mock_ollama_instance = MagicMock()
        mock_ollama_instance.get_embeddings.return_value = [0.1] * 768
        mock_ollama_class.return_value = mock_ollama_instance

        from src.knowledge_base import KnowledgeBase

        kb = KnowledgeBase()
        kb.load_knowledge_files(knowledge_dir)

        # Should have called add for the framework
        mock_collection.add.assert_called()
        # Check that framework metadata was passed
        calls = mock_collection.add.call_args_list
        assert any(
            call[1].get("metadatas", [{}])[0].get("type") == "framework"
            for call in calls
        )

    @patch('src.knowledge_base.OllamaClient')
    @patch('src.knowledge_base.chromadb')
    @patch('src.knowledge_base.settings')
    def test_add_document_error_handling(self, mock_settings, mock_chromadb, mock_ollama_class, temp_dir):
        """Test error handling when adding document fails."""
        mock_settings.chroma_persist_directory = temp_dir / "chroma"
        mock_settings.chroma_collection_name = "test_collection"

        # Mock chromadb to raise error
        mock_collection = MagicMock()
        mock_collection.add.side_effect = Exception("Database error")
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        # Mock ollama
        mock_ollama_instance = MagicMock()
        mock_ollama_instance.get_embeddings.return_value = [0.1] * 768
        mock_ollama_class.return_value = mock_ollama_instance

        from src.knowledge_base import KnowledgeBase

        kb = KnowledgeBase()

        with pytest.raises(Exception) as exc_info:
            kb.add_document(content="Test", metadata={})

        assert "Database error" in str(exc_info.value)

    @patch('src.knowledge_base.OllamaClient')
    @patch('src.knowledge_base.chromadb')
    @patch('src.knowledge_base.settings')
    def test_search_error_handling(self, mock_settings, mock_chromadb, mock_ollama_class, temp_dir):
        """Test error handling when search fails."""
        mock_settings.chroma_persist_directory = temp_dir / "chroma"
        mock_settings.chroma_collection_name = "test_collection"

        # Mock chromadb to raise error on query
        mock_collection = MagicMock()
        mock_collection.query.side_effect = Exception("Search failed")
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        # Mock ollama
        mock_ollama_instance = MagicMock()
        mock_ollama_instance.get_embeddings.return_value = [0.1] * 768
        mock_ollama_class.return_value = mock_ollama_instance

        from src.knowledge_base import KnowledgeBase

        kb = KnowledgeBase()

        with pytest.raises(Exception) as exc_info:
            kb.search("test query")

        assert "Search failed" in str(exc_info.value)

    @patch('src.knowledge_base.OllamaClient')
    @patch('src.knowledge_base.chromadb')
    @patch('src.knowledge_base.settings')
    def test_export_knowledge(self, mock_settings, mock_chromadb, mock_ollama_class, temp_dir):
        """Test exporting knowledge base to dictionary."""
        mock_settings.chroma_persist_directory = temp_dir / "chroma"
        mock_settings.chroma_collection_name = "test_collection"

        # Mock chromadb
        mock_collection = MagicMock()
        mock_collection.get.return_value = {
            "ids": ["doc_1", "doc_2", "doc_3"],
            "documents": ["Content 1", "Content 2", "Content 3"],
            "metadatas": [
                {"type": "value", "source": "knowledge_files"},
                {"type": "framework", "source": "knowledge_files"},
                {"type": "value", "source": "web"}
            ],
            "embeddings": [[0.1] * 768, [0.2] * 768, [0.3] * 768]
        }
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        # Mock ollama
        mock_ollama_instance = MagicMock()
        mock_ollama_class.return_value = mock_ollama_instance

        from src.knowledge_base import KnowledgeBase

        kb = KnowledgeBase()
        export_data = kb.export_knowledge()

        assert export_data["export_version"] == "1.0"
        assert "export_timestamp" in export_data
        assert export_data["total_documents"] == 3
        assert len(export_data["documents"]) == 3
        assert export_data["documents"][0]["id"] == "doc_1"
        assert export_data["documents"][0]["content"] == "Content 1"

    @patch('src.knowledge_base.OllamaClient')
    @patch('src.knowledge_base.chromadb')
    @patch('src.knowledge_base.settings')
    def test_export_knowledge_with_type_filter(self, mock_settings, mock_chromadb, mock_ollama_class, temp_dir):
        """Test exporting knowledge with type filter."""
        mock_settings.chroma_persist_directory = temp_dir / "chroma"
        mock_settings.chroma_collection_name = "test_collection"

        # Mock chromadb
        mock_collection = MagicMock()
        mock_collection.get.return_value = {
            "ids": ["doc_1", "doc_2", "doc_3"],
            "documents": ["Content 1", "Content 2", "Content 3"],
            "metadatas": [
                {"type": "value", "source": "knowledge_files"},
                {"type": "framework", "source": "knowledge_files"},
                {"type": "value", "source": "web"}
            ],
            "embeddings": [[0.1] * 768, [0.2] * 768, [0.3] * 768]
        }
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        # Mock ollama
        mock_ollama_instance = MagicMock()
        mock_ollama_class.return_value = mock_ollama_instance

        from src.knowledge_base import KnowledgeBase

        kb = KnowledgeBase()
        export_data = kb.export_knowledge(filter_type="value")

        # Should only export documents with type "value"
        assert export_data["total_documents"] == 2
        assert len(export_data["documents"]) == 2
        assert all(doc["metadata"]["type"] == "value" for doc in export_data["documents"])

    @patch('src.knowledge_base.OllamaClient')
    @patch('src.knowledge_base.chromadb')
    @patch('src.knowledge_base.settings')
    def test_export_knowledge_with_source_filter(self, mock_settings, mock_chromadb, mock_ollama_class, temp_dir):
        """Test exporting knowledge with source filter."""
        mock_settings.chroma_persist_directory = temp_dir / "chroma"
        mock_settings.chroma_collection_name = "test_collection"

        # Mock chromadb
        mock_collection = MagicMock()
        mock_collection.get.return_value = {
            "ids": ["doc_1", "doc_2", "doc_3"],
            "documents": ["Content 1", "Content 2", "Content 3"],
            "metadatas": [
                {"type": "value", "source": "knowledge_files"},
                {"type": "framework", "source": "knowledge_files"},
                {"type": "value", "source": "web"}
            ],
            "embeddings": [[0.1] * 768, [0.2] * 768, [0.3] * 768]
        }
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        # Mock ollama
        mock_ollama_instance = MagicMock()
        mock_ollama_class.return_value = mock_ollama_instance

        from src.knowledge_base import KnowledgeBase

        kb = KnowledgeBase()
        export_data = kb.export_knowledge(filter_source="web")

        # Should only export documents with source "web"
        assert export_data["total_documents"] == 1
        assert export_data["documents"][0]["metadata"]["source"] == "web"

    @patch('src.knowledge_base.OllamaClient')
    @patch('src.knowledge_base.chromadb')
    @patch('src.knowledge_base.settings')
    def test_import_knowledge_skip_duplicates(self, mock_settings, mock_chromadb, mock_ollama_class, temp_dir):
        """Test importing knowledge with skip duplicates strategy."""
        mock_settings.chroma_persist_directory = temp_dir / "chroma"
        mock_settings.chroma_collection_name = "test_collection"

        # Mock chromadb
        mock_collection = MagicMock()
        # Existing docs have doc_1
        mock_collection.get.return_value = {"ids": ["doc_1"]}
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        # Mock ollama
        mock_ollama_instance = MagicMock()
        mock_ollama_instance.get_embeddings.return_value = [0.1] * 768
        mock_ollama_class.return_value = mock_ollama_instance

        from src.knowledge_base import KnowledgeBase

        kb = KnowledgeBase()

        import_data = {
            "export_version": "1.0",
            "documents": [
                {"id": "doc_1", "content": "Duplicate content", "metadata": {"type": "value"}},
                {"id": "doc_2", "content": "New content", "metadata": {"type": "framework"}}
            ]
        }

        stats = kb.import_knowledge(import_data, duplicate_handling="skip")

        assert stats["total_in_file"] == 2
        assert stats["skipped"] == 1  # doc_1 skipped
        assert stats["added"] == 1   # doc_2 added
        assert stats["errors"] == 0

    @patch('src.knowledge_base.OllamaClient')
    @patch('src.knowledge_base.chromadb')
    @patch('src.knowledge_base.settings')
    def test_import_knowledge_update_duplicates(self, mock_settings, mock_chromadb, mock_ollama_class, temp_dir):
        """Test importing knowledge with update duplicates strategy."""
        mock_settings.chroma_persist_directory = temp_dir / "chroma"
        mock_settings.chroma_collection_name = "test_collection"

        # Mock chromadb
        mock_collection = MagicMock()
        mock_collection.get.return_value = {"ids": ["doc_1"]}
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        # Mock ollama
        mock_ollama_instance = MagicMock()
        mock_ollama_instance.get_embeddings.return_value = [0.1] * 768
        mock_ollama_class.return_value = mock_ollama_instance

        from src.knowledge_base import KnowledgeBase

        kb = KnowledgeBase()

        import_data = {
            "export_version": "1.0",
            "documents": [
                {"id": "doc_1", "content": "Updated content", "metadata": {"type": "value"}}
            ]
        }

        stats = kb.import_knowledge(import_data, duplicate_handling="update")

        # Should have deleted and re-added
        mock_collection.delete.assert_called_with(ids=["doc_1"])
        assert stats["added"] == 1

    @patch('src.knowledge_base.OllamaClient')
    @patch('src.knowledge_base.chromadb')
    @patch('src.knowledge_base.settings')
    def test_import_knowledge_fail_on_duplicates(self, mock_settings, mock_chromadb, mock_ollama_class, temp_dir):
        """Test importing knowledge fails on duplicates when configured."""
        mock_settings.chroma_persist_directory = temp_dir / "chroma"
        mock_settings.chroma_collection_name = "test_collection"

        # Mock chromadb
        mock_collection = MagicMock()
        mock_collection.get.return_value = {"ids": ["doc_1"]}
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        # Mock ollama
        mock_ollama_instance = MagicMock()
        mock_ollama_class.return_value = mock_ollama_instance

        from src.knowledge_base import KnowledgeBase

        kb = KnowledgeBase()

        import_data = {
            "export_version": "1.0",
            "documents": [
                {"id": "doc_1", "content": "Duplicate", "metadata": {"type": "value"}}
            ]
        }

        with pytest.raises(ValueError) as exc_info:
            kb.import_knowledge(import_data, duplicate_handling="fail")

        assert "Duplicate document ID" in str(exc_info.value)

    @patch('src.knowledge_base.OllamaClient')
    @patch('src.knowledge_base.chromadb')
    @patch('src.knowledge_base.settings')
    def test_import_knowledge_invalid_data(self, mock_settings, mock_chromadb, mock_ollama_class, temp_dir):
        """Test importing invalid data raises error."""
        mock_settings.chroma_persist_directory = temp_dir / "chroma"
        mock_settings.chroma_collection_name = "test_collection"

        # Mock chromadb
        mock_collection = MagicMock()
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        # Mock ollama
        mock_ollama_instance = MagicMock()
        mock_ollama_class.return_value = mock_ollama_instance

        from src.knowledge_base import KnowledgeBase

        kb = KnowledgeBase()

        # Missing documents key
        invalid_data = {"export_version": "1.0"}

        with pytest.raises(ValueError) as exc_info:
            kb.import_knowledge(invalid_data)

        assert "missing 'documents' key" in str(exc_info.value)

    @patch('src.knowledge_base.OllamaClient')
    @patch('src.knowledge_base.chromadb')
    @patch('src.knowledge_base.settings')
    def test_import_knowledge_uses_provided_embeddings(self, mock_settings, mock_chromadb, mock_ollama_class, temp_dir):
        """Test that import uses provided embeddings instead of generating new ones."""
        mock_settings.chroma_persist_directory = temp_dir / "chroma"
        mock_settings.chroma_collection_name = "test_collection"

        # Mock chromadb
        mock_collection = MagicMock()
        mock_collection.get.return_value = {"ids": []}
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        # Mock ollama
        mock_ollama_instance = MagicMock()
        mock_ollama_class.return_value = mock_ollama_instance

        from src.knowledge_base import KnowledgeBase

        kb = KnowledgeBase()

        provided_embedding = [0.5] * 768
        import_data = {
            "export_version": "1.0",
            "documents": [
                {
                    "id": "doc_1",
                    "content": "Content with embedding",
                    "metadata": {"type": "value"},
                    "embedding": provided_embedding
                }
            ]
        }

        kb.import_knowledge(import_data)

        # Should not have called get_embeddings since embedding was provided
        mock_ollama_instance.get_embeddings.assert_not_called()

        # Should have used the provided embedding
        call_kwargs = mock_collection.add.call_args
        assert call_kwargs[1]["embeddings"] == [provided_embedding]
