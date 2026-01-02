"""Tests for memory_system.py module."""

import pytest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch
import json

from src.memory_system import Memory, MemoryType, Context, MemorySystem


class TestMemoryType:
    """Tests for MemoryType enum."""

    def test_memory_types_exist(self):
        """Test that all expected memory types are defined."""
        expected_types = [
            "personal", "project", "insight", "goal",
            "decision", "relationship", "skill", "experience"
        ]
        for type_name in expected_types:
            assert hasattr(MemoryType, type_name.upper())
            assert MemoryType(type_name).value == type_name


class TestMemory:
    """Tests for Memory dataclass."""

    def test_memory_creation(self):
        """Test creating a Memory instance."""
        now = datetime.now()
        memory = Memory(
            id="test_mem_1",
            memory_type=MemoryType.PERSONAL,
            content="Test content",
            context="test_context",
            importance=7,
            created_at=now,
            last_accessed=now,
            access_count=1,
            tags={"tag1", "tag2"},
            projects={"project1"},
            metadata={"key": "value"}
        )

        assert memory.id == "test_mem_1"
        assert memory.memory_type == MemoryType.PERSONAL
        assert memory.content == "Test content"
        assert memory.importance == 7
        assert "tag1" in memory.tags
        assert "project1" in memory.projects

    def test_memory_to_dict(self):
        """Test Memory serialization to dict."""
        now = datetime.now()
        memory = Memory(
            id="test_mem_1",
            memory_type=MemoryType.INSIGHT,
            content="Important insight",
            context="general",
            importance=8,
            created_at=now,
            last_accessed=now,
            access_count=3,
            tags={"insight", "learning"},
            projects=set(),
            metadata={}
        )

        data = memory.to_dict()

        assert data["id"] == "test_mem_1"
        assert data["memory_type"] == "insight"
        assert data["content"] == "Important insight"
        assert data["importance"] == 8
        assert data["access_count"] == 3
        assert isinstance(data["tags"], list)
        assert isinstance(data["projects"], list)
        assert isinstance(data["created_at"], str)
        assert isinstance(data["last_accessed"], str)

    def test_memory_from_dict(self, sample_memory_data):
        """Test Memory deserialization from dict."""
        memory = Memory.from_dict(sample_memory_data)

        assert memory.id == sample_memory_data["id"]
        assert memory.memory_type == MemoryType.PERSONAL
        assert memory.content == sample_memory_data["content"]
        assert isinstance(memory.tags, set)
        assert isinstance(memory.projects, set)
        assert isinstance(memory.created_at, datetime)

    def test_memory_roundtrip(self):
        """Test Memory can be serialized and deserialized."""
        now = datetime.now()
        original = Memory(
            id="roundtrip_test",
            memory_type=MemoryType.GOAL,
            content="Achieve something great",
            context="personal_brand",
            importance=9,
            created_at=now,
            last_accessed=now,
            access_count=5,
            tags={"goal", "important"},
            projects={"personal_brand"},
            metadata={"priority": "high"}
        )

        data = original.to_dict()
        restored = Memory.from_dict(data)

        assert restored.id == original.id
        assert restored.memory_type == original.memory_type
        assert restored.content == original.content
        assert restored.importance == original.importance
        assert restored.tags == original.tags
        assert restored.projects == original.projects


class TestContext:
    """Tests for Context dataclass."""

    def test_context_creation(self):
        """Test creating a Context instance."""
        now = datetime.now()
        context = Context(
            id="test_ctx",
            name="Test Context",
            description="A test context",
            context_type="product_development",
            goals=["Goal 1", "Goal 2"],
            key_people=["Alice", "Bob"],
            current_focus="Testing",
            status="active",
            created_at=now,
            last_used=now,
            metadata={}
        )

        assert context.id == "test_ctx"
        assert context.name == "Test Context"
        assert len(context.goals) == 2
        assert context.status == "active"

    def test_context_to_dict(self):
        """Test Context serialization to dict."""
        now = datetime.now()
        context = Context(
            id="ctx_1",
            name="Development",
            description="Product development context",
            context_type="product_development",
            goals=["Ship MVP"],
            key_people=[],
            current_focus="Backend API",
            status="active",
            created_at=now,
            last_used=now,
            metadata={"priority": 1}
        )

        data = context.to_dict()

        assert data["id"] == "ctx_1"
        assert data["name"] == "Development"
        assert isinstance(data["created_at"], str)
        assert isinstance(data["last_used"], str)
        assert data["metadata"]["priority"] == 1

    def test_context_from_dict(self, sample_context_data):
        """Test Context deserialization from dict."""
        context = Context.from_dict(sample_context_data)

        assert context.id == sample_context_data["id"]
        assert context.name == sample_context_data["name"]
        assert isinstance(context.created_at, datetime)
        assert isinstance(context.last_used, datetime)
        assert len(context.goals) == 2


class TestMemorySystem:
    """Tests for MemorySystem class."""

    @pytest.fixture
    def memory_system(self, temp_dir, mock_knowledge_base, mock_ollama_client):
        """Create a MemorySystem instance with mocked dependencies."""
        with patch('src.memory_system.settings') as mock_settings:
            mock_settings.conversation_history_dir = temp_dir / "conversations"

            ms = MemorySystem(
                knowledge_base=mock_knowledge_base,
                ollama_client=mock_ollama_client
            )
            yield ms

    def test_initialization(self, memory_system):
        """Test MemorySystem initialization."""
        assert memory_system.memories is not None
        assert memory_system.contexts is not None
        assert memory_system.current_context is None

    def test_default_contexts_created(self, memory_system):
        """Test that default contexts are created on first initialization."""
        assert len(memory_system.contexts) > 0
        assert "personal_brand" in memory_system.contexts
        assert "product_dev" in memory_system.contexts
        assert "self_improvement" in memory_system.contexts

    def test_switch_context(self, memory_system):
        """Test switching between contexts."""
        result = memory_system.switch_context("personal_brand")

        assert result is True
        assert memory_system.current_context is not None
        assert memory_system.current_context.id == "personal_brand"

    def test_switch_context_invalid(self, memory_system):
        """Test switching to non-existent context."""
        result = memory_system.switch_context("nonexistent_context")

        assert result is False

    def test_create_context(self, memory_system):
        """Test creating a new context."""
        context = memory_system.create_context(
            name="New Project",
            description="A new project context",
            context_type="product_development",
            goals=["Launch product", "Get users"]
        )

        assert context is not None
        assert context.id == "new_project"
        assert context.name == "New Project"
        assert len(context.goals) == 2
        assert "new_project" in memory_system.contexts

    def test_create_context_unique_id(self, memory_system):
        """Test that duplicate context names get unique IDs."""
        ctx1 = memory_system.create_context(
            name="Test",
            description="First test",
            context_type="test"
        )
        ctx2 = memory_system.create_context(
            name="Test",
            description="Second test",
            context_type="test"
        )

        assert ctx1.id != ctx2.id
        assert ctx2.id == "test_1"

    def test_add_memory(self, memory_system, mock_knowledge_base):
        """Test adding a memory."""
        memory_system.switch_context("personal_brand")

        memory = memory_system.add_memory(
            content="User mentioned interest in AI",
            memory_type=MemoryType.PERSONAL,
            importance=7,
            tags={"AI", "interests"}
        )

        assert memory is not None
        assert memory.memory_type == MemoryType.PERSONAL
        assert memory.content == "User mentioned interest in AI"
        assert memory.importance == 7
        assert "AI" in memory.tags
        assert memory.id in memory_system.memories

        # Verify knowledge base was updated
        mock_knowledge_base.add_document.assert_called_once()

    def test_add_memory_without_context(self, memory_system, mock_knowledge_base):
        """Test adding a memory without an active context."""
        memory = memory_system.add_memory(
            content="General memory",
            memory_type=MemoryType.INSIGHT,
            importance=5
        )

        assert memory is not None
        assert memory.context == "general"
        assert len(memory.projects) == 0

    def test_recall_memories(self, memory_system, mock_knowledge_base):
        """Test recalling memories based on query."""
        # Setup mock to return a memory
        mock_knowledge_base.search.return_value = [
            {
                "content": "Memory content",
                "metadata": {"memory_id": "mem_0_20240101_120000", "type": "memory"},
                "distance": 0.1
            }
        ]

        # Add a memory first
        memory = memory_system.add_memory(
            content="User prefers Python",
            memory_type=MemoryType.PREFERENCE if hasattr(MemoryType, 'PREFERENCE') else MemoryType.PERSONAL,
            importance=6
        )

        # Update mock to return the correct memory_id
        mock_knowledge_base.search.return_value = [
            {
                "content": memory.content,
                "metadata": {"memory_id": memory.id, "type": "memory"},
                "distance": 0.1
            }
        ]

        recalled = memory_system.recall_memories("What programming languages?", limit=5)

        assert isinstance(recalled, list)
        mock_knowledge_base.search.assert_called()

    def test_get_context_summary(self, memory_system, mock_knowledge_base):
        """Test getting a context summary."""
        memory_system.switch_context("personal_brand")

        summary = memory_system.get_context_summary()

        assert summary is not None
        assert "Personal Brand Building" in summary
        assert "Context:" in summary

    def test_get_context_summary_no_context(self, memory_system):
        """Test getting summary without active context."""
        summary = memory_system.get_context_summary()

        assert summary == "No context selected"

    def test_update_context_focus(self, memory_system):
        """Test updating the current focus of a context."""
        memory_system.switch_context("personal_brand")
        original_last_used = memory_system.current_context.last_used

        memory_system.update_context_focus("Creating content")

        assert memory_system.current_context.current_focus == "Creating content"
        assert memory_system.current_context.last_used >= original_last_used

    def test_memory_persistence(self, temp_dir, mock_knowledge_base, mock_ollama_client):
        """Test that memories are persisted to disk."""
        with patch('src.memory_system.settings') as mock_settings:
            mock_settings.conversation_history_dir = temp_dir / "conversations"

            # Create first system and add memory
            ms1 = MemorySystem(mock_knowledge_base, mock_ollama_client)
            memory = ms1.add_memory(
                content="Persistent memory",
                memory_type=MemoryType.INSIGHT,
                importance=8
            )
            memory_id = memory.id

            # Create second system (should load from disk)
            ms2 = MemorySystem(mock_knowledge_base, mock_ollama_client)

            assert memory_id in ms2.memories
            assert ms2.memories[memory_id].content == "Persistent memory"

    def test_context_persistence(self, temp_dir, mock_knowledge_base, mock_ollama_client):
        """Test that contexts are persisted to disk."""
        with patch('src.memory_system.settings') as mock_settings:
            mock_settings.conversation_history_dir = temp_dir / "conversations"

            # Create first system and add context
            ms1 = MemorySystem(mock_knowledge_base, mock_ollama_client)
            ctx = ms1.create_context(
                name="Persistent Context",
                description="Should persist",
                context_type="test"
            )

            # Create second system (should load from disk)
            ms2 = MemorySystem(mock_knowledge_base, mock_ollama_client)

            assert ctx.id in ms2.contexts
            assert ms2.contexts[ctx.id].name == "Persistent Context"

    def test_get_context_for_query(self, memory_system, mock_knowledge_base):
        """Test getting context information for a query."""
        memory_system.switch_context("personal_brand")

        context_info = memory_system.get_context_for_query("How should I approach networking?")

        assert "current_context" in context_info
        assert "relevant_memories" in context_info
        assert "context_history" in context_info


class TestMemoryExtraction:
    """Tests for memory extraction from conversations."""

    @pytest.fixture
    def memory_system(self, temp_dir, mock_knowledge_base, mock_ollama_client):
        """Create a MemorySystem with mock extraction capability."""
        with patch('src.memory_system.settings') as mock_settings:
            mock_settings.conversation_history_dir = temp_dir / "conversations"

            ms = MemorySystem(
                knowledge_base=mock_knowledge_base,
                ollama_client=mock_ollama_client
            )
            yield ms

    def test_extract_memories_from_conversation(self, memory_system, mock_ollama_client):
        """Test extracting memories from conversation text."""
        # Mock the LLM response with valid JSON
        mock_ollama_client.generate.return_value = '''
        Here are the extracted memories:
        [
            {
                "type": "PERSONAL",
                "content": "User is interested in machine learning",
                "importance": 8,
                "tags": ["ml", "interests"]
            }
        ]
        '''

        conversation = """
        User: I've been learning a lot about machine learning lately.
        Assistant: That's great! What aspects interest you most?
        User: Mostly deep learning and neural networks.
        """

        memory_system.switch_context("self_improvement")
        memories = memory_system.extract_memories_from_conversation(conversation)

        assert isinstance(memories, list)
        # Note: The actual extraction depends on the mocked response being parsed correctly

    def test_extract_memories_empty_conversation(self, memory_system, mock_ollama_client):
        """Test extracting memories from empty conversation returns empty list."""
        mock_ollama_client.generate.return_value = "No important information found."

        memories = memory_system.extract_memories_from_conversation("")

        assert memories == []

    def test_extract_memories_invalid_json(self, memory_system, mock_ollama_client):
        """Test handling of invalid JSON in extraction response."""
        mock_ollama_client.generate.return_value = "This is not valid JSON"

        memories = memory_system.extract_memories_from_conversation("Some conversation")

        assert memories == []
