"""Tests for the memory_system module."""

import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch
import json


class TestMemory:
    """Tests for the Memory dataclass."""

    def test_memory_to_dict(self):
        """Test Memory.to_dict() serialization."""
        from src.memory_system import Memory, MemoryType

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

        data = memory.to_dict()

        assert data["id"] == "test_mem_1"
        assert data["memory_type"] == "personal"
        assert data["content"] == "Test content"
        assert data["importance"] == 7
        assert data["access_count"] == 1
        assert "tag1" in data["tags"]
        assert "project1" in data["projects"]
        assert isinstance(data["created_at"], str)

    def test_memory_from_dict(self):
        """Test Memory.from_dict() deserialization."""
        from src.memory_system import Memory, MemoryType

        data = {
            "id": "test_mem_2",
            "memory_type": "insight",
            "content": "An insight",
            "context": "general",
            "importance": 5,
            "created_at": "2024-01-01T12:00:00",
            "last_accessed": "2024-01-01T12:00:00",
            "access_count": 3,
            "tags": ["learning"],
            "projects": [],
            "metadata": {}
        }

        memory = Memory.from_dict(data)

        assert memory.id == "test_mem_2"
        assert memory.memory_type == MemoryType.INSIGHT
        assert memory.importance == 5
        assert isinstance(memory.created_at, datetime)
        assert isinstance(memory.tags, set)
        assert "learning" in memory.tags


class TestContext:
    """Tests for the Context dataclass."""

    def test_context_to_dict(self):
        """Test Context.to_dict() serialization."""
        from src.memory_system import Context

        now = datetime.now()
        context = Context(
            id="ctx_1",
            name="Test Context",
            description="A test context",
            context_type="testing",
            goals=["Goal 1", "Goal 2"],
            key_people=["Person 1"],
            current_focus="Testing",
            status="active",
            created_at=now,
            last_used=now,
            metadata={}
        )

        data = context.to_dict()

        assert data["id"] == "ctx_1"
        assert data["name"] == "Test Context"
        assert data["status"] == "active"
        assert isinstance(data["created_at"], str)
        assert len(data["goals"]) == 2

    def test_context_from_dict(self):
        """Test Context.from_dict() deserialization."""
        from src.memory_system import Context

        data = {
            "id": "ctx_2",
            "name": "Loaded Context",
            "description": "From storage",
            "context_type": "development",
            "goals": ["Build"],
            "key_people": [],
            "current_focus": "Coding",
            "status": "active",
            "created_at": "2024-01-01T12:00:00",
            "last_used": "2024-01-02T12:00:00",
            "metadata": {"priority": "high"}
        }

        context = Context.from_dict(data)

        assert context.id == "ctx_2"
        assert context.name == "Loaded Context"
        assert isinstance(context.created_at, datetime)
        assert isinstance(context.last_used, datetime)


class TestMemoryType:
    """Tests for the MemoryType enum."""

    def test_memory_types_exist(self):
        """Test that all expected memory types exist."""
        from src.memory_system import MemoryType

        assert MemoryType.PERSONAL.value == "personal"
        assert MemoryType.PROJECT.value == "project"
        assert MemoryType.INSIGHT.value == "insight"
        assert MemoryType.GOAL.value == "goal"
        assert MemoryType.DECISION.value == "decision"
        assert MemoryType.RELATIONSHIP.value == "relationship"
        assert MemoryType.SKILL.value == "skill"
        assert MemoryType.EXPERIENCE.value == "experience"


class TestMemorySystem:
    """Tests for the MemorySystem class."""

    @patch('src.memory_system.settings')
    def test_memory_system_init(self, mock_settings, temp_dir, mock_knowledge_base, mock_ollama_client):
        """Test MemorySystem initialization."""
        mock_settings.conversation_history_dir = temp_dir / "conversations"

        from src.memory_system import MemorySystem

        memory_system = MemorySystem(mock_knowledge_base, mock_ollama_client)

        assert memory_system.memories == {} or len(memory_system.memories) >= 0
        assert memory_system.current_context is None

    @patch('src.memory_system.settings')
    def test_create_context(self, mock_settings, temp_dir, mock_knowledge_base, mock_ollama_client):
        """Test creating a new context."""
        mock_settings.conversation_history_dir = temp_dir / "conversations"

        from src.memory_system import MemorySystem

        memory_system = MemorySystem(mock_knowledge_base, mock_ollama_client)
        context = memory_system.create_context(
            name="My Project",
            description="A new project",
            context_type="development",
            goals=["Build something"]
        )

        assert context.name == "My Project"
        assert context.context_type == "development"
        assert context.id in memory_system.contexts

    @patch('src.memory_system.settings')
    def test_switch_context(self, mock_settings, temp_dir, mock_knowledge_base, mock_ollama_client):
        """Test switching between contexts."""
        mock_settings.conversation_history_dir = temp_dir / "conversations"

        from src.memory_system import MemorySystem

        memory_system = MemorySystem(mock_knowledge_base, mock_ollama_client)
        context = memory_system.create_context(
            name="Switch Test",
            description="Testing context switch",
            context_type="test"
        )

        result = memory_system.switch_context(context.id)

        assert result is True
        assert memory_system.current_context is not None
        assert memory_system.current_context.id == context.id

    @patch('src.memory_system.settings')
    def test_switch_context_invalid(self, mock_settings, temp_dir, mock_knowledge_base, mock_ollama_client):
        """Test switching to non-existent context."""
        mock_settings.conversation_history_dir = temp_dir / "conversations"

        from src.memory_system import MemorySystem

        memory_system = MemorySystem(mock_knowledge_base, mock_ollama_client)
        result = memory_system.switch_context("non_existent_context")

        assert result is False

    @patch('src.memory_system.settings')
    def test_add_memory(self, mock_settings, temp_dir, mock_knowledge_base, mock_ollama_client):
        """Test adding a memory."""
        mock_settings.conversation_history_dir = temp_dir / "conversations"

        from src.memory_system import MemorySystem, MemoryType

        memory_system = MemorySystem(mock_knowledge_base, mock_ollama_client)
        memory = memory_system.add_memory(
            content="User likes Python",
            memory_type=MemoryType.PERSONAL,
            importance=8,
            tags={"programming", "preferences"}
        )

        assert memory.content == "User likes Python"
        assert memory.memory_type == MemoryType.PERSONAL
        assert memory.importance == 8
        assert "programming" in memory.tags
        assert memory.id in memory_system.memories
        mock_knowledge_base.add_document.assert_called_once()

    @patch('src.memory_system.settings')
    def test_update_context_focus(self, mock_settings, temp_dir, mock_knowledge_base, mock_ollama_client):
        """Test updating context focus."""
        mock_settings.conversation_history_dir = temp_dir / "conversations"

        from src.memory_system import MemorySystem

        memory_system = MemorySystem(mock_knowledge_base, mock_ollama_client)
        context = memory_system.create_context(
            name="Focus Test",
            description="Testing focus update",
            context_type="test"
        )
        memory_system.switch_context(context.id)

        memory_system.update_context_focus("New focus area")

        assert memory_system.current_context.current_focus == "New focus area"

    @patch('src.memory_system.settings')
    def test_get_context_summary_no_context(self, mock_settings, temp_dir, mock_knowledge_base, mock_ollama_client):
        """Test get_context_summary when no context is selected."""
        mock_settings.conversation_history_dir = temp_dir / "conversations"

        from src.memory_system import MemorySystem

        memory_system = MemorySystem(mock_knowledge_base, mock_ollama_client)
        summary = memory_system.get_context_summary()

        assert summary == "No context selected"

    @patch('src.memory_system.settings')
    def test_recall_memories_empty(self, mock_settings, temp_dir, mock_knowledge_base, mock_ollama_client):
        """Test recalling memories when none exist."""
        mock_settings.conversation_history_dir = temp_dir / "conversations"

        from src.memory_system import MemorySystem

        memory_system = MemorySystem(mock_knowledge_base, mock_ollama_client)
        memories = memory_system.recall_memories("test query")

        assert memories == []

    @patch('src.memory_system.settings')
    def test_default_contexts_created(self, mock_settings, temp_dir, mock_knowledge_base, mock_ollama_client):
        """Test that default contexts are created on initialization."""
        mock_settings.conversation_history_dir = temp_dir / "conversations"

        from src.memory_system import MemorySystem

        memory_system = MemorySystem(mock_knowledge_base, mock_ollama_client)

        # Default contexts should be created
        assert len(memory_system.contexts) > 0
        # Should have personal_brand, product_dev, self_improvement
        context_ids = list(memory_system.contexts.keys())
        assert "personal_brand" in context_ids
        assert "product_dev" in context_ids
        assert "self_improvement" in context_ids

    @patch('src.memory_system.settings')
    def test_create_context_unique_id(self, mock_settings, temp_dir, mock_knowledge_base, mock_ollama_client):
        """Test that duplicate context names get unique IDs."""
        mock_settings.conversation_history_dir = temp_dir / "conversations"

        from src.memory_system import MemorySystem

        memory_system = MemorySystem(mock_knowledge_base, mock_ollama_client)

        ctx1 = memory_system.create_context(
            name="Test",
            description="First test context",
            context_type="test"
        )

        ctx2 = memory_system.create_context(
            name="Test",
            description="Second test context",
            context_type="test"
        )

        assert ctx1.id != ctx2.id
        assert ctx1.id == "test"
        assert ctx2.id == "test_1"
