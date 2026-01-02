"""Tests for conversation.py module."""

import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock
import json

from src.conversation import Message, ConversationManager


class TestMessage:
    """Tests for Message dataclass."""

    def test_message_creation(self):
        """Test creating a Message instance."""
        now = datetime.now()
        message = Message(
            role="user",
            content="Hello, how are you?",
            timestamp=now,
            metadata={"source": "terminal"}
        )

        assert message.role == "user"
        assert message.content == "Hello, how are you?"
        assert message.timestamp == now
        assert message.metadata["source"] == "terminal"

    def test_message_default_metadata(self):
        """Test Message with default metadata."""
        message = Message(
            role="assistant",
            content="I'm doing well!",
            timestamp=datetime.now()
        )

        assert message.metadata is None

    def test_message_to_dict(self):
        """Test Message serialization to dict."""
        now = datetime.now()
        message = Message(
            role="user",
            content="Test message",
            timestamp=now,
            metadata={"key": "value"}
        )

        data = message.to_dict()

        assert data["role"] == "user"
        assert data["content"] == "Test message"
        assert isinstance(data["timestamp"], str)
        assert data["metadata"]["key"] == "value"

    def test_message_from_dict(self, sample_message_data):
        """Test Message deserialization from dict."""
        message = Message.from_dict(sample_message_data)

        assert message.role == sample_message_data["role"]
        assert message.content == sample_message_data["content"]
        assert isinstance(message.timestamp, datetime)
        assert message.metadata["source"] == "test"

    def test_message_roundtrip(self):
        """Test Message can be serialized and deserialized correctly."""
        now = datetime.now()
        original = Message(
            role="system",
            content="You are a helpful assistant.",
            timestamp=now,
            metadata={"type": "system_prompt"}
        )

        data = original.to_dict()
        restored = Message.from_dict(data)

        assert restored.role == original.role
        assert restored.content == original.content
        assert restored.metadata == original.metadata

    def test_message_roles(self):
        """Test different message roles."""
        roles = ["user", "assistant", "system"]

        for role in roles:
            message = Message(
                role=role,
                content="Test",
                timestamp=datetime.now()
            )
            assert message.role == role


class TestConversationManager:
    """Tests for ConversationManager class."""

    @pytest.fixture
    def conversation_manager(self, temp_dir):
        """Create a ConversationManager with temporary storage."""
        with patch('src.conversation.settings') as mock_settings:
            mock_settings.conversation_history_dir = temp_dir / "conversations"

            manager = ConversationManager()
            yield manager

    def test_initialization(self, conversation_manager, temp_dir):
        """Test ConversationManager initialization."""
        assert conversation_manager.current_conversation == []
        assert conversation_manager.conversation_id is not None
        assert (temp_dir / "conversations").exists()

    def test_add_message(self, conversation_manager):
        """Test adding a message to conversation."""
        conversation_manager.add_message(
            role="user",
            content="Hello!"
        )

        assert len(conversation_manager.current_conversation) == 1
        assert conversation_manager.current_conversation[0].role == "user"
        assert conversation_manager.current_conversation[0].content == "Hello!"

    def test_add_message_with_metadata(self, conversation_manager):
        """Test adding a message with metadata."""
        conversation_manager.add_message(
            role="assistant",
            content="How can I help?",
            metadata={"model": "llama3", "temperature": 0.7}
        )

        message = conversation_manager.current_conversation[0]
        assert message.metadata["model"] == "llama3"
        assert message.metadata["temperature"] == 0.7

    def test_add_multiple_messages(self, conversation_manager):
        """Test adding multiple messages."""
        conversation_manager.add_message("user", "First message")
        conversation_manager.add_message("assistant", "First response")
        conversation_manager.add_message("user", "Second message")
        conversation_manager.add_message("assistant", "Second response")

        assert len(conversation_manager.current_conversation) == 4

    def test_get_conversation_context(self, conversation_manager):
        """Test getting conversation context as string."""
        conversation_manager.add_message("user", "Hello!")
        conversation_manager.add_message("assistant", "Hi there!")
        conversation_manager.add_message("user", "How are you?")

        context = conversation_manager.get_conversation_context()

        assert "User: Hello!" in context
        assert "Assistant: Hi there!" in context
        assert "User: How are you?" in context

    def test_get_conversation_context_max_messages(self, conversation_manager):
        """Test conversation context respects max_messages limit."""
        # Add more messages than the limit
        for i in range(15):
            conversation_manager.add_message("user", f"Message {i}")
            conversation_manager.add_message("assistant", f"Response {i}")

        context = conversation_manager.get_conversation_context(max_messages=5)

        # Should only have the last 5 messages
        assert "Message 14" in context or "Response 14" in context
        assert "Message 0" not in context

    def test_get_conversation_context_excludes_system(self, conversation_manager):
        """Test that system messages are excluded from context."""
        conversation_manager.add_message("system", "System prompt")
        conversation_manager.add_message("user", "Hello!")
        conversation_manager.add_message("assistant", "Hi!")

        context = conversation_manager.get_conversation_context()

        assert "System prompt" not in context
        assert "Hello!" in context

    def test_get_messages_for_llm(self, conversation_manager):
        """Test getting messages formatted for LLM."""
        conversation_manager.add_message("user", "Hello!")
        conversation_manager.add_message("assistant", "Hi there!")

        messages = conversation_manager.get_messages_for_llm()

        assert len(messages) == 2
        assert messages[0] == {"role": "user", "content": "Hello!"}
        assert messages[1] == {"role": "assistant", "content": "Hi there!"}

    def test_get_messages_for_llm_max_messages(self, conversation_manager):
        """Test LLM messages respect max_messages limit."""
        for i in range(10):
            conversation_manager.add_message("user", f"Message {i}")

        messages = conversation_manager.get_messages_for_llm(max_messages=3)

        assert len(messages) == 3
        assert messages[-1]["content"] == "Message 9"

    def test_start_new_conversation(self, conversation_manager):
        """Test starting a new conversation."""
        conversation_manager.add_message("user", "Old message")
        old_id = conversation_manager.conversation_id

        conversation_manager.start_new_conversation()

        assert len(conversation_manager.current_conversation) == 0
        assert conversation_manager.conversation_id != old_id

    def test_save_conversation(self, conversation_manager, temp_dir):
        """Test that conversations are saved to disk."""
        conversation_manager.add_message("user", "Test message")

        # Check file was created
        filepath = temp_dir / "conversations" / f"{conversation_manager.conversation_id}.json"
        assert filepath.exists()

        # Verify content
        with open(filepath, 'r') as f:
            data = json.load(f)

        assert data["id"] == conversation_manager.conversation_id
        assert len(data["messages"]) == 1
        assert data["messages"][0]["content"] == "Test message"

    def test_load_conversation(self, conversation_manager, temp_dir):
        """Test loading a conversation from disk."""
        # Add some messages and get the ID
        conversation_manager.add_message("user", "Hello!")
        conversation_manager.add_message("assistant", "Hi!")
        conversation_id = conversation_manager.conversation_id

        # Start a new conversation
        conversation_manager.start_new_conversation()
        assert len(conversation_manager.current_conversation) == 0

        # Load the old conversation
        conversation_manager.load_conversation(conversation_id)

        assert conversation_manager.conversation_id == conversation_id
        assert len(conversation_manager.current_conversation) == 2
        assert conversation_manager.current_conversation[0].content == "Hello!"

    def test_load_conversation_nonexistent(self, conversation_manager):
        """Test loading a non-existent conversation raises error."""
        with pytest.raises(Exception):
            conversation_manager.load_conversation("nonexistent_id")

    def test_list_conversations(self, conversation_manager, temp_dir):
        """Test listing all saved conversations."""
        # Create multiple conversations
        conversation_manager.add_message("user", "First conversation")
        first_id = conversation_manager.conversation_id

        conversation_manager.start_new_conversation()
        conversation_manager.add_message("user", "Second conversation")
        second_id = conversation_manager.conversation_id

        conversations = conversation_manager.list_conversations()

        assert len(conversations) >= 2
        ids = [c["id"] for c in conversations]
        assert first_id in ids
        assert second_id in ids

    def test_list_conversations_includes_metadata(self, conversation_manager):
        """Test that list_conversations includes expected metadata."""
        conversation_manager.add_message("user", "Hello world!")
        conversation_manager.add_message("assistant", "Hi!")

        conversations = conversation_manager.list_conversations()

        # Find our conversation
        our_conv = next(
            c for c in conversations
            if c["id"] == conversation_manager.conversation_id
        )

        assert "started_at" in our_conv
        assert "message_count" in our_conv
        assert our_conv["message_count"] == 2
        assert "first_message" in our_conv
        assert "Hello" in our_conv["first_message"]

    def test_list_conversations_sorted_by_recency(self, conversation_manager):
        """Test that conversations are listed most recent first."""
        # Create first conversation
        conversation_manager.add_message("user", "First")
        first_id = conversation_manager.conversation_id

        # Create second conversation
        conversation_manager.start_new_conversation()
        conversation_manager.add_message("user", "Second")
        second_id = conversation_manager.conversation_id

        conversations = conversation_manager.list_conversations()

        # Most recent should be first
        assert conversations[0]["id"] == second_id

    def test_empty_conversation_not_saved(self, conversation_manager, temp_dir):
        """Test that empty conversations don't create files on start_new."""
        initial_files = list((temp_dir / "conversations").glob("*.json"))

        conversation_manager.start_new_conversation()
        # Don't add any messages

        # Starting another new conversation shouldn't save the empty one
        conversation_manager.start_new_conversation()

        # Only the initial conversation (if any) should have files
        final_files = list((temp_dir / "conversations").glob("*.json"))
        assert len(final_files) == len(initial_files)

    def test_conversation_preserves_message_order(self, conversation_manager):
        """Test that messages maintain their order."""
        messages = ["First", "Second", "Third", "Fourth", "Fifth"]

        for msg in messages:
            conversation_manager.add_message("user", msg)

        for i, msg in enumerate(messages):
            assert conversation_manager.current_conversation[i].content == msg

    def test_conversation_id_format(self, conversation_manager):
        """Test that conversation IDs follow expected format."""
        # Should be in YYYYMMDD_HHMMSS format
        conv_id = conversation_manager.conversation_id

        assert len(conv_id) == 15  # YYYYMMDD_HHMMSS
        assert "_" in conv_id
        assert conv_id.replace("_", "").isdigit()
