"""Tests for the conversation module."""

import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch
import json


class TestMessage:
    """Tests for the Message dataclass."""

    def test_message_to_dict(self):
        """Test Message.to_dict() serialization."""
        from src.conversation import Message

        now = datetime.now()
        message = Message(
            role="user",
            content="Hello, world!",
            timestamp=now,
            metadata={"source": "test"}
        )

        data = message.to_dict()

        assert data["role"] == "user"
        assert data["content"] == "Hello, world!"
        assert isinstance(data["timestamp"], str)
        assert data["metadata"]["source"] == "test"

    def test_message_from_dict(self):
        """Test Message.from_dict() deserialization."""
        from src.conversation import Message

        data = {
            "role": "assistant",
            "content": "How can I help?",
            "timestamp": "2024-01-01T12:00:00",
            "metadata": {}
        }

        message = Message.from_dict(data)

        assert message.role == "assistant"
        assert message.content == "How can I help?"
        assert isinstance(message.timestamp, datetime)

    def test_message_default_metadata(self):
        """Test Message with None metadata."""
        from src.conversation import Message

        message = Message(
            role="system",
            content="System message",
            timestamp=datetime.now(),
            metadata=None
        )

        data = message.to_dict()
        assert data["metadata"] is None


class TestConversationManager:
    """Tests for the ConversationManager class."""

    @patch('src.conversation.settings')
    def test_init_creates_directory(self, mock_settings, temp_dir):
        """Test that initialization creates the conversations directory."""
        mock_settings.conversation_history_dir = temp_dir / "conversations"

        from src.conversation import ConversationManager

        manager = ConversationManager()

        assert mock_settings.conversation_history_dir.exists()
        assert manager.current_conversation == []
        assert manager.conversation_id is not None

    @patch('src.conversation.settings')
    def test_add_message(self, mock_settings, temp_dir):
        """Test adding a message to the conversation."""
        mock_settings.conversation_history_dir = temp_dir / "conversations"

        from src.conversation import ConversationManager

        manager = ConversationManager()
        manager.add_message("user", "Hello!")

        assert len(manager.current_conversation) == 1
        assert manager.current_conversation[0].role == "user"
        assert manager.current_conversation[0].content == "Hello!"

    @patch('src.conversation.settings')
    def test_add_message_with_metadata(self, mock_settings, temp_dir):
        """Test adding a message with metadata."""
        mock_settings.conversation_history_dir = temp_dir / "conversations"

        from src.conversation import ConversationManager

        manager = ConversationManager()
        manager.add_message("assistant", "Response", metadata={"model": "llama3"})

        assert manager.current_conversation[0].metadata["model"] == "llama3"

    @patch('src.conversation.settings')
    def test_get_conversation_context(self, mock_settings, temp_dir):
        """Test getting conversation context as formatted string."""
        mock_settings.conversation_history_dir = temp_dir / "conversations"

        from src.conversation import ConversationManager

        manager = ConversationManager()
        manager.add_message("user", "What is Python?")
        manager.add_message("assistant", "Python is a programming language.")
        manager.add_message("user", "Tell me more.")

        context = manager.get_conversation_context(max_messages=10)

        assert "User: What is Python?" in context
        assert "Assistant: Python is a programming language." in context
        assert "User: Tell me more." in context

    @patch('src.conversation.settings')
    def test_get_conversation_context_excludes_system(self, mock_settings, temp_dir):
        """Test that system messages are excluded from context."""
        mock_settings.conversation_history_dir = temp_dir / "conversations"

        from src.conversation import ConversationManager, Message

        manager = ConversationManager()
        manager.current_conversation.append(
            Message(role="system", content="System prompt", timestamp=datetime.now())
        )
        manager.add_message("user", "Hello")

        context = manager.get_conversation_context()

        assert "System" not in context
        assert "User: Hello" in context

    @patch('src.conversation.settings')
    def test_get_messages_for_llm(self, mock_settings, temp_dir):
        """Test getting messages formatted for LLM consumption."""
        mock_settings.conversation_history_dir = temp_dir / "conversations"

        from src.conversation import ConversationManager

        manager = ConversationManager()
        manager.add_message("user", "Question?")
        manager.add_message("assistant", "Answer!")

        messages = manager.get_messages_for_llm()

        assert len(messages) == 2
        assert messages[0] == {"role": "user", "content": "Question?"}
        assert messages[1] == {"role": "assistant", "content": "Answer!"}

    @patch('src.conversation.settings')
    def test_get_messages_for_llm_max_limit(self, mock_settings, temp_dir):
        """Test that max_messages limit is respected."""
        mock_settings.conversation_history_dir = temp_dir / "conversations"

        from src.conversation import ConversationManager

        manager = ConversationManager()
        for i in range(10):
            manager.add_message("user", f"Message {i}")

        messages = manager.get_messages_for_llm(max_messages=5)

        assert len(messages) == 5
        # Should get the last 5 messages
        assert messages[0]["content"] == "Message 5"
        assert messages[4]["content"] == "Message 9"

    @patch('src.conversation.settings')
    def test_start_new_conversation(self, mock_settings, temp_dir):
        """Test starting a new conversation."""
        mock_settings.conversation_history_dir = temp_dir / "conversations"

        from src.conversation import ConversationManager

        manager = ConversationManager()
        old_id = manager.conversation_id
        manager.add_message("user", "Hello")

        manager.start_new_conversation()

        assert manager.current_conversation == []
        assert manager.conversation_id != old_id

    @patch('src.conversation.settings')
    def test_save_conversation(self, mock_settings, temp_dir):
        """Test that conversation is saved to disk."""
        mock_settings.conversation_history_dir = temp_dir / "conversations"

        from src.conversation import ConversationManager

        manager = ConversationManager()
        manager.add_message("user", "Save test message")

        # Check file was created
        filepath = mock_settings.conversation_history_dir / f"{manager.conversation_id}.json"
        assert filepath.exists()

        # Check content
        with open(filepath, 'r') as f:
            data = json.load(f)

        assert data["id"] == manager.conversation_id
        assert len(data["messages"]) == 1
        assert data["messages"][0]["content"] == "Save test message"

    @patch('src.conversation.settings')
    def test_load_conversation(self, mock_settings, temp_dir, sample_conversation_data):
        """Test loading a conversation from disk."""
        mock_settings.conversation_history_dir = temp_dir / "conversations"
        mock_settings.conversation_history_dir.mkdir(parents=True, exist_ok=True)

        # Save sample conversation
        filepath = mock_settings.conversation_history_dir / f"{sample_conversation_data['id']}.json"
        with open(filepath, 'w') as f:
            json.dump(sample_conversation_data, f)

        from src.conversation import ConversationManager

        manager = ConversationManager()
        manager.load_conversation(sample_conversation_data['id'])

        assert manager.conversation_id == sample_conversation_data['id']
        assert len(manager.current_conversation) == 2
        assert manager.current_conversation[0].content == "Hello, how are you?"

    @patch('src.conversation.settings')
    def test_load_conversation_not_found(self, mock_settings, temp_dir):
        """Test loading a non-existent conversation raises error."""
        mock_settings.conversation_history_dir = temp_dir / "conversations"

        from src.conversation import ConversationManager

        manager = ConversationManager()

        with pytest.raises(Exception):
            manager.load_conversation("non_existent_id")

    @patch('src.conversation.settings')
    def test_list_conversations(self, mock_settings, temp_dir):
        """Test listing all saved conversations."""
        mock_settings.conversation_history_dir = temp_dir / "conversations"
        mock_settings.conversation_history_dir.mkdir(parents=True, exist_ok=True)

        # Create some conversation files
        for i, conv_id in enumerate(["conv_001", "conv_002"]):
            data = {
                "id": conv_id,
                "started_at": f"2024-01-0{i+1}T12:00:00",
                "messages": [
                    {"role": "user", "content": f"Message in {conv_id}", "timestamp": f"2024-01-0{i+1}T12:00:00", "metadata": {}}
                ]
            }
            filepath = mock_settings.conversation_history_dir / f"{conv_id}.json"
            with open(filepath, 'w') as f:
                json.dump(data, f)

        from src.conversation import ConversationManager

        manager = ConversationManager()
        conversations = manager.list_conversations()

        assert len(conversations) == 2
        assert any(c["id"] == "conv_001" for c in conversations)
        assert any(c["id"] == "conv_002" for c in conversations)

    @patch('src.conversation.settings')
    def test_list_conversations_empty(self, mock_settings, temp_dir):
        """Test listing conversations when none exist."""
        mock_settings.conversation_history_dir = temp_dir / "conversations"

        from src.conversation import ConversationManager

        manager = ConversationManager()
        conversations = manager.list_conversations()

        # Should only have the current conversation (empty) or none
        assert isinstance(conversations, list)

    @patch('src.conversation.settings')
    def test_add_message_returns_index(self, mock_settings, temp_dir):
        """Test that add_message returns the message index."""
        mock_settings.conversation_history_dir = temp_dir / "conversations"

        from src.conversation import ConversationManager

        manager = ConversationManager()
        index0 = manager.add_message("user", "First message")
        index1 = manager.add_message("assistant", "Second message")
        index2 = manager.add_message("user", "Third message")

        assert index0 == 0
        assert index1 == 1
        assert index2 == 2

    @patch('src.conversation.settings')
    def test_get_last_assistant_message(self, mock_settings, temp_dir):
        """Test getting the last assistant message."""
        mock_settings.conversation_history_dir = temp_dir / "conversations"

        from src.conversation import ConversationManager

        manager = ConversationManager()
        manager.add_message("user", "Hello")
        manager.add_message("assistant", "Hi there!")
        manager.add_message("user", "How are you?")
        manager.add_message("assistant", "I'm doing well!")

        result = manager.get_last_assistant_message()

        assert result is not None
        index, message = result
        assert index == 3
        assert message.content == "I'm doing well!"
        assert message.role == "assistant"

    @patch('src.conversation.settings')
    def test_get_last_assistant_message_none(self, mock_settings, temp_dir):
        """Test get_last_assistant_message when no assistant messages exist."""
        mock_settings.conversation_history_dir = temp_dir / "conversations"

        from src.conversation import ConversationManager

        manager = ConversationManager()
        manager.add_message("user", "Hello")
        manager.add_message("user", "Anyone there?")

        result = manager.get_last_assistant_message()

        assert result is None

    @patch('src.conversation.settings')
    def test_get_last_user_message(self, mock_settings, temp_dir):
        """Test getting the last user message."""
        mock_settings.conversation_history_dir = temp_dir / "conversations"

        from src.conversation import ConversationManager

        manager = ConversationManager()
        manager.add_message("user", "First question")
        manager.add_message("assistant", "First answer")
        manager.add_message("user", "Second question")

        result = manager.get_last_user_message()

        assert result is not None
        index, message = result
        assert index == 2
        assert message.content == "Second question"
        assert message.role == "user"

    @patch('src.conversation.settings')
    def test_get_last_user_message_none(self, mock_settings, temp_dir):
        """Test get_last_user_message when no user messages exist."""
        mock_settings.conversation_history_dir = temp_dir / "conversations"

        from src.conversation import ConversationManager

        manager = ConversationManager()
        # Only system messages
        from src.conversation import Message
        manager.current_conversation.append(
            Message(role="system", content="System prompt", timestamp=datetime.now())
        )

        result = manager.get_last_user_message()

        assert result is None

    @patch('src.conversation.settings')
    def test_get_message_pair_for_feedback(self, mock_settings, temp_dir):
        """Test getting user-assistant message pair for feedback."""
        mock_settings.conversation_history_dir = temp_dir / "conversations"

        from src.conversation import ConversationManager

        manager = ConversationManager()
        manager.add_message("user", "What is Python?")
        manager.add_message("assistant", "Python is a programming language.")
        manager.add_message("user", "Tell me more.")
        manager.add_message("assistant", "It's widely used for web development.")

        result = manager.get_message_pair_for_feedback()

        assert result is not None
        assert result["user_index"] == 2
        assert result["user_content"] == "Tell me more."
        assert result["assistant_index"] == 3
        assert result["assistant_content"] == "It's widely used for web development."

    @patch('src.conversation.settings')
    def test_get_message_pair_for_feedback_no_assistant(self, mock_settings, temp_dir):
        """Test get_message_pair_for_feedback when no assistant messages exist."""
        mock_settings.conversation_history_dir = temp_dir / "conversations"

        from src.conversation import ConversationManager

        manager = ConversationManager()
        manager.add_message("user", "Hello")

        result = manager.get_message_pair_for_feedback()

        assert result is None

    @patch('src.conversation.settings')
    def test_get_message_pair_for_feedback_no_preceding_user(self, mock_settings, temp_dir):
        """Test get_message_pair_for_feedback when assistant has no preceding user message."""
        mock_settings.conversation_history_dir = temp_dir / "conversations"

        from src.conversation import ConversationManager, Message

        manager = ConversationManager()
        # Add only an assistant message
        manager.current_conversation.append(
            Message(role="assistant", content="Hello!", timestamp=datetime.now())
        )
        manager._save_conversation()

        result = manager.get_message_pair_for_feedback()

        assert result is not None
        assert result["user_index"] is None
        assert result["user_content"] is None
        assert result["assistant_index"] == 0
        assert result["assistant_content"] == "Hello!"
