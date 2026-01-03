"""
Unit tests for the feedback system.
"""

import json
import pytest
import tempfile
from datetime import datetime
from pathlib import Path

from src.feedback_system import (
    FeedbackManager,
    FeedbackType,
    Feedback,
    Rating
)


class TestFeedbackType:
    """Test FeedbackType enum."""

    def test_feedback_type_values(self):
        assert FeedbackType.THUMBS_UP.value == "thumbs_up"
        assert FeedbackType.THUMBS_DOWN.value == "thumbs_down"
        assert FeedbackType.RATING_1_5.value == "rating_1_5"


class TestRating:
    """Test Rating enum."""

    def test_rating_values(self):
        assert Rating.TERRIBLE.value == 1
        assert Rating.POOR.value == 2
        assert Rating.OKAY.value == 3
        assert Rating.GOOD.value == 4
        assert Rating.EXCELLENT.value == 5


class TestFeedback:
    """Test Feedback dataclass."""

    def test_feedback_creation(self):
        feedback = Feedback(
            id="test_1",
            conversation_id="conv_1",
            message_index=0,
            feedback_type=FeedbackType.THUMBS_UP,
            rating=2
        )

        assert feedback.id == "test_1"
        assert feedback.conversation_id == "conv_1"
        assert feedback.message_index == 0
        assert feedback.feedback_type == FeedbackType.THUMBS_UP
        assert feedback.rating == 2
        assert feedback.text_feedback is None
        assert isinstance(feedback.timestamp, datetime)

    def test_feedback_with_text(self):
        feedback = Feedback(
            id="test_2",
            conversation_id="conv_1",
            message_index=1,
            feedback_type=FeedbackType.THUMBS_DOWN,
            rating=1,
            text_feedback="Response was too vague"
        )

        assert feedback.text_feedback == "Response was too vague"

    def test_is_positive_thumbs_up(self):
        feedback = Feedback(
            id="test_3",
            conversation_id="conv_1",
            message_index=0,
            feedback_type=FeedbackType.THUMBS_UP,
            rating=2
        )

        assert feedback.is_positive is True

    def test_is_positive_thumbs_down(self):
        feedback = Feedback(
            id="test_4",
            conversation_id="conv_1",
            message_index=0,
            feedback_type=FeedbackType.THUMBS_DOWN,
            rating=1
        )

        assert feedback.is_positive is False

    def test_is_positive_rating_high(self):
        feedback = Feedback(
            id="test_5",
            conversation_id="conv_1",
            message_index=0,
            feedback_type=FeedbackType.RATING_1_5,
            rating=4
        )

        assert feedback.is_positive is True

    def test_is_positive_rating_low(self):
        feedback = Feedback(
            id="test_6",
            conversation_id="conv_1",
            message_index=0,
            feedback_type=FeedbackType.RATING_1_5,
            rating=2
        )

        assert feedback.is_positive is False

    def test_normalized_score_thumbs_up(self):
        feedback = Feedback(
            id="test_7",
            conversation_id="conv_1",
            message_index=0,
            feedback_type=FeedbackType.THUMBS_UP,
            rating=2
        )

        assert feedback.normalized_score == 1.0

    def test_normalized_score_thumbs_down(self):
        feedback = Feedback(
            id="test_8",
            conversation_id="conv_1",
            message_index=0,
            feedback_type=FeedbackType.THUMBS_DOWN,
            rating=1
        )

        assert feedback.normalized_score == 0.0

    def test_normalized_score_rating(self):
        # Rating 1 -> 0.0
        fb1 = Feedback(
            id="test_9a", conversation_id="conv_1", message_index=0,
            feedback_type=FeedbackType.RATING_1_5, rating=1
        )
        assert fb1.normalized_score == 0.0

        # Rating 3 -> 0.5
        fb3 = Feedback(
            id="test_9b", conversation_id="conv_1", message_index=0,
            feedback_type=FeedbackType.RATING_1_5, rating=3
        )
        assert fb3.normalized_score == 0.5

        # Rating 5 -> 1.0
        fb5 = Feedback(
            id="test_9c", conversation_id="conv_1", message_index=0,
            feedback_type=FeedbackType.RATING_1_5, rating=5
        )
        assert fb5.normalized_score == 1.0

    def test_to_dict(self):
        feedback = Feedback(
            id="test_10",
            conversation_id="conv_1",
            message_index=0,
            feedback_type=FeedbackType.THUMBS_UP,
            rating=2,
            text_feedback="Great response!"
        )

        data = feedback.to_dict()

        assert data["id"] == "test_10"
        assert data["conversation_id"] == "conv_1"
        assert data["feedback_type"] == "thumbs_up"
        assert data["rating"] == 2
        assert data["text_feedback"] == "Great response!"
        assert "timestamp" in data

    def test_from_dict(self):
        data = {
            "id": "test_11",
            "conversation_id": "conv_1",
            "message_index": 0,
            "feedback_type": "thumbs_up",
            "rating": 2,
            "text_feedback": None,
            "timestamp": "2024-01-01T12:00:00",
            "user_query": "Test query",
            "assistant_response": "Test response"
        }

        feedback = Feedback.from_dict(data)

        assert feedback.id == "test_11"
        assert feedback.feedback_type == FeedbackType.THUMBS_UP
        assert feedback.rating == 2


class TestFeedbackManager:
    """Test FeedbackManager class."""

    @pytest.fixture
    def temp_feedback_dir(self):
        """Create a temporary directory for feedback storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def feedback_manager(self, temp_feedback_dir):
        """Create a FeedbackManager with temporary storage."""
        return FeedbackManager(feedback_dir=temp_feedback_dir)

    def test_initialization(self, feedback_manager):
        assert feedback_manager.feedback == {}

    def test_add_thumbs_up(self, feedback_manager):
        feedback = feedback_manager.add_thumbs_up(
            conversation_id="conv_1",
            message_index=0,
            user_query="What is Python?",
            assistant_response="Python is a programming language."
        )

        assert feedback.feedback_type == FeedbackType.THUMBS_UP
        assert feedback.rating == 2
        assert feedback.user_query == "What is Python?"
        assert feedback.assistant_response == "Python is a programming language."
        assert feedback.id in feedback_manager.feedback

    def test_add_thumbs_down(self, feedback_manager):
        feedback = feedback_manager.add_thumbs_down(
            conversation_id="conv_1",
            message_index=0,
            text_feedback="Too short answer",
            user_query="Explain quantum physics",
            assistant_response="It's about particles."
        )

        assert feedback.feedback_type == FeedbackType.THUMBS_DOWN
        assert feedback.rating == 1
        assert feedback.text_feedback == "Too short answer"
        assert feedback.id in feedback_manager.feedback

    def test_add_rating(self, feedback_manager):
        feedback = feedback_manager.add_rating(
            conversation_id="conv_1",
            message_index=0,
            rating=4,
            user_query="How do I sort a list?",
            assistant_response="Use the sort() method."
        )

        assert feedback.feedback_type == FeedbackType.RATING_1_5
        assert feedback.rating == 4
        assert feedback.id in feedback_manager.feedback

    def test_add_rating_invalid(self, feedback_manager):
        with pytest.raises(ValueError):
            feedback_manager.add_rating(
                conversation_id="conv_1",
                message_index=0,
                rating=6  # Invalid rating
            )

    def test_get_feedback_for_message(self, feedback_manager):
        # Add multiple feedbacks
        feedback_manager.add_thumbs_up("conv_1", 0)
        feedback_manager.add_rating("conv_1", 0, 5)
        feedback_manager.add_thumbs_up("conv_1", 1)  # Different message

        results = feedback_manager.get_feedback_for_message("conv_1", 0)

        assert len(results) == 2

    def test_get_feedback_for_conversation(self, feedback_manager):
        # Add feedbacks for different conversations
        feedback_manager.add_thumbs_up("conv_1", 0)
        feedback_manager.add_thumbs_up("conv_1", 1)
        feedback_manager.add_thumbs_up("conv_2", 0)  # Different conversation

        results = feedback_manager.get_feedback_for_conversation("conv_1")

        assert len(results) == 2

    def test_get_best_responses(self, feedback_manager):
        # Add mixed feedback
        feedback_manager.add_thumbs_up("conv_1", 0, user_query="Q1", assistant_response="A1")
        feedback_manager.add_thumbs_down("conv_1", 1, user_query="Q2", assistant_response="A2")
        feedback_manager.add_rating("conv_1", 2, 5, user_query="Q3", assistant_response="A3")

        best = feedback_manager.get_best_responses(limit=10)

        # Should only include positive feedback
        assert len(best) == 2
        assert all(fb.is_positive for fb in best)

    def test_get_worst_responses(self, feedback_manager):
        # Add mixed feedback
        feedback_manager.add_thumbs_up("conv_1", 0)
        feedback_manager.add_thumbs_down("conv_1", 1, user_query="Q2", assistant_response="A2")
        feedback_manager.add_rating("conv_1", 2, 2, user_query="Q3", assistant_response="A3")

        worst = feedback_manager.get_worst_responses(limit=10)

        # Should only include negative feedback
        assert len(worst) == 2
        assert all(not fb.is_positive for fb in worst)

    def test_get_responses_with_feedback_text(self, feedback_manager):
        feedback_manager.add_thumbs_up("conv_1", 0)
        feedback_manager.add_thumbs_down("conv_1", 1, text_feedback="Needs more detail")
        feedback_manager.add_thumbs_down("conv_1", 2, text_feedback="Incorrect information")

        with_text = feedback_manager.get_responses_with_feedback_text(limit=10)

        assert len(with_text) == 2

    def test_get_statistics_empty(self, feedback_manager):
        stats = feedback_manager.get_statistics()

        assert stats["total_feedback"] == 0
        assert stats["positive_count"] == 0
        assert stats["negative_count"] == 0
        assert stats["average_score"] == 0.0

    def test_get_statistics(self, feedback_manager):
        # Add various feedback
        feedback_manager.add_thumbs_up("conv_1", 0)
        feedback_manager.add_thumbs_up("conv_1", 1)
        feedback_manager.add_thumbs_down("conv_1", 2)
        feedback_manager.add_rating("conv_1", 3, 4)

        stats = feedback_manager.get_statistics()

        assert stats["total_feedback"] == 4
        assert stats["positive_count"] == 3  # 2 thumbs up + 1 rating of 4
        assert stats["negative_count"] == 1
        assert 0 < stats["positive_rate"] < 1
        assert 0 < stats["average_score"] < 1

    def test_get_few_shot_examples(self, feedback_manager):
        # Add highly-rated responses with context
        feedback_manager.add_rating(
            "conv_1", 0, 5,
            user_query="How do I reverse a string?",
            assistant_response="You can use slicing: string[::-1]"
        )
        feedback_manager.add_rating(
            "conv_1", 1, 5,
            user_query="What is a list comprehension?",
            assistant_response="A concise way to create lists: [x*2 for x in range(10)]"
        )
        # Add without context (should be excluded)
        feedback_manager.add_rating("conv_1", 2, 5)

        examples = feedback_manager.get_few_shot_examples(limit=5)

        assert len(examples) == 2
        assert all("user_query" in ex for ex in examples)
        assert all("assistant_response" in ex for ex in examples)

    def test_get_improvement_insights(self, feedback_manager):
        # Add feedback with text
        feedback_manager.add_thumbs_down("conv_1", 0, text_feedback="Too short")
        feedback_manager.add_thumbs_down("conv_1", 1, text_feedback="Missing examples")
        feedback_manager.add_thumbs_up("conv_1", 2, text_feedback="Great explanation!")

        insights = feedback_manager.get_improvement_insights()

        assert len(insights["common_issues"]) == 2
        assert len(insights["successful_patterns"]) == 1
        assert len(insights["improvement_suggestions"]) == 3

    def test_persistence(self, temp_feedback_dir):
        # Create manager and add feedback
        manager1 = FeedbackManager(feedback_dir=temp_feedback_dir)
        manager1.add_thumbs_up("conv_1", 0, user_query="Q1", assistant_response="A1")
        manager1.add_rating("conv_1", 1, 4, user_query="Q2", assistant_response="A2")

        # Create new manager with same directory - should load saved feedback
        manager2 = FeedbackManager(feedback_dir=temp_feedback_dir)

        assert len(manager2.feedback) == 2
        stats = manager2.get_statistics()
        assert stats["total_feedback"] == 2


class TestConversationManagerFeedbackIntegration:
    """Test ConversationManager integration with feedback system."""

    def test_add_message_returns_index(self):
        from src.conversation import ConversationManager
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test ConversationManager with temp dir
            from src.config import settings
            original_dir = settings.conversation_history_dir
            settings.conversation_history_dir = Path(tmpdir)

            try:
                manager = ConversationManager()
                idx1 = manager.add_message("user", "Hello")
                idx2 = manager.add_message("assistant", "Hi there!")
                idx3 = manager.add_message("user", "How are you?")

                assert idx1 == 0
                assert idx2 == 1
                assert idx3 == 2
            finally:
                settings.conversation_history_dir = original_dir

    def test_get_last_assistant_message(self):
        from src.conversation import ConversationManager
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            from src.config import settings
            original_dir = settings.conversation_history_dir
            settings.conversation_history_dir = Path(tmpdir)

            try:
                manager = ConversationManager()
                manager.add_message("user", "Hello")
                manager.add_message("assistant", "Hi there!")
                manager.add_message("user", "How are you?")
                manager.add_message("assistant", "I'm good, thanks!")

                result = manager.get_last_assistant_message()

                assert result is not None
                index, message = result
                assert index == 3
                assert message.content == "I'm good, thanks!"
            finally:
                settings.conversation_history_dir = original_dir

    def test_get_message_pair_for_feedback(self):
        from src.conversation import ConversationManager
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            from src.config import settings
            original_dir = settings.conversation_history_dir
            settings.conversation_history_dir = Path(tmpdir)

            try:
                manager = ConversationManager()
                manager.add_message("user", "What is Python?")
                manager.add_message("assistant", "Python is a programming language.")

                pair = manager.get_message_pair_for_feedback()

                assert pair is not None
                assert pair["user_index"] == 0
                assert pair["user_content"] == "What is Python?"
                assert pair["assistant_index"] == 1
                assert pair["assistant_content"] == "Python is a programming language."
            finally:
                settings.conversation_history_dir = original_dir

    def test_get_message_pair_no_assistant(self):
        from src.conversation import ConversationManager
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            from src.config import settings
            original_dir = settings.conversation_history_dir
            settings.conversation_history_dir = Path(tmpdir)

            try:
                manager = ConversationManager()
                manager.add_message("user", "Hello")

                pair = manager.get_message_pair_for_feedback()

                assert pair is None
            finally:
                settings.conversation_history_dir = original_dir
