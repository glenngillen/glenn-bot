"""Tests for context_detector.py - Context auto-detection and switching."""

import pytest
from unittest.mock import MagicMock, patch


class TestContextDetector:
    """Test cases for ContextDetector class."""

    @pytest.fixture
    def context_detector(self, mock_ollama_client):
        """Create a ContextDetector instance for testing."""
        from src.context_detector import ContextDetector
        return ContextDetector(
            ollama_client=mock_ollama_client,
            auto_switch_threshold=0.8,
            prompt_threshold=0.6
        )

    def test_init_with_defaults(self, mock_ollama_client):
        """Test ContextDetector initialization with default values."""
        from src.context_detector import ContextDetector
        detector = ContextDetector(mock_ollama_client)

        assert detector.auto_switch_threshold == 0.8
        assert detector.prompt_threshold == 0.6
        assert detector.ollama_client == mock_ollama_client

    def test_init_with_custom_thresholds(self, mock_ollama_client):
        """Test ContextDetector initialization with custom thresholds."""
        from src.context_detector import ContextDetector
        detector = ContextDetector(
            mock_ollama_client,
            auto_switch_threshold=0.9,
            prompt_threshold=0.7
        )

        assert detector.auto_switch_threshold == 0.9
        assert detector.prompt_threshold == 0.7

    def test_keyword_classify_work_context(self, context_detector):
        """Test keyword classification for work-related messages."""
        message = "I need to finish the project report before the deadline"
        scores = context_detector._keyword_classify(message)

        assert "work" in scores
        assert "personal" in scores
        assert "creative" in scores
        assert "reflection" in scores
        # Work should have highest or significant score
        assert scores["work"] > 0

    def test_keyword_classify_personal_context(self, context_detector):
        """Test keyword classification for personal messages."""
        message = "Planning a family dinner for my friend's birthday"
        scores = context_detector._keyword_classify(message)

        assert scores["personal"] > 0

    def test_keyword_classify_creative_context(self, context_detector):
        """Test keyword classification for creative messages."""
        message = "I want to write a story and brainstorm some ideas"
        scores = context_detector._keyword_classify(message)

        assert scores["creative"] > 0

    def test_keyword_classify_reflection_context(self, context_detector):
        """Test keyword classification for reflection messages."""
        message = "I need to reflect on my goals and think about my purpose"
        scores = context_detector._keyword_classify(message)

        assert scores["reflection"] > 0

    def test_keyword_classify_scores_normalized(self, context_detector):
        """Test that keyword classification scores are normalized."""
        message = "project deadline meeting report"
        scores = context_detector._keyword_classify(message)

        # Scores should sum to approximately 1.0
        total = sum(scores.values())
        assert 0.99 <= total <= 1.01

    def test_keyword_classify_empty_message(self, context_detector):
        """Test keyword classification with empty message."""
        message = ""
        scores = context_detector._keyword_classify(message)

        # All scores should be 0 for empty message
        assert all(score == 0 for score in scores.values())

    def test_should_switch_context_auto_switch(self, context_detector):
        """Test should_switch_context with high confidence (auto-switch)."""
        scores = {"work": 0.85, "personal": 0.05, "creative": 0.05, "reflection": 0.05}
        result = context_detector.should_switch_context("personal", scores)

        assert result.should_switch is True
        assert result.recommended_context == "work"
        assert result.needs_confirmation is False
        assert result.confidence == 0.85

    def test_should_switch_context_needs_confirmation(self, context_detector):
        """Test should_switch_context with medium confidence (needs confirmation)."""
        scores = {"work": 0.7, "personal": 0.1, "creative": 0.1, "reflection": 0.1}
        result = context_detector.should_switch_context("personal", scores)

        assert result.should_switch is True
        assert result.recommended_context == "work"
        assert result.needs_confirmation is True
        assert result.confidence == 0.7

    def test_should_switch_context_low_confidence(self, context_detector):
        """Test should_switch_context with low confidence (no switch)."""
        scores = {"work": 0.4, "personal": 0.3, "creative": 0.2, "reflection": 0.1}
        result = context_detector.should_switch_context("personal", scores)

        assert result.should_switch is False
        assert result.recommended_context is None
        assert result.needs_confirmation is False

    def test_should_switch_context_already_in_best_context(self, context_detector):
        """Test should_switch_context when already in the best context."""
        scores = {"work": 0.9, "personal": 0.05, "creative": 0.03, "reflection": 0.02}
        result = context_detector.should_switch_context("work", scores)

        assert result.should_switch is False
        assert result.recommended_context == "work"

    def test_should_switch_context_empty_scores(self, context_detector):
        """Test should_switch_context with empty scores."""
        result = context_detector.should_switch_context("personal", {})

        assert result.should_switch is False
        assert result.recommended_context is None

    def test_should_switch_context_no_current_context(self, context_detector):
        """Test should_switch_context with no current context."""
        scores = {"work": 0.85, "personal": 0.05, "creative": 0.05, "reflection": 0.05}
        result = context_detector.should_switch_context(None, scores)

        assert result.should_switch is True
        assert result.recommended_context == "work"

    def test_llm_classify_success(self, context_detector, mock_ollama_client):
        """Test LLM classification with successful response."""
        mock_ollama_client.generate.return_value = '{"work": 0.8, "personal": 0.1, "creative": 0.05, "reflection": 0.05}'

        scores = context_detector._llm_classify("I need to prepare for tomorrow's meeting")

        assert scores["work"] == 0.8
        assert scores["personal"] == 0.1
        mock_ollama_client.generate.assert_called_once()

    def test_llm_classify_missing_contexts_filled(self, context_detector, mock_ollama_client):
        """Test LLM classification fills in missing context scores."""
        mock_ollama_client.generate.return_value = '{"work": 0.9}'

        scores = context_detector._llm_classify("Work task")

        assert scores["work"] == 0.9
        assert scores["personal"] == 0.0
        assert scores["creative"] == 0.0
        assert scores["reflection"] == 0.0

    def test_llm_classify_invalid_json(self, context_detector, mock_ollama_client):
        """Test LLM classification with invalid JSON response."""
        mock_ollama_client.generate.return_value = "This is not valid JSON"

        scores = context_detector._llm_classify("Some message")

        # Should return equal scores for all contexts as fallback
        assert all(score == 0.25 for score in scores.values())

    def test_classify_message_uses_keywords_for_high_confidence(self, context_detector):
        """Test classify_message uses keywords when confidence is high."""
        # Multiple work keywords should give high keyword confidence
        message = "project deadline meeting report task"
        scores = context_detector.classify_message(message)

        assert "work" in scores
        assert scores["work"] > 0

    def test_classify_message_uses_llm_for_low_keyword_confidence(self, context_detector, mock_ollama_client):
        """Test classify_message uses LLM when keyword confidence is low."""
        mock_ollama_client.generate.return_value = '{"work": 0.2, "personal": 0.6, "creative": 0.1, "reflection": 0.1}'

        # Message without clear keywords
        message = "Let's plan something nice for the evening"
        scores = context_detector.classify_message(message)

        # Should blend keyword and LLM scores
        assert "personal" in scores

    def test_detect_and_recommend_convenience_method(self, context_detector):
        """Test detect_and_recommend convenience method."""
        result = context_detector.detect_and_recommend(
            message="project deadline tomorrow",
            current_context_id="personal"
        )

        assert hasattr(result, 'should_switch')
        assert hasattr(result, 'recommended_context')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'needs_confirmation')
        assert hasattr(result, 'scores')

    def test_get_context_description(self, context_detector):
        """Test getting context descriptions."""
        assert "work" in context_detector.get_context_description("work").lower() or \
               "professional" in context_detector.get_context_description("work").lower()
        assert context_detector.get_context_description("nonexistent") == ""


class TestContextSwitchResult:
    """Test cases for ContextSwitchResult dataclass."""

    def test_context_switch_result_creation(self):
        """Test creating a ContextSwitchResult."""
        from src.context_detector import ContextSwitchResult

        result = ContextSwitchResult(
            should_switch=True,
            recommended_context="work",
            confidence=0.85,
            needs_confirmation=False,
            scores={"work": 0.85, "personal": 0.15}
        )

        assert result.should_switch is True
        assert result.recommended_context == "work"
        assert result.confidence == 0.85
        assert result.needs_confirmation is False
        assert result.scores == {"work": 0.85, "personal": 0.15}
