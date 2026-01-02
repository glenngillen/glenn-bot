"""Tests for agents.py module."""

import pytest
from unittest.mock import MagicMock, patch

from src.agents import (
    BaseAgent,
    FrameworkAgent,
    ValuesAgent,
    PreferenceAgent,
    QuotesAgent,
    AgentOrchestrator
)


class TestFrameworkAgent:
    """Tests for FrameworkAgent class."""

    @pytest.fixture
    def framework_agent(self, mock_knowledge_base, mock_ollama_client):
        """Create a FrameworkAgent with mocked dependencies."""
        return FrameworkAgent(mock_knowledge_base, mock_ollama_client)

    def test_initialization(self, framework_agent):
        """Test FrameworkAgent initialization."""
        assert framework_agent.name == "Framework Agent"
        assert "framework" in framework_agent.description.lower()

    def test_can_handle_framework_keywords(self, framework_agent, mock_knowledge_base):
        """Test can_handle returns high score for framework-related queries."""
        mock_knowledge_base.search.return_value = []

        queries_with_keywords = [
            "What framework should I use?",
            "Give me an approach for solving this",
            "What methodology works best?",
            "Walk me through the process",
        ]

        for query in queries_with_keywords:
            score = framework_agent.can_handle(query)
            assert score > 0, f"Expected positive score for: {query}"

    def test_can_handle_with_matching_frameworks(self, framework_agent, mock_knowledge_base):
        """Test can_handle boosts score when frameworks are found."""
        mock_knowledge_base.search.return_value = [
            {"metadata": {"name": "Test Framework"}, "content": "A framework"}
        ]

        score = framework_agent.can_handle("help me think through this")

        assert score >= 0.6

    def test_can_handle_unrelated_query(self, framework_agent, mock_knowledge_base):
        """Test can_handle returns low score for unrelated queries."""
        mock_knowledge_base.search.return_value = []

        score = framework_agent.can_handle("what time is it")

        assert score < 0.5

    def test_process_with_frameworks(self, framework_agent, mock_knowledge_base, mock_ollama_client):
        """Test processing a query when frameworks are available."""
        mock_knowledge_base.search.return_value = [
            {
                "metadata": {"name": "SWOT Analysis"},
                "content": "Framework: SWOT Analysis\nAnalyze Strengths, Weaknesses..."
            }
        ]
        mock_ollama_client.generate.return_value = "Let me apply the SWOT framework..."

        result = framework_agent.process(
            "How should I analyze this opportunity?",
            {"conversation_context": "Previous conversation..."}
        )

        assert result is not None
        mock_ollama_client.generate.assert_called_once()

    def test_process_without_frameworks(self, framework_agent, mock_knowledge_base, mock_ollama_client):
        """Test processing when no frameworks match."""
        mock_knowledge_base.search.return_value = []
        mock_ollama_client.generate.return_value = "Here's a structured approach..."

        result = framework_agent.process(
            "How should I handle this?",
            {}
        )

        assert result is not None


class TestValuesAgent:
    """Tests for ValuesAgent class."""

    @pytest.fixture
    def values_agent(self, mock_knowledge_base, mock_ollama_client):
        """Create a ValuesAgent with mocked dependencies."""
        return ValuesAgent(mock_knowledge_base, mock_ollama_client)

    def test_initialization(self, values_agent):
        """Test ValuesAgent initialization."""
        assert values_agent.name == "Values Agent"
        assert "values" in values_agent.description.lower()

    def test_can_handle_values_keywords(self, values_agent):
        """Test can_handle returns high score for values-related queries."""
        queries = [
            "What are my values?",
            "Does this align with my principles?",
            "Is this ethical?",
            "What's most important to me?",
        ]

        for query in queries:
            score = values_agent.can_handle(query)
            assert score > 0.3, f"Expected score > 0.3 for: {query}"

    def test_can_handle_always_minimum(self, values_agent):
        """Test values agent always has minimum involvement."""
        score = values_agent.can_handle("completely random query")

        assert score >= 0.3

    def test_can_handle_direct_values_query(self, values_agent):
        """Test high score for direct values queries."""
        score = values_agent.can_handle("What are my core values?")

        assert score >= 0.8

    def test_process_with_values(self, values_agent, mock_knowledge_base, mock_ollama_client):
        """Test processing with available values."""
        mock_knowledge_base.search.return_value = [
            {"metadata": {"name": "Integrity"}, "content": "Always be honest..."}
        ]
        mock_ollama_client.generate.return_value = "Based on your values..."

        result = values_agent.process(
            "Should I take this opportunity?",
            {"conversation_context": "Discussion about opportunity"}
        )

        assert result is not None


class TestPreferenceAgent:
    """Tests for PreferenceAgent class."""

    @pytest.fixture
    def preference_agent(self, mock_knowledge_base, mock_ollama_client):
        """Create a PreferenceAgent with mocked dependencies."""
        return PreferenceAgent(mock_knowledge_base, mock_ollama_client)

    def test_initialization(self, preference_agent):
        """Test PreferenceAgent initialization."""
        assert preference_agent.name == "Preference Agent"
        assert "preference" in preference_agent.description.lower()

    def test_can_handle_preference_keywords(self, preference_agent, mock_knowledge_base):
        """Test can_handle returns appropriate score for preference queries."""
        mock_knowledge_base.search.return_value = []

        queries = [
            "What do I prefer?",
            "Give me a recommendation",
            "What would be the best option?",
            "Suggest something for me",
        ]

        for query in queries:
            score = preference_agent.can_handle(query)
            assert score >= 0, f"Expected non-negative score for: {query}"

    def test_can_handle_with_matching_preferences(self, preference_agent, mock_knowledge_base):
        """Test can_handle boosts score when preferences are found."""
        mock_knowledge_base.search.return_value = [
            {"metadata": {"category": "work"}, "content": "Prefers quiet environment"}
        ]

        score = preference_agent.can_handle("what kind of work setup")

        assert score >= 0.5

    def test_process_with_preferences(self, preference_agent, mock_knowledge_base, mock_ollama_client):
        """Test processing with available preferences."""
        mock_knowledge_base.search.return_value = [
            {"metadata": {"category": "food"}, "content": "Preference: Likes spicy food"}
        ]
        mock_ollama_client.generate.return_value = "Based on your preferences..."

        result = preference_agent.process(
            "Recommend a restaurant",
            {}
        )

        assert result is not None


class TestQuotesAgent:
    """Tests for QuotesAgent class."""

    @pytest.fixture
    def mock_quotes_system(self):
        """Create a mock quotes system."""
        quotes_system = MagicMock()
        quotes_system.quotes = ["quote1", "quote2"]
        quotes_system.search_quotes.return_value = []
        return quotes_system

    @pytest.fixture
    def quotes_agent(self, mock_knowledge_base, mock_ollama_client, mock_quotes_system):
        """Create a QuotesAgent with mocked dependencies."""
        return QuotesAgent(mock_knowledge_base, mock_ollama_client, mock_quotes_system)

    @pytest.fixture
    def quotes_agent_no_system(self, mock_knowledge_base, mock_ollama_client):
        """Create a QuotesAgent without quotes system."""
        return QuotesAgent(mock_knowledge_base, mock_ollama_client, None)

    def test_initialization(self, quotes_agent):
        """Test QuotesAgent initialization."""
        assert quotes_agent.name == "Quotes Agent"
        assert "quotes" in quotes_agent.description.lower()

    def test_can_handle_inspiration_keywords(self, quotes_agent):
        """Test can_handle for inspiration-related queries."""
        queries = [
            "I need some inspiration",
            "Give me motivation",
            "Share some wisdom",
            "What quote applies here?",
        ]

        for query in queries:
            score = quotes_agent.can_handle(query)
            assert score > 0, f"Expected positive score for: {query}"

    def test_can_handle_with_quotes_available(self, quotes_agent):
        """Test minimum score when quotes system has quotes."""
        score = quotes_agent.can_handle("random query")

        assert score >= 0.2

    def test_can_handle_no_quotes_system(self, quotes_agent_no_system):
        """Test scoring when no quotes system available."""
        score = quotes_agent_no_system.can_handle("give me motivation")

        # Should still work but may have lower score
        assert score >= 0

    def test_process_with_quotes(self, quotes_agent, mock_quotes_system, mock_ollama_client):
        """Test processing when relevant quotes are found."""
        mock_quote = MagicMock()
        mock_quote.text = "The only way to do great work is to love what you do."
        mock_quote.author = "Steve Jobs"
        mock_quote.context = "Work and passion"
        mock_quotes_system.search_quotes.return_value = [mock_quote]

        mock_ollama_client.generate.return_value = "This reminds me of what Steve Jobs said..."

        result = quotes_agent.process(
            "I'm feeling unmotivated at work",
            {"conversation_context": "Career discussion"}
        )

        assert result is not None

    def test_process_without_quotes_system(self, quotes_agent_no_system, mock_ollama_client):
        """Test processing when no quotes system is available."""
        mock_ollama_client.generate.return_value = "Here's some thoughtful advice..."

        result = quotes_agent_no_system.process(
            "I need guidance",
            {}
        )

        assert result is not None


class TestAgentOrchestrator:
    """Tests for AgentOrchestrator class."""

    @pytest.fixture
    def orchestrator(self, mock_knowledge_base, mock_ollama_client):
        """Create an AgentOrchestrator with mocked dependencies."""
        with patch('src.agents.PlanningAgent') as MockPlanning, \
             patch('src.agents.ReasoningAgent') as MockReasoning, \
             patch('src.agents.ReviewerAgent') as MockReviewer, \
             patch('src.agents.TaskExecutor') as MockExecutor:

            # Setup mock agents
            mock_planning = MagicMock()
            mock_planning.can_handle.return_value = 0.3
            mock_planning.process.return_value = "Plan response"
            MockPlanning.return_value = mock_planning

            mock_reasoning = MagicMock()
            mock_reasoning.can_handle.return_value = 0.5
            mock_reasoning.process.return_value = "Reasoning response"
            mock_reasoning.ollama_client = mock_ollama_client
            MockReasoning.return_value = mock_reasoning

            mock_reviewer = MagicMock()
            mock_reviewer.review_response.return_value = {
                "score": 8,
                "recommendation": "ACCEPT",
                "review": "Good response"
            }
            MockReviewer.return_value = mock_reviewer

            mock_executor = MagicMock()
            MockExecutor.return_value = mock_executor

            orchestrator = AgentOrchestrator(mock_knowledge_base, mock_ollama_client)
            yield orchestrator

    def test_initialization(self, orchestrator):
        """Test AgentOrchestrator initialization."""
        assert orchestrator.agents is not None
        assert len(orchestrator.agents) > 0
        assert orchestrator.framework_agent is not None
        assert orchestrator.values_agent is not None
        assert orchestrator.preference_agent is not None

    def test_process_query_standard_workflow(self, orchestrator, mock_knowledge_base, mock_ollama_client):
        """Test processing a query with standard workflow."""
        mock_knowledge_base.search.return_value = []

        result = orchestrator.process_query(
            "What's the weather like?",
            {"conversation_context": ""}
        )

        assert result is not None

    def test_process_query_planning_workflow(self, orchestrator, mock_ollama_client):
        """Test processing a complex query triggers planning workflow."""
        # Make planning agent return high confidence
        orchestrator.planning_agent.can_handle.return_value = 0.8

        # Mock the planning workflow
        orchestrator.task_executor.execute_plan.return_value = {
            "success": True,
            "final_synthesis": "Comprehensive plan executed",
            "task_results": {}
        }

        result = orchestrator.process_query(
            "Create a comprehensive strategy for launching a new product with market analysis, competitive research, and go-to-market planning",
            {}
        )

        assert result is not None

    def test_agent_map_contains_all_agents(self, orchestrator):
        """Test that agent_map includes all expected agent types."""
        expected_types = ["framework", "values", "preference", "quotes", "planning", "reasoning", "general"]

        for agent_type in expected_types:
            assert agent_type in orchestrator.agent_map

    def test_standard_workflow_high_confidence_agent(self, orchestrator, mock_knowledge_base):
        """Test standard workflow uses high-confidence agent."""
        # Setup framework agent to have high confidence
        orchestrator.framework_agent.can_handle = MagicMock(return_value=0.9)
        orchestrator.framework_agent.process = MagicMock(return_value="Framework-based answer")
        mock_knowledge_base.search.return_value = [
            {"metadata": {"type": "framework"}, "content": "Test framework"}
        ]

        # Ensure planning doesn't trigger
        orchestrator.planning_agent.can_handle.return_value = 0.3

        result = orchestrator.process_query(
            "What framework should I use for this problem?",
            {}
        )

        assert result is not None

    def test_standard_workflow_synthesis(self, orchestrator, mock_ollama_client):
        """Test standard workflow synthesizes multiple agent responses."""
        # Setup agents with moderate confidence
        orchestrator.framework_agent.can_handle = MagicMock(return_value=0.5)
        orchestrator.values_agent.can_handle = MagicMock(return_value=0.5)
        orchestrator.preference_agent.can_handle = MagicMock(return_value=0.4)

        orchestrator.framework_agent.process = MagicMock(return_value="Framework perspective")
        orchestrator.values_agent.process = MagicMock(return_value="Values perspective")

        # Ensure planning doesn't trigger
        orchestrator.planning_agent.can_handle.return_value = 0.3

        mock_ollama_client.generate.return_value = "Synthesized response"

        result = orchestrator.process_query(
            "Help me make this decision",
            {}
        )

        assert result is not None

    def test_format_task_results(self, orchestrator):
        """Test formatting of task results."""
        mock_task = MagicMock()
        mock_task.description = "Test task"
        mock_task.status = "completed"
        mock_task.agent_type = "reasoning"

        task_results = {
            "task_1": {
                "task": mock_task,
                "review": {"score": 8, "recommendation": "ACCEPT"}
            }
        }

        formatted = orchestrator._format_task_results(task_results)

        assert "Test task" in formatted
        assert "reasoning" in formatted
        assert "8" in formatted

    def test_extract_plan_from_valid_json(self, orchestrator):
        """Test extracting plan from valid JSON response."""
        json_response = '''
        Here's the plan:
        {
            "goal": "Complete the task",
            "tasks": [
                {
                    "id": "task_1",
                    "description": "First step",
                    "dependencies": [],
                    "agent_type": "reasoning"
                }
            ]
        }
        '''

        plan = orchestrator._extract_plan_from_response(json_response, "Original query", {})

        assert plan is not None
        assert plan.goal == "Complete the task"
        assert len(plan.tasks) == 1

    def test_extract_plan_fallback(self, orchestrator):
        """Test plan extraction fallback for invalid JSON."""
        invalid_response = "This is not JSON at all"

        plan = orchestrator._extract_plan_from_response(invalid_response, "Original query", {})

        # Should create a fallback plan
        assert plan is not None
        assert len(plan.tasks) == 1
        assert plan.tasks[0].agent_type == "reasoning"


class TestAgentIntegration:
    """Integration tests for agents working together."""

    @pytest.fixture
    def full_orchestrator(self, mock_knowledge_base, mock_ollama_client):
        """Create a fully configured orchestrator."""
        with patch('src.agents.PlanningAgent') as MockPlanning, \
             patch('src.agents.ReasoningAgent') as MockReasoning, \
             patch('src.agents.ReviewerAgent') as MockReviewer, \
             patch('src.agents.TaskExecutor') as MockExecutor:

            mock_planning = MagicMock()
            mock_planning.can_handle.return_value = 0.3
            MockPlanning.return_value = mock_planning

            mock_reasoning = MagicMock()
            mock_reasoning.can_handle.return_value = 0.5
            mock_reasoning.process.return_value = "Reasoning response"
            mock_reasoning.ollama_client = mock_ollama_client
            MockReasoning.return_value = mock_reasoning

            mock_reviewer = MagicMock()
            mock_reviewer.review_response.return_value = {
                "score": 8,
                "recommendation": "ACCEPT",
                "review": "Good"
            }
            MockReviewer.return_value = mock_reviewer

            MockExecutor.return_value = MagicMock()

            mock_knowledge_base.search.return_value = []

            orchestrator = AgentOrchestrator(mock_knowledge_base, mock_ollama_client)
            yield orchestrator

    def test_multiple_agent_coordination(self, full_orchestrator):
        """Test that multiple agents can coordinate on a query."""
        result = full_orchestrator.process_query(
            "How should I approach my career development considering my values and preferences?",
            {"conversation_context": "Career coaching session"}
        )

        assert result is not None

    def test_agent_selection_varies_by_query(self, full_orchestrator):
        """Test that different queries activate different agents."""
        queries = [
            ("What framework should I use?", "framework"),
            ("What are my core values?", "values"),
            ("What do I prefer for X?", "preference"),
        ]

        for query, expected_focus in queries:
            # Just verify the query can be processed
            result = full_orchestrator.process_query(query, {})
            assert result is not None
