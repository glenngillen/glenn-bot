"""Tests for the agents module."""

import pytest
from unittest.mock import MagicMock, patch


class TestBaseAgent:
    """Tests for the BaseAgent abstract class."""

    def test_base_agent_is_abstract(self):
        """Test that BaseAgent cannot be instantiated directly."""
        from src.agents import BaseAgent

        with pytest.raises(TypeError):
            BaseAgent("test", "description", MagicMock(), MagicMock())


class TestFrameworkAgent:
    """Tests for the FrameworkAgent class."""

    def test_init(self, mock_knowledge_base, mock_ollama_client):
        """Test FrameworkAgent initialization."""
        from src.agents import FrameworkAgent

        agent = FrameworkAgent(mock_knowledge_base, mock_ollama_client)

        assert agent.name == "Framework Agent"
        assert "framework" in agent.description.lower()

    def test_can_handle_framework_keywords(self, mock_knowledge_base, mock_ollama_client):
        """Test can_handle with framework-related keywords."""
        from src.agents import FrameworkAgent

        mock_knowledge_base.search.return_value = []

        agent = FrameworkAgent(mock_knowledge_base, mock_ollama_client)

        score = agent.can_handle("What framework should I use for this approach?")
        assert score > 0

    def test_can_handle_with_relevant_frameworks(self, mock_knowledge_base, mock_ollama_client):
        """Test can_handle when frameworks are found."""
        from src.agents import FrameworkAgent

        mock_knowledge_base.search.return_value = [
            {"content": "Framework content", "metadata": {"name": "Test"}}
        ]

        agent = FrameworkAgent(mock_knowledge_base, mock_ollama_client)
        score = agent.can_handle("help me solve this problem")

        assert score >= 0.6

    def test_process_with_frameworks(self, mock_knowledge_base, mock_ollama_client):
        """Test process when frameworks are available."""
        from src.agents import FrameworkAgent

        mock_knowledge_base.search.return_value = [
            {"content": "Step 1: Analyze", "metadata": {"name": "Problem Solving"}}
        ]
        mock_ollama_client.generate.return_value = "Apply the Problem Solving framework..."

        agent = FrameworkAgent(mock_knowledge_base, mock_ollama_client)
        response = agent.process("How do I solve this?", {})

        assert mock_ollama_client.generate.called
        assert response == "Apply the Problem Solving framework..."

    def test_process_without_frameworks(self, mock_knowledge_base, mock_ollama_client):
        """Test process when no frameworks are found."""
        from src.agents import FrameworkAgent

        mock_knowledge_base.search.return_value = []
        mock_ollama_client.generate.return_value = "General advice..."

        agent = FrameworkAgent(mock_knowledge_base, mock_ollama_client)
        response = agent.process("Random question", {})

        assert mock_ollama_client.generate.called
        assert response == "General advice..."


class TestValuesAgent:
    """Tests for the ValuesAgent class."""

    def test_init(self, mock_knowledge_base, mock_ollama_client):
        """Test ValuesAgent initialization."""
        from src.agents import ValuesAgent

        agent = ValuesAgent(mock_knowledge_base, mock_ollama_client)

        assert agent.name == "Values Agent"
        assert "values" in agent.description.lower()

    def test_can_handle_values_keywords(self, mock_knowledge_base, mock_ollama_client):
        """Test can_handle with values-related keywords."""
        from src.agents import ValuesAgent

        agent = ValuesAgent(mock_knowledge_base, mock_ollama_client)
        score = agent.can_handle("What are my core values and principles?")

        assert score >= 0.8  # Should have high score for direct values query

    def test_can_handle_minimum_score(self, mock_knowledge_base, mock_ollama_client):
        """Test that ValuesAgent always has minimum involvement."""
        from src.agents import ValuesAgent

        agent = ValuesAgent(mock_knowledge_base, mock_ollama_client)
        score = agent.can_handle("completely unrelated query")

        assert score >= 0.3  # Minimum involvement for value alignment

    def test_process(self, mock_knowledge_base, mock_ollama_client):
        """Test ValuesAgent process method."""
        from src.agents import ValuesAgent

        mock_knowledge_base.search.return_value = [
            {"content": "Value: Honesty", "metadata": {"name": "Honesty"}}
        ]
        mock_ollama_client.generate.return_value = "Aligned with your values..."

        agent = ValuesAgent(mock_knowledge_base, mock_ollama_client)
        response = agent.process("Should I do this?", {"conversation_context": "test"})

        assert mock_knowledge_base.search.called
        assert mock_ollama_client.generate.called
        assert response == "Aligned with your values..."


class TestPreferenceAgent:
    """Tests for the PreferenceAgent class."""

    def test_init(self, mock_knowledge_base, mock_ollama_client):
        """Test PreferenceAgent initialization."""
        from src.agents import PreferenceAgent

        agent = PreferenceAgent(mock_knowledge_base, mock_ollama_client)

        assert agent.name == "Preference Agent"

    def test_can_handle_preference_keywords(self, mock_knowledge_base, mock_ollama_client):
        """Test can_handle with preference-related keywords."""
        from src.agents import PreferenceAgent

        mock_knowledge_base.search.return_value = []

        agent = PreferenceAgent(mock_knowledge_base, mock_ollama_client)
        score = agent.can_handle("What would you recommend based on my preferences?")

        assert score > 0

    def test_can_handle_with_preferences_found(self, mock_knowledge_base, mock_ollama_client):
        """Test can_handle when preferences are found."""
        from src.agents import PreferenceAgent

        mock_knowledge_base.search.return_value = [
            {"content": "Prefers morning meetings", "metadata": {"category": "work"}}
        ]

        agent = PreferenceAgent(mock_knowledge_base, mock_ollama_client)
        score = agent.can_handle("schedule a meeting")

        assert score >= 0.5

    def test_process(self, mock_knowledge_base, mock_ollama_client):
        """Test PreferenceAgent process method."""
        from src.agents import PreferenceAgent

        mock_knowledge_base.search.return_value = [
            {"content": "Prefers Python", "metadata": {"category": "programming"}}
        ]
        mock_ollama_client.generate.return_value = "Based on your preferences..."

        agent = PreferenceAgent(mock_knowledge_base, mock_ollama_client)
        response = agent.process("What language should I use?", {})

        assert response == "Based on your preferences..."


class TestQuotesAgent:
    """Tests for the QuotesAgent class."""

    def test_init_without_quotes_system(self, mock_knowledge_base, mock_ollama_client):
        """Test QuotesAgent initialization without quotes system."""
        from src.agents import QuotesAgent

        agent = QuotesAgent(mock_knowledge_base, mock_ollama_client)

        assert agent.name == "Quotes Agent"
        assert agent.quotes_system is None

    def test_init_with_quotes_system(self, mock_knowledge_base, mock_ollama_client):
        """Test QuotesAgent initialization with quotes system."""
        from src.agents import QuotesAgent

        mock_quotes_system = MagicMock()
        mock_quotes_system.quotes = {"quote1": MagicMock()}

        agent = QuotesAgent(mock_knowledge_base, mock_ollama_client, mock_quotes_system)

        assert agent.quotes_system is not None

    def test_can_handle_inspiration_keywords(self, mock_knowledge_base, mock_ollama_client):
        """Test can_handle with inspiration-related keywords."""
        from src.agents import QuotesAgent

        mock_quotes_system = MagicMock()
        mock_quotes_system.quotes = {"quote1": MagicMock()}

        agent = QuotesAgent(mock_knowledge_base, mock_ollama_client, mock_quotes_system)
        score = agent.can_handle("I need some motivation and inspiration")

        assert score > 0

    def test_can_handle_minimum_with_quotes(self, mock_knowledge_base, mock_ollama_client):
        """Test minimum score when quotes are available."""
        from src.agents import QuotesAgent

        mock_quotes_system = MagicMock()
        mock_quotes_system.quotes = {"quote1": MagicMock()}

        agent = QuotesAgent(mock_knowledge_base, mock_ollama_client, mock_quotes_system)
        score = agent.can_handle("any query")

        assert score >= 0.2

    def test_process_without_quotes_system(self, mock_knowledge_base, mock_ollama_client):
        """Test process when quotes system is not available."""
        from src.agents import QuotesAgent

        mock_ollama_client.generate.return_value = "General advice..."

        agent = QuotesAgent(mock_knowledge_base, mock_ollama_client)
        response = agent.process("give me advice", {"conversation_context": ""})

        assert mock_ollama_client.generate.called
        assert response == "General advice..."

    def test_process_with_quotes(self, mock_knowledge_base, mock_ollama_client):
        """Test process with quotes system."""
        from src.agents import QuotesAgent

        mock_quote = MagicMock()
        mock_quote.text = "Be the change you wish to see."
        mock_quote.author = "Gandhi"
        mock_quote.context = "Inspiration"

        mock_quotes_system = MagicMock()
        mock_quotes_system.quotes = {"quote1": mock_quote}
        mock_quotes_system.search_quotes.return_value = [mock_quote]

        mock_ollama_client.generate.return_value = "As Gandhi said..."

        agent = QuotesAgent(mock_knowledge_base, mock_ollama_client, mock_quotes_system)
        response = agent.process("inspire me", {})

        mock_quotes_system.search_quotes.assert_called_once()
        assert response == "As Gandhi said..."


class TestAgentOrchestrator:
    """Tests for the AgentOrchestrator class."""

    @patch('src.agents.PlanningAgent')
    @patch('src.agents.ReasoningAgent')
    @patch('src.agents.ReviewerAgent')
    @patch('src.agents.TaskExecutor')
    def test_init(self, mock_executor, mock_reviewer, mock_reasoning, mock_planning,
                  mock_knowledge_base, mock_ollama_client):
        """Test AgentOrchestrator initialization."""
        from src.agents import AgentOrchestrator

        orchestrator = AgentOrchestrator(mock_knowledge_base, mock_ollama_client)

        assert orchestrator.framework_agent is not None
        assert orchestrator.values_agent is not None
        assert orchestrator.preference_agent is not None
        assert orchestrator.quotes_agent is not None
        assert len(orchestrator.agents) > 0

    @patch('src.agents.PlanningAgent')
    @patch('src.agents.ReasoningAgent')
    @patch('src.agents.ReviewerAgent')
    @patch('src.agents.TaskExecutor')
    def test_init_with_quotes_system(self, mock_executor, mock_reviewer, mock_reasoning, mock_planning,
                                      mock_knowledge_base, mock_ollama_client):
        """Test AgentOrchestrator initialization with quotes system."""
        from src.agents import AgentOrchestrator

        mock_quotes_system = MagicMock()

        orchestrator = AgentOrchestrator(mock_knowledge_base, mock_ollama_client, mock_quotes_system)

        assert orchestrator.quotes_system is not None

    @patch('src.agents.PlanningAgent')
    @patch('src.agents.ReasoningAgent')
    @patch('src.agents.ReviewerAgent')
    @patch('src.agents.TaskExecutor')
    def test_process_query_standard_workflow(self, mock_executor, mock_reviewer, mock_reasoning, mock_planning,
                                              mock_knowledge_base, mock_ollama_client):
        """Test processing a query using standard workflow."""
        from src.agents import AgentOrchestrator

        # Set up mock to use standard workflow (low planning score)
        mock_planning_instance = MagicMock()
        mock_planning_instance.can_handle.return_value = 0.3
        mock_planning.return_value = mock_planning_instance

        mock_reasoning_instance = MagicMock()
        mock_reasoning_instance.can_handle.return_value = 0.4
        mock_reasoning_instance.process.return_value = "Reasoning response"
        mock_reasoning.return_value = mock_reasoning_instance

        orchestrator = AgentOrchestrator(mock_knowledge_base, mock_ollama_client)
        orchestrator.planning_agent = mock_planning_instance
        orchestrator.reasoning_agent = mock_reasoning_instance

        # Mock all agents to return low scores
        for agent in orchestrator.agents:
            if hasattr(agent, 'can_handle'):
                agent.can_handle = MagicMock(return_value=0.2)

        response = orchestrator.process_query("simple question", {})

        assert response is not None

    @patch('src.agents.PlanningAgent')
    @patch('src.agents.ReasoningAgent')
    @patch('src.agents.ReviewerAgent')
    @patch('src.agents.TaskExecutor')
    def test_agent_map_contains_all_types(self, mock_executor, mock_reviewer, mock_reasoning, mock_planning,
                                           mock_knowledge_base, mock_ollama_client):
        """Test that agent_map contains all required agent types."""
        from src.agents import AgentOrchestrator

        orchestrator = AgentOrchestrator(mock_knowledge_base, mock_ollama_client)

        assert "framework" in orchestrator.agent_map
        assert "values" in orchestrator.agent_map
        assert "preference" in orchestrator.agent_map
        assert "quotes" in orchestrator.agent_map
        assert "planning" in orchestrator.agent_map
        assert "reasoning" in orchestrator.agent_map
        assert "general" in orchestrator.agent_map

    @patch('src.agents.PlanningAgent')
    @patch('src.agents.ReasoningAgent')
    @patch('src.agents.ReviewerAgent')
    @patch('src.agents.TaskExecutor')
    def test_format_task_results(self, mock_executor, mock_reviewer, mock_reasoning, mock_planning,
                                  mock_knowledge_base, mock_ollama_client):
        """Test formatting of task results."""
        from src.agents import AgentOrchestrator

        orchestrator = AgentOrchestrator(mock_knowledge_base, mock_ollama_client)

        mock_task = MagicMock()
        mock_task.description = "Test task"
        mock_task.agent_type = "reasoning"
        mock_task.status = "completed"

        task_results = {
            "task_1": {
                "task": mock_task,
                "review": {"score": 8, "recommendation": "ACCEPT"}
            }
        }

        formatted = orchestrator._format_task_results(task_results)

        assert "Test task" in formatted
        assert "8/10" in formatted
        assert "ACCEPT" in formatted


class TestPlanningAgent:
    """Tests for the PlanningAgent class."""

    def test_init(self, mock_knowledge_base, mock_ollama_client):
        """Test PlanningAgent initialization."""
        from src.reasoning_agents import PlanningAgent

        agent = PlanningAgent(mock_knowledge_base, mock_ollama_client)

        assert agent.name == "Planning Agent"

    def test_can_handle_planning_keywords(self, mock_knowledge_base, mock_ollama_client):
        """Test can_handle with planning-related keywords."""
        from src.reasoning_agents import PlanningAgent

        agent = PlanningAgent(mock_knowledge_base, mock_ollama_client)
        score = agent.can_handle("Help me build a comprehensive plan to develop this product")

        assert score > 0.5

    def test_can_handle_how_to_questions(self, mock_knowledge_base, mock_ollama_client):
        """Test can_handle boosts score for how-to questions."""
        from src.reasoning_agents import PlanningAgent

        agent = PlanningAgent(mock_knowledge_base, mock_ollama_client)
        score = agent.can_handle("How do I implement this feature?")

        assert score >= 0.6


class TestReasoningAgent:
    """Tests for the ReasoningAgent class."""

    def test_init(self, mock_knowledge_base, mock_ollama_client):
        """Test ReasoningAgent initialization."""
        from src.reasoning_agents import ReasoningAgent

        agent = ReasoningAgent(mock_knowledge_base, mock_ollama_client)

        assert agent.name == "Reasoning Agent"

    def test_can_handle_returns_base_score(self, mock_knowledge_base, mock_ollama_client):
        """Test that can_handle returns a base score for any query."""
        from src.reasoning_agents import ReasoningAgent

        agent = ReasoningAgent(mock_knowledge_base, mock_ollama_client)
        score = agent.can_handle("any query at all")

        assert score == 0.4

    def test_process(self, mock_knowledge_base, mock_ollama_client):
        """Test ReasoningAgent process method."""
        from src.reasoning_agents import ReasoningAgent

        mock_knowledge_base.search.return_value = [
            {"content": "Relevant knowledge"}
        ]
        mock_ollama_client.generate.return_value = "Reasoned response..."

        agent = ReasoningAgent(mock_knowledge_base, mock_ollama_client)
        response = agent.process("complex question", {})

        assert mock_knowledge_base.search.called
        assert mock_ollama_client.generate.called
        assert response == "Reasoned response..."


class TestReviewerAgent:
    """Tests for the ReviewerAgent class."""

    def test_init(self, mock_knowledge_base, mock_ollama_client):
        """Test ReviewerAgent initialization."""
        from src.reasoning_agents import ReviewerAgent

        agent = ReviewerAgent(mock_knowledge_base, mock_ollama_client)

        assert agent.name == "Reviewer Agent"

    def test_can_handle_returns_zero(self, mock_knowledge_base, mock_ollama_client):
        """Test that ReviewerAgent returns 0 for can_handle."""
        from src.reasoning_agents import ReviewerAgent

        agent = ReviewerAgent(mock_knowledge_base, mock_ollama_client)
        score = agent.can_handle("any query")

        assert score == 0.0

    def test_process_returns_message(self, mock_knowledge_base, mock_ollama_client):
        """Test that process returns appropriate message."""
        from src.reasoning_agents import ReviewerAgent

        agent = ReviewerAgent(mock_knowledge_base, mock_ollama_client)
        response = agent.process("query", {})

        assert "should not" in response.lower()

    def test_review_response(self, mock_knowledge_base, mock_ollama_client):
        """Test review_response method."""
        from src.reasoning_agents import ReviewerAgent

        mock_knowledge_base.search.return_value = [
            {"content": "Value: Quality", "metadata": {"name": "Quality"}}
        ]
        mock_ollama_client.generate.return_value = """
        Quality Assessment: Good structure
        Score: 8/10
        Recommendation: ACCEPT
        """

        agent = ReviewerAgent(mock_knowledge_base, mock_ollama_client)
        review = agent.review_response(
            original_query="test query",
            response="test response",
            context={}
        )

        assert "review" in review
        assert "recommendation" in review
        assert "score" in review
        assert review["score"] == 8

    def test_review_response_extracts_revise(self, mock_knowledge_base, mock_ollama_client):
        """Test that REVISE recommendation is extracted."""
        from src.reasoning_agents import ReviewerAgent

        mock_knowledge_base.search.return_value = []
        mock_ollama_client.generate.return_value = """
        Needs improvement
        Score: 5/10
        Recommendation: REVISE
        """

        agent = ReviewerAgent(mock_knowledge_base, mock_ollama_client)
        review = agent.review_response("query", "response", {})

        assert review["recommendation"] == "REVISE"


class TestTaskDataclasses:
    """Tests for Task and Plan dataclasses."""

    def test_task_default_values(self):
        """Test Task dataclass default values."""
        from src.reasoning_agents import Task

        task = Task(
            id="task_1",
            description="Test task",
            dependencies=[],
            agent_type="reasoning"
        )

        assert task.status == "pending"
        assert task.result is None
        assert task.reasoning is None

    def test_plan_creation(self):
        """Test Plan dataclass creation."""
        from src.reasoning_agents import Plan, Task

        tasks = [
            Task(id="t1", description="First", dependencies=[], agent_type="planning"),
            Task(id="t2", description="Second", dependencies=["t1"], agent_type="reasoning")
        ]

        plan = Plan(
            goal="Test goal",
            tasks=tasks,
            context={"key": "value"}
        )

        assert plan.goal == "Test goal"
        assert len(plan.tasks) == 2
        assert plan.context["key"] == "value"


class TestTaskExecutor:
    """Tests for the TaskExecutor class."""

    def test_init(self, mock_knowledge_base, mock_ollama_client):
        """Test TaskExecutor initialization."""
        from src.reasoning_agents import TaskExecutor, ReviewerAgent

        mock_reviewer = ReviewerAgent(mock_knowledge_base, mock_ollama_client)
        agents = {"reasoning": MagicMock()}

        executor = TaskExecutor(agents, mock_reviewer)

        assert executor.agents == agents
        assert executor.reviewer == mock_reviewer

    def test_execute_plan_empty_tasks(self, mock_knowledge_base, mock_ollama_client):
        """Test executing a plan with no tasks."""
        from src.reasoning_agents import TaskExecutor, ReviewerAgent, Plan

        mock_reviewer = ReviewerAgent(mock_knowledge_base, mock_ollama_client)
        agents = {"reasoning": MagicMock()}

        executor = TaskExecutor(agents, mock_reviewer)

        plan = Plan(goal="Empty plan", tasks=[], context={})
        results = executor.execute_plan(plan)

        assert results["success"] is True
        assert results["task_results"] == {}
