from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod
import logging
import json
import re
from src.knowledge_base import KnowledgeBase
from src.ollama_client import OllamaClient

logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    """Base class for all agents."""
    
    def __init__(self, name: str, description: str, knowledge_base: KnowledgeBase, ollama_client: OllamaClient):
        self.name = name
        self.description = description
        self.knowledge_base = knowledge_base
        self.ollama_client = ollama_client
        
    @abstractmethod
    def can_handle(self, query: str) -> float:
        """Return confidence score (0-1) that this agent can handle the query."""
        pass
        
    @abstractmethod
    def process(self, query: str, context: Dict[str, Any]) -> str:
        """Process the query and return a response."""
        pass

class FrameworkAgent(BaseAgent):
    """Agent that applies problem-solving frameworks."""
    
    def __init__(self, knowledge_base: KnowledgeBase, ollama_client: OllamaClient):
        super().__init__(
            name="Framework Agent",
            description="Applies problem-solving frameworks to structure thinking",
            knowledge_base=knowledge_base,
            ollama_client=ollama_client
        )
        
    def can_handle(self, query: str) -> float:
        keywords = ["framework", "approach", "methodology", "process", "steps", "structure", "solve", "plan"]
        score = sum(1 for keyword in keywords if keyword in query.lower()) / len(keywords)
        
        # Check if we have relevant frameworks
        frameworks = self.knowledge_base.search(query, n_results=3, filter_metadata={"type": {"$eq": "framework"}})
        if frameworks:
            score = max(score, 0.6)
            
        return min(score * 2, 1.0)
        
    def process(self, query: str, context: Dict[str, Any]) -> str:
        # Search for relevant frameworks
        frameworks = self.knowledge_base.search(query, n_results=3, filter_metadata={"type": {"$eq": "framework"}})
        
        if not frameworks:
            return self.ollama_client.generate(
                prompt=query,
                system_prompt="You are a helpful assistant that structures problem-solving approaches."
            )
            
        # Build context from frameworks
        framework_context = "\n\n".join([
            f"Framework: {fw['metadata']['name']}\n{fw['content']}"
            for fw in frameworks
        ])
        
        system_prompt = f"""You are an expert at applying problem-solving frameworks. 
        
Available frameworks:
{framework_context}

Apply the most relevant framework(s) to help address the user's query. Be specific about which framework you're using and why."""
        
        return self.ollama_client.generate(
            prompt=query,
            system_prompt=system_prompt,
            context=context.get('conversation_context', '')
        )

class ValuesAgent(BaseAgent):
    """Agent that considers values and principles."""
    
    def __init__(self, knowledge_base: KnowledgeBase, ollama_client: OllamaClient):
        super().__init__(
            name="Values Agent",
            description="Ensures responses align with user's values and principles",
            knowledge_base=knowledge_base,
            ollama_client=ollama_client
        )
        
    def can_handle(self, query: str) -> float:
        keywords = ["values", "principles", "ethics", "beliefs", "important", "priority", "align", "personal"]
        score = sum(1 for keyword in keywords if keyword in query.lower()) / len(keywords)
        
        # Boost score for direct values/principles queries
        if any(word in query.lower() for word in ["values", "principles"]):
            score = max(score, 0.8)
        
        # Always have some involvement to ensure value alignment
        return max(score * 2, 0.3)
        
    def process(self, query: str, context: Dict[str, Any]) -> str:
        # Get relevant values and principles
        values = self.knowledge_base.search(query, n_results=5, filter_metadata={"type": {"$eq": "value"}})
        
        values_context = "\n".join([
            f"- {v['metadata']['name']}: {v['content']}"
            for v in values
        ])
        
        system_prompt = f"""You are an assistant that helps ensure decisions and advice align with the user's values and principles.

User's core values:
{values_context}

Consider these values when providing your response."""
        
        return self.ollama_client.generate(
            prompt=query,
            system_prompt=system_prompt,
            context=context.get('conversation_context', '')
        )

class PreferenceAgent(BaseAgent):
    """Agent that considers user preferences."""
    
    def __init__(self, knowledge_base: KnowledgeBase, ollama_client: OllamaClient):
        super().__init__(
            name="Preference Agent",
            description="Applies user preferences to recommendations",
            knowledge_base=knowledge_base,
            ollama_client=ollama_client
        )
        
    def can_handle(self, query: str) -> float:
        keywords = ["prefer", "like", "favorite", "recommendation", "suggest", "option", "choice", "best"]
        score = sum(1 for keyword in keywords if keyword in query.lower()) / len(keywords)
        
        # Check for relevant preferences
        prefs = self.knowledge_base.search(query, n_results=3, filter_metadata={"type": {"$eq": "preference"}})
        if prefs:
            score = max(score, 0.5)
            
        return min(score * 2, 1.0)
        
    def process(self, query: str, context: Dict[str, Any]) -> str:
        # Get relevant preferences
        preferences = self.knowledge_base.search(query, n_results=5, filter_metadata={"type": {"$eq": "preference"}})
        
        pref_context = "\n".join([
            f"- {p['metadata']['category']}: {p['content']}"
            for p in preferences
        ])
        
        system_prompt = f"""You are an assistant that makes personalized recommendations based on user preferences.

User's relevant preferences:
{pref_context}

Use these preferences to inform your recommendations and suggestions."""
        
        return self.ollama_client.generate(
            prompt=query,
            system_prompt=system_prompt,
            context=context.get('conversation_context', '')
        )

class QuotesAgent(BaseAgent):
    """Agent that incorporates inspirational quotes into advice."""
    
    def __init__(self, knowledge_base: KnowledgeBase, ollama_client: OllamaClient, quotes_system=None):
        super().__init__(
            name="Quotes Agent",
            description="Incorporates inspirational quotes and wisdom into responses",
            knowledge_base=knowledge_base,
            ollama_client=ollama_client
        )
        self.quotes_system = quotes_system
        
    def can_handle(self, query: str) -> float:
        keywords = ["inspire", "motivation", "wisdom", "quote", "advice", "guidance", "mindset", "perspective"]
        score = sum(1 for keyword in keywords if keyword in query.lower()) / len(keywords)
        
        # Check if we have relevant quotes
        if self.quotes_system and len(self.quotes_system.quotes) > 0:
            score = max(score, 0.2)  # Always have some involvement when quotes are available
            
        return min(score * 2, 1.0)
        
    def process(self, query: str, context: Dict[str, Any]) -> str:
        if not self.quotes_system:
            return self.ollama_client.generate(
                prompt=query,
                system_prompt="You are a helpful assistant that provides thoughtful advice.",
                context=context.get('conversation_context', '')
            )
            
        # Search for relevant quotes
        relevant_quotes = self.quotes_system.search_quotes(query, limit=3)
        
        quotes_context = ""
        if relevant_quotes:
            quotes_context = "\n\nRelevant inspirational quotes:\n" + "\n".join([
                f'"{quote.text}" - {quote.author} (Context: {quote.context})'
                for quote in relevant_quotes
            ])
            
        system_prompt = f"""You are a wise advisor who incorporates inspirational wisdom into your responses.
        
When appropriate, reference the quotes provided and explain how they relate to the user's situation.
Don't force quotes into every response - only use them when they genuinely add value.
{quotes_context}

Provide thoughtful, actionable advice that combines practical solutions with inspirational wisdom."""
        
        return self.ollama_client.generate(
            prompt=query,
            system_prompt=system_prompt,
            context=context.get('conversation_context', '')
        )

class AgentOrchestrator:
    """Orchestrates multiple agents to handle queries with reasoning capabilities."""
    
    def __init__(self, knowledge_base: KnowledgeBase, ollama_client: OllamaClient, quotes_system=None):
        self.knowledge_base = knowledge_base
        self.ollama_client = ollama_client
        self.quotes_system = quotes_system
        
        # Import reasoning agents here to avoid circular imports
        from src.reasoning_agents import PlanningAgent, ReasoningAgent, ReviewerAgent, TaskExecutor, Plan, Task
        
        # Initialize all agents
        self.framework_agent = FrameworkAgent(knowledge_base, ollama_client)
        self.values_agent = ValuesAgent(knowledge_base, ollama_client)
        self.preference_agent = PreferenceAgent(knowledge_base, ollama_client)
        self.quotes_agent = QuotesAgent(knowledge_base, ollama_client, quotes_system)
        self.planning_agent = PlanningAgent(knowledge_base, ollama_client)
        self.reasoning_agent = ReasoningAgent(knowledge_base, ollama_client)
        self.reviewer_agent = ReviewerAgent(knowledge_base, ollama_client)
        
        self.agents = [
            self.framework_agent,
            self.values_agent, 
            self.preference_agent,
            self.quotes_agent,
            self.planning_agent,
            self.reasoning_agent
        ]
        
        # Agent mapping for task execution
        self.agent_map = {
            "framework": self.framework_agent,
            "values": self.values_agent,
            "preference": self.preference_agent,
            "quotes": self.quotes_agent,
            "planning": self.planning_agent,
            "reasoning": self.reasoning_agent,
            "general": self.reasoning_agent  # Use reasoning agent as general fallback
        }
        
        self.task_executor = TaskExecutor(self.agent_map, self.reviewer_agent)
        
        # Import classes for use in methods
        self.Plan = Plan
        self.Task = Task
        
    def process_query(self, query: str, context: Dict[str, Any]) -> str:
        """Process a query using reasoning-enhanced workflow."""
        
        # Check if this needs planning/decomposition
        planning_score = self.planning_agent.can_handle(query)
        
        if planning_score > 0.6:
            logger.info(f"Using planning workflow (confidence: {planning_score:.2f})")
            return self._execute_planning_workflow(query, context)
        else:
            logger.info("Using standard agent workflow")
            return self._execute_standard_workflow(query, context)
            
    def _execute_planning_workflow(self, query: str, context: Dict[str, Any]) -> str:
        """Execute complex queries using planning and task decomposition."""
        
        try:
            # Step 1: Create plan
            plan_response = self.planning_agent.process(query, context)
            
            # Step 2: Extract plan from response
            plan = self._extract_plan_from_response(plan_response, query, context)
            
            if not plan or not plan.tasks:
                logger.warning("Failed to extract valid plan, falling back to reasoning")
                return self.reasoning_agent.process(query, context)
                
            # Step 3: Execute plan
            execution_results = self.task_executor.execute_plan(plan)
            
            # Step 4: Format response
            if execution_results["success"]:
                response = f"""# Plan Execution Results

**Goal**: {plan.goal}

## Execution Summary
✅ Completed {len([t for t in plan.tasks if t.status == 'completed'])}/{len(plan.tasks)} tasks

## Comprehensive Response
{execution_results['final_synthesis']}

## Task Breakdown
{self._format_task_results(execution_results['task_results'])}
"""
            else:
                response = f"""# Partial Plan Execution

**Goal**: {plan.goal}

⚠️ Completed {len([t for t in plan.tasks if t.status == 'completed'])}/{len(plan.tasks)} tasks

## Available Results
{execution_results['final_synthesis']}
"""
            
            # Step 5: Final review
            final_review = self.reviewer_agent.review_response(query, response, context)
            
            if final_review["recommendation"] == "REDO" and final_review["score"] < 6:
                logger.info("Plan execution quality too low, using reasoning fallback")
                return self.reasoning_agent.process(query, context)
                
            return response
            
        except Exception as e:
            logger.error(f"Planning workflow failed: {e}")
            return self.reasoning_agent.process(query, context)
            
    def _execute_standard_workflow(self, query: str, context: Dict[str, Any]) -> str:
        """Execute queries using standard agent selection with reasoning enhancement."""
        
        # Get confidence scores from all agents (excluding planning)
        agent_scores = []
        for agent in self.agents:
            if agent != self.planning_agent:  # Skip planning agent for standard workflow
                score = agent.can_handle(query)
                agent_scores.append((agent, score))
                logger.info(f"{agent.name} confidence: {score:.2f}")
                
        # Sort by confidence
        agent_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Use agents with confidence > 0.3
        active_agents = [(agent, score) for agent, score in agent_scores if score > 0.3]
        
        if not active_agents:
            # Fallback to reasoning agent
            return self.reasoning_agent.process(query, context)
            
        # If one agent has high confidence, use it with reasoning enhancement
        if active_agents[0][1] > 0.7:
            agent = active_agents[0][0]
            logger.info(f"Using {agent.name} as primary agent with reasoning")
            
            # Get response from primary agent
            response = agent.process(query, context)
            
            # Review and potentially enhance with reasoning
            review = self.reviewer_agent.review_response(query, response, context)
            
            if review["score"] >= 7:
                return response
            else:
                logger.info("Primary agent response quality low, enhancing with reasoning")
                reasoning_context = {**context, "initial_response": response, "review": review["review"]}
                return self.reasoning_agent.process(query, reasoning_context)
                
        # Otherwise, use reasoning agent to synthesize multiple perspectives
        logger.info("Using reasoning agent to synthesize multiple perspectives")
        
        # Collect responses from top agents
        responses = []
        for agent, score in active_agents[:3]:  # Use top 3 agents
            response = agent.process(query, context)
            responses.append({
                "agent": agent.name,
                "response": response,
                "confidence": score
            })
            
        # Use reasoning agent to synthesize
        synthesis_context = {
            **context,
            "agent_responses": responses
        }
        
        synthesis_prompt = f"""Synthesize these expert perspectives into a unified, comprehensive response.

Original Query: {query}

Expert Responses:
{chr(10).join([f"**{r['agent']}** (confidence: {r['confidence']:.2f}):\n{r['response']}\n" for r in responses])}

Apply your reasoning capabilities to:
1. Identify the best insights from each perspective
2. Resolve any conflicts or contradictions
3. Create a cohesive, actionable response
4. Ensure alignment with user values and preferences

Provide a unified response that's better than any individual perspective."""

        return self.reasoning_agent.ollama_client.generate(
            prompt=synthesis_prompt,
            system_prompt="You are an expert synthesizer who creates unified responses from multiple perspectives using chain-of-thought reasoning."
        )
        
    def _extract_plan_from_response(self, plan_response: str, original_query: str, context: Dict[str, Any]) -> Optional['Plan']:
        """Extract a structured plan from the planning agent's response."""
        
        try:
            # Look for JSON in the response
            json_match = re.search(r'\{.*\}', plan_response, re.DOTALL)
            if json_match:
                plan_data = json.loads(json_match.group())
                
                # Convert to Plan object
                tasks = []
                for task_data in plan_data.get("tasks", []):
                    task = self.Task(
                        id=task_data.get("id", f"task_{len(tasks)}"),
                        description=task_data.get("description", ""),
                        dependencies=task_data.get("dependencies", []),
                        agent_type=task_data.get("agent_type", "general")
                    )
                    tasks.append(task)
                    
                return self.Plan(
                    goal=plan_data.get("goal", original_query),
                    tasks=tasks,
                    context=context
                )
                
        except Exception as e:
            logger.error(f"Failed to extract plan from response: {e}")
            
        # Fallback: create simple plan
        return self.Plan(
            goal=original_query,
            tasks=[
                self.Task(
                    id="task_1",
                    description=original_query,
                    dependencies=[],
                    agent_type="reasoning"
                )
            ],
            context=context
        )
        
    def _format_task_results(self, task_results: Dict[str, Any]) -> str:
        """Format task execution results for display."""
        
        formatted = []
        for task_id, result_data in task_results.items():
            task = result_data["task"]
            review = result_data["review"]
            
            status_icon = "✅" if task.status == "completed" else "❌" if task.status == "failed" else "⏳"
            
            formatted.append(f"""
**{status_icon} {task.description}**
- Agent: {task.agent_type}
- Review Score: {review['score']}/10
- Recommendation: {review['recommendation']}
""")
            
        return "\n".join(formatted)