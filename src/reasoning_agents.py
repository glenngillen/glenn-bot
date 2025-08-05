from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import json
import re
import logging
from src.agents import BaseAgent
from src.knowledge_base import KnowledgeBase
from src.ollama_client import OllamaClient

logger = logging.getLogger(__name__)

@dataclass
class Task:
    """Represents a subtask in a plan."""
    id: str
    description: str
    dependencies: List[str]
    agent_type: str
    status: str = "pending"  # pending, in_progress, completed, failed
    result: Optional[str] = None
    reasoning: Optional[str] = None

@dataclass
class Plan:
    """Represents a complete execution plan."""
    goal: str
    tasks: List[Task]
    context: Dict[str, Any]
    
class PlanningAgent(BaseAgent):
    """Agent that breaks down complex queries into structured plans."""
    
    def __init__(self, knowledge_base: KnowledgeBase, ollama_client: OllamaClient):
        super().__init__(
            name="Planning Agent",
            description="Breaks down complex problems into structured, executable plans",
            knowledge_base=knowledge_base,
            ollama_client=ollama_client
        )
        
    def can_handle(self, query: str) -> float:
        # Look for complexity indicators
        complexity_keywords = [
            "build", "create", "develop", "design", "implement", "plan", "strategy",
            "process", "system", "product", "project", "how to", "step by step",
            "multiple", "complex", "comprehensive", "end-to-end"
        ]
        
        score = sum(1 for keyword in complexity_keywords if keyword in query.lower()) / len(complexity_keywords)
        
        # Boost score for questions that seem to need planning
        if any(phrase in query.lower() for phrase in ["how do i", "help me", "i want to", "i need to"]):
            score = max(score, 0.6)
            
        return min(score * 2, 1.0)
        
    def process(self, query: str, context: Dict[str, Any]) -> str:
        """Create a detailed plan for the query."""
        
        # Get relevant frameworks and values
        frameworks = self.knowledge_base.search(query, n_results=3, filter_metadata={"type": "framework"})
        values = self.knowledge_base.search(query, n_results=3, filter_metadata={"type": "value"})
        
        framework_context = "\n".join([f"- {fw['metadata']['name']}: {fw['content']}" for fw in frameworks])
        values_context = "\n".join([f"- {v['metadata']['name']}: {v['content']}" for v in values])
        
        planning_prompt = f"""You are an expert planning agent. Break down this complex request into a structured, executable plan.

REQUEST: {query}

CONVERSATION CONTEXT:
{context.get('conversation_context', 'None')}

RELEVANT FRAMEWORKS:
{framework_context}

RELEVANT VALUES:
{values_context}

Create a detailed plan using the following structure:

1. **GOAL ANALYSIS**: 
   - Clearly restate the main objective
   - Identify key success criteria
   - Note any constraints or requirements

2. **REASONING**: 
   - Think through the problem step by step
   - Consider which frameworks apply
   - Identify potential challenges

3. **PLAN**:
   Create a JSON structure with tasks:
   ```json
   {{
     "goal": "Clear goal statement",
     "tasks": [
       {{
         "id": "task_1",
         "description": "Specific, actionable task",
         "dependencies": ["task_id_if_any"],
         "agent_type": "framework|values|preference|general",
         "reasoning": "Why this task is needed"
       }}
     ]
   }}
   ```

4. **NEXT STEPS**: 
   - What should happen immediately
   - How to measure progress

Think carefully about task dependencies and the best agent type for each task."""

        return self.ollama_client.generate(
            prompt=planning_prompt,
            system_prompt="You are a master planner who creates detailed, actionable plans. Always think step by step and be thorough."
        )

class ReasoningAgent(BaseAgent):
    """Agent that applies chain-of-thought reasoning to problems."""
    
    def __init__(self, knowledge_base: KnowledgeBase, ollama_client: OllamaClient):
        super().__init__(
            name="Reasoning Agent",
            description="Applies structured reasoning and chain-of-thought to solve problems",
            knowledge_base=knowledge_base,
            ollama_client=ollama_client
        )
        
    def can_handle(self, query: str) -> float:
        # This agent can handle any query but with lower priority than specialized agents
        return 0.4
        
    def process(self, query: str, context: Dict[str, Any]) -> str:
        """Apply chain-of-thought reasoning to the query."""
        
        # Get relevant knowledge from multiple sources
        relevant_docs = self.knowledge_base.search(query, n_results=5)
        knowledge_context = "\n".join([f"- {doc['content'][:200]}..." for doc in relevant_docs])
        
        # Get memory context if available
        memory_context = context.get('memory_context', {})
        relevant_memories = memory_context.get('relevant_memories', [])
        current_context_info = memory_context.get('current_context', 'No context selected')
        
        # Build comprehensive context
        memory_context_str = ""
        if relevant_memories:
            memory_context_str = "\n".join([
                f"- {mem['type']}: {mem['content'][:150]}..."
                for mem in relevant_memories[:5]
            ])
        
        reasoning_prompt = f"""Apply systematic reasoning to address this request.

REQUEST: {query}

RELEVANT KNOWLEDGE:
{knowledge_context}

RELEVANT MEMORIES:
{memory_context_str if memory_context_str else 'None'}

CURRENT CONTEXT:
{current_context_info}

CONVERSATION CONTEXT:
{context.get('conversation_context', 'None')}

Use the following reasoning structure:

1. **UNDERSTANDING**: 
   - What exactly is being asked?
   - What are the key components of this problem?
   - What context or constraints should I consider?

2. **KNOWLEDGE APPLICATION**:
   - What relevant knowledge do I have?
   - Which frameworks, values, or preferences apply?
   - What similar problems have I seen?

3. **REASONING CHAIN**:
   - Step 1: [First logical step with reasoning]
   - Step 2: [Second step building on the first]
   - Step 3: [Continue the chain...]
   - [Add more steps as needed]

4. **SYNTHESIS**:
   - Combine insights from all steps
   - Consider alternative approaches
   - Identify potential issues or gaps

5. **CONCLUSION**:
   - Clear, actionable response
   - Next steps if applicable
   - Confidence level in the solution

Think through each step carefully and show your reasoning process."""

        return self.ollama_client.generate(
            prompt=reasoning_prompt,
            system_prompt="You are an expert reasoner who thinks through problems systematically. Always show your thinking process."
        )

class ReviewerAgent(BaseAgent):
    """Agent that reviews and critiques responses for quality and alignment."""
    
    def __init__(self, knowledge_base: KnowledgeBase, ollama_client: OllamaClient):
        super().__init__(
            name="Reviewer Agent", 
            description="Reviews responses for quality, completeness, and value alignment",
            knowledge_base=knowledge_base,
            ollama_client=ollama_client
        )
        
    def can_handle(self, query: str) -> float:
        # Reviewer doesn't handle initial queries directly
        return 0.0
        
    def process(self, query: str, context: Dict[str, Any]) -> str:
        # This shouldn't be called directly
        return "Reviewer agent should not process queries directly."
        
    def review_response(self, original_query: str, response: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Review a response for quality and suggest improvements."""
        
        # Get user's values for alignment check
        values = self.knowledge_base.search(original_query, n_results=5, filter_metadata={"type": "value"})
        values_context = "\n".join([f"- {v['metadata']['name']}: {v['content']}" for v in values])
        
        review_prompt = f"""Review this response for quality, completeness, and alignment with user values.

ORIGINAL QUERY: {original_query}

RESPONSE TO REVIEW:
{response}

USER VALUES:
{values_context}

CONVERSATION CONTEXT:
{context.get('conversation_context', 'None')}

Provide a structured review:

1. **QUALITY ASSESSMENT**:
   - Is the response clear and well-structured?
   - Does it fully address the original query?
   - Are there any logical gaps or inconsistencies?

2. **VALUES ALIGNMENT**:
   - Does the response align with the user's stated values?
   - Are there any conflicts or concerns?
   - How could alignment be improved?

3. **COMPLETENESS**:
   - What aspects are covered well?
   - What might be missing or needs expansion?
   - Are there important considerations overlooked?

4. **IMPROVEMENT SUGGESTIONS**:
   - Specific recommendations for enhancement
   - Alternative approaches to consider
   - Additional information that would be helpful

5. **SCORE & RECOMMENDATION**:
   - Quality score (1-10)
   - Recommend: ACCEPT, REVISE, or REDO
   - Brief justification

Be constructive and specific in your feedback."""

        review_result = self.ollama_client.generate(
            prompt=review_prompt,
            system_prompt="You are a thorough reviewer focused on quality and value alignment. Be constructive but critical."
        )
        
        # Extract recommendation
        recommendation = "ACCEPT"  # Default
        if "REVISE" in review_result.upper():
            recommendation = "REVISE"
        elif "REDO" in review_result.upper():
            recommendation = "REDO"
            
        # Extract score (simple regex)
        score_match = re.search(r'score.*?(\d+)', review_result.lower())
        score = int(score_match.group(1)) if score_match else 7
        
        return {
            "review": review_result,
            "recommendation": recommendation,
            "score": score
        }

class TaskExecutor:
    """Executes plans by running tasks through appropriate agents."""
    
    def __init__(self, agents: Dict[str, BaseAgent], reviewer: ReviewerAgent):
        self.agents = agents
        self.reviewer = reviewer
        
    def execute_plan(self, plan: Plan) -> Dict[str, Any]:
        """Execute a plan by running tasks in dependency order."""
        
        results = {
            "plan": plan,
            "task_results": {},
            "final_synthesis": "",
            "success": False
        }
        
        # Simple dependency resolution (topological sort would be better)
        completed_tasks = set()
        remaining_tasks = plan.tasks.copy()
        
        while remaining_tasks:
            # Find tasks with satisfied dependencies
            ready_tasks = [
                task for task in remaining_tasks 
                if all(dep in completed_tasks for dep in task.dependencies)
            ]
            
            if not ready_tasks:
                logger.error("Circular dependency or missing task detected")
                break
                
            for task in ready_tasks:
                # Execute task
                agent = self.agents.get(task.agent_type)
                if not agent:
                    logger.error(f"No agent available for type: {task.agent_type}")
                    continue
                    
                try:
                    # Execute task with context
                    task_context = {
                        "plan_goal": plan.goal,
                        "completed_tasks": {tid: results["task_results"][tid] for tid in completed_tasks},
                        **plan.context
                    }
                    
                    task.status = "in_progress"
                    result = agent.process(task.description, task_context)
                    
                    # Review the result
                    review = self.reviewer.review_response(task.description, result, task_context)
                    
                    task.result = result
                    task.status = "completed"
                    
                    results["task_results"][task.id] = {
                        "task": task,
                        "result": result,
                        "review": review
                    }
                    
                    completed_tasks.add(task.id)
                    
                except Exception as e:
                    logger.error(f"Error executing task {task.id}: {e}")
                    task.status = "failed"
                    
                remaining_tasks.remove(task)
                
        # Synthesize final result
        results["final_synthesis"] = self._synthesize_results(plan, results["task_results"])
        results["success"] = len(completed_tasks) == len(plan.tasks)
        
        return results
        
    def _synthesize_results(self, plan: Plan, task_results: Dict[str, Any]) -> str:
        """Synthesize task results into a final cohesive response."""
        
        # Prepare synthesis context
        task_summaries = []
        for task_id, result_data in task_results.items():
            task = result_data["task"]
            result = result_data["result"]
            task_summaries.append(f"**{task.description}**:\n{result[:500]}...")
            
        synthesis_prompt = f"""Synthesize these task results into a cohesive, comprehensive response to the original goal.

ORIGINAL GOAL: {plan.goal}

TASK RESULTS:
{chr(10).join(task_summaries)}

Create a unified response that:
1. Directly addresses the original goal
2. Integrates insights from all completed tasks
3. Provides clear, actionable guidance
4. Maintains logical flow and coherence
5. Highlights key takeaways and next steps

Be comprehensive but concise. Focus on practical value."""

        # Use reasoning agent for synthesis (it's good at integration)
        reasoning_agent = self.agents.get("reasoning")
        if reasoning_agent:
            return reasoning_agent.ollama_client.generate(
                prompt=synthesis_prompt,
                system_prompt="You are an expert at synthesizing complex information into clear, actionable guidance."
            )
        else:
            return "Synthesis agent not available."