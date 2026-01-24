# Multi-Agent System

## Overview

Glenn-bot uses a multi-agent architecture where specialized agents handle different aspects of query processing. An orchestrator routes queries to the most appropriate agent(s) based on confidence scoring, with support for planning complex tasks and synthesizing multi-agent responses.

## Key Components

- `src/agents.py`: Base agent class and specialized agents (Framework, Values, Preference, Quotes)
- `src/reasoning_agents.py`: Planning, Reasoning, and Reviewer agents plus TaskExecutor
- `AgentOrchestrator`: Routes queries and manages agent collaboration

## Agents

### Framework Agent
**Purpose**: Applies problem-solving frameworks to structure thinking

**Confidence Triggers**:
- Keywords: "framework", "approach", "methodology", "process", "steps", "structure", "solve", "plan"
- Boosts base score to at least 0.6 if relevant frameworks found; after doubling, confidence caps at 1.0

**Behavior**:
- Searches knowledge base for frameworks matching query
- Constructs prompt with available frameworks
- Asks LLM to apply most relevant framework(s)

### Values Agent
**Purpose**: Ensures responses align with user's values and principles

**Confidence Triggers**:
- Keywords: "values", "principles", "ethics", "beliefs", "important", "priority", "align", "personal"
- Minimum score of 0.3 when no keywords match; note the standard workflow only activates agents with confidence > 0.3, so a base 0.3 does not guarantee execution
- Boosts base score to at least 0.8 for direct values/principles queries; returns `max(score * 2, 0.3)` without a hard cap (can exceed 1.0)

**Behavior**:
- Retrieves relevant values from knowledge base
- Injects values context into system prompt
- Ensures recommendations consider personal principles

### Preference Agent
**Purpose**: Applies user preferences to recommendations

**Confidence Triggers**:
- Keywords: "prefer", "like", "favorite", "recommendation", "suggest", "option", "choice", "best"
- Boosts base score to at least 0.5 if relevant preferences found; after doubling, confidence caps at 1.0

**Behavior**:
- Searches for matching preferences by category
- Provides personalized recommendations
- Considers work style and communication preferences

### Quotes Agent
**Purpose**: Incorporates inspirational quotes and wisdom

**Confidence Triggers**:
- Keywords: "inspire", "motivation", "wisdom", "quote", "advice", "guidance", "mindset", "perspective"
- Base score floor of 0.2 if quotes are available (becomes 0.4 after doubling)

**Behavior**:
- Searches quotes system for relevant quotes
- Incorporates wisdom naturally when appropriate
- Does not force quotes into every response

### Planning Agent
**Purpose**: Breaks down complex problems into structured plans

**Confidence Triggers**:
- Keywords: "build", "create", "develop", "design", "implement", "plan", "strategy", "process", "system", "product", "project", "how to", "step by step", "multiple", "complex", "comprehensive", "end-to-end"
- Boosts base score to at least 0.6 for phrases like "how do i", "help me", "i want to", "i need to"; after doubling, confidence caps at 1.0

**Behavior**:
- Creates structured JSON plans with tasks
- Each task has: id, description, dependencies, agent_type
- Tasks can depend on other tasks
- Planning prompt includes relevant frameworks and values from the knowledge base
- Outputs goal analysis, reasoning, plan structure, and next steps
- Task agent types suggested in the prompt: "framework", "values", "preference", "general" (other agent types are supported by execution)
- Planning prompt asks for per-task `reasoning`, but execution currently ignores that field

### Reasoning Agent
**Purpose**: Applies chain-of-thought reasoning; acts as synthesis fallback

**Confidence Score**: Fixed at 0.4 (lower than specialized agents)

**Behavior**:
- Uses structured reasoning: Understanding → Knowledge Application → Reasoning Chain → Synthesis → Conclusion
- Synthesizes multiple agent responses when needed
- Falls back handler when other agents have low confidence
- Pulls relevant knowledge base documents and memory context into the prompt

### Reviewer Agent
**Purpose**: Reviews responses for quality and value alignment

**Confidence Score**: 0.0 (never handles queries directly)

**Behavior**:
- Evaluates responses against quality criteria
- Checks alignment with user values
- Returns structured review with score (1-10) and recommendation (ACCEPT/REVISE/REDO)
- Recommendation parsing is substring-based: REVISE is checked before REDO, and score extraction uses a simple `score.*?(\d+)` regex with a default score of 7 if parsing fails

## Agent Orchestration

### Query Processing Flow

```
Query → AgentOrchestrator.process_query()
            ↓
    Check planning_score > 0.6?
        ↓                    ↓
       YES                   NO
        ↓                    ↓
    Planning            Standard
    Workflow            Workflow
```

### Planning Workflow
1. Planning Agent creates structured plan
2. Extract Plan object from response (JSON parsing with regex); on failure, create a single-task fallback plan
3. TaskExecutor runs tasks in dependency order (including the fallback plan)
4. Each task result is reviewed by Reviewer Agent
5. Results synthesized into final response
6. Final response reviewed; falls back to Reasoning Agent only when recommendation is REDO and score < 6

### Standard Workflow
1. Score all agents against query
2. Filter agents with confidence > 0.3 (planning agent excluded from standard workflow scoring)
3. If top agent > 0.7 confidence:
   - Use as primary agent
   - Review response; enhance with Reasoning if score < 7
4. Otherwise:
   - Collect responses from top 3 agents
   - Reasoning Agent synthesizes perspectives

## Data Structures

### Task
```python
@dataclass
class Task:
    id: str
    description: str
    dependencies: List[str]  # Task IDs this depends on
    agent_type: str          # "framework"|"values"|"preference"|"quotes"|"planning"|"reasoning"|"general"
    status: str = "pending"  # "pending"|"in_progress"|"completed"|"failed"
    result: Optional[str] = None
    reasoning: Optional[str] = None
```

### Plan
```python
@dataclass
class Plan:
    goal: str
    tasks: List[Task]
    context: Dict[str, Any]
```

### Review Result
```python
{
    "review": str,           # Full review text
    "recommendation": str,   # "ACCEPT"|"REVISE"|"REDO"
    "score": int            # 1-10
}
```

## Edge Cases

- **No agents confident**: Falls back to Reasoning Agent if all scores are <= 0.3 (unlikely because Reasoning Agent returns 0.4)
- **Circular task dependencies**: Detected and logged; execution stops
- **Missing agent type**: Logged as error; task stays in the ready queue (status remains pending), which can cause the executor loop to repeat indefinitely
- **Failed tasks with dependents**: Downstream tasks never become ready; the executor stops with the same "circular dependency or missing task" log path
- **Plan extraction fails**: Uses a single-task fallback plan executed by the Reasoning Agent via TaskExecutor
- **Low quality response**: Reasoning Agent enhances or redoes response

## Dependencies

- `KnowledgeBase`: For semantic search of frameworks, values, preferences
- `OllamaClient`: For LLM generation
- `QuotesSystem`: Optional, for quotes integration
