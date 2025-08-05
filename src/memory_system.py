from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import json
import logging
from enum import Enum
from src.knowledge_base import KnowledgeBase
from src.ollama_client import OllamaClient
from src.config import settings

logger = logging.getLogger(__name__)

class MemoryType(Enum):
    PERSONAL = "personal"           # Personal info, preferences, background
    PROJECT = "project"             # Project-specific context and history
    INSIGHT = "insight"             # Key insights and learnings
    GOAL = "goal"                   # Goals and objectives
    DECISION = "decision"           # Important decisions made
    RELATIONSHIP = "relationship"   # People and relationships
    SKILL = "skill"                # Skills and competencies
    EXPERIENCE = "experience"       # Past experiences and lessons

@dataclass
class Memory:
    """Represents a single memory item."""
    id: str
    memory_type: MemoryType
    content: str
    context: str
    importance: int  # 1-10 scale
    created_at: datetime
    last_accessed: datetime
    access_count: int
    tags: Set[str]
    projects: Set[str]
    metadata: Dict[str, Any]
    
    def to_dict(self):
        data = asdict(self)
        data['memory_type'] = self.memory_type.value
        data['created_at'] = self.created_at.isoformat()
        data['last_accessed'] = self.last_accessed.isoformat()
        data['tags'] = list(self.tags)
        data['projects'] = list(self.projects)
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        data['memory_type'] = MemoryType(data['memory_type'])
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['last_accessed'] = datetime.fromisoformat(data['last_accessed'])
        data['tags'] = set(data['tags'])
        data['projects'] = set(data['projects'])
        return cls(**data)

@dataclass
class Context:
    """Represents an operating context/project."""
    id: str
    name: str
    description: str
    context_type: str  # personal_brand, product_development, self_improvement, etc.
    goals: List[str]
    key_people: List[str]
    current_focus: str
    status: str  # active, paused, completed, archived
    created_at: datetime
    last_used: datetime
    metadata: Dict[str, Any]
    
    def to_dict(self):
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['last_used'] = self.last_used.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['last_used'] = datetime.fromisoformat(data['last_used'])
        return cls(**data)

class MemorySystem:
    """Manages persistent memory and context switching."""
    
    def __init__(self, knowledge_base: KnowledgeBase, ollama_client: OllamaClient):
        self.knowledge_base = knowledge_base
        self.ollama_client = ollama_client
        self.memory_dir = settings.conversation_history_dir.parent / "memory"
        self.contexts_dir = self.memory_dir / "contexts"
        self.memories_file = self.memory_dir / "memories.json"
        
        # Create directories
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.contexts_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing data
        self.memories: Dict[str, Memory] = self._load_memories()
        self.contexts: Dict[str, Context] = self._load_contexts()
        self.current_context: Optional[Context] = None
        
        # Initialize with basic contexts if none exist
        if not self.contexts:
            self._create_default_contexts()
            
    def _load_memories(self) -> Dict[str, Memory]:
        """Load memories from storage."""
        if not self.memories_file.exists():
            return {}
            
        try:
            with open(self.memories_file, 'r') as f:
                data = json.load(f)
                return {
                    mem_id: Memory.from_dict(mem_data) 
                    for mem_id, mem_data in data.items()
                }
        except Exception as e:
            logger.error(f"Error loading memories: {e}")
            return {}
            
    def _save_memories(self):
        """Save memories to storage."""
        try:
            data = {mem_id: mem.to_dict() for mem_id, mem in self.memories.items()}
            with open(self.memories_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving memories: {e}")
            
    def _load_contexts(self) -> Dict[str, Context]:
        """Load contexts from storage."""
        contexts = {}
        
        for context_file in self.contexts_dir.glob("*.json"):
            try:
                with open(context_file, 'r') as f:
                    data = json.load(f)
                    context = Context.from_dict(data)
                    contexts[context.id] = context
            except Exception as e:
                logger.error(f"Error loading context {context_file}: {e}")
                
        return contexts
        
    def _save_context(self, context: Context):
        """Save a context to storage."""
        try:
            context_file = self.contexts_dir / f"{context.id}.json"
            with open(context_file, 'w') as f:
                json.dump(context.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Error saving context {context.id}: {e}")
            
    def _create_default_contexts(self):
        """Create default operating contexts."""
        default_contexts = [
            {
                "id": "personal_brand",
                "name": "Personal Brand Building",
                "description": "Building and maintaining personal brand, content creation, networking",
                "context_type": "personal_brand",
                "goals": ["Increase visibility", "Share expertise", "Build network"],
                "key_people": [],
                "current_focus": "Getting started"
            },
            {
                "id": "product_dev",
                "name": "Product Development",
                "description": "Working on product ideas, development, and improvement",
                "context_type": "product_development", 
                "goals": ["Build innovative products", "Solve user problems", "Generate value"],
                "key_people": [],
                "current_focus": "Ideation phase"
            },
            {
                "id": "self_improvement",
                "name": "Self Improvement",
                "description": "Personal development, learning, skill building",
                "context_type": "self_improvement",
                "goals": ["Continuous learning", "Skill development", "Personal growth"],
                "key_people": [],
                "current_focus": "Identifying areas for growth"
            }
        ]
        
        now = datetime.now()
        for ctx_data in default_contexts:
            context = Context(
                **ctx_data,
                status="active",
                created_at=now,
                last_used=now,
                metadata={}
            )
            self.contexts[context.id] = context
            self._save_context(context)
            
    def switch_context(self, context_id: str) -> bool:
        """Switch to a different operating context."""
        if context_id not in self.contexts:
            return False
            
        self.current_context = self.contexts[context_id]
        self.current_context.last_used = datetime.now()
        self._save_context(self.current_context)
        
        logger.info(f"Switched to context: {self.current_context.name}")
        return True
        
    def create_context(self, name: str, description: str, context_type: str, goals: List[str] = None) -> Context:
        """Create a new operating context."""
        context_id = name.lower().replace(" ", "_").replace("-", "_")
        
        # Ensure unique ID
        counter = 1
        original_id = context_id
        while context_id in self.contexts:
            context_id = f"{original_id}_{counter}"
            counter += 1
            
        now = datetime.now()
        context = Context(
            id=context_id,
            name=name,
            description=description,
            context_type=context_type,
            goals=goals or [],
            key_people=[],
            current_focus="Getting started",
            status="active",
            created_at=now,
            last_used=now,
            metadata={}
        )
        
        self.contexts[context_id] = context
        self._save_context(context)
        
        return context
        
    def add_memory(self, content: str, memory_type: MemoryType, 
                   importance: int = 5, tags: Set[str] = None,
                   context_override: str = None) -> Memory:
        """Add a new memory item."""
        
        memory_id = f"mem_{len(self.memories)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Determine context
        current_context_id = context_override or (self.current_context.id if self.current_context else "general")
        projects = {current_context_id} if current_context_id != "general" else set()
        
        now = datetime.now()
        memory = Memory(
            id=memory_id,
            memory_type=memory_type,
            content=content,
            context=current_context_id,
            importance=importance,
            created_at=now,
            last_accessed=now,
            access_count=1,
            tags=tags or set(),
            projects=projects,
            metadata={}
        )
        
        self.memories[memory_id] = memory
        self._save_memories()
        
        # Also add to knowledge base for semantic search
        self.knowledge_base.add_document(
            content=f"Memory ({memory_type.value}): {content}",
            metadata={
                "type": "memory",
                "memory_type": memory_type.value,
                "context": current_context_id,
                "importance": importance,
                "memory_id": memory_id
            },
            document_id=f"memory_{memory_id}"
        )
        
        logger.info(f"Added memory: {memory_type.value} in context {current_context_id}")
        return memory
        
    def recall_memories(self, query: str, context_filter: str = None, 
                       memory_types: List[MemoryType] = None,
                       limit: int = 10) -> List[Memory]:
        """Recall relevant memories based on query and filters."""
        
        # Use current context if no filter specified
        if context_filter is None and self.current_context:
            context_filter = self.current_context.id
            
        # Search knowledge base for relevant memories
        if context_filter:
            filter_metadata = {
                "$and": [
                    {"type": {"$eq": "memory"}},
                    {"context": {"$eq": context_filter}}
                ]
            }
        else:
            filter_metadata = {"type": {"$eq": "memory"}}
            
        search_results = self.knowledge_base.search(
            query, 
            n_results=limit * 2,  # Get more to filter
            filter_metadata=filter_metadata
        )
        
        # Convert to Memory objects and apply filters
        recalled_memories = []
        for result in search_results:
            memory_id = result["metadata"].get("memory_id")
            if memory_id and memory_id in self.memories:
                memory = self.memories[memory_id]
                
                # Apply memory type filter
                if memory_types and memory.memory_type not in memory_types:
                    continue
                    
                # Update access info
                memory.last_accessed = datetime.now()
                memory.access_count += 1
                
                recalled_memories.append(memory)
                
                if len(recalled_memories) >= limit:
                    break
                    
        # Save updated access info
        if recalled_memories:
            self._save_memories()
            
        return recalled_memories
        
    def get_context_summary(self, context_id: str = None) -> str:
        """Get a summary of the current or specified context."""
        
        context = self.contexts.get(context_id) if context_id else self.current_context
        if not context:
            return "No context selected"
            
        # Get recent memories for this context
        recent_memories = self.recall_memories(
            "recent activities goals progress", 
            context_filter=context.id,
            limit=5
        )
        
        memory_summary = "\n".join([
            f"- {mem.memory_type.value}: {mem.content[:100]}..."
            for mem in recent_memories
        ])
        
        summary = f"""**Context: {context.name}**

**Description**: {context.description}

**Current Focus**: {context.current_focus}

**Goals**:
{chr(10).join([f"- {goal}" for goal in context.goals])}

**Recent Context**:
{memory_summary if memory_summary else "No recent memories"}

**Status**: {context.status}
**Last Used**: {context.last_used.strftime('%Y-%m-%d %H:%M')}
"""
        
        return summary
        
    def update_context_focus(self, new_focus: str, context_id: str = None):
        """Update the current focus for a context."""
        context = self.contexts.get(context_id) if context_id else self.current_context
        if context:
            context.current_focus = new_focus
            context.last_used = datetime.now()
            self._save_context(context)
            
    def extract_memories_from_conversation(self, conversation_text: str) -> List[Memory]:
        """Extract important memories from conversation text using AI."""
        
        extraction_prompt = f"""Analyze this conversation and extract important information that should be remembered for future interactions.

CONVERSATION:
{conversation_text}

CURRENT CONTEXT: {self.current_context.name if self.current_context else "General"}

Extract memories in the following categories:
- PERSONAL: Personal information, preferences, background
- PROJECT: Project-specific information, progress, decisions  
- INSIGHT: Key insights, learnings, "aha" moments
- GOAL: Goals, objectives, aspirations mentioned
- DECISION: Important decisions or choices made
- RELATIONSHIP: People mentioned, relationships, contacts
- SKILL: Skills, competencies, areas of expertise
- EXPERIENCE: Past experiences, lessons learned

For each memory, provide:
1. Type (from above categories)
2. Content (concise but complete)
3. Importance (1-10)
4. Tags (relevant keywords)

Format as JSON:
```json
[
  {{
    "type": "PERSONAL",
    "content": "User prefers direct communication style",
    "importance": 7,
    "tags": ["communication", "preferences"]
  }}
]
```

Only extract truly important information worth remembering."""

        try:
            response = self.ollama_client.generate(
                prompt=extraction_prompt,
                system_prompt="You are an expert at identifying and extracting important information from conversations for long-term memory."
            )
            
            # Extract JSON from response
            import re
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if not json_match:
                return []
                
            memory_data = json.loads(json_match.group())
            
            # Create Memory objects
            extracted_memories = []
            for item in memory_data:
                try:
                    memory_type = MemoryType(item['type'].lower())
                    memory = self.add_memory(
                        content=item['content'],
                        memory_type=memory_type,
                        importance=item.get('importance', 5),
                        tags=set(item.get('tags', []))
                    )
                    extracted_memories.append(memory)
                except Exception as e:
                    logger.error(f"Error creating memory from extraction: {e}")
                    
            return extracted_memories
            
        except Exception as e:
            logger.error(f"Error extracting memories: {e}")
            return []
            
    def get_context_for_query(self, query: str) -> Dict[str, Any]:
        """Get relevant context and memories for a query."""
        
        context_info = {
            "current_context": self.get_context_summary() if self.current_context else "No context selected",
            "relevant_memories": [],
            "context_history": []
        }
        
        # Get relevant memories
        relevant_memories = self.recall_memories(query, limit=8)
        context_info["relevant_memories"] = [
            {
                "type": mem.memory_type.value,
                "content": mem.content,
                "importance": mem.importance,
                "context": mem.context
            }
            for mem in relevant_memories
        ]
        
        # Get context history if we have a current context
        if self.current_context:
            context_memories = self.recall_memories(
                "progress updates decisions",
                context_filter=self.current_context.id,
                limit=5
            )
            context_info["context_history"] = [
                f"{mem.created_at.strftime('%Y-%m-%d')}: {mem.content}"
                for mem in sorted(context_memories, key=lambda x: x.created_at, reverse=True)
            ]
            
        return context_info