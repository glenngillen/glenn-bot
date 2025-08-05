from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import json
import logging
from src.memory_system import MemorySystem, MemoryType
from src.knowledge_base import KnowledgeBase
from src.ollama_client import OllamaClient
from src.config import settings

logger = logging.getLogger(__name__)

@dataclass
class Quote:
    """Represents an inspirational quote."""
    id: str
    text: str
    author: str
    source: Optional[str]
    context: str  # Why this quote resonates with you
    tags: Set[str]
    category: str  # inspiration, wisdom, leadership, creativity, etc.
    created_at: datetime
    last_reflected: Optional[datetime]
    reflection_count: int
    importance: int  # 1-10 scale
    projects: Set[str]  # Which contexts/projects this relates to
    
    def to_dict(self):
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['last_reflected'] = self.last_reflected.isoformat() if self.last_reflected else None
        data['tags'] = list(self.tags)
        data['projects'] = list(self.projects)
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['last_reflected'] = datetime.fromisoformat(data['last_reflected']) if data['last_reflected'] else None
        data['tags'] = set(data['tags'])
        data['projects'] = set(data['projects'])
        return cls(**data)

class QuotesSystem:
    """Manages inspirational quotes and integrates them with memory and knowledge systems."""
    
    def __init__(self, memory_system: MemorySystem, knowledge_base: KnowledgeBase, ollama_client: OllamaClient):
        self.memory_system = memory_system
        self.knowledge_base = knowledge_base
        self.ollama_client = ollama_client
        
        self.quotes_dir = settings.conversation_history_dir.parent / "quotes"
        self.quotes_file = self.quotes_dir / "quotes.json"
        
        # Create directory
        self.quotes_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing quotes
        self.quotes: Dict[str, Quote] = self._load_quotes()
        
    def _load_quotes(self) -> Dict[str, Quote]:
        """Load quotes from storage."""
        if not self.quotes_file.exists():
            return {}
            
        try:
            with open(self.quotes_file, 'r') as f:
                data = json.load(f)
                return {
                    quote_id: Quote.from_dict(quote_data) 
                    for quote_id, quote_data in data.items()
                }
        except Exception as e:
            logger.error(f"Error loading quotes: {e}")
            return {}
            
    def _save_quotes(self):
        """Save quotes to storage."""
        try:
            data = {quote_id: quote.to_dict() for quote_id, quote in self.quotes.items()}
            with open(self.quotes_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving quotes: {e}")
            
    def add_quote(self, text: str, author: str, context: str, 
                  source: Optional[str] = None, category: str = "inspiration",
                  importance: int = 5, tags: Set[str] = None) -> Quote:
        """Add a new quote."""
        
        quote_id = f"quote_{len(self.quotes)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Get current context for project association
        current_context_id = self.memory_system.current_context.id if self.memory_system.current_context else "general"
        projects = {current_context_id} if current_context_id != "general" else set()
        
        now = datetime.now()
        quote = Quote(
            id=quote_id,
            text=text.strip(),
            author=author.strip(),
            source=source,
            context=context.strip(),
            tags=tags or set(),
            category=category,
            created_at=now,
            last_reflected=None,
            reflection_count=0,
            importance=importance,
            projects=projects
        )
        
        self.quotes[quote_id] = quote
        self._save_quotes()
        
        # Add to knowledge base for semantic search
        quote_content = f"""Quote: "{text}"
Author: {author}
Category: {category}
Context: {context}
Why this resonates: {context}"""

        if source:
            quote_content += f"\nSource: {source}"
            
        if tags:
            quote_content += f"\nTags: {', '.join(tags)}"
        
        self.knowledge_base.add_document(
            content=quote_content,
            metadata={
                "type": "quote",
                "author": author,
                "category": category,
                "importance": importance,
                "quote_id": quote_id,
                "source": "quotes_system"
            },
            document_id=f"quote_{quote_id}"
        )
        
        # Add as memory too
        self.memory_system.add_memory(
            f"Added inspiring quote by {author}: \"{text[:100]}...\" - {context}",
            MemoryType.INSIGHT,
            importance=importance,
            tags=tags or set()
        )
        
        logger.info(f"Added quote by {author}")
        return quote
        
    def get_random_quote(self, category: Optional[str] = None, 
                        context_filter: Optional[str] = None) -> Optional[Quote]:
        """Get a random quote for reflection."""
        import random
        
        # Filter quotes
        candidates = list(self.quotes.values())
        
        if category:
            candidates = [q for q in candidates if q.category == category]
            
        if context_filter:
            candidates = [q for q in candidates if context_filter in q.projects]
            
        if not candidates:
            return None
            
        # Prefer quotes that haven't been reflected on recently
        now = datetime.now()
        unvisited = [q for q in candidates if q.last_reflected is None]
        
        if unvisited:
            quote = random.choice(unvisited)
        else:
            # Weight by time since last reflection
            weights = []
            for q in candidates:
                days_since = (now - q.last_reflected).days if q.last_reflected else 365
                weights.append(max(1, days_since))
            
            quote = random.choices(candidates, weights=weights)[0]
            
        # Update reflection stats
        quote.last_reflected = now
        quote.reflection_count += 1
        self._save_quotes()
        
        return quote
        
    def search_quotes(self, query: str, limit: int = 5) -> List[Quote]:
        """Search quotes semantically."""
        
        # Search knowledge base for quotes
        search_results = self.knowledge_base.search(
            query, 
            n_results=limit,
            filter_metadata={"type": {"$eq": "quote"}}
        )
        
        # Convert back to Quote objects
        found_quotes = []
        for result in search_results:
            quote_id = result["metadata"].get("quote_id")
            if quote_id and quote_id in self.quotes:
                found_quotes.append(self.quotes[quote_id])
                
        return found_quotes
        
    def get_quotes_by_category(self, category: str) -> List[Quote]:
        """Get all quotes in a category."""
        return [q for q in self.quotes.values() if q.category == category]
        
    def get_quotes_by_author(self, author: str) -> List[Quote]:
        """Get all quotes by an author."""
        return [q for q in self.quotes.values() if author.lower() in q.author.lower()]
        
    def categorize_quote(self, text: str, author: str, context: str) -> Dict[str, Any]:
        """Use AI to suggest category and tags for a quote."""
        
        categorization_prompt = f"""Analyze this quote and suggest how to categorize it.

Quote: "{text}"
Author: {author}
User Context: {context}

Suggest:
1. Category (choose one): inspiration, wisdom, leadership, creativity, productivity, success, relationships, growth, courage, innovation, life, work, entrepreneurship
2. 3-5 relevant tags
3. Importance level (1-10) based on how profound/actionable it seems
4. Brief explanation of why this quote might be valuable

Respond with JSON:
{{
  "category": "category_name",
  "tags": ["tag1", "tag2", "tag3"],
  "importance": 7,
  "explanation": "Why this quote is valuable..."
}}"""

        try:
            response = self.ollama_client.generate(
                prompt=categorization_prompt,
                system_prompt="You are an expert at analyzing inspirational quotes and understanding their value."
            )
            
            # Extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                # Fallback
                return {
                    "category": "inspiration",
                    "tags": ["wisdom"],
                    "importance": 5,
                    "explanation": "Inspirational quote"
                }
                
        except Exception as e:
            logger.error(f"Error categorizing quote: {e}")
            return {
                "category": "inspiration", 
                "tags": ["wisdom"],
                "importance": 5,
                "explanation": "Inspirational quote"
            }
            
    def get_reflection_prompt(self, quote: Quote) -> str:
        """Generate a reflection prompt for a quote."""
        
        current_context = ""
        if self.memory_system.current_context:
            current_context = f"\nCurrent context: {self.memory_system.current_context.name} - {self.memory_system.current_context.current_focus}"
            
        return f"""Quote for Reflection:

"{quote.text}"
— {quote.author}

Why you saved this: {quote.context}
Category: {quote.category}

{current_context}

Reflection questions:
• How does this quote apply to what you're working on now?
• What specific action could you take inspired by this wisdom?
• How does this connect to your personal principles?
• What would change if you truly embodied this message?

Take a moment to reflect on how this quote speaks to your current situation."""

    def get_stats(self) -> Dict[str, Any]:
        """Get quotes statistics."""
        total_quotes = len(self.quotes)
        
        if total_quotes == 0:
            return {"total_quotes": 0}
            
        # Category breakdown
        categories = {}
        authors = {}
        importance_sum = 0
        
        for quote in self.quotes.values():
            categories[quote.category] = categories.get(quote.category, 0) + 1
            authors[quote.author] = authors.get(quote.author, 0) + 1
            importance_sum += quote.importance
            
        avg_importance = importance_sum / total_quotes
        
        # Recent activity
        now = datetime.now()
        recent_quotes = len([q for q in self.quotes.values() if (now - q.created_at).days <= 7])
        
        return {
            "total_quotes": total_quotes,
            "categories": categories,
            "top_authors": dict(sorted(authors.items(), key=lambda x: x[1], reverse=True)[:5]),
            "average_importance": avg_importance,
            "recent_quotes": recent_quotes,
            "total_reflections": sum(q.reflection_count for q in self.quotes.values())
        }