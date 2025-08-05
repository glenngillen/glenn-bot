import requests
from bs4 import BeautifulSoup
import json
from typing import Dict, Any, Optional
from pathlib import Path
from urllib.parse import urlparse
import logging
from markdownify import markdownify as md
from src.knowledge_base import KnowledgeBase
from src.ollama_client import OllamaClient
from src.config import settings

logger = logging.getLogger(__name__)

class DocumentIngestionTool:
    def __init__(self, knowledge_base: KnowledgeBase, ollama_client: OllamaClient):
        self.knowledge_base = knowledge_base
        self.ollama_client = ollama_client
        
    def fetch_webpage(self, url: str) -> str:
        """Fetch and extract text content from a webpage."""
        try:
            # Handle Google Docs URLs
            if 'docs.google.com' in url:
                # Convert Google Docs URL to export format
                if '/edit' in url:
                    doc_id = url.split('/d/')[1].split('/')[0]
                    url = f"https://docs.google.com/document/d/{doc_id}/export?format=txt"
                    
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            # For Google Docs export, content is already plain text
            if 'docs.google.com' in url and 'export?format=txt' in url:
                return response.text.strip()
                
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
                
            # Convert to markdown for better formatting
            content = md(str(soup), heading_style="ATX")
            
            return content.strip()
            
        except Exception as e:
            logger.error(f"Error fetching webpage {url}: {e}")
            raise
            
    def classify_content(self, content: str, user_context: str) -> Dict[str, Any]:
        """Use LLM to classify the content and determine how to store it."""
        
        classification_prompt = f"""Analyze this content and determine how it should be stored in a knowledge base.

Content:
{content[:2000]}{"..." if len(content) > 2000 else ""}

User context: {user_context}

Classify this content as one of:
1. "value" - Core personal/organizational values or principles
2. "framework" - Problem-solving methodology, process, or structured approach  
3. "preference" - Personal preferences, recommendations, or settings
4. "reference" - General reference material, facts, or documentation

Respond with a JSON object containing:
- "type": one of the above types
- "name": a concise name for this content
- "category": a category (if applicable)
- "description": a brief description of what this content contains
- "key_points": 3-5 key points or takeaways from the content

Example response:
{{
  "type": "framework",
  "name": "Getting Things Done",
  "category": "productivity", 
  "description": "David Allen's productivity methodology",
  "key_points": ["Capture everything", "Clarify next actions", "Organize by context", "Regular reviews", "Engage with confidence"]
}}"""

        try:
            response = self.ollama_client.generate(
                prompt=classification_prompt,
                system_prompt="You are an expert at analyzing and categorizing knowledge content. Always respond with valid JSON."
            )
            
            # Extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                # Fallback classification
                return {
                    "type": "reference",
                    "name": "Web Content",
                    "category": "general",
                    "description": "Content from web source",
                    "key_points": ["Web-based content"]
                }
                
        except Exception as e:
            logger.error(f"Error classifying content: {e}")
            # Fallback classification
            return {
                "type": "reference",
                "name": "Web Content", 
                "category": "general",
                "description": "Content from web source",
                "key_points": ["Web-based content"]
            }
            
    def add_web_content(self, url: str, user_context: str = "") -> Dict[str, Any]:
        """Fetch web content and add it to the knowledge base."""
        
        # Fetch content
        content = self.fetch_webpage(url)
        
        # Classify content
        classification = self.classify_content(content, user_context)
        
        # Prepare content for storage
        formatted_content = f"""Source: {url}
Context: {user_context}

{classification['description']}

Key Points:
{chr(10).join([f"- {point}" for point in classification['key_points']])}

Content:
{content[:3000]}{"..." if len(content) > 3000 else ""}
"""

        # Prepare metadata
        metadata = {
            "type": classification["type"],
            "name": classification["name"],
            "source": "web",
            "url": url,
            "user_context": user_context
        }
        
        if classification.get("category"):
            metadata["category"] = classification["category"]
            
        # Add to knowledge base
        self.knowledge_base.add_document(formatted_content, metadata)
        
        # Save to appropriate file if it's a structured type
        self._save_to_file(classification, url, user_context, content)
        
        return classification
        
    def _save_to_file(self, classification: Dict[str, Any], url: str, user_context: str, content: str):
        """Save structured content to appropriate JSON files."""
        
        knowledge_dir = settings.knowledge_dir
        
        if classification["type"] == "framework":
            frameworks_dir = knowledge_dir / "frameworks"
            frameworks_dir.mkdir(exist_ok=True)
            
            framework_data = {
                "name": classification["name"],
                "category": classification.get("category", "general"),
                "description": classification["description"],
                "source": url,
                "user_context": user_context,
                "steps": classification["key_points"]
            }
            
            filename = f"{classification['name'].lower().replace(' ', '_')}.json"
            filepath = frameworks_dir / filename
            
            with open(filepath, 'w') as f:
                json.dump(framework_data, f, indent=2)
                
        elif classification["type"] == "value":
            values_file = knowledge_dir / "values.json"
            
            # Load existing values or create new
            if values_file.exists():
                with open(values_file, 'r') as f:
                    values_data = json.load(f)
            else:
                values_data = {"values": []}
                
            # Add new value
            new_value = {
                "name": classification["name"],
                "description": classification["description"],
                "source": url,
                "user_context": user_context,
                "key_points": classification["key_points"]
            }
            
            values_data["values"].append(new_value)
            
            with open(values_file, 'w') as f:
                json.dump(values_data, f, indent=2)
                
        elif classification["type"] == "preference":
            preferences_file = knowledge_dir / "preferences.json"
            
            # Load existing preferences or create new
            if preferences_file.exists():
                with open(preferences_file, 'r') as f:
                    prefs_data = json.load(f)
            else:
                prefs_data = {}
                
            category = classification.get("category", "general")
            if category not in prefs_data:
                prefs_data[category] = {}
                
            prefs_data[category][classification["name"]] = {
                "description": classification["description"],
                "source": url,
                "user_context": user_context,
                "points": classification["key_points"]
            }
            
            with open(preferences_file, 'w') as f:
                json.dump(prefs_data, f, indent=2)
                
        logger.info(f"Saved {classification['type']} '{classification['name']}' to file")
        
    def add_text_content(self, content: str, user_context: str, content_name: str = "Manual Entry") -> Dict[str, Any]:
        """Add manually provided text content to the knowledge base."""
        
        # Classify content
        classification = self.classify_content(content, user_context)
        classification["name"] = content_name  # Override with user-provided name
        
        # Prepare content for storage
        formatted_content = f"""Source: Manual Entry
Context: {user_context}

{classification['description']}

Key Points:
{chr(10).join([f"- {point}" for point in classification['key_points']])}

Content:
{content}
"""

        # Prepare metadata
        metadata = {
            "type": classification["type"],
            "name": classification["name"],
            "source": "manual",
            "user_context": user_context
        }
        
        if classification.get("category"):
            metadata["category"] = classification["category"]
            
        # Add to knowledge base
        self.knowledge_base.add_document(formatted_content, metadata)
        
        # Save to appropriate file
        self._save_to_file(classification, "manual_entry", user_context, content)
        
        return classification