from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import json
import logging
from dataclasses import dataclass, asdict
from src.config import settings

logger = logging.getLogger(__name__)

@dataclass
class Message:
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: datetime
    metadata: Dict[str, Any] = None
    
    def to_dict(self):
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)

class ConversationManager:
    def __init__(self):
        self.conversations_dir = settings.conversation_history_dir
        self.conversations_dir.mkdir(parents=True, exist_ok=True)
        self.current_conversation: List[Message] = []
        self.conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> int:
        """Add a message to the current conversation.

        Returns:
            int: The index of the added message (for use with feedback system)
        """
        message = Message(
            role=role,
            content=content,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        self.current_conversation.append(message)
        self._save_conversation()
        return len(self.current_conversation) - 1

    def get_last_assistant_message(self) -> Optional[tuple]:
        """Get the last assistant message and its index.

        Returns:
            tuple: (index, message) of the last assistant message, or None if not found
        """
        for i in range(len(self.current_conversation) - 1, -1, -1):
            if self.current_conversation[i].role == "assistant":
                return (i, self.current_conversation[i])
        return None

    def get_last_user_message(self) -> Optional[tuple]:
        """Get the last user message and its index.

        Returns:
            tuple: (index, message) of the last user message, or None if not found
        """
        for i in range(len(self.current_conversation) - 1, -1, -1):
            if self.current_conversation[i].role == "user":
                return (i, self.current_conversation[i])
        return None

    def get_message_pair_for_feedback(self) -> Optional[Dict[str, Any]]:
        """Get the last user-assistant message pair for feedback.

        Returns:
            dict with 'user_index', 'user_content', 'assistant_index', 'assistant_content'
            or None if no assistant message exists
        """
        assistant_result = self.get_last_assistant_message()
        if not assistant_result:
            return None

        assistant_index, assistant_msg = assistant_result

        # Find the user message before this assistant message
        user_content = None
        user_index = None
        for i in range(assistant_index - 1, -1, -1):
            if self.current_conversation[i].role == "user":
                user_index = i
                user_content = self.current_conversation[i].content
                break

        return {
            "user_index": user_index,
            "user_content": user_content,
            "assistant_index": assistant_index,
            "assistant_content": assistant_msg.content
        }
        
    def get_conversation_context(self, max_messages: int = 10) -> str:
        """Get recent conversation context as a formatted string."""
        recent_messages = self.current_conversation[-max_messages:]
        context = []
        
        for msg in recent_messages:
            if msg.role != "system":
                context.append(f"{msg.role.capitalize()}: {msg.content}")
                
        return "\n\n".join(context)
    
    def get_messages_for_llm(self, max_messages: int = 10) -> List[Dict[str, str]]:
        """Get messages formatted for LLM consumption."""
        recent_messages = self.current_conversation[-max_messages:]
        return [
            {"role": msg.role, "content": msg.content}
            for msg in recent_messages
        ]
        
    def start_new_conversation(self):
        """Start a new conversation."""
        self.current_conversation = []
        self.conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def _save_conversation(self):
        """Save the current conversation to disk."""
        try:
            filepath = self.conversations_dir / f"{self.conversation_id}.json"
            data = {
                "id": self.conversation_id,
                "started_at": self.current_conversation[0].timestamp.isoformat() if self.current_conversation else None,
                "messages": [msg.to_dict() for msg in self.current_conversation]
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving conversation: {e}")
            
    def load_conversation(self, conversation_id: str):
        """Load a conversation from disk."""
        try:
            filepath = self.conversations_dir / f"{conversation_id}.json"
            with open(filepath, 'r') as f:
                data = json.load(f)
                
            self.conversation_id = data['id']
            self.current_conversation = [
                Message.from_dict(msg) for msg in data['messages']
            ]
            
            logger.info(f"Loaded conversation {conversation_id}")
            
        except Exception as e:
            logger.error(f"Error loading conversation: {e}")
            raise
            
    def list_conversations(self) -> List[Dict[str, Any]]:
        """List all saved conversations."""
        conversations = []
        
        for filepath in sorted(self.conversations_dir.glob("*.json"), reverse=True):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    
                conversations.append({
                    "id": data['id'],
                    "started_at": data.get('started_at'),
                    "message_count": len(data['messages']),
                    "first_message": data['messages'][0]['content'][:50] + "..." if data['messages'] else ""
                })
                
            except Exception as e:
                logger.error(f"Error reading conversation {filepath}: {e}")
                
        return conversations