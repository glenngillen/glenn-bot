import ollama
from typing import List, Dict, Any, Optional
from src.config import settings
import logging

logger = logging.getLogger(__name__)

class OllamaClient:
    def __init__(self):
        self.client = ollama.Client(host=settings.ollama_host)
        self.model = settings.ollama_model
        self.embedding_model = settings.ollama_embedding_model
        
    def generate(
        self, 
        prompt: str, 
        context: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7
    ) -> str:
        """Generate a response using Ollama."""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
            
        if context:
            messages.append({"role": "system", "content": f"Context:\n{context}"})
            
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.client.chat(
                model=self.model,
                messages=messages,
                options={"temperature": temperature}
            )
            return response['message']['content']
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise
            
    def get_embeddings(self, text: str) -> List[float]:
        """Get embeddings for text using Ollama."""
        try:
            response = self.client.embeddings(
                model=self.embedding_model,
                prompt=text
            )
            return response['embedding']
        except Exception as e:
            logger.error(f"Error getting embeddings: {e}")
            raise
            
    def list_models(self) -> List[Dict[str, Any]]:
        """List available models."""
        try:
            return self.client.list()['models']
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            raise