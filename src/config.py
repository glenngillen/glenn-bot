from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    # Ollama settings
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3:8b"
    ollama_embedding_model: str = "nomic-embed-text"
    
    # ChromaDB settings
    chroma_persist_directory: Path = Path("./data/chroma")
    chroma_collection_name: str = "glenn_knowledge"
    
    # Application settings
    log_level: str = "WARNING"
    conversation_history_dir: Path = Path("./data/conversations")
    knowledge_dir: Path = Path("./knowledge")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()