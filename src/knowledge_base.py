import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import logging
import json
from src.config import settings
from src.ollama_client import OllamaClient

logger = logging.getLogger(__name__)

class KnowledgeBase:
    def __init__(self):
        self.persist_directory = settings.chroma_persist_directory
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        
        self.collection = self.client.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        self.ollama_client = OllamaClient()
        
    def add_document(
        self, 
        content: str, 
        metadata: Dict[str, Any],
        document_id: Optional[str] = None
    ):
        """Add a document to the knowledge base."""
        try:
            embedding = self.ollama_client.get_embeddings(content)
            
            self.collection.add(
                documents=[content],
                embeddings=[embedding],
                metadatas=[metadata],
                ids=[document_id or f"doc_{len(self.collection.get()['ids'])}"]
            )
            logger.info(f"Added document with metadata: {metadata}")
        except Exception as e:
            logger.error(f"Error adding document: {e}")
            raise
            
    def search(
        self, 
        query: str, 
        n_results: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search the knowledge base."""
        try:
            query_embedding = self.ollama_client.get_embeddings(query)
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=filter_metadata
            )
            
            return [
                {
                    "content": doc,
                    "metadata": meta,
                    "distance": dist,
                    "id": doc_id
                }
                for doc, meta, dist, doc_id in zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0],
                    results['ids'][0]
                )
            ]
        except Exception as e:
            logger.error(f"Error searching knowledge base: {e}")
            raise
            
    def load_knowledge_files(self, knowledge_dir: Path):
        """Load knowledge files from a directory."""
        knowledge_dir = Path(knowledge_dir)
        
        # Check if knowledge files have been loaded by looking for a specific marker
        existing_docs = self.collection.get()
        
        # Look for any knowledge file items
        has_knowledge_files = False
        for metadata in existing_docs['metadatas']:
            if metadata and metadata.get('source') == 'knowledge_files':
                has_knowledge_files = True
                break
        
        # If we already have knowledge files loaded, skip
        if has_knowledge_files:
            logger.info("Knowledge files already loaded")
            return
        
        added_count = 0
        
        # Load values and principles
        values_file = knowledge_dir / "values.json"
        if values_file.exists():
            with open(values_file, 'r') as f:
                values = json.load(f)
                for value in values.get("values", []):
                    # Build comprehensive content
                    content_parts = [f"Value: {value['name']}", value['description']]
                    
                    # Add key points if available
                    if 'key_points' in value:
                        content_parts.append("Key Points:")
                        content_parts.extend([f"- {point}" for point in value['key_points']])
                        
                    # Add source and context if available
                    if 'source' in value:
                        content_parts.append(f"Source: {value['source']}")
                    if 'user_context' in value:
                        content_parts.append(f"Context: {value['user_context']}")
                        
                    content = "\n".join(content_parts)
                    
                    self.add_document(
                        content=content,
                        metadata={
                            "type": "value", 
                            "name": value['name'],
                            "source": "knowledge_files"
                        }
                    )
                    added_count += 1
                    
        # Load frameworks
        frameworks_dir = knowledge_dir / "frameworks"
        if frameworks_dir.exists():
            for framework_file in frameworks_dir.glob("*.json"):
                with open(framework_file, 'r') as f:
                    framework = json.load(f)
                    self.add_document(
                        content=f"Framework: {framework['name']}\n{framework['description']}\nSteps:\n" + 
                                "\n".join(f"- {step}" for step in framework.get('steps', [])),
                        metadata={
                            "type": "framework", 
                            "name": framework['name'],
                            "category": framework.get('category', 'general'),
                            "source": "knowledge_files"
                        }
                    )
                    added_count += 1
                    
        # Load preferences
        preferences_file = knowledge_dir / "preferences.json"
        if preferences_file.exists():
            with open(preferences_file, 'r') as f:
                preferences = json.load(f)
                for category, prefs in preferences.items():
                    for pref_name, pref_value in prefs.items():
                        self.add_document(
                            content=f"Preference: {category} - {pref_name}: {pref_value}",
                            metadata={
                                "type": "preference",
                                "category": category,
                                "name": pref_name,
                                "source": "knowledge_files"
                            }
                        )
                        added_count += 1
                        
        logger.info(f"Added {added_count} new knowledge items")
        
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base."""
        collection_data = self.collection.get()

        stats = {
            "total_documents": len(collection_data['ids']),
            "types": {}
        }

        for metadata in collection_data['metadatas']:
            doc_type = metadata.get('type', 'unknown')
            stats['types'][doc_type] = stats['types'].get(doc_type, 0) + 1

        return stats

    def export_knowledge(
        self,
        filter_type: Optional[str] = None,
        filter_source: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Export knowledge base to a JSON-serializable dictionary.

        Args:
            filter_type: Optional filter by document type (e.g., "value", "framework")
            filter_source: Optional filter by source (e.g., "knowledge_files", "web")

        Returns:
            Dictionary containing export metadata and documents
        """
        try:
            # Get all documents including embeddings
            collection_data = self.collection.get(include=["documents", "metadatas", "embeddings"])

            exported_documents = []

            for i, (doc_id, content, metadata, embedding) in enumerate(zip(
                collection_data['ids'],
                collection_data['documents'],
                collection_data['metadatas'],
                collection_data.get('embeddings', [None] * len(collection_data['ids']))
            )):
                # Apply filters
                if filter_type and metadata.get('type') != filter_type:
                    continue
                if filter_source and metadata.get('source') != filter_source:
                    continue

                doc_export = {
                    "id": doc_id,
                    "content": content,
                    "metadata": metadata,
                }

                # Include embeddings if available
                if embedding is not None:
                    doc_export["embedding"] = embedding

                exported_documents.append(doc_export)

            export_data = {
                "export_version": "1.0",
                "export_timestamp": datetime.now().isoformat(),
                "collection_name": settings.chroma_collection_name,
                "total_documents": len(exported_documents),
                "filters": {
                    "type": filter_type,
                    "source": filter_source
                },
                "documents": exported_documents
            }

            logger.info(f"Exported {len(exported_documents)} documents")
            return export_data

        except Exception as e:
            logger.error(f"Error exporting knowledge base: {e}")
            raise

    def import_knowledge(
        self,
        import_data: Dict[str, Any],
        duplicate_handling: str = "skip"
    ) -> Dict[str, Any]:
        """
        Import knowledge from a previously exported dictionary.

        Args:
            import_data: Dictionary from export_knowledge
            duplicate_handling: How to handle duplicates - "skip", "update", or "fail"

        Returns:
            Dictionary with import statistics
        """
        try:
            # Validate import data structure
            if "documents" not in import_data:
                raise ValueError("Invalid import data: missing 'documents' key")

            if "export_version" not in import_data:
                logger.warning("Import data missing version info, proceeding with caution")

            documents = import_data["documents"]

            # Get existing document IDs
            existing_ids = set(self.collection.get()['ids'])

            stats = {
                "total_in_file": len(documents),
                "added": 0,
                "updated": 0,
                "skipped": 0,
                "errors": 0,
                "error_messages": []
            }

            for doc in documents:
                try:
                    doc_id = doc.get("id")
                    content = doc.get("content")
                    metadata = doc.get("metadata", {})
                    embedding = doc.get("embedding")

                    if not doc_id or not content:
                        stats["errors"] += 1
                        stats["error_messages"].append(f"Document missing id or content")
                        continue

                    # Check for duplicates
                    if doc_id in existing_ids:
                        if duplicate_handling == "skip":
                            stats["skipped"] += 1
                            continue
                        elif duplicate_handling == "fail":
                            raise ValueError(f"Duplicate document ID: {doc_id}")
                        elif duplicate_handling == "update":
                            # Delete existing and re-add
                            self.collection.delete(ids=[doc_id])
                            existing_ids.remove(doc_id)

                    # Generate embedding if not provided
                    if embedding is None:
                        embedding = self.ollama_client.get_embeddings(content)

                    # Add document
                    self.collection.add(
                        documents=[content],
                        embeddings=[embedding],
                        metadatas=[metadata],
                        ids=[doc_id]
                    )

                    existing_ids.add(doc_id)

                    if duplicate_handling == "update" and doc_id in existing_ids:
                        stats["updated"] += 1
                    else:
                        stats["added"] += 1

                except Exception as e:
                    stats["errors"] += 1
                    stats["error_messages"].append(f"Error importing {doc.get('id', 'unknown')}: {str(e)}")
                    logger.error(f"Error importing document: {e}")

            logger.info(f"Import complete: {stats['added']} added, {stats['updated']} updated, "
                       f"{stats['skipped']} skipped, {stats['errors']} errors")

            return stats

        except Exception as e:
            logger.error(f"Error importing knowledge base: {e}")
            raise