"""Vector database setup and management."""
import os
from typing import List, Dict, Any, Optional
from pathlib import Path
import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
from src.config import settings


class VectorStore:
    """Manage vector database for contract embeddings."""
    
    def __init__(self, collection_name: str = "contract_documents"):
        """Initialize vector store."""
        self.collection_name = collection_name
        self.embedding_model = SentenceTransformer(settings.embedding_model)
        self.client = None
        self.collection = None
        self._initialize_db()
    
    def _initialize_db(self):
        """Initialize ChromaDB client and collection."""
        persist_directory = Path(settings.chroma_persist_directory)
        persist_directory.mkdir(parents=True, exist_ok=True)
        
        self.client = chromadb.PersistentClient(
            path=str(persist_directory),
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
        except:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Legal contract documents"}
            )
    
    def add_documents(self, chunks: List[Dict[str, Any]], batch_size: int = 100):
        """Add documents to vector store."""
        texts = []
        metadatas = []
        ids = []
        
        for idx, chunk in enumerate(chunks):
            text = chunk.get('text', '')
            if not text:
                continue
            
            texts.append(text)
            metadata = chunk.get('metadata', {})
            metadata.update({
                'page': chunk.get('page', 0),
                'chunk_index': chunk.get('chunk_index', 0),
                'section': chunk.get('section', ''),
                'clause_id': chunk.get('clause_id', ''),
                'type': chunk.get('type', 'Text')
            })
            metadatas.append(metadata)
            ids.append(f"chunk_{idx}")
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True).tolist()
        
        # Add to collection in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_embeddings = embeddings[i:i+batch_size]
            batch_metadatas = metadatas[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]
            
            self.collection.add(
                embeddings=batch_embeddings,
                documents=batch_texts,
                metadatas=batch_metadatas,
                ids=batch_ids
            )
    
    def search(self, query: str, n_results: int = 5, filter_metadata: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Search for similar documents."""
        query_embedding = self.embedding_model.encode([query]).tolist()[0]
        
        where_clause = filter_metadata if filter_metadata else None
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_clause
        )
        
        # Format results
        formatted_results = []
        if results['documents'] and len(results['documents'][0]) > 0:
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else None,
                    'id': results['ids'][0][i] if 'ids' in results else None
                })
        
        return formatted_results
    
    def delete_collection(self):
        """Delete the collection."""
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = None
        except Exception as e:
            print(f"Error deleting collection: {e}")
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        if not self.collection:
            return {}
        
        count = self.collection.count()
        return {
            'collection_name': self.collection_name,
            'document_count': count,
            'embedding_model': settings.embedding_model
        }
