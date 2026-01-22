"""RAG retriever with semantic and hybrid search."""
from typing import List, Dict, Any, Optional
from src.rag.vector_store import VectorStore
import re


class HybridRetriever:
    """Hybrid retriever combining semantic and keyword search."""
    
    def __init__(self, vector_store: VectorStore):
        """Initialize retriever with vector store."""
        self.vector_store = vector_store
    
    def retrieve(self, query: str, n_results: int = 5, 
                 use_keyword: bool = True, filter_metadata: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Retrieve relevant documents using hybrid search."""
        # Semantic search
        semantic_results = self.vector_store.search(
            query=query,
            n_results=n_results * 2,  # Get more for re-ranking
            filter_metadata=filter_metadata
        )
        
        if use_keyword:
            # Keyword search
            keyword_results = self._keyword_search(query, semantic_results)
            
            # Combine and re-rank
            combined_results = self._combine_results(semantic_results, keyword_results)
            return combined_results[:n_results]
        else:
            return semantic_results[:n_results]
    
    def _keyword_search(self, query: str, semantic_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Perform keyword-based search on semantic results."""
        query_keywords = set(re.findall(r'\b\w+\b', query.lower()))
        
        scored_results = []
        for result in semantic_results:
            text = result.get('text', '').lower()
            text_keywords = set(re.findall(r'\b\w+\b', text))
            
            # Calculate keyword overlap score
            overlap = len(query_keywords.intersection(text_keywords))
            keyword_score = overlap / len(query_keywords) if query_keywords else 0
            
            result['keyword_score'] = keyword_score
            scored_results.append(result)
        
        # Sort by keyword score
        scored_results.sort(key=lambda x: x.get('keyword_score', 0), reverse=True)
        return scored_results
    
    def _combine_results(self, semantic_results: List[Dict[str, Any]], 
                        keyword_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Combine semantic and keyword results with scoring."""
        # Create a map of results by text
        result_map = {}
        
        for result in semantic_results:
            text = result.get('text', '')
            if text not in result_map:
                result_map[text] = result
                result_map[text]['combined_score'] = 1.0 - (result.get('distance', 0) if result.get('distance') else 0)
        
        # Add keyword scores
        for result in keyword_results:
            text = result.get('text', '')
            if text in result_map:
                keyword_score = result.get('keyword_score', 0)
                result_map[text]['combined_score'] = (
                    result_map[text]['combined_score'] * 0.7 + keyword_score * 0.3
                )
        
        # Sort by combined score
        combined = list(result_map.values())
        combined.sort(key=lambda x: x.get('combined_score', 0), reverse=True)
        
        return combined
    
    def retrieve_by_section(self, query: str, section: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Retrieve documents filtered by section."""
        filter_metadata = {'section': section}
        return self.retrieve(query, n_results=n_results, filter_metadata=filter_metadata)
    
    def retrieve_by_clause(self, query: str, clause_id: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Retrieve documents filtered by clause ID."""
        filter_metadata = {'clause_id': clause_id}
        return self.retrieve(query, n_results=n_results, filter_metadata=filter_metadata)
    
    def retrieve_by_type(self, query: str, chunk_type: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Retrieve documents filtered by chunk type."""
        filter_metadata = {'type': chunk_type}
        return self.retrieve(query, n_results=n_results, filter_metadata=filter_metadata)
