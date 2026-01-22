"""LangChain RAG chain for Q&A."""
from typing import List, Dict, Any, Optional
from src.llm.claude_client import ClaudeClient
from src.llm.prompts import LegalPrompts
from src.rag.retriever import HybridRetriever


class RAGChain:
    """RAG chain combining retrieval, augmentation, and generation."""
    
    def __init__(self, retriever: HybridRetriever, claude_client: Optional[ClaudeClient] = None):
        """Initialize RAG chain."""
        self.retriever = retriever
        self.claude_client = claude_client or ClaudeClient()
        self.prompts = LegalPrompts()
    
    def query(self, question: str, n_context: int = 5, use_keyword: bool = True) -> Dict[str, Any]:
        """Query the RAG system with a question."""
        # Retrieve relevant context
        retrieved_docs = self.retriever.retrieve(
            query=question,
            n_results=n_context,
            use_keyword=use_keyword
        )
        
        # Format context from retrieved documents
        context = self._format_context(retrieved_docs)
        
        # Generate answer using Claude
        prompt = self.prompts.format_qa_prompt(context=context, question=question)
        
        messages = [{'role': 'user', 'content': prompt}]
        answer = self.claude_client.invoke(
            messages=messages,
            system_prompt=self.prompts.SYSTEM_PROMPT
        )
        
        return {
            'question': question,
            'answer': answer,
            'context_documents': retrieved_docs,
            'sources': self._extract_sources(retrieved_docs)
        }
    
    def stream_query(self, question: str, n_context: int = 5, use_keyword: bool = True):
        """Stream query response."""
        # Retrieve relevant context
        retrieved_docs = self.retriever.retrieve(
            query=question,
            n_results=n_context,
            use_keyword=use_keyword
        )
        
        # Format context
        context = self._format_context(retrieved_docs)
        
        # Generate answer
        prompt = self.prompts.format_qa_prompt(context=context, question=question)
        messages = [{'role': 'user', 'content': prompt}]
        
        # Stream response
        full_answer = ""
        for chunk in self.claude_client.stream(messages, system_prompt=self.prompts.SYSTEM_PROMPT):
            full_answer += chunk
            yield {
                'chunk': chunk,
                'answer_so_far': full_answer,
                'sources': self._extract_sources(retrieved_docs)
            }
    
    def extract_clauses(self, query: str, n_context: int = 10) -> Dict[str, Any]:
        """Extract specific clauses based on query."""
        retrieved_docs = self.retriever.retrieve(
            query=query,
            n_results=n_context,
            use_keyword=True
        )
        
        context = self._format_context(retrieved_docs)
        prompt = self.prompts.format_clause_extraction_prompt(context=context, query=query)
        
        messages = [{'role': 'user', 'content': prompt}]
        extraction_result = self.claude_client.invoke(
            messages=messages,
            system_prompt=self.prompts.SYSTEM_PROMPT
        )
        
        return {
            'query': query,
            'extracted_clauses': extraction_result,
            'source_documents': retrieved_docs
        }
    
    def classify_clause(self, clause_text: str) -> Dict[str, Any]:
        """Classify a contract clause."""
        prompt = self.prompts.format_classification_prompt(clause_text=clause_text)
        messages = [{'role': 'user', 'content': prompt}]
        
        classification = self.claude_client.invoke(
            messages=messages,
            system_prompt=self.prompts.SYSTEM_PROMPT
        )
        
        return {
            'clause_text': clause_text,
            'classification': classification
        }
    
    def _format_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """Format retrieved documents into context string."""
        context_parts = []
        
        for i, doc in enumerate(retrieved_docs, 1):
            text = doc.get('text', '')
            metadata = doc.get('metadata', {})
            page = metadata.get('page', 'Unknown')
            section = metadata.get('section', '')
            clause_id = metadata.get('clause_id', '')
            
            context_part = f"[Document {i}]"
            if section:
                context_part += f" Section: {section}"
            if clause_id:
                context_part += f" Clause: {clause_id}"
            context_part += f" (Page {page})\n{text}\n"
            
            context_parts.append(context_part)
        
        return "\n---\n".join(context_parts)
    
    def _extract_sources(self, retrieved_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract source information from retrieved documents."""
        sources = []
        for doc in retrieved_docs:
            metadata = doc.get('metadata', {})
            sources.append({
                'page': metadata.get('page', 'Unknown'),
                'section': metadata.get('section', ''),
                'clause_id': metadata.get('clause_id', ''),
                'type': metadata.get('type', 'Text'),
                'text_preview': doc.get('text', '')[:200] + '...' if len(doc.get('text', '')) > 200 else doc.get('text', '')
            })
        return sources
