"""Intelligent chunking for legal documents."""
from typing import List, Dict, Any
import re


class LegalDocumentChunker:
    """Chunk legal documents preserving context and structure."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """Initialize chunker with size and overlap parameters."""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_by_sections(self, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Chunk document by sections, preserving section context."""
        chunks = []
        
        for section in sections:
            header = section.get('header', '')
            content_items = section.get('content', [])
            
            # Combine section content
            full_text = header + '\n'
            for item in content_items:
                full_text += item.get('text', '') + '\n'
            
            # Split if too long
            if len(full_text) > self.chunk_size:
                sub_chunks = self._split_text(full_text)
                for i, sub_chunk in enumerate(sub_chunks):
                    chunks.append({
                        'text': sub_chunk,
                        'section': header,
                        'page': section.get('page', 0),
                        'chunk_index': i,
                        'metadata': {
                            'type': 'section_chunk',
                            'section_header': header
                        }
                    })
            else:
                chunks.append({
                    'text': full_text,
                    'section': header,
                    'page': section.get('page', 0),
                    'chunk_index': 0,
                    'metadata': {
                        'type': 'section_chunk',
                        'section_header': header
                    }
                })
        
        return chunks
    
    def chunk_by_clauses(self, clauses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Chunk document by legal clauses."""
        chunks = []
        
        for clause in clauses:
            clause_id = clause.get('clause_id', '')
            full_text = clause.get('full_text', '')
            
            if len(full_text) > self.chunk_size:
                sub_chunks = self._split_text(full_text)
                for i, sub_chunk in enumerate(sub_chunks):
                    chunks.append({
                        'text': sub_chunk,
                        'clause_id': clause_id,
                        'page': clause.get('page', 0),
                        'chunk_index': i,
                        'metadata': {
                            'type': 'clause_chunk',
                            'clause_id': clause_id
                        }
                    })
            else:
                chunks.append({
                    'text': full_text,
                    'clause_id': clause_id,
                    'page': clause.get('page', 0),
                    'chunk_index': 0,
                    'metadata': {
                        'type': 'clause_chunk',
                        'clause_id': clause_id
                    }
                })
        
        return chunks
    
    def chunk_text(self, text_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Chunk text preserving sentence boundaries."""
        chunks = []
        current_chunk = []
        current_length = 0
        
        for text_item in text_chunks:
            text = text_item.get('text', '')
            sentences = self._split_sentences(text)
            
            for sentence in sentences:
                sentence_length = len(sentence)
                
                if current_length + sentence_length > self.chunk_size and current_chunk:
                    # Save current chunk
                    chunk_text = ' '.join(current_chunk)
                    chunks.append({
                        'text': chunk_text,
                        'page': text_item.get('page', 0),
                        'metadata': text_item.get('metadata', {})
                    })
                    
                    # Start new chunk with overlap
                    overlap_text = ' '.join(current_chunk[-self._overlap_sentences():])
                    current_chunk = [overlap_text, sentence] if overlap_text else [sentence]
                    current_length = len(' '.join(current_chunk))
                else:
                    current_chunk.append(sentence)
                    current_length += sentence_length + 1  # +1 for space
        
        # Add remaining chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                'text': chunk_text,
                'page': text_item.get('page', 0),
                'metadata': text_item.get('metadata', {})
            })
        
        return chunks
    
    def _split_text(self, text: str) -> List[str]:
        """Split text into chunks with overlap."""
        words = text.split()
        chunks = []
        
        if len(words) <= self.chunk_size // 10:  # Rough word count estimate
            return [text]
        
        start = 0
        while start < len(words):
            end = min(start + self.chunk_size // 10, len(words))
            chunk_words = words[start:end]
            chunks.append(' '.join(chunk_words))
            
            # Move start forward with overlap
            start = end - (self.chunk_overlap // 10)
            if start < 0:
                start = end
        
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting (can be improved with NLTK)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _overlap_sentences(self) -> int:
        """Calculate number of sentences for overlap."""
        # Rough estimate: assume average sentence length
        avg_sentence_length = 100
        return max(1, self.chunk_overlap // avg_sentence_length)
