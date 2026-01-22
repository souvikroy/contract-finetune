"""Parse and extract structured data from contract JSON."""
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
from src.config import settings


class ContractParser:
    """Parse contract JSON and extract structured data."""
    
    def __init__(self, json_path: Optional[str] = None):
        """Initialize parser with contract JSON path."""
        self.json_path = json_path or settings.contract_json_path
        self.data = None
        self.chunks = []
        
    def load(self) -> Dict[str, Any]:
        """Load JSON data from file."""
        with open(self.json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        return self.data
    
    def extract_chunks(self) -> List[Dict[str, Any]]:
        """Extract chunks from parsed JSON."""
        if not self.data:
            self.load()
        
        if 'result' in self.data and 'chunks' in self.data['result']:
            self.chunks = self.data['result']['chunks']
        else:
            self.chunks = []
        
        return self.chunks
    
    def get_chunks_by_type(self, chunk_type: str) -> List[Dict[str, Any]]:
        """Get chunks filtered by type (e.g., 'Section Header', 'List Item')."""
        if not self.chunks:
            self.extract_chunks()
        
        return [chunk for chunk in self.chunks if chunk.get('type') == chunk_type]
    
    def get_sections(self) -> List[Dict[str, Any]]:
        """Extract section headers and their content."""
        if not self.chunks:
            self.extract_chunks()
        
        sections = []
        current_section = None
        
        for chunk in self.chunks:
            chunk_type = chunk.get('type', '')
            content = chunk.get('content', '').strip()
            
            if chunk_type == 'Section Header' and content:
                if current_section:
                    sections.append(current_section)
                
                current_section = {
                    'header': content,
                    'page': chunk.get('bbox', {}).get('page', chunk.get('bbox', {}).get('original_page', 0)),
                    'content': [],
                    'chunk_type': chunk_type
                }
            elif current_section and content:
                current_section['content'].append({
                    'text': content,
                    'type': chunk_type,
                    'page': chunk.get('bbox', {}).get('page', chunk.get('bbox', {}).get('original_page', 0))
                })
        
        if current_section:
            sections.append(current_section)
        
        return sections
    
    def get_clauses(self) -> List[Dict[str, Any]]:
        """Extract legal clauses from the contract."""
        sections = self.get_sections()
        clauses = []
        
        for section in sections:
            # Look for numbered clauses (e.g., "Article-1", "Clause 2.1")
            header = section['header']
            if any(keyword in header.lower() for keyword in ['article', 'clause', 'section']):
                clause_text = header
                clause_content = ' '.join([item['text'] for item in section['content']])
                
                clauses.append({
                    'clause_id': header,
                    'clause_text': clause_text,
                    'content': clause_content,
                    'full_text': f"{clause_text}\n{clause_content}",
                    'page': section['page'],
                    'type': 'clause'
                })
        
        return clauses
    
    def get_metadata(self) -> Dict[str, Any]:
        """Extract metadata from the contract."""
        if not self.data:
            self.load()
        
        metadata = {
            'job_id': self.data.get('job_id'),
            'duration': self.data.get('duration'),
            'pdf_url': self.data.get('pdf_url'),
            'usage': self.data.get('usage', {}),
            'total_chunks': len(self.chunks) if self.chunks else 0,
            'total_pages': self.data.get('usage', {}).get('num_pages', 0)
        }
        
        return metadata
    
    def get_all_text_chunks(self, min_length: int = 50) -> List[Dict[str, Any]]:
        """Get all text chunks with minimum length for embedding."""
        if not self.chunks:
            self.extract_chunks()
        
        text_chunks = []
        for idx, chunk in enumerate(self.chunks):
            content = chunk.get('content', '').strip()
            if len(content) >= min_length:
                text_chunks.append({
                    'id': idx,
                    'text': content,
                    'type': chunk.get('type', 'Text'),
                    'page': chunk.get('bbox', {}).get('page', chunk.get('bbox', {}).get('original_page', 0)),
                    'metadata': {
                        'confidence': chunk.get('confidence'),
                        'bbox': chunk.get('bbox')
                    }
                })
        
        return text_chunks
