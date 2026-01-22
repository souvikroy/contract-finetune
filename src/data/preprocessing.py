"""Preprocess and clean contract data."""
import re
from typing import List, Dict, Any


class ContractPreprocessor:
    """Clean and structure contract data for processing."""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean text by removing extra whitespace and normalizing."""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters that might interfere
        text = text.strip()
        
        return text
    
    @staticmethod
    def extract_section_numbers(text: str) -> List[str]:
        """Extract section numbers from text (e.g., 'Section 1.2', 'Article-3')."""
        patterns = [
            r'Section\s+(\d+(?:\.\d+)*)',
            r'Article\s*-?\s*(\d+)',
            r'Clause\s+(\d+(?:\.\d+)*)',
            r'(\d+\.\d+\.\d+)',  # Pattern like 1.2.3
        ]
        
        section_numbers = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            section_numbers.extend(matches)
        
        return section_numbers
    
    @staticmethod
    def identify_clause_type(text: str) -> str:
        """Identify the type of legal clause."""
        text_lower = text.lower()
        
        if any(keyword in text_lower for keyword in ['commitment', 'commit']):
            return 'commitment'
        elif any(keyword in text_lower for keyword in ['disqualification', 'exclusion']):
            return 'disqualification'
        elif any(keyword in text_lower for keyword in ['compensation', 'damage']):
            return 'compensation'
        elif any(keyword in text_lower for keyword in ['guarantee', 'security']):
            return 'guarantee'
        elif any(keyword in text_lower for keyword in ['termination', 'cancel']):
            return 'termination'
        elif any(keyword in text_lower for keyword in ['payment', 'price']):
            return 'payment'
        else:
            return 'general'
    
    @staticmethod
    def structure_chunk(chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Structure a chunk with cleaned data and metadata."""
        content = chunk.get('content', '')
        cleaned_content = ContractPreprocessor.clean_text(content)
        
        structured = {
            'text': cleaned_content,
            'original_text': content,
            'type': chunk.get('type', 'Text'),
            'page': chunk.get('bbox', {}).get('page', chunk.get('bbox', {}).get('original_page', 0)),
            'section_numbers': ContractPreprocessor.extract_section_numbers(cleaned_content),
            'clause_type': ContractPreprocessor.identify_clause_type(cleaned_content),
            'metadata': {
                'confidence': chunk.get('confidence'),
                'bbox': chunk.get('bbox'),
                'original_type': chunk.get('type')
            }
        }
        
        return structured
    
    @staticmethod
    def preprocess_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Preprocess a list of chunks."""
        return [ContractPreprocessor.structure_chunk(chunk) for chunk in chunks]
