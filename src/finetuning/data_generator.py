"""Generate training datasets from contract documents and Q&A pairs."""
import json
import random
from typing import List, Dict, Any, Optional
from pathlib import Path
from src.data.contract_parser import ContractParser
from src.data.preprocessing import ContractPreprocessor
from src.evaluation.test_cases import get_test_cases
from src.llm.prompts import LegalPrompts


class TrainingDataGenerator:
    """Generate training datasets for fine-tuning."""
    
    def __init__(self, contract_path: Optional[str] = None):
        """Initialize data generator."""
        self.contract_path = contract_path
        self.parser = ContractParser(contract_path) if contract_path else ContractParser()
        self.preprocessor = ContractPreprocessor()
        self.prompts = LegalPrompts()
    
    def generate_qa_pairs_from_contract(self) -> List[Dict[str, Any]]:
        """Generate Q&A pairs from contract sections."""
        qa_pairs = []
        
        # Load contract data
        chunks = self.parser.extract_chunks()
        sections = self.parser.get_sections()
        clauses = self.parser.get_clauses()
        
        # Generate Q&A from sections
        for section in sections:
            header = section.get('header', '')
            content = ' '.join([item.get('text', '') for item in section.get('content', [])])
            
            if header and content:
                # Generate question from header
                question = self._generate_question_from_header(header, content)
                answer = self._generate_answer_from_content(header, content)
                
                if question and answer:
                    qa_pairs.append({
                        'question': question,
                        'answer': answer,
                        'context': content[:500],  # Truncate context
                        'source': 'section',
                        'section': header
                    })
        
        # Generate Q&A from clauses
        for clause in clauses:
            clause_id = clause.get('clause_id', '')
            clause_text = clause.get('full_text', '')
            
            if clause_id and clause_text:
                question = self._generate_question_from_clause(clause_id, clause_text)
                answer = self._extract_answer_from_clause(clause_text)
                
                if question and answer:
                    qa_pairs.append({
                        'question': question,
                        'answer': answer,
                        'context': clause_text[:500],
                        'source': 'clause',
                        'clause_id': clause_id
                    })
        
        return qa_pairs
    
    def generate_from_test_cases(self) -> List[Dict[str, Any]]:
        """Generate training examples from existing test cases."""
        test_cases = get_test_cases()
        examples = []
        
        # Load contract context
        chunks = self.parser.extract_chunks()
        text_chunks = self.parser.get_all_text_chunks(min_length=50)
        
        for test_case in test_cases:
            question = test_case['question']
            category = test_case.get('category', 'general')
            
            # Find relevant context (simplified - in practice, use RAG retrieval)
            relevant_context = self._find_relevant_context(question, text_chunks)
            
            # Generate answer format
            answer = self._generate_answer_template(question, relevant_context, category)
            
            examples.append({
                'question': question,
                'answer': answer,
                'context': relevant_context,
                'source': 'test_case',
                'category': category
            })
        
        return examples
    
    def format_for_training(self, qa_pairs: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Format Q&A pairs into instruction-following format."""
        formatted_examples = []
        
        for pair in qa_pairs:
            question = pair.get('question', '')
            answer = pair.get('answer', '')
            context = pair.get('context', '')
            
            if not question or not answer:
                continue
            
            # Format as instruction-following example
            instruction = self.prompts.SYSTEM_PROMPT
            input_text = f"Contract Context:\n{context}\n\nQuestion: {question}"
            output_text = answer
            
            formatted_examples.append({
                'instruction': instruction,
                'input': input_text,
                'output': output_text
            })
        
        return formatted_examples
    
    def augment_data(self, examples: List[Dict[str, Any]], num_augmentations: int = 2) -> List[Dict[str, Any]]:
        """Augment training data with paraphrases and variations."""
        augmented = []
        
        for example in examples:
            augmented.append(example)  # Original
            
            # Paraphrase questions
            for _ in range(num_augmentations):
                question = example.get('question', '')
                paraphrased = self._paraphrase_question(question)
                
                if paraphrased and paraphrased != question:
                    augmented_example = example.copy()
                    augmented_example['question'] = paraphrased
                    augmented.append(augmented_example)
        
        return augmented
    
    def split_dataset(self, examples: List[Dict[str, Any]], 
                     train_ratio: float = 0.8, 
                     val_ratio: float = 0.1) -> Dict[str, List[Dict[str, Any]]]:
        """Split dataset into train/validation/test sets."""
        random.shuffle(examples)
        
        total = len(examples)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        
        return {
            'train': examples[:train_end],
            'validation': examples[train_end:val_end],
            'test': examples[val_end:]
        }
    
    def save_dataset(self, examples: List[Dict[str, Any]], output_path: str, format: str = 'jsonl'):
        """Save dataset to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'jsonl':
            with open(output_path, 'w', encoding='utf-8') as f:
                for example in examples:
                    f.write(json.dumps(example, ensure_ascii=False) + '\n')
        elif format == 'json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(examples, f, ensure_ascii=False, indent=2)
    
    def _generate_question_from_header(self, header: str, content: str) -> str:
        """Generate a question from section header."""
        header_lower = header.lower()
        
        if 'payment' in header_lower or 'price' in header_lower:
            return f"What are the payment terms in {header}?"
        elif 'liability' in header_lower or 'defect' in header_lower:
            return f"What is the defect liability period?"
        elif 'termination' in header_lower:
            return f"What are the termination conditions?"
        elif 'security' in header_lower or 'guarantee' in header_lower:
            return f"What are the security requirements?"
        else:
            return f"What does {header} specify?"
    
    def _generate_answer_from_content(self, header: str, content: str) -> str:
        """Generate answer from content."""
        # Extract key information
        sentences = content.split('.')
        key_sentences = [s.strip() for s in sentences if len(s.strip()) > 20][:3]
        answer = '. '.join(key_sentences)
        
        if answer:
            return f"According to {header}, {answer}."
        return ""
    
    def _generate_question_from_clause(self, clause_id: str, clause_text: str) -> str:
        """Generate question from clause."""
        return f"What does {clause_id} specify?"
    
    def _extract_answer_from_clause(self, clause_text: str) -> str:
        """Extract answer from clause text."""
        # Simple extraction - take first few sentences
        sentences = clause_text.split('.')
        answer = '. '.join([s.strip() for s in sentences[:3] if s.strip()])
        return answer if answer else clause_text[:200]
    
    def _find_relevant_context(self, question: str, chunks: List[Dict[str, Any]]) -> str:
        """Find relevant context for question."""
        # Simple keyword matching
        question_lower = question.lower()
        relevant_chunks = []
        
        for chunk in chunks[:10]:  # Limit search
            text = chunk.get('text', '').lower()
            if any(word in text for word in question_lower.split() if len(word) > 3):
                relevant_chunks.append(chunk.get('text', ''))
        
        return '\n'.join(relevant_chunks[:3])[:500]
    
    def _generate_answer_template(self, question: str, context: str, category: str) -> str:
        """Generate answer template."""
        if context:
            return f"Based on the contract, {question.lower().replace('what', '').replace('?', '')} is addressed in the following sections: {context[:200]}..."
        return f"This question relates to {category} provisions in the contract."
    
    def _paraphrase_question(self, question: str) -> str:
        """Paraphrase a question (simple synonym replacement)."""
        paraphrases = {
            'what': ['what', 'which', 'describe'],
            'are': ['are', 'is'],
            'the': ['the', 'a', 'any'],
            'terms': ['terms', 'conditions', 'provisions'],
            'period': ['period', 'duration', 'timeframe'],
            'requirements': ['requirements', 'specifications', 'criteria']
        }
        
        words = question.lower().split()
        paraphrased_words = []
        
        for word in words:
            if word in paraphrases:
                paraphrased_words.append(random.choice(paraphrases[word]))
            else:
                paraphrased_words.append(word)
        
        paraphrased = ' '.join(paraphrased_words)
        return paraphrased.capitalize() + '?' if question.endswith('?') else paraphrased.capitalize()
