"""Dataset classes for fine-tuning."""
import json
from typing import List, Dict, Any, Optional
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer
from dataclasses import dataclass


@dataclass
class TrainingExample:
    """Single training example."""
    instruction: str
    input_text: str
    output_text: str
    
    def to_prompt(self) -> str:
        """Convert to prompt format."""
        return f"{self.instruction}\n\n{self.input_text}\n\nAnswer: {self.output_text}"


class LegalDataset(Dataset):
    """PyTorch Dataset for legal contract Q&A."""
    
    def __init__(self, examples: List[Dict[str, Any]], tokenizer: PreTrainedTokenizer, 
                 max_length: int = 2048):
        """Initialize dataset."""
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get item by index."""
        example = self.examples[idx]
        
        # Format prompt
        instruction = example.get('instruction', '')
        input_text = example.get('input', '')
        output_text = example.get('output', '')
        
        # Create full prompt
        prompt = f"{instruction}\n\n{input_text}\n\nAnswer: {output_text}"
        
        # Tokenize
        encoding = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze()
        }


class InstructionDataset(Dataset):
    """Dataset formatted for instruction tuning."""
    
    def __init__(self, examples: List[Dict[str, Any]], tokenizer: PreTrainedTokenizer,
                 max_length: int = 2048):
        """Initialize instruction dataset."""
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get item by index."""
        example = self.examples[idx]
        
        instruction = example.get('instruction', '')
        input_text = example.get('input', '')
        output_text = example.get('output', '')
        
        # Format as instruction-following
        full_text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output_text}"
        
        # Tokenize
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze()
        }


class ConversationDataset(Dataset):
    """Dataset for multi-turn conversations."""
    
    def __init__(self, conversations: List[List[Dict[str, str]]], 
                 tokenizer: PreTrainedTokenizer, max_length: int = 2048):
        """Initialize conversation dataset."""
        self.conversations = conversations
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.conversations)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get item by index."""
        conversation = self.conversations[idx]
        
        # Format conversation
        formatted = []
        for turn in conversation:
            role = turn.get('role', 'user')
            content = turn.get('content', '')
            formatted.append(f"{role}: {content}")
        
        full_text = "\n".join(formatted)
        
        # Tokenize
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze()
        }


def load_dataset_from_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load dataset from JSONL file."""
    examples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples


def load_dataset_from_json(file_path: str) -> List[Dict[str, Any]]:
    """Load dataset from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_dataloader(dataset: Dataset, batch_size: int = 4, shuffle: bool = True,
                     num_workers: int = 0) -> DataLoader:
    """Create DataLoader from dataset."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
