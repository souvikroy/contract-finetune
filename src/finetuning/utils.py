"""Training utilities."""
import re
from typing import List, Dict, Any
import torch


def clean_text(text: str) -> str:
    """Clean and normalize text."""
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters that might interfere
    text = text.strip()
    
    return text


def extract_legal_terms(text: str) -> List[str]:
    """Extract legal terms from text."""
    legal_terms = [
        'contractor', 'obligation', 'liability', 'termination', 'compensation',
        'guarantee', 'defect', 'payment', 'clause', 'article', 'section',
        'penalty', 'breach', 'default', 'warranty', 'indemnity', 'dispute'
    ]
    
    found_terms = []
    text_lower = text.lower()
    
    for term in legal_terms:
        if term in text_lower:
            found_terms.append(term)
    
    return found_terms


def validate_answer_quality(answer: str, min_length: int = 20) -> bool:
    """Validate answer quality."""
    if not answer or len(answer.strip()) < min_length:
        return False
    
    # Check for meaningful content (not just punctuation)
    if not re.search(r'[a-zA-Z]{3,}', answer):
        return False
    
    return True


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Count model parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': total_params - trainable_params
    }


def format_model_size(num_params: int) -> str:
    """Format parameter count as human-readable size."""
    if num_params >= 1e9:
        return f"{num_params / 1e9:.2f}B"
    elif num_params >= 1e6:
        return f"{num_params / 1e6:.2f}M"
    elif num_params >= 1e3:
        return f"{num_params / 1e3:.2f}K"
    else:
        return str(num_params)


def merge_lora_adapters(model, adapter_path: str, output_path: str):
    """Merge LoRA adapters into base model."""
    from peft import PeftModel
    
    # Load adapters
    model = PeftModel.from_pretrained(model, adapter_path)
    
    # Merge
    merged_model = model.merge_and_unload()
    
    # Save
    merged_model.save_pretrained(output_path)
    
    return merged_model
