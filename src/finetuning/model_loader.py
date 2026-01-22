"""Model loading utilities for fine-tuned models."""
import torch
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from src.config import settings


class ModelLoader:
    """Load fine-tuned models and adapters."""
    
    def __init__(self, base_model_name: Optional[str] = None):
        """Initialize model loader."""
        self.base_model_name = base_model_name or settings.finetuning_base_model
    
    def load_base_model(self, use_quantization: bool = False) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Load base model and tokenizer."""
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        if use_quantization:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                quantization_config=bnb_config,
                device_map="auto"
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        
        return model, tokenizer
    
    def load_peft_model(
        self,
        adapter_path: str,
        merge_adapters: bool = False,
        use_quantization: bool = False
    ) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Load model with PEFT adapters."""
        # Load base model
        model, tokenizer = self.load_base_model(use_quantization=use_quantization)
        
        # Load adapters
        if merge_adapters:
            # Merge adapters into base model
            model = PeftModel.from_pretrained(model, adapter_path)
            model = model.merge_and_unload()
        else:
            # Keep adapters separate
            model = PeftModel.from_pretrained(model, adapter_path)
        
        return model, tokenizer
    
    def load_finetuned_model(
        self,
        model_path: Optional[str] = None,
        merge_adapters: Optional[bool] = None
    ) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Load fine-tuned model from settings or path."""
        adapter_path = model_path or settings.finetuning_model_path
        if not adapter_path:
            raise ValueError("No fine-tuned model path specified")
        
        merge = merge_adapters if merge_adapters is not None else settings.use_merged_model
        
        return self.load_peft_model(
            adapter_path=adapter_path,
            merge_adapters=merge
        )
    
    def save_merged_model(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, output_path: str):
        """Save merged model (base + adapters)."""
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
    
    def get_model_info(self, model: AutoModelForCausalLM) -> dict:
        """Get model information."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'trainable_percentage': 100 * trainable_params / total_params if total_params > 0 else 0
        }


def load_model_for_inference(
    model_path: Optional[str] = None,
    base_model: Optional[str] = None,
    merge: bool = True
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Convenience function to load model for inference."""
    loader = ModelLoader(base_model_name=base_model)
    
    if model_path:
        return loader.load_peft_model(model_path, merge_adapters=merge)
    elif settings.finetuning_enabled and settings.finetuning_model_path:
        return loader.load_finetuned_model(merge_adapters=merge)
    else:
        return loader.load_base_model()
