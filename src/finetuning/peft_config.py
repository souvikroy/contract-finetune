"""PEFT/LoRA configuration for fine-tuning."""
from typing import List, Optional
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import AutoModelForCausalLM
from src.config import settings


def create_lora_config(
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
    bias: str = "none",
    task_type: TaskType = TaskType.CAUSAL_LM
) -> LoraConfig:
    """Create LoRA configuration."""
    if target_modules is None:
        # Default target modules for Qwen models
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
    
    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias=bias,
        task_type=task_type
    )
    
    return config


def create_qlora_config(
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
    bias: str = "none",
    task_type: TaskType = TaskType.CAUSAL_LM,
    bnb_4bit_compute_dtype: str = "float16",
    bnb_4bit_quant_type: str = "nf4",
    bnb_4bit_use_double_quant: bool = True
):
    """Create QLoRA configuration with quantization."""
    from transformers import BitsAndBytesConfig
    import torch
    
    if target_modules is None:
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
    
    # LoRA config
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias=bias,
        task_type=task_type
    )
    
    # Quantization config
    compute_dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32
    }
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype_map.get(bnb_4bit_compute_dtype, torch.float16),
        bnb_4bit_use_double_quant=bnb_4bit_use_double_quant
    )
    
    return lora_config, bnb_config


def get_default_lora_config() -> LoraConfig:
    """Get default LoRA configuration from settings."""
    return create_lora_config(
        r=settings.lora_r,
        lora_alpha=settings.lora_alpha,
        lora_dropout=settings.lora_dropout
    )


def apply_peft_to_model(model: AutoModelForCausalLM, config: LoraConfig) -> AutoModelForCausalLM:
    """Apply PEFT configuration to model."""
    peft_model = get_peft_model(model, config)
    return peft_model


def print_trainable_parameters(model: AutoModelForCausalLM):
    """Print trainable parameters."""
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    print(
        f"trainable params: {trainable_params} || "
        f"all params: {all_param} || "
        f"trainable%: {100 * trainable_params / all_param:.2f}"
    )
