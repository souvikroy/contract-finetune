#!/usr/bin/env python
"""Training script entry point for fine-tuning."""
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.finetuning.training_pipeline import TrainingPipeline
from src.config import settings


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Fine-tune Qwen model with PEFT/LoRA')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, help='Path to training data directory')
    parser.add_argument('--contract_path', type=str, default=None,
                       help='Path to contract JSON file (for data generation)')
    
    # Model arguments
    parser.add_argument('--base_model', type=str, default=None,
                       help='Base model name (default: from config)')
    parser.add_argument('--output_dir', type=str, default='./checkpoints/qwen-legal-lora',
                       help='Output directory for checkpoints')
    
    # LoRA arguments
    parser.add_argument('--lora_r', type=int, default=None,
                       help='LoRA rank (default: from config)')
    parser.add_argument('--lora_alpha', type=int, default=None,
                       help='LoRA alpha (default: from config)')
    parser.add_argument('--lora_dropout', type=float, default=None,
                       help='LoRA dropout (default: from config)')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-4,
                       help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                       help='Gradient accumulation steps')
    parser.add_argument('--warmup_steps', type=int, default=100,
                       help='Warmup steps')
    parser.add_argument('--max_length', type=int, default=2048,
                       help='Maximum sequence length')
    
    # Other arguments
    parser.add_argument('--use_qlora', action='store_true',
                       help='Use QLoRA (4-bit quantization)')
    parser.add_argument('--merge_adapters', action='store_true', default=True,
                       help='Merge adapters after training')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                       help='Resume training from checkpoint')
    parser.add_argument('--eval_only', action='store_true',
                       help='Evaluation only mode')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file (YAML/JSON)')
    
    return parser.parse_args()


def load_config(config_path: str):
    """Load configuration from file."""
    import json
    import yaml
    
    config_path = Path(config_path)
    
    if config_path.suffix == '.json':
        with open(config_path, 'r') as f:
            return json.load(f)
    elif config_path.suffix in ['.yaml', '.yml']:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported config file format: {config_path.suffix}")


def main():
    """Main training function."""
    args = parse_args()
    
    # Load config if provided
    config = {}
    if args.config:
        config = load_config(args.config)
    
    # Override with command line arguments
    if args.base_model:
        config['base_model'] = args.base_model
    else:
        config['base_model'] = settings.finetuning_base_model
    
    if args.output_dir:
        config['output_dir'] = args.output_dir
    
    if args.lora_r is not None:
        config['lora_r'] = args.lora_r
    else:
        config['lora_r'] = settings.lora_r
    
    if args.lora_alpha is not None:
        config['lora_alpha'] = args.lora_alpha
    else:
        config['lora_alpha'] = settings.lora_alpha
    
    if args.lora_dropout is not None:
        config['lora_dropout'] = args.lora_dropout
    else:
        config['lora_dropout'] = settings.lora_dropout
    
    config.update({
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'num_epochs': args.num_epochs,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        'warmup_steps': args.warmup_steps,
        'max_length': args.max_length,
        'use_qlora': args.use_qlora,
        'merge_adapters': args.merge_adapters,
        'augment_data': True,
        'train_ratio': 0.8,
        'val_ratio': 0.1,
        'fp16': True,
        'gradient_checkpointing': True,
        'logging_steps': 100,
        'save_steps': 500,
        'eval_steps': 500
    })
    
    # Create pipeline
    pipeline = TrainingPipeline(config=config)
    
    # Run training
    if args.eval_only:
        # Evaluation only
        print("Evaluation mode - not implemented yet")
    else:
        # Full training pipeline
        results = pipeline.run(
            data_path=args.data_path,
            contract_path=args.contract_path or settings.contract_json_path
        )
        
        print(f"\nTraining completed! Results saved to: {results['output_dir']}")


if __name__ == '__main__':
    main()
