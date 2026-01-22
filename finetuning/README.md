# Fine-Tuning Guide

This directory contains scripts and configurations for fine-tuning the Qwen model on legal contract data using PEFT (Parameter-Efficient Fine-Tuning) with LoRA/QLoRA.

## Overview

Fine-tuning adapts the base Qwen model to better understand legal contract terminology, structure, and Q&A patterns. Using LoRA/QLoRA allows efficient fine-tuning with minimal computational resources.

## Quick Start

### 1. Generate Training Data

```bash
python finetuning/generate_training_data.py \
  --contract_path RFP_parsed.json \
  --output_path data/training/legal_qa.jsonl \
  --augment
```

### 2. Train Model

```bash
python finetuning/train.py \
  --data_path data/training \
  --output_dir checkpoints/qwen-legal-lora \
  --base_model Qwen/Qwen2.5-7B-Instruct \
  --lora_r 16 \
  --lora_alpha 32 \
  --batch_size 4 \
  --learning_rate 2e-4 \
  --num_epochs 3
```

### 3. Use Fine-Tuned Model

Set in `.env`:
```bash
FINETUNING_ENABLED=true
FINETUNING_MODEL_PATH=checkpoints/qwen-legal-lora
USE_MERGED_MODEL=true
```

## Configuration

### LoRA Parameters

- **r (rank)**: Controls the rank of LoRA matrices. Higher values = more parameters but better capacity. Default: 16
- **lora_alpha**: Scaling factor. Typically 2x rank. Default: 32
- **lora_dropout**: Dropout rate for LoRA layers. Default: 0.05

### Training Parameters

- **batch_size**: Training batch size. Adjust based on GPU memory. Default: 4
- **learning_rate**: Learning rate. Typical range: 1e-4 to 5e-4. Default: 2e-4
- **num_epochs**: Number of training epochs. Default: 3
- **gradient_accumulation_steps**: Effective batch size = batch_size × gradient_accumulation_steps. Default: 4

### QLoRA (Memory-Efficient)

For limited GPU memory, use QLoRA:

```bash
python finetuning/train.py \
  --use_qlora \
  --data_path data/training \
  --output_dir checkpoints/qwen-legal-qlora
```

## Training Data Format

Training data should be in JSONL format with the following structure:

```json
{
  "instruction": "You are an expert legal document analyst...",
  "input": "Contract Context: ...\n\nQuestion: What is the defect liability period?",
  "output": "Based on Article 5.2, the defect liability period is 2 years..."
}
```

## Output Structure

After training, the output directory will contain:

```
checkpoints/qwen-legal-lora/
├── adapter_config.json          # LoRA configuration
├── adapter_model.bin            # Adapter weights
├── best_model/                  # Best model checkpoint
├── checkpoint-*/                # Periodic checkpoints
├── evaluation_report.json       # Evaluation metrics
└── merged_model/                # Merged model (if merge_adapters=true)
```

## Evaluation

The training script automatically evaluates on the test set and generates:

- **BLEU score**: Measures answer quality
- **ROUGE scores**: Measures answer relevance
- **Exact match**: Exact answer accuracy
- **Legal metrics**: Citation accuracy, legal terminology usage

## Tips

1. **Start Small**: Begin with small datasets and short training runs to validate the setup
2. **Monitor Training**: Use TensorBoard to monitor training progress
3. **Experiment**: Try different LoRA ranks and learning rates
4. **Validate**: Always evaluate on a held-out test set
5. **Iterate**: Fine-tune based on evaluation results

## Troubleshooting

### Out of Memory

- Use QLoRA (`--use_qlora`)
- Reduce batch size
- Reduce max_length
- Enable gradient checkpointing

### Poor Performance

- Increase LoRA rank (r)
- Increase training epochs
- Check data quality
- Verify data format

### Slow Training

- Use gradient accumulation instead of large batch size
- Enable mixed precision training (fp16)
- Use multiple GPUs if available

## Advanced Usage

### Custom Training Loop

See `src/finetuning/trainer.py` for `CustomTrainer` class that provides more control over the training loop.

### Multiple Adapters

You can train multiple task-specific adapters and switch between them:

```python
from src.finetuning.model_loader import ModelLoader

loader = ModelLoader()
model, tokenizer = loader.load_peft_model(
    adapter_path="checkpoints/qwen-legal-lora",
    merge_adapters=False
)
```

## References

- [PEFT Documentation](https://huggingface.co/docs/peft)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
