"""End-to-end training pipeline orchestration."""
import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.finetuning.data_generator import TrainingDataGenerator
from src.finetuning.dataset import LegalDataset, load_dataset_from_jsonl, create_dataloader
from src.finetuning.peft_config import create_lora_config, get_default_lora_config
from src.finetuning.trainer import PEFTTrainer, CustomTrainer
from src.finetuning.evaluator import ModelEvaluator
from src.finetuning.model_loader import ModelLoader
from src.config import settings


class TrainingPipeline:
    """End-to-end training pipeline."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize pipeline with configuration."""
        self.config = config or {}
        self.output_dir = self.config.get('output_dir', './checkpoints')
        self.base_model = self.config.get('base_model', settings.finetuning_base_model)
        self.use_qlora = self.config.get('use_qlora', False)
        
        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
    
    def prepare_data(self, contract_path: Optional[str] = None, 
                    output_data_path: Optional[str] = None) -> Dict[str, Any]:
        """Step 1: Prepare training data."""
        print("=== Step 1: Preparing Training Data ===")
        
        generator = TrainingDataGenerator(contract_path=contract_path)
        
        # Generate Q&A pairs
        qa_pairs = generator.generate_qa_pairs_from_contract()
        test_case_examples = generator.generate_from_test_cases()
        
        # Combine and format
        all_examples = qa_pairs + test_case_examples
        formatted = generator.format_for_training(all_examples)
        
        # Augment data
        if self.config.get('augment_data', True):
            formatted = generator.augment_data(formatted, num_augmentations=2)
        
        # Split dataset
        splits = generator.split_dataset(
            formatted,
            train_ratio=self.config.get('train_ratio', 0.8),
            val_ratio=self.config.get('val_ratio', 0.1)
        )
        
        # Save datasets
        if output_data_path:
            data_dir = Path(output_data_path).parent
            data_dir.mkdir(parents=True, exist_ok=True)
            
            generator.save_dataset(splits['train'], 
                                 str(data_dir / 'train.jsonl'), format='jsonl')
            generator.save_dataset(splits['validation'], 
                                 str(data_dir / 'validation.jsonl'), format='jsonl')
            generator.save_dataset(splits['test'], 
                                 str(data_dir / 'test.jsonl'), format='jsonl')
        
        print(f"Generated {len(formatted)} examples")
        print(f"Train: {len(splits['train'])}, Val: {len(splits['validation'])}, Test: {len(splits['test'])}")
        
        return splits
    
    def load_datasets(self, data_path: str, tokenizer: AutoTokenizer) -> Dict[str, LegalDataset]:
        """Load datasets from files."""
        data_dir = Path(data_path)
        
        train_examples = load_dataset_from_jsonl(str(data_dir / 'train.jsonl'))
        val_examples = load_dataset_from_jsonl(str(data_dir / 'validation.jsonl'))
        test_examples = load_dataset_from_jsonl(str(data_dir / 'test.jsonl'))
        
        max_length = self.config.get('max_length', 2048)
        
        datasets = {
            'train': LegalDataset(train_examples, tokenizer, max_length=max_length),
            'validation': LegalDataset(val_examples, tokenizer, max_length=max_length),
            'test': LegalDataset(test_examples, tokenizer, max_length=max_length)
        }
        
        return datasets
    
    def setup_model(self) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Step 2: Setup model and tokenizer."""
        print("\n=== Step 2: Setting Up Model ===")
        
        loader = ModelLoader(base_model_name=self.base_model)
        model, tokenizer = loader.load_base_model(use_quantization=self.use_qlora)
        
        print(f"Loaded base model: {self.base_model}")
        return model, tokenizer
    
    def configure_peft(self) -> Dict[str, Any]:
        """Step 3: Configure PEFT."""
        print("\n=== Step 3: Configuring PEFT ===")
        
        lora_config = create_lora_config(
            r=self.config.get('lora_r', settings.lora_r),
            lora_alpha=self.config.get('lora_alpha', settings.lora_alpha),
            lora_dropout=self.config.get('lora_dropout', settings.lora_dropout)
        )
        
        print(f"LoRA Config: r={lora_config.r}, alpha={lora_config.lora_alpha}, dropout={lora_config.lora_dropout}")
        
        return {'lora_config': lora_config}
    
    def train(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer,
             train_dataset: LegalDataset, val_dataset: Optional[LegalDataset] = None) -> Dict[str, Any]:
        """Step 4: Train model."""
        print("\n=== Step 4: Training Model ===")
        
        lora_config = self.configure_peft()['lora_config']
        
        # Training arguments
        training_args = {
            'output_dir': self.output_dir,
            'num_train_epochs': self.config.get('num_epochs', 3),
            'per_device_train_batch_size': self.config.get('batch_size', 4),
            'gradient_accumulation_steps': self.config.get('gradient_accumulation_steps', 4),
            'learning_rate': self.config.get('learning_rate', 2e-4),
            'warmup_steps': self.config.get('warmup_steps', 100),
            'logging_steps': self.config.get('logging_steps', 100),
            'save_steps': self.config.get('save_steps', 500),
            'eval_steps': self.config.get('eval_steps', 500),
            'evaluation_strategy': 'steps' if val_dataset else 'no',
            'save_strategy': 'steps',
            'load_best_model_at_end': True if val_dataset else False,
            'fp16': self.config.get('fp16', True),
            'bf16': self.config.get('bf16', False),
            'gradient_checkpointing': self.config.get('gradient_checkpointing', True),
            'report_to': ['tensorboard'],
        }
        
        # Create trainer
        trainer = PEFTTrainer(
            model_name=self.base_model,
            output_dir=self.output_dir,
            lora_config=lora_config,
            use_qlora=self.use_qlora,
            **training_args
        )
        
        # Train
        trainer.model = model
        trainer.tokenizer = tokenizer
        
        trainer_instance = trainer.train(train_dataset, eval_dataset=val_dataset)
        
        return {'trainer': trainer_instance}
    
    def evaluate(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer,
                test_dataset: LegalDataset) -> Dict[str, Any]:
        """Step 5: Evaluate model."""
        print("\n=== Step 5: Evaluating Model ===")
        
        evaluator = ModelEvaluator(model, tokenizer)
        
        # Load test examples
        test_examples = []
        for i in range(min(len(test_dataset), 100)):  # Limit for evaluation
            example = test_dataset.examples[i]
            test_examples.append(example)
        
        references = [ex.get('output', '') for ex in test_examples]
        
        # Comprehensive evaluation
        metrics = evaluator.comprehensive_evaluate(test_examples, references)
        
        # Save report
        report_path = os.path.join(self.output_dir, 'evaluation_report.json')
        evaluator.save_evaluation_report(metrics, report_path)
        
        print(f"BLEU: {metrics['bleu']:.4f}")
        print(f"ROUGE-L: {metrics['rouge']['rougeL']:.4f}")
        print(f"Exact Match: {metrics['exact_match']:.4f}")
        
        return metrics
    
    def export_model(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer,
                    merge_adapters: bool = True):
        """Step 6: Export model."""
        print("\n=== Step 6: Exporting Model ===")
        
        loader = ModelLoader(base_model_name=self.base_model)
        
        if merge_adapters:
            # Merge adapters
            if hasattr(model, 'merge_and_unload'):
                merged_model = model.merge_and_unload()
            else:
                merged_model = model
            
            export_path = os.path.join(self.output_dir, 'merged_model')
            loader.save_merged_model(merged_model, tokenizer, export_path)
            print(f"Exported merged model to {export_path}")
        else:
            # Save adapters only
            adapter_path = os.path.join(self.output_dir, 'adapters')
            model.save_pretrained(adapter_path)
            tokenizer.save_pretrained(adapter_path)
            print(f"Exported adapters to {adapter_path}")
    
    def run(self, data_path: Optional[str] = None, contract_path: Optional[str] = None):
        """Run complete training pipeline."""
        print("=" * 60)
        print("Starting Training Pipeline")
        print("=" * 60)
        
        # Step 1: Prepare data
        if data_path and Path(data_path).exists():
            print(f"Loading existing datasets from {data_path}")
            # Load existing datasets
            tokenizer = AutoTokenizer.from_pretrained(self.base_model)
            datasets = self.load_datasets(data_path, tokenizer)
        else:
            splits = self.prepare_data(contract_path=contract_path, 
                                     output_data_path=data_path)
            # Setup model first to get tokenizer
            model, tokenizer = self.setup_model()
            datasets = {
                'train': LegalDataset(splits['train'], tokenizer, 
                                    max_length=self.config.get('max_length', 2048)),
                'validation': LegalDataset(splits['validation'], tokenizer,
                                         max_length=self.config.get('max_length', 2048)),
                'test': LegalDataset(splits['test'], tokenizer,
                                   max_length=self.config.get('max_length', 2048))
            }
        
        # Step 2: Setup model
        model, tokenizer = self.setup_model()
        
        # Step 3: Configure PEFT
        peft_config = self.configure_peft()
        
        # Step 4: Train
        training_results = self.train(
            model, tokenizer,
            datasets['train'],
            datasets.get('validation')
        )
        
        # Step 5: Evaluate
        evaluation_results = self.evaluate(
            model, tokenizer,
            datasets['test']
        )
        
        # Step 6: Export
        self.export_model(model, tokenizer, merge_adapters=self.config.get('merge_adapters', True))
        
        print("\n" + "=" * 60)
        print("Training Pipeline Complete!")
        print("=" * 60)
        
        return {
            'training': training_results,
            'evaluation': evaluation_results,
            'output_dir': self.output_dir
        }
