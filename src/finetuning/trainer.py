"""Training loop and logic for PEFT fine-tuning."""
import os
import torch
from typing import Dict, Any, Optional
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    get_linear_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model
from src.finetuning.peft_config import create_lora_config, print_trainable_parameters
from src.finetuning.dataset import LegalDataset, create_dataloader


class PEFTTrainer:
    """PEFT trainer for fine-tuning."""
    
    def __init__(
        self,
        model_name: str,
        output_dir: str,
        lora_config: Optional[LoraConfig] = None,
        use_qlora: bool = False,
        **training_args
    ):
        """Initialize trainer."""
        self.model_name = model_name
        self.output_dir = output_dir
        self.use_qlora = use_qlora
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        if use_qlora:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto"
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        
        # Apply LoRA
        if lora_config is None:
            lora_config = create_lora_config()
        
        self.model = get_peft_model(self.model, lora_config)
        print_trainable_parameters(self.model)
        
        # Training arguments
        self.training_args = TrainingArguments(
            output_dir=output_dir,
            **training_args
        )
    
    def train(
        self,
        train_dataset: LegalDataset,
        eval_dataset: Optional[LegalDataset] = None,
        callbacks: Optional[list] = None
    ) -> Trainer:
        """Train the model."""
        if callbacks is None:
            callbacks = []
        
        # Add early stopping if eval dataset provided
        if eval_dataset is not None:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=3))
        
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=callbacks,
            tokenizer=self.tokenizer
        )
        
        trainer.train()
        
        # Save model
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        
        return trainer
    
    def save_checkpoint(self, checkpoint_dir: str):
        """Save checkpoint."""
        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)
    
    def load_checkpoint(self, checkpoint_dir: str):
        """Load checkpoint."""
        from peft import PeftModel
        self.model = PeftModel.from_pretrained(self.model, checkpoint_dir)


class CustomTrainer:
    """Custom training loop with more control."""
    
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        learning_rate: float = 2e-4,
        num_epochs: int = 3,
        warmup_steps: int = 100,
        gradient_accumulation_steps: int = 4,
        max_grad_norm: float = 1.0,
        output_dir: str = "./checkpoints",
        save_steps: int = 500,
        eval_steps: int = 500,
        logging_steps: int = 100
    ):
        """Initialize custom trainer."""
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.output_dir = output_dir
        self.save_steps = save_steps
        self.eval_steps = eval_steps
        self.logging_steps = logging_steps
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # Learning rate scheduler
        total_steps = len(train_dataloader) * num_epochs // gradient_accumulation_steps
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_dataloader):
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss / self.gradient_accumulation_steps
            total_loss += loss.item()
            
            # Backward pass
            loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            # Logging
            if (batch_idx + 1) % self.logging_steps == 0:
                print(f"Epoch {epoch}, Batch {batch_idx + 1}, Loss: {loss.item() * self.gradient_accumulation_steps:.4f}")
            
            # Save checkpoint
            if (batch_idx + 1) % self.save_steps == 0:
                checkpoint_dir = os.path.join(self.output_dir, f"checkpoint-{epoch}-{batch_idx}")
                self.save_checkpoint(checkpoint_dir)
            
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return {'loss': avg_loss * self.gradient_accumulation_steps}
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate on validation set."""
        if self.val_dataloader is None:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                total_loss += outputs.loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return {'eval_loss': avg_loss}
    
    def train(self) -> Dict[str, Any]:
        """Run full training."""
        best_eval_loss = float('inf')
        training_history = []
        
        for epoch in range(1, self.num_epochs + 1):
            print(f"\n=== Epoch {epoch}/{self.num_epochs} ===")
            
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Evaluate
            eval_metrics = self.evaluate()
            
            # Log metrics
            metrics = {**train_metrics, **eval_metrics}
            training_history.append(metrics)
            print(f"Train Loss: {train_metrics.get('loss', 0):.4f}")
            if eval_metrics:
                print(f"Eval Loss: {eval_metrics.get('eval_loss', 0):.4f}")
            
            # Save best model
            if eval_metrics and eval_metrics.get('eval_loss', float('inf')) < best_eval_loss:
                best_eval_loss = eval_metrics['eval_loss']
                best_model_dir = os.path.join(self.output_dir, "best_model")
                self.save_checkpoint(best_model_dir)
                print(f"Saved best model with eval_loss: {best_eval_loss:.4f}")
        
        # Save final model
        final_model_dir = os.path.join(self.output_dir, "final_model")
        self.save_checkpoint(final_model_dir)
        
        return {
            'training_history': training_history,
            'best_eval_loss': best_eval_loss
        }
    
    def save_checkpoint(self, checkpoint_dir: str):
        """Save checkpoint."""
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)
